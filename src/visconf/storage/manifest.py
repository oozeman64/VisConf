"""Typed manifests, failure logging, and exclusive run locks."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from contextlib import AbstractContextManager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow
import torch
from pydantic import Field

from visconf.config import FrozenModel, ResolvedRunConfig
from visconf.datasets.base import prompt_template_hash
from visconf.scoring.identity import scorer_identity
from visconf.storage.schema import schema_inventory
from visconf.types import FailureRecord, RunStatus


class ManifestError(ValueError):
    """Raised when manifest state violates the run contract."""


class RunManifestEntry(FrozenModel):
    run_id: str
    run_label: str
    dataset: str
    split: str
    filter_id: str
    strategy: str
    relative_path: str
    config_hash: str
    status: RunStatus


class ExperimentManifest(FrozenModel):
    manifest_version: int = 1
    experiment_group_id: str
    created_at_utc: datetime
    updated_at_utc: datetime
    status: RunStatus
    output_root: Path
    group_config_hash: str
    git_commit: str | None
    git_dirty: bool | None
    document_hashes: dict[str, str]
    shared_configuration: dict[str, Any] = Field(default_factory=dict)
    runs: tuple[RunManifestEntry, ...]


class PartInventory(FrozenModel):
    table_name: str
    relative_path: str
    byte_size: int
    sha256: str
    row_count: int


class ShardInventory(FrozenModel):
    shard_id: str
    checkpoint_path: str
    committed_at_utc: datetime
    parts: tuple[PartInventory, ...]


class RunManifest(FrozenModel):
    manifest_version: int = 1
    experiment_group_id: str
    run_id: str
    run_label: str
    created_at_utc: datetime
    updated_at_utc: datetime
    status: RunStatus
    resolved_config: ResolvedRunConfig
    output_schema_version: int
    metric_schema_version: int
    seed_derivation_version: int
    decoder_layer_count: int = 36
    layer_ranges: dict[str, tuple[int, int]]
    metric_column_inventory: dict[str, tuple[dict[str, Any], ...]]
    dataset_source_hashes: dict[str, str] = Field(default_factory=dict)
    stop_token_ids: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    prompt_template_hashes: dict[str, str]
    scorer_versions: tuple[dict[str, str], ...]
    parquet_settings: dict[str, Any]
    environment: dict[str, Any]
    git_commit: str | None
    git_dirty: bool | None
    example_count: int | None = None
    example_selection_sha256: str | None = None
    committed_core_shards: tuple[ShardInventory, ...] = ()
    committed_score_shards: tuple[ShardInventory, ...] = ()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(value)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_manifest(path: Path, model: type[ExperimentManifest] | type[RunManifest]):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return model.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ManifestError(f"cannot load manifest {path}: {exc}") from exc


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_environment(device: str | torch.device | None = None) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    cuda_device = torch.device(device) if device is not None else None
    if cuda_available and cuda_device is None:
        cuda_device = torch.device("cuda", torch.cuda.current_device())
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": _version("transformers"),
        "datasets": _version("datasets"),
        "pyarrow": pyarrow.__version__,
        "qwen_vl_utils": _version("qwen-vl-utils"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_model": (
            torch.cuda.get_device_name(cuda_device) if cuda_available else None
        ),
        "gpu_memory_bytes": (
            torch.cuda.get_device_properties(cuda_device).total_memory
            if cuda_available
            else None
        ),
    }


def current_git_state(repository_root: Path) -> tuple[str | None, bool | None]:
    commit = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0:
        return None, None
    status = subprocess.run(
        ["git", "-C", str(repository_root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return (
        commit.stdout.strip(),
        bool(status.stdout.strip()) if status.returncode == 0 else None,
    )


def dataset_source_hashes(source: Path) -> dict[str, str]:
    """Hash configured local source files when they are available."""

    if source.is_file():
        paths = (source,)
        base = source.parent
    elif source.is_dir():
        paths = tuple(sorted(path for path in source.rglob("*") if path.is_file()))
        base = source
    else:
        return {}
    hashes: dict[str, str] = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[path.relative_to(base).as_posix()] = digest.hexdigest()
    return hashes


def build_run_manifest(
    run: ResolvedRunConfig,
    *,
    created_at: datetime,
    git_commit: str | None,
    git_dirty: bool | None,
) -> RunManifest:
    prompt_hash = prompt_template_hash(run.dataset.prompt_template)
    return RunManifest(
        experiment_group_id=run.experiment_group_id,
        run_id=run.run_id,
        run_label=run.run_label,
        created_at_utc=created_at,
        updated_at_utc=created_at,
        status=RunStatus.PLANNED,
        resolved_config=run,
        output_schema_version=run.schemas.output_schema_version,
        metric_schema_version=run.schemas.metric_schema_version,
        seed_derivation_version=run.schemas.seed_derivation_version,
        layer_ranges={
            "all_layers_all_heads": (1, 36),
            "early_visual_integration": (1, 21),
            "visual_reasoning": (22, 34),
            "last_layer": (36, 36),
        },
        metric_column_inventory=schema_inventory(),
        dataset_source_hashes=dataset_source_hashes(run.dataset.source),
        prompt_template_hashes={"dataset_prompt_template": prompt_hash},
        scorer_versions=(scorer_identity(run.scoring),),
        parquet_settings=run.storage.model_dump(mode="json"),
        environment={},
        git_commit=git_commit,
        git_dirty=git_dirty,
    )


def record_runtime_identity(
    manifest_path: Path,
    *,
    device: str | torch.device,
    stop_token_ids: dict[str, tuple[str, ...]],
    repository_root: Path,
) -> RunManifest:
    manifest = read_manifest(manifest_path, RunManifest)
    commit, dirty = current_git_state(repository_root)
    if commit != manifest.git_commit or dirty != manifest.git_dirty:
        raise ManifestError("current Git state differs from the planned run")
    source_hashes = dataset_source_hashes(
        manifest.resolved_config.dataset.source
    )
    if (
        manifest.dataset_source_hashes
        and manifest.dataset_source_hashes != source_hashes
    ):
        raise ManifestError("dataset sources changed after planning")
    expected_prompt_hashes = {
        "dataset_prompt_template": prompt_template_hash(
            manifest.resolved_config.dataset.prompt_template
        )
    }
    if manifest.prompt_template_hashes != expected_prompt_hashes:
        raise ManifestError("prompt template changed after planning")
    updated = manifest.model_copy(
        update={
            "environment": capture_environment(device),
            "stop_token_ids": stop_token_ids,
            "dataset_source_hashes": source_hashes,
            "updated_at_utc": utc_now(),
        }
    )
    atomic_write_json(manifest_path, updated)
    return updated


def record_example_selection(
    manifest_path: Path,
    *,
    example_count: int,
    selection_sha256: str,
) -> RunManifest:
    manifest = read_manifest(manifest_path, RunManifest)
    if example_count < 0 or len(selection_sha256) != 64:
        raise ManifestError("example selection identity is invalid")
    if manifest.example_count is not None and (
        manifest.example_count != example_count
        or manifest.example_selection_sha256 != selection_sha256
    ):
        raise ManifestError("examples.parquet differs from the recorded selection")
    updated = manifest.model_copy(
        update={
            "example_count": example_count,
            "example_selection_sha256": selection_sha256,
            "updated_at_utc": utc_now(),
        }
    )
    atomic_write_json(manifest_path, updated)
    return updated


_ALLOWED_TRANSITIONS = {
    (RunStatus.PLANNED, RunStatus.RUNNING),
    (RunStatus.RUNNING, RunStatus.COMPLETE),
    (RunStatus.RUNNING, RunStatus.FAILED),
    (RunStatus.FAILED, RunStatus.RUNNING),
}


def transition_run_manifest(
    path: Path,
    status: RunStatus,
    *,
    now: datetime | None = None,
) -> RunManifest:
    manifest = read_manifest(path, RunManifest)
    if status != manifest.status and (manifest.status, status) not in _ALLOWED_TRANSITIONS:
        raise ManifestError(
            f"invalid run status transition {manifest.status} -> {status}"
        )
    updated = manifest.model_copy(
        update={"status": status, "updated_at_utc": now or utc_now()}
    )
    atomic_write_json(path, updated)
    return updated


def update_experiment_run_status(
    path: Path,
    run_id: str,
    status: RunStatus,
    *,
    now: datetime | None = None,
) -> ExperimentManifest:
    manifest = read_manifest(path, ExperimentManifest)
    found = False
    runs = []
    for entry in manifest.runs:
        if entry.run_id == run_id:
            if status != entry.status and (entry.status, status) not in _ALLOWED_TRANSITIONS:
                raise ManifestError(
                    f"invalid parent run status transition {entry.status} -> {status}"
                )
            child = read_manifest(
                path.parent / entry.relative_path / "manifest.json",
                RunManifest,
            )
            if child.status is not status:
                raise ManifestError(
                    "parent run status must match the child manifest"
                )
            entry = entry.model_copy(update={"status": status})
            found = True
        runs.append(entry)
    if not found:
        raise ManifestError(f"unknown run ID {run_id}")

    statuses = {entry.status for entry in runs}
    if statuses == {RunStatus.COMPLETE}:
        group_status = RunStatus.COMPLETE
    elif RunStatus.FAILED in statuses:
        group_status = RunStatus.FAILED
    elif statuses != {RunStatus.PLANNED}:
        group_status = RunStatus.RUNNING
    else:
        group_status = RunStatus.PLANNED
    updated = manifest.model_copy(
        update={
            "runs": tuple(runs),
            "status": group_status,
            "updated_at_utc": now or utc_now(),
        }
    )
    atomic_write_json(path, updated)
    return updated


def record_shard_inventory(
    manifest_path: Path,
    checkpoint: dict[str, Any],
    *,
    score: bool = False,
) -> RunManifest:
    manifest = read_manifest(manifest_path, RunManifest)
    inventory = ShardInventory(
        shard_id=checkpoint["shard_id"],
        checkpoint_path=checkpoint["checkpoint_path"],
        committed_at_utc=checkpoint["committed_at_utc"],
        parts=tuple(PartInventory.model_validate(part) for part in checkpoint["parts"]),
    )
    field = "committed_score_shards" if score else "committed_core_shards"
    current = getattr(manifest, field)
    if any(item.shard_id == inventory.shard_id for item in current):
        raise ManifestError(f"shard already recorded: {inventory.shard_id}")
    updated = manifest.model_copy(
        update={
            field: current + (inventory,),
            "updated_at_utc": utc_now(),
        }
    )
    atomic_write_json(manifest_path, updated)
    return updated


def record_scorer_identity(
    manifest_path: Path,
    identity: dict[str, str],
) -> RunManifest:
    manifest = read_manifest(manifest_path, RunManifest)
    for current in manifest.scorer_versions:
        if (
            current["name"] == identity["name"]
            and current["version"] == identity["version"]
        ):
            if (
                current["config_hash"] != identity["config_hash"]
                or current["code_hash"] != identity["code_hash"]
            ):
                raise ManifestError(
                    "scorer version conflicts with recorded hashes"
                )
            return manifest
    updated = manifest.model_copy(
        update={
            "scorer_versions": manifest.scorer_versions + (identity,),
            "updated_at_utc": utc_now(),
        }
    )
    atomic_write_json(manifest_path, updated)
    return updated


def append_failure(path: Path, failure: FailureRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _jsonable(asdict(failure)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class RunLock(AbstractContextManager["RunLock"]):
    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / ".run.lock"
        self._handle = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+")
        try:
            fcntl.flock(
                self._handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            self._handle.close()
            self._handle = None
            raise ManifestError(f"run is already locked: {self.path.parent}") from exc
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None
