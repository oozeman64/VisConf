"""Experiment-group planning and immutable run resolution."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ConfigDict

from visconf.config import LoadedExperimentGroup, ResolvedRunConfig, sha256_json
from visconf.types import RunCell, RunStatus


class PlanningError(ValueError):
    """Raised when a plan conflicts with existing experiment state."""


class ManifestModel(ResolvedRunConfig.__base__):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunManifestEntry(ManifestModel):
    run_id: str
    run_label: str
    dataset: str
    split: str
    filter_id: str
    strategy: str
    relative_path: str
    config_hash: str
    status: RunStatus


class ExperimentManifest(ManifestModel):
    manifest_version: int = 1
    experiment_group_id: str
    created_at_utc: datetime
    status: RunStatus
    output_root: Path
    group_config_hash: str
    git_commit: str | None
    git_dirty: bool | None
    document_hashes: dict[str, str]
    runs: tuple[RunManifestEntry, ...]


class RunManifest(ManifestModel):
    manifest_version: int = 1
    experiment_group_id: str
    run_id: str
    run_label: str
    created_at_utc: datetime
    status: RunStatus
    resolved_config: ResolvedRunConfig


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    group_dir: Path
    manifest: ExperimentManifest
    runs: tuple[ResolvedRunConfig, ...]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _group_id(config: LoadedExperimentGroup, now: datetime) -> str:
    if config.group.id is not None:
        return config.group.id
    timestamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"exp-{timestamp}-{config.group_config_hash[:8]}"


def _run_id(experiment_group_id: str, cell: RunCell) -> str:
    identity = ":".join(
        (
            "visconf",
            experiment_group_id,
            cell.dataset,
            cell.split,
            cell.filter_id,
            cell.strategy,
        )
    )
    return f"run-{uuid.uuid5(uuid.NAMESPACE_URL, identity).hex[:16]}"


def _git_state(repository_root: Path) -> tuple[str | None, bool | None]:
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
    if status.returncode != 0:
        return commit.stdout.strip(), None
    return commit.stdout.strip(), bool(status.stdout.strip())


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    )
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
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_model(path: Path, model: type[ManifestModel]) -> ManifestModel:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return model.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise PlanningError(f"cannot load manifest {path}: {exc}") from exc


def _resolved_run(
    config: LoadedExperimentGroup,
    *,
    experiment_group_id: str,
    run_id: str,
    run_label: str,
    output_dir: Path,
    dataset_index: int,
    strategy_index: int,
) -> ResolvedRunConfig:
    dataset = config.datasets[dataset_index]
    sampling = config.strategies[strategy_index]
    payload = {
        "experiment_group_id": experiment_group_id,
        "run_id": run_id,
        "run_label": run_label,
        "output_dir": output_dir,
        "base_seed": config.group.base_seed,
        "dataset": dataset,
        "sampling": sampling,
        "model": config.model,
        "hardware": config.hardware,
        "generation": config.generation,
        "scoring": config.scoring,
        "storage": config.storage,
        "schemas": config.schemas,
        "document_hashes": config.document_hashes,
    }
    config_hash = sha256_json(
        {
            key: (
                value.model_dump(mode="json")
                if hasattr(value, "model_dump")
                else value
            )
            for key, value in payload.items()
        }
    )
    return ResolvedRunConfig(**payload, config_hash=config_hash)


def _expected_runs(
    config: LoadedExperimentGroup,
    *,
    experiment_group_id: str,
    group_dir: Path,
) -> tuple[ResolvedRunConfig, ...]:
    runs: list[ResolvedRunConfig] = []
    for dataset_index, dataset in enumerate(config.datasets):
        for strategy_index, strategy in enumerate(config.strategies):
            cell = RunCell(
                dataset=dataset.name,
                split=dataset.split,
                filter_id=dataset.filter_id,
                strategy=strategy.name,
            )
            run_id = _run_id(experiment_group_id, cell)
            run_label = f"{dataset.name}-{strategy.name}"
            runs.append(
                _resolved_run(
                    config,
                    experiment_group_id=experiment_group_id,
                    run_id=run_id,
                    run_label=run_label,
                    output_dir=group_dir / run_id,
                    dataset_index=dataset_index,
                    strategy_index=strategy_index,
                )
            )
    if len({run.run_id for run in runs}) != len(runs):
        raise PlanningError("run planning produced duplicate run IDs")
    return tuple(runs)


def _entry(run: ResolvedRunConfig, group_dir: Path) -> RunManifestEntry:
    return RunManifestEntry(
        run_id=run.run_id,
        run_label=run.run_label,
        dataset=run.dataset.name,
        split=run.dataset.split,
        filter_id=run.dataset.filter_id,
        strategy=run.sampling.name,
        relative_path=run.output_dir.relative_to(group_dir).as_posix(),
        config_hash=run.config_hash,
        status=RunStatus.PLANNED,
    )


def plan_experiment_group(
    config: LoadedExperimentGroup,
    *,
    experiment_group_id: str | None = None,
    output_root: str | Path | None = None,
    now: datetime | None = None,
) -> ExperimentPlan:
    """Create or reopen the six-run experiment plan."""

    created_at = now or _utc_now()
    group_id = experiment_group_id or _group_id(config, created_at)
    root = Path(output_root or config.group.output_root).resolve()
    group_dir = root / group_id
    manifest_path = group_dir / "experiment_manifest.json"

    if manifest_path.exists():
        plan = load_experiment_plan(group_dir)
        if plan.manifest.group_config_hash != config.group_config_hash:
            raise PlanningError(
                "existing experiment group has a different configuration hash"
            )
        expected = _expected_runs(
            config, experiment_group_id=group_id, group_dir=group_dir
        )
        expected_cells = {
            (
                run.run_id,
                run.dataset.name,
                run.dataset.split,
                run.dataset.filter_id,
                run.sampling.name,
                run.config_hash,
            )
            for run in expected
        }
        actual_cells = {
            (
                run.run_id,
                run.dataset.name,
                run.dataset.split,
                run.dataset.filter_id,
                run.sampling.name,
                run.config_hash,
            )
            for run in plan.runs
        }
        if actual_cells != expected_cells:
            raise PlanningError("existing experiment plan does not match configuration")
        return plan

    runs = _expected_runs(
        config, experiment_group_id=group_id, group_dir=group_dir
    )
    commit, dirty = _git_state(config.repository_root)
    manifest = ExperimentManifest(
        experiment_group_id=group_id,
        created_at_utc=created_at,
        status=RunStatus.PLANNED,
        output_root=root,
        group_config_hash=config.group_config_hash,
        git_commit=commit,
        git_dirty=dirty,
        document_hashes=config.document_hashes,
        runs=tuple(_entry(run, group_dir) for run in runs),
    )

    for run in runs:
        _atomic_write_json(
            run.output_dir / "manifest.json",
            RunManifest(
                experiment_group_id=group_id,
                run_id=run.run_id,
                run_label=run.run_label,
                created_at_utc=created_at,
                status=RunStatus.PLANNED,
                resolved_config=run,
            ),
        )
    _atomic_write_json(manifest_path, manifest)
    return ExperimentPlan(group_dir=group_dir, manifest=manifest, runs=runs)


def load_experiment_plan(group_dir: str | Path) -> ExperimentPlan:
    """Load a previously planned experiment group and its resolved runs."""

    directory = Path(group_dir).resolve()
    manifest = _read_model(
        directory / "experiment_manifest.json", ExperimentManifest
    )
    assert isinstance(manifest, ExperimentManifest)

    runs: list[ResolvedRunConfig] = []
    for entry in manifest.runs:
        run_manifest = _read_model(
            directory / entry.relative_path / "manifest.json", RunManifest
        )
        assert isinstance(run_manifest, RunManifest)
        run = run_manifest.resolved_config
        if (
            run.experiment_group_id != manifest.experiment_group_id
            or run.run_id != entry.run_id
            or run.config_hash != entry.config_hash
            or run.dataset.name != entry.dataset
            or run.dataset.split != entry.split
            or run.dataset.filter_id != entry.filter_id
            or run.sampling.name != entry.strategy
        ):
            raise PlanningError(f"run manifest does not match entry {entry.run_id}")
        runs.append(run)

    return ExperimentPlan(
        group_dir=directory,
        manifest=manifest,
        runs=tuple(runs),
    )


def load_resolved_run(
    output_root: str | Path,
    experiment_group_id: str,
    run_id: str,
) -> ResolvedRunConfig:
    """Load one immutable resolved run configuration."""

    plan = load_experiment_plan(Path(output_root) / experiment_group_id)
    for run in plan.runs:
        if run.run_id == run_id:
            return run
    raise PlanningError(f"unknown run_id {run_id!r}")

