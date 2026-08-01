"""Experiment-group planning and immutable run resolution."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from visconf.config import (
    LoadedExperimentGroup,
    ResolvedRunConfig,
    sha256_json,
)
from visconf.storage.manifest import (
    ExperimentManifest,
    RunManifest,
    RunManifestEntry,
    atomic_write_json,
    build_run_manifest,
    current_git_state,
    read_manifest,
)
from visconf.types import RunCell, RunStatus
from visconf.utils.logging import log_event


logger = logging.getLogger(__name__)


class PlanningError(ValueError):
    """Raised when a plan conflicts with existing experiment state."""


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
    config_hash = sha256_json(payload)
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
        commit, dirty = current_git_state(config.repository_root)
        if (
            commit != plan.manifest.git_commit
            or dirty != plan.manifest.git_dirty
        ):
            raise PlanningError(
                "current Git state differs from the existing experiment plan"
            )
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
    commit, dirty = current_git_state(config.repository_root)
    manifest = ExperimentManifest(
        experiment_group_id=group_id,
        created_at_utc=created_at,
        updated_at_utc=created_at,
        status=RunStatus.PLANNED,
        output_root=root,
        group_config_hash=config.group_config_hash,
        git_commit=commit,
        git_dirty=dirty,
        document_hashes=config.document_hashes,
        shared_configuration={
            "model": config.model.model_dump(mode="json"),
            "hardware": config.hardware.model_dump(mode="json"),
            "generation": config.generation.model_dump(mode="json"),
            "scoring": config.scoring.model_dump(mode="json"),
            "storage": config.storage.model_dump(mode="json"),
            "schemas": config.schemas.model_dump(mode="json"),
        },
        runs=tuple(_entry(run, group_dir) for run in runs),
    )

    for run in runs:
        atomic_write_json(
            run.output_dir / "manifest.json",
            build_run_manifest(
                run,
                created_at=created_at,
                git_commit=commit,
                git_dirty=dirty,
            ),
        )
    atomic_write_json(manifest_path, manifest)
    plan = ExperimentPlan(group_dir=group_dir, manifest=manifest, runs=runs)
    log_event(
        logger,
        "group_planned",
        experiment_group_id=group_id,
        group_dir=group_dir,
        run_count=len(runs),
    )
    return plan


def load_experiment_plan(group_dir: str | Path) -> ExperimentPlan:
    """Load a previously planned experiment group and its resolved runs."""

    directory = Path(group_dir).resolve()
    manifest = read_manifest(
        directory / "experiment_manifest.json", ExperimentManifest
    )

    runs: list[ResolvedRunConfig] = []
    for entry in manifest.runs:
        run_manifest = read_manifest(
            directory / entry.relative_path / "manifest.json", RunManifest
        )
        run = run_manifest.resolved_config
        if (
            run.experiment_group_id != manifest.experiment_group_id
            or run.run_id != entry.run_id
            or run.config_hash != entry.config_hash
            or run.dataset.name != entry.dataset
            or run.dataset.split != entry.split
            or run.dataset.filter_id != entry.filter_id
            or run.sampling.name != entry.strategy
            or run_manifest.status is not entry.status
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
