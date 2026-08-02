"""Production acceptance checks over a frozen experiment plan."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from visconf.datasets.base import prompt_template_hash
from visconf.planning import ExperimentPlan
from visconf.storage.manifest import (
    RunManifest,
    current_git_state,
    dataset_source_hashes,
    read_manifest,
)
from visconf.storage.persistence import verify_persistence_marker
from visconf.storage.resume import (
    build_resume_index,
    discover_orphan_parts,
)
from visconf.storage.schema import SCHEMAS, schema_inventory
from visconf.types import RunStatus


class ProductionValidationError(RuntimeError):
    """Raised when a production-readiness invariant is not proven."""


@dataclass(frozen=True, slots=True)
class ProductionReadiness:
    experiment_group_id: str
    run_count: int
    hardware_profile: str
    benchmark_report: str
    persistent_marker_id: str


def _document_hashes(repository_root: Path) -> dict[str, str]:
    return {
        relative: hashlib.sha256(
            (repository_root / relative).read_bytes()
        ).hexdigest()
        for relative in (
            "docs/PROBABILITY_CONFIDENCE_METRICS.md",
            "docs/ATTENTION_METRICS.MD",
            "docs/VISUAL_HIDDEN_STATE_METRICS.MD",
            "docs/OUTPUT_SCHEMA.md",
            "docs/REPO_SCHEMA.MD",
        )
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_production_readiness(
    plan: ExperimentPlan,
    repository_root: str | Path,
    benchmark_report: str | Path,
) -> ProductionReadiness:
    root = Path(repository_root).resolve()
    if len(plan.runs) != 6:
        raise ProductionValidationError(
            "production group must contain exactly six runs"
        )
    if plan.manifest.status is not RunStatus.COMPLETE:
        raise ProductionValidationError(
            "production experiment manifest must be complete"
        )
    cells = {
        (
            run.dataset.name,
            run.dataset.split,
            run.dataset.filter_id,
            run.sampling.name,
        )
        for run in plan.runs
    }
    if len(cells) != 6 or len({run.run_id for run in plan.runs}) != 6:
        raise ProductionValidationError(
            "production run cells and IDs must be unique"
        )
    if plan.manifest.document_hashes != _document_hashes(root):
        raise ProductionValidationError(
            "normative documents changed after planning"
        )
    current_commit, current_dirty = current_git_state(root)
    if (
        current_commit is None
        or current_commit != plan.manifest.git_commit
        or current_dirty is not False
        or plan.manifest.git_dirty is not False
    ):
        raise ProductionValidationError(
            "production validation requires the clean planned Git commit"
        )

    inventories = schema_inventory()
    parent_entries = {entry.run_id: entry for entry in plan.manifest.runs}
    example_ids_by_run: dict[str, set[str]] = {}
    for run in plan.runs:
        if run.model.revision is None or run.dataset.revision is None:
            raise ProductionValidationError(
                "model and dataset revisions must be frozen"
            )
        if run.generation.rollouts_per_example != 32:
            raise ProductionValidationError(
                "production runs require 32 rollouts per example"
            )
        if (
            run.generation.rollout_microbatch_size
            != run.hardware.default_rollout_microbatch_size
        ):
            raise ProductionValidationError(
                "generation microbatch differs from hardware default"
            )
        manifest = read_manifest(
            run.output_dir / "manifest.json",
            RunManifest,
        )
        if (
            parent_entries[run.run_id].status is not RunStatus.COMPLETE
            or manifest.status is not RunStatus.COMPLETE
        ):
            raise ProductionValidationError(
                "every parent and child run status must be complete"
            )
        if (
            manifest.git_commit != current_commit
            or manifest.git_dirty is not False
        ):
            raise ProductionValidationError(
                "run Git identity differs from the clean validated checkout"
            )
        expected_source_hashes = dataset_source_hashes(run.dataset.source)
        if (
            not expected_source_hashes
            or manifest.dataset_source_hashes != expected_source_hashes
        ):
            raise ProductionValidationError(
                "dataset source hashes are missing or changed"
            )
        if manifest.prompt_template_hashes != {
            "dataset_prompt_template": prompt_template_hash(
                run.dataset.prompt_template
            )
        }:
            raise ProductionValidationError("prompt template hash changed")
        if not manifest.stop_token_ids or any(
            not token_id.isdigit()
            or not names
            or any(not name for name in names)
            for token_id, names in manifest.stop_token_ids.items()
        ):
            raise ProductionValidationError("resolved stop-token IDs are missing")
        if manifest.metric_column_inventory != inventories:
            raise ProductionValidationError(
                "run metric inventory differs from current schemas"
            )
        if not all(
            identity.get("config_hash") and identity.get("code_hash")
            for identity in manifest.scorer_versions
        ):
            raise ProductionValidationError(
                "scorer identity hashes are incomplete"
            )
        examples_path = run.output_dir / "examples.parquet"
        if (
            not examples_path.is_file()
            or not pq.read_schema(examples_path).equals(SCHEMAS["examples"])
        ):
            raise ProductionValidationError(
                "examples.parquet is missing or has the wrong schema"
            )
        examples = pq.read_table(examples_path).to_pylist()
        sample_ids = [row["sample_id"] for row in examples]
        example_ids_by_run[run.run_id] = set(sample_ids)
        if (
            not sample_ids
            or len(sample_ids) != len(set(sample_ids))
            or any(
                row["run_id"] != run.run_id
                or row["dataset"] != run.dataset.name
                or row["split"] != run.dataset.split
                for row in examples
            )
        ):
            raise ProductionValidationError(
                "examples.parquet does not describe one unique run selection"
            )
        if (
            manifest.example_count != len(examples)
            or manifest.example_selection_sha256 != _sha256(examples_path)
        ):
            raise ProductionValidationError(
                "examples.parquet differs from the recorded selection artifact"
            )
        resume = build_resume_index(run)
        expected_rollouts = {
            (
                run.dataset.name,
                run.dataset.split,
                sample_id,
                run.sampling.name,
                rollout_index,
            )
            for sample_id in sample_ids
            for rollout_index in range(32)
        }
        actual_rollouts = {
            (
                key.dataset,
                key.split,
                key.sample_id,
                key.strategy,
                key.rollout_index,
            )
            for key in resume.completed_rollouts
        }
        if actual_rollouts != expected_rollouts:
            raise ProductionValidationError(
                "checkpoint completion does not contain 32 rollouts per example"
            )
        recorded_shards = {
            shard.shard_id for shard in manifest.committed_core_shards
        }
        if recorded_shards != set(resume.core_shards):
            raise ProductionValidationError(
                "run manifest core-shard inventory is incomplete"
            )
        if discover_orphan_parts(run):
            raise ProductionValidationError(
                "run contains uncommitted output parts"
            )

    report_path = Path(benchmark_report).resolve()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionValidationError(
            "hardware benchmark report is missing or invalid"
        ) from exc
    hardware = plan.runs[0].hardware
    benchmarked_run = next(
        (
            run
            for run in plan.runs
            if run.run_id == report.get("run_id")
        ),
        None,
    )
    if (
        benchmarked_run is None
        or report.get("run_config_hash") != benchmarked_run.config_hash
    ):
        raise ProductionValidationError(
            "benchmark report does not identify a resolved run"
        )
    if report.get("hardware_profile") != hardware.model_dump(mode="json"):
        raise ProductionValidationError(
            "benchmark hardware profile differs from the run plan"
        )
    if (
        report.get("dataset") != benchmarked_run.dataset.name
        or report.get("sample_id")
        not in example_ids_by_run[benchmarked_run.run_id]
        or report.get("max_new_tokens")
        != benchmarked_run.generation.max_new_tokens
    ):
        raise ProductionValidationError(
            "benchmark dataset or token limit differs from the run"
        )
    benchmark_manifest = read_manifest(
        benchmarked_run.output_dir / "manifest.json",
        RunManifest,
    )
    required_environment = {
        "python",
        "pytorch",
        "transformers",
        "datasets",
        "pyarrow",
        "qwen_vl_utils",
        "cuda_runtime",
        "cuda_available",
        "gpu_model",
        "gpu_memory_bytes",
    }
    environment = report.get("environment")
    if (
        not isinstance(environment, dict)
        or not required_environment.issubset(environment)
        or environment.get("cuda_available") is not True
        or any(environment.get(field) is None for field in required_environment)
        or any(
            environment[field] != benchmark_manifest.environment.get(field)
            for field in required_environment
        )
    ):
        raise ProductionValidationError(
            "benchmark environment differs from the completed run"
        )
    raw_measurements = report.get("measurements")
    if not isinstance(raw_measurements, list) or any(
        not isinstance(value, dict) for value in raw_measurements
    ):
        raise ProductionValidationError("benchmark measurements are malformed")
    requested = [
        (
            value.get("requested_prompt_batch_size"),
            value.get("requested_rollout_microbatch_size"),
        )
        for value in raw_measurements
    ]
    if (
        any(
            type(prompt) is not int or type(rollout) is not int
            for prompt, rollout in requested
        )
        or len(requested) != len(set(requested))
        or set(requested) != set(hardware.benchmark_batch_shapes)
    ):
        raise ProductionValidationError(
            "benchmark batch shapes are duplicated, missing, or unexpected"
        )
    measurements = {
        (
            value["requested_prompt_batch_size"],
            value["requested_rollout_microbatch_size"],
        ): value
        for value in raw_measurements
    }
    chosen_shape = (
        benchmarked_run.generation.prompt_batch_size,
        benchmarked_run.generation.rollout_microbatch_size,
    )
    if chosen_shape not in measurements:
        raise ProductionValidationError(
            "benchmark does not contain the exact configured production shape"
        )
    expected_rollout_count = max(
        rollout for _, rollout in hardware.benchmark_batch_shapes
    )
    if any(
        measurements[shape].get("status") != "complete"
        or measurements[shape].get("effective_prompt_batch_size") != shape[0]
        or measurements[shape].get("effective_rollout_microbatch_size") != shape[1]
        or measurements[shape].get("maximum_active_decode_rows")
        != shape[0] * shape[1]
        or type(measurements[shape].get("rollout_count")) is not int
        or measurements[shape]["rollout_count"] != expected_rollout_count
        or type(measurements[shape].get("oom_fallbacks")) is not int
        or measurements[shape]["oom_fallbacks"] != 0
        or type(measurements[shape].get("retained_tokens")) is not int
        or measurements[shape]["retained_tokens"] < 0
        or isinstance(measurements[shape].get("tokens_per_second"), bool)
        or not isinstance(measurements[shape].get("tokens_per_second"), (int, float))
        or measurements[shape]["tokens_per_second"] < 0
        or any(
            not isinstance(measurements[shape].get(field), (int, float))
            or measurements[shape][field] < 0
            for field in (
                "prompt_seconds", "decode_seconds", "total_seconds",
                "peak_allocated_bytes", "padding_fraction",
            )
        )
        or not 0 <= measurements[shape]["padding_fraction"] < 1
        or measurements[shape]["total_seconds"]
        < max(
            measurements[shape]["prompt_seconds"],
            measurements[shape]["decode_seconds"],
        )
        for shape in hardware.benchmark_batch_shapes
    ):
        raise ProductionValidationError(
            "every hardware batch shape must complete without fallback"
        )
    retained_hash_values = [
        measurements[shape].get("retained_ids_sha256")
        for shape in hardware.benchmark_batch_shapes
    ]
    if (
        any(not isinstance(value, str) for value in retained_hash_values)
        or len(set(retained_hash_values)) != 1
        or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in retained_hash_values
            if isinstance(value, str)
        )
    ):
        raise ProductionValidationError(
            "benchmark batch shapes produced different retained IDs"
        )

    marker = verify_persistence_marker(
        plan.manifest.output_root
    )
    lock_path = root / "environment" / "runpod-qwen3vl-metrics.lock"
    if not lock_path.is_file():
        raise ProductionValidationError(
            "locked RunPod environment record is missing"
        )
    return ProductionReadiness(
        experiment_group_id=plan.manifest.experiment_group_id,
        run_count=6,
        hardware_profile=hardware.name,
        benchmark_report=str(report_path),
        persistent_marker_id=marker["marker_id"],
    )
