"""One-run execution and experiment-group orchestration."""

from __future__ import annotations

import json
import hashlib
import logging
import traceback
import uuid
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable

import torch
import pyarrow.parquet as pq

from visconf.config import ResolvedRunConfig
from visconf.datasets import create_dataset_adapter
from visconf.generation.engine import GenerationEngine, GenerationError
from visconf.models.instrumentation import (
    InstrumentationError,
    QwenInstrumentation,
)
from visconf.models.qwen25vl import QwenFacadeError, QwenModelFacade
from visconf.planning import load_experiment_plan
from visconf.scoring.engine import score_completed_rollouts
from visconf.storage.manifest import (
    RunLock,
    append_failure,
    read_manifest,
    record_example_selection,
    record_runtime_identity,
    transition_run_manifest,
    update_experiment_run_status,
    utc_now,
    RunManifest,
)
from visconf.storage.parquet_writer import write_examples
from visconf.storage.resume import (
    RolloutResumeKey,
    build_resume_index,
    quarantine_orphan_parts,
)
from visconf.storage.schema import SCHEMAS
from visconf.storage.transaction import CoreShardTransaction
from visconf.types import (
    Example,
    ExampleRecord,
    FailureRecord,
    ImageRecord,
    PromptConfig,
    RolloutKey,
    RunStatus,
    ScoringMode,
)
from visconf.utils.logging import log_event


logger = logging.getLogger(__name__)


class RunnerError(RuntimeError):
    """Raised when a resolved run cannot be executed."""


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    status: RunStatus
    committed_rollouts: int
    committed_shards: int
    skipped_rollouts: int
    quarantined_parts: int


@dataclass(frozen=True, slots=True)
class GroupSummary:
    experiment_group_id: str
    runs: tuple[RunSummary, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _example_record(
    run: ResolvedRunConfig,
    example: Example,
    prepared: Any,
    ground_truth: Any,
) -> ExampleRecord:
    return ExampleRecord(
        run_id=run.run_id,
        dataset=run.dataset.name,
        split=run.dataset.split,
        sample_id=example.sample_id,
        source_row_index=example.source_row_index,
        question=example.question,
        rendered_prompt=prepared.prompt_record.rendered_prompt,
        prompt_token_ids=prepared.prompt_record.prompt_token_ids,
        ground_truth_json=_canonical_json(ground_truth),
        answer_type=example.answer_type,
        images=tuple(
            ImageRecord(
                source_ref=image.source_ref,
                sha256=image.sha256,
                width=image.width,
                height=image.height,
                mode=image.mode,
            )
            for image in example.images
        ),
        metadata_json=_canonical_json(example.metadata),
    )


def _read_example_rows(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file() or not pq.read_schema(path).equals(SCHEMAS["examples"]):
        raise RunnerError("examples.parquet is missing or has the wrong schema")
    rows = tuple(pq.read_table(path).to_pylist())
    keys = [
        (row["run_id"], row["dataset"], row["split"], row["sample_id"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RunnerError("examples.parquet contains duplicate example keys")
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _example_matches_row(
    run: ResolvedRunConfig,
    example: Example,
    row: dict[str, Any],
) -> bool:
    images = [
        {
            "source_ref": image.source_ref,
            "sha256": image.sha256,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
        }
        for image in example.images
    ]
    return (
        row["run_id"] == run.run_id
        and row["dataset"] == run.dataset.name == example.dataset
        and row["split"] == run.dataset.split == example.split
        and row["sample_id"] == example.sample_id
        and row["source_row_index"] == example.source_row_index
        and row["question"] == example.question
        and row["ground_truth_json"] == _canonical_json(example.ground_truth)
        and row["answer_type"] == example.answer_type
        and row["images"] == images
        and row["metadata_json"] == _canonical_json(example.metadata)
    )


def _validated_examples(
    run: ResolvedRunConfig,
    adapter: Any,
    rows: tuple[dict[str, Any], ...],
):
    for example, row in zip_longest(adapter.load_examples(run.dataset), rows):
        if example is None or row is None:
            raise RunnerError("dataset selection differs from examples.parquet")
        if not _example_matches_row(run, example, row):
            raise RunnerError(
                f"dataset example {example.sample_id!r} differs from persisted selection"
            )
        yield example, row


def _prompt_matches_row(record: ExampleRecord, row: dict[str, Any]) -> bool:
    return (
        row["rendered_prompt"] == record.rendered_prompt
        and tuple(row["prompt_token_ids"]) == record.prompt_token_ids
        and row["prompt_token_count"] == record.prompt_token_count
    )


def _generate_with_rollout_isolation(
    engine: Any,
    model_facade: Any,
    example: Example,
    prepared: Any,
    messages: list[dict[str, object]],
    rollout_keys: tuple[RolloutKey, ...],
    sampling: Any,
) -> tuple[
    tuple[Any, ...],
    tuple[tuple[RolloutKey, Exception, str], ...],
]:
    completed: list[Any] = []
    try:
        completed.extend(
            engine.generate_example(
                example,
                prepared,
                rollout_keys,
                sampling,
            )
        )
        return tuple(completed), ()
    except (InstrumentationError, QwenFacadeError, GenerationError):
        raise
    except Exception:
        completed_keys = {bundle.generation.key for bundle in completed}

    failures: list[tuple[RolloutKey, Exception, str]] = []
    for key in rollout_keys:
        if key in completed_keys:
            continue
        try:
            retry_prepared = model_facade.prepare_example(messages)
            (bundle,) = tuple(
                engine.generate_example(
                    example,
                    retry_prepared,
                    (key,),
                    sampling,
                )
            )
            completed.append(bundle)
        except (InstrumentationError, QwenFacadeError, GenerationError):
            raise
        except Exception as exc:
            failures.append((key, exc, traceback.format_exc()))
    completed.sort(key=lambda bundle: bundle.generation.key.rollout_index)
    return tuple(completed), tuple(failures)


def _resume_key(run: ResolvedRunConfig, sample_id: str, index: int):
    return RolloutResumeKey(
        dataset=run.dataset.name,
        split=run.dataset.split,
        sample_id=sample_id,
        strategy=run.sampling.name,
        rollout_index=index,
    )


def _rollout_key(run: ResolvedRunConfig, sample_id: str, index: int):
    return RolloutKey(
        run_id=run.run_id,
        dataset=run.dataset.name,
        split=run.dataset.split,
        sample_id=sample_id,
        strategy=run.sampling.name,
        rollout_index=index,
    )


def _shard_id(sample_id: str, indices: tuple[int, ...]) -> str:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]
    return f"sample-{digest}-r{indices[0]:04d}-{indices[-1]:04d}"


def _set_status(run: ResolvedRunConfig, status: RunStatus) -> None:
    transition_run_manifest(run.output_dir / "manifest.json", status)
    update_experiment_run_status(
        run.output_dir.parent / "experiment_manifest.json",
        run.run_id,
        status,
    )


def _failure(
    run: ResolvedRunConfig,
    attempt_id: str,
    stage: str,
    exc: BaseException,
    example: Example | None,
    rollout_index: int | None = None,
    traceback_text: str | None = None,
) -> FailureRecord:
    return FailureRecord(
        failure_id=f"failure-{uuid.uuid4().hex}",
        run_id=run.run_id,
        attempt_id=attempt_id,
        stage=stage,
        dataset=run.dataset.name,
        split=run.dataset.split,
        sample_id=example.sample_id if example is not None else None,
        strategy=run.sampling.name,
        rollout_index=rollout_index,
        scorer_name=None,
        scorer_version=None,
        exception_type=type(exc).__name__,
        message=str(exc),
        traceback=traceback_text or traceback.format_exc(),
        retryable=isinstance(exc, torch.cuda.OutOfMemoryError),
        created_at_utc=utc_now(),
    )


def execute_run(
    run: ResolvedRunConfig,
    *,
    adapter: Any | None = None,
    facade: Any | None = None,
    instrumentation_factory: Callable[[Any], Any] | None = None,
    engine_factory: Callable[..., Any] | None = None,
    attempt_id: str | None = None,
) -> RunSummary:
    """Execute or resume one immutable dataset-by-strategy run."""

    attempt = attempt_id or f"attempt-{uuid.uuid4().hex}"
    manifest_path = run.output_dir / "manifest.json"
    group_manifest_path = run.output_dir.parent / "experiment_manifest.json"

    with RunLock(run.output_dir):
        manifest = read_manifest(manifest_path, RunManifest)
        resume = build_resume_index(run)
        if manifest.status is RunStatus.COMPLETE:
            return RunSummary(
                run_id=run.run_id,
                status=RunStatus.COMPLETE,
                committed_rollouts=len(resume.completed_rollouts),
                committed_shards=len(resume.core_shards),
                skipped_rollouts=len(resume.completed_rollouts),
                quarantined_parts=0,
            )

        quarantined = quarantine_orphan_parts(run)
        _set_status(run, RunStatus.RUNNING)
        log_event(
            logger,
            "run_started",
            experiment_group_id=run.experiment_group_id,
            run_id=run.run_id,
            dataset=run.dataset.name,
            split=run.dataset.split,
            strategy=run.sampling.name,
            resumed=bool(resume.completed_rollouts),
        )
        current_example: Example | None = None
        stage = "initialization"
        skipped = 0

        try:
            dataset_adapter = adapter or create_dataset_adapter(
                run.dataset.adapter
            )
            model_facade = facade or QwenModelFacade.load(run.model)
            if hasattr(model_facade, "stop_token_id_names") and hasattr(
                model_facade, "device"
            ):
                record_runtime_identity(
                    manifest_path,
                    device=model_facade.device,
                    stop_token_ids=model_facade.stop_token_id_names(),
                    repository_root=Path(__file__).resolve().parents[2],
                )
            log_event(
                logger,
                "environment_summary",
                run_id=run.run_id,
                device=run.model.device,
                dtype=run.model.dtype,
                torch_version=torch.__version__,
                gpu=(
                    torch.cuda.get_device_name(run.model.device)
                    if torch.cuda.is_available()
                    else None
                ),
            )
            make_instrumentation = (
                instrumentation_factory
                or (lambda value: QwenInstrumentation(value.model))
            )
            make_engine = engine_factory or GenerationEngine
            examples_path = run.output_dir / "examples.parquet"
            if not examples_path.exists():
                stage = "examples_commit"
                example_records = []
                for example in dataset_adapter.load_examples(run.dataset):
                    current_example = example
                    if (
                        example.dataset != run.dataset.name
                        or example.split != run.dataset.split
                    ):
                        raise RunnerError(
                            "dataset adapter emitted an example outside the run"
                        )
                    messages = dataset_adapter.build_messages(
                        example,
                        PromptConfig(run.dataset.prompt_template),
                    )
                    prepared = model_facade.prepare_example(messages)
                    example_records.append(
                        _example_record(
                            run,
                            example,
                            prepared,
                            dataset_adapter.ground_truth(example),
                        )
                    )
                write_examples(
                    examples_path,
                    example_records,
                    run.storage,
                )
            example_rows = _read_example_rows(examples_path)
            record_example_selection(
                manifest_path,
                example_count=len(example_rows),
                selection_sha256=_file_sha256(examples_path),
            )

            stage = "generation"
            example_failures = 0
            with make_instrumentation(model_facade) as instrumentation:
                engine = make_engine(
                    model_facade,
                    instrumentation,
                    base_seed=run.base_seed,
                    max_new_tokens=run.generation.max_new_tokens,
                    rollout_microbatch_size=(
                        run.generation.rollout_microbatch_size
                    ),
                    seed_derivation_version=(
                        run.schemas.seed_derivation_version
                    ),
                )
                for example, persisted_row in _validated_examples(
                    run,
                    dataset_adapter,
                    example_rows,
                ):
                    current_example = example
                    log_event(
                        logger,
                        "example_started",
                        experiment_group_id=run.experiment_group_id,
                        run_id=run.run_id,
                        dataset=run.dataset.name,
                        split=run.dataset.split,
                        strategy=run.sampling.name,
                        sample_id=example.sample_id,
                    )
                    try:
                        pending_indices = tuple(
                            index
                            for index in range(
                                run.generation.rollouts_per_example
                            )
                            if _resume_key(run, example.sample_id, index)
                            not in resume.completed_rollouts
                        )
                        skipped += (
                            run.generation.rollouts_per_example
                            - len(pending_indices)
                        )
                        if not pending_indices:
                            log_event(
                                logger,
                                "example_completed",
                                run_id=run.run_id,
                                sample_id=example.sample_id,
                                resumed=True,
                            )
                            continue

                        stage = "prompt"
                        messages = dataset_adapter.build_messages(
                            example,
                            PromptConfig(run.dataset.prompt_template),
                        )
                        prepared = model_facade.prepare_example(messages)
                        persisted_record = _example_record(
                            run,
                            example,
                            prepared,
                            dataset_adapter.ground_truth(example),
                        )
                        if not _prompt_matches_row(
                            persisted_record,
                            persisted_row,
                        ):
                            raise RunnerError(
                                "rendered prompt differs from examples.parquet"
                            )

                        stage = "generation"
                        rollout_keys = tuple(
                            _rollout_key(run, example.sample_id, index)
                            for index in pending_indices
                        )
                        bundles, rollout_failures = (
                            _generate_with_rollout_isolation(
                                engine,
                                model_facade,
                                example,
                                prepared,
                                messages,
                                rollout_keys,
                                run.sampling.as_domain(),
                            )
                        )
                        for (
                            failed_key,
                            failure_exc,
                            failure_traceback,
                        ) in rollout_failures:
                            example_failures += 1
                            append_failure(
                                run.output_dir / "failures.jsonl",
                                _failure(
                                    run,
                                    attempt,
                                    "generation",
                                    failure_exc,
                                    example,
                                    failed_key.rollout_index,
                                    failure_traceback,
                                ),
                            )
                        if not bundles:
                            continue
                        successful_indices = tuple(
                            bundle.generation.key.rollout_index
                            for bundle in bundles
                        )
                        core_shard_id = _shard_id(
                            example.sample_id,
                            successful_indices,
                        )
                        stage = "core_commit"
                        CoreShardTransaction(run).commit(
                            core_shard_id,
                            attempt,
                            bundles,
                        )
                        log_event(
                            logger,
                            "core_shard_committed",
                            run_id=run.run_id,
                            dataset=run.dataset.name,
                            strategy=run.sampling.name,
                            sample_id=example.sample_id,
                            shard_id=core_shard_id,
                            rollout_count=len(bundles),
                        )
                        if run.scoring.mode is ScoringMode.ONLINE:
                            stage = "scoring"
                            try:
                                score_completed_rollouts(
                                    run,
                                    dataset_adapter,
                                    {example.sample_id: example},
                                    bundles,
                                    core_shard_id=core_shard_id,
                                    attempt_id=attempt,
                                )
                            except Exception as exc:
                                append_failure(
                                    run.output_dir / "failures.jsonl",
                                    _failure(
                                        run,
                                        attempt,
                                        stage,
                                        exc,
                                        example,
                                    ),
                                )
                        log_event(
                            logger,
                            "example_completed",
                            run_id=run.run_id,
                            dataset=run.dataset.name,
                            strategy=run.sampling.name,
                            sample_id=example.sample_id,
                            shard_id=core_shard_id,
                        )
                    except (
                        InstrumentationError,
                        QwenFacadeError,
                        GenerationError,
                    ):
                        raise
                    except Exception as exc:
                        example_failures += 1
                        append_failure(
                            run.output_dir / "failures.jsonl",
                            _failure(run, attempt, stage, exc, example),
                        )
                        log_event(
                            logger,
                            "failure_recorded",
                            run_id=run.run_id,
                            sample_id=example.sample_id,
                            stage=stage,
                            exception_type=type(exc).__name__,
                        )
                    finally:
                        stage = "generation"

            stage = "completion"
            final_resume = build_resume_index(run)
            expected_rollouts = (
                len(example_rows) * run.generation.rollouts_per_example
            )
            final_status = (
                RunStatus.COMPLETE
                if (
                    not example_failures
                    and len(final_resume.completed_rollouts)
                    == expected_rollouts
                )
                else RunStatus.FAILED
            )
            _set_status(run, final_status)
            log_event(
                logger,
                "run_completed" if final_status is RunStatus.COMPLETE else "run_incomplete",
                experiment_group_id=run.experiment_group_id,
                run_id=run.run_id,
                dataset=run.dataset.name,
                strategy=run.sampling.name,
                committed_rollouts=len(final_resume.completed_rollouts),
            )
            return RunSummary(
                run_id=run.run_id,
                status=final_status,
                committed_rollouts=len(final_resume.completed_rollouts),
                committed_shards=len(final_resume.core_shards),
                skipped_rollouts=skipped,
                quarantined_parts=len(quarantined),
            )
        except BaseException as exc:
            log_event(
                logger,
                "failure_recorded",
                run_id=run.run_id,
                dataset=run.dataset.name,
                strategy=run.sampling.name,
                sample_id=(
                    current_example.sample_id
                    if current_example is not None
                    else None
                ),
                stage=stage,
                exception_type=type(exc).__name__,
            )
            append_failure(
                run.output_dir / "failures.jsonl",
                _failure(
                    run,
                    attempt,
                    stage,
                    exc,
                    current_example,
                ),
            )
            transition_run_manifest(manifest_path, RunStatus.FAILED)
            update_experiment_run_status(
                group_manifest_path,
                run.run_id,
                RunStatus.FAILED,
            )
            raise


def execute_experiment_group(group_dir: str | Path) -> GroupSummary:
    """Execute every incomplete run in manifest order."""

    plan = load_experiment_plan(group_dir)
    summaries_list: list[RunSummary] = []
    for run in plan.runs:
        try:
            summaries_list.append(execute_run(run))
        except Exception as exc:
            log_event(
                logger,
                "run_failed",
                experiment_group_id=run.experiment_group_id,
                run_id=run.run_id,
                dataset=run.dataset.name,
                strategy=run.sampling.name,
                exception_type=type(exc).__name__,
            )
            try:
                resume = build_resume_index(run)
                committed_rollouts = len(resume.completed_rollouts)
                committed_shards = len(resume.core_shards)
            except Exception:
                committed_rollouts = committed_shards = 0
            summaries_list.append(
                RunSummary(
                    run_id=run.run_id,
                    status=RunStatus.FAILED,
                    committed_rollouts=committed_rollouts,
                    committed_shards=committed_shards,
                    skipped_rollouts=0,
                    quarantined_parts=0,
                )
            )
    summaries = tuple(summaries_list)
    summary = GroupSummary(
        experiment_group_id=plan.manifest.experiment_group_id,
        runs=summaries,
    )
    log_event(
        logger,
        (
            "group_completed"
            if all(run.status is RunStatus.COMPLETE for run in summaries)
            else "group_incomplete"
        ),
        experiment_group_id=plan.manifest.experiment_group_id,
        run_count=len(summaries),
    )
    return summary
