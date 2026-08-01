"""Versioned online and offline scoring over committed generations."""

from __future__ import annotations

import hashlib
import json
import logging
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq

from visconf.config import ResolvedRunConfig, ScoringSettings
from visconf.datasets import create_dataset_adapter
from visconf.planning import load_experiment_plan
from visconf.scoring.identity import scorer_identity
from visconf.storage.manifest import (
    RunLock,
    append_failure,
    record_scorer_identity,
    utc_now,
)
from visconf.storage.resume import (
    RolloutResumeKey,
    ScoreResumeKey,
    build_resume_index,
    quarantine_orphan_parts,
    validate_checkpoint,
)
from visconf.storage.schema import SCHEMAS
from visconf.storage.transaction import ScoreShardTransaction
from visconf.types import (
    CompletedRollout,
    Example,
    FailureRecord,
    RolloutKey,
    ScoreRecord,
)
from visconf.utils.logging import log_event


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScoringItem:
    key: RolloutKey
    generated_text: str


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    run_id: str
    scorer_name: str
    scorer_version: str
    committed_scores: int
    committed_shards: int
    skipped_scores: int
    failed_scores: int


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _load_examples(
    run: ResolvedRunConfig,
    adapter: Any,
) -> dict[str, Example]:
    path = run.output_dir / "examples.parquet"
    if not path.is_file() or not pq.read_schema(path).equals(SCHEMAS["examples"]):
        raise ValueError("examples.parquet is missing or has the wrong schema")
    rows = pq.read_table(path).to_pylist()
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("examples.parquet contains duplicate sample IDs")

    if not hasattr(adapter, "load_examples"):
        if any(row["images"] for row in rows):
            raise ValueError(
                "offline scoring needs the dataset adapter to reload images"
            )
        return {
            row["sample_id"]: Example(
                dataset=row["dataset"],
                split=row["split"],
                sample_id=row["sample_id"],
                source_row_index=row["source_row_index"],
                question=row["question"],
                images=(),
                ground_truth=json.loads(row["ground_truth_json"]),
                answer_type=row["answer_type"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        }

    examples: dict[str, Example] = {}
    source_examples = adapter.load_examples(run.dataset)
    for example, row in zip_longest(source_examples, rows):
        if example is None or row is None:
            raise ValueError("dataset selection differs from examples.parquet")
        image_identity = [
            {
                "source_ref": image.source_ref,
                "sha256": image.sha256,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
            }
            for image in example.images
        ]
        if (
            row["run_id"] != run.run_id
            or row["dataset"] != run.dataset.name
            or row["split"] != run.dataset.split
            or row["sample_id"] != example.sample_id
            or row["source_row_index"] != example.source_row_index
            or row["question"] != example.question
            or row["ground_truth_json"] != _canonical_json(example.ground_truth)
            or row["answer_type"] != example.answer_type
            or row["images"] != image_identity
            or row["metadata_json"] != _canonical_json(example.metadata)
        ):
            raise ValueError(
                f"dataset example {example.sample_id!r} differs from persisted selection"
            )
        examples[example.sample_id] = example
    return examples


def _resume_key(item: ScoringItem, config: ScoringSettings):
    key = item.key
    from visconf.storage.resume import RolloutResumeKey

    return ScoreResumeKey(
        rollout=RolloutResumeKey(
            dataset=key.dataset,
            split=key.split,
            sample_id=key.sample_id,
            strategy=key.strategy,
            rollout_index=key.rollout_index,
        ),
        scorer_name=config.scorer_name,
        scorer_version=config.scorer_version,
    )


def _items_from_checkpoint(
    run: ResolvedRunConfig,
    checkpoint_path: Path,
) -> tuple[ScoringItem, ...]:
    checkpoint = validate_checkpoint(run, checkpoint_path)
    part = next(
        item
        for item in checkpoint["parts"]
        if item["table_name"] == "generations"
    )
    rows = pq.read_table(run.output_dir / part["relative_path"]).to_pylist()
    return tuple(
        ScoringItem(
            key=RolloutKey(
                run_id=row["run_id"],
                dataset=row["dataset"],
                split=row["split"],
                sample_id=row["sample_id"],
                strategy=row["strategy"],
                rollout_index=row["rollout_index"],
            ),
            generated_text=row["generated_text"],
        )
        for row in rows
    )


def _failure(
    run: ResolvedRunConfig,
    attempt_id: str,
    config: ScoringSettings,
    exc: BaseException,
    item: ScoringItem | None,
) -> FailureRecord:
    return FailureRecord(
        failure_id=f"failure-{uuid.uuid4().hex}",
        run_id=run.run_id,
        attempt_id=attempt_id,
        stage="scoring",
        dataset=run.dataset.name,
        split=run.dataset.split,
        sample_id=item.key.sample_id if item else None,
        strategy=run.sampling.name,
        rollout_index=item.key.rollout_index if item else None,
        scorer_name=config.scorer_name,
        scorer_version=config.scorer_version,
        exception_type=type(exc).__name__,
        message=str(exc),
        traceback=traceback.format_exc(),
        retryable=False,
        created_at_utc=utc_now(),
    )


def _score_records(
    run: ResolvedRunConfig,
    adapter: Any,
    examples: Mapping[str, Example],
    items: Sequence[ScoringItem],
    config: ScoringSettings,
    scored_at: datetime,
    attempt_id: str,
) -> tuple[tuple[ScoreRecord, ...], int]:
    identity = scorer_identity(config)
    records = []
    failures = 0
    for item in items:
        try:
            example = examples[item.key.sample_id]
            result = adapter.score(
                example,
                item.generated_text,
                config,
            )
            if result is None:
                correct = None
                raw_final = None
                extracted = None
                method = "abstain"
                extra = {}
            else:
                correct_value = result.get("is_correct")
                correct = (
                    None if correct_value is None else bool(correct_value)
                )
                raw_value = result.get("raw_final_answer")
                extracted_value = result.get("extracted_answer")
                raw_final = None if raw_value is None else str(raw_value)
                extracted = (
                    None
                    if extracted_value is None
                    else str(extracted_value)
                )
                method = str(result.get("scorer_method") or "unknown")
                extra = {
                    key: value
                    for key, value in result.items()
                    if key
                    not in {
                        "is_correct",
                        "raw_final_answer",
                        "extracted_answer",
                        "scorer_method",
                    }
                }
            details = {
                "answer_type": example.answer_type,
                "ground_truth": example.ground_truth,
                "scorer_config_hash": identity["config_hash"],
                "scorer_code_hash": identity["code_hash"],
                **extra,
            }
            records.append(
                ScoreRecord(
                    key=item.key,
                    scorer_name=config.scorer_name,
                    scorer_version=config.scorer_version,
                    is_correct=correct,
                    raw_final_answer=raw_final,
                    extracted_answer=extracted,
                    scorer_method=method,
                    score_details_json=_canonical_json(details),
                    scored_at_utc=scored_at,
                )
            )
        except BaseException as exc:
            failures += 1
            append_failure(
                run.output_dir / "failures.jsonl",
                _failure(run, attempt_id, config, exc, item),
            )
    return tuple(records), failures


def _score_shard_id(
    core_shard_id: str,
    config: ScoringSettings,
    records: Sequence[ScoreRecord],
) -> str:
    identity = scorer_identity(config)
    payload = {
        "core_shard_id": core_shard_id,
        "name": identity["name"],
        "version": identity["version"],
        "config_hash": identity["config_hash"],
        "code_hash": identity["code_hash"],
        "rollouts": [
            record.key.rollout_index for record in records
        ],
    }
    digest = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()[:16]
    return f"{core_shard_id}-{digest}"


def _commit_scored_items(
    run: ResolvedRunConfig,
    adapter: Any,
    examples: Mapping[str, Example],
    items: Sequence[ScoringItem],
    *,
    core_shard_id: str,
    config: ScoringSettings,
    completed: frozenset[ScoreResumeKey],
    attempt_id: str,
    scored_at: datetime,
) -> tuple[int, int, int, int]:
    pending = tuple(
        item for item in items if _resume_key(item, config) not in completed
    )
    skipped = len(items) - len(pending)
    records, failed = _score_records(
        run,
        adapter,
        examples,
        pending,
        config,
        scored_at,
        attempt_id,
    )
    if not records:
        return 0, 0, skipped, failed
    try:
        score_shard_id = _score_shard_id(core_shard_id, config, records)
        ScoreShardTransaction(run).commit(
            score_shard_id,
            attempt_id,
            records,
        )
    except BaseException as exc:
        append_failure(
            run.output_dir / "failures.jsonl",
            _failure(run, attempt_id, config, exc, None),
        )
        return 0, 0, skipped, failed + len(records)
    log_event(
        logger,
        "score_shard_committed",
        run_id=run.run_id,
        dataset=run.dataset.name,
        strategy=run.sampling.name,
        core_shard_id=core_shard_id,
        score_shard_id=score_shard_id,
        score_count=len(records),
        scorer_name=config.scorer_name,
        scorer_version=config.scorer_version,
    )
    return len(records), 1, skipped, failed


def score_completed_rollouts(
    run: ResolvedRunConfig,
    adapter: Any,
    examples: Mapping[str, Example],
    rollouts: Sequence[CompletedRollout],
    *,
    core_shard_id: str,
    attempt_id: str,
    scoring_config: ScoringSettings | None = None,
    scored_at: datetime | None = None,
) -> ScoreSummary:
    """Score a newly committed core shard without acquiring the run lock."""

    config = scoring_config or run.scoring
    identity = scorer_identity(config)
    record_scorer_identity(run.output_dir / "manifest.json", identity)
    resume = build_resume_index(run)
    items = tuple(
        ScoringItem(
            key=bundle.generation.key,
            generated_text=bundle.generation.generated_text,
        )
        for bundle in rollouts
    )
    committed, shards, skipped, failed = _commit_scored_items(
        run,
        adapter,
        examples,
        items,
        core_shard_id=core_shard_id,
        config=config,
        completed=resume.completed_scores,
        attempt_id=attempt_id,
        scored_at=scored_at or utc_now(),
    )
    return ScoreSummary(
        run_id=run.run_id,
        scorer_name=config.scorer_name,
        scorer_version=config.scorer_version,
        committed_scores=committed,
        committed_shards=shards,
        skipped_scores=skipped,
        failed_scores=failed,
    )


def score_run(
    run: ResolvedRunConfig,
    *,
    adapter: Any | None = None,
    scoring_config: ScoringSettings | None = None,
    attempt_id: str | None = None,
    scored_at: datetime | None = None,
) -> ScoreSummary:
    """Score every missing key in checkpoint-committed core shards."""

    config = scoring_config or run.scoring
    attempt = attempt_id or f"score-attempt-{uuid.uuid4().hex}"
    with RunLock(run.output_dir):
        quarantine_orphan_parts(run)
        identity = scorer_identity(config)
        record_scorer_identity(run.output_dir / "manifest.json", identity)
        resume = build_resume_index(run)
        dataset_adapter = adapter or create_dataset_adapter(
            run.dataset.adapter
        )
        examples = _load_examples(run, dataset_adapter)
        committed = shards = skipped = failed = 0
        timestamp = scored_at or utc_now()
        checkpoint_dir = run.output_dir / "checkpoints"
        for path in sorted(checkpoint_dir.glob("shard-*.json")):
            checkpoint = validate_checkpoint(run, path)
            values = _commit_scored_items(
                run,
                dataset_adapter,
                examples,
                _items_from_checkpoint(run, path),
                core_shard_id=checkpoint["shard_id"],
                config=config,
                completed=resume.completed_scores,
                attempt_id=attempt,
                scored_at=timestamp,
            )
            committed += values[0]
            shards += values[1]
            skipped += values[2]
            failed += values[3]
        return ScoreSummary(
            run_id=run.run_id,
            scorer_name=config.scorer_name,
            scorer_version=config.scorer_version,
            committed_scores=committed,
            committed_shards=shards,
            skipped_scores=skipped,
            failed_scores=failed,
        )


def score_experiment_group(
    group_dir: str | Path,
    *,
    run_id: str | None = None,
) -> tuple[ScoreSummary, ...]:
    plan = load_experiment_plan(group_dir)
    runs = (
        tuple(run for run in plan.runs if run.run_id == run_id)
        if run_id is not None
        else plan.runs
    )
    if not runs:
        raise ValueError(f"unknown run_id {run_id!r}")
    return tuple(score_run(run) for run in runs)
