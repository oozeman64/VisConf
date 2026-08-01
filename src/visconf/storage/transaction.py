"""Checkpoint-last transactions for core and score shards."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from visconf.config import ResolvedRunConfig
from visconf.storage.manifest import (
    atomic_write_json,
    record_shard_inventory,
    utc_now,
)
from visconf.storage.parquet_writer import (
    TemporaryPart,
    publish_part,
    write_temporary_part,
)
from visconf.storage.schema import CORE_TABLES, SCHEMAS
from visconf.types import CompletedRollout, RolloutKey, ScoreRecord, TokenKey


class TransactionError(ValueError):
    """Raised when a shard violates relational or commit invariants."""


FailureInjector = Callable[[str], None]


def _rollout_identity(key: RolloutKey) -> dict[str, object]:
    return {
        "run_id": key.run_id,
        "dataset": key.dataset,
        "split": key.split,
        "sample_id": key.sample_id,
        "strategy": key.strategy,
        "rollout_index": key.rollout_index,
    }


def _score_identity(record: ScoreRecord) -> dict[str, object]:
    return {
        **_rollout_identity(record.key),
        "scorer_name": record.scorer_name,
        "scorer_version": record.scorer_version,
    }


def _assert_run_key(key: RolloutKey, run: ResolvedRunConfig) -> None:
    if (
        key.run_id != run.run_id
        or key.dataset != run.dataset.name
        or key.split != run.dataset.split
        or key.strategy != run.sampling.name
    ):
        raise TransactionError("record identity does not match the resolved run")


def _validate_core(
    run: ResolvedRunConfig,
    rollouts: Sequence[CompletedRollout],
) -> None:
    rollout_keys = [bundle.generation.key for bundle in rollouts]
    if len(rollout_keys) != len(set(rollout_keys)):
        raise TransactionError("rollout keys must be unique within a shard")

    token_keys: list[TokenKey] = []
    for bundle in rollouts:
        generation = bundle.generation
        _assert_run_key(generation.key, run)
        if generation.key.rollout_index >= run.generation.rollouts_per_example:
            raise TransactionError("rollout index exceeds the resolved run")
        if not 0 <= generation.rollout_seed <= 2**64 - 1:
            raise TransactionError("rollout seed is outside uint64")
        if generation.prompt_token_count <= 0:
            raise TransactionError("generation prompt token count must be positive")
        if (
            generation.temperature != run.sampling.temperature
            or generation.top_p != run.sampling.top_p
            or generation.top_k != run.sampling.top_k
            or generation.repetition_penalty
            != run.sampling.repetition_penalty
        ):
            raise TransactionError("generation sampling fields differ from the run")
        if generation.stop_reason == "max_new_tokens":
            valid_stop = (
                generation.terminating_token_id is None
                and generation.hit_max_new_tokens
            )
        elif generation.stop_reason == "stop_token":
            valid_stop = (
                generation.terminating_token_id is not None
                and not generation.hit_max_new_tokens
            )
        else:
            valid_stop = False
        if not valid_stop:
            raise TransactionError("generation stop fields are inconsistent")
        for token, probability, attention, hidden in zip(
            bundle.tokens,
            bundle.probability,
            bundle.attention,
            bundle.hidden_state,
            strict=True,
        ):
            for key in (
                token.key,
                probability.key,
                attention.key,
                hidden.key,
            ):
                _assert_run_key(key.rollout, run)
            expected_context = generation.prompt_token_count + token.key.step - 1
            if (
                token.context_length != expected_context
                or token.predictor_position != expected_context - 1
            ):
                raise TransactionError("token predictor context is misaligned")
            if attention.n_generated_text_tokens != token.key.step - 1:
                raise TransactionError("attention generated-token count is misaligned")
            if min(
                attention.n_image_tokens,
                attention.n_prompt_text_tokens,
                attention.n_generated_text_tokens,
            ) < 0:
                raise TransactionError("attention token counts cannot be negative")
            if probability.metrics_valid != (probability.metrics is not None):
                raise TransactionError("probability validity fields are inconsistent")
            if probability.metrics_valid == (probability.invalid_reason is not None):
                raise TransactionError("probability invalid reason is inconsistent")
            token_keys.append(token.key)

    if len(token_keys) != len(set(token_keys)):
        raise TransactionError("token keys must be unique within a shard")


def _validate_scores(
    run: ResolvedRunConfig,
    records: Sequence[ScoreRecord],
) -> None:
    identities = []
    for record in records:
        _assert_run_key(record.key, run)
        identities.append(
            (
                record.key,
                record.scorer_name,
                record.scorer_version,
            )
        )
    if len(identities) != len(set(identities)):
        raise TransactionError("score keys must be unique within a shard")


def _validate_written_parts(
    parts: Sequence[TemporaryPart],
    expected_counts: dict[str, int],
    shard_id: str,
) -> None:
    for part in parts:
        actual = pq.read_schema(part.temporary_path)
        if not actual.equals(SCHEMAS[part.table_name]):
            raise TransactionError(f"schema mismatch for {part.table_name}")
        if part.row_count != expected_counts[part.table_name]:
            raise TransactionError(f"row-count mismatch for {part.table_name}")
        shard_column = (
            "score_shard_id" if part.table_name == "scores" else "shard_id"
        )
        values = pq.read_table(
            part.temporary_path,
            columns=[shard_column],
        ).column(0).to_pylist()
        if any(value != shard_id for value in values):
            raise TransactionError(
                f"shard identity mismatch for {part.table_name}"
            )


def _call(injector: FailureInjector | None, stage: str) -> None:
    if injector is not None:
        injector(stage)


def _part_inventory(part: TemporaryPart, run_dir: Path) -> dict[str, object]:
    return {
        "table_name": part.table_name,
        "relative_path": part.final_path.relative_to(run_dir).as_posix(),
        "byte_size": part.byte_size,
        "sha256": part.sha256,
        "row_count": part.row_count,
    }


class CoreShardTransaction:
    def __init__(self, run: ResolvedRunConfig) -> None:
        self.run = run
        self.run_dir = run.output_dir

    def commit(
        self,
        shard_id: str,
        attempt_id: str,
        rollouts: Sequence[CompletedRollout],
        *,
        failure_injector: FailureInjector | None = None,
    ) -> dict[str, Any]:
        if not shard_id or not rollouts:
            raise TransactionError("a core shard needs an ID and rollout records")
        _validate_core(self.run, rollouts)

        records: dict[str, list[object]] = {
            "generations": [bundle.generation for bundle in rollouts],
            "tokens": [
                record for bundle in rollouts for record in bundle.tokens
            ],
            "token_probability_metrics": [
                record for bundle in rollouts for record in bundle.probability
            ],
            "token_attention_metrics": [
                record for bundle in rollouts for record in bundle.attention
            ],
            "token_hidden_state_metrics": [
                record for bundle in rollouts for record in bundle.hidden_state
            ],
        }

        parts: list[TemporaryPart] = []
        checkpoint_path = (
            self.run_dir / "checkpoints" / f"shard-{shard_id}.json"
        )
        if checkpoint_path.exists():
            raise TransactionError(f"checkpoint already exists: {shard_id}")
        try:
            for table_name in CORE_TABLES:
                part = write_temporary_part(
                    table_name,
                    records[table_name],
                    shard_id=shard_id,
                    final_path=(
                        self.run_dir
                        / table_name
                        / f"part-{shard_id}.parquet"
                    ),
                    settings=self.run.storage,
                )
                parts.append(part)
                _call(failure_injector, f"written:{table_name}")

            _validate_written_parts(
                parts,
                {name: len(values) for name, values in records.items()},
                shard_id,
            )
            _call(failure_injector, "validated")

            for part in parts:
                publish_part(part)
                _call(
                    failure_injector,
                    f"published:{part.table_name}",
                )

            token_steps = [
                record.key.step
                for bundle in rollouts
                for record in bundle.tokens
            ]
            committed_at = utc_now()
            checkpoint: dict[str, Any] = {
                "checkpoint_type": "core",
                "run_id": self.run.run_id,
                "shard_id": shard_id,
                "attempt_id": attempt_id,
                "output_schema_version": (
                    self.run.schemas.output_schema_version
                ),
                "metric_schema_version": (
                    self.run.schemas.metric_schema_version
                ),
                "completed_rollouts": [
                    _rollout_identity(bundle.generation.key)
                    for bundle in rollouts
                ],
                "parts": [
                    _part_inventory(part, self.run_dir) for part in parts
                ],
                "token_key_count": len(token_steps),
                "min_token_step": min(token_steps) if token_steps else None,
                "max_token_step": max(token_steps) if token_steps else None,
                "committed_at_utc": committed_at,
                "checkpoint_path": checkpoint_path.relative_to(
                    self.run_dir
                ).as_posix(),
            }
            _call(failure_injector, "before_checkpoint")
            atomic_write_json(checkpoint_path, checkpoint)
            _call(failure_injector, "checkpoint_published")

            manifest_path = self.run_dir / "manifest.json"
            if manifest_path.exists():
                record_shard_inventory(manifest_path, checkpoint)
            _call(failure_injector, "manifest_recorded")
            return checkpoint
        finally:
            for part in parts:
                part.temporary_path.unlink(missing_ok=True)


class ScoreShardTransaction:
    def __init__(self, run: ResolvedRunConfig) -> None:
        self.run = run
        self.run_dir = run.output_dir

    def commit(
        self,
        shard_id: str,
        attempt_id: str,
        records: Sequence[ScoreRecord],
        *,
        failure_injector: FailureInjector | None = None,
    ) -> dict[str, Any]:
        if not shard_id or not records:
            raise TransactionError("a score shard needs an ID and score records")
        _validate_scores(self.run, records)
        checkpoint_path = (
            self.run_dir / "checkpoints" / f"score-{shard_id}.json"
        )
        if checkpoint_path.exists():
            raise TransactionError(f"score checkpoint already exists: {shard_id}")

        part = write_temporary_part(
            "scores",
            records,
            shard_id=shard_id,
            final_path=self.run_dir / "scores" / f"part-{shard_id}.parquet",
            settings=self.run.storage,
        )
        try:
            _call(failure_injector, "written:scores")
            _validate_written_parts(
                (part,),
                {"scores": len(records)},
                shard_id,
            )
            _call(failure_injector, "validated")
            publish_part(part)
            _call(failure_injector, "published:scores")

            committed_at = utc_now()
            checkpoint: dict[str, Any] = {
                "checkpoint_type": "score",
                "run_id": self.run.run_id,
                "shard_id": shard_id,
                "attempt_id": attempt_id,
                "output_schema_version": (
                    self.run.schemas.output_schema_version
                ),
                "metric_schema_version": (
                    self.run.schemas.metric_schema_version
                ),
                "completed_scores": [
                    _score_identity(record) for record in records
                ],
                "parts": [_part_inventory(part, self.run_dir)],
                "committed_at_utc": committed_at,
                "checkpoint_path": checkpoint_path.relative_to(
                    self.run_dir
                ).as_posix(),
            }
            _call(failure_injector, "before_checkpoint")
            atomic_write_json(checkpoint_path, checkpoint)
            _call(failure_injector, "checkpoint_published")

            manifest_path = self.run_dir / "manifest.json"
            if manifest_path.exists():
                record_shard_inventory(
                    manifest_path,
                    checkpoint,
                    score=True,
                )
            _call(failure_injector, "manifest_recorded")
            return checkpoint
        finally:
            part.temporary_path.unlink(missing_ok=True)
