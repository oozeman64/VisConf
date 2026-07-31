"""Domain-record conversion and immutable Parquet part writing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from visconf.config import StorageSettings
from visconf.metrics.attention import ATTENTION_METRICS, ATTENTION_SCENARIOS
from visconf.metrics.probability import PROBABILITY_METRICS
from visconf.storage.schema import schema_for
from visconf.types import (
    AttentionMetricRecord,
    ExampleRecord,
    GenerationRecord,
    HiddenStateMetricRecord,
    ProbabilityMetricRecord,
    RolloutKey,
    ScoreRecord,
    TokenKey,
    TokenRecord,
)


class StorageError(ValueError):
    """Raised when a record cannot be represented by the storage contract."""


@dataclass(frozen=True, slots=True)
class TemporaryPart:
    table_name: str
    temporary_path: Path
    final_path: Path
    row_count: int
    byte_size: int
    sha256: str


def _rollout_key_row(key: RolloutKey) -> dict[str, object]:
    return {
        "run_id": key.run_id,
        "dataset": key.dataset,
        "split": key.split,
        "sample_id": key.sample_id,
        "strategy": key.strategy,
        "rollout_index": key.rollout_index,
    }


def _token_key_row(key: TokenKey) -> dict[str, object]:
    return {**_rollout_key_row(key.rollout), "step": key.step}


def _canonical_json(text: str) -> str:
    try:
        value = json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
        canonical = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StorageError("JSON fields must contain canonical finite JSON") from exc
    if canonical != text:
        raise StorageError("JSON fields must use canonical encoding")
    return text


def _example_row(record: ExampleRecord) -> dict[str, object]:
    return {
        "run_id": record.run_id,
        "dataset": record.dataset,
        "split": record.split,
        "sample_id": record.sample_id,
        "source_row_index": record.source_row_index,
        "question": record.question,
        "rendered_prompt": record.rendered_prompt,
        "prompt_token_ids": list(record.prompt_token_ids),
        "prompt_token_count": record.prompt_token_count,
        "ground_truth_json": _canonical_json(record.ground_truth_json),
        "answer_type": record.answer_type,
        "images": [asdict(image) for image in record.images],
        "metadata_json": _canonical_json(record.metadata_json),
    }


def _generation_row(
    record: GenerationRecord,
    shard_id: str,
) -> dict[str, object]:
    return {
        **_rollout_key_row(record.key),
        "shard_id": shard_id,
        "rollout_seed": record.rollout_seed,
        "temperature": record.temperature,
        "top_p": record.top_p,
        "top_k": record.top_k,
        "repetition_penalty": record.repetition_penalty,
        "generated_token_ids": list(record.generated_token_ids),
        "generated_text": record.generated_text,
        "num_retained_tokens": record.num_retained_tokens,
        "stop_reason": record.stop_reason,
        "terminating_token_id": record.terminating_token_id,
        "hit_max_new_tokens": record.hit_max_new_tokens,
        "prompt_token_count": record.prompt_token_count,
        "wall_time_seconds": record.wall_time_seconds,
        "tokens_per_second": record.tokens_per_second,
        "completed_at_utc": record.completed_at_utc,
    }


def _token_row(record: TokenRecord, shard_id: str) -> dict[str, object]:
    return {
        **_token_key_row(record.key),
        "shard_id": shard_id,
        "token_id": record.token_id,
        "token_piece": record.token_piece,
        "token_text": record.token_text,
        "predictor_position": record.predictor_position,
        "context_length": record.context_length,
    }


def _probability_row(
    record: ProbabilityMetricRecord,
    shard_id: str,
) -> dict[str, object]:
    values = (
        asdict(record.metrics)
        if record.metrics is not None
        else {name: None for name in PROBABILITY_METRICS}
    )
    return {
        **_token_key_row(record.key),
        "shard_id": shard_id,
        "metrics_valid": record.metrics_valid,
        "invalid_reason": record.invalid_reason,
        **values,
    }


def _attention_row(
    record: AttentionMetricRecord,
    shard_id: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        **_token_key_row(record.key),
        "shard_id": shard_id,
        "n_image_tokens": record.n_image_tokens,
        "n_prompt_text_tokens": record.n_prompt_text_tokens,
        "n_generated_text_tokens": record.n_generated_text_tokens,
        "n_prompt_generated_text_tokens": (
            record.n_prompt_generated_text_tokens
        ),
        "n_all_attn_tokens": record.n_all_attn_tokens,
    }
    for scenario in ATTENTION_SCENARIOS:
        result = getattr(record, scenario)
        row[f"{scenario}__valid"] = result.valid
        for metric in ATTENTION_METRICS:
            row[f"{scenario}__{metric}"] = getattr(result, metric)
    return row


def _hidden_row(
    record: HiddenStateMetricRecord,
    shard_id: str,
) -> dict[str, object]:
    values = asdict(record.metrics)
    return {
        **_token_key_row(record.key),
        "shard_id": shard_id,
        **values,
    }


def _score_row(record: ScoreRecord, shard_id: str) -> dict[str, object]:
    return {
        **_rollout_key_row(record.key),
        "scorer_name": record.scorer_name,
        "scorer_version": record.scorer_version,
        "score_shard_id": shard_id,
        "is_correct": record.is_correct,
        "raw_final_answer": record.raw_final_answer,
        "extracted_answer": record.extracted_answer,
        "scorer_method": record.scorer_method,
        "score_details_json": _canonical_json(record.score_details_json),
        "scored_at_utc": record.scored_at_utc,
    }


def _reject_nan(value: object) -> None:
    if isinstance(value, float) and math.isnan(value):
        raise StorageError("NaN is not a valid stored scalar")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nan(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_nan(item)


def records_to_table(
    table_name: str,
    records: Sequence[object],
    *,
    shard_id: str | None = None,
) -> pa.Table:
    converters = {
        "examples": lambda record: _example_row(record),
        "generations": lambda record: _generation_row(record, shard_id or ""),
        "tokens": lambda record: _token_row(record, shard_id or ""),
        "token_probability_metrics": lambda record: _probability_row(
            record, shard_id or ""
        ),
        "token_attention_metrics": lambda record: _attention_row(
            record, shard_id or ""
        ),
        "token_hidden_state_metrics": lambda record: _hidden_row(
            record, shard_id or ""
        ),
        "scores": lambda record: _score_row(record, shard_id or ""),
    }
    if table_name != "examples" and not shard_id:
        raise StorageError(f"{table_name} requires a shard ID")
    try:
        converter = converters[table_name]
    except KeyError as exc:
        raise StorageError(f"unknown storage table {table_name!r}") from exc

    rows = [converter(record) for record in records]
    for row in rows:
        if set(row) != set(schema_for(table_name).names):
            raise StorageError(f"{table_name} row does not match its schema")
        _reject_nan(row)
    try:
        return pa.Table.from_pylist(
            rows,
            schema=schema_for(table_name),
        )
    except (pa.ArrowException, OverflowError, TypeError, ValueError) as exc:
        raise StorageError(f"cannot convert {table_name} records") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_temporary_part(
    table_name: str,
    records: Sequence[object],
    *,
    shard_id: str | None,
    final_path: Path,
    settings: StorageSettings,
) -> TemporaryPart:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    table = records_to_table(table_name, records, shard_id=shard_id)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.",
        suffix=".tmp",
        dir=final_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        pq.write_table(
            table,
            temporary_path,
            compression=settings.compression,
            compression_level=settings.compression_level,
            use_dictionary=settings.use_dictionary,
            write_statistics=settings.write_statistics,
            row_group_size=settings.target_row_group_size,
        )
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        metadata = TemporaryPart(
            table_name=table_name,
            temporary_path=temporary_path,
            final_path=final_path,
            row_count=table.num_rows,
            byte_size=temporary_path.stat().st_size,
            sha256=_sha256(temporary_path),
        )
        return metadata
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def publish_part(part: TemporaryPart) -> None:
    if part.final_path.exists():
        raise StorageError(f"immutable part already exists: {part.final_path}")
    os.rename(part.temporary_path, part.final_path)


def write_examples(
    path: Path,
    records: Sequence[ExampleRecord],
    settings: StorageSettings,
) -> TemporaryPart:
    part = write_temporary_part(
        "examples",
        records,
        shard_id=None,
        final_path=path,
        settings=settings,
    )
    publish_part(part)
    return part
