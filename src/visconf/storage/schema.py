"""Exact Arrow schemas for VisConf output schema version 1."""

from __future__ import annotations

import pyarrow as pa

from visconf.metrics.attention import (
    ATTENTION_METRICS,
    ATTENTION_SCENARIOS,
)
from visconf.metrics.hidden_state import HIDDEN_STATE_METRICS
from visconf.metrics.probability import (
    PROBABILITY_FLOAT_METRICS,
    PROBABILITY_INTEGER_METRICS,
)


OUTPUT_SCHEMA_VERSION = 1
METRIC_SCHEMA_VERSION = 1

EXAMPLE_KEY_COLUMNS = ("run_id", "dataset", "split", "sample_id")
ROLLOUT_KEY_COLUMNS = EXAMPLE_KEY_COLUMNS + ("strategy", "rollout_index")
TOKEN_KEY_COLUMNS = ROLLOUT_KEY_COLUMNS + ("step",)
SCORE_KEY_COLUMNS = ROLLOUT_KEY_COLUMNS + ("scorer_name", "scorer_version")

CORE_TABLES = (
    "generations",
    "tokens",
    "token_probability_metrics",
    "token_attention_metrics",
    "token_hidden_state_metrics",
)
PRIMARY_TABLES = ("examples",) + CORE_TABLES + ("scores",)

_STRING = pa.string()
_INT16 = pa.int16()
_INT32 = pa.int32()
_INT64 = pa.int64()
_UINT64 = pa.uint64()
_FLOAT32 = pa.float32()
_FLOAT64 = pa.float64()
_BOOL = pa.bool_()
_TIMESTAMP = pa.timestamp("us", tz="UTC")
_INT32_LIST = pa.list_(pa.field("item", _INT32, nullable=False))


def _example_key() -> list[pa.Field]:
    return [
        pa.field("run_id", _STRING, nullable=False),
        pa.field("dataset", _STRING, nullable=False),
        pa.field("split", _STRING, nullable=False),
        pa.field("sample_id", _STRING, nullable=False),
    ]


def _rollout_key() -> list[pa.Field]:
    return _example_key() + [
        pa.field("strategy", _STRING, nullable=False),
        pa.field("rollout_index", _INT32, nullable=False),
    ]


def _token_key() -> list[pa.Field]:
    return _rollout_key() + [pa.field("step", _INT32, nullable=False)]


def _score_key() -> list[pa.Field]:
    return _rollout_key() + [
        pa.field("scorer_name", _STRING, nullable=False),
        pa.field("scorer_version", _STRING, nullable=False),
    ]


_IMAGE_STRUCT = pa.struct(
    [
        pa.field("source_ref", _STRING, nullable=True),
        pa.field("sha256", _STRING, nullable=False),
        pa.field("width", _INT32, nullable=False),
        pa.field("height", _INT32, nullable=False),
        pa.field("mode", _STRING, nullable=False),
    ]
)
_IMAGE_LIST = pa.list_(pa.field("item", _IMAGE_STRUCT, nullable=False))

EXAMPLES_SCHEMA = pa.schema(
    _example_key()
    + [
        pa.field("source_row_index", _INT64, nullable=True),
        pa.field("question", _STRING, nullable=False),
        pa.field("rendered_prompt", _STRING, nullable=False),
        pa.field("prompt_token_ids", _INT32_LIST, nullable=False),
        pa.field("prompt_token_count", _INT32, nullable=False),
        pa.field("ground_truth_json", _STRING, nullable=False),
        pa.field("answer_type", _STRING, nullable=True),
        pa.field("images", _IMAGE_LIST, nullable=False),
        pa.field("metadata_json", _STRING, nullable=False),
    ]
)

GENERATIONS_SCHEMA = pa.schema(
    _rollout_key()
    + [
        pa.field("shard_id", _STRING, nullable=False),
        pa.field("rollout_seed", _UINT64, nullable=False),
        pa.field("temperature", _FLOAT32, nullable=False),
        pa.field("top_p", _FLOAT32, nullable=False),
        pa.field("top_k", _INT32, nullable=True),
        pa.field("repetition_penalty", _FLOAT32, nullable=False),
        pa.field("generated_token_ids", _INT32_LIST, nullable=False),
        pa.field("generated_text", _STRING, nullable=False),
        pa.field("num_retained_tokens", _INT32, nullable=False),
        pa.field("stop_reason", _STRING, nullable=False),
        pa.field("terminating_token_id", _INT32, nullable=True),
        pa.field("hit_max_new_tokens", _BOOL, nullable=False),
        pa.field("prompt_token_count", _INT32, nullable=False),
        pa.field("wall_time_seconds", _FLOAT64, nullable=True),
        pa.field("tokens_per_second", _FLOAT64, nullable=True),
        pa.field("completed_at_utc", _TIMESTAMP, nullable=False),
    ]
)

TOKENS_SCHEMA = pa.schema(
    _token_key()
    + [
        pa.field("shard_id", _STRING, nullable=False),
        pa.field("token_id", _INT32, nullable=False),
        pa.field("token_piece", _STRING, nullable=False),
        pa.field("token_text", _STRING, nullable=False),
        pa.field("predictor_position", _INT32, nullable=False),
        pa.field("context_length", _INT32, nullable=False),
    ]
)

TOKEN_PROBABILITY_METRICS_SCHEMA = pa.schema(
    _token_key()
    + [
        pa.field("shard_id", _STRING, nullable=False),
        pa.field("metrics_valid", _BOOL, nullable=False),
        pa.field("invalid_reason", _STRING, nullable=True),
    ]
    + [
        pa.field(name, _FLOAT32, nullable=True)
        for name in PROBABILITY_FLOAT_METRICS
    ]
    + [
        pa.field(name, _INT32, nullable=True)
        for name in PROBABILITY_INTEGER_METRICS
    ]
)

_ATTENTION_COUNT_COLUMNS = (
    "n_image_tokens",
    "n_prompt_text_tokens",
    "n_generated_text_tokens",
    "n_prompt_generated_text_tokens",
    "n_all_attn_tokens",
)
_attention_fields: list[pa.Field] = [
    pa.field("shard_id", _STRING, nullable=False),
    *[
        pa.field(name, _INT32, nullable=False)
        for name in _ATTENTION_COUNT_COLUMNS
    ],
]
for scenario in ATTENTION_SCENARIOS:
    _attention_fields.append(
        pa.field(f"{scenario}__valid", _BOOL, nullable=False)
    )
    _attention_fields.extend(
        pa.field(f"{scenario}__{metric}", _FLOAT32, nullable=True)
        for metric in ATTENTION_METRICS
    )

TOKEN_ATTENTION_METRICS_SCHEMA = pa.schema(_token_key() + _attention_fields)

TOKEN_HIDDEN_STATE_METRICS_SCHEMA = pa.schema(
    _token_key()
    + [
        pa.field("shard_id", _STRING, nullable=False),
        pa.field("valid_layers_all", _INT16, nullable=False),
        pa.field(
            "valid_layers_early_visual_integration",
            _INT16,
            nullable=False,
        ),
        pa.field(
            "valid_layers_visual_reasoning",
            _INT16,
            nullable=False,
        ),
        pa.field("last_layer_valid", _BOOL, nullable=False),
    ]
    + [
        pa.field(name, _FLOAT32, nullable=True)
        for name in HIDDEN_STATE_METRICS
    ]
)

SCORES_SCHEMA = pa.schema(
    _score_key()
    + [
        pa.field("score_shard_id", _STRING, nullable=False),
        pa.field("is_correct", _BOOL, nullable=True),
        pa.field("raw_final_answer", _STRING, nullable=True),
        pa.field("extracted_answer", _STRING, nullable=True),
        pa.field("scorer_method", _STRING, nullable=False),
        pa.field("score_details_json", _STRING, nullable=False),
        pa.field("scored_at_utc", _TIMESTAMP, nullable=False),
    ]
)

SCHEMAS: dict[str, pa.Schema] = {
    "examples": EXAMPLES_SCHEMA,
    "generations": GENERATIONS_SCHEMA,
    "tokens": TOKENS_SCHEMA,
    "token_probability_metrics": TOKEN_PROBABILITY_METRICS_SCHEMA,
    "token_attention_metrics": TOKEN_ATTENTION_METRICS_SCHEMA,
    "token_hidden_state_metrics": TOKEN_HIDDEN_STATE_METRICS_SCHEMA,
    "scores": SCORES_SCHEMA,
}


def schema_for(table_name: str) -> pa.Schema:
    try:
        return SCHEMAS[table_name]
    except KeyError as exc:
        raise ValueError(f"unknown storage table {table_name!r}") from exc


def schema_inventory() -> dict[str, tuple[dict[str, object], ...]]:
    return {
        table_name: tuple(
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in schema
        )
        for table_name, schema in SCHEMAS.items()
    }
