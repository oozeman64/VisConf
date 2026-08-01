"""Stable model-independent domain types."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import math
from typing import TYPE_CHECKING, Mapping, TypeAlias

import torch
from PIL import Image

if TYPE_CHECKING:
    from visconf.metrics.attention import AttentionScenarioMetrics
    from visconf.metrics.hidden_state import HiddenStateMetrics
    from visconf.metrics.probability import ProbabilityMetrics


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _required_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _non_negative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


class RunStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class ScoringMode(StrEnum):
    OFFLINE = "offline"
    ONLINE = "online"


@dataclass(frozen=True, slots=True)
class RunCell:
    dataset: str
    split: str
    filter_id: str
    strategy: str


@dataclass(frozen=True, slots=True)
class RolloutKey:
    run_id: str
    dataset: str
    split: str
    sample_id: str
    strategy: str
    rollout_index: int

    def __post_init__(self) -> None:
        for name in ("run_id", "dataset", "split", "sample_id", "strategy"):
            _required_text(getattr(self, name), name)
        _non_negative_int(self.rollout_index, "rollout_index")


@dataclass(frozen=True, slots=True)
class TokenKey:
    rollout: RolloutKey
    step: int

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 1:
            raise ValueError("token step must be one-based")


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    name: str
    temperature: float
    top_p: float
    top_k: int | None
    repetition_penalty: float

    def __post_init__(self) -> None:
        _required_text(self.name, "sampling name")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be finite and in (0, 1]")
        if self.top_k is not None and (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer when specified")
        if self.repetition_penalty != 1.0:
            raise ValueError("repetition_penalty must equal 1.0")


@dataclass(frozen=True, slots=True)
class PromptConfig:
    name: str


@dataclass(frozen=True, slots=True)
class ExampleImage:
    source_ref: str | None
    image: Image.Image
    sha256: str
    width: int
    height: int
    mode: str

    def __post_init__(self) -> None:
        if (
            len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("image sha256 must be lowercase hexadecimal")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        _required_text(self.mode, "image mode")


@dataclass(frozen=True, slots=True)
class Example:
    dataset: str
    split: str
    sample_id: str
    source_row_index: int | None
    question: str
    images: tuple[ExampleImage, ...]
    ground_truth: Mapping[str, JsonValue]
    answer_type: str | None
    metadata: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class PromptRecord:
    rendered_prompt: str
    prompt_token_ids: tuple[int, ...]
    prompt_token_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.rendered_prompt, str):
            raise ValueError("rendered_prompt must be a string")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in self.prompt_token_ids
        ):
            raise ValueError("prompt_token_ids must be non-negative integers")
        if self.prompt_token_count != len(self.prompt_token_ids):
            raise ValueError("prompt_token_count must equal len(prompt_token_ids)")


@dataclass(frozen=True, slots=True)
class TokenGroups:
    image_positions: torch.LongTensor
    prompt_text_positions: torch.LongTensor
    prompt_last_position: int
    prompt_token_count: int


@dataclass(frozen=True, slots=True)
class ImageRecord:
    source_ref: str | None
    sha256: str
    width: int
    height: int
    mode: str

    def __post_init__(self) -> None:
        if (
            len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("image sha256 must be lowercase hexadecimal")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        _required_text(self.mode, "image mode")


@dataclass(frozen=True, slots=True)
class ExampleRecord:
    run_id: str
    dataset: str
    split: str
    sample_id: str
    source_row_index: int | None
    question: str
    rendered_prompt: str
    prompt_token_ids: tuple[int, ...]
    ground_truth_json: str
    answer_type: str | None
    images: tuple[ImageRecord, ...]
    metadata_json: str

    def __post_init__(self) -> None:
        for name in ("run_id", "dataset", "split", "sample_id"):
            _required_text(getattr(self, name), name)
        if self.source_row_index is not None:
            _non_negative_int(self.source_row_index, "source_row_index")
        if not isinstance(self.question, str) or not isinstance(
            self.rendered_prompt, str
        ):
            raise ValueError("question and rendered_prompt must be strings")
        if not self.prompt_token_ids or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.prompt_token_ids
        ):
            raise ValueError("prompt_token_ids must be non-empty non-negative integers")

    @property
    def prompt_token_count(self) -> int:
        return len(self.prompt_token_ids)


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    key: RolloutKey
    rollout_seed: int
    temperature: float
    top_p: float
    top_k: int | None
    repetition_penalty: float
    generated_token_ids: tuple[int, ...]
    generated_text: str
    stop_reason: str
    terminating_token_id: int | None
    hit_max_new_tokens: bool
    prompt_token_count: int
    wall_time_seconds: float | None
    tokens_per_second: float | None
    completed_at_utc: datetime

    def __post_init__(self) -> None:
        if not 0 <= self.rollout_seed <= 2**64 - 1:
            raise ValueError("rollout_seed must fit uint64")
        if self.prompt_token_count <= 0:
            raise ValueError("prompt_token_count must be positive")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.generated_token_ids
        ):
            raise ValueError("generated_token_ids must be non-negative integers")
        if self.stop_reason not in {"stop_token", "max_new_tokens"}:
            raise ValueError("stop_reason is invalid")
        if self.stop_reason == "stop_token" and (
            self.terminating_token_id is None or self.hit_max_new_tokens
        ):
            raise ValueError("stop-token fields are inconsistent")
        if self.stop_reason == "max_new_tokens" and (
            self.terminating_token_id is not None
            or not self.hit_max_new_tokens
        ):
            raise ValueError("maximum-token fields are inconsistent")
        if self.completed_at_utc.tzinfo is None:
            raise ValueError("completed_at_utc must be timezone-aware")

    @property
    def num_retained_tokens(self) -> int:
        return len(self.generated_token_ids)


@dataclass(frozen=True, slots=True)
class TokenRecord:
    key: TokenKey
    token_id: int
    token_piece: str
    token_text: str
    predictor_position: int
    context_length: int

    def __post_init__(self) -> None:
        _non_negative_int(self.token_id, "token_id")
        _non_negative_int(self.predictor_position, "predictor_position")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if not isinstance(self.token_piece, str) or not isinstance(
            self.token_text, str
        ):
            raise ValueError("token_piece and token_text must be strings")


@dataclass(frozen=True, slots=True)
class ProbabilityMetricRecord:
    key: TokenKey
    metrics_valid: bool
    invalid_reason: str | None
    metrics: "ProbabilityMetrics | None"

    def __post_init__(self) -> None:
        if self.metrics_valid != (self.metrics is not None):
            raise ValueError("probability validity does not match metrics")
        if self.metrics_valid == (self.invalid_reason is not None):
            raise ValueError("probability invalid_reason is inconsistent")


@dataclass(frozen=True, slots=True)
class AttentionMetricRecord:
    key: TokenKey
    n_image_tokens: int
    n_prompt_text_tokens: int
    n_generated_text_tokens: int
    all_layers_all_heads: "AttentionScenarioMetrics"
    early_visual_integration: "AttentionScenarioMetrics"
    visual_reasoning: "AttentionScenarioMetrics"

    def __post_init__(self) -> None:
        for name in (
            "n_image_tokens",
            "n_prompt_text_tokens",
            "n_generated_text_tokens",
        ):
            _non_negative_int(getattr(self, name), name)
        if self.n_generated_text_tokens != self.key.step - 1:
            raise ValueError("generated attention count must equal step - 1")

    @property
    def n_prompt_generated_text_tokens(self) -> int:
        return self.n_prompt_text_tokens + self.n_generated_text_tokens

    @property
    def n_all_attn_tokens(self) -> int:
        return self.n_image_tokens + self.n_prompt_generated_text_tokens


@dataclass(frozen=True, slots=True)
class HiddenStateMetricRecord:
    key: TokenKey
    metrics: "HiddenStateMetrics"


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    key: RolloutKey
    scorer_name: str
    scorer_version: str
    is_correct: bool | None
    raw_final_answer: str | None
    extracted_answer: str | None
    scorer_method: str
    score_details_json: str
    scored_at_utc: datetime

    def __post_init__(self) -> None:
        for name in ("scorer_name", "scorer_version", "scorer_method"):
            _required_text(getattr(self, name), name)
        if self.is_correct is not None and not isinstance(self.is_correct, bool):
            raise ValueError("is_correct must be bool or None")
        if self.scored_at_utc.tzinfo is None:
            raise ValueError("scored_at_utc must be timezone-aware")


@dataclass(frozen=True, slots=True)
class FailureRecord:
    failure_id: str
    run_id: str | None
    attempt_id: str
    stage: str
    dataset: str | None
    split: str | None
    sample_id: str | None
    strategy: str | None
    rollout_index: int | None
    scorer_name: str | None
    scorer_version: str | None
    exception_type: str
    message: str
    traceback: str
    retryable: bool
    created_at_utc: datetime

    def __post_init__(self) -> None:
        for name in (
            "failure_id",
            "attempt_id",
            "stage",
            "exception_type",
        ):
            _required_text(getattr(self, name), name)
        if self.rollout_index is not None:
            _non_negative_int(self.rollout_index, "rollout_index")
        if self.created_at_utc.tzinfo is None:
            raise ValueError("created_at_utc must be timezone-aware")


@dataclass(frozen=True, slots=True)
class CompletedRollout:
    generation: GenerationRecord
    tokens: tuple[TokenRecord, ...]
    probability: tuple[ProbabilityMetricRecord, ...]
    attention: tuple[AttentionMetricRecord, ...]
    hidden_state: tuple[HiddenStateMetricRecord, ...]

    def __post_init__(self) -> None:
        expected = tuple(record.key for record in self.tokens)
        for family in (self.probability, self.attention, self.hidden_state):
            if tuple(record.key for record in family) != expected:
                raise ValueError("token metric families must match token key order")
        if any(key.rollout != self.generation.key for key in expected):
            raise ValueError("token rows must belong to the generation rollout")
        if (
            tuple(record.token_id for record in self.tokens)
            != self.generation.generated_token_ids
        ):
            raise ValueError("token rows must match generated_token_ids")
        if tuple(key.step for key in expected) != tuple(
            range(1, len(expected) + 1)
        ):
            raise ValueError("token steps must be contiguous and one-based")
