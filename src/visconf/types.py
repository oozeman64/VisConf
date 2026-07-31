"""Stable model-independent domain types."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Mapping, TypeAlias

import torch
from PIL import Image

if TYPE_CHECKING:
    from visconf.metrics.attention import AttentionScenarioMetrics
    from visconf.metrics.hidden_state import HiddenStateMetrics
    from visconf.metrics.probability import ProbabilityMetrics


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


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
        if self.rollout_index < 0:
            raise ValueError("rollout_index must be zero-based and non-negative")


@dataclass(frozen=True, slots=True)
class TokenKey:
    rollout: RolloutKey
    step: int

    def __post_init__(self) -> None:
        if self.step < 1:
            raise ValueError("token step must be one-based")


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    name: str
    temperature: float
    top_p: float
    top_k: int | None
    repetition_penalty: float


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


@dataclass(frozen=True, slots=True)
class ProbabilityMetricRecord:
    key: TokenKey
    metrics_valid: bool
    invalid_reason: str | None
    metrics: "ProbabilityMetrics | None"


@dataclass(frozen=True, slots=True)
class AttentionMetricRecord:
    key: TokenKey
    n_image_tokens: int
    n_prompt_text_tokens: int
    n_generated_text_tokens: int
    all_layers_all_heads: "AttentionScenarioMetrics"
    early_visual_integration: "AttentionScenarioMetrics"
    visual_reasoning: "AttentionScenarioMetrics"

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
