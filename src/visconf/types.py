"""Stable model-independent domain types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, TypeAlias

from PIL import Image


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
