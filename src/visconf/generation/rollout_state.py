"""Mutable state for one independently seeded rollout."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from visconf.generation.stopping import StopReason
from visconf.types import (
    AttentionMetricRecord,
    HiddenStateMetricRecord,
    ProbabilityMetricRecord,
    RolloutKey,
    TokenRecord,
)


@dataclass(slots=True)
class RolloutState:
    key: RolloutKey
    rollout_seed: int
    generator: torch.Generator
    generated_token_ids: list[int] = field(default_factory=list)
    tokens: list[TokenRecord] = field(default_factory=list)
    probability: list[ProbabilityMetricRecord] = field(default_factory=list)
    attention: list[AttentionMetricRecord] = field(default_factory=list)
    hidden_state: list[HiddenStateMetricRecord] = field(default_factory=list)
    active: bool = True
    stop_reason: StopReason | None = None
    terminating_token_id: int | None = None

    @property
    def next_step(self) -> int:
        return len(self.generated_token_ids) + 1

    def finish(
        self,
        reason: StopReason,
        terminating_token_id: int | None,
    ) -> None:
        self.active = False
        self.stop_reason = reason
        self.terminating_token_id = terminating_token_id

