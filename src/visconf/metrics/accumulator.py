"""Online scalar and vector accumulators used by instrumentation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from visconf.metrics.attention import (
    ATTENTION_SCENARIO_LAYERS,
    ATTENTION_SCENARIOS,
)
from visconf.metrics.hidden_state import HiddenStateMetrics
from visconf.metrics.validation import (
    MetricInputError,
    attention_rows,
    hidden_cosine_value,
)


@dataclass(frozen=True, slots=True)
class AttentionScenarioAggregate:
    valid: bool
    vector: torch.Tensor | None


class _ScenarioAttentionAccumulator:
    def __init__(self, expected_layers: frozenset[int]) -> None:
        self.expected_layers = expected_layers
        self.seen_layers: set[int] = set()
        self.vector_sum: torch.Tensor | None = None
        self.head_count = 0
        self.valid = True

    def add(
        self,
        layer_number: int,
        rows: torch.Tensor | None,
    ) -> None:
        self.seen_layers.add(layer_number)
        if rows is None:
            self.valid = False
            return

        layer_sum = rows.sum(dim=0)
        if self.vector_sum is None:
            self.vector_sum = layer_sum
        elif self.vector_sum.shape != layer_sum.shape:
            raise MetricInputError(
                "attention key width changed within one forward pass"
            )
        else:
            self.vector_sum.add_(layer_sum)
        self.head_count += rows.shape[0]

    def finalize(self) -> AttentionScenarioAggregate:
        if self.seen_layers != self.expected_layers:
            raise MetricInputError("attention scenario is missing decoder layers")
        if not self.valid:
            return AttentionScenarioAggregate(valid=False, vector=None)
        if self.vector_sum is None or self.head_count == 0:
            raise MetricInputError("attention scenario has no head contributions")
        return AttentionScenarioAggregate(
            valid=True,
            vector=self.vector_sum / self.head_count,
        )


class AttentionAccumulator:
    """Average predictor attention online across the three layer scenarios."""

    def __init__(self) -> None:
        self._seen_layers: set[int] = set()
        self._scenarios = {
            name: _ScenarioAttentionAccumulator(ATTENTION_SCENARIO_LAYERS[name])
            for name in ATTENTION_SCENARIOS
        }

    @torch.inference_mode()
    def add_layer(
        self,
        layer_number: int,
        predictor_attention_rows: torch.Tensor,
    ) -> None:
        if not 1 <= layer_number <= 36:
            raise MetricInputError("decoder layer number must be in 1..36")
        if layer_number in self._seen_layers:
            raise MetricInputError("decoder layer was accumulated more than once")
        self._seen_layers.add(layer_number)

        rows, _ = attention_rows(predictor_attention_rows)
        for name, accumulator in self._scenarios.items():
            if layer_number in ATTENTION_SCENARIO_LAYERS[name]:
                accumulator.add(layer_number, rows)

    @torch.inference_mode()
    def finalize(self) -> dict[str, AttentionScenarioAggregate]:
        if self._seen_layers != set(range(1, 37)):
            raise MetricInputError("attention accumulation requires all 36 layers")
        return {
            name: accumulator.finalize()
            for name, accumulator in self._scenarios.items()
        }


class HiddenStateAccumulator:
    """Aggregate valid normalized layer cosines without retaining hidden tensors."""

    def __init__(self) -> None:
        self._seen_layers: set[int] = set()
        self._all_sum = 0.0
        self._all_count = 0
        self._early_sum = 0.0
        self._early_count = 0
        self._reasoning_sum = 0.0
        self._reasoning_count = 0
        self._last: float | None = None

    def add_layer(self, layer_number: int, cosine: float | None) -> None:
        if not 1 <= layer_number <= 36:
            raise MetricInputError("decoder layer number must be in 1..36")
        if layer_number in self._seen_layers:
            raise MetricInputError("decoder layer was accumulated more than once")
        self._seen_layers.add(layer_number)

        value = hidden_cosine_value(cosine)
        if value is None:
            return
        self._all_sum += value
        self._all_count += 1
        if layer_number <= 21:
            self._early_sum += value
            self._early_count += 1
        elif layer_number <= 34:
            self._reasoning_sum += value
            self._reasoning_count += 1
        if layer_number == 36:
            self._last = value

    def finalize(self) -> HiddenStateMetrics:
        if self._seen_layers != set(range(1, 37)):
            raise MetricInputError("hidden-state accumulation requires all 36 layers")

        return HiddenStateMetrics(
            valid_layers_all=self._all_count,
            valid_layers_early_visual_integration=self._early_count,
            valid_layers_visual_reasoning=self._reasoning_count,
            last_layer_valid=self._last is not None,
            cosine_gen_imgproto_hidden_avg_all_layers=(
                self._all_sum / self._all_count if self._all_count else None
            ),
            cosine_gen_imgproto_hidden_last_layer=self._last,
            cosine_gen_imgproto_hidden_early_visual_integration=(
                self._early_sum / self._early_count
                if self._early_count
                else None
            ),
            cosine_gen_imgproto_hidden_visual_reasoning=(
                self._reasoning_sum / self._reasoning_count
                if self._reasoning_count
                else None
            ),
        )
