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
    ATTENTION_EPSILON,
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


class _BatchedScenarioAttentionAccumulator:
    def __init__(self, expected_layers: frozenset[int]) -> None:
        self.expected_layers = expected_layers
        self.seen_layers: set[int] = set()
        self.vector_sum: torch.Tensor | None = None
        self.valid_rows: torch.Tensor | None = None
        self.head_count = 0

    def add(
        self,
        layer_number: int,
        layer_sum: torch.Tensor,
        valid_rows: torch.Tensor,
        head_count: int,
    ) -> None:
        self.seen_layers.add(layer_number)
        if self.vector_sum is None:
            self.vector_sum = layer_sum.clone()
            self.valid_rows = valid_rows.clone()
        elif self.vector_sum.shape != layer_sum.shape:
            raise MetricInputError(
                "attention key width changed within one forward pass"
            )
        else:
            self.vector_sum.add_(layer_sum)
            if self.valid_rows is None:
                raise MetricInputError("attention validity state is missing")
            self.valid_rows.logical_and_(valid_rows)
        self.head_count += head_count

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.seen_layers != self.expected_layers:
            raise MetricInputError("attention scenario is missing decoder layers")
        if (
            self.vector_sum is None
            or self.valid_rows is None
            or self.head_count == 0
        ):
            raise MetricInputError("attention scenario has no head contributions")
        return self.vector_sum / self.head_count, self.valid_rows


class BatchedAttentionAccumulator:
    """Accumulate predictor attention for a rollout microbatch on-device."""

    def __init__(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise MetricInputError("attention batch size must be positive")
        self.batch_size = batch_size
        self._seen_layers: set[int] = set()
        self._scenarios = {
            name: _BatchedScenarioAttentionAccumulator(
                ATTENTION_SCENARIO_LAYERS[name]
            )
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
        if not isinstance(predictor_attention_rows, torch.Tensor):
            raise MetricInputError("attention rows must be a torch.Tensor")
        if (
            predictor_attention_rows.ndim != 3
            or predictor_attention_rows.shape[0] != self.batch_size
            or predictor_attention_rows.shape[1] == 0
            or predictor_attention_rows.shape[2] == 0
        ):
            raise MetricInputError(
                "batched attention rows must have shape [batch, heads, keys]"
            )
        if not predictor_attention_rows.is_floating_point():
            raise MetricInputError("attention rows must have a floating-point dtype")
        self._seen_layers.add(layer_number)
        rows = predictor_attention_rows.detach().to(dtype=torch.float32)
        valid_rows = torch.isfinite(rows).all(dim=(1, 2)) & ~torch.any(
            rows < -ATTENTION_EPSILON,
            dim=(1, 2),
        )
        sanitized = torch.where(
            valid_rows[:, None, None],
            rows.clamp_min(0),
            torch.zeros_like(rows),
        )
        layer_sum = sanitized.sum(dim=1)
        for name, accumulator in self._scenarios.items():
            if layer_number in ATTENTION_SCENARIO_LAYERS[name]:
                accumulator.add(
                    layer_number,
                    layer_sum,
                    valid_rows,
                    rows.shape[1],
                )

    @torch.inference_mode()
    def finalize(
        self,
    ) -> tuple[dict[str, AttentionScenarioAggregate], ...]:
        if self._seen_layers != set(range(1, 37)):
            raise MetricInputError("attention accumulation requires all 36 layers")
        finalized = {
            name: accumulator.finalize()
            for name, accumulator in self._scenarios.items()
        }
        validity = torch.stack(
            [finalized[name][1] for name in ATTENTION_SCENARIOS]
        ).cpu().tolist()
        return tuple(
            {
                name: AttentionScenarioAggregate(
                    valid=bool(validity[scenario_index][row_index]),
                    vector=(
                        finalized[name][0][row_index]
                        if validity[scenario_index][row_index]
                        else None
                    ),
                )
                for scenario_index, name in enumerate(ATTENTION_SCENARIOS)
            }
            for row_index in range(self.batch_size)
        )


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
