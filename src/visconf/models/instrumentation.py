"""Restorable online instrumentation for Qwen2.5-VL decoder layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from visconf.metrics.accumulator import (
    AttentionAccumulator,
    AttentionScenarioAggregate,
    HiddenStateAccumulator,
)
from visconf.metrics.hidden_state import (
    HiddenStateMetrics,
    compute_layer_cosines,
)
from visconf.types import TokenGroups


class InstrumentationError(RuntimeError):
    """Raised when decoder instrumentation misses its alignment contract."""


@dataclass(frozen=True, slots=True)
class RowStepObservation:
    attention: dict[str, AttentionScenarioAggregate]
    hidden_state: HiddenStateMetrics


@dataclass(frozen=True, slots=True)
class StepObservations:
    rows: tuple[RowStepObservation, ...]


def discover_decoder_layers(model: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    for path in (
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
    ):
        value: Any = model
        for name in path:
            value = getattr(value, name, None)
            if value is None:
                break
        else:
            layers = tuple(value)
            if len(layers) != 36:
                raise InstrumentationError(
                    f"expected 36 decoder layers, found {len(layers)}"
                )
            return layers
    raise InstrumentationError("cannot locate Qwen language decoder layers")


class _StepCollector:
    def __init__(
        self,
        instrumentation: "QwenInstrumentation",
        mode: str,
        prompt_groups: TokenGroups,
        batch_size: int,
        attention_mask: torch.Tensor | None,
    ) -> None:
        self.instrumentation = instrumentation
        self.mode = mode
        self.prompt_groups = prompt_groups
        self.batch_size = batch_size
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, prompt_groups.prompt_token_count),
                dtype=torch.long,
            )
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if attention_mask.ndim != 2 or attention_mask.shape[0] != batch_size:
            raise InstrumentationError("attention mask has an unexpected shape")
        self.attention_mask = attention_mask
        self.valid_positions = tuple(
            torch.nonzero(attention_mask[row] == 1, as_tuple=False).flatten()
            for row in range(batch_size)
        )
        if any(positions.numel() == 0 for positions in self.valid_positions):
            raise InstrumentationError("instrumented step has no unmasked keys")
        self.prefill_query_index = (
            int(self.valid_positions[0][-1].item())
            if mode == "prefill"
            else None
        )
        self.layer_calls: set[int] = set()
        self.attention = [AttentionAccumulator() for _ in range(batch_size)]
        self.hidden = [HiddenStateAccumulator() for _ in range(batch_size)]

    @torch.inference_mode()
    def consume(
        self,
        layer_number: int,
        hidden_output: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> None:
        if layer_number in self.layer_calls:
            raise InstrumentationError("decoder layer called twice in one step")
        if hidden_output.ndim != 3 or hidden_output.shape[0] != self.batch_size:
            raise InstrumentationError("unexpected decoder hidden-state shape")
        if (
            attention_weights.ndim != 4
            or attention_weights.shape[0] != self.batch_size
        ):
            raise InstrumentationError("eager attention weights were not returned")

        if self.valid_positions[0].device != hidden_output.device:
            self.valid_positions = tuple(
                positions.to(hidden_output.device)
                for positions in self.valid_positions
            )
        query_index = (
            self.prefill_query_index
            if self.mode == "prefill"
            else hidden_output.shape[1] - 1
        )
        if query_index is None:
            raise InstrumentationError("predictor query position is unavailable")
        if (
            query_index >= hidden_output.shape[1]
            or query_index >= attention_weights.shape[-2]
        ):
            raise InstrumentationError("predictor query position is unavailable")

        image_prototype: torch.Tensor | None
        if self.mode == "prefill":
            logical_positions = self.prompt_groups.image_positions.to(
                hidden_output.device
            )
            if logical_positions.numel():
                positions = self.valid_positions[0].index_select(
                    0, logical_positions
                )
                image_prototype = (
                    hidden_output[0]
                    .index_select(0, positions)
                    .to(dtype=torch.float32)
                    .mean(dim=0)
                    .detach()
                )
            else:
                image_prototype = None
            self.instrumentation._image_prototypes[layer_number] = (
                image_prototype
            )
        else:
            image_prototype = self.instrumentation._image_prototypes[
                layer_number
            ]

        cosines = compute_layer_cosines(
            hidden_output[:, query_index, :],
            image_prototype,
        )
        for row_index, cosine in enumerate(cosines):
            key_positions = self.valid_positions[row_index]
            if key_positions.numel() > attention_weights.shape[-1]:
                raise InstrumentationError("unmasked key position is unavailable")
            selected_attention = attention_weights[
                row_index,
                :,
                query_index,
                :,
            ].index_select(1, key_positions)
            self.attention[row_index].add_layer(
                layer_number,
                selected_attention.to(dtype=torch.float32),
            )
            self.hidden[row_index].add_layer(layer_number, cosine)
        self.layer_calls.add(layer_number)

    def finalize(self) -> StepObservations:
        if self.layer_calls != set(range(1, 37)):
            raise InstrumentationError(
                "forward pass did not call all 36 decoder layers"
            )
        return StepObservations(
            rows=tuple(
                RowStepObservation(
                    attention=self.attention[index].finalize(),
                    hidden_state=self.hidden[index].finalize(),
                )
                for index in range(self.batch_size)
            )
        )


class QwenInstrumentation:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.layers: tuple[torch.nn.Module, ...] = ()
        self._original_forwards: tuple[Any, ...] = ()
        self._collector: _StepCollector | None = None
        self._image_prototypes: dict[int, torch.Tensor | None] = {}
        self._entered = False

    def __enter__(self) -> "QwenInstrumentation":
        if self._entered:
            raise InstrumentationError("instrumentation is already active")
        self.layers = discover_decoder_layers(self.model)
        self._original_forwards = tuple(layer.forward for layer in self.layers)
        for layer_number, (layer, original) in enumerate(
            zip(self.layers, self._original_forwards, strict=True),
            start=1,
        ):
            layer.forward = self._patched_forward(layer_number, original)
        self._entered = True
        return self

    def _patched_forward(self, layer_number: int, original: Any):
        def patched(*args: Any, **kwargs: Any):
            collector = self._collector
            if collector is None:
                return original(*args, **kwargs)

            requested_attention = bool(
                kwargs.get("output_attentions", False)
            )
            kwargs["output_attentions"] = True
            output = original(*args, **kwargs)
            if not isinstance(output, tuple) or len(output) < 2:
                raise InstrumentationError(
                    "decoder layer did not return eager attention"
                )
            collector.consume(layer_number, output[0], output[1])
            if requested_attention:
                return output
            return (output[0], *output[2:])

        return patched

    def begin_prefill(
        self,
        prompt_groups: TokenGroups,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        self._require_entered()
        self._image_prototypes = {}
        self._collector = _StepCollector(
            self,
            mode="prefill",
            prompt_groups=prompt_groups,
            batch_size=1,
            attention_mask=attention_mask,
        )

    def begin_decode(
        self,
        prompt_groups: TokenGroups,
        batch_size: int,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        self._require_entered()
        if set(self._image_prototypes) != set(range(1, 37)):
            raise InstrumentationError("image prototypes are not initialized")
        self._collector = _StepCollector(
            self,
            mode="decode",
            prompt_groups=prompt_groups,
            batch_size=batch_size,
            attention_mask=attention_mask,
        )

    def finish_step(self) -> StepObservations:
        if self._collector is None:
            raise InstrumentationError("no instrumented step is active")
        collector = self._collector
        self._collector = None
        return collector.finalize()

    def cancel_step(self) -> None:
        self._collector = None

    @property
    def image_prototypes(self) -> tuple[torch.Tensor | None, ...]:
        if set(self._image_prototypes) != set(range(1, 37)):
            raise InstrumentationError("image prototypes are not initialized")
        return tuple(
            self._image_prototypes[layer]
            for layer in range(1, 37)
        )

    def _require_entered(self) -> None:
        if not self._entered:
            raise InstrumentationError(
                "instrumentation context is not active"
            )
        if self._collector is not None:
            raise InstrumentationError("another instrumented step is active")

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._collector = None
        for layer, original in zip(
            self.layers,
            self._original_forwards,
            strict=True,
        ):
            layer.forward = original
        self._image_prototypes.clear()
        self._original_forwards = ()
        self.layers = ()
        self._entered = False
