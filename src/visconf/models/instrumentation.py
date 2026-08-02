"""Restorable row-aware online instrumentation for Qwen2.5-VL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from visconf.metrics.accumulator import (
    AttentionScenarioAggregate,
    BatchedAttentionAccumulator,
    HiddenStateAccumulator,
)
from visconf.metrics.hidden_state import HiddenStateMetrics, compute_layer_cosines_tensor
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


@dataclass(slots=True)
class _LayoutGroup:
    """Rows sharing one source prompt and therefore one logical layout."""

    source_prompt_index: int
    rows: tuple[int, ...]
    valid_positions: torch.Tensor
    prefill_query_index: int | None
    attention: BatchedAttentionAccumulator
    _device_rows: torch.Tensor | None = None

    def row_indices(self, device: torch.device) -> torch.Tensor:
        if self._device_rows is None or self._device_rows.device != device:
            self._device_rows = torch.tensor(
                self.rows,
                dtype=torch.long,
                device=device,
            )
        return self._device_rows

    def positions(self, device: torch.device) -> torch.Tensor:
        if self.valid_positions.device != device:
            self.valid_positions = self.valid_positions.to(device)
        return self.valid_positions


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
        prompt_groups: Sequence[TokenGroups],
        source_prompt_indices: Sequence[int],
        attention_mask: torch.Tensor | None,
    ) -> None:
        self.instrumentation = instrumentation
        self.mode = mode
        self.prompt_groups = tuple(prompt_groups)
        self.source_prompt_indices = tuple(int(value) for value in source_prompt_indices)
        self.batch_size = len(self.source_prompt_indices)
        if self.batch_size <= 0:
            raise InstrumentationError("instrumentation batch must be non-empty")
        if any(index < 0 or index >= len(self.prompt_groups) for index in self.source_prompt_indices):
            raise InstrumentationError("source prompt index is unavailable")
        if attention_mask is None:
            widths = [
                self.prompt_groups[index].prompt_token_count
                for index in self.source_prompt_indices
            ]
            if len(set(widths)) != 1:
                raise InstrumentationError("ragged prompts require an attention mask")
            attention_mask = torch.ones((self.batch_size, widths[0]), dtype=torch.long)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if attention_mask.ndim != 2 or attention_mask.shape[0] != self.batch_size:
            raise InstrumentationError("attention mask has an unexpected shape")
        self.physical_key_width = attention_mask.shape[1]
        rows_by_source: dict[int, list[int]] = {}
        for row, source_index in enumerate(self.source_prompt_indices):
            rows_by_source.setdefault(source_index, []).append(row)
        layout_groups = []
        for source_index, rows in rows_by_source.items():
            representative = rows[0]
            valid = torch.nonzero(
                attention_mask[representative] == 1,
                as_tuple=False,
            ).flatten()
            if valid.numel() == 0:
                raise InstrumentationError("instrumented step has no unmasked keys")
            layout_groups.append(
                _LayoutGroup(
                    source_prompt_index=source_index,
                    rows=tuple(rows),
                    valid_positions=valid,
                    prefill_query_index=(
                        int(valid[-1].item()) if mode == "prefill" else None
                    ),
                    attention=BatchedAttentionAccumulator(len(rows)),
                )
            )
        self.layout_groups = tuple(layout_groups)
        self.layer_calls: set[int] = set()
        self.hidden = [HiddenStateAccumulator() for _ in range(self.batch_size)]
        self.hidden_cosines: dict[int, torch.Tensor] = {}

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
        if attention_weights.ndim != 4 or attention_weights.shape[0] != self.batch_size:
            raise InstrumentationError("eager attention weights were not returned")

        prototypes_by_source: dict[int, torch.Tensor | None] = {}
        if self.mode == "prefill":
            prototypes: list[torch.Tensor | None] = [
                None for _ in self.prompt_groups
            ]
            for group in self.layout_groups:
                row = group.rows[0]
                source_index = group.source_prompt_index
                valid = group.positions(hidden_output.device)
                logical = self.prompt_groups[source_index].image_positions.to(
                    hidden_output.device
                )
                if logical.numel():
                    physical = valid.index_select(0, logical)
                    prototype = (
                        hidden_output[row]
                        .index_select(0, physical)
                        .to(dtype=torch.float32)
                        .mean(dim=0)
                        .detach()
                    )
                else:
                    prototype = None
                prototypes[source_index] = prototype
                prototypes_by_source[source_index] = prototype
            self.instrumentation._image_prototypes[layer_number] = tuple(
                prototypes
            )
        else:
            base = self.instrumentation._image_prototypes.get(layer_number)
            if base is None:
                raise InstrumentationError("image prototypes are not initialized")
            prototypes_by_source = {
                group.source_prompt_index: base[group.source_prompt_index]
                for group in self.layout_groups
            }

        layer_cosines = torch.full(
            (self.batch_size,),
            torch.nan,
            dtype=torch.float32,
            device=hidden_output.device,
        )
        cosine_hidden: list[torch.Tensor] = []
        cosine_prototypes: list[torch.Tensor] = []
        cosine_rows: list[torch.Tensor] = []
        for group in self.layout_groups:
            rows = group.row_indices(hidden_output.device)
            valid = group.positions(hidden_output.device)
            query_index = (
                group.prefill_query_index
                if self.mode == "prefill"
                else hidden_output.shape[1] - 1
            )
            if query_index is None:
                raise InstrumentationError("predictor query position is unavailable")
            if query_index >= hidden_output.shape[1] or query_index >= attention_weights.shape[-2]:
                raise InstrumentationError("predictor query position is unavailable")
            if self.physical_key_width > attention_weights.shape[-1]:
                raise InstrumentationError("unmasked key position is unavailable")
            selected = (
                attention_weights.index_select(0, rows)
                [:, :, query_index, :]
                .index_select(2, valid)
            )
            group.attention.add_layer(
                layer_number,
                selected.to(dtype=torch.float32),
            )
            prototype = prototypes_by_source[group.source_prompt_index]
            if prototype is not None:
                predictor = hidden_output.index_select(0, rows)[
                    :, query_index, :
                ]
                cosine_hidden.append(predictor)
                cosine_prototypes.append(
                    prototype.unsqueeze(0).expand(predictor.shape[0], -1)
                )
                cosine_rows.append(rows)
        if cosine_hidden:
            layer_cosines.index_copy_(
                0,
                torch.cat(cosine_rows),
                compute_layer_cosines_tensor(
                    torch.cat(cosine_hidden),
                    torch.cat(cosine_prototypes),
                ),
            )
        self.hidden_cosines[layer_number] = layer_cosines
        self.layer_calls.add(layer_number)

    def finalize(self) -> StepObservations:
        if self.layer_calls != set(range(1, 37)):
            raise InstrumentationError("forward pass did not call all 36 decoder layers")
        attention: list[dict[str, AttentionScenarioAggregate] | None] = [
            None for _ in range(self.batch_size)
        ]
        for group in self.layout_groups:
            values = group.attention.finalize()
            for row, value in zip(group.rows, values, strict=True):
                attention[row] = value
        if any(value is None for value in attention):
            raise InstrumentationError("instrumented attention rows are incomplete")
        hidden_values = torch.stack(
            [self.hidden_cosines[layer] for layer in range(1, 37)],
            dim=1,
        ).cpu().tolist()
        for row_index, values in enumerate(hidden_values):
            for layer_number, value in enumerate(values, start=1):
                self.hidden[row_index].add_layer(layer_number, value)
        return StepObservations(
            rows=tuple(
                RowStepObservation(
                    attention=attention[index],  # type: ignore[arg-type]
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
        self._image_prototypes: dict[int, tuple[torch.Tensor | None, ...]] = {}
        self._entered = False

    def __enter__(self) -> "QwenInstrumentation":
        if self._entered:
            raise InstrumentationError("instrumentation is already active")
        self.layers = discover_decoder_layers(self.model)
        self._original_forwards = tuple(layer.forward for layer in self.layers)
        for layer_number, (layer, original) in enumerate(
            zip(self.layers, self._original_forwards, strict=True), start=1
        ):
            layer.forward = self._patched_forward(layer_number, original)
        self._entered = True
        return self

    def _patched_forward(self, layer_number: int, original: Any):
        def patched(*args: Any, **kwargs: Any):
            collector = self._collector
            if collector is None:
                return original(*args, **kwargs)
            requested_attention = bool(kwargs.get("output_attentions", False))
            kwargs["output_attentions"] = True
            output = original(*args, **kwargs)
            if not isinstance(output, tuple) or len(output) < 2:
                raise InstrumentationError("decoder layer did not return eager attention")
            collector.consume(layer_number, output[0], output[1])
            return output if requested_attention else (output[0], *output[2:])
        return patched

    def begin_prefill(
        self,
        prompt_groups: TokenGroups | Sequence[TokenGroups],
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        self._require_entered()
        groups = (
            (prompt_groups,)
            if isinstance(prompt_groups, TokenGroups)
            else tuple(prompt_groups)
        )
        self._image_prototypes = {}
        self._collector = _StepCollector(
            self,
            "prefill",
            groups,
            tuple(range(len(groups))),
            attention_mask,
        )

    def begin_decode(
        self,
        prompt_groups: TokenGroups | Sequence[TokenGroups],
        batch_size: int,
        attention_mask: torch.Tensor | None = None,
        source_prompt_indices: Sequence[int] | None = None,
    ) -> None:
        self._require_entered()
        if set(self._image_prototypes) != set(range(1, 37)):
            raise InstrumentationError("image prototypes are not initialized")
        groups = (
            (prompt_groups,)
            if isinstance(prompt_groups, TokenGroups)
            else tuple(prompt_groups)
        )
        sources = (
            tuple(0 for _ in range(batch_size))
            if source_prompt_indices is None
            else tuple(source_prompt_indices)
        )
        if len(sources) != batch_size:
            raise InstrumentationError("decode source mapping differs from batch")
        self._collector = _StepCollector(
            self,
            "decode",
            groups,
            sources,
            attention_mask,
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
        if any(len(values) != 1 for values in self._image_prototypes.values()):
            raise InstrumentationError("use image_prototypes_by_prompt for a prompt batch")
        return tuple(self._image_prototypes[layer][0] for layer in range(1, 37))

    @property
    def image_prototypes_by_prompt(
        self,
    ) -> tuple[tuple[torch.Tensor | None, ...], ...]:
        if set(self._image_prototypes) != set(range(1, 37)):
            raise InstrumentationError("image prototypes are not initialized")
        count = len(self._image_prototypes[1])
        return tuple(
            tuple(self._image_prototypes[layer][row] for layer in range(1, 37))
            for row in range(count)
        )

    def _require_entered(self) -> None:
        if not self._entered:
            raise InstrumentationError("instrumentation context is not active")
        if self._collector is not None:
            raise InstrumentationError("another instrumented step is active")

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._collector = None
        for layer, original in zip(self.layers, self._original_forwards, strict=True):
            layer.forward = original
        self._image_prototypes.clear()
        self._original_forwards = ()
        self.layers = ()
        self._entered = False
