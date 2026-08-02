"""Visual hidden-state/image-prototype cosine metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from visconf.metrics.validation import (
    HIDDEN_NORM_EPSILON,
    MetricInputError,
    hidden_cosine_value,
)


HIDDEN_STATE_METRICS = (
    "cosine_gen_imgproto_hidden_avg_all_layers",
    "cosine_gen_imgproto_hidden_last_layer",
    "cosine_gen_imgproto_hidden_early_visual_integration",
    "cosine_gen_imgproto_hidden_visual_reasoning",
)


@dataclass(frozen=True, slots=True)
class HiddenStateMetrics:
    valid_layers_all: int
    valid_layers_early_visual_integration: int
    valid_layers_visual_reasoning: int
    last_layer_valid: bool
    cosine_gen_imgproto_hidden_avg_all_layers: float | None
    cosine_gen_imgproto_hidden_last_layer: float | None
    cosine_gen_imgproto_hidden_early_visual_integration: float | None
    cosine_gen_imgproto_hidden_visual_reasoning: float | None


def _average(values: Sequence[float | None]) -> tuple[float | None, int]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None, 0
    return sum(valid) / len(valid), len(valid)


@torch.inference_mode()
def compute_layer_cosine(
    predictor_hidden: torch.Tensor,
    image_prototype: torch.Tensor | None,
) -> float | None:
    """Return one normalized layer cosine, or None when it is undefined."""

    if not isinstance(predictor_hidden, torch.Tensor):
        raise MetricInputError("hidden states must be torch.Tensor values")
    if predictor_hidden.ndim != 1:
        raise MetricInputError(
            "predictor hidden state and image prototype must be equal-length vectors"
        )
    return compute_layer_cosines(
        predictor_hidden.unsqueeze(0),
        image_prototype,
    )[0]


@torch.inference_mode()
def compute_layer_cosines(
    predictor_hidden: torch.Tensor,
    image_prototype: torch.Tensor | None,
) -> tuple[float | None, ...]:
    """Return normalized layer cosines for a predictor microbatch."""

    values = compute_layer_cosines_tensor(
        predictor_hidden,
        image_prototype,
    ).cpu().tolist()
    return tuple(
        float(value) if math.isfinite(value) else None
        for value in values
    )


@torch.inference_mode()
def compute_layer_cosines_tensor(
    predictor_hidden: torch.Tensor,
    image_prototype: torch.Tensor | None,
) -> torch.Tensor:
    """Return device-resident normalized cosines, using NaN when undefined.

    A single prototype is broadcast across the batch. A two-dimensional
    prototype tensor supplies one prototype per predictor row.
    """

    if not isinstance(predictor_hidden, torch.Tensor):
        raise MetricInputError("hidden states must be torch.Tensor values")
    if predictor_hidden.ndim != 2:
        raise MetricInputError("predictor hidden states must have shape [batch, hidden]")
    if image_prototype is None:
        return torch.full(
            (predictor_hidden.shape[0],),
            torch.nan,
            device=predictor_hidden.device,
            dtype=torch.float32,
        )
    if not isinstance(image_prototype, torch.Tensor):
        raise MetricInputError("hidden states must be torch.Tensor values")
    if image_prototype.ndim == 1:
        if predictor_hidden.shape[1:] != image_prototype.shape:
            raise MetricInputError(
                "predictor hidden state and image prototype must be equal-length vectors"
            )
    elif image_prototype.ndim == 2:
        if predictor_hidden.shape != image_prototype.shape:
            raise MetricInputError(
                "batched image prototypes must match predictor hidden states"
            )
    else:
        raise MetricInputError(
            "predictor hidden state and image prototype must be equal-length vectors"
        )
    if not predictor_hidden.is_floating_point() or not image_prototype.is_floating_point():
        raise MetricInputError("hidden states must have floating-point dtypes")

    predictor = predictor_hidden.detach().to(dtype=torch.float32)
    prototype = image_prototype.detach().to(
        device=predictor.device,
        dtype=torch.float32,
    )
    if prototype.ndim == 1:
        prototype = prototype.unsqueeze(0).expand_as(predictor)
    predictor_norm = torch.linalg.vector_norm(predictor, dim=-1)
    prototype_norm = torch.linalg.vector_norm(prototype, dim=-1)
    valid = (
        torch.isfinite(predictor).all(dim=-1)
        & torch.isfinite(prototype).all(dim=-1)
        & (predictor_norm >= HIDDEN_NORM_EPSILON)
        & (prototype_norm >= HIDDEN_NORM_EPSILON)
    )
    denominator = (predictor_norm * prototype_norm).clamp_min(
        HIDDEN_NORM_EPSILON
    )
    cosine = torch.sum(predictor * prototype, dim=-1) / denominator
    normalized = torch.clamp((cosine + 1) / 2, min=0, max=1)
    return torch.where(
        valid,
        normalized,
        torch.full_like(normalized, torch.nan),
    )


@torch.inference_mode()
def aggregate_hidden_metrics(
    per_layer_cosines: Sequence[float | None],
) -> HiddenStateMetrics:
    """Aggregate 36 normalized per-layer cosines over the fixed ranges."""

    if len(per_layer_cosines) != 36:
        raise MetricInputError("per_layer_cosines must contain exactly 36 values")

    values = tuple(hidden_cosine_value(value) for value in per_layer_cosines)
    all_average, all_count = _average(values)
    early_average, early_count = _average(values[:21])
    reasoning_average, reasoning_count = _average(values[21:34])
    last = values[35]

    return HiddenStateMetrics(
        valid_layers_all=all_count,
        valid_layers_early_visual_integration=early_count,
        valid_layers_visual_reasoning=reasoning_count,
        last_layer_valid=last is not None,
        cosine_gen_imgproto_hidden_avg_all_layers=all_average,
        cosine_gen_imgproto_hidden_last_layer=last,
        cosine_gen_imgproto_hidden_early_visual_integration=early_average,
        cosine_gen_imgproto_hidden_visual_reasoning=reasoning_average,
    )
