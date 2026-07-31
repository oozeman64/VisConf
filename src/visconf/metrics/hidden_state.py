"""Visual hidden-state/image-prototype cosine metrics."""

from __future__ import annotations

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

    if image_prototype is None:
        return None
    if not isinstance(predictor_hidden, torch.Tensor) or not isinstance(
        image_prototype, torch.Tensor
    ):
        raise MetricInputError("hidden states must be torch.Tensor values")
    if (
        predictor_hidden.ndim != 1
        or image_prototype.ndim != 1
        or predictor_hidden.shape != image_prototype.shape
    ):
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
    if not torch.isfinite(predictor).all() or not torch.isfinite(prototype).all():
        return None

    predictor_norm = torch.linalg.vector_norm(predictor)
    prototype_norm = torch.linalg.vector_norm(prototype)
    if (
        predictor_norm.item() < HIDDEN_NORM_EPSILON
        or prototype_norm.item() < HIDDEN_NORM_EPSILON
    ):
        return None

    cosine = torch.dot(predictor, prototype) / (
        predictor_norm * prototype_norm
    )
    normalized = torch.clamp((cosine + 1) / 2, min=0, max=1)
    return float(normalized.item())


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
