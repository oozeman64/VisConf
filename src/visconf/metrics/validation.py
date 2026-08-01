"""Shared metric input validation."""

from __future__ import annotations

import math

import torch


ATTENTION_EPSILON = 1e-8
HIDDEN_NORM_EPSILON = 1e-8


class MetricInputError(ValueError):
    """Raised when a metric input violates its structural contract."""


def probability_logits(
    raw_logits: torch.Tensor,
    selected_token_id: int,
) -> torch.Tensor:
    if not isinstance(raw_logits, torch.Tensor):
        raise MetricInputError("raw_logits must be a torch.Tensor")
    if raw_logits.ndim != 1:
        raise MetricInputError("raw_logits must be one-dimensional")
    if raw_logits.numel() < 2:
        raise MetricInputError("raw_logits must contain at least two tokens")
    if not raw_logits.is_floating_point():
        raise MetricInputError("raw_logits must have a floating-point dtype")
    if not 0 <= selected_token_id < raw_logits.numel():
        raise MetricInputError("selected_token_id is outside the vocabulary")

    logits = raw_logits.detach().to(dtype=torch.float32)
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise MetricInputError("raw_logits contain NaN or positive infinity")
    if torch.isneginf(logits).all():
        raise MetricInputError("raw_logits cannot all be negative infinity")
    return logits


def probability_logits_only_batch(
    raw_logits: torch.Tensor,
) -> torch.Tensor:
    """Validate and normalize a probability-logit microbatch."""

    if not isinstance(raw_logits, torch.Tensor):
        raise MetricInputError("raw_logits must be a torch.Tensor")
    if raw_logits.ndim != 2:
        raise MetricInputError("raw_logits must have shape [batch, vocabulary]")
    if raw_logits.shape[0] == 0 or raw_logits.shape[1] < 2:
        raise MetricInputError("raw_logits batch is empty or vocabulary is too small")
    if not raw_logits.is_floating_point():
        raise MetricInputError("raw_logits must have a floating-point dtype")

    logits = raw_logits.detach().to(dtype=torch.float32)
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise MetricInputError("raw_logits contain NaN or positive infinity")
    if torch.isneginf(logits).all(dim=-1).any():
        raise MetricInputError("raw_logits cannot all be negative infinity")
    return logits


def probability_logits_batch(
    raw_logits: torch.Tensor,
    selected_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = probability_logits_only_batch(raw_logits)
    if not isinstance(selected_token_ids, torch.Tensor):
        raise MetricInputError("selected_token_ids must be a torch.Tensor")
    selected = selected_token_ids.to(
        device=logits.device,
        dtype=torch.long,
    ).reshape(-1)
    if selected.shape[0] != logits.shape[0]:
        raise MetricInputError("selected token batch differs from logits")
    if torch.any(selected < 0) or torch.any(selected >= logits.shape[1]):
        raise MetricInputError("selected_token_ids are outside the vocabulary")
    return logits, selected


def attention_vector(
    value: torch.Tensor,
) -> tuple[torch.Tensor | None, str | None]:
    if not isinstance(value, torch.Tensor):
        raise MetricInputError("scenario_vector must be a torch.Tensor")
    if value.ndim != 1 or value.numel() == 0:
        raise MetricInputError("scenario_vector must be a non-empty vector")
    if not value.is_floating_point():
        raise MetricInputError("scenario_vector must have a floating-point dtype")

    vector = value.detach().to(dtype=torch.float32)
    if not torch.isfinite(vector).all():
        return None, "non_finite_attention"
    if torch.any(vector < -ATTENTION_EPSILON):
        return None, "negative_attention"
    if torch.any(vector < 0):
        vector = vector.clamp_min(0)
    return vector, None


def attention_rows(
    value: torch.Tensor,
) -> tuple[torch.Tensor | None, str | None]:
    if not isinstance(value, torch.Tensor):
        raise MetricInputError("attention rows must be a torch.Tensor")
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise MetricInputError("attention rows must have shape [heads, keys]")
    if not value.is_floating_point():
        raise MetricInputError("attention rows must have a floating-point dtype")

    rows = value.detach().to(dtype=torch.float32)
    if not torch.isfinite(rows).all():
        return None, "non_finite_attention"
    if torch.any(rows < -ATTENTION_EPSILON):
        return None, "negative_attention"
    if torch.any(rows < 0):
        rows = rows.clamp_min(0)
    return rows, None


def hidden_cosine_value(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if number < -ATTENTION_EPSILON or number > 1 + ATTENTION_EPSILON:
        return None
    return min(1.0, max(0.0, number))
