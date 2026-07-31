"""Temperature, top-k, and top-p token selection."""

from __future__ import annotations

import torch

from visconf.types import SamplingConfig


class SamplingError(ValueError):
    """Raised when sampling inputs violate the decoding contract."""


def _validate(
    raw_logits: torch.Tensor,
    config: SamplingConfig,
) -> None:
    if not isinstance(raw_logits, torch.Tensor):
        raise SamplingError("raw_logits must be a torch.Tensor")
    if raw_logits.ndim != 1 or raw_logits.numel() == 0:
        raise SamplingError("raw_logits must be a non-empty vector")
    if not raw_logits.is_floating_point():
        raise SamplingError("raw_logits must have a floating-point dtype")
    if torch.isnan(raw_logits).any() or torch.isposinf(raw_logits).any():
        raise SamplingError("raw_logits contain NaN or positive infinity")
    if torch.isneginf(raw_logits).all():
        raise SamplingError("raw_logits cannot all be negative infinity")
    if config.temperature <= 0:
        raise SamplingError("temperature must be positive")
    if not 0 < config.top_p <= 1:
        raise SamplingError("top_p must be in (0, 1]")
    if config.top_k is not None and config.top_k <= 0:
        raise SamplingError("top_k must be positive when specified")
    if config.repetition_penalty != 1.0:
        raise SamplingError("repetition_penalty must equal 1.0")


@torch.inference_mode()
def apply_sampling_transforms(
    raw_logits: torch.Tensor,
    config: SamplingConfig,
) -> torch.Tensor:
    """Create selection logits without mutating the model's raw logits."""

    _validate(raw_logits, config)
    selection_logits = raw_logits.detach().to(dtype=torch.float32).clone()
    selection_logits.div_(config.temperature)

    if config.top_k is not None and config.top_k < selection_logits.numel():
        keep = torch.topk(selection_logits, k=config.top_k).indices
        remove = torch.ones_like(selection_logits, dtype=torch.bool)
        remove[keep] = False
        selection_logits.masked_fill_(remove, -torch.inf)

    if config.top_p < 1:
        sorted_logits, sorted_indices = torch.sort(
            selection_logits,
            descending=True,
        )
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        remove_sorted = torch.cumsum(sorted_probabilities, dim=-1) > config.top_p
        remove_sorted[1:] = remove_sorted[:-1].clone()
        remove_sorted[0] = False
        remove = torch.zeros_like(remove_sorted)
        remove.scatter_(0, sorted_indices, remove_sorted)
        selection_logits.masked_fill_(remove, -torch.inf)

    return selection_logits


@torch.inference_mode()
def sample_next_token(
    raw_logits: torch.Tensor,
    config: SamplingConfig,
    generator: torch.Generator,
) -> int:
    """Sample one token with the rollout's independent generator."""

    selection_logits = apply_sampling_transforms(raw_logits, config)
    probabilities = torch.softmax(selection_logits, dim=-1)
    return int(
        torch.multinomial(probabilities, 1, generator=generator).item()
    )
