"""Temperature, top-k, and top-p token selection."""

from __future__ import annotations

import torch

from visconf.types import SamplingConfig


class SamplingError(ValueError):
    """Raised when sampling inputs violate the decoding contract."""


def _validate_config(config: SamplingConfig) -> None:
    if config.temperature <= 0:
        raise SamplingError("temperature must be positive")
    if not 0 < config.top_p <= 1:
        raise SamplingError("top_p must be in (0, 1]")
    if config.top_k is not None and config.top_k <= 0:
        raise SamplingError("top_k must be positive when specified")
    if config.repetition_penalty != 1.0:
        raise SamplingError("repetition_penalty must equal 1.0")


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
    _validate_config(config)


def _validate_batch(
    raw_logits: torch.Tensor,
    config: SamplingConfig,
) -> None:
    if not isinstance(raw_logits, torch.Tensor):
        raise SamplingError("raw_logits must be a torch.Tensor")
    if raw_logits.ndim != 2 or 0 in raw_logits.shape:
        raise SamplingError("raw_logits must have shape [batch, vocabulary]")
    if not raw_logits.is_floating_point():
        raise SamplingError("raw_logits must have a floating-point dtype")
    if torch.isnan(raw_logits).any() or torch.isposinf(raw_logits).any():
        raise SamplingError("raw_logits contain NaN or positive infinity")
    if torch.isneginf(raw_logits).all(dim=-1).any():
        raise SamplingError("raw_logits cannot all be negative infinity")
    _validate_config(config)


def _validate_prepared_batch(
    raw_logits: torch.Tensor,
    sorted_indices: torch.Tensor,
    config: SamplingConfig,
) -> None:
    if not isinstance(raw_logits, torch.Tensor):
        raise SamplingError("raw_logits must be a torch.Tensor")
    if raw_logits.ndim != 2 or 0 in raw_logits.shape:
        raise SamplingError("raw_logits must have shape [batch, vocabulary]")
    if not raw_logits.is_floating_point():
        raise SamplingError("raw_logits must have a floating-point dtype")
    if not isinstance(sorted_indices, torch.Tensor):
        raise SamplingError("sorted_indices must be a torch.Tensor")
    if sorted_indices.shape != raw_logits.shape:
        raise SamplingError("sorted_indices shape differs from logits")
    if sorted_indices.device != raw_logits.device:
        raise SamplingError("sorted_indices device differs from logits")
    _validate_config(config)


@torch.inference_mode()
def apply_sampling_transforms_batch(
    raw_logits: torch.Tensor,
    config: SamplingConfig,
) -> torch.Tensor:
    """Transform a logits microbatch without mutating the raw logits."""

    _validate_batch(raw_logits, config)
    selection_logits = raw_logits.detach().to(dtype=torch.float32).clone()
    selection_logits.div_(config.temperature)

    if config.top_k is not None and config.top_k < selection_logits.shape[1]:
        keep = torch.topk(
            selection_logits,
            k=config.top_k,
            dim=-1,
        ).indices
        remove = torch.ones_like(selection_logits, dtype=torch.bool)
        remove.scatter_(1, keep, False)
        selection_logits.masked_fill_(remove, -torch.inf)

    if config.top_p < 1:
        sorted_logits, sorted_indices = torch.sort(
            selection_logits,
            dim=-1,
            descending=True,
        )
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        remove_sorted = (
            torch.cumsum(sorted_probabilities, dim=-1) >= config.top_p
        )
        remove_sorted[:, 1:] = remove_sorted[:, :-1].clone()
        remove_sorted[:, 0] = False
        remove = torch.zeros_like(remove_sorted)
        remove.scatter_(1, sorted_indices, remove_sorted)
        selection_logits.masked_fill_(remove, -torch.inf)

    return selection_logits


@torch.inference_mode()
def apply_sampling_transforms_prepared(
    raw_logits: torch.Tensor,
    sorted_indices: torch.Tensor,
    config: SamplingConfig,
) -> torch.Tensor:
    """Transform logits while reusing a validated vocabulary ordering."""

    _validate_prepared_batch(raw_logits, sorted_indices, config)
    selection_logits = raw_logits.clone()
    selection_logits.div_(config.temperature)
    vocabulary_size = selection_logits.shape[1]

    if config.top_k is not None and config.top_k < vocabulary_size:
        topk = torch.topk(selection_logits, k=config.top_k, dim=-1)
        remove = torch.ones_like(selection_logits, dtype=torch.bool)
        remove.scatter_(1, topk.indices, False)
        if config.top_p < 1:
            index_order = torch.argsort(topk.indices, dim=-1)
            candidate_indices = topk.indices.gather(1, index_order)
            candidate_logits = selection_logits.gather(
                1,
                candidate_indices,
            )
            value_order = torch.argsort(
                candidate_logits,
                dim=-1,
                descending=True,
                stable=True,
            )
            candidate_indices = candidate_indices.gather(1, value_order)
            candidate_logits = candidate_logits.gather(1, value_order)
            sorted_selection_logits = torch.full_like(
                selection_logits,
                -torch.inf,
            )
            sorted_selection_logits[:, : config.top_k] = candidate_logits
            sorted_probabilities = torch.softmax(
                sorted_selection_logits,
                dim=-1,
            )
            remove_sorted = (
                torch.cumsum(sorted_probabilities, dim=-1) >= config.top_p
            )
            remove_sorted[:, 1:] = remove_sorted[:, :-1].clone()
            remove_sorted[:, 0] = False
            remove.scatter_(
                1,
                candidate_indices,
                remove_sorted[:, : config.top_k],
            )
        selection_logits.masked_fill_(remove, -torch.inf)
    elif config.top_p < 1:
        sorted_selection_logits = selection_logits.gather(
            1,
            sorted_indices,
        )
        sorted_probabilities = torch.softmax(
            sorted_selection_logits,
            dim=-1,
        )
        remove_sorted = (
            torch.cumsum(sorted_probabilities, dim=-1) >= config.top_p
        )
        remove_sorted[:, 1:] = remove_sorted[:, :-1].clone()
        remove_sorted[:, 0] = False
        remove = torch.zeros_like(remove_sorted)
        remove.scatter_(1, sorted_indices, remove_sorted)
        selection_logits.masked_fill_(remove, -torch.inf)

    return selection_logits


@torch.inference_mode()
def apply_sampling_transforms(
    raw_logits: torch.Tensor,
    config: SamplingConfig,
) -> torch.Tensor:
    """Create selection logits without mutating the model's raw logits."""

    _validate(raw_logits, config)
    return apply_sampling_transforms_batch(
        raw_logits.unsqueeze(0),
        config,
    )[0]


@torch.inference_mode()
def sample_next_tokens(
    raw_logits: torch.Tensor,
    config: SamplingConfig,
    generators: tuple[torch.Generator, ...],
) -> tuple[int, ...]:
    """Sample a microbatch while preserving independent RNG streams."""

    if raw_logits.ndim != 2 or len(generators) != raw_logits.shape[0]:
        raise SamplingError("generator batch differs from logits")
    selection_logits = apply_sampling_transforms_batch(raw_logits, config)
    probabilities = torch.softmax(selection_logits, dim=-1)
    sampled = [
        torch.multinomial(
            probabilities[index],
            1,
            generator=generator,
        )
        for index, generator in enumerate(generators)
    ]
    return tuple(int(value) for value in torch.cat(sampled).tolist())


@torch.inference_mode()
def sample_next_tokens_prepared(
    raw_logits: torch.Tensor,
    sorted_indices: torch.Tensor,
    config: SamplingConfig,
    generators: tuple[torch.Generator, ...],
) -> tuple[int, ...]:
    """Sample using a vocabulary ordering shared with probability metrics."""

    if raw_logits.ndim != 2 or len(generators) != raw_logits.shape[0]:
        raise SamplingError("generator batch differs from logits")
    selection_logits = apply_sampling_transforms_prepared(
        raw_logits,
        sorted_indices,
        config,
    )
    probabilities = torch.softmax(selection_logits, dim=-1)
    sampled = [
        torch.multinomial(
            probabilities[index],
            1,
            generator=generator,
        )
        for index, generator in enumerate(generators)
    ]
    return tuple(int(value) for value in torch.cat(sampled).tolist())


@torch.inference_mode()
def sample_next_token(
    raw_logits: torch.Tensor,
    config: SamplingConfig,
    generator: torch.Generator,
) -> int:
    """Sample one token with the rollout's independent generator."""

    _validate(raw_logits, config)
    return sample_next_tokens(
        raw_logits.unsqueeze(0),
        config,
        (generator,),
    )[0]
