"""Tokenizer-derived logical token positions for attention groups."""

from __future__ import annotations

from typing import Any

import torch

from visconf.metrics.attention import StepTokenGroups
from visconf.types import TokenGroups


class TokenPositionError(ValueError):
    """Raised when prompt tokens cannot form valid attention groups."""


def _vector(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TokenPositionError(f"{name} must be a tensor")
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1:
        raise TokenPositionError(f"{name} must be one-dimensional")
    return value


def _special_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id < 0:
        raise TokenPositionError(f"tokenizer does not define {token}")
    if tokenizer.convert_ids_to_tokens(token_id) != token:
        raise TokenPositionError(f"tokenizer does not define {token}")
    return int(token_id)


def build_prompt_token_groups(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
) -> TokenGroups:
    ids = _vector(input_ids, "input_ids")
    mask = _vector(attention_mask, "attention_mask")
    if ids.shape != mask.shape:
        raise TokenPositionError("input_ids and attention_mask must match")

    vision_start_id = _special_token_id(tokenizer, "<|vision_start|>")
    vision_end_id = _special_token_id(tokenizer, "<|vision_end|>")
    image_pad_id = _special_token_id(tokenizer, "<|image_pad|>")
    special_ids = {
        int(token_id)
        for token_id in (getattr(tokenizer, "all_special_ids", ()) or ())
    }
    if tokenizer.pad_token_id is not None:
        special_ids.add(int(tokenizer.pad_token_id))

    valid_physical = torch.nonzero(mask == 1, as_tuple=False).flatten()
    if valid_physical.numel() == 0:
        raise TokenPositionError("prompt has no unmasked tokens")

    image_positions: list[int] = []
    prompt_text_positions: list[int] = []
    in_image_span = False
    for logical_position, physical_position in enumerate(
        valid_physical.tolist()
    ):
        token_id = int(ids[physical_position].item())
        if token_id == vision_start_id:
            if in_image_span:
                raise TokenPositionError("nested image span")
            in_image_span = True
            continue
        if token_id == vision_end_id:
            if not in_image_span:
                raise TokenPositionError("image span ends before it starts")
            in_image_span = False
            continue
        if in_image_span:
            if token_id == image_pad_id:
                image_positions.append(logical_position)
            continue
        if token_id == image_pad_id:
            raise TokenPositionError("image pad appears outside an image span")
        if token_id not in special_ids:
            prompt_text_positions.append(logical_position)

    if in_image_span:
        raise TokenPositionError("unterminated image span")

    device = ids.device
    prompt_token_count = int(valid_physical.numel())
    return TokenGroups(
        image_positions=torch.tensor(
            image_positions,
            dtype=torch.long,
            device=device,
        ),
        prompt_text_positions=torch.tensor(
            prompt_text_positions,
            dtype=torch.long,
            device=device,
        ),
        prompt_last_position=prompt_token_count - 1,
        prompt_token_count=prompt_token_count,
    )


def step_token_groups(
    prompt_groups: TokenGroups,
    retained_generated_tokens: int,
) -> StepTokenGroups:
    if retained_generated_tokens < 0:
        raise TokenPositionError(
            "retained_generated_tokens must be non-negative"
        )
    return StepTokenGroups(
        image_positions=tuple(prompt_groups.image_positions.tolist()),
        prompt_text_positions=tuple(
            prompt_groups.prompt_text_positions.tolist()
        ),
        generated_text_positions=tuple(
            range(
                prompt_groups.prompt_token_count,
                prompt_groups.prompt_token_count
                + retained_generated_tokens,
            )
        ),
    )


def predictor_position(
    prompt_groups: TokenGroups,
    retained_generated_tokens: int,
) -> int:
    if retained_generated_tokens < 0:
        raise TokenPositionError(
            "retained_generated_tokens must be non-negative"
        )
    if retained_generated_tokens == 0:
        return prompt_groups.prompt_last_position
    return (
        prompt_groups.prompt_token_count
        + retained_generated_tokens
        - 1
    )

