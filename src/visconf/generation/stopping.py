"""Stop-token trimming and maximum-length decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class StopReason(StrEnum):
    STOP_TOKEN = "stop_token"
    MAX_NEW_TOKENS = "max_new_tokens"


@dataclass(frozen=True, slots=True)
class StopDecision:
    retain_token: bool
    stop: bool
    reason: StopReason | None
    terminating_token_id: int | None


def resolve_stop_token_ids(
    eos_token_id: int | None,
    im_end_token_id: int | None,
) -> frozenset[int]:
    return frozenset(
        token_id
        for token_id in (eos_token_id, im_end_token_id)
        if token_id is not None
    )


def decide_stop(
    selected_token_id: int,
    retained_tokens_before: int,
    max_new_tokens: int,
    stop_token_ids: frozenset[int],
) -> StopDecision:
    """Decide whether the selected token is retained and whether decoding ends."""

    if retained_tokens_before < 0:
        raise ValueError("retained_tokens_before must be non-negative")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if retained_tokens_before >= max_new_tokens:
        raise ValueError("decoding is already at the maximum token count")

    if selected_token_id in stop_token_ids:
        return StopDecision(
            retain_token=False,
            stop=True,
            reason=StopReason.STOP_TOKEN,
            terminating_token_id=selected_token_id,
        )

    hit_maximum = retained_tokens_before + 1 == max_new_tokens
    return StopDecision(
        retain_token=True,
        stop=hit_maximum,
        reason=StopReason.MAX_NEW_TOKENS if hit_maximum else None,
        terminating_token_id=None,
    )
