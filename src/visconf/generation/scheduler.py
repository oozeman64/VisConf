"""Deterministic bounded scheduling for pending prompt work."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice

from visconf.types import PromptBatchTelemetry, PromptBatchWorkItem

SCHEDULER_ALGORITHM_VERSION = 1


def prompt_batch_telemetry(
    items: tuple[PromptBatchWorkItem, ...],
    configured_rollout_rows: int,
    effective_rollout_rows: int,
) -> PromptBatchTelemetry:
    lengths = [item.prompt_record.prompt_token_count for item in items]
    images = [item.image_token_count for item in items]
    padded = max(lengths) * len(lengths)
    unpadded = sum(lengths)
    return PromptBatchTelemetry(
        prompt_count=len(items),
        min_prompt_length=min(lengths),
        max_prompt_length=max(lengths),
        total_unpadded_prompt_tokens=unpadded,
        total_padded_prompt_tokens=padded,
        padding_fraction=(padded - unpadded) / padded if padded else 0.0,
        min_image_token_count=min(images),
        max_image_token_count=max(images),
        configured_rollout_rows=configured_rollout_rows,
        effective_rollout_rows=effective_rollout_rows,
    )


def schedule_prompt_batches(
    pending: Iterable[PromptBatchWorkItem],
    *,
    prompt_batch_size: int,
    strategy: str,
    bucket_window_size: int | None = None,
) -> Iterator[tuple[PromptBatchWorkItem, ...]]:
    """Schedule every pending prompt once using bounded stable windows."""
    if prompt_batch_size <= 0:
        raise ValueError("prompt_batch_size must be positive")
    if strategy not in {"contiguous", "token_count_bucketed"}:
        raise ValueError("unknown prompt batching strategy")
    if strategy == "token_count_bucketed" and (
        bucket_window_size is None or bucket_window_size < prompt_batch_size
    ):
        raise ValueError("bucket window must be at least prompt_batch_size")
    iterator = iter(pending)
    window_size = (
        prompt_batch_size
        if strategy == "contiguous" or prompt_batch_size == 1
        else int(bucket_window_size)
    )
    while True:
        window = list(islice(iterator, window_size))
        if not window:
            return
        if strategy == "token_count_bucketed" and prompt_batch_size > 1:
            window.sort(
                key=lambda item: (
                    item.prompt_record.prompt_token_count,
                    item.image_token_count,
                    item.canonical_source_ordinal,
                )
            )
        for offset in range(0, len(window), prompt_batch_size):
            yield tuple(window[offset : offset + prompt_batch_size])
