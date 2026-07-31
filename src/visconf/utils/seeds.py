"""Versioned deterministic rollout seeds and generators."""

from __future__ import annotations

import hashlib
import json

import torch


SEED_DERIVATION_VERSION = 1


class SeedError(ValueError):
    """Raised when a rollout seed identity is invalid."""


def _canonical_seed_bytes(
    *,
    base_seed: int,
    dataset: str,
    sample_id: str,
    strategy: str,
    rollout_index: int,
    seed_derivation_version: int,
) -> bytes:
    payload = {
        "base_seed": base_seed,
        "dataset": dataset,
        "rollout_index": rollout_index,
        "sample_id": sample_id,
        "seed_derivation_version": seed_derivation_version,
        "strategy": strategy,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def derive_rollout_seed(
    *,
    base_seed: int,
    dataset: str,
    sample_id: str,
    strategy: str,
    rollout_index: int,
    seed_derivation_version: int = SEED_DERIVATION_VERSION,
) -> int:
    """Derive a stable unsigned 64-bit seed from a rollout identity."""

    if not 0 <= base_seed < 2**64:
        raise SeedError("base_seed must fit uint64")
    if rollout_index < 0:
        raise SeedError("rollout_index must be non-negative")
    if seed_derivation_version != SEED_DERIVATION_VERSION:
        raise SeedError("unsupported seed derivation version")

    digest = hashlib.sha256(
        _canonical_seed_bytes(
            base_seed=base_seed,
            dataset=dataset,
            sample_id=sample_id,
            strategy=strategy,
            rollout_index=rollout_index,
            seed_derivation_version=seed_derivation_version,
        )
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def make_generator(
    seed: int,
    device: str | torch.device,
) -> torch.Generator:
    """Create and seed one independent generator on the requested device."""

    if not 0 <= seed < 2**64:
        raise SeedError("seed must fit uint64")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def make_rollout_generator(
    *,
    base_seed: int,
    dataset: str,
    sample_id: str,
    strategy: str,
    rollout_index: int,
    device: str | torch.device,
    seed_derivation_version: int = SEED_DERIVATION_VERSION,
) -> tuple[int, torch.Generator]:
    seed = derive_rollout_seed(
        base_seed=base_seed,
        dataset=dataset,
        sample_id=sample_id,
        strategy=strategy,
        rollout_index=rollout_index,
        seed_derivation_version=seed_derivation_version,
    )
    return seed, make_generator(seed, device)
