"""Stable scorer identity hashes."""

from __future__ import annotations

import hashlib
from pathlib import Path

from visconf.config import ScoringSettings, sha256_json


def _scorer_code_hash() -> str:
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in (
        "scoring/answer_normalization.py",
        "datasets/mathverse.py",
        "datasets/mathvista.py",
        "datasets/mmmu_pro.py",
    ):
        path = package / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def scorer_identity(config: ScoringSettings) -> dict[str, str]:
    scorer_config = {
        "name": config.scorer_name,
        "version": config.scorer_version,
    }
    return {
        **scorer_config,
        "mode": config.mode.value,
        "config_hash": sha256_json(scorer_config),
        "code_hash": _scorer_code_hash(),
    }
