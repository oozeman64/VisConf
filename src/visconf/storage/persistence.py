"""Persistent output-root initialization and restart verification."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from visconf.storage.manifest import atomic_write_json, utc_now


MARKER_NAME = ".visconf-persistent-storage.json"


class PersistenceError(RuntimeError):
    """Raised when an output root has not survived initialization."""


def initialize_persistence_marker(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = {
        "marker_version": 1,
        "marker_id": uuid.uuid4().hex,
        "output_root": str(root),
        "initialized_at_utc": utc_now().isoformat(),
    }
    atomic_write_json(root / MARKER_NAME, marker)
    directory = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return marker


def verify_persistence_marker(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    path = root / MARKER_NAME
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PersistenceError(
            "persistent storage marker is missing or invalid"
        ) from exc
    if (
        marker.get("marker_version") != 1
        or marker.get("output_root") != str(root)
        or not marker.get("marker_id")
    ):
        raise PersistenceError(
            "persistent storage marker does not match this output root"
        )
    return {
        **marker,
        "verified_at_utc": utc_now().isoformat(),
    }
