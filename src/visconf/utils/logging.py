"""Structured experiment event logging."""

from __future__ import annotations

import json
import logging
from typing import Any


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def log_event(
    logger: logging.Logger,
    event: str,
    **fields: Any,
) -> None:
    payload = {
        "event": event,
        **{
            name: value
            for name, value in fields.items()
            if value is not None
        },
    }
    logger.info(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )
