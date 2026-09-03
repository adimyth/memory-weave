"""Minimal structured logging helpers used by every Memory Weave component."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any


class JsonFormatter(logging.Formatter):
    """Render standard logging records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "memory_weave", None)
        if isinstance(extra, Mapping):
            payload.update(extra)
        return json.dumps(payload, default=str, sort_keys=True)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with JSON output when no handler has already been configured."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger
