"""Shared time, identifier, text-normalization, and timing utilities."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from uuid import UUID

from uuid6 import uuid7 as _uuid7

_WHITESPACE_RE = re.compile(r"\s+")


def uuid7() -> UUID:
    """Return a time-ordered UUIDv7 identifier."""

    return _uuid7()


def now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def normalize_ws(value: str) -> str:
    """Trim and collapse all whitespace in a string."""

    return _WHITESPACE_RE.sub(" ", value).strip()


def normalize_alias(value: str) -> str:
    """Normalize an entity alias for exact lookup."""

    decomposed = unicodedata.normalize("NFKD", value)
    without_diacritics = "".join(char for char in decomposed if not unicodedata.combining(char))
    return normalize_ws(without_diacritics).lower()


class Timer:
    """Collect ordered elapsed milliseconds for one observable operation."""

    def __init__(self, *, warm: bool, clock: Callable[[], float] = perf_counter) -> None:
        self.warm = warm
        self._clock = clock
        self._last = clock()
        self._timings_ms: dict[str, float] = {}

    def mark(self, name: str) -> float:
        """Record the elapsed duration since the previous mark under ``name``."""

        if name in self._timings_ms:
            raise ValueError(f"Timer stage already recorded: {name}")
        current = self._clock()
        elapsed_ms = (current - self._last) * 1000
        self._timings_ms[name] = elapsed_ms
        self._last = current
        return elapsed_ms

    def as_dict(self) -> dict[str, float]:
        """Return stages in mark order followed by their summed total."""

        return {**self._timings_ms, "total": sum(self._timings_ms.values())}
