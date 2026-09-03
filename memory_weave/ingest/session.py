"""A small process-local cache for persisted session transcripts."""

from __future__ import annotations

from memory_weave.models import Turn
from memory_weave.store import Store


class SessionBuffer:
    """Read session turns efficiently while keeping writes durable in the store.

    Call ``append_turn`` on this buffer, rather than on its store, whenever the cache is in use.
    """

    def __init__(self, store: Store) -> None:
        self._store = store
        self._cache: dict[str, tuple[Turn, ...]] = {}

    def turns(self, session_id: str | None) -> list[Turn]:
        """Return a detached transcript in turn order, or no turns for a missing session ID."""

        if session_id is None:
            return []
        cached = self._cache.get(session_id)
        if cached is None:
            cached = tuple(self._store.session_turns(session_id))
            self._cache[session_id] = cached
        return list(cached)

    def append_turn(self, turn: Turn) -> None:
        """Persist a turn and invalidate the matching cached transcript."""

        self._store.append_turn(turn)
        self._cache.pop(turn.session_id, None)
