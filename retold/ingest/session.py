"""The session transcript: a process-local cache over persisted turns, and the adapter-facing hooks."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta

from retold.config import RetoldConfig
from retold.models import Principal, Turn, TurnRole
from retold.store import Store
from retold.util import now

_SPLIT_SUFFIX = re.compile(r"~\d+$")


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

    def invalidate(self, session_id: str) -> None:
        """Drop the cached transcript so the next read sees turns another process or thread appended."""

        self._cache.pop(session_id, None)


SessionEndCallback = Callable[[str, Principal], object]


class SessionHooks:
    """LLD 12: the three calls an adapter makes, plus the idle-timeout split for hosts without a session end.

    A split closes the idle session, hands it to ``on_end`` for extraction, and continues under a derived
    session id. Evidence quoted from the earlier segment is then unfindable from the new one, so such a
    write is downgraded by the ordinary evidence rule rather than lost; a cross-session lookback is an LLD 18
    item and is not built here.
    """

    def __init__(
        self,
        store: Store,
        session_buffer: SessionBuffer,
        config: RetoldConfig,
        *,
        on_end: SessionEndCallback | None = None,
        current_time: Callable[[], datetime] = now,
    ) -> None:
        self._store = store
        self._buffer = session_buffer
        self._config = config
        self._on_end = on_end
        self._current_time = current_time

    def on_session_start(self, principal: Principal, *, at: datetime | None = None) -> None:
        """Create the session for the derived principal; a repeated start is a no-op."""

        session_id = _session_id(principal)
        if self._store.get_session(session_id) is None:
            self._store.create_session(
                session_id, principal.agent_id, principal.user_id, principal.project_id, at or self._current_time()
            )

    def on_turn(self, principal: Principal, role: TurnRole, content: str, *, at: datetime | None = None) -> Principal:
        """Append one numbered turn and return the principal to keep using, split if the session went idle."""

        current = at or self._current_time()
        session_id = _session_id(principal)
        if self._store.get_session(session_id) is None:
            self.on_session_start(principal, at=current)
        turns = self._buffer.turns(session_id)
        if turns and self._is_idle(session_id, turns[-1].at, current):
            self.on_session_end(principal, at=turns[-1].at)
            new_id = self._split_id(session_id)
            self._store.create_session(new_id, principal.agent_id, principal.user_id, principal.project_id, current)
            self._store.append_event(
                "session.split",
                principal.agent_id,
                None,
                None,
                {
                    "from": session_id,
                    "to": new_id,
                    "idle_seconds": round((current - turns[-1].at).total_seconds(), 3),
                    "idle_timeout_minutes": self._config.ingestion.session_idle_timeout_minutes,
                },
            )
            principal = replace(principal, session_id=new_id)
            session_id = new_id
            turns = []
        number = turns[-1].turn + 1 if turns else 1
        self._buffer.append_turn(Turn(session_id, number, role, content, current))
        return principal

    def on_session_end(self, principal: Principal, *, at: datetime | None = None) -> None:
        """Mark the session complete and hand it to the extraction callback exactly once."""

        session_id = _session_id(principal)
        session = self._store.get_session(session_id)
        if session is None or session.ended_at is not None:
            return
        self._store.end_session(session_id, at or self._current_time())
        if self._on_end is not None:
            self._on_end(session_id, principal)

    def _is_idle(self, session_id: str, last_at: datetime, current: datetime) -> bool:
        session = self._store.get_session(session_id)
        if session is None or session.ended_at is not None:
            return False
        idle = timedelta(minutes=self._config.ingestion.session_idle_timeout_minutes)
        return current - last_at > idle

    def _split_id(self, session_id: str) -> str:
        base = _SPLIT_SUFFIX.sub("", session_id)
        number = 2
        while self._store.get_session(f"{base}~{number}") is not None:
            number += 1
        return f"{base}~{number}"


def _session_id(principal: Principal) -> str:
    if principal.session_id is None:
        raise ValueError("Session hooks need a principal with a session id.")
    return principal.session_id
