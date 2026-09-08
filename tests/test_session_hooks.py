"""LLD 12 hooks: session start, numbered turns, explicit end, and the idle-timeout split."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memory_weave.config import EmbeddingConfig, IngestionConfig, MemoryWeaveConfig
from memory_weave.index.embedder import FakeEmbedder
from memory_weave.index.vector import VectorIndex
from memory_weave.ingest import (
    ExtractionRunner,
    FakeExtractor,
    FakeJudge,
    Ingestor,
    SessionBuffer,
    SessionHooks,
    TableReviewer,
)
from memory_weave.models import EntityMention, ExtractionOutput, Principal, Scope, SessionSummary
from memory_weave.store import Store

_NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
_PRINCIPAL = Principal("agent", "aditya", "thread-1", None)
_CONFIG = MemoryWeaveConfig(
    embedding=EmbeddingConfig(model="fake-embedder", version="1", dims=8),
    ingestion=IngestionConfig(session_idle_timeout_minutes=30),
)


class Clock:
    def __init__(self) -> None:
        self.at = _NOW

    def __call__(self) -> datetime:
        return self.at


@pytest.fixture
def store(tmp_path: Path) -> Store:
    database = Store(tmp_path / "memory.sqlite")
    database.set_grant("agent", Scope(kind="user", id="aditya"), can_read=True, can_write=True)
    yield database
    database.close()


def test_start_turn_and_end_call_the_extraction_callback_exactly_once(store: Store) -> None:
    ended: list[tuple[str, Principal]] = []
    clock = Clock()
    hooks = SessionHooks(
        store, SessionBuffer(store), _CONFIG, on_end=lambda s, p: ended.append((s, p)), current_time=clock
    )

    hooks.on_session_start(_PRINCIPAL)
    hooks.on_session_start(_PRINCIPAL)
    assert hooks.on_turn(_PRINCIPAL, "user", "hello") == _PRINCIPAL
    clock.at += timedelta(minutes=1)
    hooks.on_turn(_PRINCIPAL, "assistant", "hi")
    hooks.on_session_end(_PRINCIPAL)
    hooks.on_session_end(_PRINCIPAL)

    turns = store.session_turns("thread-1")
    assert [(turn.turn, turn.role, turn.content) for turn in turns] == [(1, "user", "hello"), (2, "assistant", "hi")]
    session = store.get_session("thread-1")
    assert session is not None and session.ended_at == clock.at
    assert ended == [("thread-1", _PRINCIPAL)]


def test_first_turn_without_a_start_creates_the_session(store: Store) -> None:
    hooks = SessionHooks(store, SessionBuffer(store), _CONFIG, current_time=Clock())
    hooks.on_turn(_PRINCIPAL, "user", "hello")
    assert store.get_session("thread-1") is not None


def test_idle_gap_splits_the_session_and_ends_the_earlier_segment(store: Store) -> None:
    ended: list[str] = []
    clock = Clock()
    hooks = SessionHooks(store, SessionBuffer(store), _CONFIG, on_end=lambda s, p: ended.append(s), current_time=clock)

    hooks.on_turn(_PRINCIPAL, "user", "first segment")
    clock.at += timedelta(minutes=29)
    assert hooks.on_turn(_PRINCIPAL, "assistant", "still the same session").session_id == "thread-1"
    clock.at += timedelta(minutes=31)
    continued = hooks.on_turn(_PRINCIPAL, "user", "back after lunch")

    assert continued.session_id == "thread-1~2"
    assert ended == ["thread-1"]
    first = store.get_session("thread-1")
    assert first is not None and first.ended_at == _NOW + timedelta(minutes=29)
    assert [turn.turn for turn in store.session_turns("thread-1~2")] == [1]
    split = store.connection.execute("SELECT payload FROM events WHERE kind = 'session.split'").fetchone()
    assert json.loads(split["payload"])["from"] == "thread-1"
    assert json.loads(split["payload"])["to"] == "thread-1~2"

    clock.at += timedelta(minutes=45)
    third = hooks.on_turn(continued, "user", "and again")
    assert third.session_id == "thread-1~3"
    assert ended == ["thread-1", "thread-1~2"]


def test_session_end_triggers_extraction_of_that_segment(store: Store) -> None:
    clock = Clock()
    buffer = SessionBuffer(store)
    ingestor = Ingestor(
        store, VectorIndex(_CONFIG.embedding), FakeEmbedder(dims=8), FakeJudge(), buffer, _CONFIG, current_time=clock
    )
    output = ExtractionOutput(
        candidates=[],
        summary=SessionSummary(
            "Aditya said hello.", [], [EntityMention(kind="person", text="nobody", role="mentions")]
        ),
    )
    extractor = FakeExtractor(output)
    runner = ExtractionRunner(store, ingestor, extractor, TableReviewer(), buffer, _CONFIG, current_time=clock)
    hooks = SessionHooks(store, buffer, _CONFIG, on_end=lambda s, p: runner.extract_session(s, p), current_time=clock)

    hooks.on_turn(_PRINCIPAL, "user", "hello")
    hooks.on_turn(_PRINCIPAL, "assistant", "hi")
    hooks.on_session_end(_PRINCIPAL)

    assert extractor.call_count == 1
    summary = store.active_session_summary("session:thread-1")
    assert summary is not None and summary.content == "Aditya said hello."
    assert store.get_session("thread-1").extracted_at == clock.at  # type: ignore[union-attr]


def test_hooks_require_a_session_id() -> None:
    with pytest.raises(ValueError):
        SessionHooks(Store(":memory:"), SessionBuffer(Store(":memory:")), _CONFIG).on_session_start(
            Principal("agent", "aditya", None, None)
        )
