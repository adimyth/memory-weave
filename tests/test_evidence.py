from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from memory_weave.config import EvidenceConfig, IngestionConfig, MemoryWeaveConfig
from memory_weave.ingest.evidence import session_turn_source_ref, validate_evidence
from memory_weave.ingest.session import SessionBuffer
from memory_weave.models import EvidenceSourceKind, Turn
from memory_weave.store import Store

_NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
_SESSION_ID = "session-1"
_CONFIG = MemoryWeaveConfig()


@pytest.fixture
def store(tmp_path: Path) -> Store:
    database = Store(tmp_path / "memory.sqlite")
    _ = database.connection
    database.create_session(_SESSION_ID, "implementation-agent", "aditya", "memory-weave", _NOW)
    yield database
    database.close()


@pytest.fixture
def session_buffer(store: Store) -> SessionBuffer:
    buffer = SessionBuffer(store)
    buffer.append_turn(Turn(_SESSION_ID, 1, "user", "yes, we can proceed.", _NOW))
    buffer.append_turn(Turn(_SESSION_ID, 3, "assistant", "I think concise answers are best.", _NOW))
    buffer.append_turn(Turn(_SESSION_ID, 5, "tool", "Repository size is 4.2 MB.", _NOW))
    buffer.append_turn(Turn(_SESSION_ID, 7, "user", "Please keep every answer concise.", _NOW))
    buffer.append_turn(Turn(_SESSION_ID, 9, "user", "“Please keep every answer concise”—thanks.", _NOW))
    return buffer


@pytest.mark.parametrize(
    ("session_id", "quote", "claimed", "turn_hint", "found", "turn", "role", "source_kind", "note"),
    [
        (
            _SESSION_ID,
            "This exact quote does not exist.",
            "user_statement",
            None,
            False,
            None,
            None,
            "agent_inference",
            "evidence not found in session",
        ),
        (
            _SESSION_ID,
            "yes",
            "user_statement",
            None,
            False,
            None,
            None,
            "agent_inference",
            "evidence quote is too short",
        ),
        (
            _SESSION_ID,
            "I think concise answers are best.",
            "user_statement",
            None,
            True,
            3,
            "assistant",
            "agent_inference",
            "downgraded from user_statement: quote is from an assistant turn",
        ),
        (
            _SESSION_ID,
            "Repository size is 4.2 MB.",
            "user_statement",
            None,
            True,
            5,
            "tool",
            "tool_result",
            "downgraded from user_statement: quote is from a tool turn",
        ),
        (
            _SESSION_ID,
            '"Please keep every answer concise"-thanks.',
            "user_statement",
            None,
            True,
            9,
            "user",
            "user_statement",
            None,
        ),
        (
            _SESSION_ID,
            "Please keep every answer concise.",
            "agent_inference",
            None,
            True,
            7,
            "user",
            "agent_inference",
            None,
        ),
        (
            "missing-session",
            "Please keep every answer concise.",
            "user_statement",
            None,
            False,
            None,
            None,
            "agent_inference",
            "evidence not found in session",
        ),
        (
            _SESSION_ID,
            "Please keep every answer concise.",
            "user_statement",
            7,
            True,
            7,
            "user",
            "user_statement",
            None,
        ),
    ],
)
def test_validate_evidence_applies_the_documented_evidence_rules(
    session_buffer: SessionBuffer,
    session_id: str,
    quote: str,
    claimed: EvidenceSourceKind,
    turn_hint: int | None,
    found: bool,
    turn: int | None,
    role: str | None,
    source_kind: EvidenceSourceKind,
    note: str | None,
) -> None:
    evidence = validate_evidence(
        session_buffer,
        session_id,
        quote,
        claimed,
        _CONFIG,
        turn_hint=turn_hint,
    )

    assert evidence.found is found
    assert evidence.turn == turn
    assert evidence.role == role
    assert evidence.source_kind == source_kind
    assert evidence.note == note
    if evidence.turn is not None:
        assert session_turn_source_ref(session_id, evidence.turn) == f"session:{session_id}<turn:{evidence.turn}>"


def test_session_buffer_invalidates_cached_turns_after_append(store: Store) -> None:
    buffer = SessionBuffer(store)
    first = Turn(_SESSION_ID, 1, "user", "First turn.", _NOW)
    second = Turn(_SESSION_ID, 2, "assistant", "Second turn.", _NOW)
    buffer.append_turn(first)

    assert buffer.turns(_SESSION_ID) == [first]

    buffer.append_turn(second)

    assert buffer.turns(_SESSION_ID) == [first, second]


def test_validate_evidence_treats_an_unknown_turn_role_as_inference(store: Store) -> None:
    store.connection.execute(
        "INSERT INTO session_turns(session_id, turn, role, content, at) VALUES (?, ?, ?, ?, ?)",
        (_SESSION_ID, 11, "system", "The deployment completed successfully.", _NOW.isoformat()),
    )

    evidence = validate_evidence(
        SessionBuffer(store),
        _SESSION_ID,
        "The deployment completed successfully.",
        "user_statement",
        _CONFIG,
    )

    assert evidence.found is True
    assert evidence.turn == 11
    assert evidence.role is None
    assert evidence.source_kind == "agent_inference"
    assert evidence.note == "unsupported transcript role; treated as agent_inference"


def test_validate_evidence_reads_its_minimum_length_from_config(session_buffer: SessionBuffer) -> None:
    config = MemoryWeaveConfig(ingestion=IngestionConfig(evidence=EvidenceConfig(min_characters=100, min_words=2)))

    evidence = validate_evidence(session_buffer, _SESSION_ID, "yes, we", "user_statement", config)

    assert evidence.found is True
    assert evidence.source_kind == "user_statement"
