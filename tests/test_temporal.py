"""Temporal metadata validation, the due-review worker, and the summary gate floor."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memory_weave.config import GateConfig
from memory_weave.ingest import flag_due_reviews, has_temporal_expression, validate_temporal
from memory_weave.models import CandidateRecord, Record, Scope, Turn
from memory_weave.retrieve.gate import _dense_floor
from memory_weave.store import Store
from memory_weave.util import render_subject

_NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
_SCOPE = Scope(kind="user", id="aditya")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Until Friday please answer in Spanish", True),
        ("We ship on 2026-10-01.", True),
        ("Let's revisit this next quarter.", True),
        ("The offsite is in March.", True),
        ("I'll be away for two weeks.", True),
        ("Deadline is by the 15th.", True),
        ("I prefer concise technical explanations.", False),
        ("Rohan runs the platform team.", False),
    ],
)
def test_temporal_expression_detection(text: str, expected: bool) -> None:
    assert has_temporal_expression(text) is expected


def _candidate(evidence: str, **temporal: datetime | None) -> CandidateRecord:
    return CandidateRecord(
        type="semantic",
        content="x",
        attribute="a",
        source_kind="user_statement",
        evidence=evidence,
        evidence_turn=1,
        entity_mentions=[],
        event_at=None,
        confidence=0.9,
        **temporal,
    )


def test_validate_temporal_rules() -> None:
    turn = Turn("s", 1, "user", "Until Friday please answer in Spanish", _NOW)
    friday = _NOW + timedelta(days=3)
    assert validate_temporal(_candidate("no dates here"), turn) is None
    assert validate_temporal(_candidate("no dates here", review_at=friday), turn) == "unsupported_temporal_metadata"
    assert validate_temporal(_candidate(turn.content, valid_until=friday), turn) is None
    assert validate_temporal(_candidate(turn.content, valid_from=friday, valid_until=_NOW), turn) == (
        "invalid_temporal_range"
    )
    assert validate_temporal(_candidate(turn.content, review_at=_NOW - timedelta(days=1)), turn) == (
        "review_at_before_evidence"
    )
    assert validate_temporal(_candidate(turn.content, review_at=datetime(2026, 9, 11)), turn) == (
        "naive_temporal_value"
    )


def _record(store: Store, record_id: str, review_at: datetime | None, status: str = "confirmed") -> Record:
    entity = store.get_entity("p") or store.create_entity(
        kind="person", canonical="aditya", scope=_SCOPE, entity_id="p"
    )
    record = Record(
        id=record_id,
        type="semantic",
        version=1,
        content="Aditya is going to Singapore in July.",
        subject=render_subject(entity.id, record_id),
        scope=_SCOPE,
        source_kind="user_statement",
        source_ref="session:s<turn:1>",
        creator_agent_id="agent",
        evidence="I am going to Singapore in July",
        created_at=_NOW - timedelta(days=60),
        event_at=_NOW - timedelta(days=60),
        expires_at=None,
        confidence=0.95,
        status=status,  # type: ignore[arg-type]
        supersedes_id=None,
        reinforcements=0,
        last_reinforced_at=None,
        tags=[],
        entity_ids=[entity.id],
        subject_entity_id=entity.id,
        attribute=record_id,
        valid_from=_NOW - timedelta(days=10),
        valid_until=_NOW - timedelta(days=1),
        review_at=review_at,
    )
    store.insert_record(record)
    return record


def test_due_review_flags_once_and_changes_nothing_else(tmp_path: Path) -> None:
    store = Store(tmp_path / "memory.sqlite")
    due = _record(store, "due", _NOW - timedelta(hours=1))
    _record(store, "later", _NOW + timedelta(days=1))
    _record(store, "never", None)
    _record(store, "gone", _NOW - timedelta(hours=1), status="superseded")

    assert flag_due_reviews(store, batch_size=10, at=_NOW) == ["due"]
    assert flag_due_reviews(store, batch_size=10, at=_NOW) == []

    flagged = store.get_record("due")
    assert flagged is not None
    assert flagged.review_flagged_at == _NOW
    assert (flagged.content, flagged.source_kind, flagged.status, flagged.evidence, flagged.valid_until) == (
        due.content,
        due.source_kind,
        due.status,
        due.evidence,
        due.valid_until,
    )
    events = store.events_for("due")
    assert [event["kind"] for event in events] == ["record.review_due"]
    assert events[0]["actor"] == "temporal_reviewer"
    assert events[0]["payload"]["review_at"] == due.review_at.isoformat()  # type: ignore[union-attr]
    assert store.get_record("later").review_flagged_at is None  # type: ignore[union-attr]
    assert store.get_record("gone").review_flagged_at is None  # type: ignore[union-attr]


def test_batch_size_bounds_one_pass(tmp_path: Path) -> None:
    store = Store(tmp_path / "memory.sqlite")
    for index in range(5):
        _record(store, f"r{index}", _NOW - timedelta(minutes=index + 1))
    assert len(flag_due_reviews(store, batch_size=2, at=_NOW)) == 2
    assert len(flag_due_reviews(store, batch_size=10, at=_NOW)) == 3


def test_two_workers_racing_on_one_due_record_emit_one_flag(tmp_path: Path) -> None:
    store = Store(tmp_path / "memory.sqlite")
    _record(store, "due", _NOW - timedelta(hours=1))
    barrier = threading.Barrier(2)
    flagged: list[list[str]] = []

    def work() -> None:
        barrier.wait()
        flagged.append(flag_due_reviews(store, batch_size=10, at=_NOW))

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(len(entry) for entry in flagged) == [0, 1]
    rows = store.connection.execute(
        "SELECT payload FROM events WHERE kind = 'record.review_due' AND record_id = 'due'"
    ).fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0]["payload"])["flagged_at"] == _NOW.isoformat()
    assert store.connection.execute("SELECT review_flagged_at FROM records WHERE id = 'due'").fetchone()[0] is not None


def test_session_summaries_use_their_own_dense_floor() -> None:
    config = GateConfig()
    summary = Record(
        id="s",
        type="episodic",
        version=1,
        content="",
        subject="",
        scope=_SCOPE,
        source_kind="session_summary",
        source_ref="session:s",
        creator_agent_id="agent",
        evidence=None,
        created_at=_NOW,
        event_at=_NOW,
        expires_at=None,
        confidence=0.8,
        status="confirmed",
        supersedes_id=None,
        reinforcements=0,
        last_reinforced_at=None,
        tags=[],
        entity_ids=[],
    )
    assert _dense_floor(summary, config) == config.dense_floor.session_summary
    summary.source_kind = "user_statement"
    assert _dense_floor(summary, config) == config.dense_floor.episodic
