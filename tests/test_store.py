from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from memory_weave.models import Record, Scope, Turn
from memory_weave.store.migrations import migrate
from memory_weave.store.store import Store

_NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
_USER_SCOPE = Scope(kind="user", id="aditya")


@pytest.fixture
def store(tmp_path: Path) -> Store:
    database = Store(tmp_path / "memory.sqlite")
    _ = database.connection
    yield database
    database.close()


def _record(record_id: str, **overrides: Any) -> Record:
    values: dict[str, Any] = {
        "id": record_id,
        "type": "semantic",
        "version": 1,
        "content": "Aditya prefers concise technical explanations.",
        "subject": "person:aditya/explanation_style",
        "scope": _USER_SCOPE,
        "source_kind": "user_statement",
        "source_ref": "session:session-1<turn:1>",
        "creator_agent_id": "research-agent",
        "evidence": "Keep answers concise.",
        "created_at": _NOW,
        "event_at": _NOW,
        "expires_at": None,
        "confidence": 0.95,
        "status": "confirmed",
        "supersedes_id": None,
        "reinforcements": 0,
        "last_reinforced_at": None,
        "tags": ["communication"],
        "entity_ids": [],
    }
    values.update(overrides)
    return Record(**values)


def _search_log_row(search_id: str) -> dict[str, Any]:
    return {
        "id": search_id,
        "at": _NOW,
        "agent_id": "research-agent",
        "user_id": "aditya",
        "session_id": "session-1",
        "request": {"queries": ["explanation preference"], "k": 8},
        "context": "Keep answers concise.",
        "rewrite_status": "disabled",
        "rewritten_queries": None,
        "readable_scopes": [{"kind": "user", "id": "aditya"}],
        "dense": [],
        "lexical": [],
        "entity": [],
        "fused": [],
        "freshness": [],
        "gated_out": [],
        "deduped_out": [],
        "reranked": None,
        "budget_out": [],
        "returned": [],
        "explanations": [],
        "config_flags": {"embedding_version": "1"},
        "warm": True,
        "timings_ms": {"total": 1.2},
    }


def test_migrate_is_idempotent_and_creates_every_phase_one_table(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "memory.sqlite")
    try:
        migrate(connection)
        migrate(connection)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
            ).fetchall()
        }
        assert {
            "schema_version",
            "records",
            "record_conflicts",
            "embeddings",
            "records_fts",
            "entities",
            "entity_aliases",
            "record_entities",
            "grants",
            "sessions",
            "session_turns",
            "events",
            "search_log",
        } <= tables
        assert connection.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1
        connection.close()
        connection = sqlite3.connect(tmp_path / "memory.sqlite")
        migrate(connection)
        assert connection.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1
    finally:
        connection.close()


def test_record_round_trip_and_lifecycle_updates(store: Store) -> None:
    assert store.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert store.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    entity = store.create_entity(kind="person", canonical="Aditya", scope=_USER_SCOPE, entity_id="entity-aditya")
    store.add_alias(entity.id, "aditya")
    older_record = _record("record-older", status="superseded")
    record = _record("record-1")
    store.insert_record(older_record)
    store.insert_record(record)
    store.link_record_entity(record.id, entity.id)

    store.set_supersedes(record.id, "record-older")
    store.reinforce_fields(
        record.id,
        confidence=0.99,
        reinforcements=2,
        last_reinforced_at=_NOW + timedelta(minutes=5),
        expires_at=None,
        status="confirmed",
    )

    restored = store.get_record(record.id)

    assert restored is not None
    assert restored.content == record.content
    assert restored.tags == ["communication"]
    assert restored.created_at == _NOW
    assert restored.last_reinforced_at == _NOW + timedelta(minutes=5)
    assert restored.supersedes_id == "record-older"
    assert restored.entity_ids == [entity.id]
    assert store.get_records(["missing", record.id]) == [restored]
    assert store.active_by_subject(_USER_SCOPE, record.subject) == [restored]


def test_eligible_ids_applies_scope_lifecycle_type_and_time_filters(store: Store) -> None:
    visible = _record("visible")
    other_scope = _record("other-scope", scope=Scope(kind="project", id="other-project"))
    superseded = _record("superseded", status="superseded")
    expired = _record("expired", expires_at=_NOW - timedelta(days=1))
    deleted = _record("deleted", status="deleted")
    episode = _record(
        "episode",
        type="episodic",
        subject="person:aditya/-",
        event_at=_NOW - timedelta(days=10),
    )
    for record in (visible, other_scope, superseded, expired, deleted, episode):
        store.insert_record(record)

    assert store.eligible_ids([_USER_SCOPE], None, None, None, False, _NOW) == {"visible", "episode"}
    assert store.eligible_ids([_USER_SCOPE], None, None, None, True, _NOW) == {
        "visible",
        "superseded",
        "expired",
        "episode",
    }
    assert store.eligible_ids([_USER_SCOPE], ["episodic"], None, None, False, _NOW) == {"episode"}
    assert store.eligible_ids([_USER_SCOPE], None, _NOW - timedelta(days=1), None, False, _NOW) == {"visible"}


def test_embeddings_and_fts_round_trip_through_store_boundaries(store: Store) -> None:
    content_match = _record("content-match", content="Service raised ERR42 during refresh.")
    alias_match = _record("alias-match", content="The token refresh failed.")
    store.insert_record(content_match)
    store.insert_record(alias_match)
    store.put_embedding(content_match.id, "fake", "1", np.array([0.5, -0.25], dtype=np.float64))
    store.upsert_fts(content_match.id, content_match.content, content_match.subject, "")
    store.upsert_fts(alias_match.id, alias_match.content, alias_match.subject, "ERR42")

    embeddings = list(store.iter_embeddings("fake", "1"))
    matches = store.fts_query("ERR42", limit=10)

    assert embeddings[0][0] == content_match.id
    assert embeddings[0][1].dtype == np.float32
    np.testing.assert_array_equal(embeddings[0][1], np.array([0.5, -0.25], dtype=np.float32))
    assert [record_id for record_id, _ in matches] == [alias_match.id, content_match.id]
    store.delete_fts(alias_match.id)
    assert [record_id for record_id, _ in store.fts_query("ERR42", limit=10)] == [content_match.id]


def test_conflicts_entities_grants_sessions_events_and_search_log(store: Store) -> None:
    first = _record("record-first")
    second = _record("record-second", subject="person:aditya/timezone")
    store.insert_record(first)
    store.insert_record(second)
    store.add_conflict(first.id, second.id, noted_at=_NOW)
    assert store.conflicts_for(first.id) == [second.id]
    assert store.conflicts_for(second.id) == [first.id]

    source = store.create_entity(kind="project", canonical="Memory Weave", scope=_USER_SCOPE, entity_id="entity-source")
    destination = store.create_entity(
        kind="project", canonical="Memory Weave Core", scope=_USER_SCOPE, entity_id="entity-dest"
    )
    store.add_alias(source.id, "memory weave")
    store.add_alias(destination.id, "memory core")
    store.link_record_entity(first.id, source.id, role="about")
    assert [entity.id for entity in store.entities_by_alias("memory weave", scopes=[_USER_SCOPE])] == [source.id]
    assert store.records_for_entities([source.id], {first.id}, limit=10) == [(first.id, source.id)]
    store.merge_entity(source.id, destination.id)
    assert store.get_entity(source.id).merged_into == destination.id  # type: ignore[union-attr]
    assert store.get_entity(destination.id).aliases == ["memory core", "memory weave"]  # type: ignore[union-attr]
    assert store.records_for_entities([destination.id], {first.id}, limit=10) == [(first.id, destination.id)]

    store.set_grant("implementation-agent", _USER_SCOPE, can_read=True, can_write=False)
    assert store.grants_for("implementation-agent", can_read=True) == [_USER_SCOPE]
    assert store.grants_for("implementation-agent", can_write=True) == []

    store.create_session("session-1", "implementation-agent", "aditya", "memory-weave", _NOW)
    turn = Turn("session-1", 1, "user", "Keep answers concise.", _NOW)
    store.append_turn(turn)
    store.end_session("session-1", _NOW + timedelta(minutes=1))
    store.mark_extracted("session-1", _NOW + timedelta(minutes=2))
    assert store.session_turns("session-1") == [turn]
    session = store.connection.execute("SELECT ended_at, extracted_at FROM sessions WHERE id = 'session-1'").fetchone()
    assert session is not None
    assert session["ended_at"] is not None
    assert session["extracted_at"] is not None

    store.append_event(
        "record.created", "implementation-agent", first.id, None, {"reason": "test"}, event_id="event-1", at=_NOW
    )
    assert store.events_for(first.id)[0]["payload"] == {"reason": "test"}
    assert not hasattr(Store, "update_event")
    assert not hasattr(Store, "delete_event")

    store.write_search_log(_search_log_row("search-1"))
    search_log = store.read_search_log("search-1")
    assert search_log is not None
    assert search_log["request"] == {"k": 8, "queries": ["explanation preference"]}
    assert search_log["warm"] is True


def test_snapshot_uses_sqlite_backup_api(store: Store, tmp_path: Path) -> None:
    record = _record("record-for-snapshot")
    store.insert_record(record)
    entity = store.create_entity(kind="project", canonical="Memory Weave", scope=_USER_SCOPE)
    store.link_record_entity(record.id, entity.id)
    store.upsert_fts(record.id, record.content, record.subject, "memory weave")
    snapshot_path = tmp_path / "snapshot.sqlite"
    table_names = ("records", "entities", "record_entities", "records_fts")
    source_counts = {
        table_name: store.connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        for table_name in table_names
    }

    store.snapshot_to(snapshot_path)

    snapshot = sqlite3.connect(snapshot_path)
    try:
        snapshot_counts = {
            table_name: snapshot.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            for table_name in table_names
        }
        assert snapshot_counts == source_counts
    finally:
        snapshot.close()
