from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from memory_weave.cli import main
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
        "trigger": "tool",
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
        "reranked_out": [],
        "budget_out": [],
        "returned": [],
        "explanations": [],
        "config_flags": {"embedding_version": "1"},
        "warm": True,
        "timings_ms": {"total": 1.2},
    }


def _legacy_schema() -> str:
    current_schema = files("memory_weave.store").joinpath("schema.sql").read_text(encoding="utf-8")
    return (
        current_schema.replace(
            "  subject         TEXT NOT NULL,              -- derived display value: "
            "<subject_entity_id>/<attribute>\n"
            "  subject_entity_id TEXT REFERENCES entities(id),\n"
            "  attribute       TEXT,\n",
            "  subject         TEXT NOT NULL,\n",
        )
        .replace(
            "CREATE INDEX records_subject ON records(scope_kind, scope_id, subject_entity_id, attribute, status);",
            "CREATE INDEX records_subject ON records(scope_kind, scope_id, subject, status);",
        )
        .replace("CREATE INDEX records_index_version ON records(index_version);\n", "")
        .replace("CREATE INDEX embeddings_index_version ON embeddings(index_version);\n", "")
        .replace("  reranked_out  TEXT NOT NULL DEFAULT '[]',\n", "")
        .replace("  extracted_at TEXT,\n  extraction_started_at TEXT\n", "  extracted_at TEXT\n")
        .replace("  trigger       TEXT NOT NULL DEFAULT 'tool',\n", "")
        .replace("  index_version   INTEGER NOT NULL DEFAULT 0,\n", "")
        .replace(
            "  vector      BLOB NOT NULL,\n  index_version INTEGER NOT NULL DEFAULT 0\n",
            "  vector      BLOB NOT NULL\n",
        )
        .replace(
            "CREATE TABLE store_meta (\n"
            "  key           TEXT PRIMARY KEY,\n"
            "  value         INTEGER NOT NULL\n"
            ");\n"
            "INSERT INTO store_meta(key, value) VALUES ('records_version', 0);\n\n",
            "",
        )
    )


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
            "migration_issues",
            "store_meta",
        } <= tables
        assert connection.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 5
        indexes = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
        assert {"records_index_version", "embeddings_index_version"} <= indexes
        connection.close()
        connection = sqlite3.connect(tmp_path / "memory.sqlite")
        migrate(connection)
        assert connection.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 5
    finally:
        connection.close()


def test_migration_two_backfills_a_structured_subject_from_an_about_link(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "legacy.sqlite")
    try:
        connection.executescript(_legacy_schema())
        connection.execute(
            "CREATE TABLE schema_version("
            "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        connection.execute("INSERT INTO schema_version(version) VALUES (1)")
        connection.execute(
            "INSERT INTO entities(id, kind, canonical, scope_kind, scope_id, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("entity-aditya", "person", "Aditya", "user", "aditya", "confirmed", _NOW.isoformat()),
        )
        connection.execute(
            """
            INSERT INTO records(
                id, type, version, content, subject, scope_kind, scope_id, source_kind, creator_agent_id,
                created_at, event_at, confidence, status, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-record",
                "semantic",
                1,
                "Aditya prefers concise answers.",
                "person:aditya/explanation_style",
                "user",
                "aditya",
                "user_statement",
                "research-agent",
                _NOW.isoformat(),
                _NOW.isoformat(),
                0.95,
                "confirmed",
                "[]",
            ),
        )
        connection.execute(
            "INSERT INTO record_entities(record_id, entity_id, role) VALUES (?, ?, 'about')",
            ("legacy-record", "entity-aditya"),
        )
        connection.execute(
            "INSERT INTO records_fts(record_id, content, subject, aliases) VALUES (?, ?, ?, ?)",
            ("legacy-record", "Aditya prefers concise answers.", "person:aditya/explanation_style", "aditya"),
        )

        result = migrate(connection)

        row = connection.execute(
            "SELECT subject_entity_id, attribute, subject FROM records WHERE id = 'legacy-record'"
        ).fetchone()
        assert row == ("entity-aditya", "explanation_style", "entity-aditya/explanation_style")
        assert connection.execute("SELECT subject FROM records_fts WHERE record_id = 'legacy-record'").fetchone()[
            0
        ] == ("entity-aditya/explanation_style")
        assert result.unmapped_subject_records == 0
    finally:
        connection.close()


def test_migration_two_reports_legacy_records_without_one_about_link(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "legacy-unmapped.sqlite")
    try:
        connection.executescript(_legacy_schema())
        connection.execute(
            "CREATE TABLE schema_version("
            "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        connection.execute("INSERT INTO schema_version(version) VALUES (1)")
        connection.execute(
            """
            INSERT INTO records(
                id, type, version, content, subject, scope_kind, scope_id, source_kind, creator_agent_id,
                created_at, event_at, confidence, status, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "unmapped-record",
                "semantic",
                1,
                "Aditya prefers concise answers.",
                "person:aditya/explanation_style",
                "user",
                "aditya",
                "user_statement",
                "research-agent",
                _NOW.isoformat(),
                _NOW.isoformat(),
                0.95,
                "confirmed",
                "[]",
            ),
        )

        result = migrate(connection)

        assert result.unmapped_subject_records == 1
        assert connection.execute(
            "SELECT record_id, issue FROM migration_issues WHERE migration_version = 2"
        ).fetchone() == ("unmapped-record", "expected exactly one about entity")
        assert migrate(connection).unmapped_subject_records == 1
    finally:
        connection.close()


def test_migrate_cli_requires_an_explicit_override_for_unmapped_legacy_subjects(tmp_path: Path) -> None:
    database_path = tmp_path / "memory.sqlite"
    store = Store(database_path)
    store.insert_record(_record("unmapped-record"))
    store.connection.execute(
        "INSERT INTO migration_issues(migration_version, record_id, issue) VALUES (2, ?, ?)",
        ("unmapped-record", "expected exactly one about entity"),
    )
    store.close()

    assert main(["--store", str(database_path), "migrate"]) == 2
    assert main(["--store", str(database_path), "migrate", "--allow-unmapped"]) == 0


def test_record_round_trip_and_lifecycle_updates(store: Store) -> None:
    assert store.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert store.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    entity = store.create_entity(kind="person", canonical="Aditya", scope=_USER_SCOPE, entity_id="entity-aditya")
    store.add_alias(entity.id, "aditya")
    older_record = _record("record-older", status="superseded")
    record = _record("record-1", subject_entity_id=entity.id, attribute="explanation_style")
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
    assert store.active_by_subject(_USER_SCOPE, entity.id, "explanation_style") == [restored]


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

    assert store.count_embeddings("fake", "1") == 1
    assert store.count_embeddings("fake", "different-version") == 0
    assert embeddings[0][0] == content_match.id
    assert embeddings[0][1].dtype == np.float32
    np.testing.assert_allclose(embeddings[0][1], np.array([0.5, -0.25], dtype=np.float32) / np.sqrt(0.3125))
    assert [record_id for record_id, _ in matches] == [alias_match.id, content_match.id]
    store.delete_fts(alias_match.id)
    assert [record_id for record_id, _ in store.fts_query("ERR42", limit=10)] == [content_match.id]


def test_transaction_is_public_reentrant_and_rolls_back_a_composite_write(store: Store) -> None:
    record = _record("record-transaction")
    entity = store.create_entity(kind="person", canonical="Aditya", scope=_USER_SCOPE)

    with store.transaction():
        store.insert_record(record)
        store.put_embedding(record.id, "fake", "1", np.array([1.0, 0.0], dtype=np.float32))
        store.upsert_fts(record.id, record.content, record.subject, "aditya")
        store.link_record_entity(record.id, entity.id)
        store.append_event("record.created", "research-agent", record.id, entity.id, {"source": "test"})

    assert store.get_record(record.id).entity_ids == [entity.id]  # type: ignore[union-attr]
    assert list(store.iter_embeddings("fake", "1"))[0][0] == record.id
    assert store.fts_query("concise", limit=1)[0][0] == record.id
    assert store.events_for(record.id)[0]["payload"] == {"source": "test"}

    with pytest.raises(RuntimeError, match="abort"):
        with store.transaction():
            store.insert_record(_record("record-rolled-back"))
            store.upsert_fts("record-rolled-back", "will disappear", "person:aditya/test", "")
            raise RuntimeError("abort")

    assert store.get_record("record-rolled-back") is None
    assert store.fts_query("disappear", limit=1) == []

    with store.transaction():
        store.insert_record(_record("outer"))
        with pytest.raises(RuntimeError, match="inner"):
            with store.transaction():
                store.insert_record(_record("inner"))
                raise RuntimeError("inner")
        store.insert_record(_record("after"))

    assert store.get_record("outer") is not None
    assert store.get_record("after") is not None
    assert store.get_record("inner") is None


def test_projection_writes_bump_the_index_version_inside_their_own_transaction(
    monkeypatch: pytest.MonkeyPatch, store: Store
) -> None:
    observed_transactions: list[bool] = []
    original_bump = store._bump_records_version

    def checked_bump(connection: sqlite3.Connection) -> int:
        observed_transactions.append(connection.in_transaction)
        return original_bump(connection)

    monkeypatch.setattr(store, "_bump_records_version", checked_bump)
    record = _record("atomic-projection")
    store.insert_record(record)
    store.update_status(record.id, "provisional")
    store.set_supersedes(record.id, None)
    store.reinforce_fields(
        record.id,
        confidence=0.96,
        reinforcements=1,
        last_reinforced_at=_NOW,
        expires_at=None,
        status="confirmed",
    )
    store.put_embedding(record.id, "fake", "1", np.array([1.0, 0.0], dtype=np.float32))

    assert observed_transactions == [True] * 5


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


def _legacy_store_with_unmapped_record(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(_legacy_schema())
        connection.execute(
            "CREATE TABLE schema_version("
            "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        connection.execute("INSERT INTO schema_version(version) VALUES (1)")
        connection.execute(
            """
            INSERT INTO records(
                id, type, version, content, subject, scope_kind, scope_id, source_kind, creator_agent_id,
                created_at, event_at, confidence, status, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "unmapped-record",
                "semantic",
                1,
                "Aditya prefers concise answers.",
                "person:aditya/Explanation Style",
                "user",
                "aditya",
                "user_statement",
                "research-agent",
                _NOW.isoformat(),
                _NOW.isoformat(),
                0.95,
                "confirmed",
                "[]",
            ),
        )
        connection.commit()
    finally:
        connection.close()


def test_store_refuses_to_open_with_unmapped_legacy_subjects_unless_allowed(tmp_path: Path) -> None:
    from memory_weave.store.migrations import MigrationIssuesError

    path = tmp_path / "legacy.sqlite"
    _legacy_store_with_unmapped_record(path)

    with pytest.raises(MigrationIssuesError):
        _ = Store(path).connection

    store = Store(path, allow_migration_issues=True)
    assert store.migration_issues() == [("unmapped-record", "expected exactly one about entity")]
    assert store.expire_migration_issues("admin") == ["unmapped-record"]
    assert store.get_record("unmapped-record").status == "expired"  # type: ignore[union-attr]
    assert store.migration_issues() == []
    store.close()

    reopened = Store(path)
    assert reopened.get_record("unmapped-record") is not None
    reopened.close()


def test_migrate_cli_lists_and_can_expire_unmapped_legacy_subjects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "legacy.sqlite"
    _legacy_store_with_unmapped_record(path)

    assert main(["--store", str(path), "migrate"]) == 2
    assert "unmapped unmapped-record: expected exactly one about entity" in capsys.readouterr().out
    assert main(["--store", str(path), "migrate", "--expire-unmapped"]) == 0
    assert main(["--store", str(path), "migrate"]) == 0


def test_migration_two_normalizes_a_legacy_attribute(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "legacy-attr.sqlite")
    try:
        connection.executescript(_legacy_schema())
        connection.execute(
            "CREATE TABLE schema_version("
            "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        connection.execute("INSERT INTO schema_version(version) VALUES (1)")
        connection.execute(
            "INSERT INTO entities(id, kind, canonical, scope_kind, scope_id, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("entity-aditya", "person", "Aditya", "user", "aditya", "confirmed", _NOW.isoformat()),
        )
        connection.execute(
            """
            INSERT INTO records(
                id, type, version, content, subject, scope_kind, scope_id, source_kind, creator_agent_id,
                created_at, event_at, confidence, status, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-record",
                "semantic",
                1,
                "Aditya prefers concise answers.",
                "person:aditya/Explanation Style",
                "user",
                "aditya",
                "user_statement",
                "research-agent",
                _NOW.isoformat(),
                _NOW.isoformat(),
                0.95,
                "confirmed",
                "[]",
            ),
        )
        connection.execute(
            "INSERT INTO record_entities(record_id, entity_id, role) VALUES (?, ?, ?)",
            ("legacy-record", "entity-aditya", "about"),
        )
        connection.execute(
            "INSERT INTO records_fts(record_id, content, subject, aliases) VALUES (?, ?, ?, ?)",
            ("legacy-record", "Aditya prefers concise answers.", "person:aditya/Explanation Style", "aditya"),
        )

        assert migrate(connection).unmapped_subject_records == 0
        row = connection.execute("SELECT subject_entity_id, attribute, subject FROM records").fetchone()
        assert row == ("entity-aditya", "explanation_style", "entity-aditya/explanation_style")
    finally:
        connection.close()


def test_merge_entity_rebuilds_the_fts_alias_column_for_moved_records(store: Store) -> None:
    short = store.create_entity(kind="person", canonical="Adi", scope=_USER_SCOPE, entity_id="adi")
    store.add_alias(short.id, "adi")
    full = store.create_entity(kind="person", canonical="Aditya Mishra", scope=_USER_SCOPE, entity_id="aditya-mishra")
    store.add_alias(full.id, "aditya mishra")
    store.add_alias(full.id, "aditya")
    record = _record("about-adi")
    store.insert_record(record)
    store.upsert_fts(record.id, record.content, record.subject, "adi")
    store.link_record_entity(record.id, short.id, "about")

    store.merge_entity(short.id, full.id)

    aliases = store.fts_rows([record.id])[record.id][2].split()
    assert set(aliases) == {"adi", "aditya", "mishra"}
    assert store.fts_query('"mishra"', limit=5)[0][0] == record.id


def test_records_for_entities_uses_a_temp_table_for_a_long_entity_list(store: Store) -> None:
    entity = store.create_entity(kind="project", canonical="Memory Weave", scope=_USER_SCOPE, entity_id="memory-weave")
    record = _record("long-entity-query")
    store.insert_record(record)
    store.link_record_entity(record.id, entity.id)
    entity_ids = [f"unmatched-{index}" for index in range(1_200)] + [entity.id]

    assert store.records_for_entities(entity_ids, {record.id}, 5) == [(record.id, entity.id)]
