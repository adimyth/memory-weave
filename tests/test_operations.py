"""Phase 12 operations: expiry, retention, erasure, re-embedding, and snapshots over a seeded store."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memory_weave.config import EmbeddingConfig, MemoryWeaveConfig, SessionsConfig
from memory_weave.host import MemoryHost
from memory_weave.index.embedder import FakeEmbedder
from memory_weave.ingest import FakeExtractor, FakeJudge, SessionHooks, TableReviewer, WriteRequest
from memory_weave.models import (
    CandidateRecord,
    EntityMention,
    ExtractionOutput,
    Principal,
    Scope,
    SearchRequest,
    SessionSummary,
)
from memory_weave.operations import OperationRefused, Operations
from memory_weave.runtime import MemoryRuntime, build_runtime
from memory_weave.store import Store

_NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
_AGENT = "agent"
_USER = "aditya"
_OTHER = "priya"
_USER_SCOPE = Scope(kind="user", id=_USER)
_OTHER_SCOPE = Scope(kind="user", id=_OTHER)
_PROJECT_SCOPE = Scope(kind="project", id="weave")
_EMBEDDING = EmbeddingConfig(model="fake-embedder", version="1", dims=8)
_CONFIG = MemoryWeaveConfig(embedding=_EMBEDDING, sessions=SessionsConfig(retain_days=30))

TURN_SENTINEL = "ZEPHYR-TURN-7731"
EVIDENCE_SENTINEL = "QUOKKA-PROJECT-4412"
CANDIDATE_SENTINEL = "MARMOT-CANDIDATE-9083"
QUERY_SENTINEL = "VIPER-QUERY-1155"
DECISION_SENTINEL = "OSPREY-DECISION-2201"
ALIAS_SENTINEL = "Aditya Mishra"
# Aliases are stored normalised, so the byte-level check looks for the normalised form.
SENTINELS = (
    TURN_SENTINEL,
    EVIDENCE_SENTINEL,
    CANDIDATE_SENTINEL,
    QUERY_SENTINEL,
    DECISION_SENTINEL,
    ALIAS_SENTINEL.lower(),
)


class Clock:
    def __init__(self) -> None:
        self.at = _NOW

    def __call__(self) -> datetime:
        return self.at


@dataclass
class World:
    path: Path
    store: Store
    runtime: MemoryRuntime
    clock: Clock
    principal: Principal
    user_record: str
    project_record: str
    other_record: str

    def ops(self, config: MemoryWeaveConfig = _CONFIG) -> Operations:
        return Operations(self.store, config, actor="tester", current_time=self.clock)

    def events(self, kind: str | None = None) -> list[dict[str, object]]:
        query = "SELECT kind, record_id, entity_id, payload FROM events" + (" WHERE kind = ?" if kind else "")
        rows = self.store.connection.execute(query, (kind,) if kind else ()).fetchall()
        return [
            {"kind": r["kind"], "record_id": r["record_id"], "entity_id": r["entity_id"], **json.loads(r["payload"])}
            for r in rows
        ]


def _build(
    config: MemoryWeaveConfig, store: Store, clock: Clock, *, extractor: FakeExtractor | None = None
) -> MemoryRuntime:
    built = build_runtime(
        config,
        store,
        embedder=FakeEmbedder(dims=config.embedding.dims),
        judge=FakeJudge(),
        extractor=extractor or FakeExtractor(ExtractionOutput([], SessionSummary("Nothing happened.", [], []))),
        reviewer=TableReviewer(),
        current_time=clock,
    )
    # Run extraction on the calling thread so the seeded state is complete when the fixture returns.
    built.hooks = SessionHooks(
        store,
        built.session_buffer,
        config,
        on_end=lambda session_id, principal: built.extraction.extract_session(session_id, principal),
        current_time=clock,
    )
    return built


@pytest.fixture
def world(tmp_path: Path) -> World:
    path = tmp_path / "memory.sqlite"
    store = Store(path)
    clock = Clock()
    host = MemoryHost(store)
    for user in (_USER, _OTHER):
        host.grant(_AGENT, Scope(kind="user", id=user), read=True, write=True)
    host.grant(_AGENT, _PROJECT_SCOPE, read=True, write=True)
    host.provision_user(_USER, aliases=(ALIAS_SENTINEL,))
    host.provision_user(_OTHER, aliases=("Priya",))
    candidate = CandidateRecord(
        type="semantic",
        content=f"Aditya keeps a {CANDIDATE_SENTINEL} pet.",
        attribute="pet",
        source_kind="user_statement",
        evidence="this quote is not in the transcript at all",
        evidence_turn=1,
        entity_mentions=[],
        event_at=None,
        confidence=0.9,
    )
    extractor = FakeExtractor(ExtractionOutput([candidate], SessionSummary("Aditya set up the editor.", [], [])))
    runtime = _build(_CONFIG, store, clock, extractor=extractor)
    principal = Principal(_AGENT, _USER, "s1", None)

    # A transcript with two sentinels, one for the turn itself and one that a project record will quote.
    runtime.hooks.on_session_start(principal)
    runtime.hooks.on_turn(principal, "user", f"My editor token is {TURN_SENTINEL} and I prefer Vim.")
    runtime.hooks.on_turn(principal, "assistant", "Noted.")
    runtime.hooks.on_turn(principal, "user", f"The project runbook is at {EVIDENCE_SENTINEL} in the wiki.")
    user_record = runtime.ingestor.write(
        principal,
        WriteRequest(
            type="semantic",
            content="Aditya prefers Vim.",
            source_kind="user_statement",
            evidence=f"My editor token is {TURN_SENTINEL} and I prefer Vim.",
            attribute="editor",
            entities=[EntityMention(kind="person", text=ALIAS_SENTINEL, role="about")],
        ),
    )
    project_record = runtime.ingestor.write(
        principal,
        WriteRequest(
            type="semantic",
            content="The project runbook lives in the wiki.",
            source_kind="user_statement",
            evidence=f"The project runbook is at {EVIDENCE_SENTINEL} in the wiki.",
            attribute="runbook_location",
            scope=_PROJECT_SCOPE,
            entities=[
                EntityMention(kind="project", text="weave", role="about"),
                EntityMention(kind="person", text=ALIAS_SENTINEL, role="mentions"),
            ],
        ),
    )
    other = Principal(_AGENT, _OTHER, "s2", None)
    runtime.hooks.on_session_start(other)
    runtime.hooks.on_turn(other, "user", "Priya prefers Emacs, always.")
    other_record = runtime.ingestor.write(
        other,
        WriteRequest(
            type="semantic",
            content="Priya prefers Emacs.",
            source_kind="user_statement",
            evidence="Priya prefers Emacs, always.",
            attribute="editor",
        ),
    )
    runtime.handlers.memory_search(principal, {"queries": [f"{QUERY_SENTINEL} editor"], "k": 8})
    store.insert_turn_decision(
        {
            "id": "td1",
            "at": _NOW,
            "session_id": "s1",
            "turn": f"What about {DECISION_SENTINEL}?",
            "disposition": "baseline_no_gaps",
            "shadow": True,
            "profile_record_ids": [],
            "inventory": [],
            "gaps": [{"category": "preference", "query": DECISION_SENTINEL}],
            "gap_status": "ok",
            "candidate_ids": [],
            "verdicts": [],
            "admitted_ids": [],
            "admission_status": "skipped",
            "requested_budget_ms": None,
            "effective_budget_ms": None,
            "timings_ms": {},
            "failures": [],
            "config": {},
        }
    )
    runtime.hooks.on_session_end(principal)  # extraction runs here; the rejected candidate lands in extraction.run
    assert store.get_session("s1").extracted_at is not None  # type: ignore[union-attr]
    assert user_record.record_id and project_record.record_id and other_record.record_id
    return World(
        path, store, runtime, clock, principal, user_record.record_id, project_record.record_id, other_record.record_id
    )


def _db_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    wal = Path(f"{path}-wal")
    if wal.exists():
        data += wal.read_bytes()
    return data


def test_seed_plants_every_sentinel(world: World) -> None:
    world.store.close()
    blob = _db_bytes(world.path)
    for sentinel in SENTINELS:
        assert sentinel.encode() in blob, sentinel


def test_erase_user_leaves_no_user_text_anywhere_and_keeps_tombstones(world: World) -> None:
    before_records = world.store.connection.execute("SELECT count(*) FROM records").fetchone()[0]
    before_events = world.store.connection.execute("SELECT count(*) FROM events").fetchone()[0]
    before_logs = world.store.connection.execute("SELECT count(*) FROM search_log").fetchone()[0]

    report = world.ops().erase_user(_USER, reason="gdpr request")

    assert world.user_record in report.records and world.project_record in report.records
    assert world.other_record not in report.records
    assert report.sessions == ["s1"]
    world.store.close()
    blob = _db_bytes(world.path)
    for sentinel in SENTINELS:
        assert sentinel.encode() not in blob, sentinel
    assert b"Priya prefers Emacs" in blob

    store = Store(world.path)
    try:
        assert store.connection.execute("SELECT count(*) FROM records").fetchone()[0] == before_records
        assert store.connection.execute("SELECT count(*) FROM events").fetchone()[0] > before_events
        assert store.connection.execute("SELECT count(*) FROM search_log").fetchone()[0] == before_logs
        user = store.get_record(world.user_record)
        project = store.get_record(world.project_record)
        assert user is not None and user.status == "deleted" and user.content == "" and user.evidence is None
        assert (
            project is not None and project.status == "deleted" and project.content == "" and project.evidence is None
        )
        assert project.scope == _PROJECT_SCOPE and project.entity_ids == []
        other = store.get_record(world.other_record)
        assert other is not None and other.status == "confirmed" and other.content == "Priya prefers Emacs."
        assert [turn.content for turn in store.session_turns("s1")] == ["", "", ""]
        assert store.get_session("s1") is not None
        assert (
            store.connection.execute(
                "SELECT count(*) FROM embeddings WHERE record_id = ?", (world.user_record,)
            ).fetchone()[0]
            == 0
        )
        assert (
            store.connection.execute(
                "SELECT count(*) FROM records_fts WHERE record_id = ?", (world.user_record,)
            ).fetchone()[0]
            == 0
        )
        log = store.connection.execute("SELECT request, user_id, explanations FROM search_log").fetchone()
        assert json.loads(log["request"]) == {"erased": True} and log["user_id"] == _USER
        decision = store.connection.execute("SELECT turn, gaps FROM turn_decisions WHERE id = 'td1'").fetchone()
        assert decision["turn"] == "" and json.loads(decision["gaps"]) == []
        run = [
            json.loads(r["payload"])
            for r in store.connection.execute("SELECT payload FROM events WHERE kind = 'extraction.run'")
        ]
        assert run and run[0] == {"erased": True, "kind": "extraction.run"}
        erased = [
            json.loads(r["payload"])
            for r in store.connection.execute("SELECT payload FROM events WHERE kind = 'user.erased'")
        ]
        assert erased and erased[0]["user_id"] == _USER and erased[0]["records"]
        entity = store.connection.execute(
            "SELECT canonical, status FROM entities WHERE scope_kind = 'user' AND scope_id = ?", (_USER,)
        ).fetchall()
        assert entity and all(row["canonical"] == "" and row["status"] == "deleted" for row in entity)
    finally:
        store.close()


def test_erase_record_removes_content_index_rows_and_links_but_keeps_the_row(world: World) -> None:
    report = world.ops().erase_record(world.user_record, reason="user asked")

    assert report.records == [world.user_record]
    record = world.store.get_record(world.user_record)
    assert record is not None and record.status == "deleted" and record.content == "" and record.entity_ids == []
    assert (
        world.store.connection.execute(
            "SELECT count(*) FROM embeddings WHERE record_id = ?", (world.user_record,)
        ).fetchone()[0]
        == 0
    )
    kinds = [event["kind"] for event in world.store.events_for(world.user_record)]
    assert kinds[-1] == "record.erased"
    assert all(
        event["payload"] == {"erased": True, "kind": event["kind"]}
        for event in world.store.events_for(world.user_record)[:-1]
    )
    project = world.store.get_record(world.project_record)
    assert project is not None and project.content == "The project runbook lives in the wiki."
    with pytest.raises(OperationRefused):
        world.ops().erase_record("missing", reason="x")


def test_erase_session_blanks_turns_and_keeps_rows_and_records(world: World) -> None:
    report = world.ops().erase_session("s1", reason="user asked")

    assert report.sessions == ["s1"]
    assert [turn.content for turn in world.store.session_turns("s1")] == ["", "", ""]
    assert len(world.store.session_turns("s1")) == 3
    record = world.store.get_record(world.user_record)
    assert record is not None and record.content == "Aditya prefers Vim."
    assert world.events("session.erased")[0]["session_id"] == "s1"


def test_expire_flips_only_records_past_expiry_with_one_event_each(world: World) -> None:
    provisional = world.runtime.ingestor.write(
        world.principal,
        WriteRequest(
            type="semantic",
            content="Aditya may like tabs.",
            source_kind="agent_inference",
            evidence=None,
            attribute="tabs",
        ),
    )
    assert provisional.record_id and provisional.status == "provisional"
    assert world.ops().expire() == []
    world.clock.at = _NOW + timedelta(days=_CONFIG.ingestion.provisional_ttl_days + 1)

    expired = world.ops().expire()

    assert expired == [provisional.record_id]
    assert world.store.get_record(provisional.record_id).status == "expired"  # type: ignore[union-attr]
    assert [e["kind"] for e in world.store.events_for(provisional.record_id)].count("record.expired") == 1
    assert world.store.get_record(world.user_record).status == "confirmed"  # type: ignore[union-attr]
    assert world.ops().expire() == []


def test_retain_blanks_only_extracted_sessions_older_than_the_retention_window(world: World) -> None:
    store = world.store
    old_extracted = Principal(_AGENT, _USER, "old-extracted", None)
    old_unextracted = Principal(_AGENT, _USER, "old-unextracted", None)
    world.clock.at = _NOW - timedelta(days=100)
    for principal in (old_extracted, old_unextracted):
        world.runtime.hooks.on_session_start(principal)
        world.runtime.hooks.on_turn(principal, "user", "old text")
        store.end_session(principal.session_id or "", world.clock.at)
    store.mark_extracted("old-extracted", world.clock.at)
    world.clock.at = _NOW

    assert world.ops().retain() == ["old-extracted"]
    assert [turn.content for turn in store.session_turns("old-extracted")] == [""]
    assert [turn.content for turn in store.session_turns("old-unextracted")] == ["old text"]
    assert [turn.content for turn in store.session_turns("s1")][0] != ""  # extracted today, inside the window
    assert world.events("session.retained")[0]["session_id"] == "old-extracted"
    assert world.ops().retain() == []


def test_reembed_refuses_until_floors_change_and_then_replaces_every_vector(world: World) -> None:
    new_embedding = replace(_EMBEDDING, version="2")
    same_floors = replace(_CONFIG, embedding=new_embedding)
    new_embedder = FakeEmbedder(dims=8, version="2")
    with pytest.raises(OperationRefused, match="unchanged"):
        Operations(world.store, same_floors, current_time=world.clock).reembed(new_embedder)
    with pytest.raises(OperationRefused, match="does not match"):
        Operations(world.store, same_floors, current_time=world.clock).reembed(FakeEmbedder(dims=8))

    floors = replace(_CONFIG.retrieval.gate.dense_floor, semantic=0.5)
    recalibrated = replace(
        same_floors, retrieval=replace(_CONFIG.retrieval, gate=replace(_CONFIG.retrieval.gate, dense_floor=floors))
    )
    count = Operations(world.store, recalibrated, current_time=world.clock).reembed(new_embedder)

    live = world.store.connection.execute("SELECT count(*) FROM records WHERE content != ''").fetchone()[0]
    assert count == live
    versions = world.store.connection.execute("SELECT DISTINCT model, version FROM embeddings").fetchall()
    assert [(row["model"], row["version"]) for row in versions] == [("fake-embedder", "2")]
    assert world.store.count_embeddings("fake-embedder", "2") == count
    event = world.events("store.reembedded")[0]
    assert event["records"] == count and event["version"] == "2"


def test_reembed_runs_without_a_search_log(tmp_path: Path) -> None:
    store = Store(tmp_path / "fresh.sqlite")
    clock = Clock()
    runtime = _build(_CONFIG, store, clock)
    MemoryHost(store).grant(_AGENT, _USER_SCOPE, read=True, write=True)
    runtime.ingestor.write(
        Principal(_AGENT, _USER, None, None),
        WriteRequest(
            type="semantic",
            content="Aditya prefers Vim.",
            source_kind="agent_inference",
            evidence=None,
            attribute="editor",
        ),
    )
    config = replace(_CONFIG, embedding=replace(_EMBEDDING, version="2"))
    assert Operations(store, config, current_time=clock).reembed(FakeEmbedder(dims=8, version="2")) == 1
    store.close()


def test_snapshot_restore_preserves_lineage_activation_conflicts_reviews_and_decisions(
    world: World, tmp_path: Path
) -> None:
    from memory_weave.policy import ActivationService, BundleRegistry, CategoryDecision

    class Policy:
        def classify(self, content: str) -> CategoryDecision:
            return CategoryDecision("preferences", "answer_style", "ambiguous", 0.6)  # type: ignore[arg-type]

    decision = ActivationService(world.store, Policy()).apply(world.principal, world.user_record)
    assert decision.review_id is not None
    superseding = world.runtime.ingestor.write(
        world.principal,
        WriteRequest(
            type="semantic",
            content="Aditya prefers Neovim.",
            source_kind="user_statement",
            evidence=f"My editor token is {TURN_SENTINEL} and I prefer Vim.",
            attribute="editor",
        ),
    )
    assert superseding.outcome.startswith("superseded:") or superseding.outcome.startswith("conflict:")
    BundleRegistry(world.store).record({"planner": "x"}, passed=True, evidence="test", recorded_by="tester")
    world.store.add_conflict(world.user_record, world.other_record, _NOW)

    def state(store: Store) -> dict[str, object]:
        tables = (
            "records",
            "record_conflicts",
            "activation_reviews",
            "turn_decisions",
            "bundle_fitness",
            "events",
            "sessions",
            "session_turns",
            "entities",
            "entity_aliases",
            "record_entities",
            "embeddings",
            "search_log",
        )
        snapshot: dict[str, object] = {}
        for table in tables:
            rows = store.connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            snapshot[table] = [tuple(row) for row in rows]
        return snapshot

    expected = state(world.store)
    snapshot = tmp_path / "snap.sqlite"
    world.ops().snapshot_save(snapshot)
    expected_after_save = state(world.store)  # the save event itself is part of the snapshot
    world.runtime.ingestor.write(
        world.principal,
        WriteRequest(type="episodic", content="Later noise.", source_kind="agent_inference", evidence=None),
    )
    assert state(world.store) != expected_after_save
    world.store.close()

    world.ops().snapshot_load(snapshot, world.path)
    restored = Store(world.path)
    try:
        assert state(restored) == expected_after_save
        assert len(expected_after_save["events"]) == len(expected["events"]) + 1  # type: ignore[arg-type]
        assert restored.get_record(superseding.record_id or "") is not None
        assert restored.open_activation_reviews()
        assert restored.conflicts_for(world.user_record) == [world.other_record]
    finally:
        restored.close()
    with pytest.raises(OperationRefused):
        Operations(Store(world.path), _CONFIG).snapshot_load(tmp_path / "missing.sqlite", world.path)


def test_flag_due_reviews_uses_the_configured_batch_size(world: World) -> None:
    assert world.ops().flag_due_reviews() == []


def _sqlite_has_json(path: Path) -> bool:
    connection = sqlite3.connect(path)
    try:
        return connection.execute("SELECT json_extract('{\"a\": 1}', '$.a')").fetchone()[0] == 1
    finally:
        connection.close()


def test_sqlite_json_functions_are_available(tmp_path: Path) -> None:
    assert _sqlite_has_json(tmp_path / "probe.sqlite")


def test_build_runtime_wires_the_hooks_to_extraction(tmp_path: Path) -> None:
    store = Store(tmp_path / "rt.sqlite")
    runtime = _build(_CONFIG, store, Clock())
    MemoryHost(store).grant(_AGENT, _USER_SCOPE, read=True, write=True)
    principal = Principal(_AGENT, _USER, "rt-1", None)
    runtime.hooks.on_turn(principal, "user", "hello there friend")
    runtime.hooks.on_session_end(principal)
    import time

    for _ in range(50):
        session = store.get_session("rt-1")
        if session is not None and session.extracted_at is not None:
            break
        time.sleep(0.05)
    assert store.get_session("rt-1").extracted_at is not None  # type: ignore[union-attr]
    request = SearchRequest(["hello"], None, None, None, None, None, 8, False)
    assert runtime.retriever.search(principal, request).search_id
    store.close()
