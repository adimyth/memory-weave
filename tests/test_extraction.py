"""Phase 10: session extraction with review before apply, on the fake extractor and the real ingestor."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from memory_weave.config import EmbeddingConfig, IngestionConfig, MemoryWeaveConfig
from memory_weave.index.embedder import FakeEmbedder
from memory_weave.index.vector import VectorIndex
from memory_weave.ingest import (
    ExtractionRunner,
    FakeExtractor,
    FakeJudge,
    Ingestor,
    ReviewDecision,
    ReviewRequest,
    SessionBuffer,
    TableReviewer,
)
from memory_weave.ingest import ingestor as ingestor_module
from memory_weave.models import (
    CandidateRecord,
    EntityMention,
    ExtractionOutput,
    Principal,
    Record,
    Scope,
    SearchRequest,
    SessionSummary,
    Turn,
)
from memory_weave.policy import ActivationService, CategoryDecision
from memory_weave.retrieve import Retriever
from memory_weave.store import Store
from memory_weave.util import render_subject

_NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
_AGENT_ID = "research-agent"
_USER_ID = "aditya"
_SESSION_ID = "session-1"
_PRINCIPAL = Principal(_AGENT_ID, _USER_ID, _SESSION_ID, None)
_USER_SCOPE = Scope(kind="user", id=_USER_ID)
_PRIVATE_SCOPE = Scope(kind="agent", id=f"{_AGENT_ID}/{_USER_ID}")
_EMBEDDING = EmbeddingConfig(model="fake-embedder", version="1", dims=8)

_TRANSCRIPT: list[tuple[str, str]] = [
    ("user", "Hi. I prefer concise technical explanations with code examples."),
    ("assistant", "Understood. I will keep explanations concise and include code examples."),
    ("user", "Rohan Mehta is my manager and he runs the platform team."),
    ("assistant", "Noted about Rohan Mehta."),
    ("tool", "The deployment completed successfully in 4 minutes."),
    ("user", "Until Friday please answer in Spanish because I am practising."),
    ("assistant", "Claro, responderé en español hasta el viernes."),
    ("user", "We decided to keep SQLite as the store for now."),
    ("assistant", "SQLite it is. I would suggest revisiting that decision next quarter."),
    ("user", "Thanks, that is all for today."),
]

C_STYLE = "Aditya prefers concise technical explanations with code examples."
C_ROHAN = "Rohan Mehta runs the platform team."
C_SPANISH = "Aditya wants answers in Spanish until Friday."
C_SQLITE = "Aditya decided to keep SQLite as the store for now."
_FRIDAY = datetime(2026, 9, 11, 23, 59, tzinfo=UTC)


class Clock:
    def __init__(self, at: datetime) -> None:
        self.at = at

    def __call__(self) -> datetime:
        return self.at

    def advance(self, **kwargs: float) -> None:
        self.at = self.at + timedelta(**kwargs)


def _candidate(
    content: str,
    evidence: str,
    turn: int,
    *,
    memory_type: str = "semantic",
    attribute: str | None = "explanation_style",
    source_kind: str = "user_statement",
    entities: list[EntityMention] | None = None,
    **temporal: datetime | None,
) -> CandidateRecord:
    return CandidateRecord(
        type=memory_type,  # type: ignore[arg-type]
        content=content,
        attribute=attribute,
        source_kind=source_kind,  # type: ignore[arg-type]
        evidence=evidence,
        evidence_turn=turn,
        entity_mentions=entities or [],
        event_at=None,
        confidence=0.9,
        **temporal,
    )


def _summary(content: str = "Aditya set an explanation style, named a manager, and chose SQLite.") -> SessionSummary:
    return SessionSummary(
        content=content,
        decisions=["Keep SQLite as the store for now."],
        entity_mentions=[EntityMention(kind="person", text="Rohan Mehta", role="mentions")],
    )


def _good_candidates() -> list[CandidateRecord]:
    return [
        _candidate(C_STYLE, "I prefer concise technical explanations with code examples.", 1),
        _candidate(
            C_ROHAN,
            "Rohan Mehta is my manager and he runs the platform team.",
            3,
            attribute="team",
            entities=[EntityMention(kind="person", text="Rohan Mehta", role="about")],
        ),
        _candidate(
            C_SPANISH,
            "Until Friday please answer in Spanish",
            6,
            attribute="response_language",
            valid_until=_FRIDAY,
        ),
        _candidate(
            C_SQLITE, "We decided to keep SQLite as the store for now.", 8, memory_type="episodic", attribute=None
        ),
    ]


def _output(candidates: list[CandidateRecord] | None = None, summary: SessionSummary | None = None) -> ExtractionOutput:
    return ExtractionOutput(
        candidates=_good_candidates() if candidates is None else candidates, summary=summary or _summary()
    )


@pytest.fixture
def clock() -> Clock:
    return Clock(_NOW)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    database = Store(tmp_path / "memory.sqlite")
    _ = database.connection
    database.create_session(_SESSION_ID, _AGENT_ID, _USER_ID, None, _NOW - timedelta(minutes=30))
    database.set_grant(_AGENT_ID, _USER_SCOPE, can_read=True, can_write=True)
    yield database
    database.close()


@pytest.fixture
def buffer(store: Store) -> SessionBuffer:
    session_buffer = SessionBuffer(store)
    for number, (role, content) in enumerate(_TRANSCRIPT, start=1):
        at = _NOW - timedelta(minutes=30 - number)
        session_buffer.append_turn(Turn(_SESSION_ID, number, role, content, at))  # type: ignore[arg-type]
    store.end_session(_SESSION_ID, _NOW - timedelta(minutes=1))
    return session_buffer


@pytest.fixture
def config() -> MemoryWeaveConfig:
    return MemoryWeaveConfig(embedding=_EMBEDDING, ingestion=IngestionConfig(dedup_candidate_cosine=0.80))


class World:
    def __init__(self, store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig, clock: Clock) -> None:
        self.store = store
        self.buffer = buffer
        self.config = config
        self.clock = clock
        self.judge = FakeJudge()
        self.embedder = FakeEmbedder(dims=config.embedding.dims)
        self.ingestor = Ingestor(
            store, VectorIndex(config.embedding), self.embedder, self.judge, buffer, config, current_time=clock
        )

    def runner(
        self,
        extractor: FakeExtractor | None = None,
        reviewer: TableReviewer | None = None,
        *,
        activation: ActivationService | None = None,
    ) -> ExtractionRunner:
        return ExtractionRunner(
            self.store,
            self.ingestor,
            extractor or FakeExtractor(_output()),
            reviewer or TableReviewer(),
            self.buffer,
            self.config,
            activation=activation,
            current_time=self.clock,
        )

    def events(self, kind_prefix: str) -> list[dict[str, Any]]:
        rows = self.store.connection.execute(
            "SELECT kind, actor, record_id, payload FROM events WHERE kind LIKE ? ORDER BY at, id", (f"{kind_prefix}%",)
        ).fetchall()
        return [
            {"kind": row["kind"], "actor": row["actor"], "record_id": row["record_id"], **json.loads(row["payload"])}
            for row in rows
        ]

    def records(self) -> list[Record]:
        ids = [row["id"] for row in self.store.connection.execute("SELECT id FROM records ORDER BY created_at, id")]
        return self.store.get_records(ids)


@pytest.fixture
def world(store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig, clock: Clock) -> World:
    return World(store, buffer, config, clock)


def _run_payload(world: World) -> dict[str, Any]:
    runs = world.events("extraction.run")
    assert runs, "no extraction.run event"
    return runs[-1]


def test_full_run_writes_reviewed_candidates_and_one_summary(world: World) -> None:
    result = world.runner().extract_session(_SESSION_ID, _PRINCIPAL)

    assert result.status == "completed"
    outcomes = {outcome.content: outcome for outcome in result.candidates}
    assert {content: o.outcome for content, o in outcomes.items()} == {
        C_STYLE: "created",
        C_ROHAN: "created",
        C_SPANISH: "created",
        C_SQLITE: "created",
    }
    assert all(outcome.review_outcome == "accept" for outcome in outcomes.values())
    records = {record.content: record for record in world.records()}
    assert records[C_SPANISH].valid_until == _FRIDAY
    assert records[C_ROHAN].subject.endswith("/team")
    assert world.store.get_session(_SESSION_ID).extracted_at == _NOW  # type: ignore[union-attr]

    payload = _run_payload(world)
    assert payload["prompt_version"] == "fake-v1"
    assert payload["reviewer_version"] == "fake-review-v1"
    assert payload["counts"]["written"] == 4
    assert payload["counts"]["rejected"] == 0
    for stage in (
        "transcript_prep",
        "extractor_model",
        "validation",
        "candidate_review",
        "writes",
        "summary_write",
        "dedup_and_contradiction",
        "total",
    ):
        assert stage in payload["timings_ms"]
    assert payload["summary"]["record_id"] == result.summary_record_id


def test_summary_is_confirmed_episodic_with_configured_expiry_and_mention_links(world: World) -> None:
    result = world.runner().extract_session(_SESSION_ID, _PRINCIPAL)

    summary = world.store.get_record(result.summary_record_id or "")
    assert summary is not None
    assert summary.source_kind == "session_summary"
    assert summary.type == "episodic"
    assert summary.status == "confirmed"
    assert summary.confidence == 0.8
    assert summary.source_ref == f"session:{_SESSION_ID}"
    assert summary.event_at == _NOW - timedelta(minutes=1)
    assert summary.expires_at == _NOW + timedelta(days=world.config.ingestion.summary_ttl_days)
    assert "Decisions: Keep SQLite as the store for now." in summary.content
    roles = world.store.record_entity_roles(summary.id)
    assert list(roles.values()).count("about") == 1
    assert summary.subject_entity_id is not None and roles[summary.subject_entity_id] == "about"
    rohan = [entity_id for entity_id, role in roles.items() if role == "mentions"]
    assert len(rohan) == 1
    assert world.store.get_entity(rohan[0]).canonical == "Rohan Mehta"  # type: ignore[union-attr]


def test_retriever_finds_extracted_records_and_summary_only_for_tool_searches(world: World) -> None:
    result = world.runner().extract_session(_SESSION_ID, _PRINCIPAL)
    retriever = Retriever(
        world.store, VectorIndex(world.config.embedding), world.embedder, world.config, current_time=world.clock
    )

    def search(trigger: str) -> set[str]:
        request = SearchRequest(
            queries=["Aditya set an explanation style, named a manager, and chose SQLite."],
            context=None,
            types=None,
            entities=None,
            since=None,
            until=None,
            k=8,
            include_history=False,
            trigger=trigger,  # type: ignore[arg-type]
        )
        return {entry.record.id for entry in retriever.search(_PRINCIPAL, request).results}

    assert result.summary_record_id in search("tool")
    assert result.summary_record_id not in search("auto")
    sqlite_ids = {r.id for r in world.records() if r.content == C_SQLITE}
    assert sqlite_ids and sqlite_ids <= search("tool")


def test_candidate_with_evidence_absent_from_the_transcript_is_rejected_and_logged(world: World) -> None:
    bad = _candidate("Aditya uses Windows.", "I use Windows for everything at work.", 1)
    runner = world.runner(FakeExtractor(_output([bad])))

    result = runner.extract_session(_SESSION_ID, _PRINCIPAL)

    assert [(o.stage, o.outcome) for o in result.candidates] == [("validation", "evidence_not_found")]
    assert not [r for r in world.records() if r.source_kind != "session_summary"]
    payload = _run_payload(world)
    assert payload["rejection_reasons"] == {"evidence_not_found": 1}
    assert payload["candidates"][0]["content"] == "Aditya uses Windows."
    assert payload["candidates"][0]["detail"]["evidence"] == "I use Windows for everything at work."


def test_reviewer_rejection_never_reaches_the_ingestor(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[str] = []
    original = world.ingestor.write

    def spy(principal: Principal, request: Any) -> Any:
        writes.append(request.content)
        return original(principal, request)

    monkeypatch.setattr(world.ingestor, "write", spy)
    reviewer = TableReviewer({C_ROHAN: ReviewDecision("reject", "assistant summary, not the user's words")})

    result = world.runner(reviewer=reviewer).extract_session(_SESSION_ID, _PRINCIPAL)

    rohan = next(o for o in result.candidates if o.content == C_ROHAN)
    assert (rohan.stage, rohan.outcome, rohan.review_reason) == (
        "review",
        "review_rejected",
        "assistant summary, not the user's words",
    )
    assert C_ROHAN not in writes
    assert sorted(writes) == sorted([C_STYLE, C_SPANISH, C_SQLITE])
    assert _run_payload(world)["counts"]["review_rejected"] == 1
    assert len(reviewer.requests) == 4


def test_reviewer_sees_evidence_turn_neighbours_and_live_records_for_the_same_subject(world: World) -> None:
    world.ingestor.write(
        _PRINCIPAL,
        ingestor_module.WriteRequest(
            type="semantic",
            content="Aditya likes long explanations.",
            source_kind="user_statement",
            evidence=None,
            attribute="explanation_style",
        ),
    )
    reviewer = TableReviewer()

    world.runner(reviewer=reviewer).extract_session(_SESSION_ID, _PRINCIPAL)

    request = next(r for r in reviewer.requests if r.candidate.content == C_STYLE)
    assert request.evidence_turn.turn == 1
    assert [turn.turn for turn in request.adjacent_turns] == [2]
    assert [record.content for record in request.live_records] == ["Aditya likes long explanations."]
    assert request.scope == _USER_SCOPE
    rohan = next(r for r in reviewer.requests if r.candidate.content == C_ROHAN)
    assert [turn.turn for turn in rohan.adjacent_turns] == [2, 4]
    assert rohan.live_records == []


def test_constrained_revision_is_written_and_escalations_are_rejected(world: World) -> None:
    narrowed = replace(_good_candidates()[0], content="Aditya prefers concise technical explanations.")
    stronger = replace(_good_candidates()[3], source_kind="user_statement")
    evidence_swap = replace(_good_candidates()[1], evidence="Noted about Rohan Mehta.", evidence_turn=4)
    entity_swap = replace(
        _good_candidates()[2], entity_mentions=[EntityMention(kind="person", text="Rohan Mehta", role="about")]
    )
    downgraded = replace(_good_candidates()[3], source_kind="agent_inference")
    candidates = [_good_candidates()[0], _good_candidates()[1], _good_candidates()[2], downgraded]
    reviewer = TableReviewer(
        {
            C_STYLE: ReviewDecision("revise", "drop the example clause", narrowed),
            C_SQLITE: ReviewDecision("revise", "raise authority", stronger),
            C_ROHAN: ReviewDecision("revise", "better quote", evidence_swap),
            C_SPANISH: ReviewDecision("revise", "attach to Rohan", entity_swap),
        }
    )

    result = world.runner(FakeExtractor(_output(candidates)), reviewer).extract_session(_SESSION_ID, _PRINCIPAL)

    by_index = {o.index: o for o in result.candidates}
    assert by_index[0].outcome == "created" and by_index[0].content == narrowed.content
    assert by_index[0].review_outcome == "revise"
    assert by_index[1].outcome == "invalid_review_revision" and by_index[1].reason == "revision replaced the evidence"
    assert by_index[2].outcome == "invalid_review_revision" and by_index[2].reason == "revision changed the entities"
    assert by_index[3].outcome == "invalid_review_revision" and by_index[3].reason == "revision changed the source kind"
    stored = {r.content for r in world.records() if r.source_kind != "session_summary"}
    assert stored == {narrowed.content}
    counts = _run_payload(world)["counts"]
    assert counts["review_revised"] == 1 and counts["written"] == 1


def test_revision_that_widens_or_invents_temporal_metadata_is_rejected(world: World) -> None:
    spanish = _good_candidates()[2]
    wider = replace(spanish, valid_until=_FRIDAY + timedelta(days=7))
    invented = replace(_good_candidates()[0], review_at=_NOW + timedelta(days=30))
    reviewer = TableReviewer(
        {
            C_SPANISH: ReviewDecision("revise", "extend", wider),
            C_STYLE: ReviewDecision("revise", "review later", invented),
        }
    )

    result = world.runner(reviewer=reviewer).extract_session(_SESSION_ID, _PRINCIPAL)

    outcomes = {o.content: o for o in result.candidates}
    assert outcomes[C_SPANISH].reason == "revision widened valid_until"
    assert outcomes[C_STYLE].reason == "revision delayed review_at"


def test_reviewer_narrowed_claim_is_entailment_checked_before_it_is_written(world: World) -> None:
    original = _good_candidates()[0]
    narrowed = replace(original, content="Aditya prefers explanations in Rust.")
    world.judge.set_entailment(original.evidence, narrowed.content, 0.1)
    reviewer = TableReviewer({C_STYLE: ReviewDecision("revise", "narrow", narrowed)})

    result = world.runner(FakeExtractor(_output([original])), reviewer).extract_session(_SESSION_ID, _PRINCIPAL)

    assert (original.evidence, narrowed.content) in world.judge.entail_calls
    written = next(o for o in result.candidates if o.stage == "write")
    record = world.store.get_record(written.record_id or "")
    assert record is not None
    assert record.content == narrowed.content
    assert record.source_kind == "agent_inference"
    assert record.status == "provisional"
    assert written.reason is not None and "evidence does not support claim" in written.reason


def test_reviewer_failure_writes_nothing_and_the_claim_can_be_reclaimed_after_the_timeout(world: World) -> None:
    reviewer = TableReviewer({C_ROHAN: TimeoutError("reviewer timed out")})
    runner = world.runner(reviewer=reviewer)

    failed = runner.extract_session(_SESSION_ID, _PRINCIPAL)

    assert failed.status == "failed"
    assert failed.reason is not None and failed.reason.startswith("reviewer: ReviewError")
    assert world.records() == []
    session = world.store.get_session(_SESSION_ID)
    assert session is not None and session.extracted_at is None and session.extraction_started_at == _NOW
    failures = world.events("extraction.failed")
    assert len(failures) == 1 and failures[0]["stage"] == "reviewer"
    assert not any(C_ROHAN in json.dumps(event) for event in failures)

    world.clock.advance(minutes=10)
    assert runner.extract_session(_SESSION_ID, _PRINCIPAL).status == "already_claimed"
    assert world.events("extraction.run") == []

    world.clock.advance(minutes=25)
    recovered = world.runner().extract_session(_SESSION_ID, _PRINCIPAL)
    assert recovered.status == "completed"
    reclaimed = world.events("extraction.reclaimed")
    assert len(reclaimed) == 1 and reclaimed[0]["previous_claim_at"] == _NOW.isoformat()
    assert _run_payload(world)["reclaimed"] is True


def test_extractor_failure_or_malformed_output_writes_nothing(world: World) -> None:
    runner = world.runner(FakeExtractor(RuntimeError("provider unavailable")))
    result = runner.extract_session(_SESSION_ID, _PRINCIPAL)

    assert result.status == "failed"
    assert world.records() == []
    assert world.store.get_session(_SESSION_ID).extracted_at is None  # type: ignore[union-attr]
    assert world.events("extraction.failed")[0]["stage"] == "extractor"


def test_a_write_phase_failure_rolls_back_every_candidate_and_the_summary(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = world.ingestor.write_session_summary

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(world.ingestor, "write_session_summary", explode)
    with pytest.raises(sqlite3.OperationalError):
        world.runner().extract_session(_SESSION_ID, _PRINCIPAL)

    assert world.records() == []
    assert world.store.get_session(_SESSION_ID).extracted_at is None  # type: ignore[union-attr]
    assert world.events("extraction.failed")[0]["stage"] == "write"
    monkeypatch.setattr(world.ingestor, "write_session_summary", original)


def test_temporal_metadata_without_support_in_the_evidence_is_rejected(world: World) -> None:
    unsupported = replace(_good_candidates()[3], review_at=_NOW + timedelta(days=90))
    result = world.runner(FakeExtractor(_output([unsupported]))).extract_session(_SESSION_ID, _PRINCIPAL)

    assert [(o.stage, o.outcome) for o in result.candidates] == [("validation", "unsupported_temporal_metadata")]


def test_user_statement_quoting_an_assistant_turn_is_downgraded_through_the_shared_helper(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []
    real = ingestor_module.validate_evidence

    def spy(buffer: Any, session_id: Any, quote: str, claimed: str, config: Any, **kwargs: Any) -> Any:
        calls.append((quote, claimed))
        return real(buffer, session_id, quote, claimed, config, **kwargs)

    monkeypatch.setattr(ingestor_module, "validate_evidence", spy)
    candidate = _candidate(
        "Aditya will revisit the SQLite decision next quarter.",
        "I would suggest revisiting that decision next quarter.",
        9,
        attribute="store_review",
    )

    result = world.runner(FakeExtractor(_output([candidate]))).extract_session(_SESSION_ID, _PRINCIPAL)

    assert (candidate.evidence, "user_statement") in calls
    written = result.candidates[0]
    assert written.outcome == "created"
    record = world.store.get_record(written.record_id or "")
    assert record is not None and record.source_kind == "agent_inference" and record.status == "provisional"
    assert written.reason is not None and "downgraded from user_statement" in written.reason


def test_ambiguous_about_entity_rejects_the_candidate_with_candidate_ids(world: World) -> None:
    first = world.store.create_entity(kind="person", canonical="Rohan Mehta", scope=_USER_SCOPE, entity_id="rohan-1")
    second = world.store.create_entity(
        kind="person", canonical="Rohan Mehta", scope=_PRIVATE_SCOPE, entity_id="rohan-2"
    )
    world.store.add_alias(first.id, "rohan mehta")
    world.store.add_alias(second.id, "rohan mehta")
    reviewer = TableReviewer()

    result = world.runner(reviewer=reviewer).extract_session(_SESSION_ID, _PRINCIPAL)

    rohan = next(o for o in result.candidates if o.content == C_ROHAN)
    assert (rohan.stage, rohan.outcome) == ("validation", "entity_ambiguous")
    assert rohan.detail["candidates"] == ["rohan-1", "rohan-2"]
    assert all(r.candidate.content != C_ROHAN for r in reviewer.requests)


def test_session_with_one_turn_writes_only_the_summary(world: World) -> None:
    world.store.create_session("short", _AGENT_ID, _USER_ID, None, _NOW)
    world.buffer.append_turn(
        Turn("short", 1, "user", "I prefer concise technical explanations with code examples.", _NOW)
    )
    extractor = FakeExtractor(_output())

    result = world.runner(extractor).extract_session("short", Principal(_AGENT_ID, _USER_ID, "short", None))

    assert result.status == "completed"
    assert extractor.call_count == 1
    assert result.candidates == []
    assert result.summary_record_id is not None
    assert [r.source_kind for r in world.records()] == ["session_summary"]
    assert _run_payload(world)["short_session"] is True


def test_empty_session_marks_extracted_without_calling_the_extractor(world: World) -> None:
    world.store.create_session("empty", _AGENT_ID, _USER_ID, None, _NOW)
    extractor = FakeExtractor(_output())

    result = world.runner(extractor).extract_session("empty", Principal(_AGENT_ID, _USER_ID, "empty", None))

    assert (result.status, result.reason) == ("completed", "empty_session")
    assert extractor.call_count == 0
    assert world.records() == []
    assert world.store.get_session("empty").extracted_at == _NOW  # type: ignore[union-attr]


def test_second_extraction_is_refused_without_force_and_idempotent_with_it(world: World) -> None:
    runner = world.runner()
    first = runner.extract_session(_SESSION_ID, _PRINCIPAL)
    before = {r.id: (r.status, r.reinforcements, r.version) for r in world.records()}
    run_events = len(world.events("extraction.run"))

    refused = runner.extract_session(_SESSION_ID, _PRINCIPAL)
    assert refused.status == "already_extracted"
    assert len(world.events("extraction.run")) == run_events
    assert world.events("extraction.rerun") == []

    world.clock.advance(minutes=45)
    forced = runner.extract_session(_SESSION_ID, _PRINCIPAL, force=True)

    assert forced.status == "completed"
    assert {o.outcome for o in forced.candidates} == {"already_reinforced"}
    assert forced.summary_outcome == "already_reinforced"
    assert forced.summary_record_id == first.summary_record_id
    after = {r.id: (r.status, r.reinforcements, r.version) for r in world.records()}
    assert after == before
    assert len(world.events("extraction.rerun")) == 1
    assert _run_payload(world)["forced"] is True


def test_changed_summary_text_on_a_forced_rerun_supersedes_the_earlier_summary(world: World) -> None:
    first = world.runner().extract_session(_SESSION_ID, _PRINCIPAL)
    world.clock.advance(minutes=45)
    second = world.runner(
        FakeExtractor(_output(summary=_summary("A different account of the session.")))
    ).extract_session(_SESSION_ID, _PRINCIPAL, force=True)

    assert second.summary_outcome == f"superseded:{first.summary_record_id}"
    old = world.store.get_record(first.summary_record_id or "")
    new = world.store.get_record(second.summary_record_id or "")
    assert old is not None and old.status == "superseded"
    assert new is not None and new.supersedes_id == old.id and new.version == 2
    assert world.store.active_session_summary(f"session:{_SESSION_ID}").id == new.id  # type: ignore[union-attr]


def test_a_second_live_summary_for_one_session_cannot_be_inserted(world: World) -> None:
    result = world.runner().extract_session(_SESSION_ID, _PRINCIPAL)
    summary = world.store.get_record(result.summary_record_id or "")
    assert summary is not None
    duplicate = replace(summary, id="duplicate-summary", entity_ids=[])

    with pytest.raises(sqlite3.IntegrityError):
        world.store.insert_record(duplicate)


def test_two_workers_on_one_session_run_the_extractor_once(world: World) -> None:
    barrier = threading.Barrier(2)

    def slow(turns: Any, context: Any) -> ExtractionOutput:
        time.sleep(0.2)
        return _output()

    extractor = FakeExtractor(slow)
    runner = world.runner(extractor)
    results: list[str] = []

    def work() -> None:
        barrier.wait()
        results.append(runner.extract_session(_SESSION_ID, _PRINCIPAL).status)

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == ["already_claimed", "completed"]
    assert extractor.call_count == 1
    assert len(world.events("extraction.run")) == 1
    assert len([r for r in world.records() if r.source_kind == "session_summary"]) == 1


def test_a_fresh_claim_is_not_reclaimed_and_a_stale_one_is(world: World) -> None:
    stale_before = _NOW - timedelta(minutes=30)
    assert world.store.claim_extraction(_SESSION_ID, _NOW - timedelta(minutes=5), stale_before) == "claimed"
    assert world.store.claim_extraction(_SESSION_ID, _NOW, stale_before) == "already_claimed"
    assert world.store.claim_extraction(_SESSION_ID, _NOW, _NOW - timedelta(minutes=4)) == "reclaimed"
    assert world.store.claim_extraction("missing", _NOW, stale_before) == "not_found"


def test_principal_must_match_the_session(world: World) -> None:
    with pytest.raises(ValueError):
        world.runner().extract_session(_SESSION_ID, Principal("other-agent", _USER_ID, _SESSION_ID, None))


def test_candidates_go_to_the_private_scope_when_the_user_scope_is_not_writable(world: World) -> None:
    world.store.revoke_grant(_AGENT_ID, _USER_SCOPE)
    world.store.set_grant(_AGENT_ID, _USER_SCOPE, can_read=True, can_write=False)

    result = world.runner().extract_session(_SESSION_ID, _PRINCIPAL)

    assert result.status == "completed"
    scopes = {r.scope for r in world.records()}
    assert scopes == {_PRIVATE_SCOPE}
    outcomes = {o.content: o.outcome for o in result.candidates}
    assert outcomes[C_ROHAN] == "created"
    assert outcomes[C_STYLE] == "invalid_subject"


def test_activation_runs_after_extraction_and_keeps_temporary_preferences_conditional(world: World) -> None:
    class Policy:
        def classify(self, content: str) -> CategoryDecision:
            category = "response_language" if "Spanish" in content else "answer_style"
            return CategoryDecision("preferences" if "Aditya" in content else "people", category, "broad", 0.9)  # type: ignore[arg-type]

    activation = ActivationService(world.store, Policy(), actor="extractor")
    result = world.runner(activation=activation).extract_session(_SESSION_ID, _PRINCIPAL)

    outcomes = {o.content: o for o in result.candidates}
    assert (outcomes[C_STYLE].activation, outcomes[C_STYLE].activation_reason) == (
        "promote",
        "eligible_broad_preference",
    )
    assert (outcomes[C_SPANISH].activation, outcomes[C_SPANISH].activation_reason) == (
        "conditional",
        "temporary_preference",
    )
    assert (outcomes[C_ROHAN].activation, outcomes[C_ROHAN].activation_reason) == ("conditional", "not_about_principal")
    assert outcomes[C_SQLITE].activation is None
    records = {r.content: r for r in world.records()}
    assert records[C_STYLE].activation == "ambient"
    assert records[C_SPANISH].activation == "conditional"
    assert outcomes[C_SPANISH].review_outcome == "accept"
    # The extraction event is complete before activation runs: two decisions, two audit trails.
    assert _run_payload(world)["candidates"][2]["activation"] is None
    kinds = [event["kind"] for event in world.store.events_for(records[C_SPANISH].id)]
    assert "record.activation_decided" in kinds and "record.created" in kinds


def test_extracted_records_serve_both_the_ambient_profile_and_host_issued_retrieval(world: World) -> None:
    """The utility-aware path reads ambient records from the profile and conditional ones through auto search."""

    from memory_weave.policy import ProfileAssembler

    class Policy:
        def classify(self, content: str) -> CategoryDecision:
            return CategoryDecision("preferences", "answer_style", "broad", 0.9)  # type: ignore[arg-type]

    activation = ActivationService(world.store, Policy(), actor="extractor")
    world.runner(activation=activation).extract_session(_SESSION_ID, _PRINCIPAL)

    profile = ProfileAssembler(world.store).build(_PRINCIPAL)
    assert C_STYLE in profile.text
    assert C_SPANISH not in profile.text and C_ROHAN not in profile.text

    retriever = Retriever(
        world.store, VectorIndex(world.config.embedding), world.embedder, world.config, current_time=world.clock
    )
    request = SearchRequest(
        queries=["Rohan Mehta runs the platform team."],
        context=None,
        types=None,
        entities=None,
        since=None,
        until=None,
        k=8,
        include_history=False,
        trigger="auto",
    )
    found = {entry.record.content for entry in retriever.search(_PRINCIPAL, request).results}
    assert C_ROHAN in found


def test_activation_is_not_run_when_no_service_is_configured(world: World) -> None:
    world.runner().extract_session(_SESSION_ID, _PRINCIPAL)
    assert {r.activation for r in world.records()} == {"conditional"}
    assert world.events("record.activation") == []


def test_schedule_runs_extraction_on_a_background_thread(world: World) -> None:
    thread = world.runner().schedule(_SESSION_ID, _PRINCIPAL)
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert world.store.get_session(_SESSION_ID).extracted_at == _NOW  # type: ignore[union-attr]


def test_script_reviewer_receives_the_request(world: World) -> None:
    seen: list[ReviewRequest] = []

    def script(request: ReviewRequest) -> ReviewDecision:
        seen.append(request)
        return ReviewDecision("accept", "ok")

    world.runner(reviewer=TableReviewer(script)).extract_session(_SESSION_ID, _PRINCIPAL)
    assert len(seen) == 4


def test_write_request_carries_temporal_metadata_into_the_record(world: World) -> None:
    result = world.ingestor.write(
        _PRINCIPAL,
        ingestor_module.WriteRequest(
            type="semantic",
            content=C_SPANISH,
            source_kind="user_statement",
            evidence="Until Friday please answer in Spanish",
            attribute="response_language",
            review_at=_FRIDAY,
        ),
    )
    record = world.store.get_record(result.record_id or "")
    assert record is not None and record.review_at == _FRIDAY and record.review_flagged_at is None
    assert record.subject == render_subject(record.subject_entity_id, "response_language")


@pytest.mark.live
def test_live_extractor_produces_at_least_one_validating_candidate(world: World) -> None:
    if os.environ.get("MEMORY_WEAVE_LIVE") != "1":
        pytest.skip("set MEMORY_WEAVE_LIVE=1 to run the hosted extractor")
    from memory_weave.ingest import StructuredLLMExtractor, StructuredLLMReviewer, completion_client_for

    ingestion = world.config.ingestion
    extraction_model = os.environ.get("MEMORY_WEAVE_EXTRACTION_MODEL", ingestion.extraction_model)
    review_model = os.environ.get("MEMORY_WEAVE_REVIEW_MODEL", ingestion.review_model)
    for model in (extraction_model, review_model):
        key = "ANTHROPIC_API_KEY" if model.startswith("claude-") else "OPENAI_API_KEY"
        if not os.environ.get(key):
            pytest.skip(f"{key} is not set")
    extractor = StructuredLLMExtractor(
        completion_client_for(extraction_model, max_output_tokens=ingestion.hosted_max_output_tokens),
        timeout_ms=ingestion.extraction_timeout_ms,
        max_candidates=ingestion.extraction_max_candidates,
    )
    reviewer = StructuredLLMReviewer(
        completion_client_for(review_model, max_output_tokens=ingestion.hosted_max_output_tokens),
        timeout_ms=ingestion.review_timeout_ms,
    )

    result = ExtractionRunner(
        world.store, world.ingestor, extractor, reviewer, world.buffer, world.config, current_time=world.clock
    ).extract_session(_SESSION_ID, _PRINCIPAL)

    assert result.status == "completed", result.reason
    assert result.summary_record_id is not None
    assert any(o.written for o in result.candidates), [o.to_payload() for o in result.candidates]
    assert all(o.review_outcome in ("accept", "revise", "reject", None) for o in result.candidates)
    print(json.dumps([o.to_payload() for o in result.candidates], indent=1))
