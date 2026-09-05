from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memory_weave.config import EmbeddingConfig, IngestionConfig, MemoryWeaveConfig
from memory_weave.index.embedder import FakeEmbedder
from memory_weave.index.vector import VectorIndex
from memory_weave.ingest import FakeJudge, Ingestor, SessionBuffer, WriteRequest
from memory_weave.models import EntityMention, Principal, Scope, Turn
from memory_weave.store import Store

_NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
_AGENT_ID = "research-agent"
_USER_ID = "aditya"
_SESSION_ID = "session-1"
_PRINCIPAL = Principal(_AGENT_ID, _USER_ID, _SESSION_ID, "memory-weave")
_AGENT_SCOPE = Scope(kind="agent", id=f"{_AGENT_ID}/{_USER_ID}")
_USER_SCOPE = Scope(kind="user", id=_USER_ID)
_PROJECT_SCOPE = Scope(kind="project", id="memory-weave")
_EMBEDDING = EmbeddingConfig(model="fake-embedder", version="1", dims=8)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    database = Store(tmp_path / "memory.sqlite")
    _ = database.connection
    database.create_session(_SESSION_ID, _AGENT_ID, _USER_ID, "memory-weave", _NOW)
    database.set_grant(_AGENT_ID, _USER_SCOPE, can_read=True, can_write=True)
    yield database
    database.close()


@pytest.fixture
def buffer(store: Store) -> SessionBuffer:
    session_buffer = SessionBuffer(store)
    session_buffer.append_turn(Turn(_SESSION_ID, 1, "user", "I prefer concise technical explanations.", _NOW))
    session_buffer.append_turn(Turn(_SESSION_ID, 2, "assistant", "I recommend concise technical explanations.", _NOW))
    session_buffer.append_turn(Turn(_SESSION_ID, 3, "tool", "The deployment completed successfully.", _NOW))
    session_buffer.append_turn(Turn(_SESSION_ID, 4, "user", "The project uses SQLite for durable memory.", _NOW))
    return session_buffer


@pytest.fixture
def config() -> MemoryWeaveConfig:
    return MemoryWeaveConfig(
        embedding=_EMBEDDING,
        ingestion=IngestionConfig(dedup_candidate_cosine=0.80, reinforcements_to_confirm=2),
    )


def _ingestor(
    store: Store,
    buffer: SessionBuffer,
    config: MemoryWeaveConfig,
    judge: FakeJudge | None = None,
    *,
    embedder: FakeEmbedder | None = None,
) -> Ingestor:
    return Ingestor(
        store,
        VectorIndex(config.embedding),
        embedder or FakeEmbedder(dims=config.embedding.dims),
        judge or FakeJudge(),
        buffer,
        config,
        current_time=lambda: _NOW,
    )


def _request(
    content: str,
    *,
    attribute: str | None = "explanation_style",
    source_kind: str = "user_statement",
    evidence: str | None = "I prefer concise technical explanations.",
    event_at: datetime | None = None,
    evidence_turn: int | None = None,
    type: str = "semantic",
    scope: Scope | None = None,
    entities: list[EntityMention] | None = None,
) -> WriteRequest:
    return WriteRequest(
        type=type,  # type: ignore[arg-type]
        content=content,
        attribute=attribute,
        scope=scope,
        source_kind=source_kind,  # type: ignore[arg-type]
        evidence=evidence,
        event_at=event_at,
        evidence_turn=evidence_turn,
        entities=entities or [],
    )


def _event(store: Store, record_id: str) -> dict[str, object]:
    events = store.events_for(record_id)
    assert events
    return events[-1]


def test_scope_not_writable_returns_before_writing_anything(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    result = _ingestor(store, buffer, config).write(
        _PRINCIPAL,
        _request("The shared project uses SQLite.", scope=_PROJECT_SCOPE),
    )

    assert result.outcome == "scope_not_writable"
    assert result.record_id is None
    assert store.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 0
    assert store.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


def test_default_write_uses_the_provisioned_user_scope_and_resolves_the_principal_person(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    result = _ingestor(store, buffer, config).write(_PRINCIPAL, _request("Aditya uses Vim."))

    assert result.outcome == "created"
    assert result.record_id is not None
    stored = store.get_record(result.record_id)
    assert stored is not None
    assert stored.scope == _USER_SCOPE
    assert stored.subject_entity_id is not None
    assert stored.attribute == "explanation_style"
    assert stored.subject == f"{stored.subject_entity_id}/explanation_style"
    assert store.get_entity(stored.subject_entity_id).canonical == _USER_ID  # type: ignore[union-attr]


def test_default_write_without_host_provisioning_returns_scope_not_writable(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    store.revoke_grant(_AGENT_ID, _USER_SCOPE)

    result = _ingestor(store, buffer, config).write(_PRINCIPAL, _request("Aditya uses Vim."))

    assert result.outcome == "scope_not_writable"
    assert result.record_id is None
    assert store.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 0


def test_semantic_and_procedural_records_require_a_current_fact_subject(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    result = _ingestor(store, buffer, config).write(
        _PRINCIPAL,
        _request("Aditya uses Vim.", attribute=None),
    )

    assert result.outcome == "invalid_subject"
    assert result.note == "semantic records require an attribute."
    assert result.record_id is None
    assert store.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 0


def test_project_scope_requires_an_about_entity_but_user_scope_uses_the_principal(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    store.set_grant(_AGENT_ID, _PROJECT_SCOPE, can_read=True, can_write=True)

    project = _ingestor(store, buffer, config).write(
        _PRINCIPAL,
        _request("The project uses SQLite.", scope=_PROJECT_SCOPE, attribute="storage"),
    )
    user = _ingestor(store, buffer, config).write(
        _PRINCIPAL,
        _request("Aditya uses Vim.", attribute="editor"),
    )

    assert project.outcome == "invalid_subject"
    assert project.note == "about entity required"
    assert user.record_id is not None
    assert store.get_record(user.record_id).subject_entity_id is not None  # type: ignore[union-attr]


def test_attribute_ignores_a_writer_supplied_entity_portion(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    result = _ingestor(store, buffer, config).write(
        _PRINCIPAL,
        _request(
            "Memory Weave uses SQLite.",
            attribute="person:someone-else/Storage Mode",
            entities=[EntityMention(kind="project", text="Memory Weave", role="about")],
        ),
    )

    assert result.record_id is not None
    stored = store.get_record(result.record_id)
    assert stored is not None
    assert stored.attribute == "storage_mode"
    assert stored.subject_entity_id is not None
    assert stored.subject == f"{stored.subject_entity_id}/storage_mode"


def test_assistant_evidence_downgrades_user_statement_and_records_the_note(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    result = _ingestor(store, buffer, config).write(
        _PRINCIPAL,
        _request(
            "Aditya prefers concise technical explanations.", evidence="I recommend concise technical explanations."
        ),
    )

    assert result.outcome == "created"
    assert result.status == "provisional"
    assert result.record_id is not None
    stored = store.get_record(result.record_id)
    assert stored is not None
    assert stored.source_kind == "agent_inference"
    assert stored.source_ref == f"session:{_SESSION_ID}<turn:2>"
    assert result.note == "downgraded from user_statement: quote is from an assistant turn"
    assert _event(store, result.record_id)["payload"]["evidence_note"] == result.note  # type: ignore[index]


def test_missing_evidence_creates_a_provisional_inference_without_a_source_reference(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    result = _ingestor(store, buffer, config).write(
        _PRINCIPAL,
        _request("Aditya prefers concise technical explanations.", evidence=None),
    )

    assert result.outcome == "created"
    assert result.note == "evidence not provided"
    assert result.record_id is not None
    stored = store.get_record(result.record_id)
    assert stored is not None
    assert stored.source_kind == "agent_inference"
    assert stored.source_ref is None
    assert stored.evidence is None
    assert stored.status == "provisional"
    assert stored.expires_at == _NOW + timedelta(days=config.ingestion.provisional_ttl_days)


def test_non_entailing_evidence_downgrades_a_direct_claim_and_records_its_score(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    content = "Aditya uses Vim."
    evidence = "I prefer concise technical explanations."
    judge = FakeJudge(entailments={(evidence, content): 0.20})

    result = _ingestor(store, buffer, config, judge).write(_PRINCIPAL, _request(content, evidence=evidence))

    assert result.record_id is not None
    stored = store.get_record(result.record_id)
    assert stored is not None
    assert stored.source_kind == "agent_inference"
    assert stored.status == "provisional"
    assert result.note == "evidence does not support claim"
    assert _event(store, result.record_id)["payload"]["evidence_entailment_score"] == 0.20  # type: ignore[index]


def test_agent_inference_is_never_sent_to_the_evidence_entailment_judge(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    judge = FakeJudge()

    result = _ingestor(store, buffer, config, judge).write(
        _PRINCIPAL,
        _request("Aditya uses Vim.", source_kind="agent_inference"),
    )

    assert result.outcome == "created"
    assert judge.entail_calls == []


def test_same_subject_reinforces_and_the_second_reinforcement_confirms(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    contents = (
        "Aditya prefers concise technical explanations.",
        "Aditya likes short technical explanations.",
        "Aditya wants brief technical explanations.",
    )
    ingestor = _ingestor(
        store,
        buffer,
        config,
        FakeJudge({(contents[0], contents[1]): "same", (contents[0], contents[2]): "same"}),
    )
    buffer.append_turn(Turn(_SESSION_ID, 5, "assistant", "I will keep technical explanations concise and brief.", _NOW))
    buffer.append_turn(Turn(_SESSION_ID, 6, "assistant", "I will use short technical explanations.", _NOW))
    initial = ingestor.write(
        _PRINCIPAL,
        _request(
            contents[0],
            source_kind="agent_inference",
            evidence="I recommend concise technical explanations.",
        ),
    )
    first = ingestor.write(
        _PRINCIPAL,
        _request(
            contents[1],
            source_kind="agent_inference",
            evidence="I will keep technical explanations concise and brief.",
        ),
    )
    second = ingestor.write(
        _PRINCIPAL,
        _request(
            contents[2],
            source_kind="agent_inference",
            evidence="I will use short technical explanations.",
        ),
    )

    assert initial.record_id is not None
    assert first.outcome == "reinforced"
    assert second.outcome == "reinforced"
    assert first.record_id == initial.record_id == second.record_id
    assert store.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1
    stored = store.get_record(initial.record_id)
    assert stored is not None
    assert stored.reinforcements == 2
    assert stored.status == "confirmed"
    assert stored.confidence == pytest.approx(0.80)


def test_repeated_source_reference_does_not_count_as_an_independent_reinforcement(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    original = "Aditya prefers concise technical explanations."
    repeated = "Aditya likes concise technical explanations."
    ingestor = _ingestor(store, buffer, config, FakeJudge({(original, repeated): "same"}))
    first = ingestor.write(
        _PRINCIPAL,
        _request(original, source_kind="agent_inference", evidence="I recommend concise technical explanations."),
    )
    duplicate = ingestor.write(
        _PRINCIPAL,
        _request(repeated, source_kind="agent_inference", evidence="I recommend concise technical explanations."),
    )

    assert first.record_id is not None
    assert duplicate.outcome == "already_reinforced"
    stored = store.get_record(first.record_id)
    assert stored is not None
    assert stored.reinforcements == 0
    assert stored.last_reinforced_at is None
    event = _event(store, first.record_id)["payload"]  # type: ignore[index]
    assert event["reinforcement_counted"] is False  # type: ignore[index]
    assert event["reinforcing_source_ref"] == f"session:{_SESSION_ID}<turn:2>"  # type: ignore[index]


def test_stronger_independent_evidence_promotes_the_existing_record_provenance(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    inference = "Aditya prefers concise technical explanations."
    user_statement = "Aditya likes short technical explanations."
    ingestor = _ingestor(store, buffer, config, FakeJudge({(inference, user_statement): "same"}))
    initial = ingestor.write(
        _PRINCIPAL,
        _request(inference, source_kind="agent_inference", evidence="I recommend concise technical explanations."),
    )
    promoted = ingestor.write(
        _PRINCIPAL,
        _request(user_statement, source_kind="user_statement", evidence="I prefer concise technical explanations."),
    )

    assert initial.record_id is not None
    assert promoted.outcome == "reinforced"
    stored = store.get_record(initial.record_id)
    assert stored is not None
    assert stored.source_kind == "user_statement"
    assert stored.source_ref == f"session:{_SESSION_ID}<turn:1>"
    assert stored.evidence == "I prefer concise technical explanations."
    assert stored.status == "confirmed"
    assert stored.confidence == pytest.approx(0.95)
    event = _event(store, initial.record_id)["payload"]  # type: ignore[index]
    assert event["provenance_promoted"] is True  # type: ignore[index]
    assert event["reinforcing_evidence"] == "I prefer concise technical explanations."  # type: ignore[index]


def test_higher_rank_same_subject_supersedes_the_active_record(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    judge = FakeJudge(
        {("The deployment completed successfully.", "Aditya prefers concise technical explanations."): "distinct"}
    )
    ingestor = _ingestor(store, buffer, config, judge)
    old = ingestor.write(
        _PRINCIPAL,
        _request(
            "The deployment completed successfully.",
            source_kind="tool_result",
            evidence="The deployment completed successfully.",
        ),
    )
    new = ingestor.write(
        _PRINCIPAL,
        _request("Aditya prefers concise technical explanations."),
    )

    assert old.record_id is not None
    assert new.outcome == f"superseded:{old.record_id}"
    assert new.record_id is not None
    assert store.get_record(old.record_id).status == "superseded"  # type: ignore[union-attr]
    stored_new = store.get_record(new.record_id)
    assert stored_new is not None
    assert stored_new.supersedes_id == old.record_id


def test_equal_rank_later_event_supersedes_and_earlier_event_is_superseded_on_arrival(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    judge = FakeJudge(
        {
            ("The deployment completed successfully.", "The deployment completed after a repair."): "distinct",
            ("The deployment completed after a repair.", "The deployment completed before a repair."): "distinct",
        }
    )
    ingestor = _ingestor(store, buffer, config, judge)
    old = ingestor.write(
        _PRINCIPAL,
        _request(
            "The deployment completed successfully.",
            source_kind="tool_result",
            evidence="The deployment completed successfully.",
            event_at=_NOW,
        ),
    )
    later = ingestor.write(
        _PRINCIPAL,
        _request(
            "The deployment completed after a repair.",
            source_kind="tool_result",
            evidence="The deployment completed successfully.",
            event_at=_NOW + timedelta(days=1),
        ),
    )
    stale = ingestor.write(
        _PRINCIPAL,
        _request(
            "The deployment completed before a repair.",
            source_kind="tool_result",
            evidence="The deployment completed successfully.",
            event_at=_NOW - timedelta(days=1),
        ),
    )

    assert old.record_id is not None
    assert later.record_id is not None
    assert stale.record_id is not None
    assert later.outcome == f"superseded:{old.record_id}"
    assert stale.outcome == f"superseded_on_arrival:{later.record_id}"
    stale_record = store.get_record(stale.record_id)
    assert stale_record is not None
    assert stale_record.status == "superseded"
    assert stale_record.supersedes_id is None
    assert _event(store, stale.record_id)["payload"]["superseded_on_arrival_by"] == later.record_id  # type: ignore[index]


def test_lower_rank_contradiction_is_provisional_and_records_symmetric_conflicts(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    judge = FakeJudge(
        {("Aditya prefers concise technical explanations.", "The deployment completed successfully."): "contradicts"}
    )
    ingestor = _ingestor(store, buffer, config, judge)
    old = ingestor.write(_PRINCIPAL, _request("Aditya prefers concise technical explanations."))
    conflict = ingestor.write(
        _PRINCIPAL,
        _request(
            "The deployment completed successfully.",
            source_kind="tool_result",
            evidence="The deployment completed successfully.",
        ),
    )

    assert old.record_id is not None
    assert conflict.record_id is not None
    assert conflict.outcome == f"conflict:{old.record_id}"
    assert conflict.status == "provisional"
    assert store.conflicts_for(old.record_id) == [conflict.record_id]
    assert store.conflicts_for(conflict.record_id) == [old.record_id]
    stored_conflict = store.get_record(conflict.record_id)
    assert stored_conflict is not None
    assert stored_conflict.expires_at == _NOW + timedelta(days=config.ingestion.provisional_ttl_days)


def test_authority_incumbent_wins_when_a_subject_also_has_a_later_provisional_conflict(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    ingestor = _ingestor(store, buffer, config, FakeJudge())
    confirmed = ingestor.write(
        _PRINCIPAL,
        _request("Aditya prefers concise technical explanations.", event_at=_NOW),
    )
    conflict = ingestor.write(
        _PRINCIPAL,
        _request(
            "Aditya prefers detailed technical explanations.",
            source_kind="tool_result",
            evidence="The deployment completed successfully.",
            event_at=_NOW + timedelta(days=1),
        ),
    )
    replacement = ingestor.write(
        _PRINCIPAL,
        _request(
            "Aditya now prefers balanced technical explanations.",
            event_at=_NOW + timedelta(days=2),
        ),
    )

    assert confirmed.record_id is not None
    assert conflict.record_id is not None
    assert replacement.outcome == f"superseded:{confirmed.record_id}"
    assert store.get_record(confirmed.record_id).status == "superseded"  # type: ignore[union-attr]
    assert store.get_record(conflict.record_id).status == "provisional"  # type: ignore[union-attr]


def test_nearby_different_subject_same_claim_reinforces_the_neighbour(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    old_content = "Aditya prefers concise technical explanations."
    new_content = "Aditya prefers brief technical explanations."
    judge = FakeJudge({(old_content, new_content): "same"})
    embedder = FakeEmbedder(dims=config.embedding.dims)
    ingestor = _ingestor(store, buffer, config, judge, embedder=embedder)
    old = ingestor.write(_PRINCIPAL, _request(old_content, attribute="explanation_style"))
    assert old.record_id is not None
    embedder.set_similarity(old_content, new_content, 0.90)

    duplicate = ingestor.write(
        _PRINCIPAL,
        _request(
            new_content,
            attribute="response_length",
            evidence="The project uses SQLite for durable memory.",
        ),
    )

    assert duplicate.outcome == "reinforced"
    assert duplicate.record_id == old.record_id
    assert store.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1
    assert _event(store, old.record_id)["payload"]["dedup_kind"] == "attribute"  # type: ignore[index]


def test_conflicting_attribute_aliases_supersede_even_when_their_vectors_are_dissimilar(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    first_content = "Aditya prefers concise technical explanations."
    second_content = "Aditya prefers detailed technical explanations."
    embedder = FakeEmbedder(dims=config.embedding.dims)
    embedder.set_similarity(first_content, second_content, 0.0)
    judge = FakeJudge({(first_content, second_content): "contradicts"})
    ingestor = _ingestor(store, buffer, config, judge, embedder=embedder)

    first = ingestor.write(_PRINCIPAL, _request(first_content, attribute="explanation_style"))
    second = ingestor.write(
        _PRINCIPAL,
        _request(second_content, attribute="answer_style", event_at=_NOW + timedelta(minutes=1)),
    )

    assert first.record_id is not None
    assert second.record_id is not None
    assert second.outcome == f"superseded:{first.record_id}"
    stored = store.get_record(second.record_id)
    assert stored is not None
    assert stored.attribute == "explanation_style"
    assert _event(store, second.record_id)["payload"]["attribute_aliased_from"] == "answer_style"  # type: ignore[index]


def test_attribute_scan_logs_when_the_configured_limit_truncates_active_attributes(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    bounded = replace(config, ingestion=replace(config.ingestion, max_entity_attributes=1))
    ingestor = _ingestor(store, buffer, bounded, FakeJudge())
    _ = ingestor.write(_PRINCIPAL, _request("Aditya uses Vim.", attribute="editor"))
    _ = ingestor.write(_PRINCIPAL, _request("Aditya lives in Bangalore.", attribute="location"))
    result = ingestor.write(_PRINCIPAL, _request("Aditya works remotely.", attribute="work_style"))

    assert result.record_id is not None
    assert _event(store, result.record_id)["payload"]["attribute_scan_truncated"] is True  # type: ignore[index]


def test_episodic_record_never_supersedes_an_existing_subject(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    ingestor = _ingestor(store, buffer, config)
    first = ingestor.write(
        _PRINCIPAL,
        _request("The deployment started.", type="episodic"),
    )
    second = ingestor.write(
        _PRINCIPAL,
        _request("The deployment completed.", type="episodic"),
    )

    assert first.outcome == "created"
    assert second.outcome == "created"
    assert first.record_id is not None
    assert second.record_id is not None
    assert store.get_record(first.record_id).status == "confirmed"  # type: ignore[union-attr]
    assert store.get_record(second.record_id).status == "confirmed"  # type: ignore[union-attr]


def test_semantic_writes_reject_an_attribute_that_normalizes_to_nothing(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    first_content = "Aditya uses Vim."
    second_content = "Aditya lives in Bangalore."
    embedder = FakeEmbedder(dims=config.embedding.dims)
    embedder.set_similarity(first_content, second_content, 0.0)
    ingestor = _ingestor(store, buffer, config, FakeJudge(), embedder=embedder)

    first = ingestor.write(_PRINCIPAL, _request(first_content, attribute=None))
    second = ingestor.write(_PRINCIPAL, _request(second_content, attribute=None))

    assert first.outcome == "invalid_subject"
    assert second.outcome == "invalid_subject"


def test_about_ambiguity_rolls_back_and_is_audited_afterward(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    first = store.create_entity(kind="project", canonical="API", scope=_AGENT_SCOPE, entity_id="project-agent")
    second = store.create_entity(kind="project", canonical="API", scope=_PROJECT_SCOPE, entity_id="project-shared")
    store.add_alias(first.id, "api")
    store.add_alias(second.id, "api")
    store.set_grant(_AGENT_ID, _PROJECT_SCOPE, can_read=True, can_write=False)

    result = _ingestor(store, buffer, config).write(
        _PRINCIPAL,
        _request(
            "The API uses SQLite for durable memory.",
            entities=[
                EntityMention(kind="person", text="Aditya", role="mentions"),
                EntityMention(kind="project", text="API", role="about"),
            ],
        ),
    )

    assert result.outcome == "entity_ambiguous"
    assert [candidate.id for candidate in result.candidates] == [first.id, second.id]
    assert store.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 0
    assert store.connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 2
    event = store.connection.execute("SELECT payload FROM events WHERE kind = 'entity.ambiguous_alias'").fetchone()
    assert event is not None
    payload = json.loads(event["payload"])
    assert [candidate["id"] for candidate in payload["candidates"]] == [first.id, second.id]


def test_ambiguous_mentions_are_dropped_while_the_record_is_written(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    first = store.create_entity(kind="project", canonical="API", scope=_AGENT_SCOPE, entity_id="project-agent")
    second = store.create_entity(kind="project", canonical="API", scope=_PROJECT_SCOPE, entity_id="project-shared")
    store.add_alias(first.id, "api")
    store.add_alias(second.id, "api")
    store.set_grant(_AGENT_ID, _PROJECT_SCOPE, can_read=True, can_write=False)

    result = _ingestor(store, buffer, config).write(
        _PRINCIPAL,
        _request(
            "The API uses SQLite for durable memory.",
            entities=[EntityMention(kind="project", text="API", role="mentions")],
        ),
    )

    assert result.outcome == "created"
    assert result.record_id is not None
    stored = store.get_record(result.record_id)
    assert stored is not None
    assert len(stored.entity_ids) == 1
    assert result.note == f"dropped ambiguous mentions: {first.id}, {second.id}"


def test_created_record_writes_its_embedding_fts_aliases_and_entity_links(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    embedder = FakeEmbedder(dims=config.embedding.dims)
    index = VectorIndex(config.embedding)
    ingestor = Ingestor(
        store,
        index,
        embedder,
        FakeJudge(),
        buffer,
        config,
        current_time=lambda: _NOW,
    )

    result = ingestor.write(
        _PRINCIPAL,
        _request(
            "It uses SQLite for durable memory.",
            attribute="storage",
            entities=[EntityMention(kind="project", text="Memory Weave", role="about")],
        ),
    )

    assert result.record_id is not None
    stored = store.get_record(result.record_id)
    assert stored is not None
    assert len(stored.entity_ids) == 1
    assert store.fts_query("memory", limit=10)[0][0] == result.record_id
    assert index.vector_for(result.record_id) is not None
    role = store.connection.execute(
        "SELECT role FROM record_entities WHERE record_id = ? AND entity_id = ?",
        (result.record_id, stored.entity_ids[0]),
    ).fetchone()
    assert role is not None
    assert role["role"] == "about"


@pytest.mark.parametrize("source_kind", ["system", "session_summary"])
def test_system_and_session_summary_sources_are_rejected(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig, source_kind: str
) -> None:
    result = _ingestor(store, buffer, config).write(
        _PRINCIPAL,
        _request("The host supplied this record.", source_kind=source_kind),
    )

    assert result.outcome == "invalid_source_kind"
    assert result.record_id is None
    assert store.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 0


def test_write_result_carries_complete_timings_after_commit_and_index_update(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    result = _ingestor(store, buffer, config).write(
        _PRINCIPAL,
        _request("Aditya prefers concise technical explanations."),
    )

    assert result.record_id is not None
    timings = result.timings_ms
    assert {
        "permission",
        "evidence",
        "entities",
        "embed",
        "dedup_search",
        "judge",
        "supersession",
        "persistence",
        "transaction",
        "event_log",
        "index_update",
        "total",
    } == set(timings)
    event = _event(store, result.record_id)["payload"]  # type: ignore[index]
    assert event["timings_pending"] == ["event_log", "transaction", "index_update"]  # type: ignore[index]


def test_contradicted_incumbent_is_superseded_even_when_a_provisional_sibling_matches(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    concise = "Aditya prefers concise technical explanations."
    detailed_guess = "Aditya prefers detailed technical explanations."
    detailed_stated = "Aditya wants detailed technical explanations."
    judge = FakeJudge(
        {
            (concise, detailed_guess): "contradicts",
            (concise, detailed_stated): "contradicts",
            (detailed_guess, detailed_stated): "same",
        }
    )
    ingestor = _ingestor(store, buffer, config, judge)
    incumbent = ingestor.write(_PRINCIPAL, _request(concise))
    sibling = ingestor.write(_PRINCIPAL, _request(detailed_guess, source_kind="agent_inference", evidence=None))
    assert sibling.outcome == f"conflict:{incumbent.record_id}"

    stated = ingestor.write(
        _PRINCIPAL,
        _request(
            detailed_stated,
            evidence="The project uses SQLite for durable memory.",
            event_at=_NOW + timedelta(minutes=1),
        ),
    )

    assert stated.outcome == f"superseded:{incumbent.record_id}"
    assert incumbent.record_id is not None and sibling.record_id is not None and stated.record_id is not None
    assert store.get_record(incumbent.record_id).status == "superseded"  # type: ignore[union-attr]
    assert store.get_record(sibling.record_id).status == "superseded"  # type: ignore[union-attr]
    assert _event(store, stated.record_id)["payload"]["also_superseded"] == [sibling.record_id]  # type: ignore[index]
    active = store.active_for_entity(_USER_SCOPE, store.get_record(stated.record_id).subject_entity_id, "semantic")  # type: ignore[union-attr, arg-type]
    assert [record.id for record in active] == [stated.record_id]


def test_attribute_cap_keeps_store_order_so_cosine_never_excludes_a_recent_attribute(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    location = "Aditya lives in Bangalore."
    editor = "Aditya uses Vim."
    incoming = "Aditya has switched to Emacs."
    embedder = FakeEmbedder(dims=config.embedding.dims)
    embedder.set_similarity(incoming, location, 0.90)
    embedder.set_similarity(incoming, editor, 0.30)
    judge = FakeJudge({(editor, incoming): "contradicts"})
    bounded = replace(config, ingestion=replace(config.ingestion, max_entity_attributes=1))
    ingestor = _ingestor(store, buffer, bounded, judge, embedder=embedder)
    ingestor_clock = [_NOW]
    ingestor._current_time = lambda: ingestor_clock[0]  # type: ignore[method-assign]

    _ = ingestor.write(_PRINCIPAL, _request(location, attribute="location"))
    ingestor_clock[0] = _NOW + timedelta(minutes=1)
    editor_result = ingestor.write(_PRINCIPAL, _request(editor, attribute="editor"))
    ingestor_clock[0] = _NOW + timedelta(minutes=2)
    result = ingestor.write(_PRINCIPAL, _request(incoming, attribute="work_style"))

    assert result.outcome == f"superseded:{editor_result.record_id}"
    payload = _event(store, result.record_id)["payload"]  # type: ignore[index, arg-type]
    assert payload["attribute_scan_truncated"] is True  # type: ignore[index]
    assert payload["attribute_aliased_from"] == "work_style"  # type: ignore[index]


def test_two_person_entities_carrying_the_principal_alias_return_entity_ambiguous(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    for entity_id in ("person-one", "person-two"):
        store.create_entity(kind="person", canonical="Aditya", scope=_USER_SCOPE, entity_id=entity_id)
        store.add_alias(entity_id, "aditya")

    result = _ingestor(store, buffer, config).write(_PRINCIPAL, _request("Aditya uses Vim.", attribute="editor"))

    assert result.outcome == "entity_ambiguous"
    assert [candidate.id for candidate in result.candidates] == ["person-one", "person-two"]
    assert store.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 0


def test_empty_attribute_reports_its_own_note(store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig) -> None:
    result = _ingestor(store, buffer, config).write(_PRINCIPAL, _request("Aditya uses Vim.", attribute="///"))

    assert result.outcome == "invalid_subject"
    assert result.note == "attribute normalizes to nothing"


def test_a_claim_downgraded_by_its_turn_role_is_not_entailment_checked_again(
    store: Store, buffer: SessionBuffer, config: MemoryWeaveConfig
) -> None:
    quote = "I recommend concise technical explanations."
    judge = FakeJudge(entailments={(quote, "Aditya prefers concise technical explanations."): 0.0})
    result = _ingestor(store, buffer, config, judge).write(
        _PRINCIPAL, _request("Aditya prefers concise technical explanations.", evidence=quote)
    )

    assert result.note == "downgraded from user_statement: quote is from an assistant turn"
    assert judge.entail_calls == []


def test_host_provisioned_aliases_make_a_mention_resolve_to_the_principal_entity(
    tmp_path: Path, config: MemoryWeaveConfig
) -> None:
    from memory_weave.host import MemoryHost

    opaque = Principal(_AGENT_ID, "u-123", _SESSION_ID, None)
    user_scope = Scope(kind="user", id="u-123")
    database = Store(tmp_path / "opaque.sqlite")
    database.create_session(_SESSION_ID, _AGENT_ID, "u-123", None, _NOW)
    session_buffer = SessionBuffer(database)
    session_buffer.append_turn(Turn(_SESSION_ID, 1, "user", "I prefer concise technical explanations.", _NOW))
    host = MemoryHost(database)
    host.grant(_AGENT_ID, user_scope, read=True, write=True)
    principal_entity_id = host.provision_user("u-123", ["Aditya", "Aditya Mishra"])
    judge = FakeJudge({("Aditya prefers concise answers.", "Aditya prefers detailed answers."): "contradicts"})
    ingestor = _ingestor(database, session_buffer, config, judge)

    first = ingestor.write(opaque, _request("Aditya prefers concise answers."))
    second = ingestor.write(
        opaque,
        _request(
            "Aditya prefers detailed answers.",
            event_at=_NOW + timedelta(minutes=1),
            entities=[EntityMention(kind="person", text="Aditya Mishra", role="about")],
        ),
    )

    assert second.outcome == f"superseded:{first.record_id}"
    assert first.record_id is not None and second.record_id is not None
    assert database.get_record(first.record_id).subject_entity_id == principal_entity_id  # type: ignore[union-attr]
    assert database.get_record(second.record_id).subject_entity_id == principal_entity_id  # type: ignore[union-attr]
    database.close()
