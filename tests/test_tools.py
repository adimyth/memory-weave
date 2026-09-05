"""Framework-neutral tool contracts and handler round trips."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from memory_weave.config import EmbeddingConfig, MemoryWeaveConfig, RetrievalConfig
from memory_weave.index.embedder import FakeEmbedder
from memory_weave.index.vector import VectorIndex
from memory_weave.ingest import FakeJudge, Ingestor, SessionBuffer
from memory_weave.models import Explanation, Principal, Record, Scope, SearchResponse, SearchResult, Turn
from memory_weave.retrieve import Retriever
from memory_weave.store import Store
from memory_weave.tools import TOOL_SCHEMAS, ToolHandlers, render_search, tool_schemas, validate_tool_input
from memory_weave.tools.schemas import ToolInputError

_NOW = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
_AGENT = "research-agent"
_USER = "aditya"
_SESSION = "session-1"
_PRINCIPAL = Principal(_AGENT, _USER, _SESSION, None)
_USER_SCOPE = Scope(kind="user", id=_USER)
_EMBEDDING = EmbeddingConfig(model="fake-embedder", version="1", dims=8)
_CONFIG = MemoryWeaveConfig(embedding=_EMBEDDING, retrieval=RetrievalConfig(per_generator_k=10, default_k=8))


@pytest.fixture
def store(tmp_path: Path) -> Store:
    database = Store(tmp_path / "memory.sqlite")
    _ = database.connection
    database.set_grant(_AGENT, _USER_SCOPE, can_read=True, can_write=True)
    database.create_session(_SESSION, _AGENT, _USER, None, _NOW)
    yield database
    database.close()


@pytest.fixture
def handlers(store: Store) -> tuple[ToolHandlers, FakeEmbedder]:
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    vector_index = VectorIndex(_EMBEDDING)
    session_buffer = SessionBuffer(store)
    session_buffer.append_turn(Turn(_SESSION, 1, "user", "I prefer concise technical explanations.", _NOW))
    ingestor = Ingestor(
        store,
        vector_index,
        embedder,
        FakeJudge(),
        session_buffer,
        _CONFIG,
        current_time=lambda: _NOW,
    )
    retriever = Retriever(store, vector_index, embedder, _CONFIG, current_time=lambda: _NOW)
    return ToolHandlers(retriever, ingestor, store, vector_index), embedder


def _write_payload(content: str = "The user prefers concise technical explanations.") -> dict[str, object]:
    return {
        "type": "semantic",
        "content": content,
        "attribute": "explanation_style",
        "source_kind": "user_statement",
        "evidence": "I prefer concise technical explanations.",
        "entities": [{"kind": "person", "name": "Aditya", "role": "about"}],
        "tags": ["communication"],
    }


def test_tool_schemas_expose_only_the_public_contract_and_reject_invalid_payloads() -> None:
    assert list(TOOL_SCHEMAS) == ["memory_search", "memory_get", "memory_write", "memory_revise", "memory_forget"]
    assert "context" not in TOOL_SCHEMAS["memory_search"]["input_schema"]["properties"]  # type: ignore[index]
    source_kinds = TOOL_SCHEMAS["memory_write"]["input_schema"]["properties"]["source_kind"]["enum"]  # type: ignore[index]
    assert source_kinds == ["user_statement", "tool_result", "agent_inference"]
    assert [schema["name"] for schema in tool_schemas()] == list(TOOL_SCHEMAS)
    copied_schemas = tool_schemas()
    copied_schemas[0]["name"] = "changed"
    assert TOOL_SCHEMAS["memory_search"]["name"] == "memory_search"

    assert validate_tool_input("memory_search", {"queries": ["editor preference"], "k": 2})["k"] == 2
    assert validate_tool_input("memory_get", {"ids": ["mem-1"]})["ids"] == ["mem-1"]
    assert validate_tool_input("memory_write", _write_payload())["type"] == "semantic"
    assert (
        validate_tool_input("memory_revise", {"id": "mem-1", "action": "expire", "reason": "obsolete"})["action"]
        == "expire"
    )
    assert validate_tool_input("memory_forget", {"id": "mem-1", "reason": "user request"})["id"] == "mem-1"

    with pytest.raises(ToolInputError, match="unknown field"):
        validate_tool_input("memory_search", {"queries": ["editor"], "context": "must come from adapter"})
    with pytest.raises(ToolInputError, match="evidence is required"):
        validate_tool_input(
            "memory_write",
            {"type": "semantic", "content": "Aditya uses Vim.", "attribute": "editor", "source_kind": "user_statement"},
        )
    with pytest.raises(ToolInputError, match="supersede.content"):
        validate_tool_input("memory_revise", {"id": "mem-1", "action": "supersede", "reason": "correction"})


def test_handlers_write_get_revise_and_forget_a_memory(
    handlers: tuple[ToolHandlers, FakeEmbedder], store: Store
) -> None:
    tool_handlers, embedder = handlers
    payload = _write_payload()
    embedder.set_similarity("concise explanations", payload["content"], 0.90)  # type: ignore[arg-type]

    written = tool_handlers.memory_write(_PRINCIPAL, payload)

    assert written["ok"] is True
    record_id = written["record_id"]
    assert isinstance(record_id, str)
    fetched = tool_handlers.memory_get(_PRINCIPAL, {"ids": [record_id]})
    record = fetched["records"][0]["record"]  # type: ignore[index]
    assert fetched["ok"] is True
    assert record["content"] == payload["content"]  # type: ignore[index]
    assert record["entities"][0]["canonical"] == "Aditya"  # type: ignore[index]
    assert fetched["records"][0]["events"][0]["kind"] == "record.created"  # type: ignore[index]
    searched_before_revision = tool_handlers.memory_search(_PRINCIPAL, {"queries": ["concise explanations"]})
    assert searched_before_revision["results"][0]["record"]["id"] == record_id  # type: ignore[index]

    revised = tool_handlers.memory_revise(
        _PRINCIPAL,
        {
            "id": record_id,
            "action": "supersede",
            "content": "The user prefers concise answers.",
            "source_kind": "user_statement",
            "evidence": "I prefer concise technical explanations.",
            "reason": "user correction",
        },
    )

    assert revised["ok"] is True
    revision_id = revised["record_id"]
    assert isinstance(revision_id, str)
    lineage = tool_handlers.memory_get(_PRINCIPAL, {"ids": [record_id]})
    assert lineage["records"][0]["lineage"]["successors"][0]["id"] == revision_id  # type: ignore[index]

    forgotten = tool_handlers.memory_forget(_PRINCIPAL, {"id": record_id, "reason": "user request"})
    searched = tool_handlers.memory_search(_PRINCIPAL, {"queries": ["concise explanations"]})
    tombstone = tool_handlers.memory_get(_PRINCIPAL, {"ids": [record_id]})

    assert forgotten == {"ok": True, "record_id": record_id, "status": "deleted", "outcome": "forgotten"}
    assert searched["results"] == []
    tombstoned = tombstone["records"][0]["record"]  # type: ignore[index]
    assert tombstoned["status"] == "deleted"  # type: ignore[index]
    assert tombstoned["tombstone"] is True  # type: ignore[index]
    assert tombstoned["content"] is None  # type: ignore[index]
    assert tombstoned["evidence"] is None  # type: ignore[index]
    assert store.fts_query('"technical"', limit=10) == []


def test_revise_requires_write_access_even_when_the_record_is_readable(
    handlers: tuple[ToolHandlers, FakeEmbedder], store: Store
) -> None:
    tool_handlers, _ = handlers
    written = tool_handlers.memory_write(_PRINCIPAL, _write_payload())
    record_id = written["record_id"]
    assert isinstance(record_id, str)
    reader = Principal("read-only-agent", _USER, _SESSION, None)
    store.set_grant(reader.agent_id, _USER_SCOPE, can_read=True, can_write=False)

    response = tool_handlers.memory_revise(reader, {"id": record_id, "action": "confirm", "reason": "reviewed"})

    assert response["ok"] is False
    assert response["error"]["code"] == "scope_not_writable"  # type: ignore[index]


def test_write_with_an_unreadable_entity_returns_a_structured_error(
    handlers: tuple[ToolHandlers, FakeEmbedder], store: Store
) -> None:
    tool_handlers, _ = handlers
    hidden_entity = store.create_entity(
        kind="person",
        canonical="Other User",
        scope=Scope(kind="user", id="other-user"),
        entity_id="other-user",
    )
    payload = _write_payload()
    payload["entities"] = [{"kind": "person", "name": "Other User", "role": "about", "entity_id": hidden_entity.id}]

    response = tool_handlers.memory_write(_PRINCIPAL, payload)

    assert response["ok"] is False
    assert response["error"]["code"] == "not_found"  # type: ignore[index]
    assert hidden_entity.id not in str(response["error"])


def test_revise_can_merge_entities_with_the_principal_write_authority(
    handlers: tuple[ToolHandlers, FakeEmbedder], store: Store
) -> None:
    tool_handlers, _ = handlers
    source = store.create_entity(kind="project", canonical="Memory Layer", scope=_USER_SCOPE, entity_id="source")
    destination = store.create_entity(
        kind="project", canonical="Memory Weave", scope=_USER_SCOPE, entity_id="destination"
    )

    response = tool_handlers.memory_revise(
        _PRINCIPAL,
        {"entity_id": source.id, "merge_into": destination.id, "reason": "same project"},
    )

    assert response == {
        "ok": True,
        "entity": {
            "id": destination.id,
            "kind": "project",
            "canonical": "Memory Weave",
            "scope": {"kind": "user", "id": _USER},
            "status": "provisional",
        },
        "outcome": "merged",
    }


def test_render_search_matches_the_agent_facing_golden_blocks() -> None:
    responses = _render_responses()
    golden_dir = Path(__file__).parent / "golden"

    assert render_search(responses[0]) == (golden_dir / "memory_search_results.txt").read_text(encoding="utf-8").rstrip(
        "\n"
    )
    assert render_search(responses[1]) == (golden_dir / "memory_search_empty.txt").read_text(encoding="utf-8").rstrip(
        "\n"
    )


def _render_responses() -> tuple[SearchResponse, SearchResponse]:
    records = [
        _record("mem-alpha", "Use concise technical explanations."),
        _record("mem-beta", "The project uses SQLite."),
        _record("mem-gamma", "Prefer direct answers."),
    ]
    results = [
        SearchResult(record, 0.90 - index / 10, _explanation(f"block {index + 1}"))
        for index, record in enumerate(records)
    ]
    return (
        SearchResponse("search-1", ["what should the response style be"], None, "disabled", results, None, {}),
        SearchResponse("search-2", ["unknown preference"], None, "disabled", [], "no matching memory", {}),
    )


def _record(record_id: str, content: str) -> Record:
    return Record(
        id=record_id,
        type="semantic",
        version=1,
        content=content,
        subject="person:aditya/style",
        subject_entity_id="person-aditya",
        attribute="style",
        scope=_USER_SCOPE,
        source_kind="user_statement",
        source_ref=None,
        creator_agent_id=_AGENT,
        evidence=None,
        created_at=_NOW,
        event_at=_NOW,
        expires_at=None,
        confidence=0.95,
        status="confirmed",
        supersedes_id=None,
        reinforcements=0,
        last_reinforced_at=None,
        tags=[],
        entity_ids=[],
    )


def _explanation(summary: str) -> Explanation:
    return Explanation(
        raw_queries=[],
        rewritten_queries=None,
        rewrite_status="disabled",
        matched_by=[],
        dense=None,
        lexical=None,
        lexical_terms=None,
        entity=None,
        fused_rank=1,
        freshness_multiplier=None,
        rerank=None,
        gate="passed",
        dedup="kept",
        budget="fit",
        source_kind="user_statement",
        status="confirmed",
        created_at=_NOW,
        event_at=_NOW,
        entity_ids=[],
        conflicts_with=[],
        summary=summary,
    )


def _foreign_record(store: Store, embedder: FakeEmbedder) -> Record:
    """Store one record in another user's scope that the test principal can never reach."""

    foreign = Record(
        id="foreign-record",
        type="semantic",
        version=1,
        content="The other user prefers detailed answers.",
        subject="foreign-entity/explanation_style",
        subject_entity_id=None,
        attribute="explanation_style",
        scope=Scope(kind="user", id="other-user"),
        source_kind="user_statement",
        source_ref=None,
        creator_agent_id="other-agent",
        evidence=None,
        created_at=_NOW,
        event_at=_NOW,
        expires_at=None,
        confidence=0.95,
        status="confirmed",
        supersedes_id=None,
        reinforcements=0,
        last_reinforced_at=None,
        tags=[],
        entity_ids=[],
    )
    store.insert_record(foreign)
    store.put_embedding(foreign.id, embedder.name, embedder.version, embedder.embed_documents([foreign.content])[0])
    return foreign


def test_get_revise_and_forget_treat_a_foreign_record_as_absent(
    handlers: tuple[ToolHandlers, FakeEmbedder], store: Store
) -> None:
    tool_handlers, embedder = handlers
    foreign = _foreign_record(store, embedder)

    fetched = tool_handlers.memory_get(_PRINCIPAL, {"ids": [foreign.id]})
    revised = tool_handlers.memory_revise(_PRINCIPAL, {"id": foreign.id, "action": "confirm", "reason": "review"})
    forgotten = tool_handlers.memory_forget(_PRINCIPAL, {"id": foreign.id, "reason": "cleanup"})
    missing = tool_handlers.memory_get(_PRINCIPAL, {"ids": ["never-existed"]})

    for response in (fetched, revised, forgotten):
        assert response["ok"] is False
        assert response["error"]["code"] == "not_found"  # type: ignore[index]
        assert foreign.content not in str(response)
    assert missing["error"]["message"].replace("never-existed", "X") == (  # type: ignore[index]
        fetched["error"]["message"].replace(foreign.id, "X")  # type: ignore[index]
    )
    assert store.get_record(foreign.id).status == "confirmed"  # type: ignore[union-attr]


def test_forget_requires_write_access_even_when_the_record_is_readable(
    handlers: tuple[ToolHandlers, FakeEmbedder], store: Store
) -> None:
    tool_handlers, _ = handlers
    written = tool_handlers.memory_write(_PRINCIPAL, _write_payload())
    record_id = written["record_id"]
    assert isinstance(record_id, str)
    reader = Principal("read-only-agent", _USER, _SESSION, None)
    store.set_grant(reader.agent_id, _USER_SCOPE, can_read=True, can_write=False)

    response = tool_handlers.memory_forget(reader, {"id": record_id, "reason": "not mine to remove"})

    assert response["error"]["code"] == "scope_not_writable"  # type: ignore[index]
    assert store.get_record(record_id).status == "confirmed"  # type: ignore[union-attr]


def test_a_forgotten_or_superseded_record_cannot_be_revised(
    handlers: tuple[ToolHandlers, FakeEmbedder], store: Store
) -> None:
    tool_handlers, _ = handlers
    written = tool_handlers.memory_write(_PRINCIPAL, _write_payload())
    record_id = written["record_id"]
    assert isinstance(record_id, str)
    assert tool_handlers.memory_forget(_PRINCIPAL, {"id": record_id, "reason": "user request"})["ok"] is True

    resurrect = tool_handlers.memory_revise(
        _PRINCIPAL,
        {
            "id": record_id,
            "action": "supersede",
            "content": "The user prefers detailed answers.",
            "source_kind": "user_statement",
            "evidence": "I prefer concise technical explanations.",
            "reason": "resurrection attempt",
        },
    )
    confirmed = tool_handlers.memory_revise(_PRINCIPAL, {"id": record_id, "action": "confirm", "reason": "attempt"})

    for response in (resurrect, confirmed):
        assert response["ok"] is False
        assert response["error"]["code"] == "invalid_input"  # type: ignore[index]
    assert store.get_record(record_id).status == "deleted"  # type: ignore[union-attr]
    assert store.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1


def test_confirm_clears_a_provisional_expiry(handlers: tuple[ToolHandlers, FakeEmbedder], store: Store) -> None:
    tool_handlers, _ = handlers
    payload = _write_payload("The user seems to prefer concise answers.")
    payload["source_kind"] = "agent_inference"
    del payload["evidence"]
    written = tool_handlers.memory_write(_PRINCIPAL, payload)
    record_id = written["record_id"]
    assert isinstance(record_id, str)
    assert store.get_record(record_id).expires_at is not None  # type: ignore[union-attr]

    response = tool_handlers.memory_revise(_PRINCIPAL, {"id": record_id, "action": "confirm", "reason": "user agreed"})

    assert response["ok"] is True
    confirmed = store.get_record(record_id)
    assert confirmed is not None
    assert confirmed.status == "confirmed"
    assert confirmed.expires_at is None


def test_supersede_requires_authority_and_keeps_evidence_provenance_and_roles(
    handlers: tuple[ToolHandlers, FakeEmbedder], store: Store
) -> None:
    tool_handlers, _ = handlers
    written = tool_handlers.memory_write(_PRINCIPAL, _write_payload())
    record_id = written["record_id"]
    assert isinstance(record_id, str)

    guessed = tool_handlers.memory_revise(
        _PRINCIPAL,
        {
            "id": record_id,
            "action": "supersede",
            "content": "The user prefers detailed answers.",
            "reason": "a guess",
        },
    )

    assert guessed["ok"] is False
    assert guessed["error"]["code"] == "invalid_input"  # type: ignore[index]
    assert store.get_record(record_id).status == "confirmed"  # type: ignore[union-attr]

    evidenced = tool_handlers.memory_revise(
        _PRINCIPAL,
        {
            "id": record_id,
            "action": "supersede",
            "content": "The user prefers concise answers.",
            "source_kind": "user_statement",
            "evidence": "I prefer concise technical explanations.",
            "reason": "user correction",
        },
    )

    assert evidenced["ok"] is True
    revision_id = evidenced["record_id"]
    assert isinstance(revision_id, str)
    revision = store.get_record(revision_id)
    original = store.get_record(record_id)
    assert revision is not None and original is not None
    assert revision.source_kind == "user_statement"
    assert revision.status == "confirmed"
    assert revision.expires_at is None
    assert revision.source_ref == f"session:{_SESSION}<turn:1>"
    assert store.record_entity_roles(revision.id) == store.record_entity_roles(record_id)
    assert original.status == "superseded"


def test_tool_input_rejects_non_object_payloads_and_blank_text() -> None:
    for payload in (None, ["queries"], [{"queries": ["x"]}], "queries"):
        with pytest.raises(ToolInputError, match="must be a JSON object"):
            validate_tool_input("memory_search", payload)  # type: ignore[arg-type]
    with pytest.raises(ToolInputError, match="must not be blank"):
        validate_tool_input(
            "memory_write",
            {"type": "semantic", "content": "   ", "attribute": "editor", "source_kind": "agent_inference"},
        )


def test_memory_revise_schema_declares_every_property_it_accepts() -> None:
    schema = TOOL_SCHEMAS["memory_revise"]["input_schema"]
    declared = set(schema["properties"])  # type: ignore[index, arg-type]
    assert schema["additionalProperties"] is False  # type: ignore[index]
    for branch in schema["oneOf"]:  # type: ignore[index]
        # A key required by a branch must be declared at the top level, or additionalProperties rejects it.
        assert set(branch["required"]) <= declared
    assert {"id", "action", "content", "reason", "entity_id", "merge_into"} <= declared
    assert validate_tool_input("memory_revise", {"entity_id": "a", "merge_into": "b", "reason": "same person"})
    with pytest.raises(ToolInputError):
        validate_tool_input("memory_revise", {"id": "m", "action": "confirm", "reason": "r", "entity_id": "e"})


def test_get_omits_entities_the_caller_cannot_read(handlers: tuple[ToolHandlers, FakeEmbedder], store: Store) -> None:
    tool_handlers, _ = handlers
    written = tool_handlers.memory_write(_PRINCIPAL, _write_payload())
    record_id = written["record_id"]
    assert isinstance(record_id, str)
    foreign_entity = store.create_entity(
        kind="project", canonical="Other Project", scope=Scope(kind="project", id="secret"), entity_id="secret-project"
    )
    store.link_record_entity(record_id, foreign_entity.id, "mentions")

    fetched = tool_handlers.memory_get(_PRINCIPAL, {"ids": [record_id]})

    entity_ids = [entity["id"] for entity in fetched["records"][0]["record"]["entities"]]  # type: ignore[index]
    assert foreign_entity.id not in entity_ids
    assert "Other Project" not in str(fetched)
