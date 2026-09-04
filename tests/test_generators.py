from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memory_weave.config import EmbeddingConfig
from memory_weave.index.embedder import FakeEmbedder
from memory_weave.index.vector import VectorIndex
from memory_weave.models import MemoryType, Record, Scope
from memory_weave.retrieve import STOPWORDS, dense_candidates, entity_candidates, lexical_candidates
from memory_weave.store import Store
from tests.fixtures import build_store

_NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
_EMBEDDING = EmbeddingConfig(model="fake-embedder", version="1", dims=8)
_AGENT_SCOPE = Scope(kind="agent", id="research-agent")
_PROJECT_SCOPE = Scope(kind="project", id="memory-weave")


@pytest.fixture
def store(tmp_path: Path) -> Store:
    database = Store(tmp_path / "memory.sqlite")
    _ = database.connection
    yield database
    database.close()


def _record(
    record_id: str,
    content: str,
    *,
    scope: Scope = _AGENT_SCOPE,
    event_at: datetime = _NOW,
    memory_type: MemoryType = "semantic",
) -> Record:
    return Record(
        id=record_id,
        type=memory_type,
        version=1,
        content=content,
        subject=f"project:memory-weave/setting_{record_id}",
        scope=scope,
        source_kind="user_statement",
        source_ref=None,
        creator_agent_id="research-agent",
        evidence=None,
        created_at=event_at,
        event_at=event_at,
        expires_at=None,
        confidence=0.95,
        status="confirmed",
        supersedes_id=None,
        reinforcements=0,
        last_reinforced_at=None,
        tags=[],
        entity_ids=[],
    )


def _store_record(store: Store, record: Record, *, aliases: str = "") -> None:
    store.insert_record(record)
    store.upsert_fts(record.id, record.content, record.subject, aliases)


def test_stopwords_match_the_documented_initial_english_list_size() -> None:
    assert len(STOPWORDS) == 100


def test_dense_candidates_union_queries_by_their_best_cosine_and_respect_the_mask() -> None:
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    index = VectorIndex(_EMBEDDING)
    first_content = "First preference record"
    second_content = "Second preference record"
    first_query = "first query"
    second_query = "second query"
    embedder.set_similarity(first_content, first_query, 0.90)
    embedder.set_similarity(second_content, second_query, 0.95)
    index.upsert("first", embedder.embed_documents([first_content])[0])
    index.upsert("second", embedder.embed_documents([second_content])[0])

    candidates = dense_candidates([first_query, second_query], embedder, index, index.mask({"first", "second"}), k=2)

    assert [(record_id, hit.rank) for record_id, hit in candidates] == [("second", 1), ("first", 2)]
    assert candidates[0][1].score == pytest.approx(0.95)
    assert candidates[1][1].score == pytest.approx(0.90)
    masked = dense_candidates([first_query], embedder, index, index.mask({"first"}), k=2)
    assert [(record_id, hit.rank) for record_id, hit in masked] == [("first", 1)]
    assert masked[0][1].score == pytest.approx(0.90)


def test_lexical_candidates_match_identifier_terms_count_matches_and_never_leak_ineligible_rows(store: Store) -> None:
    visible = _record("visible", "Deployment failed with ERR42 during refresh.")
    hidden = _record("hidden", "Private deployment also failed with ERR42.", scope=_PROJECT_SCOPE)
    _store_record(store, visible)
    _store_record(store, hidden)

    candidates = lexical_candidates(["How did ERR42 deployment fail?"], store, {visible.id}, k=2)

    assert [(record_id, hit.rank, matched, total) for record_id, hit, matched, total in candidates] == [
        (visible.id, 1, 2, 3)
    ]
    assert lexical_candidates(['" ERR42 NEAR deployment'], store, {visible.id}, k=2)[0][0] == visible.id


def test_lexical_candidates_keep_identifiers_and_proper_nouns_when_stopwords_include_them(store: Store) -> None:
    record = _record("visible", "Alice fixed ERR42.")
    _store_record(store, record)

    candidates = lexical_candidates(
        ["Tell Alice about ERR42"], store, {record.id}, k=2, stopwords={"alice", "err42", "tell", "about"}
    )

    assert [(record_id, matched, total) for record_id, _, matched, total in candidates] == [(record.id, 2, 2)]


def test_entity_candidates_find_query_aliases_in_recency_order_and_respect_scopes(store: Store) -> None:
    aditya = store.create_entity(kind="person", canonical="Aditya Mishra", scope=_AGENT_SCOPE, entity_id="aditya")
    store.add_alias(aditya.id, "aditya mishra")
    newest = _record("newest", "He approved the migration.", event_at=_NOW + timedelta(days=1))
    oldest = _record("oldest", "He created the project.", event_at=_NOW)
    for record in (newest, oldest):
        _store_record(store, record, aliases="aditya mishra")
        store.link_record_entity(record.id, aditya.id, "about")

    hidden = store.create_entity(kind="person", canonical="Merlin", scope=_PROJECT_SCOPE, entity_id="merlin")
    store.add_alias(hidden.id, "merlin")
    hidden_record = _record("hidden", "Merlin owns the project.", scope=_PROJECT_SCOPE)
    _store_record(store, hidden_record, aliases="merlin")
    store.link_record_entity(hidden_record.id, hidden.id, "about")

    candidates = entity_candidates(
        None, ["What did Aditya Mishra decide?"], store, [_AGENT_SCOPE], {newest.id, oldest.id}, 3
    )

    assert [(record_id, hit.rank, entity_id) for record_id, hit, entity_id in candidates] == [
        (newest.id, 1, aditya.id),
        (oldest.id, 2, aditya.id),
    ]
    assert entity_candidates(None, ["What did Merlin decide?"], store, [_AGENT_SCOPE], {hidden_record.id}, 3) == []
    assert entity_candidates(None, ["What did unknown person decide?"], store, [_AGENT_SCOPE], {newest.id}, 3) == []


def test_entity_store_query_filters_eligibility_before_applying_its_limit(store: Store) -> None:
    entity = store.create_entity(kind="project", canonical="Memory Weave", scope=_AGENT_SCOPE, entity_id="memory-weave")
    store.add_alias(entity.id, "memory weave")
    ineligible = _record("ineligible", "Newest memory.", event_at=_NOW + timedelta(days=2))
    eligible = _record("eligible", "Older memory.", event_at=_NOW)
    for record in (ineligible, eligible):
        _store_record(store, record)
        store.link_record_entity(record.id, entity.id, "about")

    assert store.records_for_entities([entity.id], {eligible.id}, limit=1) == [(eligible.id, entity.id)]


def test_synthetic_store_fixture_writes_records_embeddings_fts_rows_and_entities(store: Store) -> None:
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)

    records = build_store(store, 6, [_AGENT_SCOPE, _PROJECT_SCOPE], ["semantic", "episodic"], 7, embedder)

    assert len(records) == 6
    assert store.count_embeddings(embedder.name, embedder.version) == 6
    assert store.fts_query("ERR00000", limit=1)[0][0] == records[0].id
    assert store.connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 2
