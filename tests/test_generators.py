from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memory_weave.config import EmbeddingConfig
from memory_weave.index.embedder import FakeEmbedder
from memory_weave.index.vector import VectorIndex
from memory_weave.models import MemoryType, Record, Scope
from memory_weave.policy import readable_scopes
from memory_weave.retrieve import STOPWORDS, dense_candidates, entity_candidates, lexical_candidates
from memory_weave.retrieve.generators import fts_match_expression
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


def test_stopwords_use_the_documented_frequency_based_english_list() -> None:
    assert len(STOPWORDS) >= 180
    assert {"what", "which", "who", "with", "you", "your", "use"} <= STOPWORDS


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

    candidates = dense_candidates([first_query, second_query], embedder, index, {"first", "second"}, k=2)

    assert [(record_id, hit.rank) for record_id, hit in candidates] == [("second", 1), ("first", 2)]
    assert candidates[0][1].score == pytest.approx(0.95)
    assert candidates[1][1].score == pytest.approx(0.90)
    masked = dense_candidates([first_query], embedder, index, {"first"}, k=2)
    assert [(record_id, hit.rank) for record_id, hit in masked] == [("first", 1)]
    assert masked[0][1].score == pytest.approx(0.90)


def test_dense_candidates_cap_the_multi_query_union_at_k() -> None:
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    index = VectorIndex(_EMBEDDING)
    contents = ["First record", "Second record", "Third record"]
    queries = ["query one", "query two", "query three"]
    for position, (content, query) in enumerate(zip(contents, queries, strict=True)):
        embedder.set_similarity(content, query, 0.90 - position * 0.01)
        index.upsert(str(position), embedder.embed_documents([content])[0])

    candidates = dense_candidates(queries, embedder, index, {"0", "1", "2"}, k=2)

    assert [record_id for record_id, _ in candidates] == ["0", "1"]


def test_lexical_candidates_match_identifier_terms_count_matches_and_never_leak_ineligible_rows(store: Store) -> None:
    visible = _record("visible", "Deployment failed with ERR42 during refresh.")
    hidden = _record("hidden", "Private deployment also failed with ERR42.", scope=_PROJECT_SCOPE)
    _store_record(store, visible)
    _store_record(store, hidden)

    candidates = lexical_candidates(["How did ERR42 deployment fail?"], store, {visible.id}, k=2)

    assert [(record_id, hit.rank, len(match.terms), match.total_terms) for record_id, hit, match in candidates] == [
        (visible.id, 1, 2, 3)
    ]
    assert candidates[0][2].terms[0].is_identifier
    assert lexical_candidates(['" ERR42 NEAR deployment'], store, {visible.id}, k=2)[0][0] == visible.id


def test_lexical_candidates_keep_identifiers_and_proper_nouns_when_stopwords_include_them(store: Store) -> None:
    record = _record("visible", "Alice fixed ERR42.")
    _store_record(store, record)

    candidates = lexical_candidates(
        ["Tell Alice about ERR42"], store, {record.id}, k=2, stopwords={"alice", "err42", "tell", "about"}
    )

    assert [(record_id, len(match.terms), match.total_terms) for record_id, _, match in candidates] == [
        (record.id, 2, 2)
    ]


def test_lexical_candidates_flag_matched_entity_alias_terms(store: Store) -> None:
    record = _record("visible", "Aditya Mishra approved the migration.")
    _store_record(store, record, aliases="aditya mishra")

    candidates = lexical_candidates(
        ["What did Aditya Mishra decide?"],
        store,
        {record.id},
        k=2,
        entity_aliases=["Aditya Mishra"],
    )

    assert [term.value for term in candidates[0][2].terms] == ["aditya", "mishra"]
    assert all(term.is_entity_alias for term in candidates[0][2].terms)


def test_lexical_candidates_apply_eligibility_before_fts_limit_and_escape_syntax(store: Store) -> None:
    visible = _record("visible", "The deployment failed with ERR42.")
    _store_record(store, visible)
    hidden_ids: set[str] = set()
    for index in range(7):
        hidden = _record(f"hidden-{index}", "The deployment failed with ERR42.", scope=_PROJECT_SCOPE)
        _store_record(store, hidden, aliases="ERR42 deployment")
        hidden_ids.add(hidden.id)

    candidates = lexical_candidates(["What did you say about ERR42?"], store, {visible.id}, k=2)

    assert [record_id for record_id, _, _ in candidates] == [visible.id]
    match = candidates[0][2]
    assert [(term.value, term.is_identifier) for term in match.terms] == [("err42", True)]
    assert match.total_terms == 1
    assert fts_match_expression(["err42", "near"]) == '"err42" OR "near"'
    assert hidden_ids.isdisjoint({record_id for record_id, _, _ in candidates})


def test_lexical_candidates_keep_the_best_term_fraction_for_one_query_and_drop_possessive_fragments(
    store: Store,
) -> None:
    record = _record("visible", "Aditya prefers Vim as an editor.")
    _store_record(store, record)

    candidates = lexical_candidates(
        ["What is the user's editor preference?", "Aditya editor"],
        store,
        {record.id},
        k=2,
    )

    assert len(candidates) == 1
    match = candidates[0][2]
    assert match.fraction == 1.0
    assert {term.value for term in match.terms} == {"editor", "aditya"}
    assert "s" not in {term.value for term in match.terms}


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


def test_entity_candidates_fold_possessives_before_building_query_alias_n_grams(store: Store) -> None:
    aditya = store.create_entity(kind="person", canonical="Aditya Mishra", scope=_AGENT_SCOPE, entity_id="aditya")
    store.add_alias(aditya.id, "aditya")
    record = _record("editor", "Aditya uses Vim as an editor.")
    _store_record(store, record, aliases="aditya")
    store.link_record_entity(record.id, aditya.id, "about")

    candidates = entity_candidates(None, ["What is Aditya's editor?"], store, [_AGENT_SCOPE], {record.id}, 3)

    assert [(record_id, entity_id) for record_id, _, entity_id in candidates] == [(record.id, aditya.id)]


def test_entity_candidates_batch_alias_lookup_for_a_long_query(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    entity = store.create_entity(kind="person", canonical="Aditya Mishra", scope=_AGENT_SCOPE, entity_id="aditya")
    store.add_alias(entity.id, "aditya mishra")
    record = _record("visible", "Aditya approved the migration.")
    _store_record(store, record, aliases="aditya mishra")
    store.link_record_entity(record.id, entity.id)
    calls = 0
    original = store.entities_by_aliases

    def counted(*args: object, **kwargs: object) -> dict[str, list[object]]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(store, "entities_by_aliases", counted)

    candidates = entity_candidates(
        None,
        ["What did Aditya Mishra decide about the current migration today?"],
        store,
        [_AGENT_SCOPE],
        {record.id},
        2,
    )

    assert [record_id for record_id, _, _ in candidates] == [record.id]
    assert calls == 1


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


def test_private_scope_isolation_holds_for_dense_lexical_and_entity_generators(store: Store) -> None:
    agent_id = "shared-agent"
    user_u = "user-u"
    user_v = "user-v"
    private_u = Scope(kind="agent", id=f"{agent_id}/{user_u}")
    private_v = Scope(kind="agent", id=f"{agent_id}/{user_v}")
    visible = _record("visible", "Aditya uses Vim.", scope=private_u)
    hidden = _record("hidden", "Aditya uses Emacs.", scope=private_v)
    person_u = store.create_entity(kind="person", canonical="Aditya", scope=private_u, entity_id="person-u")
    person_v = store.create_entity(kind="person", canonical="Aditya", scope=private_v, entity_id="person-v")
    for entity in (person_u, person_v):
        store.add_alias(entity.id, "aditya")
    for record, entity in ((visible, person_u), (hidden, person_v)):
        _store_record(store, record, aliases="aditya")
        store.link_record_entity(record.id, entity.id)

    eligible = store.eligible_ids(readable_scopes(store, agent_id, user_u), None, None, None, False, _NOW)
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    index = VectorIndex(_EMBEDDING)
    index.upsert(visible.id, embedder.embed_documents([visible.content])[0])
    index.upsert(hidden.id, embedder.embed_documents([hidden.content])[0])

    dense = dense_candidates(["What editor does Aditya use?"], embedder, index, eligible, 2)
    lexical = lexical_candidates(["What editor does Aditya use?"], store, eligible, 2)
    entity = entity_candidates(
        None, ["What did Aditya decide?"], store, readable_scopes(store, agent_id, user_u), eligible, 2
    )

    assert [record_id for record_id, _ in dense] == [visible.id]
    assert [record_id for record_id, _, _ in lexical] == [visible.id]
    assert [record_id for record_id, _, _ in entity] == [visible.id]


def test_entity_candidates_survive_a_pasted_paragraph_without_exhausting_sql_variables(store: Store) -> None:
    from memory_weave.retrieve.generators import _entity_aliases

    long_query = " ".join(f"token{index}" for index in range(300))
    aliases = _entity_aliases((), [long_query], 4)
    assert len(aliases) < 4 * 300 + 4

    assert entity_candidates(None, [long_query], store, [_AGENT_SCOPE], {"any"}, 3) == []


def test_query_terms_drop_contractions_and_possessive_suffixes() -> None:
    from memory_weave.retrieve.generators import _query_terms

    assert [term.value for term in _query_terms("I'm on the ERR42 issue", STOPWORDS)] == ["err42", "issue"]
    assert [term.value for term in _query_terms("What\u2019s Aditya\u2019s deploy command?", STOPWORDS)][:2] == [
        "aditya",
        "deploy",
    ]
    assert "don" not in [term.value for term in _query_terms("Don't ship it", STOPWORDS)]


def test_hyphenated_words_are_not_identifiers_and_multi_part_terms_need_adjacency() -> None:
    from memory_weave.retrieve.generators import _is_identifier, _query_terms, _term_matches

    terms = {
        term.value: term for term in _query_terms("Send a follow-up e-mail about bge-m3 and deploy.yml", STOPWORDS)
    }
    assert terms["follow-up"].is_identifier is False
    assert terms["bge-m3"].is_identifier is True
    assert terms["deploy.yml"].is_identifier is True
    assert _is_identifier("e-mail") is False

    scattered = [["please", "follow", "the", "instructions", "and", "clean", "up"]]
    adjacent = [["send", "a", "follow", "up", "note"]]
    assert _term_matches("follow-up", scattered, {"follow", "up"}) is False
    assert _term_matches("follow-up", adjacent, {"follow", "up"}) is True
