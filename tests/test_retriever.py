"""End-to-end retrieval tests using deterministic vectors and a temporary SQLite store."""

from __future__ import annotations

import json
from dataclasses import asdict, fields, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Thread

import pytest

from memory_weave.config import EmbeddingConfig, MemoryWeaveConfig, RerankerConfig, RetrievalConfig
from memory_weave.index.embedder import FakeEmbedder
from memory_weave.index.reranker import Reranker
from memory_weave.index.vector import VectorIndex
from memory_weave.models import Explanation, Principal, Record, Scope, SearchRequest
from memory_weave.retrieve.retriever import Retriever
from memory_weave.store import Store

_NOW = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
_AGENT = "research-agent"
_USER = "aditya"
_USER_SCOPE = Scope(kind="user", id=_USER)
_OTHER_SCOPE = Scope(kind="user", id="other-user")
_PRINCIPAL = Principal(_AGENT, _USER, "session-1", None)
_EMBEDDING = EmbeddingConfig(model="fake-embedder", version="1", dims=8)
_CONFIG = MemoryWeaveConfig(embedding=_EMBEDDING, retrieval=RetrievalConfig(per_generator_k=10, default_k=8))


@pytest.fixture
def store(tmp_path: Path) -> Store:
    database = Store(tmp_path / "memory.sqlite")
    _ = database.connection
    database.set_grant(_AGENT, _USER_SCOPE, can_read=True, can_write=True)
    yield database
    database.close()


def _record(
    record_id: str,
    content: str,
    *,
    scope: Scope = _USER_SCOPE,
    memory_type: str = "semantic",
    source_kind: str = "user_statement",
    status: str = "confirmed",
    event_at: datetime = _NOW,
) -> Record:
    return Record(
        id=record_id,
        type=memory_type,  # type: ignore[arg-type]
        version=1,
        content=content,
        subject=f"person:aditya/{record_id}",
        scope=scope,
        source_kind=source_kind,  # type: ignore[arg-type]
        source_ref=None,
        creator_agent_id=_AGENT,
        evidence=None,
        created_at=event_at,
        event_at=event_at,
        expires_at=None,
        confidence=0.95,
        status=status,  # type: ignore[arg-type]
        supersedes_id=None,
        reinforcements=0,
        last_reinforced_at=None,
        tags=[],
        entity_ids=[],
    )


def _store_record(store: Store, embedder: FakeEmbedder, record: Record) -> None:
    store.insert_record(record)
    store.put_embedding(record.id, embedder.name, embedder.version, embedder.embed_documents([record.content])[0])
    store.upsert_fts(record.id, record.content, record.subject, "")


def _request(query: str, *, trigger: str = "tool", k: int = 8) -> SearchRequest:
    return SearchRequest([query], None, None, None, None, None, k, False, trigger=trigger)  # type: ignore[arg-type]


def _retriever(
    store: Store,
    embedder: FakeEmbedder,
    *,
    config: MemoryWeaveConfig = _CONFIG,
    reranker: Reranker | None = None,
) -> Retriever:
    return Retriever(
        store, VectorIndex(config.embedding), embedder, config, reranker=reranker, current_time=lambda: _NOW
    )


def _json_explanation(explanation: Explanation) -> dict[str, object]:
    return json.loads(json.dumps(asdict(explanation), default=lambda value: value.isoformat()))


class _ScoredReranker:
    def __init__(self, scores: dict[tuple[str, str], float]) -> None:
        self._scores = scores

    @property
    def is_loaded(self) -> bool:
        return True

    def score(self, query: str, document: str) -> float:
        return self._scores[(query, document)]


def test_search_fuses_generators_and_writes_a_replayable_log(store: Store) -> None:
    query = "Aditya editor preference"
    content = "Aditya uses Vim as the preferred editor."
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    embedder.set_similarity(query, content, 0.82)
    _store_record(store, embedder, _record("editor", content))

    response = _retriever(store, embedder).search(_PRINCIPAL, _request(query))

    assert [result.record.id for result in response.results] == ["editor"]
    explanation = response.results[0].explanation
    assert explanation.dense is not None
    assert explanation.lexical is not None
    assert explanation.matched_by == ["dense", "lexical"]
    log = store.read_search_log(response.search_id)
    assert log is not None
    assert log["returned"] == ["editor"]
    assert log["rewrite_status"] == "disabled"
    assert log["rewritten_queries"] is None
    assert log["fused"][0]["record_id"] == "editor"  # type: ignore[index]
    assert {"rewrite", "dense", "entity", "gate", "total"} <= set(log["timings_ms"])  # type: ignore[arg-type]
    assert "log" in response.timings_ms
    logged_explanation = log["explanations"][0]  # type: ignore[index]
    assert set(logged_explanation) == {field.name for field in fields(Explanation)}  # type: ignore[arg-type]
    assert logged_explanation == _json_explanation(explanation)
    assert log["dense"][0] == {"record_id": "editor", "rank": 1, "score": explanation.dense.score}  # type: ignore[index,union-attr]
    assert log["lexical"][0]["rank"] == explanation.lexical.rank  # type: ignore[index,union-attr]
    assert log["lexical"][0]["score"] == explanation.lexical.score  # type: ignore[index,union-attr]
    assert log["fused"][0]["fused_rank"] == explanation.fused_rank  # type: ignore[index]


def test_scope_filter_applies_before_dense_search(store: Store) -> None:
    query = "editor preference"
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    visible = _record("visible", "Aditya uses Vim.")
    hidden = _record("hidden", "Other user uses Emacs.", scope=_OTHER_SCOPE)
    embedder.set_similarity(query, visible.content, 0.80)
    embedder.set_similarity(query, hidden.content, 0.99)
    _store_record(store, embedder, visible)
    _store_record(store, embedder, hidden)

    response = _retriever(store, embedder).search(_PRINCIPAL, _request(query))

    assert [result.record.id for result in response.results] == [visible.id]


def test_freshness_reorders_episodic_records_without_changing_semantic_scores(store: Store) -> None:
    query = "project decision"
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    semantic = _record("semantic", "The project uses SQLite.")
    episodic = _record(
        "episodic",
        "The project decided to defer the migration.",
        memory_type="episodic",
        event_at=_NOW - timedelta(days=60),
    )
    embedder.set_similarity(query, semantic.content, 0.70)
    embedder.set_similarity(query, episodic.content, 0.90)
    _store_record(store, embedder, semantic)
    _store_record(store, embedder, episodic)

    response = _retriever(store, embedder).search(_PRINCIPAL, _request(query))

    results = {result.record.id: result for result in response.results}
    assert results[semantic.id].explanation.freshness_multiplier is None
    assert results[episodic.id].explanation.freshness_multiplier == pytest.approx(0.5)


def test_thirty_day_old_two_channel_episodic_record_survives_the_relative_floor(store: Store) -> None:
    query = "project decision"
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    current = _record("current", "The project decision is to use PostgreSQL.")
    old = _record(
        "old",
        "The project decision was to use SQLite.",
        memory_type="episodic",
        event_at=_NOW - timedelta(days=30),
    )
    embedder.set_similarity(query, current.content, 0.90)
    embedder.set_similarity(query, old.content, 0.60)
    _store_record(store, embedder, current)
    _store_record(store, embedder, old)

    response = _retriever(store, embedder).search(_PRINCIPAL, _request(query))
    results = {result.record.id: result for result in response.results}

    assert old.id in results
    assert results[old.id].explanation.matched_by == ["dense", "lexical"]
    assert results[old.id].explanation.freshness_multiplier == pytest.approx(0.5)


def test_auto_gate_excludes_session_summaries(store: Store) -> None:
    query = "what did we discuss"
    content = "The session discussed the editor preference."
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    embedder.set_similarity(query, content, 0.99)
    _store_record(store, embedder, _record("summary", content, source_kind="session_summary"))

    response = _retriever(store, embedder).search(_PRINCIPAL, _request(query, trigger="auto"))

    assert response.results == []
    assert response.empty_reason is not None
    assert "excluded source kind session_summary" in response.empty_reason


def test_auto_gate_can_reject_a_dense_hit_that_passes_a_tool_search(store: Store) -> None:
    query = "editor"
    content = "Aditya uses Vim as an editor."
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    embedder.set_similarity(query, content, 0.50)
    _store_record(store, embedder, _record("editor", content))
    retriever = _retriever(store, embedder)

    assert [result.record.id for result in retriever.search(_PRINCIPAL, _request(query)).results] == ["editor"]
    auto_response = retriever.search(_PRINCIPAL, _request(query, trigger="auto"))

    assert auto_response.results == []
    assert auto_response.empty_reason is not None
    assert "dense 0.50 < 0.55" in auto_response.empty_reason


def test_history_is_excluded_by_default_and_returned_when_requested(store: Store) -> None:
    query = "old editor preference"
    content = "Aditya used Vim."
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    embedder.set_similarity(query, content, 0.90)
    historical = _record("historical", content, status="superseded")
    _store_record(store, embedder, historical)
    retriever = _retriever(store, embedder)

    assert retriever.search(_PRINCIPAL, _request(query)).results == []
    history_request = replace(_request(query), include_history=True)
    assert [result.record.id for result in retriever.search(_PRINCIPAL, history_request).results] == [historical.id]


def test_provisional_conflict_returns_its_eligible_authority_counterpart_first(store: Store) -> None:
    query = "preferred editor"
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    authority = _record("authority", "Aditya uses Vim.")
    provisional = _record("provisional", "Aditya uses Emacs.", status="provisional")
    embedder.set_similarity(query, provisional.content, 0.90)
    embedder.set_similarity(query, authority.content, 0.60)
    _store_record(store, embedder, authority)
    _store_record(store, embedder, provisional)
    store.add_conflict(authority.id, provisional.id, _NOW)

    response = _retriever(store, embedder).search(_PRINCIPAL, _request(query, k=1))

    assert [result.record.id for result in response.results][:2] == [authority.id, provisional.id]
    assert response.results[0].explanation.dense is not None
    assert response.results[0].explanation.gate != "included as authority counterpart for a provisional conflict"
    assert response.results[1].explanation.conflicts_with == [authority.id]
    assert "conflicts with authority" in response.results[1].explanation.summary


def test_a_lower_authority_provisional_conflict_is_not_presented_as_the_authority(store: Store) -> None:
    query = "preferred editor"
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    higher = _record("higher", "Aditya uses Vim as the preferred editor.", status="provisional")
    lower = _record(
        "lower",
        "Aditya uses Emacs as the preferred editor.",
        status="provisional",
        source_kind="agent_inference",
    )
    embedder.set_similarity(query, higher.content, 0.90)
    embedder.set_similarity(query, lower.content, 0.80)
    _store_record(store, embedder, higher)
    _store_record(store, embedder, lower)
    store.add_conflict(higher.id, lower.id, _NOW)

    response = _retriever(store, embedder).search(_PRINCIPAL, _request(query, k=1))

    assert [result.record.id for result in response.results] == [higher.id, lower.id]


def test_conflict_explanations_do_not_reveal_records_outside_the_caller_scope(store: Store) -> None:
    query = "editor preference"
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    user_record = _record("u-rec", "Aditya uses Vim as an editor.")
    other_record = _record("v-rec", "The other user uses Emacs as an editor.", scope=_OTHER_SCOPE)
    embedder.set_similarity(query, user_record.content, 0.90)
    embedder.set_similarity(query, other_record.content, 0.90)
    _store_record(store, embedder, user_record)
    _store_record(store, embedder, other_record)
    store.set_grant(_AGENT, _OTHER_SCOPE, can_read=True, can_write=True)
    store.add_conflict(user_record.id, other_record.id, _NOW)

    user_response = _retriever(store, embedder).search(_PRINCIPAL, _request(query))
    other_principal = Principal(_AGENT, _OTHER_SCOPE.id, "session-2", None)
    other_response = _retriever(store, embedder).search(other_principal, _request(query))

    assert user_response.results[0].explanation.conflicts_with == []
    assert "v-rec" not in user_response.results[0].explanation.summary
    assert other_response.results[0].explanation.conflicts_with == []
    assert "u-rec" not in other_response.results[0].explanation.summary


def test_another_store_process_is_refreshed_before_search(tmp_path: Path) -> None:
    path = tmp_path / "shared.sqlite"
    reader = Store(path)
    writer = Store(path)
    _ = reader.connection
    writer.set_grant(_AGENT, _USER_SCOPE, can_read=True, can_write=True)
    query = "editor preference"
    content = "Aditya uses Vim."
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    embedder.set_similarity(query, content, 0.88)
    retriever = _retriever(reader, embedder)
    assert retriever.search(_PRINCIPAL, _request(query)).results == []
    _store_record(writer, embedder, _record("cross-process", content))

    response = retriever.search(_PRINCIPAL, _request(query))

    assert [result.record.id for result in response.results] == ["cross-process"]
    log = reader.read_search_log(response.search_id)
    assert log is not None
    assert log["config_flags"]["index_refresh"] == "delta"  # type: ignore[index]
    reader.close()
    writer.close()


def test_cross_process_write_during_refresh_is_visible_on_the_next_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "shared.sqlite"
    reader = Store(path)
    writer = Store(path)
    _ = reader.connection
    writer.set_grant(_AGENT, _USER_SCOPE, can_read=True, can_write=True)
    query = "editor preference"
    before = _record("before", "Aditya uses Vim as an editor.")
    during = _record("during", "Aditya uses Emacs as an editor.")
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    embedder.set_similarity(query, before.content, 0.90)
    embedder.set_similarity(query, during.content, 0.80)
    retriever = _retriever(reader, embedder)
    assert retriever.search(_PRINCIPAL, _request(query)).results == []
    _store_record(writer, embedder, before)

    writer_done = Event()
    writer_errors: list[BaseException] = []

    def write_during_refresh() -> None:
        try:
            _store_record(writer, embedder, during)
        except BaseException as exc:
            writer_errors.append(exc)
        finally:
            writer_done.set()

    original_changes_since = reader.index_changes_since
    write_thread = Thread(target=write_during_refresh)
    started_writer = False

    def changes_since(*args: object) -> object:
        nonlocal started_writer
        if not started_writer:
            started_writer = True
            write_thread.start()
            assert writer_done.wait(timeout=1)
        return original_changes_since(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(reader, "index_changes_since", changes_since)
    first_refresh = retriever.search(_PRINCIPAL, _request(query))
    write_thread.join(timeout=1)
    final_refresh = retriever.search(_PRINCIPAL, _request(query))

    assert writer_errors == []
    assert {result.record.id for result in first_refresh.results} == {before.id}
    assert {result.record.id for result in final_refresh.results} == {before.id, during.id}
    reader.close()
    writer.close()


def test_time_window_disables_episodic_freshness(store: Store) -> None:
    query = "past decision"
    content = "The project chose SQLite."
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    embedder.set_similarity(query, content, 0.90)
    old = _record("old", content, memory_type="episodic", event_at=_NOW - timedelta(days=60))
    _store_record(store, embedder, old)
    request = replace(_request(query), since=_NOW - timedelta(days=90))

    response = _retriever(store, embedder).search(_PRINCIPAL, request)

    assert response.results[0].explanation.freshness_multiplier is None


def test_empty_reason_names_reranking_or_budget_as_the_stage_that_removed_candidates(store: Store) -> None:
    query = "oversized editor preference"
    content = "oversized editor preference " + "x" * 500
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    embedder.set_similarity(query, content, 0.90)
    _store_record(store, embedder, _record("large", content))

    rerank_config = replace(_CONFIG, reranker=RerankerConfig(enabled=True, candidates=1, floor=0.10))
    rerank_response = _retriever(store, embedder, config=rerank_config).search(_PRINCIPAL, _request(query))
    budget_config = replace(_CONFIG, retrieval=replace(_CONFIG.retrieval, token_budget=40))
    budget_response = _retriever(store, embedder, config=budget_config).search(_PRINCIPAL, _request(query))

    assert rerank_response.results == []
    assert rerank_response.empty_reason == "all relevance-gated candidates missed reranker floor 0.10"
    assert budget_response.results == []
    assert budget_response.empty_reason == "all relevance-gated candidates exceeded the result or token budget"


def test_reranker_shortlist_omissions_are_recorded_separately_from_budget_out(store: Store) -> None:
    query = "editor preference"
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    first = _record("first", "The editor preference is Vim.")
    second = _record("second", "The editor preference is Emacs.")
    embedder.set_similarity(query, first.content, 0.90)
    embedder.set_similarity(query, second.content, 0.50)
    _store_record(store, embedder, first)
    _store_record(store, embedder, second)
    config = replace(_CONFIG, reranker=RerankerConfig(enabled=True, candidates=1, floor=0.0))

    response = _retriever(store, embedder, config=config).search(_PRINCIPAL, _request(query))
    log = store.read_search_log(response.search_id)

    assert log is not None
    assert any(
        row["reason"] == "reranker candidate limit" and row["limit"] == 1
        for row in log["reranked_out"]  # type: ignore[union-attr]
    )


def test_search_log_records_time_filters_and_each_reranker_outcome(store: Store) -> None:
    first_query = "editor preference"
    second_query = "which editor"
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    first = _record("first", "The editor preference is Vim.")
    second = _record("second", "The editor preference is Emacs.")
    third = _record("third", "The editor preference is Nano.")
    for score, record in zip((0.90, 0.80, 0.70), (first, second, third), strict=True):
        embedder.set_similarity(first_query, record.content, score)
        embedder.set_similarity(second_query, record.content, score)
        _store_record(store, embedder, record)
    config = replace(
        _CONFIG,
        retrieval=replace(_CONFIG.retrieval, dedup_cosine=0.999),
        reranker=RerankerConfig(enabled=True, candidates=2, floor=0.50),
    )
    reranker = _ScoredReranker(
        {
            (first_query, first.content): 0.40,
            (second_query, first.content): 0.90,
            (first_query, second.content): 0.30,
            (second_query, second.content): 0.20,
        }
    )
    request = replace(
        _request(first_query),
        queries=[first_query, second_query],
        since=_NOW - timedelta(days=1),
        until=_NOW + timedelta(days=1),
    )

    response = _retriever(store, embedder, config=config, reranker=reranker).search(_PRINCIPAL, request)
    log = store.read_search_log(response.search_id)

    assert log is not None
    assert log["request"]["since"] == (_NOW - timedelta(days=1)).isoformat()  # type: ignore[index]
    assert log["request"]["until"] == (_NOW + timedelta(days=1)).isoformat()  # type: ignore[index]
    winning_queries = {entry["record_id"]: entry["winning_query"] for entry in log["reranked"]}  # type: ignore[index]
    assert winning_queries == {first.id: second_query, second.id: first_query}
    assert {entry["reason"] for entry in log["reranked_out"]} == {  # type: ignore[index]
        "reranker candidate limit",
        "reranker floor",
    }
