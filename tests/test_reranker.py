"""Phase 11: the cross-encoder reranker behind the gate, with a fake model and one integration pair."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from retold.config import EmbeddingConfig, RerankerConfig, RetoldConfig, RetrievalConfig
from retold.index import BgeReranker, NoReranker, RerankError, rerank_with_timeout, reranker_from_config
from retold.index.embedder import FakeEmbedder
from retold.index.reranker import rerank
from retold.index.vector import VectorIndex
from retold.models import Candidate, GeneratorHit, Principal, Record, Scope, SearchRequest
from retold.retrieve import Retriever
from retold.store import Store

_NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
_AGENT = "research-agent"
_USER = "aditya"
_USER_SCOPE = Scope(kind="user", id=_USER)
_OTHER_SCOPE = Scope(kind="user", id="someone-else")
_PRINCIPAL = Principal(_AGENT, _USER, "session-1", None)
_EMBEDDING = EmbeddingConfig(model="fake-embedder", version="1", dims=8)


class FakeCrossEncoder:
    def __init__(self, scores: dict[tuple[str, str], float]) -> None:
        self._scores = scores
        self.batches: list[list[list[str]]] = []

    def predict(self, pairs: list[list[str]], batch_size: int, show_progress_bar: bool) -> list[float]:
        del batch_size, show_progress_bar
        self.batches.append(pairs)
        return [self._scores[(query, document)] for query, document in pairs]


def _record(record_id: str, content: str, scope: Scope = _USER_SCOPE) -> Record:
    return Record(
        id=record_id,
        type="semantic",
        version=1,
        content=content,
        subject=f"person:aditya/{record_id}",
        scope=scope,
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


def _candidate(record_id: str, rank: int) -> Candidate:
    return Candidate(
        record_id, GeneratorHit(rank, 0.9), None, None, None, None, 1.0 / rank, rank, None, 0.0, None, None, None
    )


def test_bge_reranker_loads_lazily_and_scores_one_batch() -> None:
    model = FakeCrossEncoder({("q", "a"): 0.9, ("q", "b"): 0.2})
    reranker = BgeReranker(RerankerConfig(enabled=True, floor=0.5, batch_size=16), model_factory=lambda: model)

    assert reranker.is_loaded is False
    assert reranker.score_pairs([("q", "a"), ("q", "b")]) == [0.9, 0.2]
    assert reranker.is_loaded is True
    assert reranker.score("q", "b") == 0.2
    assert reranker.score_pairs([]) == []
    assert len(model.batches) == 2


def test_rerank_scores_every_pair_in_one_call_and_keeps_the_best_query() -> None:
    model = FakeCrossEncoder(
        {("q1", "A"): 0.1, ("q2", "A"): 0.7, ("q1", "B"): 0.4, ("q2", "B"): 0.3},
    )
    reranker = BgeReranker(RerankerConfig(enabled=True, floor=0.0), model_factory=lambda: model)
    records = {"a": _record("a", "A"), "b": _record("b", "B")}

    ordered, logged = rerank([_candidate("a", 1), _candidate("b", 2)], records, ["q1", "q2"], reranker)

    assert len(model.batches) == 1 and len(model.batches[0]) == 4
    assert [candidate.record_id for candidate in ordered] == ["a", "b"]
    assert {entry["record_id"]: entry["winning_query"] for entry in logged} == {"a": "q2", "b": "q1"}
    assert {entry["record_id"]: entry["score"] for entry in logged} == {"a": 0.7, "b": 0.4}


def test_rerank_rejects_a_reranker_that_returns_the_wrong_number_of_scores() -> None:
    class Short:
        @property
        def is_loaded(self) -> bool:
            return True

        def score(self, query: str, document: str) -> float:
            return 0.0

        def score_pairs(self, pairs: Any) -> list[float]:
            return []

    with pytest.raises(ValueError):
        rerank([_candidate("a", 1)], {"a": _record("a", "A")}, ["q"], Short())


def test_reranker_from_config_is_a_no_op_unless_enabled() -> None:
    assert isinstance(reranker_from_config(RetoldConfig()), NoReranker)
    config = RetoldConfig(reranker=RerankerConfig(enabled=True, floor=0.5))
    assert isinstance(reranker_from_config(config), BgeReranker)


def test_reranker_only_sees_records_that_passed_scope_and_gate(tmp_path: Path) -> None:
    store = Store(tmp_path / "memory.sqlite")
    store.set_grant(_AGENT, _USER_SCOPE, can_read=True, can_write=True)
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    query = "preferred editor"
    mine = _record("mine", "Aditya uses Vim as the preferred editor.")
    weak = _record("weak", "Aditya keeps editor settings in dotfiles.")
    foreign = _record("foreign", "Someone else uses Emacs as the preferred editor.", _OTHER_SCOPE)
    embedder.set_similarity(query, mine.content, 0.90)
    embedder.set_similarity(query, weak.content, 0.20)
    embedder.set_similarity(query, foreign.content, 0.95)
    for record in (mine, weak, foreign):
        store.insert_record(record)
        store.put_embedding(record.id, embedder.name, embedder.version, embedder.embed_documents([record.content])[0])
        store.upsert_fts(record.id, record.content, record.subject, "")
    model = FakeCrossEncoder({(query, mine.content): 0.8, (query, weak.content): 0.1})
    config = RetoldConfig(
        embedding=_EMBEDDING,
        retrieval=RetrievalConfig(per_generator_k=10, default_k=8),
        reranker=RerankerConfig(enabled=True, floor=0.5),
    )
    reranker = BgeReranker(config.reranker, model_factory=lambda: model)
    retriever = Retriever(
        store, VectorIndex(_EMBEDDING), embedder, config, reranker=reranker, current_time=lambda: _NOW
    )

    response = retriever.search(_PRINCIPAL, SearchRequest([query], None, None, None, None, None, 8, False))

    scored = {document for batch in model.batches for _, document in batch}
    assert foreign.content not in scored
    assert [result.record.id for result in response.results] == ["mine"]
    log = store.read_search_log(response.search_id)
    assert log is not None
    assert all(entry["record_id"] != "foreign" for entry in log["reranked"])  # type: ignore[union-attr]
    store.close()


def test_enabled_reranker_without_a_floor_is_rejected_at_load(tmp_path: Path) -> None:
    from retold.config import ConfigError, load_config

    path = tmp_path / "config.yaml"
    path.write_text("reranker:\n  enabled: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="reranker.floor"):
        load_config(path)


@pytest.mark.integration
def test_real_reranker_orders_an_obvious_pair() -> None:
    if os.environ.get("RETOLD_INTEGRATION") != "1":
        pytest.skip("set RETOLD_INTEGRATION=1 to run local-model integration tests")
    reranker = BgeReranker(replace(RerankerConfig(), enabled=True, floor=0.5))
    scores = reranker.score_pairs(
        [
            ("preferred editor", "Aditya uses Vim as the preferred editor."),
            ("preferred editor", "Aditya keeps commit messages conventional."),
        ]
    )
    assert scores[0] > scores[1]
    assert 0.0 <= scores[1] <= scores[0] <= 1.0


# -- Modes, timeouts, and fallback -------------------------------------------------------------------


class SlowOrBrokenReranker:
    """Sleeps or raises on demand, and records what it was asked to score."""

    def __init__(self, *, delay_s: float = 0.0, error: Exception | None = None, score: float = 0.9) -> None:
        self.delay_s = delay_s
        self.error = error
        self.score_value = score
        self.documents: list[str] = []
        self.finished = threading.Event()

    @property
    def is_loaded(self) -> bool:
        return True

    def score(self, query: str, document: str) -> float:
        return self.score_pairs([(query, document)])[0]

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        try:
            if self.delay_s:
                time.sleep(self.delay_s)
            if self.error is not None:
                raise self.error
            self.documents.extend(document for _, document in pairs)
            return [self.score_value for _ in pairs]
        finally:
            self.finished.set()


def _seed(store: Store, embedder: FakeEmbedder, query: str, records: list[tuple[Record, float]]) -> None:
    for record, similarity in records:
        embedder.set_similarity(query, record.content, similarity)
        store.insert_record(record)
        store.put_embedding(record.id, embedder.name, embedder.version, embedder.embed_documents([record.content])[0])
        store.upsert_fts(record.id, record.content, record.subject, "")


def _search(retriever: Retriever, query: str, trigger: str = "tool") -> Any:
    return retriever.search(_PRINCIPAL, SearchRequest([query], None, None, None, None, None, 8, False, trigger=trigger))


def test_timeout_falls_back_to_the_rrf_order_and_says_so_in_the_log(tmp_path: Path) -> None:
    store = Store(tmp_path / "memory.sqlite")
    store.set_grant(_AGENT, _USER_SCOPE, can_read=True, can_write=True)
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    query = "preferred editor"
    strong = _record("strong", "Aditya uses Vim as the preferred editor.")
    weaker = _record("weaker", "Aditya's editor of choice is Vim, configured from dotfiles.")
    _seed(store, embedder, query, [(strong, 0.90), (weaker, 0.80)])
    slow = SlowOrBrokenReranker(delay_s=0.5, score=0.01)
    config = RetoldConfig(
        embedding=_EMBEDDING,
        retrieval=RetrievalConfig(per_generator_k=10, default_k=8),
        reranker=RerankerConfig(enabled=True, floor=0.5, timeout_ms=50),
    )
    retriever = Retriever(store, VectorIndex(_EMBEDDING), embedder, config, reranker=slow, current_time=lambda: _NOW)

    response = _search(retriever, query)

    # The RRF order is served in full: the reranker would have dropped both under the floor, but it did not apply.
    assert [result.record.id for result in response.results] == ["strong", "weaker"]
    assert all(result.explanation.rerank is None for result in response.results)
    log = store.read_search_log(response.search_id)
    assert log is not None
    assert log["rerank_status"] == "timeout"
    assert log["rerank_error"] == "reranker timed out after 50 ms"
    assert log["reranked"] is None and log["reranked_out"] == []
    assert log["config_flags"]["ranking"] == "rrf_cross_encoder"
    # The abandoned pass finishes later and must not have touched the candidates the search used.
    assert slow.finished.wait(2.0)
    assert all(result.explanation.rerank is None for result in response.results)
    store.close()


def test_scoring_error_falls_back_or_raises_as_configured(tmp_path: Path) -> None:
    store = Store(tmp_path / "memory.sqlite")
    store.set_grant(_AGENT, _USER_SCOPE, can_read=True, can_write=True)
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    query = "preferred editor"
    _seed(store, embedder, query, [(_record("mine", "Aditya uses Vim as the preferred editor."), 0.90)])
    broken = SlowOrBrokenReranker(error=RuntimeError("model unavailable"))
    base = RetoldConfig(embedding=_EMBEDDING, retrieval=RetrievalConfig(per_generator_k=10, default_k=8))
    vector_index = VectorIndex(_EMBEDDING)

    fallback = replace(base, reranker=RerankerConfig(enabled=True, floor=0.5))
    response = _search(Retriever(store, vector_index, embedder, fallback, reranker=broken), query)
    assert [result.record.id for result in response.results] == ["mine"]
    log = store.read_search_log(response.search_id)
    assert log is not None
    assert log["rerank_status"] == "failed"
    assert log["rerank_error"] == "RuntimeError: model unavailable"

    strict = replace(base, reranker=RerankerConfig(enabled=True, floor=0.5, on_failure="fail"))
    with pytest.raises(RerankError, match="model unavailable"):
        _search(Retriever(store, vector_index, embedder, strict, reranker=broken), query)
    store.close()


def test_cross_encoder_only_skips_relevance_floors_but_not_scope_or_policy(tmp_path: Path) -> None:
    store = Store(tmp_path / "memory.sqlite")
    store.set_grant(_AGENT, _USER_SCOPE, can_read=True, can_write=True)
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    query = "preferred editor"
    mine = _record("mine", "Aditya uses Vim as the preferred editor.")
    below_floor = _record("below", "Aditya keeps editor settings in dotfiles.")
    foreign = _record("foreign", "Someone else uses Emacs as the preferred editor.", _OTHER_SCOPE)
    summary = replace(_record("summary", "Session summary: the editor discussion."), source_kind="session_summary")
    _seed(store, embedder, query, [(mine, 0.90), (below_floor, 0.20), (foreign, 0.95), (summary, 0.85)])
    model = FakeCrossEncoder(
        {
            (query, mine.content): 0.8,
            (query, below_floor.content): 0.7,
            (query, foreign.content): 0.99,
            (query, summary.content): 0.9,
        }
    )
    reranker = BgeReranker(RerankerConfig(enabled=True, floor=0.5), model_factory=lambda: model)
    retrieval = RetrievalConfig(per_generator_k=10, default_k=8)
    rrf_then_ce = RetoldConfig(
        embedding=_EMBEDDING, retrieval=retrieval, reranker=RerankerConfig(enabled=True, floor=0.5)
    )
    ce_only = replace(rrf_then_ce, reranker=replace(rrf_then_ce.reranker, mode="cross_encoder_only"))
    vector_index = VectorIndex(_EMBEDDING)

    gated = _search(Retriever(store, vector_index, embedder, rrf_then_ce, reranker=reranker), query, "auto")
    assert [result.record.id for result in gated.results] == ["mine"]

    open_ranked = _search(Retriever(store, vector_index, embedder, ce_only, reranker=reranker), query, "auto")
    # The dense floor no longer decides: the cross-encoder admitted the record cosine rejected.
    assert [result.record.id for result in open_ranked.results] == ["mine", "below"]
    assert open_ranked.results[1].explanation.gate == "relevance floors skipped for cross-encoder-only ranking"
    scored = {document for batch in model.batches for _, document in batch}
    # Scope and the auto-retrieval source-kind policy still run first; the cross-encoder never saw either record.
    assert foreign.content not in scored
    assert summary.content not in scored
    log = store.read_search_log(open_ranked.search_id)
    assert log is not None
    assert log["config_flags"]["ranking"] == "cross_encoder_only"
    assert [entry["record_id"] for entry in log["gated_out"]] == ["summary"]
    assert log["rerank_status"] == "applied"
    store.close()


def test_rerank_with_timeout_returns_the_input_unchanged_on_timeout() -> None:
    slow = SlowOrBrokenReranker(delay_s=0.3, score=0.9)
    candidates = [_candidate("a", 1), _candidate("b", 2)]
    records = {"a": _record("a", "A"), "b": _record("b", "B")}

    outcome = rerank_with_timeout(candidates, records, ["q"], slow, timeout_ms=20, on_failure="fallback")

    assert outcome.status == "timeout" and outcome.logged is None
    assert [candidate.record_id for candidate in outcome.candidates] == ["a", "b"]
    assert slow.finished.wait(2.0)
    assert all(candidate.rerank_score is None for candidate in candidates)
    with pytest.raises(RerankError):
        rerank_with_timeout(candidates, records, ["q"], slow, timeout_ms=20, on_failure="fail")


def test_reranker_config_validation_and_ranking_name(tmp_path: Path) -> None:
    from retold.config import ConfigError, load_config

    assert RetoldConfig().reranker.ranking == "rrf_only"
    assert RerankerConfig(enabled=True, floor=0.1, mode="cross_encoder_only").ranking == "cross_encoder_only"
    for body in (
        "reranker:\n  enabled: true\n  floor: 0.1\n  mode: bm25\n",
        "reranker:\n  timeout_ms: 0\n",
        "reranker:\n  on_failure: retry\n",
    ):
        path = tmp_path / "config.yaml"
        path.write_text(body, encoding="utf-8")
        with pytest.raises(ConfigError, match="reranker\\."):
            load_config(path)


def test_new_reranker_fields_do_not_change_a_disabled_bundle_hash() -> None:
    from benchmarks.phase0_real_retrieval import recall_oriented_config
    from benchmarks.shadow_adapter import policy_bundle

    supported = policy_bundle("gpt-4o", "gpt-5.4", recall_oriented_config(), None)["retrieval_config_sha256"]
    assert supported == "e8c8c3309ab121de", "the supported bundle's retrieval hash must not move"
    enabled = recall_oriented_config(rerank_floor=0.01)
    quick = replace(enabled, reranker=replace(enabled.reranker, timeout_ms=1))
    assert (
        policy_bundle("gpt-4o", "gpt-5.4", enabled, None)["retrieval_config_sha256"]
        != policy_bundle("gpt-4o", "gpt-5.4", quick, None)["retrieval_config_sha256"]
    ), "an enabled reranker's timeout changes behaviour and so the bundle"


def test_cross_encoder_only_falls_back_to_the_gated_rrf_order_when_the_pass_times_out(tmp_path: Path) -> None:
    store = Store(tmp_path / "memory.sqlite")
    store.set_grant(_AGENT, _USER_SCOPE, can_read=True, can_write=True)
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    query = "preferred editor"
    mine = _record("mine", "Aditya uses Vim as the preferred editor.")
    below_floor = _record("below", "Aditya keeps editor settings in dotfiles.")
    _seed(store, embedder, query, [(mine, 0.90), (below_floor, 0.20)])
    slow = SlowOrBrokenReranker(delay_s=0.5, score=0.9)
    config = RetoldConfig(
        embedding=_EMBEDDING,
        retrieval=RetrievalConfig(per_generator_k=10, default_k=8),
        reranker=RerankerConfig(enabled=True, floor=0.5, mode="cross_encoder_only", timeout_ms=50),
    )
    retriever = Retriever(store, VectorIndex(_EMBEDDING), embedder, config, reranker=slow, current_time=lambda: _NOW)

    response = _search(retriever, query, "auto")

    # Without the cross-encoder there is no relevance decision left but the floors, so they apply again.
    assert [result.record.id for result in response.results] == ["mine"]
    log = store.read_search_log(response.search_id)
    assert log is not None
    assert log["rerank_status"] == "timeout"
    assert [entry["record_id"] for entry in log["gated_out"]] == ["below"]
    assert slow.finished.wait(2.0)
    store.close()
