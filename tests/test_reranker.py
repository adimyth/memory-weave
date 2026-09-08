"""Phase 11: the cross-encoder reranker behind the gate, with a fake model and one integration pair."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from memory_weave.config import EmbeddingConfig, MemoryWeaveConfig, RerankerConfig, RetrievalConfig
from memory_weave.index import BgeReranker, NoReranker, reranker_from_config
from memory_weave.index.embedder import FakeEmbedder
from memory_weave.index.reranker import rerank
from memory_weave.index.vector import VectorIndex
from memory_weave.models import Candidate, GeneratorHit, Principal, Record, Scope, SearchRequest
from memory_weave.retrieve import Retriever
from memory_weave.store import Store

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
    assert isinstance(reranker_from_config(MemoryWeaveConfig()), NoReranker)
    config = MemoryWeaveConfig(reranker=RerankerConfig(enabled=True, floor=0.5))
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
    config = MemoryWeaveConfig(
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
    from memory_weave.config import ConfigError, load_config

    path = tmp_path / "config.yaml"
    path.write_text("reranker:\n  enabled: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="reranker.floor"):
        load_config(path)


@pytest.mark.integration
def test_real_reranker_orders_an_obvious_pair() -> None:
    if os.environ.get("MEMORY_WEAVE_INTEGRATION") != "1":
        pytest.skip("set MEMORY_WEAVE_INTEGRATION=1 to run local-model integration tests")
    reranker = BgeReranker(replace(RerankerConfig(), enabled=True, floor=0.5))
    scores = reranker.score_pairs(
        [
            ("preferred editor", "Aditya uses Vim as the preferred editor."),
            ("preferred editor", "Aditya keeps commit messages conventional."),
        ]
    )
    assert scores[0] > scores[1]
    assert 0.0 <= scores[1] <= scores[0] <= 1.0
