"""Reranker protocol, the no-op placeholder, and the lazily loaded cross-encoder."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from memory_weave.config import MemoryWeaveConfig, RerankerConfig
from memory_weave.models import Candidate, Record


class Reranker(Protocol):
    """Score query-record pairs with a higher-is-better relevance value."""

    @property
    def is_loaded(self) -> bool:
        """Return whether the reranker can score without loading a model."""

    def score(self, query: str, document: str) -> float:
        """Return one relevance score for a query-record pair."""

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Return one score per pair, in order; the cross-encoder scores them as one batch."""


class NoReranker:
    """A placeholder that preserves original ordering while reranking is disabled."""

    @property
    def is_loaded(self) -> bool:
        """Return true because the no-op implementation has no model to load."""

        return True

    def score(self, query: str, document: str) -> float:
        """Return a neutral score when explicitly used in a test-only no-op path."""

        del query, document
        return 0.0

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [0.0 for _ in pairs]


class BgeReranker:
    """``BAAI/bge-reranker-v2-m3`` through sentence-transformers' ``CrossEncoder``, loaded on first use.

    The model has one output label, so ``predict`` applies a sigmoid and scores lie in (0, 1); the
    configured ``floor`` is compared on that scale.
    """

    def __init__(self, config: RerankerConfig, *, model_factory: Callable[[], Any] | None = None) -> None:
        self._config = config
        self._model_factory = model_factory or _cross_encoder_factory(config.model)
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def score(self, query: str, document: str) -> float:
        return self.score_pairs([(query, document)])[0]

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        if self._model is None:
            self._model = self._model_factory()
        scores = self._model.predict(
            [list(pair) for pair in pairs], batch_size=self._config.batch_size, show_progress_bar=False
        )
        return [float(score) for score in scores]


def reranker_from_config(config: MemoryWeaveConfig) -> Reranker:
    """The reranker the configuration asks for: a no-op unless ``reranker.enabled``."""

    return BgeReranker(config.reranker) if config.reranker.enabled else NoReranker()


def rerank(
    candidates: Sequence[Candidate], records: Mapping[str, Record], queries: Sequence[str], reranker: Reranker
) -> tuple[list[Candidate], list[dict[str, object]]]:
    """Score every query-record pair in one batch, keep each record's best score, and sort by it."""

    before = {candidate.record_id: position for position, candidate in enumerate(candidates, start=1)}
    pairs = [(query, records[candidate.record_id].content) for candidate in candidates for query in queries]
    scores = reranker.score_pairs(pairs)
    if len(scores) != len(pairs):
        raise ValueError("The reranker must return one score per pair.")
    winning_queries: dict[str, str | None] = {}
    position = 0
    for candidate in candidates:
        best_score = float("-inf")
        winning_query: str | None = None
        for query in queries:
            score = scores[position]
            position += 1
            if score > best_score:
                best_score = score
                winning_query = query
        candidate.rerank_score = best_score
        winning_queries[candidate.record_id] = winning_query
    ordered = sorted(candidates, key=lambda candidate: (-_required_score(candidate), candidate.record_id))
    logged: list[dict[str, object]] = []
    for rank, candidate in enumerate(ordered, start=1):
        assert candidate.rerank_score is not None
        candidate.rank_after_rerank = rank
        logged.append(
            {
                "record_id": candidate.record_id,
                "rank_before": before[candidate.record_id],
                "rank_after": rank,
                "score": candidate.rerank_score,
                "winning_query": winning_queries[candidate.record_id],
            }
        )
    return ordered, logged


def _required_score(candidate: Candidate) -> float:
    assert candidate.rerank_score is not None
    return candidate.rerank_score


def _cross_encoder_factory(model_name: str) -> Callable[[], Any]:
    def load() -> Any:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            message = "Reranking requires sentence-transformers. Install it with: uv sync --extra local-models"
            raise RuntimeError(message) from exc
        return CrossEncoder(model_name)

    return load
