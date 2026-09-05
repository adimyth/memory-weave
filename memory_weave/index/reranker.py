"""Reranker protocol and no-op placeholder used before the real model lands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from memory_weave.models import Candidate, Record


class Reranker(Protocol):
    """Score one query-record pair with a higher-is-better relevance value."""

    @property
    def is_loaded(self) -> bool:
        """Return whether the reranker can score without loading a model."""

    def score(self, query: str, document: str) -> float:
        """Return one relevance score for a query-record pair."""


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


def rerank(
    candidates: Sequence[Candidate], records: Mapping[str, Record], queries: Sequence[str], reranker: Reranker
) -> tuple[list[Candidate], list[dict[str, object]]]:
    """Score every query-record pair, keep the best score, and sort the candidates by it."""

    before = {candidate.record_id: position for position, candidate in enumerate(candidates, start=1)}
    winning_queries: dict[str, str | None] = {}
    for candidate in candidates:
        best_score = float("-inf")
        winning_query: str | None = None
        for query in queries:
            score = reranker.score(query, records[candidate.record_id].content)
            if score > best_score:
                best_score = score
                winning_query = query
        candidate.rerank_score = best_score
        winning_queries[candidate.record_id] = winning_query
    ordered = sorted(candidates, key=lambda candidate: (-_required_score(candidate), candidate.record_id))
    logged: list[dict[str, object]] = []
    for position, candidate in enumerate(ordered, start=1):
        assert candidate.rerank_score is not None
        candidate.rank_after_rerank = position
        logged.append(
            {
                "record_id": candidate.record_id,
                "rank_before": before[candidate.record_id],
                "rank_after": position,
                "score": candidate.rerank_score,
                "winning_query": winning_queries[candidate.record_id],
            }
        )
    return ordered, logged


def _required_score(candidate: Candidate) -> float:
    assert candidate.rerank_score is not None
    return candidate.rerank_score
