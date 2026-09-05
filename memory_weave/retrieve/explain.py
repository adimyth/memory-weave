"""Structured explanations for retrieval results and their concise agent-facing summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from memory_weave.models import Candidate, Explanation, Record, SearchResult

from .budget import token_cost


def build_results(
    candidates: Sequence[Candidate],
    records: Mapping[str, Record],
    raw_queries: list[str],
    rewritten_queries: list[str] | None,
    rewrite_status: str,
    conflicts_by_record: Mapping[str, list[str]],
) -> list[SearchResult]:
    """Build complete explanations after every ranking and budget decision has been made."""

    used_tokens = 0
    results: list[SearchResult] = []
    for position, candidate in enumerate(candidates, start=1):
        record = records[candidate.record_id]
        cost = token_cost(record)
        used_tokens += cost
        conflicts = conflicts_by_record.get(record.id, [])
        explanation = Explanation(
            raw_queries=raw_queries,
            rewritten_queries=rewritten_queries,
            rewrite_status=rewrite_status,  # type: ignore[arg-type]
            matched_by=_matched_by(candidate),
            dense=candidate.dense,
            lexical=candidate.lexical,
            lexical_terms=candidate.lexical_terms,
            entity=candidate.entity,
            fused_rank=candidate.fused_rank,
            freshness_multiplier=candidate.freshness_multiplier,
            rerank=_rerank(candidate),
            gate=candidate.gate_reason or "included as authority counterpart",
            dedup="kept",
            budget=f"fit at position {position}; {used_tokens} tokens used",
            source_kind=record.source_kind,
            status=record.status,
            created_at=record.created_at,
            event_at=record.event_at,
            entity_ids=list(record.entity_ids),
            conflicts_with=conflicts,
            summary=_summary(record, candidate, conflicts),
        )
        results.append(SearchResult(record, candidate.score, explanation))
    return results


def _matched_by(candidate: Candidate) -> list[Literal["dense", "lexical", "entity"]]:
    matched: list[Literal["dense", "lexical", "entity"]] = []
    if candidate.dense is not None:
        matched.append("dense")
    if candidate.lexical is not None:
        matched.append("lexical")
    if candidate.entity is not None:
        matched.append("entity")
    return matched


def _rerank(candidate: Candidate) -> tuple[int, int, float] | None:
    if candidate.rerank_score is None or candidate.rank_after_rerank is None:
        return None
    return candidate.fused_rank, candidate.rank_after_rerank, candidate.rerank_score


def _summary(record: Record, candidate: Candidate, conflicts: Sequence[str]) -> str:
    matches: list[str] = []
    if candidate.dense is not None:
        matches.append(f"dense {candidate.dense.score:.2f} (rank {candidate.dense.rank})")
    if candidate.lexical is not None and candidate.lexical_terms is not None:
        matches.append(
            f"lexical {len(candidate.lexical_terms.terms)}/{candidate.lexical_terms.total_terms} "
            f"(rank {candidate.lexical.rank})"
        )
    if candidate.entity is not None:
        matches.append(f"entity (rank {candidate.entity.rank})")
    suffix = f"; conflicts with {', '.join(conflicts)}" if conflicts else ""
    return (
        f"[{record.id[:10]}] {record.type} · {record.status} · {record.source_kind} · event {record.event_at.date()} · "
        f"scope {record.scope.kind}:{record.scope.id}\n{record.content}\n"
        f"matched: {', '.join(matches) or 'authority counterpart'}; "
        f"fused rank {candidate.fused_rank}; {candidate.gate_reason or 'included as authority counterpart'}{suffix}"
    )
