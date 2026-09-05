"""Reciprocal-rank fusion for the three independent candidate generators."""

from __future__ import annotations

from collections.abc import Sequence

from memory_weave.models import Candidate, GeneratorHit, LexicalMatch

type DenseHits = Sequence[tuple[str, GeneratorHit]]
type LexicalHits = Sequence[tuple[str, GeneratorHit, LexicalMatch]]
type EntityHits = Sequence[tuple[str, GeneratorHit, str]]


def fuse(dense: DenseHits, lexical: LexicalHits, entity: EntityHits, rrf_k: int) -> list[Candidate]:
    """Fuse ranked generator hits, retaining every generator contribution for later stages."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")
    candidates: dict[str, Candidate] = {}
    for record_id, hit in dense:
        candidate = _candidate(candidates, record_id)
        candidate.dense = hit
        candidate.rrf_score += 1.0 / (rrf_k + hit.rank)
    for record_id, hit, lexical_match in lexical:
        candidate = _candidate(candidates, record_id)
        candidate.lexical = hit
        candidate.lexical_terms = lexical_match
        candidate.rrf_score += 1.0 / (rrf_k + hit.rank)
    for record_id, hit, entity_id in entity:
        candidate = _candidate(candidates, record_id)
        candidate.entity = hit
        candidate.entity_id = entity_id
        candidate.rrf_score += 1.0 / (rrf_k + hit.rank)
    ordered = sorted(candidates.values(), key=lambda candidate: (-candidate.rrf_score, candidate.record_id))
    for rank, candidate in enumerate(ordered, start=1):
        candidate.fused_rank = rank
        candidate.score = candidate.rrf_score
    return ordered


def _candidate(candidates: dict[str, Candidate], record_id: str) -> Candidate:
    candidate = candidates.get(record_id)
    if candidate is None:
        candidate = Candidate(
            record_id=record_id,
            dense=None,
            lexical=None,
            lexical_terms=None,
            entity=None,
            entity_id=None,
            rrf_score=0.0,
            fused_rank=0,
            freshness_multiplier=None,
            score=0.0,
            gate_reason=None,
            rerank_score=None,
            rank_after_rerank=None,
        )
        candidates[record_id] = candidate
    return candidate
