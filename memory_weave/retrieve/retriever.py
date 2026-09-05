"""The complete retrieval pipeline from a principal and request to an auditable response."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime

from memory_weave.config import MemoryWeaveConfig
from memory_weave.index.embedder import Embedder
from memory_weave.index.reranker import NoReranker, Reranker, rerank
from memory_weave.index.vector import VectorIndex
from memory_weave.models import (
    Candidate,
    Explanation,
    GeneratorHit,
    LexicalMatch,
    Principal,
    Record,
    Scope,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from memory_weave.policy.grants import readable_scopes
from memory_weave.store import Store
from memory_weave.util import Timer, now, uuid7

from .budget import fill_budget
from .dedup import collapse_duplicates
from .explain import build_results
from .freshness import apply_freshness
from .fusion import fuse
from .gate import FloorGate, Gate, GateDecision
from .generators import dense_candidates, entity_alias_matches, entity_candidates, lexical_candidates
from .rewrite import NoRewriter, QueryRewriter, rewrite_stage


class Retriever:
    """Retrieve scope-filtered memories and persist a complete decision trail for every request."""

    def __init__(
        self,
        store: Store,
        vector_index: VectorIndex,
        embedder: Embedder,
        config: MemoryWeaveConfig,
        *,
        gate: Gate | None = None,
        rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None,
        current_time: Callable[[], datetime] = now,
    ) -> None:
        self._store = store
        self._vector_index = vector_index
        self._embedder = embedder
        self._config = config
        self._gate = gate or FloorGate(config.retrieval.gate)
        self._rewriter = rewriter or NoRewriter()
        self._reranker = reranker or NoReranker()
        self._current_time = current_time

    def search(self, principal: Principal, request: SearchRequest) -> SearchResponse:
        """Run each retrieval stage in order and record all intermediate results before returning."""

        _validate_request(request)
        timer = Timer(warm=self._embedder.is_loaded and self._vector_index.is_loaded)
        search_id = uuid7()
        rewritten_queries, rewrite_status = rewrite_stage(request, self._config.retrieval.rewrite, self._rewriter)
        timer.mark("rewrite")
        scopes = readable_scopes(self._store, principal.agent_id, principal.user_id)
        timer.mark("scopes")
        current_time = self._current_time()
        eligible = self._store.eligible_ids(
            scopes, request.types, request.since, request.until, request.include_history, current_time
        )
        timer.mark("filter")
        index_refresh = self._vector_index.refresh(self._store, self._config.embedding.incremental_reload_max)
        timer.mark("index_refresh")
        query_vectors = self._embedder.embed_queries(rewritten_queries) if eligible else None
        timer.mark("embed")
        dense = (
            dense_candidates(
                rewritten_queries,
                self._embedder,
                self._vector_index,
                eligible,
                self._config.retrieval.per_generator_k,
                query_vectors,
            )
            if eligible
            else []
        )
        timer.mark("dense")
        alias_matches = (
            entity_alias_matches(
                request.entities, rewritten_queries, self._store, scopes, self._config.retrieval.max_alias_tokens
            )
            if eligible
            else {}
        )
        lexical = (
            lexical_candidates(
                rewritten_queries,
                self._store,
                eligible,
                self._config.retrieval.per_generator_k,
                entity_aliases=alias_matches,
            )
            if eligible
            else []
        )
        timer.mark("lexical")
        entity = (
            entity_candidates(
                request.entities,
                rewritten_queries,
                self._store,
                scopes,
                eligible,
                self._config.retrieval.per_generator_k,
                self._config.retrieval.max_alias_tokens,
                alias_matches,
            )
            if eligible
            else []
        )
        timer.mark("entity")
        candidates = fuse(dense, lexical, entity, self._config.retrieval.rrf_k)
        timer.mark("fuse")
        candidate_ids = [candidate.record_id for candidate in candidates if candidate.record_id in eligible]
        records = {record.id: record for record in self._store.get_records(candidate_ids)}
        candidates = [
            candidate for candidate in candidates if candidate.record_id in eligible and candidate.record_id in records
        ]
        candidates = apply_freshness(candidates, records, request, current_time, self._config.retrieval.freshness)
        timer.mark("freshness")
        gate_decision = self._gate.apply(candidates, records, request)
        timer.mark("gate")
        deduped, deduped_out = collapse_duplicates(
            gate_decision.kept, self._vector_index, self._config.retrieval.dedup_cosine
        )
        timer.mark("dedup")
        reranked: list[dict[str, object]] | None = None
        reranked_out: list[dict[str, object]] = []
        if self._config.reranker.enabled:
            shortlist_limit = self._config.reranker.candidates
            reranked_out = [
                {
                    "record_id": candidate.record_id,
                    "reason": "reranker candidate limit",
                    "limit": shortlist_limit,
                }
                for candidate in deduped[shortlist_limit:]
            ]
            reranked_candidates, reranked = rerank(
                deduped[:shortlist_limit], records, rewritten_queries, self._reranker
            )
            assert self._config.reranker.floor is not None
            reranked_out.extend(
                {
                    "record_id": candidate.record_id,
                    "reason": "reranker floor",
                    "floor": self._config.reranker.floor,
                    "score": candidate.rerank_score,
                    "winning_query": _winning_query(reranked, candidate.record_id),
                }
                for candidate in reranked_candidates
                if candidate.rerank_score is not None and candidate.rerank_score < self._config.reranker.floor
            )
            deduped = [
                candidate
                for candidate in reranked_candidates
                if candidate.rerank_score is not None and candidate.rerank_score >= self._config.reranker.floor
            ]
        timer.mark("rerank")
        chosen, companions = self._include_conflict_authority(deduped, records, eligible)
        budgeted, budget_out = fill_budget(
            chosen, records, request.k, self._config.retrieval.token_budget, companions=companions
        )
        timer.mark("budget")
        conflicts = {
            candidate.record_id: [
                conflict_id for conflict_id in self._store.conflicts_for(candidate.record_id) if conflict_id in eligible
            ]
            for candidate in budgeted
        }
        results = build_results(
            budgeted,
            records,
            request.queries,
            rewritten_queries if rewrite_status == "applied" else None,
            rewrite_status,
            conflicts,
        )
        empty_reason = _empty_reason(
            results,
            gate_decision,
            deduped,
            self._config.reranker.enabled,
            self._config.reranker.floor,
        )
        timer.mark("explain")
        self._store.write_search_log(
            self._search_log_row(
                search_id,
                principal,
                request,
                rewritten_queries,
                rewrite_status,
                scopes,
                dense,
                lexical,
                entity,
                candidates,
                gate_decision.gated_out,
                deduped_out,
                reranked,
                reranked_out,
                budget_out,
                results,
                index_refresh,
                timer,
            )
        )
        timer.mark("log")
        return SearchResponse(
            search_id,
            request.queries,
            rewritten_queries if rewrite_status == "applied" else None,
            rewrite_status,  # type: ignore[arg-type]
            results,
            empty_reason,
            timer.as_dict(),
        )

    def _include_conflict_authority(
        self, candidates: Sequence[Candidate], records: dict[str, Record], eligible: set[str]
    ) -> tuple[list[Candidate], dict[str, Candidate]]:
        """Pair a provisional record with its eligible authority counterpart without replacing real evidence."""

        by_id = {candidate.record_id: candidate for candidate in candidates}
        positions = {candidate.record_id: position for position, candidate in enumerate(candidates)}
        companions: dict[str, Candidate] = {}
        claimed_authorities: set[str] = set()
        for position, candidate in enumerate(candidates):
            record = records[candidate.record_id]
            if record.status != "provisional":
                continue
            peers = [peer_id for peer_id in self._store.conflicts_for(record.id) if peer_id in eligible]
            peer_records = [
                peer
                for peer in self._store.get_records(peers)
                if peer.status == "confirmed" or self._authority_key(peer) > self._authority_key(record)
            ]
            if not peer_records:
                continue
            authority = max(peer_records, key=self._authority_key)
            if authority.id in claimed_authorities:
                continue
            authority_position = positions.get(authority.id)
            if authority_position is not None and authority_position < position:
                # The counterpart already outranks the provisional record on its own evidence.
                # Moving it here would demote it and could push it past the result limit.
                continue
            records[authority.id] = authority
            companions[candidate.record_id] = by_id.get(authority.id, _authority_candidate(authority.id))
            claimed_authorities.add(authority.id)
        companion_ids = {companion.record_id for companion in companions.values()}
        chosen = [candidate for candidate in candidates if candidate.record_id not in companion_ids]
        return chosen, companions

    def _authority_key(self, record: Record) -> tuple[int, datetime, datetime, str]:
        return (
            getattr(self._config.policy.source_rank, record.source_kind),
            record.event_at,
            record.created_at,
            record.id,
        )

    def _search_log_row(
        self,
        search_id: str,
        principal: Principal,
        request: SearchRequest,
        rewritten_queries: list[str],
        rewrite_status: str,
        scopes: Sequence[Scope],
        dense: Sequence[tuple[str, GeneratorHit]],
        lexical: Sequence[tuple[str, GeneratorHit, LexicalMatch]],
        entity: Sequence[tuple[str, GeneratorHit, str]],
        candidates: Sequence[Candidate],
        gated_out: Sequence[Candidate],
        deduped_out: Sequence[dict[str, object]],
        reranked: list[dict[str, object]] | None,
        reranked_out: Sequence[dict[str, object]],
        budget_out: Sequence[dict[str, object]],
        results: Sequence[SearchResult],
        index_refresh: str,
        timer: Timer,
    ) -> dict[str, object]:
        """Build the one store row that lets an offline review reconstruct this search."""

        return {
            "id": search_id,
            "at": self._current_time(),
            "agent_id": principal.agent_id,
            "user_id": principal.user_id,
            "session_id": principal.session_id,
            "trigger": request.trigger,
            "request": {
                "entities": request.entities,
                "include_history": request.include_history,
                "k": request.k,
                "queries": request.queries,
                "since": request.since,
                "types": request.types,
                "until": request.until,
            },
            "context": request.context,
            "rewrite_status": rewrite_status,
            "rewritten_queries": rewritten_queries if rewrite_status == "applied" else None,
            "readable_scopes": [{"kind": scope.kind, "id": scope.id} for scope in scopes],
            "dense": [_dense_payload(record_id, hit) for record_id, hit in dense],
            "lexical": [_lexical_payload(record_id, hit, match) for record_id, hit, match in lexical],
            "entity": [_entity_payload(record_id, hit, entity_id) for record_id, hit, entity_id in entity],
            "fused": [_candidate_payload(candidate) for candidate in candidates],
            "freshness": [_freshness_payload(candidate) for candidate in candidates],
            "gated_out": [_candidate_payload(candidate) for candidate in gated_out],
            "deduped_out": list(deduped_out),
            "reranked": reranked,
            "reranked_out": list(reranked_out),
            "budget_out": list(budget_out),
            "returned": [result.record.id for result in results],
            "explanations": [_explanation_payload(result.explanation) for result in results],
            "config_flags": {**self._config.flags(), "index_refresh": index_refresh},
            "warm": timer.warm,
            "timings_ms": timer.as_dict(),
        }


def _validate_request(request: SearchRequest) -> None:
    if not 1 <= len(request.queries) <= 3 or any(not query.strip() for query in request.queries):
        raise ValueError("Search requests require one to three non-empty queries.")
    if request.k <= 0:
        raise ValueError("Search request k must be positive.")


def _authority_candidate(record_id: str) -> Candidate:
    return Candidate(
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
        gate_reason="included as authority counterpart for a provisional conflict",
        rerank_score=None,
        rank_after_rerank=None,
    )


def _winning_query(reranked: Sequence[dict[str, object]], record_id: str) -> object | None:
    """Return the rerank query that won for one shortlisted record, if it was logged."""

    return next(
        (entry.get("winning_query") for entry in reranked if entry["record_id"] == record_id),
        None,
    )


def _dense_payload(record_id: str, hit: GeneratorHit) -> dict[str, object]:
    return {"record_id": record_id, "rank": hit.rank, "score": hit.score}


def _lexical_payload(record_id: str, hit: GeneratorHit, match: LexicalMatch) -> dict[str, object]:
    return {
        "record_id": record_id,
        "rank": hit.rank,
        "score": hit.score,
        "terms": [
            {"value": term.value, "is_identifier": term.is_identifier, "is_entity_alias": term.is_entity_alias}
            for term in match.terms
        ],
        "total_terms": match.total_terms,
    }


def _entity_payload(record_id: str, hit: GeneratorHit, entity_id: str) -> dict[str, object]:
    return {"entity_id": entity_id, "record_id": record_id, "rank": hit.rank, "score": hit.score}


def _candidate_payload(candidate: Candidate) -> dict[str, object]:
    return {
        "record_id": candidate.record_id,
        "dense": _hit_payload(candidate.dense),
        "lexical": _hit_payload(candidate.lexical),
        "entity": _hit_payload(candidate.entity),
        "entity_id": candidate.entity_id,
        "rrf_score": candidate.rrf_score,
        "fused_rank": candidate.fused_rank,
        "score": candidate.score,
        "gate_reason": candidate.gate_reason,
    }


def _freshness_payload(candidate: Candidate) -> dict[str, object]:
    return {
        "record_id": candidate.record_id,
        "multiplier": candidate.freshness_multiplier,
        "score": candidate.score,
    }


def _hit_payload(hit: GeneratorHit | None) -> dict[str, object] | None:
    if hit is None:
        return None
    return {"rank": hit.rank, "score": hit.score}


def _explanation_payload(explanation: Explanation) -> dict[str, object]:
    return {
        "raw_queries": explanation.raw_queries,
        "rewritten_queries": explanation.rewritten_queries,
        "rewrite_status": explanation.rewrite_status,
        "summary": explanation.summary,
        "matched_by": explanation.matched_by,
        "dense": _hit_payload(explanation.dense),
        "lexical": _hit_payload(explanation.lexical),
        "lexical_terms": _lexical_match_payload(explanation.lexical_terms),
        "entity": _hit_payload(explanation.entity),
        "fused_rank": explanation.fused_rank,
        "freshness_multiplier": explanation.freshness_multiplier,
        "rerank": _rerank_payload(explanation.rerank),
        "gate": explanation.gate,
        "dedup": explanation.dedup,
        "budget": explanation.budget,
        "source_kind": explanation.source_kind,
        "status": explanation.status,
        "created_at": explanation.created_at,
        "event_at": explanation.event_at,
        "entity_ids": explanation.entity_ids,
        "conflicts_with": explanation.conflicts_with,
    }


def _lexical_match_payload(match: LexicalMatch | None) -> dict[str, object] | None:
    if match is None:
        return None
    return {
        "terms": [
            {"value": term.value, "is_identifier": term.is_identifier, "is_entity_alias": term.is_entity_alias}
            for term in match.terms
        ],
        "total_terms": match.total_terms,
    }


def _rerank_payload(rerank: tuple[int, int, float] | None) -> list[object] | None:
    if rerank is None:
        return None
    return list(rerank)


def _empty_reason(
    results: Sequence[SearchResult],
    gate_decision: GateDecision,
    candidates_after_rerank: Sequence[Candidate],
    reranker_enabled: bool,
    reranker_floor: float | None,
) -> str | None:
    if results:
        return None
    if not gate_decision.kept:
        return gate_decision.empty_reason
    if reranker_enabled and not candidates_after_rerank:
        assert reranker_floor is not None
        return f"all relevance-gated candidates missed reranker floor {reranker_floor:.2f}"
    if not candidates_after_rerank:
        return "no relevance-gated candidates survived duplicate collapse"
    return "all relevance-gated candidates exceeded the result or token budget"
