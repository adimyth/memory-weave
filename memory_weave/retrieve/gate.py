"""Relevance floors for tool- and host-issued memory retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from memory_weave.config import AutoGateConfig, GateConfig
from memory_weave.models import Candidate, Record, SearchRequest


@dataclass(frozen=True, slots=True)
class GateDecision:
    """The survivors, rejected candidates, and a user-facing reason for an empty result."""

    kept: list[Candidate]
    gated_out: list[Candidate]
    empty_reason: str | None


class Gate(Protocol):
    """Apply calibrated relevance rules to fused candidates."""

    def apply(
        self, candidates: Sequence[Candidate], records: Mapping[str, Record], request: SearchRequest
    ) -> GateDecision:
        """Return candidates that have sufficient independent relevance evidence."""


class FloorGate:
    """Apply absolute signal floors before a channel-aware relative RRF floor."""

    def __init__(self, tool_config: GateConfig) -> None:
        self._tool_config = tool_config

    def apply(
        self, candidates: Sequence[Candidate], records: Mapping[str, Record], request: SearchRequest
    ) -> GateDecision:
        """Keep exact entity matches or candidates that satisfy an absolute relevance floor."""

        survivors: list[Candidate] = []
        rejected: list[Candidate] = []
        for candidate in candidates:
            record = records[candidate.record_id]
            if request.trigger == "auto" and record.source_kind in self._tool_config.auto.exclude_source_kinds:
                candidate.gate_reason = f"excluded source kind {record.source_kind} for auto retrieval"
                rejected.append(candidate)
                continue
            config = self._tool_config.auto if request.trigger == "auto" else self._tool_config
            reason = _absolute_reason(candidate, record, config)
            if reason is None:
                candidate.gate_reason = _missed_reason(candidate, record, config)
                rejected.append(candidate)
                continue
            candidate.gate_reason = reason
            survivors.append(candidate)

        channel_tops: dict[int, float] = {}
        for candidate in survivors:
            channels = _channel_count(candidate)
            channel_tops[channels] = max(channel_tops.get(channels, 0.0), candidate.rrf_score)
        kept: list[Candidate] = []
        for candidate in survivors:
            if candidate.entity is not None:
                kept.append(candidate)
                continue
            top = channel_tops[_channel_count(candidate)]
            if candidate.rrf_score < config.relative_floor * top:
                candidate.gate_reason = (
                    f"relative RRF {candidate.rrf_score:.4f} < {config.relative_floor:.2f} × {top:.4f} "
                    f"within {_channel_count(candidate)}-channel candidates"
                )
                rejected.append(candidate)
            else:
                kept.append(candidate)
        kept.sort(key=lambda candidate: (-candidate.score, candidate.fused_rank, candidate.record_id))
        rejected.sort(key=lambda candidate: (candidate.fused_rank, candidate.record_id))
        return GateDecision(kept, rejected, _empty_reason(candidates, rejected))


def _absolute_reason(candidate: Candidate, record: Record, config: GateConfig | AutoGateConfig) -> str | None:
    if candidate.entity is not None:
        return "passed exact entity match"
    if candidate.dense is not None:
        floor = getattr(config.dense_floor, record.type)
        if candidate.dense.score >= floor:
            return f"passed dense {candidate.dense.score:.2f} ≥ {floor:.2f} ({record.type})"
    if candidate.lexical is not None and candidate.lexical_terms is not None:
        matched = candidate.lexical_terms.terms
        lexical_pass = candidate.lexical_terms.fraction >= config.lexical_min_term_fraction
        precise_term = any(term.is_identifier or term.is_entity_alias for term in matched)
        enough_terms = len(matched) >= config.lexical_min_matched_terms or precise_term
        if lexical_pass and enough_terms:
            return f"passed lexical {len(matched)}/{candidate.lexical_terms.total_terms} terms"
    return None


def _missed_reason(candidate: Candidate, record: Record, config: GateConfig | AutoGateConfig) -> str:
    misses: list[str] = []
    if candidate.dense is not None:
        floor = getattr(config.dense_floor, record.type)
        misses.append(f"dense {candidate.dense.score:.2f} < {floor:.2f} ({record.type})")
    if candidate.lexical is not None and candidate.lexical_terms is not None:
        terms = candidate.lexical_terms
        misses.append(
            f"lexical {len(terms.terms)}/{terms.total_terms} terms below {config.lexical_min_term_fraction:.2f} "
            f"or {config.lexical_min_matched_terms} matches"
        )
    return "; ".join(misses) if misses else "no generator evidence"


def _channel_count(candidate: Candidate) -> int:
    return sum(hit is not None for hit in (candidate.dense, candidate.lexical, candidate.entity))


def _empty_reason(candidates: Sequence[Candidate], rejected: Sequence[Candidate]) -> str | None:
    if not candidates:
        return "no eligible records produced a candidate"
    if not rejected:
        return None
    best = min(rejected, key=lambda candidate: (candidate.fused_rank, candidate.record_id))
    return f"best candidate {best.record_id} missed: {best.gate_reason}"
