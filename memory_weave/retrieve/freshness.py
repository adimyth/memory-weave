"""Recency adjustment for episodic memories after rank fusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from memory_weave.config import FreshnessConfig
from memory_weave.models import Candidate, Record, SearchRequest


def apply_freshness(
    candidates: Sequence[Candidate],
    records: Mapping[str, Record],
    request: SearchRequest,
    current_time: datetime,
    config: FreshnessConfig,
) -> list[Candidate]:
    """Apply episodic decay unless the caller supplied an explicit event-time window."""

    skip_decay = request.since is not None or request.until is not None
    for candidate in candidates:
        record = records[candidate.record_id]
        if record.type != "episodic" or skip_decay:
            candidate.freshness_multiplier = None
            candidate.score = candidate.rrf_score
            continue
        age_days = max(0.0, (current_time - record.event_at).total_seconds() / 86_400)
        multiplier = max(config.floor, 0.5 ** (age_days / config.episodic_half_life_days))
        candidate.freshness_multiplier = multiplier
        candidate.score = candidate.rrf_score * multiplier
    return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.fused_rank, candidate.record_id))
