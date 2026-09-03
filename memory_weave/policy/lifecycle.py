"""Source trust, lifecycle defaults, reinforcement, and supersession authority."""

from __future__ import annotations

from datetime import datetime, timedelta

from memory_weave.config import MemoryWeaveConfig
from memory_weave.models import Record, RecordStatus, SourceKind

_INITIAL_CONFIDENCE: dict[SourceKind, float] = {
    "user_statement": 0.95,
    "system": 0.90,
    "tool_result": 0.85,
    "session_summary": 0.80,
    "agent_inference": 0.60,
}


def rank(source: SourceKind | Record, config: MemoryWeaveConfig) -> int:
    """Return a source's configured authority rank."""

    source_kind = source.source_kind if isinstance(source, Record) else source
    source_rank = config.policy.source_rank
    return {
        "user_statement": source_rank.user_statement,
        "system": source_rank.system,
        "tool_result": source_rank.tool_result,
        "session_summary": source_rank.session_summary,
        "agent_inference": source_rank.agent_inference,
    }[source_kind]


def initial_status(source_kind: SourceKind) -> RecordStatus:
    """Return the lifecycle status assigned to a newly created record."""

    return "provisional" if source_kind == "agent_inference" else "confirmed"


def initial_confidence(source_kind: SourceKind) -> float:
    """Return the documented starting confidence for a source kind."""

    return _INITIAL_CONFIDENCE[source_kind]


def initial_expiry(
    source_kind: SourceKind,
    current_time: datetime,
    config: MemoryWeaveConfig,
) -> datetime | None:
    """Return a provisional expiry for inference and no expiry for direct evidence."""

    if source_kind != "agent_inference":
        return None
    return current_time + timedelta(days=config.ingestion.provisional_ttl_days)


def reinforce(record: Record, current_time: datetime, config: MemoryWeaveConfig) -> Record:
    """Apply one supporting observation and promote provisional records when eligible."""

    record.reinforcements += 1
    record.last_reinforced_at = current_time
    record.confidence = min(0.99, record.confidence + 0.1)
    if record.status == "provisional":
        record.expires_at = current_time + timedelta(days=config.ingestion.provisional_ttl_days)
        if record.reinforcements >= config.ingestion.reinforcements_to_confirm:
            record.status = "confirmed"
            record.expires_at = None
    return record


def has_authority(new: Record, old: Record, config: MemoryWeaveConfig) -> bool:
    """Decide whether a new record can supersede an active record."""

    new_rank = rank(new, config)
    old_rank = rank(old, config)
    if new_rank != old_rank:
        return new_rank > old_rank
    if new.event_at != old.event_at:
        return new.event_at > old.event_at
    return new.created_at >= old.created_at
