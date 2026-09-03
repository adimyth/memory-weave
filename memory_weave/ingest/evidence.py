"""Transcript-backed evidence validation for explicit and extracted writes."""

from __future__ import annotations

from memory_weave.config import MemoryWeaveConfig
from memory_weave.models import EvidenceCheck, EvidenceSourceKind, Turn
from memory_weave.policy.lifecycle import rank
from memory_weave.util import normalize_ws

from .session import SessionBuffer

_SUPPORTED_SOURCE: dict[str, EvidenceSourceKind] = {
    "user": "user_statement",
    "tool": "tool_result",
    "assistant": "agent_inference",
}


def validate_evidence(
    session_buffer: SessionBuffer,
    session_id: str | None,
    quote: str,
    claimed: EvidenceSourceKind,
    *,
    turn_hint: int | None = None,
    config: MemoryWeaveConfig | None = None,
) -> EvidenceCheck:
    """Find a complete quote in one turn and limit its claimed source authority."""

    normalized_quote = normalize_ws(quote)
    turns = session_buffer.turns(session_id)
    candidates = _matching_turns(turns, turn_hint)
    hit = next(
        (turn for turn in candidates if normalized_quote and normalized_quote in normalize_ws(turn.content)),
        None,
    )
    if hit is None:
        return EvidenceCheck(False, None, None, "agent_inference", "evidence not found in session")

    supported = _SUPPORTED_SOURCE[hit.role]
    if rank(claimed, config) > rank(supported, config):
        note = f"downgraded from {claimed}: quote is from a {hit.role} turn"
        return EvidenceCheck(True, hit.turn, hit.role, supported, note)
    return EvidenceCheck(True, hit.turn, hit.role, claimed, None)


def session_turn_source_ref(session_id: str, turn: int) -> str:
    """Format the durable source reference for a transcript turn."""

    return f"session:{session_id}<turn:{turn}>"


def _matching_turns(turns: list[Turn], turn_hint: int | None) -> list[Turn]:
    if turn_hint is None:
        return turns
    return [turn for turn in turns if turn.turn == turn_hint]
