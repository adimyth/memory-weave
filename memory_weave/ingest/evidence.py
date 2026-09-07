"""Transcript-backed evidence validation for explicit and extracted writes."""

from __future__ import annotations

import unicodedata

from memory_weave.config import MemoryWeaveConfig
from memory_weave.models import EvidenceCheck, EvidenceSourceKind, Turn, TurnRole
from memory_weave.policy.lifecycle import rank
from memory_weave.util import normalize_ws

from .session import SessionBuffer

_SUPPORTED_SOURCE: dict[TurnRole, EvidenceSourceKind] = {
    "user": "user_statement",
    "tool": "tool_result",
    "assistant": "agent_inference",
}
# One matched pair of these, after NFKC folds the typographic forms, is punctuation the writer added.
_WRAPPING_QUOTES = {chr(34): chr(34), chr(39): chr(39)}
_EVIDENCE_FOLD = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‛": "'",
        "′": "'",
        "“": '"',
        "”": '"',
        "‟": '"',
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
    }
)


def validate_evidence(
    session_buffer: SessionBuffer,
    session_id: str | None,
    quote: str,
    claimed: EvidenceSourceKind,
    config: MemoryWeaveConfig,
    *,
    turn_hint: int | None = None,
) -> EvidenceCheck:
    """Find a complete quote in one turn and limit its claimed source authority."""

    normalized_quote = _normalize_evidence_quote(quote)
    if not _is_substantive_quote(normalized_quote, config):
        return EvidenceCheck(False, None, None, "agent_inference", "evidence quote is too short")
    turns = session_buffer.turns(session_id)
    candidates = _matching_turns(turns, turn_hint)
    hit = next(
        (turn for turn in candidates if normalized_quote in _normalize_evidence_quote(turn.content)),
        None,
    )
    if hit is None:
        return EvidenceCheck(False, None, None, "agent_inference", "evidence not found in session")

    supported = _SUPPORTED_SOURCE.get(hit.role)
    if supported is None:
        return EvidenceCheck(
            True,
            hit.turn,
            None,
            "agent_inference",
            "unsupported transcript role; treated as agent_inference",
        )
    if rank(claimed, config) > rank(supported, config):
        article = "an" if hit.role == "assistant" else "a"
        note = f"downgraded from {claimed}: quote is from {article} {hit.role} turn"
        return EvidenceCheck(True, hit.turn, hit.role, supported, note)
    return EvidenceCheck(True, hit.turn, hit.role, claimed, None)


def session_turn_source_ref(session_id: str, turn: int) -> str:
    """Format the durable source reference for a transcript turn."""

    return f"session:{session_id}<turn:{turn}>"


def _matching_turns(turns: list[Turn], turn_hint: int | None) -> list[Turn]:
    if turn_hint is None:
        return turns
    return [turn for turn in turns if turn.turn == turn_hint]


def _normalize_evidence_quote(value: str) -> str:
    folded = normalize_ws(unicodedata.normalize("NFKC", value).translate(_EVIDENCE_FOLD))
    return _strip_wrapping_quotes(folded)


def _strip_wrapping_quotes(value: str) -> str:
    """Drop one matched pair of surrounding quote marks that a writer added around the quotation itself.

    A model asked for a verbatim quote commonly returns it already quoted. The inner text must still match
    the transcript exactly, so this only removes punctuation the writer wrapped around the evidence.
    """

    while len(value) >= 2 and value[0] in _WRAPPING_QUOTES and value[-1] == _WRAPPING_QUOTES[value[0]]:
        stripped = value[1:-1].strip()
        if not stripped:
            return value
        value = stripped
    return value


def _is_substantive_quote(value: str, config: MemoryWeaveConfig) -> bool:
    evidence_config = config.ingestion.evidence
    return len(value) >= evidence_config.min_characters or len(value.split()) >= evidence_config.min_words
