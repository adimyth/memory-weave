"""Transcript-backed evidence validation for explicit and extracted writes."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from retold.config import RetoldConfig
from retold.models import EvidenceCheck, EvidenceSourceKind, Principal, Scope, Turn, TurnRole
from retold.policy.lifecycle import rank
from retold.store import Store
from retold.util import normalize_alias, normalize_ws

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
    config: RetoldConfig,
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


_PRINCIPAL_PLACEHOLDER = "the user"
_FIRST_PERSON = re.compile(r"\b(?:I|I'm|I’m|I've|I’ve|I'd|I’d|I'll|I’ll|me|my|mine|myself)\b", re.IGNORECASE)


def quote_is_first_person(quote: str) -> bool:
    """Whether the speaker is talking about themselves, which is when a claim naming them is the same claim.

    A quote about someone else ("Priya prefers tea") must not support a claim about the principal, and
    rewriting the principal's name as "the user" would make that claim generic enough to pass. So the
    rewrite applies only to first-person quotes.
    """

    return _FIRST_PERSON.search(quote) is not None


def principal_names(store: Store, principal: Principal) -> list[str]:
    """The names a claim may use for the principal: the user id, and the person entity's canonical name and aliases.

    Nothing is created here; a host that never provisioned the user gets the id alone.
    """

    names = {principal.user_id}
    for entity in store.entities_by_alias(
        normalize_alias(principal.user_id), kinds=["person"], scopes=[Scope(kind="user", id=principal.user_id)]
    ):
        names.add(entity.canonical)
        names.update(entity.aliases)
    # A one-character alias would rewrite articles and initials, so only names of two characters or more count.
    return sorted((name for name in names if len(name.strip()) >= 2), key=len, reverse=True)


def neutralise_principal(claim: str, names: Iterable[str]) -> str | None:
    """Rewrite the principal's own name in a claim as "the user", or return None when the claim names nobody.

    An NLI judge cannot know that "Aditya" is the person who said "I prefer concise answers", so a
    third-person claim about the principal scores as unsupported against a first-person quote. Judging the
    neutralised claim as well closes that one blind spot without weakening the check for any other claim.
    """

    rewritten = claim
    for name in names:
        pattern = re.compile(r"\b" + re.escape(name) + r"(?P<possessive>['’]s)?\b", re.IGNORECASE)
        rewritten = pattern.sub(
            lambda match: _PRINCIPAL_PLACEHOLDER + ("'s" if match.group("possessive") else ""), rewritten
        )
    if rewritten == claim:
        return None
    if rewritten.startswith(_PRINCIPAL_PLACEHOLDER):
        rewritten = "T" + rewritten[1:]
    return rewritten


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


def _is_substantive_quote(value: str, config: RetoldConfig) -> bool:
    evidence_config = config.ingestion.evidence
    return len(value) >= evidence_config.min_characters or len(value.split()) >= evidence_config.min_words
