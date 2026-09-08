"""Review before apply: an independent judgement on each extracted candidate.

The reviewer sees the candidate, its evidence turn with the turns around it, and the live records about the
same subject in the destination scope. It accepts, rejects, or narrows. ``check_revision`` enforces what a
revision may touch: content, the attribute hint, temporal metadata, and confidence, each only in the
narrowing direction. Source authority, the quote and its turn, the scope, the entity mentions, and the
event time are outside its reach; the extraction runner rejects any revision that changes them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

from memory_weave.models import CandidateRecord, Record, Scope, Turn

from .extractor import dump_candidate, load_prompt, parse_timestamp
from .hosted import CompletionClient, StructuredOutputError, parse_json_object

REVIEW_PROMPT_VERSION = "review-v1"

ReviewOutcome = Literal["accept", "reject", "revise"]
_OUTCOMES: frozenset[str] = frozenset({"accept", "reject", "revise"})
_REVISABLE_FIELDS: frozenset[str] = frozenset(
    {"content", "attribute", "valid_from", "valid_until", "review_at", "confidence"}
)


class ReviewError(RuntimeError):
    """The reviewer failed or answered outside its contract; the whole extraction run fails closed."""


@dataclass(frozen=True, slots=True)
class ReviewRequest:
    candidate: CandidateRecord
    scope: Scope
    evidence_turn: Turn
    adjacent_turns: list[Turn]
    live_records: list[Record]


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    outcome: ReviewOutcome
    reason: str
    revised: CandidateRecord | None = None


class CandidateReviewer(Protocol):
    @property
    def version(self) -> str: ...

    def review(self, request: ReviewRequest) -> ReviewDecision: ...


ReviewTable = Mapping[str, ReviewDecision | Exception]
ReviewScript = Callable[[ReviewRequest], ReviewDecision]


class TableReviewer:
    """Decide by candidate content from a table, or by a script; accept anything unlisted."""

    def __init__(
        self,
        decisions: ReviewTable | ReviewScript | None = None,
        *,
        version: str = "fake-review-v1",
    ) -> None:
        self._decisions = decisions
        self._version = version
        self.requests: list[ReviewRequest] = []

    @property
    def version(self) -> str:
        return self._version

    def review(self, request: ReviewRequest) -> ReviewDecision:
        self.requests.append(request)
        if self._decisions is None:
            return ReviewDecision("accept", "table default")
        if callable(self._decisions):
            return self._decisions(request)
        decision = self._decisions.get(request.candidate.content)
        if decision is None:
            return ReviewDecision("accept", "table default")
        if isinstance(decision, Exception):
            raise decision
        return decision


class StructuredLLMReviewer:
    """Ask a hosted model for accept, reject, or a narrowing revision, using the versioned prompt."""

    def __init__(self, client: CompletionClient, *, timeout_ms: int) -> None:
        self._client = client
        self._timeout_s = timeout_ms / 1000
        self._system = load_prompt("review_v1.md")

    @property
    def version(self) -> str:
        return REVIEW_PROMPT_VERSION

    def review(self, request: ReviewRequest) -> ReviewDecision:
        try:
            text = self._client.complete(self._system, render_review_request(request), timeout_s=self._timeout_s)
            return parse_review_decision(parse_json_object(text), request.candidate)
        except (StructuredOutputError, ValueError, KeyError, TypeError) as error:
            raise ReviewError(f"malformed reviewer output: {error}") from error
        except Exception as error:  # noqa: BLE001 - any provider failure fails the run closed
            raise ReviewError(f"{type(error).__name__}: {error}") from error


def render_review_request(request: ReviewRequest) -> str:
    lines = ["Candidate:", json.dumps(dump_candidate(request.candidate), indent=2, sort_keys=True), ""]
    lines.append("Evidence turn and its neighbours:")
    ordered = sorted([request.evidence_turn, *request.adjacent_turns], key=lambda turn: turn.turn)
    for turn in ordered:
        marker = " (quoted)" if turn.turn == request.evidence_turn.turn else ""
        lines.append(f"[{turn.turn}] {turn.role}{marker}: {turn.content}")
    lines.append("")
    lines.append(f"Destination scope: {request.scope.kind}:{request.scope.id}")
    lines.append("Live records about the same subject:")
    for record in request.live_records:
        lines.append(f"- [{record.attribute or '-'}] ({record.source_kind}, {record.status}) {record.content}")
    if not request.live_records:
        lines.append("- none")
    return "\n".join(lines)


def parse_review_decision(payload: dict[str, Any], candidate: CandidateRecord) -> ReviewDecision:
    outcome = str(payload.get("outcome", "")).strip().lower()
    if outcome not in _OUTCOMES:
        raise ValueError(f"unknown review outcome {outcome!r}")
    reason = str(payload.get("reason", "")).strip()
    if outcome != "revise":
        return ReviewDecision(outcome, reason)  # type: ignore[arg-type]
    revision = payload.get("revision")
    if not isinstance(revision, dict) or not revision:
        raise ValueError("a revise outcome needs a non-empty revision object")
    unknown = set(revision) - _REVISABLE_FIELDS
    if unknown:
        raise ValueError(f"revision touches non-revisable fields: {sorted(unknown)}")
    changes: dict[str, Any] = {}
    for key, value in revision.items():
        if key in ("valid_from", "valid_until", "review_at"):
            changes[key] = parse_timestamp(value)
        elif key == "confidence":
            changes[key] = float(value)
        elif key == "attribute":
            changes[key] = str(value) if value else None
        else:
            changes[key] = str(value)
    return ReviewDecision("revise", reason, replace(candidate, **changes))


def check_revision(original: CandidateRecord, revised: CandidateRecord) -> str | None:
    """Return why a revision is invalid, or ``None`` when it only narrows the candidate."""

    if revised.type != original.type:
        return "revision changed the memory type"
    if revised.source_kind != original.source_kind:
        return "revision changed the source kind"
    if revised.evidence != original.evidence or revised.evidence_turn != original.evidence_turn:
        return "revision replaced the evidence"
    if revised.entity_mentions != original.entity_mentions:
        return "revision changed the entities"
    if revised.event_at != original.event_at:
        return "revision changed the event time"
    if revised.confidence > original.confidence:
        return "revision raised the confidence"
    if not revised.content.strip():
        return "revision emptied the content"
    if revised.valid_from is not None and (original.valid_from is None or revised.valid_from < original.valid_from):
        return "revision widened valid_from"
    if revised.valid_until is not None and (original.valid_until is None or revised.valid_until > original.valid_until):
        return "revision widened valid_until"
    if revised.review_at is not None and (original.review_at is None or revised.review_at > original.review_at):
        return "revision delayed review_at"
    return None


def adjacent_turns(turns: Sequence[Turn], turn_number: int) -> list[Turn]:
    """The turns immediately before and after ``turn_number``."""

    return [turn for turn in turns if turn.turn in (turn_number - 1, turn_number + 1)]
