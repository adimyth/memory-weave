"""The extractor: a finished transcript in, candidate records and one session summary out.

The extractor proposes; it never writes. Its output is validated, reviewed, and revalidated by
``extraction.py`` before anything reaches the ingestor. The prompt lives beside this module and its version
is recorded on every ``extraction.run`` event, so a prompt change is tracked like an embedding-model change.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime
from importlib.resources import files
from typing import Any, Protocol, cast

from memory_weave.hosted import CompletionClient, StructuredOutputError, parse_json_object
from memory_weave.models import (
    CandidateRecord,
    EntityKind,
    EntityMention,
    EntityRole,
    EvidenceSourceKind,
    ExtractionContext,
    ExtractionOutput,
    MemoryType,
    SessionSummary,
    Turn,
)

EXTRACT_PROMPT_VERSION = "extract-v1"

_MEMORY_TYPES: frozenset[str] = frozenset({"semantic", "episodic", "procedural"})
_SOURCE_KINDS: frozenset[str] = frozenset({"user_statement", "tool_result", "agent_inference"})
_ENTITY_KINDS: frozenset[str] = frozenset({"person", "project", "org", "repo", "product", "other"})
_ENTITY_ROLES: frozenset[str] = frozenset({"about", "mentions"})


class ExtractionError(RuntimeError):
    """The extractor could not produce a usable ``ExtractionOutput``; the run fails closed."""


class Extractor(Protocol):
    @property
    def prompt_version(self) -> str: ...

    def extract(self, turns: Sequence[Turn], context: ExtractionContext) -> ExtractionOutput: ...


ExtractorScript = Callable[[Sequence[Turn], ExtractionContext], ExtractionOutput]


class FakeExtractor:
    """Return a canned output, run a script, or raise; records every call for assertions."""

    def __init__(
        self,
        output: ExtractionOutput | ExtractorScript | Exception,
        *,
        prompt_version: str = "fake-v1",
    ) -> None:
        self._output = output
        self._prompt_version = prompt_version
        self.calls: list[tuple[list[Turn], ExtractionContext]] = []

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def extract(self, turns: Sequence[Turn], context: ExtractionContext) -> ExtractionOutput:
        self.calls.append((list(turns), context))
        if isinstance(self._output, Exception):
            raise self._output
        if callable(self._output):
            return self._output(turns, context)
        return self._output


class StructuredLLMExtractor:
    """Call a hosted model with the versioned prompt and parse its JSON into ``ExtractionOutput``."""

    def __init__(self, client: CompletionClient, *, timeout_ms: int, max_candidates: int) -> None:
        self._client = client
        self._timeout_s = timeout_ms / 1000
        self._max_candidates = max_candidates
        self._system = load_prompt("extract_v1.md")

    @property
    def prompt_version(self) -> str:
        return EXTRACT_PROMPT_VERSION

    def extract(self, turns: Sequence[Turn], context: ExtractionContext) -> ExtractionOutput:
        try:
            text = self._client.complete(self._system, render_request(turns, context), timeout_s=self._timeout_s)
            payload = parse_json_object(text)
            return parse_extraction_output(payload, max_candidates=self._max_candidates)
        except (StructuredOutputError, ValueError, KeyError, TypeError) as error:
            raise ExtractionError(f"malformed extractor output: {error}") from error
        except Exception as error:  # noqa: BLE001 - any provider failure fails the run closed
            raise ExtractionError(f"{type(error).__name__}: {error}") from error


def load_prompt(name: str) -> str:
    """Read one versioned prompt shipped with the package."""

    return files("memory_weave.ingest").joinpath("prompts").joinpath(name).read_text(encoding="utf-8")


def render_request(turns: Sequence[Turn], context: ExtractionContext) -> str:
    """Number the transcript and attach the bounded context the prompt describes."""

    lines = [f"User id: {context.principal.user_id}", "", "Transcript:"]
    for turn in turns:
        lines.append(f"[{turn.turn}] {turn.role} at {turn.at.isoformat()}: {turn.content}")
    lines.append("")
    lines.append("Known entities (id, kind, name):")
    lines.extend(f"- {entity_id}, {kind}, {name}" for entity_id, kind, name in context.known_entities)
    if not context.known_entities:
        lines.append("- none")
    lines.append("")
    lines.append("Known subjects (entity/attribute) already stored:")
    lines.extend(f"- {subject}" for subject in context.existing_subjects)
    if not context.existing_subjects:
        lines.append("- none")
    return "\n".join(lines)


def parse_extraction_output(payload: dict[str, Any], *, max_candidates: int) -> ExtractionOutput:
    """Validate the model's JSON strictly; anything outside the contract is a malformed output."""

    raw_candidates = payload.get("candidates", [])
    raw_summary = payload.get("summary")
    if not isinstance(raw_candidates, list) or not isinstance(raw_summary, dict):
        raise ValueError("candidates must be a list and summary an object")
    candidates = [_parse_candidate(item) for item in raw_candidates[:max_candidates]]
    summary = SessionSummary(
        content=_string(raw_summary, "content"),
        decisions=[str(item) for item in raw_summary.get("decisions", []) or []],
        entity_mentions=_parse_mentions(raw_summary.get("entities", [])),
    )
    return ExtractionOutput(candidates=candidates, summary=summary)


def _parse_candidate(item: object) -> CandidateRecord:
    if not isinstance(item, dict):
        raise ValueError("candidate must be an object")
    memory_type = _string(item, "type")
    source_kind = _string(item, "source_kind")
    if memory_type not in _MEMORY_TYPES:
        raise ValueError(f"unknown memory type {memory_type!r}")
    if source_kind not in _SOURCE_KINDS:
        raise ValueError(f"unknown source kind {source_kind!r}")
    evidence_turn = item.get("evidence_turn")
    if not isinstance(evidence_turn, int) or isinstance(evidence_turn, bool):
        raise ValueError("evidence_turn must be an integer")
    confidence = item.get("confidence", 0.5)
    if not isinstance(confidence, int | float) or not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    attribute = item.get("attribute")
    return CandidateRecord(
        type=cast(MemoryType, memory_type),
        content=_string(item, "content"),
        attribute=str(attribute) if attribute else None,
        source_kind=cast(EvidenceSourceKind, source_kind),
        evidence=_string(item, "evidence"),
        evidence_turn=evidence_turn,
        entity_mentions=_parse_mentions(item.get("entities", [])),
        event_at=parse_timestamp(item.get("event_at")),
        confidence=float(confidence),
        valid_from=parse_timestamp(item.get("valid_from")),
        valid_until=parse_timestamp(item.get("valid_until")),
        review_at=parse_timestamp(item.get("review_at")),
    )


def _parse_mentions(value: object) -> list[EntityMention]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("entities must be a list")
    mentions: list[EntityMention] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("entity mention must be an object")
        kind = _string(item, "kind")
        role = str(item.get("role", "mentions"))
        if kind not in _ENTITY_KINDS or role not in _ENTITY_ROLES:
            raise ValueError(f"unknown entity kind or role: {kind!r}, {role!r}")
        mentions.append(
            EntityMention(kind=cast(EntityKind, kind), text=_string(item, "text"), role=cast(EntityRole, role))
        )
    return mentions


def _string(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def parse_timestamp(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("timestamps must be ISO 8601 strings")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must carry a timezone")
    return parsed


def dump_candidate(candidate: CandidateRecord) -> dict[str, Any]:
    """Serialise a candidate for prompts and audit payloads."""

    return {
        "type": candidate.type,
        "content": candidate.content,
        "attribute": candidate.attribute,
        "source_kind": candidate.source_kind,
        "evidence": candidate.evidence,
        "evidence_turn": candidate.evidence_turn,
        "entities": [
            {"kind": mention.kind, "text": mention.text, "role": mention.role} for mention in candidate.entity_mentions
        ],
        "event_at": candidate.event_at.isoformat() if candidate.event_at else None,
        "valid_from": candidate.valid_from.isoformat() if candidate.valid_from else None,
        "valid_until": candidate.valid_until.isoformat() if candidate.valid_until else None,
        "review_at": candidate.review_at.isoformat() if candidate.review_at else None,
        "confidence": candidate.confidence,
    }


def dump_candidate_json(candidate: CandidateRecord) -> str:
    return json.dumps(dump_candidate(candidate), indent=2, sort_keys=True)
