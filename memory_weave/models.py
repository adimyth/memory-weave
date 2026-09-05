"""Framework-neutral types shared by storage, ingestion, retrieval, and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

ScopeKind = Literal["agent", "user", "project", "org"]
MemoryType = Literal["semantic", "episodic", "procedural"]
SourceKind = Literal["user_statement", "system", "tool_result", "session_summary", "agent_inference"]
EvidenceSourceKind = Literal["user_statement", "tool_result", "agent_inference"]
RecordStatus = Literal["provisional", "confirmed", "superseded", "expired", "deleted"]
EntityKind = Literal["person", "project", "org", "repo", "product", "other"]
EntityStatus = Literal["provisional", "confirmed", "merged", "deleted"]
EntityRole = Literal["about", "mentions"]
TurnRole = Literal["user", "assistant", "tool"]
RewriteStatus = Literal["disabled", "applied", "unchanged", "failed"]
SearchTrigger = Literal["tool", "auto"]
PRIVATE_SCOPE_SEPARATOR = "/"


def validate_private_scope_component(value: str, name: str) -> None:
    """Reject a private-scope component that would make the encoded pair ambiguous."""

    if PRIVATE_SCOPE_SEPARATOR in value:
        raise ValueError(f"{name} must not contain {PRIVATE_SCOPE_SEPARATOR!r}.")


@dataclass(frozen=True, slots=True)
class Scope:
    kind: ScopeKind
    id: str


@dataclass(slots=True)
class Record:
    id: str
    type: MemoryType
    version: int
    content: str
    subject: str
    scope: Scope
    source_kind: SourceKind
    source_ref: str | None
    creator_agent_id: str
    evidence: str | None
    created_at: datetime
    event_at: datetime
    expires_at: datetime | None
    confidence: float
    status: RecordStatus
    supersedes_id: str | None
    reinforcements: int
    last_reinforced_at: datetime | None
    tags: list[str]
    entity_ids: list[str]
    subject_entity_id: str | None = None
    attribute: str | None = None


@dataclass(slots=True)
class Entity:
    id: str
    kind: EntityKind
    canonical: str
    scope: Scope
    status: EntityStatus
    merged_into: str | None
    aliases: list[str]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class EntityMention:
    kind: EntityKind
    text: str
    role: EntityRole
    entity_id: str | None = None


@dataclass(frozen=True, slots=True)
class Turn:
    session_id: str
    turn: int
    role: TurnRole
    content: str
    at: datetime


@dataclass(frozen=True, slots=True)
class Principal:
    agent_id: str
    user_id: str
    session_id: str | None
    project_id: str | None

    def __post_init__(self) -> None:
        validate_private_scope_component(self.agent_id, "agent_id")
        validate_private_scope_component(self.user_id, "user_id")


@dataclass(frozen=True, slots=True)
class ExtractionContext:
    principal: Principal
    known_entities: list[tuple[str, str, str]]
    existing_subjects: list[str]
    prompt_version: str


@dataclass(frozen=True, slots=True)
class RewriteResult:
    queries: list[str]
    status: Literal["applied", "unchanged", "failed"]


@dataclass(frozen=True, slots=True)
class EvidenceCheck:
    found: bool
    turn: int | None
    role: TurnRole | None
    source_kind: EvidenceSourceKind
    note: str | None


@dataclass(frozen=True, slots=True)
class SearchRequest:
    queries: list[str]
    context: str | None
    types: list[MemoryType] | None
    entities: list[str] | None
    since: datetime | None
    until: datetime | None
    k: int
    include_history: bool
    trigger: SearchTrigger = "tool"


@dataclass(frozen=True, slots=True)
class GeneratorHit:
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class LexicalTerm:
    """One query term that matched an indexed record."""

    value: str
    is_identifier: bool
    is_entity_alias: bool


@dataclass(frozen=True, slots=True)
class LexicalMatch:
    """The terms matched for the single query that gave a record its strongest lexical coverage."""

    terms: tuple[LexicalTerm, ...]
    total_terms: int

    @property
    def fraction(self) -> float:
        """Return the matched share of the query terms used for this lexical hit."""

        return len(self.terms) / self.total_terms if self.total_terms else 0.0


@dataclass(slots=True)
class Candidate:
    record_id: str
    dense: GeneratorHit | None
    lexical: GeneratorHit | None
    lexical_terms: LexicalMatch | None
    entity: GeneratorHit | None
    entity_id: str | None
    rrf_score: float
    fused_rank: int
    freshness_multiplier: float | None
    score: float
    gate_reason: str | None
    rerank_score: float | None
    rank_after_rerank: int | None


@dataclass(slots=True)
class Explanation:
    raw_queries: list[str]
    rewritten_queries: list[str] | None
    rewrite_status: RewriteStatus
    matched_by: list[Literal["dense", "lexical", "entity"]]
    dense: GeneratorHit | None
    lexical: GeneratorHit | None
    lexical_terms: LexicalMatch | None
    entity: GeneratorHit | None
    fused_rank: int
    freshness_multiplier: float | None
    rerank: tuple[int, int, float] | None
    gate: str
    dedup: str
    budget: str
    source_kind: SourceKind
    status: RecordStatus
    created_at: datetime
    event_at: datetime
    entity_ids: list[str]
    conflicts_with: list[str]
    summary: str


@dataclass(slots=True)
class SearchResult:
    record: Record
    score: float
    explanation: Explanation


@dataclass(slots=True)
class SearchResponse:
    search_id: str
    raw_queries: list[str]
    rewritten_queries: list[str] | None
    rewrite_status: RewriteStatus
    results: list[SearchResult]
    empty_reason: str | None
    timings_ms: dict[str, float]


@dataclass(slots=True)
class CandidateRecord:
    type: MemoryType
    content: str
    attribute: str | None
    source_kind: EvidenceSourceKind
    evidence: str
    evidence_turn: int
    entity_mentions: list[EntityMention]
    event_at: datetime | None
    confidence: float


@dataclass(slots=True)
class SessionSummary:
    content: str
    decisions: list[str]
    entity_mentions: list[EntityMention]


@dataclass(slots=True)
class ExtractionOutput:
    candidates: list[CandidateRecord]
    summary: SessionSummary


@dataclass(slots=True)
class Resolution:
    mention: EntityMention
    entity: Entity | None
    outcome: Literal["explicit", "resolved", "created", "ambiguous"]
    candidates: list[str]
