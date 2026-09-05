"""The explicit, evidence-backed memory write path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Literal, cast

import numpy as np

from memory_weave.config import MemoryWeaveConfig
from memory_weave.index.embedder import Embedder
from memory_weave.index.vector import VectorIndex
from memory_weave.models import (
    Entity,
    EntityKind,
    EntityMention,
    EntityRole,
    EvidenceCheck,
    EvidenceSourceKind,
    MemoryType,
    Principal,
    Record,
    RecordStatus,
    Resolution,
    Scope,
    SourceKind,
)
from memory_weave.policy import (
    has_authority,
    initial_confidence,
    initial_expiry,
    initial_status,
    provisional_expiry,
    rank,
    reinforce,
    writable_scopes,
)
from memory_weave.store import Store
from memory_weave.util import Timer, normalize_attribute, now, render_subject, uuid7

from .entities import PrincipalEntityAmbiguousError, aliases_text, primary_entity_for, resolve_entities
from .equivalence import EquivalenceJudge
from .evidence import session_turn_source_ref, validate_evidence
from .session import SessionBuffer

_TOOL_SOURCE_KINDS: frozenset[SourceKind] = frozenset({"user_statement", "tool_result", "agent_inference"})
_WRITE_STAGES = (
    "permission",
    "evidence",
    "entities",
    "embed",
    "dedup_search",
    "judge",
    "supersession",
    "persistence",
    "event_log",
    "transaction",
    "index_update",
)


@dataclass(frozen=True, slots=True)
class WriteRequest:
    """The validated values accepted by the framework-neutral memory write path."""

    type: MemoryType
    content: str
    source_kind: SourceKind
    evidence: str | None
    attribute: str | None = None
    scope: Scope | None = None
    event_at: datetime | None = None
    entities: list[EntityMention] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    evidence_turn: int | None = None


@dataclass(frozen=True, slots=True)
class WriteResult:
    """The durable outcome, lifecycle status, audit note, and timings of one write."""

    record_id: str | None
    status: RecordStatus | None
    outcome: str
    note: str | None
    timings_ms: dict[str, float]
    candidates: list[EntityAmbiguityCandidate] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EntityAmbiguityCandidate:
    """One readable entity returned to let a caller resolve an ambiguous mention safely."""

    id: str
    kind: EntityKind
    canonical: str
    scope: Scope


class _EntityAmbiguous(Exception):
    """Carry ambiguous resolutions out of a rolled-back write transaction."""

    def __init__(self, candidates: list[str], mentions: list[EntityMention]) -> None:
        super().__init__("An about-role entity alias is ambiguous.")
        self.candidates = candidates
        self.mentions = mentions


class _InvalidSubject(Exception):
    """Carry a subject-contract failure out of a rolled-back write transaction."""

    def __init__(self, note: str) -> None:
        super().__init__(note)
        self.note = note


@dataclass(frozen=True, slots=True)
class _WriteDecision:
    """Hold one deduplication decision and the audit facts it produced."""

    outcome: str
    existing: Record | None
    dedup_kind: str | None
    extra: dict[str, object]


@dataclass(frozen=True, slots=True)
class _PersistedOutcome:
    """Hold a committed-write outcome and the vector update that follows its transaction."""

    record_id: str
    status: RecordStatus
    outcome: str
    note: str | None
    index_record_id: str | None
    index_vector: np.ndarray | None


class Ingestor:
    """Write explicit memories with evidence, lifecycle checks, durable indexes, and audit events."""

    def __init__(
        self,
        store: Store,
        vector_index: VectorIndex,
        embedder: Embedder,
        judge: EquivalenceJudge,
        session_buffer: SessionBuffer,
        config: MemoryWeaveConfig,
        *,
        current_time: Callable[[], datetime] = now,
    ) -> None:
        if (embedder.name, embedder.version, embedder.dims) != (
            vector_index.model,
            vector_index.version,
            vector_index.dims,
        ):
            raise ValueError("The embedder and vector index must use the same model, version, and dimensions.")
        self._store = store
        self._vector_index = vector_index
        self._embedder = embedder
        self._judge = judge
        self._session_buffer = session_buffer
        self._config = config
        self._current_time = current_time

    def write(self, principal: Principal, request: WriteRequest) -> WriteResult:
        """Validate and persist one explicit memory or return its safe non-write outcome."""

        timer = Timer(warm=self._embedder.is_loaded and self._vector_index.is_loaded)
        current_time = self._current_time()
        scope = request.scope or Scope(kind="user", id=principal.user_id)
        if scope not in writable_scopes(self._store, principal.agent_id, principal.user_id):
            timer.mark("permission")
            return self._result(None, None, "scope_not_writable", None, timer)
        timer.mark("permission")

        if request.source_kind not in _TOOL_SOURCE_KINDS:
            return self._result(
                None,
                None,
                "invalid_source_kind",
                f"{request.source_kind} is reserved for the host or extractor.",
                timer,
            )

        if request.type in ("semantic", "procedural") and not (request.attribute or "").strip():
            return self._result(
                None,
                None,
                "invalid_subject",
                f"{request.type} records require an attribute.",
                timer,
            )

        evidence, entailment_score = self._validate_evidence(principal, request, timer)
        source_kind = evidence.source_kind
        source_ref = (
            session_turn_source_ref(principal.session_id, evidence.turn)
            if evidence.found and principal.session_id is not None and evidence.turn is not None
            else None
        )
        record = self._new_record(request, principal, scope, source_kind, source_ref, evidence, current_time)

        persisted: _PersistedOutcome | None = None
        try:
            with self._store.transaction():
                resolutions = resolve_entities(request.entities, scope, principal, self._store)
                ambiguous = [resolution for resolution in resolutions if resolution.outcome == "ambiguous"]
                ambiguous_about = [resolution for resolution in ambiguous if resolution.mention.role == "about"]
                if ambiguous_about:
                    candidate_ids = sorted(
                        {candidate for resolution in ambiguous_about for candidate in resolution.candidates}
                    )
                    raise _EntityAmbiguous(candidate_ids, [resolution.mention for resolution in ambiguous_about])
                entity_ids = list(
                    dict.fromkeys(resolution.entity.id for resolution in resolutions if resolution.entity is not None)
                )
                entity_roles = _entity_roles(resolutions)
                primary, attribute = self._subject_for(request, resolutions, principal, scope)
                if primary is not None:
                    entity_ids = list(dict.fromkeys([*entity_ids, primary.id]))
                    entity_roles[primary.id] = "about"
                    record.subject_entity_id = primary.id
                    record.attribute = attribute
                    record.subject = render_subject(primary.id, attribute)
                record.entity_ids = entity_ids
                timer.mark("entities")

                vector = self._embedder.embed_documents([record.content])[0]
                timer.mark("embed")

                outcome = self._decide_outcome(record, vector, current_time, timer)
                note = _combine_notes(evidence.note, _ambiguous_mentions_note(ambiguous))
                persisted = self._persist_outcome(
                    record,
                    vector,
                    outcome,
                    note,
                    entity_roles,
                    entailment_score,
                    timer,
                )
        except _EntityAmbiguous as ambiguity:
            return self._record_ambiguity(principal, ambiguity, timer)
        except _InvalidSubject as invalid:
            return self._result(None, None, "invalid_subject", invalid.note, timer)

        assert persisted is not None
        timer.mark("transaction")
        if persisted.index_record_id is not None and persisted.index_vector is not None:
            self._vector_index.upsert(persisted.index_record_id, persisted.index_vector)
        timer.mark("index_update")
        return self._result(
            persisted.record_id,
            persisted.status,
            persisted.outcome,
            persisted.note,
            timer,
        )

    def revise(
        self,
        principal: Principal,
        record_id: str,
        action: Literal["confirm", "supersede", "expire"],
        reason: str,
        *,
        content: str | None = None,
    ) -> WriteResult:
        """Apply a user-authorized lifecycle revision after checking the principal can write the record scope."""

        timer = Timer(warm=self._embedder.is_loaded and self._vector_index.is_loaded)
        existing = self._store.get_record(record_id)
        if existing is None:
            return self._result(None, None, "not_found", None, timer)
        if existing.scope not in writable_scopes(self._store, principal.agent_id, principal.user_id):
            return self._result(None, None, "scope_not_writable", None, timer)
        timer.mark("permission")
        current_time = self._current_time()

        if action == "confirm":
            return self._revise_status(existing, "confirmed", "record.confirmed", reason, principal, timer)
        if action == "expire":
            return self._revise_status(existing, "expired", "record.expired", reason, principal, timer)
        if content is None or not content.strip():
            return self._result(None, None, "invalid_input", "supersede requires content", timer)

        revision = Record(
            id=uuid7(),
            type=existing.type,
            version=existing.version + 1,
            content=content,
            subject=existing.subject,
            subject_entity_id=existing.subject_entity_id,
            attribute=existing.attribute,
            scope=existing.scope,
            source_kind="agent_inference",
            source_ref=None,
            creator_agent_id=principal.agent_id,
            evidence=None,
            created_at=current_time,
            event_at=current_time,
            expires_at=provisional_expiry(current_time, self._config),
            confidence=initial_confidence("agent_inference"),
            status="provisional",
            supersedes_id=existing.id,
            reinforcements=0,
            last_reinforced_at=None,
            tags=list(existing.tags),
            entity_ids=list(existing.entity_ids),
        )
        vector = self._embedder.embed_documents([revision.content])[0]
        timer.mark("embed")
        with self._store.transaction():
            self._store.update_status(existing.id, "superseded")
            self._store.insert_record(revision)
            self._store.put_embedding(revision.id, self._embedder.name, self._embedder.version, vector)
            self._store.upsert_fts(
                revision.id,
                revision.content,
                revision.subject,
                aliases_text(self._store, revision.entity_ids),
            )
            for entity_id in revision.entity_ids:
                role: EntityRole = "about" if entity_id == revision.subject_entity_id else "mentions"
                self._store.link_record_entity(revision.id, entity_id, role)
            self._store.append_event(
                "record.superseded",
                principal.agent_id,
                revision.id,
                None,
                {
                    "manual_revision": True,
                    "reason": reason,
                    "source_kind": revision.source_kind,
                    "supersedes_id": existing.id,
                },
            )
        timer.mark("persistence")
        timer.mark("event_log")
        timer.mark("transaction")
        self._vector_index.upsert(revision.id, vector)
        timer.mark("index_update")
        return self._result(revision.id, revision.status, "superseded", None, timer)

    def _revise_status(
        self,
        record: Record,
        status: RecordStatus,
        event_kind: str,
        reason: str,
        principal: Principal,
        timer: Timer,
    ) -> WriteResult:
        """Persist one direct lifecycle transition and its audit event."""

        with self._store.transaction():
            self._store.update_status(record.id, status)
            self._store.append_event(event_kind, principal.agent_id, record.id, None, {"reason": reason})
        timer.mark("persistence")
        timer.mark("event_log")
        timer.mark("transaction")
        return self._result(record.id, status, status, None, timer)

    def _validate_evidence(
        self, principal: Principal, request: WriteRequest, timer: Timer
    ) -> tuple[EvidenceCheck, float | None]:
        if request.evidence is None:
            evidence = EvidenceCheck(False, None, None, "agent_inference", "evidence not provided")
        else:
            claimed = cast(EvidenceSourceKind, request.source_kind)
            evidence = validate_evidence(
                self._session_buffer,
                principal.session_id,
                request.evidence,
                claimed,
                self._config,
                turn_hint=request.evidence_turn,
            )
        entailment_score: float | None = None
        direct_claim = evidence.source_kind in ("user_statement", "tool_result")
        if evidence.found and direct_claim and request.evidence is not None:
            entailment_score = self._judge.entails(request.evidence, request.content)
            if not 0.0 <= entailment_score <= 1.0:
                raise ValueError("Evidence entailment scores must be between 0 and 1.")
            if entailment_score < self._config.ingestion.evidence.entail_floor:
                evidence = replace(
                    evidence,
                    source_kind="agent_inference",
                    note=_combine_notes(evidence.note, "evidence does not support claim"),
                )
        timer.mark("evidence")
        return evidence, entailment_score

    def _new_record(
        self,
        request: WriteRequest,
        principal: Principal,
        scope: Scope,
        source_kind: EvidenceSourceKind,
        source_ref: str | None,
        evidence: EvidenceCheck,
        current_time: datetime,
    ) -> Record:
        return Record(
            id=uuid7(),
            type=request.type,
            version=1,
            content=request.content,
            subject="",
            subject_entity_id=None,
            attribute=None,
            scope=scope,
            source_kind=source_kind,
            source_ref=source_ref,
            creator_agent_id=principal.agent_id,
            evidence=request.evidence if evidence.found else None,
            created_at=current_time,
            event_at=request.event_at or current_time,
            expires_at=initial_expiry(source_kind, current_time, self._config),
            confidence=initial_confidence(source_kind),
            status=initial_status(source_kind),
            supersedes_id=None,
            reinforcements=0,
            last_reinforced_at=None,
            tags=list(request.tags),
            entity_ids=[],
        )

    def _subject_for(
        self,
        request: WriteRequest,
        resolutions: list[Resolution],
        principal: Principal,
        scope: Scope,
    ) -> tuple[Entity | None, str | None]:
        primary = next(
            (
                resolution.entity
                for resolution in resolutions
                if resolution.mention.role == "about" and resolution.entity is not None
            ),
            None,
        )
        if request.type in ("semantic", "procedural"):
            if primary is None:
                if scope.kind != "user" or scope.id != principal.user_id:
                    raise _InvalidSubject("about entity required")
                primary = self._principal_entity(principal, scope)
            attribute = normalize_attribute(request.attribute or "")
            if not attribute:
                raise _InvalidSubject("attribute normalizes to nothing")
            return primary, attribute
        if primary is None and scope.kind == "user" and scope.id == principal.user_id:
            primary = self._principal_entity(principal, scope)
        return primary, "-" if primary is not None else None

    def _principal_entity(self, principal: Principal, scope: Scope) -> Entity:
        try:
            return primary_entity_for(principal, scope, self._store)
        except PrincipalEntityAmbiguousError as ambiguity:
            mention = EntityMention(kind="person", text=principal.user_id, role="about")
            raise _EntityAmbiguous(sorted(ambiguity.candidates), [mention]) from ambiguity

    def _decide_outcome(
        self,
        record: Record,
        vector: np.ndarray,
        current_time: datetime,
        timer: Timer,
    ) -> _WriteDecision:
        self._ensure_index_loaded()
        attribute_records, scan_truncated = self._attribute_records(record, vector)
        neighbours = self._nearby_records(record, vector, {existing.id for existing in attribute_records}, current_time)
        timer.mark("dedup_search")
        extra: dict[str, object] = {"attribute_scan_truncated": True} if scan_truncated else {}

        # Judge every scanned record first. The same-attribute authority incumbent then receives the
        # supersession decision before any reinforcement, so a "same" verdict against a provisional
        # sibling can never leave a contradicted incumbent active.
        verdicts = [(existing, self._judge.judge(existing.content, record.content)) for existing in attribute_records]
        same_attribute = [existing for existing, _ in verdicts if existing.attribute == record.attribute]

        if same_attribute:
            incumbent = max(same_attribute, key=lambda candidate: _authority_key(candidate, self._config))
            incumbent_verdict = next(verdict for existing, verdict in verdicts if existing is incumbent)
            timer.mark("judge")
            if incumbent_verdict == "same":
                timer.mark("supersession")
                return _WriteDecision("reinforced", incumbent, "same_subject", extra)
            outcome, _, _ = self._supersession_outcome(record, incumbent, current_time, timer)
            if outcome == "superseded":
                subsumed = [
                    existing.id
                    for existing, verdict in verdicts
                    if existing is not incumbent and existing.attribute == record.attribute and verdict == "same"
                ]
                if subsumed:
                    extra["also_superseded"] = subsumed
            return _WriteDecision(outcome, incumbent, "same_subject", extra)

        aliased_same = [existing for existing, verdict in verdicts if verdict == "same"]
        if aliased_same:
            existing = max(aliased_same, key=lambda candidate: _authority_key(candidate, self._config))
            extra["attribute_aliased_from"] = record.attribute or ""
            timer.mark("judge")
            timer.mark("supersession")
            return _WriteDecision("reinforced", existing, "attribute", extra)

        contradictions = [existing for existing, verdict in verdicts if verdict == "contradicts"]
        if contradictions:
            existing = max(contradictions, key=lambda candidate: _authority_key(candidate, self._config))
            timer.mark("judge")
            extra["attribute_aliased_from"] = record.attribute or ""
            record.attribute = existing.attribute
            record.subject = render_subject(record.subject_entity_id, record.attribute)
            outcome, incumbent, _ = self._supersession_outcome(record, existing, current_time, timer)
            return _WriteDecision(outcome, incumbent, "attribute", extra)

        for neighbour in neighbours:
            verdict = self._judge.judge(neighbour.content, record.content)
            if verdict == "same":
                timer.mark("judge")
                timer.mark("supersession")
                return _WriteDecision("reinforced", neighbour, "nearby_subject", extra)
        timer.mark("judge")
        timer.mark("supersession")
        return _WriteDecision("created", None, None, extra)

    def _attribute_records(self, record: Record, vector: np.ndarray) -> tuple[list[Record], bool]:
        if not _is_current_fact(record):
            return [], False
        assert record.subject_entity_id is not None
        assert record.attribute is not None
        # The store returns reinforced-first order; the cap is applied to that order so a recently
        # reinforced attribute is never excluded by cosine. Cosine only orders the records kept.
        records = self._store.active_for_entity(record.scope, record.subject_entity_id, record.type)
        same_subject = [existing for existing in records if existing.attribute == record.attribute]
        incumbent = max(same_subject, key=lambda candidate: _authority_key(candidate, self._config), default=None)
        others = [existing for existing in records if existing is not incumbent]
        limit = self._config.ingestion.max_entity_attributes
        kept = others[: max(limit - (1 if incumbent is not None else 0), 0)]
        ordered = sorted(kept, key=lambda existing: (-self._attribute_cosine(existing, vector), existing.id))
        scan = ([incumbent] if incumbent is not None else []) + ordered
        return scan, len(records) > limit

    def _attribute_cosine(self, record: Record, vector: np.ndarray) -> float:
        existing_vector = self._vector_index.vector_for(record.id)
        return float(np.dot(vector, existing_vector)) if existing_vector is not None else float("-inf")

    def _supersession_outcome(
        self,
        record: Record,
        existing: Record,
        current_time: datetime,
        timer: Timer,
    ) -> tuple[str, Record, str | None]:
        if has_authority(record, existing, self._config):
            record.supersedes_id = existing.id
            timer.mark("supersession")
            return "superseded", existing, None
        if rank(record, self._config) == rank(existing, self._config) and record.event_at < existing.event_at:
            record.status = "superseded"
            timer.mark("supersession")
            return "superseded_on_arrival", existing, None
        record.status = "provisional"
        record.expires_at = provisional_expiry(current_time, self._config)
        timer.mark("supersession")
        return "conflict", existing, None

    def _ensure_index_loaded(self) -> None:
        if not self._vector_index.is_loaded:
            self._vector_index.load(self._store)

    def _nearby_records(
        self,
        record: Record,
        vector: np.ndarray,
        excluded: set[str],
        current_time: datetime,
    ) -> list[Record]:
        eligible = self._store.eligible_ids(
            [record.scope],
            [record.type],
            None,
            None,
            False,
            current_time,
        )
        eligible.difference_update(excluded)
        hits = self._vector_index.search(vector, eligible, 3)
        nearby_ids = [
            record_id for record_id, cosine in hits if cosine >= self._config.ingestion.dedup_candidate_cosine
        ]
        return self._store.get_records(nearby_ids)

    def _persist_outcome(
        self,
        record: Record,
        vector: np.ndarray,
        decision: _WriteDecision,
        note: str | None,
        entity_roles: dict[str, EntityRole],
        entailment_score: float | None,
        timer: Timer,
    ) -> _PersistedOutcome:
        outcome = decision.outcome
        existing = decision.existing
        dedup_kind = decision.dedup_kind
        if outcome == "reinforced":
            assert existing is not None
            counted = self._is_new_reinforcement(existing, record.source_ref)
            promoted = counted and rank(record, self._config) > rank(existing, self._config)
            reinforced = reinforce(existing, record.created_at, self._config) if counted else existing
            if promoted:
                reinforced.source_kind = record.source_kind
                reinforced.source_ref = record.source_ref
                reinforced.evidence = record.evidence
                reinforced.confidence = max(reinforced.confidence, initial_confidence(record.source_kind))
                reinforced.status = initial_status(record.source_kind)
                reinforced.expires_at = initial_expiry(record.source_kind, record.created_at, self._config)
            self._store.reinforce_fields(
                existing.id,
                confidence=reinforced.confidence,
                reinforcements=reinforced.reinforcements,
                last_reinforced_at=reinforced.last_reinforced_at,
                expires_at=reinforced.expires_at,
                status=reinforced.status,
                source_kind=record.source_kind if promoted else None,
                source_ref=record.source_ref if promoted else None,
                evidence=record.evidence if promoted else None,
            )
            timer.mark("persistence")
            result_outcome = "reinforced" if counted else "already_reinforced"
            result_note = _combine_notes(
                note,
                None if counted else "reinforcement source was already counted or unavailable",
            )
            payload = self._event_payload(
                record=reinforced,
                outcome=result_outcome,
                note=result_note,
                timer=timer,
                extra={
                    "dedup_kind": dedup_kind,
                    "evidence_entailment_score": entailment_score,
                    "provenance_promoted": promoted,
                    "reinforcement_counted": counted,
                    "reinforcing_evidence": record.evidence,
                    "reinforcing_source_kind": record.source_kind,
                    "reinforcing_source_ref": record.source_ref,
                    **decision.extra,
                },
            )
            self._store.append_event("record.reinforced", record.creator_agent_id, reinforced.id, None, payload)
            timer.mark("event_log")
            return _PersistedOutcome(
                reinforced.id,
                reinforced.status,
                result_outcome,
                result_note,
                None,
                None,
            )

        if outcome == "superseded":
            assert existing is not None
            self._store.update_status(existing.id, "superseded")
            for subsumed_id in cast(list[str], decision.extra.get("also_superseded", [])):
                self._store.update_status(subsumed_id, "superseded")
        self._store.insert_record(record)
        self._store.put_embedding(record.id, self._embedder.name, self._embedder.version, vector)
        self._store.upsert_fts(record.id, record.content, record.subject, aliases_text(self._store, record.entity_ids))
        for entity_id in record.entity_ids:
            self._store.link_record_entity(record.id, entity_id, entity_roles[entity_id])
        if outcome == "conflict":
            assert existing is not None
            self._store.add_conflict(record.id, existing.id, record.created_at)
        timer.mark("persistence")
        extra = {
            "evidence_entailment_score": entailment_score,
            **_outcome_event_fields(outcome, existing),
            **decision.extra,
        }
        payload = self._event_payload(record=record, outcome=outcome, note=note, timer=timer, extra=extra)
        self._store.append_event(_event_kind(outcome), record.creator_agent_id, record.id, None, payload)
        timer.mark("event_log")
        return _PersistedOutcome(
            record.id,
            record.status,
            _outcome_label(outcome, existing),
            note,
            record.id,
            vector,
        )

    def _is_new_reinforcement(self, existing: Record, source_ref: str | None) -> bool:
        if source_ref is None or source_ref == existing.source_ref:
            return False
        return all(
            event["payload"].get("source_ref") != source_ref
            and event["payload"].get("reinforcing_source_ref") != source_ref
            for event in self._store.events_for(existing.id)
        )

    def _record_ambiguity(self, principal: Principal, ambiguity: _EntityAmbiguous, timer: Timer) -> WriteResult:
        timings = timer.as_dict()
        payload = {
            "candidates": [self._entity_details(candidate) for candidate in ambiguity.candidates],
            "mentions": [
                {"kind": mention.kind, "role": mention.role, "text": mention.text} for mention in ambiguity.mentions
            ],
            "outcome": "entity_ambiguous",
            "timings_ms": timings,
            "timings_pending": [stage for stage in _WRITE_STAGES if stage not in timings],
        }
        self._store.append_event("entity.ambiguous_alias", principal.agent_id, None, None, payload)
        timer.mark("event_log")
        return self._result(
            None,
            None,
            "entity_ambiguous",
            None,
            timer,
            candidates=[self._ambiguity_candidate(candidate) for candidate in ambiguity.candidates],
        )

    def _entity_details(self, entity_id: str) -> dict[str, object]:
        entity = self._ambiguity_candidate(entity_id)
        return {
            "canonical": entity.canonical,
            "id": entity.id,
            "kind": entity.kind,
            "scope": {"id": entity.scope.id, "kind": entity.scope.kind},
        }

    def _ambiguity_candidate(self, entity_id: str) -> EntityAmbiguityCandidate:
        entity = self._store.get_entity(entity_id)
        if entity is None:
            raise ValueError(f"Ambiguous entity no longer exists: {entity_id}")
        return EntityAmbiguityCandidate(entity.id, entity.kind, entity.canonical, entity.scope)

    def _event_payload(
        self,
        *,
        record: Record,
        outcome: str,
        note: str | None,
        timer: Timer,
        extra: dict[str, object],
    ) -> dict[str, object]:
        timings = timer.as_dict()
        return {
            "evidence_note": note,
            "outcome": outcome,
            "scope": {"id": record.scope.id, "kind": record.scope.kind},
            "source_ref": record.source_ref,
            "source_kind": record.source_kind,
            "subject": record.subject,
            "subject_entity_id": record.subject_entity_id,
            "attribute": record.attribute,
            "timings_ms": timings,
            "timings_pending": [stage for stage in _WRITE_STAGES if stage not in timings],
            **extra,
        }

    def _result(
        self,
        record_id: str | None,
        status: RecordStatus | None,
        outcome: str,
        note: str | None,
        timer: Timer,
        *,
        candidates: list[EntityAmbiguityCandidate] | None = None,
    ) -> WriteResult:
        return WriteResult(record_id, status, outcome, note, self._complete_timings(timer), candidates or [])

    def _complete_timings(self, timer: Timer) -> dict[str, float]:
        marked = timer.as_dict()
        for stage in _WRITE_STAGES:
            if stage not in marked:
                timer.mark(stage)
        return timer.as_dict()

    def _timings(self, timer: Timer) -> dict[str, float]:
        return self._complete_timings(timer)


def _is_current_fact(record: Record) -> bool:
    return (
        record.type in ("semantic", "procedural")
        and record.subject_entity_id is not None
        and record.attribute not in (None, "-")
    )


def _authority_key(record: Record, config: MemoryWeaveConfig) -> tuple[int, datetime, datetime, str]:
    return rank(record, config), record.event_at, record.created_at, record.id


def _entity_roles(resolutions: list[Resolution]) -> dict[str, EntityRole]:
    roles: dict[str, EntityRole] = {}
    for resolution in resolutions:
        if resolution.entity is None:
            continue
        current = roles.get(resolution.entity.id)
        if current != "about":
            roles[resolution.entity.id] = resolution.mention.role
    return roles


def _combine_notes(*notes: str | None) -> str | None:
    resolved = [note for note in notes if note]
    return "; ".join(resolved) if resolved else None


def _ambiguous_mentions_note(resolutions: list[Resolution]) -> str | None:
    mention_ids = [
        ", ".join(resolution.candidates) for resolution in resolutions if resolution.mention.role == "mentions"
    ]
    if not mention_ids:
        return None
    return f"dropped ambiguous mentions: {'; '.join(mention_ids)}"


def _event_kind(outcome: str) -> str:
    return (
        "record.reinforced"
        if outcome == "reinforced"
        else "record.superseded"
        if outcome.startswith("superseded")
        else "record.created"
    )


def _outcome_event_fields(outcome: str, existing: Record | None) -> dict[str, object]:
    if existing is None:
        return {}
    if outcome == "superseded":
        return {"supersedes_id": existing.id}
    if outcome == "superseded_on_arrival":
        return {"superseded_on_arrival_by": existing.id}
    if outcome == "conflict":
        return {"conflicts_with": existing.id}
    return {}


def _outcome_label(outcome: str, existing: Record | None) -> str:
    if existing is None or outcome == "created":
        return outcome
    return f"{outcome}:{existing.id}"
