"""Session extraction: claim, extract, validate, review, revalidate, write, summarise, finish.

The order is the safety argument. A session is claimed atomically before any model call, so two workers
cannot both process it. The extractor and the reviewer run before anything is written, so a failure in
either writes nothing and leaves the claim to expire. Accepted candidates then go through the same
ingestor as an explicit ``memory_write``; the extractor gets no shortcut past evidence, entity, duplicate,
or supersession rules. Activation is decided afterwards by the activation service, which is a separate
decision from whether the candidate was worth keeping.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

from retold.config import RetoldConfig
from retold.models import CandidateRecord, EntityStatus, ExtractionContext, Principal, Record, Scope, Turn
from retold.policy import private_scope, readable_scopes, writable_scopes
from retold.policy.activation import ActivationService
from retold.store import SessionRow, Store
from retold.util import Timer, normalize_alias, now

from .entities import PrincipalEntityAmbiguousError, ensure_principal_entity, follow_merges
from .evidence import validate_evidence
from .extractor import ExtractionError, Extractor
from .ingestor import Ingestor, SummaryRequest, WriteRequest
from .reviewer import CandidateReviewer, ReviewError, ReviewRequest, adjacent_turns, check_revision
from .session import SessionBuffer
from .temporal import validate_temporal

EXTRACTION_ACTOR = "extractor"
ExtractionStatus = Literal["completed", "already_claimed", "already_extracted", "failed", "not_found"]
CandidateStage = Literal["validation", "review", "revalidation", "write"]

_ACTIVE_ENTITY_STATUSES: tuple[EntityStatus, ...] = ("provisional", "confirmed")
_NON_WRITE_OUTCOMES: frozenset[str] = frozenset(
    {"scope_not_writable", "entity_ambiguous", "invalid_source_kind", "invalid_subject"}
)


@dataclass(slots=True)
class CandidateOutcome:
    """What happened to one proposed candidate, and at which stage."""

    index: int
    content: str
    stage: CandidateStage
    outcome: str
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    record_id: str | None = None
    review_outcome: str | None = None
    review_reason: str | None = None
    activation: str | None = None
    activation_reason: str | None = None

    @property
    def written(self) -> bool:
        return self.stage == "write" and self.outcome not in _NON_WRITE_OUTCOMES

    def to_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "content": self.content,
            "stage": self.stage,
            "outcome": self.outcome,
            "reason": self.reason,
            "detail": self.detail,
            "record_id": self.record_id,
            "review_outcome": self.review_outcome,
            "review_reason": self.review_reason,
            "activation": self.activation,
            "activation_reason": self.activation_reason,
        }


@dataclass(slots=True)
class ExtractionResult:
    session_id: str
    status: ExtractionStatus
    reason: str | None = None
    candidates: list[CandidateOutcome] = field(default_factory=list)
    summary_record_id: str | None = None
    summary_outcome: str | None = None
    event_id: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def written_record_ids(self) -> list[str]:
        return [outcome.record_id for outcome in self.candidates if outcome.written and outcome.record_id]


@dataclass(frozen=True, slots=True)
class _Validated:
    candidate: CandidateRecord
    evidence_turn: Turn
    primary_entity_id: str | None


@dataclass(frozen=True, slots=True)
class _Rejected:
    reason: str
    detail: dict[str, Any]


class ExtractionRunner:
    """Run the asynchronous write path for one finished session."""

    def __init__(
        self,
        store: Store,
        ingestor: Ingestor,
        extractor: Extractor,
        reviewer: CandidateReviewer,
        session_buffer: SessionBuffer,
        config: RetoldConfig,
        *,
        activation: ActivationService | None = None,
        current_time: Callable[[], datetime] = now,
        actor: str = EXTRACTION_ACTOR,
    ) -> None:
        self._store = store
        self._ingestor = ingestor
        self._extractor = extractor
        self._reviewer = reviewer
        self._buffer = session_buffer
        self._config = config
        self._activation = activation
        self._current_time = current_time
        self._actor = actor

    def schedule(self, session_id: str, principal: Principal, *, force: bool = False) -> threading.Thread:
        """Run ``extract_session`` on a daemon thread; the host continues without waiting."""

        thread = threading.Thread(
            target=self.extract_session,
            args=(session_id, principal),
            kwargs={"force": force},
            name=f"retold-extract-{session_id}",
            daemon=True,
        )
        thread.start()
        return thread

    def extract_session(self, session_id: str, principal: Principal, *, force: bool = False) -> ExtractionResult:
        """LLD 8.2 steps 1 to 8. Refuses without an event when another worker holds the session."""

        timer = Timer(warm=True)
        current = self._current_time()
        session = self._store.get_session(session_id)
        if session is None:
            return ExtractionResult(session_id, "not_found", "session does not exist")
        stale_before = current - timedelta(minutes=self._config.ingestion.extraction_claim_timeout_minutes)
        claim = self._store.claim_extraction(session_id, current, stale_before, force=force)
        if claim == "not_found":
            return ExtractionResult(session_id, "not_found", "session does not exist")
        if claim in ("already_claimed", "already_extracted"):
            return ExtractionResult(session_id, claim)
        principal = _principal_for(session, principal)
        reclaimed = claim == "reclaimed"
        forced = force and session.extracted_at is not None
        if reclaimed:
            previous = session.extraction_started_at
            self._store.append_event(
                "extraction.reclaimed",
                self._actor,
                None,
                None,
                {
                    "session_id": session_id,
                    "previous_claim_at": previous.isoformat() if previous else None,
                    "claimed_at": current.isoformat(),
                },
            )
        if forced:
            self._store.append_event(
                "extraction.rerun",
                self._actor,
                None,
                None,
                {"session_id": session_id, "previous_extracted_at": session.extracted_at.isoformat()},  # type: ignore[union-attr]
            )
        try:
            return self._run(session, principal, timer, current, forced=forced, reclaimed=reclaimed)
        except ExtractionError as error:
            return self._fail(session_id, "extractor", error, timer)
        except ReviewError as error:
            return self._fail(session_id, "reviewer", error, timer)
        except Exception as error:
            self._fail(session_id, "write", error, timer)
            raise

    def _run(
        self,
        session: SessionRow,
        principal: Principal,
        timer: Timer,
        current: datetime,
        *,
        forced: bool,
        reclaimed: bool,
    ) -> ExtractionResult:
        session_id = session.id
        self._buffer.invalidate(session_id)
        turns = self._buffer.turns(session_id)
        timer.mark("transcript_prep")
        base_payload: dict[str, Any] = {
            "session_id": session_id,
            "prompt_version": self._extractor.prompt_version,
            "reviewer_version": self._reviewer.version,
            "forced": forced,
            "reclaimed": reclaimed,
        }
        if not turns:
            with self._store.transaction():
                self._store.mark_extracted(session_id, current)
                timings = _timings(timer)
                event_id = self._store.append_event(
                    "extraction.run",
                    self._actor,
                    None,
                    None,
                    {**base_payload, "empty_session": True, "counts": _counts([]), "timings_ms": timings},
                )
            return ExtractionResult(session_id, "completed", "empty_session", event_id=event_id, timings_ms=timings)

        scope = self._destination_scope(principal)
        context = self._context(principal, scope)
        try:
            output = self._extractor.extract(turns, context)
        except ExtractionError:
            raise
        except Exception as error:  # noqa: BLE001 - every extractor failure fails the run closed
            raise ExtractionError(f"{type(error).__name__}: {error}") from error
        timer.mark("extractor_model")

        short_session = len(turns) < 2
        proposed = list(output.candidates)
        considered = [] if short_session else proposed[: self._config.ingestion.extraction_max_candidates]

        outcomes: list[CandidateOutcome] = []
        validated: list[tuple[int, _Validated]] = []
        for index, candidate in enumerate(considered):
            checked = self._validate(candidate, principal, scope, turns, session_id)
            if isinstance(checked, _Rejected):
                outcomes.append(
                    CandidateOutcome(index, candidate.content, "validation", checked.reason, detail=checked.detail)
                )
                continue
            validated.append((index, checked))
        timer.mark("validation")

        # Every candidate is reviewed before any is written, so a reviewer failure writes nothing.
        accepted: list[tuple[int, _Validated, str, str]] = []
        for index, item in validated:
            request = ReviewRequest(
                candidate=item.candidate,
                scope=scope,
                evidence_turn=item.evidence_turn,
                adjacent_turns=adjacent_turns(turns, item.evidence_turn.turn),
                live_records=self._live_records(scope, item),
            )
            try:
                decision = self._reviewer.review(request)
            except ReviewError:
                raise
            except Exception as error:  # noqa: BLE001 - every reviewer failure fails the run closed
                raise ReviewError(f"{type(error).__name__}: {error}") from error
            if decision.outcome == "reject":
                outcomes.append(
                    CandidateOutcome(
                        index,
                        item.candidate.content,
                        "review",
                        "review_rejected",
                        decision.reason,
                        review_outcome="reject",
                        review_reason=decision.reason,
                    )
                )
                continue
            final = item
            if decision.outcome == "revise":
                violation = (
                    "revision missing" if decision.revised is None else check_revision(item.candidate, decision.revised)
                )
                if violation is not None or decision.revised is None:
                    outcomes.append(
                        CandidateOutcome(
                            index,
                            item.candidate.content,
                            "review",
                            "invalid_review_revision",
                            violation,
                            review_outcome="revise",
                            review_reason=decision.reason,
                        )
                    )
                    continue
                rechecked = self._validate(decision.revised, principal, scope, turns, session_id)
                if isinstance(rechecked, _Rejected):
                    outcomes.append(
                        CandidateOutcome(
                            index,
                            decision.revised.content,
                            "revalidation",
                            rechecked.reason,
                            detail=rechecked.detail,
                            review_outcome="revise",
                            review_reason=decision.reason,
                        )
                    )
                    continue
                final = rechecked
            accepted.append((index, final, decision.outcome, decision.reason))
        timer.mark("candidate_review")

        # One transaction for every write and the summary: a failure here rolls all of it back.
        write_timings: list[dict[str, float]] = []
        written: list[CandidateOutcome] = []
        with self._store.transaction():
            for index, item, review_outcome, review_reason in accepted:
                candidate = item.candidate
                write_request = WriteRequest(
                    type=candidate.type,
                    content=candidate.content,
                    source_kind=candidate.source_kind,
                    evidence=candidate.evidence,
                    attribute=candidate.attribute,
                    scope=scope,
                    event_at=candidate.event_at,
                    entities=list(candidate.entity_mentions),
                    evidence_turn=candidate.evidence_turn,
                    valid_from=candidate.valid_from,
                    valid_until=candidate.valid_until,
                    review_at=candidate.review_at,
                )
                result = self._ingestor.write(principal, write_request)
                write_timings.append(result.timings_ms)
                outcome = CandidateOutcome(
                    index,
                    candidate.content,
                    "write",
                    result.outcome,
                    result.note,
                    record_id=result.record_id,
                    review_outcome=review_outcome,
                    review_reason=review_reason,
                )
                if result.candidates:
                    outcome.detail["candidates"] = [entry.id for entry in result.candidates]
                outcomes.append(outcome)
                written.append(outcome)
            timer.mark("writes")

            ended_at = session.ended_at or turns[-1].at
            summary_text = _summary_text(output.summary.content, output.summary.decisions, self._config)
            summary = self._ingestor.write_session_summary(
                principal,
                SummaryRequest(scope, summary_text, session_id, ended_at, list(output.summary.entity_mentions)),
            )
            timer.mark("summary_write")
            self._store.mark_extracted(session_id, current)

            outcomes.sort(key=lambda entry: entry.index)
            timings = _timings(timer, write_timings)
            payload = {
                **base_payload,
                "empty_session": False,
                "short_session": short_session,
                "counts": _counts(outcomes, proposed=len(proposed), considered=len(considered)),
                "rejection_reasons": _rejection_reasons(outcomes),
                "candidates": [entry.to_payload() for entry in outcomes],
                "summary": {"record_id": summary.record_id, "outcome": summary.outcome, "note": summary.note},
                "timings_ms": timings,
            }
            event_id = self._store.append_event("extraction.run", self._actor, None, None, payload)

        # Activation is a separate decision from extraction and runs after the run has committed.
        if self._activation is not None:
            for outcome in written:
                if outcome.record_id is None or not outcome.written:
                    continue
                record = self._store.get_record(outcome.record_id)
                if record is None or record.type != "semantic":
                    continue
                activation_decision = self._activation.apply(principal, outcome.record_id)
                outcome.activation = activation_decision.outcome
                outcome.activation_reason = activation_decision.reason

        return ExtractionResult(
            session_id,
            "completed",
            candidates=outcomes,
            summary_record_id=summary.record_id,
            summary_outcome=summary.outcome,
            event_id=event_id,
            timings_ms=timings,
        )

    def _fail(self, session_id: str, stage: str, error: BaseException, timer: Timer) -> ExtractionResult:
        message = f"{type(error).__name__}: {error}"[:200]
        self._store.append_event(
            "extraction.failed",
            self._actor,
            None,
            None,
            {
                "session_id": session_id,
                "stage": stage,
                "error": message,
                "prompt_version": self._extractor.prompt_version,
                "reviewer_version": self._reviewer.version,
                "timings_ms": _timings(timer),
            },
        )
        return ExtractionResult(session_id, "failed", f"{stage}: {message}", timings_ms=_timings(timer))

    def _destination_scope(self, principal: Principal) -> Scope:
        user_scope = Scope(kind="user", id=principal.user_id)
        if user_scope in writable_scopes(self._store, principal.agent_id, principal.user_id):
            return user_scope
        return private_scope(principal.agent_id, principal.user_id)

    def _context(self, principal: Principal, scope: Scope) -> ExtractionContext:
        readable = readable_scopes(self._store, principal.agent_id, principal.user_id)
        ingestion = self._config.ingestion
        entities = self._store.entities_in_scopes(
            readable, statuses=_ACTIVE_ENTITY_STATUSES, limit=ingestion.extraction_context_max_entities
        )
        return ExtractionContext(
            principal=principal,
            known_entities=[(entity.id, entity.kind, entity.canonical) for entity in entities],
            existing_subjects=self._store.active_subjects(scope, limit=ingestion.extraction_context_max_subjects),
            prompt_version=self._extractor.prompt_version,
        )

    def _validate(
        self,
        candidate: CandidateRecord,
        principal: Principal,
        scope: Scope,
        turns: list[Turn],
        session_id: str,
    ) -> _Validated | _Rejected:
        """The immutable envelope: evidence location, temporal support, subject, and entity ambiguity."""

        check = validate_evidence(
            self._buffer,
            session_id,
            candidate.evidence,
            candidate.source_kind,
            self._config,
            turn_hint=candidate.evidence_turn,
        )
        if not check.found:
            reason = "evidence_too_short" if check.note == "evidence quote is too short" else "evidence_not_found"
            return _Rejected(reason, {"evidence": candidate.evidence, "evidence_turn": candidate.evidence_turn})
        evidence_turn = next(turn for turn in turns if turn.turn == check.turn)
        temporal = validate_temporal(candidate, evidence_turn)
        if temporal is not None:
            return _Rejected(temporal, {"evidence": candidate.evidence})
        current_fact = candidate.type in ("semantic", "procedural")
        if current_fact and not (candidate.attribute or "").strip():
            return _Rejected("invalid_subject", {"note": f"{candidate.type} records require an attribute"})

        readable = readable_scopes(self._store, principal.agent_id, principal.user_id)
        primary_id: str | None = None
        for mention in candidate.entity_mentions:
            if mention.entity_id is not None:
                if mention.role == "about":
                    primary_id = mention.entity_id
                continue
            matches = self._store.entities_by_alias(
                normalize_alias(mention.text),
                kinds=[mention.kind],
                scopes=readable,
                statuses=_ACTIVE_ENTITY_STATUSES,
            )
            if mention.role != "about":
                continue
            if len(matches) > 1:
                return _Rejected(
                    "entity_ambiguous",
                    {"mention": mention.text, "candidates": sorted(entity.id for entity in matches)},
                )
            if len(matches) == 1:
                primary_id = follow_merges(matches[0], self._store).id
        has_about = any(mention.role == "about" for mention in candidate.entity_mentions)
        if not has_about:
            in_user_scope = scope.kind == "user" and scope.id == principal.user_id
            if current_fact and not in_user_scope:
                return _Rejected("invalid_subject", {"note": "about entity required"})
            if in_user_scope:
                try:
                    primary_id = ensure_principal_entity(principal.user_id, self._store, actor=principal.agent_id).id
                except PrincipalEntityAmbiguousError as ambiguity:
                    return _Rejected(
                        "entity_ambiguous", {"mention": principal.user_id, "candidates": sorted(ambiguity.candidates)}
                    )
        return _Validated(candidate, evidence_turn, primary_id)

    def _live_records(self, scope: Scope, item: _Validated) -> list[Record]:
        if item.primary_entity_id is None:
            return []
        return self._store.active_for_entity(scope, item.primary_entity_id, item.candidate.type)


def _principal_for(session: SessionRow, principal: Principal) -> Principal:
    """The session row is trusted host state; the caller's principal must agree with it."""

    if principal.agent_id != session.agent_id or principal.user_id != session.user_id:
        raise ValueError("The principal does not match the session's agent and user.")
    if principal.session_id != session.id:
        return Principal(principal.agent_id, principal.user_id, session.id, principal.project_id)
    return principal


def _summary_text(content: str, decisions: list[str], config: RetoldConfig) -> str:
    text = content.strip()
    cleaned = [decision.strip() for decision in decisions if decision.strip()]
    if cleaned:
        text = f"{text}\nDecisions: " + " ".join(cleaned)
    limit = config.ingestion.summary_max_chars
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _timings(timer: Timer, writes: list[dict[str, float]] | None = None) -> dict[str, float]:
    timings = timer.as_dict()
    total = timings.pop("total")
    dedup_stages = ("dedup_search", "judge", "supersession")
    timings["dedup_and_contradiction"] = round(
        sum(write.get(stage, 0.0) for write in writes or [] for stage in dedup_stages), 3
    )
    timings["total"] = round(total, 3)
    return timings


def _counts(outcomes: list[CandidateOutcome], *, proposed: int = 0, considered: int = 0) -> dict[str, int]:
    def count(predicate: Callable[[CandidateOutcome], bool]) -> int:
        return sum(1 for outcome in outcomes if predicate(outcome))

    return {
        "proposed": proposed,
        "considered": considered,
        "review_accepted": count(lambda o: o.review_outcome == "accept"),
        "review_revised": count(lambda o: o.review_outcome == "revise" and o.stage == "write"),
        "review_rejected": count(lambda o: o.outcome == "review_rejected"),
        "written": count(lambda o: o.written),
        "created": count(lambda o: o.written and o.outcome == "created"),
        "reinforced": count(lambda o: o.written and o.outcome in ("reinforced", "already_reinforced")),
        "superseded": count(lambda o: o.written and o.outcome.startswith("superseded")),
        "conflicts": count(lambda o: o.written and o.outcome.startswith("conflict")),
        "rejected": count(lambda o: not o.written),
    }


def _rejection_reasons(outcomes: list[CandidateOutcome]) -> dict[str, int]:
    reasons: dict[str, int] = {}
    for outcome in outcomes:
        if outcome.written:
            continue
        reasons[outcome.outcome] = reasons.get(outcome.outcome, 0) + 1
    return reasons
