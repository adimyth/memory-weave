"""Ambient activation: which records sit in the always-present profile, decided by rule and audited.

A record is `conditional` by default and must pass retrieval and admission on each turn. A small class of
broadly applicable preferences about the principal, such as answer style or the language of code examples,
is promoted to `ambient` and rendered into a bounded profile at session start. Promotion is never a model's
decision alone: a category policy proposes a classification, the host verifies the supporting claim in the
principal's own transcript turns, and deterministic rules decide. Anything ambiguous, low-confidence, or
flagged unsafe goes to a durable review queue and stays conditional until a trusted reviewer resolves it.

Every record also receives a content-free retrieval category from a fixed taxonomy. The distinct categories
present in a principal's conditional store form the inventory a gap planner may see; record text never does.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from memory_weave.models import Entity, Principal, Record, Scope
from memory_weave.policy.grants import readable_scopes
from memory_weave.store import Store
from memory_weave.util import normalize_ws


def _principal_entity(store: Store, principal: Principal, actor: str) -> Entity:
    # Imported here because memory_weave.ingest.entities imports this package's grant rules.
    from memory_weave.ingest.entities import ensure_principal_entity

    return ensure_principal_entity(principal.user_id, store, actor=actor)

POLICY_VERSION = "activation-v2-frozen"

# Deterministic form recognition. A recognised form decides applicability by rule; the classifier's
# applicability is used only for sentences no rule recognises. Two independent classifier runs on the
# fifth scenario split labelled the same global-default sentence broad and scoped, so the classifier
# cannot be the deciding input for these forms.
_OVERRIDE_CLAUSE = re.compile(
    r"\b(unless (?:i|the user|they|you)?\s*(?:ask|asks|request|requests|say|says|tell|tells|specify|specifies|state|states)"
    r"|unless (?:another|a different|otherwise|asked|requested|told|specified|stated)"
    r"|by default|default to|as a default)\b",
    re.IGNORECASE,
)
_SCOPE_PREFIX = re.compile(
    r"^\s*(?:(?:the )?user (?:prefers|wants|likes|asks)[^.]*?\b)?(?:when|whenever|while|during|for|in|on|if)\b\s+"
    r"(?:writing|reviewing|discussing|working|answering|doing|debugging|dealing|talking|responding|handling|"
    r"drafting|editing|planning|designing|building|testing|reading|explaining|"
    r"[a-z][a-z0-9-]*\s+(?:questions|topics|tasks|code|work|reviews|meetings|discussions|threads|requests))",
    re.IGNORECASE,
)
_TEMPORARY = re.compile(
    r"\b(until|through|for the next|for the coming|for the rest of|this week|this month|this sprint|next week|"
    r"today only|for now|temporarily|for the time being|during (?:the )?(?:next|coming|current))\b",
    re.IGNORECASE,
)

FormKind = Literal["global_default", "scoped", "temporary"]


def recognise_form(content: str) -> FormKind | None:
    """Return the preference form a deterministic rule recognises in the sentence, or None."""

    text = normalize_ws(content)
    if _TEMPORARY.search(text):
        return "temporary"
    if _SCOPE_PREFIX.search(text):
        return "scoped"
    if _OVERRIDE_CLAUSE.search(text):
        return "global_default"
    return None

RETRIEVAL_CATEGORIES: dict[str, str] = {
    "time_zone": "time zone and working hours",
    "people": "people and their roles",
    "infrastructure": "infrastructure and environments",
    "decisions": "tooling and architecture decisions",
    "constraints": "runtime and version constraints",
    "limits": "limits and quotas",
    "schedules": "schedules and recurring events",
    "locations": "document and repository locations",
    "personal": "personal life",
    "preferences": "response preferences",
    "other": "other facts",
}

ActivationCategory = Literal["answer_style", "response_language", "accessibility", "code_example_language", "other"]
PROMOTABLE_CATEGORIES: tuple[ActivationCategory, ...] = (
    "answer_style",
    "response_language",
    "accessibility",
    "code_example_language",
)
Applicability = Literal["broad", "scoped", "ambiguous"]
ActivationOutcome = Literal["promote", "conditional", "review"]


@dataclass(frozen=True, slots=True)
class CategoryDecision:
    """What a category policy proposes for one record. It never decides activation on its own."""

    retrieval_category: str
    activation_category: ActivationCategory
    applicability: Applicability
    confidence: float
    unsafe: bool = False
    rationale: str = ""


class CategoryPolicy(Protocol):
    def classify(self, content: str) -> CategoryDecision: ...


@dataclass(frozen=True, slots=True)
class ActivationDecision:
    record_id: str
    outcome: ActivationOutcome
    reason: str
    retrieval_category: str | None
    activation_category: str | None
    evidence_turn: int | None
    confidence: float | None
    review_id: str | None = None


def verify_principal_evidence(store: Store, principal: Principal, record: Record) -> int | None:
    """Return the number of the principal's own user turn that contains the record's claim, if any.

    This is the host's verification, independent of any quote the writing model supplied. Assistant and
    tool turns never count: a preference the assistant proposed is not a preference the user stated.
    """

    if principal.session_id is None:
        return None
    needles = [normalize_ws(text).lower() for text in (record.evidence, record.content) if text]
    needles = [needle for needle in needles if len(needle) >= 12]
    if not needles:
        return None
    for turn in store.session_turns(principal.session_id):
        if turn.role != "user":
            continue
        haystack = normalize_ws(turn.content).lower()
        if any(needle in haystack for needle in needles):
            return turn.turn
    return None


def decide_activation(
    record: Record,
    decision: CategoryDecision | None,
    principal_entity_id: str,
    evidence_turn: int | None,
    *,
    min_confidence: float = 0.7,
) -> tuple[ActivationOutcome, str]:
    """Deterministic activation rule. The order matters: eligibility first, safety next, classification last."""

    if record.type != "semantic":
        return "conditional", "not_semantic"
    if record.subject_entity_id != principal_entity_id:
        return "conditional", "not_about_principal"
    if record.status not in ("provisional", "confirmed"):
        return "conditional", f"status_{record.status}"
    if decision is None:
        return "review", "policy_unavailable"
    if decision.unsafe:
        return "review", "flagged_unsafe"
    if evidence_turn is None:
        return "conditional", "no_host_verified_evidence"
    if decision.activation_category not in PROMOTABLE_CATEGORIES:
        return "conditional", "category_not_promotable"
    # Recognised forms are decided by rule. The classifier's applicability is consulted only below.
    form = recognise_form(record.content)
    if form == "temporary":
        return "conditional", "temporary_preference"
    if form == "scoped":
        return "conditional", "scoped_preference"
    if form == "global_default":
        return "promote", "eligible_global_default"
    if decision.applicability == "ambiguous":
        return "review", "ambiguous_applicability"
    if decision.applicability == "scoped":
        return "conditional", "scoped_preference"
    # Self-reported confidence never promotes; it can only send an unrecognised form to review.
    if decision.confidence < min_confidence:
        return "review", "low_confidence"
    return "promote", "eligible_broad_preference"


class ActivationService:
    """Run the activation policy for one committed record and persist every step of the decision."""

    def __init__(self, store: Store, policy: CategoryPolicy | None, *, min_confidence: float = 0.7, actor: str = "host") -> None:
        self._store = store
        self._policy = policy
        self._min_confidence = min_confidence
        self._actor = actor

    def apply(self, principal: Principal, record_id: str) -> ActivationDecision:
        record = self._store.get_record(record_id)
        if record is None:
            raise ValueError(f"Record {record_id} was not found.")
        principal_entity = _principal_entity(self._store, principal, self._actor).id

        decision: CategoryDecision | None = None
        policy_error: str | None = None
        if self._policy is not None:
            try:
                decision = self._policy.classify(record.content)
            except Exception as error:  # noqa: BLE001
                policy_error = type(error).__name__

        evidence_turn = verify_principal_evidence(self._store, principal, record)

        with self._store.transaction():
            if decision is not None:
                self._store.set_category(record.id, decision.retrieval_category)
                self._store.append_event(
                    "record.category_assigned",
                    self._actor,
                    record.id,
                    None,
                    {
                        "retrieval_category": decision.retrieval_category,
                        "activation_category": decision.activation_category,
                        "applicability": decision.applicability,
                        "confidence": decision.confidence,
                        "unsafe": decision.unsafe,
                        "policy_version": POLICY_VERSION,
                    },
                )
            if evidence_turn is not None:
                self._store.append_event(
                    "record.activation_evidence_verified",
                    self._actor,
                    record.id,
                    None,
                    {"session_id": principal.session_id, "turn": evidence_turn},
                )
                if record.status == "provisional" and record.subject_entity_id == principal_entity:
                    self._store.update_status(record.id, "confirmed")
                    record.status = "confirmed"

            if policy_error is not None:
                outcome: ActivationOutcome = "review"
                reason = f"policy_error:{policy_error}"
            else:
                outcome, reason = decide_activation(
                    record, decision, principal_entity, evidence_turn, min_confidence=self._min_confidence
                )

            self._store.append_event(
                "record.activation_decided",
                self._actor,
                record.id,
                None,
                {"outcome": outcome, "reason": reason, "evidence_turn": evidence_turn, "policy_version": POLICY_VERSION},
            )
            review_id: str | None = None
            if outcome == "promote":
                self._store.set_activation(record.id, "ambient")
                self._store.append_event(
                    "record.activation_changed",
                    self._actor,
                    record.id,
                    None,
                    {"from": "conditional", "to": "ambient", "reason": reason, "policy_version": POLICY_VERSION},
                )
            elif outcome == "review":
                review_id = self._store.insert_activation_review(
                    record_id=record.id,
                    evidence_turn=evidence_turn,
                    retrieval_category=decision.retrieval_category if decision else None,
                    activation_category=decision.activation_category if decision else None,
                    proposed="promote" if decision and not decision.unsafe else "conditional",
                    reason=reason,
                    confidence=decision.confidence if decision else None,
                    policy_version=POLICY_VERSION,
                )

        return ActivationDecision(
            record_id=record.id,
            outcome=outcome,
            reason=reason,
            retrieval_category=decision.retrieval_category if decision else None,
            activation_category=decision.activation_category if decision else None,
            evidence_turn=evidence_turn,
            confidence=decision.confidence if decision else None,
            review_id=review_id,
        )


@dataclass(frozen=True, slots=True)
class ProfileBlock:
    text: str
    record_ids: list[str]
    truncated: bool


class ProfileAssembler:
    """Render the principal's ambient records into one bounded block, built once per session."""

    def __init__(self, store: Store, *, max_records: int = 8, token_budget: int = 400, actor: str = "host") -> None:
        self._store = store
        self._max_records = max_records
        self._token_budget = token_budget
        self._actor = actor

    def build(self, principal: Principal) -> ProfileBlock:
        entity = _principal_entity(self._store, principal, self._actor).id
        records = self._store.ambient_records(Scope(kind="user", id=principal.user_id), entity)
        chosen: list[Record] = []
        spent = 0
        truncated = False
        for record in records:
            cost = max(1, len(record.content) // 4)
            if len(chosen) >= self._max_records or spent + cost > self._token_budget:
                truncated = True
                break
            chosen.append(record)
            spent += cost
        if not chosen:
            return ProfileBlock("", [], truncated)
        text = "Preferences that apply to every reply:\n" + "\n".join(f"- {record.content}" for record in chosen)
        return ProfileBlock(text, [record.id for record in chosen], truncated)


def inventory(store: Store, principal: Principal) -> list[str]:
    """Content-free labels of the fact categories present in the principal's readable conditional store."""

    scopes: Sequence[Scope] = readable_scopes(store, principal.agent_id, principal.user_id)
    return [RETRIEVAL_CATEGORIES.get(key, key) for key in store.active_categories(scopes)]


# -- Phase 1B: trusted review and activation operations ---------------------------------------------------

ReviewResolution = Literal["promote", "conditional", "reject"]


@dataclass(frozen=True, slots=True)
class BacklogStatus:
    open_count: int
    oldest_age_days: float
    over_count_limit: bool
    over_age_limit: bool

    @property
    def within_limits(self) -> bool:
        return not (self.over_count_limit or self.over_age_limit)


class ActivationOperations:
    """Trusted host operations: resolve reviews, change activation directly, and check the backlog."""

    def __init__(self, store: Store, *, actor: str = "host") -> None:
        self._store = store
        self._actor = actor

    def resolve_review(self, review_id: str, resolution: ReviewResolution, resolver: str, note: str = "") -> str:
        """Resolve one open review. Only a promote resolution changes activation, and only after the same
        eligibility checks a rule-based promotion needs."""

        row = self._store.get_activation_review(review_id)
        if row is None:
            raise ValueError(f"Review {review_id} was not found.")
        if row["status"] != "open":
            raise ValueError(f"Review {review_id} is already {row['status']}.")
        record_id = str(row["record_id"])
        if resolution == "promote":
            self._change_activation(record_id, "ambient", reason=f"review:{review_id}", resolver=resolver)
        self._store.resolve_activation_review(review_id, resolution=resolution, resolver=resolver)
        self._store.append_event(
            "record.activation_reviewed",
            self._actor,
            record_id,
            None,
            {"review_id": review_id, "resolution": resolution, "resolver": resolver, "note": note, "policy_version": POLICY_VERSION},
        )
        return record_id

    def set_activation(self, record_id: str, activation: Literal["ambient", "conditional"], *, resolver: str, reason: str) -> None:
        """Direct trusted change, subject to the eligibility checks that apply to any promotion."""

        self._change_activation(record_id, activation, reason=reason, resolver=resolver)

    def backlog(self, *, max_open: int, max_age_days: float, at: object = None) -> BacklogStatus:
        from datetime import datetime

        from memory_weave.util import now as _now

        rows = self._store.open_activation_reviews()
        current = at if isinstance(at, datetime) else _now()
        ages = []
        for row in rows:
            created = datetime.fromisoformat(str(row["created_at"]))
            ages.append((current - created).total_seconds() / 86400)
        oldest = max(ages) if ages else 0.0
        return BacklogStatus(len(rows), oldest, len(rows) > max_open, oldest > max_age_days)

    def _change_activation(self, record_id: str, activation: str, *, reason: str, resolver: str) -> None:
        record = self._store.get_record(record_id)
        if record is None:
            raise ValueError(f"Record {record_id} was not found.")
        if activation == "ambient":
            if record.type != "semantic":
                raise ValueError("Only semantic records can be ambient.")
            if record.status not in ("provisional", "confirmed"):
                raise ValueError(f"A {record.status} record cannot be ambient.")
            if record.subject_entity_id is None:
                raise ValueError("Only records about a principal can be ambient.")
        if record.activation == activation:
            return
        with self._store.transaction():
            self._store.set_activation(record_id, activation)
            self._store.append_event(
                "record.activation_changed",
                self._actor,
                record_id,
                None,
                {"from": record.activation, "to": activation, "reason": reason, "resolver": resolver, "policy_version": POLICY_VERSION},
            )
