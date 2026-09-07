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

POLICY_VERSION = "activation-v1"

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
    if decision.applicability == "scoped":
        return "conditional", "scoped_preference"
    if decision.applicability == "ambiguous":
        return "review", "ambiguous_applicability"
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
