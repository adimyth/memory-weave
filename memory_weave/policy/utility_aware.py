"""Utility-aware host path: gap planning, draft-relative admission, and fail-closed orchestration.

The host decides whether stored information will improve an answer, not whether it concerns the query.
On each user turn the orchestrator generates a baseline draft from public context plus the ambient
profile while a gap policy names the user-specific facts the draft may be missing. Only when there are
gaps does it retrieve, and only when an admission policy says a candidate would change the draft does it
regenerate. Every other path returns the draft unchanged, and every path writes one turn-decision row.

Policies are protocols. Hosted adapters live outside the core package. The orchestrator takes callbacks
for baseline and final generation so this module never depends on a model vendor.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol

from memory_weave.models import Principal, Record
from memory_weave.policy.activation import ProfileAssembler, ProfileBlock, inventory
from memory_weave.store import Store
from memory_weave.util import now, uuid7

GapCategory = Literal["preference", "prior_decision", "constraint", "relationship_or_event", "task_state"]
GAP_CATEGORIES: tuple[GapCategory, ...] = ("preference", "prior_decision", "constraint", "relationship_or_event", "task_state")
AdmissionVerdict = Literal["helpful", "redundant", "insufficient", "stale_or_conflicting", "potentially_harmful", "jointly_helpful"]
ADMITTING_VERDICTS: tuple[AdmissionVerdict, ...] = ("helpful", "jointly_helpful")
PolicyStatus = Literal["ok", "empty", "failed", "timeout", "skipped"]
Disposition = Literal[
    "baseline_no_gaps",
    "baseline_no_candidates",
    "baseline_empty_admission",
    "baseline_policy_failure",
    "baseline_budget_exhausted",
    "admitted_not_applied",
    "regenerated",
    "shadow_would_regenerate",
]


@dataclass(frozen=True, slots=True)
class Gap:
    category: GapCategory
    query: str


@dataclass(frozen=True, slots=True)
class GapDecision:
    gaps: list[Gap]
    policy_id: str
    status: PolicyStatus
    elapsed_ms: float = 0.0
    error: str | None = None

    @staticmethod
    def failed(policy_id: str, error: str, elapsed_ms: float = 0.0) -> GapDecision:
        return GapDecision([], policy_id, "failed", elapsed_ms, error)


class GapPolicy(Protocol):
    def plan(self, turn: str, public_context: str | None, ambient_profile: ProfileBlock, inventory: Sequence[str]) -> GapDecision: ...


@dataclass(frozen=True, slots=True)
class CandidateVerdict:
    record_id: str
    verdict: AdmissionVerdict
    reason: str


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted_ids: list[str]
    verdicts: list[CandidateVerdict]
    policy_id: str
    status: PolicyStatus
    elapsed_ms: float = 0.0
    error: str | None = None

    @staticmethod
    def failed(policy_id: str, error: str, elapsed_ms: float = 0.0) -> AdmissionDecision:
        return AdmissionDecision([], [], policy_id, "failed", elapsed_ms, error)


class AdmissionPolicy(Protocol):
    def admit(
        self,
        turn: str,
        public_context: str | None,
        ambient_profile: ProfileBlock,
        draft: str,
        candidates: Sequence[Record],
    ) -> AdmissionDecision: ...


Retrieve = Callable[[Principal, list[str], str], list[Record]]
Generate = Callable[[], str]
Regenerate = Callable[[list[Record]], str]


@dataclass(frozen=True, slots=True)
class TurnOptions:
    latency_budget_ms: int | None = None  # None: configured default; 0: baseline plus ambient profile only


@dataclass(frozen=True, slots=True)
class UtilityAwareConfig:
    gap_enabled: bool = False
    admission_mode: Literal["disabled", "hosted_judge"] = "disabled"
    shadow: bool = False
    max_gaps: int = 3
    max_candidates: int = 8
    gap_timeout_ms: int = 2000
    admission_timeout_ms: int = 2000
    latency_budget_ms: int | None = None


@dataclass(slots=True)
class TurnMemoryDecision:
    decision_id: str
    session_id: str | None
    turn: str
    disposition: Disposition
    profile_record_ids: list[str]
    inventory: list[str]
    gaps: list[Gap]
    gap_status: PolicyStatus
    candidate_ids: list[str]
    verdicts: list[CandidateVerdict]
    admitted_ids: list[str]
    admission_status: PolicyStatus
    draft: str
    final: str | None
    requested_budget_ms: int | None
    effective_budget_ms: int | None
    timings_ms: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    shadow: bool = False

    @property
    def response(self) -> str:
        return self.final if self.final is not None else self.draft


class UtilityAwareOrchestrator:
    """Deterministic orchestration of one turn. Fails closed to the draft on every error and budget path."""

    def __init__(
        self,
        store: Store,
        retrieve: Retrieve,
        gap_policy: GapPolicy | None,
        admission_policy: AdmissionPolicy | None,
        config: UtilityAwareConfig,
        *,
        profile_assembler: ProfileAssembler | None = None,
        actor: str = "host",
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._store = store
        self._retrieve = retrieve
        self._gap_policy = gap_policy
        self._admission_policy = admission_policy
        self._config = config
        self._profiles = profile_assembler or ProfileAssembler(store)
        self._actor = actor
        self._clock = clock

    def prepare_turn(
        self,
        principal: Principal,
        turn: str,
        public_context: str | None,
        baseline: Generate,
        regenerate: Regenerate,
        options: TurnOptions = TurnOptions(),
    ) -> TurnMemoryDecision:
        timings: dict[str, float] = {}
        failures: list[str] = []
        profile = self._profiles.build(principal)
        labels = inventory(self._store, principal)
        requested = options.latency_budget_ms
        effective = requested if requested is not None else self._config.latency_budget_ms

        decision = TurnMemoryDecision(
            decision_id=uuid7(),
            session_id=principal.session_id,
            turn=turn,
            disposition="baseline_no_gaps",
            profile_record_ids=list(profile.record_ids),
            inventory=labels,
            gaps=[],
            gap_status="skipped",
            candidate_ids=[],
            verdicts=[],
            admitted_ids=[],
            admission_status="skipped",
            draft="",
            final=None,
            requested_budget_ms=requested,
            effective_budget_ms=effective,
            timings_ms=timings,
            failures=failures,
            shadow=self._config.shadow,
        )

        path_enabled = self._config.gap_enabled and self._gap_policy is not None and effective != 0
        started = self._clock()
        if not path_enabled:
            decision.draft = baseline()
            timings["draft"] = (self._clock() - started) * 1000
            decision.disposition = "baseline_budget_exhausted" if effective == 0 and self._config.gap_enabled else "baseline_no_gaps"
            return self._persist(decision)

        # Baseline and gap planning run concurrently; the gap call is charged only for what outlasts the draft.
        gap_timeout = self._config.gap_timeout_ms if effective is None else min(self._config.gap_timeout_ms, effective)
        with ThreadPoolExecutor(max_workers=2) as pool:
            draft_future = pool.submit(baseline)
            gap_future = pool.submit(self._plan, turn, public_context, profile, labels)
            decision.draft = draft_future.result()
            draft_done = self._clock()
            timings["draft"] = (draft_done - started) * 1000
            budget_bound = effective is not None and effective <= self._config.gap_timeout_ms
            try:
                gap_decision = gap_future.result(timeout=max(0.0, gap_timeout / 1000))
            except FuturesTimeout:
                gap_decision = GapDecision([], "gap", "timeout", gap_timeout, "budget_exhausted" if budget_bound else "gap_timeout")
            except Exception as error:  # noqa: BLE001
                gap_decision = GapDecision.failed("gap", type(error).__name__)
            timings["gap_overhang"] = max(0.0, (self._clock() - draft_done) * 1000)

        decision.gap_status = gap_decision.status
        decision.gaps = list(gap_decision.gaps)[: self._config.max_gaps]
        if gap_decision.error:
            failures.append(f"gap:{gap_decision.error}")
        if gap_decision.status == "timeout" and gap_decision.error == "budget_exhausted":
            decision.disposition = "baseline_budget_exhausted"
            return self._persist(decision)
        if gap_decision.status in ("failed", "timeout"):
            decision.disposition = "baseline_policy_failure"
            return self._persist(decision)
        if not decision.gaps:
            decision.disposition = "baseline_no_gaps"
            return self._persist(decision)

        def remaining_ms() -> float | None:
            if effective is None:
                return None
            return effective - (self._clock() - draft_done) * 1000

        if (left := remaining_ms()) is not None and left <= 0:
            decision.disposition = "baseline_budget_exhausted"
            return self._persist(decision)

        retrieval_started = self._clock()
        try:
            candidates = self._retrieve(principal, [gap.query for gap in decision.gaps], turn)[: self._config.max_candidates]
        except Exception as error:  # noqa: BLE001
            failures.append(f"retrieval:{type(error).__name__}")
            candidates = []
        timings["retrieval"] = (self._clock() - retrieval_started) * 1000
        candidates = [record for record in candidates if record.activation == "conditional"]
        decision.candidate_ids = [record.id for record in candidates]
        if not candidates:
            decision.disposition = "baseline_no_candidates"
            return self._persist(decision)

        if self._config.admission_mode == "disabled" or self._admission_policy is None:
            decision.admission_status = "skipped"
            decision.disposition = "baseline_empty_admission"
            return self._persist(decision)
        if (left := remaining_ms()) is not None and left <= 0:
            decision.disposition = "baseline_budget_exhausted"
            return self._persist(decision)

        admission_timeout = self._config.admission_timeout_ms if (left := remaining_ms()) is None else min(self._config.admission_timeout_ms, left)
        admission_started = self._clock()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._admit, turn, public_context, profile, decision.draft, candidates)
            try:
                admission = future.result(timeout=max(0.0, admission_timeout / 1000))
            except Exception as error:  # noqa: BLE001
                admission = AdmissionDecision.failed("admission", type(error).__name__)
        timings["admission"] = (self._clock() - admission_started) * 1000
        decision.admission_status = admission.status
        decision.verdicts = list(admission.verdicts)
        if admission.error:
            failures.append(f"admission:{admission.error}")
        if admission.status in ("failed", "timeout"):
            decision.disposition = "baseline_policy_failure"
            return self._persist(decision)

        known = {record.id: record for record in candidates}
        admitting = {v.record_id for v in admission.verdicts if v.verdict in ADMITTING_VERDICTS}
        decision.admitted_ids = [i for i in admission.admitted_ids if i in known and i in admitting]
        if not decision.admitted_ids:
            decision.disposition = "baseline_empty_admission"
            return self._persist(decision)

        if self._config.shadow:
            decision.disposition = "shadow_would_regenerate"
            return self._persist(decision)

        # The best available estimate of a second generation is this turn's own draft latency.
        if (left := remaining_ms()) is not None and left < timings["draft"]:
            decision.disposition = "admitted_not_applied"
            return self._persist(decision)

        regeneration_started = self._clock()
        try:
            decision.final = regenerate([known[i] for i in decision.admitted_ids])
            decision.disposition = "regenerated"
        except Exception as error:  # noqa: BLE001
            failures.append(f"regeneration:{type(error).__name__}")
            decision.final = None
            decision.disposition = "baseline_policy_failure"
        timings["regeneration"] = (self._clock() - regeneration_started) * 1000
        return self._persist(decision)

    # -- internals ----------------------------------------------------------------------------------------

    def _plan(self, turn: str, public_context: str | None, profile: ProfileBlock, labels: list[str]) -> GapDecision:
        assert self._gap_policy is not None
        started = self._clock()
        try:
            result = self._gap_policy.plan(turn, public_context, profile, labels)
        except Exception as error:  # noqa: BLE001
            return GapDecision.failed("gap", type(error).__name__, (self._clock() - started) * 1000)
        gaps = [g for g in result.gaps if g.query.strip() and g.category in GAP_CATEGORIES]
        seen: set[str] = set()
        unique = []
        for gap in gaps:
            key = gap.query.strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(gap)
        status: PolicyStatus = result.status if result.status in ("failed", "timeout") else ("ok" if unique else "empty")
        return GapDecision(unique, result.policy_id, status, (self._clock() - started) * 1000, result.error)

    def _admit(self, turn: str, public_context: str | None, profile: ProfileBlock, draft: str, candidates: list[Record]) -> AdmissionDecision:
        assert self._admission_policy is not None
        started = self._clock()
        try:
            result = self._admission_policy.admit(turn, public_context, profile, draft, candidates)
        except Exception as error:  # noqa: BLE001
            return AdmissionDecision.failed("admission", type(error).__name__, (self._clock() - started) * 1000)
        ids = {record.id for record in candidates}
        verdicts = [v for v in result.verdicts if v.record_id in ids]
        if len({v.record_id for v in verdicts}) != len(ids):
            return AdmissionDecision.failed(result.policy_id, "incomplete_verdicts", (self._clock() - started) * 1000)
        return AdmissionDecision(list(result.admitted_ids), verdicts, result.policy_id, result.status or "ok", (self._clock() - started) * 1000, result.error)

    def _persist(self, decision: TurnMemoryDecision) -> TurnMemoryDecision:
        row = {
            "id": decision.decision_id,
            "at": now(),
            "session_id": decision.session_id,
            "turn": decision.turn,
            "disposition": decision.disposition,
            "shadow": decision.shadow,
            "profile_record_ids": decision.profile_record_ids,
            "inventory": decision.inventory,
            "gaps": [asdict(g) for g in decision.gaps],
            "gap_status": decision.gap_status,
            "candidate_ids": decision.candidate_ids,
            "verdicts": [asdict(v) for v in decision.verdicts],
            "admitted_ids": decision.admitted_ids,
            "admission_status": decision.admission_status,
            "requested_budget_ms": decision.requested_budget_ms,
            "effective_budget_ms": decision.effective_budget_ms,
            "timings_ms": decision.timings_ms,
            "failures": decision.failures,
            "config": asdict(self._config),
        }
        self._store.insert_turn_decision(row)
        self._store.append_event(
            "turn.decided",
            self._actor,
            None,
            None,
            {"decision_id": decision.decision_id, "disposition": decision.disposition, "shadow": decision.shadow, "admitted": len(decision.admitted_ids)},
        )
        return decision


def render_decision(decision: TurnMemoryDecision) -> str:
    """Compact one-line rendering for logs and tests."""

    return json.dumps(
        {"disposition": decision.disposition, "gaps": len(decision.gaps), "candidates": len(decision.candidate_ids), "admitted": decision.admitted_ids},
        sort_keys=True,
    )
