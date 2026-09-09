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
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, cast

from retold.log import get_logger
from retold.models import Principal, Record
from retold.policy.activation import ProfileAssembler, ProfileBlock, inventory
from retold.store import Store
from retold.util import now, uuid7

GapCategory = Literal["preference", "prior_decision", "constraint", "relationship_or_event", "task_state"]
GAP_CATEGORIES: tuple[GapCategory, ...] = (
    "preference",
    "prior_decision",
    "constraint",
    "relationship_or_event",
    "task_state",
)
AdmissionVerdict = Literal[
    "helpful", "redundant", "insufficient", "stale_or_conflicting", "potentially_harmful", "jointly_helpful"
]
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
    usage: dict[str, dict[str, int]] = field(default_factory=dict)  # {model: {"prompt": n, "completion": n}}

    @staticmethod
    def failed(policy_id: str, error: str, elapsed_ms: float = 0.0) -> GapDecision:
        return GapDecision([], policy_id, "failed", elapsed_ms, error)


class GapPolicy(Protocol):
    def plan(
        self, turn: str, public_context: str | None, ambient_profile: ProfileBlock, inventory: Sequence[str]
    ) -> GapDecision: ...


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
    usage: dict[str, dict[str, int]] = field(default_factory=dict)

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
    # False renders no ambient profile: the draft sees public context only and the decision records no
    # profile ids. A rollout switch like ``shadow``, recorded on every decision and not part of the bundle.
    profile_enabled: bool = True
    max_gaps: int = 3
    max_candidates: int = 8
    gap_timeout_ms: int = 2000
    admission_timeout_ms: int = 2000
    latency_budget_ms: int | None = None
    # The complete policy bundle that produced a decision: planner model and prompt version, judge model and
    # prompt version, taxonomy version, inventory-builder version, retrieval configuration hash, budget. A
    # change to any component is a new bundle and needs a fitness-test rerun before it serves.
    bundle: dict[str, object] = field(default_factory=dict)


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
    bundle_hash: str | None = None
    usage: dict[str, dict[str, int]] = field(default_factory=dict)
    # False means the bundle's retrieval hash is a claim this host could not check, because no retrieval
    # configuration was given. Serving with an unverifiable claim is refused; shadow records it here.
    bundle_retrieval_verified: bool = False

    @property
    def response(self) -> str:
        return self.final if self.final is not None else self.draft


def bundle_components(config: UtilityAwareConfig, retrieval_config: Any = None) -> dict[str, object]:
    """Everything that changes what the path does. Hashing this identifies a bundle.

    Pass the ``RetoldConfig`` the host retrieves with and the retrieval half of the hash is derived from it
    rather than taken on trust. A manifest that declares a different ``retrieval_config_sha256`` is refused:
    a bundle's fitness result was earned under one retrieval configuration, and a hash copied from that run
    into a host that retrieves differently would claim an approval nothing measured.
    """

    from retold.policy.bundles import BundleMismatchError, retrieval_config_hash

    declared = dict(config.bundle)
    if retrieval_config is not None:
        live = retrieval_config_hash(retrieval_config)
        stated = declared.get("retrieval_config_sha256")
        if stated is not None and stated != live:
            raise BundleMismatchError(
                f"The bundle declares retrieval_config_sha256 {stated!r} but this host retrieves with "
                f"{live!r}. Record a fitness result for the configuration that will serve, or serve the "
                "configuration the recorded result measured; refer retold.policy.reference."
            )
        declared["retrieval_config_sha256"] = live

    # Shadow versus serving is a mode, not a component: a fitness result earned in shadow approves the
    # same bundle for serving. Everything else that changes behaviour is part of the hash.
    return {
        **declared,
        "gap_enabled": config.gap_enabled,
        "admission_mode": config.admission_mode,
        "max_gaps": config.max_gaps,
        "max_candidates": config.max_candidates,
        "gap_timeout_ms": config.gap_timeout_ms,
        "admission_timeout_ms": config.admission_timeout_ms,
        "latency_budget_ms": config.latency_budget_ms,
    }


class _Detached[T]:
    """One call on a daemon thread, abandoned as soon as it outlasts its timeout.

    A pooled future does not bound anything on its own: the executor joins its worker when the block that
    created it exits, so a stage that has already timed out still holds the turn until the call behind it
    returns. A stage timeout is a latency promise, so the thread is abandoned instead. It is a daemon, so it
    cannot hold the process open, and nothing it computes after the timeout is read.
    """

    def __init__(self, call: Callable[..., T], *args: Any) -> None:
        self._value: T | None = None
        self._error: BaseException | None = None
        self._done = threading.Event()
        threading.Thread(target=self._run, args=(call, args), daemon=True, name="retold-turn-stage").start()

    def _run(self, call: Callable[..., T], args: tuple[Any, ...]) -> None:
        try:
            self._value = call(*args)
        except BaseException as error:  # noqa: BLE001
            self._error = error
        finally:
            self._done.set()

    def result(self, timeout_s: float) -> T:
        if not self._done.wait(timeout_s):
            raise TimeoutError("The stage outlasted its timeout and was abandoned.")
        if self._error is not None:
            raise self._error
        return cast(T, self._value)


class UtilityAwareOrchestrator:
    """Deterministic orchestration of one turn. Fails closed to the draft on every error and budget path.

    Serving, meaning shadow is off and the path is enabled, requires a bundle registry that holds a passing
    fitness result for this configuration's bundle. Without one the constructor refuses, so a host cannot
    serve an unapproved bundle by accident. Shadow mode needs no approval.

    ``retrieval_config`` is the ``RetoldConfig`` the ``retrieve`` callable searches with. Given it, the
    bundle's retrieval hash is derived from that configuration and a manifest that disagrees is refused.
    """

    def __init__(
        self,
        store: Store,
        retrieve: Retrieve,
        gap_policy: GapPolicy | None,
        admission_policy: AdmissionPolicy | None,
        config: UtilityAwareConfig,
        *,
        profile_assembler: ProfileAssembler | None = None,
        registry: Any = None,
        retrieval_config: Any = None,
        actor: str = "host",
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        from retold.policy.bundles import BundleNotApprovedError, bundle_hash

        self._store = store
        self._retrieve = retrieve
        self._gap_policy = gap_policy
        self._admission_policy = admission_policy
        self._config = config
        self._profiles = profile_assembler or ProfileAssembler(store)
        self._actor = actor
        self._clock = clock
        # With a retrieval configuration in hand the bundle describes the runtime that would serve it, not
        # whatever a manifest claims; without one the caller keeps the claim and the responsibility for it.
        components = bundle_components(config, retrieval_config)
        self._bundle_hash = bundle_hash(components)
        self._retrieval_verified = retrieval_config is not None
        declared = "retrieval_config_sha256" in components
        # Approval gates the ability to put memory in front of the user, which needs planning, admission,
        # and regeneration all on. Disabling any stage is a kill switch: it can only make the path safer,
        # so it must never be blocked by a missing fitness result for the degraded configuration.
        serving = config.gap_enabled and not config.shadow and config.admission_mode != "disabled"
        # A declared retrieval hash that nothing checked is worse than none: it reads like an approval of
        # the configuration in front of you. Serving requires the check; shadow warns and records the gap.
        if declared and not self._retrieval_verified:
            from retold.policy.bundles import BundleMismatchError

            if serving:
                raise BundleMismatchError(
                    f"Bundle {self._bundle_hash} declares a retrieval configuration but none was given, so "
                    "nothing verified it. Pass retrieval_config, or drop retrieval_config_sha256 from the "
                    "bundle if this host's retrieval is not a RetoldConfig."
                )
            get_logger(__name__).warning(
                "The bundle declares a retrieval configuration this host cannot verify.",
                extra={"retold": {"bundle_hash": self._bundle_hash, "declared": components["retrieval_config_sha256"]}},
            )
        if serving:
            if registry is None:
                raise BundleNotApprovedError(
                    f"Bundle {self._bundle_hash} cannot serve without a registry holding a passing fitness result; "
                    "use shadow mode."
                )
            registry.require_approved(components)

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
        profile = self._profiles.build(principal) if self._config.profile_enabled else ProfileBlock("", [], False)
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
            bundle_hash=self._bundle_hash,
            bundle_retrieval_verified=self._retrieval_verified,
        )

        path_enabled = self._config.gap_enabled and self._gap_policy is not None and effective != 0
        started = self._clock()
        if not path_enabled:
            decision.draft = baseline()
            timings["draft"] = (self._clock() - started) * 1000
            decision.disposition = (
                "baseline_budget_exhausted" if effective == 0 and self._config.gap_enabled else "baseline_no_gaps"
            )
            return self._persist(decision)

        # Baseline and gap planning run concurrently; the gap call is charged only for what outlasts the draft.
        gap_timeout = self._config.gap_timeout_ms if effective is None else min(self._config.gap_timeout_ms, effective)
        planning = _Detached(self._plan, turn, public_context, profile, labels)
        decision.draft = baseline()
        draft_done = self._clock()
        timings["draft"] = (draft_done - started) * 1000
        budget_bound = effective is not None and effective <= self._config.gap_timeout_ms
        try:
            gap_decision = planning.result(max(0.0, gap_timeout / 1000))
        except TimeoutError:
            gap_decision = GapDecision(
                [], "gap", "timeout", gap_timeout, "budget_exhausted" if budget_bound else "gap_timeout"
            )
        except Exception as error:  # noqa: BLE001
            gap_decision = GapDecision.failed("gap", type(error).__name__)
        timings["gap_overhang"] = max(0.0, (self._clock() - draft_done) * 1000)

        decision.gap_status = gap_decision.status
        decision.gaps = list(gap_decision.gaps)[: self._config.max_gaps]
        _merge_usage(decision.usage, gap_decision.usage)
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
            candidates = self._retrieve(principal, [gap.query for gap in decision.gaps], turn)[
                : self._config.max_candidates
            ]
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

        admission_timeout = (
            self._config.admission_timeout_ms
            if (left := remaining_ms()) is None
            else min(self._config.admission_timeout_ms, left)
        )
        admission_started = self._clock()
        judging = _Detached(self._admit, turn, public_context, profile, decision.draft, candidates)
        try:
            admission = judging.result(max(0.0, admission_timeout / 1000))
        except TimeoutError:
            admission = AdmissionDecision([], [], "admission", "timeout", admission_timeout, "admission_timeout")
        except Exception as error:  # noqa: BLE001
            admission = AdmissionDecision.failed("admission", type(error).__name__)
        timings["admission"] = (self._clock() - admission_started) * 1000
        decision.admission_status = admission.status
        decision.verdicts = list(admission.verdicts)
        _merge_usage(decision.usage, admission.usage)
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
        status: PolicyStatus = (
            result.status if result.status in ("failed", "timeout") else ("ok" if unique else "empty")
        )
        return GapDecision(
            unique, result.policy_id, status, (self._clock() - started) * 1000, result.error, dict(result.usage)
        )

    def _admit(
        self, turn: str, public_context: str | None, profile: ProfileBlock, draft: str, candidates: list[Record]
    ) -> AdmissionDecision:
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
        return AdmissionDecision(
            list(result.admitted_ids),
            verdicts,
            result.policy_id,
            result.status or "ok",
            (self._clock() - started) * 1000,
            result.error,
            dict(result.usage),
        )

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
            "bundle_hash": decision.bundle_hash,
            "usage": decision.usage,
            "bundle_retrieval_verified": decision.bundle_retrieval_verified,
        }
        self._store.insert_turn_decision(row)
        self._store.append_event(
            "turn.decided",
            self._actor,
            None,
            None,
            {
                "decision_id": decision.decision_id,
                "disposition": decision.disposition,
                "shadow": decision.shadow,
                "admitted": len(decision.admitted_ids),
            },
        )
        return decision


def _merge_usage(into: dict[str, dict[str, int]], extra: Mapping[str, Mapping[str, int]]) -> None:
    for model, usage in extra.items():
        bucket = into.setdefault(model, {"prompt": 0, "completion": 0})
        bucket["prompt"] += int(usage.get("prompt", 0))
        bucket["completion"] += int(usage.get("completion", 0))


def render_decision(decision: TurnMemoryDecision) -> str:
    """Compact one-line rendering for logs and tests."""

    return json.dumps(
        {
            "disposition": decision.disposition,
            "gaps": len(decision.gaps),
            "candidates": len(decision.candidate_ids),
            "admitted": decision.admitted_ids,
        },
        sort_keys=True,
    )
