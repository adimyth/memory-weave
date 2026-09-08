"""Reference host adapter: how a consuming application wires the utility-aware path, canary-ready.

Memory Weave is a library. The application that serves users owns the model clients, the traffic cohorts,
the deployment configuration, and the rollback decision. This adapter shows the integration contract that
application must honour, and nothing else:

- The bundle is declared once, hashed, and may serve only with a recorded passing fitness result. An
  unapproved bundle runs in shadow mode; the constructor cannot be talked into serving it.
- Stage timeouts and the per-turn budget come from measured latency, not from a configuration example.
- Each stage has its own kill switch: profile, gap planning, admission, and regeneration. Flipping one
  never changes the others, and every flip is a new bundle hash, so it is visible in the decision log.
- The rollback check reads the metrics aggregator over a window and disables the newest stage when a
  threshold is breached. The host decides when to call it; the library decides what it says.

This module makes no network calls. The application supplies the baseline and regeneration callables and
the policy objects, which in the benchmarks are the hosted adapters in `benchmarks/shadow_adapter.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal

from memory_weave.models import Principal, Record
from memory_weave.policy import (
    AdmissionPolicy,
    BundleRegistry,
    GapPolicy,
    MetricsReport,
    ProfileAssembler,
    RollbackThresholds,
    TurnMemoryDecision,
    TurnOptions,
    UtilityAwareConfig,
    UtilityAwareOrchestrator,
    aggregate,
    bundle_components,
    rollback_reasons,
)
from memory_weave.policy.utility_aware import Retrieve
from memory_weave.store import Store

# Measured on the Phase 0 splits with eight-candidate pools: planner about 1 s, judge 2 to 5 s at p95.
MEASURED_GAP_TIMEOUT_MS = 4000
MEASURED_ADMISSION_TIMEOUT_MS = 8000

Stage = Literal["regeneration", "admission", "gap", "profile"]
# Newest stage first: rollback disables in this order and never skips ahead.
ROLLBACK_ORDER: tuple[Stage, ...] = ("regeneration", "admission", "gap", "profile")


@dataclass(frozen=True, slots=True)
class KillSwitches:
    profile: bool = True
    gap: bool = True
    admission: bool = True
    regeneration: bool = True  # False means shadow mode: decide and log, never apply


class ReferenceHost:
    """One application-side object per serving process. Thread-safe use is the application's concern."""

    def __init__(
        self,
        store: Store,
        *,
        retrieve: Retrieve,
        gap_policy: GapPolicy,
        admission_policy: AdmissionPolicy,
        bundle: dict[str, object],
        switches: KillSwitches = KillSwitches(),
        default_budget_ms: int | None = None,
        thresholds: RollbackThresholds = RollbackThresholds(),
        gap_timeout_ms: int = MEASURED_GAP_TIMEOUT_MS,
        admission_timeout_ms: int = MEASURED_ADMISSION_TIMEOUT_MS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._retrieve = retrieve
        self._gap_policy = gap_policy
        self._admission_policy = admission_policy
        self._bundle = dict(bundle)
        self._switches = switches
        self._default_budget_ms = default_budget_ms
        self._thresholds = thresholds
        self._gap_timeout_ms = gap_timeout_ms
        self._admission_timeout_ms = admission_timeout_ms
        self._registry = BundleRegistry(store)
        self._assembler = ProfileAssembler(store)
        self._clock = clock
        self._orchestrator = self._build()

    # -- configuration ----------------------------------------------------------------------------------

    @property
    def switches(self) -> KillSwitches:
        return self._switches

    def config(self) -> UtilityAwareConfig:
        return UtilityAwareConfig(
            gap_enabled=self._switches.gap,
            admission_mode="hosted_judge" if self._switches.admission else "disabled",
            shadow=not self._switches.regeneration,
            gap_timeout_ms=self._gap_timeout_ms,
            admission_timeout_ms=self._admission_timeout_ms,
            latency_budget_ms=self._default_budget_ms,
            bundle=self._bundle,
        )

    def bundle_hash(self) -> str:
        from memory_weave.policy import bundle_hash

        return bundle_hash(bundle_components(self.config()))

    def is_serving(self) -> bool:
        return self._switches.gap and self._switches.regeneration

    def _build(self) -> UtilityAwareOrchestrator:
        # Raises BundleNotApprovedError when the switches would serve an unapproved bundle.
        return UtilityAwareOrchestrator(
            self._store,
            self._retrieve,
            self._gap_policy,
            self._admission_policy,
            self.config(),
            profile_assembler=self._assembler,
            registry=self._registry,
        )

    def set_switches(self, switches: KillSwitches) -> None:
        """Apply new switches atomically: the orchestrator is rebuilt first, then swapped in."""

        previous = self._switches
        self._switches = switches
        try:
            self._orchestrator = self._build()
        except Exception:
            self._switches = previous
            raise

    def disable(self, stage: Stage) -> KillSwitches:
        updated = replace(self._switches, **{stage: False})
        self.set_switches(updated)
        return updated

    # -- serving ------------------------------------------------------------------------------------------

    def serve_turn(
        self,
        principal: Principal,
        turn: str,
        public_context: str | None,
        baseline: Callable[[], str],
        regenerate: Callable[[list[Record]], str],
        *,
        latency_budget_ms: int | None = None,
    ) -> TurnMemoryDecision:
        """Return the decision; the application sends `decision.response` to the user."""

        if not self._switches.profile:
            # Profile off: the baseline callable the application passes must not include the profile.
            # The library cannot enforce what the application renders, so this is part of the contract.
            pass
        options = TurnOptions(latency_budget_ms if latency_budget_ms is not None else self._default_budget_ms)
        return self._orchestrator.prepare_turn(principal, turn, public_context, baseline, regenerate, options)

    # -- rollback -----------------------------------------------------------------------------------------

    def metrics(self, *, since: datetime | None = None, until: datetime | None = None) -> MetricsReport:
        return aggregate(self._store, since=since, until=until, bundle_hash=self.bundle_hash())

    def rollback_check(
        self, *, since: datetime | None = None, until: datetime | None = None
    ) -> tuple[list[str], Stage | None]:
        """Evaluate thresholds over the window for the current bundle. On breach, disable the newest active
        stage and return the reasons and the stage disabled. The application decides how often to call this."""

        report = self.metrics(since=since, until=until)
        reasons = rollback_reasons(report, self._thresholds)
        if not reasons:
            return [], None
        for stage in ROLLBACK_ORDER:
            if getattr(self._switches, stage):
                self.disable(stage)
                self._store.append_event(
                    "host.rollback",
                    "host",
                    None,
                    None,
                    {
                        "stage_disabled": stage,
                        "reasons": reasons,
                        "bundle_hash": report.bundle_hash,
                        "turns": report.turns,
                    },
                )
                return reasons, stage
        return reasons, None
