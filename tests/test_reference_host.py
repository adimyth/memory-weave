"""The integration contract a consuming application must honour, exercised with fakes and no network."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from examples.reference_host import MEASURED_ADMISSION_TIMEOUT_MS, MEASURED_GAP_TIMEOUT_MS, KillSwitches, ReferenceHost
from retold.host import MemoryHost
from retold.models import Principal, Record, Scope
from retold.policy import (
    AdmissionDecision,
    BundleNotApprovedError,
    BundleRegistry,
    CandidateVerdict,
    Gap,
    GapDecision,
    RollbackThresholds,
    bundle_components,
)
from retold.store import Store
from retold.util import render_subject

_AT = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
_BUNDLE = {
    "planner": "fake/gap-1",
    "judge": "fake/adm-1",
    "classifier": "fake/cat-1",
    "taxonomy": "t1",
    "inventory_builder": "i1",
    "retrieval_config_sha256": "abc",
}


class Gaps:
    def __init__(self, fire: bool, fail: bool = False) -> None:
        self.fire, self.fail = fire, fail

    def plan(self, turn, public_context, ambient_profile, inventory):
        if self.fail:
            raise RuntimeError("planner down")
        return GapDecision([Gap("preference", "tz")] if self.fire else [], "fake", "ok" if self.fire else "empty")


class Admission:
    def __init__(self, admit: bool) -> None:
        self.should_admit = admit

    def admit(self, turn, public_context, ambient_profile, draft, candidates):
        verdicts = [
            CandidateVerdict(r.id, "helpful" if self.should_admit else "insufficient", "fake") for r in candidates
        ]
        return AdmissionDecision([r.id for r in candidates] if self.should_admit else [], verdicts, "fake", "ok")


@pytest.fixture
def world(tmp_path: Path):
    store = Store(tmp_path / "host.sqlite")
    host = MemoryHost(store)
    host.grant("agent", Scope(kind="user", id="u"), read=True, write=True)
    entity = host.provision_user("u", aliases=("Dev",))
    principal = Principal("agent", "u", "s", None)
    store.create_session("s", "agent", "u", None, _AT)
    store.insert_record(
        Record(
            id="tz",
            type="semantic",
            version=1,
            content="Time zone is Europe/Berlin.",
            subject=render_subject(entity, "tz"),
            scope=Scope(kind="user", id="u"),
            source_kind="user_statement",
            source_ref=None,
            creator_agent_id="agent",
            evidence="x",
            created_at=_AT,
            event_at=_AT,
            expires_at=None,
            confidence=0.9,
            status="confirmed",
            supersedes_id=None,
            reinforcements=0,
            last_reinforced_at=None,
            tags=[],
            entity_ids=[entity],
            subject_entity_id=entity,
            attribute="tz",
            category="time_zone",
        )
    )
    return store, principal


def _host(store, gaps, admission, **kwargs) -> ReferenceHost:
    return ReferenceHost(
        store,
        retrieve=lambda p, q, c: [store.get_record("tz")],
        gap_policy=gaps,
        admission_policy=admission,
        bundle=_BUNDLE,
        **kwargs,
    )


def _approve(store, host_config) -> None:
    BundleRegistry(store).record(bundle_components(host_config), passed=True, evidence="suite", recorded_by="pytest")


def test_unapproved_bundle_can_only_run_in_shadow(world) -> None:
    store, principal = world
    with pytest.raises(BundleNotApprovedError):
        _host(store, Gaps(True), Admission(True))
    host = _host(store, Gaps(True), Admission(True), switches=KillSwitches(regeneration=False))
    decision = host.serve_turn(principal, "when?", None, lambda: "draft", lambda r: "final")
    assert decision.disposition == "shadow_would_regenerate" and decision.response == "draft"
    assert host.is_serving() is False
    # A fitness result earned in shadow approves the same bundle for serving; timeouts are the measured ones.
    from dataclasses import replace

    _approve(store, replace(host.config(), shadow=False))
    host.set_switches(KillSwitches())
    assert host.config().gap_timeout_ms == MEASURED_GAP_TIMEOUT_MS
    assert host.config().admission_timeout_ms == MEASURED_ADMISSION_TIMEOUT_MS
    served = host.serve_turn(principal, "when?", None, lambda: "draft", lambda r: "final")
    assert served.disposition == "regenerated" and served.response == "final"


def test_kill_switches_are_independent_and_change_the_bundle_hash(world) -> None:
    from dataclasses import replace

    store, principal = world
    probe = _host(store, Gaps(True), Admission(True), switches=KillSwitches(regeneration=False))
    hashes = set()
    for switches in (KillSwitches(), KillSwitches(admission=False), KillSwitches(gap=False)):
        cfg = replace(
            probe.config(),
            gap_enabled=switches.gap,
            admission_mode="hosted_judge" if switches.admission else "disabled",
            shadow=not switches.regeneration,
        )
        _approve(store, cfg)
    host = _host(store, Gaps(True), Admission(True))
    hashes.add(host.bundle_hash())
    host.disable("admission")
    hashes.add(host.bundle_hash())
    assert (
        host.serve_turn(principal, "q", None, lambda: "draft", lambda r: "final").disposition
        == "baseline_empty_admission"
    )
    host.disable("gap")
    hashes.add(host.bundle_hash())
    assert host.serve_turn(principal, "q", None, lambda: "draft", lambda r: "final").disposition == "baseline_no_gaps"
    assert len(hashes) == 3, "every switch flip is a distinct bundle"


def test_rollback_disables_the_newest_stage_on_breach_and_logs_it(world) -> None:
    from dataclasses import replace

    store, principal = world
    probe = _host(store, Gaps(True, fail=True), Admission(False), switches=KillSwitches(regeneration=False))
    for switches in (KillSwitches(), KillSwitches(regeneration=False)):
        _approve(store, replace(probe.config(), shadow=not switches.regeneration))
    thresholds = RollbackThresholds(min_turns=3, max_policy_failure_rate=0.1)
    host = _host(store, Gaps(True, fail=True), Admission(False), thresholds=thresholds)
    for _ in range(3):
        decision = host.serve_turn(principal, "q", None, lambda: "draft", lambda r: "final")
        assert decision.disposition == "baseline_policy_failure" and decision.response == "draft"

    reasons, stage = host.rollback_check()
    assert reasons and any("policy failure rate" in r for r in reasons)
    assert stage == "regeneration"
    assert host.switches.regeneration is False and host.is_serving() is False
    events = store.connection.execute("SELECT kind, payload FROM events WHERE kind = 'host.rollback'").fetchall()
    assert len(events) == 1 and "regeneration" in events[0][1]

    # Quiet window: no further rollback.
    calm = _host(
        store,
        Gaps(False),
        Admission(False),
        switches=KillSwitches(regeneration=False),
        thresholds=RollbackThresholds(min_turns=1000),
    )
    assert calm.rollback_check() == ([], None)


def test_per_request_budget_is_honoured_by_the_host(world) -> None:
    from dataclasses import replace

    store, principal = world
    probe = _host(store, Gaps(True), Admission(True), switches=KillSwitches(regeneration=False))
    _approve(store, replace(probe.config(), shadow=False))
    host = _host(store, Gaps(True), Admission(True))
    decision = host.serve_turn(principal, "q", None, lambda: "draft", lambda r: "final", latency_budget_ms=0)
    assert decision.disposition == "baseline_budget_exhausted" and decision.response == "draft"
    assert decision.requested_budget_ms == 0


def test_kill_switches_work_without_approving_the_degraded_bundles(world) -> None:
    from dataclasses import replace

    store, principal = world
    probe = _host(store, Gaps(True), Admission(True), switches=KillSwitches(regeneration=False))
    _approve(store, replace(probe.config(), shadow=False))
    host = _host(store, Gaps(True), Admission(True))
    assert host.serve_turn(principal, "q", None, lambda: "draft", lambda r: "final").disposition == "regenerated"
    host.disable("admission")  # no fitness result exists for this configuration; it must still be allowed
    assert (
        host.serve_turn(principal, "q", None, lambda: "draft", lambda r: "final").disposition
        == "baseline_empty_admission"
    )
    host.disable("gap")
    assert host.serve_turn(principal, "q", None, lambda: "draft", lambda r: "final").disposition == "baseline_no_gaps"
    # Re-enabling everything brings the approval requirement back, and the original bundle is approved.
    host.set_switches(KillSwitches())
    assert host.serve_turn(principal, "q", None, lambda: "draft", lambda r: "final").disposition == "regenerated"
