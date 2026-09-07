"""Utility-aware orchestration: every branch fails closed to the draft, the budget only shortens the path,
shadow mode never regenerates, and one turn-decision row is written on every branch."""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from memory_weave.host import MemoryHost
from memory_weave.models import Principal, Record, Scope
from memory_weave.policy import (
    AdmissionDecision,
    CandidateVerdict,
    Gap,
    GapDecision,
    ProfileBlock,
    TurnOptions,
    UtilityAwareConfig,
    UtilityAwareOrchestrator,
)
from memory_weave.store import Store
from memory_weave.util import render_subject

_AT = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


def _record(record_id: str, content: str, entity_id: str, *, activation: str = "conditional", category: str | None = "time_zone") -> Record:
    return Record(
        id=record_id, type="semantic", version=1, content=content, subject=render_subject(entity_id, record_id),
        scope=Scope(kind="user", id="user-1"), source_kind="user_statement", source_ref=None, creator_agent_id="agent",
        evidence=content, created_at=_AT, event_at=_AT, expires_at=None, confidence=0.9, status="confirmed",
        supersedes_id=None, reinforcements=0, last_reinforced_at=None, tags=[], entity_ids=[entity_id],
        subject_entity_id=entity_id, attribute=record_id, activation=activation, category=category,  # type: ignore[arg-type]
    )


class FakeGaps:
    def __init__(self, gaps: list[Gap] | Exception, delay_s: float = 0.0) -> None:
        self._gaps = gaps
        self._delay = delay_s
        self.seen_inventory: Sequence[str] | None = None

    def plan(self, turn: str, public_context: str | None, ambient_profile: ProfileBlock, inventory: Sequence[str]) -> GapDecision:
        self.seen_inventory = inventory
        time.sleep(self._delay)
        if isinstance(self._gaps, Exception):
            raise self._gaps
        return GapDecision(list(self._gaps), "fake-gap", "ok" if self._gaps else "empty")


class FakeAdmission:
    def __init__(self, admit: list[str], verdict_for_rest: str = "insufficient", incomplete: bool = False) -> None:
        self._admit = admit
        self._rest = verdict_for_rest
        self._incomplete = incomplete

    def admit(self, turn, public_context, ambient_profile, draft, candidates) -> AdmissionDecision:
        verdicts = []
        for record in candidates:
            if self._incomplete:
                break
            verdict = "helpful" if record.id in self._admit else self._rest
            verdicts.append(CandidateVerdict(record.id, verdict, "fake"))  # type: ignore[arg-type]
        return AdmissionDecision(list(self._admit), verdicts, "fake-admission", "ok")


@pytest.fixture
def world(tmp_path: Path):
    store = Store(tmp_path / "ua.sqlite")
    host = MemoryHost(store)
    host.grant("agent", Scope(kind="user", id="user-1"), read=True, write=True)
    entity = host.provision_user("user-1", aliases=("Dev",))
    principal = Principal("agent", "user-1", "s1", None)
    store.create_session("s1", "agent", "user-1", None, _AT)
    store.insert_record(_record("tz", "User's working time zone is Europe/Berlin.", entity))
    store.insert_record(_record("style", "User prefers short answers.", entity, activation="ambient", category="preferences"))
    return store, principal, entity


def _orchestrator(store, gaps, admission, **overrides):
    config = UtilityAwareConfig(gap_enabled=True, admission_mode="hosted_judge", **overrides)

    def retrieve(principal, queries, context):
        return [r for r in (store.get_record("tz"), store.get_record("style")) if r is not None]

    return UtilityAwareOrchestrator(store, retrieve, gaps, admission, config)


def _generators(calls: list[str]):
    def baseline() -> str:
        calls.append("baseline")
        return "draft"

    def regenerate(records: list[Record]) -> str:
        calls.append("regenerate:" + ",".join(r.id for r in records))
        return "final"

    return baseline, regenerate


def test_full_path_regenerates_once_with_admitted_records_and_logs_the_decision(world) -> None:
    store, principal, _ = world
    calls: list[str] = []
    gaps = FakeGaps([Gap("preference", "user's time zone")])
    orchestrator = _orchestrator(store, gaps, FakeAdmission(["tz"]))

    decision = orchestrator.prepare_turn(principal, "When should we meet?", None, *_generators(calls))

    assert decision.disposition == "regenerated"
    assert decision.response == "final"
    assert decision.admitted_ids == ["tz"]
    assert decision.candidate_ids == ["tz"], "ambient records never enter the conditional candidate pool"
    assert calls == ["baseline", "regenerate:tz"]
    assert decision.profile_record_ids == ["style"]
    assert gaps.seen_inventory == ["time zone and working hours"]
    rows = store.turn_decisions("s1")
    assert len(rows) == 1 and rows[0]["disposition"] == "regenerated" and rows[0]["admitted_ids"] == ["tz"]


@pytest.mark.parametrize(
    "gaps, admission, expected",
    [
        (FakeGaps([]), FakeAdmission(["tz"]), "baseline_no_gaps"),
        (FakeGaps(RuntimeError("down")), FakeAdmission(["tz"]), "baseline_policy_failure"),
        (FakeGaps([Gap("preference", "tz")]), FakeAdmission([]), "baseline_empty_admission"),
        (FakeGaps([Gap("preference", "tz")]), FakeAdmission(["tz"], incomplete=True), "baseline_policy_failure"),
        (FakeGaps([Gap("preference", "tz")]), FakeAdmission(["nope"]), "baseline_empty_admission"),
    ],
)
def test_every_non_admitting_branch_returns_the_draft_and_logs(world, gaps, admission, expected) -> None:
    store, principal, _ = world
    calls: list[str] = []
    decision = _orchestrator(store, gaps, admission).prepare_turn(principal, "q", None, *_generators(calls))
    assert decision.disposition == expected
    assert decision.response == "draft"
    assert calls == ["baseline"]
    assert store.turn_decisions("s1")[-1]["disposition"] == expected


def test_zero_budget_skips_the_conditional_path_but_keeps_the_profile(world) -> None:
    store, principal, _ = world
    calls: list[str] = []
    gaps = FakeGaps([Gap("preference", "tz")])
    decision = _orchestrator(store, gaps, FakeAdmission(["tz"])).prepare_turn(
        principal, "q", None, *_generators(calls), options=TurnOptions(latency_budget_ms=0)
    )
    assert decision.disposition == "baseline_budget_exhausted"
    assert decision.gap_status == "skipped"
    assert decision.profile_record_ids == ["style"]
    assert gaps.seen_inventory is None
    assert calls == ["baseline"]


def test_admitted_but_no_time_to_regenerate_is_not_applied(world) -> None:
    store, principal, _ = world
    calls: list[str] = []

    def slow_baseline() -> str:
        calls.append("baseline")
        time.sleep(0.05)
        return "draft"

    def regenerate(records):
        calls.append("regenerate")
        return "final"

    orchestrator = _orchestrator(store, FakeGaps([Gap("preference", "tz")]), FakeAdmission(["tz"]))
    decision = orchestrator.prepare_turn(principal, "q", None, slow_baseline, regenerate, TurnOptions(latency_budget_ms=40))
    assert decision.disposition == "admitted_not_applied"
    assert decision.admitted_ids == ["tz"]
    assert decision.response == "draft"
    assert "regenerate" not in calls


def test_budget_exhausted_before_admission_returns_the_draft(world) -> None:
    store, principal, _ = world
    calls: list[str] = []
    orchestrator = _orchestrator(store, FakeGaps([Gap("preference", "tz")], delay_s=0.08), FakeAdmission(["tz"]), gap_timeout_ms=5000)
    decision = orchestrator.prepare_turn(principal, "q", None, *_generators(calls), options=TurnOptions(latency_budget_ms=10))
    assert decision.disposition == "baseline_budget_exhausted"
    assert decision.response == "draft"


def test_shadow_mode_never_regenerates_but_records_what_it_would_have_done(world) -> None:
    store, principal, _ = world
    calls: list[str] = []
    orchestrator = _orchestrator(store, FakeGaps([Gap("preference", "tz")]), FakeAdmission(["tz"]), shadow=True)
    decision = orchestrator.prepare_turn(principal, "q", None, *_generators(calls))
    assert decision.disposition == "shadow_would_regenerate"
    assert decision.response == "draft"
    assert decision.admitted_ids == ["tz"]
    assert calls == ["baseline"]
    assert store.turn_decisions("s1")[-1]["shadow"] is True


def test_disabled_path_is_the_old_behaviour(world) -> None:
    store, principal, _ = world
    calls: list[str] = []
    config = UtilityAwareConfig()
    orchestrator = UtilityAwareOrchestrator(store, lambda p, q, c: [], None, None, config)
    decision = orchestrator.prepare_turn(principal, "q", None, *_generators(calls))
    assert decision.disposition == "baseline_no_gaps"
    assert decision.gap_status == "skipped"
    assert calls == ["baseline"]
    assert len(store.turn_decisions("s1")) == 1
