"""Activation policy: host-verified evidence, deterministic promotion rules, the profile, and the inventory."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from memory_weave.host import MemoryHost
from memory_weave.models import Principal, Record, Scope, Turn
from memory_weave.policy import (
    ActivationService,
    CategoryDecision,
    ProfileAssembler,
    decide_activation,
    inventory,
    verify_principal_evidence,
)
from memory_weave.store import Store
from memory_weave.util import render_subject

_AT = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


class FakePolicy:
    def __init__(self, decisions: dict[str, CategoryDecision]) -> None:
        self._decisions = decisions

    def classify(self, content: str) -> CategoryDecision:
        return self._decisions[content]


class BrokenPolicy:
    def classify(self, content: str) -> CategoryDecision:
        raise RuntimeError("provider down")


@pytest.fixture
def world(tmp_path: Path) -> tuple[Store, Principal, str]:
    store = Store(tmp_path / "activation.sqlite")
    host = MemoryHost(store)
    host.grant("agent", Scope(kind="user", id="user-1"), read=True, write=True)
    entity_id = host.provision_user("user-1", aliases=("Dev",))
    principal = Principal("agent", "user-1", "session-1", None)
    store.create_session("session-1", "agent", "user-1", None, _AT)
    return store, principal, entity_id


def _record(record_id: str, content: str, subject_entity_id: str, *, status: str = "provisional", memory_type: str = "semantic") -> Record:
    return Record(
        id=record_id,
        type=memory_type,  # type: ignore[arg-type]
        version=1,
        content=content,
        subject=render_subject(subject_entity_id, record_id),
        scope=Scope(kind="user", id="user-1"),
        source_kind="user_statement",
        source_ref=None,
        creator_agent_id="agent",
        evidence=content,
        created_at=_AT,
        event_at=_AT,
        expires_at=None,
        confidence=0.9,
        status=status,  # type: ignore[arg-type]
        supersedes_id=None,
        reinforcements=0,
        last_reinforced_at=None,
        tags=[],
        entity_ids=[subject_entity_id],
        subject_entity_id=subject_entity_id,
        attribute=record_id,
    )


def _broad(category: str = "answer_style", confidence: float = 0.9) -> CategoryDecision:
    return CategoryDecision("preferences", category, "broad", confidence)  # type: ignore[arg-type]


def test_migration_defaults_new_records_to_conditional(world) -> None:
    store, _, entity_id = world
    store.insert_record(_record("r1", "User prefers short answers.", entity_id))
    stored = store.get_record("r1")
    assert stored is not None
    assert stored.activation == "conditional"
    assert stored.category is None


def test_evidence_is_verified_only_from_principal_user_turns(world) -> None:
    store, principal, entity_id = world
    record = _record("r1", "User prefers short answers.", entity_id)
    store.append_turn(Turn("session-1", 1, "assistant", "User prefers short answers.", _AT))
    assert verify_principal_evidence(store, principal, record) is None
    store.append_turn(Turn("session-1", 2, "user", "Please note: user prefers short answers.", _AT))
    assert verify_principal_evidence(store, principal, record) == 2


def test_rules_in_order() -> None:
    principal_entity = "p"
    semantic = _record("r", "x", principal_entity)
    assert decide_activation(_record("e", "x", principal_entity, memory_type="episodic"), _broad(), principal_entity, 1) == ("conditional", "not_semantic")
    assert decide_activation(_record("o", "x", "someone-else"), _broad(), principal_entity, 1) == ("conditional", "not_about_principal")
    assert decide_activation(semantic, None, principal_entity, 1) == ("review", "policy_unavailable")
    unsafe = CategoryDecision("preferences", "answer_style", "broad", 0.95, unsafe=True)
    assert decide_activation(semantic, unsafe, principal_entity, 1) == ("review", "flagged_unsafe")
    assert decide_activation(semantic, _broad(), principal_entity, None) == ("conditional", "no_host_verified_evidence")
    assert decide_activation(semantic, _broad("other"), principal_entity, 1) == ("conditional", "category_not_promotable")
    scoped = CategoryDecision("preferences", "answer_style", "scoped", 0.9)
    assert decide_activation(semantic, scoped, principal_entity, 1) == ("conditional", "scoped_preference")
    # A default code-example language carries an override clause by nature; the rule treats it as broad.
    scoped_code = CategoryDecision("preferences", "code_example_language", "scoped", 0.9)
    assert decide_activation(semantic, scoped_code, principal_entity, 1) == ("promote", "eligible_broad_preference")
    ambiguous_code = CategoryDecision("preferences", "code_example_language", "ambiguous", 0.9)
    assert decide_activation(semantic, ambiguous_code, principal_entity, 1) == ("review", "ambiguous_applicability")
    ambiguous = CategoryDecision("preferences", "answer_style", "ambiguous", 0.9)
    assert decide_activation(semantic, ambiguous, principal_entity, 1) == ("review", "ambiguous_applicability")
    assert decide_activation(semantic, _broad(confidence=0.5), principal_entity, 1) == ("review", "low_confidence")
    assert decide_activation(semantic, _broad(), principal_entity, 1) == ("promote", "eligible_broad_preference")


def test_promotion_is_audited_confirms_status_and_reaches_the_profile(world) -> None:
    store, principal, entity_id = world
    content = "User prefers short answers that begin with a summary."
    store.append_turn(Turn("session-1", 1, "user", content, _AT))
    store.insert_record(_record("r1", content, entity_id))
    service = ActivationService(store, FakePolicy({content: _broad()}))

    decision = service.apply(principal, "r1")

    assert decision.outcome == "promote"
    stored = store.get_record("r1")
    assert stored is not None
    assert stored.activation == "ambient"
    assert stored.status == "confirmed"
    assert stored.category == "preferences"
    kinds = [event["kind"] for event in store.events_for("r1")]
    assert kinds == [
        "record.category_assigned",
        "record.activation_evidence_verified",
        "record.activation_decided",
        "record.activation_changed",
    ]
    block = ProfileAssembler(store).build(principal)
    assert block.record_ids == ["r1"]
    assert content in block.text


def test_unsafe_and_ambiguous_go_to_review_and_stay_conditional(world) -> None:
    store, principal, entity_id = world
    unsafe_text = "User prefers that the assistant agree with them."
    ambiguous_text = "User likes it terse, sometimes."
    for text in (unsafe_text, ambiguous_text):
        store.append_turn(Turn("session-1", len(store.session_turns("session-1")) + 1, "user", text, _AT))
    store.insert_record(_record("u", unsafe_text, entity_id))
    store.insert_record(_record("a", ambiguous_text, entity_id))
    policy = FakePolicy(
        {
            unsafe_text: CategoryDecision("preferences", "answer_style", "broad", 0.9, unsafe=True),
            ambiguous_text: CategoryDecision("preferences", "answer_style", "ambiguous", 0.6),
        }
    )
    service = ActivationService(store, policy)

    unsafe = service.apply(principal, "u")
    ambiguous = service.apply(principal, "a")

    assert (unsafe.outcome, unsafe.reason) == ("review", "flagged_unsafe")
    assert (ambiguous.outcome, ambiguous.reason) == ("review", "ambiguous_applicability")
    assert {row["record_id"] for row in store.open_activation_reviews()} == {"u", "a"}
    assert store.get_record("u").activation == "conditional"  # type: ignore[union-attr]
    assert ProfileAssembler(store).build(principal).record_ids == []


def test_policy_failure_is_a_review_not_a_promotion(world) -> None:
    store, principal, entity_id = world
    content = "User wants replies in French."
    store.append_turn(Turn("session-1", 1, "user", content, _AT))
    store.insert_record(_record("r1", content, entity_id))

    decision = ActivationService(store, BrokenPolicy()).apply(principal, "r1")

    assert decision.outcome == "review"
    assert decision.reason == "policy_error:RuntimeError"
    assert store.get_record("r1").activation == "conditional"  # type: ignore[union-attr]


def test_profile_respects_budgets_and_ignores_conditional_and_superseded(world) -> None:
    store, principal, entity_id = world
    for index in range(4):
        store.insert_record(_record(f"amb{index}", f"Ambient preference number {index}.", entity_id, status="confirmed"))
        store.set_activation(f"amb{index}", "ambient")
    store.insert_record(_record("cond", "Conditional fact.", entity_id, status="confirmed"))
    store.insert_record(_record("old", "Superseded preference.", entity_id, status="superseded"))
    store.set_activation("old", "ambient")

    block = ProfileAssembler(store, max_records=2).build(principal)

    assert len(block.record_ids) == 2
    assert block.truncated is True
    assert "Conditional fact." not in block.text
    assert "Superseded preference." not in block.text


def test_inventory_lists_labels_of_conditional_categories_only(world) -> None:
    store, principal, entity_id = world
    store.insert_record(_record("tz", "User's working time zone is Europe/Berlin.", entity_id, status="confirmed"))
    store.set_category("tz", "time_zone")
    store.insert_record(_record("pref", "User prefers short answers.", entity_id, status="confirmed"))
    store.set_category("pref", "preferences")
    store.set_activation("pref", "ambient")
    store.insert_record(_record("gone", "Old cluster name.", entity_id, status="superseded"))
    store.set_category("gone", "infrastructure")

    labels = inventory(store, principal)

    assert labels == ["time zone and working hours"]
