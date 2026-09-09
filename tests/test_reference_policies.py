"""The supported bundle is buildable from the installed package, and its hash describes the live runtime."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from retold.config import RetoldConfig, load_config
from retold.models import Record, Scope
from retold.policy import (
    BundleMismatchError,
    BundleRegistry,
    ProfileBlock,
    UtilityAwareConfig,
    UtilityAwareOrchestrator,
    bundle_components,
    retrieval_config_hash,
)
from retold.policy.reference import (
    ADMISSION_PROMPT_VERSION,
    CATEGORY_PROMPT_VERSION,
    GAP_PROMPT_VERSION,
    SUPPORTED_BUNDLE_ID,
    ReferenceAdmissionPolicy,
    ReferenceCategoryPolicy,
    ReferenceGapPolicy,
    supported_bundle,
    supported_retrieval_config,
)
from retold.store import Store
from retold.util import render_subject

_AT = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
_MANIFEST = Path(__file__).resolve().parents[1] / "benchmarks" / "bundles" / f"{SUPPORTED_BUNDLE_ID}.json"


class ScriptedClient:
    """A completion client that replies with a fixed body and reports what the call cost."""

    def __init__(self, reply: str, usage: Mapping[str, int] | None = None) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str, float]] = []
        self._usage = dict(usage or {})

    def complete(self, system: str, user: str, *, timeout_s: float) -> str:
        self.calls.append((system, user, timeout_s))
        return self.reply

    def last_usage(self) -> Mapping[str, int]:
        return self._usage


class SilentClient(ScriptedClient):
    """The same, without usage reporting."""

    last_usage = None  # type: ignore[assignment]


def _record(record_id: str, content: str) -> Record:
    return Record(
        id=record_id,
        type="semantic",
        version=1,
        content=content,
        subject=render_subject("e1", record_id),
        scope=Scope(kind="user", id="user-1"),
        source_kind="user_statement",
        source_ref=None,
        creator_agent_id="agent",
        evidence=content,
        created_at=_AT,
        event_at=_AT,
        expires_at=None,
        confidence=0.9,
        status="confirmed",
        supersedes_id=None,
        reinforcements=0,
        last_reinforced_at=None,
        tags=[],
        entity_ids=["e1"],
        subject_entity_id="e1",
        attribute=record_id,
    )


def test_the_package_rebuilds_the_recorded_supported_bundle_exactly() -> None:
    components = bundle_components(supported_bundle(), supported_retrieval_config())
    assert components == json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert components["planner"] == f"gpt-4o/{GAP_PROMPT_VERSION}"
    assert components["judge"] == f"gpt-5.4/{ADMISSION_PROMPT_VERSION}"
    assert components["classifier"] == f"gpt-4o/{CATEGORY_PROMPT_VERSION}"


def test_the_supported_bundle_is_not_the_shipped_retrieval_defaults() -> None:
    # The suite measured recall-oriented floors with the judge behind them. Serving the defaults instead is
    # a different bundle, and this is what makes the mismatch check worth having.
    assert retrieval_config_hash(supported_retrieval_config()) != retrieval_config_hash(load_config())


def test_a_bundle_hash_is_derived_from_the_runtime_and_a_disagreeing_manifest_is_refused() -> None:
    config = supported_bundle()
    assert bundle_components(config, supported_retrieval_config())["retrieval_config_sha256"] == retrieval_config_hash(
        supported_retrieval_config()
    )
    with pytest.raises(BundleMismatchError, match="retrieval_config_sha256"):
        bundle_components(config, load_config())


def test_the_orchestrator_refuses_a_bundle_that_does_not_describe_its_own_retrieval(tmp_path: Path) -> None:
    store = Store(tmp_path / "bundle.sqlite")
    with pytest.raises(BundleMismatchError):
        UtilityAwareOrchestrator(
            store,
            lambda principal, queries, context: [],
            None,
            None,
            supported_bundle(shadow=True),
            registry=BundleRegistry(store),
            retrieval_config=load_config(),
        )


def test_a_bundle_that_declares_no_retrieval_hash_takes_the_runtime_one() -> None:
    config = UtilityAwareConfig(gap_enabled=True, bundle={"planner": "fake"})

    assert "retrieval_config_sha256" not in bundle_components(config)
    assert bundle_components(config, RetoldConfig())["retrieval_config_sha256"] == retrieval_config_hash(RetoldConfig())


def test_the_reference_planner_parses_gaps_and_records_what_the_call_cost() -> None:
    client = ScriptedClient(
        json.dumps({"gaps": [{"category": "preference", "query": "the user's time zone"}, {"query": ""}]}),
        {"prompt": 120, "completion": 30},
    )
    decision = ReferenceGapPolicy(client, "gpt-4o").plan("When should we meet?", None, ProfileBlock("", [], False), [])

    assert [(gap.category, gap.query) for gap in decision.gaps] == [("preference", "the user's time zone")]
    assert decision.policy_id == f"gpt-4o/{GAP_PROMPT_VERSION}"
    assert decision.usage == {"gpt-4o": {"prompt": 120, "completion": 30}}
    system, _, timeout_s = client.calls[0]
    assert "(no category information available)" in system
    assert timeout_s == 4.0


def test_the_reference_judge_keeps_only_verdicts_about_candidates_it_was_given() -> None:
    client = SilentClient(
        json.dumps(
            {
                "admitted": ["r1", "ghost"],
                "verdicts": [
                    {"id": "r1", "verdict": "helpful", "reason": "changes the time"},
                    {"id": "ghost", "verdict": "helpful", "reason": "invented"},
                ],
            }
        )
    )
    decision = ReferenceAdmissionPolicy(client, "gpt-5.4").admit(
        "q", None, ProfileBlock("", [], False), "draft", [_record("r1", "Berlin time.")]
    )

    assert decision.admitted_ids == ["r1"]
    assert [verdict.record_id for verdict in decision.verdicts] == ["r1"]
    assert decision.usage == {}


def test_the_reference_classifier_falls_back_to_the_safe_label_on_an_unknown_one() -> None:
    client = SilentClient(
        json.dumps(
            {
                "retrieval_category": "not_a_category",
                "activation_category": "answer_style",
                "applicability": "nonsense",
                "confidence": 0.8,
            }
        )
    )
    decision = ReferenceCategoryPolicy(client, "gpt-4o").classify("Answer briefly by default.")

    assert decision.retrieval_category == "other"
    assert decision.activation_category == "answer_style"
    assert decision.applicability == "ambiguous"


def test_the_benchmark_harness_runs_the_policies_the_package_ships() -> None:
    from benchmarks.phase0_real_retrieval import HostedCategoryPolicy, recall_oriented_config
    from benchmarks.shadow_adapter import HostedAdmissionPolicy, HostedGapPolicy

    assert issubclass(HostedGapPolicy, ReferenceGapPolicy)
    assert issubclass(HostedAdmissionPolicy, ReferenceAdmissionPolicy)
    assert issubclass(HostedCategoryPolicy, ReferenceCategoryPolicy)
    assert retrieval_config_hash(recall_oriented_config()) == retrieval_config_hash(supported_retrieval_config())
