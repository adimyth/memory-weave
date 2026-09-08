"""Phase 14 gate: the same conversation through both adapters leaves the same semantic memory state."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

pytest.importorskip("deepagents")
pytest.importorskip("crewai")

import test_crewai_adapter as crew  # noqa: E402
import test_deepagents_adapter as deep  # noqa: E402
from adapter_contract import run_contract  # noqa: E402

from memory_weave.host import MemoryHost  # noqa: E402
from memory_weave.models import Scope  # noqa: E402
from memory_weave.store import Store  # noqa: E402


def _semantic_state(store: Store, user: str) -> list[tuple[str | None, str, str, str, str | None, str]]:
    records = store.records_in_scope(Scope(kind="user", id=user), statuses=["provisional", "confirmed"])
    return sorted(
        (r.attribute, r.content, r.source_kind, r.status, r.evidence, r.activation)
        for r in records
        if r.source_kind != "session_summary"
    )


def test_equivalent_conversations_produce_equivalent_semantic_memory(tmp_path: Path) -> None:
    deep_store = Store(tmp_path / "deep.sqlite")
    crew_store = Store(tmp_path / "crew.sqlite")
    for store, agent in ((deep_store, "assistant"), (crew_store, crew.agent_id_from_role(crew.ROLE))):
        host = MemoryHost(store)
        host.grant(agent, Scope(kind="user", id="aditya"), read=True, write=True)
        host.provision_user("aditya")

    deep_driver = deep._driver(deep_store)
    crew_driver = crew._driver(crew_store)
    run_contract(deep_driver)
    run_contract(crew_driver)

    assert _semantic_state(deep_store, "aditya") == _semantic_state(crew_store, "aditya")
    assert deep_store.active_session_summary("session:thread-1") is not None
    assert crew_store.active_session_summary("session:crew-1") is not None

    # The one documented difference: CrewAI records the task description as the user turn and each step
    # output as a tool or assistant turn, so its transcript is coarser than the Deep Agents one.
    deep_roles = [turn.role for turn in deep_store.session_turns("thread-1")]
    crew_roles = [turn.role for turn in crew_store.session_turns("crew-1")]
    assert deep_roles.count("user") == crew_roles.count("user") == 2
    assert deep_roles.count("tool") == crew_roles.count("tool") == 2
    assert deep_roles.count("assistant") >= 2 and crew_roles.count("assistant") == 2
    deep_store.close()
    crew_store.close()
