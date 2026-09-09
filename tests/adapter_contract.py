"""The framework contract suite: one set of assertions every adapter must pass.

An adapter test builds a ``ContractDriver`` for its framework and calls ``run_contract``. The driver
speaks the framework; the assertions speak Retold. A behaviour that holds for one adapter and not
the other is a contract gap, and it is caught here rather than in two diverging test files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from retold.models import Principal, SearchRequest
from retold.runtime import MemoryRuntime
from retold.store import Store

FRAMEWORK_MARKERS = ("langchain", "langgraph", "deepagents", "thread_id", "configurable", "crewai", "BaseTool")


@dataclass
class ScriptedTurn:
    """What the fake model does when it sees a user message: tool calls first, then a final answer."""

    user: str
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    answer: str = "Understood."
    # Replies queued after the answer, for paths that call the model again on the same turn.
    followups: list[str] = field(default_factory=list)


@dataclass
class TurnOutcome:
    answer: str
    tool_results: list[str]


class ContractDriver(Protocol):
    runtime: MemoryRuntime
    principal: Principal

    def run_turn(self, turn: ScriptedTurn) -> TurnOutcome: ...

    def end_session(self) -> None: ...

    def wait_for_extraction(self) -> None: ...


PREFERENCE = "I prefer concise technical answers with a short rationale."
CANDIDATE = "The user prefers concise technical answers with a short rationale."


def run_contract(driver: ContractDriver) -> None:
    """The shared scenario: state a preference, save it with evidence, recall it later, end the session."""

    store = driver.runtime.store
    principal = driver.principal

    first = driver.run_turn(
        ScriptedTurn(
            PREFERENCE,
            tool_calls=[
                (
                    "memory_write",
                    {
                        "type": "semantic",
                        "content": CANDIDATE,
                        "source_kind": "user_statement",
                        "evidence": PREFERENCE,
                        "attribute": "answer_style",
                    },
                )
            ],
            answer="Saved: concise answers with a short rationale.",
        )
    )
    write_result = json.loads(first.tool_results[0])
    assert write_result["ok"] is True and write_result["outcome"] == "created", write_result
    record = store.get_record(write_result["record_id"])
    assert record is not None
    assert record.source_kind == "user_statement"
    assert record.source_ref is not None and record.source_ref.startswith(f"session:{principal.session_id}<turn:")
    assert record.evidence == PREFERENCE
    turns = store.session_turns(principal.session_id or "")
    assert [turn.role for turn in turns][:1] == ["user"]
    assert turns[0].content == PREFERENCE

    second = driver.run_turn(
        ScriptedTurn(
            "How should you answer my questions?",
            tool_calls=[("memory_search", {"queries": ["answer style preference"], "k": 4})],
            answer="Concisely, with a short rationale.",
        )
    )
    assert record.id in second.tool_results[0], second.tool_results[0]
    assert "Recalled 1 memor" in second.tool_results[0]

    # Tool results only ever carry memory content that the search itself returned.
    for outcome in (first, second):
        for text in outcome.tool_results:
            for other in store.records_in_scope(record.scope, statuses=["provisional", "confirmed"]):
                if other.id != record.id:
                    assert other.content not in text

    # The transcript holds the user turns, the assistant answers, and the tool results, in order.
    roles = [turn.role for turn in store.session_turns(principal.session_id or "")]
    assert roles.count("user") == 2
    assert "assistant" in roles and "tool" in roles

    driver.end_session()
    driver.end_session()  # a second close is a no-op
    driver.wait_for_extraction()
    session = store.get_session(principal.session_id or "")
    assert session is not None and session.ended_at is not None and session.extracted_at is not None
    runs = store.connection.execute(
        "SELECT count(*) FROM events WHERE kind = 'extraction.run' AND json_extract(payload, '$.session_id') = ?",
        (principal.session_id,),
    ).fetchone()[0]
    assert runs == 1, "session end must trigger extraction exactly once"
    summary = store.active_session_summary(f"session:{principal.session_id}")
    assert summary is not None

    assert_no_framework_fields(store)
    response = driver.runtime.retriever.search(
        principal, SearchRequest(["answer style"], None, None, None, None, None, 8, False)
    )
    assert any(result.record.id == record.id for result in response.results)


def assert_no_framework_fields(store: Store) -> None:
    """No framework-specific value reaches a core record: not in content, subject, tags, or source refs."""

    rows = store.connection.execute("SELECT * FROM records").fetchall()
    for row in rows:
        blob = json.dumps({key: row[key] for key in row.keys()}, default=str)
        for marker in FRAMEWORK_MARKERS:
            assert not re.search(marker, blob, re.IGNORECASE), f"{marker!r} leaked into records: {blob[:200]}"


def assert_isolated(driver_a: ContractDriver, driver_b: ContractDriver) -> None:
    """Two principals sharing one store never see each other's records through the tools."""

    a_store = driver_a.runtime.store
    b = driver_b.principal
    a = driver_a.principal
    a_ids = {r.id for r in a_store.records_in_scope(_user_scope(a), statuses=["provisional", "confirmed"])}
    for driver, own, other in ((driver_a, a, b), (driver_b, b, a)):
        response = driver.runtime.retriever.search(
            own, SearchRequest(["answer style"], None, None, None, None, None, 8, False)
        )
        for result in response.results:
            assert result.record.scope.id != other.user_id
    assert a_ids


def _user_scope(principal: Principal) -> Any:
    from retold.models import Scope

    return Scope(kind="user", id=principal.user_id)


MODE_MATRIX: tuple[tuple[str, str, bool], ...] = (
    ("tool_only", "tool_only", True),
    ("auto", "tool_only", True),
    ("hybrid", "tool_only", True),
    ("tool_only", "utility_aware", True),
    # Host-issued search would put candidates in front of the model before admission decides.
    ("auto", "utility_aware", False),
    ("hybrid", "utility_aware", False),
)


def assert_mode_matrix(build: Any, tool_names: Any) -> None:
    """Every trigger mode against every memory mode: what each combination registers, and what is refused.

    ``build(trigger_mode, memory_mode)`` returns the framework's adapter; ``tool_names(adapter)`` returns the
    names it registered. The refused combinations must fail at construction, not at the turn that would have
    leaked a record past the judge.
    """

    import pytest

    for trigger, memory, allowed in MODE_MATRIX:
        if not allowed:
            with pytest.raises(ValueError, match="tool_only"):
                build(trigger, memory)
            continue
        adapter = build(trigger, memory)
        assert adapter.trigger_mode == trigger
        assert adapter.memory_mode == memory
        names = set(tool_names(adapter))
        assert ("memory_search" in names) is (trigger != "auto"), (
            f"{trigger}/{memory} registered the wrong search surface"
        )
        assert {"memory_write", "memory_get", "memory_revise", "memory_forget"} <= names
        orchestrating = getattr(adapter, "_orchestrator", None) is not None
        assert orchestrating is (memory == "utility_aware")
