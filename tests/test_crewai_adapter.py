"""Phase 14: the CrewAI adapter through a real crew with a scripted ReAct model, and the shared contract."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

pytest.importorskip("crewai")

from adapter_contract import (  # noqa: E402
    CANDIDATE,
    ScriptedTurn,
    TurnOutcome,
    assert_isolated,
    run_contract,
)
from crewai import Agent, Crew, Task  # noqa: E402
from crewai.llms.base_llm import BaseLLM  # noqa: E402
from pydantic import Field  # noqa: E402

from retold.adapters.crewai import CrewAIMemoryAdapter, agent_id_from_role, principal_from_inputs  # noqa: E402
from retold.config import EmbeddingConfig, RetoldConfig  # noqa: E402
from retold.host import MemoryHost  # noqa: E402
from retold.index.embedder import FakeEmbedder  # noqa: E402
from retold.ingest import FakeExtractor, FakeJudge, TableReviewer, WriteRequest  # noqa: E402
from retold.models import ExtractionOutput, Principal, Record, Scope, SessionSummary  # noqa: E402
from retold.policy import (  # noqa: E402
    AdmissionDecision,
    BundleRegistry,
    CandidateVerdict,
    Gap,
    GapDecision,
    UtilityAwareConfig,
    bundle_components,
)
from retold.policy.activation import ProfileBlock  # noqa: E402
from retold.runtime import MemoryRuntime, build_runtime  # noqa: E402
from retold.store import Store  # noqa: E402

_CONFIG = RetoldConfig(embedding=EmbeddingConfig(model="fake-embedder", version="1", dims=8))
MEMORY_TOOLS = {"memory_search", "memory_get", "memory_write", "memory_revise", "memory_forget"}
ROLE = "Research Assistant"


class ScriptedReActLLM(BaseLLM):
    """A text-only model: it returns queued ReAct strings and records every call."""

    queue: deque[str] = Field(default_factory=deque)
    calls: list[list[dict[str, Any]]] = Field(default_factory=list)

    def __init__(self) -> None:
        super().__init__(model="scripted")

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> Any:
        self.calls.append(
            [dict(m) for m in messages] if not isinstance(messages, str) else [{"role": "user", "content": messages}]
        )
        if not self.queue:
            raise AssertionError("the scripted model ran out of replies")
        return self.queue.popleft()

    def supports_stop_words(self) -> bool:
        return False


def _runtime(store: Store) -> MemoryRuntime:
    summary = SessionSummary("The user described how they like answers.", [], [])
    return build_runtime(
        _CONFIG,
        store,
        embedder=FakeEmbedder(dims=8),
        judge=FakeJudge(),
        extractor=FakeExtractor(ExtractionOutput([], summary)),
        reviewer=TableReviewer(),
    )


def _action(name: str, args: dict[str, Any]) -> str:
    return f"Thought: I should use {name}.\nAction: {name}\nAction Input: {json.dumps(args)}"


def _final(answer: str) -> str:
    return f"Thought: I now know the final answer.\nFinal Answer: {answer}"


class Driver:
    """A ContractDriver over a real crew: one task per turn, the same session throughout."""

    def __init__(self, runtime: MemoryRuntime, adapter: CrewAIMemoryAdapter, principal: Principal) -> None:
        self.runtime = runtime
        self.adapter = adapter
        self.principal = principal
        self.model = ScriptedReActLLM()
        self.tool_results: list[str] = []
        original = adapter.call_tool

        def capture(name: str, arguments: Any) -> str:
            result = original(name, arguments)
            self.tool_results.append(result)
            return result

        adapter.call_tool = capture  # type: ignore[method-assign]
        self.agent = Agent(
            role=ROLE,
            goal="Answer the user's request.",
            backstory=adapter.policy_text("You are a careful assistant."),
            llm=adapter.wrap_llm(self.model),
            tools=adapter.tools(),
            verbose=False,
            max_iter=6,
        )
        self.inputs = {"user_id": principal.user_id, "session_id": principal.session_id}

    def run_turn(self, turn: ScriptedTurn) -> TurnOutcome:
        self.tool_results = []
        for name, args in turn.tool_calls:
            self.model.queue.append(_action(name, args))
        self.model.queue.append(_final(turn.answer))
        self.model.queue.extend(_final(text) for text in turn.followups)
        task = Task(description=turn.user, expected_output="A short answer.", agent=self.agent)
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            step_callback=self.adapter.step_callback,
            task_callback=self.adapter.task_callback,
            verbose=False,
        )
        output = crew.kickoff(inputs=self.inputs)
        return TurnOutcome(str(output.raw), list(self.tool_results))

    def end_session(self) -> None:
        self.adapter.end_session()

    def wait_for_extraction(self) -> None:
        for _ in range(100):
            session = self.runtime.store.get_session(self.principal.session_id or "")
            if session is not None and session.extracted_at is not None:
                return
            time.sleep(0.05)
        raise AssertionError("extraction did not complete")


@pytest.fixture
def store(tmp_path: Path) -> Store:
    database = Store(tmp_path / "memory.sqlite")
    host = MemoryHost(database)
    for user in ("aditya", "priya"):
        host.grant(agent_id_from_role(ROLE), Scope(kind="user", id=user), read=True, write=True)
        host.provision_user(user)
    yield database
    database.close()


def _driver(store: Store, user: str = "aditya", session: str = "crew-1", **adapter_kwargs: Any) -> Driver:
    runtime = _runtime(store)
    principal = principal_from_inputs({"user_id": user, "session_id": session}, ROLE)
    adapter = CrewAIMemoryAdapter(runtime, principal, **adapter_kwargs)
    return Driver(runtime, adapter, principal)


def test_principal_comes_from_crew_inputs_and_the_role_slug() -> None:
    principal = principal_from_inputs({"user_id": "u", "session_id": "s", "project_id": "p"}, "Research Assistant")
    assert principal == Principal("research-assistant", "u", "s", "p")
    with pytest.raises(ValueError, match="session_id"):
        principal_from_inputs({"user_id": "u"}, ROLE)
    with pytest.raises(ValueError):
        agent_id_from_role("///")


def test_registers_five_tools_with_pydantic_argument_models(store: Store) -> None:
    driver = _driver(store)
    tools = driver.adapter.tools()
    assert {tool.name for tool in tools} == MEMORY_TOOLS
    write = next(tool for tool in tools if tool.name == "memory_write")
    fields = write.args_schema.model_fields
    assert {"type", "content", "source_kind"} <= set(fields)
    assert fields["content"].is_required() and not fields["evidence"].is_required()
    assert {tool.name for tool in _driver(store, trigger_mode="auto").adapter.tools()} == MEMORY_TOOLS - {
        "memory_search"
    }


def test_contract_suite_passes_end_to_end(store: Store) -> None:
    run_contract(_driver(store))


def test_two_users_on_one_store_are_isolated(store: Store) -> None:
    first = _driver(store, "aditya", "crew-a")
    second = _driver(store, "priya", "crew-b")
    run_contract(first)
    run_contract(second)
    assert_isolated(first, second)


def test_turn_structure_is_task_description_then_steps(store: Store) -> None:
    driver = _driver(store)
    driver.run_turn(
        ScriptedTurn("What editor do I like?", tool_calls=[("memory_search", {"queries": ["editor"]})], answer="Vim.")
    )
    turns = driver.runtime.store.session_turns("crew-1")
    assert [turn.role for turn in turns] == ["user", "tool", "assistant"]
    assert turns[0].content == "What editor do I like?"
    assert turns[-1].content == "Vim."
    row = driver.runtime.store.connection.execute(
        "SELECT context, trigger FROM search_log ORDER BY at DESC LIMIT 1"
    ).fetchone()
    assert row["trigger"] == "tool" and "User: What editor do I like?" in row["context"]


class FakeGap:
    def plan(
        self, turn: str, public_context: str | None, ambient_profile: ProfileBlock, inventory: Sequence[str]
    ) -> GapDecision:
        return GapDecision([Gap("preference", "answer style preference")], "fake-gap", "ok")


class FakeJudgeAdmitAll:
    def admit(
        self,
        turn: str,
        public_context: str | None,
        ambient_profile: ProfileBlock,
        draft: str,
        candidates: Sequence[Record],
    ) -> AdmissionDecision:
        verdicts = [CandidateVerdict(record.id, "helpful", "would change the answer") for record in candidates]
        return AdmissionDecision([record.id for record in candidates], verdicts, "fake-judge", "ok")


def _seed_preference(driver: Driver) -> Record:
    result = driver.runtime.ingestor.write(
        driver.principal,
        WriteRequest(
            type="semantic", content=CANDIDATE, source_kind="agent_inference", evidence=None, attribute="answer_style"
        ),
    )
    driver.runtime.embedder.set_similarity("answer style preference", CANDIDATE, 0.9)  # type: ignore[attr-defined]
    record = driver.runtime.store.get_record(result.record_id or "")
    assert record is not None
    return record


def test_hybrid_mode_appends_the_recalled_block_to_the_task_prompt(store: Store) -> None:
    driver = _driver(store, trigger_mode="hybrid")
    record = _seed_preference(driver)
    driver.runtime.embedder.set_similarity("How should you answer my questions about answers", CANDIDATE, 0.9)  # type: ignore[attr-defined]
    driver.run_turn(ScriptedTurn("How should you answer my questions about answers", answer="Concisely."))
    first_call = driver.model.calls[0]
    user_messages = [m for m in first_call if m["role"] == "user"]
    assert any("recalled memory" in str(m["content"]) and record.id in str(m["content"]) for m in user_messages)
    assert [turn.role for turn in store.session_turns("crew-1")] == ["user", "assistant"]
    row = store.connection.execute("SELECT trigger FROM search_log ORDER BY at DESC LIMIT 1").fetchone()
    assert row["trigger"] == "auto"


def test_utility_aware_shadow_mode_changes_nothing_served_or_stored(store: Store) -> None:
    config = UtilityAwareConfig(
        gap_enabled=True, admission_mode="hosted_judge", shadow=True, bundle={"planner": "fake"}
    )
    driver = _driver(
        store,
        memory_mode="utility_aware",
        utility_config=config,
        gap_policy=FakeGap(),
        admission_policy=FakeJudgeAdmitAll(),
    )
    record = _seed_preference(driver)
    before = store.connection.execute("SELECT id, status, activation FROM records ORDER BY id").fetchall()

    outcome = driver.run_turn(ScriptedTurn("How should you answer?", answer="Draft answer."))

    assert outcome.answer == "Draft answer."
    assert len(driver.model.calls) == 1
    decision = driver.adapter.decisions[-1]
    assert decision.shadow is True and decision.disposition == "shadow_would_regenerate"
    assert decision.admitted_ids == [record.id]
    after = store.connection.execute("SELECT id, status, activation FROM records ORDER BY id").fetchall()
    assert [tuple(row) for row in before] == [tuple(row) for row in after]


def test_utility_aware_active_mode_regenerates_with_the_admitted_record(store: Store) -> None:
    config = UtilityAwareConfig(
        gap_enabled=True, admission_mode="hosted_judge", shadow=False, bundle={"planner": "fake"}
    )
    registry = BundleRegistry(store)
    registry.record(bundle_components(config), passed=True, evidence="tests", recorded_by="tester")
    driver = _driver(
        store,
        memory_mode="utility_aware",
        utility_config=config,
        gap_policy=FakeGap(),
        admission_policy=FakeJudgeAdmitAll(),
        registry=registry,
    )
    record = _seed_preference(driver)

    outcome = driver.run_turn(
        ScriptedTurn("How should you answer?", answer="Draft answer.", followups=["Regenerated with memory."])
    )

    assert outcome.answer == "Regenerated with memory."
    assert len(driver.model.calls) == 2
    assert any(record.content in str(m["content"]) for m in driver.model.calls[1] if m["role"] == "user")
    assert driver.adapter.decisions[-1].disposition == "regenerated"
    assert store.session_turns("crew-1")[-1].content == "Regenerated with memory."


def test_ambient_profile_is_injected_into_the_system_message(store: Store) -> None:
    config = UtilityAwareConfig(gap_enabled=False, admission_mode="disabled", shadow=True, bundle={"planner": "fake"})
    driver = _driver(store, memory_mode="utility_aware", utility_config=config)
    record = _seed_preference(driver)
    store.update_status(record.id, "confirmed")
    store.set_activation(record.id, "ambient")
    driver.run_turn(ScriptedTurn("Tell me something.", answer="Something."))
    system = driver.model.calls[0][0]
    assert system["role"] == "system" and CANDIDATE in str(system["content"])


def test_tool_call_drafts_skip_the_utility_path(store: Store) -> None:
    config = UtilityAwareConfig(
        gap_enabled=True, admission_mode="hosted_judge", shadow=True, bundle={"planner": "fake"}
    )
    driver = _driver(
        store,
        memory_mode="utility_aware",
        utility_config=config,
        gap_policy=FakeGap(),
        admission_policy=FakeJudgeAdmitAll(),
    )
    driver.run_turn(ScriptedTurn("Look it up", tool_calls=[("memory_search", {"queries": ["vim"]})], answer="Done."))
    assert driver.adapter.decisions == []
