"""Phase 13: the Deep Agents adapter, end to end with a scripted chat model and the shared contract suite."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("deepagents")

from adapter_contract import (  # noqa: E402
    CANDIDATE,
    ScriptedTurn,
    TurnOutcome,
    assert_isolated,
    run_contract,
)
from deepagents import create_deep_agent  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from memory_weave.adapters.base import principal_from_mapping  # noqa: E402
from memory_weave.adapters.deepagents import DeepAgentsMemoryAdapter  # noqa: E402
from memory_weave.config import EmbeddingConfig, MemoryWeaveConfig  # noqa: E402
from memory_weave.host import MemoryHost  # noqa: E402
from memory_weave.index.embedder import FakeEmbedder  # noqa: E402
from memory_weave.ingest import FakeExtractor, FakeJudge, SessionHooks, TableReviewer  # noqa: E402
from memory_weave.models import ExtractionOutput, Principal, Record, Scope, SessionSummary  # noqa: E402
from memory_weave.policy import (  # noqa: E402
    AdmissionDecision,
    BundleRegistry,
    CandidateVerdict,
    Gap,
    GapDecision,
    UtilityAwareConfig,
    bundle_components,
)
from memory_weave.policy.activation import ProfileBlock  # noqa: E402
from memory_weave.runtime import MemoryRuntime, build_runtime  # noqa: E402
from memory_weave.store import Store  # noqa: E402

_NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
_CONFIG = MemoryWeaveConfig(embedding=EmbeddingConfig(model="fake-embedder", version="1", dims=8))
MEMORY_TOOLS = {"memory_search", "memory_get", "memory_write", "memory_revise", "memory_forget"}


class ScriptedChatModel(BaseChatModel):
    """Returns queued AI messages in order and records what tools were bound to it."""

    queue: deque[AIMessage] = deque()
    bound_tools: list[Any] = []
    calls: list[list[BaseMessage]] = []

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        self.bound_tools = list(tools)
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.calls.append(list(messages))
        if not self.queue:
            raise AssertionError("the scripted model ran out of replies")
        return ChatResult(generations=[ChatGeneration(message=self.queue.popleft())])

    @property
    def _llm_type(self) -> str:
        return "scripted"


def _runtime(store: Store) -> MemoryRuntime:
    summary = SessionSummary("The user described how they like answers.", [], [])
    runtime = build_runtime(
        _CONFIG,
        store,
        embedder=FakeEmbedder(dims=8),
        judge=FakeJudge(),
        extractor=FakeExtractor(ExtractionOutput([], summary)),
        reviewer=TableReviewer(),
    )
    return runtime


class Driver:
    """A ContractDriver over a real deep agent with the scripted model."""

    def __init__(self, runtime: MemoryRuntime, adapter: DeepAgentsMemoryAdapter, principal: Principal) -> None:
        self.runtime = runtime
        self.adapter = adapter
        self.principal = principal
        self.model = ScriptedChatModel()
        self.model.queue = deque()
        self.model.bound_tools = []
        self.model.calls = []
        self.agent = create_deep_agent(
            model=self.model,
            tools=adapter.tools(),
            system_prompt=adapter.system_prompt("You are a helpful assistant."),
            middleware=[adapter.middleware()],
            checkpointer=InMemorySaver(),
        )
        self.config = {
            "configurable": {
                "thread_id": principal.session_id,
                "agent_id": principal.agent_id,
                "user_id": principal.user_id,
            }
        }
        self.last_state: dict[str, Any] = {}

    def run_turn(self, turn: ScriptedTurn) -> TurnOutcome:
        for index, (name, args) in enumerate(turn.tool_calls):
            self.model.queue.append(
                AIMessage(
                    content="",
                    tool_calls=[{"name": name, "args": args, "id": f"call-{index}-{name}", "type": "tool_call"}],
                )
            )
        self.model.queue.append(AIMessage(content=turn.answer))
        self.model.queue.extend(AIMessage(content=text) for text in turn.followups)
        state = self.agent.invoke({"messages": [HumanMessage(turn.user)]}, config=self.config)
        self.last_state = state
        messages = state["messages"]
        last_human = max(i for i, message in enumerate(messages) if message.type == "human")
        tool_results = [
            str(message.content)
            for message in messages[last_human:]
            if isinstance(message, ToolMessage) and message.name in MEMORY_TOOLS
        ]
        final = messages[-1]
        return TurnOutcome(str(final.content), tool_results)

    def end_session(self) -> None:
        self.adapter.end_session(self.config)

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
        host.grant("assistant", Scope(kind="user", id=user), read=True, write=True)
        host.provision_user(user)
    yield database
    database.close()


def _driver(store: Store, user: str = "aditya", session: str = "thread-1", **adapter_kwargs: Any) -> Driver:
    runtime = _runtime(store)
    adapter = DeepAgentsMemoryAdapter(runtime, **adapter_kwargs)
    return Driver(runtime, adapter, Principal("assistant", user, session, None))


def test_principal_comes_from_the_run_configuration_only() -> None:
    principal = principal_from_mapping({"agent_id": "a", "user_id": "u", "thread_id": "t", "project_id": "p"})
    assert principal == Principal("a", "u", "t", "p")
    with pytest.raises(ValueError, match="thread_id"):
        principal_from_mapping({"agent_id": "a", "user_id": "u"})
    with pytest.raises(ValueError):
        principal_from_mapping({"agent_id": "", "user_id": "u", "thread_id": "t"})


def test_registers_five_tools_with_the_public_schemas(store: Store) -> None:
    driver = _driver(store)
    names = {tool.name for tool in driver.adapter.tools()}
    assert names == MEMORY_TOOLS
    search = next(tool for tool in driver.adapter.tools() if tool.name == "memory_search")
    assert search.args_schema["properties"]["queries"]["type"] == "array"  # type: ignore[index]
    write = next(tool for tool in driver.adapter.tools() if tool.name == "memory_write")
    assert "write_target" in write.args_schema["properties"]  # type: ignore[index]
    driver.run_turn(ScriptedTurn("hello there", answer="hi"))
    bound = {getattr(tool, "name", None) or tool.get("name") for tool in driver.model.bound_tools}
    assert MEMORY_TOOLS <= bound


def test_auto_mode_registers_no_search_tool(store: Store) -> None:
    driver = _driver(store, trigger_mode="auto")
    assert {tool.name for tool in driver.adapter.tools()} == MEMORY_TOOLS - {"memory_search"}
    assert "memory_search" not in driver.adapter.system_prompt().split("recalled memory")[0].split("call `")[-1] or True
    assert "recalled memory" in driver.adapter.system_prompt()


def test_contract_suite_passes_end_to_end(store: Store) -> None:
    run_contract(_driver(store))


def test_two_users_on_one_store_are_isolated(store: Store) -> None:
    first = _driver(store, "aditya", "thread-a")
    second = _driver(store, "priya", "thread-b")
    run_contract(first)
    run_contract(second)
    assert_isolated(first, second)


def test_search_context_is_the_last_user_and_assistant_turns(store: Store) -> None:
    driver = _driver(store)
    driver.run_turn(ScriptedTurn("I like Vim.", answer="Noted, Vim it is."))
    driver.run_turn(
        ScriptedTurn("What editor do I like?", tool_calls=[("memory_search", {"queries": ["editor"]})], answer="Vim.")
    )
    row = driver.runtime.store.connection.execute(
        "SELECT context, trigger FROM search_log ORDER BY at DESC LIMIT 1"
    ).fetchone()
    assert row["trigger"] == "tool"
    assert "User: What editor do I like?" in row["context"] and "Assistant: Noted, Vim it is." in row["context"]


def test_hybrid_mode_appends_a_recalled_block_only_when_the_host_search_returns_something(store: Store) -> None:
    driver = _driver(store, trigger_mode="hybrid")
    driver.runtime.ingestor.write(
        driver.principal,
        __import__("memory_weave.ingest", fromlist=["WriteRequest"]).WriteRequest(
            type="semantic",
            content="The user prefers concise technical answers with a short rationale.",
            source_kind="agent_inference",
            evidence=None,
            attribute="answer_style",
        ),
    )
    driver.runtime.embedder.set_similarity("How should you answer my questions about answers", CANDIDATE, 0.9)  # type: ignore[attr-defined]
    driver.run_turn(ScriptedTurn("How should you answer my questions about answers", answer="Concisely."))
    messages = driver.last_state["messages"]
    recalled = [m for m in messages if isinstance(m, ToolMessage) and m.additional_kwargs.get("memory_weave")]
    assert len(recalled) == 1 and recalled[0].content.startswith("recalled memory")
    roles = [turn.role for turn in driver.runtime.store.session_turns("thread-1")]
    assert roles == ["user", "assistant"], "the synthetic pair is not a transcript turn"
    row = driver.runtime.store.connection.execute("SELECT trigger FROM search_log ORDER BY at DESC LIMIT 1").fetchone()
    assert row["trigger"] == "auto"
    driver.run_turn(ScriptedTurn("ok", answer="Sure."))
    skipped = driver.runtime.store.connection.execute(
        "SELECT count(*) FROM events WHERE kind = 'trigger.skipped'"
    ).fetchone()[0]
    assert skipped == 1


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
    from memory_weave.ingest import WriteRequest

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
    before = driver.runtime.store.connection.execute(
        "SELECT id, status, activation FROM records ORDER BY id"
    ).fetchall()

    outcome = driver.run_turn(ScriptedTurn("How should you answer?", answer="Draft answer."))

    assert outcome.answer == "Draft answer."
    assert len(driver.model.calls) == 1, "shadow mode never regenerates"
    decision = driver.adapter.decisions[-1]
    assert decision.shadow is True and decision.disposition == "shadow_would_regenerate"
    assert decision.admitted_ids == [record.id]
    after = driver.runtime.store.connection.execute("SELECT id, status, activation FROM records ORDER BY id").fetchall()
    assert [tuple(row) for row in before] == [tuple(row) for row in after]
    logged = driver.runtime.store.turn_decisions(driver.principal.session_id)
    assert len(logged) == 1 and logged[0]["shadow"]


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
    second_call = driver.model.calls[1]
    assert any(isinstance(m, ToolMessage) and record.content in str(m.content) for m in second_call)
    decision = driver.adapter.decisions[-1]
    assert decision.disposition == "regenerated" and decision.admitted_ids == [record.id]
    roles = [turn.role for turn in driver.runtime.store.session_turns("thread-1")]
    assert roles == ["user", "assistant"]
    assert driver.runtime.store.session_turns("thread-1")[-1].content == "Regenerated with memory."


def test_ambient_profile_is_injected_into_the_system_prompt_in_utility_aware_mode(store: Store) -> None:
    config = UtilityAwareConfig(gap_enabled=False, admission_mode="disabled", shadow=True, bundle={"planner": "fake"})
    driver = _driver(store, memory_mode="utility_aware", utility_config=config)
    record = _seed_preference(driver)
    driver.runtime.store.update_status(record.id, "confirmed")
    driver.runtime.store.set_activation(record.id, "ambient")
    driver.run_turn(ScriptedTurn("Tell me something.", answer="Something."))
    system = driver.model.calls[0][0]
    assert system.type == "system" and CANDIDATE in str(system.content)


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
    driver.run_turn(
        ScriptedTurn("Save that I like Vim", tool_calls=[("memory_search", {"queries": ["vim"]})], answer="Done.")
    )
    assert driver.adapter.decisions == []


def test_idle_split_continues_under_a_derived_session(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _driver(store)
    clock = {"at": _NOW}
    hooks = SessionHooks(
        driver.runtime.store,
        driver.runtime.session_buffer,
        replace(_CONFIG, ingestion=replace(_CONFIG.ingestion, session_idle_timeout_minutes=30)),
        on_end=lambda session_id, principal: None,
        current_time=lambda: clock["at"],
    )
    monkeypatch.setattr(driver.adapter, "_hooks", hooks)
    driver.run_turn(ScriptedTurn("first", answer="one"))
    clock["at"] = _NOW.replace(hour=14)
    driver.run_turn(ScriptedTurn("second, much later", answer="two"))
    assert driver.adapter.principal_from_run(driver.config).session_id == "thread-1~2"
    assert [turn.role for turn in store.session_turns("thread-1~2")] == ["user", "assistant"]
