"""The Deep Agents adapter: LangChain tools, a middleware for turns and host-issued memory, and session end.

The adapter reads identity from the run configuration the host passes to the graph, records every human,
assistant, and tool message as a transcript turn through the session hooks, registers the five memory tools
as LangChain tools, and, in the utility-aware modes, wraps each answer-producing model call in the
orchestrator so memory reaches the model only when a judge says it would change the answer. Shadow mode
runs the same path and never regenerates, so the served answer is the draft.

Requires the ``deepagents`` extra.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.config import get_config
from langgraph.runtime import Runtime
from pydantic import PrivateAttr

from retold.models import Principal, Record, TurnRole
from retold.policy import (
    AdmissionPolicy,
    GapPolicy,
    ProfileAssembler,
    TurnMemoryDecision,
    TurnOptions,
    UtilityAwareConfig,
    UtilityAwareOrchestrator,
)
from retold.runtime import MemoryRuntime
from retold.tools.schemas import tool_schemas
from retold.util import normalize_ws, uuid7

from .base import (
    RECALLED_HEADER,
    MemoryMode,
    TriggerMode,
    policy_text,
    principal_from_mapping,
    recalled_memory_block,
    render_tool_result,
    tool_names_for,
    turn_context,
)

_SYNTHETIC = "retold"


class MemoryTool(BaseTool):
    """One memory tool. The principal is derived from the run configuration at call time, never from input."""

    name: str
    description: str
    args_schema: dict[str, Any]
    _adapter: Any = PrivateAttr(default=None)

    def _run(self, config: RunnableConfig, **kwargs: Any) -> str:
        adapter: DeepAgentsMemoryAdapter = self._adapter
        return adapter.call_tool(self.name, kwargs, config)


class DeepAgentsMemoryAdapter:
    """Wire one ``MemoryRuntime`` into a Deep Agents graph."""

    def __init__(
        self,
        runtime: MemoryRuntime,
        *,
        trigger_mode: TriggerMode | None = None,
        memory_mode: MemoryMode = "tool_only",
        utility_config: UtilityAwareConfig | None = None,
        gap_policy: GapPolicy | None = None,
        admission_policy: AdmissionPolicy | None = None,
        registry: Any = None,
        latency_budget_ms: int | None = None,
    ) -> None:
        self._runtime = runtime
        self._store = runtime.store
        self._handlers = runtime.handlers
        self._hooks = runtime.hooks
        self._config = runtime.config
        self.trigger_mode: TriggerMode = trigger_mode or runtime.config.retrieval.trigger.mode
        self.memory_mode: MemoryMode = memory_mode
        self._latency_budget_ms = latency_budget_ms
        self._lock = threading.Lock()
        self._sessions: dict[str, str] = {}
        self._recorded: dict[str, set[str]] = {}
        self._searched: dict[str, set[str]] = {}
        self._profiles = ProfileAssembler(self._store)
        self._orchestrator: UtilityAwareOrchestrator | None = None
        self.decisions: list[TurnMemoryDecision] = []
        if memory_mode == "utility_aware":
            if utility_config is None:
                raise ValueError("utility_aware mode needs a UtilityAwareConfig.")
            self._orchestrator = UtilityAwareOrchestrator(
                self._store,
                self._retrieve_for_orchestrator,
                gap_policy,
                admission_policy,
                utility_config,
                profile_assembler=self._profiles,
                registry=registry,
                actor="deepagents",
            )

    # -- the framework-facing surface -------------------------------------------------------------------

    def tools(self) -> list[BaseTool]:
        """The memory tools for this trigger mode, with the public schemas and no principal baked in."""

        allowed = set(tool_names_for(self.trigger_mode))
        built: list[BaseTool] = []
        for schema in tool_schemas(write_targets=("personal",)):
            name = str(schema["name"])
            if name not in allowed:
                continue
            tool = MemoryTool(
                name=name,
                description=str(schema["description"]),
                args_schema=cast(dict[str, Any], schema["input_schema"]),
            )
            tool._adapter = self
            built.append(tool)
        return built

    def middleware(self) -> RetoldMiddleware:
        return RetoldMiddleware(self)

    def system_prompt(self, base_prompt: str | None = None) -> str:
        return policy_text(self.trigger_mode, base_prompt)

    def principal_from_run(self, run_context: Any) -> Principal:
        """Identity from ``configurable``; the session id is the thread id, or its idle-split continuation."""

        configurable = _configurable(run_context)
        principal = principal_from_mapping(configurable)
        thread_id = principal.session_id or ""
        with self._lock:
            session_id = self._sessions.get(thread_id, thread_id)
        return replace(principal, session_id=session_id)

    def end_session(self, run_context: Any) -> None:
        principal = self.principal_from_run(run_context)
        self._hooks.on_session_end(principal)

    # -- tools ------------------------------------------------------------------------------------------

    def call_tool(self, name: str, arguments: Mapping[str, Any], config: RunnableConfig) -> str:
        principal = self.principal_from_run(config)
        handler = getattr(self._handlers, name)
        if name == "memory_search":
            context = turn_context(self._store, principal.session_id).text(
                self._config.retrieval.rewrite.max_context_chars
            )
            payload = handler(principal, dict(arguments), context=context, trigger="tool")
        else:
            payload = handler(principal, dict(arguments))
        return render_tool_result(name, arguments, payload)

    # -- transcript --------------------------------------------------------------------------------------

    def sync_messages(self, messages: Sequence[BaseMessage], run_context: Any) -> Principal:
        """Record every message not yet in the transcript, in order; return the principal to keep using."""

        principal = self.principal_from_run(run_context)
        thread_id = str(_configurable(run_context)["thread_id"])
        with self._lock:
            recorded = self._recorded.setdefault(thread_id, set())
            for message in messages:
                key = message.id or f"{id(message)}"
                if key in recorded or _is_synthetic(message):
                    continue
                role = _role(message)
                text = _text(message)
                if role is None or not text.strip():
                    recorded.add(key)
                    continue
                principal = self._hooks.on_turn(principal, role, text)
                self._sessions[thread_id] = principal.session_id or thread_id
                recorded.add(key)
        return principal

    # -- host-issued search (auto and hybrid) -------------------------------------------------------------

    def host_search(self, principal: Principal, user_turn: str, message_id: str) -> list[BaseMessage]:
        """LLD 14.1: one relevance search per user turn, appended as a synthetic tool-call pair when non-empty."""

        thread_id = principal.session_id or ""
        with self._lock:
            done = self._searched.setdefault(thread_id, set())
            if message_id in done:
                return []
            done.add(message_id)
        trigger = self._config.retrieval.trigger
        if len(normalize_ws(user_turn)) < trigger.auto_min_query_chars:
            self._store.append_event(
                "trigger.skipped", principal.agent_id, None, None, {"reason": "short_turn", "session_id": thread_id}
            )
            return []
        context = turn_context(self._store, principal.session_id).text(self._config.retrieval.rewrite.max_context_chars)
        payload = self._handlers.memory_search(
            principal, {"queries": [user_turn], "k": trigger.auto_k}, context=context, trigger="auto"
        )
        block = recalled_memory_block([user_turn], payload)
        if block is None:
            return []
        return _synthetic_pair([user_turn], block)

    # -- utility-aware path ------------------------------------------------------------------------------

    def _retrieve_for_orchestrator(self, principal: Principal, queries: list[str], context: str) -> list[Record]:
        payload = self._handlers.memory_search(
            principal,
            {"queries": queries[:3], "k": self._config.retrieval.trigger.auto_k},
            context=context[: self._config.retrieval.rewrite.max_context_chars] or None,
            trigger="auto",
        )
        if payload.get("ok") is not True:
            return []
        ids = [str(entry["record"]["id"]) for entry in cast(list[dict[str, Any]], payload.get("results", []))]
        return self._store.get_records(ids)

    def utility_turn(
        self,
        principal: Principal,
        turn: str,
        public_context: str | None,
        draft: Callable[[], str],
        regenerate: Callable[[list[Record]], str],
    ) -> TurnMemoryDecision:
        assert self._orchestrator is not None
        options = TurnOptions(self._latency_budget_ms)
        decision = self._orchestrator.prepare_turn(principal, turn, public_context, draft, regenerate, options)
        self.decisions.append(decision)
        return decision

    def profile_block(self, principal: Principal) -> str:
        return self._profiles.build(principal).text


class RetoldMiddleware(AgentMiddleware[Any, Any, Any]):
    """Turn capture, host-issued recall, and the utility-aware wrap around answer-producing model calls."""

    def __init__(self, adapter: DeepAgentsMemoryAdapter) -> None:
        self._adapter = adapter

    @property
    def name(self) -> str:
        return "RetoldMiddleware"

    def before_agent(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:
        config = get_config()
        self._adapter._hooks.on_session_start(self._adapter.principal_from_run(config))
        return None

    def before_model(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:
        config = get_config()
        messages: list[BaseMessage] = list(state["messages"])
        principal = self._adapter.sync_messages(messages, config)
        if self._adapter.trigger_mode == "tool_only" or not messages or messages[-1].type != "human":
            return None
        last = messages[-1]
        appended = self._adapter.host_search(principal, _text(last), last.id or str(id(last)))
        return {"messages": appended} if appended else None

    def after_agent(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:
        self._adapter.sync_messages(list(state["messages"]), get_config())
        return None

    def wrap_model_call(self, request: ModelRequest[Any], handler: Callable[[ModelRequest[Any]], Any]) -> Any:
        adapter = self._adapter
        messages = list(request.messages)
        answer_turn = bool(messages) and messages[-1].type == "human"
        if adapter.memory_mode != "utility_aware" or not answer_turn:
            return handler(request)
        config = get_config()
        principal = adapter.principal_from_run(config)
        profile = adapter.profile_block(principal)
        served_request = request
        if profile:
            served_request = replace(request, system_message=_with_profile(request.system_message, profile))
        draft_response = handler(served_request)
        draft_message = _last_ai(draft_response)
        if draft_message is None or draft_message.tool_calls:
            return draft_response
        turn = _text(messages[-1])
        public_context = turn_context(adapter._store, principal.session_id).text(
            adapter._config.retrieval.rewrite.max_context_chars
        )
        regenerated: dict[str, Any] = {}

        def baseline() -> str:
            return _text(draft_message)

        def regenerate(records: list[Record]) -> str:
            block = f"{RECALLED_HEADER}\n" + "\n\n".join(f"id={record.id}\n{record.content}" for record in records)
            pair = _synthetic_pair([turn], block)
            response = handler(replace(served_request, messages=cast(Any, [*messages, *pair])))
            regenerated["response"] = response
            message = _last_ai(response)
            return _text(message) if message is not None else ""

        decision = adapter.utility_turn(principal, turn, public_context, baseline, regenerate)
        if decision.final is not None and "response" in regenerated:
            return regenerated["response"]
        return draft_response


def _configurable(run_context: Any) -> Mapping[str, Any]:
    if isinstance(run_context, Mapping) and "configurable" in run_context:
        return cast(Mapping[str, Any], run_context["configurable"])
    if isinstance(run_context, Mapping):
        return run_context
    raise ValueError("The run context must be a RunnableConfig with a configurable mapping.")


def _role(message: BaseMessage) -> TurnRole | None:
    if message.type == "human":
        return "user"
    if message.type == "ai":
        return "assistant"
    if message.type == "tool":
        return "tool"
    return None


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(part for part in parts if part)


def _is_synthetic(message: BaseMessage) -> bool:
    return bool(getattr(message, "additional_kwargs", {}).get(_SYNTHETIC))


def _synthetic_pair(queries: Sequence[str], block: str) -> list[BaseMessage]:
    call_id = f"recall-{uuid7()}"
    call = AIMessage(
        content="",
        tool_calls=[{"name": "memory_search", "args": {"queries": list(queries)}, "id": call_id, "type": "tool_call"}],
        additional_kwargs={_SYNTHETIC: "recalled"},
        id=f"recall-ai-{call_id}",
    )
    result = ToolMessage(
        content=block,
        tool_call_id=call_id,
        name="memory_search",
        additional_kwargs={_SYNTHETIC: "recalled"},
        id=f"recall-tool-{call_id}",
    )
    return [call, result]


def _with_profile(system_message: SystemMessage | None, profile: str) -> SystemMessage:
    base = _text(system_message) if system_message is not None else ""
    text = f"{base}\n\n{profile}".strip()
    return SystemMessage(content=text)


def _last_ai(response: Any) -> AIMessage | None:
    if isinstance(response, AIMessage):
        return response
    result = getattr(response, "result", None)
    if isinstance(result, list) and result and isinstance(result[-1], AIMessage):
        return result[-1]
    return None


def dump_decision(decision: TurnMemoryDecision) -> str:
    return json.dumps(
        {
            "disposition": decision.disposition,
            "gaps": [gap.query for gap in decision.gaps],
            "admitted": decision.admitted_ids,
            "shadow": decision.shadow,
        }
    )
