"""The CrewAI adapter: tool wrappers, step callbacks, and an LLM proxy that carries the memory policy.

CrewAI drives a text-only model through a ReAct loop and hands the task to every model call, so the
adapter has three parts. Tools wrap the handlers as ``BaseTool`` subclasses with pydantic argument
models. Step callbacks record each tool result as a tool turn and each final answer as an assistant turn.
The LLM proxy records the task description as the user turn on the first call of each task, adds the
ambient profile to the system message in the utility-aware modes, appends the host-issued recalled block
in ``auto`` and ``hybrid``, and wraps an answer-producing first call in the orchestrator exactly as the
Deep Agents middleware does. Shadow mode serves the draft by construction.

Identity is bound when the adapter is built, from the crew inputs and the agent role the host supplies; a
CrewAI tool call carries no per-call configuration. Requires the ``crewai`` extra.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Mapping
from typing import Any, cast

from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool
from pydantic import PrivateAttr

from memory_weave.models import Principal, Record
from memory_weave.policy import (
    AdmissionPolicy,
    GapPolicy,
    ProfileAssembler,
    TurnMemoryDecision,
    TurnOptions,
    UtilityAwareConfig,
    UtilityAwareOrchestrator,
)
from memory_weave.runtime import MemoryRuntime
from memory_weave.tools.schemas import tool_schemas
from memory_weave.util import normalize_ws

from .base import (
    RECALLED_HEADER,
    MemoryMode,
    TriggerMode,
    args_model_from_schema,
    policy_text,
    recalled_memory_block,
    render_tool_result,
    tool_names_for,
    turn_context,
)

_FINAL_ANSWER = "Final Answer:"
_ACTION = re.compile(r"^\s*Action\s*:", re.MULTILINE)


def agent_id_from_role(role: str) -> str:
    """The agent id is the role slug: lowercase, hyphenated, and free of the private-scope separator."""

    slug = re.sub(r"[^a-z0-9]+", "-", role.lower()).strip("-")
    if not slug:
        raise ValueError("The agent role must contain at least one letter or digit.")
    return slug


def principal_from_inputs(inputs: Mapping[str, Any], agent_role: str) -> Principal:
    """The host supplies ``user_id`` and ``session_id`` through the crew inputs; the role names the agent."""

    try:
        user_id = str(inputs["user_id"])
        session_id = str(inputs["session_id"])
    except KeyError as missing:
        raise ValueError(f"Crew inputs must carry user_id and session_id: missing {missing}.") from None
    if not user_id or not session_id:
        raise ValueError("user_id and session_id must be non-empty.")
    project = inputs.get("project_id")
    return Principal(agent_id_from_role(agent_role), user_id, session_id, str(project) if project else None)


class CrewAIMemoryAdapter:
    """Wire one ``MemoryRuntime`` into a crew for one principal."""

    def __init__(
        self,
        runtime: MemoryRuntime,
        principal: Principal,
        *,
        trigger_mode: TriggerMode | None = None,
        memory_mode: MemoryMode = "tool_only",
        utility_config: UtilityAwareConfig | None = None,
        gap_policy: GapPolicy | None = None,
        admission_policy: AdmissionPolicy | None = None,
        registry: Any = None,
        latency_budget_ms: int | None = None,
    ) -> None:
        if principal.session_id is None:
            raise ValueError("The CrewAI adapter needs a principal with a session id.")
        self._runtime = runtime
        self._store = runtime.store
        self._handlers = runtime.handlers
        self._hooks = runtime.hooks
        self._config = runtime.config
        self._principal = principal
        self.trigger_mode: TriggerMode = trigger_mode or runtime.config.retrieval.trigger.mode
        self.memory_mode: MemoryMode = memory_mode
        self._latency_budget_ms = latency_budget_ms
        self._lock = threading.Lock()
        self._current_task: str | None = None
        self._searched_tasks: set[str] = set()
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
                actor="crewai",
            )

    @classmethod
    def for_crew(
        cls, runtime: MemoryRuntime, inputs: Mapping[str, Any], agent_role: str, **options: Any
    ) -> CrewAIMemoryAdapter:
        return cls(runtime, principal_from_inputs(inputs, agent_role), **options)

    # -- the framework-facing surface -------------------------------------------------------------------

    @property
    def principal(self) -> Principal:
        with self._lock:
            return self._principal

    def principal_from_run(self, run_context: Any) -> Principal:
        del run_context
        return self.principal

    def tools(self) -> list[BaseTool]:
        allowed = set(tool_names_for(self.trigger_mode))
        built: list[BaseTool] = []
        for schema in tool_schemas(write_targets=("personal",)):
            name = str(schema["name"])
            if name not in allowed:
                continue
            built.append(
                MemoryCrewTool(
                    name=name,
                    description=str(schema["description"]),
                    args_schema=args_model_from_schema(name, cast(Mapping[str, Any], schema["input_schema"])),
                    adapter=self,
                )
            )
        return built

    def policy_text(self, base_prompt: str | None = None) -> str:
        """LLD 13 text for the agent's backstory; CrewAI has no separate system prompt slot."""

        return policy_text(self.trigger_mode, base_prompt)

    def wrap_llm(self, llm: BaseLLM) -> BaseLLM:
        return MemoryLLM(llm, self)

    def step_callback(self, step: Any) -> None:
        """Record an action's tool result as a tool turn and a finish's output as an assistant turn.

        CrewAI invokes the callback twice per tool use, first with the bare tool result and then with the
        action that carries the same result, so only steps that name a tool or carry a final output count.
        """

        if hasattr(step, "tool"):
            result = getattr(step, "result", None)
            if result is not None and str(result).strip():
                self._record("tool", str(result))
            return
        output = getattr(step, "output", None)
        if output is not None and not hasattr(step, "result") and str(output).strip():
            self._record("assistant", str(output))

    def task_callback(self, output: Any) -> None:
        """Record the task output when no step callback recorded it, so a host may use either hook."""

        raw = str(getattr(output, "raw", output) or "")
        if not raw.strip():
            return
        turns = self._store.session_turns(self.principal.session_id or "")
        if turns and turns[-1].role == "assistant" and turns[-1].content == raw:
            return
        self._record("assistant", raw)

    def end_session(self, run_context: Any = None) -> None:
        del run_context
        self._hooks.on_session_end(self.principal)

    # -- tools ------------------------------------------------------------------------------------------

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> str:
        principal = self.principal
        cleaned = {key: value for key, value in arguments.items() if value is not None}
        handler = getattr(self._handlers, name)
        if name == "memory_search":
            context = self._context(principal)
            payload = handler(principal, cleaned, context=context, trigger="tool")
        else:
            payload = handler(principal, cleaned)
        return render_tool_result(name, cleaned, payload)

    # -- the proxy's hooks -------------------------------------------------------------------------------

    def begin_task(self, description: str) -> Principal:
        """The first model call of a task records its description as the user turn."""

        with self._lock:
            already = self._current_task == description
        if already:
            return self.principal
        principal = self._record("user", description)
        with self._lock:
            self._current_task = description
        return principal

    def host_search_block(self, principal: Principal, description: str) -> str | None:
        if self.trigger_mode == "tool_only":
            return None
        with self._lock:
            if description in self._searched_tasks:
                return None
            self._searched_tasks.add(description)
        trigger = self._config.retrieval.trigger
        if len(normalize_ws(description)) < trigger.auto_min_query_chars:
            self._store.append_event(
                "trigger.skipped",
                principal.agent_id,
                None,
                None,
                {"reason": "short_turn", "session_id": principal.session_id},
            )
            return None
        payload = self._handlers.memory_search(
            principal, {"queries": [description], "k": trigger.auto_k}, context=self._context(principal), trigger="auto"
        )
        return recalled_memory_block([description], payload)

    def profile_block(self, principal: Principal) -> str:
        return self._profiles.build(principal).text

    def utility_turn(
        self,
        principal: Principal,
        turn: str,
        draft: Callable[[], str],
        regenerate: Callable[[list[Record]], str],
    ) -> TurnMemoryDecision:
        assert self._orchestrator is not None
        decision = self._orchestrator.prepare_turn(
            principal, turn, self._context(principal), draft, regenerate, TurnOptions(self._latency_budget_ms)
        )
        self.decisions.append(decision)
        return decision

    # -- internals --------------------------------------------------------------------------------------

    def _record(self, role: Any, text: str) -> Principal:
        with self._lock:
            principal = self._hooks.on_turn(self._principal, role, text)
            self._principal = principal
            return principal

    def _context(self, principal: Principal) -> str | None:
        return turn_context(self._store, principal.session_id, steps_count=True).text(
            self._config.retrieval.rewrite.max_context_chars
        )

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


class MemoryCrewTool(BaseTool):
    """One memory tool for a crew; the principal is the adapter's, fixed by the host at construction."""

    adapter: Any = None

    def _run(self, *args: Any, **kwargs: Any) -> str:
        adapter: CrewAIMemoryAdapter = self.adapter
        arguments: dict[str, Any] = dict(kwargs)
        if args and isinstance(args[0], Mapping):
            arguments = {**cast(Mapping[str, Any], args[0]), **arguments}
        return adapter.call_tool(self.name, arguments)


class MemoryLLM(BaseLLM):
    """A proxy around the host's LLM. The host's model does the talking; the proxy applies the memory policy."""

    _inner: Any = PrivateAttr(default=None)
    _adapter: Any = PrivateAttr(default=None)

    def __init__(self, inner: BaseLLM, adapter: CrewAIMemoryAdapter) -> None:
        super().__init__(model=getattr(inner, "model", "memory-weave-proxy"))
        self._inner = inner
        self._adapter = adapter

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
        adapter = self._adapter
        message_list = _as_messages(messages)
        description = str(getattr(from_task, "description", "") or "")
        first_call = bool(description) and not any(message.get("role") == "assistant" for message in message_list)
        principal = adapter.begin_task(description) if description else adapter.principal

        def inner_call(prepared: list[dict[str, Any]]) -> Any:
            return self._inner.call(
                cast(Any, prepared),
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )

        prepared = [dict(message) for message in message_list]
        if adapter.memory_mode == "utility_aware":
            profile = adapter.profile_block(principal)
            if profile:
                prepared = _with_profile(prepared, profile)
        if first_call:
            block = adapter.host_search_block(principal, description)
            if block is not None:
                prepared = _append_to_last_user(prepared, block)
        if adapter.memory_mode != "utility_aware" or not first_call:
            return inner_call(prepared)

        draft = inner_call(prepared)
        draft_text = str(draft)
        if not _is_final_answer(draft_text):
            return draft
        regenerated: dict[str, Any] = {}

        def baseline() -> str:
            return _answer_of(draft_text)

        def regenerate(records: list[Record]) -> str:
            block = f"{RECALLED_HEADER}\n" + "\n\n".join(f"id={record.id}\n{record.content}" for record in records)
            response = inner_call(_append_to_last_user(prepared, block))
            regenerated["response"] = response
            return _answer_of(str(response))

        decision = adapter.utility_turn(principal, description, baseline, regenerate)
        if decision.final is not None and "response" in regenerated:
            return regenerated["response"]
        return draft

    def supports_stop_words(self) -> bool:
        return bool(self._inner.supports_stop_words())

    def get_context_window_size(self) -> int:
        return int(self._inner.get_context_window_size())


def _as_messages(messages: Any) -> list[dict[str, Any]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return [dict(message) for message in messages]


def _with_profile(messages: list[dict[str, Any]], profile: str) -> list[dict[str, Any]]:
    prepared = [dict(message) for message in messages]
    if prepared and prepared[0].get("role") == "system":
        prepared[0]["content"] = f"{prepared[0].get('content', '')}\n\n{profile}".strip()
        return prepared
    return [{"role": "system", "content": profile}, *prepared]


def _append_to_last_user(messages: list[dict[str, Any]], block: str) -> list[dict[str, Any]]:
    prepared = [dict(message) for message in messages]
    for index in range(len(prepared) - 1, -1, -1):
        if prepared[index].get("role") == "user":
            prepared[index]["content"] = f"{prepared[index].get('content', '')}\n\n{block}"
            return prepared
    prepared.append({"role": "user", "content": block})
    return prepared


def _is_final_answer(text: str) -> bool:
    return _FINAL_ANSWER in text and not _ACTION.search(text.split(_FINAL_ANSWER)[0])


def _answer_of(text: str) -> str:
    return text.split(_FINAL_ANSWER)[-1].strip() if _FINAL_ANSWER in text else text.strip()


__all__ = [
    "CrewAIMemoryAdapter",
    "MemoryCrewTool",
    "MemoryLLM",
    "agent_id_from_role",
    "principal_from_inputs",
]
