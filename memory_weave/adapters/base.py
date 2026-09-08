"""What every adapter shares: the protocol, principal derivation, policy text, context, and rendering.

An adapter owns three decisions only: who the principal is, when a turn happened, and when the session
ended. Everything else, retrieval, admission, activation, and isolation, is the core's, and an adapter
that starts making those decisions has stopped being an adapter.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from memory_weave.models import Principal
from memory_weave.policy.prompt import AUTO_MEMORY_NOTICE, AUTO_MEMORY_USE_POLICY, MEMORY_USE_POLICY
from memory_weave.store import Store
from memory_weave.tools.render import render_search_payload

TriggerMode = Literal["tool_only", "auto", "hybrid"]
MemoryMode = Literal["tool_only", "utility_aware"]
MEMORY_TOOL_NAMES = ("memory_search", "memory_get", "memory_write", "memory_revise", "memory_forget")
RECALLED_HEADER = "recalled memory"


class Adapter(Protocol):
    """The LLD 5 adapter contract, framework-neutral."""

    def tools(self) -> Sequence[Any]:
        """The memory tools to register with the framework, already filtered by trigger mode."""

    def principal_from_run(self, run_context: Any) -> Principal:
        """Derive the principal from trusted host configuration; never from model output."""

    def end_session(self, run_context: Any) -> None:
        """Close the session and schedule extraction exactly once."""


@dataclass(frozen=True, slots=True)
class TurnContext:
    """The last user turn and the last assistant turn, the context attached to every search."""

    user: str | None
    assistant: str | None

    def text(self, limit: int) -> str | None:
        parts = []
        if self.user:
            parts.append(f"User: {self.user}")
        if self.assistant:
            parts.append(f"Assistant: {self.assistant}")
        if not parts:
            return None
        return "\n".join(parts)[:limit]


def turn_context(store: Store, session_id: str | None, *, before_turn: int | None = None) -> TurnContext:
    """Read the most recent user and assistant turns of a session from the store."""

    if session_id is None:
        return TurnContext(None, None)
    turns = store.session_turns(session_id)
    if before_turn is not None:
        turns = [turn for turn in turns if turn.turn < before_turn]
    user = next((turn.content for turn in reversed(turns) if turn.role == "user"), None)
    assistant = next((turn.content for turn in reversed(turns) if turn.role == "assistant"), None)
    return TurnContext(user, assistant)


def principal_from_mapping(configurable: Mapping[str, Any], *, session_key: str = "thread_id") -> Principal:
    """Build the principal from a host-supplied mapping; every identity comes from the host, none from the model."""

    try:
        agent_id = str(configurable["agent_id"])
        user_id = str(configurable["user_id"])
        session_id = str(configurable[session_key])
    except KeyError as missing:
        raise ValueError(
            f"The run configuration must carry agent_id, user_id, and {session_key}: missing {missing}."
        ) from None
    if not agent_id or not user_id or not session_id:
        raise ValueError("agent_id, user_id, and the session id must be non-empty.")
    project = configurable.get("project_id")
    return Principal(agent_id, user_id, session_id, str(project) if project else None)


def policy_text(trigger_mode: TriggerMode, base_prompt: str | None = None) -> str:
    """LLD 13: the memory-use policy for the prompt prefix, varied by who triggers searches."""

    if trigger_mode == "tool_only":
        policy = MEMORY_USE_POLICY
    elif trigger_mode == "hybrid":
        policy = f"{MEMORY_USE_POLICY} {AUTO_MEMORY_USE_POLICY}"
    else:
        sentences = [sentence for sentence in MEMORY_USE_POLICY.split(". ") if "memory_search" not in sentence]
        policy = ". ".join(sentences).rstrip(".") + ". " + AUTO_MEMORY_NOTICE
    return f"{base_prompt.rstrip()}\n\n{policy}" if base_prompt else policy


def tool_names_for(trigger_mode: TriggerMode) -> tuple[str, ...]:
    """LLD 14.1: every tool in tool_only and hybrid; no memory_search in auto."""

    if trigger_mode == "auto":
        return tuple(name for name in MEMORY_TOOL_NAMES if name != "memory_search")
    return MEMORY_TOOL_NAMES


def render_tool_result(name: str, arguments: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    """Render one tool result for the model: search as the LLD 10.9 blocks, everything else as JSON."""

    if name == "memory_search" and payload.get("ok") is True:
        queries = [str(query) for query in arguments.get("queries", [])]
        return render_search_payload(queries, payload)
    return json.dumps(payload, sort_keys=True, default=str)


def recalled_memory_block(queries: Sequence[str], payload: Mapping[str, Any]) -> str | None:
    """The block a host-issued search appends, or ``None`` when the search returned nothing."""

    if payload.get("ok") is not True or not payload.get("results"):
        return None
    return f"{RECALLED_HEADER}\n{render_search_payload(list(queries), payload)}"
