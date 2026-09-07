"""Run the Phase 9a scripted conversation through a real serving model and Memory Weave tools."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Any, Literal, Protocol, cast

from memory_weave.config import MemoryWeaveConfig, TriggerConfig, load_config
from memory_weave.host import MemoryHost
from memory_weave.index.embedder import BgeM3Embedder, Embedder
from memory_weave.index.vector import VectorIndex
from memory_weave.ingest import EquivalenceJudge, Ingestor, NLICrossEncoderJudge, SessionBuffer
from memory_weave.models import Principal, Scope, Turn
from memory_weave.policy import AUTO_MEMORY_NOTICE, AUTO_MEMORY_USE_POLICY, MEMORY_USE_POLICY, MEMORY_USE_POLICY_VERSION
from memory_weave.retrieve import Retriever
from memory_weave.store import Store
from memory_weave.tools import ToolHandlers
from memory_weave.util import normalize_ws, now

Provider = Literal["anthropic", "openai", "openrouter"]

_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_AGENT_ID = "vertical-slice-agent"
_USER_ID = "user-aditya"
_MAX_TOOL_ROUNDS = 8
# Tool-call arguments count toward the output budget, so keep it well clear of a full memory_write payload.
_MAX_OUTPUT_TOKENS = 2000
# Chat Completions function parameters reject these top-level composition keywords.
_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset({"oneOf", "anyOf", "allOf", "not"})


class TruncatedReplyError(RuntimeError):
    """Raised when a provider stopped at the output limit, which would score a lost tool call as model behaviour."""


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One fixed user turn and the measurement bucket used for its resulting tool calls."""

    text: str
    category: Literal["preference", "correction", "entity", "memory_applies", "ordinary"]


CONVERSATION: tuple[ConversationTurn, ...] = (
    ConversationTurn("I prefer concise technical answers, with a short rationale.", "preference"),
    ConversationTurn("For code examples, use Python unless I ask for another language.", "preference"),
    ConversationTurn("My working time zone is Asia/Kolkata.", "preference"),
    ConversationTurn("Actually, keep answers concise but include the important trade-off.", "correction"),
    ConversationTurn("My colleague Priya Nair owns the deployment checklist.", "entity"),
    ConversationTurn("Priya asked us to keep the deployment checklist in the repository.", "entity"),
    ConversationTurn("Show me a small Python example that parses this configuration file.", "memory_applies"),
    ConversationTurn("What time zone should you use when suggesting a meeting for me?", "memory_applies"),
    ConversationTurn("Should I use tabs or spaces in Python?", "ordinary"),
    ConversationTurn("How often should a deployment checklist be reviewed?", "ordinary"),
    ConversationTurn("What is a clear way to format a trade-off in a design note?", "ordinary"),
    ConversationTurn("What are common mistakes when naming a Python virtual environment?", "ordinary"),
)


@dataclass(frozen=True, slots=True)
class ToolUse:
    """One tool call emitted by a serving model in the provider-neutral loop representation."""

    id: str
    name: str
    input: dict[str, object]


@dataclass(frozen=True, slots=True)
class ModelReply:
    """The provider-neutral assistant content and any calls it asks the loop to execute."""

    content: list[dict[str, object]]
    tool_uses: list[ToolUse]


class ToolModel(Protocol):
    """Minimal serving-model contract used by the live runner and its scripted-model unit test."""

    def respond(self, *, system: str, messages: list[dict[str, object]], tools: list[dict[str, object]]) -> ModelReply:
        """Return one assistant response, optionally containing tool calls."""


class AnthropicToolModel:
    """Anthropic SDK adapter kept inside this live-only example rather than the framework-neutral package."""

    def __init__(self, model: str) -> None:
        try:
            import anthropic
        except ImportError as error:
            raise RuntimeError(
                "The live slice needs the Anthropic SDK. Install it with: uv sync --extra live"
            ) from error
        self._client = anthropic.Anthropic()
        self._model = model

    def respond(self, *, system: str, messages: list[dict[str, object]], tools: list[dict[str, object]]) -> ModelReply:
        """Call Anthropic and translate its content blocks into the loop's provider-neutral reply."""

        response = self._client.messages.create(
            model=self._model,
            max_tokens=_MAX_OUTPUT_TOKENS,
            system=system,
            messages=cast(Any, messages),
            tools=cast(Any, tools),
        )
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise TruncatedReplyError(
                f"{self._model} stopped at the {_MAX_OUTPUT_TOKENS}-token output limit; "
                "a dropped tool call would be miscounted as a decision not to call one."
            )
        content: list[dict[str, object]] = []
        tool_uses: list[ToolUse] = []
        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                input_value = dict(cast(Mapping[str, object], block.input))
                content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": input_value})
                tool_uses.append(ToolUse(block.id, block.name, input_value))
        return ModelReply(content, tool_uses)


class _OpenAICompatibleToolModel:
    """Translate the provider-neutral loop into OpenAI Chat Completions function calls."""

    def __init__(self, model: str, client: Any) -> None:
        self._client = client
        self._model = model
        # Chat Completions renamed this parameter; reasoning models reject the old name and older
        # models reject the new one, so the first call settles which one this model accepts.
        self._token_parameter: str | None = None
        # Some reasoning models apply a default effort that Chat Completions refuses to combine with
        # function tools. The first rejection settles whether this model needs the effort turned off.
        self._extra_options: dict[str, object] = {}

    def respond(self, *, system: str, messages: list[dict[str, object]], tools: list[dict[str, object]]) -> ModelReply:
        """Call the OpenAI-compatible endpoint and translate its message and function-call shapes."""

        response = self._create(
            messages=cast(Any, _openai_messages(system, messages)),
            tools=cast(Any, _openai_tools(tools)),
        )
        choices = getattr(response, "choices", [])
        if not choices:
            raise RuntimeError("The OpenAI-compatible endpoint returned no completion choices.")
        if getattr(choices[0], "finish_reason", None) == "length":
            raise TruncatedReplyError(
                f"{self._model} stopped at the {_MAX_OUTPUT_TOKENS}-token output limit; "
                "a dropped tool call would be miscounted as a decision not to call one."
            )
        message = choices[0].message
        content: list[dict[str, object]] = []
        text = getattr(message, "content", None)
        if isinstance(text, str) and text:
            content.append({"type": "text", "text": text})
        tool_uses: list[ToolUse] = []
        for tool_call in getattr(message, "tool_calls", None) or []:
            function = tool_call.function
            try:
                input_value = json.loads(function.arguments)
            except (TypeError, json.JSONDecodeError) as error:
                raise RuntimeError(f"Tool call {tool_call.id} has invalid JSON arguments.") from error
            if not isinstance(input_value, Mapping):
                raise RuntimeError(f"Tool call {tool_call.id} must have an object as its arguments.")
            tool_use = ToolUse(str(tool_call.id), str(function.name), dict(input_value))
            content.append({"type": "tool_use", "id": tool_use.id, "name": tool_use.name, "input": tool_use.input})
            tool_uses.append(tool_use)
        return ModelReply(content, tool_uses)

    def _create(self, **request: Any) -> Any:
        """Send one completion, discovering the parameter shape this model accepts for tool calls."""

        names = [self._token_parameter] if self._token_parameter else ["max_tokens", "max_completion_tokens"]
        last_error: Exception | None = None
        for name in names:
            try:
                response = self._client.chat.completions.create(
                    model=self._model, **{name: _MAX_OUTPUT_TOKENS}, **self._extra_options, **request
                )
            except Exception as error:  # noqa: BLE001 - provider SDKs raise their own request errors
                if _is_reasoning_effort_error(error) and "reasoning_effort" not in self._extra_options:
                    self._extra_options["reasoning_effort"] = "none"
                    return self._create(**request)
                if not _is_token_parameter_error(error):
                    raise
                last_error = error
                continue
            self._token_parameter = name
            return response
        raise RuntimeError(
            f"{self._model} rejected both max_tokens and max_completion_tokens: {last_error}"
        ) from last_error


def _is_token_parameter_error(error: Exception) -> bool:
    """Return whether a provider rejected the output-token parameter name rather than the request itself."""

    message = str(error).lower()
    return "max_tokens" in message or "max_completion_tokens" in message


def _is_reasoning_effort_error(error: Exception) -> bool:
    """Return whether a provider refused to combine its default reasoning effort with function tools."""

    return "reasoning_effort" in str(error).lower()


class OpenAIToolModel(_OpenAICompatibleToolModel):
    """OpenAI Chat Completions adapter for the provider-neutral vertical-slice loop."""

    def __init__(self, model: str, *, client: Any | None = None) -> None:
        resolved_client = client if client is not None else _new_openai_client(_required_environment("OPENAI_API_KEY"))
        super().__init__(model, resolved_client)


class OpenRouterToolModel(_OpenAICompatibleToolModel):
    """OpenRouter adapter that uses its OpenAI-compatible Chat Completions endpoint."""

    def __init__(self, model: str, *, client: Any | None = None) -> None:
        resolved_client = client
        if resolved_client is None:
            resolved_client = _new_openai_client(
                _required_environment("OPENROUTER_API_KEY"),
                base_url=_OPENROUTER_BASE_URL,
                default_headers=_openrouter_headers(),
            )
        super().__init__(
            model,
            resolved_client,
        )


def _new_openai_client(
    api_key: str,
    *,
    base_url: str | None = None,
    default_headers: Mapping[str, str] | None = None,
) -> Any:
    """Create an OpenAI SDK client for OpenAI itself or an OpenAI-compatible provider."""

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The OpenAI and OpenRouter adapters need the OpenAI SDK. Install it with: uv sync --extra live"
        ) from error
    options: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        options["base_url"] = base_url
    if default_headers:
        options["default_headers"] = dict(default_headers)
    return OpenAI(**options)


def _required_environment(name: str) -> str:
    """Return a required non-empty environment variable without exposing its value in an error."""

    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} before running this provider.")
    return value


def _openrouter_headers() -> dict[str, str]:
    """Return optional OpenRouter attribution headers when the caller configured them."""

    values = {
        "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER"),
        "X-OpenRouter-Title": os.environ.get("OPENROUTER_APP_TITLE"),
    }
    return {name: value for name, value in values.items() if value}


def _openai_messages(system: str, messages: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Convert the loop's text, function-call, and function-result messages into Chat Completions messages."""

    converted: list[dict[str, object]] = [{"role": "system", "content": system}]
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "user" and isinstance(content, str):
            converted.append({"role": "user", "content": content})
        elif role == "assistant" and isinstance(content, list):
            converted.append(_openai_assistant_message(content))
        elif role == "user" and isinstance(content, list):
            converted.extend(_openai_tool_messages(content))
        else:
            raise RuntimeError("The provider-neutral loop produced an unsupported message shape.")
    return converted


def _openai_assistant_message(content: list[object]) -> dict[str, object]:
    """Convert one standardized assistant response into an OpenAI assistant message."""

    text_parts: list[str] = []
    tool_calls: list[dict[str, object]] = []
    for block in content:
        if not isinstance(block, Mapping):
            raise RuntimeError("Assistant content must contain objects.")
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text_parts.append(cast(str, block["text"]))
        elif block.get("type") == "tool_use":
            identifier = block.get("id")
            name = block.get("name")
            input_value = block.get("input")
            if not isinstance(identifier, str) or not isinstance(name, str) or not isinstance(input_value, Mapping):
                raise RuntimeError("Tool-use content is missing an id, name, or object input.")
            tool_calls.append(
                {
                    "id": identifier,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(dict(input_value), sort_keys=True)},
                }
            )
        else:
            raise RuntimeError("Assistant content has an unsupported block type.")
    message: dict[str, object] = {"role": "assistant", "content": "\n".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _openai_tool_messages(content: list[object]) -> list[dict[str, object]]:
    """Convert standardized tool-result blocks into one Chat Completions tool message per result."""

    converted: list[dict[str, object]] = []
    for block in content:
        if not isinstance(block, Mapping):
            raise RuntimeError("Tool-result content must contain objects.")
        identifier = block.get("tool_use_id")
        result = block.get("content")
        if block.get("type") != "tool_result" or not isinstance(identifier, str) or not isinstance(result, str):
            raise RuntimeError("Tool-result content is missing a tool call id or string result.")
        converted.append({"role": "tool", "tool_call_id": identifier, "content": result})
    return converted


def _openai_tools(tools: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Wrap the framework-neutral schemas in the OpenAI Chat Completions function-tool envelope."""

    converted: list[dict[str, object]] = []
    for tool in tools:
        name = tool.get("name")
        description = tool.get("description")
        parameters = tool.get("input_schema")
        if not isinstance(name, str) or not isinstance(description, str) or not isinstance(parameters, Mapping):
            raise RuntimeError("Tool schemas must have a name, description, and input_schema object.")
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": _openai_parameters(name, dict(parameters)),
                },
            }
        )
    return converted


def _openai_parameters(name: str, schema: dict[str, object]) -> dict[str, object]:
    """Drop top-level composition keywords that OpenAI-compatible endpoints reject in a function schema.

    Only the copy sent to the provider is relaxed. ``validate_tool_input`` still enforces the published
    contract on the way in, so a payload that violates a dropped branch rule is rejected by the handler
    exactly as it would be for any other provider.
    """

    relaxed = {key: value for key, value in schema.items() if key not in _UNSUPPORTED_SCHEMA_KEYWORDS}
    dropped = sorted(set(schema) - set(relaxed))
    if dropped:
        requirement = _branch_requirement(schema)
        description = f"Shape constraints enforced by the tool: {requirement}." if requirement else ""
        relaxed["description"] = " ".join(filter(None, (cast(str, relaxed.get("description", "")), description)))
    return relaxed


def _branch_requirement(schema: Mapping[str, object]) -> str:
    """Describe the dropped ``oneOf`` branches in prose so the model still learns the accepted shapes."""

    branches = schema.get("oneOf")
    if not isinstance(branches, list):
        return ""
    shapes = [
        ", ".join(cast(list[str], branch["required"]))
        for branch in branches
        if isinstance(branch, Mapping) and isinstance(branch.get("required"), list)
    ]
    return "supply exactly one of these field sets: " + "; or ".join(shapes) if shapes else ""


@dataclass(slots=True)
class VerticalSliceRuntime:
    """The components one fresh live-run database needs, plus a close method for the owning connection."""

    store: Store
    principal: Principal
    session_buffer: SessionBuffer
    handlers: ToolHandlers

    def close(self) -> None:
        """Close the store connection after one independent run."""

        self.store.close()


@dataclass(frozen=True, slots=True)
class ToolTrace:
    """One completed tool invocation tied to the scripted user turn that prompted it."""

    turn: int
    category: str
    name: str
    input: dict[str, object]
    result: dict[str, object]
    trigger: str = "tool"


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """Numbers derived from one store's experiment events and search logs."""

    run: int
    write_attempts: int
    write_outcomes: dict[str, int]
    evidence_not_supported: int
    downgrades: int
    direct_claims: int
    matched_evidence_quotes: int
    preference_attributes: list[str]
    memory_applies_search_calls: int
    memory_applies_searched_turns: int
    ordinary_search_calls: int
    ordinary_searched_turns: int
    ordinary_nonempty_search_calls: int
    ordinary_nonempty_search_turns: int
    host_search_calls: int
    host_nonempty_calls: int
    host_ordinary_nonempty_calls: int


def build_runtime(
    database_path: Path,
    config: MemoryWeaveConfig,
    embedder: Embedder,
    judge: EquivalenceJudge,
    *,
    run: int,
) -> VerticalSliceRuntime:
    """Create one isolated store and provision the same principal and display aliases for a single replay."""

    store = Store(database_path)
    host = MemoryHost(store)
    user_scope = Scope(kind="user", id=_USER_ID)
    host.grant(_AGENT_ID, user_scope, read=True, write=True)
    host.provision_user(_USER_ID, aliases=("Aditya", "Aditya Mishra"))
    principal = Principal(_AGENT_ID, _USER_ID, f"vertical-slice-{run}", None)
    store.create_session(
        _session_id_from_principal(principal), principal.agent_id, principal.user_id, principal.project_id, now()
    )
    session_buffer = SessionBuffer(store)
    vector_index = VectorIndex(config.embedding)
    ingestor = Ingestor(store, vector_index, embedder, judge, session_buffer, config)
    retriever = Retriever(store, vector_index, embedder, config)
    handlers = ToolHandlers(retriever, ingestor, store, vector_index)
    return VerticalSliceRuntime(store, principal, session_buffer, handlers)


def run_conversation(
    model: ToolModel,
    runtime: VerticalSliceRuntime,
    conversation: Sequence[ConversationTurn] = CONVERSATION,
    *,
    trigger_mode: str = "tool_only",
    config: MemoryWeaveConfig | None = None,
) -> list[ToolTrace]:
    """Replay the fixed conversation, persist its transcript, and return every completed tool call in order."""

    settings = (config or load_config()).retrieval.trigger
    messages: list[dict[str, object]] = []
    traces: list[ToolTrace] = []
    next_turn = 1
    schemas = runtime.handlers.tool_schemas(runtime.principal)
    if trigger_mode == "auto":
        # LLD 14.1: auto is the control that isolates the host trigger, so the model gets no search tool.
        schemas = [schema for schema in schemas if schema["name"] != "memory_search"]
    system_prompt = _system_prompt(trigger_mode)
    for user_turn, spec in enumerate(conversation, start=1):
        runtime.session_buffer.append_turn(Turn(_session_id(runtime), next_turn, "user", spec.text, now()))
        next_turn += 1
        messages.append({"role": "user", "content": spec.text})
        if trigger_mode in ("auto", "hybrid"):
            recalled = _host_search(runtime, spec, settings, user_turn)
            if recalled is not None:
                trace, block = recalled
                traces.append(trace)
                messages.extend(block)
        for _ in range(_MAX_TOOL_ROUNDS):
            reply = model.respond(system=system_prompt, messages=messages, tools=schemas)
            messages.append({"role": "assistant", "content": reply.content})
            assistant_text = _reply_text(reply)
            if assistant_text:
                runtime.session_buffer.append_turn(
                    Turn(_session_id(runtime), next_turn, "assistant", assistant_text, now())
                )
                next_turn += 1
            if not reply.tool_uses:
                break
            tool_results: list[dict[str, object]] = []
            for tool_use in reply.tool_uses:
                result = _dispatch(runtime.handlers, runtime.principal, tool_use, spec.text)
                trace = ToolTrace(user_turn, spec.category, tool_use.name, tool_use.input, result)
                traces.append(trace)
                if tool_use.name == "memory_write":
                    _append_write_attempt(runtime.store, runtime.principal, trace)
                encoded = json.dumps(result, sort_keys=True)
                runtime.session_buffer.append_turn(Turn(_session_id(runtime), next_turn, "tool", encoded, now()))
                next_turn += 1
                tool_results.append({"type": "tool_result", "tool_use_id": tool_use.id, "content": encoded})
            messages.append({"role": "user", "content": tool_results})
        else:
            raise RuntimeError(
                f"Model requested more than {_MAX_TOOL_ROUNDS} consecutive tool rounds for user turn {user_turn}."
            )
    return traces


def _host_search(
    runtime: VerticalSliceRuntime,
    spec: ConversationTurn,
    settings: TriggerConfig,
    user_turn: int,
) -> tuple[ToolTrace, list[dict[str, object]]] | None:
    """Issue one host search for a new user turn and render any non-empty result as a recalled-memory block.

    This is the adapter behaviour from LLD 14.1: one search per user turn, never on assistant or tool
    turns, appended after the existing messages so the prompt prefix is never rewritten.
    """

    query = normalize_ws(spec.text)
    if len(query) < settings.auto_min_query_chars:
        runtime.store.append_event(
            "trigger.skipped",
            runtime.principal.agent_id,
            None,
            None,
            {"query_chars": len(query), "minimum": settings.auto_min_query_chars, "turn": user_turn},
        )
        return None
    payload = {"queries": [query], "k": settings.auto_k}
    result = runtime.handlers.memory_search(runtime.principal, payload, context=_host_context(runtime), trigger="auto")
    trace = ToolTrace(user_turn, spec.category, "memory_search", payload, result, trigger="auto")
    if result.get("ok") is not True or not result.get("results"):
        return trace, []
    # The recalled block is injected context, not something the user or a tool produced, so it is
    # deliberately kept out of the transcript. Writing it there would let a later write quote a
    # recalled memory back as if it were the user's own evidence.
    call_id = f"recalled-{user_turn}"
    encoded = _render_recalled(result)
    return trace, [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": call_id, "name": "memory_search", "input": payload},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": encoded}]},
    ]


def _system_prompt(trigger_mode: str) -> str:
    """Return the policy text for the mode, since auto and hybrid must announce unrequested recall."""

    if trigger_mode == "tool_only":
        return MEMORY_USE_POLICY
    if trigger_mode == "auto":
        # The tool is not registered, so the instruction to call it is removed rather than contradicted.
        base = MEMORY_USE_POLICY.replace(
            "Before acting on anything that could depend on the user's preferences, earlier decisions, "
            "or previous sessions, call `memory_search` with one to three specific phrases. ",
            "",
        )
        return f"{base} {AUTO_MEMORY_NOTICE}"
    return f"{MEMORY_USE_POLICY} {AUTO_MEMORY_USE_POLICY}"


def _host_context(runtime: VerticalSliceRuntime) -> str:
    """Return the last user and assistant turns, which is the context LLD 14.1 gives the rewriter."""

    turns = runtime.session_buffer.turns(_session_id(runtime))
    recent = [turn for turn in turns if turn.role in ("user", "assistant")][-2:]
    return "\n".join(f"{turn.role}: {turn.content}" for turn in recent)


def _render_recalled(result: Mapping[str, object]) -> str:
    """Render recalled memory the way an agent would see a tool result, not as a raw payload dump."""

    results = cast(Sequence[Mapping[str, object]], result.get("results", []))
    lines = ["recalled memory:"]
    for entry in results:
        explanation = cast(Mapping[str, object], entry.get("explanation", {}))
        summary = explanation.get("summary")
        record = cast(Mapping[str, object], entry.get("record", {}))
        lines.append(str(summary) if summary else f"[{record.get('id')}] {record.get('content')}")
    return "\n\n".join(lines)


def run_experiment(
    model_factory: Callable[[int], ToolModel],
    runtime_factory: Callable[[Path, int], VerticalSliceRuntime],
    *,
    model_id: str,
    trigger_mode: str = "tool_only",
    provider: str = "scripted",
    runs: int = 3,
    database_dir: Path | None = None,
    config: MemoryWeaveConfig | None = None,
) -> dict[str, object]:
    """Run the conversation against fresh stores and aggregate the contract metrics the phase is meant to learn."""

    if runs <= 0:
        raise ValueError("runs must be positive.")
    if database_dir is None:
        with TemporaryDirectory(prefix="memory-weave-vertical-slice-") as temporary:
            return _run_experiment(
                Path(temporary),
                model_factory,
                runtime_factory,
                model_id,
                provider,
                runs,
                trigger_mode,
                config,
                artifact_dir=None,
            )
    database_dir.mkdir(parents=True, exist_ok=True)
    # Name the directory after the mode and start time, so an artifact folder says what produced it.
    stamp = now().strftime("%Y%m%d-%H%M%S")
    attempt_dir = Path(mkdtemp(prefix=f"{trigger_mode}-{stamp}-", dir=database_dir))
    return _run_experiment(
        attempt_dir,
        model_factory,
        runtime_factory,
        model_id,
        provider,
        runs,
        trigger_mode,
        config,
        artifact_dir=attempt_dir,
    )


def run_live(
    *,
    provider: Provider | None = None,
    model_id: str | None = None,
    config_path: Path | None = None,
    runs: int = 3,
    database_dir: Path | None = None,
    trigger_mode: str | None = None,
) -> dict[str, object]:
    """Run one provider's hosted model plus local-model experiment only after the caller explicitly enables it."""

    _load_local_env()
    if os.environ.get("MEMORY_WEAVE_LIVE") != "1":
        raise RuntimeError(
            "Live execution is disabled. Set MEMORY_WEAVE_LIVE=1 after installing the live dependencies."
        )
    resolved_provider = _live_provider(provider)
    resolved_model = _live_model(resolved_provider, model_id)
    config = load_config(config_path)
    resolved_trigger = trigger_mode or config.retrieval.trigger.mode

    def runtime_factory(path: Path, run: int) -> VerticalSliceRuntime:
        return build_runtime(
            path, config, BgeM3Embedder(config.embedding), NLICrossEncoderJudge(config.ingestion.equivalence), run=run
        )

    return run_experiment(
        lambda _run: _provider_model(resolved_provider, resolved_model),
        runtime_factory,
        model_id=resolved_model,
        provider=resolved_provider,
        trigger_mode=resolved_trigger,
        config=config,
        runs=runs,
        database_dir=database_dir or Path("benchmarks/results/vertical-slice"),
    )


def _load_local_env() -> None:
    """Load a local `.env` file when the live optional dependency is installed, without overriding real environment values."""  # noqa: E501

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _live_provider(provider: Provider | None) -> Provider:
    """Resolve and validate the requested hosted-model provider without silently changing providers."""

    value = provider or os.environ.get("MEMORY_WEAVE_VERTICAL_SLICE_PROVIDER", "anthropic")
    if value not in {"anthropic", "openai", "openrouter"}:
        raise ValueError("provider must be anthropic, openai, or openrouter.")
    return cast(Provider, value)


def _live_model(provider: Provider, model_id: str | None) -> str:
    """Resolve the model from the call or environment and retain the historical Anthropic default."""

    resolved = model_id or os.environ.get("MEMORY_WEAVE_VERTICAL_SLICE_MODEL")
    if resolved:
        return resolved
    if provider == "anthropic":
        return _DEFAULT_ANTHROPIC_MODEL
    raise ValueError(f"Set MEMORY_WEAVE_VERTICAL_SLICE_MODEL or pass --model for the {provider} provider.")


def _provider_model(provider: Provider, model_id: str) -> ToolModel:
    """Construct the requested vendor adapter while leaving the experiment loop vendor-neutral."""

    if provider == "anthropic":
        return AnthropicToolModel(model_id)
    if provider == "openai":
        return OpenAIToolModel(model_id)
    return OpenRouterToolModel(model_id)


def _run_experiment(
    database_dir: Path,
    model_factory: Callable[[int], ToolModel],
    runtime_factory: Callable[[Path, int], VerticalSliceRuntime],
    model_id: str,
    provider: str,
    runs: int,
    trigger_mode: str,
    config: MemoryWeaveConfig | None,
    *,
    artifact_dir: Path | None,
) -> dict[str, object]:
    metrics: list[RunMetrics] = []
    failures: list[dict[str, object]] = []
    for run in range(1, runs + 1):
        database_path = database_dir / f"run-{run}.sqlite"
        runtime: VerticalSliceRuntime | None = None
        stage = "setup"
        try:
            runtime = runtime_factory(database_path, run)
            stage = "conversation"
            traces = run_conversation(model_factory(run), runtime, trigger_mode=trigger_mode, config=config)
            stage = "metrics"
            metrics.append(_collect_metrics(runtime.store, runtime.principal, traces, run))
        except Exception as error:
            failures.append(
                {
                    "database_path": str(database_path),
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "run": run,
                    "stage": stage,
                }
            )
        finally:
            if runtime is not None:
                runtime.close()
    report = _aggregate_metrics(
        model_id,
        provider,
        metrics,
        requested_runs=runs,
        failures=failures,
        artifact_dir=artifact_dir,
        trigger_mode=trigger_mode,
    )
    if artifact_dir is not None:
        report_path = artifact_dir / "report.json"
        report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _dispatch(handlers: ToolHandlers, principal: Principal, tool_use: ToolUse, context: str) -> dict[str, object]:
    if tool_use.name == "memory_search":
        return handlers.memory_search(principal, tool_use.input, context=context)
    if tool_use.name == "memory_get":
        return handlers.memory_get(principal, tool_use.input)
    if tool_use.name == "memory_write":
        return handlers.memory_write(principal, tool_use.input)
    if tool_use.name == "memory_revise":
        return handlers.memory_revise(principal, tool_use.input)
    if tool_use.name == "memory_forget":
        return handlers.memory_forget(principal, tool_use.input)
    return {"ok": False, "error": {"code": "invalid_input", "message": f"Unknown tool: {tool_use.name}."}}


def _append_write_attempt(store: Store, principal: Principal, trace: ToolTrace) -> None:
    """Record enough non-sensitive experiment metadata to derive write outcome counts from durable events."""

    result = trace.result
    record_id = result.get("record_id") if result.get("ok") is True else None
    audit = _write_audit_payload(store, record_id) if isinstance(record_id, str) else {}
    error = result.get("error")
    error_code = error.get("code") if isinstance(error, Mapping) else None
    source_kind, source_ref = _claim_provenance(audit)
    store.append_event(
        "vertical_slice.tool_attempt",
        principal.agent_id,
        record_id if isinstance(record_id, str) else None,
        None,
        {
            "audit_event_found": bool(audit),
            "category": trace.category,
            "error_code": error_code,
            "evidence_note": audit.get("evidence_note") if audit else result.get("note"),
            "outcome": result.get("outcome") if result.get("ok") is True else error_code,
            "requested_source_kind": trace.input.get("source_kind"),
            "source_ref": source_ref,
            "stored_source_kind": source_kind,
            "turn": trace.turn,
        },
    )


def _write_audit_payload(store: Store, record_id: str) -> dict[str, object]:
    """Return the record event written by the immediately preceding memory_write call, if that call persisted one."""

    row = store.connection.execute(
        """
        SELECT payload FROM events
        WHERE record_id = ? AND kind IN ('record.created', 'record.reinforced', 'record.superseded')
        ORDER BY id DESC LIMIT 1
        """,
        (record_id,),
    ).fetchone()
    return {} if row is None else cast(dict[str, object], json.loads(cast(str, row["payload"])))


def _claim_provenance(audit: Mapping[str, object]) -> tuple[object | None, object | None]:
    """Use the new claim's provenance from a reinforcement event rather than the incumbent record's provenance."""

    if "reinforcing_source_kind" in audit:
        return audit.get("reinforcing_source_kind"), audit.get("reinforcing_source_ref")
    return audit.get("source_kind"), audit.get("source_ref")


def _collect_metrics(store: Store, principal: Principal, traces: Sequence[ToolTrace], run: int) -> RunMetrics:
    payloads = _write_attempt_payloads(store)
    outcomes = Counter(str(payload.get("outcome")) for payload in payloads if payload.get("outcome") is not None)
    direct_claims = [
        payload for payload in payloads if payload.get("requested_source_kind") in {"user_statement", "tool_result"}
    ]
    attributes = _attributes_for_first_preference(store, _session_id_from_principal(principal))
    search_metrics = _search_metrics(store, traces)
    return RunMetrics(
        run=run,
        write_attempts=len(payloads),
        write_outcomes=dict(sorted(outcomes.items())),
        evidence_not_supported=sum(
            "evidence does not support claim" in str(payload.get("evidence_note")) for payload in payloads
        ),
        downgrades=sum(
            payload.get("requested_source_kind") in {"user_statement", "tool_result"}
            and payload.get("stored_source_kind") == "agent_inference"
            for payload in payloads
        ),
        direct_claims=len(direct_claims),
        matched_evidence_quotes=sum(payload.get("source_ref") is not None for payload in direct_claims),
        preference_attributes=attributes,
        memory_applies_search_calls=search_metrics["memory_applies_search_calls"],
        memory_applies_searched_turns=search_metrics["memory_applies_searched_turns"],
        ordinary_search_calls=search_metrics["ordinary_search_calls"],
        ordinary_searched_turns=search_metrics["ordinary_searched_turns"],
        ordinary_nonempty_search_calls=search_metrics["ordinary_nonempty_search_calls"],
        ordinary_nonempty_search_turns=search_metrics["ordinary_nonempty_search_turns"],
        host_search_calls=search_metrics["host_search_calls"],
        host_nonempty_calls=search_metrics["host_nonempty_calls"],
        host_ordinary_nonempty_calls=search_metrics["host_ordinary_nonempty_calls"],
    )


def _write_attempt_payloads(store: Store) -> list[dict[str, object]]:
    rows = store.connection.execute(
        "SELECT payload FROM events WHERE kind = 'vertical_slice.tool_attempt' ORDER BY id"
    ).fetchall()
    return [cast(dict[str, object], json.loads(cast(str, row["payload"]))) for row in rows]


def _attributes_for_first_preference(store: Store, session_id: str) -> list[str]:
    source_ref = f"session:{session_id}<turn:1>"
    rows = store.connection.execute(
        """
        SELECT DISTINCT attribute FROM records
        WHERE source_ref = ? AND status IN ('provisional', 'confirmed') AND attribute IS NOT NULL
        ORDER BY attribute
        """,
        (source_ref,),
    ).fetchall()
    return [cast(str, row["attribute"]) for row in rows]


def _search_metrics(store: Store, traces: Sequence[ToolTrace]) -> dict[str, int]:
    memory_applies_turns: set[int] = set()
    ordinary_turns: set[int] = set()
    ordinary_nonempty_turns: set[int] = set()
    counts = {
        "memory_applies_search_calls": 0,
        "ordinary_search_calls": 0,
        "ordinary_nonempty_search_calls": 0,
        "host_search_calls": 0,
        "host_nonempty_calls": 0,
        "host_ordinary_nonempty_calls": 0,
    }
    for trace in traces:
        if trace.name != "memory_search" or trace.result.get("ok") is not True:
            continue
        if trace.trigger == "auto":
            counts["host_search_calls"] += 1
            if trace.result.get("results"):
                counts["host_nonempty_calls"] += 1
                if trace.category == "ordinary":
                    counts["host_ordinary_nonempty_calls"] += 1
            continue
        search_id = trace.result.get("search_id")
        if not isinstance(search_id, str):
            continue
        row = store.read_search_log(search_id)
        if row is None:
            raise RuntimeError(f"Search {search_id} completed without a search-log row.")
        if trace.category == "memory_applies":
            counts["memory_applies_search_calls"] += 1
            memory_applies_turns.add(trace.turn)
        if trace.category == "ordinary":
            counts["ordinary_search_calls"] += 1
            ordinary_turns.add(trace.turn)
            if row["returned"]:
                counts["ordinary_nonempty_search_calls"] += 1
                ordinary_nonempty_turns.add(trace.turn)
    return {
        **counts,
        "memory_applies_searched_turns": len(memory_applies_turns),
        "ordinary_searched_turns": len(ordinary_turns),
        "ordinary_nonempty_search_turns": len(ordinary_nonempty_turns),
    }


def _aggregate_metrics(
    model_id: str,
    provider: str,
    metrics: Sequence[RunMetrics],
    *,
    requested_runs: int,
    failures: Sequence[Mapping[str, object]],
    artifact_dir: Path | None,
    trigger_mode: str,
) -> dict[str, object]:
    per_run_attributes = [{"attributes": run.preference_attributes, "run": run.run} for run in metrics]
    contributing_attributes = [run.preference_attributes for run in metrics if run.preference_attributes]
    attributes = sorted({attribute for values in contributing_attributes for attribute in values})
    outcomes: Counter[str] = Counter()
    for run in metrics:
        outcomes.update(run.write_outcomes)
    direct_claims = sum(run.direct_claims for run in metrics)
    matched_quotes = sum(run.matched_evidence_quotes for run in metrics)
    memory_applies_turns = sum(turn.category == "memory_applies" for turn in CONVERSATION) * len(metrics)
    ordinary_turns = sum(turn.category == "ordinary" for turn in CONVERSATION) * len(metrics)
    memory_applies_search_calls = sum(run.memory_applies_search_calls for run in metrics)
    memory_applies_searched_turns = sum(run.memory_applies_searched_turns for run in metrics)
    ordinary_search_calls = sum(run.ordinary_search_calls for run in metrics)
    ordinary_searched_turns = sum(run.ordinary_searched_turns for run in metrics)
    ordinary_nonempty_search_calls = sum(run.ordinary_nonempty_search_calls for run in metrics)
    ordinary_nonempty_search_turns = sum(run.ordinary_nonempty_search_turns for run in metrics)
    host_search_calls = sum(run.host_search_calls for run in metrics)
    host_nonempty_calls = sum(run.host_nonempty_calls for run in metrics)
    host_ordinary_nonempty_calls = sum(run.host_ordinary_nonempty_calls for run in metrics)
    return {
        "model_id": model_id,
        "provider": provider,
        "trigger_mode": trigger_mode,
        "prompt_version": MEMORY_USE_POLICY_VERSION,
        "finished_at": now().isoformat(),
        "runs": requested_runs,
        "completed_runs": len(metrics),
        "failed_runs": list(failures),
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
        "report_path": None,
        "writes_attempted": sum(run.write_attempts for run in metrics),
        "write_outcomes": dict(sorted(outcomes.items())),
        "invalid_subject": outcomes["invalid_subject"],
        "entity_ambiguous": outcomes["entity_ambiguous"],
        "evidence_not_supported": sum(run.evidence_not_supported for run in metrics),
        "downgrades": sum(run.downgrades for run in metrics),
        "subject_stability": {
            "same_preference_attributes": attributes,
            "per_run_attributes": per_run_attributes,
            "runs_with_attributes": len(contributing_attributes),
            "stable": (
                len(contributing_attributes) == len(metrics)
                and bool(contributing_attributes)
                and len({tuple(values) for values in contributing_attributes}) == 1
            ),
        },
        "evidence_quote_match_rate": matched_quotes / direct_claims if direct_claims else None,
        "evidence_quote_matches": matched_quotes,
        "direct_claims": direct_claims,
        "memory_applies_search_calls": memory_applies_search_calls,
        "memory_applies_searched_turns": memory_applies_searched_turns,
        "memory_applies_search_turn_rate": (
            memory_applies_searched_turns / memory_applies_turns if memory_applies_turns else None
        ),
        "ordinary_search_calls": ordinary_search_calls,
        "ordinary_searched_turns": ordinary_searched_turns,
        "ordinary_search_turn_rate": ordinary_searched_turns / ordinary_turns if ordinary_turns else None,
        "ordinary_nonempty_search_calls": ordinary_nonempty_search_calls,
        "ordinary_nonempty_search_turns": ordinary_nonempty_search_turns,
        "ordinary_nonempty_turn_rate": (ordinary_nonempty_search_turns / ordinary_turns if ordinary_turns else None),
        "host_search_calls": host_search_calls,
        "host_nonempty_calls": host_nonempty_calls,
        "host_ordinary_nonempty_calls": host_ordinary_nonempty_calls,
        "host_injection_rate": (host_nonempty_calls / host_search_calls if host_search_calls else None),
        "host_ordinary_injection_rate": (host_ordinary_nonempty_calls / ordinary_turns if ordinary_turns else None),
        "ordinary_turns": ordinary_turns,
        "per_run": [_run_metrics_payload(run) for run in metrics],
    }


def _run_metrics_payload(metrics: RunMetrics) -> dict[str, object]:
    return {
        "run": metrics.run,
        "write_attempts": metrics.write_attempts,
        "write_outcomes": metrics.write_outcomes,
        "evidence_not_supported": metrics.evidence_not_supported,
        "downgrades": metrics.downgrades,
        "direct_claims": metrics.direct_claims,
        "matched_evidence_quotes": metrics.matched_evidence_quotes,
        "preference_attributes": metrics.preference_attributes,
        "memory_applies_search_calls": metrics.memory_applies_search_calls,
        "memory_applies_searched_turns": metrics.memory_applies_searched_turns,
        "ordinary_search_calls": metrics.ordinary_search_calls,
        "ordinary_searched_turns": metrics.ordinary_searched_turns,
        "ordinary_nonempty_search_calls": metrics.ordinary_nonempty_search_calls,
        "ordinary_nonempty_search_turns": metrics.ordinary_nonempty_search_turns,
        "host_search_calls": metrics.host_search_calls,
        "host_nonempty_calls": metrics.host_nonempty_calls,
        "host_ordinary_nonempty_calls": metrics.host_ordinary_nonempty_calls,
    }


def _reply_text(reply: ModelReply) -> str:
    return "\n".join(cast(str, block["text"]) for block in reply.content if block.get("type") == "text")


def _session_id(runtime: VerticalSliceRuntime) -> str:
    return _session_id_from_principal(runtime.principal)


def _session_id_from_principal(principal: Principal) -> str:
    if principal.session_id is None:
        raise RuntimeError("The vertical slice needs a session id for evidence validation.")
    return principal.session_id


def parse_args() -> argparse.Namespace:
    """Parse live-run controls without requiring users to edit the example file."""

    parser = argparse.ArgumentParser(description="Run the Memory Weave Phase 9a vertical slice.")
    parser.add_argument("--provider", choices=["anthropic", "openai", "openrouter"])
    parser.add_argument("--model")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--database-dir", type=Path)
    parser.add_argument("--trigger", choices=["tool_only", "auto", "hybrid"])
    return parser.parse_args()


def main() -> int:
    """Run the opt-in live experiment and print one JSON report suitable for the findings note."""

    args = parse_args()
    report = run_live(
        provider=args.provider,
        model_id=args.model,
        config_path=args.config,
        runs=args.runs,
        database_dir=args.database_dir,
        trigger_mode=args.trigger,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
