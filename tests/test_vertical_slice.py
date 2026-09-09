"""Deterministic coverage for the Phase 9a live vertical-slice harness."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from examples.vertical_slice import (
    _MAX_OUTPUT_TOKENS,
    CONVERSATION,
    ModelReply,
    OpenAIToolModel,
    OpenRouterToolModel,
    ToolUse,
    TruncatedReplyError,
    _attributes_for_first_preference,
    _openai_tools,
    _system_prompt,
    build_runtime,
    run_experiment,
    run_live,
)
from retold.config import EmbeddingConfig, RetoldConfig, RetrievalConfig
from retold.index.embedder import FakeEmbedder
from retold.ingest import FakeJudge
from retold.models import Turn
from retold.tools import tool_schemas
from retold.util import now

_CLAIM = "The user prefers concise technical answers with a short rationale."
_QUOTE = "I prefer concise technical answers, with a short rationale."
_RATIONALE_CLAIM = "The user values a short rationale with concise technical answers."


class TurnToolModel:
    """Emit the configured tool calls for each user turn and acknowledge every tool result."""

    def __init__(self, calls_by_turn: dict[int, list[ToolUse]] | None = None) -> None:
        self._calls_by_turn = calls_by_turn or {}
        self._user_turn = 0
        self.tools_seen: list[list[dict[str, object]]] = []

    def respond(self, *, system: str, messages: list[dict[str, object]], tools: list[dict[str, object]]) -> ModelReply:
        del system
        self.tools_seen.append(tools)
        latest = messages[-1]
        if latest["role"] == "user" and isinstance(latest["content"], str):
            self._user_turn += 1
            calls = self._calls_by_turn.get(self._user_turn, [])
            return ModelReply([_tool_block(call) for call in calls], calls)
        return ModelReply([{"type": "text", "text": "Acknowledged."}], [])


class FailingModel:
    """Raise at the serving-model boundary so the experiment can retain completed sibling runs."""

    def respond(self, *, system: str, messages: list[dict[str, object]], tools: list[dict[str, object]]) -> ModelReply:
        del system, messages, tools
        raise RuntimeError("scripted serving failure")


class RecordingOpenAIClient:
    """Return a prepared Chat Completions response and retain the request for adapter assertions."""

    def __init__(self, response: object) -> None:
        self._response = response
        self.request: dict[str, object] | None = None
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: object) -> object:
        self.request = kwargs
        return self._response


def _tool_block(tool_use: ToolUse) -> dict[str, object]:
    return {"type": "tool_use", "id": tool_use.id, "name": tool_use.name, "input": tool_use.input}


def _write_tool_use(
    identifier: str,
    *,
    attribute: str = "answer_style",
    content: str = _CLAIM,
    evidence: str = _QUOTE,
) -> ToolUse:
    return ToolUse(
        identifier,
        "memory_write",
        {
            "type": "semantic",
            "content": content,
            "attribute": attribute,
            "source_kind": "user_statement",
            "evidence": evidence,
            "entities": [{"kind": "person", "name": "Aditya Mishra", "role": "about"}],
        },
    )


def _search_tool_use(identifier: str) -> ToolUse:
    return ToolUse(identifier, "memory_search", {"queries": ["Aditya answer style"]})


def _openai_response() -> object:
    tool_call = SimpleNamespace(
        id="call-2",
        function=SimpleNamespace(name="memory_search", arguments='{"queries": ["Aditya answer style"]}'),
    )
    message = SimpleNamespace(content="I will check memory.", tool_calls=[tool_call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _config() -> RetoldConfig:
    return RetoldConfig(
        embedding=EmbeddingConfig(model="fake-embedder", version="1", dims=8),
        retrieval=RetrievalConfig(default_k=8, per_generator_k=8),
    )


def _runtime_factory(config: RetoldConfig, judge: FakeJudge | None = None) -> Callable[[Path, int], object]:
    def factory(path: Path, run: int):
        return build_runtime(path, config, FakeEmbedder(dims=8), judge or FakeJudge(), run=run)

    return factory


def test_scripted_vertical_slice_collects_metrics_without_a_framework(tmp_path: Path) -> None:
    report = run_experiment(
        lambda _run: TurnToolModel({1: [_write_tool_use("write-1")], 7: [_search_tool_use("search-7")]}),
        _runtime_factory(_config()),
        model_id="scripted-fake",
        runs=3,
        database_dir=tmp_path,
    )

    assert report["model_id"] == "scripted-fake"
    assert report["prompt_version"] == "v1"
    assert report["runs"] == 3
    assert report["completed_runs"] == 3
    assert report["failed_runs"] == []
    assert report["writes_attempted"] == 3
    assert report["subject_stability"] == {
        "same_preference_attributes": ["answer_style"],
        "per_run_attributes": [
            {"attributes": ["answer_style"], "run": 1},
            {"attributes": ["answer_style"], "run": 2},
            {"attributes": ["answer_style"], "run": 3},
        ],
        "runs_with_attributes": 3,
        "stable": True,
    }
    assert report["evidence_quote_match_rate"] == 1.0
    assert report["memory_applies_search_calls"] == 3
    assert report["memory_applies_searched_turns"] == 3
    assert report["memory_applies_search_turn_rate"] == 0.5
    assert report["ordinary_search_calls"] == 0
    assert report["ordinary_searched_turns"] == 0
    assert len(report["per_run"]) == 3
    assert len(CONVERSATION) == 12
    assert report["artifact_dir"] is not None
    assert report["report_path"] is not None
    report_path = Path(cast(str, report["report_path"]))
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert report["provider"] == "scripted"


def test_vertical_slice_registers_session_specific_write_targets(tmp_path: Path) -> None:
    """The live loop must register handler schemas, not the static schema that has no principal context."""

    model = TurnToolModel()
    run_experiment(
        lambda _run: model,
        _runtime_factory(_config()),
        model_id="scripted-fake",
        runs=1,
        database_dir=tmp_path,
    )

    memory_write = next(schema for schema in model.tools_seen[0] if schema["name"] == "memory_write")
    properties = memory_write["input_schema"]["properties"]  # type: ignore[index]
    assert properties["write_target"]["enum"] == ["personal"]  # type: ignore[index]
    assert "scope" not in properties


def test_reinforcement_uses_the_new_claim_provenance_for_evidence_metrics(tmp_path: Path) -> None:
    model = TurnToolModel(
        {
            1: [_write_tool_use("write-valid")],
            2: [_write_tool_use("write-fabricated", evidence="This sentence appears nowhere in the transcript.")],
        }
    )
    report = run_experiment(
        lambda _run: model,
        _runtime_factory(_config(), FakeJudge({(_CLAIM, _CLAIM): "same"})),
        model_id="scripted-fake",
        runs=1,
        database_dir=tmp_path,
    )

    assert report["write_outcomes"] == {"already_reinforced": 1, "created": 1}
    assert report["direct_claims"] == 2
    assert report["evidence_quote_matches"] == 1
    assert report["evidence_quote_match_rate"] == 0.5
    assert report["downgrades"] == 1


def test_search_rates_count_distinct_turns_not_tool_calls(tmp_path: Path) -> None:
    report = run_experiment(
        lambda _run: TurnToolModel({7: [_search_tool_use("search-a"), _search_tool_use("search-b")]}),
        _runtime_factory(_config()),
        model_id="scripted-fake",
        runs=1,
        database_dir=tmp_path,
    )

    assert report["memory_applies_search_calls"] == 2
    assert report["memory_applies_searched_turns"] == 1
    assert report["memory_applies_search_turn_rate"] == 0.5


@pytest.mark.parametrize("adapter", [OpenAIToolModel, OpenRouterToolModel])
def test_openai_compatible_adapters_translate_tool_calls_and_results(
    adapter: Callable[..., object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_PROVIDER", raising=False)
    client = RecordingOpenAIClient(_openai_response())
    model = adapter("provider/model", client=client)
    reply = model.respond(
        system="Memory policy.",
        messages=[
            {"role": "user", "content": "What answer style should you use?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "memory_search",
                        "input": {"queries": ["answer style"]},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": '{"results": []}'}],
            },
        ],
        tools=tool_schemas(),
    )

    assert reply == ModelReply(
        [
            {"type": "text", "text": "I will check memory."},
            {
                "type": "tool_use",
                "id": "call-2",
                "name": "memory_search",
                "input": {"queries": ["Aditya answer style"]},
            },
        ],
        [ToolUse("call-2", "memory_search", {"queries": ["Aditya answer style"]})],
    )
    assert client.request == {
        "model": "provider/model",
        "max_tokens": _MAX_OUTPUT_TOKENS,
        "messages": [
            {"role": "system", "content": "Memory policy."},
            {"role": "user", "content": "What answer style should you use?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "memory_search", "arguments": '{"queries": ["answer style"]}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": '{"results": []}'},
        ],
        "tools": _openai_tools(tool_schemas()),
    }


def test_openai_and_openrouter_adapters_configure_the_openai_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    client_options: list[dict[str, object]] = []

    def openai_constructor(**kwargs: object) -> object:
        client_options.append(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=openai_constructor))
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.test")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "Retold Test")

    OpenAIToolModel("openai-model")
    OpenRouterToolModel("provider/model")

    assert client_options == [
        {"api_key": "openai-test-key"},
        {
            "api_key": "openrouter-test-key",
            "base_url": "https://openrouter.ai/api/v1",
            "default_headers": {
                "HTTP-Referer": "https://example.test",
                "X-OpenRouter-Title": "Retold Test",
            },
        },
    ]


def test_openrouter_adapter_pins_configured_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_PROVIDER", "deepinfra")
    client = RecordingOpenAIClient(_openai_response())
    model = OpenRouterToolModel("z-ai/glm-5.3-flash", client=client)
    model.respond(
        system="Memory policy.",
        messages=[{"role": "user", "content": "What answer style should you use?"}],
        tools=tool_schemas(),
    )

    assert client.request is not None
    assert client.request["extra_body"] == {
        "provider": {"only": ["deepinfra"], "allow_fallbacks": False},
    }


def test_subject_stability_requires_an_active_preference_from_each_run(tmp_path: Path) -> None:
    report = run_experiment(
        lambda run: TurnToolModel({1: [_write_tool_use(f"write-{run}")]} if run == 1 else {}),
        _runtime_factory(_config()),
        model_id="scripted-fake",
        runs=3,
        database_dir=tmp_path,
    )

    assert report["subject_stability"] == {
        "same_preference_attributes": ["answer_style"],
        "per_run_attributes": [
            {"attributes": ["answer_style"], "run": 1},
            {"attributes": [], "run": 2},
            {"attributes": [], "run": 3},
        ],
        "runs_with_attributes": 1,
        "stable": False,
    }


def test_subject_stability_accepts_the_same_multiple_attributes_in_each_run(tmp_path: Path) -> None:
    report = run_experiment(
        lambda run: TurnToolModel(
            {
                1: [
                    _write_tool_use(f"answer-{run}"),
                    _write_tool_use(
                        f"rationale-{run}",
                        attribute="rationale_style",
                        content=_RATIONALE_CLAIM,
                    ),
                ]
            }
        ),
        _runtime_factory(_config()),
        model_id="scripted-fake",
        runs=3,
        database_dir=tmp_path,
    )

    assert report["subject_stability"]["same_preference_attributes"] == ["answer_style", "rationale_style"]
    assert report["subject_stability"]["runs_with_attributes"] == 3
    assert report["subject_stability"]["stable"] is True


def test_subject_stability_ignores_deleted_preference_records(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path / "status.sqlite", _config(), FakeEmbedder(dims=8), FakeJudge(), run=1)
    try:
        session_id = runtime.principal.session_id
        assert session_id is not None
        runtime.session_buffer.append_turn(Turn(session_id, 1, "user", _QUOTE, now()))
        result = runtime.handlers.memory_write(runtime.principal, _write_tool_use("write-deleted").input)
        assert result["ok"] is True
        record_id = result["record_id"]
        assert isinstance(record_id, str)
        runtime.store.update_status(record_id, "deleted")
        assert _attributes_for_first_preference(runtime.store, session_id) == []
    finally:
        runtime.close()


def test_experiment_keeps_completed_runs_and_uses_a_fresh_artifact_directory(tmp_path: Path) -> None:
    report = run_experiment(
        lambda run: FailingModel() if run == 2 else TurnToolModel(),
        _runtime_factory(_config()),
        model_id="scripted-fake",
        runs=3,
        database_dir=tmp_path,
    )
    second_report = run_experiment(
        lambda _run: TurnToolModel(),
        _runtime_factory(_config()),
        model_id="scripted-fake",
        runs=1,
        database_dir=tmp_path,
    )

    assert report["runs"] == 3
    assert report["completed_runs"] == 2
    assert report["failed_runs"] == [
        {
            "database_path": str(Path(cast(str, report["artifact_dir"])) / "run-2.sqlite"),
            "error": "scripted serving failure",
            "error_type": "RuntimeError",
            "run": 2,
            "stage": "conversation",
        }
    ]
    assert report["artifact_dir"] is not None
    assert Path(cast(str, report["artifact_dir"])).is_dir()
    assert second_report["artifact_dir"] is not None
    assert report["artifact_dir"] != second_report["artifact_dir"]
    assert second_report["completed_runs"] == 1


def test_live_runner_requires_an_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETOLD_LIVE", "")

    with pytest.raises(RuntimeError, match="Live execution is disabled"):
        run_live()


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("RETOLD_LIVE") != "1", reason="set RETOLD_LIVE=1 to run the hosted vertical slice")
def test_live_vertical_slice_reports_three_runs() -> None:
    pytest.importorskip("anthropic")
    report = run_live(runs=3)
    assert report["runs"] == 3


def test_openai_function_schemas_drop_composition_keywords_but_keep_the_shape_rule() -> None:
    """OpenAI-compatible endpoints reject top-level oneOf, so the sent copy is relaxed and described instead."""

    published = {schema["name"]: schema["input_schema"] for schema in tool_schemas()}
    sent = {tool["function"]["name"]: tool["function"]["parameters"] for tool in _openai_tools(tool_schemas())}

    assert "oneOf" in published["memory_revise"]
    assert not {"oneOf", "anyOf", "allOf", "not"} & set(sent["memory_revise"])
    assert sent["memory_revise"]["properties"] == published["memory_revise"]["properties"]
    assert sent["memory_revise"]["additionalProperties"] is False
    assert "entity_id, merge_into, reason" in sent["memory_revise"]["description"]
    assert sent["memory_search"] == published["memory_search"]


def test_openai_adapter_retries_once_with_the_renamed_output_token_parameter() -> None:
    """Reasoning models reject max_tokens; the adapter discovers the accepted name and reuses it."""

    class RejectingClient(RecordingOpenAIClient):
        def __init__(self, response: object) -> None:
            super().__init__(response)
            self.parameters_tried: list[str] = []

        def create(self, **kwargs: object) -> object:
            name = "max_tokens" if "max_tokens" in kwargs else "max_completion_tokens"
            self.parameters_tried.append(name)
            if name == "max_tokens":
                raise ValueError("Unsupported parameter: 'max_tokens' is not supported with this model.")
            return super().create(**kwargs)

    client = RejectingClient(_openai_response())
    model = OpenAIToolModel("reasoning-model", client=client)
    request = {"system": "Memory policy.", "messages": [{"role": "user", "content": "hi"}], "tools": tool_schemas()}

    model.respond(**request)  # type: ignore[arg-type]
    model.respond(**request)  # type: ignore[arg-type]

    assert client.parameters_tried == ["max_tokens", "max_completion_tokens", "max_completion_tokens"]
    assert client.request is not None
    assert client.request["max_completion_tokens"] == _MAX_OUTPUT_TOKENS


def test_adapters_reject_a_truncated_reply_instead_of_scoring_a_lost_tool_call() -> None:
    """A dropped tool call must not be recorded as the model deciding not to call one."""

    truncated = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="partial", tool_calls=[]))]
    )
    client = RecordingOpenAIClient(truncated)
    model = OpenAIToolModel("provider/model", client=client)

    with pytest.raises(TruncatedReplyError, match="output limit"):
        model.respond(system="Memory policy.", messages=[{"role": "user", "content": "hi"}], tools=tool_schemas())


def test_openai_adapter_disables_reasoning_effort_when_tools_are_rejected_with_it() -> None:
    """Some reasoning models refuse function tools unless the default effort is turned off."""

    class EffortRejectingClient(RecordingOpenAIClient):
        def __init__(self, response: object) -> None:
            super().__init__(response)
            self.efforts_tried: list[object] = []

        def create(self, **kwargs: object) -> object:
            self.efforts_tried.append(kwargs.get("reasoning_effort", "<absent>"))
            if "reasoning_effort" not in kwargs:
                raise ValueError(
                    "Function tools with reasoning_effort are not supported in /v1/chat/completions. "
                    "Set reasoning_effort to 'none'."
                )
            return super().create(**kwargs)

    client = EffortRejectingClient(_openai_response())
    model = OpenAIToolModel("reasoning-model", client=client)
    request = {"system": "Memory policy.", "messages": [{"role": "user", "content": "hi"}], "tools": tool_schemas()}

    model.respond(**request)  # type: ignore[arg-type]
    model.respond(**request)  # type: ignore[arg-type]

    assert client.efforts_tried == ["<absent>", "none", "none"]
    assert client.request is not None
    assert client.request["reasoning_effort"] == "none"


def test_hybrid_mode_issues_one_host_search_per_user_turn_and_appends_only_non_empty_results(
    tmp_path: Path,
) -> None:
    """LLD 14.1: the host searches once per user turn and appends a recalled block only when memory came back."""

    class WriteThenSilentModel:
        """Write one memory on turn 1, then never call a tool again, so every later search is host-issued."""

        def __init__(self) -> None:
            self._turn = 0

        def respond(self, *, system: str, messages: list[dict[str, object]], tools: list[dict[str, object]]):
            del system, tools
            latest = messages[-1]
            if latest["role"] == "user" and isinstance(latest["content"], str):
                self._turn += 1
                if self._turn == 1:
                    use = _write_tool_use("write-1")
                    return ModelReply([_tool_block(use)], [use])
            return ModelReply([{"type": "text", "text": "Acknowledged."}], [])

    report = run_experiment(
        lambda _run: WriteThenSilentModel(),
        lambda path, run: build_runtime(path, _config(), FakeEmbedder(dims=8), FakeJudge(), run=run),
        model_id="scripted-fake",
        trigger_mode="hybrid",
        runs=1,
        database_dir=tmp_path,
    )

    assert report["trigger_mode"] == "hybrid"
    # One host search per user turn, minus any turn shorter than the configured minimum.
    assert report["host_search_calls"] == len(CONVERSATION)
    assert report["memory_applies_search_calls"] == 0
    assert report["host_injection_rate"] is not None


def test_auto_prompt_does_not_instruct_the_model_to_call_an_unregistered_search_tool() -> None:
    """Auto mode announces recalled memory without mentioning the model-only search tool."""

    prompt = _system_prompt("auto")

    assert "call `memory_search`" not in prompt
    assert "use `memory_search` yourself" not in prompt
    assert "Relevant memories may also appear automatically" in prompt


def test_hybrid_recalled_blocks_stay_out_of_the_evidence_transcript(tmp_path: Path) -> None:
    """A recalled memory is injected context; quoting it back must never validate as user evidence."""

    class SilentModel:
        def respond(self, *, system: str, messages: list[dict[str, object]], tools: list[dict[str, object]]):
            del system, tools, messages
            return ModelReply([{"type": "text", "text": "Acknowledged."}], [])

    run_experiment(
        lambda _run: SilentModel(),
        lambda path, run: build_runtime(path, _config(), FakeEmbedder(dims=8), FakeJudge(), run=run),
        model_id="scripted-fake",
        trigger_mode="hybrid",
        runs=1,
        database_dir=tmp_path,
    )

    database = next(tmp_path.glob("*/run-1.sqlite"))
    connection = sqlite3.connect(database)
    roles = [row[0] for row in connection.execute("select distinct role from session_turns")]
    turns = connection.execute("select count(*) from session_turns").fetchone()[0]
    connection.close()

    assert set(roles) <= {"user", "assistant"}
    assert turns == 2 * len(CONVERSATION)


def test_hybrid_injects_a_rendered_recalled_block_that_never_enters_the_transcript(tmp_path: Path) -> None:
    """The non-empty branch must reach the model as rendered text and stay out of the evidence transcript."""

    seen: list[list[dict[str, object]]] = []

    class WriteThenObserveModel:
        """Write a memory on turn 1 so later host searches have something to find, then only observe."""

        def __init__(self) -> None:
            self._turn = 0

        def respond(self, *, system: str, messages: list[dict[str, object]], tools: list[dict[str, object]]):
            del system, tools
            seen.append([dict(message) for message in messages])
            latest = messages[-1]
            if latest["role"] == "user" and isinstance(latest["content"], str):
                self._turn += 1
                if self._turn == 1:
                    use = _write_tool_use("write-1")
                    return ModelReply([_tool_block(use)], [use])
            return ModelReply([{"type": "text", "text": "Acknowledged."}], [])

    embedder = FakeEmbedder(dims=8)
    # Make a later user turn resemble the stored preference so the host search returns something.
    embedder.set_similarity(CONVERSATION[3].text, _CLAIM, 0.95)
    report = run_experiment(
        lambda _run: WriteThenObserveModel(),
        lambda path, run: build_runtime(path, _config(), embedder, FakeJudge(), run=run),
        model_id="scripted-fake",
        trigger_mode="hybrid",
        runs=1,
        database_dir=tmp_path,
    )

    assert report["host_nonempty_calls"] >= 1, "the probe needs at least one non-empty host search"

    injected = [
        block
        for messages in seen
        for message in messages
        if isinstance(message.get("content"), list)
        for block in cast(list[dict[str, object]], message["content"])
        if block.get("type") == "tool_result" and str(block.get("tool_use_id", "")).startswith("recalled-")
    ]
    assert injected, "a non-empty host search must append a recalled tool-result block"
    rendered = str(injected[0]["content"])
    assert rendered.startswith("recalled memory:")
    assert not rendered.lstrip().startswith("{"), "the block must be rendered text, not a raw payload"

    database = next(tmp_path.glob("*/run-1.sqlite"))
    connection = sqlite3.connect(database)
    transcript = " ".join(row[0] for row in connection.execute("select content from session_turns"))
    roles = {row[0] for row in connection.execute("select distinct role from session_turns")}
    connection.close()
    # The model's own tool results belong in the transcript; the host's recalled block does not.
    assert "recalled memory:" not in transcript
    assert roles <= {"user", "assistant", "tool"}
