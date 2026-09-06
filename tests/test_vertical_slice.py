"""Deterministic coverage for the Phase 9a live vertical-slice harness."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from examples.vertical_slice import (
    CONVERSATION,
    ModelReply,
    ToolUse,
    _attributes_for_first_preference,
    build_runtime,
    run_experiment,
    run_live,
)
from memory_weave.config import EmbeddingConfig, MemoryWeaveConfig, RetrievalConfig
from memory_weave.index.embedder import FakeEmbedder
from memory_weave.ingest import FakeJudge
from memory_weave.models import Turn
from memory_weave.util import now

_CLAIM = "The user prefers concise technical answers with a short rationale."
_QUOTE = "I prefer concise technical answers, with a short rationale."
_RATIONALE_CLAIM = "The user values a short rationale with concise technical answers."


class TurnToolModel:
    """Emit the configured tool calls for each user turn and acknowledge every tool result."""

    def __init__(self, calls_by_turn: dict[int, list[ToolUse]] | None = None) -> None:
        self._calls_by_turn = calls_by_turn or {}
        self._user_turn = 0

    def respond(self, *, system: str, messages: list[dict[str, object]], tools: list[dict[str, object]]) -> ModelReply:
        del system, tools
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


def _config() -> MemoryWeaveConfig:
    return MemoryWeaveConfig(
        embedding=EmbeddingConfig(model="fake-embedder", version="1", dims=8),
        retrieval=RetrievalConfig(default_k=8, per_generator_k=8),
    )


def _runtime_factory(config: MemoryWeaveConfig, judge: FakeJudge | None = None) -> Callable[[Path, int], object]:
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
    monkeypatch.setenv("MEMORY_WEAVE_LIVE", "")

    with pytest.raises(RuntimeError, match="Live execution is disabled"):
        run_live()


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("MEMORY_WEAVE_LIVE") != "1", reason="set MEMORY_WEAVE_LIVE=1 to run the hosted vertical slice"
)
def test_live_vertical_slice_reports_three_runs() -> None:
    pytest.importorskip("anthropic")
    report = run_live(runs=3)
    assert report["runs"] == 3
