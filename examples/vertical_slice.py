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

from memory_weave.config import MemoryWeaveConfig, load_config
from memory_weave.host import MemoryHost
from memory_weave.index.embedder import BgeM3Embedder, Embedder
from memory_weave.index.vector import VectorIndex
from memory_weave.ingest import EquivalenceJudge, Ingestor, NLICrossEncoderJudge, SessionBuffer
from memory_weave.models import Principal, Scope, Turn
from memory_weave.policy import MEMORY_USE_POLICY, MEMORY_USE_POLICY_VERSION
from memory_weave.retrieve import Retriever
from memory_weave.store import Store
from memory_weave.tools import ToolHandlers, tool_schemas
from memory_weave.util import now

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_AGENT_ID = "vertical-slice-agent"
_USER_ID = "user-aditya"
_MAX_TOOL_ROUNDS = 8


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
            max_tokens=800,
            system=system,
            messages=cast(Any, messages),
            tools=cast(Any, tools),
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
    store.create_session(principal.session_id, principal.agent_id, principal.user_id, principal.project_id, now())
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
) -> list[ToolTrace]:
    """Replay the fixed conversation, persist its transcript, and return every completed tool call in order."""

    messages: list[dict[str, object]] = []
    traces: list[ToolTrace] = []
    next_turn = 1
    schemas = tool_schemas()
    for user_turn, spec in enumerate(conversation, start=1):
        runtime.session_buffer.append_turn(Turn(_session_id(runtime), next_turn, "user", spec.text, now()))
        next_turn += 1
        messages.append({"role": "user", "content": spec.text})
        for _ in range(_MAX_TOOL_ROUNDS):
            reply = model.respond(system=MEMORY_USE_POLICY, messages=messages, tools=schemas)
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


def run_experiment(
    model_factory: Callable[[int], ToolModel],
    runtime_factory: Callable[[Path, int], VerticalSliceRuntime],
    *,
    model_id: str,
    runs: int = 3,
    database_dir: Path | None = None,
) -> dict[str, object]:
    """Run the conversation against fresh stores and aggregate the contract metrics the phase is meant to learn."""

    if runs <= 0:
        raise ValueError("runs must be positive.")
    if database_dir is None:
        with TemporaryDirectory(prefix="memory-weave-vertical-slice-") as temporary:
            return _run_experiment(Path(temporary), model_factory, runtime_factory, model_id, runs, artifact_dir=None)
    database_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = Path(mkdtemp(prefix="attempt-", dir=database_dir))
    return _run_experiment(attempt_dir, model_factory, runtime_factory, model_id, runs, artifact_dir=attempt_dir)


def run_live(
    *,
    model_id: str = _DEFAULT_MODEL,
    config_path: Path | None = None,
    runs: int = 3,
    database_dir: Path | None = None,
) -> dict[str, object]:
    """Run the real Anthropic plus local-model experiment only after the caller explicitly enables live execution."""

    _load_local_env()
    if os.environ.get("MEMORY_WEAVE_LIVE") != "1":
        raise RuntimeError(
            "Live execution is disabled. Set MEMORY_WEAVE_LIVE=1 after installing the live dependencies."
        )
    config = load_config(config_path)

    def runtime_factory(path: Path, run: int) -> VerticalSliceRuntime:
        return build_runtime(
            path, config, BgeM3Embedder(config.embedding), NLICrossEncoderJudge(config.ingestion.equivalence), run=run
        )

    return run_experiment(
        lambda _run: AnthropicToolModel(model_id),
        runtime_factory,
        model_id=model_id,
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


def _run_experiment(
    database_dir: Path,
    model_factory: Callable[[int], ToolModel],
    runtime_factory: Callable[[Path, int], VerticalSliceRuntime],
    model_id: str,
    runs: int,
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
            traces = run_conversation(model_factory(run), runtime)
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
    return _aggregate_metrics(model_id, metrics, requested_runs=runs, failures=failures, artifact_dir=artifact_dir)


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
    }
    for trace in traces:
        if trace.name != "memory_search" or trace.result.get("ok") is not True:
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
    metrics: Sequence[RunMetrics],
    *,
    requested_runs: int,
    failures: Sequence[Mapping[str, object]],
    artifact_dir: Path | None,
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
    return {
        "model_id": model_id,
        "prompt_version": MEMORY_USE_POLICY_VERSION,
        "runs": requested_runs,
        "completed_runs": len(metrics),
        "failed_runs": list(failures),
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
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
    parser.add_argument("--model", default=os.environ.get("MEMORY_WEAVE_VERTICAL_SLICE_MODEL") or _DEFAULT_MODEL)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--database-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    """Run the opt-in live experiment and print one JSON report suitable for the findings note."""

    args = parse_args()
    report = run_live(model_id=args.model, config_path=args.config, runs=args.runs, database_dir=args.database_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
