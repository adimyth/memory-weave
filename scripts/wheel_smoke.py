"""End-to-end smoke test of an installed Retold wheel, run against the installed package only.

The unit suite runs from the source tree, where `benchmarks/`, `tests/`, and `examples/` are importable and
every file is present. A wheel carries `retold/` alone, so it can fail in ways the suite cannot see: a
module left out of the build, a policy that lives only in the benchmarks, an extra that does not install
what an adapter imports, an entry point that is not wired. This script exercises those, and nothing that
needs a network or a model.

Run it from a directory that does not contain the source tree, in an environment where the wheel is
installed with the `deepagents` and `crewai` extras:

    python -m scripts.wheel_smoke      # from the repo, against whatever `retold` resolves to
    python wheel_smoke.py              # copied next to a clean virtual environment
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

# The retrieval hash of the supported bundle, recorded in benchmarks/bundles/bundle-2026-09-08-a.json and in
# docs/usefulness-gate.md. The wheel has neither file, so the value is pinned here as well.
SUPPORTED_RETRIEVAL_HASH = "e8c8c3309ab121de"
_AT = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"ok   {name}")
        return
    failures.append(f"{name}: {detail}" if detail else name)
    print(f"FAIL {name} {detail}")


def main() -> int:
    import retold
    from retold.config import EmbeddingConfig, RetoldConfig
    from retold.index.embedder import FakeEmbedder
    from retold.ingest import FakeJudge
    from retold.models import Principal, Scope, Turn
    from retold.policy import BundleRegistry, CategoryDecision, bundle_components
    from retold.policy.activation import ActivationService
    from retold.policy.reference import (
        ADMISSION_PROMPT_VERSION,
        CATEGORY_PROMPT_VERSION,
        GAP_PROMPT_VERSION,
        supported_bundle,
        supported_retrieval_config,
    )
    from retold.runtime import build_runtime
    from retold.store import Store

    source = Path(retold.__file__).resolve()
    print(f"retold {getattr(retold, '__version__', 'unknown')} from {source}")
    installed = "site-packages" in str(source)
    if "--installed" in sys.argv:
        check("the package is installed, not imported from a source checkout", installed, str(source))
    elif not installed:
        print("note: running against the source tree; pass --installed to require a real installation")

    components = bundle_components(supported_bundle(), supported_retrieval_config())
    check(
        "the supported bundle is rebuildable from the wheel",
        components["retrieval_config_sha256"] == SUPPORTED_RETRIEVAL_HASH,
        str(components.get("retrieval_config_sha256")),
    )
    check(
        "its policy versions are the measured ones",
        components["planner"] == f"gpt-4o/{GAP_PROMPT_VERSION}"
        and components["judge"] == f"gpt-5.4/{ADMISSION_PROMPT_VERSION}"
        and components["classifier"] == f"gpt-4o/{CATEGORY_PROMPT_VERSION}",
        str(components),
    )
    check(
        "a manifest that disagrees with the runtime is refused",
        _refuses_mismatch(supported_bundle(), RetoldConfig()),
    )

    with tempfile.TemporaryDirectory() as work:
        config = RetoldConfig(embedding=EmbeddingConfig(model="fake-embedder", version="1", dims=8))
        store = Store(Path(work) / "memory.sqlite")
        store.set_grant("assistant", Scope(kind="user", id="u1"), can_read=True, can_write=True)
        store.create_session("s1", "assistant", "u1", None, _AT)
        preference = "Answer briefly by default."
        store.append_turn(Turn("s1", 1, "user", preference, _AT))

        class BroadPreference:
            def classify(self, content: str) -> CategoryDecision:
                return CategoryDecision("preferences", "answer_style", "broad", 0.9)

        runtime = build_runtime(
            config,
            store,
            embedder=FakeEmbedder(dims=8),
            judge=FakeJudge(),
            activation=ActivationService(store, BroadPreference()),
        )
        principal = Principal("assistant", "u1", "s1", None)
        written = runtime.handlers.memory_write(
            principal,
            {
                "type": "semantic",
                "content": preference,
                "attribute": "answer_style",
                "source_kind": "user_statement",
                "evidence": preference,
            },
        )
        check("a tool write succeeds", written.get("ok") is True, str(written))
        check("a tool write runs the activation policy", written.get("activation") == "promote", str(written))
        record = store.get_record(str(written.get("record_id")))
        check("the promoted record reaches the profile", record is not None and record.activation == "ambient")

        # A second fact with no supporting user turn stays conditional, which is what the judge sees.
        conditional = runtime.handlers.memory_write(
            principal,
            {
                "type": "semantic",
                "content": "The user's cluster runs in eu-central-1.",
                "attribute": "cluster_region",
                "source_kind": "agent_inference",
            },
        )
        check("a fact with no verified evidence stays conditional", conditional.get("activation") == "conditional")

        decision = _run_one_turn(runtime, store, principal, config)
        check(
            "the utility-aware path decides and logs a turn",
            decision.disposition == "shadow_would_regenerate",
            decision.disposition,
        )
        check("the decision is persisted", len(store.turn_decisions("s1")) == 1)
        check("the registry answers for a bundle", BundleRegistry(store).all() == [])

        _check_adapters(runtime, config)

    result = subprocess.run([sys.executable, "-m", "retold.cli", "--help"], capture_output=True, text=True)
    check("the CLI entry point runs", result.returncode == 0, result.stderr.strip()[:200])

    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nthe installed wheel passed every check")
    return 0


def _refuses_mismatch(bundle_config: object, retrieval_config: object) -> bool:
    from retold.policy import BundleMismatchError, bundle_components

    try:
        bundle_components(bundle_config, retrieval_config)  # type: ignore[arg-type]
    except BundleMismatchError:
        return True
    return False


def _run_one_turn(runtime: object, store: object, principal: object, config: object) -> object:
    """One shadow turn through the orchestrator with stub policies, exercising the wheel's own wiring."""

    from retold.policy import Gap, GapDecision, UtilityAwareConfig, UtilityAwareOrchestrator
    from retold.policy.utility_aware import AdmissionDecision, CandidateVerdict

    class Planner:
        def plan(self, turn, public_context, ambient_profile, inventory):
            return GapDecision([Gap("preference", "how the user likes answers")], "stub/planner", "ok")

    class Judge:
        def admit(self, turn, public_context, ambient_profile, draft, candidates):
            verdicts = [CandidateVerdict(record.id, "helpful", "stub") for record in candidates]
            return AdmissionDecision([record.id for record in candidates], verdicts, "stub/judge", "ok")

    def retrieve(who, queries, context):
        return [record for record in store.get_records(_conditional_ids(store)) if record is not None]  # type: ignore[attr-defined]

    orchestrator = UtilityAwareOrchestrator(
        store,  # type: ignore[arg-type]
        retrieve,
        Planner(),
        Judge(),
        UtilityAwareConfig(gap_enabled=True, admission_mode="hosted_judge", shadow=True, bundle={"planner": "stub"}),
        retrieval_config=config,
    )
    return orchestrator.prepare_turn(
        principal,  # type: ignore[arg-type]
        "How should you answer me?",
        None,
        lambda: "A draft written without memory.",
        lambda records: "A final answer.",
    )


def _conditional_ids(store: object) -> list[str]:
    rows = store.connection.execute("SELECT id FROM records WHERE activation = 'conditional'").fetchall()  # type: ignore[attr-defined]
    return [str(row[0]) for row in rows]


def _check_adapters(runtime: object, config: object) -> None:
    """Both adapters, across the trigger-mode and memory-mode matrix, from the installed extras."""

    from retold.adapters.crewai import CrewAIMemoryAdapter
    from retold.adapters.deepagents import DeepAgentsMemoryAdapter
    from retold.models import Principal
    from retold.policy.reference import supported_bundle

    bundle = supported_bundle(retrieval_config=config)  # type: ignore[arg-type]
    principal = Principal("assistant", "u1", "s1", None)
    for name, build in (
        ("deepagents", lambda **kwargs: DeepAgentsMemoryAdapter(runtime, **kwargs)),  # type: ignore[arg-type]
        ("crewai", lambda **kwargs: CrewAIMemoryAdapter(runtime, principal, **kwargs)),  # type: ignore[arg-type]
    ):
        for trigger in ("tool_only", "auto", "hybrid"):
            adapter = build(trigger_mode=trigger)
            names = {tool.name for tool in adapter.tools()}
            check(
                f"{name} registers the right tools in {trigger}",
                ("memory_search" in names) is (trigger != "auto") and len(names) == (4 if trigger == "auto" else 5),
                str(sorted(names)),
            )
        utility = build(trigger_mode="tool_only", memory_mode="utility_aware", utility_config=bundle)
        check(f"{name} builds the utility-aware path", utility.memory_mode == "utility_aware")
        for trigger in ("auto", "hybrid"):
            try:
                build(trigger_mode=trigger, memory_mode="utility_aware", utility_config=bundle)
            except ValueError:
                check(f"{name} refuses utility_aware with {trigger}", True)
            else:
                check(f"{name} refuses utility_aware with {trigger}", False, "it was allowed")


if __name__ == "__main__":
    raise SystemExit(main())
