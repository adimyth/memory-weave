"""Show both branches of Retold's utility-aware flow without an API key.

This example uses real lightweight retrieval and deterministic planner and admission policies so the result stays
repeatable. Production hosts replace the two policies and answer callbacks with their model clients, then approve the
resulting bundle after evaluating it on their traffic.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

from retold import Retold
from retold.models import Principal, Record
from retold.policy import (
    AdmissionDecision,
    BundleRegistry,
    CandidateVerdict,
    Gap,
    GapDecision,
    ProfileBlock,
    UtilityAwareConfig,
    UtilityAwareOrchestrator,
    bundle_components,
)


class ExamplePlanner:
    """Ask for the stored code-language preference only when the task needs code."""

    def plan(
        self, turn: str, public_context: str | None, ambient_profile: ProfileBlock, inventory: Sequence[str]
    ) -> GapDecision:
        del public_context, ambient_profile, inventory
        if "code example" in turn.casefold():
            return GapDecision(
                [Gap("preference", "preferred programming language for code examples")],
                "example-planner",
                "ok",
            )
        return GapDecision([], "example-planner", "empty")


class ExampleAdmission:
    """Admit the retrieved preference when it corrects the memory-free draft."""

    def admit(
        self,
        turn: str,
        public_context: str | None,
        ambient_profile: ProfileBlock,
        draft: str,
        candidates: Sequence[Record],
    ) -> AdmissionDecision:
        del turn, public_context, ambient_profile, draft
        verdicts = [CandidateVerdict(record.id, "helpful", "It changes the example language.") for record in candidates]
        return AdmissionDecision([record.id for record in candidates], verdicts, "example-admission", "ok")


def main() -> None:
    database = Path(tempfile.mkdtemp(prefix="retold-utility-aware-")) / "memory.sqlite"
    with Retold.open(database, profile="lite") as retold:
        with retold.session(user_id="aditya", session_id="utility-aware-demo") as memory:
            memory.remember(
                "I prefer Python for code examples.",
                evidence="I prefer Python for code examples.",
                attribute="code_example_language",
            )

            def retrieve(principal: Principal, queries: list[str], turn: str) -> list[Record]:
                del principal, turn
                found = memory.search(" ".join(queries))
                return [item.record for item in found.items]

            config = UtilityAwareConfig(
                gap_enabled=True,
                admission_mode="hosted_judge",
                shadow=False,
                bundle={"example": "deterministic-utility-aware-quickstart-v1"},
            )
            registry = BundleRegistry(retold.store)
            registry.record(
                bundle_components(config, retold.runtime.config),
                passed=True,
                evidence="deterministic quick-start scenario",
                recorded_by="example",
            )
            orchestrator = UtilityAwareOrchestrator(
                retold.store,
                retrieve,
                ExamplePlanner(),
                ExampleAdmission(),
                config,
                registry=registry,
                retrieval_config=retold.runtime.config,
            )

            turns = [
                ("Give me a code example that reads a JSON file.", "Here is a JavaScript example."),
                ("What is dependency injection?", "Dependency injection supplies dependencies from outside an object."),
            ]
            for turn, draft in turns:

                def baseline(draft: str = draft) -> str:
                    return draft

                def regenerate(records: list[Record]) -> str:
                    del records
                    return "Here is a Python example, using your saved preference."

                decision = orchestrator.prepare_turn(
                    memory.principal,
                    turn,
                    None,
                    baseline=baseline,
                    regenerate=regenerate,
                )
                print(f"question: {turn}")
                print(f"decision: {decision.disposition}")
                print(f"memory used: {len(decision.admitted_ids)}")
                print(f"answer: {decision.response}\n")


if __name__ == "__main__":
    main()
