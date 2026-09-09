"""End-to-end check of the reference host contract with real hosted adapters and the supported bundle.

Builds a real store from a scenario through the ingestor and activation policy, records the supported
bundle's fitness result through the registry the way a consuming application would, then drives the
reference host through the scenario twice: first with regeneration off (shadow), then with the switch on
(serving). Finally it runs the metrics aggregator and the rollback check over the store, so the whole
operating loop is exercised with real models: enforcement, kill switches, budget, decision log, metrics.

Usage:
    HF_HUB_OFFLINE=1 uv run --extra live --extra local-models python benchmarks/reference_host_e2e.py \
        --scenario benchmarks/scenarios/slice_conversation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.draft_delta_experiment import Models  # noqa: E402
from benchmarks.phase0_real_retrieval import HostedCategoryPolicy, RealRetrieval  # noqa: E402
from benchmarks.shadow_adapter import HostedAdmissionPolicy, HostedGapPolicy, policy_bundle  # noqa: E402
from examples.reference_host import KillSwitches, ReferenceHost  # noqa: E402
from retold.models import Principal, Record  # noqa: E402
from retold.policy import (  # noqa: E402
    BundleNotApprovedError,
    BundleRegistry,
    RollbackThresholds,
    bundle_components,
    render,
)

_DRAFT_SYSTEM = (
    "You are a helpful assistant. Answer the user's message directly and briefly, in under 150 words. "
    "You have no memory of earlier conversations with this user beyond what is stated below."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=Path("benchmarks/scenarios/slice_conversation.json"))
    parser.add_argument("--bundle-file", type=Path, default=Path("benchmarks/bundles/bundle-2026-09-07-a.json"))
    parser.add_argument("--draft-model", default="gpt-5.6-luna")
    parser.add_argument("--gap-model", default="gpt-4o")
    parser.add_argument("--admission-model", default="gpt-5.4")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results/reference-host"))
    args = parser.parse_args(argv)

    scenario = json.loads(args.scenario.read_text())
    models = Models()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    workdir = args.out_dir / "stores" / f"{args.scenario.stem}-{stamp}"
    rr = RealRetrieval(scenario, workdir, activation_policy=HostedCategoryPolicy(models, args.gap_model))
    rr.build_arm("host", [r["id"] for r in scenario["records"]])
    state = rr._arms["host"]  # noqa: SLF001
    store, handlers, principal = state["store"], state["handlers"], state["principal"]
    id_map = state["id_map"]

    def retrieve(p: Principal, queries: list[str], context: str) -> list[Record]:
        result = handlers.memory_search(p, {"queries": queries[:3], "k": 8}, context=context[:2000], trigger="auto")
        if result.get("ok") is not True:
            return []
        return store.get_records([str(e["record"]["id"]) for e in result.get("results", [])])

    bundle = policy_bundle(
        args.gap_model, args.admission_model, rr.config, None, classifier=f"{args.gap_model}/category-v2"
    )
    gap_policy = HostedGapPolicy(models, args.gap_model)
    admission = HostedAdmissionPolicy(models, args.admission_model)
    drafts: dict[str, str] = {}

    def baseline_for(turn_text: str, profile_text: str):
        def baseline() -> str:
            key = hashlib.sha256(f"{profile_text}\n{turn_text}".encode()).hexdigest()
            if key not in drafts:
                drafts[key] = models.complete(args.draft_model, f"{_DRAFT_SYSTEM}\n\n{profile_text}", turn_text)
            return drafts[key]

        return baseline

    def regenerate_for(turn_text: str, profile_text: str):
        def regenerate(records: list[Record]) -> str:
            facts = "\n".join(f"- {r.content}" for r in records)
            return models.complete(
                args.draft_model,
                f"{_DRAFT_SYSTEM}\n\n{profile_text}\n\nRelevant facts recalled from memory:\n{facts}",
                turn_text,
            )

        return regenerate

    # 1. Enforcement: serving without a recorded result must be refused.
    try:
        ReferenceHost(
            store,
            retrieve=retrieve,
            gap_policy=gap_policy,
            admission_policy=admission,
            bundle=bundle,
            retrieval_config=rr.config,
        )
        print("FAIL: an unapproved bundle was allowed to serve")
        return 1
    except BundleNotApprovedError as error:
        print(f"enforcement ok: {error}")

    # 2. Shadow with the unapproved bundle.
    host = ReferenceHost(
        store,
        retrieve=retrieve,
        gap_policy=gap_policy,
        admission_policy=admission,
        bundle=bundle,
        retrieval_config=rr.config,
        switches=KillSwitches(regeneration=False),
    )
    from retold.policy import ProfileAssembler

    profile_text = ProfileAssembler(store).build(principal).text or "Applied preferences: none."
    print(f"profile: {profile_text.replace(chr(10), ' | ')}")
    for turn in scenario["turns"]:
        d = host.serve_turn(
            principal,
            turn["text"],
            None,
            baseline_for(turn["text"], profile_text),
            regenerate_for(turn["text"], profile_text),
        )
        print(
            f"[shadow]  {turn['id']:<4} {turn['class']:<13} {d.disposition:<26} "
            f"admitted={[id_map.get(i, i) for i in d.admitted_ids]}"
        )

    # 3. Record the supported bundle's fitness the way a consuming application would, then serve.
    components = bundle_components(host.config(), rr.config)
    expected = json.loads(args.bundle_file.read_text())
    drift = {
        k: (components.get(k), expected.get(k))
        for k in set(components) | set(expected)
        if components.get(k) != expected.get(k) and k not in ("gap_enabled",)
    }
    if drift:
        print(f"note: live bundle differs from the shipped file on {sorted(drift)}; recording the live components")
    BundleRegistry(store).record(
        dict(components, gap_enabled=True),
        passed=True,
        evidence="docs/usefulness-gate.md 8k,8l",
        recorded_by="reference_host_e2e",
    )
    host.set_switches(KillSwitches(regeneration=True))
    print(f"serving bundle {host.bundle_hash()}")
    for turn in scenario["turns"]:
        d = host.serve_turn(
            principal,
            turn["text"],
            None,
            baseline_for(turn["text"], profile_text),
            regenerate_for(turn["text"], profile_text),
        )
        print(
            f"[served]  {turn['id']:<4} {turn['class']:<13} {d.disposition:<26} "
            f"admitted={[id_map.get(i, i) for i in d.admitted_ids]} budget={d.effective_budget_ms}"
        )

    # 4. Budget path and a kill switch, once each.
    tight = host.serve_turn(
        principal,
        scenario["turns"][1]["text"],
        None,
        baseline_for(scenario["turns"][1]["text"], profile_text),
        regenerate_for(scenario["turns"][1]["text"], profile_text),
        latency_budget_ms=0,
    )
    print(f"[budget0] {scenario['turns'][1]['id']:<4} {tight.disposition}")
    host.disable("admission")
    off = host.serve_turn(
        principal,
        scenario["turns"][1]["text"],
        None,
        baseline_for(scenario["turns"][1]["text"], profile_text),
        regenerate_for(scenario["turns"][1]["text"], profile_text),
    )
    print(f"[adm off] {scenario['turns'][1]['id']:<4} {off.disposition} bundle={host.bundle_hash()}")

    # 5. Metrics and rollback over everything logged.
    report = host.metrics()
    print("\n" + render(report))
    from retold.policy import aggregate, rollback_reasons

    everything = aggregate(store)
    reasons = rollback_reasons(everything, RollbackThresholds(min_turns=1))
    print(f"\nall bundles: turns={everything.turns} stages={everything.stages}")
    print(f"rollback reasons at min_turns=1: {reasons or 'none'}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{stamp}-{args.scenario.stem}.json"
    out.write_text(
        json.dumps(
            {
                "bundle": components,
                "report": report.to_dict(),
                "all": everything.to_dict(),
                "decisions": store.turn_decisions(principal.session_id),
                "usage": models.usage,
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
