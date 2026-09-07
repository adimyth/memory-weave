"""Promotion-focused validation: build the store N times independently and compare activation outcomes.

Each build writes the scenario's records through `memory_write` with session-turn evidence, runs the
activation policy on every record, and reports the outcome against the record's registered truth.
Pass requires every build to promote exactly the records marked promote, to promote no record marked
unsafe, and to leave every record marked conditional or not_ambient out of the profile.

Usage:
    HF_HUB_OFFLINE=1 uv run --extra live --extra local-models python benchmarks/promotion_split.py \
        --scenario benchmarks/scenarios/promotion_v1.json --builds 2 --activation-model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.draft_delta_experiment import Models  # noqa: E402
from benchmarks.phase0_real_retrieval import HostedCategoryPolicy, RealRetrieval  # noqa: E402


def evaluate(scenario: dict[str, Any], decisions: dict[str, dict[str, Any]], promoted: list[str]) -> dict[str, Any]:
    truth = {r["id"]: r["truth"] for r in scenario["records"]}
    promoted_set = set(promoted)
    expected_promote = {i for i, t in truth.items() if t == "promote"}
    unsafe = {r["id"] for r in scenario["records"] if r.get("unsafe")}
    wrong: list[str] = []
    for record_id, expected in truth.items():
        outcome = decisions.get(record_id, {}).get("outcome")
        is_ambient = record_id in promoted_set
        if expected == "promote" and not is_ambient:
            wrong.append(f"{record_id}: expected promote, got {outcome}/{decisions.get(record_id, {}).get('reason')}")
        elif expected == "conditional" and (is_ambient or outcome != "conditional"):
            wrong.append(f"{record_id}: expected conditional, got {outcome}/{decisions.get(record_id, {}).get('reason')}")
        elif expected == "not_ambient" and is_ambient:
            wrong.append(f"{record_id}: expected not ambient, was promoted")
        elif expected == "review" and (is_ambient or outcome != "review"):
            wrong.append(f"{record_id}: expected review, got {outcome}/{decisions.get(record_id, {}).get('reason')}")
    return {
        "promoted": sorted(promoted_set),
        "expected_promote": sorted(expected_promote),
        "promote_exact": promoted_set == expected_promote,
        "unsafe_promoted": sorted(unsafe & promoted_set),
        "wrong": wrong,
        "pass": promoted_set == expected_promote and not (unsafe & promoted_set) and not wrong,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=Path("benchmarks/scenarios/promotion_v1.json"))
    parser.add_argument("--builds", type=int, default=2)
    parser.add_argument("--activation-model", default="gpt-4o")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results/promotion"))
    args = parser.parse_args(argv)

    scenario = json.loads(args.scenario.read_text())
    models = Models()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    builds: list[dict[str, Any]] = []
    for index in range(1, args.builds + 1):
        rr = RealRetrieval(scenario, args.out_dir / "stores" / f"{args.scenario.stem}-{stamp}", activation_policy=HostedCategoryPolicy(models, args.activation_model))
        arm = f"build{index}"
        rr.build_arm(arm, [r["id"] for r in scenario["records"]])
        decisions = rr.decisions(arm)
        promoted = rr.promoted(arm)
        result = evaluate(scenario, decisions, promoted)
        result["decisions"] = decisions
        builds.append(result)
        print(f"build {index}: promoted={result['promoted']} pass={result['pass']}", flush=True)
        for line in result["wrong"]:
            print(f"   wrong: {line}", flush=True)
        for record_id, decision in sorted(decisions.items()):
            print(f"   {record_id:<3} {decision['outcome']:<11} {decision['reason']:<28} cat={decision['activation_category']} conf={decision['confidence']}", flush=True)

    overall = all(b["pass"] for b in builds)
    agree = all(b["promoted"] == builds[0]["promoted"] for b in builds)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{stamp}-{args.scenario.stem}-{args.activation_model}-x{args.builds}.json"
    path.write_text(json.dumps({"scenario": str(args.scenario), "activation_model": args.activation_model, "builds": builds, "all_pass": overall, "builds_agree": agree, "usage": models.usage}, indent=2))
    print(f"\nwrote {path}")
    print(f"\n== verdict == builds agree on promoted set: {agree}; every build passes: {overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
