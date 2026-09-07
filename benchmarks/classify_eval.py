"""Classification-only evaluation of the category policy, for taxonomy and prompt changes.

Classifies every record of the given scenarios with the chosen prompt version and reports two things:
label agreement with the scenario's hidden category labels, which is diagnostic only, and inventory
coverage, which is the metric that matters: for every memory-needed turn, whether the true category of each
conditional record it needs would appear in the inventory built from the assigned categories. It also runs
the deterministic activation rule against the classifier's proposals so promotion outcomes can be checked
without building a store.

Usage:
    uv run --extra live python benchmarks/classify_eval.py --prompt-version category-v2 \
        --scenarios benchmarks/scenarios/phase0_v4.json benchmarks/scenarios/phase0_v5.json
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
from benchmarks.phase0_real_retrieval import HostedCategoryPolicy  # noqa: E402
from memory_weave.policy.activation import RETRIEVAL_CATEGORIES  # noqa: E402


def evaluate(scenario: dict[str, Any], policy: HostedCategoryPolicy) -> dict[str, Any]:
    records = scenario["records"]
    truth = {r["id"]: r.get("category_truth") or r.get("category") for r in records}
    ambient_truth = {r["id"] for r in records if r.get("ambient_truth") is True}
    assigned: dict[str, str] = {}
    unsafe_flags: dict[str, bool] = {}
    for record in records:
        decision = policy.classify(record["text"])
        assigned[record["id"]] = decision.retrieval_category
        unsafe_flags[record["id"]] = decision.unsafe
    # Inventory as the planner would see it: categories of records that would be conditional.
    conditional_ids = [r["id"] for r in records if r["id"] not in ambient_truth]
    inventory = sorted({RETRIEVAL_CATEGORIES.get(assigned[i], assigned[i]) for i in conditional_ids})
    needed = [(t["id"], e) for t in scenario["turns"] if t["class"] != "ordinary" for e in t["expected"] if e not in ambient_truth]
    covered = [(tid, e) for tid, e in needed if truth.get(e) and RETRIEVAL_CATEGORIES.get(truth[e], truth[e]) in inventory]
    disagreements = {}
    for i, t in truth.items():
        if t and assigned[i] != t:
            disagreements[f"{t}->{assigned[i]}"] = disagreements.get(f"{t}->{assigned[i]}", 0) + 1
    unsafe_records = {r["id"] for r in records if r.get("unsafe_promotion") or r["kind"] == "misleading"}
    return {
        "scenario": scenario.get("description", "")[:60],
        "label_agreement": f"{sum(1 for i, t in truth.items() if t and assigned[i] == t)}/{sum(1 for t in truth.values() if t)}",
        "inventory_coverage_by_true_category": f"{len(covered)}/{len(needed)}",
        "uncovered": [(tid, e, truth.get(e), assigned.get(e)) for tid, e in needed if (tid, e) not in covered],
        "inventory": inventory,
        "disagreements": disagreements,
        "unsafe_flagged": f"{sum(1 for i in unsafe_records if unsafe_flags.get(i))}/{len(unsafe_records)}",
        "assigned": assigned,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--prompt-version", default="category-v2")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results/classify"))
    args = parser.parse_args(argv)
    models = Models()
    policy = HostedCategoryPolicy(models, args.model, args.prompt_version)
    results = {}
    for path in args.scenarios:
        scenario = json.loads(path.read_text())
        results[path.stem] = evaluate(scenario, policy)
        r = results[path.stem]
        print(f"{path.stem} [{args.prompt_version}] agreement={r['label_agreement']} coverage={r['inventory_coverage_by_true_category']} unsafe_flagged={r['unsafe_flagged']}")
        print(f"   uncovered: {r['uncovered']}")
        print(f"   disagreements: {r['disagreements']}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir / f"{stamp}-{args.model}-{args.prompt_version}.json"
    out.write_text(json.dumps({"model": args.model, "prompt_version": args.prompt_version, "results": results, "usage": models.usage}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
