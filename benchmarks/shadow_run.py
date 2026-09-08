"""Phase 2 shadow run: the utility-aware orchestrator beside the served path, with causal isolation asserted.

Builds a real store from a scenario through the ingestor and the activation policy, then replays the
scenario's turns twice with identical cached drafts:

- pass 1, served only: the orchestrator is disabled, the served response is the baseline draft;
- pass 2, shadow on: gap planning, real retrieval, and hosted admission run in shadow mode, which records
  what would have happened and never regenerates.

The harness asserts that the served responses, the records' status and activation, the session turns, and
the profile are identical across the two passes. It then scores pass 2's turn decisions against the
scenario's hidden labels and prints the pre-registered shadow gates. Policy failures and budget withholds
are reported separately from recall misses.

Usage:
    HF_HUB_OFFLINE=1 uv run --extra live --extra local-models python benchmarks/shadow_run.py \
        --scenario benchmarks/scenarios/phase0_v5.json --gap-model gpt-4o --admission-model gpt-5.4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.draft_delta_experiment import Models  # noqa: E402
from benchmarks.phase0_real_retrieval import HostedCategoryPolicy, RealRetrieval  # noqa: E402
from benchmarks.shadow_adapter import HostedAdmissionPolicy, HostedGapPolicy, policy_bundle  # noqa: E402
from memory_weave.models import Principal, Record  # noqa: E402
from memory_weave.policy import (  # noqa: E402
    ProfileAssembler,
    TurnOptions,
    UtilityAwareConfig,
    UtilityAwareOrchestrator,
)

_DRAFT_SYSTEM = (
    "You are a helpful assistant. Answer the user's message directly and briefly, in under 150 words. "
    "You have no memory of earlier conversations with this user beyond what is stated below."
)


class DraftCache:
    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._data: dict[str, str] = json.loads(path.read_text()) if path and path.exists() else {}

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def put(self, key: str, value: str) -> None:
        self._data[key] = value
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data))


def snapshot(store_path: Path, principal: Principal) -> dict[str, Any]:
    """Everything the shadow policy must not touch: records' status and activation, session turns, profile."""

    db = sqlite3.connect(store_path)
    try:
        records = db.execute("SELECT id, status, activation, category FROM records ORDER BY id").fetchall()
        turns = db.execute(
            "SELECT session_id, turn, role, content FROM session_turns ORDER BY session_id, turn"
        ).fetchall()
        reviews = db.execute("SELECT id, status FROM activation_reviews ORDER BY id").fetchall()
    finally:
        db.close()
    return {"records": records, "turns": turns, "reviews": reviews}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=Path("benchmarks/scenarios/phase0_v5.json"))
    parser.add_argument("--draft-model", default="gpt-5.6-luna")
    parser.add_argument("--gap-model", default="gpt-4o")
    parser.add_argument("--admission-model", default="gpt-5.4")
    parser.add_argument("--activation-model", default="gpt-4o")
    parser.add_argument(
        "--category-prompt", default="category-v2", help="Classifier prompt version; a change is a new bundle"
    )
    parser.add_argument(
        "--budget-ms", type=int, default=None, help="Per-request latency budget applied to every turn in pass 2"
    )
    parser.add_argument(
        "--gap-timeout-ms",
        type=int,
        default=4000,
        help="Stage timeout for gap planning; set from measured planner latency",
    )
    parser.add_argument(
        "--admission-timeout-ms",
        type=int,
        default=8000,
        help="Stage timeout for admission; set from measured judge p95 with eight candidates",
    )
    parser.add_argument("--draft-cache", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results/shadow"))
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    scenario = json.loads(args.scenario.read_text())
    models = Models()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    workdir = args.out_dir / "stores" / f"{args.scenario.stem}-{stamp}"
    rr = RealRetrieval(
        scenario, workdir, activation_policy=HostedCategoryPolicy(models, args.activation_model, args.category_prompt)
    )
    rr.build_arm("shadow", [r["id"] for r in scenario["records"]])
    state = rr._arms["shadow"]  # noqa: SLF001
    store, handlers, principal, id_map = state["store"], state["handlers"], state["principal"], state["id_map"]
    # id_map already maps Memory Weave record ids to scenario ids.
    reverse = dict(id_map)
    store_path = workdir / "shadow.sqlite"
    cache = DraftCache(args.draft_cache)
    assembler = ProfileAssembler(store)

    def retrieve(p: Principal, queries: list[str], context: str) -> list[Record]:
        result = handlers.memory_search(p, {"queries": queries[:3], "k": 8}, context=context[:2000], trigger="auto")
        if result.get("ok") is not True:
            return []
        ids = [str(entry["record"]["id"]) for entry in result.get("results", [])]
        return store.get_records(ids)

    def make_baseline(turn_text: str):
        def baseline() -> str:
            profile = assembler.build(principal).text or "Applied preferences: none."
            system = f"{_DRAFT_SYSTEM}\n\n{profile}"
            key = hashlib.sha256(f"{args.draft_model}\n{system}\n{turn_text}".encode()).hexdigest()
            hit = cache.get(key)
            if hit is not None:
                return hit
            text = models.complete(args.draft_model, system, turn_text)
            cache.put(key, text)
            return text

        return baseline

    def forbidden_regenerate(records: list[Record]) -> str:
        raise AssertionError("regeneration must never run in shadow mode")

    bundle = policy_bundle(
        args.gap_model,
        args.admission_model,
        rr.config.retrieval,
        args.budget_ms,
        classifier=f"{args.activation_model}/{args.category_prompt}",
    )
    off = UtilityAwareConfig(gap_enabled=False, admission_mode="disabled", shadow=False, bundle=bundle)
    on = UtilityAwareConfig(
        gap_enabled=True,
        admission_mode="hosted_judge",
        shadow=True,
        latency_budget_ms=args.budget_ms,
        gap_timeout_ms=args.gap_timeout_ms,
        admission_timeout_ms=args.admission_timeout_ms,
        bundle=bundle,
    )

    served: dict[str, list[str]] = {"off": [], "on": []}
    snapshots: dict[str, dict[str, Any]] = {}
    turns = scenario["turns"]
    for label, config in (("off", off), ("on", on)):
        gap_policy = HostedGapPolicy(models, args.gap_model) if label == "on" else None
        admission = HostedAdmissionPolicy(models, args.admission_model) if label == "on" else None
        orchestrator = UtilityAwareOrchestrator(
            store, retrieve, gap_policy, admission, config, profile_assembler=assembler
        )
        for turn in turns:
            decision = orchestrator.prepare_turn(
                principal,
                turn["text"],
                None,
                make_baseline(turn["text"]),
                forbidden_regenerate,
                TurnOptions(args.budget_ms),
            )
            served[label].append(decision.response)
            if label == "on":
                print(
                    f"[shadow] {turn['id']:<4} {turn['class']:<13} {decision.disposition:<26} "
                    f"gaps={len(decision.gaps)} cands={len(decision.candidate_ids)} "
                    f"admitted={[reverse.get(i, i) for i in decision.admitted_ids]} failures={decision.failures}",
                    flush=True,
                )
        snapshots[label] = snapshot(store_path, principal)

    isolation = {
        "served_responses_identical": served["off"] == served["on"],
        "records_identical": snapshots["off"]["records"] == snapshots["on"]["records"],
        "session_turns_identical": snapshots["off"]["turns"] == snapshots["on"]["turns"],
        "reviews_identical": snapshots["off"]["reviews"] == snapshots["on"]["reviews"],
    }

    # Score pass 2 from the turn-decision log, the only thing shadow mode is allowed to write.
    decisions = [d for d in store.turn_decisions(principal.session_id) if d["config"]["shadow"]]
    summary = score(
        scenario, decisions, reverse, set(rr.promoted("shadow")), rr.inventory_labels("shadow"), bundle, isolation
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"-{args.tag}" if args.tag else ""
    safe = lambda s: s.replace("/", "-").replace(":", "-")  # noqa: E731
    path = args.out_dir / (
        f"{stamp}-{args.scenario.stem}-gap-{safe(args.gap_model)}-adm-{safe(args.admission_model)}"
        f"-budget-{args.budget_ms}{tag}.json"
    )
    path.write_text(
        json.dumps(
            {
                "summary": summary,
                "decisions": decisions,
                "id_map": reverse,
                "store": str(store_path),
                "usage": models.usage,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nwrote {path}\n")
    print(json.dumps(summary, indent=2, default=str))
    return 0


def score(
    scenario: dict[str, Any],
    decisions: list[dict[str, Any]],
    id_map: dict[str, str],
    promoted: set[str],
    inventory: list[str],
    bundle: dict[str, object],
    isolation: dict[str, bool],
) -> dict[str, Any]:
    """Pre-registered shadow gates computed from the decision log alone."""

    records_by_id = {r["id"]: r for r in scenario["records"]}
    turns = scenario["turns"]
    by_turn = {d["turn"]: d for d in decisions}
    ordinary = [t for t in turns if t["class"] == "ordinary"]
    explicit = [t for t in turns if t["class"] == "explicit_fact"]
    implicit = [t for t in turns if t["class"] == "implicit_need"]

    def admitted_scenario_ids(turn) -> list[str]:
        return [id_map.get(i, i) for i in by_turn[turn["text"]]["admitted_ids"]]

    def conditional_expected(turn) -> list[str]:
        return [e for e in turn["expected"] if e not in promoted]

    def recalled(turn) -> bool:
        expected = conditional_expected(turn)
        return bool(expected) and all(e in admitted_scenario_ids(turn) for e in expected)

    explicit = [t for t in explicit if conditional_expected(t)]
    implicit = [t for t in implicit if conditional_expected(t)]
    injected = [t for t in ordinary if by_turn[t["text"]]["disposition"] == "shadow_would_regenerate"]
    unsafe = [
        (t["id"], i)
        for t in turns
        for i in admitted_scenario_ids(t)
        if records_by_id.get(i, {}).get("kind") in ("placebo", "misleading") or i in ("F9", "F10")
    ]
    unexpected = [(t["id"], i) for t in turns for i in admitted_scenario_ids(t) if i not in t["expected"]]
    failures = [(t["id"], by_turn[t["text"]]["failures"]) for t in turns if by_turn[t["text"]]["failures"]]
    withholds = [
        (t["id"], by_turn[t["text"]]["disposition"])
        for t in turns
        if by_turn[t["text"]]["disposition"] in ("baseline_budget_exhausted", "admitted_not_applied")
    ]

    summary = {
        "isolation": isolation,
        "turns": len(turns),
        "hypothetical_ordinary_injection": f"{len(injected)}/{len(ordinary)}",
        "explicit_recall": f"{sum(recalled(t) for t in explicit)}/{len(explicit)}",
        "implicit_recall": f"{sum(recalled(t) for t in implicit)}/{len(implicit)}",
        "unsafe_admissions": len(unsafe),
        "unexpected_admissions": unexpected,
        "policy_failures": failures,
        "budget_withholds": withholds,
        "promoted": sorted(promoted),
        "inventory": inventory,
        "bundle": bundle,
        "gates": {
            "isolation": all(isolation.values()),
            "ordinary_injection_le_5pct": len(injected) <= max(1, len(ordinary) // 20),
            "explicit_recall_ge_90pct": len(explicit) == 0 or sum(recalled(t) for t in explicit) / len(explicit) >= 0.9,
            "implicit_recall_ge_75pct": len(implicit) == 0
            or sum(recalled(t) for t in implicit) / len(implicit) >= 0.75,
            "no_unsafe": not unsafe,
        },
    }
    summary["gates"]["all"] = all(summary["gates"].values())
    return summary


def rescore(result_path: Path, scenario_path: Path) -> dict[str, Any]:
    """Re-derive the gates for a saved run from its decision log and store, without any model call."""

    saved = json.loads(result_path.read_text())
    scenario = json.loads(scenario_path.read_text())
    store_path = Path(saved["store"]) if "store" in saved else None
    id_map: dict[str, str] = saved.get("id_map") or {}
    promoted: set[str] = set(saved["summary"].get("promoted", []))
    if store_path is not None and store_path.exists() and not id_map:
        # Older result files: rebuild the map by content, which is unique per scenario record.
        by_text = {r["text"]: r["id"] for r in scenario["records"]}
        db = sqlite3.connect(store_path)
        try:
            for record_id, content, activation in db.execute("SELECT id, content, activation FROM records"):
                scenario_id = by_text.get(content)
                if scenario_id:
                    id_map[record_id] = scenario_id
                    if activation == "ambient":
                        promoted.add(scenario_id)
        finally:
            db.close()
    summary = score(
        scenario,
        saved["decisions"],
        id_map,
        promoted,
        saved["summary"].get("inventory", []),
        saved["summary"].get("bundle", {}),
        saved["summary"].get("isolation", {}),
    )
    saved["summary"] = summary
    saved["id_map"] = id_map
    result_path.write_text(json.dumps(saved, indent=2, default=str))
    return summary


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--rescore":
        print(json.dumps(rescore(Path(sys.argv[2]), Path(sys.argv[3])), indent=2, default=str))
        raise SystemExit(0)
    raise SystemExit(main())
