"""Calibrate the reranker floor on a labelled scenario split.

Builds one real store from the scenario through the ingestor, runs every turn's text as a host-issued search
with the reranker enabled and its floor at zero so every survivor's cross-encoder score is logged, then
sweeps floors offline over `search_log.reranked`. For each floor it reports precision, recall, and F1 of
expected records among reranked candidates, and the share of ordinary turns on which any record clears the
floor. The chosen floor maximises F1 subject to that ordinary-turn share staying at or below the target.

The queries here are the turn texts, not the gap planner's queries, so the sweep calibrates the reranker as a
relevance gate, which is what the LLD asks of it; the utility-aware judge remains the admission decision.

Usage:
    HF_HUB_OFFLINE=1 uv run --extra live --extra local-models python benchmarks/rerank_calibration.py \
        --scenario benchmarks/scenarios/phase0_tune.json --check benchmarks/scenarios/phase0_v5.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.phase0_real_retrieval import RealRetrieval  # noqa: E402


def collect(scenario: dict[str, Any], workdir: Path) -> list[dict[str, Any]]:
    """One row per (turn, reranked candidate) with the cross-encoder score and the hidden label."""

    rr = RealRetrieval(scenario, workdir, rerank_floor=0.0)
    rr.build_arm("calibration", [record["id"] for record in scenario["records"]])
    state = rr._arms["calibration"]  # noqa: SLF001
    store, handlers, principal, id_map = state["store"], state["handlers"], state["principal"], state["id_map"]
    rows: list[dict[str, Any]] = []
    for turn in scenario["turns"]:
        expected = set(turn.get("expected", []))
        result = handlers.memory_search(
            principal, {"queries": [turn["text"]], "k": 8}, context=turn["text"], trigger="auto"
        )
        search_id = result.get("search_id")
        log = store.read_search_log(str(search_id)) if search_id else None
        reranked = (log or {}).get("reranked") or []
        seen: set[str] = set()
        for entry in reranked:
            scenario_id = id_map.get(str(entry["record_id"]))
            if scenario_id is None:
                continue
            seen.add(scenario_id)
            rows.append(
                {
                    "turn": turn["id"],
                    "class": turn["class"],
                    "record": scenario_id,
                    "score": float(entry["score"]),
                    "expected": scenario_id in expected,
                }
            )
        for missing in expected - seen:
            rows.append(
                {"turn": turn["id"], "class": turn["class"], "record": missing, "score": None, "expected": True}
            )
    return rows


def sweep(rows: list[dict[str, Any]], ordinary_turns: int, *, target: float) -> dict[str, Any]:
    floors = [round(step / 100, 2) for step in range(0, 101)]
    table: list[dict[str, Any]] = []
    for floor in floors:
        tp = sum(1 for r in rows if r["expected"] and r["score"] is not None and r["score"] >= floor)
        fn = sum(1 for r in rows if r["expected"] and (r["score"] is None or r["score"] < floor))
        fp = sum(1 for r in rows if not r["expected"] and r["score"] is not None and r["score"] >= floor)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        admitted_ordinary = {
            r["turn"] for r in rows if r["class"] == "ordinary" and r["score"] is not None and r["score"] >= floor
        }
        ordinary_rate = len(admitted_ordinary) / ordinary_turns if ordinary_turns else 0.0
        table.append(
            {
                "floor": floor,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "ordinary_turns_with_survivors": round(ordinary_rate, 3),
            }
        )
    within = [row for row in table if row["ordinary_turns_with_survivors"] <= target]
    chosen = max(within, key=lambda row: (row["f1"], -row["floor"])) if within else None
    best = max(table, key=lambda row: (row["f1"], -row["floor"]))
    return {"table": table, "chosen": chosen, "best_f1_any": best, "target_ordinary_rate": target}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=Path("benchmarks/scenarios/phase0_tune.json"))
    parser.add_argument("--check", type=Path, action="append", default=[], help="Extra splits to report, not select on")
    parser.add_argument("--target-ordinary-rate", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results/rerank"))
    args = parser.parse_args(argv)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report: dict[str, Any] = {"selection_scenario": str(args.scenario), "splits": {}}
    for index, path in enumerate([args.scenario, *args.check]):
        scenario = json.loads(path.read_text())
        rows = collect(scenario, args.out_dir / "stores" / f"{path.stem}-{stamp}")
        ordinary = sum(1 for t in scenario["turns"] if t["class"] == "ordinary")
        result = sweep(rows, ordinary, target=args.target_ordinary_rate)
        report["splits"][path.stem] = {"rows": rows, **result, "selection": index == 0}
        print(f"== {path.stem}: {len(rows)} labelled candidates, {ordinary} ordinary turns")
        for row in result["table"]:
            if row["floor"] in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                print(
                    f"   floor={row['floor']:.2f} P={row['precision']:.2f} R={row['recall']:.2f} F1={row['f1']:.2f} "
                    f"ordinary_survivors={row['ordinary_turns_with_survivors']:.2f}"
                )
        print(f"   chosen (F1 max with ordinary <= {args.target_ordinary_rate}): {result['chosen']}")
        print(f"   best F1 ignoring the ordinary constraint: {result['best_f1_any']}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{stamp}-{args.scenario.stem}-rerank-calibration.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
