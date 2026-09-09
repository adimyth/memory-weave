"""Judge-pool statistics from saved configuration-A results, with no model call.

The cross-encoder can only earn its place as a candidate control: it decides what the judge sees, not what
is admitted. So the comparison that matters between ranking configurations is the pool it hands the judge
on every turn where the planner fired: how many candidates, how many of them the scenario never expected
("weak candidates"), whether the expected records were still in it, and what the stage cost.

    uv run python benchmarks/pool_stats.py benchmarks/results/phase0/phase11-*-v5/*.json ...
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_PLACEBO_QUERY = "forced placebo"


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({round(100 * n / d) if d else 0}%)"


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def pool_stats(result_path: Path) -> dict[str, Any]:
    data = json.loads(result_path.read_text())
    scenario = json.loads(Path(data["scenario"]).read_text())
    turns = {t["id"]: t for t in scenario["turns"]}
    out: dict[str, Any] = {
        "result": str(result_path),
        "scenario": Path(data["scenario"]).stem,
        "ranking": data.get("ranking") or ("rrf_cross_encoder" if data.get("rerank_floor") is not None else "rrf_only"),
        "rerank_floor": data.get("rerank_floor"),
        "rerank_statuses": data.get("rerank_statuses", {}),
        "rerank_stage_ms": data.get("rerank_stage_ms", {}),
        "arms": {},
    }
    for arm, results in data["results"].items():
        gap_turns = 0
        pool_sizes: list[int] = []
        weak_total = 0
        expected_total = expected_in_pool = 0
        ordinary_pools: list[int] = []
        retrieval_s: list[float] = []
        admitted_weak = 0
        for r in results:
            if not r["gaps"]:
                continue
            gap_turns += 1
            pool = [c["id"] for c in r["candidates"] if c.get("query") != _PLACEBO_QUERY]
            expected = set(turns[r["turn_id"]]["expected"])
            weak = [c for c in pool if c not in expected]
            pool_sizes.append(len(pool))
            weak_total += len(weak)
            retrieval_s.append(float(r["timing"]["retrieval_s"]))
            admitted_weak += sum(1 for c in r["admitted"] if c not in expected and c in pool)
            if turns[r["turn_id"]]["class"] == "ordinary":
                ordinary_pools.append(len(pool))
            else:
                expected_total += len(expected)
                expected_in_pool += len(expected & set(pool))
        out["arms"][arm] = {
            "gap_turns": gap_turns,
            "pool_mean": round(sum(pool_sizes) / len(pool_sizes), 2) if pool_sizes else None,
            "weak_candidates_total": weak_total,
            "weak_per_gap_turn": round(weak_total / gap_turns, 2) if gap_turns else None,
            "expected_in_pool": _pct(expected_in_pool, expected_total),
            "ordinary_gap_turns_with_a_pool": sum(1 for n in ordinary_pools if n),
            "weak_candidates_admitted": admitted_weak,
            "retrieval_s_p50": round(_quantile(retrieval_s, 0.5) or 0.0, 3) if retrieval_s else None,
            "retrieval_s_p95": round(_quantile(retrieval_s, 0.95) or 0.0, 3) if retrieval_s else None,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results/rerank"))
    args = parser.parse_args(argv)
    report: dict[str, Any] = {"computed_at": datetime.now(UTC).isoformat(), "runs": []}
    for path in args.results:
        stats = pool_stats(path)
        report["runs"].append(stats)
        for arm, values in stats["arms"].items():
            print(
                f"{path.parent.name:24} {stats['scenario']:9} {stats['ranking']:18} {arm:11} "
                f"gap_turns {values['gap_turns']:3} pool {values['pool_mean']!s:5} "
                f"weak/turn {values['weak_per_gap_turn']!s:5} expected_in_pool {values['expected_in_pool']:12} "
                f"weak_admitted {values['weak_candidates_admitted']} "
                f"retrieval p50/p95 {values['retrieval_s_p50']}/{values['retrieval_s_p95']} s "
                f"rerank {stats['rerank_statuses'].get(arm, {})} {stats['rerank_stage_ms'].get(arm, {})}"
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-pool-stats.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
