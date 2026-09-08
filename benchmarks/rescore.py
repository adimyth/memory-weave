"""Rescore saved configuration-A results under an adjudication overlay, without calling any model.

Recall is unchanged: it counts the scenario's expected (required) records. Precision counts an admitted
record as helpful when it is expected or when the overlay marks it required or helpful for that turn. Both
the strict figure and the overlay figure are printed and saved, so nothing about the original scoring is
hidden.

    uv run python benchmarks/rescore.py benchmarks/results/phase0/phase11-baseline-v5/*.json ...
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.phase0_two_arm import load_allowed  # noqa: E402


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({round(100 * n / d) if d else 0}%)"


def rescore(result_path: Path, overlay: Path) -> dict[str, Any]:
    data = json.loads(result_path.read_text())
    scenario_stem = Path(data["scenario"]).stem
    allowed = load_allowed(overlay, scenario_stem)
    scenario = json.loads(Path(data["scenario"]).read_text())
    turns = {t["id"]: t for t in scenario["turns"]}
    out: dict[str, Any] = {"result": str(result_path), "scenario": scenario_stem, "arms": {}}
    for arm, results in data["results"].items():
        admitted_total = strict = overlay_useful = 0
        extras_strict: list[tuple[str, str]] = []
        extras_overlay: list[tuple[str, str]] = []
        recall_hits = recall_total = 0
        for r in results:
            expected = set(turns[r["turn_id"]]["expected"])
            allow = allowed.get(r["turn_id"], set())
            if turns[r["turn_id"]]["class"] != "ordinary":
                recall_total += len(expected)
                recall_hits += len(expected & set(r["admitted"]))
            for record_id in r["admitted"]:
                admitted_total += 1
                if record_id in expected:
                    strict += 1
                    overlay_useful += 1
                elif record_id in allow:
                    overlay_useful += 1
                    extras_strict.append((r["turn_id"], record_id))
                else:
                    extras_strict.append((r["turn_id"], record_id))
                    extras_overlay.append((r["turn_id"], record_id))
        out["arms"][arm] = {
            "usefulness_precision_strict": _pct(strict, admitted_total),
            "usefulness_precision": _pct(overlay_useful, admitted_total),
            "recall_of_required": _pct(recall_hits, recall_total),
            "non_expected_admissions": extras_strict,
            "non_helpful_admissions": extras_overlay,
            "precision_gate_95": (overlay_useful / admitted_total >= 0.95) if admitted_total else None,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--overlay", type=Path, default=Path("benchmarks/scenarios/overlays/label_adjudication.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results/rescore"))
    args = parser.parse_args(argv)
    report = {"overlay": str(args.overlay), "rescored_at": datetime.now(UTC).isoformat(), "runs": []}
    for path in args.results:
        scored = rescore(path, args.overlay)
        report["runs"].append(scored)
        for arm, values in scored["arms"].items():
            print(
                f"{path.parent.name:22} {scored['scenario']:9} {arm:11} "
                f"strict {values['usefulness_precision_strict']:14} overlay {values['usefulness_precision']:14} "
                f"recall {values['recall_of_required']:14} "
                f"gate95 {values['precision_gate_95']}  not_helpful {values['non_helpful_admissions']}"
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-rescore.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
