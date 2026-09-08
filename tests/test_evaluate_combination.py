"""evaluate_combination.py scores a saved result without calling models."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.evaluate_combination import main as evaluate_main


def test_from_result_exits_zero_on_pass(tmp_path: Path) -> None:
    payload = {
        "scenario": "benchmarks/scenarios/phase0_v5.json",
        "gap_model": "gpt-4o",
        "gap_prompt": "v3",
        "admission_model": "gpt-5.4",
        "admission_prompt": "v3",
        "activation": "real",
        "retrieval": "real",
        "summaries": {
            "ambient": {
                "ordinary_injection": "0/20 (0%)",
                "explicit_fact_recall": "10/10 (100%)",
                "implicit_need_recall": "7/8 (88%)",
                "placebo_admitted": 0,
                "misleading_admitted": 0,
                "unrelated_private_admitted": 0,
                "unsafe_promotions": 0,
                "eligible_preferences_promoted": "3/3 (100%)",
                "scoped_preferences_kept_conditional": "1/1 (100%)",
            }
        },
    }
    path = tmp_path / "pass.json"
    path.write_text(json.dumps(payload))
    assert evaluate_main(["--from-result", str(path)]) == 0


def test_from_result_exits_one_on_fail(tmp_path: Path) -> None:
    payload = {
        "summaries": {
            "ambient": {
                "ordinary_injection": "0/20 (0%)",
                "explicit_fact_recall": "8/10 (80%)",
                "implicit_need_recall": "7/8 (88%)",
                "placebo_admitted": 0,
                "misleading_admitted": 0,
                "unrelated_private_admitted": 0,
            }
        }
    }
    path = tmp_path / "fail.json"
    path.write_text(json.dumps(payload))
    assert evaluate_main(["--from-result", str(path)]) == 1
