"""Evaluate one planner/judge/classifier combination against the committed fitness scenario.

A combination is a bundle: models, prompts, taxonomy, inventory builder, retrieval configuration, and
budgets. Changing any piece invalidates the last pass. This script is the way to find out whether a new
combination works. It prints PASS or FAIL per threshold and exits 1 on failure.

The default arguments are the combination that passed the fifth-split fitness bar. They are not a
recommendation to copy without measuring.

Live run (hosted models, real retrieval and activation):

    HF_HUB_OFFLINE=1 uv run --extra live --extra local-models python benchmarks/evaluate_combination.py

    HF_HUB_OFFLINE=1 uv run --extra live --extra local-models python benchmarks/evaluate_combination.py \\
        --gap-model gpt-4o --admission-model gpt-5.4

Score a saved phase0 result without calling models:

    uv run python benchmarks/evaluate_combination.py --from-result benchmarks/results/phase0/<file>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.fitness import combination_verdict, serving_summary
from benchmarks.phase0_two_arm import main as phase0_main

# The combination that passed configuration A on the fifth split. See docs/usefulness-gate.md section 8g
# and docs/design-contributions.md. gap-v3 here is the inventory-aware prompt; the served orchestrator
# prompt is gap-v3c, which adds a structured category on each gap.
SUPPORTED = {
    "scenario": Path("benchmarks/scenarios/phase0_v5.json"),
    "draft_model": "gpt-5.6-luna",
    "gap_model": "gpt-4o",
    "gap_prompt": "v3",
    "admission_model": "gpt-5.4",
    "admission_prompt": "v3",
    "check_model": "gpt-5.4",
    "activation": "real",
    "activation_model": "gpt-4o",
    "category_prompt": "category-v2",
    "retrieval": "real",
    "arms": "ambient",
}


def _print_verdict(arm: str, summary: dict) -> int:
    verdict = combination_verdict(summary)
    print(f"\n== combination verdict ({arm} arm) ==")
    for line in verdict.lines():
        print(line)
    return 0 if verdict.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-result",
        type=Path,
        default=None,
        help="Score a saved phase0 JSON instead of running models.",
    )
    parser.add_argument("--scenario", type=Path, default=SUPPORTED["scenario"])
    parser.add_argument("--draft-model", default=SUPPORTED["draft_model"])
    parser.add_argument("--gap-model", default=SUPPORTED["gap_model"])
    parser.add_argument("--gap-prompt", default=SUPPORTED["gap_prompt"])
    parser.add_argument("--admission-model", default=SUPPORTED["admission_model"])
    parser.add_argument("--admission-prompt", default=SUPPORTED["admission_prompt"])
    parser.add_argument("--check-model", default=SUPPORTED["check_model"])
    parser.add_argument("--activation", default=SUPPORTED["activation"])
    parser.add_argument("--activation-model", default=SUPPORTED["activation_model"])
    parser.add_argument("--category-prompt", default=SUPPORTED["category_prompt"])
    parser.add_argument("--retrieval", default=SUPPORTED["retrieval"])
    parser.add_argument("--arms", default=SUPPORTED["arms"])
    parser.add_argument("--gap-repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--draft-cache", type=Path, default=None)
    parser.add_argument("--tag", default="combination")
    parser.add_argument("--rewrite-model", default=None)
    parser.add_argument("--rerank-floor", type=float, default=None)
    parser.add_argument("--rerank-mode", choices=["rrf_cross_encoder", "cross_encoder_only"], default=None)
    parser.add_argument("--rerank-timeout-ms", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results/phase0"))
    args = parser.parse_args(argv)

    if args.from_result is not None:
        payload = json.loads(args.from_result.read_text())
        print(
            f"from {args.from_result} scenario={payload.get('scenario')} "
            f"gap={payload.get('gap_model')}/{payload.get('gap_prompt')} "
            f"admission={payload.get('admission_model')}/{payload.get('admission_prompt')} "
            f"activation={payload.get('activation')} retrieval={payload.get('retrieval')}",
            flush=True,
        )
        arm, summary = serving_summary(payload)
        return _print_verdict(arm, summary)

    forwarded = [
        "--scenario",
        str(args.scenario),
        "--draft-model",
        args.draft_model,
        "--gap-model",
        args.gap_model,
        "--gap-prompt",
        args.gap_prompt,
        "--admission-model",
        args.admission_model,
        "--admission-prompt",
        args.admission_prompt,
        "--check-model",
        args.check_model,
        "--activation",
        args.activation,
        "--activation-model",
        args.activation_model,
        "--category-prompt",
        args.category_prompt,
        "--retrieval",
        args.retrieval,
        "--arms",
        args.arms,
        "--gap-repeats",
        str(args.gap_repeats),
        "--workers",
        str(args.workers),
        "--tag",
        args.tag,
        "--out-dir",
        str(args.out_dir),
        "--shadow-judge",
        "--fail-on-verdict",
    ]
    if args.draft_cache is not None:
        forwarded.extend(["--draft-cache", str(args.draft_cache)])
    if args.rewrite_model:
        forwarded.extend(["--rewrite-model", args.rewrite_model])
    if args.rerank_floor is not None:
        forwarded.extend(["--rerank-floor", str(args.rerank_floor)])
    if args.rerank_mode is not None:
        forwarded.extend(["--rerank-mode", args.rerank_mode])
    if args.rerank_timeout_ms is not None:
        forwarded.extend(["--rerank-timeout-ms", str(args.rerank_timeout_ms)])
    return phase0_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
