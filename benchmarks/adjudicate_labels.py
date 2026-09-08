"""Blind adjudication of a disputed scenario label by an independent reviewer model.

The reviewer sees the user's turn, the records the assistant already has, and the disputed record. It
never sees the judge's verdict or reasoning, the original label, or this project's gates. It answers
whether the disputed record is required, helpful but optional, or unhelpful for answering that turn.

The result is written to an overlay file beside the scenarios. The blind labels in the scenario files are
never edited: recall keeps scoring against `expected`, and precision additionally allows the records the
overlay marks helpful.

    uv run --extra live python benchmarks/adjudicate_labels.py --reviewer openrouter:anthropic/claude-sonnet-4.6
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

OVERLAY_PATH = Path("benchmarks/scenarios/overlays/label_adjudication.json")

DISPUTES: list[dict[str, str]] = [
    {"scenario": "phase0_v5", "turn": "I5", "record": "F1"},
    {"scenario": "phase0_v4", "turn": "I5", "record": "F1"},
]

_SYSTEM = (
    "You review evaluation labels for an assistant that can draw on stored facts about a user. You are "
    "given one user message, the stored facts the assistant will already use when answering it, and one "
    "further stored fact under review. Decide how that further fact bears on answering this message well:\n"
    "- required: a good answer is wrong or incomplete without it;\n"
    "- helpful_optional: a good answer is possible without it, but using it makes the answer more useful, "
    "more specific, or saves the user a clarifying exchange;\n"
    "- unhelpful: it does not improve the answer, or it distracts.\n"
    "Judge only this message and these facts. Reply with a single JSON object: "
    '{"verdict": "required" | "helpful_optional" | "unhelpful", "reason": "<two sentences at most>"}'
)


def render(turn: dict[str, Any], already: list[dict[str, Any]], disputed: dict[str, Any]) -> str:
    lines = [f"User message:\n{turn['text']}", "", "Stored facts the assistant will already use:"]
    lines.extend(f"- {record['text']}" for record in already)
    if not already:
        lines.append("- none")
    lines.append("")
    lines.append(f"Stored fact under review:\n- {disputed['text']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer", default="openrouter:anthropic/claude-sonnet-4.6")
    parser.add_argument("--out", type=Path, default=OVERLAY_PATH)
    args = parser.parse_args(argv)

    models = Models()
    stamp = datetime.now(UTC).isoformat()
    adjudications: list[dict[str, Any]] = []
    for dispute in DISPUTES:
        scenario = json.loads(Path(f"benchmarks/scenarios/{dispute['scenario']}.json").read_text())
        turns = {t["id"]: t for t in scenario["turns"]}
        records = {r["id"]: r for r in scenario["records"]}
        turn = turns[dispute["turn"]]
        already = [records[i] for i in turn["expected"] if i != dispute["record"]]
        prompt = render(turn, already, records[dispute["record"]])
        raw = models.complete(args.reviewer, _SYSTEM, prompt, json_mode=True)
        parsed = json.loads(raw)
        verdict = str(parsed.get("verdict", "")).strip()
        if verdict not in ("required", "helpful_optional", "unhelpful"):
            raise SystemExit(f"reviewer returned an unknown verdict: {raw}")
        adjudications.append(
            {
                **dispute,
                "turn_text": turn["text"],
                "original_expected": list(turn["expected"]),
                "record_text": records[dispute["record"]]["text"],
                "verdict": verdict,
                "reason": str(parsed.get("reason", "")),
                "raw_response": raw,
            }
        )
        print(f"{dispute['scenario']} {dispute['turn']} {dispute['record']}: {verdict} — {parsed.get('reason', '')}")

    overlay = {
        "adjudicated_at": stamp,
        "reviewer": args.reviewer,
        "blind": (
            "The reviewer saw the turn, the already-expected records, and the disputed record; "
            "never the judge's reasoning or the original label."
        ),
        "prompt_version": "adjudicate-v1",
        "system_prompt": _SYSTEM,
        "scoring": {
            "recall": "against the scenario's expected list only (required records)",
            "precision": (
                "an admitted record counts as helpful when it is expected or marked required or helpful_optional here"
            ),
        },
        "adjudications": adjudications,
        "allowed": {
            f"{a['scenario']}/{a['turn']}": [a["record"]]
            for a in adjudications
            if a["verdict"] in ("required", "helpful_optional")
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(overlay, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
