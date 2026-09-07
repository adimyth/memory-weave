"""Replay logged host-issued searches to test whether the relevance gate can separate turn classes.

This reads only `search_log` and `records` from a completed vertical-slice attempt directory. It makes no
model calls and re-executes no searches, because the log stores every candidate and its score.

Usage:
    uv run python benchmarks/analyse_gate.py benchmarks/results/vertical-slice/attempt-XXXX
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.vertical_slice import CONVERSATION  # noqa: E402

_CATEGORIES = {turn.text: turn.category for turn in CONVERSATION}
_FLOORS = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One dense candidate from a logged search, tagged by whether the record is about the principal."""

    record_id: str
    score: float
    profile: bool
    attribute: str | None
    content: str


@dataclass(frozen=True, slots=True)
class LoggedSearch:
    """One host-issued search, its turn class, its candidates, and what it actually returned."""

    category: str
    dense: list[Candidate]
    entity_hits: int
    returned: list[Candidate]

    @property
    def profile(self) -> list[Candidate]:
        return [c for c in self.dense if c.profile]

    @property
    def topical(self) -> list[Candidate]:
        return [c for c in self.dense if not c.profile]


def load(attempt_dir: Path) -> list[LoggedSearch]:
    """Read every host-issued search from every run database in one attempt directory."""

    searches: list[LoggedSearch] = []
    databases = sorted(attempt_dir.glob("run-*.sqlite"))
    if not databases:
        raise SystemExit(f"No run databases under {attempt_dir}")
    for database in databases:
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            searches.extend(_load_one(connection))
        finally:
            connection.close()
    return searches


def _load_one(connection: sqlite3.Connection) -> list[LoggedSearch]:
    principal = _principal_entity(connection)
    records = {
        row["id"]: (row["subject_entity_id"], row["attribute"], row["content"])
        for row in connection.execute("SELECT id, subject_entity_id, attribute, content FROM records")
    }

    def candidate(record_id: str, score: float) -> Candidate:
        subject, attribute, content = records.get(record_id, (None, None, ""))
        return Candidate(record_id, score, subject == principal and subject is not None, attribute, content)

    searches: list[LoggedSearch] = []
    for row in connection.execute("SELECT request, dense, entity, returned FROM search_log WHERE trigger = 'auto'"):
        category = _CATEGORIES.get(json.loads(row["request"])["queries"][0])
        if category is None:
            continue
        dense = [candidate(hit["record_id"], float(hit["score"])) for hit in json.loads(row["dense"])]
        scores = {hit.record_id: hit.score for hit in dense}
        returned = [candidate(record_id, scores.get(record_id, 0.0)) for record_id in json.loads(row["returned"])]
        searches.append(LoggedSearch(category, dense, len(json.loads(row["entity"])), returned))
    return searches


def _principal_entity(connection: sqlite3.Connection) -> str | None:
    """Return the principal's own person entity, which marks a record as being about the user."""

    row = connection.execute(
        """
        SELECT e.id FROM entities AS e
        JOIN entity_aliases AS a ON a.entity_id = e.id
        WHERE e.kind = 'person' AND a.alias_norm LIKE 'user-%'
        ORDER BY e.created_at LIMIT 1
        """
    ).fetchone()
    return None if row is None else str(row["id"])


_EXPECTATION = {
    "preference": "store it; no recall needed",
    "correction": "store it and supersede the earlier preference",
    "entity": "store it against the named person",
    "memory_applies": "recall the stored preference",
    "ordinary": "recall nothing",
}


def scenarios(attempt_dir: Path) -> str:
    """Render one row per scripted turn: what was expected, what happened, and whether that is a pass."""

    runs = sorted(attempt_dir.glob("run-*.sqlite"))
    writes: dict[int, int] = {}
    recalled: dict[int, int] = {}
    index = {turn.text: position for position, turn in enumerate(CONVERSATION, start=1)}
    for database in runs:
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            for row in connection.execute("SELECT payload FROM events WHERE kind = 'vertical_slice.tool_attempt'"):
                turn = int(json.loads(row["payload"])["turn"])
                writes[turn] = writes.get(turn, 0) + 1
            for row in connection.execute("SELECT request, returned FROM search_log"):
                turn = index.get(json.loads(row["request"])["queries"][0])
                if turn is not None and json.loads(row["returned"]):
                    recalled[turn] = recalled.get(turn, 0) + 1
        finally:
            connection.close()

    total = len(runs)
    lines = [
        "| # | Class | Turn | Expected | Wrote | Recalled | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for position, turn in enumerate(CONVERSATION, start=1):
        wrote, got = writes.get(position, 0), recalled.get(position, 0)
        lines.append(
            f"| {position} | {turn.category} | {turn.text[:46]} | {_EXPECTATION[turn.category]} "
            f"| {wrote}/{total} | {got}/{total} | {_verdict(turn.category, wrote, got, total)} |"
        )
    return "\n".join(lines)


def _verdict(category: str, wrote: int, recalled: int, total: int) -> str:
    """Score one scripted turn against what its class was supposed to do."""

    if category in ("preference", "correction", "entity"):
        return "pass" if wrote == total else f"**write missed** ({wrote}/{total})"
    if category == "memory_applies":
        return "pass" if recalled else "**recall missed**"
    return "pass" if not recalled else f"**false inject** ({recalled}/{total})"


def _best(searches: Sequence[LoggedSearch], category: str, which: str) -> list[float]:
    values = []
    for search in searches:
        pool = search.dense if which == "all" else (search.profile if which == "profile" else search.topical)
        if search.category == category:
            values.append(max((c.score for c in pool), default=0.0))
    return sorted(values)


def _distribution(label: str, values: Sequence[float]) -> str:
    if not values:
        return f"  {label:>16}: no searches"
    return (
        f"  {label:>16}: n={len(values):<3} min={min(values):.2f} "
        f"median={statistics.median(values):.2f} max={max(values):.2f}"
    )


def report(searches: Sequence[LoggedSearch]) -> None:
    """Print the three findings the gate analysis depends on."""

    counts: dict[str, int] = {}
    for search in searches:
        counts[search.category] = counts.get(search.category, 0) + 1
    print(f"host-issued searches: {len(searches)}  by class: {dict(sorted(counts.items()))}")
    print(f"currently returning memory: {sum(1 for s in searches if s.returned)}\n")

    print("== 1. can a dense floor separate the two turn classes? ==")
    for which in ("all", "profile", "topical"):
        print(f" {which} candidates:")
        ordinary = _best(searches, "ordinary", which)
        applies = _best(searches, "memory_applies", which)
        print(_distribution("ordinary", ordinary))
        print(_distribution("memory applies", applies))
        if ordinary and applies:
            verdict = "YES" if min(applies) > max(ordinary) else "NO"
            print(
                f"  {'separable?':>16}: {verdict}  "
                f"(applicable min {min(applies):.2f} vs ordinary max {max(ordinary):.2f})"
            )
    print()

    print("== 2. floor sweep over all candidates ==")
    print(f"  {'floor':>6} | {'ordinary injected':>17} | {'applicable recalled':>19}")
    ordinary = [s for s in searches if s.category == "ordinary"]
    applies = [s for s in searches if s.category == "memory_applies"]
    for floor in _FLOORS:
        hit_o = sum(any(c.score >= floor for c in s.dense) for s in ordinary)
        hit_a = sum(any(c.score >= floor for c in s.dense) for s in applies)
        print(f"  {floor:>6.2f} | {hit_o:>7} of {len(ordinary):<7} | {hit_a:>8} of {len(applies):<8}")
    print()

    print("== 3. what did the turns that needed memory actually get? ==")
    profile_returned = sum(1 for s in applies for c in s.returned if c.profile)
    topical_returned = sum(1 for s in applies for c in s.returned if not c.profile)
    print(f"  profile records (about the user)   : {profile_returned}")
    print(f"  topical records (about the world)  : {topical_returned}")
    worst = sorted(
        ((c.score, c.attribute, c.content) for s in ordinary for c in s.topical),
        reverse=True,
    )[:3]
    if worst:
        print("\n  highest topical scores on ordinary turns:")
        for score, attribute, content in worst:
            print(f"    {score:.2f}  [{attribute}] {content[:60]}")


def main(argv: Sequence[str] | None = None) -> int:
    """Analyse one attempt directory and print the gate findings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_dir", type=Path)
    parser.add_argument(
        "--scenarios", action="store_true", help="Print the per-turn scenario table as Markdown and exit."
    )
    args = parser.parse_args(argv)
    if args.scenarios:
        print(scenarios(args.attempt_dir))
        return 0
    report(load(args.attempt_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
