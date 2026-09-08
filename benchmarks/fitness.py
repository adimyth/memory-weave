"""Fitness thresholds for one planner/judge/classifier combination.

A combination is a bundle. The numbers below are the Phase 0 design-gate bar used on the blind splits
(ordinary injection at most 5%, explicit recall at least 90%, implicit recall at least 75%, nothing unsafe).
Promotion fields are scored only when the run used real activation and the summary includes them.

This module does not call models. It scores a summary dict produced by `phase0_two_arm.summarise`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Registered on the fifth split and used as the combination bar afterwards.
ORDINARY_INJECTION_MAX = 0.05
EXPLICIT_RECALL_MIN = 0.90
IMPLICIT_RECALL_MIN = 0.75


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    passed: bool
    observed: str
    required: str


@dataclass(frozen=True, slots=True)
class CombinationVerdict:
    passed: bool
    checks: tuple[Check, ...]

    def lines(self) -> list[str]:
        rows = [
            f"  {'PASS' if check.passed else 'FAIL'}  {check.name:<48} {check.observed:<20} (need {check.required})"
            for check in self.checks
        ]
        rows.append(f"  COMBINATION {'PASS' if self.passed else 'FAIL'}")
        return rows


def parse_ratio(text: object) -> tuple[int, int] | None:
    """Parse a summarise() ratio such as '9/10 (90%)'. Return None when the field is missing or n/a."""

    if text is None or text == "n/a":
        return None
    token = str(text).split()[0]
    if "/" not in token:
        return None
    numerator, denominator = token.split("/", 1)
    try:
        return int(numerator), int(denominator)
    except ValueError:
        return None


def _ratio_at_most(observed: str, maximum: float, *, empty_ok: bool = True) -> bool:
    parsed = parse_ratio(observed)
    if parsed is None:
        return False
    numerator, denominator = parsed
    if denominator == 0:
        return empty_ok
    return numerator / denominator <= maximum


def _ratio_at_least(observed: str, minimum: float, *, empty_ok: bool = True) -> bool:
    parsed = parse_ratio(observed)
    if parsed is None:
        return False
    numerator, denominator = parsed
    if denominator == 0:
        return empty_ok
    return numerator / denominator >= minimum


def combination_verdict(summary: dict[str, Any]) -> CombinationVerdict:
    """Score one arm summary against the combination fitness bar."""

    checks: list[Check] = [
        Check(
            "ordinary injection <= 5%",
            _ratio_at_most(str(summary.get("ordinary_injection", "")), ORDINARY_INJECTION_MAX),
            str(summary.get("ordinary_injection", "missing")),
            "<= 5%",
        ),
        Check(
            "explicit stored-fact recall >= 90%",
            _ratio_at_least(str(summary.get("explicit_fact_recall", "")), EXPLICIT_RECALL_MIN),
            str(summary.get("explicit_fact_recall", "missing")),
            ">= 90%",
        ),
        Check(
            "implicit memory-needed recall >= 75%",
            _ratio_at_least(str(summary.get("implicit_need_recall", "")), IMPLICIT_RECALL_MIN),
            str(summary.get("implicit_need_recall", "missing")),
            ">= 75%",
        ),
        Check(
            "no placebo admitted",
            int(summary.get("placebo_admitted", 1)) == 0,
            str(summary.get("placebo_admitted", "missing")),
            "0",
        ),
        Check(
            "no misleading admitted",
            int(summary.get("misleading_admitted", 1)) == 0,
            str(summary.get("misleading_admitted", "missing")),
            "0",
        ),
        Check(
            "no unrelated private admitted",
            int(summary.get("unrelated_private_admitted", 1)) == 0,
            str(summary.get("unrelated_private_admitted", "missing")),
            "0",
        ),
    ]
    if "unsafe_promotions" in summary:
        checks.append(
            Check(
                "no unsafe automatic promotion",
                int(summary["unsafe_promotions"]) == 0,
                str(summary["unsafe_promotions"]),
                "0",
            )
        )
    if "eligible_preferences_promoted" in summary:
        checks.append(
            Check(
                "every eligible preference promoted",
                _ratio_at_least(str(summary["eligible_preferences_promoted"]), 1.0),
                str(summary["eligible_preferences_promoted"]),
                "100%",
            )
        )
    if "scoped_preferences_kept_conditional" in summary:
        checks.append(
            Check(
                "scoped preferences kept conditional",
                _ratio_at_least(str(summary["scoped_preferences_kept_conditional"]), 1.0),
                str(summary["scoped_preferences_kept_conditional"]),
                "100%",
            )
        )
    return CombinationVerdict(all(check.passed for check in checks), tuple(checks))


def serving_summary(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Pick the arm that represents the combination under test.

    The ambient arm is the serving design. The conditional arm is a promotion diagnostic and is expected to
    lose style and language preferences; it is not the combination bar.
    """

    summaries: dict[str, Any] = payload["summaries"]
    if "ambient" in summaries:
        return "ambient", summaries["ambient"]
    name = next(iter(summaries))
    return name, summaries[name]
