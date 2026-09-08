"""Combination fitness scoring: pass/fail from a phase0 summary, no model calls."""

from __future__ import annotations

from benchmarks.fitness import combination_verdict, parse_ratio, serving_summary


def _passing_summary(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
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
    base.update(overrides)
    return base


def test_parse_ratio() -> None:
    assert parse_ratio("9/10 (90%)") == (9, 10)
    assert parse_ratio("0/20 (0%)") == (0, 20)
    assert parse_ratio("n/a") is None
    assert parse_ratio(None) is None


def test_passing_fifth_split_summary_passes() -> None:
    verdict = combination_verdict(_passing_summary())
    assert verdict.passed
    assert all(check.passed for check in verdict.checks)
    assert verdict.lines()[-1] == "  COMBINATION PASS"


def test_recall_miss_fails_the_combination() -> None:
    verdict = combination_verdict(_passing_summary(**{"implicit_need_recall": "5/8 (62%)"}))
    assert not verdict.passed
    failed = [check.name for check in verdict.checks if not check.passed]
    assert failed == ["implicit memory-needed recall >= 75%"]
    assert verdict.lines()[-1] == "  COMBINATION FAIL"


def test_unsafe_admission_fails_the_combination() -> None:
    verdict = combination_verdict(_passing_summary(misleading_admitted=1))
    assert not verdict.passed
    assert any(check.name == "no misleading admitted" and not check.passed for check in verdict.checks)


def test_fixture_activation_skips_promotion_checks() -> None:
    summary = _passing_summary()
    del summary["unsafe_promotions"]
    del summary["eligible_preferences_promoted"]
    del summary["scoped_preferences_kept_conditional"]
    verdict = combination_verdict(summary)
    assert verdict.passed
    assert all("promot" not in check.name for check in verdict.checks)


def test_partial_eligible_promotion_fails() -> None:
    verdict = combination_verdict(_passing_summary(**{"eligible_preferences_promoted": "2/3 (67%)"}))
    assert not verdict.passed


def test_serving_summary_prefers_the_ambient_arm() -> None:
    payload = {
        "summaries": {
            "conditional": {"ordinary_injection": "0/20 (0%)"},
            "ambient": {"ordinary_injection": "1/20 (5%)"},
        }
    }
    name, summary = serving_summary(payload)
    assert name == "ambient"
    assert summary["ordinary_injection"] == "1/20 (5%)"
