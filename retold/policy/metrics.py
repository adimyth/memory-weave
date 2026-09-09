"""Metrics over the turn-decision log, for reports and for a host's rollback decisions.

Every turn decision is assigned exactly one stage outcome, so a change in served injection or recall can be
attributed to one stage:

- path_disabled: the utility-aware path was off or the budget was zero, so nothing ran;
- planner_silence: the planner returned no gaps, so no retrieval happened;
- retrieval_miss: the planner named gaps but retrieval returned no conditional candidates;
- judge_rejection: candidates were judged and none admitted;
- policy_failure: a policy failed, timed out beyond its stage limit, or returned malformed output;
- budget_withheld: the per-request budget cut the path short, before admission or before regeneration;
- admitted: memory was admitted; served when it regenerated, shadow when it only would have.

Production turns carry no labels, so unsafe and unexpected admissions cannot be counted here; those come
from the fitness suite. What a host can watch is the served and shadow admission rates, the stage counts,
latency, cost, the admitted-record distribution, and the review backlog, per bundle and per time window.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

from retold.store import Store

StageOutcome = Literal[
    "path_disabled",
    "planner_silence",
    "retrieval_miss",
    "judge_rejection",
    "policy_failure",
    "budget_withheld",
    "admitted",
]
STAGE_OUTCOMES: tuple[StageOutcome, ...] = (
    "path_disabled",
    "planner_silence",
    "retrieval_miss",
    "judge_rejection",
    "policy_failure",
    "budget_withheld",
    "admitted",
)


def stage_outcome(decision: Mapping[str, Any]) -> StageOutcome:
    """Map one turn decision to its single stage outcome."""

    disposition = decision["disposition"]
    gap_status = decision.get("gap_status")
    config = decision.get("config") or {}
    if disposition == "baseline_no_gaps":
        return "path_disabled" if gap_status == "skipped" or not config.get("gap_enabled", False) else "planner_silence"
    if disposition == "baseline_budget_exhausted":
        return "path_disabled" if gap_status == "skipped" else "budget_withheld"
    if disposition == "baseline_no_candidates":
        return "retrieval_miss"
    if disposition == "baseline_empty_admission":
        return "judge_rejection"
    if disposition == "baseline_policy_failure":
        return "policy_failure"
    if disposition == "admitted_not_applied":
        return "budget_withheld"
    if disposition in ("regenerated", "shadow_would_regenerate"):
        return "admitted"
    return "policy_failure"


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[index]


@dataclass(slots=True)
class MetricsReport:
    window_start: str | None
    window_end: str | None
    bundle_hash: str | None
    turns: int
    served_turns: int
    shadow_turns: int
    stages: dict[str, int]
    served_admission_rate: float
    shadow_admission_rate: float
    planner_fire_rate: float
    added_latency_ms_p50: float
    added_latency_ms_p95: float
    added_latency_gap_turns_ms_p50: float
    added_latency_gap_turns_ms_p95: float
    tokens: dict[str, dict[str, int]]
    admitted_records: dict[str, int]
    harmful_verdicts: int
    failures: dict[str, int]
    review_backlog_open: int
    review_backlog_oldest_days: float
    by_bundle: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate(
    store: Store,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    bundle_hash: str | None = None,
    session_id: str | None = None,
) -> MetricsReport:
    """Aggregate the turn-decision log into one report. Reads only; a host may call this on a schedule."""

    decisions = store.turn_decisions(session_id, since=since, until=until, bundle_hash=bundle_hash)
    stages: dict[str, int] = {name: 0 for name in STAGE_OUTCOMES}
    by_bundle: dict[str, dict[str, int]] = {}
    served = [d for d in decisions if not d["shadow"]]
    shadow = [d for d in decisions if d["shadow"]]
    added_all: list[float] = []
    added_gap: list[float] = []
    tokens: dict[str, dict[str, int]] = {}
    admitted_records: dict[str, int] = {}
    harmful = 0
    failures: dict[str, int] = {}
    fired = 0
    eligible = 0
    for d in decisions:
        outcome = stage_outcome(d)
        stages[outcome] += 1
        key = d.get("bundle_hash") or "unhashed"
        by_bundle.setdefault(key, {name: 0 for name in STAGE_OUTCOMES})[outcome] += 1
        timings = d.get("timings_ms") or {}
        added = sum(v for k, v in timings.items() if k != "draft")
        added_all.append(added)
        if d.get("gaps"):
            added_gap.append(added)
        if outcome != "path_disabled":
            eligible += 1
            if d.get("gaps"):
                fired += 1
        for model, usage in (d.get("usage") or {}).items():
            bucket = tokens.setdefault(model, {"prompt": 0, "completion": 0})
            bucket["prompt"] += int(usage.get("prompt", 0))
            bucket["completion"] += int(usage.get("completion", 0))
        for record_id in d.get("admitted_ids") or []:
            admitted_records[record_id] = admitted_records.get(record_id, 0) + 1
        for verdict in d.get("verdicts") or []:
            if verdict.get("verdict") == "potentially_harmful":
                harmful += 1
        for failure in d.get("failures") or []:
            failures[failure] = failures.get(failure, 0) + 1

    served_admitted = sum(1 for d in served if stage_outcome(d) == "admitted")
    shadow_admitted = sum(1 for d in shadow if stage_outcome(d) == "admitted")
    open_reviews = store.open_activation_reviews()
    oldest = 0.0
    if open_reviews:
        from retold.util import now

        current = now()
        oldest = max(
            (current - datetime.fromisoformat(str(r["created_at"]))).total_seconds() / 86400 for r in open_reviews
        )

    return MetricsReport(
        window_start=since.isoformat() if since else None,
        window_end=until.isoformat() if until else None,
        bundle_hash=bundle_hash,
        turns=len(decisions),
        served_turns=len(served),
        shadow_turns=len(shadow),
        stages=stages,
        served_admission_rate=served_admitted / len(served) if served else 0.0,
        shadow_admission_rate=shadow_admitted / len(shadow) if shadow else 0.0,
        planner_fire_rate=fired / eligible if eligible else 0.0,
        added_latency_ms_p50=_percentile(added_all, 0.5),
        added_latency_ms_p95=_percentile(added_all, 0.95),
        added_latency_gap_turns_ms_p50=_percentile(added_gap, 0.5),
        added_latency_gap_turns_ms_p95=_percentile(added_gap, 0.95),
        tokens=tokens,
        admitted_records=dict(sorted(admitted_records.items(), key=lambda kv: -kv[1])[:20]),
        harmful_verdicts=harmful,
        failures=failures,
        review_backlog_open=len(open_reviews),
        review_backlog_oldest_days=oldest,
        by_bundle=by_bundle,
    )


@dataclass(frozen=True, slots=True)
class RollbackThresholds:
    """Limits a host applies to a report. Any breach is a reason to disable the newest stage."""

    max_served_admission_rate: float = 0.05
    max_policy_failure_rate: float = 0.02
    max_budget_withheld_rate: float = 0.20
    max_added_latency_gap_turns_ms_p95: float = 8000.0
    max_review_backlog_open: int = 50
    max_review_backlog_oldest_days: float = 7.0
    min_turns: int = 50


def rollback_reasons(report: MetricsReport, thresholds: RollbackThresholds) -> list[str]:
    """Return every threshold the report breaches; empty means no rollback signal."""

    reasons: list[str] = []
    if report.turns < thresholds.min_turns:
        return reasons
    if report.served_turns and report.served_admission_rate > thresholds.max_served_admission_rate:
        reasons.append(
            f"served admission rate {report.served_admission_rate:.3f} > {thresholds.max_served_admission_rate}"
        )
    failure_rate = report.stages["policy_failure"] / report.turns
    if failure_rate > thresholds.max_policy_failure_rate:
        reasons.append(f"policy failure rate {failure_rate:.3f} > {thresholds.max_policy_failure_rate}")
    withheld_rate = report.stages["budget_withheld"] / report.turns
    if withheld_rate > thresholds.max_budget_withheld_rate:
        reasons.append(f"budget withheld rate {withheld_rate:.3f} > {thresholds.max_budget_withheld_rate}")
    if report.added_latency_gap_turns_ms_p95 > thresholds.max_added_latency_gap_turns_ms_p95:
        reasons.append(
            f"gap-turn added latency p95 {report.added_latency_gap_turns_ms_p95:.0f} ms > "
            f"{thresholds.max_added_latency_gap_turns_ms_p95:.0f} ms"
        )
    if report.review_backlog_open > thresholds.max_review_backlog_open:
        reasons.append(f"review backlog {report.review_backlog_open} > {thresholds.max_review_backlog_open}")
    if report.review_backlog_oldest_days > thresholds.max_review_backlog_oldest_days:
        reasons.append(
            f"oldest review {report.review_backlog_oldest_days:.1f} days > {thresholds.max_review_backlog_oldest_days}"
        )
    return reasons


def render(report: MetricsReport) -> str:
    """Human-readable rendering for the CLI."""

    lines = [
        f"turns={report.turns} served={report.served_turns} shadow={report.shadow_turns} "
        f"bundle={report.bundle_hash or 'all'}",
        "stage outcomes: " + ", ".join(f"{name}={report.stages[name]}" for name in STAGE_OUTCOMES),
        f"served admission rate={report.served_admission_rate:.3f}  "
        f"shadow admission rate={report.shadow_admission_rate:.3f}  "
        f"planner fire rate={report.planner_fire_rate:.3f}",
        f"added latency ms p50/p95 all={report.added_latency_ms_p50:.0f}/{report.added_latency_ms_p95:.0f}  "
        f"gap turns={report.added_latency_gap_turns_ms_p50:.0f}/{report.added_latency_gap_turns_ms_p95:.0f}",
        "tokens: "
        + (", ".join(f"{m}: {u['prompt']}+{u['completion']}" for m, u in report.tokens.items()) or "none recorded"),
        f"harmful verdicts seen={report.harmful_verdicts}  failures={report.failures or {}}",
        f"review backlog open={report.review_backlog_open} oldest_days={report.review_backlog_oldest_days:.1f}",
    ]
    if report.admitted_records:
        lines.append(
            "most admitted records: " + ", ".join(f"{k}x{v}" for k, v in list(report.admitted_records.items())[:8])
        )
    if len(report.by_bundle) > 1:
        for key, stages in report.by_bundle.items():
            lines.append(f"  bundle {key}: " + ", ".join(f"{n}={stages[n]}" for n in STAGE_OUTCOMES if stages[n]))
    return "\n".join(lines)


def summarise_many(reports: Iterable[MetricsReport]) -> dict[str, Any]:
    """Convenience for hosts comparing windows or bundles."""

    return {r.bundle_hash or "all": r.to_dict() for r in reports}
