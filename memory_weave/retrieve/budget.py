"""Bound retrieval output without truncating a memory record's content."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from memory_weave.models import Candidate, Record

_ENVELOPE_TOKENS = 30


def fill_budget(
    candidates: Sequence[Candidate],
    records: Mapping[str, Record],
    k: int,
    token_budget: int,
    *,
    companions: Mapping[str, Candidate] | None = None,
) -> tuple[list[Candidate], list[dict[str, object]]]:
    """Choose complete records that fit within both limits, keeping an authority pair together."""

    if k <= 0 or token_budget <= 0:
        return [], [
            {"record_id": candidate.record_id, "reason": "result or token budget is zero"} for candidate in candidates
        ]
    chosen: list[Candidate] = []
    excluded: list[dict[str, object]] = []
    tokens_used = 0
    companions = companions or {}
    for candidate in candidates:
        companion = companions.get(candidate.record_id)
        group = (companion, candidate) if companion is not None else (candidate,)
        if len([chosen_candidate for chosen_candidate in chosen if chosen_candidate.record_id not in companions]) >= k:
            excluded.extend(
                {"record_id": group_candidate.record_id, "reason": "result count reached"} for group_candidate in group
            )
            continue
        cost = sum(token_cost(records[group_candidate.record_id]) for group_candidate in group)
        if tokens_used + cost > token_budget:
            excluded.extend(
                {
                    "record_id": group_candidate.record_id,
                    "reason": "token budget",
                    "cost": token_cost(records[group_candidate.record_id]),
                    "group_cost": cost,
                }
                for group_candidate in group
            )
            continue
        chosen.extend(group)
        tokens_used += cost
    return chosen, excluded


def token_cost(record: Record) -> int:
    """Return the approximate context cost used by budget filling and explanations."""

    return len(record.content) // 4 + _ENVELOPE_TOKENS
