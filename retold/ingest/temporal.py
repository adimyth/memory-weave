"""Temporal metadata validation and the bounded due-review worker.

Temporal fields are accepted only when the evidence quote itself contains a time expression. The worker
flags a record once when its ``review_at`` arrives; it never rewrites content, asserts that a plan happened,
or changes lifecycle status. The passage of time is not evidence.
"""

from __future__ import annotations

import re
from datetime import datetime

from retold.models import CandidateRecord, Turn
from retold.store import Store
from retold.util import now

TEMPORAL_REVIEW_ACTOR = "temporal_reviewer"

_TEMPORAL_EXPRESSION = re.compile(
    r"\b(?:"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"today|tonight|tomorrow|yesterday|"
    r"(?:next|this|last|coming|following) (?:week|month|year|quarter|sprint|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday|weekend)|"
    r"(?:in|for|within|after|over the next) (?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"twelve|few|couple of) (?:day|days|week|weeks|month|months|year|years|hour|hours|quarter|quarters)|"
    r"(?:until|till|through|by|before|after|from|starting|ending|due|deadline) [^.,;]{1,40}?"
    r"(?:\d{1,2}(?:st|nd|rd|th)?|week|month|year|quarter|monday|tuesday|wednesday|thursday|friday|saturday|"
    r"sunday|q[1-4])|"
    r"q[1-4](?: \d{4})?|"
    r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:19|20)\d{2}|"
    r"end of (?:the )?(?:week|month|year|quarter|day|sprint)"
    r")\b",
    re.IGNORECASE,
)


def has_temporal_expression(text: str) -> bool:
    """Whether ``text`` names a date, a weekday, a month, a relative period, or a deadline."""

    return _TEMPORAL_EXPRESSION.search(text) is not None


def validate_temporal(candidate: CandidateRecord, evidence_turn: Turn | None) -> str | None:
    """Return a rejection reason when temporal metadata is unsupported, malformed, or implausible."""

    values = (candidate.valid_from, candidate.valid_until, candidate.review_at)
    if all(value is None for value in values):
        return None
    if any(value is not None and value.tzinfo is None for value in values):
        return "naive_temporal_value"
    if not has_temporal_expression(candidate.evidence):
        return "unsupported_temporal_metadata"
    if (
        candidate.valid_from is not None
        and candidate.valid_until is not None
        and candidate.valid_from > candidate.valid_until
    ):
        return "invalid_temporal_range"
    if candidate.review_at is not None and evidence_turn is not None and candidate.review_at < evidence_turn.at:
        return "review_at_before_evidence"
    return None


def flag_due_reviews(
    store: Store,
    *,
    batch_size: int,
    at: datetime | None = None,
    actor: str = TEMPORAL_REVIEW_ACTOR,
) -> list[str]:
    """Claim and flag one bounded batch of due records; return the ids this call flagged.

    Each row is claimed by a conditional update inside the same transaction as its event, so two workers
    racing on one record produce one flag and one ``record.review_due`` event between them.
    """

    current = at or now()
    flagged: list[str] = []
    for record in store.due_review_records(batch_size, current):
        with store.transaction(immediate=True):
            if not store.flag_review_due(record.id, current):
                continue
            store.append_event(
                "record.review_due",
                actor,
                record.id,
                None,
                {
                    "review_at": record.review_at.isoformat() if record.review_at else None,
                    "valid_from": record.valid_from.isoformat() if record.valid_from else None,
                    "valid_until": record.valid_until.isoformat() if record.valid_until else None,
                    "flagged_at": current.isoformat(),
                    "status": record.status,
                },
            )
            flagged.append(record.id)
    return flagged
