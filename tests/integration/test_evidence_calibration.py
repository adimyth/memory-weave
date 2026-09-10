"""Calibration of the evidence check against the real NLI judge on a labelled set of quote and claim pairs.

Every pair is scored the way the ingestor scores it: the claim as written, and, when the claim names the
principal, the claim with that name rewritten as "the user"; the higher score is compared with the floor.
Pairs labelled ``supported`` must clear the floor and pairs labelled ``unsupported`` must not. Pairs labelled
``borderline`` are reported and not asserted: a downgrade there is the floor being conservative, not a defect.
Pairs labelled ``known-limit`` are misjudged by the model itself and are reported so a model change is measured
against them; the rewrite must never be what turns an unsupported pair into a supported one.

Run with the local models:

    HF_HUB_OFFLINE=1 RETOLD_INTEGRATION=1 uv run --extra local-models \
        pytest tests/integration/test_evidence_calibration.py -s
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from retold.config import EquivalenceConfig, EvidenceConfig
from retold.ingest import NLICrossEncoderJudge
from retold.ingest.evidence import neutralise_principal, quote_is_first_person

PRINCIPAL_NAMES = ["Aditya Mishra", "aditya", "Adi"]
FLOOR = EvidenceConfig().entail_floor


@dataclass(frozen=True)
class Pair:
    group: str
    quote: str
    claim: str
    label: str  # supported | unsupported | borderline | known-limit


PAIRS: list[Pair] = [
    # A. the same words
    Pair("same words", "I prefer concise answers.", "I prefer concise answers.", "supported"),
    Pair("same words", "My working time zone is Asia/Kolkata.", "My working time zone is Asia/Kolkata.", "supported"),
    Pair(
        "same words",
        "We decided to use Postgres logical replication.",
        "We decided to use Postgres logical replication.",
        "supported",
    ),
    # B. first-person paraphrase
    Pair("paraphrase", "I prefer concise answers.", "I like short answers.", "supported"),
    Pair("paraphrase", "I prefer concise answers.", "I want brief replies.", "supported"),
    Pair("paraphrase", "My working time zone is Asia/Kolkata.", "I work in the Asia/Kolkata time zone.", "supported"),
    Pair(
        "paraphrase",
        "Please use Python for code examples unless I ask otherwise.",
        "I want code examples in Python by default.",
        "known-limit",  # scores 0.22: the model reads the instruction and the preference as different claims
    ),
    # C. the principal named in the third person; the neutralised claim must carry these
    Pair("named principal", "I prefer concise answers.", "Aditya prefers concise answers.", "supported"),
    Pair("named principal", "I prefer concise answers.", "Aditya Mishra prefers concise answers.", "supported"),
    Pair("named principal", "I prefer concise answers.", "Adi likes short answers.", "supported"),
    Pair(
        "named principal",
        "My working time zone is Asia/Kolkata.",
        "Aditya's working time zone is Asia/Kolkata.",
        "supported",
    ),
    Pair("named principal", "I use Vim as my editor.", "aditya uses Vim as their editor.", "supported"),
    Pair("named principal", "I am allergic to peanuts.", "Aditya is allergic to peanuts.", "supported"),
    # D. "the user" form
    Pair("the user", "I prefer concise answers.", "The user prefers concise answers.", "supported"),
    Pair("the user", "I use Vim as my editor.", "The user's editor is Vim.", "supported"),
    # E. third parties named in the quote
    Pair(
        "third party",
        "My colleague Priya Nair owns the deployment checklist.",
        "Priya Nair owns the deployment checklist.",
        "supported",
    ),
    Pair("third party", "My manager is Rohan and he approves releases.", "Rohan approves releases.", "supported"),
    Pair(
        "third party",
        "Priya asked us to keep the checklist in the wiki.",
        "The deployment checklist lives in the wiki.",
        "borderline",
    ),
    Pair("third party", "Rohan left Nimbus in April and joined Lattice.", "Rohan works at Lattice.", "supported"),
    # F. tool output supporting a claim
    Pair("tool result", "Deploy finished at 14:02 with exit code 0.", "The deploy at 14:02 succeeded.", "supported"),
    Pair("tool result", "pytest: 391 passed, 12 skipped in 8.4s", "The test suite passed with 391 tests.", "supported"),
    Pair(
        "tool result",
        "Error: connection refused on port 5432",
        "The database on port 5432 refused the connection.",
        "known-limit",  # scores 0.00: terse tool output is far from the model's training distribution
    ),
    # G. claims that need a step of reasoning
    Pair("reasoning", "I moved to Berlin last month.", "Aditya lives in Berlin.", "supported"),
    Pair(
        "reasoning", "I moved to Berlin last month.", "Aditya used to live somewhere other than Berlin.", "borderline"
    ),
    Pair("reasoning", "My daughter starts school in September.", "Aditya has a daughter.", "supported"),
    # H. unrelated
    Pair("unrelated", "I prefer concise answers.", "Aditya is allergic to peanuts.", "unsupported"),
    Pair("unrelated", "I like tea.", "The user prefers concise answers.", "unsupported"),
    Pair("unrelated", "Deploy finished at 14:02 with exit code 0.", "The user works in Asia/Kolkata.", "unsupported"),
    Pair(
        "unrelated",
        "My colleague Priya Nair owns the deployment checklist.",
        "Aditya owns the deployment checklist.",
        "unsupported",
    ),
    # I. contradiction
    Pair("contradiction", "I prefer concise answers.", "Aditya prefers detailed answers.", "unsupported"),
    Pair("contradiction", "I use Vim as my editor.", "The user's editor is Emacs.", "unsupported"),
    Pair("contradiction", "Rohan left Nimbus in April and joined Lattice.", "Rohan works at Nimbus.", "unsupported"),
    Pair(
        "contradiction",
        "Deploy finished at 14:02 with exit code 1.",
        "The deploy at 14:02 succeeded.",
        "known-limit",  # scores 0.99: the model does not know what an exit code means
    ),
    # J. over-claims: the quote hedges, the claim does not
    Pair(
        "over-claim",
        "I might switch to Emacs at some point.",
        "Aditya uses Emacs.",
        "known-limit",  # scores 0.83: the model reads "might" as support
    ),
    Pair(
        "over-claim",
        "We are considering Postgres logical replication.",
        "We decided to use Postgres logical replication.",
        "unsupported",
    ),
    Pair(
        "over-claim",
        "I sometimes like a longer answer for design questions.",
        "Aditya always prefers detailed answers.",
        "unsupported",
    ),
    # K. wrong person
    Pair("wrong person", "Priya prefers tea.", "Aditya prefers tea.", "unsupported"),
    Pair("wrong person", "My manager Rohan works in Berlin.", "Aditya works in Berlin.", "unsupported"),
    # L. intent paraphrases the floor may reject
    Pair("intent", "Please keep replies short and skip the preamble.", "Aditya prefers concise answers.", "borderline"),
    Pair(
        "intent",
        "Stop explaining the basics every time.",
        "The user prefers answers without introductory explanation.",
        "borderline",
    ),
]


def _score(judge: NLICrossEncoderJudge, pair: Pair) -> tuple[float, str, bool]:
    """Score a pair the way the ingestor does; the flag says whether the rewrite is what cleared the floor."""

    score = judge.entails(pair.quote, pair.claim)
    used = pair.claim
    flipped = False
    neutral = neutralise_principal(pair.claim, PRINCIPAL_NAMES) if quote_is_first_person(pair.quote) else None
    if neutral is not None:
        alternative = judge.entails(pair.quote, neutral)
        if alternative > score:
            flipped = score < FLOOR <= alternative
            score, used = alternative, neutral
    return score, used, flipped


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RETOLD_INTEGRATION") != "1",
    reason="set RETOLD_INTEGRATION=1 to run local-model integration tests",
)
def test_evidence_check_calibration_on_the_labelled_pairs() -> None:
    judge = NLICrossEncoderJudge(EquivalenceConfig())
    rows: list[tuple[Pair, float, str, bool]] = []
    flipped_unsupported: list[Pair] = []
    for pair in PAIRS:
        score, used, flipped = _score(judge, pair)
        rows.append((pair, score, used, score >= FLOOR))
        if flipped and pair.label == "unsupported":
            flipped_unsupported.append(pair)

    print(f"\nfloor {FLOOR:.2f}")
    for pair, score, used, passed in rows:
        verdict = "supported" if passed else "downgraded"
        asserted = pair.label in ("supported", "unsupported")
        flag = "" if not asserted or (pair.label == "supported") == passed else "  <-- WRONG"
        print(f"{pair.group:15} {score:5.2f} {verdict:10} [{pair.label:11}] {pair.quote!r} -> {used!r}{flag}")

    wrong = [
        (pair, score)
        for pair, score, _, passed in rows
        if pair.label in ("supported", "unsupported") and (pair.label == "supported") != passed
    ]
    labelled = [row for row in rows if row[0].label in ("supported", "unsupported")]
    limits = [row for row in rows if row[0].label == "known-limit"]
    borderline = [row for row in rows if row[0].label == "borderline"]
    print(
        f"\n{len(labelled) - len(wrong)} of {len(labelled)} asserted pairs correct; "
        f"{len(limits)} known model limits and {len(borderline)} borderline pairs reported only"
    )
    assert not flipped_unsupported, "the rewrite must never make an unsupported claim pass: " + "; ".join(
        p.claim for p in flipped_unsupported
    )
    assert not wrong, "misjudged pairs: " + "; ".join(f"{p.group}: {p.claim!r} scored {s:.2f}" for p, s in wrong)
