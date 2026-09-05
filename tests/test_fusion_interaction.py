"""Focused tests for Phase 8 ranking stages and their calibration interactions."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from memory_weave.config import EmbeddingConfig, MemoryWeaveConfig
from memory_weave.index.vector import VectorIndex
from memory_weave.models import Candidate, GeneratorHit, LexicalMatch, LexicalTerm, Record, Scope, SearchRequest
from memory_weave.retrieve.budget import fill_budget
from memory_weave.retrieve.dedup import collapse_duplicates
from memory_weave.retrieve.fusion import fuse
from memory_weave.retrieve.gate import FloorGate

_NOW = datetime(2026, 9, 5, tzinfo=UTC)
_SCOPE = Scope(kind="user", id="aditya")


def _record(record_id: str, *, content: str = "A concise memory.", memory_type: str = "semantic") -> Record:
    return Record(
        id=record_id,
        type=memory_type,  # type: ignore[arg-type]
        version=1,
        content=content,
        subject=record_id,
        scope=_SCOPE,
        source_kind="user_statement",
        source_ref=None,
        creator_agent_id="agent",
        evidence=None,
        created_at=_NOW,
        event_at=_NOW,
        expires_at=None,
        confidence=0.95,
        status="confirmed",
        supersedes_id=None,
        reinforcements=0,
        last_reinforced_at=None,
        tags=[],
        entity_ids=[],
    )


def _request() -> SearchRequest:
    return SearchRequest(["concise memory"], None, None, None, None, None, 8, False)


def test_rrf_matches_the_documented_formula_to_four_decimal_places() -> None:
    lexical = LexicalMatch((LexicalTerm("shared", False, False),), 1)
    candidates = fuse(
        [("shared", GeneratorHit(2, 0.7)), ("dense", GeneratorHit(1, 0.8))],
        [("shared", GeneratorHit(1, -3.0), lexical)],
        [("entity", GeneratorHit(1, 0.0), "person-aditya")],
        60,
    )

    scores = {candidate.record_id: candidate.rrf_score for candidate in candidates}
    assert scores["shared"] == pytest.approx(1 / 62 + 1 / 61, abs=0.0001)
    assert scores["dense"] == pytest.approx(1 / 61, abs=0.0001)
    assert scores["entity"] == pytest.approx(1 / 61, abs=0.0001)


def test_relative_floor_compares_candidates_with_the_same_channel_count() -> None:
    config = MemoryWeaveConfig()
    lexical = LexicalMatch((LexicalTerm("two", False, False),), 1)
    candidates = fuse(
        [
            ("two-channel", GeneratorHit(1, 0.9)),
            ("single-one", GeneratorHit(1, 0.9)),
            ("single-two", GeneratorHit(2, 0.9)),
        ],
        [("two-channel", GeneratorHit(1, -2.0), lexical)],
        [],
        config.retrieval.rrf_k,
    )
    records = {candidate.record_id: _record(candidate.record_id) for candidate in candidates}

    decision = FloorGate(config.retrieval.gate).apply(candidates, records, _request())

    assert {candidate.record_id for candidate in decision.kept} == {"two-channel", "single-one", "single-two"}


def test_gate_drops_weak_dense_and_weak_lexical_candidates() -> None:
    config = MemoryWeaveConfig()
    weak_dense = Candidate("dense", GeneratorHit(1, 0.2), None, None, None, None, 0.02, 1, None, 0.02, None, None, None)
    weak_lexical = Candidate(
        "lexical",
        None,
        GeneratorHit(1, -1.0),
        LexicalMatch((LexicalTerm("concise", False, False),), 4),
        None,
        None,
        0.02,
        2,
        None,
        0.02,
        None,
        None,
        None,
    )
    decision = FloorGate(config.retrieval.gate).apply(
        [weak_dense, weak_lexical], {"dense": _record("dense"), "lexical": _record("lexical")}, _request()
    )

    assert decision.kept == []
    assert "dense 0.20" in decision.empty_reason  # type: ignore[operator]


def test_exact_entity_match_passes_the_gate_without_dense_or_lexical_evidence() -> None:
    config = MemoryWeaveConfig()
    entity = Candidate(
        "entity",
        None,
        None,
        None,
        GeneratorHit(1, 0.0),
        "person-aditya",
        0.02,
        1,
        None,
        0.02,
        None,
        None,
        None,
    )

    decision = FloorGate(config.retrieval.gate).apply([entity], {"entity": _record("entity")}, _request())

    assert decision.kept == [entity]
    assert entity.gate_reason == "passed exact entity match"


def test_duplicate_collapse_keeps_the_first_candidate_and_budget_skips_oversized_records() -> None:
    index = VectorIndex(EmbeddingConfig(model="test", version="1", dims=2))
    index.upsert("first", np.array([1.0, 0.0]))
    index.upsert("second", np.array([0.99, 0.1]))
    first = Candidate("first", None, None, None, None, None, 0.03, 1, None, 0.03, "passed", None, None)
    second = Candidate("second", None, None, None, None, None, 0.02, 2, None, 0.02, "passed", None, None)
    kept, dropped = collapse_duplicates([first, second], index, 0.92)
    records = {"first": _record("first", content="x" * 500), "second": _record("second", content="short")}

    chosen, budget_out = fill_budget(kept + [second], records, 2, 40)

    assert kept == [first]
    assert dropped[0]["kept_id"] == "first"
    assert chosen == [second]
    assert budget_out[0]["record_id"] == "first"
