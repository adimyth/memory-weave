"""Sweep the dense floor over the labelled queries on the 1K fixture with the real embedder."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from memory_weave.config import EmbeddingConfig, MemoryWeaveConfig
from memory_weave.index.embedder import BgeM3Embedder
from memory_weave.index.vector import VectorIndex
from memory_weave.models import Principal
from memory_weave.policy import readable_scopes
from memory_weave.store import Store

from .fixture_1k import NOW, copy_fixture

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("MEMORY_WEAVE_INTEGRATION") != "1",
        reason="set MEMORY_WEAVE_INTEGRATION=1 to run local-model integration tests",
    ),
]

QUERIES = Path(__file__).with_name("labelled_queries.yaml")


def _labelled() -> dict[str, list[dict[str, Any]]]:
    loaded = yaml.safe_load(QUERIES.read_text(encoding="utf-8"))
    assert len(loaded["relevant"]) == 50 and len(loaded["irrelevant"]) == 50
    return loaded  # type: ignore[no-any-return]


def test_configured_dense_floor_sits_in_the_best_f1_band(tmp_path: Path) -> None:
    store = Store(copy_fixture(tmp_path))
    config = MemoryWeaveConfig(embedding=EmbeddingConfig())
    embedder = BgeM3Embedder(config.embedding)
    index = VectorIndex(config.embedding)
    index.load(store)
    labelled = _labelled()

    relevant_scores: list[tuple[str, float, str]] = []  # (query, cosine of the expected record, type)
    for item in labelled["relevant"]:
        agent, user = item["principal"]
        principal = Principal(agent, user, None, None)
        eligible = store.eligible_ids(readable_scopes(store, agent, user), None, None, None, False, NOW)
        vector = embedder.embed_queries([item["query"]])[0]
        hits = dict(index.search(vector, eligible, 50))
        expected = item["expected"][0]
        assert expected in eligible, f"{expected} is not readable by {principal}"
        record = store.get_record(expected)
        assert record is not None
        relevant_scores.append((item["query"], hits.get(expected, 0.0), record.type))
    irrelevant_scores: list[tuple[str, float]] = []
    for item in labelled["irrelevant"]:
        agent, user = item["principal"]
        eligible = store.eligible_ids(readable_scopes(store, agent, user), None, None, None, False, NOW)
        vector = embedder.embed_queries([item["query"]])[0]
        hits = index.search(vector, eligible, 1)
        irrelevant_scores.append((item["query"], hits[0][1] if hits else 0.0))

    floors = [round(0.30 + step * 0.01, 2) for step in range(41)]
    table = []
    for floor in floors:
        tp = sum(1 for _, score, _ in relevant_scores if score >= floor)
        fn = len(relevant_scores) - tp
        fp = sum(1 for _, score in irrelevant_scores if score >= floor)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        table.append((floor, tp, fp, fn, round(precision, 3), round(recall, 3), round(f1, 3)))
    best = max(row[6] for row in table)
    band = [row[0] for row in table if row[6] >= 0.9 * best]
    print("\nfloor  tp  fp  fn  precision recall f1")
    for row in table:
        if row[0] * 100 % 5 == 0:
            print("  ".join(str(value) for value in row))
    print(f"best F1 {best:.3f}; floors within 90% of best: {band[0]} to {band[-1]}")
    print("weakest relevant:", sorted(relevant_scores, key=lambda item: item[1])[:5])
    print("strongest irrelevant:", sorted(irrelevant_scores, key=lambda item: -item[1])[:5])
    configured = config.retrieval.gate.dense_floor.semantic
    assert band[0] <= configured <= band[-1], (
        f"configured floor {configured} outside the best band {band[0]}..{band[-1]}"
    )
    store.close()
