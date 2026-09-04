from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pytest

from memory_weave.config import EmbeddingConfig
from memory_weave.index.embedder import BgeM3Embedder, FakeEmbedder
from memory_weave.index.vector import VectorIndex
from memory_weave.models import Record, Scope
from memory_weave.store import Store

_NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
_EMBEDDING_CONFIG = EmbeddingConfig(model="fake-embedder", version="1", dims=3, max_chars=200)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    database = Store(tmp_path / "memory.sqlite")
    _ = database.connection
    yield database
    database.close()


def _record(record_id: str) -> Record:
    return Record(
        id=record_id,
        type="semantic",
        version=1,
        content=f"Content for {record_id}.",
        subject=f"project:memory-weave/{record_id}",
        scope=Scope(kind="project", id="memory-weave"),
        source_kind="system",
        source_ref=None,
        creator_agent_id="implementation-agent",
        evidence=None,
        created_at=_NOW,
        event_at=_NOW,
        expires_at=None,
        confidence=0.90,
        status="confirmed",
        supersedes_id=None,
        reinforcements=0,
        last_reinforced_at=None,
        tags=[],
        entity_ids=[],
    )


def test_fake_embedder_is_deterministic_normalized_and_can_set_similarity() -> None:
    embedder = FakeEmbedder(dims=8)
    embedder.set_similarity("concise responses", "short technical answers", 0.75)

    vectors = embedder.embed(["concise responses", "short technical answers"])

    assert vectors.shape == (2, 8)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])
    assert float(np.dot(vectors[0], vectors[1])) == pytest.approx(0.75)
    np.testing.assert_array_equal(embedder.embed(["concise responses"]), vectors[:1])


def test_bge_embedder_loads_lazily_uses_exact_string_cache_and_normalizes() -> None:
    model = _StubSentenceTransformer()
    config = EmbeddingConfig(model="stub", version="test", dims=3, max_chars=2)
    embedder = BgeM3Embedder(config, model_factory=lambda: model)

    assert embedder.is_loaded is False
    vectors = embedder.embed(["alpha", "beta", "alpha"])

    assert embedder.is_loaded is True
    assert model.calls == [["al", "be"]]
    assert vectors.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(vectors[0], vectors[2])
    np.testing.assert_array_equal(embedder.embed(["alpha", "beta"]), vectors[:2])
    assert model.calls == [["al", "be"]]


def test_vector_index_loads_store_vectors_and_applies_masks_updates_and_removals(store: Store) -> None:
    vectors = {
        "record-a": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "record-b": np.array([0.8, 0.6, 0.0], dtype=np.float32),
        "record-c": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    for record_id, vector in vectors.items():
        store.insert_record(_record(record_id))
        store.put_embedding(record_id, _EMBEDDING_CONFIG.model, _EMBEDDING_CONFIG.version, vector)

    index = VectorIndex(_EMBEDDING_CONFIG)
    assert index.is_loaded is False
    index.load(store)

    assert index.is_loaded is True
    assert index.ids == ["record-a", "record-b", "record-c"]
    np.testing.assert_array_equal(index.matrix, np.vstack(list(vectors.values())))
    assert index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), index.mask(set(vectors)), 3) == [
        ("record-a", 1.0),
        ("record-b", pytest.approx(0.8)),
        ("record-c", 0.0),
    ]

    allowed = index.mask({"record-a", "record-c"})
    assert [record_id for record_id, _ in index.search(np.array([1.0, 0.0, 0.0]), allowed, 3)] == [
        "record-a",
        "record-c",
    ]

    prior_position = index.pos["record-b"]
    index.upsert("record-b", np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert index.pos["record-b"] == prior_position
    np.testing.assert_array_equal(index.matrix[prior_position], np.array([0.0, 1.0, 0.0], dtype=np.float32))

    index.remove("record-b")
    assert "record-b" not in [
        record_id
        for record_id, _ in index.search(np.array([0.0, 1.0, 0.0], dtype=np.float32), index.mask(set(vectors)), 3)
    ]


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("MEMORY_WEAVE_RUN_SLOW") != "1",
    reason="set MEMORY_WEAVE_RUN_SLOW=1 to run scale tests",
)
def test_vector_index_searches_fifty_thousand_vectors_under_ten_milliseconds() -> None:
    config = EmbeddingConfig(model="scale", version="1", dims=1024)
    generator = np.random.default_rng(42)
    vectors = generator.standard_normal((50_000, config.dims), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    index = VectorIndex(config)
    for position, vector in enumerate(vectors):
        index.upsert(f"record-{position}", vector)

    allowed = np.ones(len(index.ids), dtype=np.bool_)
    index.search(vectors[0], allowed, 30)
    started_at = perf_counter()
    results = index.search(vectors[0], allowed, 30)
    elapsed = perf_counter() - started_at

    assert len(results) == 30
    assert results[0][0] == "record-0"
    assert elapsed < 0.010


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("MEMORY_WEAVE_INTEGRATION") != "1",
    reason="set MEMORY_WEAVE_INTEGRATION=1 to run local-model integration tests",
)
def test_bge_m3_places_paraphrases_closer_than_unrelated_text() -> None:
    embedder = BgeM3Embedder(EmbeddingConfig())

    vectors = embedder.embed(
        [
            "Aditya prefers concise technical explanations.",
            "Aditya likes short technical answers.",
            "The monsoon arrived in Bangalore this afternoon.",
        ]
    )

    assert float(np.dot(vectors[0], vectors[1])) > 0.8
    assert float(np.dot(vectors[0], vectors[2])) < 0.5


class _StubSentenceTransformer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self.calls.append(texts)
        assert kwargs == {
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "output_value": "sentence_embedding",
            "show_progress_bar": False,
        }
        return np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32)
