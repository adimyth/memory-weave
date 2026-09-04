from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock
from time import perf_counter, sleep
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

    vectors = embedder.embed_documents(["concise responses", "short technical answers"])

    assert embedder.is_loaded is True
    assert vectors.shape == (2, 8)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])
    assert float(np.dot(vectors[0], vectors[1])) == pytest.approx(0.75)
    np.testing.assert_array_equal(embedder.embed_queries(["concise responses"]), vectors[:1])


def test_bge_embedder_loads_lazily_caches_queries_only_and_normalizes() -> None:
    model = _StubSentenceTransformer()
    config = EmbeddingConfig(model="stub", version="test", dims=3, max_chars=2)
    embedder = BgeM3Embedder(config, model_factory=lambda: model)

    assert embedder.is_loaded is False
    vectors = embedder.embed_queries(["alpha", "beta", "alpha"])

    assert embedder.is_loaded is True
    assert model.calls == [["al", "be"]]
    assert vectors.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(vectors[0], vectors[2])
    np.testing.assert_array_equal(embedder.embed_documents(["alpha", "beta"]), vectors[:2])
    np.testing.assert_array_equal(embedder.embed_queries(["alpha", "beta"]), vectors[:2])
    assert model.calls == [["al", "be"], ["al", "be"]]


def test_bge_embedder_evicts_least_recently_used_query_vectors() -> None:
    model = _StubSentenceTransformer()
    embedder = BgeM3Embedder(
        EmbeddingConfig(model="stub", version="test", dims=3, query_cache_entries=2),
        model_factory=lambda: model,
    )

    embedder.embed_queries(["alpha", "beta"])
    embedder.embed_queries(["alpha"])
    embedder.embed_queries(["gamma"])
    embedder.embed_queries(["beta"])

    assert model.calls == [["alpha", "beta"], ["gamma"], ["beta"]]


def test_bge_embedder_serializes_query_and_document_inference() -> None:
    model = _BlockingSentenceTransformer()
    embedder = BgeM3Embedder(EmbeddingConfig(model="stub", version="test", dims=3), model_factory=lambda: model)
    second_started = Event()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(embedder.embed_queries, ["query"])
        assert model.entered.wait(timeout=1)
        second = executor.submit(_embed_document, embedder, second_started)
        assert second_started.wait(timeout=1)
        try:
            assert model.overlap.wait(timeout=0.05) is False
        finally:
            model.release.set()
        first.result(timeout=1)
        second.result(timeout=1)

    assert model.max_active == 1


def test_bge_embedder_rechecks_the_cache_after_waiting_for_inference() -> None:
    model = _BlockingSentenceTransformer()
    embedder = _ObservedCacheEmbedder(
        EmbeddingConfig(model="stub", version="test", dims=3),
        model_factory=lambda: model,
    )
    second_started = Event()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(embedder.embed_queries, ["same query"])
        assert model.entered.wait(timeout=1)
        second = executor.submit(_embed_query, embedder, second_started)
        assert second_started.wait(timeout=1)
        assert embedder.second_cache_miss.wait(timeout=1)
        model.release.set()
        first.result(timeout=1)
        second.result(timeout=1)

    assert model.calls == [["same query"]]


def test_bge_embedder_loads_one_model_when_two_threads_arrive_together() -> None:
    model = _StubSentenceTransformer()
    config = EmbeddingConfig(model="stub", version="test", dims=3)
    factory_calls = 0
    factory_lock = Lock()

    def factory() -> _StubSentenceTransformer:
        nonlocal factory_calls
        with factory_lock:
            factory_calls += 1
        sleep(0.02)
        return model

    embedder = BgeM3Embedder(config, model_factory=factory)
    with ThreadPoolExecutor(max_workers=2) as executor:
        vectors = list(executor.map(lambda text: embedder.embed_documents([text]), ["first", "second"]))

    assert factory_calls == 1
    assert embedder.is_loaded is True
    assert all(vector.shape == (1, 3) for vector in vectors)


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
    assert not hasattr(index, "matrix")
    np.testing.assert_array_equal(index.vector_for("record-a"), vectors["record-a"])
    assert index.cosine("record-a", "record-b") == pytest.approx(0.8)
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
    np.testing.assert_array_equal(index.vector_for("record-b"), np.array([0.0, 1.0, 0.0], dtype=np.float32))

    index.remove("record-b")
    assert "record-b" not in [
        record_id
        for record_id, _ in index.search(np.array([0.0, 1.0, 0.0], dtype=np.float32), index.mask(set(vectors)), 3)
    ]


def test_vector_index_load_rejects_an_embedding_with_unexpected_dimensions(store: Store) -> None:
    record = _record("wrong-dimensions")
    store.insert_record(record)
    store.put_embedding(record.id, _EMBEDDING_CONFIG.model, _EMBEDDING_CONFIG.version, np.array([1.0, 0.0]))
    index = VectorIndex(_EMBEDDING_CONFIG)

    with pytest.raises(ValueError, match=r"wrong-dimensions has 2 dimensions, expected 3"):
        index.load(store)

    assert index.ids == []
    assert index.is_loaded is False


def test_vector_index_excludes_rows_added_after_the_allowed_mask_was_built() -> None:
    index = VectorIndex(_EMBEDDING_CONFIG)
    index.upsert("existing", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    allowed = index.mask({"existing"})
    index.upsert("new", np.array([1.0, 0.0, 0.0], dtype=np.float32))

    assert index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), allowed, 2) == [("existing", 1.0)]


def test_vector_index_returns_single_vector_copies_and_reports_unknown_records() -> None:
    index = VectorIndex(_EMBEDDING_CONFIG)
    index.upsert("first", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    index.upsert("second", np.array([0.0, 1.0, 0.0], dtype=np.float32))

    vector = index.vector_for("first")
    assert vector is not None
    vector[0] = 0.0

    np.testing.assert_array_equal(index.vector_for("first"), np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert index.cosine("first", "second") == 0.0
    assert index.vector_for("missing") is None
    with pytest.raises(KeyError, match="missing"):
        index.cosine("first", "missing")


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

    vectors = embedder.embed_documents(
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
        vectors = {
            "al": [2.0, 0.0, 0.0],
            "be": [0.0, 3.0, 4.0],
        }
        return np.array([vectors.get(text, [1.0, 0.0, 0.0]) for text in texts], dtype=np.float32)


def _embed_document(embedder: BgeM3Embedder, started: Event) -> np.ndarray:
    started.set()
    return embedder.embed_documents(["document"])


def _embed_query(embedder: BgeM3Embedder, started: Event) -> np.ndarray:
    started.set()
    return embedder.embed_queries(["same query"])


class _BlockingSentenceTransformer(_StubSentenceTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.entered = Event()
        self.overlap = Event()
        self.release = Event()
        self._active = 0
        self._active_lock = Lock()
        self.max_active = 0

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        with self._active_lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            if self._active > 1:
                self.overlap.set()
            self.entered.set()
        try:
            assert self.release.wait(timeout=1)
            return super().encode(texts, **kwargs)
        finally:
            with self._active_lock:
                self._active -= 1


class _ObservedCacheEmbedder(BgeM3Embedder):
    def __init__(self, config: EmbeddingConfig, *, model_factory: Any) -> None:
        super().__init__(config, model_factory=model_factory)
        self._cache_checks = 0
        self._cache_checks_lock = Lock()
        self.second_cache_miss = Event()

    def _cached_queries(self, texts: list[str]) -> dict[str, np.ndarray]:
        cached = super()._cached_queries(texts)
        with self._cache_checks_lock:
            self._cache_checks += 1
            if self._cache_checks == 3:
                self.second_cache_miss.set()
        return cached
