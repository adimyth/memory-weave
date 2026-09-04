"""Embedding implementations that keep model loading outside unit tests."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable
from math import sqrt
from typing import Any, Protocol

import numpy as np

from memory_weave.config import EmbeddingConfig
from memory_weave.util import normalize_vector


class Embedder(Protocol):
    """Convert queries and documents to L2-normalized float32 vectors."""

    name: str
    version: str
    dims: int

    @property
    def is_loaded(self) -> bool:
        """Return whether the embedder can serve requests without loading a model."""

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Return one normalized vector for each query, using the query cache when available."""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return one normalized vector for each document without retaining document vectors in a cache."""


class FakeEmbedder:
    """Produce deterministic unit vectors without loading a model."""

    def __init__(self, *, dims: int, name: str = "fake-embedder", version: str = "1") -> None:
        if dims <= 0:
            raise ValueError("Embedding dimensions must be positive.")
        self.name = name
        self.version = version
        self.dims = dims
        self._similarity_overrides: dict[str, tuple[str, float]] = {}

    @property
    def is_loaded(self) -> bool:
        """Return true because the fake never loads a model."""

        return True

    def set_similarity(self, first: str, second: str, cosine: float) -> None:
        """Make ``second`` have the requested cosine similarity to ``first``."""

        if not -1.0 <= cosine <= 1.0:
            raise ValueError("Cosine similarity must be between -1 and 1.")
        if first == second and cosine != 1.0:
            raise ValueError("A text can only have cosine similarity 1 with itself.")
        if self.dims == 1 and abs(cosine) != 1.0:
            raise ValueError("One-dimensional vectors can only have cosine similarity -1 or 1.")
        self._similarity_overrides[second] = (first, cosine)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Return deterministic L2-normalized vectors for query texts."""

        return self._embed(texts)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return deterministic L2-normalized vectors for document texts."""

        return self._embed(texts)

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Return deterministic L2-normalized vectors for ``texts``."""

        if not texts:
            return np.empty((0, self.dims), dtype=np.float32)
        return np.vstack([self._vector(text) for text in texts]).astype(np.float32, copy=False)

    def _vector(self, text: str) -> np.ndarray:
        override = self._similarity_overrides.get(text)
        if override is None:
            return self._base_vector(text)
        anchor, cosine = override
        anchor_vector = self._base_vector(anchor)
        if abs(cosine) == 1.0:
            return anchor_vector * cosine
        perpendicular = self._perpendicular_vector(anchor, text, anchor_vector)
        return np.ascontiguousarray(cosine * anchor_vector + sqrt(1.0 - cosine**2) * perpendicular)

    def _base_vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        generator = np.random.default_rng(seed)
        return normalize_vector(generator.standard_normal(self.dims).astype(np.float32), self.dims)

    def _perpendicular_vector(self, anchor: str, text: str, anchor_vector: np.ndarray) -> np.ndarray:
        candidate = self._base_vector(f"{anchor}\x00{text}\x00perpendicular")
        candidate -= np.dot(candidate, anchor_vector) * anchor_vector
        if np.linalg.norm(candidate) == 0.0:
            candidate = np.zeros(self.dims, dtype=np.float32)
            candidate[int(np.argmin(np.abs(anchor_vector)))] = 1.0
            candidate -= np.dot(candidate, anchor_vector) * anchor_vector
        return normalize_vector(candidate, self.dims)


class BgeM3Embedder:
    """Embed with BGE-M3, loading the local model only when embeddings are requested."""

    def __init__(self, config: EmbeddingConfig, *, model_factory: Callable[[], Any] | None = None) -> None:
        self.name = config.model
        self.version = config.version
        self.dims = config.dims
        self._max_chars = config.max_chars
        self._query_cache_entries = config.query_cache_entries
        self._model_factory = model_factory or _sentence_transformer_factory(config.model, config.device)
        self._model: Any | None = None
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._model_load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._query_cache_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Return whether the SentenceTransformer model has been instantiated."""

        with self._model_load_lock:
            return self._model is not None

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Return cached or newly computed dense vectors for exact query strings."""

        if not texts:
            return np.empty((0, self.dims), dtype=np.float32)

        vectors_by_text = self._cached_queries(texts)
        if len(vectors_by_text) < len(set(texts)):
            with self._inference_lock:
                vectors_by_text.update(self._cached_queries(texts))
                missing = list(dict.fromkeys(text for text in texts if text not in vectors_by_text))
                if missing:
                    encoded = self._encode(missing)
                    vectors_by_text.update(zip(missing, encoded, strict=True))
                    self._cache_queries(zip(missing, encoded, strict=True))
        return np.vstack([vectors_by_text[text] for text in texts]).astype(np.float32, copy=False)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return newly computed dense vectors for documents without adding them to the query cache."""

        if not texts:
            return np.empty((0, self.dims), dtype=np.float32)

        with self._inference_lock:
            return self._encode(texts)

    def _cached_queries(self, texts: list[str]) -> dict[str, np.ndarray]:
        """Return cached query vectors and refresh their LRU order without holding the inference lock."""

        cached: dict[str, np.ndarray] = {}
        with self._query_cache_lock:
            for text in dict.fromkeys(texts):
                vector = self._query_cache.get(text)
                if vector is not None:
                    self._query_cache.move_to_end(text)
                    cached[text] = vector
        return cached

    def _cache_queries(self, vectors: Iterable[tuple[str, np.ndarray]]) -> None:
        """Store newly encoded query vectors and evict least-recently-used entries."""

        with self._query_cache_lock:
            for text, vector in vectors:
                self._query_cache[text] = vector
                self._query_cache.move_to_end(text)
            while len(self._query_cache) > self._query_cache_entries:
                self._query_cache.popitem(last=False)

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode one batch while the caller holds the inference lock."""

        model = self._load_model()
        encoded = model.encode(
            [text[: self._max_chars] for text in texts],
            convert_to_numpy=True,
            normalize_embeddings=True,
            output_value="sentence_embedding",
            show_progress_bar=False,
        )
        return _normalize_matrix(np.asarray(encoded, dtype=np.float32), self.dims, len(texts))

    def _load_model(self) -> Any:
        with self._model_load_lock:
            if self._model is None:
                self._model = self._model_factory()
            return self._model


def _sentence_transformer_factory(model_name: str, device: str) -> Callable[[], Any]:
    def load() -> Any:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            message = "BGE-M3 support requires sentence-transformers. Install it with: uv sync --extra local-models"
            raise RuntimeError(message) from exc
        if device == "auto":
            return SentenceTransformer(model_name)
        return SentenceTransformer(model_name, device=device)

    return load


def _normalize_matrix(values: np.ndarray, dims: int, expected_rows: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1 and expected_rows == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape != (expected_rows, dims):
        raise ValueError(f"Embedder returned shape {matrix.shape}, expected ({expected_rows}, {dims}).")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding vectors must contain only finite values.")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("Embedding vectors must be non-zero.")
    return np.ascontiguousarray(matrix / norms[:, np.newaxis], dtype=np.float32)
