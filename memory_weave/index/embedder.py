"""Embedding implementations that keep model loading outside unit tests."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from math import sqrt
from typing import Any, Protocol

import numpy as np

from memory_weave.config import EmbeddingConfig


class Embedder(Protocol):
    """Convert texts to L2-normalized float32 vectors."""

    name: str
    version: str
    dims: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one normalized vector for each input text."""


class FakeEmbedder:
    """Produce deterministic unit vectors without loading a model."""

    def __init__(self, *, dims: int, name: str = "fake-embedder", version: str = "1") -> None:
        if dims <= 0:
            raise ValueError("Embedding dimensions must be positive.")
        self.name = name
        self.version = version
        self.dims = dims
        self._similarity_overrides: dict[str, tuple[str, float]] = {}

    def set_similarity(self, first: str, second: str, cosine: float) -> None:
        """Make ``second`` have the requested cosine similarity to ``first``."""

        if not -1.0 <= cosine <= 1.0:
            raise ValueError("Cosine similarity must be between -1 and 1.")
        if first == second and cosine != 1.0:
            raise ValueError("A text can only have cosine similarity 1 with itself.")
        if self.dims == 1 and abs(cosine) != 1.0:
            raise ValueError("One-dimensional vectors can only have cosine similarity -1 or 1.")
        self._similarity_overrides[second] = (first, cosine)

    def embed(self, texts: list[str]) -> np.ndarray:
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
        return _normalize_vector(generator.standard_normal(self.dims).astype(np.float32), self.dims)

    def _perpendicular_vector(self, anchor: str, text: str, anchor_vector: np.ndarray) -> np.ndarray:
        candidate = self._base_vector(f"{anchor}\x00{text}\x00perpendicular")
        candidate -= np.dot(candidate, anchor_vector) * anchor_vector
        if np.linalg.norm(candidate) == 0.0:
            candidate = np.zeros(self.dims, dtype=np.float32)
            candidate[int(np.argmin(np.abs(anchor_vector)))] = 1.0
            candidate -= np.dot(candidate, anchor_vector) * anchor_vector
        return _normalize_vector(candidate, self.dims)


class BgeM3Embedder:
    """Embed with BGE-M3, loading the local model only when embeddings are requested."""

    def __init__(self, config: EmbeddingConfig, *, model_factory: Callable[[], Any] | None = None) -> None:
        self.name = config.model
        self.version = config.version
        self.dims = config.dims
        self._max_chars = config.max_chars
        self._model_factory = model_factory or _sentence_transformer_factory(config.model, config.device)
        self._model: Any | None = None
        self._cache: dict[str, np.ndarray] = {}

    @property
    def is_loaded(self) -> bool:
        """Return whether the SentenceTransformer model has been instantiated."""

        return self._model is not None

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return cached or newly computed dense vectors for exact input strings."""

        if not texts:
            return np.empty((0, self.dims), dtype=np.float32)
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        if missing:
            model = self._load_model()
            encoded = model.encode(
                [text[: self._max_chars] for text in missing],
                convert_to_numpy=True,
                normalize_embeddings=True,
                output_value="sentence_embedding",
                show_progress_bar=False,
            )
            vectors = _normalize_matrix(np.asarray(encoded, dtype=np.float32), self.dims, len(missing))
            self._cache.update(zip(missing, vectors, strict=True))
        return np.vstack([self._cache[text] for text in texts]).astype(np.float32, copy=False)

    def _load_model(self) -> Any:
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


def _normalize_vector(value: np.ndarray, dims: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size != dims:
        raise ValueError(f"Embedding vector has {vector.size} values, expected {dims}.")
    if not np.isfinite(vector).all():
        raise ValueError("Embedding vectors must contain only finite values.")
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("Embedding vectors must be non-zero.")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)
