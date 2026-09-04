"""Exact cosine retrieval over a rebuildable in-memory embedding matrix."""

from __future__ import annotations

import numpy as np

from memory_weave.config import EmbeddingConfig
from memory_weave.store import Store

from .embedder import _normalize_vector


class VectorIndex:
    """Keep compatible embeddings in a normalized matrix and search them exactly."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.model = config.model
        self.version = config.version
        self.dims = config.dims
        self.ids: list[str] = []
        self.pos: dict[str, int] = {}
        self._matrix = np.empty((0, self.dims), dtype=np.float32)
        self._live = np.empty(0, dtype=np.bool_)
        self._is_loaded = False

    @property
    def matrix(self) -> np.ndarray:
        """Return the populated rows of the normalized embedding matrix."""

        return self._matrix[: len(self.ids)]

    @property
    def live(self) -> np.ndarray:
        """Return the populated liveness flags aligned with ``ids``."""

        return self._live[: len(self.ids)]

    @property
    def is_loaded(self) -> bool:
        """Return whether the index has completed a load from durable storage."""

        return self._is_loaded

    def load(self, store: Store) -> None:
        """Rebuild the index from embeddings matching this index's model and version."""

        self._reset()
        for record_id, vector in store.iter_embeddings(self.model, self.version):
            if vector.size != self.dims:
                continue
            self.upsert(record_id, vector)
        self._is_loaded = True

    def upsert(self, record_id: str, vector: np.ndarray) -> None:
        """Append or replace one normalized vector and mark its row live."""

        normalized = _normalize_vector(vector, self.dims)
        existing = self.pos.get(record_id)
        if existing is not None:
            self._matrix[existing] = normalized
            self._live[existing] = True
            return
        self._ensure_capacity(len(self.ids) + 1)
        position = len(self.ids)
        self.ids.append(record_id)
        self.pos[record_id] = position
        self._matrix[position] = normalized
        self._live[position] = True

    def remove(self, record_id: str) -> None:
        """Hide a record without compacting the matrix or changing its position."""

        position = self.pos.get(record_id)
        if position is not None:
            self._live[position] = False

    def mask(self, eligible_ids: set[str]) -> np.ndarray:
        """Return an eligibility mask aligned with the populated matrix rows."""

        return np.fromiter((record_id in eligible_ids for record_id in self.ids), dtype=np.bool_, count=len(self.ids))

    def search(self, query_vector: np.ndarray, allowed: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return the best live and allowed record IDs with their exact cosine scores."""

        if k <= 0 or not self.ids:
            return []
        allowed_mask = np.asarray(allowed, dtype=np.bool_)
        if allowed_mask.ndim != 1 or allowed_mask.size != len(self.ids):
            raise ValueError("Allowed mask must have one boolean value for each indexed record.")
        active = self.live & allowed_mask
        count = int(active.sum())
        if count == 0:
            return []
        query = _normalize_vector(query_vector, self.dims)
        scores = self.matrix @ query
        scores[~active] = -np.inf
        result_count = min(k, count)
        top = np.argpartition(-scores, result_count - 1)[:result_count]
        ordered = top[np.argsort(-scores[top], kind="stable")]
        return [(self.ids[position], float(scores[position])) for position in ordered]

    def _reset(self) -> None:
        self.ids.clear()
        self.pos.clear()
        self._matrix = np.empty((0, self.dims), dtype=np.float32)
        self._live = np.empty(0, dtype=np.bool_)
        self._is_loaded = False

    def _ensure_capacity(self, required: int) -> None:
        current = self._matrix.shape[0]
        if current >= required:
            return
        capacity = max(1, current)
        while capacity < required:
            capacity *= 2
        matrix = np.empty((capacity, self.dims), dtype=np.float32)
        live = np.zeros(capacity, dtype=np.bool_)
        if self.ids:
            matrix[: len(self.ids)] = self.matrix
            live[: len(self.ids)] = self.live
        self._matrix = matrix
        self._live = live
