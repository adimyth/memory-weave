"""Exact cosine retrieval over a rebuildable in-memory embedding matrix."""

from __future__ import annotations

import threading
from typing import cast

import numpy as np

from memory_weave.config import EmbeddingConfig
from memory_weave.store import Store
from memory_weave.util import normalize_vector


class VectorIndex:
    """Keep compatible embeddings in a normalized matrix and search them exactly."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.model = config.model
        self.version = config.version
        self.dims = config.dims
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._matrix = np.empty((0, self.dims), dtype=np.float32)
        self._live = np.empty(0, dtype=np.bool_)
        self._is_loaded = False
        self._lock = threading.RLock()

    @property
    def ids(self) -> list[str]:
        """Return a snapshot of record IDs aligned with the matrix rows."""

        with self._lock:
            return self._ids.copy()

    @property
    def pos(self) -> dict[str, int]:
        """Return a snapshot mapping record IDs to matrix positions."""

        with self._lock:
            return self._pos.copy()

    @property
    def live(self) -> np.ndarray:
        """Return the populated liveness flags aligned with ``ids``."""

        with self._lock:
            return self._live[: len(self._ids)].copy()

    @property
    def is_loaded(self) -> bool:
        """Return whether the index has completed a load from durable storage."""

        with self._lock:
            return self._is_loaded

    def load(self, store: Store) -> None:
        """Rebuild the index from embeddings matching this index's model and version."""

        with self._lock:
            self._reset()
            self._allocate(store.count_embeddings(self.model, self.version))
            try:
                for record_id, vector in store.iter_embeddings(self.model, self.version):
                    if vector.size != self.dims:
                        raise ValueError(
                            f"Embedding {record_id} has {vector.size} dimensions, expected {self.dims} for this index."
                        )
                    self._append(record_id, vector)
                self._validate_loaded_vectors()
            except BaseException:
                self._reset()
                raise
            self._is_loaded = True

    def upsert(self, record_id: str, vector: np.ndarray) -> None:
        """Append or replace one normalized vector and mark its row live."""

        normalized = normalize_vector(vector, self.dims)
        with self._lock:
            existing = self._pos.get(record_id)
            if existing is not None:
                self._matrix[existing] = normalized
                self._live[existing] = True
                return
            self._append(record_id, normalized)

    def remove(self, record_id: str) -> None:
        """Hide a record without compacting the matrix or changing its position."""

        with self._lock:
            position = self._pos.get(record_id)
            if position is not None:
                self._live[position] = False

    def vector_for(self, record_id: str) -> np.ndarray | None:
        """Return a defensive copy of one normalized vector, or ``None`` when the record is absent."""

        with self._lock:
            position = self._pos.get(record_id)
            if position is None:
                return None
            return cast(np.ndarray, self._matrix[position].copy())

    def cosine(self, first_id: str, second_id: str) -> float:
        """Return the exact cosine between two indexed records without exposing the full matrix."""

        with self._lock:
            first_position = self._pos.get(first_id)
            second_position = self._pos.get(second_id)
            if first_position is None or second_position is None:
                missing_id = first_id if first_position is None else second_id
                raise KeyError(f"Record {missing_id} is not indexed.")
            return float(np.dot(self._matrix[first_position], self._matrix[second_position]))

    def mask(self, eligible_ids: set[str]) -> np.ndarray:
        """Return an eligibility mask aligned with the populated matrix rows."""

        with self._lock:
            return np.fromiter(
                (record_id in eligible_ids for record_id in self._ids), dtype=np.bool_, count=len(self._ids)
            )

    def search(self, query_vector: np.ndarray, allowed: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return the best live and allowed record IDs with their exact cosine scores."""

        with self._lock:
            if k <= 0 or not self._ids:
                return []
            allowed_mask = np.asarray(allowed, dtype=np.bool_)
            if allowed_mask.ndim != 1 or allowed_mask.size > len(self._ids):
                raise ValueError("Allowed mask must have at most one boolean value for each indexed record.")
            if allowed_mask.size < len(self._ids):
                extended_mask = np.zeros(len(self._ids), dtype=np.bool_)
                extended_mask[: allowed_mask.size] = allowed_mask
                allowed_mask = extended_mask
            active = self._live[: len(self._ids)] & allowed_mask
            count = int(active.sum())
            if count == 0:
                return []
            query = normalize_vector(query_vector, self.dims)
            scores = self._matrix[: len(self._ids)] @ query
            scores[~active] = -np.inf
            result_count = min(k, count)
            top = np.argpartition(-scores, result_count - 1)[:result_count]
            ordered = top[np.argsort(-scores[top], kind="stable")]
            return [(self._ids[position], float(scores[position])) for position in ordered]

    def _reset(self) -> None:
        self._ids.clear()
        self._pos.clear()
        self._matrix = np.empty((0, self.dims), dtype=np.float32)
        self._live = np.empty(0, dtype=np.bool_)
        self._is_loaded = False

    def _append(self, record_id: str, normalized: np.ndarray) -> None:
        """Append one already-normalized vector while holding the index lock."""

        self._ensure_capacity(len(self._ids) + 1)
        position = len(self._ids)
        self._ids.append(record_id)
        self._pos[record_id] = position
        self._matrix[position] = normalized
        self._live[position] = True

    def _allocate(self, capacity: int) -> None:
        """Allocate initial storage for a store rebuild while holding the index lock."""

        if capacity <= 0:
            return
        self._matrix = np.empty((capacity, self.dims), dtype=np.float32)
        self._live = np.zeros(capacity, dtype=np.bool_)

    def _validate_loaded_vectors(self) -> None:
        """Verify that durable vectors obey the normalized index invariant."""

        if not self._ids:
            return
        vectors = self._matrix[: len(self._ids)]
        if not np.isfinite(vectors).all():
            raise ValueError("Stored embeddings must contain only finite values.")
        norms = np.linalg.norm(vectors, axis=1)
        invalid = np.flatnonzero(~np.isclose(norms, 1.0, rtol=1e-5, atol=1e-6))
        if invalid.size:
            position = int(invalid[0])
            raise ValueError(f"Embedding {self._ids[position]} is not L2-normalized.")

    def _ensure_capacity(self, required: int) -> None:
        current = self._matrix.shape[0]
        if current >= required:
            return
        capacity = max(1, current)
        while capacity < required:
            capacity *= 2
        matrix = np.empty((capacity, self.dims), dtype=np.float32)
        live = np.zeros(capacity, dtype=np.bool_)
        if self._ids:
            matrix[: len(self._ids)] = self._matrix[: len(self._ids)]
            live[: len(self._ids)] = self._live[: len(self._ids)]
        self._matrix = matrix
        self._live = live
