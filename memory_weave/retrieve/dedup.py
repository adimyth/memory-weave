"""Cosine-based duplicate collapse after relevance gating."""

from __future__ import annotations

from collections.abc import Sequence

from memory_weave.index.vector import VectorIndex
from memory_weave.models import Candidate


def collapse_duplicates(
    candidates: Sequence[Candidate], vector_index: VectorIndex, cosine_floor: float
) -> tuple[list[Candidate], list[dict[str, object]]]:
    """Keep the first fused survivor from every near-duplicate group and record each removal."""

    kept: list[Candidate] = []
    dropped: list[dict[str, object]] = []
    for candidate in sorted(candidates, key=lambda candidate: (candidate.fused_rank, candidate.record_id)):
        duplicate: tuple[Candidate, float] | None = None
        for accepted in kept:
            try:
                cosine = vector_index.cosine(candidate.record_id, accepted.record_id)
            except KeyError:
                continue
            if cosine >= cosine_floor:
                duplicate = (accepted, cosine)
                break
        if duplicate is None:
            kept.append(candidate)
            continue
        accepted, cosine = duplicate
        dropped.append({"dropped_id": candidate.record_id, "kept_id": accepted.record_id, "cosine": cosine})
    return sorted(kept, key=lambda candidate: (-candidate.score, candidate.fused_rank, candidate.record_id)), dropped
