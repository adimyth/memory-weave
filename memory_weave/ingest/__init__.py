"""Ingestion helpers shared by current and future write paths."""

from .entities import (
    EntityMergeError,
    EntityNotFoundError,
    EntityNotReadableError,
    EntityNotWritableError,
    EntityResolutionError,
    aliases_text,
    follow_merges,
    merge_entities,
    primary_entity_for,
    resolve_entities,
)
from .equivalence import EquivalenceJudge, FakeJudge, NLICrossEncoderJudge
from .evidence import session_turn_source_ref, validate_evidence
from .ingestor import EntityAmbiguityCandidate, Ingestor, WriteRequest, WriteResult
from .session import SessionBuffer

__all__ = [
    "EntityMergeError",
    "EntityAmbiguityCandidate",
    "EntityNotFoundError",
    "EntityNotReadableError",
    "EntityNotWritableError",
    "EntityResolutionError",
    "EquivalenceJudge",
    "FakeJudge",
    "Ingestor",
    "NLICrossEncoderJudge",
    "SessionBuffer",
    "WriteRequest",
    "WriteResult",
    "aliases_text",
    "follow_merges",
    "merge_entities",
    "primary_entity_for",
    "resolve_entities",
    "session_turn_source_ref",
    "validate_evidence",
]
