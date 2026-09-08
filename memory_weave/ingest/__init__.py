"""Ingestion: the explicit write path, session extraction, and the helpers both share."""

from memory_weave.hosted import (
    AnthropicCompletionClient,
    CompletionClient,
    OpenAICompletionClient,
    StructuredOutputError,
    completion_client_for,
    parse_json_object,
)

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
from .extraction import (
    EXTRACTION_ACTOR,
    CandidateOutcome,
    ExtractionResult,
    ExtractionRunner,
    ExtractionStatus,
)
from .extractor import (
    EXTRACT_PROMPT_VERSION,
    ExtractionError,
    Extractor,
    FakeExtractor,
    StructuredLLMExtractor,
)
from .ingestor import EntityAmbiguityCandidate, Ingestor, SummaryRequest, WriteRequest, WriteResult
from .reviewer import (
    REVIEW_PROMPT_VERSION,
    CandidateReviewer,
    ReviewDecision,
    ReviewError,
    ReviewRequest,
    StructuredLLMReviewer,
    TableReviewer,
    check_revision,
)
from .session import SessionBuffer, SessionHooks
from .temporal import TEMPORAL_REVIEW_ACTOR, flag_due_reviews, has_temporal_expression, validate_temporal

__all__ = [
    "EXTRACTION_ACTOR",
    "EXTRACT_PROMPT_VERSION",
    "REVIEW_PROMPT_VERSION",
    "TEMPORAL_REVIEW_ACTOR",
    "AnthropicCompletionClient",
    "CandidateOutcome",
    "CandidateReviewer",
    "CompletionClient",
    "EntityAmbiguityCandidate",
    "EntityMergeError",
    "EntityNotFoundError",
    "EntityNotReadableError",
    "EntityNotWritableError",
    "EntityResolutionError",
    "EquivalenceJudge",
    "ExtractionError",
    "ExtractionResult",
    "ExtractionRunner",
    "ExtractionStatus",
    "Extractor",
    "FakeExtractor",
    "FakeJudge",
    "Ingestor",
    "NLICrossEncoderJudge",
    "OpenAICompletionClient",
    "ReviewDecision",
    "ReviewError",
    "ReviewRequest",
    "SessionBuffer",
    "SessionHooks",
    "StructuredLLMExtractor",
    "StructuredLLMReviewer",
    "StructuredOutputError",
    "SummaryRequest",
    "TableReviewer",
    "WriteRequest",
    "WriteResult",
    "aliases_text",
    "check_revision",
    "completion_client_for",
    "flag_due_reviews",
    "follow_merges",
    "has_temporal_expression",
    "merge_entities",
    "parse_json_object",
    "primary_entity_for",
    "resolve_entities",
    "session_turn_source_ref",
    "validate_evidence",
    "validate_temporal",
]
