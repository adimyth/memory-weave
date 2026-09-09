"""Public types and configuration for Retold."""

from importlib.metadata import PackageNotFoundError, version

from .config import ConfigError, RetoldConfig, load_config
from .models import (
    Candidate,
    CandidateRecord,
    Entity,
    EntityMention,
    EvidenceCheck,
    Explanation,
    ExtractionContext,
    ExtractionOutput,
    GeneratorHit,
    Principal,
    Record,
    Resolution,
    RewriteResult,
    Scope,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SessionSummary,
    Turn,
)

try:
    __version__ = version("retold")
except PackageNotFoundError:  # running from a source checkout that was never installed
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "Candidate",
    "CandidateRecord",
    "ConfigError",
    "Entity",
    "EntityMention",
    "EvidenceCheck",
    "Explanation",
    "ExtractionContext",
    "ExtractionOutput",
    "GeneratorHit",
    "RetoldConfig",
    "Principal",
    "Record",
    "Resolution",
    "RewriteResult",
    "Scope",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SessionSummary",
    "Turn",
    "load_config",
]
