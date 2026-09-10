"""Public types and configuration for Retold."""

from importlib.metadata import PackageNotFoundError, version

from .config import ConfigError, RetoldConfig, load_config
from .errors import EmbeddingProfileMismatch, LocalModelUnavailable, UnsupportedEvidenceError
from .facade import MemorySearchResult, MemorySession, Retold, RetoldSetupError
from .host import MemoryHost
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
from .profiles import lite_config
from .runtime import MemoryRuntime, build_runtime
from .store import Store

try:
    __version__ = version("retold")
except PackageNotFoundError:  # running from a source checkout that was never installed
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "Candidate",
    "CandidateRecord",
    "ConfigError",
    "EmbeddingProfileMismatch",
    "Entity",
    "EntityMention",
    "EvidenceCheck",
    "Explanation",
    "ExtractionContext",
    "ExtractionOutput",
    "GeneratorHit",
    "LocalModelUnavailable",
    "MemoryHost",
    "MemorySearchResult",
    "MemorySession",
    "MemoryRuntime",
    "RetoldConfig",
    "Retold",
    "RetoldSetupError",
    "Principal",
    "Record",
    "Resolution",
    "RewriteResult",
    "Scope",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SessionSummary",
    "Store",
    "Turn",
    "UnsupportedEvidenceError",
    "build_runtime",
    "load_config",
    "lite_config",
]
