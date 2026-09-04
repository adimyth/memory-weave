"""Typed configuration loading and validation for Memory Weave."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import UnionType
from typing import Any, Literal, cast, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when a Memory Weave configuration is invalid."""


@dataclass(frozen=True, slots=True)
class StoreConfig:
    path: str = "./memory.sqlite"


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    model: str = "BAAI/bge-m3"
    version: str = "1"
    dims: int = 1024
    device: str = "auto"
    max_chars: int = 2000
    query_cache_entries: int = 4096


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    enabled: bool = False
    model: str = "BAAI/bge-reranker-v2-m3"
    candidates: int = 30
    floor: float | None = None
    budget_mean_ms: int = 100


@dataclass(frozen=True, slots=True)
class RewriteConfig:
    enabled: bool = False
    model: str = "claude-haiku-4-5-20251001"
    max_context_chars: int = 2000
    timeout_ms: int = 800


@dataclass(frozen=True, slots=True)
class DenseFloorConfig:
    """Per-type cosine floors. Long episodic summaries score lower against short queries than short facts do."""

    semantic: float = 0.45
    episodic: float = 0.40
    procedural: float = 0.45


@dataclass(frozen=True, slots=True)
class GateConfig:
    dense_floor: DenseFloorConfig = field(default_factory=DenseFloorConfig)
    lexical_min_term_fraction: float = 0.5
    # Lexical-only passes need this many matched terms unless one matched term is an identifier or an entity alias.
    lexical_min_matched_terms: int = 2
    # Survivors below this fraction of the top fused score are dropped. Entity hits are exempt.
    relative_floor: float = 0.5


TriggerMode = Literal["tool_only", "auto", "hybrid"]
TRIGGER_MODES: tuple[TriggerMode, ...] = ("tool_only", "auto", "hybrid")


@dataclass(frozen=True, slots=True)
class TriggerConfig:
    """Who calls memory_search. The retrieval pipeline is identical in every mode; only the caller changes."""

    mode: TriggerMode = "tool_only"
    auto_k: int = 4  # k for host-issued searches; smaller than default_k because nothing asked for them
    auto_min_query_chars: int = 12  # host-issued search is skipped for very short user turns such as "ok" or "yes"


@dataclass(frozen=True, slots=True)
class FreshnessConfig:
    episodic_half_life_days: int = 30
    floor: float = 0.5


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    rewrite: RewriteConfig = field(default_factory=RewriteConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    per_generator_k: int = 30
    rrf_k: int = 60
    default_k: int = 8
    token_budget: int = 1500
    dedup_cosine: float = 0.92
    gate: GateConfig = field(default_factory=GateConfig)
    freshness: FreshnessConfig = field(default_factory=FreshnessConfig)


@dataclass(frozen=True, slots=True)
class EquivalenceConfig:
    model: str = "cross-encoder/nli-deberta-v3-small"
    entail_floor: float = 0.70
    contradict_floor: float = 0.70


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    dedup_candidate_cosine: float = 0.85
    equivalence: EquivalenceConfig = field(default_factory=EquivalenceConfig)
    provisional_ttl_days: int = 30
    reinforcements_to_confirm: int = 2
    extraction_model: str = "claude-haiku-4-5-20251001"
    extraction_max_candidates: int = 20


@dataclass(frozen=True, slots=True)
class SourceRankConfig:
    user_statement: int = 4
    system: int = 3
    tool_result: int = 2
    session_summary: int = 2
    agent_inference: int = 1


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    source_rank: SourceRankConfig = field(default_factory=SourceRankConfig)


@dataclass(frozen=True, slots=True)
class MemoryWeaveConfig:
    store: StoreConfig = field(default_factory=StoreConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def flags(self) -> dict[str, Any]:
        """Return the feature and calibration values persisted with each search."""

        return {
            "embedding_model": self.embedding.model,
            "embedding_version": self.embedding.version,
            "embedding_query_cache_entries": self.embedding.query_cache_entries,
            "rewrite_enabled": self.retrieval.rewrite.enabled,
            "reranker_enabled": self.reranker.enabled,
            "trigger_mode": self.retrieval.trigger.mode,
            "gate": asdict(self.retrieval.gate),
            "reranker_floor": self.reranker.floor,
        }


def load_config(path: str | Path | None = None) -> MemoryWeaveConfig:
    """Load defaults, optionally applying a nested YAML mapping from ``path``."""

    raw: Mapping[str, Any]
    if path is None:
        raw = {}
    else:
        config_path = Path(path)
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigError(f"Unable to read configuration file {config_path}: {exc}") from exc
        if loaded is None:
            raw = {}
        elif isinstance(loaded, Mapping):
            raw = loaded
        else:
            raise ConfigError("Configuration root must be a mapping.")

    _reject_unknown_keys(raw, {"store", "embedding", "reranker", "retrieval", "ingestion", "policy"}, "root")
    config = MemoryWeaveConfig(
        store=_load_dataclass(StoreConfig, raw.get("store"), "store"),
        embedding=_load_dataclass(EmbeddingConfig, raw.get("embedding"), "embedding"),
        reranker=_load_dataclass(RerankerConfig, raw.get("reranker"), "reranker"),
        retrieval=_load_retrieval(raw.get("retrieval")),
        ingestion=_load_ingestion(raw.get("ingestion")),
        policy=_load_policy(raw.get("policy")),
    )
    _validate(config)
    return config


def _load_retrieval(raw: object) -> RetrievalConfig:
    values = _mapping(raw, "retrieval")
    try:
        return RetrievalConfig(
            rewrite=_load_dataclass(RewriteConfig, values.pop("rewrite", None), "retrieval.rewrite"),
            trigger=_load_dataclass(TriggerConfig, values.pop("trigger", None), "retrieval.trigger"),
            gate=_load_gate(values.pop("gate", None)),
            freshness=_load_dataclass(FreshnessConfig, values.pop("freshness", None), "retrieval.freshness"),
            **_coerce_dataclass_values(RetrievalConfig, values, "retrieval"),
        )
    except TypeError as exc:
        raise ConfigError(f"Invalid keys or values in retrieval: {exc}") from exc


def _load_gate(raw: object) -> GateConfig:
    values = _mapping(raw, "retrieval.gate")
    try:
        return GateConfig(
            dense_floor=_load_dataclass(
                DenseFloorConfig, values.pop("dense_floor", None), "retrieval.gate.dense_floor"
            ),
            **_coerce_dataclass_values(GateConfig, values, "retrieval.gate"),
        )
    except TypeError as exc:
        raise ConfigError(f"Invalid keys or values in retrieval.gate: {exc}") from exc


def _load_ingestion(raw: object) -> IngestionConfig:
    values = _mapping(raw, "ingestion")
    try:
        return IngestionConfig(
            equivalence=_load_dataclass(EquivalenceConfig, values.pop("equivalence", None), "ingestion.equivalence"),
            **_coerce_dataclass_values(IngestionConfig, values, "ingestion"),
        )
    except TypeError as exc:
        raise ConfigError(f"Invalid keys or values in ingestion: {exc}") from exc


def _load_policy(raw: object) -> PolicyConfig:
    values = _mapping(raw, "policy")
    try:
        return PolicyConfig(
            source_rank=_load_dataclass(SourceRankConfig, values.pop("source_rank", None), "policy.source_rank"),
            **_coerce_dataclass_values(PolicyConfig, values, "policy"),
        )
    except TypeError as exc:
        raise ConfigError(f"Invalid keys or values in policy: {exc}") from exc


def _load_dataclass[T](cls: type[T], raw: object, section: str) -> T:
    values = _mapping(raw, section)
    coerced_values = _coerce_dataclass_values(cls, values, section)
    try:
        return cls(**coerced_values)
    except TypeError as exc:
        raise ConfigError(f"Invalid keys or values in {section}: {exc}") from exc


def _coerce_dataclass_values[T](cls: type[T], values: Mapping[str, Any], section: str) -> dict[str, Any]:
    dataclass_fields = {config_field.name for config_field in fields(cast(Any, cls))}
    _reject_unknown_keys(values, dataclass_fields, section)
    annotations = get_type_hints(cls)
    return {name: _coerce_value(value, annotations[name], f"{section}.{name}") for name, value in values.items()}


def _mapping(raw: object, section: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Configuration section {section} must be a mapping.")
    return dict(raw)


def _reject_unknown_keys(raw: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ConfigError(f"Unknown configuration key(s) in {section}: {joined}")


def _coerce_value(value: Any, annotation: Any, key: str) -> Any:
    if value is None:
        if type(None) in get_args(annotation):
            return None
        raise ConfigError(f"Configuration value {key} must not be null.")

    if annotation is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"Configuration value {key} must be a string.")

    if annotation is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"Configuration value {key} must be a boolean.")

    if annotation is int:
        if isinstance(value, bool):
            raise ConfigError(f"Configuration value {key} must be an integer.")
        if isinstance(value, float) and not value.is_integer():
            raise ConfigError(f"Configuration value {key} must be an integer.")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Configuration value {key} must be an integer.") from exc

    if annotation is float:
        if isinstance(value, bool):
            raise ConfigError(f"Configuration value {key} must be a number.")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Configuration value {key} must be a number.") from exc

    if get_origin(annotation) in (UnionType,):
        non_none = next(item for item in get_args(annotation) if item is not type(None))
        return _coerce_value(value, non_none, key)

    return value


def _validate(config: MemoryWeaveConfig) -> None:
    if config.reranker.enabled and config.reranker.floor is None:
        raise ConfigError("reranker.floor must be set when reranker.enabled is true.")
    if config.embedding.dims <= 0:
        raise ConfigError("embedding.dims must be positive.")
    if config.embedding.max_chars <= 0:
        raise ConfigError("embedding.max_chars must be positive.")
    if config.embedding.query_cache_entries <= 0:
        raise ConfigError("embedding.query_cache_entries must be positive.")
    if config.reranker.candidates <= 0:
        raise ConfigError("reranker.candidates must be positive.")
    if config.retrieval.per_generator_k <= 0 or config.retrieval.default_k <= 0:
        raise ConfigError("retrieval candidate limits must be positive.")
    if config.retrieval.rrf_k <= 0 or config.retrieval.token_budget <= 0:
        raise ConfigError("retrieval.rrf_k and retrieval.token_budget must be positive.")
    if not 0.0 <= config.retrieval.dedup_cosine <= 1.0:
        raise ConfigError("retrieval.dedup_cosine must be between 0 and 1.")
    if not 0.0 <= config.ingestion.dedup_candidate_cosine <= 1.0:
        raise ConfigError("ingestion.dedup_candidate_cosine must be between 0 and 1.")
    for memory_type in ("semantic", "episodic", "procedural"):
        if not 0.0 <= getattr(config.retrieval.gate.dense_floor, memory_type) <= 1.0:
            raise ConfigError(f"retrieval.gate.dense_floor.{memory_type} must be between 0 and 1.")
    if not 0.0 <= config.retrieval.gate.lexical_min_term_fraction <= 1.0:
        raise ConfigError("retrieval.gate.lexical_min_term_fraction must be between 0 and 1.")
    if config.retrieval.gate.lexical_min_matched_terms < 1:
        raise ConfigError("retrieval.gate.lexical_min_matched_terms must be at least 1.")
    if not 0.0 <= config.retrieval.gate.relative_floor <= 1.0:
        raise ConfigError("retrieval.gate.relative_floor must be between 0 and 1.")
    if config.retrieval.trigger.mode not in TRIGGER_MODES:
        raise ConfigError(f"retrieval.trigger.mode must be one of {', '.join(TRIGGER_MODES)}.")
    if config.retrieval.trigger.auto_k <= 0:
        raise ConfigError("retrieval.trigger.auto_k must be positive.")
    if config.retrieval.trigger.auto_min_query_chars < 0:
        raise ConfigError("retrieval.trigger.auto_min_query_chars must not be negative.")
    if not 0.0 <= config.retrieval.freshness.floor <= 1.0:
        raise ConfigError("retrieval.freshness.floor must be between 0 and 1.")
    if config.retrieval.freshness.episodic_half_life_days <= 0:
        raise ConfigError("retrieval.freshness.episodic_half_life_days must be positive.")
    if config.ingestion.provisional_ttl_days <= 0 or config.ingestion.reinforcements_to_confirm <= 0:
        raise ConfigError("ingestion lifecycle limits must be positive.")
    if config.ingestion.extraction_max_candidates <= 0:
        raise ConfigError("ingestion.extraction_max_candidates must be positive.")
    if not 0.0 <= config.ingestion.equivalence.entail_floor <= 1.0:
        raise ConfigError("ingestion.equivalence.entail_floor must be between 0 and 1.")
    if not 0.0 <= config.ingestion.equivalence.contradict_floor <= 1.0:
        raise ConfigError("ingestion.equivalence.contradict_floor must be between 0 and 1.")
