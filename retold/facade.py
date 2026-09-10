"""Adoption-first API for private, evidence-backed memory sessions."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Literal

from retold.config import RetoldConfig, load_config
from retold.errors import LocalModelUnavailable, UnsupportedEvidenceError
from retold.ingest import WriteRequest, WriteResult
from retold.models import EntityMention, MemoryType, Principal, SearchRequest, SearchResult, TurnRole
from retold.policy import private_scope
from retold.profiles import lite_config
from retold.runtime import MemoryRuntime, build_runtime
from retold.store import Store
from retold.tools import render_search
from retold.util import normalize_attribute, uuid7

EvidenceSourceKind = Literal["user_statement", "tool_result", "agent_inference"]


class RetoldSetupError(RuntimeError):
    """Explain how to install or configure an optional Retold capability."""


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    """A typed search result with stable text rendering for logs and quick starts."""

    items: list[SearchResult]
    empty_reason: str | None
    search_id: str
    text: str


class Retold:
    """Open Retold once, then create private sessions for users."""

    def __init__(self, runtime: MemoryRuntime, *, owns_store: bool = False) -> None:
        self.runtime = runtime
        self.store = runtime.store
        self._owns_store = owns_store
        self._closed = False

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        config: RetoldConfig | str | Path | None = None,
        profile: Literal["standard", "lite"] = "standard",
    ) -> Retold:
        """Open a SQLite store and build a runtime with default or supplied configuration."""

        if profile == "lite":
            if config is not None:
                raise ValueError("profile='lite' cannot be combined with config. Start from lite_config() instead.")
            resolved = lite_config()
        elif profile == "standard":
            resolved = config if isinstance(config, RetoldConfig) else load_config(config)
        else:
            raise ValueError("profile must be 'standard' or 'lite'.")
        store = Store(path)
        return cls(build_runtime(resolved, store), owns_store=True)

    def session(
        self,
        user_id: str,
        *,
        agent_id: str = "assistant",
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> MemorySession:
        """Start a session whose default destination is the agent and user's private scope."""

        if self._closed:
            raise RuntimeError("Retold is closed. Open a new Retold instance before starting a session.")
        principal = Principal(agent_id, user_id, session_id or uuid7(), project_id)
        self.runtime.hooks.on_session_start(principal)
        return MemorySession(self, principal)

    def close(self) -> None:
        """Close the store when this object opened it."""

        if not self._closed and self._owns_store:
            self.store.close()
        self._closed = True

    def __enter__(self) -> Retold:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class MemorySession:
    """Private memory operations bound to a trusted agent, user, and session identity."""

    def __init__(self, retold: Retold, principal: Principal) -> None:
        self._retold = retold
        self._runtime = retold.runtime
        self.principal = principal
        self._finished = False

    def remember(
        self,
        content: str,
        *,
        evidence: str,
        type: MemoryType = "semantic",
        attribute: str | None = None,
        event_at: datetime | None = None,
        tags: list[str] | None = None,
        source_kind: EvidenceSourceKind = "user_statement",
        allow_inference: bool = False,
    ) -> WriteResult:
        """Record trusted evidence and write one memory through the normal ingestion policy.

        A ``user_statement`` or ``tool_result`` claim the evidence does not support is not stored silently as
        an expiring inference: ``remember`` raises :class:`UnsupportedEvidenceError` unless
        ``allow_inference`` is true.
        """

        self._require_active()
        if not content.strip():
            raise ValueError("content must not be empty.")
        if not evidence.strip():
            raise ValueError("evidence must not be empty and must quote the source statement.")

        roles: dict[EvidenceSourceKind, TurnRole] = {
            "user_statement": "user",
            "tool_result": "tool",
            "agent_inference": "assistant",
        }
        role = roles[source_kind]
        self.principal = self._runtime.hooks.on_turn(self.principal, role, evidence)
        resolved_attribute = attribute
        if type in ("semantic", "procedural") and not normalize_attribute(attribute or ""):
            resolved_attribute = _generated_attribute(content)
        request = WriteRequest(
            type=type,
            content=content,
            source_kind=source_kind,
            evidence=evidence,
            attribute=resolved_attribute,
            scope=private_scope(self.principal.agent_id, self.principal.user_id),
            event_at=event_at,
            entities=[EntityMention(kind="person", text=self.principal.user_id, role="about")],
            tags=list(tags or []),
            require_supported_evidence=not allow_inference,
        )
        try:
            result = self._runtime.ingestor.write(self.principal, request)
        except (ImportError, OSError, LocalModelUnavailable) as error:
            raise _local_model_failure(error) from error
        if result.outcome == "unsupported_evidence":
            # Nothing was written: the ingestor refused the claim before persisting anything.
            raise UnsupportedEvidenceError(content, evidence, result.note)
        if result.record_id is None:
            raise ValueError(result.note or f"Retold did not write the memory: {result.outcome}.")
        return result

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        types: list[MemoryType] | None = None,
        entities: list[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> MemorySearchResult:
        """Search this session's readable memory and retain every retrieval explanation."""

        self._require_active()
        if not query.strip():
            raise ValueError("query must not be empty.")
        if k is not None and k <= 0:
            raise ValueError("k must be positive.")
        request = SearchRequest(
            queries=[query],
            context=None,
            types=types,
            entities=entities,
            since=since,
            until=until,
            k=self._runtime.config.retrieval.default_k if k is None else k,
            include_history=False,
        )
        try:
            response = self._runtime.retriever.search(self.principal, request)
        except (ImportError, OSError, LocalModelUnavailable) as error:
            raise _local_model_failure(error) from error
        return MemorySearchResult(
            items=response.results,
            empty_reason=response.empty_reason,
            search_id=response.search_id,
            text=render_search(response),
        )

    def finish(self, *, extract: bool = False) -> threading.Thread | None:
        """End this session and return the background extraction thread when one was started."""

        if self._finished:
            return None
        worker: threading.Thread | None = None
        if extract:
            _require_hosted_extraction(self._runtime.config)
            worker = self._runtime.hooks.on_session_end(self.principal)
        else:
            session_id = self.principal.session_id
            if session_id is not None:
                session = self._runtime.store.get_session(session_id)
                if session is not None and session.ended_at is None:
                    self._runtime.store.end_session(session_id, datetime.now().astimezone())
        self._finished = True
        return worker

    def __enter__(self) -> MemorySession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.finish()

    def _require_active(self) -> None:
        if self._finished:
            raise RuntimeError("This memory session has finished. Start a new session before using memory.")


def _generated_attribute(content: str) -> str:
    digest = hashlib.sha256(content.strip().casefold().encode("utf-8")).hexdigest()[:16]
    return f"facade-memory-{digest}"


def _local_model_failure(error: Exception) -> RetoldSetupError:
    return RetoldSetupError(
        "A local embedding or evidence model could not load. Install the local-models extra with the command in "
        "the Retold README, and allow Hugging Face network access on the first run. "
        f"Model error: {error}"
    )


def _require_hosted_extraction(config: RetoldConfig) -> None:
    models: set[str] = {config.ingestion.extraction_model, config.ingestion.review_model}
    for model in models:
        if model.startswith("claude-"):
            if importlib.util.find_spec("anthropic") is None:
                raise RetoldSetupError(
                    "Background extraction uses Anthropic. Install Retold's live extra and set ANTHROPIC_API_KEY."
                )
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RetoldSetupError("Background extraction uses Anthropic. Set ANTHROPIC_API_KEY first.")
        else:
            if importlib.util.find_spec("openai") is None:
                raise RetoldSetupError(
                    "Background extraction uses OpenAI. Install Retold's live extra and set OPENAI_API_KEY."
                )
            if not os.environ.get("OPENAI_API_KEY"):
                raise RetoldSetupError("Background extraction uses OpenAI. Set OPENAI_API_KEY first.")
