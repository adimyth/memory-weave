"""Optional query rewriting: a no-op default and a hosted implementation behind the same protocol.

The rewriter sees raw queries and the current-turn context, never candidate records, the store, or earlier
results. A hosted rewrite may only resolve references and name the subject. A rewrite that introduces a
capitalised name absent from the queries and the context is refused and the raw queries are used, so the
stage cannot invent an entity or a private fact on the way into retrieval.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from importlib.resources import files
from typing import Protocol

from retold.config import RetoldConfig, RewriteConfig
from retold.hosted import CompletionClient, StructuredOutputError, completion_client_for, parse_json_object
from retold.models import RewriteResult, SearchRequest

REWRITE_PROMPT_VERSION = "rewrite-v1"

_NAME_TOKEN = re.compile(r"[A-Z][A-Za-z0-9'\-]+")
_WORD = re.compile(r"[A-Za-z0-9'\-]+")


class RewriteError(RuntimeError):
    """Raised when a configured query rewriter cannot produce a safe result."""


class QueryRewriter(Protocol):
    """Rewrite raw search queries using only the supplied current-turn context."""

    def rewrite(self, queries: list[str], context: str) -> RewriteResult:
        """Return the same number of standalone queries, or raise RewriteError."""


class NoRewriter:
    """The default slot that leaves raw queries unchanged without a hosted call."""

    def rewrite(self, queries: list[str], context: str) -> RewriteResult:
        """Return the supplied raw queries unchanged."""

        del context
        return RewriteResult(list(queries), "unchanged")


class HostedLLMQueryRewriter:
    """One structured call with the versioned prompt; anything outside the contract is a failed rewrite."""

    def __init__(self, client: CompletionClient, *, timeout_ms: int) -> None:
        self._client = client
        self._timeout_s = timeout_ms / 1000
        self._system = files("retold.retrieve").joinpath("prompts").joinpath("rewrite_v1.md").read_text("utf-8")

    @property
    def prompt_version(self) -> str:
        return REWRITE_PROMPT_VERSION

    def rewrite(self, queries: list[str], context: str) -> RewriteResult:
        try:
            text = self._client.complete(
                self._system, render_rewrite_request(queries, context), timeout_s=self._timeout_s
            )
            payload = parse_json_object(text)
        except StructuredOutputError as error:
            raise RewriteError(f"malformed rewriter output: {error}") from error
        except Exception as error:  # noqa: BLE001 - a provider failure is a failed rewrite, never an exception
            raise RewriteError(f"{type(error).__name__}: {error}") from error
        rewritten = payload.get("queries")
        if not isinstance(rewritten, list) or len(rewritten) != len(queries):
            raise RewriteError("rewriter did not return one query per input query")
        if not all(isinstance(query, str) and query.strip() for query in rewritten):
            raise RewriteError("rewriter returned an empty query")
        cleaned = [query.strip() for query in rewritten]
        invented = invented_names(cleaned, [*queries, context])
        if invented:
            raise RewriteError(f"rewrite introduced names absent from the queries and context: {invented}")
        return RewriteResult(cleaned, "applied")


def render_rewrite_request(queries: Sequence[str], context: str) -> str:
    lines = ["Context:", context.strip() or "(none)", "", "Queries:"]
    lines.extend(f"{index}. {query}" for index, query in enumerate(queries, start=1))
    return "\n".join(lines)


def invented_names(rewritten: Sequence[str], sources: Sequence[str]) -> list[str]:
    """Capitalised tokens in the rewrites, other than the first word of a query, that no source contains."""

    known = {word.lower() for source in sources for word in _WORD.findall(source)}
    invented: list[str] = []
    for query in rewritten:
        tokens = _NAME_TOKEN.findall(query)
        first = _WORD.match(query.strip())
        for token in tokens:
            if first is not None and token == first.group(0):
                continue
            if token.lower() not in known and token not in invented:
                invented.append(token)
    return invented


def rewriter_from_config(config: RetoldConfig) -> QueryRewriter:
    """The rewriter the configuration asks for: a no-op unless ``retrieval.rewrite.enabled``."""

    rewrite = config.retrieval.rewrite
    if not rewrite.enabled:
        return NoRewriter()
    client = completion_client_for(rewrite.model, max_output_tokens=rewrite.max_output_tokens)
    return HostedLLMQueryRewriter(client, timeout_ms=rewrite.timeout_ms)


def rewrite_stage(request: SearchRequest, config: RewriteConfig, rewriter: QueryRewriter) -> tuple[list[str], str]:
    """Run a configured rewriter and fall back to raw queries on an invalid or failed response."""

    if not config.enabled:
        return request.queries, "disabled"
    context = (request.context or "")[: config.max_context_chars]
    try:
        result = rewriter.rewrite(request.queries, context)
    except (TimeoutError, RewriteError):
        return request.queries, "failed"
    if len(result.queries) != len(request.queries) or any(not query.strip() for query in result.queries):
        return request.queries, "failed"
    if result.queries == request.queries:
        return request.queries, "unchanged"
    return result.queries, "applied"
