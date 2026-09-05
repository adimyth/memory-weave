"""Optional query rewriting with a no-op default implementation."""

from __future__ import annotations

from typing import Protocol

from memory_weave.config import RewriteConfig
from memory_weave.models import RewriteResult, SearchRequest


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
