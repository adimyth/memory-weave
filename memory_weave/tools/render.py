"""Stable plain-text rendering for framework adapters that expose memory search results to an agent."""

from __future__ import annotations

from memory_weave.models import SearchResponse


def render_search(response: SearchResponse) -> str:
    """Render the search header followed by the complete result blocks or one explicit empty result."""

    searched_for = _searched_for(response)
    if not response.results:
        reason = response.empty_reason or "no eligible records matched"
        return f"No recalled memory for {searched_for}. Reason: {reason}."
    header = f"Recalled {len(response.results)} memor{'y' if len(response.results) == 1 else 'ies'} for {searched_for}."
    blocks = "\n\n".join(result.explanation.summary for result in response.results)
    return f"{header}\n\n{blocks}"


def _searched_for(response: SearchResponse) -> str:
    raw = "; ".join(f'"{query}"' for query in response.raw_queries)
    if response.rewritten_queries is None:
        return raw
    rewritten = "; ".join(f'"{query}"' for query in response.rewritten_queries)
    return f"{rewritten} (rewritten from {raw})"
