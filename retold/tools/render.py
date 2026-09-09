"""Stable plain-text rendering for framework adapters that expose memory search results to an agent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from retold.models import SearchResponse


def render_search(response: SearchResponse) -> str:
    """Render the search header followed by the complete result blocks or one explicit empty result."""

    searched_for = _searched_for(response)
    if not response.results:
        reason = response.empty_reason or "no eligible records matched"
        return f"No recalled memory for {searched_for}. Reason: {reason}."
    header = f"Recalled {len(response.results)} memor{'y' if len(response.results) == 1 else 'ies'} for {searched_for}."
    blocks = "\n\n".join(result.explanation.summary for result in response.results)
    return f"{header}\n\n{blocks}"


def render_search_payload(queries: Sequence[str], payload: Mapping[str, Any]) -> str:
    """Render the handler's search payload the same way ``render_search`` renders a response."""

    raw = "; ".join(f'"{query}"' for query in queries)
    rewritten = payload.get("rewritten_queries")
    if isinstance(rewritten, list) and rewritten:
        searched_for = "; ".join(f'"{query}"' for query in rewritten) + f" (rewritten from {raw})"
    else:
        searched_for = raw
    results = payload.get("results") or []
    if not results:
        reason = payload.get("empty_reason") or "no eligible records matched"
        return f"No recalled memory for {searched_for}. Reason: {reason}."
    header = f"Recalled {len(results)} memor{'y' if len(results) == 1 else 'ies'} for {searched_for}."
    blocks = "\n\n".join(_payload_block(result) for result in results)
    return f"{header}\n\n{blocks}"


def _payload_block(result: Mapping[str, Any]) -> str:
    explanation = result.get("explanation") or {}
    record = result.get("record") or {}
    summary = str(explanation.get("summary") or record.get("content") or "")
    return f"id={record.get('id')}\n{summary}"


def _searched_for(response: SearchResponse) -> str:
    raw = "; ".join(f'"{query}"' for query in response.raw_queries)
    if response.rewritten_queries is None:
        return raw
    rewritten = "; ".join(f'"{query}"' for query in response.rewritten_queries)
    return f"{rewritten} (rewritten from {raw})"
