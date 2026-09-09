"""Phase 11: the optional query-rewrite stage, with a fake rewriter and a fake hosted client."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from retold.config import EmbeddingConfig, RetoldConfig, RetrievalConfig, RewriteConfig
from retold.index.embedder import FakeEmbedder
from retold.index.vector import VectorIndex
from retold.models import Principal, Record, RewriteResult, Scope, SearchRequest
from retold.retrieve import HostedLLMQueryRewriter, Retriever, RewriteError, invented_names
from retold.retrieve.rewrite import rewrite_stage
from retold.store import Store

_NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
_AGENT = "research-agent"
_USER = "aditya"
_USER_SCOPE = Scope(kind="user", id=_USER)
_PRINCIPAL = Principal(_AGENT, _USER, "session-1", None)
_EMBEDDING = EmbeddingConfig(model="fake-embedder", version="1", dims=8)
_CONFIG = RetoldConfig(
    embedding=_EMBEDDING,
    retrieval=RetrievalConfig(per_generator_k=10, default_k=8, rewrite=RewriteConfig(enabled=True)),
)
_CONTENT = "Aditya uses Vim as the preferred editor."
_RAW = "what does he prefer"
_REWRITTEN = "Aditya preferred editor"
_CONTEXT = "User: Aditya said he set up his editor yesterday.\nAssistant: Noted."


class FakeRewriter:
    def __init__(self, table: dict[str, str] | None = None, *, error: Exception | None = None) -> None:
        self._table = table or {}
        self._error = error
        self.calls: list[tuple[list[str], str]] = []

    def rewrite(self, queries: list[str], context: str) -> RewriteResult:
        self.calls.append((list(queries), context))
        if self._error is not None:
            raise self._error
        return RewriteResult([self._table.get(query, query) for query in queries], "applied")


class FakeClient:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[tuple[str, str, float]] = []

    def complete(self, system: str, user: str, *, timeout_s: float) -> str:
        self.calls.append((system, user, timeout_s))
        return self._reply


@pytest.fixture
def store(tmp_path: Path) -> Store:
    database = Store(tmp_path / "memory.sqlite")
    database.set_grant(_AGENT, _USER_SCOPE, can_read=True, can_write=True)
    yield database
    database.close()


def _record(record_id: str, content: str) -> Record:
    return Record(
        id=record_id,
        type="semantic",
        version=1,
        content=content,
        subject=f"person:aditya/{record_id}",
        scope=_USER_SCOPE,
        source_kind="user_statement",
        source_ref=None,
        creator_agent_id=_AGENT,
        evidence=None,
        created_at=_NOW,
        event_at=_NOW,
        expires_at=None,
        confidence=0.95,
        status="confirmed",
        supersedes_id=None,
        reinforcements=0,
        last_reinforced_at=None,
        tags=[],
        entity_ids=[],
    )


def _world(
    store: Store, rewriter: FakeRewriter, *, similarities: dict[str, float] | None = None
) -> tuple[Retriever, FakeEmbedder]:
    embedder = FakeEmbedder(dims=_EMBEDDING.dims)
    record = _record("editor", _CONTENT)
    # The fake embedder fixes vectors when similarities are declared, so declare them before storing.
    for query, cosine in (similarities or {_REWRITTEN: 0.85}).items():
        embedder.set_similarity(query, _CONTENT, cosine)
    store.insert_record(record)
    store.put_embedding(record.id, embedder.name, embedder.version, embedder.embed_documents([_CONTENT])[0])
    store.upsert_fts(record.id, record.content, record.subject, "")
    retriever = Retriever(
        store, VectorIndex(_EMBEDDING), embedder, _CONFIG, rewriter=rewriter, current_time=lambda: _NOW
    )
    return retriever, embedder


def _request(queries: list[str], *, entities: list[str] | None = None) -> SearchRequest:
    return SearchRequest(queries, _CONTEXT, None, None, None, None, 8, False, trigger="tool")  # noqa: PLR2004


def test_rewritten_queries_drive_dense_and_lexical_and_both_forms_are_logged(store: Store) -> None:
    rewriter = FakeRewriter({_RAW: _REWRITTEN})
    retriever, _ = _world(store, rewriter)

    response = retriever.search(_PRINCIPAL, _request([_RAW]))

    assert [result.record.id for result in response.results] == ["editor"]
    assert response.rewrite_status == "applied"
    assert response.raw_queries == [_RAW]
    assert response.rewritten_queries == [_REWRITTEN]
    explanation = response.results[0].explanation
    assert explanation.dense is not None and explanation.lexical is not None
    log = store.read_search_log(response.search_id)
    assert log is not None
    assert log["request"]["queries"] == [_RAW]  # type: ignore[index]
    assert log["rewritten_queries"] == [_REWRITTEN]
    assert log["rewrite_status"] == "applied"
    assert log["context"] == _CONTEXT
    # The rewriter saw the raw queries and the context, and nothing from the store.
    assert rewriter.calls == [([_RAW], _CONTEXT)]
    assert "timings_ms" in log and "rewrite" in log["timings_ms"]  # type: ignore[operator]


def test_entity_hints_pass_through_the_rewrite_untouched(store: Store) -> None:
    rewriter = FakeRewriter({_RAW: _REWRITTEN})
    retriever, _ = _world(store, rewriter)
    request = replace(_request([_RAW]), entities=["fixture-entity"])

    response = retriever.search(_PRINCIPAL, request)

    log = store.read_search_log(response.search_id)
    assert log is not None
    assert log["request"]["entities"] == ["fixture-entity"]  # type: ignore[index]
    assert rewriter.calls[0][0] == [_RAW]


def test_rewriter_timeout_falls_back_to_the_raw_queries(store: Store) -> None:
    rewriter = FakeRewriter(error=TimeoutError("slow"))
    retriever, _ = _world(store, rewriter, similarities={_RAW: 0.80})

    response = retriever.search(_PRINCIPAL, _request([_RAW]))

    assert response.rewrite_status == "failed"
    assert response.rewritten_queries is None
    assert [result.record.id for result in response.results] == ["editor"]
    log = store.read_search_log(response.search_id)
    assert log is not None and log["rewrite_status"] == "failed" and log["rewritten_queries"] is None


def test_context_is_truncated_to_the_configured_length(store: Store) -> None:
    rewriter = FakeRewriter()
    config = replace(
        _CONFIG, retrieval=replace(_CONFIG.retrieval, rewrite=RewriteConfig(enabled=True, max_context_chars=10))
    )
    retriever = Retriever(store, VectorIndex(_EMBEDDING), FakeEmbedder(dims=8), config, rewriter=rewriter)

    retriever.search(_PRINCIPAL, _request([_RAW]))

    assert rewriter.calls[0][1] == _CONTEXT[:10]


def test_hosted_rewriter_parses_a_valid_reply_and_never_sees_records() -> None:
    client = FakeClient('{"queries": ["Aditya preferred editor"]}')
    rewriter = HostedLLMQueryRewriter(client, timeout_ms=800)

    result = rewriter.rewrite([_RAW], _CONTEXT)

    assert result == RewriteResult([_REWRITTEN], "applied")
    system, user, timeout = client.calls[0]
    assert "standalone" in system.lower()
    assert _RAW in user and _CONTEXT in user
    assert _CONTENT not in user
    assert timeout == 0.8


@pytest.mark.parametrize(
    "reply",
    [
        "not json at all",
        '{"queries": ["one", "two"]}',
        '{"queries": [""]}',
        '{"queries": "Aditya preferred editor"}',
        '{"queries": ["editor preferred by Rohan"]}',
    ],
)
def test_hosted_rewriter_refuses_malformed_or_inventive_replies(reply: str) -> None:
    rewriter = HostedLLMQueryRewriter(FakeClient(reply), timeout_ms=800)
    with pytest.raises(RewriteError):
        rewriter.rewrite([_RAW], _CONTEXT)
    request = SearchRequest([_RAW], _CONTEXT, None, None, None, None, 8, False)
    assert rewrite_stage(request, RewriteConfig(enabled=True), rewriter) == ([_RAW], "failed")


def test_hosted_rewriter_reports_a_provider_failure_as_a_failed_rewrite() -> None:
    class Broken:
        def complete(self, system: str, user: str, *, timeout_s: float) -> str:
            raise ConnectionError("down")

    rewriter = HostedLLMQueryRewriter(Broken(), timeout_ms=800)
    with pytest.raises(RewriteError, match="ConnectionError"):
        rewriter.rewrite([_RAW], _CONTEXT)


def test_invented_names_ignore_the_leading_word_and_known_names() -> None:
    sources = [_RAW, _CONTEXT]
    assert invented_names(["Which editor does Aditya prefer"], sources) == []
    assert invented_names(["Aditya preferred editor"], sources) == []
    assert invented_names(["What does Rohan prefer in Vim"], sources) == ["Rohan", "Vim"]
    assert invented_names(["what does he prefer"], sources) == []


@pytest.mark.live
def test_live_rewriter_names_the_subject_from_the_context() -> None:
    if os.environ.get("RETOLD_LIVE") != "1":
        pytest.skip("set RETOLD_LIVE=1 to run the hosted rewriter")
    from retold.hosted import completion_client_for

    model = os.environ.get("RETOLD_REWRITE_MODEL", RewriteConfig().model)
    key = "ANTHROPIC_API_KEY" if model.startswith("claude-") else "OPENAI_API_KEY"
    if not os.environ.get(key):
        pytest.skip(f"{key} is not set")
    rewriter = HostedLLMQueryRewriter(completion_client_for(model, max_output_tokens=512), timeout_ms=10000)

    result = rewriter.rewrite([_RAW], _CONTEXT)

    assert len(result.queries) == 1
    assert "aditya" in result.queries[0].lower(), result.queries
