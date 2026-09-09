"""The composition root: one place that wires a store and a configuration into the working components.

The CLI, the adapters, and the benchmarks all need the same graph: embedder, vector index, equivalence
judge, session buffer, ingestor, retriever, tool handlers, extraction runner, and session hooks. Building it
here keeps the wiring in one place and lets tests substitute fakes for the model-backed pieces.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from retold.config import RetoldConfig
from retold.hosted import completion_client_for
from retold.index.embedder import BgeM3Embedder, Embedder
from retold.index.reranker import Reranker, reranker_from_config
from retold.index.vector import VectorIndex
from retold.ingest import (
    CandidateReviewer,
    EquivalenceJudge,
    ExtractionRunner,
    Extractor,
    Ingestor,
    NLICrossEncoderJudge,
    SessionBuffer,
    SessionHooks,
    StructuredLLMExtractor,
    StructuredLLMReviewer,
)
from retold.policy.activation import ActivationService
from retold.retrieve import QueryRewriter, Retriever, rewriter_from_config
from retold.store import Store
from retold.tools import ToolHandlers
from retold.util import now


@dataclass(slots=True)
class MemoryRuntime:
    config: RetoldConfig
    store: Store
    embedder: Embedder
    vector_index: VectorIndex
    judge: EquivalenceJudge
    session_buffer: SessionBuffer
    ingestor: Ingestor
    retriever: Retriever
    handlers: ToolHandlers
    extractor: Extractor
    reviewer: CandidateReviewer
    extraction: ExtractionRunner
    hooks: SessionHooks


def build_runtime(
    config: RetoldConfig,
    store: Store,
    *,
    embedder: Embedder | None = None,
    judge: EquivalenceJudge | None = None,
    rewriter: QueryRewriter | None = None,
    reranker: Reranker | None = None,
    extractor: Extractor | None = None,
    reviewer: CandidateReviewer | None = None,
    activation: ActivationService | None = None,
    current_time: Callable[[], datetime] = now,
) -> MemoryRuntime:
    """Wire the components for ``config`` over ``store``; any argument given replaces the configured piece.

    Hosted pieces are created lazily by their own classes, so building a runtime makes no model call and
    loads no local model until the first write, search, or extraction.
    """

    embedder = embedder or BgeM3Embedder(config.embedding)
    judge = judge or NLICrossEncoderJudge(config.ingestion.equivalence)
    vector_index = VectorIndex(config.embedding)
    session_buffer = SessionBuffer(store)
    ingestor = Ingestor(store, vector_index, embedder, judge, session_buffer, config, current_time=current_time)
    retriever = Retriever(
        store,
        vector_index,
        embedder,
        config,
        rewriter=rewriter or rewriter_from_config(config),
        reranker=reranker or reranker_from_config(config),
        current_time=current_time,
    )
    handlers = ToolHandlers(
        retriever, ingestor, store, vector_index, default_k=config.retrieval.default_k, activation=activation
    )
    ingestion = config.ingestion
    if extractor is None:
        extractor = StructuredLLMExtractor(
            completion_client_for(ingestion.extraction_model, max_output_tokens=ingestion.hosted_max_output_tokens),
            timeout_ms=ingestion.extraction_timeout_ms,
            max_candidates=ingestion.extraction_max_candidates,
        )
    if reviewer is None:
        reviewer = StructuredLLMReviewer(
            completion_client_for(ingestion.review_model, max_output_tokens=ingestion.hosted_max_output_tokens),
            timeout_ms=ingestion.review_timeout_ms,
        )
    extraction = ExtractionRunner(
        store,
        ingestor,
        extractor,
        reviewer,
        session_buffer,
        config,
        activation=activation,
        current_time=current_time,
    )
    hooks = SessionHooks(
        store,
        session_buffer,
        config,
        on_end=lambda session_id, principal: extraction.schedule(session_id, principal),
        current_time=current_time,
    )
    return MemoryRuntime(
        config=config,
        store=store,
        embedder=embedder,
        vector_index=vector_index,
        judge=judge,
        session_buffer=session_buffer,
        ingestor=ingestor,
        retriever=retriever,
        handlers=handlers,
        extractor=extractor,
        reviewer=reviewer,
        extraction=extraction,
        hooks=hooks,
    )
