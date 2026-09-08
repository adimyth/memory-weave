"""LLD 17: a 50K synthetic store over 200 users and 10 agents, with a fixed 25 ms embedding cost.

The fake embedder sleeps for 25 ms per query call, the LLD 15 estimate for one warm bge-m3 forward pass,
so the measurement isolates everything the retriever does around the model: scope resolution, the
eligibility mask, the dense matmul over 50K rows, the two FTS queries, fusion, the gate, the log. The LLD
15 estimate for the scopes and filter stages together is 3 ms, and both are printed separately.
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from memory_weave.config import EmbeddingConfig, MemoryWeaveConfig, RetrievalConfig
from memory_weave.host import MemoryHost
from memory_weave.index.embedder import FakeEmbedder
from memory_weave.models import Principal, Record, Scope, SearchRequest
from memory_weave.runtime import build_runtime
from memory_weave.store import Store
from memory_weave.util import render_subject

from .fixture_1k import NOW

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("MEMORY_WEAVE_RUN_SLOW") != "1", reason="set MEMORY_WEAVE_RUN_SLOW=1 to run scale tests"
    ),
]

RECORDS = 50_000
USERS = 200
AGENTS = 10
DIMS = 256
EMBED_SLEEP_S = 0.025
SEARCH_P50_BUDGET_MS = 80.0


class SleepingEmbedder(FakeEmbedder):
    """The fake embedder with the LLD's warm forward-pass cost on every query call."""

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        time.sleep(EMBED_SLEEP_S)
        return super().embed_queries(texts)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def _build(path: Path, embedder: FakeEmbedder) -> Store:
    store = Store(path)
    host = MemoryHost(store)
    users = [f"user-{index:03d}" for index in range(USERS)]
    agents = [f"agent-{index:02d}" for index in range(AGENTS)]
    for user_index, user in enumerate(users):
        host.grant(agents[user_index % AGENTS], Scope(kind="user", id=user), read=True, write=True)
        host.grant(agents[(user_index + 1) % AGENTS], Scope(kind="user", id=user), read=True, write=False)
    generator = np.random.default_rng(7)
    vectors = generator.standard_normal((RECORDS, DIMS), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    topics = ("deploy", "invoice", "editor", "meeting", "cluster", "pipeline", "budget", "retry", "schema", "alert")
    with store.transaction():
        for index in range(RECORDS):
            user = users[index % USERS]
            agent = agents[index % AGENTS]
            scope = Scope(kind="user", id=user) if index % 3 else Scope(kind="agent", id=f"{agent}/{user}")
            topic = topics[index % len(topics)]
            content = f"{user} {topic} fact {index}: the {topic} setting for run {index % 97} is {index % 13}."
            record = Record(
                id=f"scale-{index:06d}",
                type="semantic" if index % 5 else "episodic",
                version=1,
                content=content,
                subject=render_subject(None, f"{topic}_{index}"),
                scope=scope,
                source_kind="user_statement",
                source_ref=None,
                creator_agent_id=agent,
                evidence=None,
                created_at=NOW,
                event_at=NOW,
                expires_at=None,
                confidence=0.9,
                status="confirmed",
                supersedes_id=None,
                reinforcements=0,
                last_reinforced_at=None,
                tags=[],
                entity_ids=[],
                attribute=f"{topic}_{index}" if index % 5 else "-",
            )
            store.insert_record(record)
            store.put_embedding(record.id, embedder.name, embedder.version, vectors[index])
            store.upsert_fts(record.id, content, record.subject, "")
    return store


def test_fifty_thousand_records_search_under_the_lld_budget(tmp_path: Path) -> None:
    embedder = SleepingEmbedder(dims=DIMS)
    started = perf_counter()
    store = _build(tmp_path / "scale.sqlite", embedder)
    print(f"\nbuilt {RECORDS} records over {USERS} users and {AGENTS} agents in {perf_counter() - started:.1f} s")
    config = MemoryWeaveConfig(
        embedding=EmbeddingConfig(model=embedder.name, version=embedder.version, dims=DIMS),
        retrieval=RetrievalConfig(per_generator_k=30, default_k=8),
    )
    runtime = build_runtime(config, store, embedder=embedder)
    principal = Principal("agent-03", "user-003", None, None)
    runtime.retriever.search(principal, SearchRequest(["deploy setting"], None, None, None, None, None, 8, False))

    latencies: list[float] = []
    stages: dict[str, list[float]] = {}
    for index in range(30):
        query = f"{'deploy invoice editor meeting cluster'.split()[index % 5]} setting for run {index}"
        started = perf_counter()
        response = runtime.retriever.search(principal, SearchRequest([query], None, None, None, None, None, 8, False))
        latencies.append((perf_counter() - started) * 1000)
        log = store.read_search_log(response.search_id)
        assert log is not None and log["warm"] == 1
        for stage, value in log["timings_ms"].items():  # type: ignore[union-attr]
            stages.setdefault(stage, []).append(float(value))
    p50 = statistics.median(latencies)
    print(
        f"warm search p50 {p50:.1f} ms p95 {_percentile(latencies, 0.95):.1f} ms with a {EMBED_SLEEP_S * 1000:.0f} ms embed"  # noqa: E501
    )
    for stage in (
        "scopes",
        "filter",
        "index_refresh",
        "embed",
        "dense",
        "lexical",
        "entity",
        "fuse",
        "gate",
        "log",
        "total",
    ):
        if stage in stages:
            print(
                f"  {stage:<14} p50 {statistics.median(stages[stage]):6.2f} ms  p95 {_percentile(stages[stage], 0.95):6.2f} ms"  # noqa: E501
            )
    assert p50 < SEARCH_P50_BUDGET_MS, f"p50 {p50:.1f} ms"
    store.close()
