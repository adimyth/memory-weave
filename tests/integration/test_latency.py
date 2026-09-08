"""Warm and cold latency of search and write on the 1K fixture with the real embedder and judge."""

from __future__ import annotations

import os
import statistics
from pathlib import Path
from time import perf_counter

import pytest

from memory_weave.config import MemoryWeaveConfig
from memory_weave.ingest import WriteRequest
from memory_weave.models import Principal, SearchRequest
from memory_weave.runtime import build_runtime
from memory_weave.store import Store

from .fixture_1k import copy_fixture

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("MEMORY_WEAVE_INTEGRATION") != "1",
        reason="set MEMORY_WEAVE_INTEGRATION=1 to run local-model integration tests",
    ),
]

SEARCH_P50_BUDGET_MS = 80.0
WRITE_P50_BUDGET_MS = 150.0
QUERIES = [
    "which editor does Aditya use",
    "commit message convention",
    "where is the release runbook",
    "billing API rate limit",
    "who is my manager",
    "how to roll back a billing deployment",
    "time zone for scheduling",
    "test runner preference",
    "change data capture decision",
    "deployment freeze window",
    "which shell does Aditya use now",
    "Priya's team since July",
    "on-call handover time",
    "where is the engineering handbook kept",
    "Meera's chart preferences",
    "Rohan's meeting length preference",
    "what happened at the Goa offsite",
    "invoice queue backlog incident",
    "embedding model for memory-weave",
    "hotfix procedure for the billing service",
]


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def test_warm_search_and_write_meet_the_lld_budgets(tmp_path: Path) -> None:
    store = Store(copy_fixture(tmp_path))
    runtime = build_runtime(MemoryWeaveConfig(), store)
    principal = Principal("coding-agent", "aditya", "latency-session", None)
    store.create_session("latency-session", "coding-agent", "aditya", None, __import__("memory_weave.util").util.now())

    def search(query: str) -> tuple[float, str]:
        request = SearchRequest([query], None, None, None, None, None, 8, False)
        started = perf_counter()
        response = runtime.retriever.search(principal, request)
        return (perf_counter() - started) * 1000, response.search_id

    cold_ms, cold_id = search(QUERIES[0])
    cold_log = store.read_search_log(cold_id)
    assert cold_log is not None and cold_log["warm"] == 0
    warm: list[float] = []
    stage_totals: dict[str, list[float]] = {}
    for query in QUERIES:  # twenty distinct queries, so the exact-string query cache never answers
        elapsed, search_id = search(query)
        warm.append(elapsed)
        log = store.read_search_log(search_id)
        assert log is not None and log["warm"] == 1
        for stage, value in log["timings_ms"].items():  # type: ignore[union-attr]
            stage_totals.setdefault(stage, []).append(float(value))
    search_p50 = statistics.median(warm)
    print(f"\ncold search {cold_ms:.1f} ms; warm search p50 {search_p50:.1f} ms p95 {_percentile(warm, 0.95):.1f} ms")
    for stage, values in stage_totals.items():
        print(f"  {stage:<14} p50 {statistics.median(values):6.1f} ms  p95 {_percentile(values, 0.95):6.1f} ms")

    writes: list[float] = []
    for index in range(10):
        request = WriteRequest(
            type="semantic",
            content=f"Aditya keeps the latency probe number {index} in his notes.",
            source_kind="agent_inference",
            evidence=None,
            attribute=f"latency_probe_{index}",
        )
        started = perf_counter()
        result = runtime.ingestor.write(principal, request)
        writes.append((perf_counter() - started) * 1000)
        assert result.record_id is not None
    write_p50 = statistics.median(writes[1:])
    print(f"write p50 {write_p50:.1f} ms p95 {_percentile(writes[1:], 0.95):.1f} ms (first {writes[0]:.1f} ms)")

    assert search_p50 < SEARCH_P50_BUDGET_MS, f"warm search p50 {search_p50:.1f} ms"
    assert write_p50 < WRITE_P50_BUDGET_MS, f"write p50 {write_p50:.1f} ms"
    store.close()
