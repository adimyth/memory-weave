"""Interactive writes from four threads while two extraction workers run against the same store.

SQLite has one writer at a time; this test gives that ceiling a number. Every store transaction begins
IMMEDIATE and waits on the busy timeout, so the expected count of lock errors is zero and the cost shows up
as write latency instead. Reported: write p50 and p95 under contention and the number of lock errors.
"""

from __future__ import annotations

import os
import statistics
import threading
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pytest

from memory_weave.config import EmbeddingConfig, MemoryWeaveConfig
from memory_weave.host import MemoryHost
from memory_weave.index.embedder import FakeEmbedder
from memory_weave.ingest import FakeExtractor, FakeJudge, TableReviewer, WriteRequest
from memory_weave.models import CandidateRecord, ExtractionOutput, Principal, Scope, SessionSummary, Turn
from memory_weave.runtime import build_runtime
from memory_weave.store import Store

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("MEMORY_WEAVE_RUN_SLOW") != "1", reason="set MEMORY_WEAVE_RUN_SLOW=1 to run scale tests"
    ),
]

WRITERS = 4
WRITES_PER_THREAD = 40
EXTRACTORS = 2
SESSIONS_PER_EXTRACTOR = 10
_NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def test_writes_under_extraction_contention_report_latency_and_lock_errors(tmp_path: Path) -> None:
    config = MemoryWeaveConfig(embedding=EmbeddingConfig(model="fake-embedder", version="1", dims=32))
    store = Store(tmp_path / "contention.sqlite")
    host = MemoryHost(store)
    users = [f"user-{index}" for index in range(WRITERS)]
    for user in users:
        host.grant("agent", Scope(kind="user", id=user), read=True, write=True)
        host.provision_user(user)
    extraction_users = [f"extract-{index}" for index in range(EXTRACTORS)]
    for user in extraction_users:
        host.grant("agent", Scope(kind="user", id=user), read=True, write=True)
        host.provision_user(user)

    def extractor_output(session_id: str) -> ExtractionOutput:
        candidates = [
            CandidateRecord(
                type="semantic",
                content=f"The user of {session_id} prefers option {index}.",
                attribute=f"option_{index}",
                source_kind="user_statement",
                evidence=f"I prefer option {index} for {session_id}",
                evidence_turn=index + 1,
                entity_mentions=[],
                event_at=None,
                confidence=0.9,
            )
            for index in range(5)
        ]
        return ExtractionOutput(candidates, SessionSummary(f"Session {session_id} covered five options.", [], []))

    def make_runtime(session_ids: list[str]):  # type: ignore[no-untyped-def]
        outputs = {session_id: extractor_output(session_id) for session_id in session_ids}
        extractor = FakeExtractor(lambda turns, context: outputs[turns[0].session_id])
        return build_runtime(
            config,
            store,
            embedder=FakeEmbedder(dims=32),
            judge=FakeJudge(),
            extractor=extractor,
            reviewer=TableReviewer(),
        )

    # Seed the sessions the extraction workers will consume.
    sessions: dict[str, list[str]] = {}
    for worker_index, user in enumerate(extraction_users):
        ids = [f"session-{worker_index}-{n}" for n in range(SESSIONS_PER_EXTRACTOR)]
        sessions[user] = ids
        for session_id in ids:
            store.create_session(session_id, "agent", user, None, _NOW)
            for turn in range(5):
                store.append_turn(Turn(session_id, turn + 1, "user", f"I prefer option {turn} for {session_id}", _NOW))
            store.end_session(session_id, _NOW)

    write_latencies: list[float] = []
    lock_errors: list[str] = []
    latency_lock = threading.Lock()

    def writer(user: str) -> None:
        runtime = make_runtime([])
        principal = Principal("agent", user, None, None)
        for index in range(WRITES_PER_THREAD):
            request = WriteRequest(
                type="semantic",
                content=f"{user} keeps fact number {index} in the store.",
                source_kind="agent_inference",
                evidence=None,
                attribute=f"fact_{index}",
            )
            started = perf_counter()
            try:
                result = runtime.ingestor.write(principal, request)
                assert result.record_id is not None
            except Exception as error:  # noqa: BLE001
                with latency_lock:
                    lock_errors.append(f"{type(error).__name__}: {error}")
                continue
            with latency_lock:
                write_latencies.append((perf_counter() - started) * 1000)

    def extraction_worker(user: str) -> None:
        runtime = make_runtime(sessions[user])
        for session_id in sessions[user]:
            principal = Principal("agent", user, session_id, None)
            try:
                result = runtime.extraction.extract_session(session_id, principal)
                assert result.status == "completed", result.reason
            except Exception as error:  # noqa: BLE001
                with latency_lock:
                    lock_errors.append(f"extraction {type(error).__name__}: {error}")

    threads = [threading.Thread(target=writer, args=(user,)) for user in users]
    threads += [threading.Thread(target=extraction_worker, args=(user,)) for user in extraction_users]
    started = perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = perf_counter() - started

    written = store.connection.execute("SELECT count(*) FROM records").fetchone()[0]
    print(
        f"\n{WRITERS} writers x {WRITES_PER_THREAD} writes beside {EXTRACTORS} extraction workers x "
        f"{SESSIONS_PER_EXTRACTOR} sessions in {elapsed:.1f} s; {written} records; "
        f"write p50 {statistics.median(write_latencies):.1f} ms p95 {_percentile(write_latencies, 0.95):.1f} ms; "
        f"lock errors {len(lock_errors)}"
    )
    assert lock_errors == [], lock_errors[:3]
    assert len(write_latencies) == WRITERS * WRITES_PER_THREAD
    for user in extraction_users:
        for session_id in sessions[user]:
            assert store.get_session(session_id).extracted_at is not None  # type: ignore[union-attr]
    store.close()
