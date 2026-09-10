"""The adoption-first facade stays simple without bypassing Retold's policies."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from retold import EmbeddingProfileMismatch, MemorySession, Retold, RetoldSetupError
from retold.config import EmbeddingConfig, IngestionConfig, RetoldConfig
from retold.index.embedder import FakeEmbedder
from retold.ingest import FakeExtractor, FakeJudge, TableReviewer
from retold.models import ExtractionOutput, Scope, SessionSummary
from retold.policy import private_scope, readable_scopes
from retold.runtime import build_runtime
from retold.store import Store

_NOW = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
_CONFIG = RetoldConfig(embedding=EmbeddingConfig(model="fake-embedder", version="1", dims=8))


def _retold(tmp_path: Path) -> tuple[Retold, FakeEmbedder]:
    store = Store(tmp_path / "memory.sqlite")
    embedder = FakeEmbedder(dims=8)
    runtime = build_runtime(
        _CONFIG,
        store,
        embedder=embedder,
        judge=FakeJudge(),
        extractor=FakeExtractor(ExtractionOutput([], SessionSummary("No summary.", [], []))),
        reviewer=TableReviewer(),
        current_time=lambda: _NOW,
    )
    return Retold(runtime), embedder


def test_session_generates_identity_and_writes_private_memory_without_a_grant(tmp_path: Path) -> None:
    retold, _ = _retold(tmp_path)

    with retold.session(user_id="aditya") as memory:
        written = memory.remember("Aditya prefers concise answers.", evidence="I prefer concise answers.")

    assert isinstance(memory, MemorySession)
    assert memory.principal.agent_id == "assistant"
    assert memory.principal.session_id
    assert written.record_id is not None
    record = retold.store.get_record(written.record_id)
    assert record is not None
    assert record.scope == private_scope("assistant", "aditya")
    assert record.attribute is not None and record.attribute.startswith("facade_memory_")
    assert retold.store.grants_for("assistant") == []
    assert retold.store.get_session(memory.principal.session_id).ended_at is not None  # type: ignore[union-attr]


def test_supplied_session_id_attribute_and_typed_explained_search(tmp_path: Path) -> None:
    retold, embedder = _retold(tmp_path)
    content = "Aditya prefers concise technical answers."
    embedder.set_similarity("answer style", content, 0.9)

    with retold.session(user_id="aditya", session_id="session-1") as memory:
        memory.remember(content, evidence="I prefer concise technical answers.", attribute="answer_style")
        result = memory.search("answer style", k=2)

    assert memory.principal.session_id == "session-1"
    assert result.search_id
    assert result.empty_reason is None
    assert result.items[0].record.content == content
    assert content in result.text
    assert result.items[0].explanation.summary in result.text


def test_explicit_attribute_supersedes_a_changed_current_fact(tmp_path: Path) -> None:
    retold, _ = _retold(tmp_path)
    judge = retold.runtime.judge
    assert isinstance(judge, FakeJudge)
    concise = "I prefer concise answers."
    detailed = "I prefer long, detailed answers."
    judge.set_verdict(concise, detailed, "contradicts")

    with retold.session(user_id="aditya") as memory:
        first = memory.remember(concise, evidence=concise, attribute="answer_style", event_at=_NOW)
        second = memory.remember(
            detailed,
            evidence=detailed,
            attribute="answer_style",
            event_at=_NOW + timedelta(minutes=1),
        )

    assert first.record_id is not None
    assert second.outcome == f"superseded:{first.record_id}"
    assert retold.store.get_record(first.record_id).status == "superseded"  # type: ignore[union-attr]


def test_empty_search_has_a_stable_explanation(tmp_path: Path) -> None:
    retold, _ = _retold(tmp_path)

    with retold.session(user_id="aditya") as memory:
        result = memory.search("a fact that was never stored")

    assert result.items == []
    assert result.empty_reason
    assert result.empty_reason in result.text


def test_two_users_cannot_read_each_others_private_memory(tmp_path: Path) -> None:
    retold, embedder = _retold(tmp_path)
    content = "Aditya prefers concise technical answers."
    embedder.set_similarity("answer style", content, 0.9)

    with retold.session(user_id="aditya") as aditya:
        aditya.remember(content, evidence="I prefer concise technical answers.", attribute="answer_style")
    with retold.session(user_id="maya") as maya:
        result = maya.search("answer style")

    assert result.items == []


def test_project_memory_needs_an_explicit_grant(tmp_path: Path) -> None:
    retold, _ = _retold(tmp_path)
    project = Scope(kind="project", id="retold")
    assert project not in readable_scopes(retold.store, "assistant", "aditya")

    retold.store.set_grant("assistant", project, can_read=True, can_write=True)

    assert project in readable_scopes(retold.store, "assistant", "aditya")


def test_context_manager_does_not_schedule_extraction(tmp_path: Path) -> None:
    retold, _ = _retold(tmp_path)
    extractor = retold.runtime.extractor

    with retold.session(user_id="aditya", session_id="no-extract") as memory:
        memory.remember("Aditya prefers concise answers.", evidence="I prefer concise answers.")

    assert isinstance(extractor, FakeExtractor)
    assert extractor.call_count == 0
    with pytest.raises(RuntimeError, match="finished"):
        memory.search("answer style")


def test_explicit_extraction_fails_before_session_end_without_provider_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = RetoldConfig(
        embedding=EmbeddingConfig(model="fake-embedder", version="1", dims=8),
        ingestion=IngestionConfig(extraction_model="claude-test", review_model="claude-test"),
    )
    store = Store(tmp_path / "memory.sqlite")
    runtime = build_runtime(
        config,
        store,
        embedder=FakeEmbedder(dims=8),
        judge=FakeJudge(),
        extractor=FakeExtractor(ExtractionOutput([], SessionSummary("No summary.", [], []))),
        reviewer=TableReviewer(),
        current_time=lambda: _NOW,
    )
    memory = Retold(runtime).session(user_id="aditya", session_id="extract")

    with pytest.raises(RetoldSetupError, match="ANTHROPIC_API_KEY"):
        memory.finish(extract=True)

    assert store.get_session("extract").ended_at is None  # type: ignore[union-attr]


def test_open_owns_and_closes_its_store(tmp_path: Path) -> None:
    retold = Retold.open(tmp_path / "memory.sqlite", config=_CONFIG)
    retold.close()

    with pytest.raises(RuntimeError, match="closed"):
        retold.session(user_id="aditya")


def test_open_lite_uses_the_calibrated_minilm_profile(tmp_path: Path) -> None:
    retold = Retold.open(tmp_path / "memory.sqlite", profile="lite")

    assert retold.runtime.config.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert retold.runtime.config.embedding.dims == 384
    assert retold.runtime.config.retrieval.gate.dense_floor.semantic == 0.32
    retold.close()


def test_open_lite_refuses_a_second_configuration_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        Retold.open(tmp_path / "memory.sqlite", profile="lite", config=_CONFIG)


def test_open_rejects_an_unknown_profile(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="standard.*lite"):
        Retold.open(tmp_path / "memory.sqlite", profile="small")  # type: ignore[arg-type]


def test_runtime_refuses_embeddings_from_a_different_profile(tmp_path: Path) -> None:
    retold, _ = _retold(tmp_path)
    with retold.session(user_id="aditya") as memory:
        memory.remember(
            "I prefer concise answers.",
            evidence="I prefer concise answers.",
            attribute="answer_style",
        )
    changed = replace(_CONFIG, embedding=replace(_CONFIG.embedding, model="another-embedder", dims=16))

    with pytest.raises(EmbeddingProfileMismatch, match="retold reembed"):
        build_runtime(changed, retold.store)


def test_open_closes_the_store_after_an_embedding_profile_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite"
    retold, _ = _retold(tmp_path)
    with retold.session(user_id="aditya") as memory:
        memory.remember(
            "I prefer concise answers.",
            evidence="I prefer concise answers.",
            attribute="answer_style",
        )
    retold.store.close()

    changed = replace(_CONFIG, embedding=replace(_CONFIG.embedding, model="another-embedder", dims=16))
    with pytest.raises(EmbeddingProfileMismatch):
        Retold.open(path, config=changed)

    reopened = Store(path)
    reopened.close()


def test_local_model_failure_explains_the_install_and_network_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retold, _ = _retold(tmp_path)

    def unavailable(*args: object) -> object:
        raise OSError("model is not cached")

    monkeypatch.setattr(retold.runtime.ingestor, "write", unavailable)
    with retold.session(user_id="aditya") as memory:
        with pytest.raises(RetoldSetupError, match="local-models.*Hugging Face"):
            memory.remember("I prefer concise answers.", evidence="I prefer concise answers.")


@pytest.mark.parametrize("value", ["", "   "])
def test_facade_rejects_empty_content_evidence_and_query(tmp_path: Path, value: str) -> None:
    retold, _ = _retold(tmp_path)
    with retold.session(user_id="aditya") as memory:
        with pytest.raises(ValueError, match="content"):
            memory.remember(value, evidence="A sufficiently long quote.")
        with pytest.raises(ValueError, match="evidence"):
            memory.remember("A sufficiently long memory.", evidence=value)
        with pytest.raises(ValueError, match="query"):
            memory.search(value)
        with pytest.raises(ValueError, match="k must be positive"):
            memory.search("answer style", k=0)


def test_only_model_loading_failures_are_reported_as_setup_problems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from retold import LocalModelUnavailable

    retold, _ = _retold(tmp_path)

    def other_runtime_error(*args: object) -> object:
        raise RuntimeError("store_meta is missing the records_version row.")

    def model_missing(*args: object) -> object:
        raise LocalModelUnavailable("BGE-M3 support requires sentence-transformers.")

    with retold.session(user_id="aditya") as memory:
        monkeypatch.setattr(retold.runtime.retriever, "search", other_runtime_error)
        with pytest.raises(RuntimeError, match="records_version") as caught:
            memory.search("answer style")
        assert not isinstance(caught.value, RetoldSetupError)
        monkeypatch.setattr(retold.runtime.retriever, "search", model_missing)
        with pytest.raises(RetoldSetupError, match="local-models"):
            memory.search("answer style")


def test_finish_with_extraction_returns_the_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr("retold.facade.importlib.util.find_spec", lambda name: object())
    retold, _ = _retold(tmp_path)
    memory = retold.session(user_id="aditya", session_id="extract-thread")
    memory.remember("I prefer concise answers.", evidence="I prefer concise answers.")

    worker = memory.finish(extract=True)

    assert isinstance(worker, threading.Thread)
    worker.join(timeout=10)
    assert not worker.is_alive()


def test_unsupported_evidence_raises_unless_inference_is_allowed(tmp_path: Path) -> None:
    from retold import UnsupportedEvidenceError

    retold, _ = _retold(tmp_path)
    judge = retold.runtime.judge
    assert isinstance(judge, FakeJudge)
    judge.set_entailment("I prefer concise answers.", "Aditya is allergic to peanuts.", 0.0)
    judge.set_entailment("I prefer concise answers.", "The user is allergic to peanuts.", 0.0)

    with retold.session(user_id="aditya") as memory:
        with pytest.raises(UnsupportedEvidenceError, match="allow_inference"):
            memory.remember("Aditya is allergic to peanuts.", evidence="I prefer concise answers.")
        # The refusal happened before anything was persisted: no record in any status, no write event.
        scope = private_scope("assistant", "aditya")
        assert retold.store.records_in_scope(scope, statuses=["provisional", "confirmed"]) == []
        written_events = retold.store.connection.execute(
            "SELECT COUNT(*) FROM events WHERE kind LIKE 'record.%'"
        ).fetchone()[0]
        assert written_events == 0
        accepted = memory.remember(
            "Aditya is allergic to peanuts.", evidence="I prefer concise answers.", allow_inference=True
        )

    assert accepted.source_kind == "agent_inference"
    record = retold.store.get_record(accepted.record_id)
    assert record is not None and record.status == "provisional"


def test_a_claim_naming_the_user_is_judged_as_the_user(tmp_path: Path) -> None:
    retold, _ = _retold(tmp_path)
    judge = retold.runtime.judge
    assert isinstance(judge, FakeJudge)
    # The judge does not know that "Aditya" is the speaker, but it accepts "the user".
    judge.set_entailment("I prefer concise answers.", "Aditya prefers concise answers.", 0.01)
    judge.set_entailment("I prefer concise answers.", "The user prefers concise answers.", 0.99)

    with retold.session(user_id="aditya") as memory:
        written = memory.remember("Aditya prefers concise answers.", evidence="I prefer concise answers.")

    assert written.source_kind == "user_statement"
    record = retold.store.get_record(written.record_id)
    assert record is not None and record.status == "confirmed" and record.source_kind == "user_statement"


def _counts(store: Store) -> dict[str, int]:
    tables = ("records", "embeddings", "record_entities", "entities", "records_fts", "record_conflicts")
    counts = {table: store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables}
    counts["record_events"] = store.connection.execute(
        "SELECT COUNT(*) FROM events WHERE kind LIKE 'record.%'"
    ).fetchone()[0]
    return counts


def test_a_refused_write_leaves_no_residue_and_repeating_it_is_side_effect_free(tmp_path: Path) -> None:
    from retold import UnsupportedEvidenceError

    retold, _ = _retold(tmp_path)
    judge = retold.runtime.judge
    assert isinstance(judge, FakeJudge)
    judge.set_entailment("I prefer concise answers.", "Aditya is allergic to peanuts.", 0.0)
    judge.set_entailment("I prefer concise answers.", "The user is allergic to peanuts.", 0.0)

    with retold.session(user_id="aditya") as memory:
        before = _counts(retold.store)
        for _ in range(2):
            with pytest.raises(UnsupportedEvidenceError):
                memory.remember("Aditya is allergic to peanuts.", evidence="I prefer concise answers.")
            assert _counts(retold.store) == before, "a refused write must leave the store as it found it"
        accepted = memory.remember(
            "Aditya is allergic to peanuts.", evidence="I prefer concise answers.", allow_inference=True
        )
        after = _counts(retold.store)

    assert after["records"] == before["records"] + 1 and after["embeddings"] == before["embeddings"] + 1
    record = retold.store.get_record(accepted.record_id)
    assert record is not None and record.status == "provisional" and record.expires_at is not None
    assert accepted.note == "evidence does not support claim"


def test_tool_result_claims_follow_the_same_contract(tmp_path: Path) -> None:
    from retold import UnsupportedEvidenceError

    retold, _ = _retold(tmp_path)
    judge = retold.runtime.judge
    assert isinstance(judge, FakeJudge)
    judge.set_entailment("Deploy finished with exit code 1.", "The deploy succeeded.", 0.0)

    with retold.session(user_id="aditya") as memory:
        before = _counts(retold.store)
        with pytest.raises(UnsupportedEvidenceError):
            memory.remember(
                "The deploy succeeded.", evidence="Deploy finished with exit code 1.", source_kind="tool_result"
            )
        assert _counts(retold.store) == before
        supported = memory.remember(
            "The deploy at 14:02 succeeded.",
            evidence="Deploy finished at 14:02 with exit code 0.",
            source_kind="tool_result",
        )

    record = retold.store.get_record(supported.record_id)
    assert record is not None and record.source_kind == "tool_result" and record.status == "confirmed"


def test_low_level_writes_still_downgrade_and_persist(tmp_path: Path) -> None:
    from retold.ingest import WriteRequest
    from retold.models import EntityMention

    retold, _ = _retold(tmp_path)
    judge = retold.runtime.judge
    assert isinstance(judge, FakeJudge)
    judge.set_entailment("I prefer concise answers.", "Aditya is allergic to peanuts.", 0.0)
    judge.set_entailment("I prefer concise answers.", "The user is allergic to peanuts.", 0.0)
    memory = retold.session(user_id="aditya", session_id="low-level")
    memory.principal = retold.runtime.hooks.on_turn(memory.principal, "user", "I prefer concise answers.")

    result = retold.runtime.ingestor.write(
        memory.principal,
        WriteRequest(
            type="semantic",
            content="Aditya is allergic to peanuts.",
            source_kind="user_statement",
            evidence="I prefer concise answers.",
            attribute="allergy",
            scope=private_scope("assistant", "aditya"),
            entities=[EntityMention(kind="person", text="aditya", role="about")],
        ),
    )

    assert result.outcome == "created" and result.source_kind == "agent_inference"
    assert result.note == "evidence does not support claim"
    memory.finish()
