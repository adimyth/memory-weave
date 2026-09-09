"""Phase 12 CLI: every operational command against a temporary store, with fakes wired through the runtime."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from retold import cli, runtime
from retold.host import MemoryHost
from retold.index.embedder import FakeEmbedder
from retold.ingest import FakeExtractor, FakeJudge, TableReviewer
from retold.models import ExtractionOutput, Principal, Scope, SessionSummary, Turn
from retold.store import Store

_NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
_CONFIG_YAML = """
embedding:
  model: fake-embedder
  version: "1"
  dims: 8
sessions:
  retain_days: 30
"""


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    config = tmp_path / "config.yaml"
    config.write_text(_CONFIG_YAML, encoding="utf-8")
    store = tmp_path / "memory.sqlite"

    def fake_runtime(config_obj, store_obj, **overrides):  # type: ignore[no-untyped-def]
        overrides.setdefault(
            "embedder", FakeEmbedder(dims=config_obj.embedding.dims, version=config_obj.embedding.version)
        )
        overrides.setdefault("judge", FakeJudge())
        overrides.setdefault(
            "extractor", FakeExtractor(ExtractionOutput([], SessionSummary("The user set up an editor.", [], [])))
        )
        overrides.setdefault("reviewer", TableReviewer())
        return original(config_obj, store_obj, **overrides)

    original = runtime.build_runtime
    monkeypatch.setattr(runtime, "build_runtime", fake_runtime)
    return store, config


def _seed(store_path: Path) -> str:
    store = Store(store_path)
    host = MemoryHost(store)
    host.grant("agent", Scope(kind="user", id="aditya"), read=True, write=True)
    host.provision_user("aditya", aliases=("Aditya",))
    store.create_session("s1", "agent", "aditya", None, _NOW)
    store.append_turn(Turn("s1", 1, "user", "I prefer Vim for editing code.", _NOW))
    store.append_turn(Turn("s1", 2, "assistant", "Noted.", _NOW))
    store.end_session("s1", _NOW)
    store.close()
    return "s1"


def _run(args: list[str], store: Path, config: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = cli.main(["--store", str(store), "--config", str(config), *args])
    return code, capsys.readouterr().out


def test_migrate_search_get_and_dump(paths: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    store, config = paths
    assert _run(["migrate"], store, config, capsys) == (0, "migrated\n")
    _seed(store)
    code, out = _run(["extract", "s1"], store, config, capsys)
    assert code == 0 and "completed" in out and "summary" in out

    code, out = _run(["search", "--agent", "agent", "--user", "aditya", "editor", "set", "up"], store, config, capsys)
    assert code == 0 and "search " in out and "returned=" in out
    code, out = _run(["dump", "--scope", "user:aditya"], store, config, capsys)
    assert code == 0 and "session_summary" in out and "1 record(s)" in out
    record_id = out.split()[0]
    code, out = _run(["get", record_id, "--agent", "agent", "--user", "aditya"], store, config, capsys)
    assert code == 0 and json.loads(out)["ok"] is True
    code, out = _run(["get", record_id, "--agent", "agent", "--user", "someone-else"], store, config, capsys)
    assert code == cli.EXIT_REFUSED and json.loads(out)["ok"] is False
    code, out = _run(["search", "--agent", "agent", "--user", "aditya", "--json", "editor"], store, config, capsys)
    assert code == 0 and "search_id" in json.loads(out)


def test_extract_is_idempotent_and_reports_a_missing_session(
    paths: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    store, config = paths
    _seed(store)
    assert _run(["extract", "s1"], store, config, capsys)[0] == 0
    code, out = _run(["extract", "s1", "--json"], store, config, capsys)
    assert code == 0 and json.loads(out)["status"] == "already_extracted"
    code, out = _run(["extract", "s1", "--force"], store, config, capsys)
    assert code == 0 and "already_reinforced" in out
    code, out = _run(["extract", "nope"], store, config, capsys)
    assert code == cli.EXIT_REFUSED and "does not exist" in out


def test_expire_retain_and_review_due_report_counts(
    paths: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    store, config = paths
    _seed(store)
    assert _run(["expire"], store, config, capsys) == (0, "expired 0 record(s)\n")
    assert _run(["retain"], store, config, capsys) == (0, "retained 0 session(s)\n")
    assert _run(["review-due"], store, config, capsys) == (0, "flagged 0 record(s)\n")


def test_erase_requires_confirmation_and_reports(paths: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    store, config = paths
    _seed(store)
    _run(["extract", "s1"], store, config, capsys)
    code, out = _run(["erase", "--user", "aditya", "--reason", "request"], store, config, capsys)
    assert code == cli.EXIT_REFUSED and "--yes" in out
    code, out = _run(["erase", "--session", "s1", "--reason", "request", "--yes"], store, config, capsys)
    assert code == 0 and json.loads(out)["sessions"] == ["s1"]
    code, out = _run(["erase", "--record", "missing", "--reason", "request", "--yes"], store, config, capsys)
    assert code == cli.EXIT_REFUSED and "does not exist" in out
    code, out = _run(["erase", "--user", "aditya", "--reason", "request", "--yes"], store, config, capsys)
    report = json.loads(out)
    assert code == 0 and report["sessions"] == ["s1"] and len(report["records"]) == 1
    db = store.read_bytes()
    assert b"Vim for editing" not in db and b"set up an editor" not in db


def test_reembed_refuses_without_recalibrated_floors(
    paths: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    store, config = paths
    _seed(store)
    _run(["extract", "s1"], store, config, capsys)
    _run(["search", "--agent", "agent", "--user", "aditya", "editor"], store, config, capsys)
    code, out = _run(["reembed", "--model", "fake-embedder", "--version", "2"], store, config, capsys)
    assert code == cli.EXIT_REFUSED and "recalibrate" in out
    recalibrated = config.with_name("recalibrated.yaml")
    recalibrated.write_text(
        _CONFIG_YAML + "retrieval:\n  gate:\n    dense_floor:\n      semantic: 0.5\n", encoding="utf-8"
    )
    code, out = _run(["reembed", "--model", "fake-embedder", "--version", "2"], store, recalibrated, capsys)
    assert code == 0 and "re-embedded 1 record(s)" in out


def test_snapshot_save_and_load_round_trip(
    paths: tuple[Path, Path], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    store, config = paths
    _seed(store)
    _run(["extract", "s1"], store, config, capsys)
    snapshot = tmp_path / "snap.sqlite"
    assert _run(["snapshot", "save", str(snapshot)], store, config, capsys)[0] == 0
    _run(["erase", "--user", "aditya", "--reason", "x", "--yes"], store, config, capsys)
    assert "0 record(s)" in _run(["dump", "--scope", "user:aditya"], store, config, capsys)[1]
    code, out = _run(["snapshot", "load", str(snapshot)], store, config, capsys)
    assert code == cli.EXIT_REFUSED
    code, out = _run(["snapshot", "load", str(snapshot), "--yes"], store, config, capsys)
    assert code == 0 and "restored" in out
    assert "1 record(s)" in _run(["dump", "--scope", "user:aditya"], store, config, capsys)[1]
    code, out = _run(["snapshot", "load", str(tmp_path / "missing.sqlite"), "--yes"], store, config, capsys)
    assert code == cli.EXIT_REFUSED


def test_grant_search_scope_follows_the_principal(paths: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    store, config = paths
    _seed(store)
    _run(["extract", "s1"], store, config, capsys)
    code, out = _run(["search", "--agent", "other-agent", "--user", "aditya", "editor"], store, config, capsys)
    assert code == 0 and "no results" in out
    assert _run(["grant", "other-agent", "user:aditya", "--read"], store, config, capsys)[0] == 0
    code, out = _run(
        ["search", "--agent", "other-agent", "--user", "aditya", "--json", "editor", "set", "up"], store, config, capsys
    )
    assert code == 0 and json.loads(out)["results"]


def test_principal_helper_builds_the_principal() -> None:
    import argparse

    args = argparse.Namespace(agent="a", user="u", session="s", project=None)
    assert cli._principal(args) == Principal("a", "u", "s", None)
