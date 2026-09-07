"""Review-queue CLI: list, resolve, backlog, and direct activation, all over the audited operations."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from memory_weave import cli
from memory_weave.host import MemoryHost
from memory_weave.models import Principal, Record, Scope, Turn
from memory_weave.policy import ActivationService, CategoryDecision
from memory_weave.store import Store
from memory_weave.util import render_subject

_AT = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


class _Policy:
    def classify(self, content: str) -> CategoryDecision:
        return CategoryDecision("preferences", "answer_style", "ambiguous", 0.6)


def _seed(path: Path) -> tuple[str, str]:
    store = Store(path)
    host = MemoryHost(store)
    host.grant("agent", Scope(kind="user", id="user-1"), read=True, write=True)
    entity = host.provision_user("user-1", aliases=("Dev",))
    principal = Principal("agent", "user-1", "s1", None)
    store.create_session("s1", "agent", "user-1", None, _AT)
    text = "User likes it terse, sometimes."
    store.append_turn(Turn("s1", 1, "user", text, _AT))
    store.insert_record(
        Record(
            id="r1", type="semantic", version=1, content=text, subject=render_subject(entity, "r1"),
            scope=Scope(kind="user", id="user-1"), source_kind="user_statement", source_ref=None, creator_agent_id="agent",
            evidence=text, created_at=_AT, event_at=_AT, expires_at=None, confidence=0.9, status="provisional",
            supersedes_id=None, reinforcements=0, last_reinforced_at=None, tags=[], entity_ids=[entity],
            subject_entity_id=entity, attribute="r1",
        )
    )
    decision = ActivationService(store, _Policy()).apply(principal, "r1")
    store.close()
    assert decision.review_id is not None
    return decision.review_id, "r1"


def test_reviews_list_resolve_and_backlog(tmp_path: Path, capsys) -> None:
    db = tmp_path / "m.sqlite"
    review_id, record_id = _seed(db)

    assert cli.main(["--store", str(db), "reviews", "list"]) == 0
    out = capsys.readouterr().out
    assert review_id in out and "ambiguous_applicability" in out and "User likes it terse" in out

    assert cli.main(["--store", str(db), "reviews", "backlog", "--max-open", "0", "--max-age-days", "30"]) == 3

    assert cli.main(["--store", str(db), "reviews", "resolve", review_id, "--as", "promote", "--resolver", "ops-1"]) == 0
    store = Store(db)
    assert store.get_record(record_id).activation == "ambient"  # type: ignore[union-attr]
    assert store.open_activation_reviews() == []
    kinds = [e["kind"] for e in store.events_for(record_id)]
    assert "record.activation_reviewed" in kinds and "record.activation_changed" in kinds
    store.close()

    assert cli.main(["--store", str(db), "reviews", "backlog", "--max-open", "0", "--max-age-days", "30"]) == 0
    assert cli.main(["--store", str(db), "reviews", "resolve", review_id, "--as", "reject", "--resolver", "ops-1"]) == 2
    assert "refused" in capsys.readouterr().out


def test_direct_activation_is_checked(tmp_path: Path, capsys) -> None:
    db = tmp_path / "m.sqlite"
    _, record_id = _seed(db)
    assert cli.main(["--store", str(db), "activation", record_id, "ambient", "--resolver", "ops-1", "--reason", "manual"]) == 0
    assert cli.main(["--store", str(db), "activation", record_id, "conditional", "--resolver", "ops-1", "--reason", "demote"]) == 0
    assert cli.main(["--store", str(db), "activation", "missing", "ambient", "--resolver", "ops-1", "--reason", "x"]) == 2
    assert "refused" in capsys.readouterr().out


def test_metrics_and_bundle_commands(tmp_path: Path, capsys) -> None:
    import json

    from memory_weave.policy import UtilityAwareConfig, UtilityAwareOrchestrator, bundle_components

    db = tmp_path / "m.sqlite"
    _seed(db)
    store = Store(db)
    principal = Principal("agent", "user-1", "s1", None)
    shadow = UtilityAwareConfig(gap_enabled=True, admission_mode="hosted_judge", shadow=True, bundle={"planner": "p/1"})

    class Gaps:
        def plan(self, turn, public_context, ambient_profile, inventory):
            from memory_weave.policy import GapDecision

            return GapDecision([], "fake", "empty")

    UtilityAwareOrchestrator(store, lambda p, q, c: [], Gaps(), None, shadow).prepare_turn(principal, "hello", None, lambda: "draft", lambda r: "final")
    store.close()

    assert cli.main(["--store", str(db), "metrics"]) == 0
    out = capsys.readouterr().out
    assert "planner_silence=1" in out and "turns=1" in out
    assert cli.main(["--store", str(db), "metrics", "--json", "--rollback-check"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stages"]["planner_silence"] == 1 and payload["rollback_reasons"] == []

    components = tmp_path / "bundle.json"
    components.write_text(json.dumps(bundle_components(shadow)))
    assert cli.main(["--store", str(db), "bundles", "list"]) == 0
    assert "no fitness results" in capsys.readouterr().out
    assert cli.main(["--store", str(db), "bundles", "record", str(components), "--passed", "--evidence", "results/x", "--by", "ops"]) == 0
    assert "PASS" in capsys.readouterr().out
    assert cli.main(["--store", str(db), "bundles", "record", str(components), "--evidence", "x", "--by", "ops"]) == 2
    assert cli.main(["--store", str(db), "bundles", "list"]) == 0
    assert "PASS" in capsys.readouterr().out
