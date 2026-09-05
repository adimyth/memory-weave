from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memory_weave.config import IngestionConfig, MemoryWeaveConfig
from memory_weave.models import Record, Scope, SourceKind
from memory_weave.policy.grants import readable_scopes, writable_scopes
from memory_weave.policy.lifecycle import (
    has_authority,
    initial_confidence,
    initial_expiry,
    initial_status,
    reinforce,
)
from memory_weave.store import Store

_NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
_AGENT_ID = "implementation-agent"
_USER_ID = "aditya"
_CONFIG = MemoryWeaveConfig()


@pytest.fixture
def store(tmp_path: Path) -> Store:
    database = Store(tmp_path / "memory.sqlite")
    _ = database.connection
    yield database
    database.close()


def _record(record_id: str, source_kind: SourceKind, **overrides: object) -> Record:
    values: dict[str, object] = {
        "id": record_id,
        "type": "semantic",
        "version": 1,
        "content": "Aditya prefers concise technical explanations.",
        "subject": "person:aditya/explanation_style",
        "scope": Scope(kind="user", id=_USER_ID),
        "source_kind": source_kind,
        "source_ref": "session:session-1<turn:1>",
        "creator_agent_id": _AGENT_ID,
        "evidence": "Keep answers concise.",
        "created_at": _NOW,
        "event_at": _NOW,
        "expires_at": None,
        "confidence": 0.95,
        "status": "confirmed",
        "supersedes_id": None,
        "reinforcements": 0,
        "last_reinforced_at": None,
        "tags": [],
        "entity_ids": [],
    }
    values.update(overrides)
    return Record(**values)  # type: ignore[arg-type]


def test_grants_add_to_the_implicit_agent_and_user_private_scope(store: Store) -> None:
    own_agent_scope = Scope(kind="agent", id=f"{_AGENT_ID}/{_USER_ID}")
    own_user_scope = Scope(kind="user", id=_USER_ID)
    another_user_scope = Scope(kind="user", id="someone-else")
    project_scope = Scope(kind="project", id="memory-weave")
    store.set_grant(_AGENT_ID, own_user_scope, can_read=True, can_write=True)
    store.set_grant(_AGENT_ID, another_user_scope, can_read=True, can_write=True)
    store.set_grant(_AGENT_ID, project_scope, can_read=True, can_write=True)

    readable = readable_scopes(store, _AGENT_ID, _USER_ID)
    writable = writable_scopes(store, _AGENT_ID, _USER_ID)

    assert own_agent_scope in readable
    assert own_agent_scope in writable
    assert own_user_scope in readable
    assert own_user_scope in writable
    assert another_user_scope not in readable
    assert another_user_scope not in writable
    assert project_scope in readable
    assert project_scope in writable


def test_project_grant_does_not_create_user_scope_access(store: Store) -> None:
    project_scope = Scope(kind="project", id="memory-weave")
    user_scope = Scope(kind="user", id=_USER_ID)
    store.set_grant(_AGENT_ID, project_scope, can_read=True, can_write=True)

    assert readable_scopes(store, _AGENT_ID, _USER_ID) == [
        Scope(kind="agent", id=f"{_AGENT_ID}/{_USER_ID}"),
        project_scope,
    ]
    assert user_scope not in writable_scopes(store, _AGENT_ID, _USER_ID)


def test_a_granted_private_scope_is_visible_only_to_its_encoded_user(store: Store) -> None:
    private_scope = Scope(kind="agent", id="agent-a/user-u")
    store.set_grant("agent-b", private_scope, can_read=True, can_write=True)

    assert private_scope in readable_scopes(store, "agent-b", "user-u")
    assert private_scope in writable_scopes(store, "agent-b", "user-u")
    assert private_scope not in readable_scopes(store, "agent-b", "user-v")
    assert private_scope not in writable_scopes(store, "agent-b", "user-v")


@pytest.mark.parametrize(
    ("source_kind", "status", "confidence", "expires"),
    [
        ("user_statement", "confirmed", 0.95, False),
        ("system", "confirmed", 0.90, False),
        ("tool_result", "confirmed", 0.85, False),
        ("session_summary", "confirmed", 0.80, False),
        ("agent_inference", "provisional", 0.60, True),
    ],
)
def test_initial_lifecycle_values_cover_every_source_kind(
    source_kind: SourceKind, status: str, confidence: float, expires: bool
) -> None:
    config = MemoryWeaveConfig(ingestion=IngestionConfig(provisional_ttl_days=14))

    assert initial_status(source_kind) == status
    assert initial_confidence(source_kind) == confidence
    assert initial_expiry(source_kind, _NOW, config) == (_NOW + timedelta(days=14) if expires else None)


def test_reinforce_refreshes_provisional_expiry_and_confirms_at_the_configured_count() -> None:
    provisional = _record(
        "provisional",
        "agent_inference",
        status="provisional",
        confidence=0.60,
        expires_at=_NOW + timedelta(days=1),
        reinforcements=1,
    )
    config = MemoryWeaveConfig(ingestion=IngestionConfig(provisional_ttl_days=14, reinforcements_to_confirm=3))

    reinforced = reinforce(provisional, _NOW + timedelta(hours=1), config)

    assert reinforced is provisional
    assert reinforced.reinforcements == 2
    assert reinforced.confidence == pytest.approx(0.70)
    assert reinforced.status == "provisional"
    assert reinforced.expires_at == _NOW + timedelta(days=14, hours=1)

    confirm_config = MemoryWeaveConfig(ingestion=IngestionConfig(provisional_ttl_days=14, reinforcements_to_confirm=2))
    confirmed = reinforce(
        _record(
            "confirm",
            "agent_inference",
            status="provisional",
            confidence=0.95,
            expires_at=_NOW + timedelta(days=1),
            reinforcements=1,
        ),
        _NOW,
        confirm_config,
    )

    assert confirmed.reinforcements == 2
    assert confirmed.confidence == 0.99
    assert confirmed.status == "confirmed"
    assert confirmed.expires_at is None
    assert confirmed.last_reinforced_at == _NOW


def test_has_authority_uses_rank_then_event_time_then_write_time() -> None:
    old = _record("old", "tool_result")

    assert has_authority(_record("higher", "user_statement"), old, _CONFIG)
    later_event = _record("later-event", "tool_result", event_at=_NOW + timedelta(seconds=1))
    earlier_event = _record("earlier-event", "tool_result", event_at=_NOW - timedelta(seconds=1))
    later_write = _record("later-write", "tool_result", created_at=_NOW + timedelta(seconds=1))
    earlier_write = _record("earlier-write", "tool_result", created_at=_NOW - timedelta(seconds=1))

    assert has_authority(later_event, old, _CONFIG)
    assert not has_authority(earlier_event, old, _CONFIG)
    assert has_authority(later_write, old, _CONFIG)
    assert not has_authority(earlier_write, old, _CONFIG)
