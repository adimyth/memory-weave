from __future__ import annotations

from pathlib import Path

import pytest

from memory_weave.host import MemoryHost
from memory_weave.models import Scope
from memory_weave.policy import readable_scopes, writable_scopes
from memory_weave.store import Store


def test_host_grant_and_revoke_record_the_administrative_actions(tmp_path: Path) -> None:
    store = Store(tmp_path / "memory.sqlite")
    scope = Scope(kind="agent", id="operations-agent")
    host = MemoryHost(store)

    host.grant("research-agent", scope, read=True, write=True)

    assert scope in readable_scopes(store, "research-agent", "aditya")
    assert scope in writable_scopes(store, "research-agent", "someone-else")
    assert store.connection.execute("SELECT kind FROM events WHERE kind = 'grant.changed'").fetchone() is not None

    host.revoke("research-agent", scope)

    assert scope not in readable_scopes(store, "research-agent", "aditya")
    assert store.connection.execute("SELECT kind FROM events WHERE kind = 'grant.revoked'").fetchone() is not None
    store.close()


@pytest.mark.parametrize(
    ("agent_id", "scope"),
    [
        ("research/agent", Scope(kind="project", id="memory-weave")),
        ("research-agent", Scope(kind="user", id="aditya/test")),
        ("research-agent", Scope(kind="agent", id="research-agent/aditya")),
    ],
)
def test_host_rejects_ambiguous_or_private_scope_grants(tmp_path: Path, agent_id: str, scope: Scope) -> None:
    store = Store(tmp_path / "memory.sqlite")
    host = MemoryHost(store)

    with pytest.raises(ValueError):
        host.grant(agent_id, scope, read=True, write=True)

    assert store.connection.execute("SELECT COUNT(*) FROM grants").fetchone()[0] == 0
    store.close()
