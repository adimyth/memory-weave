"""Scope access rules shared by ingestion and retrieval."""

from __future__ import annotations

from memory_weave.models import Scope
from memory_weave.store import Store


def readable_scopes(store: Store, agent_id: str, user_id: str, project_id: str | None) -> list[Scope]:
    """Return the scopes a principal may read without crossing a user boundary."""

    del project_id
    return _allowed_scopes(store, agent_id, user_id, can_read=True)


def writable_scopes(store: Store, agent_id: str, user_id: str, project_id: str | None) -> list[Scope]:
    """Return the scopes a principal may write without crossing a user boundary."""

    del project_id
    return _allowed_scopes(store, agent_id, user_id, can_read=False)


def _allowed_scopes(store: Store, agent_id: str, user_id: str, *, can_read: bool) -> list[Scope]:
    grants = store.grants_for(agent_id, can_read=True) if can_read else store.grants_for(agent_id, can_write=True)
    visible = [Scope(kind="agent", id=agent_id)]
    visible.extend(scope for scope in grants if scope.kind != "user" or scope.id == user_id)
    return list(dict.fromkeys(visible))
