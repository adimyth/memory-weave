"""Scope access rules shared by ingestion and retrieval."""

from __future__ import annotations

from memory_weave.models import PRIVATE_SCOPE_SEPARATOR, Scope, validate_private_scope_component
from memory_weave.store import Store


def readable_scopes(store: Store, agent_id: str, user_id: str) -> list[Scope]:
    """Return the agent-and-user private scope and every explicitly readable grant."""

    return _allowed_scopes(store, agent_id, user_id, can_read=True)


def writable_scopes(store: Store, agent_id: str, user_id: str) -> list[Scope]:
    """Return the agent-and-user private scope and every explicitly writable grant."""

    return _allowed_scopes(store, agent_id, user_id, can_read=False)


def _allowed_scopes(store: Store, agent_id: str, user_id: str, *, can_read: bool) -> list[Scope]:
    grants = store.grants_for(agent_id, can_read=True) if can_read else store.grants_for(agent_id, can_write=True)
    visible = [private_scope(agent_id, user_id)]
    visible.extend(scope for scope in grants if _grant_is_visible(scope, user_id))
    return list(dict.fromkeys(visible))


def private_scope(agent_id: str, user_id: str) -> Scope:
    """Return the implicit private scope for one principal pair."""

    validate_private_scope_component(agent_id, "agent_id")
    validate_private_scope_component(user_id, "user_id")
    return Scope(kind="agent", id=f"{agent_id}{PRIVATE_SCOPE_SEPARATOR}{user_id}")


def _grant_is_visible(scope: Scope, user_id: str) -> bool:
    if scope.kind == "user":
        return scope.id == user_id
    if scope.kind == "agent" and PRIVATE_SCOPE_SEPARATOR in scope.id:
        _, _, private_user_id = scope.id.rpartition(PRIVATE_SCOPE_SEPARATOR)
        return private_user_id == user_id
    return True
