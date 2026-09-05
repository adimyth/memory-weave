"""Trusted host operations for provisioning and revoking principal access."""

from __future__ import annotations

from collections.abc import Iterable

from memory_weave.ingest.entities import ensure_principal_entity
from memory_weave.models import PRIVATE_SCOPE_SEPARATOR, EntityStatus, Scope, validate_private_scope_component
from memory_weave.store import Store
from memory_weave.util import normalize_alias

_ACTIVE_ENTITY_STATUSES: tuple[EntityStatus, ...] = ("provisional", "confirmed")


class MemoryHost:
    """Provision grants before a host starts work for an agent and user pair."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def grant(self, agent_id: str, scope: Scope, *, read: bool, write: bool) -> None:
        """Create or replace one explicit scope grant and record the administrative action."""

        validate_private_scope_component(agent_id, "agent_id")
        if scope.kind == "user":
            validate_private_scope_component(scope.id, "user_id")
        if scope.kind == "agent" and PRIVATE_SCOPE_SEPARATOR in scope.id:
            raise ValueError("Private agent scopes are implicit and cannot be granted.")
        self._store.set_grant(agent_id, scope, can_read=read, can_write=write)
        self._store.append_event(
            "grant.changed",
            "host",
            None,
            None,
            {"agent_id": agent_id, "read": read, "scope": {"id": scope.id, "kind": scope.kind}, "write": write},
        )

    def provision_user(self, user_id: str, aliases: Iterable[str] = ()) -> str:
        """Create the principal's person entity in its user scope and attach the names people use for it.

        Registering display names such as "Aditya" or "Aditya Mishra" makes a later ``about`` mention of
        that name resolve to the same entity a subject-less write uses, so facts about the user cannot
        split across two person entities when ``user_id`` is an opaque identifier.
        """

        validate_private_scope_component(user_id, "user_id")
        scope = Scope(kind="user", id=user_id)
        with self._store.transaction():
            entity = ensure_principal_entity(user_id, self._store, actor="host")
            added: list[str] = []
            for alias in aliases:
                normalized = normalize_alias(alias)
                if not normalized or normalized in entity.aliases:
                    continue
                others = [
                    other.id
                    for other in self._store.entities_by_alias(
                        normalized, kinds=["person"], scopes=[scope], statuses=_ACTIVE_ENTITY_STATUSES
                    )
                    if other.id != entity.id
                ]
                if others:
                    raise ValueError(f"Alias {alias!r} already names person entity {others[0]} in user:{user_id}.")
                self._store.add_alias(entity.id, normalized)
                added.append(normalized)
            self._store.append_event(
                "principal.provisioned",
                "host",
                None,
                entity.id,
                {"aliases_added": added, "scope": {"id": scope.id, "kind": scope.kind}, "user_id": user_id},
            )
        return entity.id

    def revoke(self, agent_id: str, scope: Scope) -> None:
        """Remove one explicit scope grant and record the administrative action."""

        self._store.revoke_grant(agent_id, scope)
        self._store.append_event(
            "grant.revoked",
            "host",
            None,
            None,
            {"agent_id": agent_id, "scope": {"id": scope.id, "kind": scope.kind}},
        )
