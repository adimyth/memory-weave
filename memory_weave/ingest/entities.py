"""Resolve entity mentions without guessing across scopes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from memory_weave.models import Entity, EntityMention, EntityStatus, Principal, Resolution, Scope
from memory_weave.policy import readable_scopes, writable_scopes
from memory_weave.store import Store
from memory_weave.util import normalize_alias, normalize_ws

_ACTIVE_ENTITY_STATUSES: tuple[EntityStatus, ...] = ("provisional", "confirmed")
# One message for a missing entity and for an unreadable one, so an error never reveals that an id exists.
_ENTITY_NOT_FOUND = 'entity_not_found'


class EntityResolutionError(ValueError):
    """Raised when a mention cannot be resolved safely."""


class EntityNotFoundError(EntityResolutionError):
    """Raised when an operation names no durable entity."""


class EntityNotReadableError(EntityResolutionError):
    """Raised when an explicit entity is outside the caller's readable scopes."""


class EntityNotWritableError(EntityResolutionError):
    """Raised when an entity operation targets a scope the caller cannot write."""


class EntityMergeError(EntityResolutionError):
    """Raised when an entity merge is invalid or its chain is corrupt."""


class PrincipalEntityAmbiguousError(EntityResolutionError):
    """Raised when more than one person entity in a user scope carries the principal alias."""

    def __init__(self, candidates: list[str]) -> None:
        super().__init__("The principal user alias resolves to more than one person entity.")
        self.candidates = candidates


MergeActor = Principal | Literal["admin"]


def resolve_entities(
    mentions: Sequence[EntityMention],
    scope: Scope,
    principal: Principal,
    store: Store,
) -> list[Resolution]:
    """Resolve exact aliases in readable scopes and create provisional entities for misses."""

    readable = readable_scopes(store, principal.agent_id, principal.user_id)
    resolutions: list[Resolution] = []
    for mention in mentions:
        if mention.entity_id is not None:
            entity = _readable_entity(mention.entity_id, readable, store)
            resolved = follow_merges(entity, store)
            if resolved.kind != mention.kind:
                raise EntityResolutionError(
                    f"entity_kind_mismatch: mention is {mention.kind}, entity {resolved.id} is {resolved.kind}"
                )
            resolutions.append(Resolution(mention, resolved, "explicit", []))
            continue

        alias = normalize_alias(mention.text)
        matches = store.entities_by_alias(
            alias,
            kinds=[mention.kind],
            scopes=readable,
            statuses=_ACTIVE_ENTITY_STATUSES,
        )
        if len(matches) == 1:
            resolutions.append(Resolution(mention, matches[0], "resolved", []))
        elif matches:
            candidate_ids = [entity.id for entity in matches]
            resolutions.append(Resolution(mention, None, "ambiguous", candidate_ids))
        else:
            resolutions.append(_create_entity_resolution(store, principal, mention, scope, alias))
    return resolutions


def follow_merges(entity: Entity, store: Store) -> Entity:
    """Return the surviving entity at the end of one merge chain."""

    seen: set[str] = set()
    current = entity
    while current.merged_into is not None:
        if current.id in seen:
            raise EntityMergeError(f"Entity merge cycle includes {current.id}.")
        seen.add(current.id)
        next_entity = store.get_entity(current.merged_into)
        if next_entity is None:
            raise EntityMergeError(f"Entity {current.id} merges into missing entity {current.merged_into}.")
        current = next_entity
    return current


def merge_entities(
    source_id: str,
    destination_id: str,
    actor: MergeActor,
    reason: str,
    store: Store,
) -> Entity:
    """Merge one entity into another after checking the actor may write both scopes."""

    if source_id == destination_id:
        raise EntityMergeError("An entity cannot be merged into itself.")

    with store.transaction():
        # Authorization runs before any check whose message would describe an entity the actor cannot write.
        source = _entity_or_error(source_id, store)
        destination = follow_merges(_entity_or_error(destination_id, store), store)
        _require_merge_access(actor, source.scope, destination.scope, store)
        if source.status == "merged":
            raise EntityMergeError(f"Entity {source.id} is already merged into {source.merged_into}.")
        if source.kind != destination.kind:
            raise EntityMergeError(f"Entity kinds must match: {source.kind} cannot merge into {destination.kind}.")
        if source.status == "deleted" or destination.status == "deleted":
            raise EntityMergeError("Deleted entities cannot participate in a merge.")
        if source.id == destination.id:
            raise EntityMergeError("An entity cannot be merged into itself.")
        store.merge_entity(source.id, destination.id)
        store.append_event(
            "entity.merged",
            _actor_id(actor),
            None,
            source.id,
            {"merged_into": destination.id, "reason": reason},
        )
        return _entity_or_error(destination.id, store)


def aliases_text(store: Store, entity_ids: Iterable[str]) -> str:
    """Return normalized aliases for linked entities, ready for an FTS aliases column."""

    aliases: dict[str, None] = {}
    for entity_id in entity_ids:
        entity = follow_merges(_entity_or_error(entity_id, store), store)
        aliases.setdefault(normalize_alias(entity.canonical), None)
        for alias in entity.aliases:
            aliases.setdefault(alias, None)
    return " ".join(aliases)


def primary_entity_for(principal: Principal, scope: Scope, store: Store) -> Entity:
    """Resolve or create the principal's person entity in that principal's user scope."""

    if scope.kind != "user" or scope.id != principal.user_id:
        raise EntityResolutionError("A principal user entity can only be created in that user's scope.")
    return ensure_principal_entity(principal.user_id, store, actor=principal.agent_id)


def ensure_principal_entity(user_id: str, store: Store, *, actor: str) -> Entity:
    """Return the person entity that stands for ``user_id`` in its own user scope, creating it if absent."""

    scope = Scope(kind="user", id=user_id)
    alias = normalize_alias(user_id)
    matches = store.entities_by_alias(
        alias,
        kinds=["person"],
        scopes=[scope],
        statuses=_ACTIVE_ENTITY_STATUSES,
    )
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise PrincipalEntityAmbiguousError([entity.id for entity in matches])
    with store.transaction():
        entity = store.create_entity(kind="person", canonical=user_id, scope=scope)
        store.add_alias(entity.id, alias)
        store.append_event(
            "entity.created",
            actor,
            None,
            entity.id,
            {
                "alias": alias,
                "canonical": entity.canonical,
                "kind": entity.kind,
                "scope": {"id": scope.id, "kind": scope.kind},
            },
        )
        return entity


def _create_entity_resolution(
    store: Store,
    principal: Principal,
    mention: EntityMention,
    scope: Scope,
    alias: str,
) -> Resolution:
    writable = writable_scopes(store, principal.agent_id, principal.user_id)
    if scope not in writable:
        raise EntityNotWritableError(f"entity_not_writable: {scope.kind}:{scope.id}")
    with store.transaction():
        canonical = normalize_ws(mention.text)
        entity = store.create_entity(kind=mention.kind, canonical=canonical, scope=scope)
        store.add_alias(entity.id, alias)
        store.append_event(
            "entity.created",
            principal.agent_id,
            None,
            entity.id,
            {
                "alias": alias,
                "canonical": canonical,
                "kind": mention.kind,
                "scope": {"id": scope.id, "kind": scope.kind},
            },
        )
        return Resolution(mention, _entity_or_error(entity.id, store), "created", [])


def _readable_entity(entity_id: str, readable: Sequence[Scope], store: Store) -> Entity:
    entity = _entity_or_error(entity_id, store)
    if entity.scope not in readable:
        raise EntityNotReadableError(_ENTITY_NOT_FOUND)
    return entity


def _entity_or_error(entity_id: str, store: Store) -> Entity:
    entity = store.get_entity(entity_id)
    if entity is None:
        raise EntityNotFoundError(_ENTITY_NOT_FOUND)
    return entity


def _require_merge_access(actor: MergeActor, source_scope: Scope, destination_scope: Scope, store: Store) -> None:
    if actor == "admin":
        return
    writable = writable_scopes(store, actor.agent_id, actor.user_id)
    missing = [scope for scope in (source_scope, destination_scope) if scope not in writable]
    if missing:
        scopes = ", ".join(f"{scope.kind}:{scope.id}" for scope in missing)
        raise EntityNotWritableError(f"entity_not_writable: {scopes}")


def _actor_id(actor: MergeActor) -> str:
    return actor if actor == "admin" else actor.agent_id
