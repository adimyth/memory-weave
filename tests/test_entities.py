from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from memory_weave.ingest.entities import (
    EntityMergeError,
    EntityNotReadableError,
    EntityNotWritableError,
    EntityResolutionError,
    aliases_text,
    follow_merges,
    merge_entities,
    resolve_entities,
)
from memory_weave.models import EntityMention, Principal, Record, Scope
from memory_weave.store import Store
from memory_weave.util import render_subject

_NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
_AGENT_ID = "research-agent"
_USER_ID = "aditya"
_PRINCIPAL = Principal(agent_id=_AGENT_ID, user_id=_USER_ID, session_id="session-1", project_id="memory-weave")
_AGENT_SCOPE = Scope(kind="agent", id=f"{_AGENT_ID}/{_USER_ID}")
_PROJECT_SCOPE = Scope(kind="project", id="memory-weave")


@pytest.fixture
def store(tmp_path: Path) -> Store:
    database = Store(tmp_path / "memory.sqlite")
    _ = database.connection
    yield database
    database.close()


def _record(record_id: str) -> Record:
    return Record(
        id=record_id,
        type="semantic",
        version=1,
        content="Memory Weave stores durable facts outside the conversation.",
        subject="project:memory-weave/architecture",
        scope=_AGENT_SCOPE,
        source_kind="user_statement",
        source_ref="session:session-1<turn:1>",
        creator_agent_id=_AGENT_ID,
        evidence="Store this as a durable fact.",
        created_at=_NOW,
        event_at=_NOW,
        expires_at=None,
        confidence=0.95,
        status="confirmed",
        supersedes_id=None,
        reinforcements=0,
        last_reinforced_at=None,
        tags=[],
        entity_ids=[],
    )


def test_resolve_entities_returns_the_only_readable_exact_alias(store: Store) -> None:
    entity = store.create_entity(kind="project", canonical="Memory Weave", scope=_AGENT_SCOPE, entity_id="project-1")
    store.add_alias(entity.id, "memory weave")

    resolutions = resolve_entities(
        [EntityMention(kind="project", text="Memory Weave", role="about")],
        _AGENT_SCOPE,
        _PRINCIPAL,
        store,
    )

    assert [
        (resolution.outcome, resolution.entity.id if resolution.entity else None) for resolution in resolutions
    ] == [("resolved", entity.id)]


def test_resolve_entities_creates_a_normalized_provisional_alias_and_audits_it(store: Store) -> None:
    mention = EntityMention(kind="person", text="  Áditya\tMehta  ", role="about")

    resolution = resolve_entities([mention], _AGENT_SCOPE, _PRINCIPAL, store)[0]

    assert resolution.outcome == "created"
    assert resolution.entity is not None
    assert resolution.entity.canonical == "Áditya Mehta"
    assert resolution.entity.scope == _AGENT_SCOPE
    assert resolution.entity.status == "provisional"
    assert resolution.entity.aliases == ["aditya mehta"]
    event = store.connection.execute(
        "SELECT actor, entity_id, payload FROM events WHERE kind = 'entity.created'"
    ).fetchone()
    assert event is not None
    assert tuple(event[:2]) == (_AGENT_ID, resolution.entity.id)
    assert json.loads(event["payload"])["alias"] == "aditya mehta"
    assert json.loads(event["payload"])["canonical"] == "Áditya Mehta"
    assert aliases_text(store, [resolution.entity.id]) == "aditya mehta"


def test_resolve_entities_returns_ambiguity_without_creating_or_selecting_an_entity(store: Store) -> None:
    first = store.create_entity(kind="repo", canonical="API", scope=_AGENT_SCOPE, entity_id="repo-agent")
    second = store.create_entity(kind="repo", canonical="API", scope=_PROJECT_SCOPE, entity_id="repo-project")
    store.add_alias(first.id, "api")
    store.add_alias(second.id, "api")
    store.set_grant(_AGENT_ID, _PROJECT_SCOPE, can_read=True, can_write=False)

    resolution = resolve_entities(
        [EntityMention(kind="repo", text="API", role="mentions")],
        _AGENT_SCOPE,
        _PRINCIPAL,
        store,
    )[0]

    assert resolution.outcome == "ambiguous"
    assert resolution.entity is None
    assert resolution.candidates == [first.id, second.id]
    assert store.connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 2
    assert (
        store.connection.execute("SELECT COUNT(*) FROM events WHERE kind = 'entity.ambiguous_alias'").fetchone()[0] == 0
    )


def test_explicit_entity_id_must_be_readable(store: Store) -> None:
    private = store.create_entity(kind="person", canonical="Someone Else", scope=Scope(kind="user", id="someone-else"))

    with pytest.raises(EntityNotReadableError, match="entity_not_readable"):
        resolve_entities(
            [EntityMention(kind="person", text="ignored", role="about", entity_id=private.id)],
            _AGENT_SCOPE,
            _PRINCIPAL,
            store,
        )


def test_explicit_entity_id_must_match_the_mention_kind(store: Store) -> None:
    project = store.create_entity(kind="project", canonical="Memory Weave", scope=_AGENT_SCOPE)

    with pytest.raises(EntityResolutionError, match="entity_kind_mismatch"):
        resolve_entities(
            [EntityMention(kind="person", text="Memory Weave", role="about", entity_id=project.id)],
            _AGENT_SCOPE,
            _PRINCIPAL,
            store,
        )


def test_creating_an_entity_requires_write_access_to_its_scope(store: Store) -> None:
    with pytest.raises(EntityNotWritableError, match="entity_not_writable"):
        resolve_entities(
            [EntityMention(kind="project", text="Shared Project", role="about")],
            _PROJECT_SCOPE,
            _PRINCIPAL,
            store,
        )

    assert store.connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0


def test_resolution_creations_roll_back_with_the_callers_transaction(store: Store) -> None:
    with pytest.raises(RuntimeError, match="abort"):
        with store.transaction():
            resolve_entities(
                [EntityMention(kind="project", text="Memory Weave", role="about")],
                _AGENT_SCOPE,
                _PRINCIPAL,
                store,
            )
            raise RuntimeError("abort")

    assert store.connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0
    assert store.connection.execute("SELECT COUNT(*) FROM events WHERE kind = 'entity.created'").fetchone()[0] == 0


def test_two_unknown_mentions_of_one_alias_create_only_one_entity(store: Store) -> None:
    resolutions = resolve_entities(
        [
            EntityMention(kind="project", text="Memory Weave", role="about"),
            EntityMention(kind="project", text="  memory\tweave  ", role="mentions"),
        ],
        _AGENT_SCOPE,
        _PRINCIPAL,
        store,
    )

    assert [resolution.outcome for resolution in resolutions] == ["created", "resolved"]
    assert resolutions[0].entity is not None
    assert resolutions[1].entity is not None
    assert resolutions[0].entity.id == resolutions[1].entity.id
    assert store.connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 1


def test_explicit_resolution_follows_a_merge_chain(store: Store) -> None:
    source = store.create_entity(kind="project", canonical="Old Name", scope=_AGENT_SCOPE, entity_id="project-old")
    destination = store.create_entity(
        kind="project", canonical="Current Name", scope=_AGENT_SCOPE, entity_id="project-current"
    )

    merged = merge_entities(source.id, destination.id, _PRINCIPAL, "renamed project", store)
    resolution = resolve_entities(
        [EntityMention(kind="project", text="Old Name", role="about", entity_id=source.id)],
        _AGENT_SCOPE,
        _PRINCIPAL,
        store,
    )[0]

    assert merged.id == destination.id
    assert resolution.outcome == "explicit"
    assert resolution.entity is not None
    assert resolution.entity.id == destination.id


def test_follow_merges_rejects_a_corrupt_cycle(store: Store) -> None:
    first = store.create_entity(kind="project", canonical="First", scope=_AGENT_SCOPE)
    second = store.create_entity(kind="project", canonical="Second", scope=_AGENT_SCOPE)
    store.connection.execute(
        "UPDATE entities SET status = 'merged', merged_into = ? WHERE id = ?", (second.id, first.id)
    )
    store.connection.execute(
        "UPDATE entities SET status = 'merged', merged_into = ? WHERE id = ?", (first.id, second.id)
    )
    corrupt = store.get_entity(first.id)
    assert corrupt is not None

    with pytest.raises(EntityMergeError, match="merge cycle"):
        follow_merges(corrupt, store)


def test_merge_entities_unions_aliases_repoints_links_and_keeps_the_primary_role(store: Store) -> None:
    source = store.create_entity(
        kind="project", canonical="Memory Weave", scope=_AGENT_SCOPE, entity_id="project-source"
    )
    destination = store.create_entity(
        kind="project", canonical="Memory Weave Core", scope=_AGENT_SCOPE, entity_id="project-destination"
    )
    store.add_alias(source.id, "memory weave")
    store.add_alias(destination.id, "memory weave core")
    record = _record("record-1")
    record.subject_entity_id = source.id
    record.attribute = "architecture"
    record.subject = render_subject(source.id, record.attribute)
    store.insert_record(record)
    store.upsert_fts(record.id, record.content, record.subject, "memory weave")
    store.link_record_entity(record.id, source.id, role="about")
    store.link_record_entity(record.id, destination.id, role="mentions")

    merged = merge_entities(source.id, destination.id, _PRINCIPAL, "same project", store)

    stored_source = store.get_entity(source.id)
    assert stored_source is not None
    assert stored_source.status == "merged"
    assert stored_source.merged_into == destination.id
    assert merged.aliases == ["memory weave", "memory weave core"]
    assert store.records_for_entities([destination.id], {record.id}, limit=10) == [(record.id, destination.id)]
    rewritten = store.get_record(record.id)
    assert rewritten is not None
    assert rewritten.subject_entity_id == destination.id
    assert rewritten.subject == f"{destination.id}/architecture"
    assert store.active_by_subject(_AGENT_SCOPE, destination.id, "architecture") == [rewritten]
    assert store.fts_rows([record.id])[record.id][1] == f"{destination.id}/architecture"
    role = store.connection.execute(
        "SELECT role FROM record_entities WHERE record_id = ? AND entity_id = ?",
        (record.id, destination.id),
    ).fetchone()["role"]
    assert role == "about"
    event = store.connection.execute(
        "SELECT actor, entity_id, payload FROM events WHERE kind = 'entity.merged'"
    ).fetchone()
    assert event is not None
    assert tuple(event[:2]) == (_AGENT_ID, source.id)
    assert json.loads(event["payload"]) == {"merged_into": destination.id, "reason": "same project"}
    assert aliases_text(store, [source.id, destination.id]) == "memory weave core memory weave"


def test_merge_entities_rejects_different_entity_kinds(store: Store) -> None:
    person = store.create_entity(kind="person", canonical="Aditya", scope=_AGENT_SCOPE)
    project = store.create_entity(kind="project", canonical="Memory Weave", scope=_AGENT_SCOPE)

    with pytest.raises(EntityMergeError, match="Entity kinds must match"):
        merge_entities(person.id, project.id, _PRINCIPAL, "incorrect target", store)

    stored_person = store.get_entity(person.id)
    stored_project = store.get_entity(project.id)
    assert stored_person is not None
    assert stored_project is not None
    assert stored_person.status == "provisional"
    assert stored_project.status == "provisional"


def test_merge_entities_requires_write_access_to_both_scopes_except_for_admin(store: Store) -> None:
    source = store.create_entity(kind="project", canonical="Personal", scope=_AGENT_SCOPE)
    destination = store.create_entity(kind="project", canonical="Shared", scope=_PROJECT_SCOPE)

    with pytest.raises(EntityNotWritableError, match="entity_not_writable"):
        merge_entities(source.id, destination.id, _PRINCIPAL, "consolidate", store)

    merged = merge_entities(source.id, destination.id, "admin", "consolidate", store)

    assert merged.id == destination.id
    assert store.get_entity(source.id).merged_into == destination.id  # type: ignore[union-attr]
