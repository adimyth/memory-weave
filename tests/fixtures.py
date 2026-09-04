"""Reusable deterministic stores for retrieval and benchmark tests."""

from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from memory_weave.index.embedder import FakeEmbedder
from memory_weave.models import MemoryType, Record, Scope
from memory_weave.store import Store

_FIXTURE_START = datetime(2026, 1, 1, tzinfo=UTC)


def build_store(
    store: Store,
    n: int,
    scopes: Sequence[Scope],
    types: Sequence[MemoryType],
    seed: int,
    embedder: FakeEmbedder,
) -> list[Record]:
    """Fill ``store`` with deterministic records, FTS rows, embeddings, and scoped fixture entities."""

    if n < 0:
        raise ValueError("Fixture size cannot be negative.")
    if not scopes:
        raise ValueError("Fixture stores require at least one scope.")
    if not types:
        raise ValueError("Fixture stores require at least one memory type.")

    generator = random.Random(seed)
    entities = []
    for index, scope in enumerate(scopes):
        entity = store.create_entity(
            kind="project",
            canonical=f"Fixture Project {seed}-{index}",
            scope=scope,
            entity_id=f"fixture-{seed}-entity-{index}",
            created_at=_FIXTURE_START,
        )
        store.add_alias(entity.id, f"fixture project {seed} {index}")
        entities.append(entity)

    records: list[Record] = []
    with store.transaction():
        for index in range(n):
            scope_index = generator.randrange(len(scopes))
            scope = scopes[scope_index]
            memory_type = types[generator.randrange(len(types))]
            event_at = _FIXTURE_START + timedelta(days=index)
            record_id = f"fixture-{seed}-record-{index:06d}"
            content = f"Fixture memory {index} records ERR{index:05d} for project {scope_index}."
            entity = entities[scope_index]
            subject = (
                f"project:fixture-{seed}-{scope_index}/setting_{index}"
                if memory_type != "episodic"
                else f"project:fixture-{seed}-{scope_index}/-"
            )
            record = Record(
                id=record_id,
                type=memory_type,
                version=1,
                content=content,
                subject=subject,
                scope=scope,
                source_kind="user_statement",
                source_ref=None,
                creator_agent_id="fixture-agent",
                evidence=None,
                created_at=event_at,
                event_at=event_at,
                expires_at=None,
                confidence=0.95,
                status="confirmed",
                supersedes_id=None,
                reinforcements=0,
                last_reinforced_at=None,
                tags=[],
                entity_ids=[entity.id],
            )
            store.insert_record(record)
            store.put_embedding(record.id, embedder.name, embedder.version, embedder.embed_documents([content])[0])
            store.upsert_fts(record.id, record.content, record.subject, f"fixture project {seed} {scope_index}")
            store.link_record_entity(record.id, entity.id)
            records.append(record)
    return records
