"""Trusted operator maintenance: expiry, transcript retention, erasure, re-embedding, and snapshots.

These are administrative operations over one store. None of them is reachable from a model-facing tool.
Erasure is the one irreversible operation and it is complete by construction: it walks every table that can
hold a user's text, replaces the text with tombstone markers that keep ids, kinds, and timestamps, and then
compacts the database file so freed pages carry no residue.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from memory_weave.config import MemoryWeaveConfig
from memory_weave.index.embedder import Embedder
from memory_weave.ingest import flag_due_reviews
from memory_weave.models import Scope
from memory_weave.policy.grants import private_scope
from memory_weave.store import Store
from memory_weave.util import now

DEFAULT_ACTOR = "admin"


class OperationRefused(RuntimeError):
    """The operation would be unsafe or meaningless as requested; nothing was changed."""


@dataclass(slots=True)
class ErasureReport:
    records: list[str] = field(default_factory=list)
    sessions: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    search_logs: int = 0
    turn_decisions: int = 0
    events: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "sessions": self.sessions,
            "entities": self.entities,
            "search_logs": self.search_logs,
            "turn_decisions": self.turn_decisions,
            "events": self.events,
        }


class Operations:
    """Every method appends the events that make the change auditable."""

    def __init__(
        self,
        store: Store,
        config: MemoryWeaveConfig,
        *,
        actor: str = DEFAULT_ACTOR,
        current_time: Callable[[], datetime] = now,
    ) -> None:
        self._store = store
        self._config = config
        self._actor = actor
        self._current_time = current_time

    # -- expiry and retention ----------------------------------------------------------------------------

    def expire(self) -> list[str]:
        """Mark every active record past its expiry as expired, one event each; retrieval already ignored them."""

        current = self._current_time()
        with self._store.transaction(immediate=True):
            ids = self._store.expire_due(current)
            for record_id in ids:
                self._store.append_event(
                    "record.expired", self._actor, record_id, None, {"reason": "expired", "at": current.isoformat()}
                )
        return ids

    def retain(self) -> list[str]:
        """Blank the transcripts of extracted sessions older than ``sessions.retain_days``; never an unextracted one."""

        current = self._current_time()
        cutoff = current - timedelta(days=self._config.sessions.retain_days)
        blanked: list[str] = []
        with self._store.transaction(immediate=True):
            for session_id in self._store.sessions_due_for_retention(cutoff):
                session = self._store.get_session(session_id)
                if session is None or session.extracted_at is None:
                    continue
                turns = self._store.blank_session_turns(session_id)
                self._store.append_event(
                    "session.retained",
                    self._actor,
                    None,
                    None,
                    {
                        "session_id": session_id,
                        "turns_blanked": turns,
                        "retain_days": self._config.sessions.retain_days,
                    },
                )
                blanked.append(session_id)
        return blanked

    def flag_due_reviews(self) -> list[str]:
        return flag_due_reviews(
            self._store, batch_size=self._config.ingestion.temporal_review_batch_size, at=self._current_time()
        )

    # -- erasure -----------------------------------------------------------------------------------------

    def erase_record(self, record_id: str, *, reason: str) -> ErasureReport:
        """Erase one record's durable content; the tombstone row, its id, and its audit trail remain."""

        report = ErasureReport()
        with self._store.transaction(immediate=True):
            if self._store.get_record(record_id) is None:
                raise OperationRefused(f"record {record_id} does not exist")
            self._store.erase_record(record_id)
            report.records.append(record_id)
            report.events = self._store.erase_event_payloads(record_ids=[record_id])
            self._store.append_event("record.erased", self._actor, record_id, None, {"reason": reason})
        self._store.compact()
        return report

    def erase_session(self, session_id: str, *, reason: str) -> ErasureReport:
        """Blank one transcript, keep the turn rows so source references still resolve to a tombstone."""

        report = ErasureReport()
        with self._store.transaction(immediate=True):
            if self._store.get_session(session_id) is None:
                raise OperationRefused(f"session {session_id} does not exist")
            self._store.blank_session_turns(session_id)
            report.sessions.append(session_id)
            report.turn_decisions = self._store.erase_turn_decisions_for_sessions([session_id])
            report.events = self._store.erase_event_payloads(session_ids=[session_id])
            self._store.append_event(
                "session.erased", self._actor, None, None, {"session_id": session_id, "reason": reason}
            )
        self._store.compact()
        return report

    def erase_user(self, user_id: str, *, reason: str) -> ErasureReport:
        """Remove every place a user's text can reach: transcripts, records, evidence, indexes, logs, events, aliases.

        Covered, in this order: every session for the user; every record in the user scope and in every
        private ``agent:<agent>/<user>`` scope; the content and evidence of any record in any scope whose
        source reference points at one of those sessions; the user's search logs and turn decisions; the
        payload of every event about an erased record, entity, or session; and the entities and aliases in the
        user's scopes. Rows, ids, kinds, and timestamps remain as tombstones.
        """

        report = ErasureReport()
        with self._store.transaction(immediate=True):
            sessions = self._store.sessions_for_user(user_id)
            session_ids = [session.id for session in sessions]
            agent_ids = sorted({session.agent_id for session in sessions})
            scopes = [Scope(kind="user", id=user_id), *[private_scope(agent_id, user_id) for agent_id in agent_ids]]
            record_ids = list(
                dict.fromkeys(
                    [
                        *self._store.record_ids_in_scopes(scopes),
                        *self._store.record_ids_sourced_from_sessions(session_ids),
                    ]
                )
            )
            for session_id in session_ids:
                self._store.blank_session_turns(session_id)
            report.sessions = session_ids
            for record_id in record_ids:
                self._store.erase_record(record_id)
            report.records = record_ids
            entity_ids = self._store.entity_ids_in_scopes(scopes)
            for entity_id in entity_ids:
                self._store.erase_entity(entity_id)
            report.entities = entity_ids
            report.search_logs = self._store.erase_search_logs_for_user(user_id)
            report.turn_decisions = self._store.erase_turn_decisions_for_sessions(session_ids)
            report.events = self._store.erase_event_payloads(
                record_ids=record_ids, entity_ids=entity_ids, session_ids=session_ids
            )
            self._store.append_event(
                "user.erased",
                self._actor,
                None,
                None,
                {"user_id": user_id, "reason": reason, **report.to_dict()},
            )
        self._store.compact()
        return report

    # -- re-embedding and snapshots ----------------------------------------------------------------------

    def reembed(self, embedder: Embedder) -> int:
        """Re-embed every record with ``embedder`` and drop the old vectors; refused until floors were recalibrated.

        The gate floors are calibrated per embedding model. If the latest logged search ran under a different
        embedding version with the same dense floors this configuration still carries, nobody has recalibrated,
        and re-embedding would silently move every score under floors chosen for another model.
        """

        embedding = self._config.embedding
        if (embedder.name, embedder.version, embedder.dims) != (embedding.model, embedding.version, embedding.dims):
            raise OperationRefused("the embedder does not match embedding.model, embedding.version, and embedding.dims")
        flags = self._store.latest_search_flags()
        if flags is not None:
            previous_version = f"{flags.get('embedding_model')}/{flags.get('embedding_version')}"
            current_version = f"{embedding.model}/{embedding.version}"
            previous_floors = ((flags.get("gate") or {}).get("dense_floor")) or {}
            current_floors = self._config.flags()["gate"]["dense_floor"]
            if previous_version != current_version and previous_floors == current_floors:
                raise OperationRefused(
                    f"gate floors are unchanged since searches ran under {previous_version}; "
                    "recalibrate retrieval.gate.dense_floor for the new embedding before re-embedding"
                )
        count = 0
        batch: list[tuple[str, str]] = []

        def flush() -> None:
            nonlocal count
            if not batch:
                return
            vectors = embedder.embed_documents([content for _, content in batch])
            with self._store.transaction():
                for (record_id, _), vector in zip(batch, vectors, strict=True):
                    self._store.put_embedding(record_id, embedder.name, embedder.version, vector)
            count += len(batch)
            batch.clear()

        for record_id, content in list(self._store.iter_records_for_reembedding()):
            batch.append((record_id, content))
            if len(batch) >= embedding.reembed_batch_size:
                flush()
        flush()
        with self._store.transaction():
            removed = self._store.delete_embeddings_except(embedder.name, embedder.version)
            self._store.append_event(
                "store.reembedded",
                self._actor,
                None,
                None,
                {"model": embedder.name, "version": embedder.version, "records": count, "old_vectors_removed": removed},
            )
        return count

    def snapshot_save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._store.append_event("store.snapshot_saved", self._actor, None, None, {"path": str(destination)})
        self._store.compact()
        self._store.snapshot_to(destination)
        return destination

    def snapshot_load(self, path: str | Path, store_path: str | Path) -> Path:
        """Replace the store file with a snapshot. The caller must have closed every connection first."""

        source = Path(path)
        if not source.exists():
            raise OperationRefused(f"snapshot {source} does not exist")
        target = Path(store_path)
        for sidecar in (f"{target}-wal", f"{target}-shm"):
            Path(sidecar).unlink(missing_ok=True)
        shutil.copyfile(source, target)
        return target
