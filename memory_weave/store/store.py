"""Typed SQLite boundary for Memory Weave's durable state."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from memory_weave.models import (
    Entity,
    EntityKind,
    EntityRole,
    EntityStatus,
    MemoryType,
    Record,
    RecordStatus,
    Scope,
    SourceKind,
    Turn,
    TurnRole,
)
from memory_weave.util import normalize_alias, normalize_vector, now, render_subject, uuid7

from .migrations import MigrationIssuesError, MigrationResult, migrate

_ACTIVE_STATUSES: tuple[RecordStatus, ...] = ("provisional", "confirmed")
# Keep every alias lookup far below SQLite's historical 999 bound-parameter limit once scopes are added.
_ALIAS_LOOKUP_BATCH = 400
_SQLITE_SAFE_BINDINGS = 900
_HISTORY_STATUSES: tuple[RecordStatus, ...] = (*_ACTIVE_STATUSES, "superseded", "expired")
_SEARCH_LOG_JSON_COLUMNS = frozenset(
    {
        "request",
        "rewritten_queries",
        "readable_scopes",
        "dense",
        "lexical",
        "entity",
        "fused",
        "freshness",
        "gated_out",
        "deduped_out",
        "reranked",
        "reranked_out",
        "budget_out",
        "returned",
        "explanations",
        "config_flags",
        "timings_ms",
    }
)
_SEARCH_LOG_COLUMNS = (
    "id",
    "at",
    "agent_id",
    "user_id",
    "session_id",
    "trigger",
    "request",
    "context",
    "rewrite_status",
    "rewritten_queries",
    "readable_scopes",
    "dense",
    "lexical",
    "entity",
    "fused",
    "freshness",
    "gated_out",
    "deduped_out",
    "reranked",
    "reranked_out",
    "budget_out",
    "returned",
    "explanations",
    "config_flags",
    "warm",
    "timings_ms",
)


class Store:
    """Own SQLite connections and expose persistence without policy or ranking decisions."""

    def __init__(self, path: str | Path, *, allow_migration_issues: bool = False) -> None:
        self._path = str(path)
        self._local = threading.local()
        self._allow_migration_issues = allow_migration_issues

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the current thread's migrated SQLite connection."""

        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self._path, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            result = migrate(connection)
            if result.unmapped_subject_records and not self._allow_migration_issues:
                connection.close()
                raise MigrationIssuesError(result.unmapped_subject_records)
            self._local.migration_result = result
            self._local.connection = connection
        return cast(sqlite3.Connection, connection)

    def migration_issues(self) -> list[tuple[str, str]]:
        """Return unresolved migration issues as (record_id, issue) pairs."""

        rows = self.connection.execute(
            """
            SELECT issues.record_id, issues.issue
            FROM migration_issues AS issues
            JOIN records ON records.id = issues.record_id
            WHERE records.subject_entity_id IS NULL OR records.attribute IS NULL
            ORDER BY issues.record_id
            """
        ).fetchall()
        return [(cast(str, row["record_id"]), cast(str, row["issue"])) for row in rows]

    def expire_migration_issues(self, actor: str) -> list[str]:
        """Expire every unmapped legacy record so it leaves default retrieval, and clear its issue row."""

        expired: list[str] = []
        with self.transaction() as connection:
            for record_id, issue in self.migration_issues():
                index_version = self._bump_records_version(connection)
                connection.execute(
                    "UPDATE records SET status = 'expired', index_version = ? WHERE id = ?",
                    (index_version, record_id),
                )
                connection.execute("DELETE FROM migration_issues WHERE record_id = ?", (record_id,))
                self.append_event(
                    "record.status",
                    actor,
                    record_id,
                    None,
                    {"status": "expired", "reason": f"migration issue: {issue}"},
                )
                expired.append(record_id)
        return expired

    @property
    def migration_result(self) -> MigrationResult:
        """Return conditions reported by migrations on this thread's connection."""

        _ = self.connection
        return cast(MigrationResult, self._local.migration_result)

    def close(self) -> None:
        """Close the current thread's connection, if it was opened."""

        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            del self._local.connection
        if hasattr(self._local, "transaction_depth"):
            del self._local.transaction_depth
        if hasattr(self._local, "migration_result"):
            del self._local.migration_result

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run store operations atomically and nest inner calls with savepoints."""

        connection = self.connection
        depth = getattr(self._local, "transaction_depth", 0)
        owns_transaction = depth == 0 and not connection.in_transaction
        savepoint = f"memory_weave_transaction_{depth}"
        if owns_transaction:
            connection.execute("BEGIN")
        else:
            connection.execute(f"SAVEPOINT {savepoint}")
        self._local.transaction_depth = depth + 1

        try:
            yield connection
        except BaseException:
            if owns_transaction:
                connection.rollback()
            else:
                connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
        else:
            if owns_transaction:
                connection.commit()
            else:
                connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        finally:
            self._local.transaction_depth = depth

    def insert_record(self, record: Record) -> None:
        """Persist one canonical memory record."""

        with self.transaction() as connection:
            index_version = self._bump_records_version(connection)
            connection.execute(
                """
                INSERT INTO records(
                    id, type, version, content, subject, scope_kind, scope_id, source_kind, source_ref,
                    creator_agent_id, evidence, created_at, event_at, expires_at, confidence, status,
                    supersedes_id, reinforcements, last_reinforced_at, index_version, tags, subject_entity_id, attribute
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.type,
                    record.version,
                    record.content,
                    record.subject,
                    record.scope.kind,
                    record.scope.id,
                    record.source_kind,
                    record.source_ref,
                    record.creator_agent_id,
                    record.evidence,
                    _dump_datetime(record.created_at),
                    _dump_datetime(record.event_at),
                    _dump_datetime(record.expires_at),
                    record.confidence,
                    record.status,
                    record.supersedes_id,
                    record.reinforcements,
                    _dump_datetime(record.last_reinforced_at),
                    index_version,
                    _dump_json(record.tags),
                    record.subject_entity_id,
                    record.attribute,
                ),
            )

    def get_record(self, record_id: str) -> Record | None:
        """Return one record with its linked entity IDs."""

        row = self.connection.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
        if row is None:
            return None
        return self._records_from_rows([row])[0]

    def get_records(self, record_ids: Sequence[str]) -> list[Record]:
        """Return records in the caller's requested ID order, omitting unknown IDs."""

        if not record_ids:
            return []
        placeholders = _placeholders(len(record_ids))
        rows = self.connection.execute(
            f"SELECT * FROM records WHERE id IN ({placeholders})", tuple(record_ids)
        ).fetchall()
        records_by_id = {record.id: record for record in self._records_from_rows(rows)}
        return [records_by_id[record_id] for record_id in record_ids if record_id in records_by_id]

    def update_status(self, record_id: str, status: RecordStatus) -> None:
        """Set a record lifecycle status."""

        with self.transaction() as connection:
            index_version = self._bump_records_version(connection)
            connection.execute(
                "UPDATE records SET status = ?, index_version = ? WHERE id = ?", (status, index_version, record_id)
            )

    def set_supersedes(self, record_id: str, supersedes_id: str | None) -> None:
        """Set the record this row replaces."""

        with self.transaction() as connection:
            index_version = self._bump_records_version(connection)
            connection.execute(
                "UPDATE records SET supersedes_id = ?, index_version = ? WHERE id = ?",
                (supersedes_id, index_version, record_id),
            )

    def reinforce_fields(
        self,
        record_id: str,
        *,
        confidence: float,
        reinforcements: int,
        last_reinforced_at: datetime | None,
        expires_at: datetime | None,
        status: RecordStatus,
        source_kind: SourceKind | None = None,
        source_ref: str | None = None,
        evidence: str | None = None,
    ) -> None:
        """Persist lifecycle fields and an optional stronger evidence provenance during reinforcement."""

        with self.transaction() as connection:
            index_version = self._bump_records_version(connection)
            connection.execute(
                """
                UPDATE records
                SET confidence = ?, reinforcements = ?, last_reinforced_at = ?, expires_at = ?, status = ?,
                    source_kind = COALESCE(?, source_kind), source_ref = COALESCE(?, source_ref),
                    evidence = COALESCE(?, evidence), index_version = ?
                WHERE id = ?
                """,
                (
                    confidence,
                    reinforcements,
                    _dump_datetime(last_reinforced_at),
                    _dump_datetime(expires_at),
                    status,
                    source_kind,
                    source_ref,
                    evidence,
                    index_version,
                    record_id,
                ),
            )

    def active_by_subject(self, scope: Scope, subject_entity_id: str, attribute: str) -> list[Record]:
        """Return active records for one scope and structured current-fact subject."""

        rows = self.connection.execute(
            """
            SELECT * FROM records
            WHERE scope_kind = ? AND scope_id = ? AND subject_entity_id = ? AND attribute = ?
              AND status IN (?, ?)
            ORDER BY event_at DESC, created_at DESC, id DESC
            """,
            (scope.kind, scope.id, subject_entity_id, attribute, *_ACTIVE_STATUSES),
        ).fetchall()
        return self._records_from_rows(rows)

    def active_for_entity(self, scope: Scope, entity_id: str, memory_type: MemoryType) -> list[Record]:
        """Return active current facts for one entity, prioritizing recently reinforced attributes."""

        rows = self.connection.execute(
            """
            SELECT * FROM records
            WHERE scope_kind = ? AND scope_id = ? AND subject_entity_id = ? AND type = ?
              AND attribute IS NOT NULL AND attribute != '-' AND status IN (?, ?)
            ORDER BY COALESCE(last_reinforced_at, created_at) DESC, event_at DESC, created_at DESC, id DESC
            """,
            (scope.kind, scope.id, entity_id, memory_type, *_ACTIVE_STATUSES),
        ).fetchall()
        return self._records_from_rows(rows)

    def eligible_ids(
        self,
        scopes: Sequence[Scope],
        types: Sequence[MemoryType] | None,
        since: datetime | None,
        until: datetime | None,
        include_history: bool,
        current_time: datetime,
    ) -> set[str]:
        """Apply scope, lifecycle, type, and time filters before candidate generation."""

        if not scopes:
            return set()

        scope_predicates = " OR ".join("(scope_kind = ? AND scope_id = ?)" for _ in scopes)
        parameters: list[object] = [part for scope in scopes for part in (scope.kind, scope.id)]
        statuses = _HISTORY_STATUSES if include_history else _ACTIVE_STATUSES
        status_predicates = _placeholders(len(statuses))
        query = [f"SELECT id FROM records WHERE ({scope_predicates})", f"AND status IN ({status_predicates})"]
        parameters.extend(statuses)

        if not include_history:
            query.append("AND (expires_at IS NULL OR expires_at >= ?)")
            parameters.append(_dump_datetime(current_time))
        if types:
            query.append(f"AND type IN ({_placeholders(len(types))})")
            parameters.extend(types)
        if since is not None:
            query.append("AND event_at >= ?")
            parameters.append(_dump_datetime(since))
        if until is not None:
            query.append("AND event_at <= ?")
            parameters.append(_dump_datetime(until))

        rows = self.connection.execute(" ".join(query), tuple(parameters)).fetchall()
        return {cast(str, row["id"]) for row in rows}

    def put_embedding(self, record_id: str, model: str, version: str, vector: np.ndarray) -> None:
        """Persist one normalized float32 embedding as a SQLite BLOB."""

        raw_values = np.asarray(vector, dtype=np.float32).reshape(-1)
        values = normalize_vector(raw_values, raw_values.size)
        with self.transaction() as connection:
            index_version = self._bump_records_version(connection)
            connection.execute(
                """
                INSERT INTO embeddings(record_id, model, version, dims, vector, index_version)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET model = excluded.model, version = excluded.version,
                    dims = excluded.dims, vector = excluded.vector, index_version = excluded.index_version
                """,
                (record_id, model, version, values.size, values.tobytes(), index_version),
            )

    def iter_embeddings(self, model: str, version: str) -> Iterator[tuple[str, np.ndarray]]:
        """Yield compatible normalized embeddings as detached float32 arrays."""

        rows = self.connection.execute(
            "SELECT record_id, dims, vector FROM embeddings WHERE model = ? AND version = ? ORDER BY record_id",
            (model, version),
        )
        for row in rows:
            vector = np.frombuffer(cast(bytes, row["vector"]), dtype=np.float32)
            if vector.size != cast(int, row["dims"]):
                raise ValueError(
                    f"Embedding {row['record_id']} has {vector.size} values but declares {row['dims']} dimensions."
                )
            yield cast(str, row["record_id"]), vector.copy()

    def count_embeddings(self, model: str, version: str) -> int:
        """Return how many embeddings match one model and version."""

        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM embeddings WHERE model = ? AND version = ?", (model, version)
        ).fetchone()
        return cast(int, row["count"])

    def records_version(self) -> int:
        """Return the monotonic version changed by record and embedding writes."""

        row = self.connection.execute("SELECT value FROM store_meta WHERE key = 'records_version'").fetchone()
        if row is None:
            raise RuntimeError("store_meta is missing the records_version row.")
        return cast(int, row["value"])

    def index_changes_since(
        self, index_version: int, model: str, embedding_version: str
    ) -> list[tuple[str, RecordStatus, np.ndarray | None]]:
        """Return records whose vector or lifecycle changed after one index projection version."""

        rows = self.connection.execute(
            """
            WITH changed_ids AS (
                SELECT id AS record_id FROM records WHERE index_version > ?
                UNION
                SELECT record_id FROM embeddings WHERE index_version > ?
            )
            SELECT records.id, records.status, embeddings.dims, embeddings.vector
            FROM changed_ids
            JOIN records ON records.id = changed_ids.record_id
            LEFT JOIN embeddings ON embeddings.record_id = records.id
              AND embeddings.model = ? AND embeddings.version = ?
            ORDER BY records.id
            """,
            (index_version, index_version, model, embedding_version),
        ).fetchall()
        changes: list[tuple[str, RecordStatus, np.ndarray | None]] = []
        for row in rows:
            raw = cast(bytes | None, row["vector"])
            vector = None if raw is None else np.frombuffer(raw, dtype=np.float32).copy()
            if vector is not None and vector.size != cast(int, row["dims"]):
                raise ValueError(
                    f"Embedding {row['id']} has {vector.size} values but declares {row['dims']} dimensions."
                )
            changes.append((cast(str, row["id"]), cast(RecordStatus, row["status"]), vector))
        return changes

    def upsert_fts(self, record_id: str, content: str, subject: str, aliases: str) -> None:
        """Replace one derived FTS5 row."""

        with self.transaction() as connection:
            connection.execute("DELETE FROM records_fts WHERE record_id = ?", (record_id,))
            connection.execute(
                "INSERT INTO records_fts(record_id, content, subject, aliases) VALUES (?, ?, ?, ?)",
                (record_id, content, subject, aliases),
            )

    def delete_fts(self, record_id: str) -> None:
        """Remove a record from lexical retrieval."""

        self.connection.execute("DELETE FROM records_fts WHERE record_id = ?", (record_id,))

    def fts_query(self, match_expr: str, limit: int, eligible: set[str] | None = None) -> list[tuple[str, float]]:
        """Return sanitized FTS5 matches, applying an optional eligibility set before the limit."""

        if limit <= 0 or eligible == set():
            return []
        if eligible is None:
            rows = self.connection.execute(
                """
                SELECT record_id, bm25(records_fts, 0.0, 1.0, 2.0, 3.0) AS score
                FROM records_fts
                WHERE records_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (match_expr, limit),
            ).fetchall()
        else:
            with self.transaction() as connection:
                connection.execute(
                    "CREATE TEMP TABLE IF NOT EXISTS memory_weave_eligible_fts (record_id TEXT PRIMARY KEY)"
                )
                connection.execute("DELETE FROM memory_weave_eligible_fts")
                connection.executemany(
                    "INSERT INTO memory_weave_eligible_fts(record_id) VALUES (?)",
                    ((record_id,) for record_id in eligible),
                )
                rows = connection.execute(
                    """
                    SELECT records_fts.record_id, bm25(records_fts, 0.0, 1.0, 2.0, 3.0) AS score
                    FROM records_fts
                    JOIN memory_weave_eligible_fts AS eligible ON eligible.record_id = records_fts.record_id
                    WHERE records_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (match_expr, limit),
                ).fetchall()
                connection.execute("DELETE FROM memory_weave_eligible_fts")
        return [(cast(str, row["record_id"]), cast(float, row["score"])) for row in rows]

    def fts_rows(self, record_ids: Sequence[str]) -> dict[str, tuple[str, str, str]]:
        """Return the indexed content, subject, and alias text for the requested FTS rows."""

        if not record_ids:
            return {}
        placeholders = _placeholders(len(record_ids))
        rows = self.connection.execute(
            f"SELECT record_id, content, subject, aliases FROM records_fts WHERE record_id IN ({placeholders})",
            tuple(record_ids),
        ).fetchall()
        return {
            cast(str, row["record_id"]): (
                cast(str, row["content"]),
                cast(str, row["subject"]),
                cast(str, row["aliases"]),
            )
            for row in rows
        }

    def add_conflict(self, first_id: str, second_id: str, noted_at: datetime | None = None) -> None:
        """Record a symmetric conflict relationship."""

        timestamp = _dump_datetime(noted_at or now())
        with self.transaction() as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO record_conflicts(record_id, other_id, noted_at) VALUES (?, ?, ?)",
                ((first_id, second_id, timestamp), (second_id, first_id, timestamp)),
            )

    def conflicts_for(self, record_id: str) -> list[str]:
        """Return conflicting record IDs in deterministic order."""

        rows = self.connection.execute(
            "SELECT other_id FROM record_conflicts WHERE record_id = ? ORDER BY other_id", (record_id,)
        ).fetchall()
        return [cast(str, row["other_id"]) for row in rows]

    def create_entity(
        self,
        *,
        kind: EntityKind,
        canonical: str,
        scope: Scope,
        status: EntityStatus = "provisional",
        entity_id: str | None = None,
        created_at: datetime | None = None,
        merged_into: str | None = None,
    ) -> Entity:
        """Create and return a durable entity row."""

        resolved_id = entity_id or uuid7()
        self.connection.execute(
            """
            INSERT INTO entities(id, kind, canonical, scope_kind, scope_id, status, merged_into, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_id,
                kind,
                canonical,
                scope.kind,
                scope.id,
                status,
                merged_into,
                _dump_datetime(created_at or now()),
            ),
        )
        entity = self.get_entity(resolved_id)
        assert entity is not None
        return entity

    def get_entity(self, entity_id: str) -> Entity | None:
        """Return one entity and all of its normalized aliases."""

        row = self.connection.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        if row is None:
            return None
        return self._entity_from_row(row)

    def add_alias(self, entity_id: str, alias_norm: str) -> None:
        """Attach a normalized alias to an entity."""

        self.connection.execute(
            "INSERT OR IGNORE INTO entity_aliases(entity_id, alias_norm) VALUES (?, ?)", (entity_id, alias_norm)
        )

    def entities_by_alias(
        self,
        alias_norm: str,
        *,
        kinds: Sequence[EntityKind] | None = None,
        scopes: Sequence[Scope] | None = None,
        statuses: Sequence[EntityStatus] | None = None,
    ) -> list[Entity]:
        """Find entities with an exact alias, constrained by optional scope and lifecycle filters."""

        query = [
            "SELECT DISTINCT e.* FROM entities AS e JOIN entity_aliases AS a ON a.entity_id = e.id",
            "WHERE a.alias_norm = ?",
        ]
        parameters: list[object] = [alias_norm]
        if kinds:
            query.append(f"AND e.kind IN ({_placeholders(len(kinds))})")
            parameters.extend(kinds)
        if scopes is not None:
            if not scopes:
                return []
            scope_predicates = " OR ".join("(e.scope_kind = ? AND e.scope_id = ?)" for _ in scopes)
            query.append(f"AND ({scope_predicates})")
            parameters.extend(part for scope in scopes for part in (scope.kind, scope.id))
        if statuses:
            query.append(f"AND e.status IN ({_placeholders(len(statuses))})")
            parameters.extend(statuses)
        query.append("ORDER BY e.id")
        rows = self.connection.execute(" ".join(query), tuple(parameters)).fetchall()
        return [self._entity_from_row(row) for row in rows]

    def entities_by_aliases(
        self,
        aliases: Sequence[str],
        *,
        kinds: Sequence[EntityKind] | None = None,
        scopes: Sequence[Scope] | None = None,
        statuses: Sequence[EntityStatus] | None = None,
    ) -> dict[str, list[Entity]]:
        """Find entities for multiple exact aliases with one database query."""

        if not aliases or scopes == []:
            return {}
        distinct_aliases = list(dict.fromkeys(aliases))
        filter_bindings = len(kinds or ()) + 2 * len(scopes or ()) + len(statuses or ())
        batch_size = min(_ALIAS_LOOKUP_BATCH, max(1, _SQLITE_SAFE_BINDINGS - filter_bindings))
        if len(distinct_aliases) > batch_size:
            batched: dict[str, list[Entity]] = {}
            for start in range(0, len(distinct_aliases), batch_size):
                batched.update(
                    self.entities_by_aliases(
                        distinct_aliases[start : start + batch_size],
                        kinds=kinds,
                        scopes=scopes,
                        statuses=statuses,
                    )
                )
            return batched
        aliases = distinct_aliases
        query = [
            "SELECT a.alias_norm AS lookup_alias, e.id, e.kind, e.canonical, e.scope_kind, e.scope_id,",
            "e.status, e.merged_into, e.created_at, GROUP_CONCAT(all_aliases.alias_norm) AS aliases",
            "FROM entity_aliases AS a",
            "JOIN entities AS e ON e.id = a.entity_id",
            "JOIN entity_aliases AS all_aliases ON all_aliases.entity_id = e.id",
            f"WHERE a.alias_norm IN ({_placeholders(len(aliases))})",
        ]
        parameters: list[object] = list(aliases)
        if kinds:
            query.append(f"AND e.kind IN ({_placeholders(len(kinds))})")
            parameters.extend(kinds)
        if scopes is not None:
            scope_predicates = " OR ".join("(e.scope_kind = ? AND e.scope_id = ?)" for _ in scopes)
            query.append(f"AND ({scope_predicates})")
            parameters.extend(part for scope in scopes for part in (scope.kind, scope.id))
        if statuses:
            query.append(f"AND e.status IN ({_placeholders(len(statuses))})")
            parameters.extend(statuses)
        query.append("GROUP BY a.alias_norm, e.id ORDER BY a.alias_norm, e.id")
        rows = self.connection.execute(" ".join(query), tuple(parameters)).fetchall()
        result: dict[str, list[Entity]] = {}
        for row in rows:
            alias = cast(str, row["lookup_alias"])
            entity = Entity(
                id=cast(str, row["id"]),
                kind=cast(EntityKind, row["kind"]),
                canonical=cast(str, row["canonical"]),
                scope=Scope(kind=cast(Any, row["scope_kind"]), id=cast(str, row["scope_id"])),
                status=cast(EntityStatus, row["status"]),
                merged_into=cast(str | None, row["merged_into"]),
                aliases=sorted(cast(str, row["aliases"]).split(",")),
                created_at=_load_datetime(cast(str, row["created_at"])),
            )
            result.setdefault(alias, []).append(entity)
        return result

    def link_record_entity(self, record_id: str, entity_id: str, role: EntityRole = "about") -> None:
        """Create or update a record-to-entity link."""

        self.connection.execute(
            """
            INSERT INTO record_entities(record_id, entity_id, role) VALUES (?, ?, ?)
            ON CONFLICT(record_id, entity_id) DO UPDATE SET role = excluded.role
            """,
            (record_id, entity_id, role),
        )

    def records_for_entities(self, entity_ids: Sequence[str], eligible: set[str], limit: int) -> list[tuple[str, str]]:
        """Return eligible record and entity IDs in event-time order."""

        if limit <= 0 or not entity_ids or not eligible:
            return []
        with self.transaction() as connection:
            connection.execute(
                "CREATE TEMP TABLE IF NOT EXISTS memory_weave_eligible_entities (record_id TEXT PRIMARY KEY)"
            )
            connection.execute(
                "CREATE TEMP TABLE IF NOT EXISTS memory_weave_entity_candidates (entity_id TEXT PRIMARY KEY)"
            )
            connection.execute("DELETE FROM memory_weave_eligible_entities")
            connection.execute("DELETE FROM memory_weave_entity_candidates")
            connection.executemany(
                "INSERT INTO memory_weave_eligible_entities(record_id) VALUES (?)",
                ((record_id,) for record_id in eligible),
            )
            connection.executemany(
                "INSERT INTO memory_weave_entity_candidates(entity_id) VALUES (?)",
                ((entity_id,) for entity_id in entity_ids),
            )
            rows = connection.execute(
                """
                SELECT re.record_id, MIN(re.entity_id) AS entity_id, MAX(r.event_at) AS event_at
                FROM record_entities AS re
                JOIN records AS r ON r.id = re.record_id
                JOIN memory_weave_eligible_entities AS eligible ON eligible.record_id = re.record_id
                JOIN memory_weave_entity_candidates AS candidates ON candidates.entity_id = re.entity_id
                GROUP BY re.record_id
                ORDER BY event_at DESC, re.record_id DESC, entity_id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            connection.execute("DELETE FROM memory_weave_eligible_entities")
            connection.execute("DELETE FROM memory_weave_entity_candidates")
        return [(cast(str, row["record_id"]), cast(str, row["entity_id"])) for row in rows]

    def merge_entity(self, source_id: str, destination_id: str) -> None:
        """Merge aliases and record links into the destination entity."""

        if source_id == destination_id:
            raise ValueError("An entity cannot be merged into itself.")

        with self.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO entity_aliases(entity_id, alias_norm) "
                "SELECT ?, alias_norm FROM entity_aliases WHERE entity_id = ?",
                (destination_id, source_id),
            )
            connection.execute(
                """
                UPDATE record_entities AS destination
                SET role = 'about'
                WHERE destination.entity_id = ?
                  AND destination.role = 'mentions'
                  AND EXISTS (
                      SELECT 1
                      FROM record_entities AS source
                      WHERE source.record_id = destination.record_id
                        AND source.entity_id = ?
                        AND source.role = 'about'
                  )
                """,
                (destination_id, source_id),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO record_entities(record_id, entity_id, role)
                SELECT record_id, ?, role FROM record_entities WHERE entity_id = ?
                """,
                (destination_id, source_id),
            )
            connection.execute("DELETE FROM record_entities WHERE entity_id = ?", (source_id,))
            subject_rows = connection.execute(
                "SELECT id, attribute FROM records WHERE subject_entity_id = ?", (source_id,)
            ).fetchall()
            index_version = self._bump_records_version(connection) if subject_rows else None
            for row in subject_rows:
                record_id = cast(str, row["id"])
                subject = render_subject(destination_id, cast(str | None, row["attribute"]))
                connection.execute(
                    "UPDATE records SET subject_entity_id = ?, subject = ?, index_version = ? WHERE id = ?",
                    (destination_id, subject, index_version, record_id),
                )
                connection.execute("UPDATE records_fts SET subject = ? WHERE record_id = ?", (subject, record_id))
            connection.execute(
                "UPDATE entities SET status = 'merged', merged_into = ? WHERE id = ?", (destination_id, source_id)
            )
            self._rebuild_fts_aliases_for_entity(connection, destination_id)

    def aliases_text_for(self, entity_ids: Sequence[str]) -> str:
        """Return the space-joined normalized canonical names and aliases of the given entities."""

        if not entity_ids:
            return ""
        placeholders = _placeholders(len(entity_ids))
        rows = self.connection.execute(
            f"""
            SELECT e.id, e.canonical, a.alias_norm
            FROM entities AS e
            LEFT JOIN entity_aliases AS a ON a.entity_id = e.id
            WHERE e.id IN ({placeholders})
            ORDER BY e.id, a.alias_norm
            """,
            tuple(entity_ids),
        ).fetchall()
        aliases: dict[str, None] = {}
        for row in rows:
            aliases.setdefault(normalize_alias(cast(str, row["canonical"])), None)
            alias = cast(str | None, row["alias_norm"])
            if alias:
                aliases.setdefault(alias, None)
        return " ".join(aliases)

    def _rebuild_fts_aliases_for_entity(self, connection: sqlite3.Connection, entity_id: str) -> None:
        record_rows = connection.execute(
            "SELECT DISTINCT record_id FROM record_entities WHERE entity_id = ?", (entity_id,)
        ).fetchall()
        for record_row in record_rows:
            record_id = cast(str, record_row["record_id"])
            linked = [
                cast(str, row["entity_id"])
                for row in connection.execute(
                    "SELECT entity_id FROM record_entities WHERE record_id = ? ORDER BY entity_id", (record_id,)
                ).fetchall()
            ]
            connection.execute(
                "UPDATE records_fts SET aliases = ? WHERE record_id = ?", (self.aliases_text_for(linked), record_id)
            )

    def grants_for(self, agent_id: str, *, can_read: bool = False, can_write: bool = False) -> list[Scope]:
        """Return scopes granted to an agent, optionally filtered by capability."""

        query = ["SELECT scope_kind, scope_id FROM grants WHERE agent_id = ?"]
        parameters: list[object] = [agent_id]
        if can_read:
            query.append("AND can_read = 1")
        if can_write:
            query.append("AND can_write = 1")
        query.append("ORDER BY scope_kind, scope_id")
        rows = self.connection.execute(" ".join(query), tuple(parameters)).fetchall()
        return [Scope(kind=cast(Any, row["scope_kind"]), id=cast(str, row["scope_id"])) for row in rows]

    def set_grant(self, agent_id: str, scope: Scope, *, can_read: bool, can_write: bool) -> None:
        """Create or replace an agent's grant for one scope."""

        self.connection.execute(
            """
            INSERT INTO grants(agent_id, scope_kind, scope_id, can_read, can_write) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, scope_kind, scope_id) DO UPDATE SET
                can_read = excluded.can_read, can_write = excluded.can_write
            """,
            (agent_id, scope.kind, scope.id, int(can_read), int(can_write)),
        )

    def revoke_grant(self, agent_id: str, scope: Scope) -> None:
        """Remove one explicit grant without affecting the implicit private scope."""

        self.connection.execute(
            "DELETE FROM grants WHERE agent_id = ? AND scope_kind = ? AND scope_id = ?",
            (agent_id, scope.kind, scope.id),
        )

    def create_session(
        self,
        session_id: str,
        agent_id: str,
        user_id: str,
        project_id: str | None,
        started_at: datetime,
    ) -> None:
        """Create one session transcript container."""

        self.connection.execute(
            "INSERT INTO sessions(id, agent_id, user_id, project_id, started_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, agent_id, user_id, project_id, _dump_datetime(started_at)),
        )

    def append_turn(self, turn: Turn) -> None:
        """Append one immutable transcript turn."""

        self.connection.execute(
            "INSERT INTO session_turns(session_id, turn, role, content, at) VALUES (?, ?, ?, ?, ?)",
            (turn.session_id, turn.turn, turn.role, turn.content, _dump_datetime(turn.at)),
        )

    def session_turns(self, session_id: str) -> list[Turn]:
        """Return session turns in transcript order."""

        rows = self.connection.execute(
            "SELECT session_id, turn, role, content, at FROM session_turns WHERE session_id = ? ORDER BY turn",
            (session_id,),
        ).fetchall()
        return [
            Turn(
                session_id=cast(str, row["session_id"]),
                turn=cast(int, row["turn"]),
                role=cast(TurnRole, row["role"]),
                content=cast(str, row["content"]),
                at=_load_datetime(cast(str, row["at"])),
            )
            for row in rows
        ]

    def end_session(self, session_id: str, ended_at: datetime) -> None:
        """Mark a session closed."""

        self.connection.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (_dump_datetime(ended_at), session_id))

    def mark_extracted(self, session_id: str, extracted_at: datetime) -> None:
        """Record successful end-of-session extraction."""

        self.connection.execute(
            "UPDATE sessions SET extracted_at = ? WHERE id = ?", (_dump_datetime(extracted_at), session_id)
        )

    def append_event(
        self,
        kind: str,
        actor: str,
        record_id: str | None,
        entity_id: str | None,
        payload: Mapping[str, Any],
        *,
        event_id: str | None = None,
        at: datetime | None = None,
    ) -> str:
        """Append one immutable audit event and return its ID."""

        resolved_id = event_id or uuid7()
        self.connection.execute(
            "INSERT INTO events(id, at, kind, actor, record_id, entity_id, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (resolved_id, _dump_datetime(at or now()), kind, actor, record_id, entity_id, _dump_json(payload)),
        )
        return resolved_id

    def events_for(self, record_id: str) -> list[dict[str, Any]]:
        """Return decoded record audit events in chronological order."""

        rows = self.connection.execute(
            "SELECT * FROM events WHERE record_id = ? ORDER BY at, id", (record_id,)
        ).fetchall()
        return [
            {
                "id": cast(str, row["id"]),
                "at": cast(str, row["at"]),
                "kind": cast(str, row["kind"]),
                "actor": cast(str, row["actor"]),
                "record_id": cast(str | None, row["record_id"]),
                "entity_id": cast(str | None, row["entity_id"]),
                "payload": json.loads(cast(str, row["payload"])),
            }
            for row in rows
        ]

    def write_search_log(self, row: Mapping[str, Any]) -> None:
        """Persist one complete retrieval trace."""

        missing = [column for column in _SEARCH_LOG_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"Search log row is missing columns: {', '.join(missing)}")
        values: list[object] = []
        for column in _SEARCH_LOG_COLUMNS:
            value = row[column]
            if column == "at":
                value = _dump_datetime(cast(datetime, value))
            elif column in _SEARCH_LOG_JSON_COLUMNS and value is not None:
                value = _dump_json(value)
            elif column == "warm":
                value = int(cast(bool, value))
            values.append(value)
        placeholders = _placeholders(len(_SEARCH_LOG_COLUMNS))
        columns = ", ".join(_SEARCH_LOG_COLUMNS)
        self.connection.execute(f"INSERT INTO search_log({columns}) VALUES ({placeholders})", tuple(values))

    def read_search_log(self, search_id: str) -> dict[str, Any] | None:
        """Read one search trace and decode its JSON columns."""

        row = self.connection.execute("SELECT * FROM search_log WHERE id = ?", (search_id,)).fetchone()
        if row is None:
            return None
        result = {column: row[column] for column in _SEARCH_LOG_COLUMNS}
        for column in _SEARCH_LOG_JSON_COLUMNS:
            if result[column] is not None:
                result[column] = json.loads(cast(str, result[column]))
        result["warm"] = bool(result["warm"])
        return result

    def snapshot_to(self, path: str | Path) -> None:
        """Copy the database through SQLite's consistent backup API."""

        destination = sqlite3.connect(str(path))
        try:
            self.connection.backup(destination)
        finally:
            destination.close()

    @staticmethod
    def _bump_records_version(connection: sqlite3.Connection) -> int:
        connection.execute("UPDATE store_meta SET value = value + 1 WHERE key = 'records_version'")
        row = connection.execute("SELECT value FROM store_meta WHERE key = 'records_version'").fetchone()
        if row is None:
            raise RuntimeError("store_meta is missing the records_version row.")
        return cast(int, row["value"])

    def _records_from_rows(self, rows: Sequence[sqlite3.Row]) -> list[Record]:
        if not rows:
            return []
        record_ids = [cast(str, row["id"]) for row in rows]
        entity_rows = self.connection.execute(
            "SELECT record_id, entity_id FROM record_entities "
            f"WHERE record_id IN ({_placeholders(len(record_ids))}) ORDER BY entity_id",
            tuple(record_ids),
        ).fetchall()
        entity_ids: dict[str, list[str]] = {record_id: [] for record_id in record_ids}
        for entity_row in entity_rows:
            entity_ids[cast(str, entity_row["record_id"])].append(cast(str, entity_row["entity_id"]))
        return [self._record_from_row(row, entity_ids[cast(str, row["id"])]) for row in rows]

    def _record_from_row(self, row: sqlite3.Row, entity_ids: list[str]) -> Record:
        return Record(
            id=cast(str, row["id"]),
            type=cast(MemoryType, row["type"]),
            version=cast(int, row["version"]),
            content=cast(str, row["content"]),
            subject=cast(str, row["subject"]),
            subject_entity_id=cast(str | None, row["subject_entity_id"]),
            attribute=cast(str | None, row["attribute"]),
            scope=Scope(kind=cast(Any, row["scope_kind"]), id=cast(str, row["scope_id"])),
            source_kind=cast(SourceKind, row["source_kind"]),
            source_ref=cast(str | None, row["source_ref"]),
            creator_agent_id=cast(str, row["creator_agent_id"]),
            evidence=cast(str | None, row["evidence"]),
            created_at=_load_datetime(cast(str, row["created_at"])),
            event_at=_load_datetime(cast(str, row["event_at"])),
            expires_at=_load_datetime_or_none(cast(str | None, row["expires_at"])),
            confidence=cast(float, row["confidence"]),
            status=cast(RecordStatus, row["status"]),
            supersedes_id=cast(str | None, row["supersedes_id"]),
            reinforcements=cast(int, row["reinforcements"]),
            last_reinforced_at=_load_datetime_or_none(cast(str | None, row["last_reinforced_at"])),
            tags=cast(list[str], json.loads(cast(str, row["tags"]))),
            entity_ids=entity_ids,
        )

    def _entity_from_row(self, row: sqlite3.Row) -> Entity:
        aliases = self.connection.execute(
            "SELECT alias_norm FROM entity_aliases WHERE entity_id = ? ORDER BY alias_norm", (row["id"],)
        ).fetchall()
        return Entity(
            id=cast(str, row["id"]),
            kind=cast(EntityKind, row["kind"]),
            canonical=cast(str, row["canonical"]),
            scope=Scope(kind=cast(Any, row["scope_kind"]), id=cast(str, row["scope_id"])),
            status=cast(EntityStatus, row["status"]),
            merged_into=cast(str | None, row["merged_into"]),
            aliases=[cast(str, alias["alias_norm"]) for alias in aliases],
            created_at=_load_datetime(cast(str, row["created_at"])),
        )


def _placeholders(count: int) -> str:
    return ", ".join("?" for _ in range(count))


def _dump_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        raise ValueError("Timestamps must be timezone-aware.")
    return value.astimezone(UTC).isoformat()


def _load_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"Stored timestamp is not timezone-aware: {value}")
    return parsed


def _load_datetime_or_none(value: str | None) -> datetime | None:
    return _load_datetime(value) if value is not None else None


def _dump_json(value: Any) -> str:
    return json.dumps(value, default=_json_default, separators=(",", ":"), sort_keys=True)


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return cast(str, _dump_datetime(value))
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON.")
