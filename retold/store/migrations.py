"""Forward-only SQLite schema migrations."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files

from retold.util import normalize_attribute, render_subject

_LOG = logging.getLogger(__name__)


def _initial_schema() -> str:
    return files("retold.store").joinpath("schema.sql").read_text(encoding="utf-8")


Migration = Callable[[sqlite3.Connection], None]


@dataclass(frozen=True, slots=True)
class MigrationResult:
    """Report migration conditions that require an operator's decision."""

    unmapped_subject_records: int


class MigrationIssuesError(RuntimeError):
    """Raised when a store holds legacy records that migration could not map and the caller did not opt in."""

    def __init__(self, unmapped_subject_records: int) -> None:
        super().__init__(
            f"{unmapped_subject_records} legacy record(s) could not be mapped to a structured subject. "
            "Run `retold migrate` to list them, then `--expire-unmapped` or open the store with "
            "allow_migration_issues=True after reviewing them."
        )
        self.unmapped_subject_records = unmapped_subject_records


def _migration_1(connection: sqlite3.Connection) -> None:
    connection.executescript(_initial_schema())


def _migration_2(connection: sqlite3.Connection) -> None:
    """Add structured subjects, trigger provenance, and extraction claim bookkeeping."""

    if not _has_column(connection, "records", "subject_entity_id"):
        connection.execute("ALTER TABLE records ADD COLUMN subject_entity_id TEXT REFERENCES entities(id)")
    if not _has_column(connection, "records", "attribute"):
        connection.execute("ALTER TABLE records ADD COLUMN attribute TEXT")
    if not _has_column(connection, "search_log", "trigger"):
        connection.execute("ALTER TABLE search_log ADD COLUMN trigger TEXT NOT NULL DEFAULT 'tool'")
    if not _has_column(connection, "sessions", "extraction_started_at"):
        connection.execute("ALTER TABLE sessions ADD COLUMN extraction_started_at TEXT")
    _ensure_migration_issues_table(connection)

    connection.execute("DROP INDEX IF EXISTS records_subject")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS records_subject "
        "ON records(scope_kind, scope_id, subject_entity_id, attribute, status)"
    )
    _backfill_structured_subjects(connection)


def _migration_3(connection: sqlite3.Connection) -> None:
    """Add durable index versions so one process can refresh after another process writes."""

    if not _has_column(connection, "records", "index_version"):
        connection.execute("ALTER TABLE records ADD COLUMN index_version INTEGER NOT NULL DEFAULT 0")
    if not _has_column(connection, "embeddings", "index_version"):
        connection.execute("ALTER TABLE embeddings ADD COLUMN index_version INTEGER NOT NULL DEFAULT 0")
    connection.execute("CREATE TABLE IF NOT EXISTS store_meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL)")
    connection.execute("INSERT OR IGNORE INTO store_meta(key, value) VALUES ('records_version', 0)")


def _migration_4(connection: sqlite3.Connection) -> None:
    """Index version columns used to find a small cross-process refresh delta."""

    connection.execute("CREATE INDEX IF NOT EXISTS records_index_version ON records(index_version)")
    connection.execute("CREATE INDEX IF NOT EXISTS embeddings_index_version ON embeddings(index_version)")


def _migration_5(connection: sqlite3.Connection) -> None:
    """Record every reranker exclusion independently from budget exclusions."""

    if not _has_column(connection, "search_log", "reranked_out"):
        connection.execute("ALTER TABLE search_log ADD COLUMN reranked_out TEXT NOT NULL DEFAULT '[]'")


def _migration_6(connection: sqlite3.Connection) -> None:
    """Ambient activation, retrieval category, and a durable queue for activation decisions needing review."""

    if not _has_column(connection, "records", "activation"):
        connection.execute("ALTER TABLE records ADD COLUMN activation TEXT NOT NULL DEFAULT 'conditional'")
    if not _has_column(connection, "records", "category"):
        connection.execute("ALTER TABLE records ADD COLUMN category TEXT")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS activation_reviews (
            id                  TEXT PRIMARY KEY,
            record_id           TEXT NOT NULL REFERENCES records(id),
            evidence_turn       INTEGER,
            retrieval_category  TEXT,
            activation_category TEXT,
            proposed            TEXT NOT NULL,
            reason              TEXT NOT NULL,
            confidence          REAL,
            policy_version      TEXT NOT NULL,
            status              TEXT NOT NULL DEFAULT 'open',
            created_at          TEXT NOT NULL,
            resolved_at         TEXT,
            resolver            TEXT,
            resolution          TEXT
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS records_activation ON records(scope_kind, scope_id, activation, status)"
    )


def _migration_7(connection: sqlite3.Connection) -> None:
    """One row per host-policy turn, including turns where gap planning returned nothing and no search ran."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS turn_decisions (
            id                  TEXT PRIMARY KEY,
            at                  TEXT NOT NULL,
            session_id          TEXT,
            turn                TEXT NOT NULL,
            disposition         TEXT NOT NULL,
            shadow              INTEGER NOT NULL DEFAULT 0,
            profile_record_ids  TEXT NOT NULL,
            inventory           TEXT NOT NULL,
            gaps                TEXT NOT NULL,
            gap_status          TEXT NOT NULL,
            candidate_ids       TEXT NOT NULL,
            verdicts            TEXT NOT NULL,
            admitted_ids        TEXT NOT NULL,
            admission_status    TEXT NOT NULL,
            requested_budget_ms INTEGER,
            effective_budget_ms INTEGER,
            timings_ms          TEXT NOT NULL,
            failures            TEXT NOT NULL,
            config              TEXT NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX IF NOT EXISTS turn_decisions_session ON turn_decisions(session_id, at)")


def _migration_8(connection: sqlite3.Connection) -> None:
    """Bundle hash and token usage on turn decisions, and the registry of bundle fitness results."""

    if not _has_column(connection, "turn_decisions", "bundle_hash"):
        connection.execute("ALTER TABLE turn_decisions ADD COLUMN bundle_hash TEXT")
    if not _has_column(connection, "turn_decisions", "usage"):
        connection.execute("ALTER TABLE turn_decisions ADD COLUMN usage TEXT NOT NULL DEFAULT '{}'")
    connection.execute("CREATE INDEX IF NOT EXISTS turn_decisions_bundle ON turn_decisions(bundle_hash, at)")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS bundle_fitness (
            id            TEXT PRIMARY KEY,
            bundle_hash   TEXT NOT NULL,
            bundle        TEXT NOT NULL,
            suite_version TEXT NOT NULL,
            passed        INTEGER NOT NULL,
            evidence      TEXT NOT NULL,
            recorded_by   TEXT NOT NULL,
            recorded_at   TEXT NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX IF NOT EXISTS bundle_fitness_hash ON bundle_fitness(bundle_hash, recorded_at)")


def _migration_9(connection: sqlite3.Connection) -> None:
    """Temporal metadata, the due-review index, and one live session summary per session."""

    for column in ("valid_from", "valid_until", "review_at", "review_flagged_at"):
        if not _has_column(connection, "records", column):
            connection.execute(f"ALTER TABLE records ADD COLUMN {column} TEXT")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS records_review_due ON records(review_at) "
        "WHERE review_at IS NOT NULL AND review_flagged_at IS NULL"
    )
    # The summary's idempotency key. A re-run finds the live summary by it and supersedes or keeps it;
    # superseded summaries stay as history, so uniqueness covers active rows only.
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS records_live_session_summary ON records(source_ref) "
        "WHERE source_kind = 'session_summary' AND status IN ('provisional', 'confirmed')"
    )


def _has_column(connection: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row[1] == column for row in connection.execute(f"PRAGMA table_info({table})"))


def _ensure_migration_issues_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS migration_issues (
            migration_version INTEGER NOT NULL,
            record_id TEXT NOT NULL REFERENCES records(id),
            issue TEXT NOT NULL,
            PRIMARY KEY (migration_version, record_id)
        )
        """
    )


def _backfill_structured_subjects(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT r.id, r.subject, r.subject_entity_id, r.attribute,
               GROUP_CONCAT(CASE WHEN re.role = 'about' THEN re.entity_id END) AS about_entity_ids
        FROM records AS r
        LEFT JOIN record_entities AS re ON re.record_id = r.id
        WHERE r.subject_entity_id IS NULL OR r.attribute IS NULL
        GROUP BY r.id
        """
    ).fetchall()
    for record_id, subject, subject_entity_id, attribute, about_entity_ids in rows:
        if subject_entity_id is not None and attribute is not None:
            continue
        entity_ids = [] if about_entity_ids is None else str(about_entity_ids).split(",")
        if len(entity_ids) != 1:
            _record_migration_issue(connection, record_id, "expected exactly one about entity")
            continue
        resolved_attribute = _attribute_from_legacy_subject(str(subject))
        if resolved_attribute is None:
            _record_migration_issue(connection, record_id, "legacy subject has no attribute segment")
            continue
        entity_id = entity_ids[0]
        rendered = render_subject(entity_id, resolved_attribute)
        connection.execute(
            "UPDATE records SET subject_entity_id = ?, attribute = ?, subject = ? WHERE id = ?",
            (entity_id, resolved_attribute, rendered, record_id),
        )
        connection.execute("UPDATE records_fts SET subject = ? WHERE record_id = ?", (rendered, record_id))
        connection.execute("DELETE FROM migration_issues WHERE migration_version = 2 AND record_id = ?", (record_id,))


def _record_migration_issue(connection: sqlite3.Connection, record_id: str, issue: str) -> None:
    _LOG.warning("Migration 2 could not map record %s: %s.", record_id, issue)
    connection.execute(
        """
        INSERT INTO migration_issues(migration_version, record_id, issue) VALUES (2, ?, ?)
        ON CONFLICT(migration_version, record_id) DO UPDATE SET issue = excluded.issue
        """,
        (record_id, issue),
    )


def _attribute_from_legacy_subject(subject: str) -> str | None:
    if "/" not in subject:
        return None
    attribute = subject.rsplit("/", 1)[1]
    if attribute == "-":
        return attribute
    return normalize_attribute(attribute) or None


def _migration_10(connection: sqlite3.Connection) -> None:
    """Record whether the cross-encoder pass applied, timed out, or failed on each search."""

    if not _has_column(connection, "search_log", "rerank_status"):
        connection.execute("ALTER TABLE search_log ADD COLUMN rerank_status TEXT NOT NULL DEFAULT 'disabled'")
    if not _has_column(connection, "search_log", "rerank_error"):
        connection.execute("ALTER TABLE search_log ADD COLUMN rerank_error TEXT")


MIGRATIONS: tuple[tuple[int, Migration], ...] = (
    (1, _migration_1),
    (2, _migration_2),
    (3, _migration_3),
    (4, _migration_4),
    (5, _migration_5),
    (6, _migration_6),
    (7, _migration_7),
    (8, _migration_8),
    (9, _migration_9),
    (10, _migration_10),
)


def migrate(connection: sqlite3.Connection) -> MigrationResult:
    """Apply each pending schema migration exactly once."""

    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_version "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    current_version = connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()[0]

    for version, apply in MIGRATIONS:
        if version <= current_version:
            continue
        apply(connection)
        connection.execute("INSERT INTO schema_version(version) VALUES (?)", (version,))
    connection.commit()
    _ensure_migration_issues_table(connection)
    unresolved = connection.execute(
        """
        SELECT COUNT(*)
        FROM migration_issues AS issues
        JOIN records AS records ON records.id = issues.record_id
        WHERE issues.migration_version = 2
          AND (records.subject_entity_id IS NULL OR records.attribute IS NULL)
        """
    ).fetchone()[0]
    return MigrationResult(unmapped_subject_records=int(unresolved))
