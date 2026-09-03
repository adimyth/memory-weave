"""Forward-only SQLite schema migrations."""

from __future__ import annotations

import sqlite3
from importlib.resources import files


def _initial_schema() -> str:
    return files("memory_weave.store").joinpath("schema.sql").read_text(encoding="utf-8")


MIGRATIONS: tuple[tuple[int, str], ...] = ((1, _initial_schema()),)


def migrate(connection: sqlite3.Connection) -> None:
    """Apply each pending schema migration exactly once."""

    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_version "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    current_version = connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()[0]

    for version, statements in MIGRATIONS:
        if version <= current_version:
            continue
        connection.executescript(statements)
        connection.execute("INSERT INTO schema_version(version) VALUES (?)", (version,))
    connection.commit()
