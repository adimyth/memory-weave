"""Small operational CLI for trusted host grant management and migrations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import cast

from memory_weave.host import MemoryHost
from memory_weave.models import Scope, ScopeKind
from memory_weave.store import Store


def main(argv: Sequence[str] | None = None) -> int:
    """Run the grant and revoke commands used by a trusted host operator."""

    parser = argparse.ArgumentParser(prog="memory-weave")
    parser.add_argument("--store", default="./memory.sqlite", help="Path to the SQLite memory store.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    migrate = subcommands.add_parser("migrate", help="Apply schema migrations and report unmapped legacy subjects.")
    migrate.add_argument(
        "--allow-unmapped",
        action="store_true",
        help="Complete with unresolved legacy subject records after listing them.",
    )
    migrate.add_argument(
        "--expire-unmapped",
        action="store_true",
        help="Expire every unresolved legacy subject record so it leaves default retrieval, then complete.",
    )
    grant = subcommands.add_parser("grant", help="Create or update an explicit scope grant.")
    revoke = subcommands.add_parser("revoke", help="Remove an explicit scope grant.")
    for command in (grant, revoke):
        command.add_argument("agent_id")
        command.add_argument("scope", help="Scope in the form user:U, project:P, org:O, or agent:A.")
    grant.add_argument("--read", action="store_true")
    grant.add_argument("--write", action="store_true")

    args = parser.parse_args(argv)
    store = Store(args.store, allow_migration_issues=args.command == "migrate")
    try:
        if args.command == "migrate":
            issues = store.migration_issues()
            for record_id, issue in issues:
                print(f"unmapped {record_id}: {issue}")
            if issues and args.expire_unmapped:
                expired = store.expire_migration_issues("admin")
                print(f"Expired {len(expired)} unmapped legacy subject record(s).")
                return 0
            if issues and not args.allow_unmapped:
                print(
                    f"Migration found {len(issues)} unmapped legacy subject record(s). "
                    "Rerun with --expire-unmapped to retire them, or --allow-unmapped after reviewing them.",
                )
                return 2
            return 0

        try:
            scope = _parse_scope(args.scope)
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
        host = MemoryHost(store)
        if args.command == "grant":
            host.grant(args.agent_id, scope, read=args.read, write=args.write)
        else:
            host.revoke(args.agent_id, scope)
    finally:
        store.close()
    return 0


def _parse_scope(value: str) -> Scope:
    kind, separator, scope_id = value.partition(":")
    if not separator or not scope_id or kind not in {"agent", "user", "project", "org"}:
        raise argparse.ArgumentTypeError("scope must be one of agent:ID, user:ID, project:ID, or org:ID")
    return Scope(kind=cast(ScopeKind, kind), id=scope_id)
