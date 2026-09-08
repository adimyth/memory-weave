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

    reviews = subcommands.add_parser("reviews", help="Activation review queue: list, resolve, or check the backlog.")
    review_commands = reviews.add_subparsers(dest="review_command", required=True)
    review_commands.add_parser("list", help="List open activation reviews, oldest first.")
    resolve = review_commands.add_parser("resolve", help="Resolve one open review.")
    resolve.add_argument("review_id")
    resolve.add_argument("--as", dest="resolution", choices=("promote", "conditional", "reject"), required=True)
    resolve.add_argument("--resolver", required=True, help="Identity of the reviewer, recorded on the event.")
    resolve.add_argument("--note", default="")
    backlog = review_commands.add_parser(
        "backlog", help="Report the open backlog against limits; exit 3 when breached."
    )
    backlog.add_argument("--max-open", type=int, required=True)
    backlog.add_argument("--max-age-days", type=float, required=True)

    activation = subcommands.add_parser(
        "activation", help="Directly promote or demote one record, subject to eligibility checks."
    )
    activation.add_argument("record_id")
    activation.add_argument("target", choices=("ambient", "conditional"))
    activation.add_argument("--resolver", required=True)
    activation.add_argument("--reason", required=True)

    metrics = subcommands.add_parser(
        "metrics", help="Aggregate the turn-decision log: stage outcomes, admission rates, latency, cost, backlog."
    )
    metrics.add_argument("--since", default=None, help="ISO 8601 start of the window, inclusive.")
    metrics.add_argument("--until", default=None, help="ISO 8601 end of the window, exclusive.")
    metrics.add_argument("--bundle", default=None, help="Restrict to one bundle hash.")
    metrics.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    metrics.add_argument(
        "--rollback-check", action="store_true", help="Apply default rollback thresholds; exit 4 when any is breached."
    )

    bundles = subcommands.add_parser("bundles", help="Bundle fitness registry: list results or record one.")
    bundle_commands = bundles.add_subparsers(dest="bundle_command", required=True)
    bundle_commands.add_parser("list", help="List recorded fitness results, newest first.")
    record = bundle_commands.add_parser("record", help="Record a fitness result for a bundle described in a JSON file.")
    record.add_argument("components_json", help="Path to a JSON object with the bundle components.")
    record.add_argument("--passed", action="store_true")
    record.add_argument("--failed", action="store_true")
    record.add_argument(
        "--evidence", required=True, help="Where the suite results live, e.g. a results path or commit."
    )
    record.add_argument("--by", required=True, dest="recorded_by")

    args = parser.parse_args(argv)
    store = Store(args.store, allow_migration_issues=args.command == "migrate")
    try:
        if args.command == "metrics":
            return _metrics_command(store, args)
        if args.command == "bundles":
            return _bundle_commands(store, args)
        if args.command in ("reviews", "activation"):
            return _activation_commands(store, args)
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


def _metrics_command(store: Store, args: argparse.Namespace) -> int:
    import json
    from datetime import datetime

    from memory_weave.policy import RollbackThresholds, aggregate, render, rollback_reasons

    since = datetime.fromisoformat(args.since) if args.since else None
    until = datetime.fromisoformat(args.until) if args.until else None
    report = aggregate(store, since=since, until=until, bundle_hash=args.bundle)
    reasons = rollback_reasons(report, RollbackThresholds()) if args.rollback_check else []
    if args.json:
        payload = report.to_dict()
        if args.rollback_check:
            payload["rollback_reasons"] = reasons
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(render(report))
        if args.rollback_check:
            print("rollback: " + ("; ".join(reasons) if reasons else "no threshold breached"))
    return 4 if reasons else 0


def _bundle_commands(store: Store, args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from memory_weave.policy import BundleRegistry

    registry = BundleRegistry(store)
    if args.bundle_command == "list":
        rows = registry.all()
        if not rows:
            print("no fitness results recorded")
            return 0
        for row in rows:
            print(
                f"{row.bundle_hash}  {'PASS' if row.passed else 'FAIL'}  {row.suite_version}  {row.recorded_at}  "
                f"by={row.recorded_by}  {row.evidence}"
            )
        return 0
    if args.passed == args.failed:
        print("refused: pass exactly one of --passed or --failed")
        return 2
    components = json.loads(Path(args.components_json).read_text())
    if not isinstance(components, dict):
        print("refused: the components file must hold a JSON object")
        return 2
    record = registry.record(components, passed=args.passed, evidence=args.evidence, recorded_by=args.recorded_by)
    print(f"recorded {record.bundle_hash} {'PASS' if record.passed else 'FAIL'}")
    return 0


def _activation_commands(store: Store, args: argparse.Namespace) -> int:
    """Review-queue and activation operations for a trusted operator. Every change is audited by the service."""

    from memory_weave.policy import ActivationOperations

    operations = ActivationOperations(store)
    if args.command == "activation":
        try:
            operations.set_activation(args.record_id, args.target, resolver=args.resolver, reason=args.reason)
        except ValueError as error:
            print(f"refused: {error}")
            return 2
        print(f"{args.record_id} -> {args.target}")
        return 0
    if args.review_command == "list":
        rows = store.open_activation_reviews()
        if not rows:
            print("no open reviews")
            return 0
        for row in rows:
            record = store.get_record(str(row["record_id"]))
            content = record.content if record else "<missing record>"
            print(
                f"{row['id']}  record={row['record_id']}  proposed={row['proposed']}  reason={row['reason']}  "
                f"category={row['activation_category']}  since={row['created_at']}\n    {content}"
            )
        return 0
    if args.review_command == "resolve":
        try:
            record_id = operations.resolve_review(
                args.review_id, args.resolution, resolver=args.resolver, note=args.note
            )
        except ValueError as error:
            print(f"refused: {error}")
            return 2
        print(f"resolved {args.review_id} as {args.resolution}; record {record_id}")
        return 0
    status = operations.backlog(max_open=args.max_open, max_age_days=args.max_age_days)
    print(
        f"open={status.open_count} oldest_age_days={status.oldest_age_days:.1f} "
        f"over_count_limit={status.over_count_limit} over_age_limit={status.over_age_limit}"
    )
    return 0 if status.within_limits else 3


def _parse_scope(value: str) -> Scope:
    kind, separator, scope_id = value.partition(":")
    if not separator or not scope_id or kind not in {"agent", "user", "project", "org"}:
        raise argparse.ArgumentTypeError("scope must be one of agent:ID, user:ID, project:ID, or org:ID")
    return Scope(kind=cast(ScopeKind, kind), id=scope_id)
