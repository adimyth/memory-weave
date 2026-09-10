"""The operator CLI: every LLD 16 command over one store, without writing Python.

Commands that need models build the runtime through ``build_runtime``; tests substitute fakes there.
Irreversible commands require ``--yes``. Exit codes: 0 success, 2 refused or invalid request, 3 review
backlog breached, 4 rollback threshold breached.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from retold import runtime
from retold.config import RetoldConfig, load_config
from retold.host import MemoryHost
from retold.models import Principal, Scope, ScopeKind, SearchRequest
from retold.operations import OperationRefused, Operations
from retold.store import Store

EXIT_REFUSED = 2
EXIT_BACKLOG = 3
EXIT_ROLLBACK = 4


def main(argv: Sequence[str] | None = None) -> int:
    """Parse one command and run it against the store named by ``--store``."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else load_config()
    if args.command == "snapshot" and args.snapshot_command == "load":
        return _snapshot_load(args, config)
    store = Store(
        args.store,
        allow_migration_issues=args.command == "migrate",
        busy_timeout_s=config.store.busy_timeout_seconds,
    )
    try:
        return _dispatch(parser, store, config, args)
    except OperationRefused as error:
        print(f"refused: {error}")
        return EXIT_REFUSED
    finally:
        store.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retold")
    parser.add_argument("--store", default="./memory.sqlite", help="Path to the SQLite memory store.")
    parser.add_argument("--config", default=None, help="YAML configuration file; defaults apply without one.")
    parser.add_argument("--as", dest="actor", default="admin", help="Operator identity recorded on audit events.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    migrate = subcommands.add_parser("migrate", help="Apply schema migrations and report unmapped legacy subjects.")
    migrate.add_argument("--allow-unmapped", action="store_true", help="Complete with unresolved legacy records.")
    migrate.add_argument(
        "--expire-unmapped", action="store_true", help="Expire unresolved legacy records, then complete."
    )

    grant = subcommands.add_parser("grant", help="Create or update an explicit scope grant.")
    revoke = subcommands.add_parser("revoke", help="Remove an explicit scope grant.")
    for command in (grant, revoke):
        command.add_argument("agent_id")
        command.add_argument("scope", help="Scope in the form user:U, project:P, org:O, or agent:A.")
    grant.add_argument("--read", action="store_true")
    grant.add_argument("--write", action="store_true")

    search = subcommands.add_parser("search", help="Run memory_search as one principal and print the log row.")
    search.add_argument("query", nargs="+")
    _principal_arguments(search)
    search.add_argument("--k", type=int, default=None)
    search.add_argument("--trigger", choices=("tool", "auto"), default="tool")
    search.add_argument("--context", default=None)
    search.add_argument("--json", action="store_true")

    get = subcommands.add_parser("get", help="Print a readable record with lineage, conflicts, and events.")
    get.add_argument("record_id")
    _principal_arguments(get)

    dump = subcommands.add_parser("dump", help="Print the active records in one scope, oldest event first.")
    dump.add_argument("--scope", required=True)
    dump.add_argument("--all-statuses", action="store_true", help="Include superseded, expired, and deleted rows.")
    dump.add_argument("--json", action="store_true")

    subcommands.add_parser("expire", help="Mark active records past their expiry as expired, one event each.")
    subcommands.add_parser("retain", help="Blank transcripts of extracted sessions older than sessions.retain_days.")
    subcommands.add_parser("review-due", help="Flag one bounded batch of records whose review_at has arrived.")

    reembed = subcommands.add_parser("reembed", help="Re-embed every record with the configured model and version.")
    reembed.add_argument("--model", required=True)
    reembed.add_argument("--version", required=True)
    reembed.add_argument("--dims", type=int, default=None, help="Vector width when it differs from the configuration.")

    erase = subcommands.add_parser("erase", help="Irreversibly erase durable content for a record, session, or user.")
    target = erase.add_mutually_exclusive_group(required=True)
    target.add_argument("--record")
    target.add_argument("--session")
    target.add_argument("--user")
    erase.add_argument("--reason", required=True)
    erase.add_argument("--yes", action="store_true", help="Confirm the irreversible erasure.")

    extract = subcommands.add_parser("extract", help="Run session extraction through candidate review.")
    extract.add_argument("session_id")
    extract.add_argument("--force", action="store_true", help="Re-run an already extracted session.")
    extract.add_argument("--json", action="store_true")

    snapshot = subcommands.add_parser("snapshot", help="Copy the database for fixtures, or restore one.")
    snapshot_commands = snapshot.add_subparsers(dest="snapshot_command", required=True)
    save = snapshot_commands.add_parser("save")
    save.add_argument("path")
    load = snapshot_commands.add_parser("load")
    load.add_argument("path")
    load.add_argument("--yes", action="store_true", help="Confirm replacing the store file.")

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
        "activation", help="Directly promote or demote one record, with eligibility checks."
    )
    activation.add_argument("record_id")
    activation.add_argument("target", choices=("ambient", "conditional"))
    activation.add_argument("--resolver", required=True)
    activation.add_argument("--reason", required=True)

    metrics = subcommands.add_parser(
        "metrics", help="Aggregate the turn-decision log: stages, rates, latency, cost, backlog."
    )
    metrics.add_argument("--since", default=None, help="ISO 8601 start of the window, inclusive.")
    metrics.add_argument("--until", default=None, help="ISO 8601 end of the window, exclusive.")
    metrics.add_argument("--bundle", default=None, help="Restrict to one bundle hash.")
    metrics.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    metrics.add_argument(
        "--rollback-check", action="store_true", help="Apply rollback thresholds; exit 4 when breached."
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
    return parser


def _principal_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--agent", required=True, help="Agent id the command acts as; scopes follow its grants.")
    command.add_argument("--user", required=True, help="User id the command acts for.")
    command.add_argument("--session", default=None)
    command.add_argument("--project", default=None)


def _dispatch(parser: argparse.ArgumentParser, store: Store, config: RetoldConfig, args: argparse.Namespace) -> int:
    operations = Operations(store, config, actor=args.actor)
    if args.command == "metrics":
        return _metrics_command(store, args)
    if args.command == "bundles":
        return _bundle_commands(store, args)
    if args.command in ("reviews", "activation"):
        return _activation_commands(store, args)
    if args.command == "migrate":
        return _migrate(store, args)
    if args.command in ("grant", "revoke"):
        try:
            scope = _parse_scope(args.scope)
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
        host = MemoryHost(store)
        if args.command == "grant":
            host.grant(args.agent_id, scope, read=args.read, write=args.write)
        else:
            host.revoke(args.agent_id, scope)
        return 0
    if args.command == "search":
        return _search(store, config, args)
    if args.command == "get":
        return _get(store, config, args)
    if args.command == "dump":
        return _dump(store, args)
    if args.command == "expire":
        ids = operations.expire()
        print(f"expired {len(ids)} record(s)" + (": " + " ".join(ids) if ids else ""))
        return 0
    if args.command == "retain":
        ids = operations.retain()
        print(f"retained {len(ids)} session(s)" + (": " + " ".join(ids) if ids else ""))
        return 0
    if args.command == "review-due":
        ids = operations.flag_due_reviews()
        print(f"flagged {len(ids)} record(s)" + (": " + " ".join(ids) if ids else ""))
        return 0
    if args.command == "reembed":
        return _reembed(store, config, operations, args)
    if args.command == "erase":
        return _erase(operations, args)
    if args.command == "extract":
        return _extract(store, config, args)
    if args.command == "snapshot":
        path = operations.snapshot_save(args.path)
        print(f"saved snapshot to {path}")
        return 0
    parser.error(f"unknown command {args.command}")


def _migrate(store: Store, args: argparse.Namespace) -> int:
    issues = store.migration_issues()
    for record_id, issue in issues:
        print(f"unmapped {record_id}: {issue}")
    if issues and args.expire_unmapped:
        expired = store.expire_migration_issues(args.actor)
        print(f"Expired {len(expired)} unmapped legacy subject record(s).")
        return 0
    if issues and not args.allow_unmapped:
        print(
            f"Migration found {len(issues)} unmapped legacy subject record(s). "
            "Rerun with --expire-unmapped to retire them, or --allow-unmapped after reviewing them.",
        )
        return EXIT_REFUSED
    print("migrated")
    return 0


def _principal(args: argparse.Namespace) -> Principal:
    return Principal(args.agent, args.user, args.session, args.project)


def _search(store: Store, config: RetoldConfig, args: argparse.Namespace) -> int:
    components = runtime.build_runtime(config, store)
    request = SearchRequest(
        queries=[" ".join(args.query)],
        context=args.context,
        types=None,
        entities=None,
        since=None,
        until=None,
        k=args.k or config.retrieval.default_k,
        include_history=False,
        trigger=args.trigger,
    )
    response = components.retriever.search(_principal(args), request)
    log = store.read_search_log(response.search_id) or {}
    if args.json:
        payload = {
            "search_id": response.search_id,
            "results": [
                {"record": _record_payload(entry.record), "score": entry.score, "summary": entry.explanation.summary}
                for entry in response.results
            ],
            "empty_reason": response.empty_reason,
            "timings_ms": response.timings_ms,
            "log": {key: log.get(key) for key in ("rewrite_status", "returned", "gated_out", "reranked_out")},
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if not response.results:
        print(f"no results ({response.empty_reason})")
    for entry in response.results:
        record = entry.record
        print(f"{record.id}  {record.type}  {record.status}  score={entry.score:.3f}  {record.content}")
        print(f"    {entry.explanation.summary}")
    print(
        f"search {response.search_id}: rewrite={log.get('rewrite_status')} returned={len(response.results)} "
        f"gated_out={len(log.get('gated_out') or [])} total_ms={response.timings_ms.get('total', 0):.1f}"
    )
    return 0


def _get(store: Store, config: RetoldConfig, args: argparse.Namespace) -> int:
    components = runtime.build_runtime(config, store)
    result = components.handlers.memory_get(_principal(args), {"ids": [args.record_id]})
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else EXIT_REFUSED


def _dump(store: Store, args: argparse.Namespace) -> int:
    scope = _parse_scope(args.scope)
    statuses: list[Any] = (
        ["provisional", "confirmed", "superseded", "expired", "deleted"]
        if args.all_statuses
        else ["provisional", "confirmed"]
    )
    records = store.records_in_scope(scope, statuses=statuses)
    if args.json:
        print(json.dumps([_record_payload(record) for record in records], indent=2, default=str))
        return 0
    for record in records:
        print(
            f"{record.id}  {record.type}  {record.status}  {record.activation}  {record.source_kind}  "
            f"{record.event_at.date().isoformat()}  {record.subject}  {record.content}"
        )
    print(f"{len(records)} record(s) in {scope.kind}:{scope.id}")
    return 0


def _reembed(store: Store, config: RetoldConfig, operations: Operations, args: argparse.Namespace) -> int:
    from dataclasses import replace

    embedding = replace(
        config.embedding, model=args.model, version=args.version, dims=args.dims or config.embedding.dims
    )
    target = replace(config, embedding=embedding)
    components = runtime.build_runtime(target, store, allow_embedding_mismatch=True)
    count = Operations(store, target, actor=args.actor).reembed(components.embedder)
    print(f"re-embedded {count} record(s) with {args.model}/{args.version}")
    return 0


def _erase(operations: Operations, args: argparse.Namespace) -> int:
    if not args.yes:
        print("refused: erasure is irreversible; pass --yes to confirm")
        return EXIT_REFUSED
    if args.record:
        report = operations.erase_record(args.record, reason=args.reason)
    elif args.session:
        report = operations.erase_session(args.session, reason=args.reason)
    else:
        report = operations.erase_user(args.user, reason=args.reason)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def _extract(store: Store, config: RetoldConfig, args: argparse.Namespace) -> int:
    session = store.get_session(args.session_id)
    if session is None:
        print(f"refused: session {args.session_id} does not exist")
        return EXIT_REFUSED
    components = runtime.build_runtime(config, store)
    principal = Principal(session.agent_id, session.user_id, session.id, session.project_id)
    result = components.extraction.extract_session(args.session_id, principal, force=args.force)
    if args.json:
        payload = asdict(result)
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"{args.session_id}: {result.status}" + (f" ({result.reason})" if result.reason else ""))
        for outcome in result.candidates:
            print(f"  [{outcome.stage}] {outcome.outcome}: {outcome.content}")
        if result.summary_record_id:
            print(f"  summary {result.summary_record_id}: {result.summary_outcome}")
    return 0 if result.status in ("completed", "already_extracted") else EXIT_REFUSED


def _snapshot_load(args: argparse.Namespace, config: RetoldConfig) -> int:
    if not args.yes:
        print("refused: restoring replaces the store file; pass --yes to confirm")
        return EXIT_REFUSED
    try:
        placeholder = Store(args.store, busy_timeout_s=config.store.busy_timeout_seconds)
        Operations(placeholder, config, actor=args.actor).snapshot_load(args.path, args.store)
    except OperationRefused as error:
        print(f"refused: {error}")
        return EXIT_REFUSED
    store = Store(args.store, busy_timeout_s=config.store.busy_timeout_seconds)
    try:
        store.append_event("store.restored", args.actor, None, None, {"path": str(args.path)})
    finally:
        store.close()
    print(f"restored {args.store} from {args.path}")
    return 0


def _record_payload(record: Any) -> dict[str, Any]:
    payload = asdict(record)
    payload["scope"] = {"kind": record.scope.kind, "id": record.scope.id}
    return payload


def _metrics_command(store: Store, args: argparse.Namespace) -> int:
    from retold.policy import RollbackThresholds, aggregate, render, rollback_reasons

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
    return EXIT_ROLLBACK if reasons else 0


def _bundle_commands(store: Store, args: argparse.Namespace) -> int:
    from retold.policy import BundleRegistry

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
        return EXIT_REFUSED
    components = json.loads(Path(args.components_json).read_text())
    if not isinstance(components, dict):
        print("refused: the components file must hold a JSON object")
        return EXIT_REFUSED
    record = registry.record(components, passed=args.passed, evidence=args.evidence, recorded_by=args.recorded_by)
    print(f"recorded {record.bundle_hash} {'PASS' if record.passed else 'FAIL'}")
    return 0


def _activation_commands(store: Store, args: argparse.Namespace) -> int:
    """Review-queue and activation operations for a trusted operator. Every change is audited by the service."""

    from retold.policy import ActivationOperations

    operations = ActivationOperations(store)
    if args.command == "activation":
        try:
            operations.set_activation(args.record_id, args.target, resolver=args.resolver, reason=args.reason)
        except ValueError as error:
            print(f"refused: {error}")
            return EXIT_REFUSED
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
            return EXIT_REFUSED
        print(f"resolved {args.review_id} as {args.resolution}; record {record_id}")
        return 0
    status = operations.backlog(max_open=args.max_open, max_age_days=args.max_age_days)
    print(
        f"open={status.open_count} oldest_age_days={status.oldest_age_days:.1f} "
        f"over_count_limit={status.over_count_limit} over_age_limit={status.over_age_limit}"
    )
    return 0 if status.within_limits else EXIT_BACKLOG


def _parse_scope(value: str) -> Scope:
    kind, separator, scope_id = value.partition(":")
    if not separator or not scope_id or kind not in {"agent", "user", "project", "org"}:
        raise argparse.ArgumentTypeError("scope must be one of agent:ID, user:ID, project:ID, or org:ID")
    return Scope(kind=cast(ScopeKind, kind), id=scope_id)
