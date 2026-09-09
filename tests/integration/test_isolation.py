"""The named isolation class for the many-agents, many-users target.

The same assertions run twice: on a synthetic multi-user store with the fake embedder, in the ordinary
suite, and on the 1K fixture with the real embedder, in the integration suite. For every pair of users
sharing an agent and every pair of agents sharing a user, no channel returns a record from the other party's
user scope or private scope, at any stage of the log, not only in the returned list; grants widen exactly
to the granted scopes; `include_history` never widens scope; and `memory_get` on a foreign id is
`not_found`. The pass criterion is zero violations.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest

from retold.config import EmbeddingConfig, RetoldConfig
from retold.host import MemoryHost
from retold.index.embedder import Embedder, FakeEmbedder
from retold.models import Principal, Record, Scope, SearchRequest
from retold.policy import readable_scopes
from retold.runtime import MemoryRuntime, build_runtime
from retold.store import Store
from retold.util import render_subject

from .fixture_1k import NOW, PROJECT_GRANTS, USER_GRANTS, copy_fixture

_LOG_STAGES = ("dense", "lexical", "entity", "fused", "freshness", "gated_out", "deduped_out", "budget_out", "returned")


@dataclass(slots=True)
class Violation:
    principal: Principal
    kind: str
    detail: str


def _foreign_scopes(store: Store, agent: str, user: str) -> set[Scope]:
    """Every user scope and private scope in the store that this principal must never see."""

    readable = set(readable_scopes(store, agent, user))
    rows = store.connection.execute("SELECT DISTINCT scope_kind, scope_id FROM records").fetchall()
    foreign: set[Scope] = set()
    for row in rows:
        scope = Scope(kind=row["scope_kind"], id=row["scope_id"])
        if scope in readable:
            continue
        if scope.kind == "user" or (scope.kind == "agent" and "/" in scope.id):
            foreign.add(scope)
    return foreign


def _record_scope(store: Store, record_id: str) -> Scope | None:
    record = store.get_record(record_id)
    return record.scope if record else None


def _ids_in_stage(value: Any) -> list[str]:
    ids: list[str] = []
    if isinstance(value, list):
        for entry in value:
            if isinstance(entry, str):
                ids.append(entry)
            elif isinstance(entry, dict):
                for key in ("record_id", "id", "dropped_id", "kept_id"):
                    if isinstance(entry.get(key), str):
                        ids.append(entry[key])
            elif isinstance(entry, list) and entry and isinstance(entry[0], str):
                ids.append(entry[0])
    return ids


def check_isolation(runtime: MemoryRuntime, grants: dict[str, tuple[str, ...]]) -> list[Violation]:
    store = runtime.store
    violations: list[Violation] = []
    pairs: list[tuple[Principal, Principal]] = []
    for agent, users in grants.items():
        for first, second in combinations(users, 2):
            pairs.append((Principal(agent, first, None, None), Principal(agent, second, None, None)))
    users_to_agents: dict[str, list[str]] = {}
    for agent, users in grants.items():
        for user in users:
            users_to_agents.setdefault(user, []).append(agent)
    for user, agents in users_to_agents.items():
        for first, second in combinations(agents, 2):
            pairs.append((Principal(first, user, None, None), Principal(second, user, None, None)))

    for me, other in pairs:
        for principal, target in ((me, other), (other, me)):
            foreign = _foreign_scopes(store, principal.agent_id, principal.user_id)
            # Query with the other party's own text: the strongest possible lure for every channel.
            lures = [
                record.content
                for scope in (
                    Scope(kind="user", id=target.user_id),
                    Scope(kind="agent", id=f"{target.agent_id}/{target.user_id}"),
                )
                for record in store.records_in_scope(scope, statuses=["provisional", "confirmed"])[:3]
                if scope in foreign
            ]
            for lure in lures:
                for include_history in (False, True):
                    request = SearchRequest([lure], None, None, None, None, None, 8, include_history)
                    response = runtime.retriever.search(principal, request)
                    for result in response.results:
                        if result.record.scope in foreign:
                            violations.append(
                                Violation(principal, "returned", f"{result.record.id} from {result.record.scope}")
                            )
                    log = store.read_search_log(response.search_id)
                    assert log is not None
                    logged_scopes = {
                        Scope(**scope) if isinstance(scope, dict) else scope for scope in log["readable_scopes"]
                    }  # type: ignore[arg-type]
                    for stage in _LOG_STAGES:
                        for record_id in _ids_in_stage(log.get(stage)):
                            scope = _record_scope(store, record_id)
                            if scope is not None and scope in foreign:
                                violations.append(Violation(principal, f"log:{stage}", f"{record_id} from {scope}"))
                    for scope in logged_scopes:
                        if isinstance(scope, Scope) and scope in foreign:
                            violations.append(Violation(principal, "readable_scopes", str(scope)))
            # A foreign id through memory_get answers exactly like a missing one.
            for scope in foreign:
                for record in store.records_in_scope(scope, statuses=["provisional", "confirmed"])[:2]:
                    payload = runtime.handlers.memory_get(principal, {"ids": [record.id]})
                    if payload.get("ok") is not False or payload.get("error", {}).get("code") != "not_found":  # type: ignore[union-attr]
                        violations.append(
                            Violation(principal, "memory_get", f"{record.id} answered {json.dumps(payload)[:80]}")
                        )
            # Grants widen exactly to the granted scopes.
            expected = {
                Scope(kind="agent", id=f"{principal.agent_id}/{principal.user_id}"),
                Scope(kind="user", id=principal.user_id),
            }
            expected |= {Scope(kind="project", id=p) for p in PROJECT_GRANTS.get(principal.agent_id, ())}
            actual = set(readable_scopes(store, principal.agent_id, principal.user_id))
            unexpected = {scope for scope in actual - expected if scope.kind != "org"}
            if unexpected:
                violations.append(Violation(principal, "grants", str(sorted(str(s) for s in unexpected))))
    return violations


def _synthetic_store(path: Path) -> tuple[Store, MemoryRuntime]:
    config = RetoldConfig(embedding=EmbeddingConfig(model="fake-embedder", version="1", dims=16))
    store = Store(path)
    host = MemoryHost(store)
    embedder: Embedder = FakeEmbedder(dims=16)
    for user in ("aditya", "priya", "rohan", "meera"):
        host.provision_user(user)
    for agent, users in USER_GRANTS.items():
        for user in users:
            host.grant(agent, Scope(kind="user", id=user), read=True, write=True)
        for project in PROJECT_GRANTS[agent]:
            host.grant(agent, Scope(kind="project", id=project), read=True, write=True)
    index = 0
    with store.transaction():
        for agent, users in USER_GRANTS.items():
            for user in users:
                for scope in (Scope(kind="user", id=user), Scope(kind="agent", id=f"{agent}/{user}")):
                    for n in range(4):
                        index += 1
                        content = f"{user} secret {scope.kind} fact number {n} about {agent} topic {index}"
                        record = Record(
                            id=f"iso-{index}",
                            type="semantic",
                            version=1,
                            content=content,
                            subject=render_subject(None, f"fact_{index}"),
                            scope=scope,
                            source_kind="user_statement",
                            source_ref=None,
                            creator_agent_id=agent,
                            evidence=None,
                            created_at=NOW,
                            event_at=NOW,
                            expires_at=None,
                            confidence=0.9,
                            status="confirmed",
                            supersedes_id=None,
                            reinforcements=0,
                            last_reinforced_at=None,
                            tags=[],
                            entity_ids=[],
                            attribute=f"fact_{index}",
                        )
                        store.insert_record(record)
                        store.put_embedding(
                            record.id, embedder.name, embedder.version, embedder.embed_documents([content])[0]
                        )
                        store.upsert_fts(record.id, content, record.subject, user)
    runtime = build_runtime(config, store, embedder=embedder)
    return store, runtime


def test_synthetic_multi_user_store_has_zero_isolation_violations(tmp_path: Path) -> None:
    store, runtime = _synthetic_store(tmp_path / "iso.sqlite")
    violations = check_isolation(runtime, USER_GRANTS)
    assert violations == [], "\n".join(f"{v.principal} {v.kind}: {v.detail}" for v in violations)
    store.close()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RETOLD_INTEGRATION") != "1",
    reason="set RETOLD_INTEGRATION=1 to run local-model integration tests",
)
def test_fixture_store_has_zero_isolation_violations(tmp_path: Path) -> None:
    store = Store(copy_fixture(tmp_path))
    runtime = build_runtime(RetoldConfig(), store)
    violations = check_isolation(runtime, USER_GRANTS)
    assert violations == [], "\n".join(f"{v.principal} {v.kind}: {v.detail}" for v in violations)
    store.close()
