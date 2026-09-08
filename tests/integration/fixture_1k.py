"""The 1K-record integration fixture: a hand-written seed expanded deterministically, embedded for real.

Four users, three agents, two projects, and one organisation share one store. Every named record below is
hand-written and carries a stable id, so `labelled_queries.yaml` can point at it. Around nine hundred more
records come from templates over the same people, projects, services, and dates, so retrieval has the
distractor mass a real store has. Some current facts have superseded chains, some inferences are past
their expiry, and some rows are already expired.

Build the snapshot once with the real embedder (about a minute on an M-series laptop):

    HF_HUB_OFFLINE=1 MEMORY_WEAVE_INTEGRATION=1 uv run --extra local-models python -m tests.integration.fixture_1k

Tests copy `tests/fixtures/memory_1k.sqlite` into a temporary directory and open it there.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from memory_weave.config import EmbeddingConfig, MemoryWeaveConfig
from memory_weave.host import MemoryHost
from memory_weave.index.embedder import Embedder
from memory_weave.models import Record, Scope
from memory_weave.store import Store
from memory_weave.util import render_subject

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "memory_1k.sqlite"
NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
EPOCH = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)

USERS = ("aditya", "priya", "rohan", "meera")
AGENTS = ("research-agent", "ops-agent", "coding-agent")
USER_ALIASES = {
    "aditya": ("Aditya", "Aditya Mishra"),
    "priya": ("Priya", "Priya Nair"),
    "rohan": ("Rohan", "Rohan Mehta"),
    "meera": ("Meera", "Meera Iyer"),
}
# Which agents may read and write which user scopes; the private agent/user scopes exist for every pair.
USER_GRANTS = {
    "research-agent": ("aditya", "priya"),
    "ops-agent": ("priya", "rohan"),
    "coding-agent": ("aditya", "rohan", "meera"),
}
PROJECT_GRANTS = {
    "research-agent": ("memory-weave",),
    "ops-agent": ("billing",),
    "coding-agent": ("memory-weave", "billing"),
}
ORG = "acme"


def user_scope(user: str) -> Scope:
    return Scope(kind="user", id=user)


def private_scope(agent: str, user: str) -> Scope:
    return Scope(kind="agent", id=f"{agent}/{user}")


def project_scope(project: str) -> Scope:
    return Scope(kind="project", id=project)


@dataclass(slots=True)
class Seed:
    id: str
    type: str
    content: str
    scope: Scope
    attribute: str | None = None
    entity: tuple[str, str, str] | None = None  # (kind, canonical, alias)
    about_user: str | None = None
    source_kind: str = "user_statement"
    status: str = "confirmed"
    event_at: datetime = NOW - timedelta(days=30)
    expires_at: datetime | None = None
    evidence: str | None = None
    supersedes: str | None = None
    tags: list[str] = field(default_factory=list)


def _pref(user: str, key: str, text: str, *, days_ago: int = 40) -> Seed:
    return Seed(
        f"pref-{user}-{key}",
        "semantic",
        text,
        user_scope(user),
        attribute=key,
        about_user=user,
        event_at=NOW - timedelta(days=days_ago),
        evidence=text,
    )


def named_seeds() -> list[Seed]:
    """The hand-written records the labelled queries point at."""

    seeds: list[Seed] = [
        _pref("aditya", "editor", "Aditya uses Neovim as his editor and keeps his configuration in Lua."),
        _pref("aditya", "answer_style", "Aditya prefers concise technical answers with a short rationale."),
        _pref("aditya", "code_language", "Aditya wants code examples in Python unless he asks for another language."),
        _pref("aditya", "time_zone", "Aditya works in the Asia/Kolkata time zone."),
        _pref("aditya", "commit_style", "Aditya writes commit messages in the Conventional Commits format."),
        _pref(
            "aditya", "test_runner", "Aditya runs the test suite with pytest and expects new code to ship with tests."
        ),
        _pref("priya", "editor", "Priya uses VS Code with the Vim keybindings extension."),
        _pref("priya", "answer_style", "Priya likes answers as short bullet lists with the decision first."),
        _pref("priya", "time_zone", "Priya works in the Europe/Lisbon time zone."),
        _pref("priya", "deploy_window", "Priya only deploys the billing service on Tuesday or Wednesday mornings."),
        _pref("rohan", "diet", "Rohan is vegetarian and avoids restaurants without a vegetarian menu."),
        _pref("rohan", "time_zone", "Rohan works in the America/Toronto time zone."),
        _pref("rohan", "meeting_length", "Rohan prefers meetings of thirty minutes or less."),
        _pref("rohan", "notebook", "Rohan takes notes in Obsidian and links every meeting note to its project page."),
        _pref("meera", "language", "Meera wants replies written in British English."),
        _pref("meera", "chart_style", "Meera prefers charts with a neutral palette and no gridlines."),
        _pref("meera", "code_language", "Meera writes data pipelines in Scala and wants examples in Scala."),
        _pref("meera", "working_hours", "Meera does not take meetings before 10:00 in the Australia/Sydney time zone."),
        # Superseded chains: the live record is the last one; the earlier versions stay as history.
        Seed(
            "pref-aditya-shell-old",
            "semantic",
            "Aditya uses bash as his interactive shell.",
            user_scope("aditya"),
            attribute="shell",
            about_user="aditya",
            status="superseded",
            event_at=NOW - timedelta(days=200),
        ),
        Seed(
            "pref-aditya-shell",
            "semantic",
            "Aditya switched to the fish shell and no longer uses bash interactively.",
            user_scope("aditya"),
            attribute="shell",
            about_user="aditya",
            event_at=NOW - timedelta(days=20),
            supersedes="pref-aditya-shell-old",
            evidence="I switched to fish, I don't use bash interactively any more.",
        ),
        Seed(
            "pref-priya-team-old",
            "semantic",
            "Priya is on the payments team.",
            user_scope("priya"),
            attribute="team",
            about_user="priya",
            status="superseded",
            event_at=NOW - timedelta(days=300),
        ),
        Seed(
            "pref-priya-team-mid",
            "semantic",
            "Priya moved to the platform team in March 2026.",
            user_scope("priya"),
            attribute="team",
            about_user="priya",
            status="superseded",
            event_at=NOW - timedelta(days=150),
            supersedes="pref-priya-team-old",
        ),
        Seed(
            "pref-priya-team",
            "semantic",
            "Priya leads the billing platform team since July 2026.",
            user_scope("priya"),
            attribute="team",
            about_user="priya",
            event_at=NOW - timedelta(days=60),
            supersedes="pref-priya-team-mid",
            evidence="I've been leading the billing platform team since July.",
        ),
        # Third-party facts in user scopes.
        Seed(
            "fact-aditya-manager",
            "semantic",
            "Rohan Mehta is Aditya's manager and runs the platform team.",
            user_scope("aditya"),
            attribute="role",
            entity=("person", "Rohan Mehta", "rohan mehta"),
            event_at=NOW - timedelta(days=90),
            evidence="Rohan Mehta is my manager, he runs the platform team.",
        ),
        Seed(
            "fact-priya-checklist-owner",
            "semantic",
            "Priya Nair owns the deployment checklist for the billing service.",
            user_scope("aditya"),
            attribute="owns",
            entity=("person", "Priya Nair", "priya nair"),
            event_at=NOW - timedelta(days=80),
        ),
        Seed(
            "fact-meera-pipeline",
            "semantic",
            "Meera Iyer maintains the nightly analytics pipeline that feeds the finance dashboard.",
            user_scope("rohan"),
            attribute="maintains",
            entity=("person", "Meera Iyer", "meera iyer"),
            event_at=NOW - timedelta(days=45),
        ),
        # Project and organisation facts.
        Seed(
            "proj-weave-store",
            "semantic",
            "The memory-weave project uses SQLite with FTS5 as its canonical store and rebuilds indexes from it.",
            project_scope("memory-weave"),
            attribute="store",
            entity=("project", "memory-weave", "memory-weave"),
            event_at=NOW - timedelta(days=120),
        ),
        Seed(
            "proj-weave-embedder",
            "semantic",
            "The memory-weave project embeds records with BAAI/bge-m3 at 1024 dimensions.",
            project_scope("memory-weave"),
            attribute="embedding_model",
            entity=("project", "memory-weave", "memory-weave"),
            event_at=NOW - timedelta(days=100),
        ),
        Seed(
            "proj-weave-runbook",
            "semantic",
            "The memory-weave release runbook lives in the repository under docs/runbooks/release.md.",
            project_scope("memory-weave"),
            attribute="runbook_location",
            entity=("repo", "memory-weave repo", "memory-weave repo"),
            event_at=NOW - timedelta(days=70),
        ),
        Seed(
            "proj-billing-cdc",
            "semantic",
            "The billing project chose Postgres logical replication over Debezium for change data capture.",
            project_scope("billing"),
            attribute="cdc_choice",
            entity=("project", "billing", "billing"),
            event_at=NOW - timedelta(days=140),
        ),
        Seed(
            "proj-billing-rate-limit",
            "semantic",
            "The billing API allows 600 requests per minute per tenant, and enterprise tenants get triple that.",
            project_scope("billing"),
            attribute="rate_limit",
            entity=("project", "billing", "billing"),
            event_at=NOW - timedelta(days=50),
        ),
        Seed(
            "proj-billing-freeze",
            "semantic",
            "The billing project freezes deployments during the last three business days of each quarter.",
            project_scope("billing"),
            attribute="freeze_window",
            entity=("project", "billing", "billing"),
            event_at=NOW - timedelta(days=200),
        ),
        Seed(
            "org-acme-oncall",
            "semantic",
            "Acme's on-call rotation hands over every Monday at 10:00 UTC.",
            Scope(kind="org", id=ORG),
            attribute="oncall_handover",
            entity=("org", "Acme", "acme"),
            event_at=NOW - timedelta(days=250),
        ),
        Seed(
            "org-acme-handbook",
            "semantic",
            "The Acme engineering handbook is kept in the wiki under Engineering/Handbook.",
            Scope(kind="org", id=ORG),
            attribute="handbook_location",
            entity=("org", "Acme", "acme"),
            event_at=NOW - timedelta(days=300),
        ),
        # Procedures.
        Seed(
            "proc-weave-release",
            "procedural",
            "To release memory-weave: run the full test suite, bump the version in pyproject.toml, tag the commit, and publish with uv.",  # noqa: E501
            project_scope("memory-weave"),
            attribute="release_steps",
            entity=("project", "memory-weave", "memory-weave"),
            event_at=NOW - timedelta(days=60),
        ),
        Seed(
            "proc-weave-reembed",
            "procedural",
            "To change the embedding model in memory-weave, recalibrate the gate floors first and then run the reembed command.",  # noqa: E501
            project_scope("memory-weave"),
            attribute="reembed_steps",
            entity=("project", "memory-weave", "memory-weave"),
            event_at=NOW - timedelta(days=30),
        ),
        Seed(
            "proc-billing-rollback",
            "procedural",
            "To roll back a billing deployment, redeploy the previous image tag and replay the outbox from the last checkpoint.",  # noqa: E501
            project_scope("billing"),
            attribute="rollback_steps",
            entity=("project", "billing", "billing"),
            event_at=NOW - timedelta(days=90),
        ),
        Seed(
            "proc-aditya-venv",
            "procedural",
            "Aditya sets up a project by running uv sync with the local-models extra and exporting HF_HUB_OFFLINE=1.",
            user_scope("aditya"),
            attribute="project_setup",
            about_user="aditya",
            event_at=NOW - timedelta(days=15),
        ),
        Seed(
            "proc-priya-hotfix",
            "procedural",
            "Priya's hotfix procedure is to branch from the release tag, cherry-pick the fix, and open a PR against both main and the release branch.",  # noqa: E501
            user_scope("priya"),
            attribute="hotfix_procedure",
            about_user="priya",
            event_at=NOW - timedelta(days=25),
        ),
        # Episodes.
        Seed(
            "event-weave-gate-finding",
            "episodic",
            "On 6 September 2026 the team found that the relevance gate could not separate ordinary turns from memory-needed turns.",  # noqa: E501
            project_scope("memory-weave"),
            entity=("project", "memory-weave", "memory-weave"),
            event_at=datetime(2026, 9, 6, 15, 0, tzinfo=UTC),
        ),
        Seed(
            "event-billing-outage",
            "episodic",
            "On 14 August 2026 the billing service was down for forty minutes after a migration locked the invoices table.",  # noqa: E501
            project_scope("billing"),
            entity=("project", "billing", "billing"),
            event_at=datetime(2026, 8, 14, 11, 0, tzinfo=UTC),
        ),
        Seed(
            "event-aditya-offsite",
            "episodic",
            "Aditya attended the platform offsite in Goa on 22 July 2026 and presented the memory roadmap.",
            user_scope("aditya"),
            about_user="aditya",
            event_at=datetime(2026, 7, 22, 9, 0, tzinfo=UTC),
        ),
        Seed(
            "event-priya-oncall-page",
            "episodic",
            "Priya was paged on 3 September 2026 because the invoice queue backed up past 50,000 messages.",
            user_scope("priya"),
            about_user="priya",
            event_at=datetime(2026, 9, 3, 2, 0, tzinfo=UTC),
        ),
        Seed(
            "event-rohan-hiring",
            "episodic",
            "On 28 August 2026 Rohan decided to open two backend positions on the platform team.",
            user_scope("rohan"),
            about_user="rohan",
            event_at=datetime(2026, 8, 28, 10, 0, tzinfo=UTC),
        ),
        Seed(
            "event-meera-dashboard",
            "episodic",
            "Meera shipped the finance dashboard redesign on 1 September 2026 after two rounds of review.",
            user_scope("meera"),
            about_user="meera",
            event_at=datetime(2026, 9, 1, 16, 0, tzinfo=UTC),
        ),
        Seed(
            "event-acme-allhands",
            "episodic",
            "At the Acme all-hands on 15 August 2026 the CEO announced the move to a four-day release train.",
            Scope(kind="org", id=ORG),
            entity=("org", "Acme", "acme"),
            event_at=datetime(2026, 8, 15, 14, 0, tzinfo=UTC),
        ),
        # Private agent scopes.
        Seed(
            "private-coding-aditya-branch",
            "semantic",
            "Aditya's current working branch for memory-weave is feat/utility-aware-memory.",
            private_scope("coding-agent", "aditya"),
            attribute="working_branch",
            about_user="aditya",
            event_at=NOW - timedelta(days=3),
            source_kind="agent_inference",
            status="provisional",
            expires_at=NOW + timedelta(days=27),
        ),
        Seed(
            "private-ops-rohan-cluster",
            "semantic",
            "Rohan's staging cluster for the billing service is billing-staging-eu in Frankfurt.",
            private_scope("ops-agent", "rohan"),
            attribute="staging_cluster",
            about_user="rohan",
            event_at=NOW - timedelta(days=10),
        ),
        # Expired and past-expiry rows.
        Seed(
            "expired-aditya-laptop",
            "semantic",
            "Aditya's laptop is due for replacement in March 2026.",
            user_scope("aditya"),
            attribute="laptop_replacement",
            about_user="aditya",
            source_kind="agent_inference",
            status="expired",
            event_at=NOW - timedelta(days=220),
            expires_at=NOW - timedelta(days=190),
        ),
        Seed(
            "stale-priya-ticket",
            "semantic",
            "Priya is probably still waiting on the vendor ticket about the card processor outage.",
            user_scope("priya"),
            attribute="vendor_ticket",
            about_user="priya",
            source_kind="agent_inference",
            status="provisional",
            event_at=NOW - timedelta(days=70),
            expires_at=NOW - timedelta(days=40),
        ),
    ]
    return seeds


_SERVICES = (
    "invoices",
    "payments",
    "ledger",
    "notifications",
    "search",
    "ingestion",
    "reporting",
    "identity",
    "gateway",
    "scheduler",
    "exports",
    "webhooks",
    "audit",
    "rates",
    "tax",
    "refunds",
    "disputes",
    "catalog",
    "pricing",
    "metering",
)
_REGIONS = ("eu-central", "us-east", "ap-south", "ap-southeast")
_PROJECTS = ("memory-weave", "billing")
_TOPICS = (
    "retry budgets",
    "queue depth alerts",
    "schema migrations",
    "index rebuilds",
    "cache warmup",
    "backfill jobs",
    "rate limiting",
    "canary rollouts",
    "log retention",
    "secret rotation",
    "feature flags",
    "load testing",
)
_DECISIONS = (
    "keep the default timeout at thirty seconds",
    "move the nightly job to 02:00 UTC",
    "require two reviewers on schema changes",
    "cap batch size at five hundred rows",
    "retire the legacy exporter",
    "store audit logs for ninety days",
    "add a dashboard panel for the error budget",
)


def generated_seeds() -> list[Seed]:
    """Templated distractors over the same vocabulary; deterministic and content-distinct."""

    seeds: list[Seed] = []
    day = 0
    for project in _PROJECTS:
        for service in _SERVICES:
            for region in _REGIONS:
                day += 1
                seeds.append(
                    Seed(
                        f"gen-{project}-{service}-{region}-region",
                        "semantic",
                        f"The {service} service of the {project} project runs in the {region} region.",
                        project_scope(project),
                        attribute=f"{service}_{region}_region",
                        entity=("project", project, project),
                        event_at=EPOCH + timedelta(days=day % 200),
                    )
                )
            day += 1
            seeds.append(
                Seed(
                    f"gen-{project}-{service}-limit",
                    "semantic",
                    f"The {service} service of the {project} project allows {100 + 25 * (day % 17)} requests per minute.",  # noqa: E501
                    project_scope(project),
                    attribute=f"{service}_rate_limit",
                    entity=("project", project, project),
                    event_at=EPOCH + timedelta(days=day % 200),
                )
            )
    for project in _PROJECTS:
        for topic_index, topic in enumerate(_TOPICS):
            for decision_index, decision in enumerate(_DECISIONS):
                day += 1
                when = EPOCH + timedelta(days=(topic_index * 7 + decision_index * 13 + day) % 240)
                seeds.append(
                    Seed(
                        f"gen-{project}-episode-{topic_index}-{decision_index}",
                        "episodic",
                        f"On {when.strftime('%-d %B %Y')} the {project} team discussed {topic} and agreed to {decision}.",  # noqa: E501
                        project_scope(project),
                        entity=("project", project, project),
                        event_at=when,
                    )
                )
    for user in USERS:
        for service in _SERVICES:
            day += 1
            seeds.append(
                Seed(
                    f"gen-{user}-{service}-owner",
                    "semantic",
                    f"{USER_ALIASES[user][0]} is the on-call owner for the {service} service this quarter.",
                    user_scope(user),
                    attribute=f"{service}_oncall",
                    about_user=user,
                    event_at=EPOCH + timedelta(days=day % 200),
                )
            )
        for topic_index, topic in enumerate(_TOPICS):
            day += 1
            seeds.append(
                Seed(
                    f"gen-{user}-howto-{topic_index}",
                    "procedural",
                    f"{USER_ALIASES[user][0]} handles {topic} by opening a ticket first and pairing with the owner before changing anything.",  # noqa: E501
                    user_scope(user),
                    attribute=f"{topic.replace(' ', '_')}_howto",
                    about_user=user,
                    event_at=EPOCH + timedelta(days=day % 200),
                )
            )
    for user_index, user in enumerate(USERS):
        for topic_index, topic in enumerate(_TOPICS):
            for decision_index, decision in enumerate(_DECISIONS):
                day += 1
                when = EPOCH + timedelta(days=(user_index * 31 + topic_index * 5 + decision_index * 11 + day) % 240)
                agent = AGENTS[(user_index + topic_index) % len(AGENTS)]
                seeds.append(
                    Seed(
                        f"gen-{user}-episode-{topic_index}-{decision_index}",
                        "episodic",
                        f"On {when.strftime('%-d %B %Y')} {USER_ALIASES[user][0]} paired with the {agent} on {topic} "
                        f"and decided to {decision}.",
                        user_scope(user),
                        about_user=user,
                        event_at=when,
                    )
                )
    for topic_index, topic in enumerate(_TOPICS):
        for decision_index, decision in enumerate(_DECISIONS):
            day += 1
            seeds.append(
                Seed(
                    f"gen-acme-policy-{topic_index}-{decision_index}",
                    "semantic",
                    f"Acme policy on {topic}: teams {decision}.",
                    Scope(kind="org", id=ORG),
                    attribute=f"policy_{topic_index}_{decision_index}",
                    entity=("org", "Acme", "acme"),
                    event_at=EPOCH + timedelta(days=day % 200),
                )
            )
    for project in _PROJECTS:
        for topic_index, topic in enumerate(_TOPICS):
            day += 1
            seeds.append(
                Seed(
                    f"gen-{project}-howto-{topic_index}",
                    "procedural",
                    f"In the {project} project, {topic} are handled by opening a change request and running the checklist.",  # noqa: E501
                    project_scope(project),
                    attribute=f"{topic.replace(' ', '_')}_procedure",
                    entity=("project", project, project),
                    event_at=EPOCH + timedelta(days=day % 200),
                )
            )
    for agent, users in USER_GRANTS.items():
        for user in users:
            for index in range(12):
                day += 1
                expired = index % 4 == 3
                seeds.append(
                    Seed(
                        f"gen-private-{agent}-{user}-{index}",
                        "semantic",
                        f"{USER_ALIASES[user][0]} was last seen working on {_TOPICS[index]} with the {agent}.",
                        private_scope(agent, user),
                        attribute=f"last_seen_{index}",
                        about_user=user,
                        source_kind="agent_inference",
                        status="provisional",
                        event_at=NOW - timedelta(days=index + 1),
                        expires_at=NOW - timedelta(days=1) if expired else NOW + timedelta(days=30 - index),
                    )
                )
    return seeds


def all_seeds() -> list[Seed]:
    seeds = named_seeds() + generated_seeds()
    ids = [seed.id for seed in seeds]
    assert len(ids) == len(set(ids)), "seed ids must be unique"
    return seeds


def build_fixture(path: Path, embedder: Embedder, *, config: MemoryWeaveConfig | None = None) -> int:
    """Write the fixture store at ``path`` with ``embedder`` and return the record count."""

    config = config or MemoryWeaveConfig(
        embedding=EmbeddingConfig(model=embedder.name, version=embedder.version, dims=embedder.dims)
    )
    if path.exists():
        path.unlink()
    for sidecar in (f"{path}-wal", f"{path}-shm"):
        Path(sidecar).unlink(missing_ok=True)
    store = Store(path)
    host = MemoryHost(store)
    principal_entities: dict[str, str] = {}
    for user in USERS:
        principal_entities[user] = host.provision_user(user, aliases=USER_ALIASES[user])
    for agent, users in USER_GRANTS.items():
        for user in users:
            host.grant(agent, user_scope(user), read=True, write=True)
        for project in PROJECT_GRANTS[agent]:
            host.grant(agent, project_scope(project), read=True, write=True)
        host.grant(agent, Scope(kind="org", id=ORG), read=True, write=False)
    entities: dict[tuple[str, str], str] = {}
    seeds = all_seeds()
    contents = [seed.content for seed in seeds]
    vectors = []
    batch = 64
    for start in range(0, len(contents), batch):
        vectors.extend(embedder.embed_documents(contents[start : start + batch]))
    with store.transaction():
        for seed, vector in zip(seeds, vectors, strict=True):
            entity_id: str | None = None
            aliases = ""
            if seed.about_user is not None:
                entity_id = principal_entities[seed.about_user]
                aliases = " ".join(alias.lower() for alias in USER_ALIASES[seed.about_user])
            elif seed.entity is not None:
                kind, canonical, alias = seed.entity
                key = (seed.scope.id, canonical)
                if key not in entities:
                    created = store.create_entity(kind=kind, canonical=canonical, scope=seed.scope)  # type: ignore[arg-type]
                    store.add_alias(created.id, alias)
                    entities[key] = created.id
                entity_id = entities[key]
                aliases = alias
            attribute = seed.attribute if seed.type != "episodic" else "-"
            record = Record(
                id=seed.id,
                type=seed.type,  # type: ignore[arg-type]
                version=1,
                content=seed.content,
                subject=render_subject(entity_id, attribute) if entity_id else "",
                scope=seed.scope,
                source_kind=seed.source_kind,  # type: ignore[arg-type]
                source_ref=None,
                creator_agent_id=AGENTS[len(seed.id) % len(AGENTS)],
                evidence=seed.evidence,
                created_at=seed.event_at,
                event_at=seed.event_at,
                expires_at=seed.expires_at,
                confidence=0.95 if seed.source_kind == "user_statement" else 0.6,
                status=seed.status,  # type: ignore[arg-type]
                supersedes_id=seed.supersedes,
                reinforcements=0,
                last_reinforced_at=None,
                tags=list(seed.tags),
                entity_ids=[entity_id] if entity_id else [],
                subject_entity_id=entity_id,
                attribute=attribute if entity_id else None,
            )
            store.insert_record(record)
            store.put_embedding(record.id, embedder.name, embedder.version, vector)
            store.upsert_fts(record.id, record.content, record.subject, aliases)
            if entity_id:
                store.link_record_entity(record.id, entity_id, "about")
    store.compact()
    count = store.connection.execute("SELECT count(*) FROM records").fetchone()[0]
    store.close()
    return int(count)


def copy_fixture(destination_dir: Path) -> Path:
    """Copy the committed snapshot into a scratch directory and return the copy's path."""

    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(f"{FIXTURE_PATH} is missing; build it with python -m tests.integration.fixture_1k")
    target = destination_dir / "memory_1k.sqlite"
    shutil.copyfile(FIXTURE_PATH, target)
    return target


def fixture_config() -> MemoryWeaveConfig:
    return MemoryWeaveConfig()


def scope_of(record: Record) -> Scope:
    return record.scope


def describe(seeds: list[Seed]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    by_scope: dict[str, int] = {}
    for seed in seeds:
        by_type[seed.type] = by_type.get(seed.type, 0) + 1
        by_scope[f"{seed.scope.kind}:{seed.scope.id}"] = by_scope.get(f"{seed.scope.kind}:{seed.scope.id}", 0) + 1
    return {"records": len(seeds), "by_type": by_type, "by_scope": by_scope}


if __name__ == "__main__":
    import json

    from memory_weave.index.embedder import BgeM3Embedder

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    total = build_fixture(FIXTURE_PATH, BgeM3Embedder(EmbeddingConfig()))
    print(json.dumps({"path": str(FIXTURE_PATH), **describe(all_seeds()), "written": total}, indent=1))
