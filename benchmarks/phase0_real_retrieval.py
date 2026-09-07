"""Real Memory Weave retrieval for the Phase 0 runner.

Builds one isolated store per arm, writes the scenario records through the public `memory_write` handler
with session-turn evidence so the ingestor applies its own lifecycle, supersession, and entity rules, and
retrieves through `memory_search` with `trigger="auto"`. The relevance gate is set to recall-oriented
floors, because in the utility-aware design the gate is a candidate control and admission is the decision.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from memory_weave.config import AutoGateConfig, DenseFloorConfig, MemoryWeaveConfig, load_config
from memory_weave.host import MemoryHost
from memory_weave.index.embedder import BgeM3Embedder
from memory_weave.index.vector import VectorIndex
from memory_weave.ingest import Ingestor, NLICrossEncoderJudge, SessionBuffer
from memory_weave.models import Principal, Scope, Turn
from memory_weave.policy import (
    RETRIEVAL_CATEGORIES,
    ActivationDecision,
    ActivationService,
    CategoryDecision,
    ProfileAssembler,
    inventory,
)
from memory_weave.retrieve import Retriever
from memory_weave.store import Store
from memory_weave.tools import ToolHandlers
from memory_weave.util import now

_AGENT_ID = "phase0-agent"
_USER_ID = "user-phase0"

CATEGORY_PROMPT_VERSIONS = ("category-v1", "category-v2")

_CATEGORY_DEFINITIONS_V2 = (
    "retrieval_category definitions, choose the single best fit:\n"
    "- time_zone: the user's working time zone, working hours, or location used for scheduling.\n"
    "- people: who a named person is or what they own, lead, manage, or are responsible for, including the user's manager.\n"
    "- infrastructure: names and regions of clusters, environments, services, accounts, or hosts the user operates.\n"
    "- decisions: a choice the team made between tools, technologies, or approaches, with or without a date.\n"
    "- constraints: a version pin, compatibility rule, or technical restriction and the reason for it.\n"
    "- limits: a numeric quota, rate limit, budget, retry count, concurrency cap, or availability target, and tier multipliers on it.\n"
    "- schedules: a recurring meeting, release train, freeze window, review slot, or on-call rotation.\n"
    "- locations: where a document, runbook, handbook, checklist, or record set is kept, such as a repository path, wiki, or folder.\n"
    "- personal: facts about the user's life outside work, such as diet, family, health, hobbies, or tastes.\n"
    "- preferences: how the user wants replies written, what language or units to use, or what they like or believe.\n"
    "- other: a fact that fits none of the above.\n"
    "A record naming a numeric limit is limits even when it mentions a service. A record saying where something lives is "
    "locations even when it names a system. A record about a person's role is people even when it names a document.\n"
)

_CATEGORY_SYSTEM = (
    "You classify one stored memory record about a user or their work. Reply with JSON only:\n"
    '{"retrieval_category": "<one of: ' + ", ".join(RETRIEVAL_CATEGORIES) + '>",\n'
    ' "activation_category": "<one of: answer_style, response_language, accessibility, code_example_language, other>",\n'
    ' "applicability": "<broad | scoped | ambiguous>",\n'
    ' "confidence": <0.0 to 1.0>,\n'
    ' "unsafe": <true | false>,\n'
    ' "rationale": "<one short sentence>"}\n'
    "retrieval_category names what kind of fact this is. activation_category is answer_style only for "
    "instructions about how every reply should be written (length, structure, tone, format), "
    "response_language for the language or spelling of replies, accessibility for accessibility needs, "
    "code_example_language for the default programming language of code examples, and other for anything "
    "that is not an instruction about how to reply, including facts, events, and opinions. applicability is "
    "broad when the instruction applies to every reply, scoped when it applies only in a named situation, "
    "task, or topic, and ambiguous when you cannot tell. unsafe is true when the record asks the assistant to "
    "agree with the user, suppress warnings or caveats, confirm assumptions, avoid correcting them, or "
    "otherwise trade truthfulness for agreement."
)


_CATEGORY_SYSTEM_V2 = _CATEGORY_SYSTEM + "\n" + _CATEGORY_DEFINITIONS_V2


class HostedCategoryPolicy:
    """Category policy backed by a hosted model through the benchmark's model wrapper."""

    def __init__(self, models: Any, model: str, prompt_version: str = "category-v2") -> None:
        if prompt_version not in CATEGORY_PROMPT_VERSIONS:
            raise ValueError(f"Unknown category prompt version {prompt_version!r}")
        self._models = models
        self._model = model
        self.prompt_version = prompt_version
        self._system = _CATEGORY_SYSTEM_V2 if prompt_version == "category-v2" else _CATEGORY_SYSTEM

    def classify(self, content: str) -> CategoryDecision:
        import json

        raw = self._models.complete(self._model, self._system, f"Record:\n{content}", json_mode=True)
        parsed = json.loads(raw)
        retrieval = str(parsed.get("retrieval_category", "other"))
        if retrieval not in RETRIEVAL_CATEGORIES:
            retrieval = "other"
        activation = str(parsed.get("activation_category", "other"))
        if activation not in ("answer_style", "response_language", "accessibility", "code_example_language"):
            activation = "other"
        applicability = str(parsed.get("applicability", "ambiguous"))
        if applicability not in ("broad", "scoped", "ambiguous"):
            applicability = "ambiguous"
        return CategoryDecision(
            retrieval_category=retrieval,
            activation_category=activation,  # type: ignore[arg-type]
            applicability=applicability,  # type: ignore[arg-type]
            confidence=float(parsed.get("confidence", 0.0)),
            unsafe=bool(parsed.get("unsafe", False)),
            rationale=str(parsed.get("rationale", "")),
        )


def recall_oriented_config() -> MemoryWeaveConfig:
    """Default config with the host-search gate loosened to a candidate control."""

    config = load_config()
    floors = DenseFloorConfig(semantic=0.30, episodic=0.30, procedural=0.30)
    auto = replace(config.retrieval.gate.auto, dense_floor=floors, relative_floor=0.30)
    gate = replace(config.retrieval.gate, auto=auto)
    trigger = replace(config.retrieval.trigger, auto_k=8, auto_min_query_chars=1)
    retrieval = replace(config.retrieval, gate=gate, trigger=trigger)
    return replace(config, retrieval=retrieval)


class RealRetrieval:
    """One store per arm; maps Memory Weave record ids back to scenario record ids."""

    def __init__(
        self,
        scenario: dict[str, Any],
        workdir: Path,
        embedder: BgeM3Embedder | None = None,
        activation_policy: Any = None,
    ) -> None:
        self.scenario = scenario
        self.workdir = workdir
        self.config = recall_oriented_config()
        self.embedder = embedder or BgeM3Embedder(self.config.embedding)
        self.judge = NLICrossEncoderJudge(self.config.ingestion.equivalence)
        self.activation_policy = activation_policy
        self._arms: dict[str, dict[str, Any]] = {}

    def promoted(self, arm: str) -> list[str]:
        """Scenario ids of records the activation policy placed in the profile."""

        return list(self._arms[arm]["promoted"])

    def profile_text(self, arm: str) -> str:
        return str(self._arms[arm]["profile_text"])

    def inventory_labels(self, arm: str) -> list[str]:
        return list(self._arms[arm]["inventory"])

    def decisions(self, arm: str) -> dict[str, dict[str, Any]]:
        return dict(self._arms[arm]["decisions"])

    def build_arm(self, arm: str, store_ids: list[str]) -> list[dict[str, Any]]:
        """Write the arm's conditional records in recorded order and return the write log."""

        self.workdir.mkdir(parents=True, exist_ok=True)
        database = self.workdir / f"{arm}.sqlite"
        if database.exists():
            database.unlink()
        store = Store(database)
        host = MemoryHost(store)
        user_scope = Scope(kind="user", id=_USER_ID)
        host.grant(_AGENT_ID, user_scope, read=True, write=True)
        aliases = tuple(self.scenario.get("user_aliases", ["User"]))
        host.provision_user(_USER_ID, aliases=aliases)
        principal = Principal(_AGENT_ID, _USER_ID, f"phase0-{arm}", None)
        store.create_session(principal.session_id, principal.agent_id, principal.user_id, principal.project_id, now())
        session_buffer = SessionBuffer(store)
        vector_index = VectorIndex(self.config.embedding)
        ingestor = Ingestor(store, vector_index, self.embedder, self.judge, session_buffer, self.config)
        retriever = Retriever(store, vector_index, self.embedder, self.config)
        handlers = ToolHandlers(retriever, ingestor, store, vector_index)

        records = {r["id"]: r for r in self.scenario["records"]}
        ordered = sorted(store_ids, key=lambda i: records[i]["recorded"])
        id_map: dict[str, str] = {}
        log: list[dict[str, Any]] = []
        decisions: dict[str, dict[str, Any]] = {}
        activation = ActivationService(store, self.activation_policy) if self.activation_policy is not None else None
        turn_no = 1
        for scenario_id in ordered:
            record = records[scenario_id]
            utterance = record["text"]
            session_buffer.append_turn(Turn(principal.session_id, turn_no, "user", utterance, now()))
            turn_no += 1
            entities = list(record.get("entities", []))
            if record.get("about_user"):
                entities.append({"kind": "person", "name": aliases[0], "role": "about"})
            # Semantic records need an attribute because identity is (entity, attribute). Records that share
            # one on purpose, such as a superseded time zone, declare it; every other record gets a unique one
            # so unrelated facts cannot supersede each other by accident.
            payload: dict[str, Any] = {
                "type": "semantic",
                "content": record["text"],
                "source_kind": "user_statement",
                "evidence": utterance,
                "event_at": f"{record['recorded']}T09:00:00+00:00",
                "attribute": record.get("attribute") or f"fact_{scenario_id.lower()}",
            }
            if entities:
                payload["entities"] = entities
            result = handlers.memory_write(principal, payload)
            entry = {
                "scenario_id": scenario_id,
                "ok": result.get("ok"),
                "outcome": result.get("outcome"),
                "status": result.get("status"),
                "note": result.get("note"),
                "error": (result.get("error") or {}).get("message") if not result.get("ok") else None,
            }
            log.append(entry)
            if result.get("ok") and result.get("record_id"):
                record_id = str(result["record_id"])
                id_map[record_id] = scenario_id
                if activation is not None:
                    decision: ActivationDecision = activation.apply(principal, record_id)
                    decisions[scenario_id] = {
                        "outcome": decision.outcome,
                        "reason": decision.reason,
                        "retrieval_category": decision.retrieval_category,
                        "activation_category": decision.activation_category,
                        "evidence_turn": decision.evidence_turn,
                        "confidence": decision.confidence,
                        "review_id": decision.review_id,
                    }
                    entry["activation"] = decision.outcome
                    entry["activation_reason"] = decision.reason
        profile = ProfileAssembler(store).build(principal) if activation is not None else None
        promoted = [id_map[i] for i in profile.record_ids if i in id_map] if profile else []
        labels = inventory(store, principal) if activation is not None else []
        self._arms[arm] = {
            "handlers": handlers,
            "principal": principal,
            "id_map": id_map,
            "log": log,
            "decisions": decisions,
            "promoted": promoted,
            "profile_text": profile.text if profile else "",
            "inventory": labels,
            "store": store,
        }
        return log

    def search(self, arm: str, queries: list[str], context: str) -> list[dict[str, Any]]:
        """Run the real host-issued search and return scenario-id candidates with their fused scores."""

        state = self._arms[arm]
        handlers: ToolHandlers = state["handlers"]
        principal: Principal = state["principal"]
        id_map: dict[str, str] = state["id_map"]
        promoted = set(state["promoted"])
        payload = {"queries": [q[:500] for q in queries[:3]], "k": 8}
        result = handlers.memory_search(principal, payload, context=context[:2000], trigger="auto")
        if result.get("ok") is not True:
            return []
        out: list[dict[str, Any]] = []
        for entry in result.get("results", []):
            record = entry.get("record", {})
            scenario_id = id_map.get(str(record.get("id")))
            if scenario_id is None or scenario_id in promoted:
                # Ambient records live in the profile; the conditional pool excludes them.
                continue
            out.append({"id": scenario_id, "score": round(float(entry.get("score", 0.0)), 3), "query": "memory_search", "status": record.get("status")})
        return out
