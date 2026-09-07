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
from memory_weave.retrieve import Retriever
from memory_weave.store import Store
from memory_weave.tools import ToolHandlers
from memory_weave.util import now

_AGENT_ID = "phase0-agent"
_USER_ID = "user-phase0"


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

    def __init__(self, scenario: dict[str, Any], workdir: Path, embedder: BgeM3Embedder | None = None) -> None:
        self.scenario = scenario
        self.workdir = workdir
        self.config = recall_oriented_config()
        self.embedder = embedder or BgeM3Embedder(self.config.embedding)
        self.judge = NLICrossEncoderJudge(self.config.ingestion.equivalence)
        self._arms: dict[str, tuple[ToolHandlers, Principal, dict[str, str], list[dict[str, Any]]]] = {}

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
                id_map[str(result["record_id"])] = scenario_id
        self._arms[arm] = (handlers, principal, id_map, log)
        return log

    def search(self, arm: str, queries: list[str], context: str) -> list[dict[str, Any]]:
        """Run the real host-issued search and return scenario-id candidates with their fused scores."""

        handlers, principal, id_map, _ = self._arms[arm]
        payload = {"queries": [q[:500] for q in queries[:3]], "k": 8}
        result = handlers.memory_search(principal, payload, context=context[:2000], trigger="auto")
        if result.get("ok") is not True:
            return []
        out: list[dict[str, Any]] = []
        for entry in result.get("results", []):
            record = entry.get("record", {})
            scenario_id = id_map.get(str(record.get("id")))
            if scenario_id is None:
                continue
            out.append({"id": scenario_id, "score": round(float(entry.get("score", 0.0)), 3), "query": "memory_search", "status": record.get("status")})
        return out
