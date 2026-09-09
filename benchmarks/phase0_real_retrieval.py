"""Real Retold retrieval for the Phase 0 runner.

Builds one isolated store per arm, writes the scenario records through the public `memory_write` handler
with session-turn evidence so the ingestor applies its own lifecycle, supersession, and entity rules, and
retrieves through `memory_search` with `trigger="auto"`. The relevance gate is set to recall-oriented
floors, because in the utility-aware design the gate is a candidate control and admission is the decision.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from retold.config import RetoldConfig
from retold.host import MemoryHost
from retold.index.embedder import BgeM3Embedder
from retold.index.reranker import reranker_from_config
from retold.index.vector import VectorIndex
from retold.ingest import Ingestor, NLICrossEncoderJudge, SessionBuffer
from retold.models import Principal, Scope, Turn
from retold.policy import (
    ActivationDecision,
    ActivationService,
    ProfileAssembler,
    inventory,
)
from retold.policy.reference import (
    CATEGORY_DEFINITIONS,
    CATEGORY_SYSTEM,
    CATEGORY_SYSTEM_BASE,
    ReferenceCategoryPolicy,
    supported_retrieval_config,
)
from retold.retrieve import Retriever, rewriter_from_config
from retold.store import Store
from retold.tools import ToolHandlers
from retold.util import now

_AGENT_ID = "phase0-agent"
_USER_ID = "user-phase0"

CATEGORY_PROMPT_VERSIONS = ("category-v1", "category-v2")

_CATEGORY_DEFINITIONS_V2 = CATEGORY_DEFINITIONS

_CATEGORY_SYSTEM = CATEGORY_SYSTEM_BASE


_CATEGORY_SYSTEM_V2 = CATEGORY_SYSTEM


class HostedCategoryPolicy(ReferenceCategoryPolicy):
    """The shipped classifier over the benchmark's model wrapper, with the retired v1 prompt still runnable."""

    def __init__(self, models: Any, model: str, prompt_version: str = "category-v2") -> None:
        from benchmarks.shadow_adapter import ModelsClient

        if prompt_version not in CATEGORY_PROMPT_VERSIONS:
            raise ValueError(f"Unknown category prompt version {prompt_version!r}")
        super().__init__(ModelsClient(models, model), model)
        self.prompt_version = prompt_version
        self.policy_id = f"{model}/{prompt_version}"
        self._system = _CATEGORY_SYSTEM_V2 if prompt_version == "category-v2" else _CATEGORY_SYSTEM


def recall_oriented_config(
    *,
    rewrite_model: str | None = None,
    rewrite_timeout_ms: int | None = None,
    rerank_floor: float | None = None,
    rerank_mode: str | None = None,
    rerank_timeout_ms: int | None = None,
) -> RetoldConfig:
    """Default config with the host-search gate loosened to a candidate control.

    ``rewrite_model`` enables query rewriting through that hosted model; ``rerank_floor`` enables the
    cross-encoder reranker with that floor. Either is a new retrieval configuration and so a new bundle.
    ``rerank_mode`` picks ``rrf_cross_encoder`` or ``cross_encoder_only``; the latter raises the shortlist
    cap to the whole fused pool so the cross-encoder scores every candidate the channels produced. The
    benchmark's ``rerank_timeout_ms`` defaults to a minute, so runs measure the ranking, not the fallback;
    the shipped default is 2 s and the runs report how long the stage actually took.
    """

    config = supported_retrieval_config()
    rewrite = config.retrieval.rewrite
    if rewrite_model is not None:
        rewrite = replace(
            rewrite, enabled=True, model=rewrite_model, timeout_ms=rewrite_timeout_ms or rewrite.timeout_ms
        )
    retrieval = replace(config.retrieval, rewrite=rewrite)
    reranker = config.reranker
    if rerank_floor is not None:
        reranker = replace(reranker, enabled=True, floor=rerank_floor, timeout_ms=rerank_timeout_ms or 60000)
        if rerank_mode is not None:
            reranker = replace(reranker, mode=rerank_mode)  # type: ignore[arg-type]
        if reranker.mode == "cross_encoder_only":
            reranker = replace(reranker, candidates=3 * retrieval.per_generator_k)
    return replace(config, retrieval=retrieval, reranker=reranker)


class RealRetrieval:
    """One store per arm; maps Retold record ids back to scenario record ids."""

    def __init__(
        self,
        scenario: dict[str, Any],
        workdir: Path,
        embedder: BgeM3Embedder | None = None,
        activation_policy: Any = None,
        *,
        rewrite_model: str | None = None,
        rewrite_timeout_ms: int | None = None,
        rerank_floor: float | None = None,
        rerank_mode: str | None = None,
        rerank_timeout_ms: int | None = None,
    ) -> None:
        self.scenario = scenario
        self.workdir = workdir
        self.config = recall_oriented_config(
            rewrite_model=rewrite_model,
            rewrite_timeout_ms=rewrite_timeout_ms,
            rerank_floor=rerank_floor,
            rerank_mode=rerank_mode,
            rerank_timeout_ms=rerank_timeout_ms,
        )
        # Per arm: how many host-issued searches the cross-encoder pass applied to, timed out on, or failed on,
        # and the retriever's own rerank stage time per search, so a run reports the stage's real cost.
        self.rerank_statuses: dict[str, dict[str, int]] = {}
        self.rerank_stage_ms: dict[str, list[float]] = {}
        self.embedder = embedder or BgeM3Embedder(self.config.embedding)
        self.judge = NLICrossEncoderJudge(self.config.ingestion.equivalence)
        self.rewriter = rewriter_from_config(self.config)
        self.reranker = reranker_from_config(self.config)
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
        retriever = Retriever(
            store, vector_index, self.embedder, self.config, rewriter=self.rewriter, reranker=self.reranker
        )
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

    def _note_rerank(self, arm: str, store: Store, search_id: str) -> None:
        if not self.config.reranker.enabled or not search_id:
            return
        log = store.read_search_log(search_id)
        if log is None:
            return
        counts = self.rerank_statuses.setdefault(arm, {})
        status = str(log.get("rerank_status"))
        counts[status] = counts.get(status, 0) + 1
        timings = log.get("timings_ms") or {}
        if isinstance(timings, dict) and "rerank" in timings:
            self.rerank_stage_ms.setdefault(arm, []).append(float(timings["rerank"]))

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
        self._note_rerank(arm, state["store"], str(result.get("search_id", "")))
        out: list[dict[str, Any]] = []
        for entry in result.get("results", []):
            record = entry.get("record", {})
            scenario_id = id_map.get(str(record.get("id")))
            if scenario_id is None or scenario_id in promoted:
                # Ambient records live in the profile; the conditional pool excludes them.
                continue
            out.append(
                {
                    "id": scenario_id,
                    "score": round(float(entry.get("score", 0.0)), 3),
                    "query": "memory_search",
                    "status": record.get("status"),
                }
            )
        return out
