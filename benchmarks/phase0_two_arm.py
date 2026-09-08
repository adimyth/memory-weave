"""Phase 0 of the utility-aware memory plan: two-arm validation on a hand-authored scenario set.

Runs only the proposed online path, gap planning followed by hosted draft-relative admission, over the
committed scenario set in `benchmarks/scenarios/phase0.json`. The same turns, models, and prompts run twice:

- ambient arm: ambient-eligible preferences are rendered into the baseline profile and excluded from the
  conditional store;
- conditional arm: the same preferences sit in the conditional store and must pass gap planning,
  retrieval, and admission like any other record.

Retrieval is dense-only over the scenario records with the project's BGE-M3 embedder, driven by the gap
queries rather than the turn. A placebo record is appended to every candidate set so admission is tested
against it on every candidate-bearing turn. Expected record sets and reference facts are never shown to
any policy; they are used only for scoring.

Usage:
    HF_HUB_OFFLINE=1 uv run --extra live --extra local-models python benchmarks/phase0_two_arm.py \
        --draft-model gpt-5.6-luna --policy-model gpt-5.4

    HF_HUB_OFFLINE=1 uv run --extra live --extra local-models python benchmarks/phase0_two_arm.py \
        --scenario benchmarks/scenarios/phase0_tune.json --draft-model gpt-5.6-luna \
        --gap-model gpt-4o --admission-model openrouter:anthropic/claude-sonnet-4.6 \
        --check-model gpt-5.4 --gap-prompt v2 --admission-prompt v3 --arms ambient
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.draft_delta_experiment import Models  # noqa: E402
from benchmarks.fitness import combination_verdict, serving_summary  # noqa: E402

_ADMITTING = ("helpful", "jointly_helpful")
_VERDICTS = ("helpful", "redundant", "insufficient", "stale_or_conflicting", "potentially_harmful", "jointly_helpful")
_PLACEBO_ID = "P1"
_RETRIEVAL_FLOOR = 0.30
_TOP_K_PER_QUERY = 3
_MAX_CANDIDATES = 8

_DRAFT_SYSTEM = (
    "You are a helpful assistant. Answer the user's message directly and briefly, in under 150 words. "
    "You have no memory of earlier conversations with this user beyond what is stated below."
)

_GAP_SYSTEM = (
    "You help a memory system decide what to look up before answering a user. Given the user's message and "
    "the preferences already applied, list up to 3 short search queries. Each query must describe one piece "
    "of user-specific or prior-interaction information that public knowledge cannot supply and that would "
    "materially change how the message is answered: a preference, an earlier decision, a constraint, a fact "
    "about the user's own situation, team, or systems, or something the user told the assistant before. Do "
    "not restate the topic of the message, do not ask for public facts, and do not ask for anything already "
    "covered by the applied preferences. If the message can be answered equally well for anyone, return an "
    'empty list. Reply with JSON only: {"queries": ["...", "..."]}'
)

_GAP_SYSTEM_V2 = (
    "You help a memory system decide what to look up before answering a user. You see the user's message "
    "and the preferences already applied. Decide whether the correct or appropriate response depends on "
    "facts specific to this user that public knowledge cannot supply. Ask: would two users in different "
    "situations receive different correct responses?\n"
    "- If the message asks for an explanation of a general concept, a best practice, how a tool works, or a "
    "generic how-to, the answer is the same for everyone. Return an empty list, even if the user's "
    "organisation might hold related records.\n"
    "- If the message asks the assistant to act on the user's behalf or produce something tailored to their "
    "situation, such as proposing or scheduling a time, writing a message to a specific person or role, "
    "producing a command or code for their own systems or project, checking a plan against their team's "
    "calendar, freezes, or policies, or stating what their team decided, uses, or owns, then the user-specific "
    "inputs to that task are gaps: their time zone, the people and roles involved, system and environment "
    "names, team schedules, prior decisions, and tooling or language choices for their project.\n"
    "List up to 3 gaps as short search queries that name the missing fact, not the topic. Do not ask for "
    "public facts or for anything already covered by the applied preferences. Reply with JSON only: "
    '{"queries": ["...", "..."]}'
)

_GAP_SYSTEM_V3 = (
    _GAP_SYSTEM_V2
    + "\n\nThe memory store for this user holds facts in these categories only, and nothing outside them:\n"
    "{inventory}\n"
    "A question about a decision, constraint, limit, schedule, location, system, or person that falls in one "
    "of these categories may need memory even when it does not say my, our, or we. A question that asks how "
    "something works in general, or how to do something in general, does not."
)

_GAP_PROMPTS = {"v1": _GAP_SYSTEM, "v2": _GAP_SYSTEM_V2, "v3": _GAP_SYSTEM_V3}

_ADMISSION_SYSTEM = (
    "You decide which stored memory records, if any, should be given to an assistant before it answers. You "
    "see the user's message, the preferences already applied, a draft answer written without any of the "
    "candidate records, and the candidate records with their recorded date and status. Evaluate the "
    "candidates together. Give every candidate exactly one verdict:\n"
    "- helpful: on its own it would materially improve the answer's correctness, personalisation, or next "
    "action compared with the draft.\n"
    "- jointly_helpful: it helps only in combination with another candidate you are also admitting.\n"
    "- redundant: it repeats what the draft, public knowledge, or the applied preferences already cover.\n"
    "- insufficient: it concerns the request but cannot support a material change to the answer.\n"
    "- stale_or_conflicting: it is superseded, outdated, or contradicted by a stronger candidate.\n"
    "- potentially_harmful: using it risks factual distortion, agreeing with a false belief, inappropriate "
    "personalisation, or exposing private information unrelated to the request.\n"
    "Admit only candidates with verdict helpful or jointly_helpful. When improvement is uncertain, do not "
    'admit. An empty admitted list is the normal outcome. Reply with JSON only: {"admitted": ["<id>", ...], '
    '"verdicts": [{"id": "<id>", "verdict": "<verdict>", "reason": "<one short sentence>"}]}. Include every '
    "candidate id exactly once in verdicts."
)

_ADMISSION_SYSTEM_V2 = (
    "You decide which stored memory records, if any, should be given to an assistant before it answers. You "
    "see the user's message, the preferences already applied, a draft answer written without any of the "
    "candidate records, and the candidate records with their recorded date and status. Evaluate the "
    "candidates together. Give every candidate exactly one verdict:\n"
    "- helpful: the requested answer itself would say something different with this record: a fact, value, "
    "name, date, choice, or recommendation stated in the answer would change, or the draft asked the user "
    "for exactly this information.\n"
    "- jointly_helpful: it meets the helpful test only in combination with another candidate you are also "
    "admitting.\n"
    "- redundant: it repeats what the draft, public knowledge, or the applied preferences already cover.\n"
    "- insufficient: it concerns the request but the requested answer would not say anything different. "
    "Background, attribution, context the user did not ask for, and usefulness for possible follow-up work "
    "are insufficient, not helpful.\n"
    "- stale_or_conflicting: it is marked superseded or conflicting, or a stronger candidate contradicts it. "
    "Disagreeing with the draft is not a reason for this verdict; the draft was written without the records.\n"
    "- potentially_harmful: using it risks factual distortion, agreeing with a false belief, suppressing "
    "warnings, inappropriate personalisation, or exposing private information unrelated to the request.\n"
    "Admit only candidates with verdict helpful or jointly_helpful. When you are not sure the answer would "
    'change, do not admit. An empty admitted list is the normal outcome. Reply with JSON only: {"admitted": '
    '["<id>", ...], "verdicts": [{"id": "<id>", "verdict": "<verdict>", "reason": "<one short sentence>"}]}. '
    "Include every candidate id exactly once in verdicts."
)

_ADMISSION_SYSTEM_V3 = (
    "You decide which stored memory records, if any, should be given to an assistant before it answers. You "
    "see the user's message, the preferences already applied, a draft answer written without any of the "
    "candidate records, and the candidate records with their recorded date and status. Evaluate the "
    "candidates together. Give every candidate exactly one verdict:\n"
    "- helpful: the record changes what the answer recommends, the time or date it proposes, the person it "
    "addresses or names, the command or code it gives, a constraint or value it states, or a warning it "
    "should raise. The draft asking the user for exactly this information also counts.\n"
    "- jointly_helpful: it meets the helpful test only together with another candidate you are also admitting.\n"
    "- redundant: it repeats what the draft, public knowledge, or the applied preferences already cover.\n"
    "- insufficient: it concerns the request but changes none of the things listed under helpful. Background, "
    "attribution, and usefulness for other work the user did not ask about are insufficient.\n"
    "- stale_or_conflicting: it is marked superseded or conflicting, or a stronger candidate contradicts it. "
    "Disagreeing with the draft is not a reason for this verdict; the draft was written without the records.\n"
    "- potentially_harmful: using it risks factual distortion, agreeing with a false belief, suppressing "
    "warnings, inappropriate personalisation, or exposing private information unrelated to the request.\n"
    "Admit only candidates with verdict helpful or jointly_helpful. When you are not sure one of the listed "
    "things would change, do not admit. An empty admitted list is the normal outcome. Reply with JSON only: "
    '{"admitted": ["<id>", ...], "verdicts": [{"id": "<id>", "verdict": "<verdict>", "reason": "<one short '
    'sentence>"}]}. Include every candidate id exactly once in verdicts.'
)

_ADMISSION_SYSTEM_V4 = (
    _ADMISSION_SYSTEM_V3
    + "\n\nGap anchoring. Before retrieval, a planner named the specific missing facts it was looking for; they "
    "are listed as numbered gaps in the message. A candidate can be admitted only if it resolves one of those "
    'gaps. For every candidate you admit, add "resolves_gap": <gap number> to its verdict entry. A candidate '
    "that is useful but does not resolve a listed gap is insufficient, however relevant it seems."
)

_ADMISSION_PROMPTS = {
    "v1": _ADMISSION_SYSTEM,
    "v2": _ADMISSION_SYSTEM_V2,
    "v3": _ADMISSION_SYSTEM_V3,
    "v4": _ADMISSION_SYSTEM_V4,
}
_GAP_ANCHORED_PROMPTS = {"v4"}

_CORRECTNESS_SYSTEM = (
    "You check whether an assistant's answer contains a set of reference facts. Reply with JSON only: "
    '{"contains_reference": true|false, "contradicts_reference": true|false, "note": "<one short sentence>"}. '
    "contains_reference is true only if the answer states the substance of every reference fact. "
    "contradicts_reference is true if the answer asserts something incompatible with the reference."
)


@dataclass(slots=True)
class Timing:
    draft_s: float = 0.0
    gap_s: float = 0.0
    retrieval_s: float = 0.0
    admission_s: float = 0.0
    regeneration_s: float = 0.0

    @property
    def added_s(self) -> float:
        overhang = max(0.0, self.gap_s - self.draft_s)
        return overhang + self.retrieval_s + self.admission_s + self.regeneration_s


@dataclass(slots=True)
class TurnResult:
    arm: str
    turn_id: str
    turn_class: str
    turn: str
    expected: list[str]
    draft: str
    gaps: list[str]
    candidates: list[dict[str, Any]]
    verdicts: list[dict[str, Any]]
    admitted: list[str]
    disposition: str
    final: str | None
    draft_correct: bool | None
    final_correct: bool | None
    timing: Timing
    failures: list[str] = field(default_factory=list)
    gap_repeats_nonempty: list[bool] = field(default_factory=list)
    shadow_candidates: list[dict[str, Any]] = field(default_factory=list)
    shadow_verdicts: list[dict[str, Any]] = field(default_factory=list)
    shadow_admitted: list[str] = field(default_factory=list)


class Phase0:
    def __init__(
        self,
        scenario: dict[str, Any],
        draft_model: str,
        gap_model: str,
        admission_model: str,
        check_model: str,
        gap_prompt: str,
        gap_repeats: int,
        workers: int,
        draft_cache: Path | None,
        admission_prompt: str = "v1",
        retrieval: str = "dense",
        shadow_judge: bool = False,
        real_workdir: Path | None = None,
        activation: str = "fixture",
        activation_model: str = "gpt-4o",
        category_prompt: str = "category-v1",
        rewrite_model: str | None = None,
        rewrite_timeout_ms: int | None = None,
        rerank_floor: float | None = None,
    ) -> None:
        self.scenario = scenario
        self.rewrite_model = rewrite_model
        self.rewrite_timeout_ms = rewrite_timeout_ms
        self.rerank_floor = rerank_floor
        self.draft_model = draft_model
        self.gap_model = gap_model
        self.admission_model = admission_model
        self.check_model = check_model
        self.gap_prompt = gap_prompt
        self.admission_prompt = admission_prompt
        self.retrieval = retrieval
        self.shadow_judge = shadow_judge
        self._real: Any = None
        self._real_workdir = real_workdir or Path("benchmarks/results/phase0/stores")
        self.write_logs: dict[str, list[dict[str, Any]]] = {}
        self.activation = activation
        self.activation_model = activation_model
        self.category_prompt = category_prompt
        self.promoted: dict[str, list[str]] = {}
        self.activation_decisions: dict[str, dict[str, dict[str, Any]]] = {}
        self.inventory_labels: dict[str, list[str]] = {}
        self._inventory_override: dict[str, str] = {}
        self.gap_repeats = max(1, gap_repeats)
        self.workers = workers
        self.models = Models()
        self.records = {r["id"]: r for r in scenario["records"]}
        self._embedder = None
        self._doc_vectors: dict[str, Any] = {}
        self._draft_cache_path = draft_cache
        self._draft_cache: dict[str, list[Any]] = {}
        if draft_cache and draft_cache.exists():
            self._draft_cache = json.loads(draft_cache.read_text())
        import threading

        self._cache_lock = threading.Lock()

    def save_cache(self) -> None:
        if self._draft_cache_path:
            self._draft_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._draft_cache_path.write_text(json.dumps(self._draft_cache))

    # -- embeddings -------------------------------------------------------------------------------------

    def _embed_store(self, store_ids: list[str]) -> None:
        from memory_weave.config import EmbeddingConfig
        from memory_weave.index.embedder import BgeM3Embedder

        if self._embedder is None:
            self._embedder = BgeM3Embedder(EmbeddingConfig())
        missing = [i for i in store_ids if i not in self._doc_vectors]
        if missing:
            vectors = self._embedder.embed_documents([self.records[i]["text"] for i in missing])
            self._doc_vectors.update(zip(missing, vectors, strict=True))

    def _inventory_text(self, store_ids: list[str]) -> str:
        """Content-free list of the fact categories present in the conditional store."""

        labels = self.scenario.get("categories", {})
        present = sorted({self.records[i].get("category", "") for i in store_ids} - {""})
        if not present:
            return "- (no category information available)"
        return "\n".join(f"- {labels.get(c, c)}" for c in present)

    def retrieve_any(self, arm: str, queries: list[str], store_ids: list[str], context: str) -> list[dict[str, Any]]:
        if self.retrieval == "real":
            return self._real.search(arm, queries, context)
        return self.retrieve(queries, store_ids)

    def retrieve(self, queries: list[str], store_ids: list[str]) -> list[dict[str, Any]]:
        assert self._embedder is not None
        query_vectors = self._embedder.embed_queries(queries)
        best: dict[str, dict[str, Any]] = {}
        for query, vector in zip(queries, query_vectors, strict=True):
            scored = sorted(((float(self._doc_vectors[i] @ vector), i) for i in store_ids), reverse=True)[
                :_TOP_K_PER_QUERY
            ]
            for score, record_id in scored:
                if score < _RETRIEVAL_FLOOR:
                    continue
                if record_id not in best or score > best[record_id]["score"]:
                    best[record_id] = {"id": record_id, "score": round(score, 3), "query": query}
        ranked = sorted(best.values(), key=lambda h: -h["score"])[:_MAX_CANDIDATES]
        return ranked

    # -- prompts ----------------------------------------------------------------------------------------

    @staticmethod
    def _profile_text(profile: list[dict[str, Any]]) -> str:
        if not profile:
            return "Applied preferences: none."
        return "Applied preferences:\n" + "\n".join(f"- {r['text']}" for r in profile)

    def _timed(self, model: str, system: str, user: str, *, json_mode: bool = False) -> tuple[str, float]:
        started = time.perf_counter()
        text = self.models.complete(model, system, user, json_mode=json_mode)
        return text, time.perf_counter() - started

    def _draft(self, system: str, user: str) -> tuple[str, float]:
        """Draft calls are cached on disk so every configuration is judged against the same drafts."""

        import hashlib

        key = hashlib.sha256(f"{self.draft_model}\n{system}\n{user}".encode()).hexdigest()
        with self._cache_lock:
            hit = self._draft_cache.get(key)
        if hit is not None:
            return str(hit[0]), float(hit[1])
        text, elapsed = self._timed(self.draft_model, system, user)
        with self._cache_lock:
            self._draft_cache[key] = [text, elapsed]
        return text, elapsed

    def _candidate_block(self, candidates: list[dict[str, Any]]) -> str:
        lines = []
        for c in candidates:
            r = self.records[c["id"]]
            status = r["status"] + (f", {r['note']}" if r.get("note") else "")
            lines.append(f"- id={c['id']} (recorded {r['recorded']}, status: {status}): {r['text']}")
        return "\n".join(lines)

    # -- one turn ---------------------------------------------------------------------------------------

    def _judge(
        self,
        profile_text: str,
        turn_text: str,
        draft: str,
        candidates: list[dict[str, Any]],
        gaps: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[str], float]:
        """One joint admission call. Returns (verdicts, admitted ids, seconds). Raises on malformed output."""

        anchored = self.admission_prompt in _GAP_ANCHORED_PROMPTS
        gap_block = ""
        if anchored:
            listed = "\n".join(f"{i}. {g}" for i, g in enumerate(gaps or [], start=1)) or "(none)"
            gap_block = f"\n\nGaps the planner named:\n{listed}"
        admission_user = (
            f"{profile_text}\n\nUser message:\n{turn_text}{gap_block}\n\n"
            f"Draft answer written without the candidates:\n{draft}\n\n"
            f"Candidate records:\n{self._candidate_block(candidates)}"
        )
        raw, seconds = self._timed(
            self.admission_model, _ADMISSION_PROMPTS[self.admission_prompt], admission_user, json_mode=True
        )
        parsed = json.loads(raw)
        known = {c["id"] for c in candidates}
        verdicts = []
        resolves: dict[str, int | None] = {}
        for item in parsed.get("verdicts", []):
            verdict = str(item.get("verdict", "")).strip().lower()
            if str(item.get("id")) in known:
                entry = {
                    "id": str(item["id"]),
                    "verdict": verdict if verdict in _VERDICTS else f"unparsed:{verdict}",
                    "reason": str(item.get("reason", "")),
                }
                if anchored:
                    try:
                        entry["resolves_gap"] = (
                            int(item.get("resolves_gap")) if item.get("resolves_gap") is not None else None
                        )
                    except (TypeError, ValueError):
                        entry["resolves_gap"] = None
                    resolves[entry["id"]] = entry["resolves_gap"]
                verdicts.append(entry)
        admitted = [str(i) for i in parsed.get("admitted", []) if str(i) in known]
        admitted = [i for i in admitted if any(v["id"] == i and v["verdict"] in _ADMITTING for v in verdicts)]
        if anchored:
            valid = range(1, len(gaps or []) + 1)
            admitted = [i for i in admitted if resolves.get(i) in valid]
        return verdicts, admitted, seconds

    def run_turn(
        self, arm: str, turn: dict[str, Any], profile: list[dict[str, Any]], store_ids: list[str]
    ) -> TurnResult:
        timing = Timing()
        failures: list[str] = []
        profile_text = self._profile_text(profile)
        draft_user = turn["text"]
        gap_user = f"{profile_text}\n\nUser message:\n{turn['text']}"

        gap_system = _GAP_PROMPTS[self.gap_prompt]
        if "{inventory}" in gap_system:
            override = self._inventory_override.get(arm)
            gap_system = gap_system.replace(
                "{inventory}", override if override is not None else self._inventory_text(store_ids)
            )

        def plan() -> list[str]:
            raw, _ = self._timed(self.gap_model, gap_system, gap_user, json_mode=True)
            return [str(q).strip() for q in json.loads(raw).get("queries", []) if str(q).strip()][:3]

        with ThreadPoolExecutor(max_workers=2) as pool:
            draft_future = pool.submit(self._draft, f"{_DRAFT_SYSTEM}\n\n{profile_text}", draft_user)
            gap_future = pool.submit(self._timed, self.gap_model, gap_system, gap_user, json_mode=True)
            draft, timing.draft_s = draft_future.result()
            try:
                gap_raw, timing.gap_s = gap_future.result()
                gaps = [str(q).strip() for q in json.loads(gap_raw).get("queries", []) if str(q).strip()][:3]
            except Exception as error:  # noqa: BLE001
                failures.append(f"gap:{type(error).__name__}")
                gaps = []

        repeats = [bool(gaps)]
        for _ in range(self.gap_repeats - 1):
            try:
                repeats.append(bool(plan()))
            except Exception:  # noqa: BLE001
                repeats.append(False)

        result = TurnResult(
            arm,
            turn["id"],
            turn["class"],
            turn["text"],
            list(turn["expected"]),
            draft,
            gaps,
            [],
            [],
            [],
            "baseline_no_gaps",
            None,
            None,
            None,
            timing,
            failures,
            repeats,
        )
        if not gaps:
            if self.shadow_judge:
                # Evaluation only: what would the judge do if the planner had fired? Retrieve on the raw turn,
                # which is the old relevance path, and judge without applying the result.
                shadow = self.retrieve_any(arm, [turn["text"]], store_ids, turn["text"])
                if _PLACEBO_ID in store_ids and all(c["id"] != _PLACEBO_ID for c in shadow):
                    shadow.append({"id": _PLACEBO_ID, "score": None, "query": "forced placebo"})
                result.shadow_candidates = shadow
                if shadow:
                    try:
                        result.shadow_verdicts, result.shadow_admitted, _ = self._judge(
                            profile_text, turn["text"], draft, shadow
                        )
                    except Exception as error:  # noqa: BLE001
                        failures.append(f"shadow:{type(error).__name__}")
            self._score_correctness(result, turn)
            return result

        started = time.perf_counter()
        candidates = self.retrieve_any(arm, gaps, store_ids, turn["text"])
        timing.retrieval_s = time.perf_counter() - started
        if _PLACEBO_ID in store_ids and all(c["id"] != _PLACEBO_ID for c in candidates):
            candidates.append({"id": _PLACEBO_ID, "score": None, "query": "forced placebo"})
        result.candidates = candidates
        if not candidates:
            result.disposition = "baseline_no_candidates"
            self._score_correctness(result, turn)
            return result

        try:
            verdicts, admitted, timing.admission_s = self._judge(profile_text, turn["text"], draft, candidates, gaps)
            result.verdicts = verdicts
            result.admitted = admitted
        except Exception as error:  # noqa: BLE001
            failures.append(f"admission:{type(error).__name__}")
            result.disposition = "baseline_policy_failure"
            self._score_correctness(result, turn)
            return result

        if not admitted:
            result.disposition = "baseline_empty_admission"
            self._score_correctness(result, turn)
            return result

        evidence = "\n".join(f"- {self.records[i]['text']}" for i in admitted)
        final_system = f"{_DRAFT_SYSTEM}\n\n{profile_text}\n\nRelevant facts recalled from memory:\n{evidence}"
        result.final, timing.regeneration_s = self._draft(final_system, draft_user)
        result.disposition = "regenerated"
        self._score_correctness(result, turn)
        return result

    def _score_correctness(self, result: TurnResult, turn: dict[str, Any]) -> None:
        reference = turn.get("reference")
        if not reference:
            return
        result.draft_correct = self._contains(turn["text"], reference, result.draft)
        if result.final is not None:
            result.final_correct = self._contains(turn["text"], reference, result.final)

    def _contains(self, question: str, reference: str, answer: str) -> bool | None:
        try:
            raw, _ = self._timed(
                self.check_model,
                _CORRECTNESS_SYSTEM,
                f"Question:\n{question}\n\nReference facts:\n{reference}\n\nAnswer:\n{answer}",
                json_mode=True,
            )
            return bool(json.loads(raw).get("contains_reference"))
        except Exception:  # noqa: BLE001
            return None

    # -- one arm ----------------------------------------------------------------------------------------

    def run_arm(self, arm: str) -> list[TurnResult]:
        ambient_ids = [r["id"] for r in self.scenario["records"] if r["kind"] == "ambient_eligible"]
        all_ids = [r["id"] for r in self.scenario["records"]]
        real_activation = self.retrieval == "real" and self.activation == "real" and arm == "ambient"
        if real_activation:
            # Every record is written conditional; the activation policy decides what becomes ambient.
            profile: list[dict[str, Any]] = []
            store_ids = list(all_ids)
        elif arm == "ambient":
            profile = [self.records[i] for i in ambient_ids]
            store_ids = [i for i in all_ids if i not in ambient_ids]
        else:
            profile = []
            store_ids = list(all_ids)
        self._embed_store(store_ids)
        if self.retrieval == "real":
            from benchmarks.phase0_real_retrieval import HostedCategoryPolicy, RealRetrieval

            if self._real is None:
                policy = (
                    HostedCategoryPolicy(self.models, self.activation_model, self.category_prompt)
                    if self.activation == "real"
                    else None
                )
                self._real = RealRetrieval(
                    self.scenario,
                    self._real_workdir,
                    self._embedder,
                    policy,
                    rewrite_model=self.rewrite_model,
                    rewrite_timeout_ms=self.rewrite_timeout_ms,
                    rerank_floor=self.rerank_floor,
                )
            self.write_logs[arm] = self._real.build_arm(arm, store_ids)
            outcomes = {}
            for entry in self.write_logs[arm]:
                key = f"{entry['outcome']}/{entry['status']}"
                outcomes[key] = outcomes.get(key, 0) + 1
            print(f"[{arm}] real store written: {outcomes}", flush=True)
            if real_activation:
                promoted = self._real.promoted(arm)
                self.promoted[arm] = promoted
                self.activation_decisions[arm] = self._real.decisions(arm)
                profile = [self.records[i] for i in promoted]
                store_ids = [i for i in all_ids if i not in promoted]
                labels = self._real.inventory_labels(arm)
                self.inventory_labels[arm] = labels
                self._inventory_override[arm] = (
                    "\n".join(f"- {label}" for label in labels) or "- (no category information available)"
                )
                print(f"[{arm}] promoted to ambient: {promoted}", flush=True)
                print(f"[{arm}] store-generated inventory: {labels}", flush=True)
                summary = {}
                for d in self.activation_decisions[arm].values():
                    summary[d["outcome"] + ":" + d["reason"]] = summary.get(d["outcome"] + ":" + d["reason"], 0) + 1
                print(f"[{arm}] activation decisions: {summary}", flush=True)
        turns = self.scenario["turns"]
        results: list[TurnResult] = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for result in pool.map(lambda t: self.run_turn(arm, t, profile, store_ids), turns):
                results.append(result)
                print(
                    f"[{arm}] {result.turn_id:<4} {result.turn_class:<13} gaps={len(result.gaps)} "
                    f"cands={len(result.candidates)} admitted={result.admitted} {result.disposition} "
                    f"added={result.timing.added_s:.1f}s",
                    flush=True,
                )
        return results


# -- scoring ----------------------------------------------------------------------------------------------


def _pct(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}" + (f" ({100 * numerator / denominator:.0f}%)" if denominator else "")


def _p(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[index]


def load_allowed(overlay: Path | None, scenario_stem: str) -> dict[str, set[str]]:
    """Records an adjudication overlay allows for precision on each turn of one scenario, or nothing."""

    if overlay is None or not overlay.exists():
        return {}
    data = json.loads(overlay.read_text())
    allowed: dict[str, set[str]] = {}
    for key, ids in data.get("allowed", {}).items():
        stem, _, turn_id = key.partition("/")
        if stem == scenario_stem:
            allowed.setdefault(turn_id, set()).update(ids)
    return allowed


def summarise(
    arm: str,
    results: list[TurnResult],
    records: dict[str, dict[str, Any]],
    promoted: list[str] | None = None,
    decisions: dict[str, dict[str, Any]] | None = None,
    inventory_labels: list[str] | None = None,
    allowed: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Score one arm. Recall counts only the scenario's expected (required) records; precision also counts
    records an adjudication overlay marks helpful for the turn, and the strict label-only figure is kept."""

    allowed = allowed or {}
    by_class: dict[str, list[TurnResult]] = {}
    for r in results:
        by_class.setdefault(r.turn_class, []).append(r)
    ordinary = by_class.get("ordinary", [])
    explicit = by_class.get("explicit_fact", [])
    implicit = by_class.get("implicit_need", [])

    if promoted is not None:
        ambient_ids = set(promoted)
    else:
        ambient_ids = {i for i, r in records.items() if r["kind"] == "ambient_eligible"}

    promotion: dict[str, Any] = {}
    if promoted is not None:
        eligible = [i for i, r in records.items() if r.get("ambient_truth") is True]
        unsafe = [i for i, r in records.items() if r.get("unsafe_promotion") is True]
        scoped = [i for i, r in records.items() if r.get("ambient_truth") is False and r["kind"] == "conditional"]
        truth = {i: r["category_truth"] for i, r in records.items() if r.get("category_truth")}
        assigned = {i: (decisions or {}).get(i, {}).get("retrieval_category") for i in truth}
        # Inventory coverage: for each memory-needed turn, did the inventory the planner saw contain the true
        # category of the conditional record it needed? This is the metric that matters; label agreement is not.
        from memory_weave.policy import RETRIEVAL_CATEGORIES

        covered = 0
        needed = 0
        for r in results:
            if r.turn_class == "ordinary":
                continue
            for expected in r.expected:
                if expected in ambient_ids or expected not in truth:
                    continue
                needed += 1
                label = RETRIEVAL_CATEGORIES.get(truth[expected], truth[expected])
                if label in (inventory_labels or []):
                    covered += 1
        promotion = {
            "eligible_preferences_promoted": _pct(sum(1 for i in eligible if i in ambient_ids), len(eligible)),
            "unsafe_promotions": sum(1 for i in unsafe if i in ambient_ids),
            "scoped_preferences_kept_conditional": _pct(sum(1 for i in scoped if i not in ambient_ids), len(scoped)),
            "reviews_opened": sum(1 for d in (decisions or {}).values() if d.get("outcome") == "review"),
            "inventory_coverage_of_needed_records": _pct(covered, needed) if inventory_labels is not None else "n/a",
            "retrieval_category_label_agreement": _pct(
                sum(1 for i in truth if assigned.get(i) == truth[i]), len(truth)
            ),
        }

    def injected(r: TurnResult) -> bool:
        return bool(r.admitted)

    def conditional_expected(r: TurnResult) -> list[str]:
        # In the ambient arm an ambient record is in the profile, so it cannot be a retrieval miss.
        return [e for e in r.expected if not (arm == "ambient" and e in ambient_ids)]

    def recalled(r: TurnResult) -> bool:
        expected = conditional_expected(r)
        return bool(expected) and all(e in r.admitted for e in expected)

    # Turns whose only expected record is ambient leave the recall denominator in the ambient arm; their
    # answer-correctness check still measures whether the profile did its job.
    explicit = [r for r in explicit if conditional_expected(r)]
    implicit = [r for r in implicit if conditional_expected(r)]

    admitted_total = sum(len(r.admitted) for r in results)
    admitted_useful_strict = sum(sum(1 for i in r.admitted if i in r.expected) for r in results)
    admitted_useful = sum(
        sum(1 for i in r.admitted if i in r.expected or i in allowed.get(r.turn_id, set())) for r in results
    )
    kinds_admitted: dict[str, int] = {}
    for r in results:
        for i in r.admitted:
            kinds_admitted[records[i]["kind"]] = kinds_admitted.get(records[i]["kind"], 0) + 1
    leakage = sum(1 for r in results for i in r.admitted if i in ("F9", "F10"))

    empty_gap = [r.timing.added_s for r in results if not r.gaps]
    gap_turns = [r.timing.added_s for r in results if r.gaps]

    with_ref = [r for r in results if r.draft_correct is not None]
    memory_ref = [r for r in with_ref if r.turn_class != "ordinary"]

    def final_or_draft(r: TurnResult) -> bool:
        return bool(r.final_correct) if r.final is not None else bool(r.draft_correct)

    shadowed = [r for r in results if r.shadow_candidates]
    shadow_ordinary = [r for r in shadowed if r.turn_class == "ordinary"]
    shadow_ordinary_admit = sum(1 for r in shadow_ordinary if r.shadow_admitted)
    shadow_unsafe = sum(
        1
        for r in shadowed
        for i in r.shadow_admitted
        if records[i]["kind"] in ("placebo", "misleading") or i in ("F9", "F10")
    )

    repeated = [r for r in results if len(r.gap_repeats_nonempty) > 1]
    agree = sum(1 for r in repeated if len(set(r.gap_repeats_nonempty)) == 1)
    memory_turns = explicit + implicit
    any_repeat = sum(1 for r in memory_turns if any(r.gap_repeats_nonempty))
    all_repeat = sum(1 for r in memory_turns if r.gap_repeats_nonempty and all(r.gap_repeats_nonempty))
    ordinary_any = sum(1 for r in ordinary if any(r.gap_repeats_nonempty))

    return {
        "arm": arm,
        "turns": len(results),
        "gap_decision_agreement_across_repeats": _pct(agree, len(repeated)) if repeated else "n/a",
        "memory_turns_with_gaps_every_repeat": _pct(all_repeat, len(memory_turns)),
        "memory_turns_with_gaps_any_repeat": _pct(any_repeat, len(memory_turns)),
        "ordinary_turns_with_gaps_any_repeat": _pct(ordinary_any, len(ordinary)),
        "ordinary_injection": _pct(sum(injected(r) for r in ordinary), len(ordinary)),
        "explicit_fact_recall": _pct(sum(recalled(r) for r in explicit), len(explicit)),
        "implicit_need_recall": _pct(sum(recalled(r) for r in implicit), len(implicit)),
        "ambient_pref_admitted_on_turns": _pct(
            sum(1 for r in results if any(i in ambient_ids for i in r.admitted)), len(results)
        ),
        "placebo_admitted": kinds_admitted.get("placebo", 0),
        "misleading_admitted": kinds_admitted.get("misleading", 0),
        "stale_or_conflicting_admitted": kinds_admitted.get("stale", 0) + kinds_admitted.get("conflicting", 0),
        "redundant_admitted": kinds_admitted.get("redundant", 0),
        "unrelated_private_admitted": leakage,
        "usefulness_precision": _pct(admitted_useful, admitted_total),
        "usefulness_precision_strict": _pct(admitted_useful_strict, admitted_total),
        "answer_correct_memory_turns_draft": _pct(sum(bool(r.draft_correct) for r in memory_ref), len(memory_ref)),
        "answer_correct_memory_turns_final": _pct(sum(final_or_draft(r) for r in memory_ref), len(memory_ref)),
        "empty_gap_turns": len(empty_gap),
        "added_latency_empty_gap_p50_p95_s": (round(_p(empty_gap, 0.5), 2), round(_p(empty_gap, 0.95), 2)),
        "added_latency_gap_turns_p50_p95_s": (round(_p(gap_turns, 0.5), 2), round(_p(gap_turns, 0.95), 2)),
        "draft_latency_p50_s": round(_p([r.timing.draft_s for r in results], 0.5), 2),
        "policy_failures": sum(len(r.failures) for r in results),
        "shadow_judge_ordinary_turns_with_candidates": len(shadow_ordinary),
        "shadow_judge_ordinary_turns_it_would_inject": _pct(shadow_ordinary_admit, len(shadow_ordinary))
        if shadow_ordinary
        else "n/a",
        "shadow_judge_unsafe_admissions": shadow_unsafe if shadowed else "n/a",
        **promotion,
    }


def gates(summary_ambient: dict[str, Any], summary_conditional: dict[str, Any]) -> list[str]:
    def ratio(text: str) -> float:
        n, d = text.split(" ")[0].split("/")
        return int(n) / int(d) if int(d) else 0.0

    lines = []
    a = summary_ambient
    design = [
        ("ordinary injection <= 5%", ratio(a["ordinary_injection"]) <= 0.05),
        ("explicit stored-fact recall >= 90%", ratio(a["explicit_fact_recall"]) >= 0.90),
        ("no placebo admitted", a["placebo_admitted"] == 0),
        ("no misleading admitted", a["misleading_admitted"] == 0),
    ]
    for name, ok in design:
        lines.append(f"  design gate  {'PASS' if ok else 'FAIL'}  {name}")
    lines.append(f"  design gate  {'PASS' if all(ok for _, ok in design) else 'FAIL'}  overall")
    c = summary_conditional
    promo_breach = ratio(c["ordinary_injection"]) > 0.05
    promo_recall = ratio(c["ambient_pref_admitted_on_turns"])
    lines.append(
        f"  promotion gate: conditional arm ordinary injection {c['ordinary_injection']} "
        f"({'breaches' if promo_breach else 'within'} 5%); "
        f"style/language admitted on {c['ambient_pref_admitted_on_turns']} of turns "
        f"({'lost' if promo_recall < 0.95 else 'kept'} their recall). "
        f"Promotion is {'a BLOCKER' if promo_breach or promo_recall < 0.95 else 'not a blocker'} for rollout."
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=Path("benchmarks/scenarios/phase0.json"))
    parser.add_argument("--draft-model", default="gpt-5.6-luna")
    parser.add_argument("--policy-model", default="gpt-5.4", help="Default for both gap and admission models")
    parser.add_argument("--gap-model", default=None)
    parser.add_argument("--admission-model", default=None)
    parser.add_argument("--check-model", default="gpt-5.4", help="Reference checker; keep fixed across configurations")
    parser.add_argument("--gap-prompt", choices=sorted(_GAP_PROMPTS), default="v1")
    parser.add_argument("--admission-prompt", choices=sorted(_ADMISSION_PROMPTS), default="v1")
    parser.add_argument(
        "--retrieval",
        choices=("dense", "real"),
        default="dense",
        help="dense: BGE-M3 top-k stand-in; real: the Memory Weave store, ingestor, and retriever",
    )
    parser.add_argument(
        "--shadow-judge",
        action="store_true",
        help="On no-gap turns, retrieve on the raw turn and run the judge without applying it",
    )
    parser.add_argument(
        "--activation",
        choices=("fixture", "real"),
        default="fixture",
        help=(
            "fixture: ambient records come from the scenario file; "
            "real: every record is written conditional and the activation policy decides"
        ),
    )
    parser.add_argument(
        "--activation-model", default="gpt-4o", help="Hosted model behind the category policy when --activation real"
    )
    parser.add_argument("--category-prompt", default="category-v2", help="Classifier prompt version")
    parser.add_argument(
        "--gap-repeats", type=int, default=1, help="Extra gap-planning calls per turn to measure decision stability"
    )
    parser.add_argument("--arms", default="ambient,conditional")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--draft-cache", type=Path, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--rewrite-model", default=None, help="Enable query rewriting through this hosted model; a new bundle"
    )
    parser.add_argument("--rewrite-timeout-ms", type=int, default=None)
    parser.add_argument(
        "--rerank-floor",
        type=float,
        default=None,
        help="Enable the cross-encoder reranker with this floor; a new bundle",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results/phase0"))
    parser.add_argument(
        "--label-overlay",
        type=Path,
        default=Path("benchmarks/scenarios/overlays/label_adjudication.json"),
        help="Adjudicated labels: records allowed for precision per scenario turn; recall ignores it.",
    )
    parser.add_argument(
        "--fail-on-verdict",
        action="store_true",
        help="Exit 1 if the serving arm fails the combination fitness bar. evaluate_combination.py sets this.",
    )
    args = parser.parse_args(argv)

    gap_model = args.gap_model or args.policy_model
    admission_model = args.admission_model or args.policy_model
    scenario = json.loads(args.scenario.read_text())
    runner = Phase0(
        scenario,
        args.draft_model,
        gap_model,
        admission_model,
        args.check_model,
        args.gap_prompt,
        args.gap_repeats,
        args.workers,
        args.draft_cache,
        args.admission_prompt,
        args.retrieval,
        args.shadow_judge,
        args.out_dir / "stores" / args.scenario.stem,
        args.activation,
        args.activation_model,
        args.category_prompt,
        rewrite_model=args.rewrite_model,
        rewrite_timeout_ms=args.rewrite_timeout_ms,
        rerank_floor=args.rerank_floor,
    )
    print(
        f"scenario={args.scenario} draft={args.draft_model} gap={gap_model}/{args.gap_prompt} "
        f"admission={admission_model}/{args.admission_prompt} "
        f"check={args.check_model} repeats={args.gap_repeats} retrieval={args.retrieval} "
        f"shadow_judge={args.shadow_judge} "
        f"activation={args.activation}/{args.activation_model}",
        flush=True,
    )
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    all_results: dict[str, list[TurnResult]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for arm in arms:
        print(f"\n=== arm: {arm} ===", flush=True)
        all_results[arm] = runner.run_arm(arm)
        summaries[arm] = summarise(
            arm,
            all_results[arm],
            runner.records,
            runner.promoted.get(arm),
            runner.activation_decisions.get(arm),
            runner.inventory_labels.get(arm),
            allowed=load_allowed(args.label_overlay, args.scenario.stem),
        )

    runner.save_cache()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tag = f"-{args.tag}" if args.tag else ""
    stages = ("-rewrite" if args.rewrite_model else "") + (
        f"-rerank{args.rerank_floor}" if args.rerank_floor is not None else ""
    )

    def _safe(name: str) -> str:
        return (
            name.replace("local:", "local-").replace("openrouter:", "openrouter-").replace("/", "-").replace(":", "-")
        )

    path = args.out_dir / (
        f"{stamp}-{args.scenario.stem}-gap-{_safe(gap_model)}-{args.gap_prompt}"
        f"-adm-{_safe(admission_model)}-{args.admission_prompt}-act-{args.activation}{stages}{tag}.json"
    )
    payload = {
        "scenario": str(args.scenario),
        "draft_model": args.draft_model,
        "gap_model": gap_model,
        "gap_prompt": args.gap_prompt,
        "admission_model": admission_model,
        "admission_prompt": args.admission_prompt,
        "retrieval": args.retrieval,
        "rewrite_model": args.rewrite_model,
        "rewrite_timeout_ms": args.rewrite_timeout_ms,
        "rerank_floor": args.rerank_floor,
        "shadow_judge": args.shadow_judge,
        "activation": args.activation,
        "activation_model": args.activation_model,
        "category_prompt": args.category_prompt,
        "promoted": runner.promoted,
        "activation_decisions": runner.activation_decisions,
        "inventory_labels": runner.inventory_labels,
        "write_logs": runner.write_logs,
        "check_model": args.check_model,
        "gap_repeats": args.gap_repeats,
        "usage": runner.models.usage,
        "calls": runner.models.calls,
        "summaries": summaries,
        "results": {arm: [asdict(r) for r in rs] for arm, rs in all_results.items()},
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {path}")

    print("\n== summary ==")
    keys = list(next(iter(summaries.values())).keys())
    for key in keys:
        if key == "arm":
            continue
        print(f"  {key:<40} " + "   ".join(f"{arm}: {summaries[arm][key]}" for arm in arms))
    if "ambient" in summaries and "conditional" in summaries:
        print("\n== gates ==")
        for line in gates(summaries["ambient"], summaries["conditional"]):
            print(line)
    print("\n== cost ==")
    for model, calls in runner.models.calls.items():
        usage = runner.models.usage[model]
        print(
            f"  {model:<14} calls={calls:<4} prompt_tokens={usage['prompt']:<7} completion_tokens={usage['completion']}"
        )

    print("\n== admissions that were not expected ==")
    for arm, rs in all_results.items():
        for r in rs:
            wrong = [i for i in r.admitted if i not in r.expected]
            if wrong:
                reasons = {v["id"]: v["reason"] for v in r.verdicts}
                for i in wrong:
                    print(
                        f"  [{arm}] {r.turn_id} {r.turn[:50]!r} admitted {i} "
                        f"({runner.records[i]['kind']}): {reasons.get(i, '')}"
                    )
    print("\n== shadow judge: what it would have admitted on no-gap turns ==")
    for arm, rs in all_results.items():
        for r in rs:
            if r.shadow_admitted:
                reasons = {v["id"]: v["reason"] for v in r.shadow_verdicts}
                for i in r.shadow_admitted:
                    print(
                        f"  [{arm}] {r.turn_id} {r.turn_class:<13} {r.turn[:50]!r} would admit {i} "
                        f"({runner.records[i]['kind']}): {reasons.get(i, '')}"
                    )
    print("\n== expected records not admitted ==")
    for arm, rs in all_results.items():
        for r in rs:
            missed = [i for i in r.expected if i not in r.admitted]
            if missed:
                reasons = {v["id"]: f"{v['verdict']}: {v['reason']}" for v in r.verdicts}
                retrieved = {c["id"] for c in r.candidates}
                for i in missed:
                    why = (
                        reasons.get(i) or ("not retrieved" if i not in retrieved else "no verdict")
                        if r.gaps
                        else "no gaps generated"
                    )
                    print(f"  [{arm}] {r.turn_id} {r.turn[:50]!r} missed {i}: {why}")

    arm, summary = serving_summary(payload)
    verdict = combination_verdict(summary)
    print(f"\n== combination verdict ({arm} arm) ==")
    for line in verdict.lines():
        print(line)
    if args.fail_on_verdict and not verdict.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
