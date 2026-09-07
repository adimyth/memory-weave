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
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.draft_delta_experiment import Models  # noqa: E402

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

_GAP_PROMPTS = {"v1": _GAP_SYSTEM, "v2": _GAP_SYSTEM_V2}

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
    ) -> None:
        self.scenario = scenario
        self.draft_model = draft_model
        self.gap_model = gap_model
        self.admission_model = admission_model
        self.check_model = check_model
        self.gap_prompt = gap_prompt
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

    def retrieve(self, queries: list[str], store_ids: list[str]) -> list[dict[str, Any]]:
        assert self._embedder is not None
        query_vectors = self._embedder.embed_queries(queries)
        best: dict[str, dict[str, Any]] = {}
        for query, vector in zip(queries, query_vectors, strict=True):
            scored = sorted(
                ((float(self._doc_vectors[i] @ vector), i) for i in store_ids), reverse=True
            )[:_TOP_K_PER_QUERY]
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

    def run_turn(self, arm: str, turn: dict[str, Any], profile: list[dict[str, Any]], store_ids: list[str]) -> TurnResult:
        timing = Timing()
        failures: list[str] = []
        profile_text = self._profile_text(profile)
        draft_user = turn["text"]
        gap_user = f"{profile_text}\n\nUser message:\n{turn['text']}"

        gap_system = _GAP_PROMPTS[self.gap_prompt]

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

        result = TurnResult(arm, turn["id"], turn["class"], turn["text"], list(turn["expected"]), draft, gaps, [], [], [], "baseline_no_gaps", None, None, None, timing, failures, repeats)
        if not gaps:
            self._score_correctness(result, turn)
            return result

        started = time.perf_counter()
        candidates = self.retrieve(gaps, store_ids)
        timing.retrieval_s = time.perf_counter() - started
        if _PLACEBO_ID in store_ids and all(c["id"] != _PLACEBO_ID for c in candidates):
            candidates.append({"id": _PLACEBO_ID, "score": None, "query": "forced placebo"})
        result.candidates = candidates
        if not candidates:
            result.disposition = "baseline_no_candidates"
            self._score_correctness(result, turn)
            return result

        admission_user = (
            f"{profile_text}\n\nUser message:\n{turn['text']}\n\nDraft answer written without the candidates:\n{draft}\n\n"
            f"Candidate records:\n{self._candidate_block(candidates)}"
        )
        try:
            raw, timing.admission_s = self._timed(self.admission_model, _ADMISSION_SYSTEM, admission_user, json_mode=True)
            parsed = json.loads(raw)
            known = {c["id"] for c in candidates}
            verdicts = []
            for item in parsed.get("verdicts", []):
                verdict = str(item.get("verdict", "")).strip().lower()
                if str(item.get("id")) in known:
                    verdicts.append({"id": str(item["id"]), "verdict": verdict if verdict in _VERDICTS else f"unparsed:{verdict}", "reason": str(item.get("reason", ""))})
            admitted = [str(i) for i in parsed.get("admitted", []) if str(i) in known]
            admitted = [i for i in admitted if any(v["id"] == i and v["verdict"] in _ADMITTING for v in verdicts)]
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
            raw, _ = self._timed(self.check_model, _CORRECTNESS_SYSTEM, f"Question:\n{question}\n\nReference facts:\n{reference}\n\nAnswer:\n{answer}", json_mode=True)
            return bool(json.loads(raw).get("contains_reference"))
        except Exception:  # noqa: BLE001
            return None

    # -- one arm ----------------------------------------------------------------------------------------

    def run_arm(self, arm: str) -> list[TurnResult]:
        ambient_ids = [r["id"] for r in self.scenario["records"] if r["kind"] == "ambient_eligible"]
        if arm == "ambient":
            profile = [self.records[i] for i in ambient_ids]
            store_ids = [r["id"] for r in self.scenario["records"] if r["id"] not in ambient_ids]
        else:
            profile = []
            store_ids = [r["id"] for r in self.scenario["records"]]
        self._embed_store(store_ids)
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


def summarise(arm: str, results: list[TurnResult], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, list[TurnResult]] = {}
    for r in results:
        by_class.setdefault(r.turn_class, []).append(r)
    ordinary = by_class.get("ordinary", [])
    explicit = by_class.get("explicit_fact", [])
    implicit = by_class.get("implicit_need", [])

    ambient_ids = {i for i, r in records.items() if r["kind"] == "ambient_eligible"}

    def injected(r: TurnResult) -> bool:
        return bool(r.admitted)

    def recalled(r: TurnResult) -> bool:
        return bool(r.expected) and all(e in r.admitted for e in r.expected)

    admitted_total = sum(len(r.admitted) for r in results)
    admitted_useful = sum(sum(1 for i in r.admitted if i in r.expected) for r in results)
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
        "ambient_pref_admitted_on_turns": _pct(sum(1 for r in results if any(i in ambient_ids for i in r.admitted)), len(results)),
        "placebo_admitted": kinds_admitted.get("placebo", 0),
        "misleading_admitted": kinds_admitted.get("misleading", 0),
        "stale_or_conflicting_admitted": kinds_admitted.get("stale", 0) + kinds_admitted.get("conflicting", 0),
        "redundant_admitted": kinds_admitted.get("redundant", 0),
        "unrelated_private_admitted": leakage,
        "usefulness_precision": _pct(admitted_useful, admitted_total),
        "answer_correct_memory_turns_draft": _pct(sum(bool(r.draft_correct) for r in memory_ref), len(memory_ref)),
        "answer_correct_memory_turns_final": _pct(sum(final_or_draft(r) for r in memory_ref), len(memory_ref)),
        "empty_gap_turns": len(empty_gap),
        "added_latency_empty_gap_p50_p95_s": (round(_p(empty_gap, 0.5), 2), round(_p(empty_gap, 0.95), 2)),
        "added_latency_gap_turns_p50_p95_s": (round(_p(gap_turns, 0.5), 2), round(_p(gap_turns, 0.95), 2)),
        "draft_latency_p50_s": round(_p([r.timing.draft_s for r in results], 0.5), 2),
        "policy_failures": sum(len(r.failures) for r in results),
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
        f"({'breaches' if promo_breach else 'within'} 5%); style/language admitted on {c['ambient_pref_admitted_on_turns']} of turns "
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
    parser.add_argument("--gap-repeats", type=int, default=1, help="Extra gap-planning calls per turn to measure decision stability")
    parser.add_argument("--arms", default="ambient,conditional")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--draft-cache", type=Path, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results/phase0"))
    args = parser.parse_args(argv)

    gap_model = args.gap_model or args.policy_model
    admission_model = args.admission_model or args.policy_model
    scenario = json.loads(args.scenario.read_text())
    runner = Phase0(scenario, args.draft_model, gap_model, admission_model, args.check_model, args.gap_prompt, args.gap_repeats, args.workers, args.draft_cache)
    print(f"scenario={args.scenario} draft={args.draft_model} gap={gap_model}/{args.gap_prompt} admission={admission_model} check={args.check_model} repeats={args.gap_repeats}", flush=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    all_results: dict[str, list[TurnResult]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for arm in arms:
        print(f"\n=== arm: {arm} ===", flush=True)
        all_results[arm] = runner.run_arm(arm)
        summaries[arm] = summarise(arm, all_results[arm], runner.records)

    runner.save_cache()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tag = f"-{args.tag}" if args.tag else ""
    path = args.out_dir / f"{stamp}-{args.scenario.stem}-gap-{gap_model}-{args.gap_prompt}-adm-{admission_model}{tag}.json"
    payload = {
        "scenario": str(args.scenario),
        "draft_model": args.draft_model,
        "gap_model": gap_model,
        "gap_prompt": args.gap_prompt,
        "admission_model": admission_model,
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
        print(f"  {model:<14} calls={calls:<4} prompt_tokens={usage['prompt']:<7} completion_tokens={usage['completion']}")

    print("\n== admissions that were not expected ==")
    for arm, rs in all_results.items():
        for r in rs:
            wrong = [i for i in r.admitted if i not in r.expected]
            if wrong:
                reasons = {v["id"]: v["reason"] for v in r.verdicts}
                for i in wrong:
                    print(f"  [{arm}] {r.turn_id} {r.turn[:50]!r} admitted {i} ({runner.records[i]['kind']}): {reasons.get(i, '')}")
    print("\n== expected records not admitted ==")
    for arm, rs in all_results.items():
        for r in rs:
            missed = [i for i in r.expected if i not in r.admitted]
            if missed:
                reasons = {v["id"]: f"{v['verdict']}: {v['reason']}" for v in r.verdicts}
                retrieved = {c["id"] for c in r.candidates}
                for i in missed:
                    why = reasons.get(i) or ("not retrieved" if i not in retrieved else "no verdict") if r.gaps else "no gaps generated"
                    print(f"  [{arm}] {r.turn_id} {r.turn[:50]!r} missed {i}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
