"""Replay logged host-issued searches through the utility signals of RUMS and TRACE-Memory.

Both papers define a memory record's value by how it changes the generator's response distribution, not by
how similar it is to the query. This script computes those signals with a local frozen model on the
candidates already stored in `search_log`, so no search is re-run and no serving model is called for the
signals themselves.

RUMS (Response-Aware User Memory Selection, ICML 2026). Utility of a memory set S for query x is
    U(S) = H(Y | x) - H(Y | x, S)
where H is the response entropy, estimated by sampling N short continuations and averaging the per-token
entropy of the next-token distribution along them. The paper used Llama-3.1-8B, N=5, T_max=20. It then
trained a classifier on these labels; this script computes the labels directly, which is the oracle the
classifier was trained to imitate.

TRACE-Memory (2026). Stage 1 turns the request into short queries describing the user-specific information
that public knowledge cannot supply, and retrieves against those. Stage 2 admits evidence by its incremental
value over the public-only path, measured against a reference response y* under a frozen generator:
    dL(S) = mean_t [ log p(y*_t | x, S) - log p(y*_t | x) ]
    dH(S) = mean_t [ H(p_t(. | x)) - H(p_t(. | x, S)) ]
with EMPTY worth exactly zero. The paper trains a small policy on that reward; this script computes the
reward directly, using the assistant reply the serving model actually gave on that turn as y*.

Usage:
    HF_HUB_OFFLINE=1 uv run --extra live --extra local-models python benchmarks/utility_signal_experiment.py \
        benchmarks/results/vertical-slice/hybrid-20260906-reproduction
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.draft_delta_experiment import Models, _load_with_turns  # noqa: E402

_SYSTEM = "You are a helpful assistant."
_PLACEBO = "User's favourite colour is green."
_GAP_SYSTEM = (
    "You help a memory system decide what to look up before answering a user. Given the user's message, "
    "list up to 3 short search queries. Each query must describe one piece of user-specific information "
    "that public knowledge cannot supply and that would change how the message is answered: a preference, "
    "a constraint, a fact about the user's own situation or organisation, or something the user told the "
    "assistant earlier. Do not restate the topic of the message and do not ask for public facts. If the "
    "message can be answered equally well for anyone, return an empty list. Reply with JSON only: "
    '{"queries": ["...", "..."]}'
)


@dataclass(slots=True)
class RecordSignal:
    record_id: str
    attribute: str | None
    profile: bool
    content: str
    dense_score: float
    rums_entropy: float
    rums_utility: float
    trace_dl: float
    trace_dh: float


@dataclass(slots=True)
class SearchSignals:
    run: str
    category: str
    turn: str
    reference: str
    rums_entropy_base: float
    records: list[RecordSignal]
    rums_utility_all: float
    trace_dl_all: float
    trace_dh_all: float
    placebo_utility: float
    placebo_dl: float
    placebo_dh: float
    gate_returned: list[str]
    stage1_queries: list[str] = field(default_factory=list)
    stage1_retrieved: list[dict[str, object]] = field(default_factory=list)


class FrozenModel:
    """A local causal LM used only to score distributions. Nothing it generates is shown to anyone."""

    def __init__(self, name: str, samples: int, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.samples = samples
        self.max_new_tokens = max_new_tokens
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(self.device).eval()
        print(f"loaded {name} on {self.device} in {time.perf_counter() - started:.0f}s", flush=True)
        self._entropy_cache: dict[tuple[str, tuple[str, ...]], float] = {}
        self._teacher_cache: dict[tuple[str, tuple[str, ...], str], tuple[float, float]] = {}

    def _prompt_ids(self, turn: str, facts: Sequence[str]):
        system = _SYSTEM
        if facts:
            system += "\n\nFacts you know about this user:\n" + "\n".join(f"- {fact}" for fact in facts)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": turn}]
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    @staticmethod
    def _entropy(logits):
        log_probs = logits.float().log_softmax(dim=-1)
        return -(log_probs.exp() * log_probs).sum(dim=-1)

    def response_entropy(self, turn: str, facts: Sequence[str]) -> float:
        """RUMS: mean per-token entropy along N sampled continuations of at most T_max tokens."""

        key = (turn, tuple(facts))
        if key in self._entropy_cache:
            return self._entropy_cache[key]
        torch = self.torch
        ids = self._prompt_ids(turn, facts).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                ids,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.samples,
                output_logits=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = out.sequences[:, ids.shape[1] :]
        steps = len(out.logits)
        per_sample: list[float] = []
        eos_ids = set(
            self.tokenizer.eos_token_id
            if isinstance(self.tokenizer.eos_token_id, list)
            else [self.tokenizer.eos_token_id]
        )
        extra = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(extra, int) and extra >= 0:
            eos_ids.add(extra)
        for sample in range(self.samples):
            entropies = []
            for step in range(steps):
                entropies.append(float(self._entropy(out.logits[step][sample])))
                if int(generated[sample, step]) in eos_ids:
                    break
            per_sample.append(sum(entropies) / len(entropies))
        value = sum(per_sample) / len(per_sample)
        self._entropy_cache[key] = value
        return value

    def teacher_forced(self, turn: str, facts: Sequence[str], reference: str) -> tuple[float, float]:
        """Return (mean log-likelihood of the reference, mean entropy along it) under the given facts."""

        key = (turn, tuple(facts), reference)
        if key in self._teacher_cache:
            return self._teacher_cache[key]
        torch = self.torch
        prompt = self._prompt_ids(turn, facts)
        target = self.tokenizer(reference, add_special_tokens=False, return_tensors="pt").input_ids
        ids = torch.cat([prompt, target], dim=1).to(self.device)
        with torch.no_grad():
            logits = self.model(ids).logits[0]
        start = prompt.shape[1]
        positions = logits[start - 1 : ids.shape[1] - 1]
        log_probs = positions.float().log_softmax(dim=-1)
        token_lp = log_probs.gather(1, ids[0, start:].unsqueeze(1)).squeeze(1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        value = (float(token_lp.mean()), float(entropy.mean()))
        self._teacher_cache[key] = value
        return value


def _references(attempt_dir: Path) -> dict[tuple[str, str], str]:
    """Map (run, user turn text) to the assistant reply that followed it in that run."""

    refs: dict[tuple[str, str], str] = {}
    for database in sorted(attempt_dir.glob("run-*.sqlite")):
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            rows = list(connection.execute("SELECT role, content FROM session_turns ORDER BY rowid"))
        finally:
            connection.close()
        pending: str | None = None
        for row in rows:
            if row["role"] == "user":
                pending = row["content"]
            elif row["role"] == "assistant" and pending is not None:
                refs[(database.stem, pending)] = row["content"]
                pending = None
    return refs


def _all_records(attempt_dir: Path) -> dict[str, list[tuple[str, str, str | None, bool]]]:
    """Every active record per run: (id, content, attribute, profile)."""

    import benchmarks.analyse_gate as gate

    out: dict[str, list[tuple[str, str, str | None, bool]]] = {}
    for database in sorted(attempt_dir.glob("run-*.sqlite")):
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            principal = gate._principal_entity(connection)  # noqa: SLF001
            out[database.stem] = [
                (row["id"], row["content"], row["attribute"], row["subject_entity_id"] == principal)
                for row in connection.execute(
                    "SELECT id, content, attribute, subject_entity_id FROM records WHERE status != 'superseded'"
                )
            ]
        finally:
            connection.close()
    return out


def stage1(models: Models, policy_model: str, turns: Sequence[str]) -> dict[str, list[str]]:
    queries: dict[str, list[str]] = {}
    for turn in dict.fromkeys(turns):
        raw = models.complete(policy_model, _GAP_SYSTEM, f"User message:\n{turn}", json_mode=True)
        parsed = json.loads(raw).get("queries", [])
        queries[turn] = [str(q) for q in parsed][:3]
        print(f"stage1 {turn[:50]!r} -> {queries[turn]}", flush=True)
    return queries


def stage1_retrieve(
    queries: dict[str, list[str]], records: dict[str, list[tuple[str, str, str | None, bool]]], top_k: int
):
    from memory_weave.config import EmbeddingConfig
    from memory_weave.index.embedder import BgeM3Embedder

    embedder = BgeM3Embedder(EmbeddingConfig())
    all_queries = sorted({q for qs in queries.values() for q in qs})
    query_vectors = dict(zip(all_queries, embedder.embed_queries(all_queries), strict=True)) if all_queries else {}
    hits: dict[tuple[str, str], list[dict[str, object]]] = {}
    for run, rows in records.items():
        doc_vectors = embedder.embed_documents([content for _, content, _, _ in rows])
        for turn, qs in queries.items():
            best: dict[str, dict[str, object]] = {}
            for q in qs:
                scores = doc_vectors @ query_vectors[q]
                order = sorted(range(len(rows)), key=lambda i: -float(scores[i]))[:top_k]
                for i in order:
                    record_id, content, attribute, profile = rows[i]
                    score = float(scores[i])
                    if record_id not in best or score > float(best[record_id]["score"]):
                        best[record_id] = {
                            "record_id": record_id,
                            "attribute": attribute,
                            "profile": profile,
                            "content": content,
                            "score": round(score, 3),
                            "query": q,
                        }
            hits[(run, turn)] = sorted(best.values(), key=lambda h: -float(h["score"]))
    return hits


def run(
    attempt_dir: Path,
    model_name: str,
    samples: int,
    max_new_tokens: int,
    policy_model: str | None,
    out_dir: Path,
    limit: int | None,
) -> None:
    searches = _load_with_turns(attempt_dir)
    if limit:
        searches = searches[:limit]
    refs = _references(attempt_dir)
    print(f"evaluation searches: {len(searches)}", flush=True)

    stage1_queries: dict[str, list[str]] = {}
    stage1_hits: dict[tuple[str, str], list[dict[str, object]]] = {}
    if policy_model:
        models = Models()
        stage1_queries = stage1(models, policy_model, [turn for _, turn, _ in searches])
        stage1_hits = stage1_retrieve(stage1_queries, _all_records(attempt_dir), top_k=3)

    frozen = FrozenModel(model_name, samples, max_new_tokens)
    results: list[SearchSignals] = []
    for index, (run_name, turn, search) in enumerate(searches, start=1):
        started = time.perf_counter()
        reference = refs[(run_name, turn)]
        facts_all = [c.content for c in search.dense]
        h_base = frozen.response_entropy(turn, [])
        lp_base, h_ref_base = frozen.teacher_forced(turn, [], reference)
        records: list[RecordSignal] = []
        for c in search.dense:
            h_one = frozen.response_entropy(turn, [c.content])
            lp_one, h_ref_one = frozen.teacher_forced(turn, [c.content], reference)
            records.append(
                RecordSignal(
                    c.record_id,
                    c.attribute,
                    c.profile,
                    c.content,
                    c.score,
                    h_one,
                    h_base - h_one,
                    lp_one - lp_base,
                    h_ref_base - h_ref_one,
                )
            )
        h_all = frozen.response_entropy(turn, facts_all) if facts_all else h_base
        lp_all, h_ref_all = frozen.teacher_forced(turn, facts_all, reference) if facts_all else (lp_base, h_ref_base)
        h_pl = frozen.response_entropy(turn, [_PLACEBO])
        lp_pl, h_ref_pl = frozen.teacher_forced(turn, [_PLACEBO], reference)
        results.append(
            SearchSignals(
                run_name,
                search.category,
                turn,
                reference,
                h_base,
                records,
                h_base - h_all,
                lp_all - lp_base,
                h_ref_base - h_ref_all,
                h_base - h_pl,
                lp_pl - lp_base,
                h_ref_base - h_ref_pl,
                [c.record_id for c in search.returned],
                stage1_queries.get(turn, []),
                stage1_hits.get((run_name, turn), []),
            )
        )
        print(
            f"[{index}/{len(searches)}] {run_name} {search.category:<14} {turn[:40]!r} H0={h_base:.3f} "
            f"maxU={max((r.rums_utility for r in records), default=0):.3f} maxdL={
                max((r.trace_dl for r in records), default=0):.3f} "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{stamp}-{model_name.split('/')[-1]}-n{samples}-t{max_new_tokens}.json"
    path.write_text(
        json.dumps(
            {
                "attempt_dir": str(attempt_dir),
                "frozen_model": model_name,
                "samples": samples,
                "max_new_tokens": max_new_tokens,
                "policy_model": policy_model,
                "searches": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    print(f"\nwrote {path}")
    report(results)


def _dist(label: str, values: Sequence[float]) -> str:
    if not values:
        return f"  {label:>28}: none"
    return f"  {label:>28}: n={len(values):<3} min={min(values):+.3f} median={statistics.median(values):+.3f} max={max(values):+.3f}"  # noqa: E501


def report(results: Sequence[SearchSignals]) -> None:
    ordinary = [r for r in results if r.category == "ordinary"]
    applies = [r for r in results if r.category == "memory_applies"]

    def best(rows: Sequence[SearchSignals], attr: str, topical_only: bool) -> list[float]:
        out = []
        for r in rows:
            pool = [x for x in r.records if not (topical_only and x.profile)]
            out.append(max((getattr(x, attr) for x in pool), default=0.0))
        return out

    for name, attr, placebo_attr in (
        ("RUMS utility  U = H0 - H(S)", "rums_utility", "placebo_utility"),
        ("TRACE  dL (likelihood gain of reference)", "trace_dl", "placebo_dl"),
        ("TRACE  dH (entropy reduction along reference)", "trace_dh", "placebo_dh"),
    ):
        print(f"\n== {name}: best candidate per search ==")
        for topical_only in (False, True):
            tag = "topical only" if topical_only else "all candidates"
            o, a = best(ordinary, attr, topical_only), best(applies, attr, topical_only)
            print(f" {tag}:")
            print(_dist("ordinary", o))
            print(_dist("memory applies", a))
            if o and a:
                print(f"  {'separable?':>28}: {'YES' if min(a) > max(o) else 'NO'}")
        print(_dist("placebo fact, all searches", [getattr(r, placebo_attr) for r in results]))

    print("\n== RUMS threshold sweep on max utility, all candidates (paper used tau=0.29) ==")
    for tau in (0.0, 0.05, 0.1, 0.2, 0.29, 0.4, 0.6):
        o = sum(1 for v in best(ordinary, "rums_utility", False) if v > tau)
        a = sum(1 for v in best(applies, "rums_utility", False) if v > tau)
        print(
            f"  tau={tau:<5} ordinary injected {o:>2} of {len(ordinary):<3} applicable recalled {a:>2} of {len(applies)}"  # noqa: E501
        )

    print("\n== TRACE admission sweep on dL > delta, EMPTY = 0 ==")
    for delta in (0.0, 0.01, 0.02, 0.05, 0.1):
        for topical_only in (False, True):
            o = sum(1 for v in best(ordinary, "trace_dl", topical_only) if v > delta)
            a = sum(1 for v in best(applies, "trace_dl", topical_only) if v > delta)
            print(
                f"  delta={delta:<5} {'topical' if topical_only else 'all    '} ordinary injected {o:>2} of {len(ordinary):<3} applicable recalled {a:>2} of {len(applies)}"  # noqa: E501
            )

    print("\n== per record: the two cases that matter ==")
    for r in results:
        for x in r.records:
            if (r.turn.startswith("What time zone") and x.attribute == "working_timezone") or (
                r.turn.startswith("How often should a deployment checklist")
                and x.attribute in ("ownership", "deployment_checklist_owner")
            ):
                print(
                    f"  {r.run} {r.turn[:32]!r:<36} [{x.attribute}] dense={x.dense_score:.2f} U={x.rums_utility:+.3f} dL={x.trace_dl:+.3f} dH={x.trace_dh:+.3f}"  # noqa: E501
                )

    print("\n== per record, all searches ==")
    for r in results:
        print(
            f"{r.run} {r.category:<14} {r.turn[:44]!r} H0={r.rums_entropy_base:.3f} U(all)={r.rums_utility_all:+.3f} "
            f"dL(all)={r.trace_dl_all:+.3f} placebo U={r.placebo_utility:+.3f} dL={r.placebo_dl:+.3f}"
        )
        for x in r.records:
            print(
                f"    {'P' if x.profile else 'T'} {x.attribute or '-':<32} dense={x.dense_score:.2f} U={x.rums_utility:+.3f} dL={x.trace_dl:+.3f} dH={x.trace_dh:+.3f}"  # noqa: E501
            )

    if any(r.stage1_queries or r.stage1_retrieved for r in results):
        print("\n== TRACE stage 1: public-conditioned queries and what they retrieve ==")
        seen: set[str] = set()
        for r in results:
            if r.turn in seen:
                continue
            seen.add(r.turn)
            print(f"  {r.category:<14} {r.turn[:60]!r}\n    queries: {r.stage1_queries}")
            for h in r.stage1_retrieved[:3]:
                print(
                    f"    -> {h['score']:.2f} [{'P' if h['profile'] else 'T'} {h['attribute']}] {str(h['content'])[:60]}"  # noqa: E501
                )
        print("\n  turn-level: stage 1 produced no queries")
        for category, rows in (("ordinary", ordinary), ("memory_applies", applies)):
            empty = sum(1 for r in rows if not r.stage1_queries)
            print(f"    {category:<14} {empty} of {len(rows)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_dir", type=Path)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--samples", type=int, default=5, help="RUMS Monte Carlo samples N")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="RUMS T_max")
    parser.add_argument(
        "--policy-model", default=None, help="Hosted model standing in for TRACE stage 1; omit to skip stage 1"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results/utility-signals"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    run(args.attempt_dir, args.model, args.samples, args.max_new_tokens, args.policy_model, args.out_dir, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
