"""Replay logged host-issued searches through a draft-then-judge usefulness gate.

For each evaluation turn in a completed vertical-slice attempt, this asks a draft model to answer the turn
with no memory at all, then asks a judge model, once per search, whether each logged candidate record
would change that draft. A record is admitted when the judge says it contradicts the draft or fills a gap
the draft left open. The result is compared with what the relevance gate actually returned.

Drafts are cached per turn text, so the number of draft calls is the number of distinct evaluation turns.
Judge calls are one per logged search. Nothing is re-retrieved; candidates come from `search_log.dense`.

Usage:
    uv run --extra live python benchmarks/draft_delta_experiment.py \
        benchmarks/results/vertical-slice/hybrid-20260906-reproduction \
        --draft-model gpt-5.6-luna --judge-model gpt-5-nano
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.analyse_gate import Candidate, LoggedSearch  # noqa: E402

_EVAL_CATEGORIES = ("memory_applies", "ordinary")
_ADMITTING = ("contradicts", "fills_gap")

_DRAFT_SYSTEM = (
    "You are a helpful assistant. Answer the user's message directly and briefly, in under 120 words. "
    "You have no memory of this user or of any earlier conversation."
)

_JUDGE_SYSTEM = (
    "You decide whether a stored fact about a user would change an assistant's answer. "
    "You will see the user's message, a draft answer written with no knowledge of the user, and a list of "
    "stored facts. For each fact, choose exactly one verdict:\n"
    "- contradicts: the draft asserts or assumes something that this fact shows is wrong for this user, so "
    "the answer must change.\n"
    "- fills_gap: the draft is missing, hedging on, or asking for something this fact supplies, and knowing "
    "it would materially change what the answer says or how it is written.\n"
    "- unchanged: knowing this fact would not change the substance of the answer. Being about the same "
    "topic as the message is not enough; the fact must actually alter the answer.\n"
    'Reply with JSON only: {"verdicts": [{"id": "<fact id>", "verdict": "<one of the three>", '
    '"reason": "<one short sentence>"}]}. Include every fact id exactly once.'
)


_JUDGE_SYSTEM_STRICT = (
    "You decide whether a stored fact about a user would change an assistant's answer. "
    "You will see the user's message, a draft answer written with no knowledge of the user, and a list of "
    "stored facts. For each fact, choose exactly one verdict:\n"
    "- contradicts: the draft asserts or assumes something that this fact shows is wrong for this user.\n"
    "- fills_gap: the draft explicitly asks the user for this information, or says it does not know it, or "
    "cannot complete the task without it.\n"
    "- unchanged: everything else. If the fact would only add context, attribution, background, or a "
    "nice-to-have detail, or would only change tone or length, the verdict is unchanged. Being about the same "
    "topic is not enough. When unsure, answer unchanged.\n"
    'Reply with JSON only: {"verdicts": [{"id": "<fact id>", "verdict": "<one of the three>", '
    '"reason": "<one short sentence>"}]}. Include every fact id exactly once.'
)

_JUDGE_PROMPTS = {"default": _JUDGE_SYSTEM, "strict": _JUDGE_SYSTEM_STRICT}


@dataclass(slots=True)
class Verdict:
    record_id: str
    verdict: str
    reason: str
    profile: bool
    attribute: str | None
    content: str
    dense_score: float


@dataclass(slots=True)
class JudgedSearch:
    run: str
    category: str
    turn: str
    draft: str
    verdicts: list[Verdict]
    gate_returned: list[str]
    judge_admitted: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.judge_admitted = [v.record_id for v in self.verdicts if v.verdict in _ADMITTING]


class _LocalBackend:
    """A local open-weight chat model behind the same complete() contract, for vendor-portability runs.

    Model names are given as ``local:<hf repo id>``. Generation is greedy and serialised with a lock, since
    one process holds one copy of the weights. JSON mode appends an instruction and extracts the first
    JSON object from the reply, because open-weight chat models have no structured-output switch.
    """

    def __init__(self, repo: str) -> None:
        import threading

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(repo)
        self._model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to(self._device).eval()
        self._lock = threading.Lock()

    def complete(self, system: str, user: str, *, json_mode: bool) -> tuple[str, int, int]:
        if json_mode:
            system = system + "\n\nReply with a single JSON object and nothing else."
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self._device)
        with self._lock, self._torch.no_grad():
            out = self._model.generate(ids, do_sample=False, max_new_tokens=700, pad_token_id=self._tokenizer.eos_token_id)
        generated = out[0, ids.shape[1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        if json_mode:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                text = text[start : end + 1]
        return text, int(ids.shape[1]), int(generated.shape[0])


class Models:
    """Thin wrapper over the OpenAI SDK, plus a local open-weight backend, that records token usage per model."""

    def __init__(self) -> None:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        self._client = OpenAI()
        self.usage: dict[str, dict[str, int]] = {}
        self.calls: dict[str, int] = {}
        self.latency_s: dict[str, float] = {}
        self._local: dict[str, _LocalBackend] = {}
        import threading

        self._local_init_lock = threading.Lock()

    def _complete_local(self, model: str, system: str, user: str, *, json_mode: bool) -> str:
        repo = model.split(":", 1)[1]
        # Concurrent turns must share one copy of the weights: two threads racing through a lazy
        # construction each loaded a 16 GB model, and the second load failed inside the gap call.
        with self._local_init_lock:
            backend = self._local.get(repo)
            if backend is None:
                backend = self._local[repo] = _LocalBackend(repo)
        started = time.perf_counter()
        text, prompt_tokens, completion_tokens = backend.complete(system, user, json_mode=json_mode)
        elapsed = time.perf_counter() - started
        bucket = self.usage.setdefault(model, {"prompt": 0, "completion": 0})
        bucket["prompt"] += prompt_tokens
        bucket["completion"] += completion_tokens
        self.calls[model] = self.calls.get(model, 0) + 1
        self.latency_s[model] = self.latency_s.get(model, 0.0) + elapsed
        if not text:
            raise RuntimeError(f"{model} returned an empty message")
        return text

    def complete(
        self, model: str, system: str, user: str, *, json_mode: bool = False, reasoning_effort: str | None = None
    ) -> str:
        if model.startswith("local:"):
            return self._complete_local(model, system, user, json_mode=json_mode)
        kwargs: dict[str, object] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": 4000,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        started = time.perf_counter()
        response = self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        elapsed = time.perf_counter() - started
        usage = response.usage
        bucket = self.usage.setdefault(model, {"prompt": 0, "completion": 0})
        if usage is not None:
            bucket["prompt"] += int(usage.prompt_tokens or 0)
            bucket["completion"] += int(usage.completion_tokens or 0)
        self.calls[model] = self.calls.get(model, 0) + 1
        self.latency_s[model] = self.latency_s.get(model, 0.0) + elapsed
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(f"{model} returned an empty message (finish={response.choices[0].finish_reason})")
        return content


def _load_with_turns(attempt_dir: Path) -> list[tuple[str, str, LoggedSearch]]:
    """Return (run name, turn text, search) for every evaluation-turn host search."""

    from examples.vertical_slice import CONVERSATION

    categories = {turn.text: turn.category for turn in CONVERSATION}
    out: list[tuple[str, str, LoggedSearch]] = []
    for database in sorted(attempt_dir.glob("run-*.sqlite")):
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            turns = [
                json.loads(row["request"])["queries"][0]
                for row in connection.execute("SELECT request FROM search_log WHERE trigger = 'auto'")
            ]
        finally:
            connection.close()
        per_db = _load_one_db(database)
        for turn, search in zip(turns, per_db, strict=True):
            if categories.get(turn) in _EVAL_CATEGORIES:
                out.append((database.stem, turn, search))
    return out


def _load_one_db(database: Path) -> list[LoggedSearch]:
    """Load one run database using the same reader as analyse_gate, preserving log order."""

    import benchmarks.analyse_gate as gate

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        # analyse_gate skips turns it cannot classify; every auto search here is a scripted turn.
        return gate._load_one(connection)  # noqa: SLF001
    finally:
        connection.close()


def _judge(
    models: Models,
    judge_model: str,
    turn: str,
    draft: str,
    candidates: Sequence[Candidate],
    effort: str | None,
    prompt_name: str,
) -> dict[str, tuple[str, str]]:
    """Judge every candidate in one call. Facts are labelled F1..Fn because record ids share long prefixes."""

    labels = {f"F{i}": c.record_id for i, c in enumerate(candidates, start=1)}
    facts = "\n".join(f"- id={label}: {c.content}" for label, c in zip(labels, candidates, strict=True))
    prompt = f"User message:\n{turn}\n\nDraft answer (written with no knowledge of the user):\n{draft}\n\nStored facts:\n{facts}"  # noqa: E501
    raw = models.complete(judge_model, _JUDGE_PROMPTS[prompt_name], prompt, json_mode=True, reasoning_effort=effort)
    parsed = json.loads(raw)
    verdicts: dict[str, tuple[str, str]] = {}
    for item in parsed.get("verdicts", []):
        verdict = str(item.get("verdict", "")).strip().lower()
        if verdict not in (*_ADMITTING, "unchanged"):
            verdict = "unparsed:" + verdict
        record_id = labels.get(str(item.get("id", "")).strip())
        if record_id is not None:
            verdicts[record_id] = (verdict, str(item.get("reason", "")))
    return verdicts


def run(
    attempt_dir: Path,
    draft_model: str,
    judge_model: str,
    out_dir: Path,
    judge_effort: str | None,
    judge_prompt: str,
) -> None:
    models = Models()
    searches = _load_with_turns(attempt_dir)
    print(f"evaluation searches: {len(searches)} across {len({r for r, _, _ in searches})} runs")

    drafts: dict[str, str] = {}
    for _, turn, _ in searches:
        if turn not in drafts:
            drafts[turn] = models.complete(draft_model, _DRAFT_SYSTEM, turn)
            print(f"draft [{draft_model}] {turn[:50]!r} -> {drafts[turn][:80]!r}")

    judged: list[JudgedSearch] = []
    for run_name, turn, search in searches:
        candidates = search.dense
        if not candidates:
            judged.append(JudgedSearch(run_name, search.category, turn, drafts[turn], [], []))
            continue
        verdicts = _judge(models, judge_model, turn, drafts[turn], candidates, judge_effort, judge_prompt)
        rows = [
            Verdict(
                c.record_id, *verdicts.get(c.record_id, ("missing", "")), c.profile, c.attribute, c.content, c.score
            )
            for c in candidates
        ]
        judged.append(
            JudgedSearch(run_name, search.category, turn, drafts[turn], rows, [c.record_id for c in search.returned])
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "attempt_dir": str(attempt_dir),
        "draft_model": draft_model,
        "judge_model": judge_model,
        "judge_reasoning_effort": judge_effort,
        "judge_prompt": judge_prompt,
        "usage": models.usage,
        "calls": models.calls,
        "latency_s": models.latency_s,
        "searches": [asdict(j) for j in judged],
    }
    path = out_dir / f"{stamp}-draft-{draft_model}-judge-{judge_model}-{judge_effort or 'default'}-{judge_prompt}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path}")
    report(judged, models)


def report(judged: Sequence[JudgedSearch], models: Models) -> None:
    print("\n== per search ==")
    for j in judged:
        marks = " ".join(f"[{'P' if v.profile else 'T'} {v.attribute}: {v.verdict}]" for v in j.verdicts)
        print(
            f"{j.run} {j.category:<14} {j.turn[:48]:<48} gate={len(j.gate_returned)} judge={len(j.judge_admitted)}  {marks}"  # noqa: E501
        )

    print("\n== turn-level: searches where at least one record was admitted ==")
    for category in _EVAL_CATEGORIES:
        rows = [j for j in judged if j.category == category]
        gate_hits = sum(1 for j in rows if j.gate_returned)
        judge_hits = sum(1 for j in rows if j.judge_admitted)
        print(f"  {category:<14} n={len(rows):<3} relevance gate: {gate_hits:>2}   draft-delta judge: {judge_hits:>2}")

    print("\n== record-level on ordinary turns (should all be unchanged) ==")
    counts: dict[str, int] = {}
    for j in judged:
        if j.category != "ordinary":
            continue
        for v in j.verdicts:
            key = f"{'profile' if v.profile else 'topical'}/{v.verdict}"
            counts[key] = counts.get(key, 0) + 1
    for key in sorted(counts):
        print(f"  {key:<24} {counts[key]}")

    print("\n== record-level on memory-applies turns ==")
    counts = {}
    for j in judged:
        if j.category != "memory_applies":
            continue
        for v in j.verdicts:
            key = f"{'profile' if v.profile else 'topical'}/{v.attribute}/{v.verdict}"
            counts[key] = counts.get(key, 0) + 1
    for key in sorted(counts):
        print(f"  {key:<48} {counts[key]}")

    print("\n== the residual case: checklist-ownership record on the review-cadence turn ==")
    for j in judged:
        if j.turn.startswith("How often should a deployment checklist"):
            for v in j.verdicts:
                if v.attribute in ("ownership", "deployment_checklist_owner"):
                    print(f"  {j.run}: {v.verdict} (dense {v.dense_score:.2f}) -- {v.reason}")

    print("\n== cost ==")
    for model, calls in models.calls.items():
        usage = models.usage[model]
        print(
            f"  {model:<14} calls={calls:<3} prompt_tokens={usage['prompt']:<6} completion_tokens={usage['completion']:<6} "  # noqa: E501
            f"mean_latency={models.latency_s[model] / calls:.1f}s"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_dir", type=Path)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results/draft-delta"))
    parser.add_argument(
        "--judge-reasoning-effort",
        default=None,
        help="Reasoning effort for the judge model, e.g. low. Omit to use the provider default.",
    )
    parser.add_argument("--judge-prompt", choices=sorted(_JUDGE_PROMPTS), default="default")
    args = parser.parse_args(argv)
    run(
        args.attempt_dir,
        args.draft_model,
        args.judge_model,
        args.out_dir,
        args.judge_reasoning_effort,
        args.judge_prompt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
