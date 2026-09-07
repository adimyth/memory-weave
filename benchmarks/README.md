# Experiments

What has been run against Memory Weave, what each experiment was supposed to show, and what it showed.

## The vertical slice

One fixed twelve-turn conversation replayed through a real serving model and the five memory tools, three
times against three fresh stores, once per trigger mode. Nothing is judged by reading the model's replies.
Every number comes from the audit events and the search log.

The conversation is built so each turn has a known correct behaviour, which is what makes it scorable:

| Class | Turns | What the system should do |
| --- | --- | --- |
| preference | 3 | Store a durable preference about the user. |
| correction | 1 | Store it and supersede the preference it contradicts. |
| entity | 2 | Store a fact about a named third party. |
| memory_applies | 2 | Recall a stored preference, because the answer depends on it. |
| ordinary | 4 | Recall nothing. The turn overlaps the store's subject matter but is not answered by it. |

The last class is the hard one and the reason the script exists. Public memory benchmarks do not contain
it: they only ask questions whose answers are in the store, so they cannot measure a system returning
memory when it should have stayed quiet.

## Running it

```bash
# one-time: model weights and the live SDKs
uv sync --extra live --extra local-models

# set OPENAI_API_KEY and the model in .env, then
HF_HUB_OFFLINE=1 MEMORY_WEAVE_LIVE=1 uv run python examples/vertical_slice.py \
  --provider openai --model <model-id> --runs 3 --trigger hybrid
```

`HF_HUB_OFFLINE=1` is not optional once the cache is warm. A partial model cache hangs indefinitely
inside the hub transfer library instead of failing.

Artifacts land in `results/vertical-slice/<mode>-<timestamp>-<id>/`, one SQLite store per run plus
`report.json`.

## Analysing it

```bash
uv run python benchmarks/analyse_gate.py <artifact-dir>              # gate separability and floor sweep
uv run python benchmarks/analyse_gate.py <artifact-dir> --scenarios  # the per-turn table below
```

Both read only the stored databases. No model calls, no searches re-executed.

```bash
uv run --extra live python benchmarks/draft_delta_experiment.py <artifact-dir> \
  --draft-model gpt-5.6-luna --judge-model gpt-5.4
```

This one does make model calls. It replays the logged host-issued searches through a draft-then-judge
usefulness gate: a draft answer with no memory, then one judge call per search asking whether each logged
candidate would change that draft. Results land in `results/draft-delta/`. Method and findings are in
[../docs/usefulness-gate.md](../docs/usefulness-gate.md) section 8.

```bash
HF_HUB_OFFLINE=1 uv run --extra live --extra local-models python benchmarks/utility_signal_experiment.py \
  <artifact-dir> --policy-model gpt-5.4
```

Replays the same searches through the utility signals of RUMS (entropy reduction) and TRACE-Memory (gap
queries, then likelihood gain of the recorded reply), using a local Llama-3.1-8B-Instruct as the frozen
model. Takes about eight minutes on an M4 Pro. Results land in `results/utility-signals/`; findings are in
[../docs/usefulness-gate.md](../docs/usefulness-gate.md) section 8b.

```bash
HF_HUB_OFFLINE=1 uv run --extra live --extra local-models python benchmarks/phase0_two_arm.py \
  --draft-model gpt-5.6-luna --policy-model gpt-5.4
```

Phase 0 of the utility-aware memory plan: the gap-planning plus draft-relative admission path, run twice
over the hand-authored scenario set in `scenarios/phase0.json`, once with style and language preferences
ambient and once with them conditional. Prints the design gate and the promotion gate. Results land in
`results/phase0/`; findings are in [../docs/usefulness-gate.md](../docs/usefulness-gate.md) section 8c.

## Results, 6 September 2026

Model `gpt-5.6-luna`, three runs per mode, reproduced from a deleted store.
Artifacts: `results/vertical-slice/tool_only-20260906-reproduction` and `hybrid-20260906-reproduction`.

### Per turn, hybrid mode

| # | Class | Turn | Expected | Wrote | Recalled | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | preference | I prefer concise technical answers, with a sho | store it; no recall needed | 3/3 | 0/3 | pass |
| 2 | preference | For code examples, use Python unless I ask for | store it; no recall needed | 3/3 | 0/3 | pass |
| 3 | preference | My working time zone is Asia/Kolkata. | store it; no recall needed | 3/3 | 0/3 | pass |
| 4 | correction | Actually, keep answers concise but include the | store it and supersede the earlier preference | 3/3 | 3/3 | pass |
| 5 | entity | My colleague Priya Nair owns the deployment ch | store it against the named person | 3/3 | 0/3 | pass |
| 6 | entity | Priya asked us to keep the deployment checklis | store it against the named person | 3/3 | 3/3 | pass |
| 7 | memory_applies | Show me a small Python example that parses thi | recall the stored preference | 0/3 | 0/3 | **recall missed** |
| 8 | memory_applies | What time zone should you use when suggesting  | recall the stored preference | 0/3 | 3/3 | pass |
| 9 | ordinary | Should I use tabs or spaces in Python? | recall nothing | 0/3 | 0/3 | pass |
| 10 | ordinary | How often should a deployment checklist be rev | recall nothing | 0/3 | 3/3 | **false inject** (3/3) |
| 11 | ordinary | What is a clear way to format a trade-off in a | recall nothing | 0/3 | 2/3 | **false inject** (2/3) |
| 12 | ordinary | What are common mistakes when naming a Python  | recall nothing | 0/3 | 0/3 | pass |

### Mode comparison

| Measure | tool_only | hybrid |
| --- | --- | --- |
| Writes attempted | 18 | 18 |
| Supersessions | 1 | 3 |
| Model searched when memory applied | 0 of 6 turns | 0 of 6 turns |
| Host searches issued | 0 | 36 |
| Ordinary turns receiving memory | 0 of 12 | 5 of 12 |

### What the results say

**Writing works.** All six turns that should write did so, in every run, in both modes.

**The model does not read.** It searched on zero of six applicable turns. Memory reached it only when the
host asked. `tool_only` is therefore not a working mode on this evidence.

**`hybrid` delivers and pollutes.** It recalled on the turns that needed it, and on 5 of 12 that did
not, against a target of under 5 percent.

**The gate cannot be tuned out of this.** Score distributions for "needs memory" and "does not" overlap
almost completely, with ordinary turns reaching the higher maximum. No floor separates them. Full
treatment in [../docs/gate.md](../docs/gate.md).

**But most of it dissolves.** Every record returned on a turn that needed memory was a preference about
the user. Those apply to every turn, so gating them is meaningless; they belong in an always-present
block. Of the two false injections, turn 11 matched a preference and disappears under that change, and
turn 10 is the accepted residual case: a turn about the deployment checklist matching checklist facts
that do not answer it.

**Two things are accepted rather than fixed:** that residual adjacency, and attribute-name drift, where
three executions produced four different slugs for one preference. Both are recorded in
[../docs/next-phases.md](../docs/next-phases.md).

## Not built yet

The comparative benchmark in [../docs/benchmark-plan.md](../docs/benchmark-plan.md), covering LongMemEval
and LoCoMo against Mem0 and LangMem, is specified and unbuilt. This directory holds only the vertical
slice.
