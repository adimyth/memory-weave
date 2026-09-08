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
HF_HUB_OFFLINE=1 uv run --extra live --extra local-models python benchmarks/evaluate_combination.py \
  --gap-model gpt-4o --admission-model gpt-5.4
```

Scores one planner/judge/classifier combination against the committed fifth-split scenario. Prints `PASS` or `FAIL` per threshold and a final `COMBINATION PASS` or `COMBINATION FAIL`, and exits 1 on failure. Defaults are the combination that already passed; any other models or prompts are a new combination and need this run. To score a saved `phase0` JSON without calling models: `--from-result benchmarks/results/phase0/<file>.json`. The thresholds and the combination that passed are in [../docs/design-contributions.md](../docs/design-contributions.md).

```bash
HF_HUB_OFFLINE=1 uv run --extra live --extra local-models python benchmarks/phase0_two_arm.py \
  --draft-model gpt-5.6-luna --policy-model gpt-5.4
```

Phase 0 of the utility-aware memory plan: the gap-planning plus draft-relative admission path, run twice over the hand-authored scenario set in `scenarios/phase0.json`, once with style and language preferences ambient and once with them conditional. Prints the design gate, the promotion gate, and a combination verdict on the ambient arm. Results land in `results/phase0/`; findings are in [../docs/usefulness-gate.md](../docs/usefulness-gate.md) section 8c. `--fail-on-verdict` exits 1 when that combination bar is missed; `evaluate_combination.py` sets it.

`--gap-model`, `--admission-model`, `--gap-prompt v1|v2`, and `--gap-repeats N` vary one stage at a time
and measure gap-decision stability; `--draft-cache` keeps drafts identical across configurations.
`scenarios/phase0_tune.json` is the separate tuning split used to choose the gap model and prompt, so
that `phase0.json` stays a held-out set. That selection is in section 8d of the same document.
`scenarios/phase0_final.json` and `scenarios/phase0_v4.json` are blind splits, each run once with the
configuration fixed in advance; sections 8e and 8f. `--retrieval real` writes the records through the
real ingestor and retrieves through `memory_search` (`phase0_real_retrieval.py`); `--shadow-judge` runs
the judge on relevance-path candidates for no-gap turns without applying the result, which is how the
judge's own ordinary-turn admission rate is measured; `--gap-prompt v3` adds the content-free category
inventory and `--admission-prompt v3` is the decision-impact rule.

```bash
HF_HUB_OFFLINE=1 uv run --extra live --extra local-models python benchmarks/shadow_run.py \
  --scenario benchmarks/scenarios/phase0_v5.json --gap-model gpt-4o --admission-model gpt-5.4
uv run memory-weave --store <store.sqlite> metrics --json --rollback-check
uv run memory-weave --store <store.sqlite> bundles record benchmarks/bundles/bundle-2026-09-07-a.json \
  --passed --evidence "docs/usefulness-gate.md 8k,8l" --by <you>
```

```bash
HF_HUB_OFFLINE=1 uv run --extra live --extra local-models python benchmarks/reference_host_e2e.py \
  --scenario benchmarks/scenarios/slice_conversation.json
```

Drives `examples/reference_host.py` with the real hosted adapters and the supported bundle through the
whole operating loop: refusal while unapproved, shadow, fitness recording, serving, budget withhold, a kill
switch, metrics, and rollback reasons. Results land in `results/reference-host/`.

```bash
HF_HUB_OFFLINE=1 uv run --extra live --extra local-models python benchmarks/rerank_calibration.py \
  --scenario benchmarks/scenarios/phase0_tune.json --check benchmarks/scenarios/phase0_v5.json
```

Sweeps the cross-encoder reranker's floor on a labelled split: every turn's text runs as a host-issued search
with the reranker enabled and its floor at zero, then precision, recall, and F1 of expected records, and the
share of ordinary turns with any survivor, are computed offline for every floor. Results land in
`results/rerank/`. `--rewrite-model <model>` and `--rerank-floor <x>` on `phase0_two_arm.py` and
`shadow_run.py` enable the two optional retrieval stages for a run; either is a new retrieval configuration,
so the bundle hash changes and the run cannot reuse an earlier fitness result. The bundle hash covers the
`retrieval` and `reranker` sections of the configuration together.

```bash
uv run --extra live python benchmarks/adjudicate_labels.py --reviewer openrouter:anthropic/claude-sonnet-4.6
uv run python benchmarks/rescore.py benchmarks/results/phase0/phase11-baseline-v5/*.json
```

A disputed scenario label is adjudicated blind by an independent reviewer model that sees the turn, the
records already expected, and the record under review, never the judge's reasoning or the label. The verdicts
land in `scenarios/overlays/label_adjudication.json`; the blind labels are never edited. Recall counts only
the expected (required) records; precision also counts records the overlay marks helpful, and `summarise`
reports both `usefulness_precision` and `usefulness_precision_strict`. `rescore.py` recomputes saved results
under the overlay with no model call. Findings in [../docs/usefulness-gate.md](../docs/usefulness-gate.md)
section 8o.

The shadow harness runs the orchestrator beside the served path with isolation asserted; `metrics`
aggregates any store's turn-decision log into stage outcomes, rates, latency, cost, and backlog; `bundles
record` writes the supported bundle's fitness result into a store so that store may serve it. Without that
record the orchestrator refuses to serve and allows shadow only.

Unprefixed model names go to OpenAI. `openrouter:<slug>` uses `OPENROUTER_API_KEY` (for example `--admission-model openrouter:anthropic/claude-sonnet-4.6` for a second-vendor judge). `local:<hf repo>` loads an open-weight model. The vertical slice's `OPENROUTER_PROVIDER` pin is not applied to these names.

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
