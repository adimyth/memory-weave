# Phase 9a vertical-slice findings

## Status

Recorded on 2026-09-06. Two experiments, three runs each, against a hosted model: one in `tool_only`
and one in `hybrid`. Both completed with no failed runs.

## Run metadata

| Field | Value |
| --- | --- |
| Provider / model | `openai` / `gpt-5.6-luna` |
| Prompt version | `v1` |
| Runs | 3 per mode, 0 failed |
| tool_only artifacts | `benchmarks/results/vertical-slice/tool_only-20260906-reproduction` |
| hybrid artifacts | `benchmarks/results/vertical-slice/hybrid-20260906-reproduction` |

## Headline comparison

| Measure | `tool_only` | `hybrid` |
| --- | --- | --- |
| Writes attempted | 18 | 18 |
| Outcomes | 15 created, 3 conflict | 15 created, **2 superseded**, 1 conflict |
| Model searched on preference-applicable turns | 1 of 6 turns | **0 of 6 turns** |
| Host searches issued | 0 | 36 (one per user turn) |
| Host searches returning memory | n/a | 15 of 36 (42%) |
| Ordinary turns that received injected memory | 0 of 12 | **5 of 12 (42%)** |
| Evidence quotes located in the transcript | 12 of 18 | 0 of 18 |
| Quotes located but rejected by entailment | 6 | 0 |

These are the figures from the reproduction run of 2026-09-06, after all Phase 9a fixes were in place.
An earlier run of the same script, before `evidence_turn` was removed, is described in section
"Reproduction and variance" below; the structural findings held, the per-run rates did not.

## What hybrid changes

**It is the only mode in which memory reaches the model at all.** The model searched once in three
`tool_only` runs and never in three `hybrid` runs. Left to itself this model writes memory
constantly and reads it almost never. Recall happened only because the host asked.

**Supersession started working.** `tool_only` produced three conflicts and no supersessions; `hybrid`
produced two supersessions. The turn-4 correction could finally replace the turn-1 preference, because
removing `evidence_turn` let more claims keep the source rank needed to supersede.

**The injection rate is far above the promotion target.** Half of the ordinary turns received recalled
memory. The benchmark plan sets an ordinary-turn injection target below 5 percent before `hybrid` can
become the default. At 50 percent this model plus these gate floors is not close. The gate is doing
something, since 21 of 36 host searches returned nothing, but not nearly enough.

**Conclusion: neither mode is shippable as-is.** `tool_only` does not recall. `hybrid` recalls but
pollutes. The gap between them is the gate, which is where the next work belongs.

## Subject stability: unstable, and the earlier reading was wrong

An earlier note recorded stability as holding. That was measured by a metric that only inspects records
citing transcript turn 1, so it saw one attribute per run and nothing else. Reading every attribute the
runs actually produced:

| Run | Attributes chosen |
| --- | --- |
| 1 | `code_language`, `response_style`, `working_timezone`, `ownership` |
| 2 | `code_language`, `response_style`, `working_timezone`, `deployment_checklist_location`, `deployment_checklist_owner` |
| 3 | `code_language_preference`, `communication_preference`, `working_timezone`, `ownership`, `storage_location` |

Only `working_timezone` is stable across all three runs. Run 3 renamed almost everything:
`response_style` became `communication_preference`, `code_language` became `code_language_preference`.
Facts stated identically in identical conversations landed under different current-fact keys.

**This triggers the deferred LLD 18 item on a system-proposed attribute vocabulary.** Supersession keys
on `(entity, attribute)`, so a fact recorded as `response_style` in one session and
`communication_preference` in the next will never supersede its predecessor. Both persist and both are
retrievable. The Phase 7a attribute-aliasing pass through the judge mitigates this within a scope, but
it only fires when the cosine or subject scan surfaces the older record, and it will not fire reliably
across sessions.

## Evidence: the contract has two gates and the model clears both unreliably

Per-run behaviour differed sharply, which matters more than the aggregate:

| Run | Quotes located | Rejected by entailment | Net accepted |
| --- | --- | --- | --- |
| 1 | 0 of 6 | 0 | 0 |
| 2 | 6 of 6 | 3 | 3 |
| 3 | 0 of 6 | 0 | 0 |

One run quoted the user verbatim every time. Two runs paraphrased every time and matched nothing. Same
conversation, same prompt, same model. When quotes did land, the NLI judge rejected half of them as not
supporting the claim.

So the evidence rule is not a stable filter on this model. It behaves closer to a coin flip on whether a
whole run produces usable provenance, which means most records land as unconfirmed inferences that expire
in 30 days.

## Metric defect found while reading these results

`_attributes_for_first_preference` keys on `source_ref = session:<id><turn:1>`. It therefore reports
nothing for any run whose evidence failed, and it only ever examines one preference. It reported
`stable: true` for `tool_only` and `runs_with_attributes: 1` for `hybrid`, both misleading. It should
compare the full per-run attribute set for the principal's entity, independent of evidence success.

## Contract problems this phase found and fixed

- `memory_write` exposed a raw `{kind, id}` scope. The model invented `{"kind": "user", "id": "user"}`
  and every write was rejected. Replaced with host-resolved symbolic targets.
- `evidence_turn` exposed harness-internal turn numbering. The model counted conversational turns, so
  every write after the first pointed at an assistant turn. Removed from the model-facing schema.
- Evidence quotes wrapped in quotation marks failed the substring check. Normalization now strips one
  matched surrounding pair.
- The model rejects `max_tokens` and refuses function tools alongside its default reasoning effort.
  Both are discovered and adapted to on first rejection.

Three of these four were the same defect: the tool asked the model for an identifier it could not know.

## Environment notes

Pin `sentence-transformers<5`; version 6 with transformers 5.x cannot load BGE-M3. The DeBERTa tokenizer
needs `protobuf` and `sentencepiece`. Run with `HF_HUB_OFFLINE=1` once the model cache is warm, because a
partial cache hangs indefinitely inside the hub transfer library rather than failing.

## Next

1. Fix the stability metric, then re-measure. The attribute instability above is read from the databases
   directly and does not depend on it, but the reported number should match.
2. Take the attribute-vocabulary item in LLD 18 off the deferred list. This is now evidenced.
3. The gate is the blocking problem for `hybrid`. The analysis, why every memory system faces it, and
   three ways to attack it that can each be validated offline are in [gate.md](gate.md). A 50 percent ordinary-turn injection rate against a
   5 percent target is the calibration work the benchmark plan describes, and the search log holds every
   candidate score needed to sweep the floors offline.

## Run command

```bash
HF_HUB_OFFLINE=1 MEMORY_WEAVE_LIVE=1 uv run --extra live --extra local-models \
  python examples/vertical_slice.py --provider openai --model <model-id> --runs 3 --trigger hybrid
```


## Reproduction and variance

The whole experiment was deleted and re-run from scratch on 2026-09-06 with the final code, to check
which findings are stable and which are run-to-run noise.

**Stable across both executions.**

- The model never searched on its own when memory applied. Zero of six applicable turns in `hybrid` in
  both executions, and zero to one in `tool_only`.
- `hybrid` is the only mode that delivers memory. 14 to 15 of 36 host searches returned records.
- Ordinary-turn injection sits at 42 to 50 percent, against a 5 percent target.
- Supersession works in `hybrid` and barely fires in `tool_only`.
- The gate cannot separate the two turn classes, at any floor, on any candidate subset.
- Every record returned on a turn that needed memory was a profile record. Zero topical, both times.
- The highest topical scores on ordinary turns are always the deployment-checklist records against the
  "how often should a checklist be reviewed" turn.

**Not stable.** The evidence-quote match rate swung from 3 of 18, to 12 of 18, to 0 of 18 across
executions of the same script. Entailment rejections swung from 0 to 6. This is the same run-to-run
variance seen within a single execution, where one run quoted verbatim six times out of six and the other
two paraphrased everything. Any single number for evidence quality on this model is not meaningful; the
variance is the finding.

**Attribute stability was worse on re-run.** Run 3 produced `response_style_preference` where runs 1 and
3 of the earlier execution produced `response_style` and `communication_preference`. Three executions have
now produced four different slugs for the same stated preference. The vocabulary item is not marginal.
