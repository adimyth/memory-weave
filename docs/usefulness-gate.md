# Utility-aware memory experiments

This document records the evidence behind Retold's utility-aware host path. It keeps the conclusions and reproducibility details that remain useful after implementation. The authoritative runtime design is [utility-aware-memory-architecture.md](utility-aware-memory-architecture.md), and the release-level verdict is [acceptance-report.md](acceptance-report.md).

## The question

Retrieval relevance asks whether a record concerns the user query. It does not establish whether giving that record to the response model will improve the answer. The original relevance-gate experiment found that the best similarity scores for ordinary turns and memory-needed turns overlap at every threshold. Raising the threshold removes useful records along with unwanted ones.

Retold therefore evaluates usefulness relative to an answer:

1. Generate a baseline draft without conditional memory.
2. Identify specific missing context using a content-free inventory of the memory categories available to the caller.
3. Retrieve only for that missing context.
4. Compare the bounded candidate set with the baseline draft.
5. Admit a record only when it would materially improve the answer, with an empty set as a valid result.

Stable response preferences are handled separately as audited ambient memory because they shape many answers and should not be rediscovered on each turn.

## Evaluation design

The final evaluation uses two blind scenario splits plus a shadow run through the real orchestrator. The scenarios contain explicit questions about stored facts, implicit turns that require prior state, ordinary turns adjacent to stored records, placebos, misleading records, private records, conflicts, superseded records, and jointly useful evidence. Expected records are hidden from the planner and admission judge.

The primary gates are ordinary-turn conditional injection, explicit and implicit recall, helpful precision, unsafe admission, ambient-preference promotion, isolation, and fail-closed behavior. Retrieval and write latency are measured separately in the acceptance report.

## Final result

| Measure | Required | Fifth blind split | Fourth blind split | Verdict |
| --- | --- | --- | --- | --- |
| Ordinary turns receiving conditional memory | at most 5% | 1 of 20 | 0 of 20 | pass |
| Explicit stored-fact recall | at least 90% | 10 of 10 | 9 of 10 | pass |
| Implicit memory-needed recall | at least 75% | 7 of 8 | 6 of 7 | pass |
| Helpful precision among admitted records | at least 95% | 19 of 20 | 17 of 17 | pass |
| Unsafe admissions or promotions | zero | zero | zero | pass |

The shadow harness preserved cross-principal isolation on every check, admitted nothing unsafe, and returned the baseline draft on policy failure and timeout. The supported bundle is `benchmarks/bundles/bundle-2026-09-08-a.json`; its retrieval configuration hash is `e8c8c3309ab121de`.

## What the experiments established

- A similarity floor can filter weak candidates but cannot decide whether a turn needs memory.
- Asking for the specific missing fact produces better retrieval queries than searching with the raw user query.
- Comparing candidates with a baseline draft separates useful context from context that merely shares a topic.
- The planner carries most of the ordinary-turn precision because it can choose not to retrieve. The admission judge carries safety and final usefulness.
- An admission judge must see the bounded candidate set together because some records become useful only in combination.
- Utility judgments are model-dependent. The planner role transferred to an open-weight 8B model, but the admission role did not meet the full bar outside the selected frontier model.
- The path must be evaluated as one versioned bundle containing models, prompts, taxonomy, inventory builder, retrieval configuration, limits, and timeouts.

## Approaches measured and not selected

| Approach | Result | Decision |
| --- | --- | --- |
| Similarity threshold as a turn-level gate | Ordinary and memory-needed score distributions overlap | Keep it only as a candidate relevance filter |
| Query rewriting before retrieval | Usually left planner queries unchanged and added roughly 1.7 to 1.9 seconds per applied search | Built, disabled by default |
| Cross-encoder after reciprocal-rank fusion | Removed expected records often enough to fail explicit recall | Built, disabled by default |
| Cross-encoder instead of the relevance floors | Reduced candidate volume but failed explicit and implicit recall | Built, disabled by default |
| Small reasoning model as admission judge | Slower and less reliable than the selected judge | Not selected |
| Alternative frontier judges on the shadow set | Matched recall but admitted misleading or private records | Not selected |
| RUMS-style entropy reduction | Distinguished some useful facts but misread response-style preferences and requires model logits | Retained as research context only |
| TRACE-style likelihood gain | Separated the key useful and merely related cases offline but requires a reference response | Useful as an offline label, not a request-path policy |

RRF-only retrieval remains the supported configuration. The admission judge already rejects most weak candidates, so an aggressive pre-judge reranker loses recall without producing a corresponding safety or precision gain.

## Label adjudication

One recurring admission was not covered by the original required-record labels: a user time-zone record on a scheduling query. A blind reviewer, shown the user query, the records already required, and the disputed record without seeing the original label or judge verdict, classified the time-zone record as helpful but optional. The original labels remain unchanged. Precision is reported both under the strict labels and under the adjudication overlay, and recall continues to count only required records.

The prompt and verdicts are stored in `benchmarks/scenarios/overlays/label_adjudication.json`. `benchmarks/rescore.py` applies the overlay to saved results without calling a model.

## Reproducing the evidence

The committed inputs and harnesses are:

- `benchmarks/scenarios/phase0_v4.json` and `phase0_v5.json` for the two blind recall splits.
- `benchmarks/scenarios/promotion_v1.json` and `benchmarks/promotion_split.py` for ambient activation.
- `benchmarks/phase0_two_arm.py` for the end-to-end planner, retrieval, admission, and regeneration evaluation.
- `benchmarks/shadow_adapter.py` for the real orchestrator path and isolation checks.
- `benchmarks/evaluate_combination.py` and `benchmarks/fitness.py` for bundle-level fitness.
- `benchmarks/rerank_calibration.py` and `benchmarks/pool_stats.py` for the rejected cross-encoder placements.
- `benchmarks/rescore.py` for label-overlay rescoring.

Raw result directories are intentionally gitignored. The accepted aggregate values are recorded here and in [acceptance-report.md](acceptance-report.md); the bundle manifest records the exact selected components.

## Research basis

The architecture is closest to two lines of work:

- *Response-Aware User Memory Selection for LLM Personalization* (RUMS), ICML 2026, evaluates memory by its effect on the response distribution.
- *TRACE-Memory: Public-Conditioned Retrieval and Utility-Aware Evidence Admission for Personalized Generation*, 2026, separates missing-information planning from evidence admission.

Retold uses the same answer-relative principle with a text-only, provider-neutral runtime, audited ambient activation, a content-free inventory, joint admission, fail-closed execution, and bundle fitness gating.
