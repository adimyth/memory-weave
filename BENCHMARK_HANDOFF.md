# Benchmark handoff

What the evaluation work needs from the repository, in one place. Everything here exists and is tested; the
numbers quoted are from the runs named, not estimates.

## The fixture

`tests/fixtures/memory_1k.sqlite` is a 1,074-record store built by `tests/integration/fixture_1k.py` with the
real `BAAI/bge-m3` embedder at 1,024 dimensions. Rebuild it after any change to the seed or the embedder:

```bash
HF_HUB_OFFLINE=1 MEMORY_WEAVE_INTEGRATION=1 uv run --extra local-models python -m tests.integration.fixture_1k
```

Shape: four users (`aditya`, `priya`, `rohan`, `meera`), three agents (`research-agent`, `ops-agent`,
`coding-agent`), two projects (`memory-weave`, `billing`), one organisation (`acme`); 486 semantic, 77
procedural, and 511 episodic records; 58 hand-written named records with stable ids that the labelled
queries point at, the rest templated distractors over the same vocabulary; two superseded chains
(`pref-aditya-shell`, `pref-priya-team`), one expired row and one past-expiry provisional inference, and
twelve provisional private-scope inferences per agent and user pair, a quarter of them past expiry. Grants
are in `USER_GRANTS` and `PROJECT_GRANTS` in the fixture module; every agent reads `org:acme`.

`tests/integration/labelled_queries.yaml` holds 50 relevant queries, each naming the record that answers it
and a principal that may read it, and 50 irrelevant queries on subjects the store does not hold. Tests copy
the fixture into a temporary directory with `copy_fixture`; never open the committed file directly.

## Running the suites

| Suite | Command | What it measures |
| --- | --- | --- |
| Unit | `uv run pytest` | Everything with fakes, including the synthetic isolation class and both adapters when their extras are installed. |
| Integration | `HF_HUB_OFFLINE=1 MEMORY_WEAVE_INTEGRATION=1 uv run --extra local-models pytest tests/integration` | Real embedder and judge on the fixture: the dense-floor sweep, warm and cold latency, isolation. |
| Slow | `MEMORY_WEAVE_RUN_SLOW=1 uv run pytest tests/integration/test_scale.py tests/integration/test_writer_contention.py -s` | The 50K store over 200 users and 10 agents, and four writers beside two extraction workers. |
| Live | `MEMORY_WEAVE_LIVE=1` with keys in `.env` | Hosted extractor, reviewer, and rewriter once each; the utility-aware fitness suite in `benchmarks/`. |

Both adapters' suites run when their extras are present: `uv run --extra deepagents --extra crewai pytest`.

## Where the log lives and which columns carry which stage

Every `memory_search` writes one row to `search_log` (LLD 3.6). `timings_ms` carries the stages in order:
`rewrite`, `scopes`, `filter`, `index_refresh`, `embed`, `dense`, `lexical`, `entity`, `fuse`, `freshness`,
`gate`, `dedup`, `rerank`, `budget`, `explain`, `log`, and `total`. The candidate columns are `dense`,
`lexical`, `entity` (per-channel hits), `fused`, `freshness`, `gated_out` (with the gate reason), `deduped_out`,
`reranked` and `reranked_out` (when the reranker is on), `budget_out`, `returned`, and `explanations`;
`trigger` says whether the model or the host issued the search; `config_flags` records the embedding
version, feature flags, and gate floors that produced the row; `warm` is 0 on the first search after a
process opens the store.

Writes carry their stages in the `record.created`, `record.reinforced`, `record.superseded`, and
`record.confirmed` events (`permission`, `evidence`, `entities`, `embed`, `dedup_search`, `judge`,
`supersession`, `persistence`, `event_log`, and the stages still pending when the event was written); the
`WriteResult.timings_ms` the caller receives is complete. Extraction runs carry `transcript_prep`,
`extractor_model`, `validation`, `candidate_review`, `writes`, `summary_write`, `dedup_and_contradiction`,
and `total` on the `extraction.run` event. Utility-aware turns write one `turn_decisions` row each, with the
stage outcome derivable by `memory_weave.policy.metrics.stage_outcome`; `memory-weave metrics` aggregates
them.

## Configuration flags that matter to a benchmark

`retrieval.trigger.mode` (`tool_only`, `auto`, `hybrid`); `retrieval.rewrite.enabled` and `reranker.enabled`
with `reranker.floor`, both off and both measured off in `docs/usefulness-gate.md` section 8n;
`retrieval.gate.dense_floor.<type>` and `.session_summary`, `lexical_min_term_fraction`,
`lexical_min_matched_terms`, `relative_floor`, and the stricter `retrieval.gate.auto` block for host-issued
searches; `retrieval.per_generator_k`, `rrf_k`, `default_k`, `token_budget`, `dedup_cosine`;
`embedding.model` and `embedding.version`, which every stored vector and every search log carry; the
utility-aware bundle, hashed by `memory_weave.policy.bundles.bundle_hash`, whose supported manifest is
`benchmarks/bundles/bundle-2026-09-08-a.json`. A change to any of these is a new configuration; a change to
any bundle component needs the full fitness suite before it serves.

## Measured on 8 September 2026

| Measurement | Result | Where |
| --- | --- | --- |
| Dense-floor sweep on the labelled set | Best F1 0.99 at floor 0.50; the configured 0.45 sits in the band 0.44 to 0.66 whose F1 is within 90 percent of the best. The weakest relevant query scored 0.52, the strongest irrelevant 0.51. | `test_gate_calibration.py` |
| Isolation, synthetic four-user store and the 1K fixture | Zero violations across every user pair sharing an agent and every agent pair sharing a user, on every log stage, through `memory_get`, and on grants. | `test_isolation.py` |
| Warm search on the fixture, real embedder, flags off, twenty distinct queries | p50 23.2 ms, p95 28.0 ms; embedding 19.8 ms of it; cold first search 2.6 s. Write p50 26.3 ms, p95 471 ms when the judge runs. | `test_latency.py` |
| 50K records, 200 users, 10 agents, 25 ms embed | Warm search p50 73.7 ms, p95 78.3 ms; scopes and filter 0.36 ms together; lexical 38.3 ms is the dominant stage at this size. | `test_scale.py` |
| Four writers beside two extraction workers | 160 writes and 20 extractions in 0.2 s, write p50 0.6 ms, zero lock errors. | `test_writer_contention.py` |

## Scenario categories the benchmark should cover

From HLD 16: desired write, prohibited write, desired retrieval, prohibited retrieval, stale record,
conflicting records, entity ambiguity, cross-scope boundary, search versus no-search, and downstream task
effect. The fixture's superseded chains, expired rows, private scopes, and aliased people are there for the
stale, conflicting, boundary, and ambiguity classes; the ordinary-turn class that public benchmarks lack is
what the utility-aware scenario splits in `benchmarks/scenarios/` were written for, and
`benchmarks/README.md` explains how to score a new model or prompt against them.
