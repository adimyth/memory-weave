# Acceptance report: Retold v1 (tagged as Memory Weave v1.0.0)

Written 8 September 2026 at the end of Core Phase 15 and completed 9 September with the adjudication in section 3, against the sequence agreed that morning: merge the utility-aware branch into `main`, then Core Phases 10 to 15 in order, each gated. Every number below comes from a test or a result file named beside it. Where a gate is not met, or is met only under a reading the reader may not accept, this report says so rather than rounding it.

## 1. What was built since the merge

| Phase | Commit | What it delivered |
| --- | --- | --- |
| Merge | `36e84a0` | `feat/utility-aware-memory` fast-forwarded into `main` after the full fitness suite passed again (`usefulness-gate.md` section 8m). |
| 10 | `9c22b8d` | Session extraction with review before apply, temporal metadata and the due-review worker, session hooks with the idle split, schema migration 9. |
| 11 | `490057d`, `aa43ef6` | The hosted query rewriter and the cross-encoder reranker behind their flags; both measured in four configurations and left off (section 8n); the supported bundle re-recorded as `bundle-2026-09-08-a`. |
| 12 | `ba607e2` | The operator surface: search, get, dump, expire, retain, review-due, reembed, erase, extract, snapshot; the composition root; immediate write transactions with a busy timeout. |
| 13 | `b51432d` | The Deep Agents adapter and the shared framework contract suite. |
| 14 | `3a6014e` | The CrewAI adapter, and the equivalence test showing both adapters leave the same semantic records from the same conversation. |
| 15 | `a827db6` | The 1K fixture, the labelled queries, the dense-floor sweep, warm and cold latency, the isolation class, the 50K scale test, the writer-contention test, and the benchmark handoff. |

## 2. Final quality gates

| Gate | Required | Measured | Status |
| --- | --- | --- | --- |
| Ordinary conditional injection | at most 5% | 1 of 20 and 0 of 20 on the fifth and fourth splits (Phase 11 baseline, configuration A); 0 of 20 and 0 of 4 in the shadow harness | met |
| Explicit stored-fact recall | at least 90% | 10 of 10 and 9 of 10 | met |
| Implicit memory-needed recall | at least 75% | 7 of 8 and 6 of 7 | met |
| Ambient-preference recall | at least 95% | 3 of 3 eligible preferences promoted on the fifth split; the promotion split passes on both independent builds with identical promoted sets | met |
| Helpful precision among admitted records | at least 95% | 19 of 20 (95%) and 17 of 17 (100%) with the adjudicated overlay; 18 of 20 (90%) and 16 of 17 (94%) under the blind labels alone | met after the blind adjudication in section 3 |
| Unsafe admissions and promotions | zero | zero placebo, misleading, private, or unsafe admissions in every run of the suite; zero unsafe promotions | met |
| Provider and budget failures return the baseline | always | every fail-closed branch of the orchestrator is tested; the one planner timeout observed in the Phase 11 baseline shadow run served the draft | met |
| Cross-principal leakage | zero | zero violations on the synthetic four-user store and on the 1K fixture, across every log stage, `memory_get`, and grants | met |
| Full tests, Ruff, strict mypy from a clean checkout | pass | see section 5 | see section 5 |

Sources: `benchmarks/results/phase0/phase11-baseline-v5` and `-v4`, `benchmarks/results/shadow/*phase11-baseline.json`, `benchmarks/results/promotion/20260908T101548Z-*.json`, `tests/integration/test_isolation.py`, `tests/test_utility_aware.py`, `tests/test_reference_host.py`.

## 3. The precision gate, diagnosed by stage and adjudicated

The suite counts an admitted record as helpful only when the scenario's `expected` list names it. Across the six recorded configuration A runs of the supported bundle, the non-expected admissions are of two kinds: the user's time-zone record on the scheduling turn `I5` of both splits, in every run, and an ordinary-turn injection in two of six runs, which the 5 percent gate already counts.

On 9 September 2026 an independent reviewer, Claude Sonnet 4.6 through OpenRouter, adjudicated the disputed record blind: it saw the turn, the records the assistant would already use, and the record under review, and never the judge's reasoning or the original label. Its verdict on both splits was **helpful but optional**. The blind labels are unchanged; the verdicts live in `benchmarks/scenarios/overlays/label_adjudication.json`; scoring now distinguishes required records, which recall counts, from allowed records, which precision also counts; and `benchmarks/rescore.py` recomputed the saved runs with no model call. On the two runs that are the bundle's fitness evidence, precision is 19 of 20 and 17 of 17. Section 8o of `usefulness-gate.md` has the table for all six. No bundle component changed.

## 4. What Phase 15 measured

| Measurement | Result |
| --- | --- |
| Dense-floor sweep, 50 relevant and 50 irrelevant labelled queries on the 1K fixture | Best F1 0.99 at floor 0.50. The configured semantic floor of 0.45 sits inside the band, 0.44 to 0.66, whose F1 is within 90 percent of the best. At 0.45 all 50 relevant queries pass and 6 irrelevant ones do. The weakest relevant score was 0.52 for an episodic all-hands query; the strongest irrelevant was 0.51 for a chess-opening query. |
| Isolation | Zero violations on both stores. |
| Warm search on the fixture, real embedder and judge, flags off, twenty distinct queries | p50 23.2 ms, p95 28.0 ms, against the 80 ms budget; the embedding pass is 19.8 ms of it and lexical 1.2 ms. The first search after opening the store, which loads the model and the index, took 2.6 s. |
| Write on the fixture, real embedder and judge | p50 26.3 ms against the 150 ms budget; p95 471 ms on the write whose attribute scan reached the NLI judge. Before Phase 15 the ingestor judged every active attribute of the entity, up to 64 pairs, and write p50 was 269 ms; it now judges only the same-attribute and cosine-adjacent records whose verdicts it consults, which changes no outcome and is covered by the existing ingestor tests. |
| 50K records over 200 users and 10 agents, fake embedder with a fixed 25 ms cost | Warm search p50 73.7 ms, p95 78.3 ms, against the 80 ms budget. Scopes and filter together 0.36 ms against the LLD 15 estimate of 3 ms; dense 0.8 ms; lexical 38.3 ms, the dominant stage at this size, because FTS5 joins the eligible-id table before its limit; the store took 160 s to build. |
| Four writer threads, 40 writes each, beside two extraction workers of ten sessions each, on one file | 160 writes and 20 extractions in 0.2 s; write p50 0.6 ms, p95 1.0 ms; zero lock errors, because every transaction begins IMMEDIATE and waits on the busy timeout. Fake models, so this is the SQLite ceiling alone. |

## 5. Clean checkout

A fresh clone of `main` at `a827db6` with `uv sync --all-extras`: Ruff clean, every file formatted, strict mypy clean over 59 source files, and 356 tests passed with 12 skipped, the skips being the integration, slow, and live suites that need the environment variables in `BENCHMARK_HANDOFF.md`. Repeated on 9 September at `6b86be8`, the commit carrying the adjudication and the foreign fitness work: Ruff clean, formatted, mypy clean, 365 passed with 12 skipped. Run separately on the same day: the integration suite (calibration, latency, isolation) 4 passed; the slow suite (scale, contention) 2 passed; the live extractor, reviewer, and rewriter once each against `gpt-4o`.

## 6. Supported configurations

| Configuration | Status |
| --- | --- |
| `retrieval.trigger.mode = tool_only`, rewrite off, reranker off | The default. Conservative, and the only mode in which the model decides every search. |
| `utility_aware`, bundle `bundle-2026-09-08-a` (`gpt-4o/gap-v3c`, `gpt-5.4/admission-v3`, `gpt-4o/category-v2`, `activation-v2-frozen`, `inventory-v1`, retrieval hash `e8c8c3309ab121de`, timeouts 4 s and 8 s) | Offline validated on the scripted splits and the shadow harness; a host may serve it only after recording that fitness result in its own store. Real-traffic validation is pending a consuming host. |
| Rewrite on, reranker on, or both | Built, measured, and not supported: neither produced a measurable benefit (section 8n). Enabling either is a new bundle. |
| Deep Agents and CrewAI adapters | Supported in `tool_only`, and in `tool_only` with `memory_mode="utility_aware"`, with fakes end to end; neither has run against a real serving model. `auto` and `hybrid` with `utility_aware` are refused at construction (section 8). |

## 7. Tag

Every gate is met and measured. `main` is tagged `v1.0.0`: `tool_only` is the conservative default, the approved `utility_aware` bundle `bundle-2026-09-08-a` is offline validated, and real-traffic production validation remains pending a consuming host. Future work should follow measured demand: a production integration, retrieval misses, judge cost, latency pressure, or enough labelled traffic to justify distillation.


## 8. v1.0.1 closure, 9 and 10 September 2026

A review of the tagged tree before publication found eight gaps. This section records what each one was, what
was done, and what was measured afterwards. Three commits carry the work: `db98a62`, `9662bc1`, and `dcda14b`.

### 8.1 What was found and what closed it

| # | Finding | Closed by | Evidence |
| --- | --- | --- | --- |
| 1 | Stage timeouts failed closed but did not bound latency: the executor joined its worker when the block exited, so a timed-out planner or judge still held the turn. | The planner and judge run on daemon threads that are abandoned at their timeout. | `test_a_hung_planner_stops_costing_the_turn_at_its_timeout`, `..._judge_...`: a 5 s policy against a 50 ms timeout returns in under 1 s. |
| 2 | `memory_mode="utility_aware"` with `auto` or `hybrid` let host-issued search put records in front of the model before admission decided anything. | `check_modes` refuses the combination in both adapters at construction. | `test_every_trigger_mode_against_every_memory_mode`, both adapters, all six combinations. |
| 3 | The validated planner, judge, and classifier lived under `benchmarks/`, which the wheel excludes, so a package user could not build the supported bundle. | They ship in `retold/policy/reference.py`; the harnesses import them. | `test_the_package_rebuilds_the_recorded_supported_bundle_exactly`; the wheel smoke rebuilds the bundle from an installed wheel alone. |
| 4 | The bundle's retrieval hash was a written-down claim, and the fitness runs used recall-oriented settings the shipped defaults do not have. | `supported_retrieval_config()` ships those settings; the hash is derived from the runtime `RetoldConfig` and a disagreeing manifest is refused. | `bundle_components(supported_bundle(), supported_retrieval_config())` equals `bundles/bundle-2026-09-08-a.json`, retrieval hash `e8c8c3309ab121de`. |
| 5 | A record written through `memory_write` never ran the activation policy, so a preference the model saved mid-session stayed conditional. | The tool handlers run it on writes and revisions and report the outcome. | `test_a_preference_written_through_the_tool_runs_the_activation_policy`. |
| 6 | The ambient profile was rebuilt every turn, against a design that says it is fixed for the session. | `ProfileAssembler` assembles once per session and the adapters drop it at session end. | `test_the_profile_is_assembled_once_a_session_and_reassembled_for_the_next`. |
| 7 | No pull-request CI; Python 3.13 advertised but never validated; known advisories in the optional extras; the artifact's metadata unproven against the publishing toolchain. | A `ci` workflow, a dependency bump, and a build-backend ceiling. | Section 8.3. |
| 8 | Public documents described pre-implementation states. | Historical documents say so at the top; the README separates supported, experimental, and historical. | `README.md` section 5.1. |

A ninth was found while closing the fourth: a bundle could declare a retrieval hash while no configuration was
given to check it against, which reads like an approval of whatever retrieval is in front of it. Serving that
now refuses; shadow warns and records `bundle_retrieval_verified` on every turn decision (migration 11).

### 8.2 The regression tests were checked against the defect

The six tests above were run against `bdce864`, the commit before the fixes, in a separate worktree. All six
fail there, each on the behaviour it names: the planner test measures 5.00 s against its 1.0 s bound, the judge
reports `failed` rather than `timeout`, the activation test cannot construct the handlers, the profile changes
mid-session, and both adapters accept the refused combinations. They pass on `dcda14b`.

### 8.3 Release gates

| Gate | Required | Measured | Status |
| --- | --- | --- | --- |
| Standard suite | pass | 391 passed, 12 deselected, on 3.12 and on 3.13 | met |
| Local-model and slow suites | pass | 9 passed in 191.7 s, macOS arm64, with `transformers` 5.16.1 and `sentence-transformers` 5.7.0 | met |
| Ruff, format, strict mypy | clean | clean on both versions, 60 source files | met |
| CI on every advertised version | pass on Ubuntu | run `34388291557`: `python 3.12`, `python 3.13`, `docs`, `wheel`, and `dependency audit` all green | met |
| Documentation builds strictly | pass | `mkdocs build --strict`, no warnings | met |
| Distribution metadata | accepted by the upload path | `twine check` passes on the wheel and the sdist. It failed before this round: hatchling 1.30 emits Metadata-Version 2.5 and the toolchain rejects it, so the build pins `hatchling>=1.27,<1.30`, which emits 2.4 | met |
| Clean installation | wheel and sdist install and run | the wheel installs with both adapter extras into fresh 3.12 and 3.13 environments and passes 24 smoke checks run from outside the checkout; the sdist installs and imports | met for the built artifacts |
| Known vulnerabilities in the locked set | none unfixed | five `transformers` advisories cleared by the version bump; four `chromadb` advisories have no fixed release, arrive only through the `crewai` extra, and are ignored by identifier in the audit job. Retold neither imports nor runs chromadb | met with that exception recorded |
| Published package | `pip install retold` works | **not met.** The package is not on PyPI: the name is unregistered and the trusted publisher has not been created, so `v1.0.1` is not tagged | open |

### 8.4 What is still open

Publication. Everything upstream of it is verified, including the metadata defect that would have failed the
upload, but `retold` has no PyPI project and no pending publisher, so the tag has deliberately not been pushed:
a tag whose publish job cannot succeed is worse than no tag. When a publisher exists for project `retold`, owner
`adimyth`, repository `retold`, workflow `publish.yml`, environment `pypi`, pushing `v1.0.1` completes the
release, and `pip install retold` in a clean environment is the last check.
