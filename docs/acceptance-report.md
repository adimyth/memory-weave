# Acceptance report: Retold v1

## Final quality gates

| Gate | Required | Measured | Status |
| --- | --- | --- | --- |
| Ordinary conditional injection | at most 5% | 1 of 20 and 0 of 20 on the fifth and fourth blind splits (configuration A); 0 of 20 and 0 of 4 in the shadow harness | met |
| Explicit stored-fact recall | at least 90% | 10 of 10 and 9 of 10 | met |
| Implicit memory-needed recall | at least 75% | 7 of 8 and 6 of 7 | met |
| Ambient-preference recall | at least 95% | 3 of 3 eligible preferences promoted on the fifth split; the promotion split passes on both independent builds with identical promoted sets | met |
| Helpful precision among admitted records | at least 95% | 19 of 20 (95%) and 17 of 17 (100%). One disputed label, the user's time-zone record on a scheduling turn, was adjudicated blind by an independent reviewer as helpful; under the blind labels alone the figures are 18 of 20 and 16 of 17. The adjudication is recorded in `usefulness-gate.md`. No bundle component changed. | met |
| Unsafe admissions and promotions | zero | zero placebo, misleading, private, or unsafe admissions in every run of the suite; zero unsafe promotions | met |
| Provider and budget failures return the baseline | always | every fail-closed branch of the orchestrator is tested; the one planner timeout observed in a shadow run served the draft | met |
| Stage timeouts bound the turn | a hung planner or judge costs no more than its timeout | a 5 s policy against a 50 ms timeout returns in under 1 s for both stages | met |
| Cross-principal leakage | zero | zero violations on the synthetic four-user store and on the 1,074-record fixture, across every log stage, `memory_get`, and grants | met |
| Dense-floor calibration | the configured floor inside the band whose F1 is within 90% of the best | best F1 0.99 at 0.50; the configured 0.45 sits inside the 0.44 to 0.66 band on 50 relevant and 50 irrelevant labelled queries | met |
| Warm search latency | p50 under 80 ms | p50 23.2 ms, p95 28.0 ms on the 1K fixture with the real embedder; p50 73.7 ms, p95 78.3 ms on 50K records with a fixed 25 ms embedding cost | met |
| Write latency | p50 under 150 ms | p50 26.3 ms; p95 471 ms on a write whose attribute scan reaches the NLI judge | met |
| Concurrent writers beside extraction | zero lock errors | four writers and two extraction workers on one file: 160 writes and 20 extractions in 0.2 s, write p50 0.6 ms | met |
| The supported bundle rebuilds from the package alone | components equal the recorded manifest | `bundle_components(supported_bundle(), supported_retrieval_config())` equals `bundles/bundle-2026-09-08-a.json`, retrieval hash `e8c8c3309ab121de` | met |
| Standard suite, Ruff, format, strict mypy | pass on every advertised Python version | 391 passed on 3.12 and on 3.13; clean on both, 60 source files | met |
| Local-model and slow suites | pass | 9 passed in 191.7 s on macOS arm64 | met |
| CI | green on every advertised version | run `34388291557`: `python 3.12`, `python 3.13`, `docs`, `wheel`, and `dependency audit` | met |
| Documentation builds strictly | no warnings | `mkdocs build --strict` clean | met |
| Distribution metadata | accepted by the upload path | `twine check` passes on the wheel and the sdist | met |
| Clean installation | the wheel installs and runs outside the checkout | installs with both adapter extras into fresh 3.12 and 3.13 environments and passes 24 smoke checks | met |
| Known vulnerabilities in the locked set | none unfixed | five `transformers` advisories cleared; four `chromadb` advisories have no fixed release, arrive only through the `crewai` extra, and are ignored by identifier | met, exception recorded |
| Tagged release with installable artifacts | wheel and sdist attached to a GitHub release | `v1.0.1` released with the artifacts CI built; the wheel installs from the release URL | met |
| Published package | `pip install retold` works | not on PyPI: the trusted publisher has not been created | open |

Sources: `benchmarks/results/phase0/phase11-baseline-v5` and `-v4`, `benchmarks/results/shadow/*phase11-baseline.json`, `benchmarks/results/promotion/20260908T101548Z-*.json`, `benchmarks/results/rescore/20260908T184734Z-rescore.json`, `tests/integration/`, `tests/test_utility_aware.py`, `tests/test_reference_host.py`, `scripts/wheel_smoke.py`.
