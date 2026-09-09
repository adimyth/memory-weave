# Changelog

All notable changes to Retold. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [1.0.1] - 2026-09-09

The first release under the Retold name, with the closure work a v1 needed: four runtime defects fixed, the
measured policies moved into the package, and the release path proved end to end on a clean install. The wheel
and the sdist are attached to the GitHub release; PyPI publication follows once the trusted publisher exists.

### Fixed

- A stage timeout now bounds the turn. The planner and the judge run on daemon threads that are abandoned when they outlast their timeout, so `gap_timeout_ms` and `admission_timeout_ms` are latency promises rather than labels on a decision the turn still waited for.
- `memory_mode="utility_aware"` requires `trigger_mode="tool_only"`, and both adapters refuse the other combinations when they are constructed. In `auto` and `hybrid` the host-issued search appended its results to the turn before the utility-aware path ran, which put records in front of the model whatever admission then decided.
- A record written through `memory_write`, or left behind by a `memory_revise` correction, runs the activation policy the way an extracted record does, and the tool result reports its `activation` and `activation_reason`. A preference the model saved mid-session used to stay conditional however plainly it applied.
- The ambient profile is assembled on a session's first turn and held until the session ends, as the design says. It was rebuilt on every turn, so a promotion or an expiry could change the standing instructions between one turn and the next.

### Added

- `retold.policy.reference`: the planner, judge, and record classifier the fitness suite measured, with their prompts and versions, the retrieval settings every run used (`supported_retrieval_config`), and `supported_bundle()`, which returns the `UtilityAwareConfig` whose components hash to `benchmarks/bundles/bundle-2026-09-08-a.json`. The benchmark harnesses now import these rather than defining their own, so a package user can instantiate the supported bundle and the measured code and the shipped code cannot drift.
- `bundle_components(config, retrieval_config)` and the orchestrator's `retrieval_config` argument derive `retrieval_config_sha256` from the configuration a host actually retrieves with, and raise `BundleMismatchError` when a declared manifest names a different one. Both adapters pass their runtime's configuration.
- `retold.__version__`.
- `scripts/wheel_smoke.py`, an end-to-end check of an installed wheel: the bundle rebuilds, a tool write activates, one utility-aware turn decides and logs, both adapters build across the trigger-mode and memory-mode matrix, and the console script runs. CI installs the wheel with both adapter extras into a clean environment and runs it from outside the checkout.
- A `ci` workflow: ruff, ruff format, mypy, and the test suite on Python 3.12 and 3.13 with both adapters installed, a strict documentation build, the wheel job above, and a `pip-audit` of the locked dependency set.
- Turn decisions record `bundle_retrieval_verified` (migration 11), and a bundle that declares a retrieval hash while no retrieval configuration was given may not serve: the orchestrator refuses, and in shadow it warns and records the gap. A hash nothing checked reads like an approval of the configuration in front of it, so it is never allowed to pass silently.
- The README says which capabilities are supported, which are experimental and why they are off, and which documents are historical.

### Changed

- Renamed the project from Memory Weave to Retold, because the old name was taken on PyPI. The package is `retold`, the console script is `retold`, the configuration class is `RetoldConfig`, the Deep Agents middleware is `RetoldMiddleware`, and the test environment variables are `RETOLD_INTEGRATION`, `RETOLD_RUN_SLOW`, and `RETOLD_LIVE`. No behaviour changed.
- The cross-encoder reranker gained two placements (`reranker.mode`: after the RRF relevance floors, or in place of them), a stage timeout, and a configured fallback; the search log records whether each pass applied, timed out, or failed. Both placements were measured against RRF alone and neither is enabled: each removed the expected record from the judge's pool on about one turn in five (`docs/usefulness-gate.md` section 8p).
- The README was rewritten for integrators and every design document was brought up to the v1 state. `implementation-plan.md` and `docs/next-phases.md` now say plainly that they are historical, and `docs/bedrock-agentcore-comparison.md` names the code for the parts of the design that have since been built.
- The build pins `hatchling>=1.27,<1.30`. Hatchling 1.30 emits Metadata-Version 2.5, which the current publishing toolchain rejects outright: `twine check` failed with "'2.5' is not a valid metadata version", so an upload would not have got through. 1.29 emits 2.4, which carries the PEP 639 license expression.
- The `local-models` extra requires `sentence-transformers>=5.1` and `transformers>=5.10`, which clears five known transformers advisories. The integration and scale suites pass on the new versions. The four remaining `chromadb` advisories have no fixed release, reach the tree only through the `crewai` extra, and are ignored by name in the audit job; Retold neither imports nor runs chromadb.

- `UtilityAwareConfig.profile_enabled` makes the ambient profile a real switch: off, the draft sees no profile and the decision records no profile ids. The reference host's `profile` kill switch maps onto it, and the adapters honour it.
- The reference host now defaults to shadow mode (`KillSwitches.regeneration=False`); an application turns serving on deliberately.

### Added

- The MIT license.
- A documentation site at https://adimyth.in/retold/, built from the README, the design documents, and the docstrings.
- `py.typed`, so type checkers use the package's annotations.

## [1.0.0] - 2026-09-09

Tagged as `v1.0.0` under the name Memory Weave; never published to PyPI.

- Evidence-backed records in SQLite with scope, grants, source authority, supersession, and lifecycle.
- Retrieval through dense, lexical, and entity channels fused by reciprocal-rank fusion, with a gate that can return nothing and a search log that explains every decision.
- Session extraction with a separate reviewer before anything is written, temporal validation, and a due-review worker.
- The utility-aware host path: an audited ambient profile, gap planning against a content-free inventory, gap-driven retrieval, and draft-relative joint admission, fail-closed at every stage, gated on a recorded bundle fitness result. Validated offline on blind splits and a shadow harness (`docs/acceptance-report.md`).
- Deep Agents and CrewAI adapters that pass one shared contract suite.
- The operator surface: expiry, retention, erasure, re-embedding, snapshots, review queue, metrics with rollback thresholds, and the bundle registry.

[1.0.1]: https://github.com/adimyth/retold/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/adimyth/retold/releases/tag/v1.0.0
