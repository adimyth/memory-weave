# Changelog

All notable changes to Retold. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [1.0.1] - 2026-09-09

The first release on PyPI.

### Changed

- Renamed the project from Memory Weave to Retold, because the old name was taken on PyPI. The package is `retold`, the console script is `retold`, the configuration class is `RetoldConfig`, the Deep Agents middleware is `RetoldMiddleware`, and the test environment variables are `RETOLD_INTEGRATION`, `RETOLD_RUN_SLOW`, and `RETOLD_LIVE`. No behaviour changed.
- The cross-encoder reranker gained two placements (`reranker.mode`: after the RRF relevance floors, or in place of them), a stage timeout, and a configured fallback; the search log records whether each pass applied, timed out, or failed. Both placements were measured against RRF alone and neither is enabled: each removed the expected record from the judge's pool on about one turn in five (`docs/usefulness-gate.md` section 8p).
- The README was rewritten for integrators and every design document was brought up to the v1 state.

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
