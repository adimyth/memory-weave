"""Policy bundles: the complete set of components that produced a decision, hashed, and their fitness.

A bundle is every component that changes what the utility-aware path does: planner model and prompt, judge
model and prompt, classifier, taxonomy, inventory builder, retrieval configuration, stage timeouts, budget,
and candidate and gap caps. Any change to any of them is a new bundle hash. Active serving is allowed only
for a bundle with a recorded passing fitness result; an unapproved bundle may run in shadow mode only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from retold.store import Store

SUITE_VERSION = "fitness-suite-2026-09-07"


def bundle_hash(components: Mapping[str, Any]) -> str:
    """Stable hash of a bundle's components; key order and whitespace do not matter."""

    canonical = json.dumps(dict(components), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class BundleNotApprovedError(RuntimeError):
    """Raised when a host tries to serve with a bundle that has no recorded passing fitness result."""


@dataclass(frozen=True, slots=True)
class FitnessRecord:
    bundle_hash: str
    passed: bool
    suite_version: str
    evidence: str
    recorded_by: str
    recorded_at: str


class BundleRegistry:
    """Records fitness results and answers whether a bundle may serve."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def record(
        self,
        components: Mapping[str, Any],
        *,
        passed: bool,
        evidence: str,
        recorded_by: str,
        suite_version: str = SUITE_VERSION,
    ) -> FitnessRecord:
        digest = bundle_hash(components)
        self._store.record_bundle_fitness(
            bundle_hash=digest,
            bundle=components,
            suite_version=suite_version,
            passed=passed,
            evidence=evidence,
            recorded_by=recorded_by,
        )
        latest = self.latest(digest)
        assert latest is not None
        return latest

    def latest(self, digest: str) -> FitnessRecord | None:
        rows = self._store.bundle_fitness(digest)
        if not rows:
            return None
        row = rows[0]
        return FitnessRecord(
            row["bundle_hash"],
            row["passed"],
            row["suite_version"],
            row["evidence"],
            row["recorded_by"],
            row["recorded_at"],
        )

    def is_approved(self, components: Mapping[str, Any]) -> bool:
        latest = self.latest(bundle_hash(components))
        return latest is not None and latest.passed

    def require_approved(self, components: Mapping[str, Any]) -> str:
        digest = bundle_hash(components)
        latest = self.latest(digest)
        if latest is None:
            raise BundleNotApprovedError(
                f"Bundle {digest} has no recorded fitness result; it may run in shadow mode only."
            )
        if not latest.passed:
            raise BundleNotApprovedError(
                f"Bundle {digest} last failed the fitness suite ({latest.suite_version}); "
                "it may run in shadow mode only."
            )
        return digest

    def all(self) -> list[FitnessRecord]:
        return [
            FitnessRecord(
                row["bundle_hash"],
                row["passed"],
                row["suite_version"],
                row["evidence"],
                row["recorded_by"],
                row["recorded_at"],
            )
            for row in self._store.bundle_fitness()
        ]
