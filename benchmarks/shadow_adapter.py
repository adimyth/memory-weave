"""The benchmark's view of the shipped reference policies: the same code, wired to the run's model wrapper.

The planner, the judge, and the bundle manifest live in `retold.policy.reference` so that an application
installing the package gets the policies the fitness suite measured. This module only adapts the benchmark's
`Models` wrapper, which routes by model name and counts tokens, to the one-call client contract those
policies take. Nothing about a prompt, a parser, or a bundle component is decided here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from retold.policy.reference import (
    ADMISSION_PROMPT_VERSION,
    ADMISSION_SYSTEM,
    CATEGORY_PROMPT_VERSION,
    GAP_PROMPT_VERSION,
    GAP_SYSTEM,
    INVENTORY_BUILDER_VERSION,
    ReferenceAdmissionPolicy,
    ReferenceGapPolicy,
    policy_bundle,
)

__all__ = [
    "ADMISSION_PROMPT_VERSION",
    "ADMISSION_SYSTEM",
    "CATEGORY_PROMPT_VERSION",
    "GAP_PROMPT_VERSION",
    "GAP_SYSTEM",
    "INVENTORY_BUILDER_VERSION",
    "HostedAdmissionPolicy",
    "HostedGapPolicy",
    "ModelsClient",
    "policy_bundle",
]


class ModelsClient:
    """Adapt the benchmark's `Models` wrapper to the package's completion-client contract.

    The wrapper counts tokens cumulatively per model, so `last_usage` reports the difference across the call
    just made and the policies record that on their decision. It ignores `timeout_s`: a run measures the
    model, and the orchestrator's stage timeout is what bounds a turn.
    """

    def __init__(self, models: Any, model: str) -> None:
        self._models = models
        self._model = model
        self._before: Mapping[str, int] = {"prompt": 0, "completion": 0}
        self._after: Mapping[str, int] = {"prompt": 0, "completion": 0}

    def complete(self, system: str, user: str, *, timeout_s: float) -> str:
        del timeout_s
        self._before = self._snapshot()
        raw = str(self._models.complete(self._model, system, user, json_mode=True))
        self._after = self._snapshot()
        return raw

    def last_usage(self) -> Mapping[str, int]:
        return {key: self._after.get(key, 0) - self._before.get(key, 0) for key in ("prompt", "completion")}

    def _snapshot(self) -> Mapping[str, int]:
        return dict((getattr(self._models, "usage", {}) or {}).get(self._model, {"prompt": 0, "completion": 0}))


class HostedGapPolicy(ReferenceGapPolicy):
    def __init__(self, models: Any, model: str) -> None:
        super().__init__(ModelsClient(models, model), model)


class HostedAdmissionPolicy(ReferenceAdmissionPolicy):
    def __init__(self, models: Any, model: str) -> None:
        super().__init__(ModelsClient(models, model), model)
