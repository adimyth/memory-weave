"""Hosted adapters for the utility-aware orchestrator, kept out of the core package.

The gap policy and admission policy here are the prompts selected in Phase 0, expressed against the core
`GapPolicy` and `AdmissionPolicy` contracts. The gap prompt is the v3 prompt with one change: each gap
carries a category from the fixed set the orchestrator accepts. The admission prompt is the decision-impact
rule. Both return structured JSON and fail loudly on malformed output, which the orchestrator turns into a
fail-closed baseline.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from benchmarks.phase0_two_arm import _ADMISSION_SYSTEM_V3, _GAP_SYSTEM_V2
from memory_weave.models import Record
from memory_weave.policy import AdmissionDecision, CandidateVerdict, Gap, GapDecision, ProfileBlock
from memory_weave.policy.activation import POLICY_VERSION as TAXONOMY_VERSION
from memory_weave.policy.utility_aware import GAP_CATEGORIES

GAP_PROMPT_VERSION = "gap-v3c"
ADMISSION_PROMPT_VERSION = "admission-v3"
INVENTORY_BUILDER_VERSION = "inventory-v1"

_GAP_SYSTEM_V3C = (
    _GAP_SYSTEM_V2
    + "\n\nThe memory store for this user holds facts in these categories only, and nothing outside them:\n"
    "{inventory}\n"
    "A question about a decision, constraint, limit, schedule, location, system, or person that falls in one "
    "of these categories may need memory even when it does not say my, our, or we. A question that asks how "
    "something works in general, or how to do something in general, does not.\n\n"
    "Each gap carries a category from exactly this set: preference, prior_decision, constraint, "
    'relationship_or_event, task_state. Reply with JSON only: {"gaps": [{"category": "<category>", '
    '"query": "<short search query naming the missing fact>"}]}. Return {"gaps": []} when nothing is needed.'
)


def _usage_delta(models: Any, model: str, before: dict[str, int] | None) -> dict[str, dict[str, int]]:
    """Token usage of the call just made, from the wrapper's cumulative counters."""

    after = dict((models.usage or {}).get(model, {"prompt": 0, "completion": 0}))
    prior = before or {"prompt": 0, "completion": 0}
    return {model: {"prompt": after.get("prompt", 0) - prior.get("prompt", 0), "completion": after.get("completion", 0) - prior.get("completion", 0)}}


def _usage_snapshot(models: Any, model: str) -> dict[str, int]:
    return dict((getattr(models, "usage", {}) or {}).get(model, {"prompt": 0, "completion": 0}))


class HostedGapPolicy:
    def __init__(self, models: Any, model: str) -> None:
        self._models = models
        self.model = model
        self.policy_id = f"{model}/{GAP_PROMPT_VERSION}"

    def plan(self, turn: str, public_context: str | None, ambient_profile: ProfileBlock, inventory: Sequence[str]) -> GapDecision:
        labels = "\n".join(f"- {label}" for label in inventory) or "- (no category information available)"
        system = _GAP_SYSTEM_V3C.replace("{inventory}", labels)
        profile_text = ambient_profile.text or "Applied preferences: none."
        user = f"{profile_text}\n\nUser message:\n{turn}"
        if public_context:
            user = f"Public context:\n{public_context}\n\n{user}"
        before = _usage_snapshot(self._models, self.model)
        raw = self._models.complete(self.model, system, user, json_mode=True)
        usage = _usage_delta(self._models, self.model, before)
        parsed = json.loads(raw)
        gaps: list[Gap] = []
        for item in parsed.get("gaps", []):
            if not isinstance(item, dict):
                continue
            query = str(item.get("query", "")).strip()
            category = str(item.get("category", "task_state")).strip()
            if not query:
                continue
            if category not in GAP_CATEGORIES:
                category = "task_state"
            gaps.append(Gap(category, query))  # type: ignore[arg-type]
        return GapDecision(gaps, self.policy_id, "ok" if gaps else "empty", usage=usage)


class HostedAdmissionPolicy:
    def __init__(self, models: Any, model: str) -> None:
        self._models = models
        self.model = model
        self.policy_id = f"{model}/{ADMISSION_PROMPT_VERSION}"

    def admit(self, turn: str, public_context: str | None, ambient_profile: ProfileBlock, draft: str, candidates: Sequence[Record]) -> AdmissionDecision:
        lines = []
        for record in candidates:
            lines.append(f"- id={record.id} (recorded {record.event_at.date().isoformat()}, status: {record.status}): {record.content}")
        profile_text = ambient_profile.text or "Applied preferences: none."
        user = (
            f"{profile_text}\n\nUser message:\n{turn}\n\nDraft answer written without the candidates:\n{draft}\n\n"
            f"Candidate records:\n" + "\n".join(lines)
        )
        before = _usage_snapshot(self._models, self.model)
        raw = self._models.complete(self.model, _ADMISSION_SYSTEM_V3, user, json_mode=True)
        usage = _usage_delta(self._models, self.model, before)
        parsed = json.loads(raw)
        known = {record.id for record in candidates}
        verdicts: list[CandidateVerdict] = []
        for item in parsed.get("verdicts", []):
            if not isinstance(item, dict) or str(item.get("id")) not in known:
                continue
            verdict = str(item.get("verdict", "")).strip().lower()
            if verdict not in ("helpful", "redundant", "insufficient", "stale_or_conflicting", "potentially_harmful", "jointly_helpful"):
                verdict = "insufficient"
            verdicts.append(CandidateVerdict(str(item["id"]), verdict, str(item.get("reason", ""))))  # type: ignore[arg-type]
        admitted = [str(i) for i in parsed.get("admitted", []) if str(i) in known]
        return AdmissionDecision(admitted, verdicts, self.policy_id, "ok", usage=usage)


def policy_bundle(
    gap_model: str,
    admission_model: str,
    retrieval_config: Any,
    latency_budget_ms: int | None,
    classifier: str = "gpt-4o/category-v2",
) -> dict[str, object]:
    """The versioned bundle recorded with every decision."""

    from dataclasses import asdict, is_dataclass

    retrieval_repr = json.dumps(asdict(retrieval_config) if is_dataclass(retrieval_config) else retrieval_config, sort_keys=True, default=str)
    return {
        "planner": f"{gap_model}/{GAP_PROMPT_VERSION}",
        "judge": f"{admission_model}/{ADMISSION_PROMPT_VERSION}",
        "classifier": classifier,
        "taxonomy": TAXONOMY_VERSION,
        "inventory_builder": INVENTORY_BUILDER_VERSION,
        "retrieval_config_sha256": hashlib.sha256(retrieval_repr.encode()).hexdigest()[:16],
        "latency_budget_ms": latency_budget_ms,
    }
