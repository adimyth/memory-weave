"""The reference policies the fitness suite measured, in the package that ships them.

A host that installs Retold and wants the supported bundle has to be able to build it from what the wheel
contains. These are the planner, the judge, and the record classifier the suite ran, with their prompts and
their versions, together with the retrieval settings those runs used and the bundle manifest that names all
of it. Nothing here is a benchmark harness: the harnesses in `benchmarks/` import this module, so the
policies a run measures and the policies an application installs are one implementation.

The classes take a `CompletionClient`, which is the same one-call contract the extractor and the reviewer
take, so an application supplies its own model access. A client that also reports the token usage of its
last call, through a `last_usage()` method, has that usage recorded on the decision; one that does not
simply records none.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Protocol

from retold.config import DenseFloorConfig, RetoldConfig, load_config
from retold.hosted import CompletionClient
from retold.models import Record
from retold.policy.activation import POLICY_VERSION as TAXONOMY_VERSION
from retold.policy.activation import RETRIEVAL_CATEGORIES, CategoryDecision, ProfileBlock
from retold.policy.bundles import retrieval_config_hash
from retold.policy.utility_aware import (
    GAP_CATEGORIES,
    AdmissionDecision,
    CandidateVerdict,
    Gap,
    GapDecision,
    UtilityAwareConfig,
)

SUPPORTED_BUNDLE_ID = "bundle-2026-09-08-a"
SUPPORTED_PLANNER_MODEL = "gpt-4o"
SUPPORTED_JUDGE_MODEL = "gpt-5.4"
SUPPORTED_CLASSIFIER_MODEL = "gpt-4o"
GAP_PROMPT_VERSION = "gap-v3c"
ADMISSION_PROMPT_VERSION = "admission-v3"
CATEGORY_PROMPT_VERSION = "category-v2"
INVENTORY_BUILDER_VERSION = "inventory-v1"
# Measured on the Phase 0 splits with eight-candidate pools: planner about 1 s, judge 2 to 5 s at p95.
GAP_TIMEOUT_MS = 4000
ADMISSION_TIMEOUT_MS = 8000

GAP_SYSTEM_BASE = (
    "You help a memory system decide what to look up before answering a user. You see the user's message "
    "and the preferences already applied. Decide whether the correct or appropriate response depends on "
    "facts specific to this user that public knowledge cannot supply. Ask: would two users in different "
    "situations receive different correct responses?\n"
    "- If the message asks for an explanation of a general concept, a best practice, how a tool works, or a "
    "generic how-to, the answer is the same for everyone. Return an empty list, even if the user's "
    "organisation might hold related records.\n"
    "- If the message asks the assistant to act on the user's behalf or produce something tailored to their "
    "situation, such as proposing or scheduling a time, writing a message to a specific person or role, "
    "producing a command or code for their own systems or project, checking a plan against their team's "
    "calendar, freezes, or policies, or stating what their team decided, uses, or owns, then the user-specific "
    "inputs to that task are gaps: their time zone, the people and roles involved, system and environment "
    "names, team schedules, prior decisions, and tooling or language choices for their project.\n"
    "List up to 3 gaps as short search queries that name the missing fact, not the topic. Do not ask for "
    "public facts or for anything already covered by the applied preferences. Reply with JSON only: "
    '{"queries": ["...", "..."]}'
)

GAP_SYSTEM = (
    GAP_SYSTEM_BASE
    + "\n\nThe memory store for this user holds facts in these categories only, and nothing outside them:\n"
    "{inventory}\n"
    "A question about a decision, constraint, limit, schedule, location, system, or person that falls in one "
    "of these categories may need memory even when it does not say my, our, or we. A question that asks how "
    "something works in general, or how to do something in general, does not.\n\n"
    "Each gap carries a category from exactly this set: preference, prior_decision, constraint, "
    'relationship_or_event, task_state. Reply with JSON only: {"gaps": [{"category": "<category>", '
    '"query": "<short search query naming the missing fact>"}]}. Return {"gaps": []} when nothing is needed.'
)

ADMISSION_SYSTEM = (
    "You decide which stored memory records, if any, should be given to an assistant before it answers. You "
    "see the user's message, the preferences already applied, a draft answer written without any of the "
    "candidate records, and the candidate records with their recorded date and status. Evaluate the "
    "candidates together. Give every candidate exactly one verdict:\n"
    "- helpful: the record changes what the answer recommends, the time or date it proposes, the person it "
    "addresses or names, the command or code it gives, a constraint or value it states, or a warning it "
    "should raise. The draft asking the user for exactly this information also counts.\n"
    "- jointly_helpful: it meets the helpful test only together with another candidate you are also admitting.\n"
    "- redundant: it repeats what the draft, public knowledge, or the applied preferences already cover.\n"
    "- insufficient: it concerns the request but changes none of the things listed under helpful. Background, "
    "attribution, and usefulness for other work the user did not ask about are insufficient.\n"
    "- stale_or_conflicting: it is marked superseded or conflicting, or a stronger candidate contradicts it. "
    "Disagreeing with the draft is not a reason for this verdict; the draft was written without the records.\n"
    "- potentially_harmful: using it risks factual distortion, agreeing with a false belief, suppressing "
    "warnings, inappropriate personalisation, or exposing private information unrelated to the request.\n"
    "Admit only candidates with verdict helpful or jointly_helpful. When you are not sure one of the listed "
    "things would change, do not admit. An empty admitted list is the normal outcome. Reply with JSON only: "
    '{"admitted": ["<id>", ...], "verdicts": [{"id": "<id>", "verdict": "<verdict>", "reason": "<one short '
    'sentence>"}]}. Include every candidate id exactly once in verdicts.'
)

CATEGORY_SYSTEM_BASE = (
    "You classify one stored memory record about a user or their work. Reply with JSON only:\n"
    '{"retrieval_category": "<one of: ' + ", ".join(RETRIEVAL_CATEGORIES) + '>",\n'
    ' "activation_category": '
    '"<one of: answer_style, response_language, accessibility, code_example_language, other>",\n'
    ' "applicability": "<broad | scoped | ambiguous>",\n'
    ' "confidence": <0.0 to 1.0>,\n'
    ' "unsafe": <true | false>,\n'
    ' "rationale": "<one short sentence>"}\n'
    "retrieval_category names what kind of fact this is. activation_category is answer_style only for "
    "instructions about how every reply should be written (length, structure, tone, format), "
    "response_language for the language or spelling of replies, accessibility for accessibility needs, "
    "code_example_language for the default programming language of code examples, and other for anything "
    "that is not an instruction about how to reply, including facts, events, and opinions. applicability is "
    "broad when the instruction applies to every reply, scoped when it applies only in a named situation, "
    "task, or topic, and ambiguous when you cannot tell. unsafe is true when the record asks the assistant to "
    "agree with the user, suppress warnings or caveats, confirm assumptions, avoid correcting them, or "
    "otherwise trade truthfulness for agreement."
)

CATEGORY_DEFINITIONS = (
    "retrieval_category definitions, choose the single best fit:\n"
    "- time_zone: the user's working time zone, working hours, or location used for scheduling.\n"
    "- people: who a named person is or what they own, lead, manage, or are responsible for, "
    "including the user's manager.\n"
    "- infrastructure: names and regions of clusters, environments, services, accounts, or hosts the user operates.\n"
    "- decisions: a choice the team made between tools, technologies, or approaches, with or without a date.\n"
    "- constraints: a version pin, compatibility rule, or technical restriction and the reason for it.\n"
    "- limits: a numeric quota, rate limit, budget, retry count, concurrency cap, or availability target, "
    "and tier multipliers on it.\n"
    "- schedules: a recurring meeting, release train, freeze window, review slot, or on-call rotation.\n"
    "- locations: where a document, runbook, handbook, checklist, or record set is kept, "
    "such as a repository path, wiki, or folder.\n"
    "- personal: facts about the user's life outside work, such as diet, family, health, hobbies, or tastes.\n"
    "- preferences: how the user wants replies written, what language or units to use, or what they like or believe.\n"
    "- other: a fact that fits none of the above.\n"
    "A record naming a numeric limit is limits even when it mentions a service. "
    "A record saying where something lives is "
    "locations even when it names a system. A record about a person's role is people even when it names a document.\n"
)

CATEGORY_SYSTEM = CATEGORY_SYSTEM_BASE + "\n" + CATEGORY_DEFINITIONS


class UsageReporting(Protocol):
    """A completion client that can report what its last call cost."""

    def last_usage(self) -> Mapping[str, int]: ...


def _usage(client: CompletionClient, model: str) -> dict[str, dict[str, int]]:
    report = getattr(client, "last_usage", None)
    if report is None:
        return {}
    counts = report()
    return {model: {"prompt": int(counts.get("prompt", 0)), "completion": int(counts.get("completion", 0))}}


class ReferenceGapPolicy:
    """The planner of the supported bundle: names the user-specific facts a draft may be missing."""

    def __init__(self, client: CompletionClient, model: str, *, timeout_ms: int = GAP_TIMEOUT_MS) -> None:
        self._client = client
        self._timeout_s = timeout_ms / 1000
        self.model = model
        self.policy_id = f"{model}/{GAP_PROMPT_VERSION}"

    def plan(
        self, turn: str, public_context: str | None, ambient_profile: ProfileBlock, inventory: Sequence[str]
    ) -> GapDecision:
        labels = "\n".join(f"- {label}" for label in inventory) or "- (no category information available)"
        system = GAP_SYSTEM.replace("{inventory}", labels)
        profile_text = ambient_profile.text or "Applied preferences: none."
        user = f"{profile_text}\n\nUser message:\n{turn}"
        if public_context:
            user = f"Public context:\n{public_context}\n\n{user}"
        raw = self._client.complete(system, user, timeout_s=self._timeout_s)
        usage = _usage(self._client, self.model)
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


class ReferenceAdmissionPolicy:
    """The judge of the supported bundle: admits only records that would change the draft answer."""

    def __init__(self, client: CompletionClient, model: str, *, timeout_ms: int = ADMISSION_TIMEOUT_MS) -> None:
        self._client = client
        self._timeout_s = timeout_ms / 1000
        self.model = model
        self.policy_id = f"{model}/{ADMISSION_PROMPT_VERSION}"

    def admit(
        self,
        turn: str,
        public_context: str | None,
        ambient_profile: ProfileBlock,
        draft: str,
        candidates: Sequence[Record],
    ) -> AdmissionDecision:
        lines = []
        for record in candidates:
            lines.append(
                f"- id={record.id} (recorded {record.event_at.date().isoformat()}, status: {record.status}): "
                f"{record.content}"
            )
        profile_text = ambient_profile.text or "Applied preferences: none."
        user = (
            f"{profile_text}\n\nUser message:\n{turn}\n\nDraft answer written without the candidates:\n{draft}\n\n"
            f"Candidate records:\n" + "\n".join(lines)
        )
        raw = self._client.complete(ADMISSION_SYSTEM, user, timeout_s=self._timeout_s)
        usage = _usage(self._client, self.model)
        parsed = json.loads(raw)
        known = {record.id for record in candidates}
        verdicts: list[CandidateVerdict] = []
        for item in parsed.get("verdicts", []):
            if not isinstance(item, dict) or str(item.get("id")) not in known:
                continue
            verdict = str(item.get("verdict", "")).strip().lower()
            if verdict not in (
                "helpful",
                "redundant",
                "insufficient",
                "stale_or_conflicting",
                "potentially_harmful",
                "jointly_helpful",
            ):
                verdict = "insufficient"
            verdicts.append(CandidateVerdict(str(item["id"]), verdict, str(item.get("reason", ""))))  # type: ignore[arg-type]
        admitted = [str(i) for i in parsed.get("admitted", []) if str(i) in known]
        return AdmissionDecision(admitted, verdicts, self.policy_id, "ok", usage=usage)


class ReferenceCategoryPolicy:
    """The record classifier of the supported bundle: proposes a retrieval and an activation category.

    It proposes only. `decide_activation` holds the rules, and the host verifies the supporting claim in the
    principal's own turns before anything reaches the ambient profile.
    """

    def __init__(self, client: CompletionClient, model: str, *, timeout_ms: int = ADMISSION_TIMEOUT_MS) -> None:
        self._client = client
        self._timeout_s = timeout_ms / 1000
        self._system = CATEGORY_SYSTEM
        self.model = model
        self.policy_id = f"{model}/{CATEGORY_PROMPT_VERSION}"

    def classify(self, content: str) -> CategoryDecision:
        raw = self._client.complete(self._system, f"Record:\n{content}", timeout_s=self._timeout_s)
        parsed = json.loads(raw)
        retrieval = str(parsed.get("retrieval_category", "other"))
        if retrieval not in RETRIEVAL_CATEGORIES:
            retrieval = "other"
        activation = str(parsed.get("activation_category", "other"))
        if activation not in ("answer_style", "response_language", "accessibility", "code_example_language"):
            activation = "other"
        applicability = str(parsed.get("applicability", "ambiguous"))
        if applicability not in ("broad", "scoped", "ambiguous"):
            applicability = "ambiguous"
        return CategoryDecision(
            retrieval_category=retrieval,
            activation_category=activation,  # type: ignore[arg-type]
            applicability=applicability,  # type: ignore[arg-type]
            confidence=float(parsed.get("confidence", 0.0)),
            unsafe=bool(parsed.get("unsafe", False)),
            rationale=str(parsed.get("rationale", "")),
        )


def supported_retrieval_config(base: RetoldConfig | None = None) -> RetoldConfig:
    """The retrieval settings every fitness run of the supported bundle used.

    The shipped defaults gate host-issued search for a host that has no judge behind it. The utility-aware
    path does have one, so the gate is a candidate control there and admission is the decision: the floors
    come down, the host-issued search returns more, and the judge throws away what does not change the
    answer. Serving the supported bundle on the shipped defaults is a different bundle that no run measured,
    which is why `bundle_components` refuses the combination rather than letting the manifest claim it.
    """

    config = base or load_config()
    floors = DenseFloorConfig(semantic=0.30, episodic=0.30, procedural=0.30)
    auto = replace(config.retrieval.gate.auto, dense_floor=floors, relative_floor=0.30)
    gate = replace(config.retrieval.gate, auto=auto)
    trigger = replace(config.retrieval.trigger, auto_k=8, auto_min_query_chars=1)
    return replace(config, retrieval=replace(config.retrieval, gate=gate, trigger=trigger))


def policy_bundle(
    gap_model: str,
    admission_model: str,
    retrieval_config: Any,
    latency_budget_ms: int | None,
    classifier: str = f"gpt-4o/{CATEGORY_PROMPT_VERSION}",
) -> dict[str, object]:
    """The versioned bundle recorded with every decision.

    ``retrieval_config`` is the whole ``RetoldConfig`` when the caller has one: the hash then covers
    the retrieval section and the reranker section together, because an enabled reranker changes what the
    judge sees as much as a gate floor does. A bare retrieval section or a plain mapping is hashed as given.
    """

    return {
        "planner": f"{gap_model}/{GAP_PROMPT_VERSION}",
        "judge": f"{admission_model}/{ADMISSION_PROMPT_VERSION}",
        "classifier": classifier,
        "taxonomy": TAXONOMY_VERSION,
        "inventory_builder": INVENTORY_BUILDER_VERSION,
        "retrieval_config_sha256": retrieval_config_hash(retrieval_config),
        "latency_budget_ms": latency_budget_ms,
    }


def supported_bundle(
    *,
    planner_model: str = SUPPORTED_PLANNER_MODEL,
    judge_model: str = SUPPORTED_JUDGE_MODEL,
    classifier_model: str = SUPPORTED_CLASSIFIER_MODEL,
    retrieval_config: RetoldConfig | None = None,
    latency_budget_ms: int | None = None,
    shadow: bool = True,
) -> UtilityAwareConfig:
    """The one configuration the fitness suite passed, ready to hand to the orchestrator.

    Every component is named here rather than left to a caller: the planner and judge prompts and their
    models, the classifier, the taxonomy, the inventory builder, the stage timeouts measured on the Phase 0
    splits, the candidate and gap caps, and the retrieval configuration whose hash the manifest records.
    Change any of them and the hash changes, which is the point: the result is a different bundle and needs
    its own fitness run before it may serve.

    It is built in shadow mode. Serving is a deliberate second step: record a passing fitness result for
    this bundle in the store, then construct with ``shadow=False``.
    """

    config = retrieval_config if retrieval_config is not None else supported_retrieval_config()
    return UtilityAwareConfig(
        gap_enabled=True,
        admission_mode="hosted_judge",
        shadow=shadow,
        max_gaps=3,
        max_candidates=8,
        gap_timeout_ms=GAP_TIMEOUT_MS,
        admission_timeout_ms=ADMISSION_TIMEOUT_MS,
        latency_budget_ms=latency_budget_ms,
        bundle=policy_bundle(
            planner_model,
            judge_model,
            config,
            latency_budget_ms,
            classifier=f"{classifier_model}/{CATEGORY_PROMPT_VERSION}",
        ),
    )
