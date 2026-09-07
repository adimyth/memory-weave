# Utility-aware memory implementation plan

This plan implements [utility-aware-memory-architecture.md](utility-aware-memory-architecture.md). It has three phases before real traffic, followed by controlled rollout. Distillation is deferred until production data volume and hosted-judge operating cost both justify it.

## Delivery rules

- Preserve the current `tool_only` path and public tool behavior by default.
- Use provider-neutral protocols in `memory_weave`; keep hosted clients in adapters and examples.
- Add a fake for every model-facing dependency.
- Persist the model, prompt, configuration, record IDs, and outcome for every evaluated policy decision.
- Use one Conventional Commit per completed phase and do not begin a runtime phase whose preceding evaluation gate failed.

## Phase 0: validate the target

Build a committed held-out scenario set before implementing runtime components. It must contain ordinary turns, implicit memory-needed turns, redundant records, jointly useful record pairs, stale or contradictory records, placebo records, and explicit conditional-fact questions such as “What did we decide about X?” and “Where did Priya say to keep the checklist?” The explicit questions are required because the current vertical slice does not measure recall when a stored conditional fact is the answer.

Run only the proposed online path in this phase: gap planning followed by the hosted draft-relative judge. Run the identical scenario set twice, with identical models and prompts. In the ambient arm, style and language records are preclassified as ambient and visible in the baseline profile. In the conditional arm, the same records remain conditional and must pass gap planning, retrieval, and admission. Use independently hand-authored reference answers to score both arms.

Report both arms separately and side by side. Include ordinary conditional-injection rate, explicit conditional-fact recall, implicit memory-needed recall, ambient-preference recall, placebo admission, misleading-memory admission, usefulness precision, latency, and estimated cost. Report added latency as two distributions, empty-gap turns and gap turns, each with p50 and p95, and record the baseline draft latency per turn. These numbers set the default `latency_budget_ms` and the Phase 3 monitoring thresholds; do not choose either before this measurement exists. Count ambient-promotion recall from scenario ground truth—the preferences the user actually stated—not merely from records that extraction successfully quoted or classified.

This phase has two gates. The design gate passes when the ambient arm has at most 5% ordinary conditional injection, at least 90% recall on explicit stored-fact questions, no admitted placebo or misleading memory, and no regression on implicit memory-needed turns. The promotion gate is the measured difference between the ambient and conditional arms: if leaving style and language conditional either breaches the 5% injection limit or loses their recall, reliable promotion is a blocker for rollout. Record both failures without tuning against the held-out set.

### Phase 0 outcome, 7 September 2026

Both gates ran. Full numbers are in `usefulness-gate.md` sections 8c and 8d.

- Design gate: pass on the held-out set with gap planner `gpt-4o` and gap prompt v2, admission `gpt-5.4`. Ordinary injection 0 of 20, explicit stored-fact recall 9 of 10, implicit memory-needed recall 5 of 6, no placebo, misleading, stale, redundant, or unrelated private record admitted, gap decisions identical across three repeats on every turn.
- Promotion gate: blocker. With style and language preferences left conditional they reached the answer on no turn, because gap planning never asks for how-to-answer preferences. Phase 1 is required before rollout.
- Model selection was made on a separate tuning split and confirmed once on the held-out set. `gpt-5-nano` is excluded from both roles: unstable and slow as a planner, and it admitted a misleading record as a judge. `gpt-4o` is the cost fallback for admission with a measured recall penalty on implicit turns.
- Carried into Phase 2: tighten the admission prompt's definition of helpful against a third split, because the judge over-admitted "useful for the next action" records when candidate pools were larger; keep the sentence-transformers style of record wording, a constraint explained by a failure in a later version, in the scenario set as a known hard case.
- Blind third split, pre-registered, `usefulness-gate.md` section 8e: ordinary injection 0 of 20 and no safety admission, again, but explicit recall 8 of 10 and implicit recall 5 of 7 against thresholds of 90% and 75%. Not passed. The planner decides from surface cues such as "our" and "my" and stays silent on organisation-specific questions that lack them; the tightened admission prompt traded one over-admission for one missed warning; dense-only replay retrieval missed once. The code-example-language preference as an ambient category is confirmed: the profile alone produced the right language with no retrieval.
- Consequence for Phase 1 and Phase 2: Phase 1 proceeds, since the promotion blocker and the ambient categories are settled and none of the recall misses touch it. Phase 2 does not start from the current gap prompt. It needs a structural planner input, a bounded content-free index of the fact categories the store holds for the principal, evaluated on a fourth blind split with the judge's ordinary-turn false-admission rate measured for the first time, because no ordinary turn has yet reached the judge in any run.

## Phase 1: ambient profile foundation

Add `MemoryActivation = Literal["ambient", "conditional"]` and `Record.activation`, with `conditional` as the default. Add a schema migration that backfills every existing record to `conditional`, update record persistence and serialization, and include activation in inspection responses.

Keep model-visible `memory_write` conditional-only. After a semantic record about the principal commits, run deterministic activation eligibility checks for active lifecycle state, expiration, and unresolved conflicts. The host, not the extractor or activation policy, verifies direct evidence by locating the supporting preference claim in a persisted principal-user turn and recording that turn reference. Successful verification establishes confirmed direct-evidence status for activation even if the extractor omitted or misquoted evidence; append `record.activation_evidence_verified`. A missing model-supplied quote does not make a stated preference ineligible.

Add an activation policy with `promote`, `conditional`, and `review` outcomes plus the assigned category `answer_style`, `response_language`, `accessibility`, or `other`. Automatic promotion requires host-verified direct evidence, an explicit broadly applicable instruction, and an allowed policy category. The record's normalized attribute slug is never used to authorize promotion; log it beside the assigned category so vocabulary drift is visible. Triggered or task-specific preferences remain conditional. Ambiguous, low-confidence, malformed, and timed-out decisions enter review and cannot change activation.

Add an activation-review table and trusted host operations to resolve review items or directly change activation after applying the same eligibility checks. Persist the record ID, activation-evidence turn reference, extracted attribute slug, assigned category, proposed decision, reason, confidence, policy and prompt versions, status, reviewer, and timestamps. Append `record.activation_evidence_verified`, `record.activation_decided`, `record.activation_reviewed`, and `record.activation_changed` events as applicable.

Add `ProfileConfig(enabled=False, max_records=8, token_budget=400)` and a deterministic `ProfileAssembler`. Select only active, unexpired, confirmed semantic records about the principal with ambient activation; exclude unresolved conflicts; follow authoritative supersession; then order by source authority, reinforcement time, event time, and record ID before applying count and token budgets.

Extend the vertical-slice fixture so answer style and response language are ambient while time-zone, ownership, checklist, and episodic records remain conditional. Assemble once per session and pass the immutable `ProfileBlock` to the adapter.

Tests cover migration, default activation, host discovery of a principal-user claim without a model-supplied quote, rejection of assistant and other-user claims, every deterministic eligibility rejection, safe automatic promotion independent of attribute slug, logged category-to-slug drift, task-specific conditional decisions, ambiguous and failed decisions entering review, rejected untrusted promotion, audited review resolution, host promotion and demotion, supersession of ambient records, lifecycle and conflict exclusion, deterministic profile ordering, count and token budgets, and session stability.

Done when `profile.enabled=false` is byte-for-byte compatible at the tool response level, the enabled fixture automatically promotes and includes direct-evidence style and language records without returning them from conditional retrieval, no ineligible fixture is automatically promoted, every ambiguous fixture has a resolvable review item, and the Phase 0 promotion gate passes through the implemented path.

## Phase 2: utility-aware host runtime

Add `Gap`, `GapDecision`, and `GapPolicy` to a provider-neutral policy module. A gap contains `category` and `query`; categories are `preference`, `prior_decision`, `constraint`, `relationship_or_event`, and `task_state`. Enforce zero to three non-empty, deduplicated queries and convert invalid structured output into a failed decision.

Add `AdmissionVerdict`, `CandidateVerdict`, `AdmissionDecision`, and `AdmissionPolicy`. Verdicts are `helpful`, `redundant`, `insufficient`, `stale_or_conflicting`, `potentially_harmful`, and `jointly_helpful`. Admission evaluates the complete bounded candidate set jointly and supports an explicit `EMPTY` result.

Add hosted reference adapters and fakes for both policies. The gap prompt is the v2 prompt selected in Phase 0: it asks whether two users in different situations would receive different correct answers, treats explanations of general concepts as gap-free, and treats tasks tailored to the user's situation as having that task's user-specific inputs as gaps. The reference adapter defaults to `gpt-4o` for gap planning and `gpt-5.4` for admission, with `gpt-4o` admission available as a cost option; both defaults are configuration, not code. The admission prompt compares at most eight source-traceable candidates against the baseline draft and defaults to `EMPTY` when improvement is uncertain. Any change to either prompt or model reruns the tuning and held-out scenario sets before merge.

Add disabled-by-default gap and admission configuration. In the reference turn loop, run baseline generation and gap planning concurrently. Empty or failed gaps return the baseline. For non-empty gaps, issue one host search with gap queries, preserve the original turn as context, exclude ambient records, and retain `trigger="auto"`. Empty candidates, admission timeout, malformed verdicts, unknown IDs, and inconclusive decisions return the baseline. A non-empty admitted subset causes exactly one final regeneration.

Add `UtilityAwareOrchestrator` with injected baseline generation, final generation, profile assembler, retriever, gap policy, and admission policy. Existing model-issued searches remain unchanged, and relevance scores remain candidate features rather than utility decisions.

Add `TurnOptions(latency_budget_ms: int | None = None)` as a parameter of `prepare_turn`, and `retrieval.trigger.latency_budget_ms: null` as the configured default. A per-request value overrides the default. Implement the semantics in architecture section 3.3: zero skips gap planning, retrieval, and admission and returns the baseline with the ambient profile; a positive value is a deadline measured from baseline completion; the effective timeout of gap planning and admission is the smaller of the stage timeout and the remaining budget; regeneration runs only when the remaining budget is at least the observed baseline draft latency for this turn, otherwise the decision is `admitted_not_applied` with reason `budget_exhausted`. Every budget-driven withholding returns the baseline and is logged with the requested budget, effective budget, per-stage elapsed times, and disposition. The budget never alters an admission verdict and a budget-withheld record is never recorded as rejected.

Add the turn-decision table and migration in this phase. Store one row for every host-policy turn, including empty-gap and failed-policy turns, with session and turn references, policy and serving model identifiers, ambient IDs, gaps, search IDs, candidate IDs, admission verdicts, admitted IDs, baseline/final assistant-turn references, configuration, timings, status, and failures. Reuse canonical records and session turns instead of duplicating their text.

Add shadow mode that persists the complete hypothetical decision without injecting candidates. Hosted-judge mode is the only shipping admission policy.

Tests cover structured parsing, empty gaps, gap limits, timeouts, context preservation, ambient exclusion, scope filtering, joint admission, stale and harmful rejection, unknown IDs, candidate caps, every fail-closed branch, concurrent baseline execution, baseline reuse, exactly one regeneration on admission, persisted decisions on every branch, foreign-key integrity, shadow non-injection, and unchanged tool-issued retrieval.

Budget tests cover: unset budget uses the configured default; zero budget issues no host search, no policy call, and logs a `baseline_budget_exhausted` decision; a per-request value overrides the configured default; a budget smaller than a stage timeout clamps that stage and a larger one does not extend it; budget exhaustion before admission returns the baseline with the reason recorded; an admitted subset with insufficient remaining budget for regeneration returns the baseline as `admitted_not_applied` and leaves the verdicts intact; the ambient profile is present on every budget path; shadow mode logs the would-be outcome under the given budget without injecting; and the empty-gap path reports zero added latency when gap planning completes before the draft.

Done when one acceptance run through the implemented provider-neutral path passes both Phase 0 gates, shadow mode cannot alter the final response, a zero budget reproduces the pre-existing conditional-memory behavior byte-for-byte at the tool response level, and the existing model-issued search suite remains unchanged.

## Phase 3: controlled rollout and production labels

Roll out in this order: activation-policy shadow decisions and review-queue operations; audited automatic activation for a small cohort; ambient profile rendering; full utility-aware decisions in shadow mode; hosted admission for an experimental cohort; and reconsideration of the recommended trigger mode only after production metrics meet the acceptance gates.

Every stage has an independent configuration kill switch. Monitor ordinary conditional injection, explicit conditional-fact recall from reviewed samples, ambient-preference recall, usefulness precision, harmful admission, empty rates, policy failures, added calls, token cost, and p50/p95 latency.

Report added latency separately for empty-gap turns and gap turns, and track the budget-withhold rate, the share of turns with an admitted subset that returned the baseline because the budget ran out, broken down by reason and by caller-supplied budget. A rising withhold rate on a cohort means that cohort's budget is set below what the path needs there; it is a configuration signal, not a quality regression, and must not be folded into recall. Set each cohort's default budget from the Phase 0 gap-turn p95 before enabling hosted admission for it.

Store two production label signals only: the hosted judge's original admission verdict and a deletion test. The deletion test regenerates without the admitted memory and asks the judge whether the response materially improved, worsened, or stayed equivalent. Mark these labels as weak, model-judged evidence rather than independent ground truth.

For the hand-authored scenario set only, optionally compute TRACE-style likelihood gain against the independent reference answer. Do not compute likelihood gain for production turns, and do not build a composite label.

Rollback disables the newest stage without migrating or deleting records. Ambient records remain stored when profile rendering is disabled, and conditional retrieval falls back to the existing `tool_only` behavior.

Rollout cannot complete while eligible ambient-preference recall is below 95%, an unsafe automatic promotion exists in the safety suite, review items have no assigned owner, the unresolved review backlog exceeds its configured age or size budget, ordinary conditional injection exceeds 5%, or explicit conditional-fact recall is below 90%. Done when the selected cohort remains within these gates for a full observation window and the operational owner records whether `tool_only` remains the default or `hybrid` becomes recommended.

## Deferred until production evidence justifies distillation

The hosted judge remains the shipping admission policy. A distilled policy and per-serving-model-family calibration enter planning only when both measurable conditions hold:

1. The turn-decision log contains at least 10,000 production turns with both a hosted-judge verdict and completed deletion-test label for the same serving-model family.
2. Over a rolling seven-day window for that family, hosted admission exceeds either a p95 added-latency budget of 750 ms or a cost budget of USD 2.00 per 1,000 candidate-bearing turns.

When both triggers hold, write a separate implementation plan for dataset splitting, feature design, classifier training, family-specific calibration, shadow comparison, artifact versioning, and fallback. Do not begin that work merely because Phase 3 completed, and do not pool model families to satisfy the volume trigger.

## Cross-phase acceptance suite

- Ambient preferences appear on every applicable session turn and never enter conditional retrieval.
- Eligible style, language, and accessibility preferences reach ambient activation with at least 95% recall measured against scenario ground truth, while the safety suite has zero unsafe automatic promotions.
- Ambiguous activation decisions remain conditional until an audited review resolves them, and review backlog limits block production rollout.
- Ordinary conditional-memory injection is at most 5%.
- Explicit stored-conditional-fact recall is at least 90%.
- Placebo and misleading-memory admission are zero in the safety suite.
- Existing scope, lifecycle, conflict, and tool-issued retrieval tests do not regress.
- Policy failures withhold conditional memory and preserve the baseline answer.
- A per-request latency budget of zero reproduces the pre-existing conditional-memory behavior, and budget exhaustion at any stage returns the baseline draft with the reason logged.
- Every decision is attributable to persisted model IDs, prompt versions, configuration, record IDs, and timings.
- All new features remain disabled by default, and all new records default to conditional activation.
