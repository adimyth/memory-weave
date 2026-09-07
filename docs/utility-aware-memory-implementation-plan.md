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
- Blind fourth split, pre-registered, `usefulness-gate.md` section 8f: pass. With the category inventory, the decision-impact admission rule, the real ingestor and retriever, and shadow judging: ordinary injection 1 of 20, explicit recall 9 of 10, implicit recall 7 of 7, no unsafe admission, gap decisions stable on every turn. The planner fired on every memory-needed turn including the no-possessive questions. The shadow judge would have injected an adjacent fact on 3 of 19 ordinary turns and nothing unsafe: the planner carries precision, the judge carries safety.

### Phase 0 verdict, 7 September 2026

Decision: Phase 0 is complete and the design gate is met. Phase 1 starts. Phase 2 starts from the requirements listed below, not from the original gap prompt.

Confidence that the plan is solid and worth pursuing, as stated at each checkpoint:

| Checkpoint | Evidence at that point | Confidence |
| --- | --- | --- |
| Plan written, before any Phase 0 run | 18 logged searches from one scripted conversation | 7 of 10 |
| First held-out split, section 8c | Design gate passed, implicit recall 2 of 6, planner unstable between arms | 8 |
| Tuning split and confirmation, section 8d | Implicit recall 5 of 6, planner stable, but the prompt had seen the first split's failures | 8.5 |
| Blind third split, section 8e | Pre-registered run failed both recall thresholds; cause isolated to the planner's reliance on possessives | 8 |
| Blind fourth split, section 8f | Pre-registered run passed all four criteria through the real pipeline; judge measured alone for the first time | 9 |

The move from 8.5 to 9 is not the same 8.5 recovered. Earlier the uncertainty was unknowns: whether the planner would generalise, whether the real pipeline matched the stand-in, and what the judge would do when reached. Now the uncertainty is knowns: the injection gate was met with no margin on a sample of 20, the judge admits an adjacent fact on about 16% of ordinary turns when it is reached, and the category inventory was hand-assigned for the scenario rather than derived from the store. Fewer things can still surprise, and the ones that can are listed.

What would move it further: the same pre-registered run with a second vendor's models in both roles, which is also the test of the model-agnostic claim, and a fifth split with the inventory generated from the store's activation categories rather than from the scenario file.

### Fifth split, pre-registration, 7 September 2026

Written before Phase 1A existed and before any classifier, inventory generator, or gap-anchored judge produced output. The split is `benchmarks/scenarios/phase0_v5.json`. Unlike earlier splits, no record carries a category the pipeline can read: retrieval categories, activation categories, and ambient promotion all come from the Phase 1A activation policy operating on the store, and the planner's inventory is generated from the store.

Two configurations, both run once, neither selected afterwards:

- **A**: store-generated inventory, `gpt-4o` planner with prompt v3, `gpt-5.4` judge with the decision-impact rule. A answers whether automatic inventory generation and automatic promotion work.
- **B**: identical, except the judge is gap-anchored: a record is admissible only if the judge names the planner gap it resolves, and the runner drops any admission without a valid gap reference. B answers whether Phase 2 can leave shadow mode without further judge work.

Thresholds for each configuration:

| Measure | Threshold |
| --- | --- |
| Ordinary conditional injection | at most 1 of 20 |
| Explicit stored-fact recall | at least 9 of 10 |
| Implicit memory-needed recall, conditional records | at least 75% |
| Placebo, misleading, or unrelated private admission | 0 |
| Eligible ambient preferences promoted automatically | 3 of 3 |
| Unsafe automatic promotion | 0, the "agree when confident" record must not become ambient |
| Scoped preference kept conditional | the SQL-style record must not become ambient |

Also reported, not gated: retrieval-category assignment accuracy against the split's hidden labels, shadow-judge ordinary admission rate, and for B the recall lost to gap anchoring relative to A.

If A passes, confidence moves above 9. If A fails on promotion or inventory, Phase 1A is the cause and the fix is in code, not prompts. If A passes and B fails on recall, the judge keeps the decision-impact rule and Phase 2 stays in shadow mode until the adjacency rate is addressed another way.

**Outcome, 7 September 2026.** A passed every threshold: 0 of 20 ordinary injections, 10 of 10 explicit, 7 of 8 implicit, no unsafe admission, 3 of 3 eligible preferences promoted, no unsafe promotion, scoped preference kept conditional. B met every threshold except promotion, 2 of 3, because its independent store build classified the code-example-language preference as scoped where A's had classified it as broad; over both builds eligible promotion was 5 of 6. B's gap-anchored judge is rejected on the evidence from the fourth-split offline evaluation, where it admitted a placebo, and from the fifth split, where it cost a scoped-preference recall while adding no precision. Full numbers in `usefulness-gate.md` section 8g.

Consequences. The decision-impact judge is the Phase 2 admission policy. The activation rule now treats `code_example_language` as broad unless the policy flags it ambiguous or unsafe, since the override clause is the normal shape of that preference; this change is unit-tested and must be confirmed on the next blind split before the promotion gate is considered stable. Phase 1B tightens the category taxonomy and prompt, since assignment accuracy was 15 of 24 even though inventory coverage sufficed. Confidence is held at 9 rather than raised, because the promotion gate's pass in A was not reproduced by B's build.

### Phase 1A complete, 7 September 2026

Phase 1A is complete and validated through the real ingestion, storage, and retrieval path on a blind pre-registered split. Confidence that the plan is solid and worth pursuing: **9 of 10, 90%**, held rather than raised because the promotion gate's pass was not reproduced by an independent store build.

Established:

- The activation policy separates ambient, conditional, unsafe, and review outcomes correctly on real records, with host-verified evidence and an audit event for every step.
- A store-generated inventory preserves the planner improvement: explicit recall 10 of 10, implicit 7 of 8, ordinary injection 0 of 20, no unsafe admission.
- Gap-anchored admission adds no value and is rejected permanently. It admitted a placebo on the fourth split and cost a recall on the fifth.
- The decision-impact judge and permanent shadow judging remain.

Open, and narrower than an architectural risk:

- Promotion is not yet deterministic for one form, the global default with an override clause. The fix is a frozen rule that treats recognised global-default forms as broad and uses the classifier's applicability only for forms the rule does not recognise. Self-reported classifier confidence is not a promotion input; it may only route a case to review.
- Category assignment agreement with hidden labels is low. The metric that matters is inventory coverage: whether the inventory the planner saw contained the assigned category of each record a memory-needed turn required.

Next steps, in order:

1. Reject gap-anchored admission permanently.
2. Freeze the global-default promotion rule.
3. Run a small blind promotion-focused split with global defaults, explicit overrides, topic-scoped preferences, temporary preferences, and unsafe preferences; pass requires two independent builds to promote exactly the eligible set with no unsafe promotion.
4. Proceed with Phase 1B while that validation runs.
5. Keep the decision-impact judge and permanent shadow judging.
6. Start Phase 2 only in shadow mode after the promotion rule passes.
7. Run the second-vendor matrix before claiming provider neutrality.
8. Monitor planner fire rate and shadow-judge adjacency separately.

The system has earned implementation confidence. The remaining work is making activation deterministic and proving portability, not rescuing the core retrieval architecture.

**Progress on the next steps, 7 September 2026.**

- Step 1 done: gap-anchored admission is rejected permanently; the prompt remains in the runner only as a record.
- Step 2 done: the promotion rule is frozen as `activation-v2-frozen`, form recognition first, classifier applicability only for unrecognised forms, confidence never a promotion input.
- Step 3 done: the blind promotion split passes across two independent builds with identical promoted sets, `usefulness-gate.md` section 8h. It exposed and fixed one ingestion defect: a temporary statement no longer supersedes a standing preference.
- Step 4, Phase 1B, started: trusted review resolution, direct activation change with the same eligibility checks, and backlog limits are implemented and tested as `ActivationOperations`; the operator surface is not built.
- Step 7 partly done, `usefulness-gate.md` section 8i: with Llama 3.1 8B Instruct as the second vendor, the planner role is portable, recall 8 of 8 and 8 of 8 with zero injection and nothing unsafe, while the judge role is not, implicit recall 3 of 8 and twelve unrelated admissions. Provider neutrality is established for the planner role across a hosted and an open-weight model, and remains unproven for the judge role beyond one frontier vendor. A second frontier vendor's judge run is required before the claim, and needs a configured key.
- Step 6 started: the Phase 2 core is built and tested, `memory_weave/policy/utility_aware.py`, with provider-neutral contracts, the fail-closed orchestrator, the per-request latency budget, shadow mode, and the turn-decision table. Not yet wired to a hosted adapter or run on live traffic; disabled by default.

**Confidence after these steps: 9.2 of 10, 92%.** Up from 9 because the one open defect at Phase 1A completion, non-deterministic promotion, is now closed by a frozen rule confirmed across two independent builds, and because the planner role has been shown portable. Held short of higher because the judge role is validated on one vendor only, the judge's adjacency rate when reached is unchanged, Phase 2 has not run in shadow on real traffic, and category assignment accuracy is still low even though inventory coverage sufficed.

- Step 7 done, `usefulness-gate.md` section 8i: with an OpenRouter key configured, Claude Sonnet 4.6 and Gemini 2.5 Pro both pass the judge fitness test on the tuning split, 0 of 12 injections, 8 of 8 explicit, 8 of 8 implicit, nothing unsafe. The judge role is validated on frontier models from three vendors and has failed on every non-frontier candidate. Provider neutrality is claimed for both roles with that class caveat. **Confidence: 9.3 of 10, 93%.** The remaining evidence that can move it materially is the live shadow run.

### Phase 2 shadow run, pre-registration, 7 September 2026

Recorded before the hosted adapter or the shadow harness exists.

Pass criteria for the first isolated shadow run on the vertical-slice conversation and on a scripted conversation that includes explicit stored-fact turns:

- Zero effect on served responses: the served transcript, tool schemas, session turns, activation state, and profile are byte-identical with shadow on and off, asserted by the harness on every run.
- At most 5% hypothetical ordinary-turn injection.
- At least 90% hypothetical explicit conditional-fact recall.
- At least 75% hypothetical implicit recall.
- Zero unsafe admissions, placebo, misleading, or unrelated private.
- Every unexpected admission reported, including those on memory-needed turns.
- Policy failures and latency-budget withholds reported separately from recall misses, never folded into them.

Safeguards that apply from the first shadow run:

1. **Causal isolation.** The shadow policy reads a snapshot and writes only its decision log. It never modifies prompts, tool availability, session state, memory activation, or the served response. This is a tested property, not a rule in prose.
2. **Policy bundle versioning.** Every turn decision records the bundle that produced it: planner model and prompt version, judge model and prompt version, category taxonomy version, inventory-builder version, retrieval configuration hash, and latency budget. A change to any component is a new bundle and requires a fitness-test rerun before it serves.
3. **Review surface before ambient rollout.** A minimal CLI over `ActivationOperations`, list, resolve, promote, demote, backlog, must exist before the ambient profile reaches real traffic. It may follow the first shadow run.
4. **Judge comparison on both axes.** When comparing judges on the shadow set, report adjacency admissions and useful recall together, with latency and cost. A judge that removes adjacency by rejecting useful records is a recall loss, not an improvement.

Order: hosted adapter and isolated shadow run; category taxonomy improvements measured by inventory coverage; minimal review-queue CLI; judge comparison across the three frontier vendors on adjacency, recall, latency, and cost; fitness rerun on any bundle change; a small canary only after the pre-registered shadow gates pass.

**Outcome, 7 September 2026, `usefulness-gate.md` section 8j.** The isolated shadow run passes every pre-registered gate on both the vertical-slice conversation and the fifth split: isolation identical on all four checks, hypothetical ordinary injection 0 of 4 and 0 of 20, explicit recall 1 of 1 and 10 of 10, implicit recall 7 of 8, no unsafe admission, no policy failures, no budget withholds. The first attempt failed on the orchestrator's 2-second default admission timeout, below the judge's measured 2 to 3 seconds with eight candidates; it failed closed with served responses untouched and the failures reported as failures. Stage timeouts are now harness flags set from measurement; the core default is not a serving value and the reference adapter must set it from the Phase 0 numbers.

**Confidence: 9.5 of 10, 95%.** Up from 9.3 because the live shadow evidence the plan named as the next material step is in: the full path runs through the real store with a proven zero effect on what is served and reproduces the blind-split results from its own decision log. What remains is operational rather than architectural: the review CLI before ambient rollout, the taxonomy work, the three-vendor judge comparison on the shadow set, and a canary on production traffic.

### Execution and rollout sequence, agreed 7 September 2026

This is an execution sequence, not further architectural exploration. A change to any policy-bundle component, planner model or prompt, judge model or prompt, taxonomy, inventory builder, retrieval configuration, or budget, is a new bundle: it runs in shadow and passes the complete fitness suite before it is merged into the supported bundle.

1. Improve the category taxonomy and classifier prompt, measured by inventory coverage on memory-needed turns, never by exact label agreement. Run the change in shadow.
2. Run the updated bundle against the full fitness suite: the tuning split, the held-out and blind splits, the promotion split across two builds, and the shadow harness on both conversations.
3. Build and test the minimal review-queue CLI over `ActivationOperations`: list open reviews, resolve, promote, demote, backlog status.
4. Compare the three frontier judges on the shadow set, reporting useful recall, adjacency admissions, unsafe admissions, latency, and cost together.
5. Select and version one supported policy bundle. Record the alternatives as evaluated configurations with their numbers, not as options.
6. Require a complete fitness rerun whenever any bundle component changes.
7. Start a small production canary with independent kill switches per stage and rollback thresholds defined before it starts.

The canary does not begin until all of the following hold:

- Review operations are usable through the CLI.
- The selected bundle passes the existing injection and recall gates on the full suite.
- Stage timeouts and the per-turn latency budget are configured from measured latency, not from the configuration example.
- Monitoring distinguishes planner silence, retrieval misses, judge rejection, policy failure, and budget withholding as separate counts, so a change in served injection or recall can be attributed to one stage.

**Steps 1 and 2 done, 7 September 2026, `usefulness-gate.md` section 8k.** Classifier prompt v2 adds definitions and disambiguation rules to the unchanged taxonomy; inventory coverage on memory-needed records rose from 16 of 19 to 19 of 19 on both splits. The bundle with `classifier: gpt-4o/category-v2` passed the complete fitness suite: promotion split across two builds, configuration A on the fourth and fifth splits with real activation, and the shadow harness on both conversations. It is now the default classifier in the harnesses and part of the supported bundle. Next: step 3, the review-queue CLI.

**Step 3 done.** `memory-weave reviews list | resolve | backlog` and `memory-weave activation` over `ActivationOperations`, tested end to end; `reviews backlog` exits non-zero on a breached limit so it can gate rollout.

**Steps 4 and 5 done, `usefulness-gate.md` section 8l.** On the shadow set with real retrieval and eight-candidate pools, Claude Sonnet 4.6 and Gemini 2.5 Pro match or beat `gpt-5.4` on recall and fail on everything else: each admitted a misleading record on the served path, Gemini also a private one, and both would inject on 40 to 50 percent of ordinary turns against 5 to 11 percent. The judge fitness test is the shadow-set measurement from now on; the tuning split understates precision and safety failures. Supported bundle selected and versioned as `bundle-2026-09-07-a`: `gpt-4o/gap-v3c`, `gpt-5.4/admission-v3`, `gpt-4o/category-v2`, `activation-v2-frozen`, `inventory-v1`, retrieval hash `f1f7b134bcfb7ad9`, timeouts 4 s and 8 s. Alternatives are recorded as evaluated configurations with their numbers. The judge-role portability claim is narrowed to what was measured: one model meets the full bar; two frontier alternatives meet recall only.

**Remaining before the canary:** monitoring that separates planner silence, retrieval misses, judge rejection, policy failure, and budget withholding, which the turn-decision log already carries per turn and needs only aggregation; then the canary itself with kill switches and rollback thresholds. Confidence unchanged at 9.5: the selected bundle has passed every gate, and the comparison strengthened the selection while narrowing a claim.

### Requirements Phase 0 adds to Phase 2

- The gap policy receives a bounded, content-free category inventory for the principal's readable conditional store, built from the activation policy's fixed taxonomy and never from extractor attribute slugs or record text. The inventory is part of the turn-decision log.
- The gap prompt is v3 and the admission prompt is the decision-impact rule, as committed in `benchmarks/phase0_two_arm.py`. Any change to either reruns all four scenario splits before merge.
- Host-issued retrieval runs the full candidate pipeline with recall-oriented floors; the relevance gate is a candidate control. Candidate caps stay at eight.
- Shadow judging is a permanent evaluation mode: on no-gap turns the judge is run on relevance-path candidates without effect, and its ordinary-turn admission rate is reported next to the served injection rate. A planner change that raises the planner's firing rate on ordinary turns must show the served injection rate stays within budget under that shadow rate.
- The known judge weakness is topical adjacency, not harm. Phase 2 tests must include adjacent-fact ordinary turns with the judge exercised, and the acceptance run reports both numbers.

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
