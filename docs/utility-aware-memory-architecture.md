# Utility-aware memory architecture

Status: authoritative design for host-issued memory use. The implementation is evaluation-gated and disabled by default until the acceptance criteria in [utility-aware-memory-implementation-plan.md](utility-aware-memory-implementation-plan.md) are met.

This document defines how Memory Weave decides whether stored information will improve an answer. [usefulness-gate.md](usefulness-gate.md) remains the research and experiment record that motivated this design; where the two documents differ, this document is authoritative. [design-contributions.md](design-contributions.md) is a plain-language account of what is distinctive relative to RUMS and TRACE.

## 1. Problem and decision boundary

The current retrieval gate answers whether a record concerns a query. Host-issued retrieval needs a different decision: whether giving a record to the serving model will improve the response compared with withholding it.

For a request (q), public context (c), bounded ambient profile (P), and conditional-memory subset (S), the target is:

\[
\Delta U(S)=U(y\mid q,c,P,S)-U(y\mid q,c,P,\varnothing)-C(S)
\]

The cost term covers tokens, latency, privacy exposure, distraction, staleness, and harm. Conditional memory is admitted only when its expected net utility is positive.

Three related signals must not be conflated:

- Relevance asks whether a record concerns the request.
- Informativeness asks whether a record changes the response distribution.
- Helpfulness asks whether the changed response is better for the user and task.

RUMS entropy reduction is an informativeness signal, not a production label in this design. TRACE-style reference likelihood gain measures compatibility with a chosen answer and is retained only as an optional analysis on the hand-authored scenario set. Neither is the definition of helpfulness.

## 2. Memory activation tiers

Every record has an explicit activation mode:

- `ambient` records are stable, broadly applicable preferences that shape how the assistant responds.
- `conditional` records are facts, events, decisions, and procedures that must pass retrieval and admission for a particular turn.

`conditional` is the storage default, not the permanent outcome for every new record. Extraction and model-issued writes cannot silently set ambient activation in their write payload. After the normal evidence and lifecycle checks commit a record, an activation policy evaluates eligible principal preferences and either promotes them, leaves them conditional, or sends them to review. Every decision is audited.

The ambient profile contains only active, non-expired, confirmed semantic records about the principal whose activation is `ambient`. The assembler follows supersession, excludes unresolved conflicts, orders records deterministically, and applies count and token budgets. It builds the profile once at session start and does not mutate the session's copy afterward.

Style, response language, accessibility requirements, and similarly broad preferences are eligible for ambient activation. Episodic events, task-specific facts, ownership records, and procedures remain conditional.

The session-stable profile preserves prefix caching within a session. A later session sees promotions, revisions, expirations, and supersessions that occurred after the earlier profile was assembled.

### 2.1 Audited activation policy

Automatic consideration is limited to semantic records about the principal that are active, non-expired, and free of unresolved conflicts. Direct evidence is host-verified from the persisted transcript: the host locates a supporting claim in the principal user's own turn and stores that turn reference as activation evidence. Successful host verification establishes confirmed direct-evidence status for activation even when the extractor omitted or misquoted evidence; this transition is audited and does not depend on a model-authored quote. Failing any eligibility check leaves the record conditional and records the reason without invoking the activation policy.

Eligible records receive one of three activation decisions:

- `promote` applies only when the host-verified claim explicitly states a broadly applicable response instruction and the policy assigns the reviewed category `answer_style`, `response_language`, or `accessibility`. Examples include “Always answer me concisely” and “Reply in Hindi.”
- `conditional` applies when the preference has a task, topic, time, location, or other trigger that should be evaluated per turn.
- `review` applies when broad applicability, attribute classification, or safety is ambiguous.

The classifier or reviewer cannot expand the category allowlist. A promotion requires both host-verified eligibility and a `promote` decision; otherwise the record remains conditional. The stored attribute slug is not an eligibility input: it is logged beside the assigned category so drift between extractor vocabulary and policy categories is measurable. Policy timeout, malformed output, or low confidence creates a review item rather than promoting the record.

The activation review queue stores the record ID, activation-evidence turn reference, proposed decision, reason, policy and prompt versions, confidence, creation time, status, and reviewer resolution. Successful host evidence verification appends `record.activation_evidence_verified`; a trusted reviewer may then promote, keep conditional, or reject the record. Resolution appends `record.activation_reviewed`; automatic decisions append `record.activation_decided`; actual activation changes continue to append `record.activation_changed`.

Supersession, contradiction, expiry, deletion, or loss of confirmed status removes an ambient record from newly assembled profiles without waiting for review. A new direct user statement that contradicts an ambient preference follows normal authority rules and triggers activation evaluation for the surviving record. Existing sessions keep their immutable profile until the next session boundary.

## 3. Runtime flow

The host performs baseline generation and gap planning concurrently. The baseline already includes public context and the ambient profile, so conditional memory is measured only by what it adds beyond both.

```text
Session start
    -> assemble bounded ambient profile

User turn
    +-> generate baseline draft from public context + ambient profile
    +-> generate zero to three missing-information gaps
          -> no gaps: return baseline draft
          -> retrieve conditional candidates using the gap queries
               -> no candidates: return baseline draft
               -> jointly admit a compact subset against the draft
                    -> EMPTY: return baseline draft
                    -> admitted subset: regenerate with admitted evidence
```

Gap generation, retrieval, and admission fail closed. A timeout, malformed policy response, unavailable provider, or internal policy error withholds conditional memory and returns the baseline draft. Failure does not remove the already assembled ambient profile.

The orchestration applies only to host-issued searches. Model-issued `memory_search` calls remain available in `tool_only` and `hybrid` modes and continue through the existing relevance pipeline. This preserves explicit searches for historical questions, named entities, time ranges, and provenance checks.

Added latency is not uniform across turns. When gap planning returns an empty list before the baseline draft finishes, the draft is the final response and the path adds nothing. Only turns with a non-empty gap pay for retrieval, admission, and a second generation. The per-request latency budget in section 3.3 lets each caller decide how much of that tail it will accept.

### 3.1 Gap planning

The gap policy receives the user turn, public context, and ambient profile. It returns zero to three descriptions of user-specific or prior-interaction information that could materially change the baseline answer.

Each gap contains a stable category and a retrieval query. Initial categories are `preference`, `prior_decision`, `constraint`, `relationship_or_event`, and `task_state`. An empty list is a successful no-search decision, not an error.

The gap query, rather than the raw turn, drives host retrieval. The original turn remains `SearchRequest.context` and is retained in the audit trail. Candidate generation stays scope-filtered and uses the existing dense, lexical, entity, fusion, freshness, deduplication, reranking, and budget stages.

### 3.2 Candidate admission

Admission receives the complete bounded candidate set in one request so it can recognize complementary evidence. The initial cap is eight records. Existing relevance and freshness scores are features and candidate controls, not admission verdicts.

The admission policy compares the candidates with the baseline draft and either selects a compact subset or returns `EMPTY`. Every candidate receives one reason code:

- `helpful`: independently improves correctness, personalization, or the next action.
- `redundant`: repeats information already present in the baseline, public context, or ambient profile.
- `insufficient`: concerns the request but cannot support a material response change.
- `stale_or_conflicting`: is outdated, unresolved, or contradicted by stronger evidence.
- `potentially_harmful`: risks factual distortion, inappropriate personalization, privacy exposure, or unsafe behavior.
- `jointly_helpful`: is useful only with one or more other admitted records.

When admission returns `EMPTY`, the draft is the final response. When it admits evidence, the serving model generates once more from the same baseline inputs plus the admitted, source-traceable records.

### 3.3 Per-request latency budget

Every call to the orchestrator may carry `latency_budget_ms`, the maximum time the caller will accept beyond what the baseline draft already costs. The host sets it per turn, so one deployment can serve a chat surface that tolerates a few seconds and a completion or voice surface that tolerates none, without separate configurations or code paths.

The budget covers everything the utility-aware path adds: the portion of gap planning that outlasts the draft, retrieval, admission, and the final regeneration. The ambient profile is never charged against it, because the profile is assembled at session start and is present in the baseline.

Semantics, in order of evaluation:

- Unset means the configured default `retrieval.trigger.latency_budget_ms` applies. A default of `null` means no per-request bound, and only the per-stage timeouts limit the path.
- A budget of zero skips gap planning, retrieval, and admission for that turn. The response is the baseline draft with the ambient profile. This is the pre-existing behavior for conditional memory and requires no separate mode flag.
- A positive budget is a deadline measured from the moment the baseline draft completes. Gap planning still starts concurrently with the draft; its effective timeout is the smaller of `gap_policy.timeout_ms` and the remaining budget. The same clamp applies to admission.
- Before regeneration, the orchestrator compares the remaining budget with the observed latency of this turn's baseline draft, which is the best available estimate of a second generation. If the remaining budget is smaller, admitted records are not applied, the draft is returned, and the decision is recorded as `admitted_not_applied` with reason `budget_exhausted`.

Every budget-driven withholding fails closed in the same way as a timeout: the baseline draft is returned, conditional memory is withheld, and the turn-decision log records the requested budget, elapsed time per stage, and the reason. Shadow mode records what the budget would have permitted and never injects.

The budget does not change what admission decides, only whether the path runs to completion. A record withheld for budget reasons is not a rejected candidate and must not be labeled as one.

## 4. Provider-neutral contracts

The core package owns contracts and deterministic orchestration. Provider SDK clients remain in adapters and examples.

```python
class GapPolicy(Protocol):
    def plan(
        self,
        turn: str,
        public_context: str | None,
        ambient_profile: ProfileBlock,
    ) -> GapDecision: ...


class AdmissionPolicy(Protocol):
    def admit(
        self,
        turn: str,
        public_context: str | None,
        ambient_profile: ProfileBlock,
        draft: str,
        candidates: Sequence[SearchResult],
    ) -> AdmissionDecision: ...


class ProfileAssembler:
    def build(self, principal: Principal) -> ProfileBlock: ...


@dataclass(frozen=True, slots=True)
class TurnOptions:
    latency_budget_ms: int | None = None   # None: configured default; 0: baseline plus ambient profile only


class UtilityAwareOrchestrator:
    def prepare_turn(..., options: TurnOptions = TurnOptions()) -> TurnMemoryDecision: ...
```

`GapDecision` contains the structured gaps, policy identity, status, and timing. `AdmissionDecision` contains the admitted record IDs, per-record verdicts, policy identity, status, and timing. `ProfileBlock` contains rendered text, included record IDs, and budget metadata. `TurnMemoryDecision` joins the profile, gaps, search IDs, candidates, admission result, draft reference, final-response disposition, failure states, and timings. Its disposition is one of `baseline_no_gaps`, `baseline_no_candidates`, `baseline_empty_admission`, `baseline_policy_failure`, `baseline_budget_exhausted`, `admitted_not_applied`, and `regenerated`, and it carries the requested and effective latency budget alongside the per-stage elapsed times.

The orchestrator accepts callbacks for baseline and final generation so the core package does not depend on a model vendor. The reference adapter runs baseline generation and gap planning concurrently, returns the draft unchanged on the fail-closed paths, and performs at most one final regeneration.

## 5. Configuration and compatibility

All features are opt-in:

```yaml
profile:
  enabled: false
  max_records: 8
  token_budget: 400

retrieval:
  trigger:
    latency_budget_ms: null # default per-turn budget for the utility-aware path; null: per-stage timeouts only
    gap_policy:
      enabled: false
      max_gaps: 3
      timeout_ms: 2000
  admission:
    mode: disabled          # disabled | hosted_judge
    max_candidates: 8
    timeout_ms: 2000
```

Existing stores migrate records to `activation="conditional"`. Existing callers, tool schemas, and `tool_only` behavior remain compatible. Enabling only the profile feature does not enable host retrieval. Enabling a gap policy without admission may run in shadow mode but must not inject conditional memory.

`latency_budget_ms` is overridable per request through `TurnOptions`. A per-request value always wins over the configured default. Per-stage timeouts remain hard caps, so a generous budget cannot extend a stage beyond its own timeout.

## 6. Audit and learning data

`search_log` remains the audit record for an executed retrieval. A separate turn-decision log covers the wider policy, including turns where gap planning returns an empty list and no search occurs.

Each turn decision records its session and turn references, serving and policy model identifiers, ambient record IDs, gaps, associated search IDs, candidate and admitted record IDs, per-record verdicts, timings, configuration snapshot, failure states, and references to the baseline and final assistant turns. It stores identifiers rather than duplicate record bodies where the canonical record or session turn already exists.

It also records the requested and effective latency budget, the elapsed time of each stage, the baseline draft latency used as the regeneration estimate, and the final disposition. Added latency is reported separately for empty-gap turns and gap turns, because the two populations differ by an order of magnitude and a single average hides the trade being made. Decisions with disposition `admitted_not_applied` are excluded from usefulness-precision and recall calculations and counted under a separate budget-withhold rate.

Offline utility labels are keyed by turn decision, candidate subset, labeler version, serving-model family, and label provenance. Production stores two signals only: the hosted judge's admission verdict and the deletion-test verdict.

The committed scenario set uses independently hand-authored reference answers and may calculate TRACE likelihood gain as an offline experimental feature. Production turns do not have isolated references: their offline evidence is the hosted admission judge's verdict plus a deletion test that regenerates without the admitted memory and judges whether the response improved, worsened, or stayed equivalent. Production labels are explicitly marked as weak, model-judged evidence rather than independent ground truth. Likelihood gain is not calculated for production turns, RUMS entropy and a composite label are out of scope, and no offline signal runs on the request path.

The hosted judge is the shipping admission policy. Distillation and per-serving-model-family calibration are deferred until one family has at least 10,000 production turns containing both a judge verdict and deletion-test label and, over a rolling seven-day window, hosted admission exceeds either 750 ms p95 added latency or USD 2.00 per 1,000 candidate-bearing turns. Both conditions must hold, and model families cannot be pooled to satisfy the volume threshold.

## 7. Safety and operational invariants

- Scope and lifecycle filtering occur before any record text reaches a policy model.
- Conditional memory is withheld on policy failure.
- Ambient activation requires deterministic eligibility plus an audited automatic or human review decision; ambiguous and failed decisions remain conditional.
- Admission always supports `EMPTY` and treats it as the default when evidence is inconclusive.
- A per-request latency budget can only shorten the path; exhausting it returns the baseline draft and never a partially applied result.
- The final answer receives only admitted record text, never rejected candidates or offline labels.
- Model-issued search behavior does not regress when host policy features are disabled.
- Every behavioral decision is attributable to stored record IDs, model IDs, prompts, configuration, and timings.

## 8. Acceptance gates

The first validation runs the same gap-planning and hosted draft-relative admission pipeline twice: once with style and language preferences ambient and once with them conditional. The design gate requires at most 5% conditional-memory injection on ordinary turns and at least 90% recall on turns whose correct answer is a stored conditional fact. The promotion gate requires at least 95% end-to-end recall of the ambient preferences present in scenario ground truth, counts missing or unsupported extracted records as misses, permits zero unsafe automatic promotions in the safety suite, and requires a bounded review backlog with an assigned owner.
