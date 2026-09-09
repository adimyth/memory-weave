# Next phases: background writes

Local working note. Written 6 September 2026 after checking the current landscape against what Weave actually does.

## Where Weave is today

**Updated 8 September 2026.** Phase 10 is built. `ingest/extraction.py` claims a session atomically, runs the extractor and the reviewer before anything is written, validates evidence, temporal support, subject, and entity ambiguity twice, writes every accepted candidate through the same ingestor as `memory_write`, writes one summary keyed by `session:<id>`, and hands written semantic records to the activation service afterwards. `ingest/temporal.py` flags due records once and rewrites nothing. `SessionHooks` in `ingest/session.py` gives adapters the three LLD 12 calls plus the idle split. The hosted extractor and reviewer were run once together against the ten-turn test transcript with `gpt-4o`: two candidates accepted and written, one rejected by the reviewer because the quote did not support the duration the extractor had resolved, one rejected by the temporal validator because the extractor set a validity window from "for now", which the deterministic expression list does not count as temporal support, and the summary written. That is one run on one transcript, enough to show the prompts and parsers work, not an evaluation of extraction precision.

One consequence for the utility-aware bundle: the phase added `retrieval.gate.dense_floor.session_summary`, which changes the retrieval configuration hash the bundle records. Nothing the fitness scenarios exercise touches summaries, but the rule is mechanical: the Phase 11 baseline run, with neither optional stage enabled, re-records the supported bundle under the new hash.

Phases 11 and 12 followed the same day. The optional rewrite and rerank stages are built and measured in `usefulness-gate.md` section 8n, and both stay off. The operator surface in `memory_weave/operations.py` and the CLI covers expiry, retention, erasure of a record, a session, or a user, re-embedding with the floor-recalibration refusal, and snapshots; `memory_weave/runtime.py` is the composition root the adapters will use.

Phase 13 added the Deep Agents adapter, `memory_weave/adapters/deepagents.py`, behind the `deepagents` extra, with the shared contract suite it must pass in `tests/adapter_contract.py`. Phase 14 added the CrewAI adapter, `memory_weave/adapters/crewai.py`, behind the `crewai` extra; it passes the same suite, and `tests/test_adapter_equivalence.py` shows both leave the same semantic records from the same conversation. The core contracts needed no change for the second framework.

Phase 15 (9 September) added the integration suite, the 1K fixture, the calibration sweep, latency, isolation, scale, and contention tests, and `docs/acceptance-report.md`. The precision gate was settled by a blind adjudication of one disputed label without changing the bundle (`usefulness-gate.md` 8o), and `main` was tagged `v1.0.0`. After the tag, the cross-encoder was given two placements, a timeout, and a fallback, and measured against RRF alone (8p): it stays off. The items still open below are the trigger-mode question and attribute-name drift.

The paragraphs below are the note as written on 6 September, before the build.

Weave has one write path: the chatting agent calls `memory_write` during the session. That is it.

`memory_weave/ingest/session.py` is a `SessionBuffer`, a process-local cache over persisted transcripts. It is not extraction. There is no `ingest/extractor.py` and no `ingest/extraction.py`. Phase 10 in `implementation-plan.md` specifies both and neither is built.

So every claim in the README and in `current-landscape.md` about a session-end extractor describes a specification, not running code. The landscape document says so; this note is the plan for closing it.

## What the rest of the field does

Checked against each product's current documentation on 6 September 2026. Sources are listed at the bottom of `current-landscape.md`.

| System | Background pass | Reads | Default |
| --- | --- | --- | --- |
| ChatGPT (Dreaming, 4 June 2026) | Yes, called dreaming | Across many past conversations | On, rolling out |
| Letta | Yes, called dreaming | Recent conversations | Off unless configured at agent creation |
| LangMem | Yes, `create_memory_store_manager` | The transcript, after the chat | Host runs it |
| AgentCore | Yes, per enabled strategy | Events, after `CreateEvent` or `IngestData` | Strategies must be enabled |
| Mem0 | Extraction on `add`, not a separate pass | The messages handed to `add` | On |
| Claude Code | No. Claude writes memory files during the session | The session | Auto memory on by default |
| Memory Weave | Specified, not built | | |

Four independent products converged on the same shape: the chatting model is not the only thing that decides what gets remembered, and a second pass reads the transcript afterwards. Two of them independently named it dreaming. Weave's Phase 10 is in that family, so building it is not a novelty, it is catching up to the table stakes.

## What Phase 10 now covers, and what remains deferred

Phase 10 now covers single-session extraction, review-before-apply, temporal metadata derived from explicit evidence, and a bounded worker that flags due records without rewriting them. One transcript produces reviewed and validated candidate records plus one session summary, and the run is claimed atomically.

Cross-session consolidation remains deliberately deferred.

### Review before applying: moved into Phase 10

Letta has an optional setting where a second background conversation revises the proposed memory updates before they are committed. Weave has an adjacent mechanism and it is not the same one: `policy/lifecycle.py` writes `provisional` for `agent_inference` and `confirmed` for direct evidence, with a TTL on provisional records. That is a judgement about the **source**, made at write time by a rule. It is not a reviewer looking at the candidate.

The gap matters most for extraction, because an extractor proposes in bulk from a whole transcript with no user in the loop. Evidence location and entailment validation catch unsupported claims; the reviewer adds a separate judgement about durability, contextual completeness, and whether the candidate would change a future action.

**Decision.** Candidates remain transient until a separate reviewer accepts, rejects, or narrows them. Accepted candidates then enter the ingestor, and source plus evidence still determine lifecycle status. The reviewer cannot strengthen authority, replace evidence, change scope, or substitute the primary entity.

### Cross-session consolidation: next phase

Dreaming reads across many conversations, not one. AgentCore consolidates newly extracted information against existing records per strategy. Phase 10 extracts from one session and writes through `Ingestor.write`, which handles conflict on a current-fact key one record at a time.

What is missing is a pass that looks at the store as a whole when no new candidate touches an older record. Ordinary writes already compare every bounded active attribute for the same entity and can collapse alternate attribute names, so a periodic pass is justified only if evaluation still finds fragmentation outside that path.

**Decision.** Keep this for the next phase. Use the completed Phase 9a findings and later extraction metrics to decide whether the problem is vocabulary control, entity repair, conflict review, summary compaction, or true whole-store consolidation.

### Time-driven review: moved into Phase 10

The most interesting one, and nothing in Weave addresses it. OpenAI's stated example is a memory reading "you're going to Singapore in July" rewriting itself to "you went to Singapore in July 2026" once the trip is over, with no user action and no contradicting statement. The fact changed because the calendar moved.

Weave's supersession is triggered by a new claim on the same current-fact key. Nothing revisits a record because time passed. The store has the raw material: `event_at` on the record, `expires_at` on provisional records and session summaries. What it does not have is a pass that reads them.

This is not the same as Graphiti's validity windows. Graphiti answers "what was true in February" by keeping an interval. Time-driven revision answers "this record is now describing the past" by rewriting the record. A store could do either, both, or neither, and Weave currently does neither.

**Decision.** The extractor may set `valid_from`, `valid_until`, and `review_at` only from temporal expressions in evidence, resolving relative expressions against the turn timestamp. A scheduled pass atomically flags records when `review_at` arrives and records `review_flagged_at`; it does not infer that a plan happened or rewrite the source-backed claim.

## Ordering

1. Complete and run Phase 9a before beginning the extractor.
2. Build Phase 10 with extraction, review-before-apply, temporal metadata, and due-review flagging.
3. Evaluate cross-session consolidation as a later phase only if measured fragmentation remains after normal ingestion and attribute aliasing.

## What this does not change

The gate stays as it is. None of the above is about retrieval. The background write path decides what is in the store; the gate decides what leaves it. They are separate arguments and conflating them is how a memory system ends up writing more and returning more at the same time.

---

## Two behaviours the experiment harness surfaced, 6 September 2026

Found while running Weave through `agent-memory-experiments` against the same
transcript as Mem0, LangMem, and Graphiti. Both reproduce in isolation, outside that
harness, with two writes and no distractors. Neither is a harness artefact.

### 1. The supplied `attribute` is ignored

Two semantic writes about the same subject with different explicit attributes both
land under the first one's attribute, and the second supersedes the first:

```
write attr='test_runner'  -> {'ok': True, 'outcome': 'created'}
write attr='commit_style' -> {'ok': True, 'outcome': 'superseded:01a07728-19f3...'}

stored: attr='test_runner' status=superseded  :: Aditya writes tests in pytest
stored: attr='test_runner' status=provisional :: Aditya keeps commit messages conventional
```

"Aditya writes tests in pytest" and "Aditya keeps commit messages conventional" are
not the same claim and do not conflict. The result is that a person can hold only one
live semantic fact at a time, which contradicts the documented contract: a record's
identity is `entity + attribute`, and one live record exists per key.

Worth checking whether the caller's `attribute` is being used at all in conflict
detection, or whether the equivalence judge's entailment floor is deciding the key
before the attribute is consulted. The floor is a plausible culprit: an NLI
cross-encoder asked whether one statement about Aditya entails another statement
about Aditya may score high on subject overlap alone.

### 2. Supersession is inconsistent in the other direction

In the same run, Rohan's genuine employer change did **not** supersede.
`Rohan works at Nimbus` and `Rohan works at Lattice` were both left live, both under
`attribute='employer'`, with `supersedes_id` unset on the newer record. Default
`memory_search` for the March question returned both.

So the same mechanism merges two facts that should stay separate and keeps separate
two facts that should merge. Whatever is deciding, it is not the current-fact key.

### 3. Retrieval behaviour, for the record

Against a 45-row store, Weave returned fewer rows than its `k` on 9 of 15 probes,
which is the only system in that comparison where anything but the result cap ever
binds. But on the two ordinary turns, the case the gate exists for, it returned a
full eight rows every time. The floors are uncalibrated starting values, so this is a
measurement of the current configuration and not of the design. It does mean the
"type 3 returns empty" claim has now been tested once and does not hold yet.

### Ordering impact

These move ahead of everything in the list above. A background extractor that writes
in bulk into a store where unrelated facts supersede each other would multiply the
problem rather than expose it. Fix the key handling, then run Phase 9a, then build
Phase 10.

## Open: do the three trigger modes still earn their place?

Raised 6 September 2026, after the Phase 9a reproduction. Not decided.

The three modes exist to answer "who calls `memory_search`". Two results make that framing look weaker
than it did:

- The serving model searched on zero of six applicable turns in `hybrid` and zero to one in `tool_only`,
  across two full executions. `tool_only` is not a working mode on this evidence, it is the absence of one.
- Every record that mattered was a profile record about the principal, and those should not be gated or
  searched per turn at all. They belong in a bounded always-present block. See [gate.md](gate.md).

If profile records become ambient, the remaining question is not "who triggers a search" but "when is a
topical lookup worth doing", which is a narrower question and may not need three modes to express. `auto`
in particular exists only as an experimental control and has never been run.

Revisit after the ambient-profile experiment. Deciding now would be deciding without the result that
changes the shape of the question.


## Accepted for now, 6 September 2026

Two known defects are accepted rather than fixed, so that Phase 10 is not blocked on them.

**Topical adjacency in the gate.** A turn sharing a subject with a stored fact that does not answer it
still admits that fact. Bounded, measured, and covered in [gate.md](gate.md) section 9a.

**Attribute-name drift.** Three executions produced four different slugs for the same stated preference:
`response_style`, `communication_preference`, `response_style_preference`, and `answer_style`. Facts
recorded under different slugs never supersede each other, so both rows stay live and retrievable. The
Phase 7a attribute-aliasing pass catches some of this within a scope but will not catch it reliably
across sessions.

Neither is closed. Both are the argument for the attribute-vocabulary item, which stays on the list.
