# The relevance gate, and the question it cannot answer

This document is about the one unsolved problem in Retold. Everything else in the system works:
records are written with evidence, scoped correctly, superseded when they change, and retrieved through
three channels with a full audit trail. The gate is what decides whether any of that reaches the model,
and on host-issued searches it currently decides wrong about half the time.

The measurements here come from the Phase 9a vertical slice, three runs of a fixed twelve-turn
conversation against a hosted model. Full method and raw numbers are in
[vertical-slice-findings.md](vertical-slice-findings.md).

**Status, 9 September 2026.** This document is the measurement that motivated the utility-aware path. Direction 5.1, the profile block, became ambient activation. Direction 5.2 became the gap planner and draft-relative judge, which meet the injection, recall, and safety gates on scripted blind splits (`usefulness-gate.md` 8c onward, `acceptance-report.md`). Direction 5.3, the reranker, was measured in two placements and rejected (8n, 8p). The relevance gate described here still runs on every search as a candidate filter; it is no longer asked to decide whether a turn needs memory.

## 1. What the gate does today

Every `memory_search` ends at the gate. It receives fused candidates, each carrying whatever evidence
the three retrieval channels produced, and it decides which of them, if any, the caller sees. It runs in
three steps, specified in [LLD 10.5](agent-memory-lld.md).

**Step 1, absolute floors, per candidate.** A candidate survives if any one signal holds on its own:

| Signal | Passing condition |
| --- | --- |
| Dense | Cosine is at least `dense_floor` for the record's type. |
| Lexical | Enough query terms matched, or one matched term is an identifier or entity alias. |
| Entity | The candidate came from an exact entity match. Model-issued searches only, see section 6. |

**Step 2, relative floor, across survivors.** Survivors are compared against the strongest survivor with
the same number of contributing channels, and anything far below it is dropped. Comparing within a
channel count matters: a record found by two channels scores roughly twice one found by one, so a flat
comparison would delete every single-channel result the moment any corroborated one existed.

**Step 3, the empty decision.** If nothing survives, the result is empty and `empty_reason` names the
floors the best candidate missed. Every dropped candidate keeps a `gate_reason`, and every score is
written to `search_log`, so floors can be re-swept offline against past searches without re-running them.

The gate is the reason host-issued search is viable at all. Systems that inject memory on every turn do
so unconditionally. Here a host search passes through the same gate as a model search, and the intended
outcome on an ordinary turn is nothing.

## 2. The gap

One mechanism is being asked two different questions.

**"Is this record about the same subject as this query?"** That is a property of a query-record pair.
Cosine similarity estimates it, imperfectly but usefully. This is the question the gate was built for and
the question it answers.

**"Does this turn need remembered state?"** That is a property of the turn alone. It does not depend on
which records exist, and no comparison between a query and a record can reveal it.

These two questions agree most of the time, which is why the design held together until it was measured.
They come apart on exactly the case the gate exists to catch: a turn whose subject matter overlaps the
store but whose answer does not depend on anything remembered.

The clearest example from the run. The store holds a preference that the user wants Python for code
examples. The user then asks:

> Should I use tabs or spaces in Python?

The stored record is genuinely about Python. The query is genuinely about Python. Cosine is right that
they are related. But nothing about the answer changes because of what is remembered, so injecting the
record is pure noise. The gate has no signal that distinguishes this from:

> What time zone should you use when suggesting a meeting for me?

where the stored timezone is the whole answer. Both are on-topic. Only one needs memory.

## 3. What this costs Retold

Measured across 36 host-issued searches in the Phase 9a hybrid run, taking the best dense score in each:

| Turn class | min | median | max |
| --- | --- | --- | --- |
| Ordinary, no memory needed | 0.45 | 0.54 | 0.64 |
| Memory genuinely applies | 0.52 | 0.57 | 0.62 |

The medians are three hundredths apart. The ordinary turns reach a *higher* maximum than the turns that
genuinely need memory. The classes are not separable by a threshold on this signal.

Sweeping the dense floor confirms it. Each row is the same 36 searches replayed offline at a different
floor:

| Dense floor | Ordinary turns injected | Applicable turns recalled |
| --- | --- | --- |
| 0.45 | 12 of 12 | 6 of 6 |
| 0.50 | 9 of 12 | 6 of 6 |
| 0.55 (current) | 5 of 12 | 3 of 6 |
| 0.60 | 5 of 12 | 3 of 6 |
| 0.65 | 0 of 12 | 0 of 6 |
| 0.75 | 0 of 12 | 0 of 6 |

Every threshold trades a useful recall for an unwanted injection at roughly one to one, and the window
between "recalls something useful" and "recalls nothing at all" is narrower than the overlap between the
two classes. There is no floor that keeps applicable recall while cutting ordinary injection. The benchmark plan sets a target of under 5
percent injection on ordinary turns before `hybrid` can become the default mode. That target is not
reachable by tuning `dense_floor`. The gate needs a different signal, not a better number.

This matters because `hybrid` is the only mode in which memory reached the model at all. The same run
showed the serving model searching once in three `tool_only` runs and never in three `hybrid` runs. Left
alone it wrote memory constantly and read it almost never. So the choice today is between a mode that
does not recall and a mode that pollutes, and the gate is the whole distance between them.

## 4. Every system has this problem

Almost none of them see it, because almost none of them try to answer the second question.

| System | What it does on recall | Does it face the problem? |
| --- | --- | --- |
| ChatGPT | Places saved memories into context every conversation. No per-request relevance decision. | Permanently, by design. Never measured as a failure. |
| Claude chat memory | A memory block held in context, plus explicit past-chat search. | Same for the always-on block. |
| Mem0 | `Memory.search` applies a similarity threshold, default `0.1`, then returns up to `top_k`. | A cutoff that low admits nearly any candidate, so the caller effectively always receives results. |
| LangMem | `BaseStore.search` returns up to `limit`, default 10, with no score threshold. | Nothing is ever withheld. |
| Letta | Persona and user blocks live in the prompt; archival content is searched on demand. | The in-prompt half is always injected. |
| File-based agents | A hot markdown slice is loaded at session start. | The hot slice is always present. |
| Retold | Every search passes a gate that can return nothing, and logs why. | Yes, and it is measured. |

The pattern is that returning something on every search is the norm, so the question "should anything
have been returned here" is never asked and therefore never fails. Retold is unusual in asking it,
which is why the failure is visible and quantified here and nowhere else in the table. That is the
design working as intended. The audit trail did its job. What it revealed is that the chosen signal is
not up to the question.

Nothing above suggests the other systems are getting better results. It suggests they are paying the
same cost silently, spread across every turn, and calling it personalization.

## 5. Three directions

Ordered by expected leverage, not by effort.

### 5.1 Take the always-applicable memories out of the gate's job

Look at what the model actually needed across the run: preferred answer style, preferred code language,
working timezone. Those shape *how* to answer almost any turn. Putting them behind a per-turn relevance
gate is a category error, because the honest answer to "is this relevant to this turn" is yes, always,
and the gate is being asked to make a distinction that does not exist for this class of record.

Move them into a small bounded profile block that is always present, and the gate's remaining job shrinks
to episodic and factual recall, where topical gating genuinely works. A record that answers *what* is
usefully gated by subject. A record that shapes *how* is not.

This is the ambient-profile experiment already deferred in HLD 15. The Phase 9a run is the first evidence
for it, and it changes the cost-benefit: the risk of a bounded profile is staleness and a less stable
prefix, and the measured alternative is a gate that cannot do the job at all for this class.

### 5.2 Split the decision in two

Add a turn-level judgement that runs before retrieval and answers only "could this turn depend on prior
state". It sees the turn, not the store, so it cannot be fooled by topical overlap. Only when it says yes
does the relevance gate run.

This is the piece the current design has no equivalent of. It is what separates "should I use tabs or
spaces in Python" from "what timezone should you use for me", which no query-record comparison can do.
It also fails safe: a false no costs a missed recall, which is the failure mode the design already
prefers, and it is cheap because it needs no retrieval.

### 5.3 Use the reranker as the gate for host-issued searches

The design already names this as the next lever in LLD 10.5. A cross-encoder scores the query and record
jointly rather than comparing two independently produced vectors, and it is materially better calibrated
than bi-encoder cosine.

The honest expectation is that it sharpens the boundary without moving it, because it still answers the
first question. It is worth measuring first anyway, because it is the cheapest of the three and it is
already specified, configured, and wired behind a flag.

## 6. What has already changed

One tuning change was justified by the data and is now the default. A host-issued search no longer
exempts exact entity matches from the gate. Those matches admitted memory on 3 of 12 ordinary turns and
on 0 of 6 turns where memory applied, so the exemption was pure injection when the host, rather than the
model, asked. A model-issued search keeps the exemption, because there the model named the entity on
purpose. See `retrieval.gate.auto.entity_exempt`.

This is a real improvement and it is not a fix. It removes three injections that no threshold could
remove, and leaves the overlap in section 3 untouched.

## 7. How to validate each direction cheaply

None of these needs a new live model run to get a first answer, because `search_log` stores every
candidate and every score for all 36 host-issued searches already recorded.

**The reranker, 5.3, is testable today with no model calls beyond a local cross-encoder.** Replay the
logged searches, score each candidate pair with `bge-reranker-v2-m3`, and produce the same two
distributions as section 3. If the ordinary and applicable ranges separate where cosine's did not, the
reranker is the answer and the change is a config flag. If they overlap the same way, that is settled
cheaply and permanently. This is the first thing to run.

**The profile block, 5.1, has been run. See section 9 for the result.**

**The turn classifier, 5.2, needs the smallest new experiment.** The twelve scripted turns are already
labelled by category, so the ground truth exists. Run a single cheap model call per turn asking only
whether the turn could depend on prior state, and compare against the labels. Twelve turns times three
runs is 36 calls of a few hundred tokens. If it separates the classes cleanly, it is worth building; if
it does not, the two harder directions remain.

Run them in that order. The first two cost no model calls at all, and either could remove the need for
the third.

## 8. Where this leaves the design

`tool_only` remains the supported default, and it is honest about what it gives you: memory when the
model asks, which this model rarely did. `hybrid` is the mode that delivers, and it stays experimental
until the ordinary-turn injection rate comes down by a mechanism other than threshold tuning.

The gate is not broken. It answers the question it was built to answer. The design asked it to answer a
second one, and that is the thing to fix.


## 9. Result: direction 5.1, validated offline

Run on the 36 logged host-issued searches with no new model calls. Records were split by a signal already
in the schema and needing no new field: a record whose `subject_entity_id` is the principal's own person
entity is **profile** (about the user), anything else is **topical** (about the world). On this
conversation that split is exactly preferences versus facts: `response_style`, `code_language` and
`working_timezone` on one side, `ownership` and `deployment_checklist_location` on the other.

**Finding 1: every useful recall came from a profile record.**

| Records returned on turns where memory genuinely applied | Count |
| --- | --- |
| Profile, about the user | 3 |
| Topical, about other entities | 0 |

Reproduced exactly on a second full execution from a deleted store: 3 profile, 0 topical again.

Across three runs the topical channel contributed nothing to the turns that needed memory.

**Finding 2: profile records cannot be gated, and should not be.**

| Profile candidates only | min | median | max |
| --- | --- | --- | --- |
| Ordinary turns | 0.36 | 0.49 | 0.61 |
| Memory applies | 0.52 | 0.57 | 0.62 |

Still overlapping. This is the category error made concrete. "Answer concisely" is genuinely relevant to
every turn, so a relevance score for it carries no information about whether this turn needs it. Gating
these records means admitting them by topical accident.

**Finding 3: removing profile records sharpens the topical picture, but one hard case survives.**

| Topical candidates only | min | median | max |
| --- | --- | --- | --- |
| Ordinary turns | 0.37 | 0.41 | 0.64 |
| Memory applies | 0.42 | 0.46 | 0.53 |

The medians separate where before they were one hundredth apart, but the ordinary maximum is still the
higher of the two. Every high topical score on an ordinary turn comes from one turn:

> How often should a deployment checklist be reviewed?

scoring 0.60 to 0.63 against "Priya Nair owns the deployment checklist". The turn is about the checklist,
the record is about the checklist, and knowing who owns it does not answer how often to review it. This
is the irreducible type-3 case, and no threshold reaches it because the applicable topical scores top out
at 0.47, below the noise.

### What this implies

On this transcript, host-issued topical search delivered zero value and non-zero harm. The configuration
the data supports is a bounded profile block carrying the principal's own records, always present and
never gated, plus host-issued topical search either suppressed or held to a floor above the adjacency
noise. That yields full coverage of the turns that needed memory and no ordinary-turn injection, which
neither current mode achieves.

### What this does not establish

The sample is one conversation, one model, three runs, 12 ordinary and 6 applicable turns. More
importantly the scripted conversation contains no turn that asks for a fact from an earlier session, the
"what did we decide about X" case that the topical channel exists to serve. Its absence is why the
topical channel looks worthless here, and that is a property of the script, not a finding about topical
retrieval. Before acting on the second half of the recommendation, add such turns and re-measure.

This result also says nothing about `tool_only`, where the model chooses to search and the topical
channel is doing a different job.

### Costs a profile block still carries

It must be appended after the prompt prefix rather than edited into it, or prefix caching breaks. It must
be bounded, which needs a selection rule once a user has more preferences than fit. And it reintroduces
staleness, since a preference sits in context whether or not it still holds, which is the risk the
tool-mediated design was built to avoid. Those are the trade-offs the HLD 15 experiment was written to
weigh, and this result changes their weighting rather than removing them.


## 9a. Accepted limitation

Decided 6 September 2026: the residual topical case is accepted, not fixed.

A turn that shares a subject with a stored fact but is not answered by it, such as "how often should a
deployment checklist be reviewed" against "Priya Nair owns the deployment checklist", will still admit
that record. It scored 0.63 and 0.64 in the two executions, above every useful topical score. No floor
reaches it, and the profile block does not cover it because the record is genuinely about another entity.

The cost is bounded: it is one adjacency per topically-overlapping turn, the record is true, and it is
rendered as recalled memory rather than asserted as an answer. The alternative is a usefulness judgement
per turn, which is the direction 5.2 work. That is deferred, not rejected. A literature survey of that
question and a concrete direction for it are in [usefulness-gate.md](usefulness-gate.md).

## 10. Reproducing this

The analysis is a script, not a hand calculation. It reads only `search_log` and `records` from a
completed attempt directory, makes no model calls, and re-executes no searches.

```bash
uv run python benchmarks/analyse_gate.py benchmarks/results/vertical-slice/attempt-XXXX
```

It prints the three findings this document rests on: whether a dense floor separates the turn classes on
all, profile-only, and topical-only candidates; the floor sweep; and what the turns that needed memory
actually received.

Every structural claim here survived deleting the artifacts and re-running the whole experiment. The
distributions moved by a few hundredths, the separability verdict, the shape of the sweep, and the
profile-versus-topical result did not move at all.
