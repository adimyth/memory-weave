# Usefulness, not relevance: a direction for the gate's second question

Written 6 September 2026. Companion to [gate.md](gate.md), which establishes that the gate answers "is this record about the same subject as the turn" and is being asked "will this record change the answer". This note surveys what the 2024 to 2026 literature says about that second question and proposes a direction that can be validated on the artifacts already in `benchmarks/results`.

## 1. The question, stated precisely

The gate scores a pair, `(turn, record)`. Usefulness is not a property of that pair. It is a property of the triple `(turn, record, answer)`: a record is useful exactly when the best answer with it differs from the best answer without it.

That definition is not new. It is the value of information from decision theory, and it is the definition every recent utility-centric retrieval paper converges on independently. FILCO calls it conditional cross-mutual information, the ratio of the output likelihood with the passage to the likelihood without it. The LLM-specific utility paper defines a passage as utilitarian "if its inclusion leads to improved answer generation compared to answering without any external evidence". RUMS defines memory utility as the entropy of the response without the memory minus the entropy with it. TRACE-Memory defines it as "how much the selected evidence improves the final response beyond what the public-only path already achieves".

Three things follow, and they explain the Phase 9a result exactly.

1. **No pair score can compute it**, because the answer is missing from the pair. Cosine, lexical overlap, entity match, and a cross-encoder reranker all score the pair. Direction 5.3 in gate.md would sharpen the boundary and could not move it, for this reason.
2. **No turn-only score can compute it either**, because the record is missing. A turn classifier answers "could this turn depend on prior state", which is a prior, not the decision. Direction 5.2 is the weak form of something better, covered in section 4.
3. **It is generator-specific.** The same record is useful to a model that does not know the fact and useless to one that does. Utility judgments transfer poorly across models. A vendor-neutral host cannot hard-code it; it has to measure it per serving model.

## 2. What the literature says

Grouped by what each method judges. The column that matters for Memory Weave is whether the method works against a hosted model that exposes only text.

### 2.1 Judging the turn alone

| Work | What it does | Black-box | Result |
| --- | --- | --- | --- |
| RAGate, Findings of NAACL 2025 | Gate on whether a conversational turn needs augmentation. Prompted, fine-tuned, and attention-encoder variants. | Yes | On KETOD, where 12.1% of turns need knowledge, the best gate reaches F1 0.41. Prompt-only variants reach F1 0.12. |
| When2Call, NAACL 2025 | Benchmark for when to call a tool, ask, or abstain. | n/a | Frontier tool-calling models have "significant room for improvement". Preference optimisation helps more than fine-tuning. |
| Adaptive retrieval survey, ACL 2025 | 35 methods for deciding whether to retrieve, compared on 6 datasets. | Mixed | Plain uncertainty estimation matches or beats elaborate self-knowledge pipelines. The decision is best made from the model's own uncertainty, not from reasoning about the question. |
| TARG, 2025 | Decode a 20-token prefix without context and gate on its logit margin. | No, needs logprobs | Cuts retrieval 70 to 90% against always-retrieve while improving accuracy. Threshold set by quantile to hit a retrieval budget. |
| ENPMR-Bench, 2026 | Proactive memory retrieval from inferred latent needs. | Yes | Chain of thought lifts type-match from 0.53 to 0.59 and empathy from 4.59 to 4.66 against a gold ceiling of 4.91. |
| TriggerBench, 2026 | Detect that a stored constraint applies to a later turn when the two share no content words. | Yes | Reasoning adds 14 points. Models show an "always remind" bias: 98% on positives, 44% on negatives. Always-visible core memory beats embedding retrieval, 62% to 52%. |

Reading: a turn-only yes-or-no judgement is weak and does not get strong with more reasoning. The methods that work in this row use the model's uncertainty about its own answer, which is information about the answer, not about the turn. That matches the concern in the request that more reasoning is not the direction.

### 2.2 Judging the record given a pseudo-answer

| Work | What it does | Black-box | Result |
| --- | --- | --- | --- |
| Are LLMs good at utility judgments, SIGIR 2024 | Distinguish utility from relevance for QA passages. | Yes | Instructed LLMs can separate the two. Listwise judgement with sampling reduces order sensitivity. |
| ITEM, Findings of ACL 2026 | Interleave pseudo-answer generation, relevance ranking, and utility filtering. | Yes | Cumulative gains over a single-shot utility filter. |
| Utility-focused annotation, EMNLP 2025 | Label passages by utility against an LLM pseudo-answer instead of by relevance. | Yes | Utility labels predict downstream answer quality better than relevance labels. |
| LLM-specific utility, 2025 | Measure utility as answer improvement for a specific model; test verbalised judgements. | Yes | Pseudo-answers add 5 to 10 F1 points to utility judgements. Verbalised listwise selection still only reaches about 50 to 54 F1 on open-domain QA. Utility does not transfer across models. |
| UsefulBench, 2026 | Human labels for both relevance and usefulness on the same pairs. | Yes | Embedding retrievers align with relevance more than usefulness. LLM rankers improve usefulness but plateau. A fine-tuned 8B classifier beats GPT-4.1 on usefulness. |
| Decision-aware memory cards, CICL, 2026 | Score each context unit by whether it would change the agent's next action, via a structured judge. | Yes, hosted judges | hit@1 0.78 against BM25 0.58 on file selection. Heuristic judge weights beat learned rankers. Cards carry an explicit "trigger: when to consult" field. |

Reading: the consistent lever is conditioning the judgement on a draft answer. That is what turns the pair into a triple. The honest caveat is that verbalised utility judgement is far from perfect on hard open-domain sets. Memory Weave's setting is easier: at most eight short candidates, one principal, and records that are single claims.

### 2.3 Judging the answer delta directly

| Work | What it does | Black-box | Result |
| --- | --- | --- | --- |
| FILCO, 2023 | Filter context by CXMI, entailment, or lexical overlap; train a filter on those labels. | Labels need logprobs, filter does not | Better answers with prompts up to 64% shorter. |
| RUMS, ICML 2026 | Select user memories by entropy reduction of the response distribution; empty set allowed. | No, needs logits | Detects whether personalisation is needed at 92.3% recall and 96.0% specificity on real chats. Selects 1.58 items instead of 50. |
| TRACE-Memory, 2026 | Stage 1 generates "public-conditioned information gaps" and retrieves against them. Stage 2 admits evidence by incremental likelihood gain and entropy reduction over the public-only response, with an EMPTY action. | Training needs logprobs, inference does not | Admission F1 65.3 against 21.5 for SFT-only. EMPTY rate rises from 7.7% to 12.5% as public context gets richer, while reward rises. |
| Hindsight Memory-PRM, 2026 | After answering, delete the cited memory, re-answer, and record whether the answer flipped. Use that as credit for the memory manager. | Yes | 68% of deletions flip the answer. Intervention-calibrated credit adds 7.3 points over observational feedback. |
| AttriMem, 2026 | Black-box attribution of the answer to memory tokens by masking subsets and fitting a sparse surrogate. | Yes | Token-level attribution rewards beat outcome-only rewards on LoCoMo and LongMemEval. |

Reading: when the answer delta is measured rather than predicted, the "should memory be used at all" decision becomes accurate. RUMS is the cleanest evidence and it is white-box. TRACE-Memory shows the same decision structure can be trained offline with logprobs and then run with a text-only judge. Hindsight deletion shows the labels can be collected against a black-box generator after the fact.

### 2.4 Evidence that injection is a real cost

| Work | Finding |
| --- | --- |
| PersistBench, 2026 | Across 18 models, median failure of 53% on cross-domain leakage and 97% on memory-induced sycophancy. |
| Structured memory for over-personalisation, 2026 | Retrieval with a similarity threshold cut leakage from 56% to 24% but raised beneficial-memory failure from 23% to 71%. Reorganising the prompt cut leakage by only 4 to 9%. |
| When personalization misleads, Findings of ACL 2026 | Personalised models answer factual questions in line with user history instead of truth. |
| RootMem, 2026 | Two failure classes named explicitly: records semantically similar but logically irrelevant, and records semantically distant but logically essential. Strongest retrieval baseline reaches 43.5% on that split. |

Reading: two of these directly reproduce the Phase 9a findings on other people's systems. A similarity threshold trades beneficial recall for reduced injection at a bad rate, which is the sweep in gate.md section 3. And an always-visible core block beats retrieval for records that apply broadly, which is the profile-block result in gate.md section 9.

## 3. The thesis

Change the object being judged, not the amount of reasoning applied to it.

The pair `(turn, record)` cannot carry the usefulness signal. The turn alone cannot either. The literature that succeeds on this decision, whether white-box or black-box, always gets a draft answer into the judgement, either as a generated pseudo-answer, as the response distribution, or as an after-the-fact deletion test. Everything that stays on the turn or the pair plateaus.

For Memory Weave this is good news, because the host loop sits in exactly the right place to obtain a draft answer, and the design already prefers a missed recall over an unwanted injection, which is the failure asymmetry a utility gate needs.

## 4. The proposal: judge the answer, in three tiers

The three tiers are independent and cumulative. Each is testable offline first.

### Tier A. Elicit the information gaps, then retrieve against them

Before retrieval, ask a cheap model one question about the turn: *list the specific facts about this user, their organisation, or your prior conversations whose value would change your answer; write `none` if the answer is the same for everyone.* The output is a short list of gap descriptions, or empty.

This is TRACE-Memory's public-conditioned retrieval, and it differs from direction 5.2 in one way that matters. A yes-or-no classifier is the task RAGate showed to be weak. Naming the missing fact is a generative, grounded task, and its empty case is the "no". It also produces the retrieval query: retrieval runs against the gap descriptions, not against the raw turn.

Worked on the two hard turns:

- *"What time zone should you use when suggesting a meeting for me?"* Gaps: the user's working time zone. Retrieval on that phrase against "working timezone: Asia/Kolkata" is a query-to-answer match, which is far sharper than question-to-statement cosine.
- *"How often should a deployment checklist be reviewed?"* Gaps: none, or at most the organisation's review policy. Nothing in that retrieves "Priya Nair owns the deployment checklist". The record that scored 0.63 on the raw turn is not adjacent to the gap.
- *"Should I use tabs or spaces in Python?"* Gaps: none. The code-language preference does not enter, and it does not need to, because it is a profile record and the profile block carries it.

ENPMR-Bench is the caution here: inferring latent emotional needs gained only modestly from reasoning. Factual slots are a much narrower target, and the pass criterion in section 7 will show whether that difference holds.

### Tier B. Admit each candidate by whether it changes a draft

After retrieval, obtain a draft answer produced without memory, then ask a judge one question per candidate: *given this draft, does the record contradict it, fill a gap the draft left open, or leave it unchanged?* Only the first two admit the record.

This is the pseudo-answer utility judgement of section 2.2, applied to a small candidate set. It resolves the residual case in gate.md section 9a on its own: knowing who owns the checklist does not change a draft about review cadence, and a judge asked that question says so. It is a property of the triple, which is why the pair gate could never see it.

Two ways to obtain the draft, in order of cost:

1. **A cheap model writes a sketch.** A few hundred tokens from a small hosted or local model. This is the version to measure first.
2. **The serving model answers speculatively.** In `hybrid` today the host searches first and then calls the model once. Invert it: call the model once without memory, judge the candidates against that reply, and call again only when a candidate is admitted. On an ordinary turn this is one serving call, as today, plus one small judge call. On a turn where memory applies it is two serving calls. Since ordinary turns dominate, the expected cost rises modestly and the ordinary-turn injection rate falls to the judge's false-positive rate. The draft also reveals self-declared gaps, because a model that lacks the time zone tends to say so.

A zero-model-call variant exists for the contradiction half: the DeBERTa NLI cross-encoder already in the ingestion path can score `record` against `draft` for contradiction. It will not catch the gap-filling half, but it is free to measure.

### Tier C. Collect hindsight labels and learn a calibrated, budgeted gate

Every host-issued search already writes every candidate and every score to `search_log`. What is missing is the outcome. Add it after the answer is produced, by black-box attribution:

- **Deletion test, offline.** For each record that was injected, re-answer the turn without it and ask a judge whether the answer materially changed. This is the 68%-flip measurement from Hindsight Memory-PRM, and it needs nothing from the serving model beyond text.
- **Use test, online and cheap.** Does the final answer entail or cite the injected record? NLI again, or a one-line judge.

Stored next to the existing scores, those labels turn the gate from tuned to learned: a small calibrated classifier over cosine, reranker score, gap-match score, Tier B verdict, record type, and serving model. Three consequences:

1. **The threshold becomes a budget, not a cosine.** As TARG does, set the admission threshold by quantile so that host injection lands under a stated rate on ordinary turns. The benchmark plan's 5% target becomes a knob.
2. **Calibration is per serving model**, which is what the non-transferability result in section 2.2 demands and what no vendor-neutral system can get from a fixed number.
3. **Memory Weave gets a metric nobody else reports**: usefulness precision, the fraction of injected records that changed an answer. Given gate.md section 4, that is the differentiator the audit trail was built to enable.

UsefulBench and FILCO both found that a small fine-tuned classifier on utility labels beats prompting a frontier model for the same judgement. Once Tier C has labels, Tier B's judge is the thing it replaces.

### How the tiers sit with the existing design

- The **profile block** from gate.md section 9 stays. Records that shape *how* to answer are not gated by any tier. TriggerBench and PersistBench both support this.
- **Model-issued searches** keep the current gate. The model named its intent, so the relevance question is the right one there.
- **Host-issued searches** run Tier A, then existing retrieval with floors lowered to a recall setting, then Tier B as the final decision. `dense_floor` stops being the decision and becomes a candidate cap.
- The **mode question** in next-phases.md narrows further. If host-issued search is gap-first and admission-judged, `hybrid` is no longer "inject on every turn through a threshold". It is "inject when a named gap is filled by a record that changes the draft", which is a defensible default in a way the current `hybrid` is not.

## 5. Why this rather than the three earlier directions

Direction 5.1 is validated and kept. Direction 5.3, the reranker, still scores the pair and so cannot reach the usefulness question. It remains worth the free replay as a better candidate cap. Direction 5.2 is Tier A with the useful part removed: it asks for the verdict without asking for the gap, and the literature says the verdict alone is the weak form.

The proposal adds one thing the earlier directions all lack: a source of ground truth that accumulates in production. Without Tier C the gate can only be tuned on scripted conversations. With it every host search becomes a labelled example.

## 6. Risks the literature names

- **Verbalised utility judgements are imperfect.** About 50 to 54 F1 on open-domain QA with many passages. The setting here is smaller, but this is the number to beat, and section 7 measures it before anything is built.
- **Judges have an "always remind" bias.** TriggerBench measured 44% on negatives. Tier B's prompt must make "unchanged" the default and the pass criterion must be measured on ordinary turns, not on applicable ones.
- **Self-declared gaps can be invented.** A model asked what it is missing may list things it does not need. Tier A's output is a retrieval query, not an admission, so an invented gap costs a retrieval that Tier B then rejects. Measure the empty rate on ordinary turns.
- **Latency.** Tier A and Tier B each add a small-model call on the host path. The speculative-answer variant adds a second serving call only on admitted turns. Budget both explicitly and log them in `timings_ms`.
- **Utility is generator-specific.** A Tier B judge that is a different model from the serving model is estimating general utility, not this model's. That is acceptable for admission and is why Tier C calibrates per serving model afterwards.
- **The sample is still one scripted conversation.** The topical channel has never been exercised by a "what did we decide about X" turn. That gap in the script must be closed before any conclusion about topical recall.

## 7. Validation, in order, on artifacts that already exist

All three experiments read the two completed attempt directories under `benchmarks/results/vertical-slice`. The twelve turns are labelled by category in `examples/vertical_slice.py`.

**Experiment 1, Tier A, about 36 cheap calls.** For each turn in each of the three hybrid runs, run the gap-elicitation prompt. Record whether the list is empty. Embed each non-empty gap and score it against the stored records with the existing embedder. Produce the two distributions from gate.md section 3, but on gap-to-record scores instead of turn-to-record scores.
Pass: empty on at least 11 of 12 ordinary-turn instances per run, non-empty with the right slot on the two applicable turns that are not covered by the profile block, and the checklist-ownership record no longer the top topical score on the review-cadence turn.

**Experiment 2, Tier B, about 36 draft calls plus one judge call per logged candidate.** For each host search in `search_log`, generate a no-memory draft for the turn, then run the admission judge over every fused candidate. Also run the NLI-only contradiction variant with zero model calls. Report ordinary-turn injection and applicable recall exactly as the floor sweep does.
Pass: ordinary-turn injection at or below 1 in 12 with applicable recall unchanged. The interesting failure is the judge admitting the ownership record on the review-cadence turn; if it does, the prompt is wrong, not the approach.

**Experiment 3, Tier C, first labels, about 15 re-answer calls.** For each of the 15 host searches in the hybrid run that returned memory, re-answer the turn without the returned record and judge whether the answer changed. This tells us how many of the 5 ordinary-turn injections actually altered an answer and how many of the applicable-turn recalls did. It is the first measurement of usefulness precision and the seed of the label set.

**Then** add the missing episodic turns to the script, re-run the slice once, and repeat the three experiments on the new log before deciding what to build.

## 8. First result: Tier B replayed on the logged searches

Run on 6 September 2026 with `benchmarks/draft_delta_experiment.py` against the three hybrid run databases. Inputs are the 18 host-issued searches on the six evaluation turns (two where memory applies, four ordinary, times three runs), with the five dense candidates each search logged. The draft is written by `gpt-5.6-luna`, the same model that served the original run, from the bare user turn with no memory and no conversation history. The judge sees the turn, the draft, and all five candidates in one call and returns one verdict per candidate. Result files are under `benchmarks/results/draft-delta/`.

An earlier pass of this script was discarded: it labelled facts by the first eight characters of the record id, which are identical across records, so the judge's verdicts were assigned to the wrong candidates. The runs below use positional labels.

**Turn level. A turn counts as injected when at least one candidate is admitted.**

| Configuration | Ordinary turns injected, of 12 | Applicable turns recalled, of 6 |
| --- | --- | --- |
| Relevance gate as shipped | 5 | 3 |
| Judge `gpt-5-nano`, default prompt | 3 | 6 |
| Judge `gpt-5-nano`, default prompt, low reasoning effort | 4 | 4 |
| Judge `gpt-5-nano`, strict prompt | 0 | 0 |
| Draft also from `gpt-5-nano`, judge `gpt-5-nano` | 6 | 6 |
| Judge `gpt-5.4`, default prompt | 8 | 3 |

Read on its own that table says the approach is unstable. Split by record type it says something specific.

**Record level, topical candidates only, on ordinary turns.** These are the records the judge would actually be deciding on once profile records live in the always-present block. There are 24 such judgments per configuration.

| Judge | False admits of 24 | Ownership record on the review-cadence turn |
| --- | --- | --- |
| `gpt-5-nano`, default | 3 | unchanged 2 of 3 |
| `gpt-5-nano`, low effort | 4 | admitted 3 of 3 |
| `gpt-5-nano`, strict | 0 | unchanged 3 of 3 |
| `gpt-5.4`, default | 0 | unchanged 3 of 3 |

With `gpt-5.4` as judge the residual case from gate.md section 9a is closed on every run, with the reason "who owns the checklist does not change the recommended review cadence", and no topical record is admitted on any ordinary turn. The location record, which scored as high as the ownership record, is also judged unchanged 4 of 4 times.

**Record level, the time-zone turn.** The draft from the serving model was "Which time zone are you in?" on every run. The `gpt-5.4` judge and the default-prompt `gpt-5-nano` judge both admitted the time-zone record 3 of 3 times. The strict-prompt `gpt-5-nano` judge called it unchanged 3 of 3 times, reasoning that the draft "asks for the user's time zone without asserting a value". That is a judge failure, not an approach failure, and it is why the strict row recalls nothing.

**Why the `gpt-5.4` row shows 8 ordinary turns injected.** Every one of those admissions is the response-style preference, "concise answers that include the important trade-off", judged as filling a gap because the draft is not written that way. That verdict is correct: a style preference does change every answer. It is the category error from gate.md section 9 confirmed from the other side, and it is the reason profile records must bypass the judge and sit in the profile block. Excluding profile records, the `gpt-5.4` configuration injects on 0 of 12 ordinary turns and recalls the time-zone turn 3 of 3 times.

**What did not work.** The Python-example turn was never recalled through the code-language record, in any configuration, because the draft asked the user to paste the file rather than writing code, so there was no code for the preference to change. The draft's shape decides which gaps exist. This is a real limit of judging against a single draft, and it is another record the profile block covers.

**Cost and latency per turn, measured.**

| Call | Model | Prompt tokens | Completion tokens | Mean latency |
| --- | --- | --- | --- | --- |
| Draft | `gpt-5.6-luna` | 55 | 130 | 2.5 s |
| Judge | `gpt-5.4` | 420 | 160 | 1.9 s |
| Judge | `gpt-5-nano`, default | 425 | 1440 | 7.4 s |
| Judge | `gpt-5-nano`, low effort | 430 | 560 | 3.5 s |

The small reasoning model was slower and produced worse verdicts than the larger model, because it spent its budget on reasoning tokens and still misjudged. In the speculative design the draft is the real first answer, so the added cost on an ordinary turn is one judge call, about two seconds and 600 tokens with `gpt-5.4`. A second serving call happens only on admitted turns, which with the profile block in place was 3 of 18 searches here.

**What this establishes and what it does not.** On this conversation, a draft-conditioned judge with a competent model separates topical records that change the answer from topical records that merely share a subject, which no floor on any pair score could do. The sample is 18 searches, 108 record judgments per configuration, six distinct turns, one conversation, and no turn that asks for an earlier decision. The next measurement is the same replay on a script that contains such turns.

## 8b. Second result: the RUMS and TRACE-Memory signals replayed on the same searches

Run on 7 September 2026 with `benchmarks/utility_signal_experiment.py`. Same 18 searches and same logged candidates as section 8. The frozen model is Llama-3.1-8B-Instruct run locally, the model family RUMS used. RUMS settings follow the paper: N=5 sampled continuations of at most 20 tokens, utility is the drop in mean per-token entropy when a record is added. TRACE Stage 1 uses `gpt-5.4` as a prompted stand-in for the paper's trained query policy, with retrieval by the project's BGE-M3 embedder over every active record in the run. TRACE Stage 2 computes the paper's reward directly: the mean per-token log-likelihood gain of a reference reply with the record versus without, where the reference is the reply the serving model actually gave on that turn. A placebo record, "User's favourite colour is green", is scored on every search to show the noise floor. Result file under `benchmarks/results/utility-signals/`.

Both papers use these signals offline to train a small model that runs at inference. What is measured here is whether the signals themselves separate the classes on this data, which is the precondition for anything trained on them to work.

**The two cases that matter, per run.**

| Case | Dense cosine | RUMS utility | TRACE likelihood gain |
| --- | --- | --- | --- |
| Time-zone record on the time-zone turn | 0.61 to 0.62 | +0.19, +0.28, +0.37 | +0.95, +0.88, +0.66 |
| Ownership record on the review-cadence turn | 0.62 to 0.63 | -0.13, -0.13, +0.04 | -0.14, -0.20, -0.13 |
| Placebo record, all 18 searches | n/a | median -0.05, max +0.14 | median -0.05, max +0.19 |

Cosine cannot tell these two apart. Both utility signals do, on every run, with the time-zone record well above the placebo noise and the ownership record at or below it.

**RUMS threshold sweep on the best candidate per search, all candidates.**

| tau | Ordinary injected, of 12 | Applicable recalled, of 6 |
| --- | --- | --- |
| 0.05 | 5 | 6 |
| 0.10 | 0 | 3 |
| 0.29, the paper's value | 0 | 1 |

At tau 0.10 the three recalls are exactly the three time-zone turns and nothing ordinary passes. The three misses are the Python-example turn, where the reply asks for the file and no preference can sharpen it. The style preference gets negative RUMS utility everywhere, between -0.2 and -0.45, because telling the model how to write widens its choices rather than narrowing them. Entropy reduction measures certainty, not quality, so it is the wrong signal for how-to-answer records.

**TRACE likelihood gain, topical records only, on ordinary turns.** 24 judgments.

| Admission rule | Ordinary turns injected, of 12 | Applicable recalled, of 6 |
| --- | --- | --- |
| gain > 0.05 | 2 | 6 |
| gain > 0.10 | 1 | 6 |

The remaining false admits are the two checklist records on the trade-off-formatting turn, at +0.06 to +0.15, inside the placebo band. With all candidates included the style preference is admitted on every turn with gains of +0.4 to +1.15, and under TRACE's definition that is correct: the reference replies were written under that preference, so it genuinely raises their likelihood. Same conclusion as section 8: profile records belong in the profile block, not in front of any gate.

**TRACE Stage 1 alone.** The prompted policy returned an empty query list on 12 of 12 ordinary searches and on the 3 Python-example searches, and on the time-zone turn returned "user time zone or current location", "preferred working hours for meetings", "organization/team default time zone for scheduling". Retrieval on those queries put the time-zone record first at cosine 0.75, against 0.51 for the next record. The turn itself scored it at 0.62 against 0.49. Searching for the gap instead of the turn widened the margin from 0.13 to 0.24 and retrieved nothing at all on every ordinary turn.

**Cost.** RUMS needs eight sampled generations per search on the local 8B model, 16 to 25 seconds each here. That is not a serving-path cost in either paper and should not be one here. The TRACE likelihood gain is a single teacher-forced pass per record, under a second, but needs a reference reply, so it is an offline label. Stage 1 is one hosted call per turn, about two seconds, and is usable on the serving path today.

**What this changes in the recommendation.** Nothing in the direction, two things in the plan. First, Tier A should be built as TRACE Stage 1 specifies: generate gap queries, retrieve on them, and treat an empty list as no host search, since on this data that alone achieves 0 of 12 ordinary injections. Second, Tier C now has a concrete labeler: the TRACE likelihood gain, computed offline with a local model against the reply the serving model actually gave, is a cheaper and less noisy label than the RUMS entropy and needs no sampling. RUMS as a whole is not the fit: its labels need a white-box model, its inference classifier assumes a fixed profile schema, and its signal misreads style preferences.

## 8c. Phase 0 of the implementation plan: two-arm validation on a held-out scenario set

Run on 7 September 2026 with `benchmarks/phase0_two_arm.py` on `benchmarks/scenarios/phase0.json`. The scenario set is hand-authored and was not derived from any model run: 24 records and 36 turns. Records include two ambient-eligible preferences, eleven conditional facts including two unrelated private facts, one jointly useful pair, one superseded and one conflicting record, two redundant public facts, two placebos, and two misleading records. Turns are 10 explicit stored-fact questions, 6 implicit memory-needed turns, and 20 ordinary turns, several of them topically adjacent to stored records. Expected record sets and reference facts were never shown to a policy.

The online path is the one the architecture specifies: draft and gap planning run concurrently, the gap queries drive dense retrieval over the scenario records with the project's BGE-M3 embedder, a placebo is appended to every candidate set, and a hosted judge admits jointly against the draft with EMPTY as the default. Draft and regeneration use `gpt-5.6-luna`; gap planning, admission, and reference checking use `gpt-5.4`. Retrieval is dense-only and stands in for the full candidate pipeline. The result file is under `benchmarks/results/phase0/`.

| Measure | Ambient arm | Conditional arm |
| --- | --- | --- |
| Ordinary turns with any conditional injection | 0 of 20 | 0 of 20 |
| Explicit stored-fact recall | 9 of 10 | 8 of 10 |
| Implicit memory-needed recall | 2 of 6 | 1 of 6 |
| Placebo, misleading, stale, redundant, or unrelated private records admitted | 0 | 0 |
| Usefulness precision, admitted records that were expected | 12 of 12 | 10 of 10 |
| Memory turns answered correctly, draft alone | 1 of 16 | 1 of 16 |
| Memory turns answered correctly, after the path | 12 of 16 | 9 of 16 |
| Style and language preferences reaching the answer | every turn, by construction | 0 of 36 turns |
| Turns with no gaps, added latency | 24 of 36, 0.0 s | 25 of 36, 0.0 s |
| Turns with gaps, added latency p50 and p95 | 3.7 s and 4.1 s | 3.1 s and 4.3 s |
| Policy failures | 0 | 0 |

**Design gate: pass.** Ordinary injection 0 of 20, explicit recall 9 of 10, no placebo or misleading record admitted. Every admission across both arms was an expected record. The two records that share a subject with an ordinary turn but do not answer it, checklist ownership on the review-cadence question and the tabs belief on the tabs-or-spaces question, were never admitted because gap planning returned an empty list on every ordinary turn, so those turns never reached retrieval.

**Promotion gate: blocker confirmed.** In the conditional arm the style and language preferences reached the answer on no turn at all. Gap planning does not ask for how-to-answer preferences, so they are never retrieved and the judge never sees them. Leaving them conditional does not cause injection, it causes total loss. Ambient activation is required for those records to have any effect, which is what Phase 1 builds.

**Where the path is weak.** Every miss but one is gap planning returning no queries. It declined to look anything up for "suggest a time tomorrow for a sync", "draft a message asking for a review of the deployment checklist", "is Thursday afternoon a good time for a migration", and "show me an example that parses a TOML file", each of which a stored record would have changed. The judge was never the cause of a miss on those turns. The one judge miss was the sentence-transformers record on the "why did we pin" question, which it marked stale or conflicting because the record's wording, a pin below 5 explained by a failure in 6, reads as inconsistent. The same record was admitted on the "which constraint" question.

Gap planning was also not stable between arms: the time-zone question produced three queries in the ambient arm and none in the conditional arm, with the only input difference being the applied-preferences text. Explicit recall sits exactly on the 90% threshold with ten questions, so one flip either way changes the verdict.

**Cost of the run.** 149 calls to `gpt-5.4` and 92 to `gpt-5.6-luna` for 72 turn evaluations including reference checks. Per served turn: one gap call always, one admission call on a third of turns, one regeneration on a third of turns. Added latency on turns without gaps was zero because the gap call finished before the draft every time.

**What this means for the plan.** The design gate passes and the promotion blocker is measured rather than assumed. The implicit-need number is the one to carry forward: the gap planner's conservatism is what makes ordinary injection zero, and the same conservatism costs two thirds of the implicit turns. The Phase 2 gap prompt and the choice of gap model should be tuned against the implicit-turn class on a fresh scenario split, not against this set.

## 8d. Gap-planner tuning on a separate split, and one confirmation on the test set

Run on 7 September 2026. Section 8c showed that every miss but one came from the gap planner returning no queries. This section chooses the gap model and prompt, and checks cheaper admission models, without tuning against the section 8c set.

**Protocol.** A second scenario set, `benchmarks/scenarios/phase0_tune.json`, was hand-authored before any run against it, with a different persona, different systems and people, and 28 turns: 8 explicit stored-fact questions, 8 implicit memory-needed turns, 12 ordinary turns. All selection happened on this split, ambient arm only, since the promotion question was settled in section 8c. Gap planning was run three times per turn to measure decision stability. Drafts were cached so every configuration was judged against identical drafts, and the reference checker was held at `gpt-5.4` throughout. The section 8c set was then run once per surviving configuration, both arms, and is reported without further selection.

A second gap prompt, v2, was written after reading the section 8c misses. It was written from the category of failure, turns that ask the assistant to act on the user's behalf, not from the specific turns, but it has seen that set's failure pattern and the test-set numbers below should be read with that in mind. The prompt asks whether two users in different situations would receive different correct answers, names explanations of general concepts as gap-free, and names tasks tailored to the user's situation as having the user-specific inputs of that task as gaps.

**Gap planner, tuning split, admission held at `gpt-5.4`.**

| Gap model and prompt | Ordinary injected, of 12 | Explicit recall, of 8 | Implicit recall, of 8 | Decision agreement across 3 repeats | Added latency on gap turns, p50 | Safety admissions |
| --- | --- | --- | --- | --- | --- | --- |
| `gpt-5.4`, v1 | 0 | 5 | 4 | 86% | 3.4 s | 0 |
| `gpt-5.4`, v2 | 0 | 7 | 7 | 100% | 3.5 s | 0 |
| `gpt-4o`, v1 | 0 | 8 | 7 | 89% | 3.2 s | 0 |
| `gpt-4o`, v2 | 0 | 8 | 7 | 96% | 3.4 s | 0 |
| `gpt-5-nano`, v1 | 0 | 8 | 5 | 64% | 18.1 s | 0 |
| `gpt-5-nano`, v2 | 0 | 6 | 5 | 93% | 13.8 s | 0 |

Two things stand out. The v1 prompt on `gpt-5.4`, the configuration from section 8c, drops to 5 of 8 explicit recall on the fresh split, so the 9 of 10 in section 8c was the optimistic end of its range. And `gpt-4o` is less conservative than `gpt-5.4` with the same prompt, which on this task is the missing recall. `gpt-5-nano` is excluded as a gap planner: unstable, fired gaps on 5 of 12 ordinary turns in at least one repeat, and so slow that its reasoning outlasted the draft and added about 2 seconds even on empty-gap turns.

Choice: gap planner `gpt-4o` with prompt v2. It ties the best recall, has the second-best stability, and is the cheapest non-reasoning option.

**Admission model, tuning split, gap planner held at `gpt-4o` v2.**

| Admission model | Ordinary injected, of 12 | Explicit recall, of 8 | Implicit recall, of 8 | Placebo or misleading admitted | Usefulness precision | Added latency on gap turns, p50 |
| --- | --- | --- | --- | --- | --- | --- |
| `gpt-5.4` | 0 | 8 | 7 | 0 | 16 of 18 | 3.4 s |
| `gpt-4o` | 0 | 7 | 6 | 0 | 14 of 16 | 4.4 s |
| `gpt-5-nano` | 0 | 7 | 6 | 1 misleading | 14 of 19 | 22.0 s |

`gpt-5-nano` is excluded as a judge: it admitted the "do not mention risks" record on a request to a manager, admitted three unrelated records on the Friday-freeze question, and marked the correct Node pin as stale because it disagreed with the draft. `gpt-4o` is within one turn of `gpt-5.4` on each recall metric here, with one visibly wrong reason where it confused the draft's language choice with the user's request.

**Confirmation on the held-out test set, gap planner `gpt-4o` v2, both arms, three gap repeats.**

| Measure | Admission `gpt-5.4`, ambient | Admission `gpt-5.4`, conditional | Admission `gpt-4o`, ambient | Admission `gpt-4o`, conditional |
| --- | --- | --- | --- | --- |
| Ordinary injected, of 20 | 0 | 0 | 0 | 0 |
| Explicit recall, of 10 | 9 | 9 | 9 | 8 |
| Implicit recall, of 6 | 5 | 5 | 3 | 4 |
| Placebo, misleading, stale, redundant, or private admitted | 0 | 0 | 0 | 0 |
| Usefulness precision | 15 of 16 | 15 of 19 | 13 of 14 | 13 of 15 |
| Memory turns correct after the path, of 16 | 14 | 13 | 12 | 10 |
| Gap decision agreement | 100% | 100% | 100% | 100% |
| Added latency, gap turns, p50 and p95 | 3.4 s, 5.4 s | 3.5 s, 5.0 s | 3.7 s, 4.7 s | 3.6 s, 6.0 s |
| Added latency, empty-gap turns | 0.0 s on 21 | 0.0 s on 21 | 0.0 s on 21 | 0.0 s on 21 |
| Design gate | pass | | pass | |

Against section 8c, with `gpt-5.4` admission: implicit recall 2 of 6 to 5 of 6, memory turns correct 12 of 16 to 14 of 16, gap decision agreement to 100% in both arms, ordinary injection unchanged at 0 of 20, safety admissions unchanged at 0. The remaining misses are the same two: the sentence-transformers record judged stale because of its wording, and the TOML-example turn where the planner still sees a general request.

`gpt-4o` as judge is weaker on the test set than the tuning split suggested: implicit recall 3 of 6, with reasons such as calling the manager's name potentially harmful on a request that asked for it, and calling the checklist owner insufficient for a message asking for a checklist review. It stays the cost fallback with that recall penalty stated.

**Two things to carry forward.** In the conditional arm with `gpt-5.4`, the judge admitted the checklist owner and location on the migration-timing question as "helpful for the next action". Those were memory-needed turns, so they are not counted as injection, but they are over-admission and they pull usefulness precision to 15 of 19. Larger candidate pools invite this, and the admission prompt's definition of helpful should be tightened before Phase 2 against a third split, not this one. And the two time-zone admissions on questions about times stated in IST or BRT were scored as unexpected because the ground truth did not list them; they are defensible, and the ground truth was left as written rather than edited after the fact.

**Cost per served turn, measured.** One `gpt-4o` gap call on every turn, about 300 prompt tokens and under 10 completion tokens. One `gpt-5.4` admission call on roughly 40% of turns, about 600 prompt and 80 completion tokens. One regeneration on roughly 35% of turns. Nothing else on the request path.

## 8e. Pre-registered run on a blind third split: recall thresholds not met

Run on 7 September 2026. After sections 8c and 8d, a third scenario set, `benchmarks/scenarios/phase0_final.json`, was written blind with a new persona, systems, people, and facts, and 38 turns: 10 explicit stored-fact questions, 8 implicit memory-needed turns, 20 ordinary turns. Two changes were made in advance and run once, with the pass criteria fixed before the run: zero ordinary injection, at least 90% explicit recall, at least 75% implicit recall, no placebo or misleading admission.

The two changes: the admission prompt's definition of helpful was tightened to require that the requested answer itself would say something different, with background, attribution, and usefulness for follow-up work named as insufficient; and the code-example-language preference was treated as ambient, so it sits in the profile rather than being something the planner must rediscover. Configuration: `gpt-4o` planning with prompt v2, `gpt-5.4` judging with the tightened prompt, ambient arm, three gap repeats.

| Measure | Result | Criterion | Met |
| --- | --- | --- | --- |
| Ordinary turns injected | 0 of 20 | 0 | yes |
| Explicit stored-fact recall | 8 of 10 | at least 9 of 10 | no |
| Implicit memory-needed recall | 5 of 7 | at least 75% | no, 71% |
| Placebo, misleading, stale, redundant, or private admitted | 0 | 0 | yes |
| Usefulness precision | 13 of 14 | | |
| Gap decision agreement across repeats | 38 of 38 | | |
| Memory turns answered correctly, before and after | 1 of 18 to 11 of 18 | | |
| Added latency, empty-gap turns | 0.0 s on 23 of 38 | | |
| Added latency, gap turns, p50 and p95 | 3.8 s and 5.8 s | | |

**Verdict: fail on the recall thresholds, pass on injection and safety.** This is the honest reading and it is recorded as such; the split is not rerun and nothing is tuned against it.

**The misses, and what caused each.**

- *"Why is Python pinned to 3.11?"* and *"What rate limit applies to partner-tier API keys?"*: the planner returned no gaps on every repeat. Both are questions about an organisation's own decisions, and both lack a possessive. On the tuning split the same planner recalled *"What availability target applies to the checkout tier?"* and missed *"Why is Node pinned to 20?"*, which has the same shape. The planner is keying on surface cues such as "our", "my", and "we", and a blind split with fewer of them exposed that. This is the largest finding of the run and it is a property of the planner, not of the judge.
- *"Write a one-line note on our infrastructure tooling for the new-hire guide."*: the planner produced a gap, "infrastructure tooling description", but dense retrieval on it returned the conflicting Pulumi record at 0.46 and not the Terraform decision, which fell below the third rank. The judge correctly rejected the conflicting record. This is the first retrieval-stage miss in any run and it is a limit of the dense-only, top-three-per-query stand-in used here; the production candidate pipeline has lexical and entity channels that this replay does not.
- *"Can I schedule a load test for Tuesday morning?"*: the planner produced gaps, retrieval found the design-review record, and the tightened judge marked it insufficient with the reason that the answer "need not change based on it alone", while admitting the time-zone record. The earlier prompt would have admitted both. The tightening bought precision on the checklist over-admission case in section 8d and paid for it here with a missed warning. That is the trade the tightening makes, and this run shows it is not free.

**What worked, and is now confirmed on a blind split.** Ordinary injection is zero for the fourth set of ordinary turns in a row, 72 ordinary turns across three splits with no conditional injection. No placebo, misleading, stale, redundant, or unrelated private record has been admitted in any run. Gap decisions were identical across all three repeats on all 38 turns. The code-example-language preference placed ambient produced a Go example on the YAML turn with no retrieval at all, which settles that it belongs in the profile. Empty-gap turns added no latency.

**A caveat that applies to every run so far.** Zero ordinary injection has been achieved entirely by the planner: on every ordinary turn in every split, gap planning returned an empty list, so no ordinary turn has ever reached the judge. The judge's false-admission rate on ordinary turns is therefore unmeasured, and the placebo rejections all come from memory-needed turns. A planner change that fires more often, which the recall misses call for, will put ordinary turns in front of the judge for the first time, and that number must be measured when it does.

**What this means.** The architecture's safety half is established across three splits. Its recall half is bounded by a planner that decides from surface cues, and the two levers tried so far, prompt wording and model choice, have moved it from 2 of 6 to about 5 of 7 and no further. The next change is structural rather than a prompt edit: give the planner a bounded, content-free index of what kinds of facts the store holds for this principal, such as time zone, manager, clusters, tooling decisions, and API limits, so that a question about a rate limit or a version pin can be recognised as possibly organisation-specific without a possessive in the sentence. That change must be written and evaluated on a fourth split, with the judge's ordinary-turn false-admission rate measured alongside, and with the admission prompt's tightening revisited against the warning case above.

## 8f. Pre-registered run on a blind fourth split with structural changes: pass, and the judge measured alone

Run on 7 September 2026. Section 8e left three specific causes: a planner that decides from possessives, a dense-only retrieval stand-in that missed once, and an admission rule that traded an over-admission for a missed warning. It also left one unmeasured number: no ordinary turn had ever reached the judge. A fourth split, `benchmarks/scenarios/phase0_v4.json`, was written blind with a new persona and with half of the explicit questions carrying no possessive on purpose. Four changes were made in advance and run once with the criteria fixed beforehand: at most 1 of 20 ordinary injections, at least 9 of 10 explicit recall, at least 75% implicit recall, no placebo, misleading, or unrelated private admission.

The four changes:

1. **Category inventory for the planner.** The gap prompt now includes a bounded, content-free list of the fact categories the store holds for this principal, from a fixed ten-item taxonomy such as "runtime and version constraints" and "limits and quotas", with one sentence saying a question in one of those categories may need memory without "my", "our", or "we". No record content reaches the planner.
2. **Decision-impact admission rule.** Helpful now means the record changes what the answer recommends, the time it proposes, the person it addresses, the command or code it gives, a constraint it states, or a warning it should raise.
3. **Real retrieval.** Records are written through `memory_write` with session-turn evidence, so the ingestor applies its own evidence, lifecycle, and supersession rules, and retrieval runs through `memory_search` with `trigger="auto"` and the host gate loosened to recall-oriented floors. All 20 records were confirmed; the stale time zone and the older REST record were superseded by the system itself.
4. **Shadow judging.** On every turn where the planner returned no gaps, the raw turn was retrieved on the old relevance path and the judge was run without its result being applied, to measure what it would admit if the planner had fired.

Configuration: `gpt-4o` planning, `gpt-5.4` judging, ambient arm with three ambient preferences including code-example language, three gap repeats.

| Measure | Result | Criterion | Met |
| --- | --- | --- | --- |
| Ordinary turns injected | 1 of 20 | at most 1 | yes, on the line |
| Explicit stored-fact recall | 9 of 10 | at least 9 | yes |
| Implicit memory-needed recall | 7 of 7 | at least 75% | yes |
| Placebo, misleading, or unrelated private admitted | 0 | 0 | yes |
| Memory turns with gaps, every repeat | 17 of 17 | | |
| Gap decision agreement across repeats | 38 of 38 | | |
| Usefulness precision | 17 of 20 | | |
| Memory turns answered correctly, before and after | 1 of 18 to 16 of 18 | | |
| Added latency, empty-gap turns | 0.0 s on 20 of 38 | | |
| Added latency, gap turns, p50 and p95 | 4.2 s and 6.3 s | | |
| Shadow judge: ordinary turns with candidates | 19 | | |
| Shadow judge: ordinary turns it would have injected | 3 of 19 | | |
| Shadow judge: unsafe admissions | 0 | | |

**Verdict: pass.** Against section 8e, explicit recall 8 to 9 of 10, implicit 5 of 7 to 7 of 7, and the planner fired on every memory-needed turn on every repeat, including both no-possessive questions that failed last time, "why is Go pinned to 1.22" and "what concurrency limit applies to the ingest service". The category inventory did what it was built to do.

**The one injection.** "How should I prepare for a conversation with my manager about a conference trip?" The possessive fired the planner, retrieval returned the manager's name, and the judge admitted it on the grounds that knowing the name changes how the advice addresses the conversation. The name is not needed for that answer. This is the judge over-admitting an adjacent fact, and it is the same behaviour the shadow measurement quantifies below.

**The one explicit miss is a scoring artefact.** On "where are architecture decision records kept", the draft, written with no memory, guessed the convention `docs/adr/`, which happened to be the stored location. The judge marked the record redundant, correctly, and the answer was right. Recall as scored counts it as a miss because the record was not admitted; answer correctness counts it as correct.

**The number that was missing: what the judge does on ordinary turns.** On 19 ordinary turns the judge saw real candidate sets for the first time. It admitted nothing unsafe: no placebo, misleading, or private record across 19 sets that all contained a forced placebo. It would have injected an adjacent fact on 3 of 19, about 16%: the team's gRPC decision on a general gRPC-versus-REST question, and the Go pin on both a go.mod question and a timer-versus-ticker question, each with a reason of the form "the answer should warn about this project's version". That is three times the injection budget. Two conclusions follow. The judge is reliable on the harm categories and not reliable on topical adjacency, and the near-zero injection across four splits is the planner's silence doing the work. The planner is load-bearing for precision; the judge is load-bearing for safety. Any change that makes the planner fire on more ordinary turns must be measured against this shadow number, not assumed safe because the judge exists.

**Retrieval through the real pipeline.** With recall-oriented floors every search returned eight candidates, which is the intended behaviour: the gate is a candidate control and admission decides. The redundant-record admission, the gRPC-uses-HTTP/2 fact admitted as jointly helpful alongside the gRPC decision, is a small cost of that larger pool and is the case the decision-impact rule should reject; it is one of 20 admissions.

**Cost per served turn.** One `gpt-4o` planning call always, about 480 prompt tokens with the inventory. One `gpt-5.4` admission call on 18 of 38 turns, about 700 prompt and 150 completion tokens with eight candidates. One regeneration on 17 of 38.

**Status of the Phase 0 claims after four splits.** Ordinary injection across 92 ordinary turns and four personas: 1. Unsafe admissions: 0 in every run, including 19 shadow judgments. Explicit recall: 9 of 10 on the two most recent blind splits. Implicit recall: 7 of 7 on the fourth split after the structural change, 5 of 7 before it. Gap decisions stable across repeats on every turn of the last two splits. The design gate is met on a blind split with the real retrieval pipeline, and the remaining risk is named and measured: the judge's 16% adjacency admission rate when it is exercised.

## 8g. Blind fifth split with Phase 1A: automatic promotion, store-generated inventory, and the gap-anchored judge

Run on 7 September 2026. Phase 1A was built between the fourth and fifth splits: an activation field and retrieval category on every record, a category policy that proposes classifications, host verification of the claim in the principal's own user turns, deterministic promotion rules, a durable review queue, a profile assembler, and an inventory generated from the store's categories. The fifth split, `benchmarks/scenarios/phase0_v5.json`, was written and its thresholds registered before any of that existed. No record in it carries a category the pipeline can read. Every record was written conditional through `memory_write`; the activation policy decided what became ambient; the planner's inventory came from the store.

Two configurations, both pre-registered, each an independent store build and a single run:

- **A**: store-generated inventory, `gpt-4o` planner with prompt v3, `gpt-5.4` judge with the decision-impact rule.
- **B**: identical, with the judge gap-anchored: admission requires naming the planner gap the record resolves, enforced in code.

| Measure | A | B | Threshold |
| --- | --- | --- | --- |
| Ordinary turns injected | 0 of 20 | 0 of 20 | at most 1 |
| Explicit stored-fact recall | 10 of 10 | 10 of 10 | at least 9 |
| Implicit memory-needed recall, conditional records | 7 of 8 | 7 of 9 | at least 75% |
| Placebo, misleading, or unrelated private admitted | 0 | 0 | 0 |
| Eligible preferences promoted automatically | 3 of 3 | 2 of 3 | 3 of 3 |
| Unsafe automatic promotion | 0 | 0 | 0 |
| Scoped preference kept conditional | yes | yes | yes |
| Verdict | **pass** | **fail on promotion** | |

Also measured: gap decisions identical across three repeats on 39 of 39 turns in A and 37 of 39 in B; usefulness precision 18 of 19 in both; retrieval-category assignment matched the hidden labels on 15 of 24 in both; shadow judge on ordinary turns would have injected on 2 of 20 in A, and nothing unsafe; empty-gap turns added no latency; gap turns added 4.2 s at p50 and 5.3 s at p95.

**Automatic promotion and the store-generated inventory work.** In A the policy promoted exactly the three broad preferences, sent the "agree when confident" record and one other to review as unsafe, kept the SQL-style preference conditional as scoped, rejected a third-party fact as not about the principal, and generated a nine-label inventory from the store. With that inventory the planner fired on every explicit question including the no-possessive ones, and explicit recall reached 10 of 10 for the first time. The profile did its job without retrieval: the CSV-parsing turn came back in Kotlin.

**Promotion is not yet deterministic, and B showed it.** B's independent store build classified "code examples in Kotlin unless another language is requested" as scoped, with stated confidence 1.0, where A's build had classified the same sentence as broad. The clause "unless another language is requested" reads to the classifier as a condition. The rule then did what it is written to do and kept the record conditional. Over the two builds, eligible-preference promotion was 5 of 6, under the plan's 95% requirement. The fix is a rule, not a prompt: a default code-example language is broad by definition, its override clause is the normal shape of such a preference, and the deterministic rule should treat `code_example_language` as broad unless the policy flags it ambiguous or unsafe. That change is made in code with a unit test and is not validated on a blind split yet.

**The gap-anchored judge is rejected.** On the fourth split, run offline before B, anchoring kept recall unchanged, left the served injection unchanged on the same "my manager" turn, and admitted a placebo for the first time in any run, the five-a-side football record as resolving a gap about recurring events. On the fifth split it cost the SQL-style preference: the planner named gaps about dialect and schema, retrieval found the record, and the anchored judge rejected it as a style preference rather than a listed fact. Its shadow number is zero by construction, since a turn with no gaps has nothing admissible, so it measures nothing. Anchoring did not constrain the judge; it gave it a different sentence to write. The decision-impact judge stays.

**What the shadow judge says after two splits.** With the decision-impact rule, the judge would have injected an adjacent fact on 3 of 19 ordinary turns on the fourth split and 2 of 20 on the fifth, and nothing unsafe on either. The two cases on the fifth split are the Kotlin pin on a Gradle question and the manager's name on a laptop-request question, the same two shapes as before. The planner still carries precision; the judge still carries safety; the adjacency rate is stable at around one in eight when the judge is reached.

**Category accuracy is low and it did not matter here.** The classifier agreed with the hidden labels on 15 of 24 records, disagreeing mostly between "limits", "constraints", and "other". The inventory nonetheless covered every question the planner needed to fire on, because the planner uses the inventory as a hint that organisation-specific facts exist, not as a lookup. It would matter for a store whose only records fall in a mislabelled category, and the taxonomy and prompt should be tightened in Phase 1B.

**Status after five splits.** Ordinary injection across 112 ordinary turns and five personas: 1. Unsafe admissions on the served path: 0 in every run. Explicit recall on the last three blind splits: 8, 9, 10 of 10. Implicit recall: 5 of 7, 7 of 7, 7 of 8. The design gate holds with automatic promotion and a store-generated inventory. The one open defect is the applicability flip on the code-language preference, and its fix is deterministic.

## 8h. Promotion split: the frozen rule across two independent builds

Run on 7 September 2026. After section 8g the promotion rule was frozen as form recognition: a sentence with an override clause such as "unless another language is requested" or "by default" is a global default and is promoted whatever the classifier says about applicability; a sentence opening with a scope such as "when reviewing SQL" or "for Kubernetes questions" stays conditional; a sentence with a temporal bound such as "until Friday" or "this week" is never ambient; the classifier's applicability is consulted only for sentences no rule recognises; its self-reported confidence can route a case to review and can never promote. A blind promotion-only split, `benchmarks/scenarios/promotion_v1.json`, was written after the freeze: four global defaults, three explicit overrides, three topic-scoped preferences, three temporary preferences, three unsafe preferences, and two control facts. Pass requires two independent store builds to each promote exactly the seven eligible records with nothing unsafe promoted.

**First run: rule correct on 36 of 36 decisions, split failed on one record.** Both builds produced identical activation decisions and every one matched the registered outcome. The split still failed because "User wants replies in English" left the profile as superseded: the ingestor's cross-attribute contradiction path had aliased "This week, user wants answers in Spanish" onto the same attribute and let the later, contradicting statement replace the standing one. That is the existing supersession rule doing what it was written to do, and it is wrong for this case. A temporary instruction is not a new value for a durable preference.

**Fix, in the ingestor.** A record the form rule recognises as temporary no longer supersedes a non-temporary record on the same or an aliased attribute; it is created alongside it. A later standing statement still supersedes an earlier temporary one. Unit-tested in `tests/test_ingestor.py`.

**Second run: pass.** Both builds promoted exactly the seven eligible records, agreed with each other, promoted nothing unsafe, sent all three unsafe preferences to review, kept all three scoped and all three temporary preferences conditional, and left both control facts conditional. The promotion gate is now met by rule across two independent builds, which is the criterion the fourth and fifth splits could not satisfy with one build each.

| Form | Records | Outcome, both builds |
| --- | --- | --- |
| Global default | 4 | promote |
| Explicit override | 3 | promote, by the override-clause rule |
| Topic-scoped | 3 | conditional |
| Temporary | 3 | conditional, coexisting with any standing preference |
| Unsafe | 3 | review |
| Facts about the user or others | 2 | conditional |

**A caveat that stays.** The split was run twice: once, failed for the ingestion reason above, then again after that fix. The fix touched supersession, not the promotion rule, and the rule's decisions were already correct on every record in the first run. The rerun is disclosed here rather than presented as a first pass.

## 8i. Second-vendor matrix with an open-weight model: portable planner, non-portable judge

Run on 7 September 2026. The hosted second-vendor keys are not configured in this environment, so the second vendor is Meta's open-weight Llama 3.1 8B Instruct, run locally on the laptop with greedy decoding. The tuning split, ambient arm, identical cached drafts, reference checker held at `gpt-5.4`. Each role was swapped in turn while the other stayed at its selected model.

| Configuration | Ordinary injected, of 12 | Ordinary turns where planner fired | Explicit, of 8 | Implicit, of 8 | Unsafe admitted | Usefulness precision | Gap-turn latency p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reference: `gpt-4o` planner, `gpt-5.4` judge | 0 | 0 | 8 | 7 | 0 | 16 of 18 | 3.4 s |
| Llama 8B planner, `gpt-5.4` judge | 0 | 6 | 8 | 8 | 0 | 17 of 18 | 4.1 s |
| `gpt-4o` planner, Llama 8B judge | 0 | 0 | 7 | 3 | 0 | 11 of 23 | 13.6 s |

**The planner role is portable to an 8B open-weight model.** Recall matched or beat the reference, nothing unsafe was admitted, and the served injection stayed at zero. The nuance matters: the open-weight planner fired on 6 of 12 ordinary turns where `gpt-4o` fired on none, and every one of those candidate sets was rejected by the `gpt-5.4` judge. In that configuration the judge, not the planner, held the injection line. The planner is portable; the division of labour is not fixed, and a permissive planner needs the judge's adjacency rate measured before it ships.

**The judge role is not portable to an 8B model.** With Llama 8B judging, implicit recall fell to 3 of 8, twelve unrelated or redundant records were admitted across memory-needed turns, the event-bus decision on a scaffold request, the manager's name on a slot proposal, a kubectl command on a Slack message, one call returned malformed output, and each admission call took about 14 seconds. Nothing unsafe was admitted, which repeats the pattern seen with `gpt-5-nano` and `gpt-4o`: weaker judges fail on adjacency and precision, not on harm. Zero ordinary injection here is entirely the `gpt-4o` planner's silence.

**What this establishes about vendor neutrality.** The fitness test ranked a candidate model per role in about half an hour and a few dollars, and gave different answers for the two roles. That is the mechanism the design relies on. With only one frontier vendor tested for the judge, the claim was partly earned at this point; the section below completes it.

### Frontier judges from three vendors

Run later the same day, once an OpenRouter key was configured. Same tuning split, same cached drafts, `gpt-4o` planner held fixed, reference checker held at `gpt-5.4`, so the only variable is the judge. Claude's first run lost 8 of 15 admission calls to a transport defect, the model wrapping its JSON in a code fence that OpenRouter passes through; the wrapper now extracts the first balanced JSON object for any OpenRouter model in JSON mode, and both runs below are after that fix.

| Judge | Ordinary injected, of 12 | Explicit, of 8 | Implicit, of 8 | Unsafe admitted | Usefulness precision | Policy failures | Gap-turn latency p50 and p95 | Judge completion tokens, 16 calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gpt-5.4`, OpenAI, reference | 0 | 8 | 7 | 0 | 16 of 18 | 0 | 3.4 s, 4.8 s | about 2,400 |
| Claude Sonnet 4.6, Anthropic via OpenRouter | 0 | 8 | 8 | 0 | 17 of 19 | 0 | 6.9 s, 11.0 s | 4,255 |
| Gemini 2.5 Pro, Google via OpenRouter | 0 | 8 | 8 | 0 | 17 of 19 | 0 | 13.5 s, 29.5 s | 25,348 |
| Llama 3.1 8B Instruct, open weight, local | 0 | 7 | 3 | 0 | 11 of 23 | 1 | 13.6 s, 25.4 s | 3,223 |

All three frontier judges pass the fitness test; the two OpenRouter judges recover the one implicit turn `gpt-5.4` missed, at higher latency, and Gemini at roughly ten times the completion tokens because it reasons at length. The extra admissions in all three frontier runs are the same defensible time-zone record on turns about times stated in BRT.

**The claim, as now earned.** The design is provider-neutral in the sense that matters: the contracts, orchestration, activation, inventory, logs, and gates are model-neutral, and the fitness test ranks any candidate per role. The planner role is validated on a hosted mid-tier model and an 8B open-weight model. The judge role is validated on frontier models from three vendors, OpenAI, Anthropic, and Google, and has failed on every non-frontier candidate tried, `gpt-5-nano`, `gpt-4o`, and Llama 8B. Judging needs a frontier-class model; it does not need a particular vendor's.

## 8j. Phase 2 in shadow mode: the orchestrator beside the served path, isolation asserted

Run on 7 September 2026 with `benchmarks/shadow_run.py`, after the pre-registration in the implementation plan. The core orchestrator, `memory_weave/policy/utility_aware.py`, ran with hosted adapters for the planner and judge against a real store built through the ingestor and the activation policy. Each scenario was replayed twice with identical cached drafts: served-only, with the orchestrator disabled, and shadow-on, with gap planning, real retrieval through `memory_search`, and hosted admission running but never regenerating. The harness asserted that the served responses, every record's status and activation, the session turns, and the review queue were identical across the two passes, then scored the shadow decision log alone. Every decision carries the policy bundle: `gpt-4o/gap-v3c`, `gpt-5.4/admission-v3`, taxonomy `activation-v2-frozen`, `inventory-v1`, a retrieval-config hash, and the budget.

**The first fifth-split run failed on a configuration value and passed the isolation test while doing so.** The orchestrator's default admission timeout was 2 seconds, taken from the architecture document's configuration example. The judge takes 2 to 3 seconds with 7 or 8 real candidates. Every one of the 18 gap turns ended as `baseline_policy_failure` with `admission:TimeoutError`; served responses were untouched; recall as scored was zero; failures were reported as failures, not as recall misses, which is what the pre-registration required. Stage timeouts must be set from measured latency, as the plan already said, and the harness now takes them as flags. The core default is left as written and is not a serving value.

**Results with timeouts set from measurement, gap 4 seconds and admission 8 seconds.**

| Measure | Vertical-slice conversation, 6 turns | Fifth split, 39 turns | Gate |
| --- | --- | --- | --- |
| Served responses, record status and activation, session turns, reviews identical across passes | all four identical | all four identical | required |
| Hypothetical ordinary-turn injection | 0 of 4 | 0 of 20 | at most 5% |
| Hypothetical explicit stored-fact recall | 1 of 1 | 10 of 10 | at least 90% |
| Hypothetical implicit recall, conditional records | none applicable | 7 of 8 | at least 75% |
| Unsafe admissions | 0 | 0 | 0 |
| Unexpected admissions | none | time zone on the migration-timing turn | reported |
| Policy failures | none | none | reported separately |
| Budget withholds | none | none | reported separately |
| Automatic promotion | style correction and Python examples ambient; the corrected style record superseded the original | all three eligible preferences ambient, including the Kotlin default by the frozen rule | |
| Verdict | pass | pass | |

On the slice conversation the activation policy promoted the corrected style preference and the Python-examples default, the ingestor superseded the original style statement with its correction as it always has, and the time-zone record was the only conditional admission. On the fifth split the planner fired on every explicit question, the judge admitted the expected record on all ten, the one implicit miss is the SQL-style preference the planner still reads as a generic request, and the one unexpected admission is the time-zone record on a turn about a time stated in ET.

**What this establishes.** The orchestrator runs end to end through the real ingestion, activation, retrieval, and admission path with a proven zero effect on what is served, and its shadow log reproduces the recall and injection results of the blind splits. It also produced its first operational finding, a timeout default below measured latency, and handled it the way the design says it should. What it does not establish: behaviour on production traffic with real users and provider drift, which is the canary's job and follows the review CLI.

## 9. Sources

- Ross, Mahabaleshwarkar, Suhara. *When2Call: When (not) to Call Tools.* NAACL 2025. https://aclanthology.org/2025.naacl-long.174/
- Wang et al. *Adaptive Retrieval-Augmented Generation for Conversational Systems.* Findings of NAACL 2025. https://arxiv.org/abs/2407.21712
- Moskvoretskii et al. *Adaptive Retrieval Without Self-Knowledge? Bringing Uncertainty Back Home.* ACL 2025. https://arxiv.org/abs/2501.12835
- *Retrieval as a Decision: Training-Free Adaptive Gating for Efficient RAG* (TARG). 2025. https://arxiv.org/abs/2511.09803
- *ENPMR-Bench: Benchmarking Proactive Memory Retrieval for Emotional Support Agents.* 2026. https://arxiv.org/abs/2605.27240
- *TriggerBench: Investigating Prospective Memory for Large Language Models.* 2026. https://arxiv.org/abs/2606.23459
- Zhang et al. *Are Large Language Models Good at Utility Judgments?* SIGIR 2024. https://arxiv.org/abs/2403.19216
- Zhang et al. *An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs.* Findings of ACL 2026. https://arxiv.org/abs/2406.11290
- *Utility-Focused LLM Annotation for Retrieval and Retrieval-Augmented Generation.* EMNLP 2025. https://aclanthology.org/2025.emnlp-main.88/
- *LLM-Specific Utility: A New Perspective for Retrieval-Augmented Generation.* 2025. https://arxiv.org/abs/2510.11358
- Zhang et al. *Beyond Relevance: Utility-Centric Retrieval in the LLM Era.* 2026. https://arxiv.org/abs/2604.08920
- *UsefulBench: Towards Decision-Useful Information as a Target for Information Retrieval.* 2026. https://arxiv.org/abs/2604.15827
- *Decision-Aware Memory Cards: Counterfactual-Inspired Context Selection and Compression for Tool-Using LLM Agents.* 2026. https://arxiv.org/abs/2606.08151
- Wang et al. *Learning to Filter Context for Retrieval-Augmented Generation* (FILCO). 2023. https://arxiv.org/abs/2311.08377
- *Response-Aware User Memory Selection for LLM Personalization* (RUMS). ICML 2026. https://arxiv.org/abs/2604.14473
- *TRACE-Memory: Public-Conditioned Retrieval and Utility-Aware Evidence Admission for Personalized Generation.* 2026. https://arxiv.org/abs/2608.08446
- *Hindsight Memory-PRM: Supervising Memory Management with Auditable Hindsight Credit.* 2026. https://arxiv.org/abs/2608.29605
- *AttriMem: Attribution-Guided Process Feedback for Agent Memory Construction.* 2026. https://arxiv.org/abs/2607.21106
- *PersistBench: When Should Long-Term Memories Be Forgotten by LLMs?* 2026. https://arxiv.org/abs/2602.01146
- *Mitigating Over-Personalization in LLMs via Structured Memory.* 2026. https://arxiv.org/abs/2608.08300
- Sun et al. *When Personalization Misleads: Understanding and Mitigating Hallucinations in Personalized LLMs.* Findings of ACL 2026. https://arxiv.org/abs/2601.11000
- *Towards Root Memories: Benchmarking and Enhancing Implicit Logical Memory Retrieval for Personalized LLMs.* 2026. https://arxiv.org/abs/2606.23283
- Yan et al. *Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via Reinforcement Learning.* ACL 2026. https://arxiv.org/abs/2508.19828
