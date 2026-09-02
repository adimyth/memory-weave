---
title: "Agent Memory: Research Notes"
description: "Raw material for an essay and future research on how agents store, retrieve, and use durable memory."
date: "2026-09-03"
readTime: "Research notes"
draft: true
---

> These are structured research notes, not a finished essay or an implementation plan. They capture the current aim, the directions we prefer, the systems worth studying, and the questions that must be answered before any prototype is designed.

## 1. Aim

The essay is about agent memory: how a stateless language model is made to act as though it has a past, what kinds of information should survive a turn or session, and why retrieval policy matters more than simply storing more information.

The eventual goal is to reason our way toward a separate, public experimental memory-system repository. This portfolio repository remains essay-only. The implementation, benchmarks, experiments, and reproducible results will live elsewhere, as they do for the [LLM inference experiments](https://github.com/adimyth/llm-inference-experiments). The essay should link directly to that future repository rather than contain the experimental code itself.

We are not ready to design or build that system. The immediate output is a well-researched design brief: clear enough to discuss with another model and the team, identify the experiments worth running, and later turn into an implementation plan.

## 2. The starting point: an LLM has no yesterday

An individual LLM invocation does not inherently retain a prior interaction. The surrounding application creates continuity by providing representations of earlier state: messages, summaries, files, tool state, project rules, retrieved records, and other context.

Appending earlier user and assistant messages is conversational memory. It is useful because the model needs the recent dialogue to resolve references, preserve turn-taking, and continue a task coherently. It is still not sufficient as a broader memory system. A raw transcript grows without deciding what is durable, current, authoritative, private, or relevant.

The central claim to test in the essay is: **chat history is raw material for memory, not memory itself.** A memory system has to decide what is worth keeping, how it is represented, who can use it, how long it survives, and when it should influence a later task.

“LLMs are stateless” needs careful wording. Providers and applications can retain conversation objects, caches, and logs. The practical point is that an earlier interaction affects the next response only if some representation of it is made available to the model or to the agent through a tool.

## 3. A working vocabulary: four types of memory

The useful starting taxonomy is [Cognitive Architectures for Language Agents (CoALA)](https://arxiv.org/abs/2309.02427). CoALA distinguishes working memory from optional long-term episodic, semantic, and procedural memories. We should use these as engineering categories, not claim that an agent has human cognition.

| Memory type | What it means for an agent | Example | Main design question |
| --- | --- | --- | --- |
| Working memory | What is active in the current task: recent messages, tool results, scratchpad, opened files, and a session handoff. | “We are editing the agent-memory essay, and the latest decision is to keep durable memory behind tools.” | What must be visible right now for this task? |
| Semantic memory | Durable facts, preferences, policies, conventions, and generalized knowledge. | “This user prefers concise technical explanations.” “Markdown prose should not be hard-wrapped.” | What tends to remain true across many tasks? |
| Procedural memory | Reusable ways of doing things: skills, workflows, playbooks, and operational patterns. | “Before publishing, compile the MDX and run the relevant checks.” | How should this class of task be performed? |
| Episodic memory | Dated experiences, outcomes, decisions, and their rationale. | “We tried full transcript injection, found it caused context pressure, and decided to study tool-based retrieval.” | What happened before, and why did we choose this path? |

These are not four labels for one undifferentiated database. They need different write rules, retrieval rules, expiry policies, provenance, and safety controls. A stale preference is annoying; a stale deployment procedure can be dangerous; a fabricated episode can make an agent repeat a bad strategy.

`AGENTS.md` is a practical example of externalized semantic memory. It records project constraints that should influence many future tasks. A skill or playbook is procedural memory because it tells an agent how to perform a task. A dated decision record is episodic memory. The current task state is working memory.

## 4. Long-term memory: what it must be

Long-term memory must be external to the conversation so it survives a session reset. It must persist across sessions. It must be inspectable through explicit ingestion and retrieval capabilities so the agent can see the record, source, scope, and status behind anything it claims to remember.

External memory can also be transferable. If it is centred on the user rather than on one specific agent, a support agent, research agent, and calendar agent can act consistently on the same confirmed preference or profile fact. Transferability must not mean universal access. Every retrieval has to know whose memory is being searched and which agent is requesting it.

Every durable record should eventually answer at least these questions:

- What does this record say?
- What memory type is it?
- When was it created, and when did the underlying event occur?
- When does it expire, if at all?
- Who created it, and from what source or evidence?
- Who owns it, and which agents may retrieve or modify it?
- Is it confirmed, provisional, superseded, deleted, or subject to review?

Creation time and event time are different. An agent can create a record today about a failure that occurred last week. Both dates can matter when deciding whether the record is relevant.

The memory lifecycle is observe, decide whether it matters, create a candidate record, validate and write it, retrieve it when relevant, use it, revise or supersede it, and eventually expire or delete it. The difficult design work sits in the write and retrieval policies, not in the existence of a database table.

## 5. The shift toward building our own memory system

The original essay idea has expanded from a taxonomy into a research programme. We want to understand enough to eventually build our own memory layer and test it in a separate public repository. The aim is not to reproduce a commercial memory product or assume that a vector database is the answer.

The memory layer should be model-, framework-, and provider-neutral. Its durable contract should own records, provenance, scope, lifecycle, ingestion decisions, retrieval requests, and returned memory results. Adapters should translate a chosen agent framework’s message and tool representation into that contract, call a chosen extraction, embedding, or reranking model, and format tool results for the serving model.

That separation should allow a serving model from provider A, an extraction model from provider B, open-source embeddings, and a different retrieval backend without changing what a memory record means. It does not make switching free. Changing an embedding model may require re-embedding an index, changing an extractor may alter what gets written, and changing a reranker may alter score distributions. Those should be explicit migrations and evaluations, not hidden dependencies.

## 6. Four contrasting implementations: a structural reference

Manthan Gupta’s [“Reverse Engineering ChatGPT, Claude, OpenClaw, and Hermes Convinced Me Most AI Products Shouldn't Ship Memory”](https://manthanguptaa.in/posts/memory_is_a_mistake/) is a useful reference for the essay’s structure. Its value is not just that it describes four systems. It compares them through the same lens: what becomes prompt context, what stays outside it, and how retrieval is decided.

The article’s synthesis, which we must independently verify from primary sources before making technical claims, is useful as a source map.

| System | High-level approach described in the reference | Retrieval-policy contrast |
| --- | --- | --- |
| ChatGPT | A long-term profile plus summaries and metadata are injected into prompts. | Stored facts are ambient context. |
| Claude | A smaller active memory block is present, while past-conversation context is accessed through tools. | The model decides when to search. |
| OpenClaw | Durable notes and daily logs live in a Markdown workspace with search. | The agent searches notes when needed. |
| Hermes | A bounded hot prompt memory is separated from cold episodic recall and procedural skills. | Memory is tiered, with explicit rules for what belongs in each tier. |

For our eventual essay, any comparison should make every system answer the same questions.

| Comparison field | Question |
| --- | --- |
| Persistent representation | Is memory a profile, summary, Markdown workspace, structured record store, graph, or something else? |
| Hot context | What, if anything, is always injected, and how is it bounded? |
| Cold recall | What is kept outside the prompt and brought in only when needed? |
| Retrieval policy | Is memory always injected, automatically retrieved, selected by the model through a tool, or handled through a tiered combination? |
| Write policy | Which user, agent, tool, or outcome events create or revise memory? |
| Memory types | Are facts, episodes, procedures, and user modelling kept separate? |
| Scope and control | Who owns the record, who can access it, and can a user inspect, correct, or delete it? |
| Failure mode | What happens when the system retrieves too much, too little, stale information, or information about the wrong entity? |

This comparison may become a stronger essay structure than starting with a taxonomy alone. It lets us establish that systems remember differently, then earn the broader claim that storage format is secondary to the policy that decides what reaches the model, when, and under whose authority.

## 7. Our preferred direction: the Hermes-shaped path

The direction we currently align with is: **keep the prompt prefix stable for caching, and push durable memory to tools.** Memory should be something the agent actively reaches for, rather than a personality or profile that follows it into every prompt.

The stable, cacheable prompt prefix should contain the system instructions, tool-use policies, and memory-tool definitions. The live interaction still changes as the user and agent exchange messages. Durable memory should remain outside the prompt by default. When the agent decides it needs prior context, it should explicitly search, inspect, write, correct, expire, or supersede memory through a bounded tool interface.

This does not mean the full prompt is frozen. It means mutable long-term memory should not constantly alter the reusable prefix and fragment prompt caching. A memory result enters the working context only after a visible tool call. That call can be traced: which query the agent issued, which scope it searched, what results came back, and whether the agent used them.

The appeal of this direction is that it avoids ambient-memory injection. It keeps unrelated or stale records out of context, makes the retrieval decision observable, and treats memory as a deliberate capability rather than an invisible prompt mutation.

The cost is equally important. The agent must recognise when memory is relevant, decide to search, construct a useful query, and correctly use or reject the returned records. The evaluation set therefore needs search cases, no-search cases, and cases where a search result should be rejected as stale, unauthorized, contradictory, or irrelevant.

We have not yet decided whether there should be any always-present user or project state beyond stable instructions. The default preference is to minimize mutable ambient state and make memory access explicit. Any exception must be bounded and justified by evaluation.

## 8. The two core jobs: storage and retrieval

When we build our own system, the two most important pieces are storage and retrieval. Storage is the integrity layer. Retrieval is the behaviour layer. Retrieval matters even more because it determines whether stored information changes an agent’s action at the right moment.

### Storage correctness

Storage correctness means that a record is faithful, attributable, governable, and correctable. The system must preserve the record’s type, owner, source, creator, dates, confidence, scope, lifecycle, and relationship to records it supersedes or contradicts. It must support correction and deletion without making historical reasoning impossible.

If storage is weak, an inference can become a durable lie, records can leak between users or agents, and there may be no way to explain why an agent believes something. A good retrieval system cannot fully compensate for bad provenance or incorrect scope.

### Retrieval correctness

Retrieval correctness asks a more immediate question: should this agent see this memory for this task now? A perfectly stored record that is retrieved in an irrelevant task is still a product failure. The agent experiences the retrieval policy, not the database schema.

Retrieval must decide whether to search at all, construct an appropriate query, enforce identity and permissions, generate candidates, rank or rerank them, apply thresholds, fit results into a context budget, and allow the outcome that no memory should be returned. It also needs to make the final results inspectable to the agent and, eventually, to the user.

The design priority is therefore to make storage trustworthy enough to serve as evidence, then spend most of the design and evaluation effort on retrieval. Storage gives memory a past. Retrieval decides whether that past gets to influence the present.

## 9. Storage design questions

The following questions are not settled.

### Source of truth

Should the source of truth be a relational record store with embeddings as a secondary index, an event log that produces materialized memory views, a graph of entities and relations, or a combination? What must be editable, superseded, deleted, or retained for audit? The answer determines whether memory is primarily a set of records, a derived view over events, or both.

### Scope, identity, and access

Memory needs more than a single user identifier. Candidate scopes include agent-private memory, user memory shared across authorized agents, project memory, team memory, and organization memory. A retrieval must state both the owner scope and the requesting agent, because transferability without an access policy becomes leakage.

### Time and lifecycle

Which records expire automatically? Should provisional inferences expire unless confirmed? How should a newer user statement supersede an older fact? Do we delete records, tombstone them, retain them as history but exclude them from retrieval, or preserve temporal facts as separate records? These choices determine whether the system can reason about change without accumulating unmanageable clutter.

### Entity memory

Entity memory is likely a modelling layer rather than a fifth kind of memory. Semantic facts and episodes are about people, companies, projects, repositories, tickets, products, and conversations. Stable entities and relationships can help an agent distinguish a user preference from a project decision or an event involving a deployment.

We have not chosen an entity-resolution strategy. Is a name, email address, CRM contact, and account identifier one entity? Who establishes the link? Can an agent create an entity or only attach a record to an entity resolved by an authoritative external system? Entity mistakes can lead to the most serious failures: a memory leak or a false merge between two people.

### Where vector databases fit

A vector database is not synonymous with memory. It is usually a candidate-retrieval index over durable records. The durable record, not its embedding, must carry the scope, provenance, dates, confidence, and lifecycle. A system that only embeds conversation chunks has useful semantic retrieval over history, but has not yet solved authority, transfer, expiry, or correction.

## 10. Ingestion and extraction

Ingestion is the write path from an interaction, tool result, document, or outcome into a candidate memory record. It needs to be a deliberate system capability, not an invisible job that writes every model output into storage.

We have not decided when extraction should run. Candidate triggers include every message, session boundaries, tool results, explicit user corrections, completed tasks, or agent-initiated ingestion. Each choice changes latency, cost, recall, and the chance that a weak inference becomes durable state.

We have not selected an extraction model. The possibilities include the serving model, a separate structured-output model, a rules-first path, or a model with a user or human confirmation step. The important question is not whether a model can emit a record-shaped response. It is what evidence must exist before a candidate is allowed to outlive the turn.

Ingestion must distinguish explicit user statements, trusted-system data, tool results, agent claims, and model inferences. An agent should not be able to turn an unverified guess into a confirmed fact. Records need confidence, source, creator, and a path for correction or expiration.

## 11. Retrieval design questions

Retrieval is not one generic search call. It is a contract with several independent decisions.

| Retrieval decision | Open question |
| --- | --- |
| Search decision | Does the agent search at all, or is a search triggered automatically in limited cases? |
| Query | Is the input the latest user message, a task/session summary, a rewritten standalone query, extracted entities, multiple queries, or a combination? |
| Owner and requester | Whose memory is searched, and which agent is requesting it? |
| Candidate count | How many candidates does each first-stage retriever return before fusion or reranking? |
| Final count | How many records can enter working context after ranking, deduplication, and budget checks? |
| Threshold | When should the system return no memory, and how are scores calibrated when models or ranking methods change? |
| Freshness | How should recency, event time, expiry, and supersession affect ranking? |
| Context budget | How much retrieved information can fit without becoming a distraction? |
| Explainability | Can the agent inspect the record, source, and reason it was retrieved? |

### Query construction and rewriting

We have not decided whether query rewriting is necessary. A raw conversational follow-up such as “what about the second one?” may not contain useful retrieval terms. A task summary, standalone rewritten query, entity-aware query, or multi-query approach may improve recall. Each option adds another model call or heuristic and another way to distort user intent.

Different memory types may need different query strategies. Semantic memory may benefit from a task-focused query. Entity memory may be better served by explicit entity resolution. Episodic memory may need time and event constraints. Procedural memory may be selected from task classification rather than semantic search. Query rewriting should be evaluated as a hypothesis, not adopted as an architectural default.

### First-stage retrieval

We have not selected an embedding model or committed to dense retrieval as the only candidate path. Open-source candidates include `bge-m3` and `nomic-embed-text-v1.5`. Closed-source embedding providers are also candidates. `GLM 5.3 Flash` needs role validation before it belongs in an embedding comparison: it may be useful for extraction or query rewriting, but an embedding model must be assessed as a stable vector-representation model.

The model comparison should use our own memory-retrieval test set rather than a generic embedding leaderboard. Relevant criteria include retrieval quality on our query types, multilingual and domain behaviour, vector size, latency, cost, hosting and privacy constraints, licensing, model stability, and reproducibility.

Approximate nearest-neighbour search is another open choice. At small scale, exact search is an important baseline. At larger scale, ANN indexes trade some recall for speed. We first need corpus size, update rate, latency target, and acceptable recall loss. Index configuration must be evaluated separately from the embedding model so a weak index is not misdiagnosed as weak embeddings.

### Keyword, entity, and hybrid retrieval

Keyword retrieval needs a real design rather than a placeholder reference to BM25. We need to determine whether lexical scoring is plain BM25, BM25 over normalized text, another lexical method, or a hybrid over several fields. Lemmatization may help in some languages and query forms, but it is not automatically correct. Exact proper nouns, identifiers, filenames, product names, and user-created vocabulary may benefit more from exact matching. This requires language-specific evaluation.

Entity matching may be a hard filter, a lexical boost, a separate candidate path, or an input to a learned reranker. It should not be assumed that automatic entity extraction is always safe. The system needs to know when an entity match is authoritative and when it is merely a weak signal.

### Reranking

Reranking adds another candidate count, model, cost, and latency profile. We need to decide whether a cross-encoder, an LLM-based reranker, or no reranker is appropriate for the first experiment. The important measure is not whether offline rankings look nicer, but whether reranking improves the records that enter context and the task outcome that follows.

The likely pipeline is some combination of dense, lexical, and entity candidates; hard filtering by scope and lifecycle; score fusion; optional reranking; deduplication; and a final context budget. None of those stages should be assumed until we have representative evaluation cases.

## 12. How well should the system perform?

Performance is not only latency. The system needs storage, retrieval, and task-level evaluation.

### Storage evaluation

- Does the system create a record when it should?
- Does it avoid creating a record when it should not?
- Are source, owner, scope, dates, and confidence correct?
- Are corrections, supersessions, expiry, and deletion handled correctly?
- Can an agent and user inspect why a record exists?

### Retrieval evaluation

- Does the agent search when memory is needed?
- Does it avoid searching when memory is irrelevant or harmful?
- Does the query retrieve the right candidate records?
- Does filtering prevent cross-user, cross-project, expired, or unauthorized records from appearing?
- Does ranking select the correct records over stale or merely similar ones?
- Does the system correctly return no memory when no record should influence the task?
- Does the retrieved context improve the downstream task rather than distract the model or cause over-personalization?

### Operational evaluation

- What are ingestion latency and cost?
- What are retrieval, reranking, and end-to-end tool-call latency?
- What is the impact on prompt caching and token usage?
- How does quality change as the memory store grows?
- How do model, framework, embedding, index, and reranker swaps affect reproducibility and thresholds?

The test set must contain positive and negative cases. It needs desired writes, prohibited writes, desired retrievals, prohibited retrievals, stale records, conflicting records, entity ambiguity, sensitive-scope boundaries, search/no-search decisions, and downstream task behaviour. A memory system is easy to demo if it remembers something. It is hard to trust until it reliably knows what not to remember and what not to retrieve.

## 13. Related systems and research leads

### Mem0

[Mem0](https://arxiv.org/abs/2504.19413) is worth studying as a concrete memory system focused on extracting and retrieving useful long-term information. Its current documentation describes a single-pass, add-only extraction approach, hybrid retrieval, entity linking, and optional reranking. Its platform exposes separate controls for `top_k`, thresholding, and reranking, and its documentation warns that score thresholds need retuning as the retrieval algorithm changes. The [current memory-algorithm documentation](https://docs.mem0.ai/migration/platform-v2-to-v3) and [graph-memory documentation](https://docs.mem0.ai/platform/features/graph-memory) should be read alongside the paper.

Mem0 is a case study, not a conclusion. Questions to test include whether add-only extraction preserves useful temporal history or creates too much retrieval burden, how it treats agent-generated facts, whether hybrid scoring improves our representative queries, when automatic entity linking helps or harms, and whether reranking justifies its latency.

### Hermes

Hermes is especially relevant because the reference article presents its hot/cold split and separation of facts, episodic recall, and skills as a deliberate alternative to ambient memory. We need to identify and read the exact Hermes primary sources and version before treating any implementation detail as factual. The questions are the same as for our own design: what survives a session, what is externalized, what remains stable in the prompt, how the agent accesses cold memory, how skills are selected, and which choices are model- or framework-specific.

### Skills, self-improvement, and GEPA

Memory, skills, and optimization are related but not identical.

| Layer | What it changes | Core question |
| --- | --- | --- |
| Memory | Durable facts, experiences, decisions, and task state. | What did the system observe or learn? |
| Skill | A reusable procedure, workflow, or playbook. | How should the agent perform this class of task? |
| GEPA-style optimization | A textual artifact selected by evaluation, such as a prompt, query-rewriting instruction, retrieval policy, or skill. | Which version of this artifact performs better on the metric? |

[GEPA](https://arxiv.org/abs/2507.19457) is a reflective evolutionary optimizer, not a memory store. It runs candidates, reads execution traces, proposes changes, evaluates them, and maintains a set of strong alternatives. Its relevance is as a possible way to improve the policies around memory after we have a fixed contract and a representative evaluation set.

Self-improving skills are the closest bridge to memory. A successful or failed episode can provide evidence for a candidate procedure. After repeated evidence and evaluation, that procedure can be promoted into a versioned skill, which is procedural memory. [MemSkill](https://arxiv.org/abs/2602.02474) is a research lead because it explicitly explores evolving skills for extracting, consolidating, and pruning memories.

GEPA should be a later experiment, not a dependency of the first memory layer. It could eventually test a narrow hypothesis such as whether a query-rewriting instruction improves retrieval or whether a policy improves reranking. It should not mutate user-scoped facts or rewrite memory records simply because it produced a plausible reflection. Any policy change needs offline evaluation, versioning, and rollback.

### Voice Intelligence Platform

There may be a useful connection to the Voice Intelligence Platform. Context builders assembled task-specific context, while bring-your-own tools controlled the external systems an agent could interact with. Memory retrieval could be another context-builder input, and ingestion/retrieval could be tools alongside the external-system tools.

This needs a factual recovery pass before it appears in the essay. We need to establish what those context builders actually combined, where inputs came from, and whether any state persisted beyond a voice session. The conceptual distinction is promising: context construction decides what enters the immediate working set; tool exposure decides what the agent can observe or change outside it; long-term memory provides durable, governed context across sessions and agents.

## 14. Future demo constraints

Any experiment or demo must be built in a separate public repository. The portfolio essay should contain the explanation and direct links to reproducible code, measurements, and results.

The future system should demonstrate provider and framework neutrality rather than merely claim it. The minimum meaningful target is two genuinely different model providers, an adapter for the Deep Agents framework, and an adapter for CrewAI. The same records, scope and permission rules, ingestion semantics, retrieval contract, and evaluation scenarios should work through both frameworks.

The integration work itself is part of the experiment. Frameworks may differ in conversation state, agent identity, tool calls, run/session identifiers, callbacks, asynchronous behaviour, and how they format tool results for a model. The memory layer should absorb those differences in adapters rather than force framework-specific state into the durable store.

## 15. Research order and unresolved decisions

Before design or implementation, work through the following sequence.

1. Read CoALA for the taxonomy and boundaries between memory types.
2. Use the four-system article as a comparison map, then verify each system from primary sources.
3. Study Mem0’s paper, documentation, extraction, hybrid retrieval, entity, and lifecycle choices.
4. Identify and study the exact Hermes system and its primary sources.
5. Study query construction, dense retrieval, lexical retrieval, entity resolution, ANN, reranking, and no-memory decisions as separate retrieval questions.
6. Study skills and GEPA as adjacent procedural-learning and policy-optimization systems, without collapsing them into memory.
7. Define the smallest representative evaluation set before choosing models, indexes, or frameworks.
8. Only then design the separate experimental repository and implementation plan.

The unresolved questions that need deliberate answers include the source of truth, record schema, scope model, entity resolution, lifecycle semantics, extraction triggers, extraction model, validation policy, retrieval query construction, top-k values, thresholding, embedding model, ANN index, lexical strategy, reranking, model/provider adapters, and framework adapters.

## 16. Candidate essay thesis and shape

The candidate thesis is: **an agent becomes stateful not because it can quote an old turn, but because the system can turn an experience into a justified record, retrieve that record when it matters, and let a later action change because of it.**

The essay should not become a catalogue of vector stores or a survey of every self-improving-agent technique. It should lead with the stateless-model problem, explain the four memory types, compare contrasting real implementations, show why the retrieval policy is the critical design decision, and use the future experiment as a way to test the claims rather than decorate them.

The current strongest conclusion is: **storage gives memory a past; retrieval decides whether that past gets to influence the present.**
