# Agent Memory System: Research Notes and Initial Design Specification

These notes capture the research base, architectural intent, current baseline, and open experiments for the Agentic Memory System. They provide decision context for the high-level design and low-level design. The high-level design records current choices; these notes explain their rationale, limits, and validation work.

## 1. Problem statement

An individual language-model invocation has no inherent access to earlier interactions. Continuity is created by the surrounding system: conversation messages, summaries, files, tool state, project rules, retrieved records, and caches. Appending prior messages creates conversational memory. It is necessary for a coherent session, but it does not decide what is durable, current, authorized, private, or relevant across sessions.

The system needs durable state that agents can inspect and use without treating a growing transcript as a database. It must turn an observation into a justified record, govern that record through time and ownership changes, and retrieve it when it should influence an action.

The practical interpretation of “LLMs are stateless” is limited but useful: a prior interaction only affects a later response when some representation of it is made available to the model or agent. Providers may retain conversation objects, logs, and caches, but those mechanisms do not replace an application-owned memory policy.

## 2. Scope and success criteria

The initial system is a long-term, text-only memory layer for agents. It must preserve attributable semantic facts, episodic experiences, and procedural knowledge; retrieve relevant records under strict scope controls; support correction and expiry; and explain how every write and retrieval decision was made.

The current design excludes general document ingestion, full transcript archival, transactional business data, image and audio memory, file-content memory, distributed storage, cross-user entity merges, ambient prompt injection, and self-optimizing policy changes.

Success has three dimensions:

1. **Integrity:** records are faithful, attributable, scoped correctly, and correctable.
2. **Retrieval quality:** the right agent receives the right records at the right time, and receives nothing when nothing applies.
3. **Operational quality:** interactive writes and searches are fast enough to use routinely, while every stage remains observable and benchmarkable.

## 3. Mental model: conversation, RAG, and memory

Conversation history provides working context for references and live tasks. A transcript does not distinguish a temporary observation from a confirmed preference, an outdated decision, or an authorized project rule.

Retrieval-augmented generation supplies retrieval techniques. Dense search, keyword search, fusion, reranking, and context budgeting form the RAG portion of this system. The memory layer governs writes, ownership, source, lifecycle, time, correction, and auditability.

| Concern | Typical RAG corpus | Agent memory system |
| --- | --- | --- |
| Primary data | Imported documents and chunks. | Facts, preferences, decisions, outcomes, procedures, and entity relationships. |
| Write policy | Ingest or re-index source content. | Validate evidence, deduplicate, classify trust, apply lifecycle and lineage rules, then write. |
| Time | Often external to retrieval or handled as metadata. | Event time, creation time, expiry, supersession, and historical audit are first-class. |
| Ownership | Commonly one application corpus. | Every record has a scope; grants decide which agent may read or write it. |
| Retrieval outcome | Supporting passages. | Governed prior state that may change an agent’s action, with an explanation of why it was returned. |

The memory layer uses RAG for retrieval and governs the durable state around it.

## 4. Memory taxonomy and policies

The working taxonomy comes from [Cognitive Architectures for Language Agents (CoALA)](https://arxiv.org/abs/2309.02427). It supplies engineering categories for the system.

| Type | Meaning | Example | Initial policy |
| --- | --- | --- | --- |
| Working | Information active in the current run: recent messages, scratchpad, tool results, opened files, and session state. | “The active task is to evaluate query rewriting.” | Owned by the host framework, not stored as durable memory. |
| Semantic | A durable statement that tends to remain true until corrected. | “The user prefers concise technical answers.” | Supersede with a stronger or newer statement on the same subject; provisional records expire if never reinforced. |
| Episodic | A dated account of an event, outcome, decision, and rationale. | “On 3 September, the team chose SQLite because the system is single-process.” | Append rather than supersede; use event time and optional time filters in retrieval. |
| Procedural | A reusable, versioned way to perform a class of task. | “After changing an embedding model, re-embed the store and recalibrate retrieval gates.” | Keep deliberately small; a new version supersedes the prior procedure. |

Each category needs distinct write, retrieval, and lifecycle rules. A fabricated episode can repeat a bad strategy. A stale semantic preference can annoy a user. A stale procedure can produce unsafe operational behaviour. One undifferentiated store cannot apply the required controls.

## 5. Architectural stance

The current design uses a tool-mediated memory layer. Durable memory stays outside the prompt prefix. The stable prefix contains system instructions, tool definitions, and a short memory-use policy. The agent searches, inspects, writes, revises, or forgets memory through a bounded tool surface.

This approach protects prompt-cache stability and exposes retrieval decisions. A visible tool call adds a memory result to working context and records the query, readable scopes, candidates, ranking decisions, and final result.

Agents must recognize when a search helps. The evaluation suite includes search, no-search, and reject-result cases. A later experiment may test a small user or project profile; the current design does not include one.

The contract stays model-, provider-, and framework-neutral. It defines durable records, scopes, grants, lifecycle rules, tool schemas, and ingestion events. Framework adapters translate their tool and session semantics into that contract. Versioned interfaces isolate embedding, extraction, and reranking models; a model change requires a migration and evaluation.

## 6. Initial system baseline

The current baseline supports inspection and benchmarking on a laptop while preserving the boundaries needed for later scale.

| Area | Initial position | Rationale |
| --- | --- | --- |
| Source of truth | One SQLite database. | Strong local transactional semantics, simple inspection, and a single canonical record store. |
| Dense retrieval | `bge-m3` dense embeddings, exact cosine search over an in-memory matrix. | Provides an open, multilingual baseline without a network hop or ANN configuration. |
| Lexical retrieval | SQLite FTS5 with BM25 over content, normalized subject, and entity aliases. | Preserves exact matches for names, identifiers, products, filenames, and user vocabulary. |
| Entity retrieval | Exact alias matches within authorized scope, ordered by recency. | Avoids false merges and treats identity as a hard safety concern. |
| Candidate fusion | Reciprocal-rank fusion. | Dense, BM25, and entity scores are not naturally calibrated against one another. |
| Reranking | `bge-reranker-v2-m3`, specified behind a disabled feature flag. | Establishes a measurable option without paying its latency until it proves useful. |
| Query rewriting | Retrieval-owned optional stage, disabled by default. | Keeps query improvement inside retrieval while requiring evidence before it becomes a default dependency. |
| Context budget | Default maximum of 1,500 tokens. | Prevents retrieval from displacing the active task. |
| Observability | Structured write and search logs with per-stage timing and explanations. | Makes quality failures diagnosable and enables reproducible benchmarks. |

### Store, vector index, and full-text search

SQLite is the canonical store. It persists records, scope and grant data, source data, lifecycle state, entity links, events, logs, and embedding blobs. The vector index uses an in-memory matrix and record-id map rebuilt from persisted embeddings at startup. It performs exact cosine search over eligible records.

SQLite FTS5 is the lexical index. It uses an inverted index and BM25 ranking to find records containing relevant words, identifiers, and aliases. The vector index finds similar meaning; FTS5 protects exact terms; entity retrieval provides an exact identity signal. The three candidate lists are complementary and are fused only after scope and lifecycle filtering.

## 7. Durable records and lifecycle

Every durable record contains content and an envelope. Content states the memory. The envelope records authorization, trust, currency, and retrieval controls.

| Field group | Required information | Why it exists |
| --- | --- | --- |
| Identity | Record id, memory type, version. | Addresses the record and its revision lineage. |
| Content | Readable content and a normalized subject. | Supports model use, duplicate detection, and semantic contradiction checks. |
| Scope | Scope kind and id. | Identifies whether the record belongs to an agent, user, project, or organization. |
| Source and evidence | Source kind, source reference, creator agent, and evidence quote. | Establishes what supports the record and enables later inspection. |
| Time | Creation time, event time, and optional expiry time. | Distinguishes when the system learned something from when it happened. |
| Trust | Confidence and lifecycle status. | Distinguishes provisional inference from confirmed fact. |
| Lineage | Supersedes and conflicts-with references. | Preserves change history and contradictions. |
| Links | Entity links and tags. | Supplies exact retrieval handles and structured filtering. |

Default retrieval considers `provisional` and `confirmed` records. Audit retains `superseded`, `expired`, and `deleted` records outside the default result set. Expired records stay indexed for historical inspection but fail the eligibility filter. Deletion creates a tombstone, removes the FTS entry, marks the RAM vector dead, and sends durable content and embedding erasure through the controlled deletion process.

Source kinds have an explicit trust order: user statement, authoritative system data, tool result, then agent inference. A record can supersede only a record with equal or lower source rank. An inference remains provisional and expires after 30 days unless it is reinforced by later independent evidence, an agent revision, or direct user confirmation. Two independent observations promote it to confirmed.

## 8. Ingestion and extraction

The system exposes two write paths.

1. **Explicit agent write:** the agent calls `memory_write` with a type, content, normalized subject, scope, source kind, and evidence. The tool validates scope permission, verifies claimed user evidence against the current transcript, checks duplicates and contradictions, assigns status, creates embeddings and indexes, then returns a result synchronously.
2. **End-of-session extraction:** after a session ends, a separate structured-output model reads the transcript and proposes facts, decisions, outcomes, and one episodic session summary. Each candidate must include a verbatim evidence span. A validator rejects unsupported candidates, reinforces duplicates, applies contradiction and supersession rules, assigns status from the source, and writes accepted records asynchronously.

The current design excludes per-message extraction. It adds cost and can store partial conversation state or weak inference. Explicit writes cover facts that need immediate storage; session extraction runs once after the completed interaction. The evaluation suite compares missed writes with store-pollution rates.

The extractor runs apart from the serving model and uses a low-cost structured-output model. Evaluate a candidate such as `GLM 5.3 Flash` for extraction or query rewriting separately from embedding quality.

## 9. Retrieval policy

`memory_search` runs synchronously because an agent needs the result before choosing the next action. Dense, lexical, and entity candidate generators run in parallel inside the call. Session extraction runs in the background.

The pipeline has the following stages:

1. Optionally rewrite the raw retrieval request using the current-turn context. The stage is specified but disabled by default, and logs the raw and rewritten forms.
2. Resolve readable scopes for the requesting agent and principal user.
3. Apply hard filters for scope, lifecycle status, expiry, type, and requested time window.
4. Generate dense, lexical, and entity candidates in parallel, with up to 30 candidates per generator.
5. Fuse candidate rankings with reciprocal-rank fusion, then apply episodic freshness adjustment.
6. Gate weak results. If all signals are weak and there is no entity match, return no memory.
7. Collapse near-duplicates, optionally rerank survivors, and fill the context budget in rank order.
8. Return records with explanations and write a complete search log.

Scope filtering acts as a hard predicate rather than a ranking signal. The pipeline blocks records outside readable scopes before dense, lexical, entity, or reranking stages. An empty result signals that no record qualified.

Returned records must explain the raw and rewritten query, matching generators, per-generator score and rank, fused rank, freshness adjustment, reranker movement, source, status, dates, entity links, and why the record survived the gate, duplicate collapse, and token budget. An empty response records why no candidate qualified.

## 10. Scope, access, and entities

Scope answers “whose memory is this?” Access answers “which agent can read or write it?” They are distinct controls.

The initial scope kinds are `agent`, `user`, `project`, and `org`. A record has one scope. A grant table resolves the scopes an agent may read or write; each agent has access to its own agent scope. The policy requires explicit grants across scopes. A user grant does not grant project access, and a project grant does not grant organization access.

Entity memory models the subjects of records rather than adding a fifth memory type. Records may concern people, projects, organizations, repositories, products, tickets, or other durable entities. Each entity has a kind, canonical name, scope, and aliases. The current design resolves aliases by exact normalized match within readable scope. It creates a provisional entity when no match exists and requires an explicit revision with an audit event for each merge. A false merge can leak records across users.

## 11. Evaluation and performance

Benchmark the system as a state-management system. Include desired and prohibited writes; desired and prohibited retrievals; stale and conflicting records; entity ambiguity; cross-scope boundaries; search, no-search, and reject-result decisions; and downstream task effects.

| Evaluation layer | Questions |
| --- | --- |
| Write integrity | Did the system write when it should, avoid writes when it should not, preserve evidence and scope, and handle correction, supersession, expiry, and deletion correctly? |
| Retrieval quality | Did the system search when useful, avoid irrelevant search, retrieve the right candidates, exclude unauthorized and stale records, rank the right final context, and return nothing appropriately? |
| Task impact | Did retrieved memory improve the downstream action without distraction or over-personalization? |
| Operational behaviour | What are the latency, cost, token, cache, and scalability consequences of each model, index, and framework choice? |

Benchmarking must report end-to-end and per-stage timings. Search timing includes optional query rewriting, scope resolution, hard filtering, query embedding, dense retrieval, FTS5/BM25 retrieval, entity lookup, fusion, freshness, gating, deduplication, optional reranking, context assembly, explanation construction, and logging. Write timing includes permission and evidence validation, duplicate and contradiction checks, entity linking, embedding, index update, SQLite transaction, and logging.

The initial targets assume a warm single-process laptop deployment and a store of up to 50,000 records: `memory_search` without reranking targets p50 below 40 ms and p95 below 120 ms; `memory_write` targets p50 below 60 ms and p95 below 150 ms. The reranker budget adds 100 ms mean latency for 30 candidates. Benchmarks record p50, p95, mean, corpus size, candidate count, model version, feature flags, and cold-versus-warm state.

## 12. Comparative research and design implications

### CoALA

[CoALA](https://arxiv.org/abs/2309.02427) provides the taxonomy used here: working memory plus optional semantic, episodic, and procedural long-term memories. Its value is conceptual separation. The implementation must give each class different write, lifecycle, and retrieval treatment rather than placing all history into one vector index.

### ChatGPT, Claude, OpenClaw, and Hermes

Manthan Gupta’s [comparison of ChatGPT, Claude, OpenClaw, and Hermes](https://manthanguptaa.in/posts/memory_is_a_mistake/) is a useful system-comparison map, but its implementation claims require verification against each system’s primary sources. The important comparison dimensions are persistent representation, hot prompt context, cold recall, write policy, retrieval policy, memory-type separation, scope controls, and failure modes.

The current design is closest to the tool-mediated end of that spectrum: durable records are cold by default, the prompt prefix remains stable, and retrieval is explicit and logged. The reference article’s description of Hermes is useful for framing a bounded hot/cold split, but the exact Hermes source and version must be identified before treating its details as factual. The current design does not adopt a hot durable-fact block.

### Mem0

[Mem0](https://arxiv.org/abs/2504.19413) is a relevant memory-system case study. Its [memory-algorithm documentation](https://docs.mem0.ai/migration/platform-v2-to-v3) and [graph-memory documentation](https://docs.mem0.ai/platform/features/graph-memory) describe extraction, hybrid retrieval, entity linking, and reranking choices that are directly comparable to this design.

Evaluate add-only extraction, entity linking under ambiguous identity, hybrid retrieval on representative cases, source trust, and reranker latency. Mem0 supplies a case study rather than an implementation template.

### Skills, GEPA, and self-improvement

Memory, skills, and optimization are adjacent but distinct layers.

| Layer | Role |
| --- | --- |
| Memory | Stores durable observations, facts, decisions, and experiences. |
| Skill | Stores a reusable procedure for a task class. |
| Optimization | Selects or improves an artifact, such as a query-rewrite instruction, retrieval policy, or skill, using evaluation. |

[GEPA](https://arxiv.org/abs/2507.19457) is a reflective evolutionary optimizer. It may later improve bounded policy artifacts after the contract and evaluation set stabilize. It must not mutate user-scoped facts or rewrite memory records from a plausible reflection alone.

[MemSkill](https://arxiv.org/abs/2602.02474) is relevant because it explores the bridge between episodic evidence and evolving procedures. The boundary remains explicit: agents or human authors write procedures deliberately; automatic promotion of episodes into procedures is out of scope.

## 13. Open experiments and research questions

The baseline makes firm choices while leaving the following decisions open to experiment.

| Question | Baseline | Evidence needed before changing it |
| --- | --- | --- |
| Query rewriting | Implement as a disabled retrieval stage. | Compare raw and rewritten queries on conversational follow-ups, entity ambiguity, and no-memory cases. |
| Reranking | `bge-reranker-v2-m3`, disabled. | Measure final-context precision and downstream outcome improvement against latency by candidate count. |
| Ambient memory | No hot user or project profile. | Compare a bounded profile against tool-only retrieval for search misses, stale context, cache behaviour, and over-personalization. |
| Embedding model | `bge-m3` baseline. | Compare a Nomic Embed Text model, hosted candidates, and other models against this task’s labelled retrieval set, latency, privacy, licensing, and migration cost. |
| Vector search algorithm | Exact cosine through 50,000 records. | Revisit ANN only above 200,000 records or when exact search misses the latency budget. |
| Extraction cadence | Explicit writes plus session-end extraction. | Compare per-message extraction against write recall, cost, and store-pollution rate. |
| Entity strategy | Exact scoped aliases; no automatic merge. | Test whether more recall is worth the identity and cross-scope risk. |
| Procedural promotion | Manual only. | Define evidence and evaluation standards before allowing episodes to become procedures automatically. |

## 14. Next research and design actions

1. Verify comparative-system claims from primary sources, especially the exact Hermes implementation and version.
2. Build the initial labelled evaluation set before optimizing any retriever or model selection.
3. Define representative records, scopes, grants, conflicts, and user corrections for the evaluation fixtures.
4. Implement the baseline contract and direct retrieval tests before framework adapters.
5. Add Deep Agents and CrewAI adapters, then test the same contract with at least two model providers.
6. Benchmark write and retrieval stages independently, then run end-to-end task evaluations.
7. Run the deferred query-rewrite, reranker, ambient-memory, extraction-cadence, and model experiments one at a time against a fixed baseline.

Storage preserves prior state. Retrieval determines whether that state affects the next action.
