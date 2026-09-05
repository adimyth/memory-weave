# Agent Memory System Components

This guide explains the runtime components with concrete examples. Read the component map first to see how a search or write moves through the system. The [high-level design](agent-memory-hld.md) defines the architecture; the [low-level design](agent-memory-lld.md) defines the schemas and implementation contracts.

## 1. System component map

Memory Weave has one canonical store and three retrieval channels. The vector index and FTS5 index are derived search structures, not separate sources of truth.

```text
Agent framework
  -> Adapter -> tool handler -> policy and session buffer
  -> Ingestor -> SQLite store -> FTS5 index, entity links, vector index
  -> Retriever -> dense search + lexical FTS5 search + entity search -> RRF -> tool result
```

| Component | Owns | Reads | Writes or returns |
| --- | --- | --- | --- |
| Framework adapter | Framework integration and current-run identity. | Framework run state and current turns. | Derives `Principal`, records session hooks, registers tools, and adds search context. |
| Tool handlers | Framework-neutral entry points for the five memory tools. | Validated tool input and `Principal`. | Calls the policy, ingestor, or retriever and returns tool-safe results. |
| Policy service | Scope access, source authority, lifecycle, and evidence rules. | Grants, principal, source kind, record state, and session turns. | Eligible scopes, write decisions, statuses, and evidence checks. |
| Session buffer and extractor | The transcript needed for evidence and durable session memory. | User, assistant, and tool-result turns. | Numbered turns, extracted candidates, and one session summary. |
| Ingestor | The write path and current-fact decisions. | A tool request or extractor candidate, policy results, entities, and nearby records. | A new, reinforced, superseding, conflicting, or rejected record. |
| SQLite store | Canonical records, entities, grants, sessions, events, and logs. | Queries from all runtime services. | Transactional durable state and eligible record IDs. |
| FTS5 lexical index | Exact-word and identifier retrieval. | `content`, `subject`, and linked entity aliases from a record. | BM25-ranked lexical candidate IDs, term-match counts, and scores. |
| Vector index | Semantic similarity retrieval in process memory. | Compatible embedding rows and an eligible record-ID set. | Cosine-ranked dense candidate IDs and scores. |
| Entity resolver | Exact, scope-aware entity and alias lookup. | Entity mentions, aliases, grants, and scopes. | Entity IDs and entity-ranked candidate records, or an ambiguity error. |
| Retriever | The search pipeline and result explanation. | Search request, eligible IDs, three candidate channels, and configuration. | Fused, gated, deduplicated, budgeted results plus `search_log`. |
| Audit and search log | Why the system changed or returned a result. | Ingestor and retriever decisions and timers. | Append-only events and one complete search trace per request. |

The store owns the durable record and entity state. The ingestor updates the store, FTS5 row, entity links, and vector index as one write operation. At search time, the retriever applies the same eligible-ID set to dense, lexical, and entity channels before RRF combines their rankings.

## 2. Memory record

A memory record is one durable item an agent may retrieve in a later session. Its metadata tells the retrieval service whether the caller may access it, whether it remains current, and why the service can trust it.

```text
id: mem_01J9K4D0S6H2V8P3X7M5Q1R9T
type: semantic
version: 1
content: "Aditya prefers concise technical explanations."
subject_entity_id: ent_person_aditya
attribute: explanation_style
subject: ent_person_aditya/explanation_style
scope_kind: user
scope_id: aditya
source_kind: user_statement
source_ref: session:sess_42<turn:7>
creator_agent_id: research-agent
evidence: "Keep answers concise."
created_at: 2026-09-03T12:00:00Z
event_at: 2026-09-03T12:00:00Z
expires_at: null
confidence: 0.95
status: confirmed
supersedes_id: null
reinforcements: 0
last_reinforced_at: null
tags: ["communication"]
```

| Question | Fields that answer it |
| --- | --- |
| What does the memory say? | `content`, `type`, `tags` |
| Which current claim does it represent? | `subject_entity_id` and `attribute`; `subject` is their derived display and FTS value. |
| Which bucket owns it? | `scope_kind`, `scope_id` |
| Who supplied it, and what supports it? | `source_kind`, `source_ref`, `creator_agent_id`, `evidence` |
| When does it apply? | `created_at`, `event_at`, `expires_at` |
| What is its current state? | `confidence`, `status`, `supersedes_id`, `reinforcements`, `last_reinforced_at` |

## 3. Memory type and lifecycle

| Type | Use it for | Example |
| --- | --- | --- |
| Semantic | Durable facts and preferences | “The project uses Conventional Commits.” |
| Episodic | A dated experience, outcome, or decision | “On 2026-08-12, vector-only retrieval missed exact error messages.” |
| Procedural | A reusable method or workflow | “Run the benchmark with a warm embedder, then record per-stage timings.” |

| Status | Meaning | Record and index handling | Default retrieval behaviour |
| --- | --- | --- | --- |
| `provisional` | An agent inference or an unresolved lower-trust conflict. | Record and indexes remain active. | Eligible. |
| `confirmed` | Source or reinforcement meets the confirmation policy. | Record and indexes remain active. | Eligible. |
| `superseded` | A newer record replaced this current claim. | Record and indexes remain so history can be searched on request. | Excluded unless the caller asks for history. |
| `expired` | Its expiry date passed. | Record and indexes remain so history can be inspected or reactivated. The eligibility filter blocks it from normal search. | Excluded. |
| `deleted` | The user requested removal. | The row remains as an audit tombstone; the service removes its FTS row and marks its RAM vector entry dead. A controlled erase path removes stored content and embeddings. | Excluded. |

Semantic and procedural records describe current knowledge. Episodic records preserve history and carry `event_at`, so the retriever can reduce the rank of old episodes for a non-time-bounded query.

## 4. Scope

Scope is the ownership boundary on a memory record. It answers: “Where does this memory belong?”

| Scope | Example | Use it for |
| --- | --- | --- |
| Agent | `agent:researcher/aditya` | Private working knowledge for one agent-user pair. A plain `agent:researcher` scope requires a grant. |
| User | `user:aditya` | Preferences and facts about one user. |
| Project | `project:agentic-memory-system` | Shared decisions, conventions, and findings for one project. |
| Organization | `org:acme` | Knowledge shared across an organization. |

For example, a commit convention belongs in project scope:

```text
content: "Commit messages follow Conventional Commits."
scope_kind: project
scope_id: agentic-memory-system
subject_entity_id: ent_project_agentic_memory_system
attribute: commit_convention
subject: ent_project_agentic_memory_system/commit_convention
```

The same text in `user:aditya` scope would represent a personal preference, rather than a rule that project contributors must follow.

## 5. Principal, grants, and enforcement

A principal identifies the caller of a memory tool.

```text
agent_id: implementation-agent
user_id: aditya
project_id: agentic-memory-system
session_id: sess_42
```

A grant gives that agent access to one scope.

```text
agent_id: implementation-agent
scope_kind: project
scope_id: agentic-memory-system
can_read: true
can_write: false
```

The agent may search project memory but cannot change it. Scope describes ownership. A grant describes permission.

The service enforces grants and scopes before it ranks records:

1. The framework adapter derives the principal from the current agent run.
2. The policy service resolves the agent’s readable or writable scopes. The implicit scope is `agent:<agent_id>/<user_id>`, private to that principal pair. Agent and user IDs cannot contain `/`, which keeps that encoded scope unique. User, project, organization, and plain agent scopes require an explicit grant from the host. The host refuses grants on private agent scopes, and policy ignores a direct-store private grant when its encoded user does not match the current principal.
3. `memory_search` runs an indexed SQL filter for readable scopes, lifecycle state, expiry, type, and time range. It produces the eligible record IDs.
4. Dense search receives those IDs and builds its position mask while holding the index lock. FTS and entity search intersect their hits with the same eligible IDs. The retriever also drops any fused candidate outside that set. No candidate generator can return an out-of-scope record.
5. `memory_write` checks that the requested scope appears in the caller’s writable scopes before it creates a record. A failed check returns `scope_not_writable`.

A grant for `user:aditya` is valid only when the current principal is `user:aditya`. Project and organization scope support deliberate sharing across users.

## 6. Source and evidence

Source identifies where a memory came from. Evidence preserves the supporting material.

```text
source_kind: tool_result
source_ref: tool:github-commit-abc123
evidence: "docs(memory): add component guide"
creator_agent_id: implementation-agent
```

| Source kind | Source rank | Initial confidence | Initial status |
| --- | ---: | ---: | --- |
| `user_statement` | 4 | 0.95 | Confirmed |
| `system` | 3 | 0.90 | Confirmed |
| `tool_result` | 2 | 0.85 | Confirmed |
| `session_summary` | 2 | 0.80 | Confirmed |
| `agent_inference` | 1 | 0.60 | Provisional |

An agent inference expires after the provisional time-to-live unless later evidence reinforces it. `confidence` records lifecycle state, but current retrieval does not use it to rank, gate, or render a result.

A user or tool claim needs two checks. Its evidence must appear verbatim in the cited turn, and the evidence must entail the stored content at or above `ingestion.evidence.entail_floor`. A failed entailment check downgrades the claim to `agent_inference` and records the score and note in the audit event.

## 7. Current-fact keys, conflicts, supersession, and reinforcement

`subject_entity_id` and `attribute` identify the current claim a non-episodic record represents. `subject` is the derived form `<entity_id>/<attribute>`, used for display and FTS indexing.

```text
ent_person_aditya/explanation_style
ent_project_agentic_memory_system/commit_convention
```

Consider these records in the same user scope:

```text
Old record
subject_entity_id: ent_person_aditya
attribute: explanation_style
subject: ent_person_aditya/explanation_style
content: "Aditya prefers concise technical explanations."
source_kind: user_statement
status: confirmed

New candidate
subject_entity_id: ent_person_aditya
attribute: explanation_style
subject: ent_person_aditya/explanation_style
content: "Aditya prefers detailed architecture explanations."
source_kind: agent_inference
```

Both records attempt to answer the same current question: “What explanation style does Aditya prefer?” The ingestion service handles the relationship as follows:

| Situation | System action |
| --- | --- |
| The content has the same meaning. | Reinforce the old record. The service does not insert a duplicate. |
| The new source has an equal or higher source rank. | Insert the new record, set `supersedes_id` to the old record ID, and mark the old record `superseded`. |
| The new source has a lower source rank and conflicts with the old record. | Store the candidate as `provisional`, link both records in `record_conflicts`, and keep the confirmed record as the normal search result. |

Supersession preserves the previous fact for audit and historical inspection. Normal retrieval excludes the older record. `memory_get` and `include_history` can retrieve it.

In the example above, the final state is unambiguous: the user statement has source rank 4, and the agent inference has source rank 1. The old record remains `confirmed` and appears in default search. The candidate remains `provisional`, receives a conflict link to the old record, and remains visible through `memory_get` for inspection. It expires after the provisional time-to-live unless later evidence reinforces or confirms it.

Reinforcement means later evidence supports the same memory. The service increments `reinforcements`, raises confidence up to `0.99`, refreshes the provisional expiry date, and confirms a provisional record after the configured number of reinforcements.

The service also scans active records about the same entity for equivalent attribute names. If a candidate labelled `answer_style` contradicts an existing `explanation_style` record, the judge treats those labels as one current fact. The authoritative survivor keeps the existing attribute, and the audit event records `attribute_aliased_from: answer_style`. The scan is capped by `ingestion.max_entity_attributes`; the event records `attribute_scan_truncated` if it reaches that cap.

Episodic records never supersede one another. “Vector-only retrieval missed exact error messages on Tuesday” and “FTS improved error lookup on Friday” describe separate events worth retaining.

## 8. Entities, aliases, and record links

An entity represents a named subject such as a person, project, repository, product, or organization. An alias gives that entity alternate names.

The entity rows define the names:

| Entity ID | Kind | Canonical name | Aliases |
| --- | --- | --- | --- |
| `ent_project_memory` | Project | `agentic-memory-system` | `agent-memory-system`, `memory system` |
| `ent_repo_cli` | Repository | `agent-memory-cli` | `memory CLI` |

The memory row holds the statement:

| Memory ID | Content |
| --- | --- |
| `mem_01J9K4D0S6H2V8P3X7M5Q1R9T` | “The project uses Conventional Commits.” |

The `record_entities` join table links that memory to the two entity rows:

| Record ID | Entity ID | Role |
| --- | --- | --- |
| `mem_01J9K4D0S6H2V8P3X7M5Q1R9T` | `ent_project_memory` | `about` |
| `mem_01J9K4D0S6H2V8P3X7M5Q1R9T` | `ent_repo_cli` | `mentions` |

The `about` link says that the memory’s main topic is the project. The `mentions` link says that the memory also refers to the command-line repository.

If an agent searches for `memory system`, alias lookup resolves that phrase to `ent_project_memory`. The service then reads `record_entities`, finds `mem_01J9K4D0S6H2V8P3X7M5Q1R9T`, and adds that memory to entity-search candidates. A search for the command-line repository can also find this memory through its `mentions` link, subject to scope and other retrieval filters.

The service creates a provisional entity for an unknown name. It leaves an ambiguous name unmerged for review, rather than guessing that two similarly named things are the same entity.

## 9. Embeddings and the vector index

An embedding is a numeric representation of a record’s meaning. BGE-M3 produces a 1,024-number vector for each record.

The SQLite `embeddings` table stores that vector on disk:

```text
record_id: mem_01J9K4D0S6H2V8P3X7M5Q1R9T
model: BAAI/bge-m3
version: 1
dims: 1024
vector: [0.012, -0.084, ...]
```

The vector index lives in process memory. On boot, the service scans compatible embedding rows from SQLite and recreates the index.

| In-memory structure | Purpose |
| --- | --- |
| `ids` | Maps each matrix row to a memory record ID. |
| `matrix` | Holds one normalized embedding per row. |
| `pos` | Maps a record ID back to its matrix row. |
| `live` | Marks deleted records unavailable without rebuilding the matrix during that process run. |

The current design targets and benchmarks a store of up to 50,000 records, rather than enforcing a hard maximum. A 1,024-dimension `float32` vector occupies 4 KiB, so 50,000 vectors occupy about 200 MB before Python and index overhead. The current estimate for a full SQLite scan and index rebuild at that size is about one second, or about 0.02 ms per record on average. That is an estimate, not a measurement. The benchmark must report cold-start and warm-start p50 and p95 times at several store sizes.

For a search, the embedder turns the query into a vector. The index calculates cosine similarity between that query vector and the eligible memory vectors, then returns the highest-scoring record IDs. The current design uses exact search over the in-memory matrix rather than approximate nearest-neighbor search.

SQLite remains the durable source of truth. The RAM index is a rebuildable search accelerator.

## 10. Full-text search (FTS)

FTS5 is the lexical retrieval component. It is a SQLite virtual table called `records_fts`, not a separate search service. It answers queries where exact terms matter more than semantic similarity: error text, identifiers, commands, attribute names, and entity aliases.

| Indexed field | Source | Why FTS indexes it |
| --- | --- | --- |
| `record_id` | The canonical memory record ID. | Maps an FTS match back to the record. FTS does not tokenize this field. |
| `content` | `records.content`. | Finds exact words and phrases in the memory text. |
| `subject` | Derived `records.subject`. | Finds attribute keys such as `ent_project_agentic_memory_system/commit_convention`. |
| `aliases` | Linked entity aliases, joined into one string at write time. | Lets a name query find a record even when its content uses a pronoun or a canonical name. |

The ingestor creates or replaces the FTS row in the same transaction that writes a memory record and its entity links. `memory_forget` removes the FTS row because deleted records must not appear in lexical results. FTS5 remains rebuildable from canonical records and entity links.

SQLite FTS5 builds an inverted word index over `content`, `subject`, and entity `aliases`.

```text
Query: "OAuth refresh failure"

Matches:
- content: "The OAuth refresh token expired after one hour."
- subject: "ent_product_auth_service/oauth_refresh"
- aliases: "auth", "OAuth service"
```

The lexical generator runs these steps for each search query:

1. It detects identifiers from raw whitespace tokens, then tokenizes with FTS5's `unicode61` rules.
2. It removes a frequency-based English stopword list and terms shorter than two characters, but keeps identifiers and proper nouns.
3. It joins the remaining terms with `OR` and runs `bm25(records_fts, 0.0, 1.0, 2.0, 3.0)`. The leading `0.0` skips the unindexed `record_id` column. The remaining weights favor `aliases` over `subject`, and `subject` over `content`.
4. It joins FTS5 to a temporary table of eligible record IDs before applying `per_generator_k`, so inaccessible high-ranking rows cannot crowd out results.
5. It returns each record's BM25 rank and score, plus the exact matched terms from the query that gives it the best coverage. Each term says whether it is an identifier or an entity alias.

FTS ranks literal-word matches with BM25. It complements vector search, which works better when the query paraphrases a memory rather than repeating its terms.

The alias text is denormalized into each FTS row. When an alias changes or two entities merge, the ingestor must rewrite every affected row so lexical search does not use stale alias text.

## 11. Retrieval channels and ranking

The retrieval service runs three candidate generators after it applies scope, grant, lifecycle, type, and time filters.

| Channel | Input | Strength |
| --- | --- | --- |
| Dense | Query embedding against the vector index | Finds semantic similarity. |
| Lexical | Query terms against FTS5 | Finds exact words and identifiers. |
| Entity | Entity and alias lookup | Finds memories about a known named subject. |

### Reciprocal Rank Fusion (RRF)

The three channels use incompatible scores. Cosine similarity and BM25 values have different scales, and entity lookup has no comparable numeric score. RRF combines ranking positions instead of raw scores.

For a query about commit rules, two channels return these lists:

| Rank | Lexical search | Dense search |
| ---: | --- | --- |
| 1 | Memory A | Memory C |
| 2 | Memory B | Memory A |
| 3 | Memory C | Memory D |

The current configuration uses `rrf_k = 60`. For each list where a record appears, the service adds `1 / (60 + rank)` to that record’s fused score.

| Memory | Calculation | Fused score | Final rank |
| --- | --- | ---: | ---: |
| A | `1/61 + 1/62` | 0.03252 | 1 |
| C | `1/63 + 1/61` | 0.03227 | 2 |
| B | `1/62` | 0.01613 | 3 |
| D | `1/63` | 0.01587 | 4 |

Memory A ranks first because both channels found it near the top. Memory C has the best dense rank but a lower lexical rank, so it follows A. The entity channel uses the same calculation when it returns a record. A record found by all three channels receives three contributions.

### Result control after RRF

| Stage | Purpose |
| --- | --- |
| Episodic freshness | Reduces the score of an old episodic record. Semantic and procedural records keep their score. |
| Relevance gate | Removes weak dense-only or lexical-only matches. An entity match passes the gate. |
| Duplicate collapse | Keeps the higher-ranked record when two surviving records have near-identical embeddings. |
| Optional reranker | Uses `bge-reranker-v2-m3` to score up to 30 survivors against the query. The initial configuration keeps it off until evaluation proves that its extra latency helps. |
| Token budget | Selects whole records that fit the tool-result budget. |

The response includes an explanation for each returned record:

```text
matched: dense 0.71 (rank 2), lexical 3/3 terms (rank 1)
fused rank: 1
gate: passed dense threshold
```

The service writes this reasoning and per-stage timings to `search_log` for inspection and evaluation.

## 12. Query rewriting

The serving agent does not rewrite its own query. It sends a raw request to `memory_search`.

```text
Agent request: "what does he prefer?"
Adapter context: "Aditya asked for concise technical explanations."
Retrieval query: "Aditya's preferred explanation style"
```

`QueryRewriter` is the interface for this step. `HostedLLMQueryRewriter` is the initial implementation of that interface. It calls the configured hosted model, `claude-haiku-4-5-20251001`, through a structured-output request. `NoRewriter` is another implementation: it returns the agent’s raw query unchanged while rewriting remains disabled.

When rewriting is enabled, the framework adapter adds the latest user and assistant turns to `SearchRequest.context`. `HostedLLMQueryRewriter` combines that context with the raw request and returns a clearer search phrase. The retriever sends that phrase to dense and lexical search.

The initial configuration keeps rewriting off because the model call adds 300 to 800 ms to the retrieval path. The service uses the raw query if the call times out or fails, then records `rewrite_status: failed` in `search_log`.

## 13. Ingestion, sessions, and audit trail

An agent can call `memory_write` during a session. The request checks writable scope and evidence, resolves entity links, checks duplicates and conflicts, creates an embedding, writes SQLite rows, updates indexes, and writes an audit event. The agent waits for that tool result.

The adapter stores user, assistant, and tool turns in `session_turns`. The extractor runs once after the session ends. It reads the complete transcript, proposes evidence-backed candidate records, and writes one episodic session summary.

The append-only `events` table records memory changes. `search_log` records retrieval decisions. Together, those tables show who changed a memory, which source supported it, and why retrieval returned or rejected it.

## 14. Memory tools and framework adapters

| Tool | Agent action |
| --- | --- |
| `memory_search` | Find relevant, authorized memories. |
| `memory_get` | Inspect complete records, including conflicts and supersession history. |
| `memory_write` | Create a memory with source and evidence. |
| `memory_revise` | Confirm, supersede, expire, or merge an entity. |
| `memory_forget` | Mark a memory deleted with a reason. |

An adapter integrates those tools with an agent framework. It registers the tool schemas, derives the principal, records session hooks, and attaches current-turn context to each search. The memory contract remains independent of the agent framework and model provider.

## 15. Prompts and instructions

Two runtime components turn transcript text into a decision by calling a hosted model: session extraction and query rewriting. Each needs a natural-language prompt that tells the model how to reason, plus a schema that tells it how to answer. Neither prompt exists in the codebase yet.

| Component | Config that names the model | Code location once implemented | Status |
| --- | --- | --- | --- |
| Session extraction | `ingestion.extraction_model` | `ingest/extractor.py`, `StructuredLLMExtractor` | The `Extractor` protocol and its output dataclasses (`ExtractionContext`, `CandidateRecord`, `SessionSummary`) are specified in `models.py`. The implementation and its prompt are not yet written; only `FakeExtractor` exists for tests. |
| Query rewriting | `retrieval.rewrite.model` | `retrieve/rewrite.py`, `HostedLLMQueryRewriter` | Only `NoRewriter` exists today; it returns queries unchanged and never calls a model. The hosted implementation and its prompt are not yet written. |

`ExtractionContext.prompt_version` already anticipates a versioned prompt: it exists so a later change to the extraction prompt can be tracked on the `extraction.run` event the same way an embedding-model change is tracked by `embeddings.version`, rather than silently changing behavior.

When these are implemented, the instructions belong next to the code that uses them, not in `config.yaml`. The YAML config holds thresholds and model *names*, values a calibration pass would sweep. Prompt text is reasoning guidance, not a tunable, so it belongs as a module-level constant in `ingest/extractor.py` and `retrieve/rewrite.py` respectively, or a small `ingest/prompts.py` if it grows past one string.
