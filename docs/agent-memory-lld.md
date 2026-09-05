# Agent Memory System: Low-Level Design

This document specifies the implementation of decisions in `agent-memory-hld.md`.

Language is Python 3.12. The package is called `memory_weave` and is distributed as `memory-weave`; the CLI command is `memory-weave`. Everything runs in one process with one SQLite file.

Memory Weave gives an agent durable, scoped memory. It stores evidence-backed records in SQLite, searches them with dense, lexical, and entity retrieval, fuses the rankings with RRF, and returns only records the caller may read. It also keeps an audit trail for each write and search.

The document moves from static contracts to runtime behavior: package and configuration, storage and types, policy, ingestion, retrieval, integration, and tests. Sections 3 and 10 are the main references for data ownership and the search path.

## 1. Package layout

The package mirrors the path a memory takes through the system. `store` holds durable state, `index` holds model-backed retrieval structures, `ingest` validates and writes memories, and `retrieve` reads them back.

| Area | Responsibility |
| --- | --- |
| Root modules | Shared configuration, data types, logging, and the CLI. |
| `store` | Schema migrations and all SQLite reads and writes. |
| `index` | Embedding, vector search, and optional reranking. |
| `ingest` | Evidence checks, duplicate and contradiction decisions, extraction, and session capture. |
| `retrieve` | Query rewriting, candidate generation, RRF, gating, freshness, deduplication, and explanations. |
| `policy` | Scope access and record lifecycle rules. |
| `tools` and `adapters` | Framework-neutral tool handlers and framework-specific integration. |

```bash
memory_weave/
  __init__.py
  config.py          # typed config, loaded from YAML, all thresholds live here
  models.py          # dataclasses for Record, Entity, SearchRequest, SearchResult, Candidate
  store/
    schema.sql       # DDL, applied by migrations
    migrations.py    # versioned, forward-only
    store.py         # Store: CRUD, FTS, event log, search log
  index/
    embedder.py      # Embedder protocol + BgeM3Embedder + FakeEmbedder for tests
    vector.py        # VectorIndex: in-memory matrix over the store
    reranker.py      # Reranker protocol + BgeReranker + NoReranker
  ingest/
    ingestor.py      # write path: validate, dedup, contradiction, supersede, link, embed
    evidence.py      # validate_evidence: the one helper both write paths use to locate a quote and verify the claimed source
    equivalence.py   # EquivalenceJudge protocol + NLICrossEncoderJudge + FakeJudge; decides same / contradicts / distinct
    extractor.py     # Extractor protocol + StructuredLLMExtractor + FakeExtractor
    session.py       # SessionBuffer: turns for the running session, used by extraction and evidence checks
  retrieve/
    retriever.py     # the pipeline
    rewrite.py       # QueryRewriter protocol + HostedLLMQueryRewriter + NoRewriter (default)
    fusion.py        # Reciprocal Rank Fusion
    gate.py          # empty-result decision
    freshness.py     # episodic recency
    explain.py       # builds the per-result Explanation object and the response-level empty reason
  policy/
    grants.py        # readable/writable scope resolution
    lifecycle.py     # source ranks, status transitions, expiry
  tools/
    schemas.py       # JSON schema for the five tools
    handlers.py      # tool name -> handler, framework-agnostic
  adapters/
    base.py          # Adapter protocol
    deepagents.py
    crewai.py
  log.py             # structured logging helpers
  cli.py             # inspect, search, dump, migrate, reembed
```

## 2. Configuration

Memory Weave loads its tunables from one YAML file through `load_config`. The following sections define that file's blocks.

### 2.1 Store and embedding

```yaml
store:
  path: ./memory.sqlite

embedding:
  model: BAAI/bge-m3
  version: "1"
  dims: 1024
  device: auto
  max_chars: 2000
  query_cache_entries: 4096
  incremental_reload_max: 512
```

- `version` tags each stored vector. `VectorIndex.load` accepts rows that match the current model and version. Bump it when the model or preprocessing changes, then run `memory-weave reembed`.
- `dims` is the vector width. It must match the model's output.
- `device` selects `mps` on Apple silicon, `cuda` when available, and `cpu` otherwise.
- `max_chars` truncates content before embedding. Stored content is never truncated.
- `query_cache_entries` bounds the least-recently-used cache of exact query embeddings. Document embeddings never enter this cache. The cap prevents long-running `auto` and `hybrid` adapters from retaining one vector for every user turn.
- `incremental_reload_max` is the largest durable index delta that a running process applies in place. A larger delta rebuilds the vector index, which keeps a long period of inactivity from turning the next search into many small updates.

### 2.2 Reranker

```yaml
reranker:
  enabled: false
  model: BAAI/bge-reranker-v2-m3
  candidates: 30
  floor: null
  budget_mean_ms: 100
```

- `enabled` adds a cross-encoder pass after duplicate collapse. The cross-encoder scores each query-record pair and reorders the survivors; refer section 10.7.
- `candidates` is the maximum number of records sent to the reranker after the initial gate and duplicate collapse. The reranker scores every selected record against every query, then keeps the record's best score. For three queries and 30 records, that is at most `3 × 30 = 90` query-record scores.
- `floor` is the minimum best reranker score a record needs to be returned. Dense and lexical floors still keep weak records out of the reranker shortlist, but, when reranking is enabled, `floor` makes the final keep-or-drop decision. `null` means the threshold has not been calibrated, so `load_config` rejects an enabled reranker without a floor.
- `budget_mean_ms` sets the expected cost for 30 candidates on the target laptop. The benchmark compares measured p50 and p95 against it.

### 2.3 Retrieval

```yaml
retrieval:
  rewrite:
    enabled: false
    model: claude-haiku-4-5-20251001
    max_context_chars: 2000
    timeout_ms: 800
  trigger:
    mode: tool_only           # tool_only | auto | hybrid; refer section 14.1
    auto_k: 4                 # k for host-issued searches in auto and hybrid modes
    auto_min_query_chars: 12  # host-issued search is skipped for shorter user turns
  per_generator_k: 30
  max_alias_tokens: 4
  rrf_k: 60
  default_k: 8
  token_budget: 1500
  dedup_cosine: 0.92
  gate:
    dense_floor:
      semantic: 0.45
      episodic: 0.40
      procedural: 0.45
    lexical_min_term_fraction: 0.5
    lexical_min_matched_terms: 2
    relative_floor: 0.5
    auto:
      dense_floor:
        semantic: 0.55
        episodic: 0.50
        procedural: 0.55
      lexical_min_term_fraction: 0.6
      lexical_min_matched_terms: 2
      relative_floor: 0.6
      exclude_source_kinds: [session_summary]
  freshness:
    episodic_half_life_days: 30
    floor: 0.5
```

The retriever applies these settings in the order shown. Section 10 defines the pipeline.

- `trigger.mode` decides who calls `memory_search`: the model through its tool (`tool_only`), the host once per user turn (`auto`), or both (`hybrid`); refer section 14.1. The pipeline, the gate, and the log are identical in every mode. Only the caller changes.
- `trigger.auto_k` caps host-issued searches. It is smaller than `default_k` because nobody asked for those results; they must earn their place.
- `trigger.auto_min_query_chars` skips the host-issued search when the user turn is shorter than this after whitespace normalization. "ok", "yes", and "do it" are not queries.

- `rewrite.enabled` rewrites the query before searching, using the last user and assistant turns: "what does he prefer?" becomes "Aditya's preferred explanation style"; refer section 10.0. The feature defaults to off because the hosted call sits on the hot path.
- `rewrite.max_context_chars` caps the combined current-turn context that the adapter sends to the rewrite model. By default, this context is the most recent user turn plus assistant turn. It does not truncate the stored session transcript or the search queries.
- `rewrite.timeout_ms` bounds the wait. On timeout the search uses the raw query and logs `rewrite_status = failed`.
- `per_generator_k` limits each candidate generator to its top records: dense vector search, lexical FTS search, and entity search. At `30`, RRF receives at most 30 ranked positions from each generator, or 90 positions total. The same record can appear in more than one list, so the number of unique candidates can be lower. Increasing this value raises work in later stages.
- `max_alias_tokens` bounds each query-derived entity alias to this many whitespace tokens. Alias lookup batches its candidates and uses temporary tables for matching entity IDs, so a long pasted query cannot exhaust SQLite bound parameters.
- `rrf_k` is the fusion constant. Each list a record appears in adds `1 / (60 + rank)` to its score; refer section 10.3.
- `freshness.episodic_half_life_days` halves an episodic record's score for every 30 days of event age; refer section 10.4. Semantic and procedural records do not decay.
- `freshness.floor` is the lowest decay multiplier.
- `gate.dense_floor.<type>` is the lowest cosine a dense-only candidate of that record type may have; refer section 10.5. Episodic summaries are long, so their cosine against a short query runs lower than a short semantic fact's, and one shared floor would under-retrieve episodes while over-retrieving facts.
- `gate.lexical_min_term_fraction` is the fraction of query terms a lexical-only candidate must contain.
- `gate.lexical_min_matched_terms` is the smallest number of matched terms a lexical-only candidate needs, unless one matched term is an entity alias or an identifier token. It stops a one-word query such as "deployment" from admitting every record that mentions deployment.
- `gate.relative_floor` drops survivors whose fused score is below this fraction of the strongest survivor with the same number of contributing channels. Entity hits are exempt. This preserves strong single-channel candidates when a different record is corroborated by several generators.
- `gate.auto` replaces every gate floor for a host-issued search and excludes the listed source kinds. Its default excludes session summaries because generated summaries are broad and should not arrive unasked.
- `dedup_cosine` drops a survivor this similar to a record already kept; refer section 10.6.
- `default_k` is how many records a search returns when the caller omits `SearchRequest.k`; refer section 10.8.
- `token_budget` is the token ceiling on the tool result.

The gate keeps a candidate that clears an absolute floor or matches an entity exactly, then applies the relative floor. Recalibrate every `dense_floor` value and `relative_floor` and record them with `embedding.version` after an embedder change; section 10.5 describes the three calibration classes.

Budget filling stops at `default_k` or `token_budget`, whichever it reaches first. Long records can leave a response below `k`.

### 2.4 Ingestion

```yaml
ingestion:
  dedup_candidate_cosine: 0.85
  equivalence:
    model: cross-encoder/nli-deberta-v3-small
    entail_floor: 0.70
    contradict_floor: 0.70
  evidence:
    min_characters: 15
    min_words: 3
    entail_floor: 0.70
  provisional_ttl_days: 30
  reinforcements_to_confirm: 2
  max_entity_attributes: 64
  extraction_model: claude-haiku-4-5-20251001
  extraction_max_candidates: 20
```

- `dedup_candidate_cosine` is the similarity threshold for comparing active records in the same scope and type but about different entities. At or above this cosine, the ingestor asks the equivalence judge whether the claims are the same, contradictory, or distinct.
- `equivalence.entail_floor` is the minimum directed entailment score for calling two claims the same. The judge compares the existing claim to the new claim and the new claim to the existing claim; both scores must meet this floor before the ingestor reinforces the existing record.
- `equivalence.contradict_floor` is the minimum directed contradiction score for recording a conflict. If either direction meets this floor, the ingestor treats the claims as contradictory rather than as a duplicate.
- `evidence.min_characters` and `evidence.min_words` make a quote substantial enough to support a direct claim.
- `evidence.entail_floor` is the minimum score when the judge compares the evidence quote with a direct user or tool claim. It is calibrated separately from the equivalence floors because it answers a different question.
- `provisional_ttl_days` sets the lifetime for every provisional record, including an unsupported inference and a lower-authority conflict; refer sections 6.2 and 6.4. An independent reinforcement extends the expiry date.
- `reinforcements_to_confirm` sets the reinforcement count that promotes a provisional record to `confirmed`; refer section 6.4.
- `max_entity_attributes` bounds the active attributes that the ingestor asks the judge to compare for one entity. It starts with the most recently reinforced attributes and records `attribute_scan_truncated` when it reaches the cap.
- `extraction_max_candidates` caps the candidates from one session extraction; refer section 8.2.

### 2.5 Policy

```yaml
policy:
  source_rank:
    user_statement: 4
    system: 3
    tool_result: 2
    session_summary: 2
    agent_inference: 1
```

`source_rank` determines which record wins a disagreement: a record may supersede an equal- or lower-ranked record. The rank also sets initial status and confidence; refer section 6.2. `session_summary` shares rank 2 with `tool_result`, and the extractor writes it.

## 3. Schema

The store uses one SQLite file with WAL mode and foreign keys enabled. Timestamps use ISO 8601 UTC strings. IDs use UUIDv7, so creation order also sorts by ID.

Read this section in two passes: use the table map to learn where data lives, then use the field descriptions and DDL when implementing migrations or store methods.

| Area | Tables | Purpose |
| --- | --- | --- |
| Memories | `records`, `record_conflicts` | Store current and historical memory claims, plus disagreements. |
| Retrieval indexes | `embeddings`, `records_fts`, `store_meta` | Store derived vector and full-text search data, plus the version used to refresh another process's vector projection. |
| Entity graph | `entities`, `entity_aliases`, `record_entities` | Link memories to people, projects, repos, and other named subjects. |
| Access and sessions | `grants`, `sessions`, `session_turns` | Enforce scope access and retain the transcript needed for evidence validation. |
| Migration review | `migration_issues` | Preserve legacy rows that migration 2 could not safely map to a structured subject. |
| Audit and observability | `events`, `search_log` | Explain writes, lifecycle changes, and each retrieval decision. |

### Reference DDL

```sql
CREATE TABLE records (
  id              TEXT PRIMARY KEY,
  type            TEXT NOT NULL CHECK (type IN ('semantic','episodic','procedural')),
  version         INTEGER NOT NULL DEFAULT 1,
  content         TEXT NOT NULL,
  subject         TEXT NOT NULL,              -- derived '<subject_entity_id>/<attribute>' display and lookup value
  subject_entity_id TEXT REFERENCES entities(id), -- primary entity for a current fact; NULL when no subject applies
  attribute       TEXT,                       -- normalized lowercase snake_case current-fact attribute
  scope_kind      TEXT NOT NULL CHECK (scope_kind IN ('agent','user','project','org')),
  scope_id        TEXT NOT NULL,
  source_kind     TEXT NOT NULL CHECK (source_kind IN ('user_statement','system','tool_result','session_summary','agent_inference')),
  source_ref      TEXT,                       -- 'session:session_id<turn:n>' or a tool call id or a document id
  creator_agent_id TEXT NOT NULL,
  evidence        TEXT,                       -- verbatim quote from source_ref
  created_at      TEXT NOT NULL,
  event_at        TEXT NOT NULL,              -- defaults to created_at
  expires_at      TEXT,                       -- NULL means never
  confidence      REAL NOT NULL,              -- 0..1
  status          TEXT NOT NULL CHECK (status IN ('provisional','confirmed','superseded','expired','deleted')),
  supersedes_id   TEXT REFERENCES records(id),
  reinforcements  INTEGER NOT NULL DEFAULT 0,
  last_reinforced_at TEXT,
  index_version   INTEGER NOT NULL DEFAULT 0, -- durable change version for vector-index refresh
  tags            TEXT NOT NULL DEFAULT '[]'  -- JSON array
);
CREATE INDEX records_scope ON records(scope_kind, scope_id, status);
CREATE INDEX records_subject ON records(scope_kind, scope_id, subject_entity_id, attribute, status);
CREATE INDEX records_event ON records(type, event_at);
CREATE INDEX records_index_version ON records(index_version);

CREATE TABLE migration_issues (
  migration_version INTEGER NOT NULL,
  record_id   TEXT NOT NULL REFERENCES records(id),
  issue       TEXT NOT NULL,
  PRIMARY KEY (migration_version, record_id)
);

CREATE TABLE record_conflicts (
  record_id   TEXT NOT NULL REFERENCES records(id),
  other_id    TEXT NOT NULL REFERENCES records(id),
  noted_at    TEXT NOT NULL,
  PRIMARY KEY (record_id, other_id)
);

CREATE TABLE embeddings (
  record_id   TEXT PRIMARY KEY REFERENCES records(id),
  model       TEXT NOT NULL,
  version     TEXT NOT NULL,
  dims        INTEGER NOT NULL,
  vector      BLOB NOT NULL,                  -- float32, L2-normalized
  index_version INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX embeddings_index_version ON embeddings(index_version);

CREATE TABLE store_meta (
  key         TEXT PRIMARY KEY,
  value       INTEGER NOT NULL
);
INSERT INTO store_meta(key, value) VALUES ('records_version', 0);

CREATE VIRTUAL TABLE records_fts USING fts5(
  record_id UNINDEXED,
  content,
  subject,
  aliases,                                    -- space-joined aliases of linked entities, denormalized at write time
  tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE entities (
  id            TEXT PRIMARY KEY,
  kind          TEXT NOT NULL CHECK (kind IN ('person','project','org','repo','product','other')),
  canonical     TEXT NOT NULL,
  scope_kind    TEXT NOT NULL,
  scope_id      TEXT NOT NULL,
  status        TEXT NOT NULL CHECK (status IN ('provisional','confirmed','merged','deleted')),
  merged_into   TEXT REFERENCES entities(id),
  created_at    TEXT NOT NULL
);

CREATE TABLE entity_aliases (
  entity_id   TEXT NOT NULL REFERENCES entities(id),
  alias_norm  TEXT NOT NULL,                  -- lowercased, whitespace-collapsed, diacritics stripped
  PRIMARY KEY (entity_id, alias_norm)
);
CREATE INDEX entity_aliases_lookup ON entity_aliases(alias_norm);

CREATE TABLE record_entities (
  record_id   TEXT NOT NULL REFERENCES records(id),
  entity_id   TEXT NOT NULL REFERENCES entities(id),
  role        TEXT NOT NULL DEFAULT 'about',  -- 'about' | 'mentions'
  PRIMARY KEY (record_id, entity_id)
);
CREATE INDEX record_entities_by_entity ON record_entities(entity_id);

CREATE TABLE grants (
  agent_id    TEXT NOT NULL,
  scope_kind  TEXT NOT NULL,
  scope_id    TEXT NOT NULL,
  can_read    INTEGER NOT NULL DEFAULT 1,
  can_write   INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (agent_id, scope_kind, scope_id)
);

CREATE TABLE sessions (
  id          TEXT PRIMARY KEY,
  agent_id    TEXT NOT NULL,
  user_id     TEXT NOT NULL,
  project_id  TEXT,
  started_at  TEXT NOT NULL,
  ended_at    TEXT,
  extraction_started_at TEXT,
  extracted_at TEXT
);

CREATE TABLE session_turns (
  session_id  TEXT NOT NULL REFERENCES sessions(id),
  turn        INTEGER NOT NULL,
  role        TEXT NOT NULL,                  -- 'user' | 'assistant' | 'tool'
  content     TEXT NOT NULL,
  at          TEXT NOT NULL,
  PRIMARY KEY (session_id, turn)
);

CREATE TABLE events (                         -- append-only audit log, never updated or deleted
  id          TEXT PRIMARY KEY,
  at          TEXT NOT NULL,
  kind        TEXT NOT NULL,                  -- record.created | record.reinforced | record.superseded | record.status | entity.created | entity.merged | grant.changed | extraction.run
  actor       TEXT NOT NULL,                  -- agent id, 'extractor', 'admin', or a user id
  record_id   TEXT,
  entity_id   TEXT,
  payload     TEXT NOT NULL                   -- JSON
);

CREATE TABLE search_log (
  id            TEXT PRIMARY KEY,
  at            TEXT NOT NULL,
  agent_id      TEXT NOT NULL,
  user_id       TEXT NOT NULL,
  session_id    TEXT,
  trigger       TEXT NOT NULL DEFAULT 'tool', -- 'tool' when the model called memory_search, 'auto' when the host did (section 14.1)
  request       TEXT NOT NULL,                -- JSON of the SearchRequest as received (raw queries, filters, k)
  context       TEXT,                         -- host-supplied current-turn context, as passed to the rewriter; NULL when none
  rewrite_status TEXT NOT NULL,               -- 'disabled' | 'applied' | 'unchanged' | 'failed'
  rewritten_queries TEXT,                     -- JSON [str, ...] or NULL when rewrite_status is 'disabled' or 'failed'
  readable_scopes TEXT NOT NULL,              -- JSON
  dense         TEXT NOT NULL,                -- JSON [[record_id, rank, cosine], ...]
  lexical       TEXT NOT NULL,                -- JSON [[record_id, rank, bm25, matched_terms, total_terms], ...]
  entity        TEXT NOT NULL,                -- JSON [[record_id, rank, entity_id], ...]
  fused         TEXT NOT NULL,                -- JSON [[record_id, fused_rank, rrf_score], ...] before freshness
  freshness     TEXT NOT NULL,                -- JSON [[record_id, multiplier], ...] for adjusted (episodic) records only
  gated_out     TEXT NOT NULL,                -- JSON [[record_id, reason], ...]
  deduped_out   TEXT NOT NULL,                -- JSON [[dropped_id, kept_id, cosine], ...]
  reranked      TEXT,                         -- JSON [[record_id, rank_before, rank_after, score, winning_query], ...] or NULL when disabled
  reranked_out  TEXT NOT NULL DEFAULT '[]',   -- JSON records excluded by the reranker shortlist or floor, with the reason
  budget_out    TEXT NOT NULL,                -- JSON [record_id, ...] survivors that did not fit k or the token budget
  returned      TEXT NOT NULL,                -- JSON [record_id, ...]
  explanations  TEXT NOT NULL,                -- JSON [Explanation, ...] one per returned record, plus empty_reason when none
  config_flags  TEXT NOT NULL,                -- JSON {embedding_model, embedding_version, rewrite_enabled, reranker_enabled, gate floors}
  warm          INTEGER NOT NULL,             -- 1 if the embedder and index were already loaded when the call started
  timings_ms    TEXT NOT NULL                 -- JSON {rewrite, scopes, filter, embed, dense, lexical, entity, fuse, freshness, gate, dedup, rerank, budget, explain, log, total}
);
```

### Key relationships

- `subject_entity_id` and `attribute` are the current-fact key for semantic and procedural records. `subject` is a derived value in the format `<entity_id>/<attribute>`, which the service rewrites after an entity merge. The writer may suggest only the attribute; the resolved primary `about` entity supplies the entity ID. Episodic records may use `<entity_id>/-` when they have a primary entity, but that form is never a supersession key.
- The ingestor copies entity aliases into each FTS row, so a name query can find records whose content uses a pronoun.
- SQLite stores embedding vectors as blobs. The process rebuilds its in-memory matrix from those blobs at startup.
- `events` provides the audit trail. `records` stores the current materialized state. The system uses the log for explanation, not replay.
- Write-path events carry the completed stages and list stages still pending when the event is appended; section 8.1 defines the complete `WriteResult` timing contract. `extraction.run` carries the stages from section 8.2. The benchmark reads search timing from `search_log` and complete write timing from `WriteResult` values collected by the harness.

### 3.1 `records`: the canonical memory row

`records` stores the memory text, who owns it, what supports it, and its lifecycle. Retrieval indexes derive from this table.

| Field | Meaning |
| --- | --- |
| `id` | UUIDv7 record identifier. |
| `type` | `semantic`, `episodic`, or `procedural`. |
| `version` | Revision number for the record lineage. |
| `content` | Memory text returned to an agent. |
| `subject` | Derived display and FTS value: `<subject_entity_id>/<attribute>`. |
| `subject_entity_id` | Primary `about` entity for the current fact. The ingestor updates it when entities merge. |
| `attribute` | Normalized current-fact attribute such as `explanation_style`. |
| `scope_kind` | Ownership boundary: `agent`, `user`, `project`, or `org`. |
| `scope_id` | Identifier inside the ownership boundary. |
| `source_kind` | Evidence class: `user_statement`, `system`, `tool_result`, `session_summary`, or `agent_inference`. |
| `source_ref` | Session turn, tool call, or document that supports the record. |
| `creator_agent_id` | Agent that created the record. |
| `evidence` | Verbatim supporting quote. |
| `created_at` | Time when Memory Weave stored the row. |
| `event_at` | Time when the fact or event occurred. Defaults to `created_at`. |
| `expires_at` | Time after which normal retrieval excludes the row. `NULL` means no expiry. |
| `confidence` | Confidence value from 0 to 1. |
| `status` | `provisional`, `confirmed`, `superseded`, `expired`, or `deleted`. |
| `supersedes_id` | Record replaced by this record. |
| `reinforcements` | Count of later observations that supported this claim. |
| `last_reinforced_at` | Time of the latest reinforcement. |
| `tags` | JSON array of caller-supplied labels. |

`records_scope` supports the hard scope and lifecycle filter. `records_subject` supports structured current-fact lookup during ingestion. `records_event` supports time-ordered retrieval.

### 3.2 `migration_issues`: legacy rows requiring review

Migration 2 moves the old subject string to the structured `subject_entity_id` and `attribute` fields. It can do that only when a legacy record has exactly one `about` link and an attribute segment in its old subject. It records every other row here instead of silently treating it as a subject-less record.

| Field | Meaning |
| --- | --- |
| `migration_version` | The migration that detected the issue, currently `2`. |
| `record_id` | Legacy memory that needs a primary entity or attribute repair. |
| `issue` | Why the migration could not safely map the row, such as zero or multiple `about` links. |

`migrate()` returns the number of unresolved rows. `Store` refuses to open a database that has unresolved rows unless constructed with `allow_migration_issues=True`, because an unmapped legacy record is an active fact that new writes can neither supersede nor conflict with. `memory-weave migrate` lists each unresolved record and exits with status 2 until an operator retires them with `--expire-unmapped` (which sets `status = expired` and appends a `record.status` event per row) or acknowledges them with `--allow-unmapped`. The backfill normalizes the legacy attribute segment with `normalize_attribute`, so a legacy key matches the slug a new write produces.

### 3.3 Conflict and retrieval-index tables

`record_conflicts` retains incompatible claims that cannot replace each other. The ingestor writes the relationship in both directions.

| Field | Meaning |
| --- | --- |
| `record_id` | One record in the conflict. |
| `other_id` | The conflicting record. |
| `noted_at` | Time when the ingestor recorded the conflict. |

`embeddings` stores the durable vector for each record. The process rebuilds the vector index from rows whose `model` and `version` match the active configuration.

| Field | Meaning |
| --- | --- |
| `record_id` | Memory record represented by the vector. |
| `model` | Embedding model name. |
| `version` | Application-managed model or preprocessing version. |
| `dims` | Vector width. |
| `vector` | L2-normalized `float32` vector stored as a BLOB. |

`records_fts` is the lexical search index. FTS5 is a virtual table: it holds an inverted index from words to rows rather than authoritative data, so it can be dropped and rebuilt from `records`. The ingestor writes its row in the same transaction as the record and `memory_forget` removes it. The channel exists because dense vectors rank paraphrases above exact strings such as error messages, identifiers, and commands.

| Field | Meaning |
| --- | --- |
| `record_id` | Record ID returned by an FTS match. FTS does not tokenize this field. |
| `content` | Indexed memory text. |
| `subject` | Indexed current-fact key. |
| `aliases` | Space-separated aliases for linked entities. |
| `tokenize` | `unicode61 remove_diacritics 2`, which preserves Unicode terms and removes diacritics for matching. |

The retriever loads eligible record IDs into a temporary SQLite table and joins it to FTS5 before applying the result limit; refer section 10.2. `aliases` is denormalized at write time, so adding an alias or merging two entities rewrites every affected FTS row.

### 3.4 Entity tables

`entities` stores canonical names. `entity_aliases` maps normalized spellings to those names. `record_entities` applies those names to memory records.

| `entities` field | Meaning |
| --- | --- |
| `id` | Entity identifier. |
| `kind` | `person`, `project`, `org`, `repo`, `product`, or `other`. |
| `canonical` | Preferred display name with leading, trailing, and repeated whitespace collapsed. It preserves accents. |
| `scope_kind` | Scope that owns the entity. |
| `scope_id` | Identifier inside that scope. |
| `status` | `provisional`, `confirmed`, `merged`, or `deleted`. |
| `merged_into` | Surviving entity after a manual merge. |
| `created_at` | Entity creation time. |

| `entity_aliases` field | Meaning |
| --- | --- |
| `entity_id` | Canonical entity that owns the alias. |
| `alias_norm` | Lowercased, whitespace-collapsed, diacritic-free alias used for exact lookup. |

`entity_aliases_lookup` supports exact alias resolution.

| `record_entities` field | Meaning |
| --- | --- |
| `record_id` | Linked memory record. |
| `entity_id` | Linked entity. |
| `role` | `about` marks the primary subject. `mentions` records a secondary reference. |

`record_entities_by_entity` supports entity-based retrieval.

### 3.5 Grants and session transcript

`grants` gives an agent access to one shared scope. The only implicit scope is `agent:<agent_id>/<user_id>`, which is private to one agent-user pair and has no grant row. Agent and user IDs must not contain `/`, so this representation cannot collide. A plain `agent:<agent_id>` scope requires an explicit grant. `MemoryHost.grant` refuses an agent scope containing `/`, so hosts and the CLI cannot grant access to a private scope.

| Field | Meaning |
| --- | --- |
| `agent_id` | Agent receiving the grant. |
| `scope_kind` | Granted scope category. |
| `scope_id` | Granted scope identifier. |
| `can_read` | `1` when the agent may retrieve records from the scope. |
| `can_write` | `1` when the agent may create or revise records in the scope. |

`sessions` identifies an agent run. `session_turns` stores the transcript used by evidence validation and session extraction.

| `sessions` field | Meaning |
| --- | --- |
| `id` | Session identifier supplied by the adapter. |
| `agent_id` | Agent that ran the session. |
| `user_id` | User associated with the session. |
| `project_id` | Optional project associated with the session. |
| `started_at` | Session start time. |
| `ended_at` | Session end time, set by `on_session_end` or the idle timeout. |
| `extraction_started_at` | Time when a worker claimed the session for extraction. It prevents duplicate workers from processing the same session. |
| `extracted_at` | Time when background extraction completed. |

| `session_turns` field | Meaning |
| --- | --- |
| `session_id` | Parent session. |
| `turn` | Monotonic turn number inside the session. |
| `role` | `user`, `assistant`, or `tool`. |
| `content` | Turn text. Tool turns store result text, not tool-call arguments. |
| `at` | Turn timestamp. |

### 3.6 Audit events and search logs

`events` is append-only. It records writes, lifecycle changes, entity work, grant changes, and extraction runs. `records` remains the current materialized state; events explain changes and do not rebuild it.

| Field | Meaning |
| --- | --- |
| `id` | Event identifier. |
| `at` | Time when the actor made the change. |
| `kind` | Change type such as `record.created`, `record.reinforced`, `entity.merged`, or `extraction.run`. |
| `actor` | Agent ID, `extractor`, `admin`, or user ID that made the change. |
| `record_id` | Related record when relevant. |
| `entity_id` | Related entity when relevant. |
| `payload` | JSON details, including reasons and write or extraction timings. |

`search_log` captures one complete `memory_search` execution for debugging, evaluation, and latency analysis.

| Field | Meaning |
| --- | --- |
| `id` | Search identifier, created before pipeline work begins. |
| `at` | Search start time. |
| `agent_id` | Requesting agent. |
| `user_id` | Principal user for the request. |
| `session_id` | Optional source session. |
| `trigger` | `tool` when the model issued the search, `auto` when the host issued it after a user turn; refer section 14.1. Added in schema migration 2. |
| `request` | Original `SearchRequest`, including raw queries and filters. |
| `context` | Host-supplied context passed to the rewriter. `NULL` when absent. |
| `rewrite_status` | `disabled`, `applied`, `unchanged`, or `failed`. |
| `rewritten_queries` | Rewriter output. `NULL` when rewriting is disabled or fails. |
| `readable_scopes` | Scopes resolved before candidate generation. |
| `dense` | Dense candidates with rank and cosine. |
| `lexical` | Lexical candidates with rank, BM25 score, and term counts. |
| `entity` | Entity candidates with rank and matching entity ID. |
| `fused` | RRF ranking before freshness adjustment. |
| `freshness` | Episodic freshness multipliers. |
| `gated_out` | Candidates rejected by the relevance gate and their reasons. |
| `deduped_out` | Duplicates removed after gating. |
| `reranked` | Reranker results, including the winning query for each score. `NULL` when reranking is disabled. |
| `reranked_out` | Candidates excluded by the reranker shortlist limit or reranker floor, with the reason and applicable score. Added in schema migration 5. |
| `budget_out` | Candidates that survived ranking but did not fit `k` or the token budget. |
| `returned` | Final record IDs. |
| `explanations` | One `Explanation` per returned record and the empty-result reason when relevant. |
| `config_flags` | Model versions, feature flags, and gate settings used for the search. |
| `warm` | `1` when the embedder and vector index were already loaded. |
| `timings_ms` | Per-stage search timing from rewrite through logging. |

Write events carry the completed stages and their `timings_pending` list from section 8.1. The benchmark harness records complete write timings from `WriteResult`; `extraction.run` events carry the stages from section 8.2. The benchmark reads retrieval timing from `search_log`.

## 4. Core types

These dataclasses form the framework-neutral contract. Storage, ingestion, retrieval, tools, and adapters exchange these types instead of framework-specific objects.

```python
@dataclass(frozen=True)
class Scope:
    kind: Literal["agent", "user", "project", "org"]
    id: str


@dataclass
class Record:
    id: str
    type: Literal["semantic", "episodic", "procedural"]
    version: int
    content: str
    subject: str
    subject_entity_id: str | None
    attribute: str | None
    scope: Scope
    source_kind: Literal["user_statement", "system", "tool_result", "session_summary", "agent_inference"]
    source_ref: str | None
    creator_agent_id: str
    evidence: str | None
    created_at: datetime
    event_at: datetime
    expires_at: datetime | None
    confidence: float
    status: Literal["provisional", "confirmed", "superseded", "expired", "deleted"]
    supersedes_id: str | None
    reinforcements: int
    last_reinforced_at: datetime | None
    tags: list[str]
    entity_ids: list[str]


@dataclass
class Entity:
    id: str
    kind: Literal["person", "project", "org", "repo", "product", "other"]
    canonical: str
    scope: Scope
    status: Literal["provisional", "confirmed", "merged", "deleted"]
    merged_into: str | None
    aliases: list[str]  # normalized forms
    created_at: datetime


@dataclass(frozen=True)
class EntityMention:
    kind: Literal["person", "project", "org", "repo", "product", "other"]
    text: str  # as written in the source
    role: Literal["about", "mentions"]
    entity_id: str | None = None  # explicit id supplied by the writer; skips alias resolution when set


@dataclass(frozen=True)
class Turn:
    session_id: str
    turn: int
    role: Literal["user", "assistant", "tool"]
    content: str
    at: datetime


@dataclass(frozen=True)
class ExtractionContext:
    principal: Principal
    known_entities: list[tuple[str, str, str]]  # (entity_id, kind, canonical) readable by the agent
    existing_subjects: list[str]  # active subjects in the agent's writable scopes
    prompt_version: str


@dataclass(frozen=True)
class RewriteResult:
    queries: list[str]
    status: Literal["applied", "unchanged", "failed"]


@dataclass(frozen=True)
class EvidenceCheck:
    found: bool
    turn: int | None
    role: Literal["user", "assistant", "tool"] | None
    source_kind: Literal["user_statement", "tool_result", "agent_inference"]  # source kind supported by the evidence
    note: str | None  # set when the claimed kind was downgraded


@dataclass(frozen=True)
class Principal:
    agent_id: str
    user_id: str
    session_id: str | None
    project_id: str | None


@dataclass(frozen=True)
class SearchRequest:
    queries: list[str]  # 1 to 3, as the agent wrote them (the raw retrieval request)
    context: str | None  # host-supplied current-turn context, set by the adapter, never by the agent
    types: list[str] | None
    entities: list[str] | None  # alias strings, resolved by the retriever
    since: datetime | None  # applies to event_at
    until: datetime | None
    k: int
    include_history: bool  # superseded and expired become eligible
    trigger: Literal["tool", "auto"] = "tool"  # set by the adapter; "auto" for host-issued searches (section 14.1)


@dataclass
class GeneratorHit:
    rank: int  # 1-based rank within that generator's list
    score: float  # cosine, bm25, or 0.0 for entity


@dataclass(frozen=True)
class LexicalTerm:
    value: str
    is_identifier: bool
    is_entity_alias: bool


@dataclass(frozen=True)
class LexicalMatch:
    terms: tuple[LexicalTerm, ...]
    total_terms: int


@dataclass
class Candidate:
    record_id: str
    dense: GeneratorHit | None
    lexical: GeneratorHit | None
    lexical_terms: LexicalMatch | None
    entity: GeneratorHit | None
    entity_id: str | None
    rrf_score: float
    fused_rank: int
    freshness_multiplier: float | None  # set for episodic records only
    score: float  # rrf_score * freshness_multiplier
    gate_reason: str | None  # why it passed, or why it was dropped
    rerank_score: float | None
    rank_after_rerank: int | None


@dataclass
class Explanation:  # one per returned record; the HLD's "explanation object"
    raw_queries: list[str]
    rewritten_queries: list[str] | None
    rewrite_status: Literal["disabled", "applied", "unchanged", "failed"]
    matched_by: list[Literal["dense", "lexical", "entity"]]
    dense: GeneratorHit | None
    lexical: GeneratorHit | None
    lexical_terms: LexicalMatch | None
    entity: GeneratorHit | None
    fused_rank: int
    freshness_multiplier: float | None
    rerank: tuple[int, int, float] | None  # rank_before, rank_after, score; None when the reranker is disabled
    gate: str  # which floor or match let it through
    dedup: str  # "kept" or "kept over <dropped_id> at cosine 0.94"
    budget: str  # "fit at position 3 of 8, 412 tokens used"
    source_kind: str
    status: str
    created_at: datetime
    event_at: datetime
    entity_ids: list[str]
    conflicts_with: list[str]  # unresolved conflict record IDs shown with this result
    summary: str  # one line, human readable, rendered to the model


@dataclass
class SearchResult:
    record: Record
    score: float
    explanation: Explanation


@dataclass
class SearchResponse:
    search_id: str
    raw_queries: list[str]
    rewritten_queries: list[str] | None
    rewrite_status: Literal["disabled", "applied", "unchanged", "failed"]
    results: list[SearchResult]
    empty_reason: str | None  # set when results is empty; states which floors the best candidate missed
    timings_ms: dict[str, float]
```

### Type guide

| Type | Used by | Purpose |
| --- | --- | --- |
| `Scope` | Policy, store, tools | Names one ownership boundary. |
| `Principal` | Adapters, policy, tools | Identifies the calling agent, user, session, and project. |
| `Record` | Store, ingestion, retrieval | Represents one memory with its evidence, lifecycle, and entity links. |
| `Entity`, `EntityMention`, `Resolution` | Entity resolver, ingestion, retrieval | Represent a named subject, a proposed mention, and the resolver outcome. |
| `Turn` | Session buffer, evidence, extraction | Represents one transcript turn. |
| `ExtractionContext` | Extraction | Gives the extractor the principal, readable entities, existing subjects, and prompt version. |
| `CandidateRecord`, `SessionSummary`, `ExtractionOutput` | Extraction, ingestion | Represent extractor output before the ingestor validates it. |
| `EvidenceCheck` | Explicit writes, extraction | Records whether a quote exists and which source kind it supports. |
| `SearchRequest` | Tools, adapters, retrieval | Carries raw queries and caller-selected filters. The adapter owns `context`. |
| `GeneratorHit`, `LexicalTerm`, `LexicalMatch`, `Candidate` | Retrieval | Preserve generator ranks, scores, and exact lexical evidence through fusion, gating, dedupe, and reranking. |
| `RewriteResult` | Query rewrite stage | Carries rewritten queries and the rewrite status. |
| `Explanation`, `SearchResult`, `SearchResponse` | Retrieval, tools | Return a memory, its retrieval evidence, response metadata, and timings. |

Use `Record` for persisted state and `CandidateRecord` for untrusted extractor output. The ingestor converts a candidate into a record only after it validates evidence, scope, entities, and lifecycle rules.

## 5. Interfaces

Each model-facing dependency uses a `Protocol`. Unit tests use fakes, so they do not download models or call hosted services.

```python
class Embedder(Protocol):
    name: str
    version: str
    dims: int
    @property
    def is_loaded(self) -> bool: ...
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...    # (n, dims) float32, L2-normalized; exact-string query cache
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...  # (n, dims) float32, L2-normalized; never enters the query cache

class Reranker(Protocol):
    def score(self, query: str, docs: list[str]) -> list[float]: ...

class QueryRewriter(Protocol):
    def rewrite(self, queries: list[str], context: str | None) -> RewriteResult: ...

class EquivalenceJudge(Protocol):
    def judge(self, a: str, b: str) -> Literal["same", "contradicts", "distinct"]: ...
    def entails(self, premise: str, hypothesis: str) -> float: ...
    # NLICrossEncoderJudge scores both directions with a local NLI cross-encoder; FakeJudge is table-driven for tests.

class Extractor(Protocol):
    def extract(self, transcript: list[Turn], context: ExtractionContext) -> ExtractionOutput: ...

class Adapter(Protocol):
    def register_tools(self, handlers: ToolHandlers) -> None: ...
    def principal_from_run(self, run_ctx: Any) -> Principal: ...
    def on_session_start(self, ...) -> None: ...
    def on_turn(self, ...) -> None: ...
    def on_session_end(self, ...) -> None: ...
```

`ExtractionOutput` contains candidate records plus one session summary:

```python
@dataclass
class CandidateRecord:
    type: Literal["semantic", "episodic", "procedural"]
    content: str
    attribute: str | None  # hint only; the ingestor resolves the primary entity and derived subject
    source_kind: Literal["user_statement", "tool_result", "agent_inference"]
    evidence: str  # verbatim
    evidence_turn: int
    entity_mentions: list[EntityMention]  # (kind, text, role)
    event_at: datetime | None
    confidence: float


@dataclass
class SessionSummary:
    content: str  # 3 to 8 sentences: goal, what happened, decisions, open threads
    decisions: list[str]
    entity_mentions: list[EntityMention]
```

| Protocol | Production implementation | Test implementation | Contract |
| --- | --- | --- | --- |
| `Embedder` | `BgeM3Embedder` | `FakeEmbedder` | Returns L2-normalized vectors with configured dimensions. |
| `Reranker` | `BgeReranker` | `FakeReranker` | Scores one query against candidate documents. |
| `QueryRewriter` | `HostedLLMQueryRewriter` | `FakeRewriter` | Returns the same number of standalone queries or a failed status. |
| `EquivalenceJudge` | `NLICrossEncoderJudge` | `FakeJudge` | Labels two claims as `same`, `contradicts`, or `distinct`, and scores evidence-to-claim entailment. |
| `Extractor` | `StructuredLLMExtractor` | `FakeExtractor` | Produces `ExtractionOutput` from a transcript and extraction context. |
| `Adapter` | Deep Agents or CrewAI adapter | Adapter fixture | Registers tools, derives the principal, records turns, and closes sessions. |

## 6. Policy

Policy code answers four questions: which scopes a caller may use, how much authority a source has, whether a new claim replaces an old claim, and whether a quote supports its claimed source.

### 6.1 Grants

```python
def readable_scopes(store, agent_id, user_id) -> list[Scope]:
    scopes = [Scope("agent", f"{agent_id}/{user_id}")]  # implicit private scope
    scopes += store.grants_for(agent_id, can_read=True)
    return [s for s in scopes if s.kind != "user" or s.id == user_id]
```

The user-scope condition blocks a grant on `user:X` unless the current principal is `X`. The same safety belt applies to a legacy or direct-store grant on a private agent scope: its encoded user suffix must match the current principal user. Cross-user reads use `org`, `project`, or an explicitly granted plain agent scope.

`writable_scopes` follows the same rule with `can_write=True`. The caller always has read and write access to the agent-and-user private scope. `Principal` rejects `/` in `agent_id` and `user_id`, and `MemoryHost.grant` applies the same validation to an agent ID and user-scope ID. A host provisions every other scope through `MemoryHost.grant(agent_id, scope, read, write)` and may remove it with `MemoryHost.revoke(...)`. The `memory-weave grant` command wraps that host API.

### 6.2 Source rank and initial status

```python
def initial_status(source_kind) -> str:
    return "provisional" if source_kind == "agent_inference" else "confirmed"


def initial_confidence(source_kind) -> float:
    return {"user_statement": 0.95, "system": 0.9, "tool_result": 0.85, "session_summary": 0.8, "agent_inference": 0.6}[
        source_kind
    ]


def initial_expiry(source_kind, now) -> datetime | None:
    return now + timedelta(days=cfg.provisional_ttl_days) if source_kind == "agent_inference" else None
```

| Source kind | Rank | Initial status | Initial confidence | Expiry |
| --- | ---: | --- | ---: | --- |
| `user_statement` | 4 | `confirmed` | 0.95 | None |
| `system` | 3 | `confirmed` | 0.90 | None |
| `tool_result` | 2 | `confirmed` | 0.85 | None |
| `session_summary` | 2 | `confirmed` | 0.80 | None |
| `agent_inference` | 1 | `provisional` | 0.60 | `provisional_ttl_days` after creation |

The policy assigns `session_summary` its status and expiry rules. The summary references the whole transcript through `source_ref = "session:<id>"`. The extractor writes it as an episodic record with the principal's person entity and `attribute = "-"`; it does not supersede or reinforce another record. The summary describes a dated session. Store a user fact through a separate evidenced candidate.

`confidence` is stored for audit and lifecycle reporting. The current retriever, gate, and rendering logic do not read it as a ranking signal. Phase 15 data decides whether to make it a ranking prior or remove it.

When the supersession rule stores any source as a provisional conflict, it gives that row the same `provisional_ttl_days` expiry as an inference. A lower-authority conflict is useful for inspection, but it must not remain active forever without independent support.

### 6.3 Current facts, attribute aliases, and supersession

For a semantic or procedural write, the ingestor resolves one primary `about` entity and normalizes the writer's attribute hint to lowercase `snake_case`. It then loads every active record with the same scope, type, and `subject_entity_id`. This finds both an exact attribute match and older equivalent attribute names such as `explanation_style` and `answer_style`.

The scan is limited by `ingestion.max_entity_attributes`. The cap is applied to the store's order, most recently reinforced first, with the same-attribute authority incumbent always kept. Cosine similarity then orders the records that survived the cap; it never decides which records are excluded. If the cap is reached, the event records `attribute_scan_truncated`.

The ingestor judges every scanned record before it decides anything, then applies the rules in this order:

1. If active records share the candidate's attribute, the authority incumbent among them (highest source rank, then latest `event_at`, then latest `created_at`) receives the decision. A `same` verdict reinforces it. Any other verdict runs the supersession rule against it. When the candidate supersedes the incumbent, every same-attribute sibling the judge called `same` is marked superseded as well and listed in the event as `also_superseded`, so a provisional conflict that agrees with the new fact cannot outlive the fact it contradicted.
2. Otherwise, if any record under another attribute is `same`, the ingestor reinforces the highest-authority one and records the candidate attribute in `attribute_aliased_from`.
3. Otherwise, if a record under another attribute `contradicts` the candidate, the attributes describe one current fact. The supersession rule runs against the highest-authority such record, the survivor keeps the incumbent attribute, and the event records `attribute_aliased_from`.
4. Otherwise the nearby-record dedup search runs as for any record type.

Rule 1 runs before rule 2 on purpose. A `same` verdict against a provisional sibling must never short-circuit the decision the confirmed incumbent is owed; before this ordering, a user statement that agreed with a provisional conflict promoted the conflict and left the contradicted confirmed record active.

```
verdict = judge.judge(old.content, new.content)          # "same" | "contradicts" | "distinct"

if verdict == "same":
    reinforce(old); do not insert

elif has_authority(new, old):                            # see below
    insert new with supersedes_id=old.id; old.status = superseded

else:                                                    # lower-authority contradiction or distinct claim
    insert new as provisional; add conflict rows both ways
```

Authority is rank first, then time:

```python
def has_authority(new, old) -> bool:
    if rank(new) != rank(old):
        return rank(new) > rank(old)
    if new.event_at != old.event_at:
        return new.event_at > old.event_at  # equal rank: the later event wins, whenever it arrived
    return new.created_at >= old.created_at  # same event time: the later write wins
```

| Judge result and authority | Ingestor action |
| --- | --- |
| `same` | Reinforce the existing record. Do not insert another row. |
| `contradicts` or `distinct`, and new record has authority | Insert the new record, set `supersedes_id`, and mark the old record `superseded`. |
| `contradicts` or `distinct`, and old record has authority | Insert the new record as `provisional` and add conflict rows in both directions. |

Authority compares source rank first, then `event_at`, then `created_at`. A user statement therefore outranks a tool result even when the tool result describes a later event.

A stale, equal-ranked record cannot supersede a newer fact. The ingestor stores it with `status = superseded`, leaves `supersedes_id` empty, and records `superseded_on_arrival_by = old.id` in the event. `include_history` and `memory_get` expose it; default retrieval excludes it. A higher-ranked record supersedes regardless of event time because an explicit user statement outranks a tool observation. The log retains both timestamps for evaluation.

The ingestor does not use cosine similarity as an equivalence verdict. Opposing short preferences such as "prefers concise answers" and "prefers detailed answers" can have nearby embeddings. Cosine proposes records about other entities for the judge when they meet `ingestion.dedup_candidate_cosine`. The structured entity-and-attribute scan handles records about the same entity before that nearby search.

The retriever excludes superseded records, so a lower-ranked contradiction never hides a confirmed fact, but it is kept and surfaced in `memory_get` so an agent can ask the user.

### 6.4 Reinforcement

```python
def reinforce(record, now):
    record.reinforcements += 1
    record.last_reinforced_at = now
    record.confidence = min(0.99, record.confidence + 0.1)
    if record.status == "provisional":
        record.expires_at = now + timedelta(days=cfg.provisional_ttl_days)
        if record.reinforcements >= cfg.reinforcements_to_confirm:
            record.status = "confirmed"
            record.expires_at = None
```

Reinforcement records an independent later observation of the same claim. The source reference must differ from the record's original reference and every prior reinforcing reference; replaying the same transcript turn records an audit event but does not increase the count. A counted observation adds `0.1` confidence up to `0.99`, refreshes provisional expiry, and promotes a provisional record after `reinforcements_to_confirm` observations. A higher-ranked observation, such as a user statement supporting an inference, also replaces the record's stored `source_kind`, `source_ref`, and evidence with the stronger provenance. The reinforcement event retains the incoming evidence and source reference either way.

### 6.5 Expiry

Retrieval treats `expires_at < now` as ineligible before candidate generation. The stored row, embedding, FTS row, and entity links remain available for `include_history` and audit. `memory-weave expire` changes eligible expired rows to `expired` and writes audit events. Retrieval does not depend on this maintenance command.

### 6.6 Evidence validation

Both write paths call this helper, so explicit writes and extractor candidates use the same evidence standard.

```python
def validate_evidence(session_id, quote, claimed: str, turn_hint: int | None = None) -> EvidenceCheck:
    turns = session_buffer.turns(session_id)
    q = normalize_ws(quote)
    candidates = [turns[turn_hint]] if turn_hint is not None else turns
    hit = next((t for t in candidates if q and q in normalize_ws(t.content)), None)
    if hit is None:
        return EvidenceCheck(False, None, None, "agent_inference", "evidence not found in session")
    supported = {"user": "user_statement", "tool": "tool_result", "assistant": "agent_inference"}[hit.role]
    if rank(claimed) > rank(supported):
        return EvidenceCheck(
            True, hit.turn, hit.role, supported, f"downgraded from {claimed}: quote is from a {hit.role} turn"
        )
    return EvidenceCheck(True, hit.turn, hit.role, claimed, None)
```

The helper normalizes whitespace, then looks for the complete quote in one transcript turn. A paraphrase is not evidence. The quote must meet `ingestion.evidence.min_characters` and `ingestion.evidence.min_words`.

| Supporting turn | Highest supported source kind | Handler action when the request claims more authority |
| --- | --- | --- |
| User turn | `user_statement` | Keep the claim. |
| Tool turn | `tool_result` | Downgrade and record the note. |
| Assistant turn | `agent_inference` | Downgrade and record the note. |
| No matching turn or no session | `agent_inference` | Store no `source_ref` and record why validation failed. |
| No evidence supplied | `agent_inference` | Store no `source_ref`, use `evidence = NULL`, and add the `evidence not provided` note. |

For a matched user or tool quote, the ingestor also calls `judge.entails(evidence, content)`. A score below `ingestion.evidence.entail_floor` downgrades the requested source to `agent_inference`, records `evidence does not support claim`, and stores the score in the event. An `agent_inference` is not entailment-checked.

The caller may choose a lower source kind than the evidence supports. The helper returns the matched turn so the caller can write `source_ref = "session:<id><turn:n>"`. The host writes `system` records through a separate API.

## 7. Vector index

The vector index is an in-memory projection of compatible rows in `embeddings`. It runs exact cosine search against an eligible ID set from the store, so vector search cannot return an unreadable or expired record.

```python
class VectorIndex:
    ids: list[str]  # diagnostic snapshot of record IDs
    pos: dict[str, int]  # diagnostic snapshot of row positions
    live: np.ndarray  # diagnostic snapshot of liveness flags

    def load(self, store): ...  # read all embeddings for the configured model/version
    def refresh(self, store, incremental_reload_max) -> str: ...  # apply another process's delta or rebuild
    def upsert(self, record_id, vec): ...  # append or overwrite a normalized row; grows the live index by doubling
    def remove(self, record_id): ...  # live[pos] = False
    def vector_for(self, record_id) -> np.ndarray | None: ...
    def cosine(self, first_id, second_id) -> float: ...

    def search(
        self, qvec, eligible_ids: set[str], k
    ) -> list[tuple[str, float]]: ...  # builds the matching positions and scores the private normalized matrix under one lock hold
```

| Structure | Holds | Why it exists |
| --- | --- | --- |
| `ids` | Record ID for each matrix row. | Converts a selected row back to a record. |
| `pos` | Matrix row for each record ID. | Builds filters and updates rows without scanning. |
| Internal matrix | One L2-normalized vector per record. | Makes cosine search a matrix multiplication. It is intentionally not exposed because copying it at 50K records costs about 200 MB. |
| `live` | Boolean flag for each matrix row. | Hides deleted records without rebuilding the matrix. |

The store increments `store_meta.records_version` in the same transaction as every record, status, and embedding update. Each row records the version that last affected the index. Before dense retrieval, `VectorIndex.refresh` compares its loaded version with the durable version. It finds changed record IDs through indexed `records.index_version` and `embeddings.index_version` lookups, applies a small delta in place, or reloads the projection when the delta exceeds `embedding.incremental_reload_max`. Two processes using one SQLite file therefore observe each other's writes on the next search without a restart.

The store returns eligible record IDs after scope, lifecycle, type, and time filtering. `VectorIndex.search` converts only those IDs into matrix positions, so selection cost grows with the caller's eligible set rather than every indexed ID. It builds that position mask and runs `live & allowed` under one index-lock hold. A concurrent refresh therefore cannot move a row between filtering and matrix multiplication. The retriever also rejects any fused candidate outside `eligible` before it loads records or writes a log. Retrieval code uses `vector_for` for an isolated record vector and `cosine` for comparisons between indexed records; neither API exposes or copies the full matrix.

For multiple queries, dense search embeds each query and keeps the highest cosine for each record before RRF. At 50K records, a 1,024-dimension `float32` matrix uses about 200 MB. Startup loads those vectors from SQLite in about one second; search uses one matrix multiplication and one partition per query, with a design estimate below 5 ms.

## 8. Ingestion

Ingestion puts memories into the store through two paths. Both paths apply the same permission, evidence, entity, duplicate, contradiction, and lifecycle rules. The only difference is the source of the candidate.

| Path | Starts when | Caller waits | Purpose |
| --- | --- | --- | --- |
| Explicit write | An agent calls `memory_write` during a session. | Yes. | Save a known fact, decision, or procedure now. |
| Session extraction | The adapter calls `on_session_end`. | No. | Review the completed transcript for durable memories and write one session summary. |

### 8.1 Explicit write: `memory_write`

Use this path when the agent has one memory worth saving and can provide the supporting evidence.

#### Steps

1. **Validate the request.** The handler validates the tool arguments and derives the caller's `Principal`.
2. **Choose and authorize the scope.** The requested scope wins. If the request omits it, the handler uses `Scope("user", principal.user_id)`. The host must first grant the agent read and write access to that user scope. Otherwise the handler returns `scope_not_writable`. The only implicit scope is `Scope("agent", f"{agent_id}/{user_id}")`.
3. **Validate evidence.** `validate_evidence` checks the quote against the current session, assigns the supported `source_kind`, and returns the matching turn. The handler stores that turn as `source_ref = "session:<session_id><turn:n>"`. For a claimed user or tool source, it also scores whether the quote entails the content. A score below `ingestion.evidence.entail_floor` downgrades the claim to `agent_inference`. The tool rejects `session_summary`, and the host reserves `system` for its separate API.
4. **Resolve entities.** The handler resolves each entity mention in the requested scope. An ambiguous `about` entity stops the write and returns `entity_ambiguous` with candidate entity IDs. An ambiguous `mentions` entity drops that link and adds a note.
5. **Set the structured subject.** The writer supplies an `attribute` hint, not a subject string. The handler normalizes it and combines it with the resolved primary `about` entity as `<entity_id>/<attribute>`. Semantic and procedural records require an attribute. Without an `about` entity, they are valid only in the principal's user scope, where the handler resolves or creates that principal's `person` entity. Other scopes return `invalid_subject` with `about entity required`. An episodic record may use `<entity_id>/-`; that form is never a supersession key.
6. **Embed the content.** The embedder produces one normalized vector for the record content.
7. **Check for an existing memory.** For semantic and procedural records, the handler checks active records with the same scope, type, and primary entity, then applies the attribute-alias and supersession rules in section 6.3. For all record types, it also searches nearby records about other entities. A nearby record at or above `ingestion.dedup_candidate_cosine` goes to the equivalence judge. A `same` verdict reinforces the existing record instead of inserting a new one.
8. **Write one transaction.** The handler writes the record, embedding, FTS row, entity links, and audit event, then updates the in-memory vector index.
9. **Return the result.** The response contains `record_id`, `status`, `outcome`, `note`, and `timings_ms`.

#### Results

| Result | Meaning |
| --- | --- |
| `created` | The handler stored a new active memory. |
| `reinforced` | The candidate matched an existing memory, so the handler increased its reinforcement count instead. |
| `already_reinforced` | The candidate used evidence that the record already counted, so the handler kept the record unchanged and recorded the duplicate observation. |
| `superseded:<old_id>` | The new record replaced the active record for the same subject. |
| `superseded_on_arrival:<old_id>` | The candidate has the same rank as a newer active fact, so the handler stores it as history without making it active. |
| `conflict:<old_id>` | The handler stored a lower-authority contradiction as provisional and linked both records as conflicts. |
| `scope_not_writable` | The caller cannot write to the requested scope. |
| `entity_ambiguous` | The handler could not safely identify the record's primary entity. This includes the principal's own person entity when more than one person entity in the user scope carries the principal alias, which an entity merge can produce. |
| `invalid_source_kind` | The caller attempted to write a host- or extractor-owned source kind. |
| `invalid_subject` | A semantic or procedural request omitted its attribute (`note` is `attribute normalizes to nothing` when a hint was given but reduced to an empty slug), or an out-of-user scope omitted its primary `about` entity (`note` is `about entity required`). |

When a request has no evidence, the handler writes it as an `agent_inference` with no source reference and returns the `evidence not provided` note. When a quote exists but does not support a direct claim, it writes the same provisional source with the `evidence does not support claim` note. The entailment check runs only while the claim is still direct: a claim already downgraded by its turn role is not scored again and carries one note. The record remains provisional and expires unless later, independent evidence supports it.

The handler estimates 25 ms for one warm MPS embedding, under 5 ms for one or two index searches, and under 5 ms for the database transaction.

The response records the complete timing sequence: `permission`, `evidence`, `entities`, `embed`, `dedup_search`, `judge`, `supersession`, `persistence`, `event_log`, `transaction`, `index_update`, and `total`. `persistence` covers record, embedding, FTS, entity-link, and conflict writes. `event_log` measures the append-only audit insertion, `transaction` measures the SQLite commit after the transaction context exits, and `index_update` measures the post-commit in-memory vector update. The event cannot include its own insertion, commit, or index-update duration, so its payload records the completed stages and names the remaining stages in `timings_pending`. Benchmarks use the returned `WriteResult` timings for complete write latency. `judge` measures the NLI cross-encoder from section 6.3.

### 8.2 Session extraction

`on_session_end` schedules extraction in a worker thread or a separate process. The host continues without waiting for it.

#### Steps

1. **Read the session.** The worker loads the session turns. A session with fewer than two turns skips candidate extraction and writes the summary only.
2. **Build extraction context.** The worker gives the extractor the caller's principal, entity aliases visible to that caller, active subjects in writable scopes, and the extractor prompt version.
3. **Request candidates.** The extractor returns `ExtractionOutput`, which contains candidate records and one session summary. The worker considers at most `ingestion.extraction_max_candidates` candidates.
4. **Validate each candidate.** The worker validates evidence against the declared turn, resolves entities, and then reuses the explicit-write logic from steps 5 through 8 above. It records accepted, reinforced, superseded, conflicting, and rejected candidates.
5. **Write the session summary.** The worker writes one episodic record from `out.summary` with the principal's person entity as `subject_entity_id`, `attribute = "-"`, `event_at = session.ended_at`, `source_kind = "session_summary"`, and `source_ref = "session:<session_id>"`. Policy assigns the summary `confirmed` status, confidence `0.8`, and no expiry. The worker records other summary entities as `mentions`; it drops ambiguous aliases.
6. **Finish the run.** The worker sets `sessions.extracted_at` and appends one `extraction.run` event.

#### Candidate validation

| Condition | Worker action |
| --- | --- |
| Evidence quote is absent from the named turn. | Reject the candidate with `evidence_not_found`. |
| Candidate claims `user_statement` but quotes an assistant turn. | Downgrade the source to `agent_inference`, using the same rule as an explicit write. |
| Primary (`about`) entity is ambiguous. | Reject the candidate and record the candidate entity IDs. |
| Mention-only entity is ambiguous. | Write the record without that link and record the ambiguity. |
| Candidate passes validation. | Reuse the explicit-write subject, duplicate, contradiction, persistence, index, and event steps. |

The `extraction.run` event records counts for `proposed`, `written`, `reinforced`, `superseded`, `rejected`, and rejection reasons. Its `timings_ms` payload contains `transcript_prep`, `extractor_model`, `validation`, `dedup_and_contradiction`, per-record `writes`, `summary_write`, and `total`. Rejected candidate content and reasons stay in this event so evaluation can measure extractor precision separately from validator precision.

### 8.3 Extractor contract

The extractor receives numbered transcript turns, readable entity aliases, active writable subjects, and a prompt version. It returns JSON matching `ExtractionOutput`.

| Requirement | Extractor behavior |
| --- | --- |
| Memory value | Propose a candidate only when it would change a future action. |
| Candidate scope | Put one fact in each candidate. Do not combine facts. |
| Content | Write one standalone declarative sentence that remains clear without the transcript. |
| Evidence | Copy a verbatim quote from one specified turn. Do not paraphrase. |
| Attribute | Reuse an existing attribute for the same fact when one is known. Otherwise propose a short attribute hint. The ingestor resolves the primary entity and normalizes the final key. |
| Source kind | Use `user_statement` for the user's own words. Use `agent_inference` for an assistant conclusion. |
| Third-party facts | Propose them only when the user stated them. |
| Episodes | Capture decisions, rationale, outcomes, and failures. Put routine progress in the session summary. |

The repository versions the extractor prompt. Each `extraction.run` event records that prompt version.

## 9. Entity resolution

Entity resolution links a record to the person, project, repository, or product it concerns. It uses exact, scope-aware alias lookup because a wrong link can expose unrelated memory.

```python
@dataclass
class Resolution:
    mention: EntityMention
    entity: Entity | None
    outcome: Literal["explicit", "resolved", "created", "ambiguous"]
    candidates: list[str]                                            # entity ids, set when ambiguous

def resolve_entities(mentions, scope, principal) -> list[Resolution]:
    readable = readable_scopes(principal)
    out = []
    for m in mentions:
        if m.entity_id:
            e = store.entity(m.entity_id)
            assert e and e.scope in readable                          # else error "entity_not_readable"
            e = follow_merges(e)
            assert e.kind == m.kind                                   # else error "entity_kind_mismatch"
            out.append(Resolution(m, e, "explicit", [])); continue
        norm = normalize(m.text)                                     # lower, collapse ws, strip diacritics
        hits = store.entities_by_alias(norm, kinds=[m.kind], scopes=readable, status in (provisional, confirmed))
        if len(hits) == 1:
            out.append(Resolution(m, hits[0], "resolved", []))
        elif len(hits) > 1:
            out.append(Resolution(m, None, "ambiguous", [h.id for h in hits]))   # never pick one
        else:
            assert scope in writable_scopes(principal)
            e = store.create_entity(kind=m.kind, canonical=normalize_ws(m.text), scope=scope, status="provisional")
            store.add_alias(e.id, norm); append event entity.created
            out.append(Resolution(m, e, "created", []))
    return out
```

| Resolution outcome | Meaning | Write behavior |
| --- | --- | --- |
| `explicit` | The request supplied a readable `entity_id`. | Link to the entity after following any merge. |
| `resolved` | One readable entity owns the normalized alias. | Link to that entity. |
| `created` | No readable entity owns the alias. | Create a provisional entity in the writer's scope, add the alias, and link it. |
| `ambiguous` | Several readable entities own the alias. | Do not choose one. |

Ambiguity has different effects for the two link roles.

| Role | On ambiguity |
| --- | --- |
| `about` | Stop the explicit write with `entity_ambiguous`. Extraction rejects the candidate and logs the entity IDs. |
| `mentions` | Write the record without that link and add `ambiguous_mention` to the event. |

The resolver only returns an ambiguity in `Resolution`; it does not write an event. The ingestor logs each ambiguity as `ambiguous_alias` after its write transaction succeeds or rolls back, so the audit event survives a rejected `about` mention. Repeated ambiguity requires a more specific alias or a manual merge.

An agent with write access to both scopes, or the CLI, performs `memory_revise(entity_id=..., merge_into=...)`. Both entities must have the same kind. The merge sets `status=merged` and `merged_into`, repoints `record_entities`, and unions aliases. Alias lookup follows `merged_into` to the surviving entity.

The first `about` mention is the primary entity. When an episodic record has no `about` mention and omits its subject, the handler uses the principal user as the primary entity for its `/-` subject.

### 9.1 The principal entity

The principal's person entity lives in `user:<user_id>` and carries `normalize_alias(user_id)` as its first alias. A subject-less semantic write resolves to it. When `user_id` is an opaque identifier, an `about` mention such as "Aditya" would otherwise create a second person entity in the same scope, and facts about one human would split across two subjects that never meet in the attribute scan. The host closes that gap at provisioning time: `MemoryHost.provision_user(user_id, aliases)` creates the principal entity if absent and attaches the display names people use for that user. It refuses an alias that already names a different person entity in the scope, and it appends a `principal.provisioned` event. Adapters call it alongside `grant` before the first session for a user.

## 10. Retrieval pipeline

`memory_search` finds useful memories without leaking ineligible records or returning weak matches. It filters before ranking, combines three retrieval methods, and records enough detail to explain the final response.

| Stage | Input and output | Purpose |
| --- | --- | --- |
| 1. Rewrite | Raw queries → standalone queries. | Resolve references such as pronouns when the feature is enabled. |
| 2. Resolve scopes | `Principal` → readable scopes. | Define the caller's access boundary. |
| 3. Hard filter | Scopes and request filters → eligible record IDs. | Exclude unreadable, expired, deleted, and unwanted record types before ranking. |
| 4. Generate candidates | Eligible IDs → dense, lexical, and entity rankings. | Find semantic matches, exact terms, and named subjects. |
| 5. Fuse | Three rankings → one ranking. | Use RRF because cosine, BM25, and entity matches use different score scales. |
| 6. Freshness | Fused rankings → recency-adjusted rankings. | Reduce the rank of old episodic memories. |
| 7. Gate | Candidates → relevant candidates. | Return an empty response when no candidate has enough evidence. |
| 8. Deduplicate | Relevant candidates → distinct candidates. | Avoid sending near-identical memories to the agent. |
| 9. Rerank | Distinct candidates → reordered candidates. | Use the optional cross-encoder on a small shortlist. |
| 10. Budget | Ranked candidates → bounded candidates. | Respect `k` and the token budget. |
| 11. Explain and log | Bounded candidates → `SearchResponse` and `search_log`. | Return concise results and preserve the full decision trail. |

### 10.0 Query rewriting (optional, off by default)

The agent supplies raw queries. With `retrieval.rewrite.enabled`, a hosted model rewrites them into standalone search queries from the raw queries and host-supplied `context`. The rewriter cannot access candidate records, the store, or prior search results.

```python
def rewrite_stage(req: SearchRequest) -> tuple[list[str], str]:
    if not cfg.retrieval.rewrite.enabled:
        return req.queries, "disabled"
    ctx = (req.context or "")[: cfg.retrieval.rewrite.max_context_chars]
    try:
        out = rewriter.rewrite(req.queries, ctx)  # one structured-output call, timeout_ms
    except (TimeoutError, RewriteError):
        return req.queries, "failed"  # raw queries proceed; nothing else changes
    return (out.queries, "applied") if out.queries != req.queries else (req.queries, "unchanged")
```

Rules:

- The rewriter returns the same number of queries it received, each a standalone phrase that names its subject. It may expand a pronoun or a "the second one" reference using `context`; it may not invent subjects absent from both inputs.
- The adapter supplies current-turn context: by default, the last user and assistant turns, truncated. With no context, the rewriter uses the raw queries alone.
- Dense and lexical generators receive rewritten queries. The retriever leaves entity hints unchanged.
- The response and log retain raw and rewritten queries, and each `Explanation` carries both, so evaluation can attribute a hit or miss to rewriting.
- On rewrite failure, the search proceeds on raw queries and records `failed` in the log.

The deployment leaves rewriting off by default because the hosted call sits on the hot path. The timer records rewrite latency as its own stage.

### 10.1 Pipeline orchestration

The handler below runs the stages in the table. It creates `search_id` first, records timing after each stage, and writes one `search_log` row before it returns.

```python
def memory_search(principal, req: SearchRequest) -> SearchResponse:
    t = Timer(warm=embedder.is_loaded and vector_index.is_loaded)
    search_id = uuid7()  # generated first; it is the search_log primary key and appears in the response
    reranked = None  # stays None when the reranker is disabled; the log column is NULL

    queries, rewrite_status = rewrite_stage(req)  # 10.0
    t.mark("rewrite")

    scopes = readable_scopes(store, principal.agent_id, principal.user_id)
    t.mark("scopes")

    eligible = store.eligible_ids(
        scopes, req.types, req.since, req.until, req.include_history, now
    )  # SQL, returns set[str]
    t.mark("filter")

    index_refresh = vector_index.refresh(store, cfg.embedding.incremental_reload_max)
    t.mark("index_refresh")

    qvecs = embedder.embed_queries(queries)
    t.mark("embed")

    # Run generators sequentially until the concurrency benchmark justifies a thread pool.
    dense = {}  # record_id -> max cosine
    for qv in qvecs:
        for rid, cos in vector_index.search(qv, eligible, cfg.per_generator_k):
            dense[rid] = max(dense.get(rid, -1), cos)
    dense = ranked(dense)[: cfg.per_generator_k]  # cap the post-union list, not each query alone
    t.mark("dense")

    alias_matches = resolve_entity_aliases(req.entities, queries, scopes)
    lexical = lexical_search(queries, eligible, cfg.per_generator_k, entity_aliases=alias_matches)  # 10.2
    t.mark("lexical")

    entity_hits = entity_search(alias_matches, eligible, limit=cfg.per_generator_k)
    t.mark("entity")

    candidates = rrf([dense, ranked(lexical), ranked(entity_hits)], k=cfg.rrf_k)  # 10.3, sets rrf_score and fused_rank
    candidates = [candidate for candidate in candidates if candidate.record_id in eligible]  # hard isolation invariant
    t.mark("fuse")

    candidates = apply_freshness(candidates, store, now)  # 10.4, sets freshness_multiplier and score
    t.mark("freshness")

    kept, gated_out = gate(candidates, queries)  # 10.5, sets gate_reason
    t.mark("gate")

    kept, deduped_out = collapse_duplicates(kept, vector_index, cfg.dedup_cosine)  # 10.6
    t.mark("dedup")

    reranked_out = []
    if cfg.reranker.enabled:
        reranked_out.extend(record_reranker_shortlist_omissions(kept[cfg.reranker.candidates :]))
        kept, reranked = rerank(
            kept[: cfg.reranker.candidates], queries
        )  # 10.7, scores every query, keeps the max per record
        kept, below_floor = apply_reranker_floor(kept, cfg.reranker.floor)
        reranked_out.extend(below_floor)
    t.mark("rerank")

    chosen, budget_out = fill_budget_with_authority_pairs(kept, req.k, cfg.token_budget)  # 10.8
    t.mark("budget")

    results, empty_reason = explain(
        chosen, candidates, req.queries, queries, rewrite_status, gated_out, deduped_out
    )  # 10.9
    t.mark("explain")

    log_search(
        principal,
        req,
        queries,
        rewrite_status,
        scopes,
        dense,
        lexical,
        entity_hits,
        candidates,
        gated_out,
        deduped_out,
        reranked,
        reranked_out,
        budget_out,
        results,
        empty_reason,
        cfg.flags(),
        t,
    )
    t.mark("log")
    return SearchResponse(
        search_id,
        req.queries,
        queries if rewrite_status == "applied" else None,
        rewrite_status,
        results,
        empty_reason,
        t.as_dict(),
    )
```

The returned response includes the measured `log` duration. The row cannot contain the duration of its own SQLite insert, so `search_log.timings_ms` ends at `explain`; this avoids calling an unmeasured interval a log duration. The row records `index_refresh` in `config_flags` as `loaded`, `unchanged`, `delta`, or `reloaded`.

### 10.2 Lexical search

Lexical search handles identifiers, error messages, commands, and exact phrases.

1. Split raw whitespace tokens before FTS tokenization so identifier detection preserves values such as `bge-m3` and `deploy.yml`.
2. Normalize with FTS5-compatible `unicode61` token rules, fold typographic apostrophes, reduce contractions and possessives to their stem ("aditya's" to "aditya", "don't" to "do", "I'm" to "i"), discard terms shorter than two characters, and remove the frequency-based English stopword list while preserving identifiers and proper nouns.
3. Join remaining terms with `OR` and query FTS5 with `bm25(records_fts, 0.0, 1.0, 2.0, 3.0)`. The leading `0.0` skips the unindexed `record_id` column, leaving weights 1 for `content`, 2 for `subject`, and 3 for `aliases`.
4. Load `eligible` into a temporary SQLite table, join it to FTS5, and apply `per_generator_k` after that join. Ineligible rows therefore cannot crowd eligible rows out of the result limit.
5. For each returned record, calculate term coverage separately for every query and retain the best coverage. A single-token term matches anywhere in the indexed fields. A multi-token term such as `bge-m3` or `follow-up` matches only where its tokens appear contiguously in one field, so a record that merely contains the parts somewhere does not count. Return the exact matched terms, with `is_identifier` and `is_entity_alias` flags, plus the total terms from that winning query for the relevance gate.

The entity generator normalizes apostrophes and removes possessive or contraction endings before it scans query text for aliases. It then uses every n-gram up to `retrieval.max_alias_tokens` tokens long (default 4) and looks them up in batches. The entity IDs enter a temporary table before `records_for_entities` applies its limit. Lexical search receives the same resolved aliases, so it can mark a matching term as an entity alias for the gate.

The BM25 weights are 1 for `content`, 2 for `subject`, and 3 for `aliases`.

### 10.3 Reciprocal rank fusion

```python
def rrf(rankings: list[list[str]], k: int) -> dict[str, float]:
    score = defaultdict(float)
    for ranking in rankings:
        for rank, rid in enumerate(ranking, start=1):
            score[rid] += 1.0 / (k + rank)
    return dict(sorted(score.items(), key=lambda kv: -kv[1]))
```

An empty generator adds no RRF score. The remaining generators still produce a valid ranking.

### 10.4 Freshness

The retriever adjusts episodic records after RRF:

```text
age_days = (now - event_at).days
multiplier = max(freshness.floor, 0.5 ** (age_days / freshness.episodic_half_life_days))
score = rrf_score * multiplier
```

Semantic and procedural records retain their fused score because supersession handles stale current facts. A request with `since` or `until` skips freshness because the caller already supplied a time range.

### 10.5 Gate

The gate decides whether any candidate is strong enough to reach the agent, and it drops weak candidates even when stronger ones pass. It is the component that makes an empty result common rather than exceptional, and it is what makes host-issued searches (section 14.1) safe. It runs in three steps.

**Step 1, absolute floors, per candidate.** Keep a candidate when any one signal holds:

| Signal | Passing condition |
| --- | --- |
| Entity | The candidate came from an exact entity match. |
| Dense | Cosine is at least `cfg.gate.dense_floor[record.type]`. Floors are per record type because long episodic summaries score lower against short queries than short semantic facts do. |
| Lexical | `matched_terms / total_terms` is at least `cfg.gate.lexical_min_term_fraction`, and either `matched_terms` is at least `cfg.gate.lexical_min_matched_terms` or one matched term is an entity alias or an identifier token. |

An identifier token contains a digit, underscore, dot, or slash, or mixes case inside the token: `ERR42`, `bge-m3`, `deploy.yml`, `camelCase`. A hyphen on its own does not qualify, so `follow-up`, `e-mail`, and `and/or`-style compounds of plain words stay plain words. Such tokens are precise enough to pass on their own. A plain word is not, which is why a one-word query such as "deployment" cannot admit every record that mentions deployment.

**Step 2, relative floor, within each channel count.** Group step-1 survivors by the number of contributing channels, then let `top` be the highest fused score in each group. Drop a survivor whose fused score is below `cfg.gate.relative_floor * top` for its own group, except entity hits. With one survivor in a group this is a no-op. This keeps the useful comparison between records with the same evidence shape while preventing a two- or three-channel record from eliminating every strong single-channel record.

For `trigger = auto`, the gate uses `cfg.gate.auto` instead of the normal settings and rejects its excluded source kinds before evaluating the floors. The default excludes `session_summary` from host-issued retrieval. The gate still measures relevance rather than usefulness; the ordinary-turn benchmark decides whether the stricter auto settings are sufficient.

**Step 3, the empty decision.** If nothing survives, `results` is empty and `empty_reason` names the best candidate's missed floors, for example `"best dense 0.38 < 0.45 (semantic); best lexical 1/4 terms, 1 matched < 2; no entity match"`. Every dropped candidate carries a `gate_reason` naming the step and floor that dropped it, so the log can be replayed offline with different floors.

**Calibration.** The floors depend on the embedding model and are re-calibrated on every embedder change. Calibration uses three query classes, and the third is the one most systems never test:

| Class | Query | Correct outcome |
| --- | --- | --- |
| Evidence present | A question whose answer is in the store. | Return the evidence record. |
| Evidence absent | A question with no supporting record anywhere in the store. | Return nothing. |
| Ordinary turn | A conversational turn sampled from a real transcript, with a populated store that does not bear on it. | Return nothing, or at most a record the judge deems useful. |

Because `search_log` keeps every candidate's scores, floor sweeps run offline against logged searches without re-executing them. Pick per-type dense floors and the relative floor that maximize F1 on classes 1 and 2, subject to an injection rate on class 3 below the target the benchmark sets. Record the chosen values with `embedding.version` in `config.py`.

**What the gate does not do.** It judges relevance, not need. A record about the user's coffee habit passes on any coffee query, and whether that is personalization or pollution depends on the task. In `tool_only` mode the model absorbs that judgement by deciding to search. In `auto` and `hybrid` modes, the ordinary-turn class is how it is measured, and the reranker, which is better calibrated than cosine, is the next lever if the class-3 rate is too high.

### 10.6 Duplicate collapse

Walk candidates in fused order. Compare each candidate with records already accepted through `VectorIndex.cosine`, which reads the two private rows under the index lock. If cosine is at least `retrieval.dedup_cosine`, drop the later candidate and log `(dropped_id, kept_id, cosine)`. This removes near-duplicates that entered through different scopes or types without copying the full index matrix.

### 10.7 Reranker (optional, off by default)

The reranker runs after gating and duplicate collapse. It receives at most `reranker.candidates` records and does not score the full store.

For each survivor, it scores every query-record pair, keeps the record's maximum score, and sorts the shortlist by that score. With three queries and 30 records, it evaluates at most 90 pairs. The log records rank before reranking, rank after reranking, score, and the query that produced the score. Records outside the shortlist and records below `reranker.floor` go in `reranked_out`, not `budget_out`, so the reason for each omission remains unambiguous.

`reranker.floor` drops weak reranked candidates. When reranking is enabled, that floor becomes the final gate; dense and lexical floors only create the shortlist. `load_config` rejects an enabled reranker until evaluation supplies a floor.

`reranker.budget_mean_ms` starts at 100 ms mean for 30 candidates on the target laptop. The benchmark reports p50 and p95 by candidate count and hardware, then updates the HLD latency table. Make reranking the default after evaluation improves final context and downstream task outcomes.

### 10.8 Budget fill

Budget fill walks the ranked survivors. `k` comes from `req.k` or `retrieval.default_k`.

Each result costs `len(content) // 4` tokens plus 30 tokens for its envelope line. The retriever stops after `k` primary results or after the token budget. It skips a record that does not fit and tries the next record; it does not truncate content. When a provisional conflict needs an authority counterpart, the two records are admitted or rejected together. The counterpart does not consume a primary `k` slot, but both records count toward the token budget.

### 10.9 Explanations and result formatting

`explain()` builds one `Explanation` for each returned record and one response-level `empty_reason` when the result is empty. The structured payload and `search_log` keep every `Explanation` field. When the gate rejected every candidate, `empty_reason` names the best missed floor. When a later stage removed all candidates, it names that later stage instead, such as the reranker floor or token budget. The agent sees the concise `summary` line.

Each result is rendered to the tool result as one block:

```
[mem_01J...] semantic · confirmed · user_statement · event 2026-08-12 · scope user:aditya
Prefers concise technical explanations without preamble.
matched: dense 0.71 (rank 2), lexical 3/3 (rank 1); fused rank 1; passed gate on dense
```

The response header identifies a rewritten query, for example `searched for: "user's preferred answer length" (rewritten from "what does he prefer")`. An empty response includes `empty_reason` in the header.

The rendered block uses a short ID prefix. The structured payload retains the full ID. When a returned record has unresolved conflicts, `Explanation.conflicts_with` lists only conflicting records in the caller's eligible set and the rendered block adds `conflicts with <id>`. If a provisional conflict is returned and an eligible confirmed or higher-authority counterpart exists, the retriever keeps the counterpart's real candidate evidence when available and inserts it immediately before the provisional record. The pair is admitted or rejected together by the budget. Adapters render this block; the payload shape remains fixed.

## 11. Tool surface

The memory layer exposes five framework-neutral tools. They use plain JSON Schema, so each adapter can register the same contract.

| Tool | Use it when | Result | Important boundary |
| --- | --- | --- | --- |
| `memory_search` | The current task may depend on a prior preference, decision, or session. | Relevant active memories, or an explicit empty result. | It never returns a record outside the caller's readable scopes. |
| `memory_get` | The agent needs a full record, its source, or its history. | Full records, including conflicts and superseded lineage. | Call it only with IDs already known to the agent. |
| `memory_write` | A current-session fact, preference, decision, or procedure should affect future work. | A new, reinforced, superseding, or conflicting memory. | Evidence must come from the current session. |
| `memory_revise` | A known record needs confirmation, replacement, or expiry. | An updated lifecycle state and audit event. | The caller needs write access to the record scope. |
| `memory_forget` | A user asks to remove a known memory from normal use. | A tombstone and audit event. | Durable content erasure remains an admin operation. |

The agent should use search before work that may depend on prior context. It should use write only for information that changes a future action. Every handler derives the `Principal` from the adapter; tool input never supplies an agent identity.

### `memory_search`

```json
{
  "name": "memory_search",
  "description": "Search long-term memory for facts, past decisions, and procedures relevant to the current task. Call this before acting on anything that may depend on the user's preferences, prior decisions, or earlier sessions. Returns nothing when nothing relevant is stored; treat an empty result as 'no memory', not as an error.",
  "input_schema": {
    "type": "object",
    "properties": {
      "queries":  {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3,
                   "description": "One to three standalone search phrases. Write them as if to a colleague: name the subject, not the pronoun."},
      "types":    {"type": "array", "items": {"enum": ["semantic", "episodic", "procedural"]}},
      "entities": {"type": "array", "items": {"type": "string"}, "description": "Names of people, projects, or repos the task is about."},
      "since":    {"type": "string", "format": "date-time"},
      "until":    {"type": "string", "format": "date-time"},
      "k":        {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
      "include_history": {"type": "boolean", "default": false}
    },
    "required": ["queries"]
  }
}
```

The schema intentionally omits `context`. The adapter attaches a small current-turn context to `SearchRequest` before the handler runs; refer to section 14. Letting agents supply context would encourage transcript-sized search calls.

### `memory_get`

Input: `{"ids": [string]}`.

It returns the full records, including superseded lineage and conflicts. An agent uses it to inspect the source or show a disagreement to the user.

### `memory_write`

```json
{
  "type": {"enum": ["semantic", "episodic", "procedural"]},
  "content": {"type": "string", "maxLength": 1000},
  "attribute": {"type": "string", "description": "Current-fact attribute such as 'timezone'. Required for semantic and procedural memories. Any entity prefix is ignored."},
  "scope": {"type": "object", "properties": {"kind": {...}, "id": {...}}, "description": "Defaults to the principal user's scope. The host must provision a grant for it."},
  "source_kind": {"enum": ["user_statement", "tool_result", "agent_inference"]},
  "evidence": {"type": "string", "description": "Verbatim quote from this session. Required for user_statement."},
  "event_at": {"type": "string", "format": "date-time"},
  "entities": {"type": "array", "items": {"type": "object", "properties": {"kind": {...}, "name": {...}, "role": {"enum": ["about", "mentions"]},
               "entity_id": {"type": "string", "description": "Pass this when a previous call returned entity_ambiguous, or when memory_get showed you the exact entity."}}}},
  "tags": {"type": "array", "items": {"type": "string"}}
}
```

The description tells the agent to write one fact per call and only information that would change a future action. It cannot write `system` or `session_summary`: the host and extractor own those source kinds, and the input enum excludes them.

When an `about` entity is ambiguous, the error returns the candidate IDs, kinds, canonical names, and scopes. The agent either retries with one `entity_id` or asks the user to disambiguate.

### `memory_revise`

Input: `{"id": string, "content"?: string, "action": "confirm" | "supersede" | "expire", "reason": string}`.

- `supersede` requires `content` and creates a new record with `supersedes_id`.
- `confirm` promotes a provisional record to `confirmed`; the principal must have write access to its scope.
- `expire` sets the record status to `expired`.

Each action appends an event with the reason. Entity merge uses the same tool with a separate argument shape: `{"entity_id", "merge_into", "reason"}`.

### `memory_forget`

Input: `{"id": string, "reason": string}`.

The handler sets the record to `deleted`, retains a tombstone, removes its FTS row, and marks its in-memory vector dead. Durable content and its embedding remain until a controlled erase removes them. Content erasure is an admin CLI operation (`memory-weave erase <id>`), not an agent tool: a person must control that irreversible step.

## 12. Session buffer and ingestion hooks

The session buffer is the short-term transcript used to verify evidence and to extract memories after a session. It is not the retrieval index. The adapter stores every turn as it arrives so `memory_write` and the extractor see the same text.

| Hook | When the adapter calls it | What the memory layer does |
| --- | --- | --- |
| `on_session_start` | A new conversation or task begins. | Creates a session tied to the derived principal. |
| `on_turn` | A user, assistant, or tool result is available. | Appends a numbered turn. Tool turns contain the result text, not the tool call. |
| `on_session_end` | The host explicitly closes the conversation or task. | Marks the session complete and schedules asynchronous extraction. |

```python
def on_session_start(principal):
    store.create_session(...)


def on_turn(principal, role, content):
    store.append_turn(...)  # tool turns store the tool result text, not the call


def on_session_end(principal):
    store.end_session(...)
    schedule(extract_session, principal.session_id)
```

If a framework cannot signal session end reliably, the adapter uses a 30-minute idle timeout. A later turn starts a new session. Every `source_ref` includes the session ID, so a split affects summary quality but does not make evidence point to the wrong transcript.

## 13. Memory-use policy text

The adapter adds the following policy to the agent's prompt prefix. Keep the wording short and stable so prompt-prefix caching remains effective.

> You have long-term memory available through tools. Before acting on anything that could depend on the user's preferences, earlier decisions, or previous sessions, call `memory_search` with one to three specific phrases. Do not search for general knowledge or for facts already visible in this conversation. When results come back, check their status, source, and date before relying on them; a provisional or old record may be wrong, and you can ask the user. When the user states a preference, a fact about themselves, or a decision, save it with `memory_write` and quote their words as evidence. Save decisions you make together as episodic records with the reason. Do not save guesses as facts.

In `auto` and `hybrid` modes (section 14.1) the adapter appends one more sentence pair, since the model needs to know that some memory arrives without asking:

> Relevant memories may also appear automatically before you answer, marked as recalled memory. Treat them exactly like search results: check their status, source, and date, and use `memory_search` yourself for anything more specific.

In `auto` mode the sentence about calling `memory_search` is removed, because the tool is not registered.

The policy text is versioned with the extractor prompt. Evaluation records both versions on every run, so a behavior change can be tied to the instructions that produced it.

## 14. Adapters

Adapters translate framework state into the framework-neutral types and hooks defined above. They do not change policy, ingestion, or retrieval behavior.

| Concern | Deep Agents | CrewAI |
| --- | --- | --- |
| Tool registration | Register the five JSON schemas as LangChain tools. | Wrap each handler in a `BaseTool` subclass. |
| Principal | Read `agent_id`, `user_id`, and `thread_id` (the session ID) from `configurable` run config. | Use the agent role slug for `agent_id`; the host supplies user ID and session ID through crew inputs. |
| Turn capture | Record each human and AI message with callbacks. | Record the task description as the first user turn and each step output as an assistant turn. |
| Session end | The host closes explicitly, or the idle timeout closes it. | The same explicit-close or idle-timeout path. |
| Search context | Last user turn plus last assistant turn, truncated to `retrieval.rewrite.max_context_chars`. | Task description plus most recent step output, truncated to the same limit. |
| Result rendering | Join the section 10.9 blocks with blank lines and add a one-line count and search header. | Use the same output format. |

CrewAI produces coarser dialogue than Deep Agents because it exposes step output rather than per-turn conversation. The experiment measures the effect of that difference.

Both adapters attach current-turn context to every search, even while rewriting is disabled. That avoids an adapter change when the flag is enabled later. If a framework needs framework-specific state in `records`, change the core contract instead of adding adapter-only fields.

### 14.1 Trigger policy

`retrieval.trigger.mode` decides who calls `memory_search`. The adapter owns this decision. The retrieval pipeline, the gate, the explanations, and the log are the same in every mode; only the caller changes.

| Mode | Who searches | Tools registered for the model | Intended use |
| --- | --- | --- | --- |
| `tool_only` | The model, when it decides to. | All five. | The default. Right for task agents whose work signals when memory matters. |
| `auto` | The host, once per user turn (never on assistant or tool turns). | `memory_get`, `memory_write`, `memory_revise`, `memory_forget`. `memory_search` is not registered. | An experimental control that isolates the host trigger from the model's own searching. |
| `hybrid` | The host once per user turn, and the model whenever it decides to. | All five. | The production candidate for user-facing assistants, once the gate meets the benchmark's ordinary-turn target. |

`hybrid` is not a blend of two triggers on one search. It is two different searches with different callers. The host's search once per user turn catches the silently relevant cases the model would never think to search for, such as a preference that should shape the answer. The model's own searches cover what the raw user turn cannot surface: a targeted follow-up, a time window, an entity hint, or a `memory_get` to inspect provenance before trusting a record.

**Host-issued search.** Before the model call that follows a new user turn in `auto` or `hybrid` mode, the adapter:

1. Takes the new user turn. If its whitespace-normalized length is below `trigger.auto_min_query_chars`, skips the search and logs an `events` row of kind `trigger.skipped` with the reason.
2. Builds `SearchRequest(queries=[user turn text], context=<last user and assistant turns>, k=trigger.auto_k, trigger="auto")` with no type, time, or entity filters, and calls the same handler the tool uses. The rewrite stage applies when enabled, and it matters more here than for model-written queries, because a raw user turn is a poor query.
3. If the response is non-empty, appends it to the conversation as a tool-result-shaped message, rendered with section 10.9's format under the header `recalled memory`. In frameworks that require a tool call to precede a tool result, the adapter appends a synthetic `memory_search` call and its result as a pair. If the response is empty, appends nothing.
4. Never edits the prompt prefix or any earlier message. Appending keeps provider-side prefix caching intact; the recalled block simply becomes part of the history from that turn on, exactly as a model-issued tool result would.

One host-issued search per user turn, never on assistant or tool turns. The `search_log` row records `trigger = 'auto'`, so every metric in the benchmark can be split by who asked.

**Why the gate is the precondition.** Systems that inject memory on every turn pollute because they inject top-k unconditionally. A host-issued search here is subject to the same three-step gate as any other, and the expected outcome on most turns is empty. The benchmark's ordinary-turn class (section 10.5) measures how often that expectation fails. `hybrid` becomes the recommended default only when that rate is acceptably low and accuracy on the memory-needed cases rises; until then `tool_only` stays the default, which is why it is the initial value.

**Adapter obligations by mode.**

| Concern | Deep Agents | CrewAI |
| --- | --- | --- |
| Where the host search runs | A pre-model hook on the agent graph, before each model call that follows a human message. | Before each task starts and before each step that follows new task input. |
| Appending the recalled block | Append a tool-call and tool-result message pair to the state's message list. | Append the rendered block to the task context passed into the step. |
| Tool registration | Register all five in `tool_only` and `hybrid`; omit `memory_search` in `auto`. | Same rule, applied to the `BaseTool` wrappers. |
| Policy text | Section 13's base text, plus the recalled-memory sentences in `auto` and `hybrid`. | Same. |

## 15. Latency accounting

This budget describes warm `memory_search` on an Apple M-series laptop with two queries, 50K records, and the reranker disabled. It is a design target, not a measured benchmark result.

| Stage                        | Expected                        | Dominant cost                                                                     |
| ---------------------------- | ------------------------------- | --------------------------------------------------------------------------------- |
| rewrite                      | 0 ms off; 300 to 800 ms on      | A hosted structured-output call. This is why the flag is off by default.          |
| scopes, filter               | 3 ms                            | One indexed SQL query returning ids; building the mask.                           |
| embed                        | 20 to 40 ms                     | bge-m3 forward pass for two short texts, warm.                                    |
| dense                        | 2 to 5 ms                       | Two matmuls over the matrix.                                                      |
| lexical                      | 3 to 8 ms                       | Two FTS5 queries, intersection.                                                   |
| entity                       | 1 to 3 ms                       | Alias lookup plus an indexed join.                                                |
| fuse, freshness, gate, dedup | under 2 ms                      | Python over at most 90 candidates.                                                |
| rerank                       | 0 ms off; 100 ms mean budget on | Cross-encoder over at most 30 survivors. Measured p50 and p95 replace the budget. |
| budget, explain              | under 1 ms                      | Building at most 20 explanation objects.                                          |
| log                          | 1 to 2 ms                       | One insert with JSON columns.                                                     |
| total, flags off             | 35 to 65 ms                     |                                                                                   |

With rewriting and reranking off, embedding dominates latency. Keep the model loaded and cache exact-string query embeddings in the bounded LRU defined by `embedding.query_cache_entries`. A smaller query-side encoder is out of scope.

`memory_write` has the same embedding cost plus one small index search and one transaction. It can also pay NLI-judge cost when it finds a possible duplicate or contradiction.

### Benchmark instrumentation

Use the same `Timer` for search, writes, and extraction. This makes timings comparable across paths.

| Work | Where timings are stored | Required benchmark context |
| --- | --- | --- |
| Search | `search_log.timings_ms` | Corpus size, generator counts, models and versions, flags, hardware, and cold/warm state. |
| Explicit write | Write event payload | The same model, flag, hardware, and cold/warm context. |
| Session extraction | `extraction.run` event payload | Transcript size, candidate count, model versions, and the same runtime context. |

The `warm` column separates the first cold call from later calls. The harness reports dense, lexical, entity, rewrite, and reranker distributions separately from end-to-end totals. It runs search at several store sizes and reranking at several candidate counts.

## 16. Operations and CLI

The CLI supports maintenance, debugging, and reproducible evaluation. It is not an alternate policy path: commands use the same store and lifecycle rules as the runtime.

| Command | Purpose |
| --- | --- |
| `memory-weave migrate` | Apply forward-only schema migrations and list any legacy record migration 2 could not map to one primary entity and attribute. Exits with status 2 while any remain; `--expire-unmapped` retires them, `--allow-unmapped` acknowledges them. |
| `memory-weave search --agent A --user U "..."` | Run retrieval and print the corresponding log row. |
| `memory-weave get <id>` | Print a full record with lineage and events. |
| `memory-weave dump --scope user:U` | Print active records in one scope. |
| `memory-weave expire` | Mark provisional records past expiry as `expired`. |
| `memory-weave reembed --model M --version V` | Re-embed every record, then swap the index. Refuses to run until gate floors are reset. |
| `memory-weave erase <id>` | Erase durable content and append an event. |
| `memory-weave grant A user:U --read --write` | Create or update a scope grant. |
| `memory-weave extract <session_id>` | Re-run extraction for one session. It is idempotent because deduplication absorbs repeats. |
| `memory-weave snapshot save|load <path>` | Copy the SQLite database for evaluation fixtures. |

## 17. Test plan for the implementation

The test suite proves policy and retrieval behavior with deterministic fakes first, then verifies the real embedding path separately. Unit tests must not download models.

### 17.1 Test fixtures and test boundaries

| Fixture | Use | Why it is deterministic |
| --- | --- | --- |
| `FakeEmbedder` | Unit tests for dense search, duplicate collapse, and gating. | Hash-based vectors with controllable similarity. |
| `FakeExtractor` | Unit tests for session extraction. | Returns predefined candidates and summaries. |
| `FakeJudge` | Unit tests for reinforcement and contradiction. | Table-driven `same`, `contradicts`, and `distinct` results. |
| `FakeRewriter` | Unit tests for query rewriting. | Captures inputs and returns a predefined rewrite or failure. |
| `FakeReranker` | Unit tests for reranker ordering and limits. | Returns known query-record scores. |

The unit suite tests storage and policy with these fakes. The integration suite is the only suite that loads the real embedder.

### 17.2 Access, evidence, and write-path tests

#### Grants

- An agent reads and writes its own `agent:<agent_id>/<user_id>` scope without a grant.
- A user-scope grant is not honored for a different principal user.
- The default write target is `user:<user_id>` only after the host provisions a read-write grant; without it, the write returns `scope_not_writable` and changes nothing.
- A user-scoped record written for user `U` never appears for user `V` through dense, lexical, or entity retrieval, even when both use the same agent.
- A plain `agent:<agent_id>` scope is unreachable without a grant and intentionally cross-user-readable when that explicit grant exists.
- A direct-store grant on `agent:A/U` is visible only in sessions for user `U`; `MemoryHost.grant` refuses to create that grant.
- `Principal` rejects `/` in agent and user IDs, so `agent:a/b/c` cannot ambiguously represent two principal pairs.
- A project grant does not imply access to a user scope.

#### Evidence

Run one parameterized suite through both `memory_write` and session extraction. It verifies that:

- A missing quote downgrades the candidate to `agent_inference`.
- A write with no evidence stores an expiring inference with no `source_ref` and the `evidence not provided` note.
- A claimed `user_statement` backed by an assistant turn downgrades with a note.
- A tool turn supports `tool_result` but not `user_statement`.
- A caller may claim a lower source kind than the supporting turn allows.
- `source_ref` points to the matched turn.

#### Explicit writes and lifecycle

- A duplicate judged `same` reinforces the original instead of inserting a new record.
- Repeated evidence from the same source reference does not count as an independent reinforcement.
- The second independent reinforcement confirms a provisional record.
- A higher-ranked independent observation upgrades the stored evidence provenance.
- The tool rejects `session_summary` and `system` as `source_kind` values.
- The tool rejects a semantic or procedural write that omits its attribute.
- A semantic or procedural write without an `about` entity is valid only in the principal user's own user scope, where the ingestor creates or resolves the principal's `person` entity.
- An attribute containing an entity prefix does not control the stored entity ID; the ingestor uses the resolved `about` entity and normalizes only the attribute part.
- Contradicting claims under different attributes of the same entity apply supersession against the authority incumbent and record `attribute_aliased_from`.
- An entity with more active attributes than `ingestion.max_entity_attributes` records `attribute_scan_truncated` and still judges the most recently reinforced records.
- A direct user or tool claim with a located quote that does not entail its content is downgraded to `agent_inference` and records the entailment score.
- A higher-rank new fact supersedes an old fact regardless of event time.
- When a subject has a later provisional conflict, the authority incumbent still receives the next supersession decision.
- An equal-rank fact with a later `event_at` supersedes the old fact.
- An equal-rank fact with an earlier `event_at` is stored as superseded on arrival and is absent from default retrieval.
- A lower-rank contradiction becomes provisional and has conflict rows in both directions.
- A provisional conflict has a `provisional_ttl_days` expiry.
- An episodic record never supersedes another record.
- Two opposing preferences above the dedup cosine floor receive the `contradicts` verdict, not `same`.

### 17.3 Extraction and entity tests

#### Session extraction

- A candidate with missing evidence is rejected and logged.
- The worker writes a session summary even when extraction returns no candidates. It has `source_kind = session_summary`, `confirmed` status, and no expiry.
- A candidate with an ambiguous `about` entity is rejected with the candidate entity IDs.

#### Entities

- Alias lookup resolves only within readable scopes.
- An unknown alias creates a provisional entity in the writer's scope.
- An alias matching two readable entities returns `entity_ambiguous` for an `about` mention and writes no record.
- For an ambiguous `mentions` entity, the write succeeds without that link.
- An explicit `entity_id` skips alias resolution and must still be readable.
- An entity merge repoints record links and unions aliases.
- An entity merge rewrites `subject_entity_id`, the derived `subject`, and the FTS subject for all records of the source entity, and rebuilds the FTS `aliases` column for every record linked to the surviving entity so the lexical channel sees the union of names.
- A contradicted same-attribute incumbent is superseded even when a provisional sibling is judged the same as the candidate, and the sibling is listed in `also_superseded`.
- Under `max_entity_attributes`, the cap follows the store's reinforced-first order; a low-cosine, recently reinforced attribute is still judged.
- Two person entities carrying the principal alias produce `entity_ambiguous`, not an exception.
- `MemoryHost.provision_user` aliases make an `about` mention of the user's display name resolve to the principal entity.
- A 300-token query produces a bounded alias set and no SQLite error.
- `follow-up` is not an identifier and matches only where its tokens are adjacent; `bge-m3` and `deploy.yml` are identifiers.
- "I'm", "what's", and "don't" never survive as query terms.
- A `Store` refuses to open with unresolved migration issues; `--expire-unmapped` retires them.
- Migration 2 returns the count of legacy rows it could not map because they have zero or multiple `about` links, and records each in `migration_issues`. The `memory-weave migrate` command requires `--allow-unmapped` to complete successfully with any unresolved rows.

### 17.4 Retrieval and result-contract tests

#### Core retrieval

- Default retrieval excludes superseded records; `include_history` includes them.
- Retrieval excludes expired records.
- The scope filter applies before dense search.
- A dense union from multiple queries is capped at `per_generator_k` after records keep their best score.
- An eligible FTS match is returned even when more than `3 * per_generator_k` ineligible matches score above it.
- A lexical match keeps the best term fraction from one query, drops one-character terms, and preserves identifier and entity-alias flags for the gate.
- Entity alias lookup batches all query n-grams into one store call.
- RRF handles an empty generator.
- The gate returns an empty response with a reason when no candidate passes.
- Duplicate collapse keeps the higher-ranked record.
- Budget filling never truncates record content.

#### Gate

- A dense-only episodic candidate at cosine 0.42 passes while a dense-only semantic candidate at 0.42 is dropped, with the per-type floor named in `gate_reason`.
- A lexical-only candidate matching one plain word is dropped; the same candidate matching one identifier token such as `ERR42` passes; matching two plain words passes.
- A lexical-only candidate whose single matched term is an entity alias passes.
- With one corroborated candidate at fused score `s` and a single-signal candidate below `relative_floor * s`, the second is dropped with a step-2 reason; an entity hit at the same score is kept.
- With a single survivor, the relative floor is a no-op.
- `empty_reason` names every missed floor for the best candidate, including the record type of the dense floor.
- Replaying a logged search offline with different floors reproduces the gate decision from the logged candidate scores alone.

#### Trigger

- In `tool_only` mode the adapter never issues a search on its own and registers all five tools.
- In `auto` mode `memory_search` is not registered; a host search runs once per user turn, never on assistant or tool turns; a user turn shorter than `auto_min_query_chars` logs `trigger.skipped` and issues no search.
- In `hybrid` mode both the host search and the model's tool search reach the handler, and their `search_log` rows carry `trigger = 'auto'` and `trigger = 'tool'` respectively.
- A non-empty host search appends exactly one tool-result-shaped message; an empty one appends nothing and still writes a log row.
- Host-issued searches use `auto_k`, and model-issued searches use the requested or default `k`.
- The prompt prefix is byte-identical across turns in every mode; the recalled block only ever appears after existing messages.

#### Freshness and rewrite

- Episodic records decay; semantic records do not.
- A `since` or `until` time window disables freshness decay.
- With rewriting disabled, queries stay unchanged and the log records `disabled`.
- With `FakeRewriter` enabled, the log stores raw and rewritten queries and retrieval uses the rewritten queries.
- A rewriter timeout or error falls back to raw queries and logs `failed`.
- The rewriter receives only queries and context, never candidates. Assert this from the fake's captured arguments.
- With no context, the rewriter runs on the raw queries alone.

#### Reranking, explanations, and logs

- Enabling a reranker without a floor is a configuration error.
- With `FakeReranker`, reranking runs only on gate and duplicate-collapse survivors, caps input at `reranker.candidates`, and records rank before and after.
- Every returned record has an `Explanation` whose generator ranks and scores match `search_log`.
- An empty response has an `empty_reason` that names the missed floors.
- One `search_log` row reconstructs every retrieval stage, including rewrite status, freshness multipliers, and records omitted by the budget.
- Every write event includes `timings_ms` for every named stage.

### 17.5 Latency and integration tests

The 50K scale benchmark uses a synthetic store. With rewriting and reranking off and the fake embedder replaced by a fixed 25 ms sleep, it asserts p50 under 80 ms. It also verifies `warm = 0` for the first call and `warm = 1` afterward.

Integration tests run once with the real embedder against a 1K-record fixture. They check that the calibrated dense floor separates a hand-labeled set of 50 relevant queries from 50 irrelevant queries.

## 18. Open items carried into the implementation plan

These choices do not block the initial implementation. Each has a safe default and a specific evaluation result that should trigger revisiting it.

| Question | Initial default | Revisit when |
| --- | --- | --- |
| Stopwords and lexical tokenization | Use a frequency-based English list of about 180 stopwords, while preserving identifiers and proper nouns. | Multilingual cases enter evaluation. |
| Exact subject lookup | Do not add a `subject` filter to `memory_search` yet. | Evaluation shows agents frequently need exact attribute lookups. |
| Session end | Deep Agents supports both explicit end and idle-timeout splitting from the first version. | Adapter behavior shows one source is unreliable. |
| Reranker floor | Leave it unset and reject `reranker.enabled` until the reranker experiment runs. | The experiment supplies a calibrated floor. |
| Long session summaries | Write one summary record capped at 1,200 characters. | Long-session evaluation shows that one summary loses important context. |
| Rewriter latitude | Resolve references, name the subject, and preserve query count. | Follow-up-question cases show misses that need broader rewrites. |
| Generator concurrency | Run dense, lexical, and entity generators sequentially. | Measurement shows a thread pool improves latency enough to justify its complexity. |
| Equivalence-judge floors | Treat `nli-deberta-v3-small` `entail_floor` and `contradict_floor` as starting values. | A labeled set of same, contradictory, and distinct pairs calibrates them, or the judge model changes. |
| Rank versus event time | Higher source rank beats a later `event_at` in supersession. | Stale-record evaluation, including a recent tool result versus an older user statement on the same subject, shows the policy is harmful. |
| Trigger mode | `tool_only`. The `auto` and `hybrid` paths are specified in section 14.1 and built in the adapter phases. | The agent-in-the-loop benchmark shows `hybrid` raises accuracy on memory-needed turns while keeping the ordinary-turn injection rate under the target. |
| Gate floors | Per-type dense floors, two matched terms for lexical-only passes, relative floor 0.5. | The three-class calibration sweep in section 10.5 picks different values, or the reranker proves a better gate for host-issued searches. |
| `search_log.trigger` column | Added as schema migration 2 when phase 8 first writes the log. | Never; it is required by the benchmark split. |
| Confidence | Store it for lifecycle and audit only; do not use it in ranking or gating. | Phase 15 data shows that a ranking prior helps, or confirms the field should be removed. |
