# Agent Memory System: Low-Level Design

Status: draft v1, 2026-09-03. Implements the decisions in `agent-memory-hld.md`. Read that first. This document is written so that an implementation plan can be derived from it section by section.

Language is Python 3.12. The package is called `memlayer` throughout; rename freely. Everything runs in one process with one SQLite file.

## 1. Package layout

```
memlayer/
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
    extractor.py     # Extractor protocol + StructuredLLMExtractor + FakeExtractor
    session.py       # SessionBuffer: turns for the running session, used by extraction and evidence checks
  retrieve/
    retriever.py     # the pipeline
    fusion.py        # RRF
    gate.py          # empty-result decision
    freshness.py     # episodic recency
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

All tunables live in one file. The defaults are the v1 defaults; the evaluation re-calibrates the gate floors.

```yaml
store:
  path: ./memory.sqlite

embedding:
  model: BAAI/bge-m3
  version: "1"              # bumped by hand on any change to model or preprocessing
  dims: 1024
  device: auto              # mps on Apple silicon, cuda if present, else cpu
  max_chars: 2000           # content is truncated for embedding, never for storage

reranker:
  enabled: false
  model: BAAI/bge-reranker-v2-m3
  candidates: 30

retrieval:
  per_generator_k: 30
  rrf_k: 60
  default_k: 8
  token_budget: 1500
  dedup_cosine: 0.92
  gate:
    dense_floor: 0.45       # cosine, bge-m3 dense head. Calibrate.
    lexical_min_term_fraction: 0.5
  freshness:
    episodic_half_life_days: 30
    floor: 0.5              # multiplier never drops below this

ingestion:
  dedup_cosine: 0.92
  contradiction_cosine: 0.80
  provisional_ttl_days: 30
  reinforcements_to_confirm: 2
  extraction_model: claude-haiku-4-5-20251001
  extraction_max_candidates: 20

policy:
  source_rank:
    user_statement: 4
    system: 3
    tool_result: 2
    agent_inference: 1
```

## 3. Schema

SQLite, WAL mode, foreign keys on. Timestamps are ISO 8601 UTC strings. Ids are UUIDv7 so they sort by creation time.

```sql
CREATE TABLE records (
  id              TEXT PRIMARY KEY,
  type            TEXT NOT NULL CHECK (type IN ('semantic','episodic','procedural')),
  version         INTEGER NOT NULL DEFAULT 1,
  content         TEXT NOT NULL,
  subject         TEXT NOT NULL,              -- '<entity_ref>/<attribute>' or '<entity_ref>/-' when no attribute applies
  scope_kind      TEXT NOT NULL CHECK (scope_kind IN ('agent','user','project','org')),
  scope_id        TEXT NOT NULL,
  source_kind     TEXT NOT NULL CHECK (source_kind IN ('user_statement','system','tool_result','agent_inference')),
  source_ref      TEXT,                       -- 'session:<id>#turn:<n>' or a tool call id or a document id
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
  tags            TEXT NOT NULL DEFAULT '[]'  -- JSON array
);
CREATE INDEX records_scope ON records(scope_kind, scope_id, status);
CREATE INDEX records_subject ON records(scope_kind, scope_id, subject, status);
CREATE INDEX records_event ON records(type, event_at);

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
  vector      BLOB NOT NULL                   -- float32, L2-normalized
);

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
  request       TEXT NOT NULL,                -- JSON of the SearchRequest
  readable_scopes TEXT NOT NULL,              -- JSON
  dense         TEXT NOT NULL,                -- JSON [[record_id, cosine], ...]
  lexical       TEXT NOT NULL,                -- JSON [[record_id, bm25, matched_terms, total_terms], ...]
  entity        TEXT NOT NULL,                -- JSON [[record_id, entity_id], ...]
  fused         TEXT NOT NULL,                -- JSON [[record_id, score], ...] after freshness
  gated_out     TEXT NOT NULL,                -- JSON [record_id, ...]
  deduped_out   TEXT NOT NULL,                -- JSON [[dropped_id, kept_id], ...]
  reranked      TEXT,                         -- JSON or NULL
  returned      TEXT NOT NULL,                -- JSON [record_id, ...]
  timings_ms    TEXT NOT NULL                 -- JSON {embed, filter, dense, lexical, entity, fuse, gate, dedup, rerank, total}
);
```

Notes on the schema:

- `subject` is the contradiction key. Format is `<entity_ref>/<attribute>` where `entity_ref` is `<kind>:<canonical>` of the record's primary entity (role `about`) and `attribute` is a lowercase snake_case slug chosen by the writer. Example: `person:aditya/explanation_style`, `project:my-portfolio/commit_convention`. Episodic records use `<entity_ref>/-` and never participate in supersession.
- The FTS table denormalizes entity aliases into the row so a lexical query for a name hits records about that entity even when the content uses a pronoun.
- The vector blob is the source of truth for embeddings. The in-memory matrix is rebuilt from it on startup.
- `events` is the audit trail. `records` is the materialized current state. Rebuilding `records` from `events` is not supported in v1; the log exists to explain, not to replay.

## 4. Core types

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
    scope: Scope
    source_kind: Literal["user_statement", "system", "tool_result", "agent_inference"]
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
    tags: list[str]
    entity_ids: list[str]

@dataclass(frozen=True)
class Principal:
    agent_id: str
    user_id: str
    session_id: str | None
    project_id: str | None

@dataclass(frozen=True)
class SearchRequest:
    queries: list[str]                        # 1 to 3
    types: list[str] | None
    entities: list[str] | None                # alias strings, resolved by the retriever
    since: datetime | None                    # applies to event_at
    until: datetime | None
    k: int
    include_history: bool                     # superseded and expired become eligible

@dataclass
class Candidate:
    record_id: str
    dense: float | None                       # cosine
    lexical: tuple[float, int, int] | None    # bm25, matched_terms, total_terms
    entity_hit: bool
    fused: float
    passed_gate: bool

@dataclass
class SearchResult:
    record: Record
    score: float
    matched_by: list[Literal["dense", "lexical", "entity"]]
    dense: float | None
    lexical_terms: tuple[int, int] | None
    why: str                                  # one line, human readable

@dataclass
class SearchResponse:
    search_id: str
    results: list[SearchResult]
    empty_reason: str | None                  # set when results is empty
```

## 5. Interfaces

Every model-facing dependency is a Protocol with a fake implementation for tests.

```python
class Embedder(Protocol):
    name: str
    version: str
    dims: int
    def embed(self, texts: list[str]) -> np.ndarray: ...     # (n, dims) float32, L2-normalized

class Reranker(Protocol):
    def score(self, query: str, docs: list[str]) -> list[float]: ...

class Extractor(Protocol):
    def extract(self, transcript: list[Turn], context: ExtractionContext) -> ExtractionOutput: ...

class Adapter(Protocol):
    def register_tools(self, handlers: ToolHandlers) -> None: ...
    def principal_from_run(self, run_ctx: Any) -> Principal: ...
    def on_session_start(self, ...) -> None: ...
    def on_turn(self, ...) -> None: ...
    def on_session_end(self, ...) -> None: ...
```

`ExtractionOutput` is a list of `CandidateRecord` plus one `SessionSummary`:

```python
@dataclass
class CandidateRecord:
    type: Literal["semantic", "episodic", "procedural"]
    content: str
    subject: str
    source_kind: Literal["user_statement", "tool_result", "agent_inference"]
    evidence: str                             # verbatim
    evidence_turn: int
    entity_mentions: list[EntityMention]      # (kind, text, role)
    event_at: datetime | None
    confidence: float

@dataclass
class SessionSummary:
    content: str                              # 3 to 8 sentences: goal, what happened, decisions, open threads
    decisions: list[str]
    entity_mentions: list[EntityMention]
```

## 6. Policy

### 6.1 Grants

```python
def readable_scopes(store, agent_id, user_id, project_id) -> list[Scope]:
    scopes = [Scope("agent", agent_id)]                      # implicit
    scopes += store.grants_for(agent_id, can_read=True)
    return [s for s in scopes if s.kind != "user" or s.id == user_id]   # never read another user's scope, even if granted
```

The last line is a safety belt: a grant on `user:X` is only honoured when the current principal is `X`. Cross-user reads require an `org` or `project` scope by design.

`writable_scopes` is the same with `can_write=True`. The implicit agent scope is always writable.

### 6.2 Source rank and initial status

```python
def initial_status(source_kind) -> str:
    return "provisional" if source_kind == "agent_inference" else "confirmed"

def initial_confidence(source_kind) -> float:
    return {"user_statement": 0.95, "system": 0.9, "tool_result": 0.85, "agent_inference": 0.6}[source_kind]

def initial_expiry(source_kind, now) -> datetime | None:
    return now + timedelta(days=cfg.provisional_ttl_days) if source_kind == "agent_inference" else None
```

### 6.3 Supersession rule

Given existing record `old` and candidate `new` with the same `subject` in the same scope, both non-episodic:

```
if content_equivalent(old, new):        reinforce(old); do not insert
elif rank(new) >= rank(old):            insert new with supersedes_id=old.id; old.status = superseded
else:                                   insert new as provisional; add conflict rows both ways
```

`content_equivalent` is cosine ≥ `ingestion.dedup_cosine` on the two embeddings. The retriever excludes superseded records, so a lower-ranked contradiction never hides a confirmed fact, but it is kept and surfaced in `memory_get` so an agent can ask the user.

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

### 6.5 Expiry

Lazy. The retrieval filter treats `expires_at < now` as ineligible. A CLI command `memlayer expire` flips those rows to `expired` and writes events, for tidiness. Nothing depends on the sweep having run.

## 7. Vector index

In-process, exact, filtered.

```python
class VectorIndex:
    ids: list[str]
    pos: dict[str, int]
    matrix: np.ndarray            # (n, dims) float32, rows L2-normalized
    live: np.ndarray              # (n,) bool

    def load(self, store): ...    # read all embeddings for the configured model/version
    def upsert(self, record_id, vec): ...   # append or overwrite row; grows matrix by doubling
    def remove(self, record_id): ...        # live[pos] = False

    def search(self, qvec, allowed: np.ndarray, k) -> list[tuple[str, float]]:
        scores = self.matrix @ qvec                     # (n,)
        scores[~(self.live & allowed)] = -inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > -inf]
```

`allowed` is a boolean mask built from the SQL hard filter: the store returns the set of eligible record ids, and the retriever maps them to positions. At 50K rows this is one matmul and one partition, under 5 ms.

Multiple queries: embed all, search each, union by max cosine per record before fusion.

Startup: load takes a full scan of `embeddings`. At 50K rows of 4 KB this is 200 MB and about one second. Acceptable.

## 8. Ingestion

### 8.1 `memory_write` handler (synchronous)

```
1. principal, args -> validate schema
2. scope = args.scope or Scope("agent", principal.agent_id)
   assert scope in writable_scopes(principal)              -> error "scope_not_writable"
3. source_kind = args.source_kind
   if source_kind == "user_statement":
       if not args.evidence or not session_buffer.contains(principal.session_id, args.evidence):
           source_kind = "agent_inference"; note = "downgraded: evidence not found in session"
4. entities = resolve_or_create(args.entities, scope, principal)   (section 9)
5. subject = args.subject or f"{primary_entity_ref}/-"
6. vec = embedder.embed([content])[0]
7. if type != "episodic":
       existing = store.active_by_subject(scope, subject)
       apply supersession rule (6.3) using vec
       if reinforced: return {"record_id": existing.id, "outcome": "reinforced"}
   near = vector_index.search(vec, allowed=same scope & same type & active, k=3)
   if near and near[0].cosine >= cfg.ingestion.dedup_cosine:
       reinforce(near[0]); return {"outcome": "reinforced", ...}
8. insert record, embedding, fts row, entity links; append events
9. return {"record_id", "status", "outcome": "created" | "superseded:<old_id>" | "conflict:<old_id>", "note"}
```

Latency: one embed (25 ms warm on MPS), one or two index searches (under 5 ms), a handful of inserts in one transaction (under 5 ms).

### 8.2 Session extraction (asynchronous)

Triggered by `on_session_end`. Runs in a worker thread or a separate process; the host does not wait.

```
1. turns = store.session_turns(session_id)
   if len(turns) < 2: write only the session summary; return
2. ctx = ExtractionContext(principal, known_entities=aliases readable by the agent, existing_subjects=active subjects in writable scopes)
3. out = extractor.extract(turns, ctx)
4. for cand in out.candidates[:cfg.extraction_max_candidates]:
       a. evidence check: normalize whitespace on both sides; require cand.evidence in turns[cand.evidence_turn].content
          else reject with reason "evidence_not_found"
       b. source check: if cand.source_kind == "user_statement", the evidence turn must have role == "user"
          else downgrade to agent_inference
       c. run steps 4 to 8 of 8.1 with creator_agent_id = principal.agent_id, source_ref = f"session:{sid}#turn:{n}"
5. write one episodic record from out.summary:
       subject = f"session:{sid}/-", event_at = session.ended_at, source_kind = "agent_inference",
       status = "confirmed" (session summaries are always kept; they are the episodic backbone),
       confidence = 0.8, no expiry
6. sessions.extracted_at = now; append event extraction.run with counts {proposed, written, reinforced, superseded, rejected, reasons}
```

Rejected candidates are logged in the event payload with their content, so the evaluation can measure extractor precision independently of validator precision.

### 8.3 Extractor prompt contract

The extractor receives the transcript with turn numbers, the list of known entity aliases, and the list of existing subjects. It must return JSON matching `ExtractionOutput`. Instructions that matter:

- Extract only what would change a future action. Not everything said is memory.
- One fact per candidate. No compound statements.
- `content` is a standalone declarative sentence that makes sense without the transcript.
- `evidence` is copied verbatim from exactly one turn; do not paraphrase.
- Prefer an existing `subject` when the fact is about the same attribute. Only invent a new attribute slug when none fits.
- Mark `user_statement` only when the user said it in their own words. Anything the assistant concluded is `agent_inference`.
- Do not extract facts about third parties unless the user stated them.
- Episodic candidates are for decisions with rationale, outcomes, and failures. Put routine progress in the session summary instead.

The prompt is versioned in the repo and its version is recorded on every `extraction.run` event.

## 9. Entity resolution

```python
def resolve_or_create(mentions, scope, principal):
    readable = readable_scopes(principal)
    out = []
    for m in mentions:
        norm = normalize(m.text)                                     # lower, collapse ws, strip diacritics
        hits = store.entities_by_alias(norm, kinds=[m.kind], scopes=readable, status in (provisional, confirmed))
        if len(hits) == 1:   out.append(hits[0])
        elif len(hits) > 1:  out.append(most_specific_scope(hits))   # agent > user > project > org; log "ambiguous_alias"
        else:
            e = store.create_entity(kind=m.kind, canonical=m.text, scope=scope, status="provisional")
            store.add_alias(e.id, norm); append event entity.created
            out.append(e)
    return out
```

Merging is manual: `memory_revise(entity_id=..., merge_into=...)` from an agent with write on both scopes, or the CLI. Merge sets `status=merged`, `merged_into`, repoints `record_entities`, and unions aliases. Every alias lookup follows `merged_into` to the surviving entity.

The primary entity of a record (role `about`) is the first mention with role `about`, or the principal user when none is given and the type is semantic.

## 10. Retrieval pipeline

### 10.1 Handler

```python
def memory_search(principal, req: SearchRequest) -> SearchResponse:
    t = Timer()
    scopes = readable_scopes(store, principal.agent_id, principal.user_id, principal.project_id)

    eligible = store.eligible_ids(scopes, req.types, req.since, req.until, req.include_history, now)   # SQL, returns set[str]
    allowed = vector_index.mask(eligible)
    t.mark("filter")

    qvecs = embedder.embed(req.queries); t.mark("embed")

    dense = {}                                   # record_id -> max cosine
    for qv in qvecs:
        for rid, cos in vector_index.search(qv, allowed, cfg.per_generator_k):
            dense[rid] = max(dense.get(rid, -1), cos)
    t.mark("dense")

    lexical = lexical_search(req.queries, eligible, cfg.per_generator_k)     # 10.2
    t.mark("lexical")

    entity_ids = resolve_aliases(req.entities, scopes) + entities_in_queries(req.queries, scopes)
    entity_hits = store.records_for_entities(entity_ids, eligible, order="event_at desc", limit=cfg.per_generator_k)
    t.mark("entity")

    fused = rrf([ranked(dense), ranked(lexical), ranked(entity_hits)], k=cfg.rrf_k)      # 10.3
    fused = apply_freshness(fused, store, now)                                            # 10.4
    t.mark("fuse")

    kept, gated_out = gate(fused, dense, lexical, entity_hits, req.queries)              # 10.5
    t.mark("gate")

    kept, deduped_out = collapse_duplicates(kept, vector_index, cfg.dedup_cosine)         # 10.6
    t.mark("dedup")

    if cfg.reranker.enabled:
        kept = rerank(kept[:cfg.reranker.candidates], req.queries[0])                     # 10.7
    t.mark("rerank")

    results = fill_budget(kept, req.k, cfg.token_budget)                                  # 10.8
    log_search(...); t.mark("total")
    return SearchResponse(search_id, results, empty_reason)
```

### 10.2 Lexical search

Query preparation: tokenize with the same unicode61 rules, drop a short stopword list, keep identifiers and proper nouns intact, join with `OR`. FTS5 query per user query, take `bm25(records_fts, 1.0, 2.0, 3.0)` as the score (content weight 1, subject 2, aliases 3). Results are intersected with `eligible` in Python (FTS5 cannot see the scope predicate). Fetch `3 * per_generator_k` from FTS to survive the intersection, then cut.

For each hit, count how many of the query's non-stopword terms appear in `content + subject + aliases`. That count and the total go into the candidate for the gate.

### 10.3 Reciprocal rank fusion

```python
def rrf(rankings: list[list[str]], k: int) -> dict[str, float]:
    score = defaultdict(float)
    for ranking in rankings:
        for rank, rid in enumerate(ranking, start=1):
            score[rid] += 1.0 / (k + rank)
    return dict(sorted(score.items(), key=lambda kv: -kv[1]))
```

An empty ranking contributes nothing and breaks nothing.

### 10.4 Freshness

Only episodic records are adjusted. For each, `age_days = (now - event_at).days`, `mult = max(cfg.freshness.floor, 0.5 ** (age_days / cfg.freshness.episodic_half_life_days))`, `fused *= mult`. Semantic and procedural records are not decayed; supersession handles their staleness.

If `req.since` or `req.until` is set, freshness is skipped: the caller has already expressed a time intent.

### 10.5 Gate

Per candidate, keep if any of:

- `entity_hit` is true.
- `dense >= cfg.gate.dense_floor`.
- `lexical.matched_terms / lexical.total_terms >= cfg.gate.lexical_min_term_fraction` and `total_terms >= 1`.

A candidate that came only from dense with cosine below the floor, or only from lexical with a low term fraction, is dropped. If nothing survives, `results` is empty and `empty_reason` says which floors were missed by the best candidate, for example `"best dense 0.38 < 0.45; best lexical 1/4 terms; no entity match"`.

The dense floor is model-specific. The evaluation sweeps it on the no-memory cases and the desired-retrieval cases and picks the value that maximizes F1 over "should have returned something". Record the chosen value with the embedding version in `config.py`.

### 10.6 Duplicate collapse

Walk `kept` in fused order. For each record, compare its vector against every already-accepted record; if cosine ≥ `retrieval.dedup_cosine`, drop it and log `(dropped, kept)`. This catches records that survived ingestion dedup because they sat in different scopes or types.

### 10.7 Reranker (optional)

`Reranker.score(query, [r.content for r in kept])`, sort descending, and replace the gate with `score >= cfg.reranker.floor` (a separate calibrated value). When enabled, the reranker's floor is the gate; the dense and lexical floors then act only as candidate pre-filters.

### 10.8 Budget fill

Token count is `len(content) // 4` plus 30 for the envelope line. Walk in order, stop when either `k` results or the budget is reached. Never truncate a record's content; a record that does not fit is skipped and the next one is tried.

### 10.9 Result formatting for the model

Each result is rendered to the tool result as one block:

```
[mem_01J...] semantic · confirmed · user_statement · event 2026-08-12 · scope user:aditya
Prefers concise technical explanations without preamble.
matched: dense 0.71, lexical 3/3
```

Ids are short-prefixed for readability but the full id is returned in the structured payload. The rendering is the adapter's job; the payload shape is fixed.

## 11. Tool surface

Five tools. Schemas are plain JSON Schema so any framework can register them.

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

### `memory_get`

`{"ids": [string]}`. Returns full records including superseded lineage and conflicts, so the agent can inspect provenance or show a disagreement to the user.

### `memory_write`

```json
{
  "type": {"enum": ["semantic", "episodic", "procedural"]},
  "content": {"type": "string", "maxLength": 1000},
  "subject": {"type": "string", "description": "'<kind>:<name>/<attribute>' e.g. 'person:aditya/timezone'. Omit for episodic."},
  "scope": {"type": "object", "properties": {"kind": {...}, "id": {...}}, "description": "Defaults to this agent's private scope."},
  "source_kind": {"enum": ["user_statement", "tool_result", "agent_inference"]},
  "evidence": {"type": "string", "description": "Verbatim quote from this session. Required for user_statement."},
  "event_at": {"type": "string", "format": "date-time"},
  "entities": {"type": "array", "items": {"type": "object", "properties": {"kind": {...}, "name": {...}, "role": {"enum": ["about", "mentions"]}}}},
  "tags": {"type": "array", "items": {"type": "string"}}
}
```

The description tells the agent: write only what would change a future action, one fact per call, and never write `system` (that source kind is reserved for the host).

### `memory_revise`

`{"id": string, "content"?: string, "action": "confirm" | "supersede" | "expire", "reason": string}`. `supersede` requires `content` and creates a new record with `supersedes_id`; `confirm` promotes provisional to confirmed and requires the principal to have write on the record's scope; `expire` sets `expired`. All three append events with the reason. Entity merge is a separate argument shape on the same tool: `{"entity_id", "merge_into", "reason"}`.

### `memory_forget`

`{"id": string, "reason": string}`. Sets `deleted`, keeps the row, removes the FTS row and marks the vector dead. Content erasure is an admin CLI operation (`memlayer erase <id>`), not a tool, because a user's deletion request should be honoured by a human-controlled path, not an agent's judgement.

## 12. Session buffer and ingestion hooks

Adapters call three hooks. The memory layer stores turns as they arrive so evidence checks in `memory_write` and end-of-session extraction see the same text.

```python
def on_session_start(principal): store.create_session(...)
def on_turn(principal, role, content): store.append_turn(...)        # tool turns store the tool result text, not the call
def on_session_end(principal): store.end_session(...); schedule(extract_session, principal.session_id)
```

If a framework cannot signal session end reliably, the adapter treats an idle timeout (default 30 minutes) as the end, and a new turn after that starts a new session. The session id is part of every `source_ref`, so a split session costs nothing but a slightly less useful summary.

## 13. Memory-use policy text

The only memory-related content in the prompt prefix. Kept short, and kept stable so the prefix caches.

> You have long-term memory available through tools. Before acting on anything that could depend on the user's preferences, earlier decisions, or previous sessions, call `memory_search` with one to three specific phrases. Do not search for general knowledge or for facts already visible in this conversation. When results come back, check their status, source, and date before relying on them; a provisional or old record may be wrong, and you can ask the user. When the user states a preference, a fact about themselves, or a decision, save it with `memory_write` and quote their words as evidence. Save decisions you make together as episodic records with the reason. Do not save guesses as facts.

This text is versioned with the extractor prompt; the evaluation records both versions on every run.

## 14. Adapters

### Deep Agents

- Tools: register the five JSON schemas as LangChain tools; handlers receive the run config and derive `Principal` from `configurable.agent_id`, `configurable.user_id`, `configurable.thread_id` (as session id).
- Hooks: `on_turn` from a callback on each human and AI message; `on_session_end` from an explicit call by the host or the idle timeout.
- Tool result formatting: the block format in 10.9, joined with blank lines, prefixed by a one-line count.

### CrewAI

- Tools: wrap each handler in a `BaseTool` subclass. Agent id is the CrewAI agent role slug; user id and session id come from crew inputs, which the host must supply.
- Hooks: CrewAI step callbacks give per-step output, not per-turn dialogue; the adapter records the task description as the first user turn and each step output as an assistant turn. This is coarser than Deep Agents and is part of what the experiment measures.

Both adapters are thin. If either needs to store framework-specific state in `records`, that is a contract failure and the contract, not the adapter, gets fixed.

## 15. Latency accounting

For `memory_search` with two queries, no reranker, 50K records, Apple M-series:

| Stage | Expected | Dominant cost |
| --- | --- | --- |
| filter | 3 ms | One indexed SQL query returning ids; building the mask. |
| embed | 20 to 40 ms | bge-m3 forward pass for two short texts, warm. |
| dense | 2 to 5 ms | Two matmuls over the matrix. |
| lexical | 3 to 8 ms | Two FTS5 queries, intersection. |
| entity | 1 to 3 ms | Alias lookup plus an indexed join. |
| fuse, gate, dedup | under 2 ms | Python over at most 90 candidates. |
| log | 1 to 2 ms | One insert. |
| total | 35 to 65 ms | |

The embedder is the budget. Two mitigations are in scope: keep the model loaded in the process, and cache query embeddings by exact string for the life of the process. Out of scope for v1: a smaller query-side encoder.

`memory_write` is the same embed cost plus one small index search and one transaction.

## 16. Operations and CLI

```
memlayer migrate                         # apply schema migrations
memlayer search --agent A --user U "..." # run the pipeline, print the log row
memlayer get <id>                        # full record with lineage and events
memlayer dump --scope user:U             # all active records in a scope
memlayer expire                          # sweep provisional records past expiry
memlayer reembed --model M --version V   # re-embed every record, then swap the index; refuses if gate floors are not re-set
memlayer erase <id>                      # content erasure with an event
memlayer grant A user:U --read --write
memlayer extract <session_id>            # re-run extraction for a session, idempotent (dedup absorbs repeats)
memlayer snapshot save|load <path>       # copy the sqlite file, for evaluation fixtures
```

## 17. Test plan for the implementation

Unit tests use `FakeEmbedder` (deterministic hash-based vectors with controllable similarity) and `FakeExtractor` (returns canned candidates). No model downloads in unit tests.

| Area | Cases |
| --- | --- |
| Grants | Agent reads own scope without grant. Grant on another user's scope is not honoured. Project grant does not imply user grant. |
| Write | `user_statement` without findable evidence is downgraded. Duplicate reinforces instead of inserting. Second reinforcement confirms a provisional record. |
| Supersession | Higher-rank new fact supersedes. Lower-rank new fact is stored provisional with conflict rows. Episodic never supersedes. |
| Extraction | Candidate with missing evidence is rejected and logged. `user_statement` claimed on an assistant turn is downgraded. Session summary is written even with zero candidates. |
| Entities | Alias resolves within readable scopes only. Unknown alias creates a provisional entity in the writer's scope. Merge repoints links and unions aliases. |
| Retrieval | Superseded records excluded by default and included with `include_history`. Expired records excluded. Scope filter applied before dense search. RRF handles an empty generator. Gate returns empty with a reason. Duplicate collapse keeps the higher rank. Budget never truncates content. |
| Freshness | Episodic decays, semantic does not. Time window disables decay. |
| Log | Every stage of a search is reconstructible from one `search_log` row. |
| Latency | Benchmark test over a 50K synthetic store asserts p50 under 80 ms with the fake embedder replaced by a fixed 25 ms sleep. |

Integration tests run once with the real embedder against a 1K-record fixture and check that the calibrated dense floor separates a hand-labelled set of 50 relevant and 50 irrelevant queries.

## 18. Open items carried into the implementation plan

1. Stopword list and tokenization for lexical search: start with FTS5 defaults plus a 100-word English list; revisit when multilingual cases enter the evaluation.
2. Whether `memory_search` should accept a `subject` filter for exact attribute lookups. Cheap to add; decide once the evaluation shows whether agents use it.
3. Idle-timeout session splitting versus explicit end: the Deep Agents adapter should support both from day one.
4. The reranker floor is unset until the reranker experiment runs.
5. Whether the session summary should be chunked for long sessions. Start with a single record capped at 1200 characters.
