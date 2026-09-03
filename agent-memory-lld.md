# Agent Memory System: Low-Level Design

This document implements the decisions in `agent-memory-hld.md`.

Language is Python 3.12. The package is called `memlayer` throughout. Everything runs in one process with one SQLite file.

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

All tunables live in one file. Evaluation recalibrates the gate floors.

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
  enabled: false            # feature flag. Runs after duplicate collapse, on survivors only.
  model: BAAI/bge-reranker-v2-m3
  candidates: 30
  floor: null               # becomes the gate when enabled. Unset until the reranker experiment calibrates it.
  budget_mean_ms: 100       # initial latency budget for 30 candidates on the target laptop; the benchmark replaces this with measured p50/p95

retrieval:
  rewrite:
    enabled: false          # feature flag. First stage of the pipeline when on.
    model: claude-haiku-4-5-20251001
    max_context_chars: 2000 # host-supplied current-turn context is truncated to this before the rewrite call
    timeout_ms: 800         # on timeout or error, fall back to the raw queries and log rewrite_status = failed
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
  dedup_candidate_cosine: 0.85   # cosine only proposes a possible duplicate; the equivalence judge decides (section 6.3)
  equivalence:
    model: cross-encoder/nli-deberta-v3-small   # local NLI cross-encoder, run in both directions
    entail_floor: 0.70                          # both directions must entail at or above this to count as "same"
    contradict_floor: 0.70                      # either direction at or above this counts as "contradicts"
  provisional_ttl_days: 30
  reinforcements_to_confirm: 2
  extraction_model: claude-haiku-4-5-20251001
  extraction_max_candidates: 20

policy:
  source_rank:
    user_statement: 4
    system: 3
    tool_result: 2
    session_summary: 2         # derived from the transcript by the extractor; only the extractor may write it
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
  reranked      TEXT,                         -- JSON [[record_id, rank_before, rank_after, score], ...] or NULL when disabled
  budget_out    TEXT NOT NULL,                -- JSON [record_id, ...] survivors that did not fit k or the token budget
  returned      TEXT NOT NULL,                -- JSON [record_id, ...]
  explanations  TEXT NOT NULL,                -- JSON [Explanation, ...] one per returned record, plus empty_reason when none
  config_flags  TEXT NOT NULL,                -- JSON {embedding_model, embedding_version, rewrite_enabled, reranker_enabled, gate floors}
  warm          INTEGER NOT NULL,             -- 1 if the embedder and index were already loaded when the call started
  timings_ms    TEXT NOT NULL                 -- JSON {rewrite, scopes, filter, embed, dense, lexical, entity, fuse, freshness, gate, dedup, rerank, budget, explain, log, total}
);
```

Notes on the schema:

- `subject` is the contradiction key. Format is `<entity_ref>/<attribute>` where `entity_ref` is `<kind>:<canonical>` of the record's primary entity (role `about`) and `attribute` is a lowercase snake_case slug chosen by the writer. Example: `person:aditya/explanation_style`, `project:agentic-memory-system/commit_convention`. Episodic records use `<entity_ref>/-` and never participate in supersession.
- The FTS table denormalizes entity aliases into the row so a lexical query for a name hits records about that entity even when the content uses a pronoun.
- The vector blob is the source of truth for embeddings. The in-memory matrix is rebuilt from it on startup.
- `events` is the audit trail. `records` is the materialized current state. The current design does not rebuild `records` from `events`; the log exists to explain, not to replay.
- Write-path timing lives in the event payload, not in a separate table. Every `record.created`, `record.reinforced`, and `record.superseded` event carries `timings_ms` with the stages named in section 8.1, and every `extraction.run` event carries the stages named in section 8.2. The benchmark reads search timing from `search_log` and write timing from `events`.

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
    aliases: list[str]                        # normalized forms
    created_at: datetime

@dataclass(frozen=True)
class EntityMention:
    kind: Literal["person", "project", "org", "repo", "product", "other"]
    text: str                                 # as written in the source
    role: Literal["about", "mentions"]
    entity_id: str | None = None              # explicit id supplied by the writer; skips alias resolution when set

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
    known_entities: list[tuple[str, str, str]]   # (entity_id, kind, canonical) readable by the agent
    existing_subjects: list[str]                 # active subjects in the agent's writable scopes
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
    source_kind: Literal["user_statement", "tool_result", "agent_inference"]   # source kind supported by the evidence
    note: str | None                          # set when the claimed kind was downgraded

@dataclass(frozen=True)
class Principal:
    agent_id: str
    user_id: str
    session_id: str | None
    project_id: str | None

@dataclass(frozen=True)
class SearchRequest:
    queries: list[str]                        # 1 to 3, as the agent wrote them (the raw retrieval request)
    context: str | None                       # host-supplied current-turn context, set by the adapter, never by the agent
    types: list[str] | None
    entities: list[str] | None                # alias strings, resolved by the retriever
    since: datetime | None                    # applies to event_at
    until: datetime | None
    k: int
    include_history: bool                     # superseded and expired become eligible

@dataclass
class GeneratorHit:
    rank: int                                 # 1-based rank within that generator's list
    score: float                              # cosine, bm25, or 0.0 for entity

@dataclass
class Candidate:
    record_id: str
    dense: GeneratorHit | None
    lexical: GeneratorHit | None
    lexical_terms: tuple[int, int] | None     # matched_terms, total_terms
    entity: GeneratorHit | None
    entity_id: str | None
    rrf_score: float
    fused_rank: int
    freshness_multiplier: float | None        # set for episodic records only
    score: float                              # rrf_score * freshness_multiplier
    gate_reason: str | None                   # why it passed, or why it was dropped
    rerank_score: float | None
    rank_after_rerank: int | None

@dataclass
class Explanation:                            # one per returned record; the HLD's "explanation object"
    raw_queries: list[str]
    rewritten_queries: list[str] | None
    rewrite_status: Literal["disabled", "applied", "unchanged", "failed"]
    matched_by: list[Literal["dense", "lexical", "entity"]]
    dense: GeneratorHit | None
    lexical: GeneratorHit | None
    lexical_terms: tuple[int, int] | None
    entity: GeneratorHit | None
    fused_rank: int
    freshness_multiplier: float | None
    rerank: tuple[int, int, float] | None     # rank_before, rank_after, score; None when the reranker is disabled
    gate: str                                 # which floor or match let it through
    dedup: str                                # "kept" or "kept over <dropped_id> at cosine 0.94"
    budget: str                               # "fit at position 3 of 8, 412 tokens used"
    source_kind: str
    status: str
    created_at: datetime
    event_at: datetime
    entity_ids: list[str]
    summary: str                              # one line, human readable, rendered to the model

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
    empty_reason: str | None                  # set when results is empty; states which floors the best candidate missed
    timings_ms: dict[str, float]
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

class QueryRewriter(Protocol):
    def rewrite(self, queries: list[str], context: str | None) -> RewriteResult: ...

class EquivalenceJudge(Protocol):
    def judge(self, a: str, b: str) -> Literal["same", "contradicts", "distinct"]: ...
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
    return {"user_statement": 0.95, "system": 0.9, "tool_result": 0.85, "session_summary": 0.8, "agent_inference": 0.6}[source_kind]

def initial_expiry(source_kind, now) -> datetime | None:
    return now + timedelta(days=cfg.provisional_ttl_days) if source_kind == "agent_inference" else None
```

`session_summary` is its own source kind, so the policy table, not special extraction code, explains why it is confirmed and never expires. Its source is the whole transcript, referenced by `source_ref = "session:<id>"`. Only the extractor can write it; it must be an episodic record with subject `session:<id>/-`; and it cannot supersede or reinforce another record. A summary describes a dated session, not a fact about the user. A separate evidenced candidate is required to store one.

### 6.3 Supersession rule

Applies when a new non-episodic record lands on a `subject` that already has an active record in the same scope. Two inputs decide the outcome: what the equivalence judge says about the two contents, and which record has authority.

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
        return new.event_at > old.event_at                # equal rank: the later event wins, whenever it arrived
    return new.created_at >= old.created_at               # same event time: the later write wins
```

A stale record of equal rank that arrives after a newer fact does not supersede it. It is inserted with `status = superseded` on arrival and no `supersedes_id`, and the event records `superseded_on_arrival_by = old.id`. It is visible through `include_history` and `memory_get` but never through default retrieval. A higher-ranked record supersedes regardless of event time, because a user's explicit statement about a subject outranks a tool's observation of it even when the observation is more recent; the log records both timestamps so the evaluation can check whether that rule is right.

Cosine similarity never decides equivalence on its own. Two opposing short preferences ("prefers concise answers", "prefers detailed answers") sit within a few hundredths of each other in embedding space. Cosine is used only to propose candidates for the judge: same-subject records are always judged, and records on other subjects are judged when cosine is at or above `ingestion.dedup_candidate_cosine`. The judge is a local NLI cross-encoder and costs roughly 20 to 40 ms per pair on the target laptop; it runs on at most four pairs per write.

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

### 6.5 Expiry

Expiry is lazy. The retrieval filter treats `expires_at < now` as ineligible before candidate generation. The record, embedding blob, FTS row, and entity links remain available for `include_history` and audit. A CLI command `memlayer expire` flips those rows to `expired` and writes events for tidiness. Nothing depends on the sweep having run.

### 6.6 Evidence validation

One helper, used by both write paths, so that an explicit agent write and an extractor candidate are held to the same standard.

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
        return EvidenceCheck(True, hit.turn, hit.role, supported, f"downgraded from {claimed}: quote is from a {hit.role} turn")
    return EvidenceCheck(True, hit.turn, hit.role, claimed, None)
```

Rules the helper enforces:

- The quote must be found verbatim after whitespace normalization. Fuzzy matching is not allowed; a paraphrase is not evidence.
- The turn the quote is found in caps the source kind. A quote from a user turn supports `user_statement`; a tool turn supports `tool_result`; an assistant turn supports only `agent_inference`. Claiming higher is downgraded, never rejected, and the downgrade is recorded on the event.
- Claiming lower than the turn supports is allowed. An agent may mark a user quote as `agent_inference` if it is unsure what the user meant.
- Without a session, or without a quote, the only kind available is `agent_inference`. `system` never goes through this helper; it is written by the host through a separate API, not by an agent or the extractor.
- The helper returns the matched turn. The caller writes `source_ref = "session:<id><turn:n>"`, so every record points to the exact turn that supports it.

## 7. Vector index

The vector index is in-process, exact, and filtered.

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

On startup, `load` scans `embeddings`. At 50K rows of 4 KB, that is 200 MB and about one second.

## 8. Ingestion

### 8.1 `memory_write` handler (synchronous)

```
1. principal, args -> validate schema
2. scope = args.scope or Scope("agent", principal.agent_id)
   assert scope in writable_scopes(principal)              -> error "scope_not_writable"
3. ev = validate_evidence(principal.session_id, args.evidence, args.source_kind)      (section 6.6)
   source_kind = ev.source_kind; source_ref = f"session:{sid}<turn:{ev.turn}>" if ev.found else None; note = ev.note
   (args.source_kind == "session_summary" is rejected by the schema; "system" is rejected by the handler)
4. resolved = resolve_entities(args.entities, scope, principal)   (section 9)
   if any about-role mention is ambiguous: return error "entity_ambiguous" with the candidate entity ids; nothing is written
   mentions-role ambiguities drop the link and add a note
5. subject = args.subject or f"{primary_entity_ref}/-"
6. vec = embedder.embed([content])[0]
7. if type != "episodic":
       existing = store.active_by_subject(scope, subject)
       if existing: apply supersession rule (6.3); if reinforced: return {"record_id": existing.id, "outcome": "reinforced"}
   near = vector_index.search(vec, allowed=same scope & same type & active & subject != this subject, k=3)
   for n in near where n.cosine >= cfg.ingestion.dedup_candidate_cosine:
       if judge.judge(n.content, content) == "same": reinforce(n); return {"outcome": "reinforced", ...}
8. insert record, embedding, fts row, entity links; update the vector index; append events
9. return {"record_id", "status", "outcome": "created" | "superseded:<old_id>" | "conflict:<old_id>", "note", "timings_ms"}
```

Latency: one embed (25 ms warm on MPS), one or two index searches (under 5 ms), a handful of inserts in one transaction (under 5 ms).

Every stage is timed and the timings go into both the tool response and the event payload, under these names: `permission`, `evidence`, `entities`, `embed`, `dedup_search`, `judge`, `supersession`, `index_update`, `transaction`, `event_log`, `total`. The `judge` stage is the NLI cross-encoder from section 6.3 and is the second largest cost after `embed`; the benchmark reports it separately so its price is visible. The handler is wrapped in the same `Timer` the retriever uses, so the benchmark harness reads write and search timings the same way.

### 8.2 Session extraction (asynchronous)

Triggered by `on_session_end`. Runs in a worker thread or a separate process; the host does not wait.

```
1. turns = store.session_turns(session_id)
   if len(turns) < 2: write only the session summary; return
2. ctx = ExtractionContext(principal, known_entities=aliases readable by the agent, existing_subjects=active subjects in writable scopes)
3. out = extractor.extract(turns, ctx)
4. for cand in out.candidates[:cfg.extraction_max_candidates]:
       a. ev = validate_evidence(sid, cand.evidence, cand.source_kind, turn_hint=cand.evidence_turn)   (section 6.6)
          if not ev.found: reject with reason "evidence_not_found"
          source_kind = ev.source_kind (a claimed user_statement on an assistant turn is downgraded, same as in 8.1)
       b. resolved = resolve_entities(cand.entity_mentions, scope, principal)   (section 9)
          if any about-role mention is ambiguous: reject with reason "entity_ambiguous" and the candidate ids
       c. run steps 5 to 8 of 8.1 with creator_agent_id = principal.agent_id, source_ref = f"session:{sid}<turn:{ev.turn}>"
5. write one episodic record from out.summary:
       subject = f"session:{sid}/-", event_at = session.ended_at, source_kind = "session_summary",
       source_ref = f"session:{sid}", status and confidence and expiry from the policy table (confirmed, 0.8, none)
       entity links from out.summary.entity_mentions, all with role "mentions"; ambiguous ones are dropped, never guessed
6. sessions.extracted_at = now; append event extraction.run with counts {proposed, written, reinforced, superseded, rejected, reasons}
   and timings_ms {transcript_prep, extractor_model, validation, dedup_and_contradiction, writes: [per accepted record], summary_write, total}
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
            out.append(Resolution(m, follow_merges(e), "explicit", [])); continue
        norm = normalize(m.text)                                     # lower, collapse ws, strip diacritics
        hits = store.entities_by_alias(norm, kinds=[m.kind], scopes=readable, status in (provisional, confirmed))
        if len(hits) == 1:
            out.append(Resolution(m, hits[0], "resolved", []))
        elif len(hits) > 1:
            out.append(Resolution(m, None, "ambiguous", [h.id for h in hits]))   # never pick one
        else:
            e = store.create_entity(kind=m.kind, canonical=m.text, scope=scope, status="provisional")
            store.add_alias(e.id, norm); append event entity.created
            out.append(Resolution(m, e, "created", []))
    return out
```

An alias that matches more than one readable entity is an ambiguity, and the system never resolves it by guessing. What happens next depends on the mention's role:

- Role `about`: the record is not written. `memory_write` returns `entity_ambiguous` with the candidate ids so the agent can call again with an explicit `entity_id`, or ask the user. Extraction rejects the candidate with the same reason and logs the ids. A fact attached to the wrong person is the leak the design is built to prevent, so a missing fact is the right failure.
- Role `mentions`: the link is dropped, the record is written without it, and the event notes `ambiguous_mention` with the ids. A dropped mention costs one entity-generator hit; a wrong one poisons entity retrieval for two entities.

Ambiguity is expected to be rare in practice because aliases are scoped, and every occurrence is logged as `ambiguous_alias` so the evaluation can count them. If the count is high, the fix is a merge or a more specific alias, not a heuristic.

Merging is manual: `memory_revise(entity_id=..., merge_into=...)` from an agent with write on both scopes, or the CLI. Merge sets `status=merged`, `merged_into`, repoints `record_entities`, and unions aliases. Every alias lookup follows `merged_into` to the surviving entity.

The primary entity of a record (role `about`) is the first mention with role `about`, or the principal user when none is given and the type is semantic.

## 10. Retrieval pipeline

The stage order matches the HLD diagram: optional rewrite, scope resolution, hard filter, three parallel generators, fusion, freshness, gate, duplicate collapse, optional rerank, budget, explanations, log.

### 10.0 Query rewriting (optional, off by default)

The agent supplies raw queries. When `retrieval.rewrite.enabled` is true, a hosted model turns them into standalone search queries using the raw queries and host-supplied `context`. It has no access to candidate records, the store, or prior search results.

```python
def rewrite_stage(req: SearchRequest) -> tuple[list[str], str]:
    if not cfg.retrieval.rewrite.enabled:
        return req.queries, "disabled"
    ctx = (req.context or "")[: cfg.retrieval.rewrite.max_context_chars]
    try:
        out = rewriter.rewrite(req.queries, ctx)                # one structured-output call, timeout_ms
    except (TimeoutError, RewriteError):
        return req.queries, "failed"                            # raw queries proceed; nothing else changes
    return (out.queries, "applied") if out.queries != req.queries else (req.queries, "unchanged")
```

Rules:

- The rewriter returns the same number of queries it received, each a standalone phrase that names its subject. It may expand a pronoun or a "the second one" reference using `context`; it may not invent subjects absent from both inputs.
- The adapter supplies current-turn context: by default, the last user and assistant turns, truncated. With no context, the rewriter uses the raw queries alone.
- Rewritten queries feed the dense and lexical generators. Entity hints from the request are used as given; the rewriter does not alter them.
- Both raw and rewritten queries are logged and returned in the response, and each `Explanation` carries both, so the evaluation can attribute a hit or miss to the rewrite.
- A failed rewrite is not an error to the agent. The search proceeds on the raw queries and the log records `failed`.

The rewriter is a hosted model call on the hot path, which is why it is off by default. Its latency is timed as its own stage.

### 10.1 Handler

```python
def memory_search(principal, req: SearchRequest) -> SearchResponse:
    t = Timer(warm=embedder.is_loaded and vector_index.is_loaded)
    search_id = uuid7()                                   # generated first; it is the search_log primary key and appears in the response
    reranked = None                                       # stays None when the reranker is disabled; the log column is NULL

    queries, rewrite_status = rewrite_stage(req)                                          # 10.0
    t.mark("rewrite")

    scopes = readable_scopes(store, principal.agent_id, principal.user_id, principal.project_id)
    t.mark("scopes")

    eligible = store.eligible_ids(scopes, req.types, req.since, req.until, req.include_history, now)   # SQL, returns set[str]
    allowed = vector_index.mask(eligible)
    t.mark("filter")

    qvecs = embedder.embed(queries); t.mark("embed")

    # the three generators are independent and may run concurrently (open item 7); each is timed separately either way
    dense = {}                                   # record_id -> max cosine
    for qv in qvecs:
        for rid, cos in vector_index.search(qv, allowed, cfg.per_generator_k):
            dense[rid] = max(dense.get(rid, -1), cos)
    t.mark("dense")

    lexical = lexical_search(queries, eligible, cfg.per_generator_k)         # 10.2
    t.mark("lexical")

    entity_ids = resolve_aliases(req.entities, scopes) + entities_in_queries(queries, scopes)
    entity_hits = store.records_for_entities(entity_ids, eligible, order="event_at desc", limit=cfg.per_generator_k)
    t.mark("entity")

    candidates = rrf([ranked(dense), ranked(lexical), ranked(entity_hits)], k=cfg.rrf_k)   # 10.3, sets rrf_score and fused_rank
    t.mark("fuse")

    candidates = apply_freshness(candidates, store, now)                                  # 10.4, sets freshness_multiplier and score
    t.mark("freshness")

    kept, gated_out = gate(candidates, queries)                                           # 10.5, sets gate_reason
    t.mark("gate")

    kept, deduped_out = collapse_duplicates(kept, vector_index, cfg.dedup_cosine)         # 10.6
    t.mark("dedup")

    if cfg.reranker.enabled:
        kept, reranked = rerank(kept[:cfg.reranker.candidates], queries)                  # 10.7, scores every query, keeps the max per record
    t.mark("rerank")

    chosen, budget_out = fill_budget(kept, req.k, cfg.token_budget)                       # 10.8
    t.mark("budget")

    results, empty_reason = explain(chosen, candidates, req.queries, queries, rewrite_status, gated_out, deduped_out)   # 10.9
    t.mark("explain")

    log_search(principal, req, queries, rewrite_status, scopes, dense, lexical, entity_hits,
               candidates, gated_out, deduped_out, reranked, budget_out, results, empty_reason, cfg.flags(), t)
    t.mark("log")
    return SearchResponse(search_id, req.queries, queries if rewrite_status == "applied" else None,
                          rewrite_status, results, empty_reason, t.as_dict())
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

### 10.7 Reranker (optional, off by default)

Runs only on survivors of the gate and duplicate collapse, capped at `reranker.candidates`; it never scores the whole store. With several queries, it scores each survivor against each query and keeps the record's maximum score. This matches dense search, which keeps each record's maximum cosine across queries: a record needs to answer one requested phrase well. Cost is `len(queries) * len(kept)` pairs, at most 90 with three queries. Sort descending and record `rank_before`, `rank_after`, `score`, and the query index that produced the score. Drop candidates below `reranker.floor`. When enabled, that floor is the final gate; dense and lexical floors only pre-filter candidates. Because the floor is unset until its experiment calibrates it, enabling the reranker without one is a configuration error.

The initial latency budget is `reranker.budget_mean_ms` (100 ms mean for 30 candidates on the target laptop). The benchmark reports measured p50 and p95 by candidate count and hardware, and those numbers replace the budget in the HLD latency table. The reranker becomes the default only if the evaluation shows it improves which records enter context and the downstream task outcome.

### 10.8 Budget fill

Token count is `len(content) // 4` plus 30 for the envelope line. Walk in order, stop when either `k` results or the budget is reached. Never truncate a record's content; a record that does not fit is skipped and the next one is tried.

### 10.9 Explanations and result formatting

`explain()` builds one `Explanation` per returned record from the `Candidate` it came from, plus the response-level `empty_reason` when nothing was returned. The full object goes into the structured payload and the `search_log` row. The model sees only the `summary` line, so the tool result stays short.

Each result is rendered to the tool result as one block:

```
[mem_01J...] semantic · confirmed · user_statement · event 2026-08-12 · scope user:aditya
Prefers concise technical explanations without preamble.
matched: dense 0.71 (rank 2), lexical 3/3 (rank 1); fused rank 1; passed gate on dense
```

When the rewriter changed the query, the response header says so: `searched for: "user's preferred answer length" (rewritten from "what does he prefer")`. When the result is empty, the header carries `empty_reason`.

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

`context` is deliberately absent from the schema. The agent never supplies it; the adapter attaches the current-turn context to the `SearchRequest` before the handler runs (section 14). Exposing it to the agent would invite it to paste the conversation into the search call.

### `memory_get`

`{"ids": [string]}`. Returns full records including superseded lineage and conflicts, so the agent can inspect the source or show a disagreement to the user.

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
  "entities": {"type": "array", "items": {"type": "object", "properties": {"kind": {...}, "name": {...}, "role": {"enum": ["about", "mentions"]},
               "entity_id": {"type": "string", "description": "Pass this when a previous call returned entity_ambiguous, or when memory_get showed you the exact entity."}}}},
  "tags": {"type": "array", "items": {"type": "string"}}
}
```

The description tells the agent: write only what would change a future action, one fact per call, and never write `system` or `session_summary` (those source kinds are reserved for the host and the extractor, and the enum does not offer them). On `entity_ambiguous`, the error lists the candidate entity ids with their kinds, canonical names, and scopes; the agent either passes one back as `entity_id` or asks the user which one they mean.

### `memory_revise`

`{"id": string, "content"?: string, "action": "confirm" | "supersede" | "expire", "reason": string}`. `supersede` requires `content` and creates a new record with `supersedes_id`; `confirm` promotes provisional to confirmed and requires the principal to have write on the record's scope; `expire` sets `expired`. All three append events with the reason. Entity merge is a separate argument shape on the same tool: `{"entity_id", "merge_into", "reason"}`.

### `memory_forget`

`{"id": string, "reason": string}`. Sets `deleted`, keeps the row as a tombstone, removes the FTS row, and marks the in-memory vector dead. The record’s durable content and embedding remain until the controlled erase operation removes them. Content erasure is an admin CLI operation (`memlayer erase <id>`), not a tool, because a user's deletion request should be honoured by a human-controlled path, not an agent's judgement.

## 12. Session buffer and ingestion hooks

Adapters call three hooks. The memory layer stores turns as they arrive so evidence checks in `memory_write` and end-of-session extraction see the same text.

```python
def on_session_start(principal): store.create_session(...)
def on_turn(principal, role, content): store.append_turn(...)        # tool turns store the tool result text, not the call
def on_session_end(principal): store.end_session(...); schedule(extract_session, principal.session_id)
```

If a framework cannot signal session end reliably, the adapter uses an idle timeout (default 30 minutes); a later turn starts a new session. Every `source_ref` carries the session id, so the only consequence of a split is a less useful summary.

## 13. Memory-use policy text

The only memory-related content in the prompt prefix. Kept short, and kept stable so the prefix caches.

> You have long-term memory available through tools. Before acting on anything that could depend on the user's preferences, earlier decisions, or previous sessions, call `memory_search` with one to three specific phrases. Do not search for general knowledge or for facts already visible in this conversation. When results come back, check their status, source, and date before relying on them; a provisional or old record may be wrong, and you can ask the user. When the user states a preference, a fact about themselves, or a decision, save it with `memory_write` and quote their words as evidence. Save decisions you make together as episodic records with the reason. Do not save guesses as facts.

This text is versioned with the extractor prompt; the evaluation records both versions on every run.

## 14. Adapters

### Deep Agents

- Tools: register the five JSON schemas as LangChain tools; handlers receive the run config and derive `Principal` from `configurable.agent_id`, `configurable.user_id`, `configurable.thread_id` (as session id).
- Hooks: `on_turn` from a callback on each human and AI message; `on_session_end` from an explicit call by the host or the idle timeout.
- Search context: on every `memory_search` call the adapter sets `SearchRequest.context` to the last user turn plus the last assistant turn from the session buffer, truncated to `retrieval.rewrite.max_context_chars`. It does this whether or not the rewriter is enabled, so enabling the flag later needs no adapter change.
- Tool result formatting: the block format in 10.9, joined with blank lines, prefixed by a one-line count and the search header.

### CrewAI

- Tools: wrap each handler in a `BaseTool` subclass. Agent id is the CrewAI agent role slug; user id and session id come from crew inputs, which the host must supply.
- Hooks: CrewAI step callbacks give per-step output, not per-turn dialogue; the adapter records the task description as the first user turn and each step output as an assistant turn. This is coarser than Deep Agents and is part of what the experiment measures.
- Search context: the task description plus the most recent step output, truncated. Coarser than the Deep Agents context for the same reason as the hooks.

Both adapters are thin. Their obligations are exactly four: register the tool schemas, derive the `Principal`, call the three session hooks, and attach the current-turn `context` to each search. If either needs to store framework-specific state in `records`, that is a contract failure and the contract, not the adapter, gets fixed.

## 15. Latency accounting

For `memory_search` with two queries, no reranker, 50K records, Apple M-series:

| Stage | Expected | Dominant cost |
| --- | --- | --- |
| rewrite | 0 ms off; 300 to 800 ms on | A hosted structured-output call. This is why the flag is off by default. |
| scopes, filter | 3 ms | One indexed SQL query returning ids; building the mask. |
| embed | 20 to 40 ms | bge-m3 forward pass for two short texts, warm. |
| dense | 2 to 5 ms | Two matmuls over the matrix. |
| lexical | 3 to 8 ms | Two FTS5 queries, intersection. |
| entity | 1 to 3 ms | Alias lookup plus an indexed join. |
| fuse, freshness, gate, dedup | under 2 ms | Python over at most 90 candidates. |
| rerank | 0 ms off; 100 ms mean budget on | Cross-encoder over at most 30 survivors. Measured p50 and p95 replace the budget. |
| budget, explain | under 1 ms | Building at most 20 explanation objects. |
| log | 1 to 2 ms | One insert with JSON columns. |
| total, flags off | 35 to 65 ms | |

These are design estimates, not measurements. With both flags off, embedding dominates latency. Keep the model loaded and cache exact-string query embeddings for the life of the process. A smaller query-side encoder is out of scope.

`memory_write` is the same embed cost plus one small index search and one transaction.

### Benchmark instrumentation

Use the same `Timer` for search, writes, and extraction. Search timings live in `search_log.timings_ms`; write and extraction timings live in event payloads. Each benchmark run also records corpus size, generator and reranker candidate counts, model names and versions, feature flags, hardware, and whether the first call was cold or warm. The `warm` column separates cold-start calls afterward.

The harness must report dense, lexical, entity, rewrite, and reranker timing as separate distributions, not only end-to-end totals, and must run the search benchmark at several store sizes and the reranker benchmark at several candidate counts.

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
| Evidence | One parametrized suite runs the same cases through both `memory_write` and extraction: quote not found downgrades to `agent_inference`; quote from an assistant turn with claimed `user_statement` downgrades with a note; quote from a tool turn supports `tool_result` but not `user_statement`; claiming lower than the turn supports is kept as claimed; `source_ref` points at the matched turn. |
| Write | Duplicate judged `same` reinforces instead of inserting. Second reinforcement confirms a provisional record. `session_summary` and `system` are rejected as `source_kind` on the tool. |
| Supersession | Higher-rank new fact supersedes regardless of event time. Equal rank with later `event_at` supersedes. Equal rank with earlier `event_at` is inserted as superseded on arrival and never returned by default. Lower-rank contradiction is stored provisional with conflict rows. Episodic never supersedes. Two opposing preferences with cosine above the dedup floor are judged `contradicts`, not `same` (table-driven `FakeJudge`). |
| Extraction | Candidate with missing evidence is rejected and logged. Session summary is written even with zero candidates, with `source_kind = session_summary`, confirmed, no expiry. Candidate with an ambiguous about-role entity is rejected with the candidate ids. |
| Entities | Alias resolves within readable scopes only. Unknown alias creates a provisional entity in the writer's scope. Alias matching two readable entities returns `entity_ambiguous` on an about-role mention and writes nothing; on a mentions-role mention the link is dropped and the record is written. Explicit `entity_id` skips resolution and must be readable. Merge repoints links and unions aliases. |
| Retrieval | Superseded records excluded by default and included with `include_history`. Expired records excluded. Scope filter applied before dense search. RRF handles an empty generator. Gate returns empty with a reason. Duplicate collapse keeps the higher rank. Budget never truncates content. |
| Freshness | Episodic decays, semantic does not. Time window disables decay. |
| Rewrite | Disabled flag leaves queries untouched and logs `disabled`. Enabled with `FakeRewriter` logs both raw and rewritten queries and searches on the rewritten ones. Rewriter timeout or error falls back to raw queries and logs `failed`. Rewriter receives only queries and context, never candidates (asserted on the fake's call args). Missing context runs the rewriter on queries alone. |
| Reranker | Enabled without a floor is a configuration error. Enabled with `FakeReranker` runs only on gate and dedup survivors, caps at `reranker.candidates`, records rank before and after. |
| Explanations | Every returned record has an `Explanation` whose generator ranks and scores match the `search_log` columns. Empty response carries `empty_reason` naming the missed floors. |
| Log | Every stage of a search is reconstructible from one `search_log` row, including rewrite status, freshness multipliers, and budget leftovers. Every write event carries `timings_ms` with all named stages. |
| Latency | Benchmark test over a 50K synthetic store asserts p50 under 80 ms with both flags off and the fake embedder replaced by a fixed 25 ms sleep. `warm` is 0 on the first call and 1 afterwards. |

Integration tests run once with the real embedder against a 1K-record fixture and check that the calibrated dense floor separates a hand-labelled set of 50 relevant and 50 irrelevant queries.

## 18. Open items carried into the implementation plan

1. Stopword list and tokenization for lexical search: start with FTS5 defaults plus a 100-word English list; revisit when multilingual cases enter the evaluation.
2. Whether `memory_search` should accept a `subject` filter for exact attribute lookups. Cheap to add; decide once the evaluation shows whether agents use it.
3. Idle-timeout session splitting versus explicit end: the Deep Agents adapter should support both from day one.
4. The reranker floor is unset until the reranker experiment runs. Enabling the flag before then is a configuration error by design.
5. Whether the session summary should be chunked for long sessions. Start with a single record capped at 1200 characters.
6. The rewriter prompt: how much latitude it has to expand a query beyond pronoun and reference resolution. Start narrow (resolve references, name the subject, keep the count) and widen only if the follow-up-question cases show misses.
7. Whether the three generators should run in a thread pool or sequentially. Sequential is simpler and the generators are single-digit milliseconds each; measure before adding the pool.
8. The equivalence judge's floors (`entail_floor`, `contradict_floor`) are starting values for `nli-deberta-v3-small`. Calibrate them on a hand-labelled set of same, contradicting, and distinct pairs before trusting the write path, and re-calibrate on any model change, the same way the gate floors follow the embedder.
9. Whether higher rank should always beat a later event time in supersession (section 6.3). The current rule says yes. The evaluation's stale-record cases should include a recent tool result versus an older user statement on the same subject to test it.
