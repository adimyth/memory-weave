CREATE TABLE records (
  id              TEXT PRIMARY KEY,
  type            TEXT NOT NULL CHECK (type IN ('semantic','episodic','procedural')),
  version         INTEGER NOT NULL DEFAULT 1,
  content         TEXT NOT NULL,
  subject         TEXT NOT NULL,
  scope_kind      TEXT NOT NULL CHECK (scope_kind IN ('agent','user','project','org')),
  scope_id        TEXT NOT NULL,
  source_kind     TEXT NOT NULL CHECK (source_kind IN ('user_statement','system','tool_result','session_summary','agent_inference')),
  source_ref      TEXT,
  creator_agent_id TEXT NOT NULL,
  evidence        TEXT,
  created_at      TEXT NOT NULL,
  event_at        TEXT NOT NULL,
  expires_at      TEXT,
  confidence      REAL NOT NULL,
  status          TEXT NOT NULL CHECK (status IN ('provisional','confirmed','superseded','expired','deleted')),
  supersedes_id   TEXT REFERENCES records(id),
  reinforcements  INTEGER NOT NULL DEFAULT 0,
  last_reinforced_at TEXT,
  tags            TEXT NOT NULL DEFAULT '[]'
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
  vector      BLOB NOT NULL
);

CREATE VIRTUAL TABLE records_fts USING fts5(
  record_id UNINDEXED,
  content,
  subject,
  aliases,
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
  alias_norm  TEXT NOT NULL,
  PRIMARY KEY (entity_id, alias_norm)
);
CREATE INDEX entity_aliases_lookup ON entity_aliases(alias_norm);

CREATE TABLE record_entities (
  record_id   TEXT NOT NULL REFERENCES records(id),
  entity_id   TEXT NOT NULL REFERENCES entities(id),
  role        TEXT NOT NULL DEFAULT 'about',
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
  role        TEXT NOT NULL,
  content     TEXT NOT NULL,
  at          TEXT NOT NULL,
  PRIMARY KEY (session_id, turn)
);

CREATE TABLE events (
  id          TEXT PRIMARY KEY,
  at          TEXT NOT NULL,
  kind        TEXT NOT NULL,
  actor       TEXT NOT NULL,
  record_id   TEXT,
  entity_id   TEXT,
  payload     TEXT NOT NULL
);

CREATE TABLE search_log (
  id            TEXT PRIMARY KEY,
  at            TEXT NOT NULL,
  agent_id      TEXT NOT NULL,
  user_id       TEXT NOT NULL,
  session_id    TEXT,
  request       TEXT NOT NULL,
  context       TEXT,
  rewrite_status TEXT NOT NULL,
  rewritten_queries TEXT,
  readable_scopes TEXT NOT NULL,
  dense         TEXT NOT NULL,
  lexical       TEXT NOT NULL,
  entity        TEXT NOT NULL,
  fused         TEXT NOT NULL,
  freshness     TEXT NOT NULL,
  gated_out     TEXT NOT NULL,
  deduped_out   TEXT NOT NULL,
  reranked      TEXT,
  budget_out    TEXT NOT NULL,
  returned      TEXT NOT NULL,
  explanations  TEXT NOT NULL,
  config_flags  TEXT NOT NULL,
  warm          INTEGER NOT NULL,
  timings_ms    TEXT NOT NULL
);
