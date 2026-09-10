# Next phase: background memory writes

Retold currently writes memory in two ways: an agent can call `memory_write` during a session, and session-end extraction can propose records after the conversation closes. Both paths validate evidence, resolve entities, apply authority and lifecycle rules, and preserve an audit trail.

The next capability worth investigating is a background worker that revisits accumulated sessions and records after the request path has finished. Other memory systems use background processing to consolidate repeated observations, connect related experiences across sessions, and reconsider records when later evidence arrives. Retold should adopt that pattern only where it improves memory quality without weakening provenance.

## Proposed boundary

A background worker may propose changes, but it must not bypass the existing ingestion contract. Every proposal should carry source references, evidence, the policy and model versions that produced it, and a deterministic explanation of the intended operation. The normal evidence, scope, authority, conflict, supersession, activation, and audit rules should still decide what is committed.

Useful first tasks are:

- Consolidate repeated observations from multiple sessions into one durable record while retaining every source reference.
- Detect records whose later evidence suggests a revision, conflict, expiry, or review.
- Produce a compact episodic summary across related sessions without replacing the source records.
- Schedule this work outside the response path so it adds no latency to a user query.

## Conditions for building it

The feature should not ship until an evaluation shows that background proposals improve precision or recall over session-end extraction alone. The evaluation must also demonstrate zero cross-scope leakage, no unsupported claims, preserved lineage, bounded review volume, idempotent retries, and safe recovery after worker failure.

Until those conditions are measured, background consolidation remains a focused research direction rather than a roadmap commitment.
