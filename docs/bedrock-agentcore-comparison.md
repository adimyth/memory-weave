# Memory Weave vs. Amazon Bedrock AgentCore Memory

This document compares Memory Weave's design against Amazon Bedrock AgentCore Memory, a managed long-term memory service for agents.
It is a comparison, not a roadmap.
Nothing here changes the current design; where AgentCore does something we do not, the note says so and stops.

Sources read for this comparison, all fetched 2026-09-05:

- [Memory terminology](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-terminology.html)
- [Memory types](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-types.html#memory-long-term-memory)
- [RetrieveMemoryRecords API](https://docs.aws.amazon.com/bedrock-agentcore/latest/APIReference/API_RetrieveMemoryRecords.html)
- [GetMemoryRecord API](https://docs.aws.amazon.com/bedrock-agentcore/latest/APIReference/API_GetMemoryRecord.html)
- [Built-in strategies](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/built-in-strategies.html), plus the semantic, user-preference, summary, and episodic strategy pages it links to

Our own design is described in [`components.md`](components.md), [`agent-memory-hld.md`](agent-memory-hld.md), and [`agent-memory-lld.md`](agent-memory-lld.md).

## 1. Terminology mapping

| AgentCore term | Memory Weave equivalent | Notes |
| --- | --- | --- |
| AgentCore Memory (the resource) | The SQLite store, i.e. one Memory Weave deployment | AgentCore treats "a memory" as a provisioned, named resource you configure with strategies. We have one store per deployment; there is no per-resource strategy configuration because there is no strategy concept at all (section 5). |
| Memory strategy | No equivalent | AgentCore's strategies are pluggable extraction policies you attach to a resource. Our extraction path is a single, fixed pipeline that always produces the same shapes: `semantic`, `episodic`, `procedural` records plus one session summary. Section 6 covers this gap directly. |
| Namespace | `scope_kind` + `scope_id`, refined by `subject_entity_id`/`attribute` | AgentCore namespaces are arbitrary hierarchical path strings you design per strategy (`/strategy/{id}/actor/{id}/session/{id}/`), searchable by prefix. Our scope is a fixed four-way enum (`agent`, `user`, `project`, `org`) with no path hierarchy or prefix search; the current-fact key (`subject_entity_id`/`attribute`) does the job of narrowing to "this one thing" that a namespace path does for AgentCore. |
| Memory record | `Record` | Direct equivalent: a structured, identified, retrievable unit of long-term memory. |
| Session | `sessions` row, `session_id` | Direct equivalent: one continuous run, grouping events/turns under one ID. |
| Actor | `Principal` (`agent_id` + `user_id`) | AgentCore's `actorId` is a single ID that can name a human, another agent, or a system. We split that into a compound principal because an agent acts *for* a user, and scope/grant enforcement needs both: `agent:<agent_id>/<user_id>` is our private scope, and it has no AgentCore analogue. |
| Event | `session_turns` row | Direct equivalent: the atomic, immutable, timestamped unit of short-term memory, tied to one actor/session. AgentCore's `CreateEvent` accepts richer payload types (conversational turns or arbitrary structured data); we store role + text. |
| Event metadata | No equivalent | AgentCore lets you attach key-value metadata to an event for filtered retrieval (e.g. "all events mentioning Destination X"). We have no per-turn metadata field; `tags` exist only on `records`, not on `session_turns`. |

The two systems agree closely on the short-term side (session, actor, event) and diverge on the long-term side, where AgentCore organizes around configurable strategies and path-shaped namespaces, and we organize around a fixed record schema, a flat scope enum, and an explicit current-fact key.

## 2. Long-term memory: extraction, consolidation, and the read APIs

### Extraction and consolidation

AgentCore names the same two operations we already use for writes, with one addition:

- **Extraction** – pull candidate insights out of raw events.
- **Consolidation** – decide whether a new insight becomes a new record or updates an existing one.
- **Reflection** – a third, optional step, used only by the episodic strategy (section 4).

This matches our ingestion vocabulary closely. Our [HLD](agent-memory-hld.md#6-ingestion) and [LLD](agent-memory-lld.md) use the same two words for the same two decisions. Where we go further than AgentCore's public description is *how* consolidation decides:

| Consolidation concern | AgentCore (as documented) | Memory Weave |
| --- | --- | --- |
| Same claim as an existing record | Not detailed publicly beyond "writes to a new or existing record" | Equivalence judge (`ingest/equivalence.py`, NLI cross-encoder) checks directed entailment both ways before reinforcing instead of duplicating. |
| Conflicting claim about the same subject | Not detailed publicly | Source-rank table (`user_statement` > `system` > `tool_result`/`session_summary` > `agent_inference`) decides supersession vs. provisional-conflict; both outcomes are logged (`record_conflicts`, `events`). |
| Renamed/aliased attribute for the same fact | Not detailed publicly | Entity-attribute alias scan (`ingestion.max_entity_attributes`) treats e.g. `answer_style` and `explanation_style` as the same current fact and records `attribute_aliased_from`. |
| Trigger | Asynchronous, runs automatically after `CreateEvent` or `IngestData` | Two paths: synchronous `memory_write` (explicit) and asynchronous session-end extraction (implicit). Consolidation logic is shared by both. |

AgentCore's extraction/consolidation is asynchronous and strategy-scoped: each active strategy on a resource runs its own extraction pass. Ours is asynchronous only for the session-extractor path; an explicit `memory_write` consolidates synchronously because the calling agent is waiting on the tool result.

### The read APIs

| AgentCore operation | Purpose | Memory Weave equivalent | Gap |
| --- | --- | --- | --- |
| `GetMemoryRecord` | Fetch one record by ID | `memory_get` tool, backed by `Store.get_record` | We already have this, and it returns more than AgentCore's flat fetch: conflicts (`record_conflicts`) and supersession lineage (`supersedes_id` chain), not just the record body. |
| `ListMemoryRecords` | Enumerate records under a namespace, paginated, no ranking | No agent-facing equivalent | `Store` has internal listing primitives (`get_records`, `eligible_ids`) used by retrieval, but nothing exposed as a tool for "list everything in this scope." This is the one AgentCore read operation we do not cover at all today. |
| `RetrieveMemoryRecords` | Semantic search over records, ranked by relevance, paginated | `memory_search` tool | Same purpose, different mechanism: see below. |

### `RetrieveMemoryRecords` vs. `memory_search`

Both exist to answer "what does the store know that's relevant to this query," and both return a ranked list rather than a raw dump. The differences, without either being clearly better:

- **Ranking source.** AgentCore's `RetrieveMemoryRecords` is described as a semantic search — vector similarity is the entire ranking signal, filtered by `namespace`/`namespacePath` and an optional `metadataFilters` list. `memory_search` fuses three signals (dense, lexical, entity) with RRF, then applies a relevance gate that can return nothing at all. AgentCore's docs do not describe an equivalent "return nothing" gate; its API always returns up to `maxResults` summaries ordered by score.
- **Filters.** AgentCore's `metadataFilters` is a generic `{left, operator, right}` expression list over arbitrary event/record metadata. Ours is a fixed filter set: `types`, `since`/`until` on `event_at`, `entities`, and `include_history`. Generic metadata filtering is more flexible in the abstract; our fixed set is easier to reason about and index (`records_scope`, `records_event`), and matches our smaller, fixed record schema.
- **Pagination.** `RetrieveMemoryRecords` takes `maxResults` (default 20, max 100) and `nextToken`/paginated responses. `memory_search` has no pagination: it returns up to `default_k` records that also fit `token_budget`, and stops there. You already noted we don't need to change this — a single bounded tool result matches the "don't inject unbounded context" design goal in the [HLD](agent-memory-hld.md#1-goals-and-non-goals), whereas AgentCore's API shape assumes a caller who might legitimately want more than 100 records across multiple calls (e.g., building a UI list), which is not our use case.
- **Strategy scoping.** `searchCriteria.memoryStrategyId` lets a caller search within just one strategy's records (e.g., only summaries). We have no strategy concept, so the nearest equivalent is `types` (search only `episodic`, say) — coarser than strategy scoping, since AgentCore could have multiple named strategies of the same underlying type.

### `GetMemoryRecord` — do we need it as a distinct tool?

We already have the underlying capability (`Store.get_record`, exposed through `memory_get`), so the honest answer to "do we have this" is yes, and to "do we need a *separate* API surface for it" is probably not: `memory_get` already does this job and does it with more context (conflicts, supersession) than AgentCore's endpoint returns. If a future consumer needs a minimal, schema-stable "just the record, nothing else" contract — for example, a UI that only wants to resolve an ID it already has — that is a thin wrapper around `Store.get_record`, not new logic, and is worth adding only when something actually needs that narrower contract rather than speculatively.

## 3. Built-in strategies: what "Reflection" and "Instructions" mean

AgentCore frames every strategy as a small pipeline of named steps, each driven by "a system prompt, which is a combination of Instructions and an Output schema":

- **Instructions** guide the model's reasoning: what to look for, what counts as worth keeping, how to phrase the result. This is the natural-language half of the prompt, and it is the part AWS exposes as a configuration field you can override per strategy.
- **Output schema** is the structured shape the model must return (AgentCore's strategies return JSON for semantic/preference facts and XML for summaries and episodes).
- **Reflection** is a step used only by the episodic strategy. It runs after episodes are captured and looks *across* multiple already-extracted episodes to produce a higher-order insight — a pattern, a recurring failure mode, a "this approach worked twice" conclusion — rather than a fact about a single interaction. AWS's own framing: reflections "analyze past episodes to surface insights, patterns, and higher level conclusions," turning "raw experience into guidance the application can use immediately."

So Instructions is the answer to your question 8: it is where AgentCore lets you steer *what the extraction/consolidation/summarization model pays attention to*, without touching code — e.g., telling the summary strategy to focus on decisions and open threads rather than pleasantries, or telling the semantic strategy to prioritize numeric facts. Section 7 covers where the equivalent lives (or would live) in our design.

Reflection is a capability we do not have at all today. Section 5 covers the gap in the context of the episodic strategy specifically.

## 4. Do we cover the four strategies? A precise mapping

You supposed we support all four strategies and store all four kinds of memory. That is roughly true for two of them, folded into an existing type for the third, and clearly not true for the fourth's most distinctive feature (Reflection). Details:

| AgentCore strategy | What it extracts | Memory Weave coverage |
| --- | --- | --- |
| **Semantic memory** | Standalone JSON facts ("order #XYZ-123 is linked to this case") | Covered directly by our `semantic` type: a declarative statement plus an entity link, written via explicit `memory_write` or session extraction. Close to a 1:1 match. |
| **User preference memory** | JSON `{context, preference, categories}` — choices, styles, tastes | Not a distinct type. A preference like "prefers concise answers" is just a `semantic` record whose `attribute` happens to be `explanation_style`. The current-fact key (`subject_entity_id` + `attribute`) does the work AgentCore does with a dedicated schema, but there is no separate record kind, no `categories` field, and no strategy you could disable independently of general semantic facts. |
| **Summary strategy** | Real-time, per-session, chunked XML summaries, retrievable by namespace listing or semantic search | Covered by our mandatory end-of-session summary: one `episodic` record with `source_kind = session_summary`, written once per session, not incrementally chunked during the session. AgentCore can hand back multiple summary chunks per long session via `ListMemoryRecords`; we always produce exactly one summary record per session. |
| **Episodic memory** | Structured per-episode records (situation/intent/assessment/justification), with automatic mid-conversation episode-boundary detection, *plus* Reflection across episodes | Partially covered. Our `episodic` type stores dated events/decisions/outcomes as free text with `event_at`, written by the session extractor or an explicit agent write — but extraction only happens at session end, never mid-conversation, so there is no automatic "episode complete" detection the way AgentCore describes. We have no structured situation/intent/assessment schema. And we have no Reflection step at all: episodic records are never re-analyzed in aggregate to surface cross-episode patterns. |

So the accurate framing is: we cover the *use cases* behind three of the four strategies (facts, preferences-as-facts, session recall) through three memory types and one current-fact mechanism, rather than through four independently configurable extraction pipelines with their own schemas — and we do not have anything resembling Reflection, which is the one genuinely novel idea in AgentCore's strategy set rather than a re-packaging of extraction.

## 5. Four selectable strategies vs. one fixed pipeline

**AgentCore's model:** you attach a subset of strategies to a memory resource at creation time. Only the strategies you enable run extraction/consolidation on each event or session.

Benefits of that model, as described in AgentCore's own docs and API shape:

- **Cost and latency scale with what you actually need.** Each active strategy is effectively its own extraction/consolidation LLM pass over the same short-term events. A pure Q&A bot can enable only semantic memory and pay for one pass, not four.
- **Narrower, predictable namespaces.** Each strategy gets its own namespace tree, so retrieval and access control can be scoped per strategy (`RetrieveMemoryRecords` even accepts a `memoryStrategyId` to search within just one).
- **Independent tuning.** Instructions and output schema are per strategy, so you can tune the summary strategy's prompt without touching the semantic strategy's.
- **Composability.** You can combine strategies ("You can combine multiple strategies when creating memories"), so the four are building blocks, not an exclusive choice.

**Our model:** one fixed extraction pass (per the [LLD](agent-memory-lld.md), a single `Extractor.extract(transcript, context) -> ExtractionOutput`, not yet implemented in code — section 7) proposes candidates across all three record types plus one summary, in one call, at session end. There is no per-agent switch to say "this agent only cares about facts, skip episodic."

Trade-offs, stated plainly rather than as a recommendation to change anything:

- **We pay less per session, in exchange for less separation of concerns.** One hosted call instead of up to four is cheaper and simpler to operate, but an agent that never needs episodic memory still has episodic candidates proposed (and then, per the design, validated and possibly rejected) for it.
- **We get richer consolidation semantics AgentCore doesn't describe** (source-rank-aware supersession, contradiction tracking, attribute aliasing) as a property of the whole store, not per strategy — because we have one ingestion path with one authority model, rather than N independently consolidating strategies that would each need their own conflict-resolution story if they ever extracted overlapping facts.
- **AgentCore's approach scales better to "I only want strategy X" tenants**, which matters for a managed multi-tenant service selling to many different customers with different needs. That scaling problem does not obviously apply to a single-process, single-store deployment the way it does to a managed AWS service — the pressure to let customers opt out of unwanted extraction cost is much stronger when you are billing per strategy-invocation across many tenants than when you are running one local extractor once per session.
- **Instructions being a first-class, per-strategy config field is a real ergonomic win we don't have an equivalent for yet** (section 7), independent of whether you'd want one fixed pipeline or four selectable ones.

## 6. Instructions in our own design: current status

Two places in our design call a hosted model and would need AgentCore-style Instructions: session extraction and query rewriting. As of this comparison, **neither has an instructions file, because neither is implemented yet**:

| Component | Config that names the model | Code that would hold the prompt | Status |
| --- | --- | --- | --- |
| Session extraction | `ingestion.extraction_model: claude-haiku-4-5-20251001` | `ingest/extractor.py`, `StructuredLLMExtractor` | Not present in the repository. Only the `Extractor` protocol and its output dataclasses (`ExtractionContext`, `CandidateRecord`, `SessionSummary`) are specified, in `models.py`. `FakeExtractor` (the test double) is the only thing the design currently requires to exist. |
| Query rewriting | `retrieval.rewrite.model: claude-haiku-4-5-20251001` | `retrieve/rewrite.py`, `HostedLLMQueryRewriter` | Only `NoRewriter` exists today (verified by reading the file); it returns queries unchanged and never calls a model. `HostedLLMQueryRewriter` and its prompt are specified in the HLD/LLD but not written. |

This maps directly onto AgentCore's Instructions/Output-schema split: once these are built, each one needs exactly that — natural-language instructions plus a structured output contract (`ExtractionOutput`'s dataclasses already are that contract for extraction).

One detail in the current design already anticipates this: `ExtractionContext.prompt_version` exists specifically so that a future change to the extraction prompt can be tracked the same way an embedding-model change is tracked by `embeddings.version` — i.e., recorded on the resulting `extraction.run` event, so a behavior change from re-prompting isn't silently invisible in the audit trail.

**Where these should live, when built:** next to the code that uses them, not in `config.yaml`. The YAML config in `config.py` holds thresholds and model *names* (numbers you'd sweep in a calibration pass); the prompt text itself is reasoning guidance, not a tunable, so it belongs as a module-level constant (or a small `ingest/prompts.py` if it grows past one string) in the same module as `StructuredLLMExtractor`/`HostedLLMQueryRewriter`. This has been added to [`components.md`](components.md#15-prompts-and-instructions) as its own section so the location is documented before the code exists, not after.

## 7. Summary

| Dimension | AgentCore | Memory Weave |
| --- | --- | --- |
| Extraction model | Per-strategy, pluggable, async after every event/session | Single fixed pipeline, async at session end, plus sync explicit writes |
| Consolidation | New vs. existing record, per strategy; conflict handling not publicly detailed | Source-rank supersession, NLI-based equivalence/contradiction, entity-attribute aliasing, all logged |
| Reflection | Yes, episodic strategy only: cross-episode pattern extraction | No equivalent |
| Instructions as config | Yes, first-class per-strategy prompt field | Not yet implemented for either extraction or rewriting; when built, belongs in code, not YAML |
| Record identity | Flat record + free-form namespace path | Structured current-fact key (`subject_entity_id` + `attribute`) plus flat scope |
| Get-by-ID | `GetMemoryRecord` | `memory_get` (richer: includes conflicts and supersession lineage) |
| List/enumerate | `ListMemoryRecords`, paginated | No agent-facing equivalent today |
| Semantic search | `RetrieveMemoryRecords`, vector-only, paginated, generic metadata filters | `memory_search`, dense+lexical+entity fused with RRF, gated to allow an empty result, no pagination, fixed filter set |
| Access control | Not covered by the pages reviewed here | Scope + grant table, enforced as a hard SQL filter before ranking |
