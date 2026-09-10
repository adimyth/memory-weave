<h1 align="center">Retold</h1>

<p align="center"><strong>Long-term memory for AI agents that stays quiet until it changes the answer.</strong></p>

<p align="center">
  <a href="https://adimyth.in/retold/"><img alt="Documentation" src="https://img.shields.io/badge/docs-adimyth.in%2Fretold-black"></a>
  <a href="https://github.com/adimyth/retold/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/adimyth/retold/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.12 | 3.13" src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue">
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

Retold gives an agent durable, scoped memory in one SQLite file: facts, decisions, and dated experiences about a user or a project, each backed by a verbatim quote and a lifecycle. It retrieves through dense, lexical, and entity channels, returns only what the caller may read, and can return nothing. Every search explains itself.

Retold exposes five tools as plain JSON schemas, so any agent framework can register them. Adapters for Deep Agents and CrewAI ship as worked examples. **[Documentation](https://adimyth.in/retold/)** · `pip install "retold[local-models]"`

## Decide before you retrieve

Every memory layer retrieves by similarity and injects the top results. Similarity answers ***"is this record about the same subject as the turn?"*** It cannot answer ***"does this turn need remembered state?"***, and the two come apart on exactly the turns that matter: the store holds "use Python for code examples" and the user asks "tabs or spaces in Python?". Related, and irrelevant.

Measured on a scripted conversation, the best similarity score per turn looked like this:[^1]

| Turn class | min | median | max |
| --- | :-: | :-: | :-: |
| Ordinary, no memory needed | 0.44 | 0.56 | **0.63** |
| Memory genuinely applies | 0.50 | 0.57 | 0.62 |

No threshold separates them. Every floor trades one useful recall for one unwanted injection.

Retold answers the second question with a different mechanism. The host drafts an answer without conditional memory while a planner identifies any specific stored facts that may be needed. Retold searches only for those facts, and a judge admits a record only if it would change the draft.

![Utility-aware memory path: draft first, retrieve only for specific missing context, and revise only with admitted records](docs/assets/utility-aware-path.svg)

### How the decision works

1. **The host receives the user query.** The host is the application running the agent. It supplies the query, public conversation context, and the fixed ambient profile for the session.
2. **The host starts two operations concurrently.** The response model drafts an answer without conditional memory. At the same time, the planner checks whether answering the query requires a specific user preference, prior decision, constraint, relationship, event, or task state.
3. **The planner sees categories, not memory contents.** It receives a content-free inventory such as `prior decisions` or `project constraints`. When information is missing, it produces a short retrieval query that names the fact needed, such as `user's preferred language for code examples`.
4. **If no stored context is needed, the host serves the original draft.** Retold performs no retrieval and no conditional memory enters the model context.
5. **If stored context may be needed, Retold searches for that specific information.** It first restricts the search to records the caller may read, then retrieves and ranks candidates through its dense, lexical, and entity channels.
6. **The admission judge compares the candidates with the original draft.** A record is admitted only when it would materially correct, complete, or personalize that draft. Topical similarity by itself is not enough.
7. **If no candidate is admitted, the host serves the original draft.** Retrieved but unnecessary records never reach the response model.
8. **If one or more records are admitted, the host revises the answer once.** Only the admitted records are supplied to the response model, and the resulting revision becomes the final response.

Every stage fails closed to the draft: a timeout, a malformed policy reply, an unavailable provider, or an exhausted latency budget serves the draft and logs why. When the planner identifies no missing context, its work usually adds no latency because it runs alongside the initial draft.

> [!IMPORTANT]
> **Measured on two blind scenario splits and a shadow harness.** At most 1 of 20 ordinary turns received memory. Explicit stored facts were recalled 10 of 10 and 9 of 10 times, implicit needs 7 of 8 and 6 of 7. Nothing placebo, misleading, private, or unsafe was ever admitted.

Two more things make this work in practice:

- **An ambient profile.** Stable response preferences ("answer concisely", "reply in Hindi") never go through this path. An audited activation policy promotes them into a small profile that is fixed at session start.
- **A bundle.** The models, prompts, taxonomy, and retrieval settings that produced those numbers are hashed together. A store serves a bundle only after a passing fitness result is recorded for it, and runs it in shadow mode otherwise.

The path sits in the family opened by [RUMS](https://arxiv.org/abs/2604.14473) and [TRACE-Memory](https://arxiv.org/abs/2608.08446). What Retold adds:

- the ambient-versus-conditional split;
- a content-free inventory the planner sees instead of the records;
- joint, draft-relative admission;
- the control plane around them: shadow mode, per-stage kill switches, a decision log, and bundle gating.

[The comparison in full](docs/design-contributions.md).

## Three rules

- **A bad record is worse than a missing one.** A claim attributed to the user must be backed by a transcript quote that entails it, or it becomes an inference that expires unless it is seen again. Contradictions are settled by source authority, and the losing record stays as lineage.
- **An irrelevant retrieval is worse than an empty one.** Search ends with a gate that can return nothing and says why.
- **Nothing enters the prompt prefix.** Memory arrives as a tool result, so provider-side prompt caching keeps working. The ambient profile is the one exception, and it is fixed for the session.

## Quick start

```bash
pip install "retold[local-models]"     # bge-m3 embedder, NLI judge; add deepagents or crewai for an adapter
```

Three lines open a store, allow an agent to read and write one user's memory, and build the runtime:

```python
from retold import MemoryHost, Scope, Store, build_runtime, load_config

store = Store("memory.sqlite")
MemoryHost(store).grant("assistant", Scope(kind="user", id="aditya"), read=True, write=True)
runtime = build_runtime(load_config(), store)
```

#### Integrating with Deep Agents or CrewAI

Hand the runtime to an adapter and the framework does the rest: tools registered, identity taken from the run, turns captured, extraction scheduled at session end. The CrewAI adapter takes the same runtime; see the [guide](https://adimyth.in/retold/guide/using/).

```python
from deepagents import create_deep_agent
from retold.adapters.deepagents import DeepAgentsMemoryAdapter

adapter = DeepAgentsMemoryAdapter(runtime)
agent = create_deep_agent(model=model, tools=adapter.tools(),
                          system_prompt=adapter.system_prompt("You are a helpful assistant."),
                          middleware=[adapter.middleware()])
agent.invoke({"messages": [...]}, {"configurable": {"thread_id": "s1", "agent_id": "assistant", "user_id": "aditya"}})
```

The grant is the one deliberate step: isolation is enforced by scope and grant before anything is ranked, so an agent reads nothing it was not given.

#### Integrating with any other framework

The runtime works on its own. `runtime.handlers` is the five tools as Python calls, `runtime.hooks` records the session, and `runtime.handlers.tool_schemas(principal)` gives any framework the schemas to register.

```python
from retold import Principal

me = Principal("assistant", "aditya", "session-1", None)
runtime.hooks.on_session_start(me)
runtime.hooks.on_turn(me, "user", "I prefer concise answers.")
runtime.handlers.memory_write(me, {"type": "semantic", "content": "Aditya prefers concise answers.",
                                   "attribute": "answer_style", "source_kind": "user_statement",
                                   "evidence": "I prefer concise answers."})
runtime.handlers.memory_search(me, {"queries": ["answer style"]})
runtime.hooks.on_session_end(me)          # hands the transcript to extraction
```

To turn on the utility-aware path, build the adapter with `memory_mode="utility_aware"` and the measured bundle from `retold.policy.reference`. The [guide](https://adimyth.in/retold/guide/using/) walks through it, and `examples/` runs both adapters with fakes and no keys.

## How it works

### The record

One envelope for every memory. A current fact is keyed by entity and attribute, so "Aditya's editor" has one live answer and a chain of superseded rows behind it.

| | |
| --- | --- |
| **Type** | `semantic`, `episodic`, `procedural` |
| **Scope** | `user`, `project`, `org`, or a private agent scope; access is a separate grant |
| **Source** | `user_statement` › `system` › `tool_result` › `agent_inference`, ranked by authority, plus the verbatim `evidence` |
| **Status** | `provisional` → `confirmed` → `superseded` / `expired` / `deleted` |
| **Activation** | `conditional` (must pass retrieval and admission) or `ambient` (in the session profile) |
| **Time** | created, event, expiry, stated validity bounds, scheduled review |

A user statement or tool result starts confirmed. An inference starts provisional and expires in 30 days unless it is seen again.

### Two ways in

- **During the session**, the agent calls `memory_write` with content, evidence, and entity mentions. The quote is located in the transcript and checked for entailment with a local NLI model. The record is then compared with what exists and created, reinforced, superseded, or marked conflicting.
- **After the session**, an extractor proposes candidates from the whole transcript. A separate reviewer accepts, rejects, or narrows each one before it takes the same validation path.

There is no per-message extraction. It charges every turn and stores guesses from an unfinished conversation.

### One way out

`memory_search` runs the same pipeline whoever calls it:

1. Filter by scope, lifecycle, type, and time before any candidate exists.
2. Generate candidates from three channels: dense (`bge-m3`), lexical (FTS5 BM25), and exact entity aliases.
3. Fuse by reciprocal rank; decay old episodes.
4. Gate each survivor on its own evidence, or return nothing and say why.
5. Collapse near-duplicates; fill the result within a token budget.
6. Log every candidate, score, decision, and stage timing.

Warm searches take about 23 ms on a thousand records and 74 ms on fifty thousand.

### Who searches

| Mode | Who calls `memory_search` | Status |
| --- | --- | --- |
| `tool_only` | The model, when it decides to. The utility-aware path runs beside it for the turns the model would never search for. | Default |
| `auto`, `hybrid` | The host, on every user turn. | Experimental controls. Refused together with the utility-aware path, since they would put records in front of the model before admission decided anything. |

Full detail, with diagrams: [Architecture](https://adimyth.in/retold/guide/architecture/) · [Components](docs/components.md) · [Low-level design](docs/agent-memory-lld.md)

## Configuration

One YAML file, every value defaulted and validated. These are the keys an integrator is likely to touch; [the low-level design](docs/agent-memory-lld.md#2-configuration) explains all of them.

| Key | Default | Allowed values | What it does |
| --- | --- | --- | --- |
| `retrieval.trigger.mode` | `tool_only` | `tool_only`, `auto`, `hybrid` | Who calls `memory_search`. |
| `retrieval.gate.dense_floor.<type>` | 0.45 semantic, 0.40 episodic, 0.45 procedural, 0.50 session_summary | 0.0 to 1.0 per type | The cosine a dense-only candidate must reach. The defaults sit inside the band whose F1 is within 90% of the best on a labelled fixture. |
| `retrieval.gate.auto.*` | stricter floors, `entity_exempt: false` | the same keys as `gate`, plus `exclude_source_kinds` | The gate for host-issued searches. |
| `retrieval.default_k`, `retrieval.token_budget` | 8, 1500 | positive integers | Results returned and the tool-result ceiling. |
| `embedding.model`, `embedding.version` | `BAAI/bge-m3`, `"1"` | any sentence-transformers model id; any string | Every stored vector carries both. Changing either is a migration (`retold reembed`) that recalibrates the floors. |
| `ingestion.evidence.entail_floor` | 0.70 | 0.0 to 1.0 | How strongly a quote must support a direct claim. |
| `ingestion.provisional_ttl_days`, `ingestion.reinforcements_to_confirm` | 30, 2 | positive integers | Lifecycle of inferences. |
| `ingestion.extraction_model`, `ingestion.review_model` | `claude-haiku-4-5-20251001` | any `claude-*` id (Anthropic) or OpenAI model id | The two hosted models in the background write path. |
| `reranker.enabled`, `retrieval.rewrite.enabled` | `false` | `true`, `false`; the reranker also needs `reranker.floor` | A cross-encoder and a query rewriter. Both built, both measured, neither helped. |

The utility-aware path is configured by the host that owns the model clients, through `UtilityAwareConfig` and `supported_bundle()`, not through this file.

## What the numbers say

- **It stays quiet.** On ordinary turns, the ones public memory benchmarks never test, the utility-aware path injected on at most 1 of 20, against a target of 5%.
- **It still remembers.** Explicit stored facts came back 10 of 10 and 9 of 10 times; implicit needs 7 of 8 and 6 of 7. Every eligible response preference was promoted to the ambient profile, and the one unsafe candidate went to review.
- **It never admitted anything harmful.** Zero placebo, misleading, private, or unsafe records across every run of the suite, and every fail-closed branch is tested.
- **It leaks nothing.** Zero cross-principal violations on a synthetic four-user store and a 1,074-record fixture, checked at every log stage, through `memory_get`, and against grants.
- **It is cheap to call.** Warm search p50 23 ms on 1K records and 74 ms on 50K; write p50 26 ms; four concurrent writers beside two extraction workers with zero lock errors.
- **It knows what does not work.** A cross-encoder reranker and a query rewriter were built and measured in every placement; both removed the records the judge needed and neither is enabled. The only judge that passed the safety bar was one of three frontier models tried, and that result is pinned in the bundle rather than assumed of the next model.

All of it is in the [acceptance report](docs/acceptance-report.md) and the [experiment record](docs/usefulness-gate.md), with the run behind each number.

## Documentation

**Guide:** [Architecture](https://adimyth.in/retold/guide/architecture/) · [Using it](https://adimyth.in/retold/guide/using/) · [Configuration](https://adimyth.in/retold/guide/configuration/) · [API reference](https://adimyth.in/retold/api/)

**Design:** [Components](docs/components.md) · [High-level design](docs/agent-memory-hld.md) · [Low-level design](docs/agent-memory-lld.md) · [Utility-aware architecture](docs/utility-aware-memory-architecture.md) · [What is distinctive](docs/design-contributions.md)

**Evidence:** [Acceptance report](docs/acceptance-report.md) · [Utility-aware experiments](docs/usefulness-gate.md) · [Benchmark handoff](BENCHMARK_HANDOFF.md) · [Experiments](benchmarks/README.md)

## License

MIT. Commits follow [Conventional Commits](https://www.conventionalcommits.org/); after cloning, run `git config core.hooksPath .githooks`.

[^1]: The Phase 9a vertical slice: one fixed twelve-turn conversation, replayed three times through a hosted model in `hybrid` mode, giving 36 host-issued searches. For each search the table takes the highest dense cosine among its candidates and groups the searches by whether the turn needed memory. It is a small sample from one scripted conversation with one embedder, and it was reproduced from scratch with the same shape. It measures the dense channel's best score, not the full gate. Method and per-turn results: [benchmarks/README.md](benchmarks/README.md).
