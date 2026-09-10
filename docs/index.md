# Retold

**Long-term memory for AI agents that stays quiet until it changes the answer.**

Retold gives an agent durable, scoped memory that outlives a conversation. Records are facts, decisions, and dated experiences about a user or a project, each carrying a verbatim evidence quote, a source, and a lifecycle. Everything lives in one SQLite file. Every search can return nothing, and every result can explain why it got through.

```bash
pip install "retold[local-models]"
```

## Three rules

<div class="grid cards" markdown>

-   **A bad record is worse than a missing one**

    ---

    A claim attributed to the user must be backed by a quote from the transcript that entails it. Otherwise it is an inference that expires unless it is seen again. Contradictions are settled by source authority, and the losing record stays as lineage.

-   **An irrelevant retrieval is worse than an empty one**

    ---

    Search ends with a gate that can return nothing and says why. For host-issued memory a judge admits a record only if it would change a draft answer written without it.

-   **Nothing enters the prompt prefix**

    ---

    Memory arrives as a tool result, so provider-side prompt caching keeps working. The one exception is a small profile of confirmed response preferences, fixed at session start.

</div>

## Sixty seconds

```python
from retold.config import load_config
from retold.host import MemoryHost
from retold.models import Principal, Scope
from retold.runtime import build_runtime
from retold.store import Store

store = Store("memory.sqlite")
MemoryHost(store).grant("assistant", Scope(kind="user", id="aditya"), read=True, write=True)
runtime = build_runtime(load_config(), store)

me = Principal("assistant", "aditya", "session-1", None)
runtime.hooks.on_session_start(me)
runtime.hooks.on_turn(me, "user", "I prefer concise answers.")
runtime.handlers.memory_write(me, {
    "type": "semantic", "content": "Aditya prefers concise answers.", "attribute": "answer_style",
    "source_kind": "user_statement", "evidence": "I prefer concise answers.",
})
print(runtime.handlers.memory_search(me, {"queries": ["answer style"]}))
runtime.hooks.on_session_end(me)
```

Five tools do the same work from inside an agent: `memory_search`, `memory_get`, `memory_write`, `memory_revise`, `memory_forget`. Any framework can register them; the [Deep Agents and CrewAI adapters](guide/using.md) show how, deriving the caller's identity from the run and never from tool input.

## How a turn is decided

The similarity gate answers "is this record about the same subject". It cannot answer "does this turn need memory", and it was measured failing at that on every threshold. The utility-aware host path answers the second question:

```text
draft an answer without conditional memory
  -> a planner names the information the draft is missing, from a content-free inventory
  -> retrieve against those gaps, not the turn
  -> a judge admits a record only if it changes the draft
  -> regenerate once, or serve the draft
```

Every stage fails closed to the draft. On scripted blind splits it injects on at most 1 of 20 ordinary turns, recalls 9 or 10 of 10 explicit stored facts, and admits nothing unsafe. It is off by default until a consuming host validates it on real traffic. [The evidence](acceptance-report.md).

## Where to go next

| | |
| --- | --- |
| [Architecture](guide/architecture.md) | The record, both write paths, the search pipeline, the host decision, trigger modes, isolation. |
| [Using it](guide/using.md) | Install, wire a host, use an adapter, operate the CLI, run the suites. |
| [Configuration](guide/configuration.md) | Every key an integrator is likely to touch, with defaults and what was measured about them. |
| [What has been measured](guide/measured.md) | Each claim and the run it comes from. |
| [Design documents](components.md) | Components, high-level and low-level design, the utility-aware architecture. |
| [API reference](api/index.md) | Generated from the docstrings. |

Version 1.0.1, MIT licensed, [source on GitHub](https://github.com/adimyth/retold). Python 3.12, one process, one database file.
