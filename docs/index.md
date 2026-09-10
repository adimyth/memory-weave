# Retold

**Long-term memory for AI agents, with evidence, isolation, and permission to return nothing.**

Retold stores durable facts, decisions, and experiences in SQLite. Local embedding, lexical, and entity retrieval can return an explained result or nothing; source quotes, lifecycle rules, and scope checks keep the store useful as it grows.

## Try it

```bash
python -m pip install "retold[local-models] @ https://github.com/adimyth/retold/releases/download/v1.2.1/retold-1.2.1-py3-none-any.whl"
```

```python
from retold import Retold

with Retold.open("memory.sqlite", profile="lite") as retold:
    with retold.session(user_id="aditya") as memory:
        memory.remember(
            "I prefer concise answers.",
            evidence="I prefer concise answers.",
            attribute="answer_style",
        )
        print(memory.search("What kind of answers do I prefer?").text)
```

The first run downloads about 640 MB of local model files. It needs no API key. Start with the [usage guide](guide/using.md), then read the [architecture](guide/architecture.md) or [API reference](api/index.md) when you need the lower-level controls.

## What Retold protects

<div class="grid cards" markdown>

-   **Record quality**

    A direct claim keeps the transcript quote that supports it. Inferences expire unless later evidence reinforces them.

-   **Retrieval quality**

    Dense, lexical, and entity channels feed a relevance gate that can return nothing and explain why.

-   **Isolation**

    Private agent-user memory needs no setup. Shared user, project, and organization scopes require grants before ranking begins.

-   **Prompt stability**

    The default mode exposes memory as tools. The opt-in utility-aware mode admits conditional memory only when it would change a draft.

</div>

## Current boundary

Retold 1.1 is a Python 3.12 or 3.13 library backed by one SQLite file. Background extraction and utility-aware admission require hosted model clients. The utility-aware measurements come from controlled offline fixtures, so production adopters should start that path in shadow mode.
