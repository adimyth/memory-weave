# API reference

The public surface a host integrates against, generated from the docstrings. Start from `build_runtime`, which wires every component over one store; the adapters, the CLI, and the benchmarks all go through it.

| Page | What it covers |
| --- | --- |
| [Configuration](config.md) | `RetoldConfig` and every section: store, embedding, reranker, retrieval, gate, ingestion, policy, sessions. `load_config` reads one YAML file. |
| [Runtime and host](runtime.md) | `build_runtime` and `MemoryRuntime`; `MemoryHost` for grants and user provisioning. |
| [Models](models.md) | The record envelope, `Principal`, `Scope`, search requests and results, candidates, and explanations. |
| [Tools](tools.md) | The five tool handlers and the JSON schemas the adapters register. |
| [Session hooks and extraction](ingest.md) | `SessionHooks`, the session buffer, the ingestor, and the extraction runner. |
| [Retrieval](retrieve.md) | The retriever, the gate, and the reranker with its timeout and fallback. |
| [Utility-aware policy](policy.md) | Gap and admission protocols, the orchestrator, turn decisions, activation, bundles, and metrics. |
| [Adapters](adapters.md) | The Deep Agents and CrewAI adapters and what every adapter shares. |
| [Operations](operations.md) | Expiry, retention, erasure, re-embedding, snapshots, and due-review flagging. |

Everything not listed here is internal and may change without notice.
