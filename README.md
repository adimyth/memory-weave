# Agentic Memory System

Agentic Memory System is a research and implementation project for a provider- and framework-neutral long-term memory layer for AI agents.

The system keeps durable memory outside the prompt and exposes it through explicit tools. It is designed to store attributable semantic facts, episodic experiences, and procedural knowledge; retrieve only relevant, authorized records; and explain every write and retrieval decision.

## Current design

- SQLite is the durable source of truth, with vector, full-text, and entity-based retrieval over the same records.
- Memory is tool-mediated rather than automatically injected into an agent’s prompt.
- Every record carries scope, provenance, lifecycle, trust, and lineage information.
- Retrieval combines dense, lexical, and entity candidates; it can return no result when nothing is relevant.
- The project will benchmark every write and retrieval stage, including optional query rewriting and reranking.

## Documents

- [Research notes](agent-memory-working-notes.md) capture the questions, references, and design direction.
- [High-level design](agent-memory-hld.md) records the current architecture and decisions.
- [Low-level design](agent-memory-lld.md) specifies the data model, contracts, and implementation plan.

## Status

The design is documented; implementation and benchmarks come next. The first implementation will test the same memory contract through Deep Agents and CrewAI, and across at least two model providers.
