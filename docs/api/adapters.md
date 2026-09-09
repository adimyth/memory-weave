# Adapters

Each adapter wires one `MemoryRuntime` into a framework and owns nothing else: identity derivation, turn capture, tool registration, result rendering, and the host-issued search in `auto` and `hybrid`. Both pass one shared contract suite. Install with `--extra deepagents` or `--extra crewai`.

::: retold.adapters.base

::: retold.adapters.deepagents
    options:
      members:
        - DeepAgentsMemoryAdapter
        - RetoldMiddleware

::: retold.adapters.crewai
    options:
      members:
        - CrewAIMemoryAdapter
        - MemoryCrewTool
        - MemoryLLM
