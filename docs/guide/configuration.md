# Configuration

`Retold.open("memory.sqlite")` uses the measured local retrieval defaults. Pass a `RetoldConfig` or the path to a YAML file through `config=` when an application needs different models, budgets, lifecycle settings, or trigger behavior.

```python
from retold import Retold

retold = Retold.open("memory.sqlite", config="retold.yaml")
```

The complete configuration contract and the reasoning behind each default live in [section 2 of the low-level design](../agent-memory-lld.md#2-configuration). The generated [configuration API](../api/config.md) lists every typed field.

The utility-aware path uses host-owned model clients and `UtilityAwareConfig`, so its planner, admission, timeout, and bundle settings do not live in this YAML file. See the [utility-aware architecture](../utility-aware-memory-architecture.md).
