"""Framework adapters. Each translates one framework's state into the core hooks and tools and nothing else.

Importing this package imports no framework. Import ``retold.adapters.deepagents`` or
``retold.adapters.crewai`` directly; each needs its optional extra.
"""

from .base import (
    Adapter,
    MemoryMode,
    TriggerMode,
    TurnContext,
    policy_text,
    principal_from_mapping,
    recalled_memory_block,
    render_tool_result,
)

__all__ = [
    "Adapter",
    "MemoryMode",
    "TriggerMode",
    "TurnContext",
    "policy_text",
    "principal_from_mapping",
    "recalled_memory_block",
    "render_tool_result",
]
