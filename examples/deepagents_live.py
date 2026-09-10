"""Run Retold with Deep Agents, Anthropic, and real local memory models.

Install the local-models, live, and deepagents extras, then set ANTHROPIC_API_KEY before running this file.
"""

import os
from typing import Any, cast

from deepagents import create_deep_agent
from langchain_anthropic import ChatAnthropic

from retold import MemoryHost, Retold, Scope
from retold.adapters.deepagents import DeepAgentsMemoryAdapter


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY before running this example.")

    with Retold.open("memory.sqlite") as retold:
        MemoryHost(retold.store).grant("assistant", Scope("user", "aditya"), read=True, write=True)
        adapter = DeepAgentsMemoryAdapter(retold.runtime)
        model = ChatAnthropic(model_name="claude-haiku-4-5-20251001", timeout=None, stop=None)
        agent = create_deep_agent(
            model=model,
            tools=adapter.tools(),
            system_prompt=adapter.system_prompt("You are a helpful assistant."),
            middleware=[adapter.middleware()],
        )
        run = {"configurable": {"thread_id": "demo-1", "agent_id": "assistant", "user_id": "aditya"}}
        response = agent.invoke(
            cast(Any, {"messages": [{"role": "user", "content": "Remember that I prefer concise answers."}]}),
            config=cast(Any, run),
        )
        print(response["messages"][-1].content)
        worker = adapter.end_session(run)
        if worker is not None:
            worker.join()


if __name__ == "__main__":
    main()
