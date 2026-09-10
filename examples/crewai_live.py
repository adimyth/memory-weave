"""Run Retold with CrewAI, Anthropic, and real local memory models.

Install the local-models, live, and crewai extras, then set ANTHROPIC_API_KEY before running this file.
"""

import os

from crewai import LLM, Agent, Crew, Task

from retold import MemoryHost, Retold, Scope
from retold.adapters.crewai import CrewAIMemoryAdapter


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY before running this example.")

    with Retold.open("memory.sqlite") as retold:
        inputs = {"user_id": "aditya", "session_id": "demo-1"}
        role = "Research Assistant"
        MemoryHost(retold.store).grant("research-assistant", Scope("user", "aditya"), read=True, write=True)
        adapter = CrewAIMemoryAdapter.for_crew(retold.runtime, inputs, role)
        agent = Agent(
            role=role,
            goal="Answer the user's request.",
            backstory=adapter.policy_text("You are a helpful assistant."),
            llm=LLM(model="anthropic/claude-haiku-4-5-20251001"),
            tools=adapter.tools(),
        )
        task = Task(
            description="Remember that I prefer concise answers.",
            expected_output="A short confirmation.",
            agent=agent,
        )
        print(Crew(agents=[agent], tasks=[task], step_callback=adapter.step_callback).kickoff(inputs=inputs))
        worker = adapter.end_session()
        if worker is not None:
            worker.join()


if __name__ == "__main__":
    main()
