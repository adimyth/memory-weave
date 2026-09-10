"""Run Retold with CrewAI, OpenAI, and the lightweight local-memory profile.

Install the local-models, live, and crewai extras, then set OPENAI_API_KEY before running this file.
The agent model, background extractor, and candidate reviewer all use OpenAI. Retrieval and evidence checks stay local.
"""

from __future__ import annotations

import os
from dataclasses import replace

from crewai import LLM, Agent, Crew, Task

from retold import MemoryHost, Retold, Scope, lite_config
from retold.adapters.crewai import CrewAIMemoryAdapter


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running this example.")

    config = lite_config()
    config = replace(
        config,
        ingestion=replace(
            config.ingestion,
            extraction_model="gpt-4.1-mini",
            review_model="gpt-4.1-mini",
        ),
    )
    with Retold.open("memory.sqlite", config=config) as retold:
        inputs = {"user_id": "aditya", "session_id": "openai-demo-1"}
        role = "Research Assistant"
        MemoryHost(retold.store).grant("research-assistant", Scope("user", "aditya"), read=True, write=True)
        adapter = CrewAIMemoryAdapter.for_crew(retold.runtime, inputs, role)
        agent = Agent(
            role=role,
            goal="Answer the user's request and use memory when it will help later.",
            backstory=adapter.policy_text("You are a careful assistant."),
            llm=LLM(model="openai/gpt-4.1-mini"),
            tools=adapter.tools(),
        )
        task = Task(
            description="Remember that I prefer concise answers, then confirm what you saved.",
            expected_output="A short confirmation.",
            agent=agent,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            step_callback=adapter.step_callback,
            task_callback=adapter.task_callback,
        )
        print(crew.kickoff(inputs=inputs))
        worker = adapter.end_session()
        if worker is not None:
            worker.join()


if __name__ == "__main__":
    main()
