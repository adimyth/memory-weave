"""The same scripted conversation as the Deep Agents demo, through a CrewAI crew.

Runs with fakes for every model so it needs no keys and no weights:

    CREWAI_DISABLE_TELEMETRY=true uv run --extra crewai python examples/crewai_demo.py
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import Agent, Crew, Task  # noqa: E402
from crewai.llms.base_llm import BaseLLM  # noqa: E402
from pydantic import Field  # noqa: E402

from retold.adapters.crewai import CrewAIMemoryAdapter  # noqa: E402
from retold.config import EmbeddingConfig, RetoldConfig  # noqa: E402
from retold.host import MemoryHost  # noqa: E402
from retold.index.embedder import FakeEmbedder  # noqa: E402
from retold.ingest import FakeExtractor, FakeJudge, TableReviewer  # noqa: E402
from retold.models import ExtractionOutput, Scope, SessionSummary  # noqa: E402
from retold.runtime import build_runtime  # noqa: E402
from retold.store import Store  # noqa: E402


class ScriptedReActLLM(BaseLLM):
    queue: deque[str] = Field(default_factory=deque)

    def __init__(self) -> None:
        super().__init__(model="scripted")

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> Any:
        return self.queue.popleft()

    def supports_stop_words(self) -> bool:
        return False


def main() -> None:
    workdir = Path(tempfile.mkdtemp())
    store = Store(workdir / "memory.sqlite")
    inputs = {"user_id": "aditya", "session_id": "demo-1"}
    role = "Research Assistant"
    host = MemoryHost(store)
    host.grant("research-assistant", Scope(kind="user", id="aditya"), read=True, write=True)
    host.provision_user("aditya", aliases=("Aditya",))
    config = RetoldConfig(embedding=EmbeddingConfig(model="fake-embedder", version="1", dims=8))
    embedder = FakeEmbedder(dims=8)
    summary = SessionSummary("Aditya said how they like answers.", [], [])
    runtime = build_runtime(
        config,
        store,
        embedder=embedder,
        judge=FakeJudge(),
        extractor=FakeExtractor(ExtractionOutput([], summary)),
        reviewer=TableReviewer(),
    )
    adapter = CrewAIMemoryAdapter.for_crew(runtime, inputs, role)
    model = ScriptedReActLLM()
    agent = Agent(
        role=role,
        goal="Answer the user's request.",
        backstory=adapter.policy_text("You are a careful assistant."),
        llm=adapter.wrap_llm(model),
        tools=adapter.tools(),
        verbose=False,
    )

    preference = "I prefer concise technical answers with a short rationale."
    saved = "Aditya prefers concise technical answers with a short rationale."
    embedder.set_similarity("answer style", saved, 0.9)
    write_args = {
        "type": "semantic",
        "content": saved,
        "source_kind": "user_statement",
        "evidence": preference,
        "attribute": "answer_style",
    }
    script = [
        (
            preference,
            [
                f"Thought: I should save this.\nAction: memory_write\nAction Input: {json.dumps(write_args)}",
                "Thought: I now know the final answer.\nFinal Answer: Noted: concise, with a short rationale.",
            ],
        ),
        (
            "How should you answer my questions?",
            [
                'Thought: I should check memory.\nAction: memory_search\nAction Input: {"queries": ["answer style"]}',
                "Thought: I now know the final answer.\nFinal Answer: Concisely, with a short rationale, as you asked.",
            ],
        ),
    ]
    for description, replies in script:
        model.queue.extend(replies)
        task = Task(description=description, expected_output="A short answer.", agent=agent)
        crew = Crew(
            agents=[agent],
            tasks=[task],
            step_callback=adapter.step_callback,
            task_callback=adapter.task_callback,
            verbose=False,
        )
        output = crew.kickoff(inputs=inputs)
        print(f"task      > {description}")
        print(f"assistant > {getattr(output, 'raw', output)}")
    adapter.end_session()
    for turn in store.session_turns("demo-1"):
        print(f"turn {turn.turn} {turn.role:<9} {turn.content.splitlines()[0][:90]}")
    print(f"\nstore: {workdir / 'memory.sqlite'}")


if __name__ == "__main__":
    main()
