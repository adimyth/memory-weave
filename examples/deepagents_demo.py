"""A short scripted conversation through the Deep Agents adapter, with the memory tool calls visible.

Runs with fakes for every model so it needs no keys and no weights:

    uv run --extra deepagents python examples/deepagents_demo.py

Swap the scripted model for a real one and the fake embedder and judge for the local models to run it for
real; the adapter does not change.
"""

from __future__ import annotations

import tempfile
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from retold.adapters.deepagents import DeepAgentsMemoryAdapter
from retold.config import EmbeddingConfig, RetoldConfig
from retold.host import MemoryHost
from retold.index.embedder import FakeEmbedder
from retold.ingest import FakeExtractor, FakeJudge, TableReviewer
from retold.models import ExtractionOutput, Scope, SessionSummary
from retold.runtime import build_runtime
from retold.store import Store


class ScriptedChatModel(BaseChatModel):
    queue: deque[AIMessage] = deque()

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.queue.popleft())])

    @property
    def _llm_type(self) -> str:
        return "scripted"


def main() -> None:
    workdir = Path(tempfile.mkdtemp())
    store = Store(workdir / "memory.sqlite")
    host = MemoryHost(store)
    host.grant("assistant", Scope(kind="user", id="aditya"), read=True, write=True)
    host.provision_user("aditya", aliases=("Aditya",))
    config = RetoldConfig(embedding=EmbeddingConfig(model="fake-embedder", version="1", dims=8))
    embedder = FakeEmbedder(dims=8)
    summary = SessionSummary("Aditya said how they like answers and asked for a Python example.", [], [])
    runtime = build_runtime(
        config,
        store,
        embedder=embedder,
        judge=FakeJudge(),
        extractor=FakeExtractor(ExtractionOutput([], summary)),
        reviewer=TableReviewer(),
    )
    adapter = DeepAgentsMemoryAdapter(runtime)
    model = ScriptedChatModel()
    agent = create_deep_agent(
        model=model,
        tools=adapter.tools(),
        system_prompt=adapter.system_prompt("You are a helpful assistant."),
        middleware=[adapter.middleware()],
        checkpointer=InMemorySaver(),
    )
    run_config = {"configurable": {"thread_id": "demo-1", "agent_id": "assistant", "user_id": "aditya"}}

    preference = "I prefer concise technical answers with a short rationale."
    saved = "Aditya prefers concise technical answers with a short rationale."
    embedder.set_similarity("answer style", saved, 0.9)
    script = [
        (
            preference,
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "memory_write",
                            "args": {
                                "type": "semantic",
                                "content": saved,
                                "source_kind": "user_statement",
                                "evidence": preference,
                                "attribute": "answer_style",
                            },
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Noted: concise, with a short rationale."),
            ],
        ),
        (
            "How should you answer my questions?",
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "memory_search",
                            "args": {"queries": ["answer style"]},
                            "id": "call-2",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Concisely, with a short rationale, as you asked before."),
            ],
        ),
    ]
    for user_text, replies in script:
        model.queue.extend(replies)
        state = agent.invoke(cast(Any, {"messages": [HumanMessage(user_text)]}), config=cast(Any, run_config))
        last_human = max(i for i, message in enumerate(state["messages"]) if message.type == "human")
        for message in state["messages"][last_human:]:
            if message.type == "human":
                print(f"user      > {message.content}")
            elif message.type == "ai" and message.tool_calls:
                for call in message.tool_calls:
                    print(f"tool call > {call['name']} {call['args']}")
            elif message.type == "tool":
                print(f"tool      > {str(message.content).splitlines()[0]}")
            elif message.type == "ai":
                print(f"assistant > {message.content}")
    adapter.end_session(run_config)
    print(f"\nstore: {workdir / 'memory.sqlite'}")


if __name__ == "__main__":
    main()
