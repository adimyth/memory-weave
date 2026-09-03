from __future__ import annotations

from datetime import UTC, datetime

from memory_weave.models import EntityMention, Principal, Scope, SearchRequest, Turn


def test_core_models_preserve_the_framework_neutral_contract() -> None:
    scope = Scope(kind="user", id="aditya")
    mention = EntityMention(kind="person", text="Aditya", role="about")
    principal = Principal(
        agent_id="research-agent",
        user_id="aditya",
        session_id="session-1",
        project_id="memory-weave",
    )
    request = SearchRequest(
        queries=["Aditya explanation preference"],
        context=None,
        types=["semantic"],
        entities=["Aditya"],
        since=None,
        until=None,
        k=8,
        include_history=False,
    )
    turn = Turn(
        session_id="session-1",
        turn=1,
        role="user",
        content="Keep answers concise.",
        at=datetime(2026, 9, 3, tzinfo=UTC),
    )

    assert scope.kind == "user"
    assert mention.entity_id is None
    assert principal.project_id == "memory-weave"
    assert request.k == 8
    assert turn.role == "user"
