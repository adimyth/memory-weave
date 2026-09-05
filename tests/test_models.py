from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

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
    assert request.trigger == "tool"
    assert turn.role == "user"


def test_principal_and_scope_are_immutable() -> None:
    scope = Scope(kind="user", id="aditya")
    principal = Principal(
        agent_id="research-agent",
        user_id="aditya",
        session_id="session-1",
        project_id="memory-weave",
    )

    with pytest.raises(FrozenInstanceError):
        scope.id = "another-user"
    with pytest.raises(FrozenInstanceError):
        principal.user_id = "another-user"


def test_principal_rejects_private_scope_delimiter_collisions() -> None:
    with pytest.raises(ValueError, match="agent_id"):
        Principal(agent_id="a/b", user_id="c", session_id=None, project_id=None)
    with pytest.raises(ValueError, match="user_id"):
        Principal(agent_id="a", user_id="b/c", session_id=None, project_id=None)
