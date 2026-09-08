"""Routing for the benchmark Models wrapper."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from benchmarks.draft_delta_experiment import Models


class _FakeCompletions:
    def __init__(self, owner: _FakeOpenAI) -> None:
        self._owner = owner

    def create(self, **kwargs: object) -> object:
        self._owner.requests.append(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5),
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
        )


class _FakeOpenAI:
    def __init__(self, **kwargs: object) -> None:
        self.options = kwargs
        self.requests: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=_FakeCompletions(self))


def _install_openai(monkeypatch: pytest.MonkeyPatch, constructed: list[_FakeOpenAI]) -> None:
    def constructor(**kwargs: object) -> _FakeOpenAI:
        client = _FakeOpenAI(**kwargs)
        constructed.append(client)
        return client

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=constructor))


def test_models_routes_openrouter_prefix_to_openrouter_client(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed: list[_FakeOpenAI] = []
    _install_openai(monkeypatch, constructed)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.test")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "Memory Weave Test")
    monkeypatch.setenv("OPENROUTER_PROVIDER", "deepinfra")

    text = Models().complete("openrouter:anthropic/claude-sonnet-4.6", "system", "user")

    assert text == "ok"
    assert constructed[0].options == {
        "api_key": "openrouter-test-key",
        "base_url": "https://openrouter.ai/api/v1",
        "default_headers": {
            "HTTP-Referer": "https://example.test",
            "X-OpenRouter-Title": "Memory Weave Test",
        },
    }
    assert constructed[0].requests[0]["model"] == "anthropic/claude-sonnet-4.6"
    assert "extra_body" not in constructed[0].requests[0]


def test_models_keeps_unprefixed_names_on_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed: list[_FakeOpenAI] = []
    _install_openai(monkeypatch, constructed)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")

    Models().complete("gpt-5.4", "system", "user")

    assert constructed[0].options == {}
    assert constructed[0].requests[0]["model"] == "gpt-5.4"


def test_models_requires_openrouter_key_and_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed: list[_FakeOpenAI] = []
    _install_openai(monkeypatch, constructed)
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    models = Models()

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        models.complete("openrouter:anthropic/claude-sonnet-4.6", "system", "user")
    with pytest.raises(ValueError, match="slug"):
        models.complete("openrouter:", "system", "user")
    assert constructed == []


def test_openrouter_json_mode_extracts_the_object_from_fenced_or_wrapped_replies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.draft_delta_experiment import _extract_json_object

    fenced = (
        "Here you go:\n```json\n"
        '{"admitted": ["F1"], "verdicts": [{"id": "F1", "reason": "a } inside \\" quotes"}]}\n'
        "```\nDone."
    )
    assert (
        _extract_json_object(fenced)
        == '{"admitted": ["F1"], "verdicts": [{"id": "F1", "reason": "a } inside \\" quotes"}]}'
    )
    assert _extract_json_object("no json here") == "no json here"
