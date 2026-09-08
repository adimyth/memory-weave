"""Provider-neutral structured completions for the extractor and the reviewer.

The core package never depends on a model vendor at import time. A ``CompletionClient`` turns a system
prompt and a user message into text; the two hosted clients here wrap the official SDKs behind lazy imports
and are chosen by model name. Both ask for one JSON object and ``parse_json_object`` tolerates the fences
and prose a model sometimes wraps around it.
"""

from __future__ import annotations

import json
from typing import Any, Protocol


class CompletionClient(Protocol):
    def complete(self, system: str, user: str, *, timeout_s: float) -> str: ...


class StructuredOutputError(RuntimeError):
    """The model's reply was not the JSON object the caller asked for."""


class AnthropicCompletionClient:
    """One Messages API call through the official ``anthropic`` SDK; installed with the ``live`` extra."""

    def __init__(self, model: str, *, max_output_tokens: int) -> None:
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._client: Any = None

    def complete(self, system: str, user: str, *, timeout_s: float) -> str:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        response = self._client.with_options(timeout=timeout_s).messages.create(
            model=self._model,
            max_tokens=self._max_output_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAICompletionClient:
    """One chat completion in JSON mode through the official ``openai`` SDK; installed with the ``live`` extra."""

    def __init__(self, model: str, *, max_output_tokens: int) -> None:
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._client: Any = None

    def complete(self, system: str, user: str, *, timeout_s: float) -> str:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            max_completion_tokens=self._max_output_tokens,
            timeout=timeout_s,
        )
        return str(response.choices[0].message.content or "")


def completion_client_for(model: str, *, max_output_tokens: int) -> CompletionClient:
    """Pick the SDK by model name: ``claude-*`` goes to Anthropic, anything else to OpenAI."""

    if model.startswith("claude-"):
        return AnthropicCompletionClient(model, max_output_tokens=max_output_tokens)
    return OpenAICompletionClient(model, max_output_tokens=max_output_tokens)


def parse_json_object(text: str) -> dict[str, Any]:
    """Return the one JSON object in ``text``, allowing a code fence or surrounding prose."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise StructuredOutputError("response contains no JSON object") from None
        try:
            loaded = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as error:
            raise StructuredOutputError(f"response is not valid JSON: {error}") from None
    if not isinstance(loaded, dict):
        raise StructuredOutputError("response JSON is not an object")
    return loaded
