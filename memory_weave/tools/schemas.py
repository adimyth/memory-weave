"""JSON Schema contracts and small dependency-free validation for the five agent tools."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import cast

_SCOPE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "kind": {"type": "string", "enum": ["agent", "user", "project", "org"]},
        "id": {"type": "string", "minLength": 1},
    },
    "required": ["kind", "id"],
}

_ENTITY_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "kind": {"type": "string", "enum": ["person", "project", "org", "repo", "product", "other"]},
        "name": {"type": "string", "minLength": 1},
        "role": {"type": "string", "enum": ["about", "mentions"]},
        "entity_id": {"type": "string", "minLength": 1},
    },
    "required": ["kind", "name", "role"],
}

TOOL_SCHEMAS: dict[str, dict[str, object]] = {
    "memory_search": {
        "name": "memory_search",
        "description": "Search durable memory for facts, past decisions, and procedures relevant to the current task.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "queries": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1, "maxItems": 3},
                "types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["semantic", "episodic", "procedural"]},
                },
                "entities": {"type": "array", "items": {"type": "string", "minLength": 1}},
                "since": {"type": "string", "format": "date-time"},
                "until": {"type": "string", "format": "date-time"},
                "k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
                "include_history": {"type": "boolean", "default": False},
            },
            "required": ["queries"],
        },
    },
    "memory_get": {
        "name": "memory_get",
        "description": "Inspect known memory records, their conflicts, events, and supersession lineage.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"ids": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1}},
            "required": ["ids"],
        },
    },
    "memory_write": {
        "name": "memory_write",
        "description": "Save one durable fact, decision, procedure, or event that should change future work.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {"type": "string", "enum": ["semantic", "episodic", "procedural"]},
                "content": {"type": "string", "minLength": 1, "maxLength": 1000},
                "attribute": {"type": "string", "minLength": 1},
                "scope": _SCOPE_SCHEMA,
                "source_kind": {"type": "string", "enum": ["user_statement", "tool_result", "agent_inference"]},
                "evidence": {"type": "string", "minLength": 1},
                "event_at": {"type": "string", "format": "date-time"},
                "entities": {"type": "array", "items": _ENTITY_SCHEMA},
                "tags": {"type": "array", "items": {"type": "string", "minLength": 1}},
                "evidence_turn": {"type": "integer", "minimum": 1},
            },
            "required": ["type", "content", "source_kind"],
        },
    },
    "memory_revise": {
        "name": "memory_revise",
        "description": "Confirm, supersede, expire a known memory, or merge two known entities.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            # Every property is declared here because `additionalProperties: false` cannot see into `oneOf`
            # branches; the branches then select one shape by requiring and forbidding the right keys.
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "action": {"type": "string", "enum": ["confirm", "supersede", "expire"]},
                "content": {"type": "string", "minLength": 1, "maxLength": 1000},
                "source_kind": {"type": "string", "enum": ["user_statement", "tool_result", "agent_inference"]},
                "evidence": {"type": "string", "minLength": 1},
                "evidence_turn": {"type": "integer", "minimum": 1},
                "reason": {"type": "string", "minLength": 1},
                "entity_id": {"type": "string", "minLength": 1},
                "merge_into": {"type": "string", "minLength": 1},
            },
            "oneOf": [
                {
                    "required": ["id", "action", "reason"],
                    "not": {"anyOf": [{"required": ["entity_id"]}, {"required": ["merge_into"]}]},
                },
                {
                    "required": ["entity_id", "merge_into", "reason"],
                    "not": {
                        "anyOf": [
                            {"required": ["id"]},
                            {"required": ["action"]},
                            {"required": ["content"]},
                            {"required": ["source_kind"]},
                            {"required": ["evidence"]},
                            {"required": ["evidence_turn"]},
                        ]
                    },
                },
            ],
        },
    },
    "memory_forget": {
        "name": "memory_forget",
        "description": "Remove a known memory from normal retrieval while keeping a durable audit tombstone.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["id", "reason"],
        },
    },
}


class ToolInputError(ValueError):
    """Raised when a tool payload does not satisfy the public JSON Schema contract."""


def tool_schemas() -> list[dict[str, object]]:
    """Return detached schemas in stable registration order."""

    return deepcopy([TOOL_SCHEMAS[name] for name in TOOL_SCHEMAS])


def validate_tool_input(tool_name: str, payload: Mapping[str, object]) -> dict[str, object]:
    """Validate the supported JSON Schema subset before a handler reads a caller-controlled value."""

    if tool_name not in TOOL_SCHEMAS:
        raise ToolInputError(f"Unknown tool: {tool_name}.")
    if not isinstance(payload, Mapping):
        raise ToolInputError(f"{tool_name} input must be a JSON object.")
    if tool_name == "memory_revise":
        _validate_revise(payload)
    else:
        schema = cast(dict[str, object], TOOL_SCHEMAS[tool_name]["input_schema"])
        _validate_object(payload, schema, tool_name)
    _validate_semantics(tool_name, payload)
    return dict(payload)


def _validate_revise(payload: Mapping[str, object]) -> None:
    expected = {
        "id",
        "action",
        "content",
        "source_kind",
        "evidence",
        "evidence_turn",
        "reason",
        "entity_id",
        "merge_into",
    }
    _reject_unknown_keys(payload, expected, "memory_revise")
    if "entity_id" in payload or "merge_into" in payload:
        _require_keys(payload, {"entity_id", "merge_into", "reason"}, "memory_revise entity merge")
        _require_strings(payload, ("entity_id", "merge_into", "reason"), "memory_revise entity merge")
        if set(payload) != {"entity_id", "merge_into", "reason"}:
            raise ToolInputError("memory_revise entity merge accepts only entity_id, merge_into, and reason.")
        return
    _require_keys(payload, {"id", "action", "reason"}, "memory_revise")
    _require_strings(payload, ("id", "action", "reason"), "memory_revise")
    action = payload["action"]
    if action not in {"confirm", "supersede", "expire"}:
        raise ToolInputError("memory_revise.action must be confirm, supersede, or expire.")
    supersede_only = {"content", "source_kind", "evidence", "evidence_turn"} & set(payload)
    if action == "supersede":
        _require_strings(payload, ("content",), "memory_revise supersede")
        if payload.get("source_kind") == "user_statement" and "evidence" not in payload:
            raise ToolInputError("memory_revise.evidence is required for user_statement.")
    elif supersede_only:
        fields = ", ".join(sorted(supersede_only))
        raise ToolInputError(f"memory_revise fields are only valid with action supersede: {fields}.")
    schema = cast(dict[str, object], TOOL_SCHEMAS["memory_revise"]["input_schema"])
    properties = cast(Mapping[str, Mapping[str, object]], schema["properties"])
    for key, value in payload.items():
        _validate_value(value, properties[key], f"memory_revise.{key}")


def _validate_object(payload: Mapping[str, object], schema: Mapping[str, object], path: str) -> None:
    properties = cast(Mapping[str, Mapping[str, object]], schema["properties"])
    _reject_unknown_keys(payload, set(properties), path)
    _require_keys(payload, set(cast(list[str], schema.get("required", []))), path)
    for key, value in payload.items():
        _validate_value(value, properties[key], f"{path}.{key}")


def _validate_value(value: object, schema: Mapping[str, object], path: str) -> None:
    expected_type = schema.get("type")
    if expected_type == "string":
        if not isinstance(value, str):
            raise ToolInputError(f"{path} must be a string.")
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if isinstance(minimum, int) and len(value) < minimum:
            raise ToolInputError(f"{path} must not be empty.")
        if isinstance(maximum, int) and len(value) > maximum:
            raise ToolInputError(f"{path} must be at most {maximum} characters.")
    elif expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ToolInputError(f"{path} must be an integer.")
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, int) and value < minimum:
            raise ToolInputError(f"{path} must be at least {minimum}.")
        if isinstance(maximum, int) and value > maximum:
            raise ToolInputError(f"{path} must be at most {maximum}.")
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            raise ToolInputError(f"{path} must be a boolean.")
    elif expected_type == "array":
        if not isinstance(value, list):
            raise ToolInputError(f"{path} must be an array.")
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            raise ToolInputError(f"{path} requires at least {minimum} item(s).")
        if isinstance(maximum, int) and len(value) > maximum:
            raise ToolInputError(f"{path} allows at most {maximum} item(s).")
        item_schema = cast(Mapping[str, object], schema["items"])
        for index, item in enumerate(value):
            _validate_value(item, item_schema, f"{path}[{index}]")
    elif expected_type == "object":
        if not isinstance(value, Mapping):
            raise ToolInputError(f"{path} must be an object.")
        _validate_object(cast(Mapping[str, object], value), schema, path)
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        choices = ", ".join(cast(list[str], enum))
        raise ToolInputError(f"{path} must be one of: {choices}.")


def _validate_semantics(tool_name: str, payload: Mapping[str, object]) -> None:
    if tool_name == "memory_write" and payload.get("source_kind") == "user_statement" and "evidence" not in payload:
        raise ToolInputError("memory_write.evidence is required for user_statement.")
    content = payload.get("content")
    if isinstance(content, str) and not content.strip():
        raise ToolInputError(f"{tool_name}.content must not be blank.")
    evidence = payload.get("evidence")
    if isinstance(evidence, str) and not evidence.strip():
        raise ToolInputError(f"{tool_name}.evidence must not be blank.")
    if tool_name == "memory_write" and payload.get("type") in {"semantic", "procedural"} and "attribute" not in payload:
        raise ToolInputError("memory_write.attribute is required for semantic and procedural memories.")


def _reject_unknown_keys(payload: Mapping[str, object], allowed: set[str], path: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ToolInputError(f"{path} has unknown field(s): {', '.join(unknown)}.")


def _require_keys(payload: Mapping[str, object], required: set[str], path: str) -> None:
    missing = sorted(required - set(payload))
    if missing:
        raise ToolInputError(f"{path} is missing required field(s): {', '.join(missing)}.")


def _require_strings(payload: Mapping[str, object], names: tuple[str, ...], path: str) -> None:
    for name in names:
        value = payload.get(name)
        if not isinstance(value, str) or not value:
            raise ToolInputError(f"{path}.{name} must be a non-empty string.")
