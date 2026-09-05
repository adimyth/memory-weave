"""Framework-neutral handlers that apply tool schemas, scope checks, and durable lifecycle operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, cast

from memory_weave.index.vector import VectorIndex
from memory_weave.ingest import (
    EntityMergeError,
    EntityNotFoundError,
    EntityNotReadableError,
    EntityNotWritableError,
    EntityResolutionError,
    Ingestor,
    WriteRequest,
    merge_entities,
)
from memory_weave.models import EntityMention, MemoryType, Principal, Record, Scope, SearchRequest
from memory_weave.policy import readable_scopes, writable_scopes
from memory_weave.retrieve import Retriever
from memory_weave.store import Store

from .schemas import ToolInputError, validate_tool_input


class ToolHandlers:
    """Execute the five memory tools without taking framework identity or authorization from tool input."""

    def __init__(
        self,
        retriever: Retriever,
        ingestor: Ingestor,
        store: Store,
        vector_index: VectorIndex,
        *,
        default_k: int = 8,
    ) -> None:
        self._retriever = retriever
        self._ingestor = ingestor
        self._store = store
        self._vector_index = vector_index
        self._default_k = default_k

    def memory_search(
        self,
        principal: Principal,
        payload: Mapping[str, object],
        *,
        context: str | None = None,
        trigger: Literal["tool", "auto"] = "tool",
    ) -> dict[str, object]:
        """Search readable memory using adapter-provided context that does not appear in the public schema."""

        parsed = self._validated("memory_search", payload)
        if isinstance(parsed, dict) and "ok" in parsed:
            return parsed
        assert isinstance(parsed, Mapping)
        try:
            request = SearchRequest(
                queries=cast(list[str], parsed["queries"]),
                context=context,
                types=cast(list[MemoryType] | None, parsed.get("types")),
                entities=cast(list[str] | None, parsed.get("entities")),
                since=_datetime_value(parsed.get("since"), "memory_search.since"),
                until=_datetime_value(parsed.get("until"), "memory_search.until"),
                k=cast(int, parsed.get("k", self._default_k)),
                include_history=cast(bool, parsed.get("include_history", False)),
                trigger=trigger,
            )
            response = self._retriever.search(principal, request)
        except ValueError as error:
            return _error("invalid_input", str(error))
        return {
            "ok": True,
            "search_id": response.search_id,
            "results": [
                _search_result_payload(result.record, result.score, result.explanation) for result in response.results
            ],
            "empty_reason": response.empty_reason,
            "rewritten_queries": response.rewritten_queries,
        }

    def memory_get(self, principal: Principal, payload: Mapping[str, object]) -> dict[str, object]:
        """Return known readable records with conflicts, audit events, and both directions of their lineage."""

        parsed = self._validated("memory_get", payload)
        if isinstance(parsed, dict) and "ok" in parsed:
            return parsed
        assert isinstance(parsed, Mapping)
        records: list[dict[str, object]] = []
        for record_id in cast(list[str], parsed["ids"]):
            record = self._readable_record(principal, record_id)
            if record is None:
                return _error("not_found", f"Record {record_id} was not found.")
            records.append(self._get_payload(principal, record))
        return {"ok": True, "records": records}

    def memory_write(self, principal: Principal, payload: Mapping[str, object]) -> dict[str, object]:
        """Validate one public write payload and delegate durable policy decisions to the ingestor."""

        parsed = self._validated("memory_write", payload)
        if isinstance(parsed, dict) and "ok" in parsed:
            return parsed
        assert isinstance(parsed, Mapping)
        try:
            request = WriteRequest(
                type=cast(Any, parsed["type"]),
                content=cast(str, parsed["content"]),
                source_kind=cast(Any, parsed["source_kind"]),
                evidence=cast(str | None, parsed.get("evidence")),
                attribute=cast(str | None, parsed.get("attribute")),
                scope=_scope_value(parsed.get("scope")),
                event_at=_datetime_value(parsed.get("event_at"), "memory_write.event_at"),
                entities=_entity_mentions(cast(Sequence[Mapping[str, object]], parsed.get("entities", []))),
                tags=cast(list[str], parsed.get("tags", [])),
                evidence_turn=cast(int | None, parsed.get("evidence_turn")),
            )
        except ValueError as error:
            return _error("invalid_input", str(error))
        try:
            result = self._ingestor.write(principal, request)
        except (EntityNotFoundError, EntityNotReadableError):
            return _error("not_found", "A requested entity was not found.")
        except EntityNotWritableError:
            return _error("scope_not_writable", "The principal cannot write the entity scope.")
        except EntityResolutionError as error:
            return _error("invalid_input", str(error))
        if result.outcome in {"scope_not_writable", "entity_ambiguous", "invalid_source_kind", "invalid_subject"}:
            details: dict[str, object] = {}
            if result.candidates:
                details["candidates"] = [_ambiguity_candidate_payload(candidate) for candidate in result.candidates]
            return _error(result.outcome, result.note, **details)
        return {
            "ok": True,
            "record_id": result.record_id,
            "status": result.status,
            "outcome": result.outcome,
            "note": result.note,
        }

    def memory_revise(self, principal: Principal, payload: Mapping[str, object]) -> dict[str, object]:
        """Apply a direct record lifecycle revision or a checked entity merge."""

        parsed = self._validated("memory_revise", payload)
        if isinstance(parsed, dict) and "ok" in parsed:
            return parsed
        assert isinstance(parsed, Mapping)
        reason = cast(str, parsed["reason"])
        if "entity_id" in parsed:
            try:
                entity = merge_entities(
                    cast(str, parsed["entity_id"]), cast(str, parsed["merge_into"]), principal, reason, self._store
                )
            except (EntityNotFoundError, EntityNotReadableError, EntityNotWritableError):
                # One answer for missing, unreadable, and unwritable entities, so an error never
                # confirms that a foreign entity id exists.
                return _error("not_found", "One or both entities were not found.")
            except (EntityMergeError, EntityResolutionError) as error:
                return _error("invalid_input", str(error))
            return {"ok": True, "entity": _entity_payload(entity), "outcome": "merged"}

        record_id = cast(str, parsed["id"])
        if self._readable_record(principal, record_id) is None:
            return _error("not_found", f"Record {record_id} was not found.")

        result = self._ingestor.revise(
            principal,
            record_id,
            cast(Any, parsed["action"]),
            reason,
            content=cast(str | None, parsed.get("content")),
            source_kind=cast(Any, parsed.get("source_kind", "agent_inference")),
            evidence=cast(str | None, parsed.get("evidence")),
            evidence_turn=cast(int | None, parsed.get("evidence_turn")),
        )
        if result.outcome in {"not_found", "scope_not_writable", "invalid_input", "invalid_source_kind"}:
            return _error(result.outcome, result.note)
        return {"ok": True, "record_id": result.record_id, "status": result.status, "outcome": result.outcome}

    def memory_forget(self, principal: Principal, payload: Mapping[str, object]) -> dict[str, object]:
        """Tombstone a writable record, remove its lexical row, and hide its live vector after commit."""

        parsed = self._validated("memory_forget", payload)
        if isinstance(parsed, dict) and "ok" in parsed:
            return parsed
        assert isinstance(parsed, Mapping)
        record_id = cast(str, parsed["id"])
        record = self._readable_record(principal, record_id)
        if record is None:
            return _error("not_found", f"Record {record_id} was not found.")
        if record.scope not in writable_scopes(self._store, principal.agent_id, principal.user_id):
            return _error("scope_not_writable", "The principal cannot write the record scope.")
        with self._store.transaction():
            self._store.update_status(record.id, "deleted")
            self._store.delete_fts(record.id)
            self._store.append_event(
                "record.deleted", principal.agent_id, record.id, None, {"reason": parsed["reason"]}
            )
        self._vector_index.remove(record.id, index_version=self._store.record_index_version(record.id))
        return {"ok": True, "record_id": record.id, "status": "deleted", "outcome": "forgotten"}

    def _validated(self, tool_name: str, payload: Mapping[str, object]) -> Mapping[str, object] | dict[str, object]:
        try:
            return validate_tool_input(tool_name, payload)
        except ToolInputError as error:
            return _error("invalid_input", str(error))

    def _readable_record(self, principal: Principal, record_id: str) -> Record | None:
        record = self._store.get_record(record_id)
        if record is None:
            return None
        scopes = readable_scopes(self._store, principal.agent_id, principal.user_id)
        return record if record.scope in scopes else None

    def _get_payload(self, principal: Principal, record: Record) -> dict[str, object]:
        ancestors = self._ancestors(principal, record)
        successors = self._successors(principal, record)
        scopes = readable_scopes(self._store, principal.agent_id, principal.user_id)
        conflicts = [
            _record_payload(conflict, self._store, scopes)
            for conflict_id in self._store.conflicts_for(record.id)
            if (conflict := self._readable_record(principal, conflict_id)) is not None
        ]
        return {
            "record": _record_payload(record, self._store, scopes),
            "conflicts": conflicts,
            "lineage": {
                "ancestors": [_record_payload(item, self._store, scopes) for item in ancestors],
                "successors": [_record_payload(item, self._store, scopes) for item in successors],
            },
            "events": self._store.events_for(record.id),
        }

    def _ancestors(self, principal: Principal, record: Record) -> list[Record]:
        ancestors: list[Record] = []
        current = record
        while current.supersedes_id is not None:
            parent = self._readable_record(principal, current.supersedes_id)
            if parent is None:
                break
            ancestors.append(parent)
            current = parent
        return ancestors

    def _successors(self, principal: Principal, record: Record) -> list[Record]:
        successors: list[Record] = []
        pending = [record]
        while pending:
            parent = pending.pop(0)
            children = [
                child
                for child in self._store.records_superseded_by(parent.id)
                if self._readable_record(principal, child.id) is not None
            ]
            successors.extend(children)
            pending.extend(children)
        return successors


def _scope_value(value: object) -> Scope | None:
    if value is None:
        return None
    mapping = cast(Mapping[str, object], value)
    return Scope(kind=cast(Any, mapping["kind"]), id=cast(str, mapping["id"]))


def _entity_mentions(values: Sequence[Mapping[str, object]]) -> list[EntityMention]:
    return [
        EntityMention(
            kind=cast(Any, value["kind"]),
            text=cast(str, value["name"]),
            role=cast(Any, value["role"]),
            entity_id=cast(str | None, value.get("entity_id")),
        )
        for value in values
    ]


def _datetime_value(value: object, path: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{path} must be an ISO 8601 date-time string.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{path} must be an ISO 8601 date-time string.") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{path} must include a timezone.")
    return parsed


def _error(code: str, message: str | None, **details: object) -> dict[str, object]:
    return {"ok": False, "error": {"code": code, "message": message, **details}}


def _record_payload(
    record: Record, store: Store | None = None, scopes: Sequence[Scope] | None = None
) -> dict[str, object]:
    # A forgotten record keeps its audit shape but never returns the text the user asked to remove;
    # durable erasure of that text stays an admin operation.
    forgotten = record.status == "deleted"
    payload: dict[str, object] = {
        "id": record.id,
        "type": record.type,
        "version": record.version,
        "content": None if forgotten else record.content,
        "tombstone": forgotten,
        "subject": record.subject,
        "subject_entity_id": record.subject_entity_id,
        "attribute": record.attribute,
        "scope": {"kind": record.scope.kind, "id": record.scope.id},
        "source_kind": record.source_kind,
        "source_ref": None if forgotten else record.source_ref,
        "creator_agent_id": record.creator_agent_id,
        "evidence": None if forgotten else record.evidence,
        "created_at": record.created_at.isoformat(),
        "event_at": record.event_at.isoformat(),
        "expires_at": record.expires_at.isoformat() if record.expires_at is not None else None,
        "confidence": record.confidence,
        "status": record.status,
        "supersedes_id": record.supersedes_id,
        "reinforcements": record.reinforcements,
        "last_reinforced_at": record.last_reinforced_at.isoformat() if record.last_reinforced_at is not None else None,
        "tags": record.tags,
        "entity_ids": record.entity_ids,
    }
    if store is not None:
        payload["entities"] = [
            _entity_payload(entity)
            for entity_id in record.entity_ids
            if (entity := store.get_entity(entity_id)) is not None and (scopes is None or entity.scope in scopes)
        ]
    return payload


def _search_result_payload(record: Record, score: float, explanation: object) -> dict[str, object]:
    return {"record": _record_payload(record), "score": score, "explanation": _explanation_payload(explanation)}


def _explanation_payload(explanation: object) -> dict[str, object]:
    values = cast(Any, explanation)
    return {
        "summary": values.summary,
        "matched_by": values.matched_by,
        "fused_rank": values.fused_rank,
        "gate": values.gate,
        "conflicts_with": values.conflicts_with,
    }


def _ambiguity_candidate_payload(candidate: object) -> dict[str, object]:
    values = cast(Any, candidate)
    return {"id": values.id, "kind": values.kind, "canonical": values.canonical, "scope": _scope_payload(values.scope)}


def _entity_payload(entity: object) -> dict[str, object]:
    values = cast(Any, entity)
    return {
        "id": values.id,
        "kind": values.kind,
        "canonical": values.canonical,
        "scope": _scope_payload(values.scope),
        "status": values.status,
    }


def _scope_payload(scope: Scope) -> dict[str, str]:
    return {"kind": scope.kind, "id": scope.id}
