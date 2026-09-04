"""Dense, lexical, and entity candidate generators used by the retrieval pipeline."""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence

import numpy as np

from memory_weave.index.embedder import Embedder
from memory_weave.index.vector import VectorIndex
from memory_weave.models import EntityStatus, GeneratorHit, Scope
from memory_weave.store import Store
from memory_weave.util import normalize_alias

from .stopwords import STOPWORDS

type DenseCandidate = tuple[str, GeneratorHit]
type LexicalCandidate = tuple[str, GeneratorHit, int, int]
type EntityCandidate = tuple[str, GeneratorHit, str]
_ENTITY_SEARCH_STATUSES: tuple[EntityStatus, ...] = ("provisional", "confirmed")
_UNICODE61_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def dense_candidates(
    queries: Sequence[str],
    embedder: Embedder,
    vector_index: VectorIndex,
    allowed: np.ndarray,
    k: int,
) -> list[DenseCandidate]:
    """Return the top dense candidates after taking each record's best cosine across queries."""

    if k <= 0 or not queries:
        return []

    scores: dict[str, float] = {}
    for query_vector in embedder.embed_queries(list(queries)):
        for record_id, cosine in vector_index.search(query_vector, allowed, k):
            scores[record_id] = max(scores.get(record_id, float("-inf")), cosine)
    return [
        (record_id, GeneratorHit(rank, score))
        for rank, (record_id, score) in enumerate(sorted(scores.items(), key=lambda item: (-item[1], item[0])), start=1)
    ]


def lexical_candidates(
    queries: Sequence[str],
    store: Store,
    eligible: set[str],
    k: int,
    stopwords: Collection[str] = STOPWORDS,
) -> list[LexicalCandidate]:
    """Return eligible FTS candidates with BM25 scores and matched query-term counts."""

    if k <= 0 or not eligible:
        return []
    terms = _query_terms(queries, stopwords)
    if not terms:
        return []

    matches = store.fts_query(" OR ".join(terms), limit=3 * k)
    eligible_matches = [(record_id, score) for record_id, score in matches if record_id in eligible]
    indexed_rows = store.fts_rows([record_id for record_id, _ in eligible_matches])
    total_terms = len(terms)
    candidates: list[LexicalCandidate] = []
    for record_id, score in eligible_matches:
        fields = indexed_rows.get(record_id)
        if fields is None:
            continue
        indexed_terms = _index_terms(fields)
        matched_terms = sum(term in indexed_terms for term in terms)
        candidates.append((record_id, GeneratorHit(len(candidates) + 1, score), matched_terms, total_terms))
        if len(candidates) == k:
            break
    return candidates


def entity_candidates(
    entity_hints: Sequence[str] | None,
    queries: Sequence[str],
    store: Store,
    scopes: Sequence[Scope],
    eligible: set[str],
    k: int,
) -> list[EntityCandidate]:
    """Return records linked to exact aliases from entity hints or query text, ordered by recency."""

    if k <= 0 or not eligible or not scopes:
        return []

    entity_ids: list[str] = []
    for alias in _entity_aliases(entity_hints or (), queries):
        entity_ids.extend(
            entity.id for entity in store.entities_by_alias(alias, scopes=scopes, statuses=_ENTITY_SEARCH_STATUSES)
        )
    distinct_entity_ids = list(dict.fromkeys(entity_ids))
    return [
        (record_id, GeneratorHit(rank, 0.0), entity_id)
        for rank, (record_id, entity_id) in enumerate(
            store.records_for_entities(distinct_entity_ids, eligible, k), start=1
        )
    ]


def _query_terms(queries: Sequence[str], stopwords: Collection[str]) -> list[str]:
    normalized_stopwords = {normalize_alias(word) for word in stopwords}
    terms: list[str] = []
    for query in queries:
        for position, raw_term in enumerate(_raw_terms(query)):
            term = normalize_alias(raw_term)
            if not term:
                continue
            if (
                term in normalized_stopwords
                and not _is_identifier(raw_term)
                and not _is_proper_noun(raw_term, position)
            ):
                continue
            if term not in terms:
                terms.append(term)
    return terms


def _index_terms(fields: tuple[str, str, str]) -> set[str]:
    return {normalize_alias(raw_term) for field in fields for raw_term in _raw_terms(field)}


def _entity_aliases(entity_hints: Sequence[str], queries: Sequence[str]) -> list[str]:
    aliases = {normalize_alias(hint) for hint in entity_hints if normalize_alias(hint)}
    for query in queries:
        terms = _raw_terms(query)
        for start in range(len(terms)):
            for end in range(start + 1, len(terms) + 1):
                aliases.add(normalize_alias(" ".join(terms[start:end])))
    return sorted(alias for alias in aliases if alias)


def _raw_terms(value: str) -> list[str]:
    return _UNICODE61_TOKEN.findall(value)


def _is_identifier(term: str) -> bool:
    if any(character.isdigit() for character in term):
        return True
    if term[:1].isupper() and term[1:].islower():
        return False
    return term.lower() != term and term.upper() != term


def _is_proper_noun(term: str, position: int) -> bool:
    return position > 0 and len(term) > 1 and term[0].isupper() and term[1:].islower()
