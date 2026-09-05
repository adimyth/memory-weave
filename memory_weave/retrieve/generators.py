"""Dense, lexical, and entity candidate generators used by the retrieval pipeline."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from memory_weave.index.embedder import Embedder
from memory_weave.index.vector import VectorIndex
from memory_weave.models import Entity, EntityStatus, GeneratorHit, LexicalMatch, LexicalTerm, Scope
from memory_weave.store import Store
from memory_weave.util import normalize_alias

from .stopwords import STOPWORDS

type DenseCandidate = tuple[str, GeneratorHit]
type LexicalCandidate = tuple[str, GeneratorHit, LexicalMatch]
type EntityCandidate = tuple[str, GeneratorHit, str]
_ENTITY_SEARCH_STATUSES: tuple[EntityStatus, ...] = ("provisional", "confirmed")
_RAW_TOKEN = re.compile(r"\S+", re.UNICODE)
_FTS_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_TOKEN_TRIM = "\"'`.,;:!?()[]{}<>"
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"})
_DEFAULT_MAX_ALIAS_TOKENS = 4


@dataclass(frozen=True, slots=True)
class _QueryTerm:
    value: str
    raw: str
    is_identifier: bool


def dense_candidates(
    queries: Sequence[str],
    embedder: Embedder,
    vector_index: VectorIndex,
    eligible_ids: set[str],
    k: int,
    query_vectors: np.ndarray | None = None,
) -> list[DenseCandidate]:
    """Return at most ``k`` dense candidates after each record keeps its best query cosine."""

    if k <= 0 or not queries:
        return []

    scores: dict[str, float] = {}
    vectors = embedder.embed_queries(list(queries)) if query_vectors is None else query_vectors
    if len(vectors) != len(queries):
        raise ValueError("query_vectors must contain exactly one vector for each query.")
    for query_vector in vectors:
        for record_id, cosine in vector_index.search(query_vector, eligible_ids, k):
            scores[record_id] = max(scores.get(record_id, float("-inf")), cosine)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    return [(record_id, GeneratorHit(rank, score)) for rank, (record_id, score) in enumerate(ordered, start=1)]


def lexical_candidates(
    queries: Sequence[str],
    store: Store,
    eligible: set[str],
    k: int,
    stopwords: Collection[str] = STOPWORDS,
    entity_aliases: Collection[str] = (),
) -> list[LexicalCandidate]:
    """Return eligible FTS candidates with the best per-query matched-term evidence for each record."""

    if k <= 0 or not eligible:
        return []
    terms_by_query = [_query_terms(query, stopwords) for query in queries]
    all_terms = list(dict.fromkeys(term.value for terms in terms_by_query for term in terms))
    if not all_terms:
        return []

    matches = store.fts_query(fts_match_expression(all_terms), k, eligible)
    indexed_rows = store.fts_rows([record_id for record_id, _ in matches])
    alias_tokens = {token for alias in entity_aliases for token in _fts_tokens(alias)}
    candidates: list[LexicalCandidate] = []
    for record_id, score in matches:
        fields = indexed_rows.get(record_id)
        if fields is None:
            continue
        lexical_match = _best_lexical_match(terms_by_query, fields, alias_tokens)
        candidates.append((record_id, GeneratorHit(len(candidates) + 1, score), lexical_match))
    return candidates


def entity_candidates(
    entity_hints: Sequence[str] | None,
    queries: Sequence[str],
    store: Store,
    scopes: Sequence[Scope],
    eligible: set[str],
    k: int,
    max_alias_tokens: int = _DEFAULT_MAX_ALIAS_TOKENS,
    alias_matches: Mapping[str, Sequence[Entity]] | None = None,
) -> list[EntityCandidate]:
    """Return records linked to exact aliases from entity hints or query text, ordered by recency."""

    if k <= 0 or not eligible or not scopes:
        return []

    matches = (
        entity_alias_matches(entity_hints, queries, store, scopes, max_alias_tokens)
        if alias_matches is None
        else alias_matches
    )
    entity_ids = list(dict.fromkeys(entity.id for entities in matches.values() for entity in entities))
    return [
        (record_id, GeneratorHit(rank, 0.0), entity_id)
        for rank, (record_id, entity_id) in enumerate(store.records_for_entities(entity_ids, eligible, k), start=1)
    ]


def entity_alias_matches(
    entity_hints: Sequence[str] | None,
    queries: Sequence[str],
    store: Store,
    scopes: Sequence[Scope],
    max_alias_tokens: int = _DEFAULT_MAX_ALIAS_TOKENS,
) -> dict[str, list[Entity]]:
    """Resolve every explicit or query-derived entity alias once so lexical and entity stages share its evidence."""

    return store.entities_by_aliases(
        _entity_aliases(entity_hints or (), queries, max_alias_tokens),
        scopes=scopes,
        statuses=_ENTITY_SEARCH_STATUSES,
    )


def fts_match_expression(terms: Sequence[str]) -> str:
    """Build an FTS5 OR expression that treats every normalized term as literal text."""

    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def _query_terms(query: str, stopwords: Collection[str]) -> list[_QueryTerm]:
    normalized_stopwords = {normalize_alias(word) for word in stopwords}
    terms: list[_QueryTerm] = []
    for position, raw_term in enumerate(_raw_terms(query)):
        raw_term = raw_term.translate(_APOSTROPHES)
        term = _strip_contraction(normalize_alias(raw_term))
        if len(term) < 2:
            continue
        identifier = _is_identifier(raw_term)
        if term in normalized_stopwords and not identifier and not _is_proper_noun(raw_term, position):
            continue
        if all(existing.value != term for existing in terms):
            terms.append(_QueryTerm(term, raw_term, identifier))
    return terms


def _best_lexical_match(
    terms_by_query: Sequence[Sequence[_QueryTerm]], fields: tuple[str, str, str], alias_tokens: set[str]
) -> LexicalMatch:
    field_tokens = [_fts_tokens(field) for field in fields]
    indexed_tokens = {token for tokens in field_tokens for token in tokens}
    best: LexicalMatch | None = None
    for terms in terms_by_query:
        matched = tuple(
            LexicalTerm(
                value=term.value,
                is_identifier=term.is_identifier,
                is_entity_alias=bool(_fts_tokens(term.value)) and set(_fts_tokens(term.value)) <= alias_tokens,
            )
            for term in terms
            if _term_matches(term.value, field_tokens, indexed_tokens)
        )
        candidate = LexicalMatch(matched, len(terms))
        if (
            best is None
            or candidate.fraction > best.fraction
            or (candidate.fraction == best.fraction and len(candidate.terms) > len(best.terms))
        ):
            best = candidate
    return best or LexicalMatch((), 0)


def _term_matches(term: str, field_tokens: Sequence[Sequence[str]], indexed_tokens: set[str]) -> bool:
    """A single-token term matches anywhere; a multi-token term must appear contiguously in one field."""

    parts = _fts_tokens(term)
    if not parts:
        return False
    if len(parts) == 1:
        return parts[0] in indexed_tokens
    return any(_contains_sequence(tokens, parts) for tokens in field_tokens)


def _contains_sequence(tokens: Sequence[str], parts: Sequence[str]) -> bool:
    width = len(parts)
    return any(list(tokens[start : start + width]) == list(parts) for start in range(len(tokens) - width + 1))


def _strip_contraction(term: str) -> str:
    """Reduce "aditya's" to "aditya" and "don't" to "do" so contractions never survive as query terms."""

    if "'" not in term:
        return term
    if term.endswith("n't"):
        return term[:-3]
    return term.split("'", 1)[0]


def _entity_aliases(entity_hints: Sequence[str], queries: Sequence[str], max_alias_tokens: int) -> list[str]:
    """Return candidate aliases: explicit hints plus every query n-gram up to ``max_alias_tokens`` long."""

    aliases = {normalize_alias(hint) for hint in entity_hints if normalize_alias(hint)}
    for query in queries:
        terms = [_alias_term(term) for term in _raw_terms(query)]
        terms = [term for term in terms if term]
        for start in range(len(terms)):
            for end in range(start + 1, min(start + max_alias_tokens, len(terms)) + 1):
                alias = normalize_alias(" ".join(terms[start:end]))
                if alias:
                    aliases.add(alias)
    return sorted(aliases)


def _alias_term(term: str) -> str:
    """Normalize query alias tokens with the same apostrophe and contraction rules as lexical terms."""

    return _strip_contraction(normalize_alias(term.translate(_APOSTROPHES)))


def _raw_terms(value: str) -> list[str]:
    return [term for raw in _RAW_TOKEN.findall(value) if (term := raw.strip(_TOKEN_TRIM))]


def _fts_tokens(value: str) -> list[str]:
    return [token for token in _FTS_TOKEN.findall(normalize_alias(value)) if len(token) >= 2]


def _is_identifier(term: str) -> bool:
    """Digits, dotted or slashed paths, underscores, and mixed case are identifiers; a hyphenated word is not."""

    if any(character.isdigit() for character in term) or any(character in "._/" for character in term):
        return True
    if term[:1].isupper() and term[1:].islower():
        return False
    return term.lower() != term and term.upper() != term


def _is_proper_noun(term: str, position: int) -> bool:
    return position > 0 and len(term) > 1 and term[0].isupper() and term[1:].islower()
