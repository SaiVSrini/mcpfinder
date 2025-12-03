from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from openai import OpenAI

from .config import get_embedding_model, get_openai_api_key, has_openai_api_key
from .intent import QueryIntent
from .models import CatalogEntry

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_LOCAL_ENTRY_KEYWORDS = {
    "local",
    "filesystem",
    "fs",
    "offline",
    "self-hosted",
    "self hosted",
    "on-prem",
    "air-gapped",
    "airgapped",
    "air gap",
}
_FREE_ENTRY_KEYWORDS = {
    "free",
    "no-cost",
    "no cost",
    "open-source",
    "open source",
    "community",
    "self-hosted",
    "self hosted",
}


@dataclass(frozen=True)
class QueryVector:
    weights: Dict[str, float]
    norm: float


@dataclass(frozen=True)
class LexicalDocument:
    entry: CatalogEntry
    weights: Dict[str, float]
    norm: float


@dataclass(frozen=True)
class ScoredEntry:
    entry: CatalogEntry
    score: float


class LexicalIndex:

    def __init__(self, entries: Sequence[CatalogEntry]):
        self.document_count = len(entries)
        self.idf: Dict[str, float] = {}
        self.documents: List[LexicalDocument] = []
        self._build(entries)

    def _build(self, entries: Sequence[CatalogEntry]) -> None:
        tokenized: List[tuple[CatalogEntry, List[str]]] = []
        doc_freq: Counter[str] = Counter()
        for entry in entries:
            tokens = tokenize(entry.combined_text)
            tokenized.append((entry, tokens))
            doc_freq.update(set(tokens))

        if not doc_freq:
            return

        # Calculate IDF scores - rare words get higher scores
        for token, freq in doc_freq.items():
            self.idf[token] = math.log((self.document_count + 1) / (freq + 1)) + 1.0

        for entry, tokens in tokenized:
            tf_counter = Counter(tokens)
            weights: Dict[str, float] = {}
            for token, freq in tf_counter.items():
                idf_val = self.idf.get(token)
                if not idf_val:
                    continue
                weights[token] = (1.0 + math.log(freq)) * idf_val
            norm = math.sqrt(sum(value * value for value in weights.values())) or 1.0
            self.documents.append(LexicalDocument(entry=entry, weights=weights, norm=norm))

    def build_query_vector(self, tokens: Sequence[str]) -> QueryVector:
        tf_counter = Counter(tokens)
        weights: Dict[str, float] = {}
        for token, freq in tf_counter.items():
            idf_val = self.idf.get(token)
            if not idf_val:
                continue
            weights[token] = (1.0 + math.log(freq)) * idf_val
        norm = math.sqrt(sum(value * value for value in weights.values())) or 1.0
        return QueryVector(weights=weights, norm=norm)

    def score(self, document: LexicalDocument, query_vector: QueryVector) -> float:
        dot_product = 0.0
        for token, weight in query_vector.weights.items():
            dot_product += weight * document.weights.get(token, 0.0)
        denom = query_vector.norm * document.norm
        if denom == 0.0:
            return 0.0
        return dot_product / denom


_LEXICAL_INDEX_CACHE: Dict[int, LexicalIndex] = {}


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text.lower())


def _get_lexical_index(entries: Sequence[CatalogEntry]) -> LexicalIndex:
    cache_key = id(entries)
    cached = _LEXICAL_INDEX_CACHE.get(cache_key)
    if cached and cached.document_count == len(entries):
        return cached
    index = LexicalIndex(entries)
    _LEXICAL_INDEX_CACHE[cache_key] = index
    return index


def cosine_similarity(first_vector: List[float], second_vector: List[float]) -> float:
    if not first_vector or not second_vector or len(first_vector) != len(second_vector):
        return 0.0
    dot_product = sum(x * y for x, y in zip(first_vector, second_vector))
    first_norm = math.sqrt(sum(x * x for x in first_vector))
    second_norm = math.sqrt(sum(y * y for y in second_vector))
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    return dot_product / (first_norm * second_norm)


def generate_query_embedding(user_query: str) -> Optional[List[float]]:
    """
    Turn the user query into an embedding vector using OpenAI, if possible.
    """
    if not has_openai_api_key():
        return None

    api_key = get_openai_api_key()
    if not api_key:
        return None

    try:
        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(
            model=get_embedding_model(),
            input=user_query,
        )
        return resp.data[0].embedding
    except Exception as exc:  
        print(f"WARNING: Failed to generate query embedding: {exc}")
        return None


def _entry_matches(entry: CatalogEntry, keywords: set[str]) -> bool:
    haystack = " ".join(
        [
            entry.tool_description or "",
            entry.server_description or "",
            entry.embedded_text or "",
            " ".join(entry.capability_tags),
        ]
    ).lower()
    return any(keyword in haystack for keyword in keywords)


def _entry_is_local(entry: CatalogEntry) -> bool:
    return _entry_matches(entry, _LOCAL_ENTRY_KEYWORDS)


def _entry_is_free(entry: CatalogEntry) -> bool:
    return _entry_matches(entry, _FREE_ENTRY_KEYWORDS)


def _capability_bonus(
    entry: CatalogEntry,
    filter_tags: set[str],
    intent_tags: set[str],
) -> float:
    bonus = 0.0
    entry_tags = {tag.lower() for tag in entry.capability_tags}
    if filter_tags and entry_tags & filter_tags:
        bonus += 0.2
    if intent_tags:
        overlap = entry_tags & intent_tags
        if overlap:
            bonus += min(0.15 * len(overlap), 0.3)
    return bonus


def _intent_alignment(entry: CatalogEntry, intent: QueryIntent) -> float:
    bonus = 0.0
    if intent.must_be_local and _entry_is_local(entry):
        bonus += 0.15
    if intent.must_be_free and _entry_is_free(entry):
        bonus += 0.1

    preferred_auth = (intent.preferred_auth or "").lower()
    entry_auth = (entry.auth_type or "").lower()
    if preferred_auth:
        if preferred_auth == "none":
            if not entry_auth or entry_auth in {"none", "local"}:
                bonus += 0.08
            else:
                bonus -= 0.04
        elif preferred_auth in entry_auth:
            bonus += 0.08
        else:
            bonus -= 0.04
    return bonus


def _intent_penalty(entry: CatalogEntry, intent: QueryIntent) -> float:
    penalty = 0.0

    if intent.must_be_local and not _entry_is_local(entry):
        penalty += 0.20  
    if intent.must_be_free and not _entry_is_free(entry):
        penalty += 0.15 

    preferred_auth = (intent.preferred_auth or "").lower()
    entry_auth = (entry.auth_type or "").lower()
    if preferred_auth:
        if preferred_auth == "none" and entry_auth and entry_auth not in {"none", "local"}:
            penalty +=0.08 
        elif preferred_auth not in entry_auth and preferred_auth != "none":
            penalty += 0.03 
    return penalty


def _combine_scores(
    lexical_score: float,
    embedding_score: float,
    has_embedding: bool,
) -> float:
    embedding_score = max(embedding_score, 0.0)
    if not has_embedding:
        return 0.85 * lexical_score
    # I give more weight to keyword matching (70%) than embeddings (20%)
    return 0.70 * lexical_score + 0.20 * embedding_score


def score_entries(
    user_query: str,
    entries: List[CatalogEntry],
    intent: QueryIntent,
    filter_tags: Optional[List[str]] = None,
    use_embeddings: bool = True,
    apply_intent: bool = True,
    apply_capability_bonus: bool = True,
) -> List[ScoredEntry]:
    """
    Score all entries according to the hybrid retriever and return sorted scores.
    """
    if not entries:
        return []

    lexical_index = _get_lexical_index(entries)
    query_tokens = tokenize(user_query)
    query_vector = lexical_index.build_query_vector(query_tokens)
    query_embedding: Optional[List[float]] = None
    if use_embeddings:
        query_embedding = generate_query_embedding(user_query)

    filter_tag_set = {tag.lower() for tag in filter_tags or [] if tag}
    intent_tag_set = {tag.lower() for tag in intent.capability_hints}

    scored: List[ScoredEntry] = []
    for document in lexical_index.documents:
        lexical_score = lexical_index.score(document, query_vector)
        embedding_score = 0.0
        if query_embedding is not None and document.entry.embedded_vector:
            embedding_score = cosine_similarity(query_embedding, document.entry.embedded_vector)

        combined = _combine_scores(
            lexical_score,
            embedding_score,
            has_embedding=query_embedding is not None,
        )
        capability_bonus = (
            _capability_bonus(document.entry, filter_tag_set, intent_tag_set)
            if apply_capability_bonus
            else 0.0
        )
        intent_alignment = _intent_alignment(document.entry, intent) if apply_intent else 0.0
        penalty = _intent_penalty(document.entry, intent) if apply_intent else 0.0

        final_score = max(combined + capability_bonus + intent_alignment - penalty, 0.0)
        if final_score > 0.0:
            scored.append(ScoredEntry(entry=document.entry, score=final_score))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored


def select_candidates(
    user_query: str,
    entries: List[CatalogEntry],
    intent: QueryIntent,
    max_candidates: int = 20,
    filter_tags: Optional[List[str]] = None,
) -> List[ScoredEntry]:
    """
    Score all entries and return the strongest ones as ScoredEntry objects.
    """
    scored = score_entries(
        user_query=user_query,
        entries=entries,
        intent=intent,
        filter_tags=filter_tags,
        use_embeddings=True,
        apply_intent=True,
        apply_capability_bonus=True,
    )

    if max_candidates is None:
        limit = len(scored)
    else:
        limit = max(0, max_candidates)
    return scored[:limit]
