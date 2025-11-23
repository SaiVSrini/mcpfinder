from __future__ import annotations

import math
import re
from typing import List, Optional

from openai import OpenAI

from .config import get_embedding_model, get_openai_api_key, has_openai_api_key
from .models import ToolEntry

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """
    Lowercase and split text into alphanumeric tokens.
    """
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text.lower())


def keyword_overlap_score(query_tokens: List[str], text_tokens: List[str]) -> float:
    """
    Compute the fraction of unique query tokens that appear in the text tokens.
    """
    if not query_tokens:
        return 0.0
    q_set = set(query_tokens)
    if not q_set:
        return 0.0
    t_set = set(text_tokens)
    overlap = len(q_set & t_set)
    return overlap / float(len(q_set))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Standard cosine similarity with zero-norm checks.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def generate_query_embedding(user_query: str) -> Optional[List[float]]:
    """
    Generate an embedding for the user query using OpenAI, or return None if unavailable.
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
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: Failed to generate query embedding: {exc}")
        return None


def basic_score(
    user_query: str,
    entry: ToolEntry,
    query_embedding: Optional[List[float]],
    filter_tags: Optional[List[str]] = None,
) -> float:
    """
    Hybrid score combining keyword overlap, embedding similarity, and tag bonus.
    """
    query_tokens = tokenize(user_query)
    entry_tokens = tokenize(entry.combined_text)

    keyword_score = keyword_overlap_score(query_tokens, entry_tokens)

    embed_score = 0.0
    if query_embedding is not None and entry.embedded_vector is not None:
        embed_score = cosine_similarity(query_embedding, entry.embedded_vector)

    tag_bonus = 0.0
    if filter_tags:
        filter_lower = {t.lower() for t in filter_tags}
        entry_tags_lower = {t.lower() for t in entry.capability_tags}
        if filter_lower & entry_tags_lower:
            tag_bonus = 0.2

    if query_embedding is None:
        final_score = keyword_score + tag_bonus
    else:
        final_score = 0.6 * keyword_score + 0.4 * embed_score + tag_bonus

    return max(final_score, 0.0)


def select_candidates(
    user_query: str,
    entries: List[ToolEntry],
    max_candidates: int = 20,
    filter_tags: Optional[List[str]] = None,
) -> List[ToolEntry]:
    """
    Score all entries and return up to max_candidates with positive scores.
    """
    query_embedding: Optional[List[float]] = generate_query_embedding(user_query)

    scored: List[tuple[ToolEntry, float]] = []
    for entry in entries:
        score = basic_score(
            user_query=user_query,
            entry=entry,
            query_embedding=query_embedding,
            filter_tags=filter_tags,
        )
        if score > 0.0:
            scored.append((entry, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    top_entries = [entry for entry, _ in scored[: max_candidates or 0]]
    return top_entries

