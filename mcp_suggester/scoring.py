from __future__ import annotations

import math
import re
from typing import List, Optional

from openai import OpenAI

from .config import get_embedding_model, get_openai_api_key, has_openai_api_key
from .models import ToolEntry

# Very small, forgiving tokenizer: anything alphanumeric is a token.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """
    Turn a piece of text into simple lowercase tokens.

    This is deliberately light‑weight; we do not try to be clever here.
    """
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text.lower())


def keyword_overlap_score(query_tokens: List[str], text_tokens: List[str]) -> float:
    """
    Simple overlap: how many unique query tokens show up in the text.
    """
    if not query_tokens:
        return 0.0
    q_set = set(query_tokens)
    if not q_set:
        return 0.0
    t_set = set(text_tokens)
    overlap = len(q_set & t_set)
    return overlap / float(len(q_set))


def cosine_similarity(first_vector: List[float], second_vector: List[float]) -> float:
    """
    Standard cosine similarity with zero-norm checks.
    """
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
    Compute a single score for how well one catalog entry matches the query.

    In words:
    - Look at how much the words overlap.
    - Blend in embedding similarity if we have both vectors.
    - Add a small bump if the caller passed tags that match this tool.
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
    Score all entries and return the strongest ones.

    I think of this as the "first pass" filter before we involve the LLM.
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
