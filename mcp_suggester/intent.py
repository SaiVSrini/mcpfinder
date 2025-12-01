from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set

from openai import OpenAI

from .config import get_model_name, get_openai_api_key, has_openai_api_key

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

CAPABILITY_KEYWORDS = {
    "pdf": "pdf",
    "document": "documents",
    "image": "vision",
    "images": "vision",
    "vision": "vision",
    "docker": "docker",
    "kubernetes": "kubernetes",
    "aws": "aws",
    "azure": "azure",
    "gcp": "gcp",
    "sql": "database",
    "database": "database",
    "postgres": "database",
    "vector": "vector",
    "security": "security",
    "vulnerability": "security",
    "code": "code",
    "repo": "code",
    "git": "code",
    "design": "design",
    "imagegen": "generation",
    "generate": "generation",
    "analysis": "analysis",
    "ml": "ml",
    "ai": "ml",
    "local": "local",
    "offline": "local",
    "browser": "browser",
    "web": "web",
    "scrape": "web",
}

LOCAL_HINTS = {"local", "offline", "airgapped", "air-gapped", "airgap", "onprem", "on-prem"}
FREE_HINTS = {"free", "opensource", "open-source", "no-cost"}


def _tokenize(text: str) -> List[str]:
    """
    Tiny tokenizer for intent extraction.
    """
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text.lower())


def _normalize_auth(preferred_auth: Optional[str]) -> Optional[str]:
    """
    Map natural language strings to a small set of auth values.
    """
    if not preferred_auth:
        return None
    normalized = preferred_auth.strip().lower()
    if not normalized:
        return None
    if "api" in normalized and "key" in normalized:
        return "api-key"
    if "oauth" in normalized or "sso" in normalized:
        return "oauth"
    if "none" in normalized or "no auth" in normalized or "local" in normalized:
        return "none"
    return normalized


@dataclass
class QueryIntent:
    """
    Structured interpretation of a free-form user query.
    """

    raw_query: str
    capability_hints: List[str] = field(default_factory=list)
    must_be_local: bool = False
    must_be_free: bool = False
    preferred_auth: Optional[str] = None
    keywords: List[str] = field(default_factory=list)

    def keywords_set(self) -> Set[str]:
        return set(self.keywords)

    def add_capabilities(self, hints: Sequence[str]) -> None:
        current = set(self.capability_hints)
        for hint in hints:
            normalized = hint.strip().lower()
            if normalized:
                current.add(normalized)
        self.capability_hints = sorted(current)


def _heuristic_intent(user_query: str) -> QueryIntent:
    """
    Lightweight keyword-based intent extraction.
    """
    tokens = _tokenize(user_query)
    hints: Set[str] = set()
    for token in tokens:
        mapped = CAPABILITY_KEYWORDS.get(token)
        if mapped:
            hints.add(mapped)

    lower_query = user_query.lower()
    must_be_local = any(keyword in lower_query for keyword in LOCAL_HINTS)
    must_be_free = any(keyword in lower_query for keyword in FREE_HINTS)

    preferred_auth: Optional[str] = None
    if "api key" in lower_query or "apikey" in lower_query:
        preferred_auth = "api-key"
    elif "oauth" in lower_query or "sso" in lower_query:
        preferred_auth = "oauth"
    elif "no auth" in lower_query or "local only" in lower_query:
        preferred_auth = "none"

    intent = QueryIntent(
        raw_query=user_query,
        capability_hints=sorted(hints),
        must_be_local=must_be_local,
        must_be_free=must_be_free,
        preferred_auth=preferred_auth,
        keywords=tokens,
    )
    return intent


def _call_llm_for_intent(user_query: str) -> Optional[dict]:
    """
    Ask a lightweight OpenAI model to classify the query intent.
    """
    if not has_openai_api_key():
        return None
    api_key = get_openai_api_key()
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are an intent extraction engine for routing Model Context Protocol tools. "
        "Read the user query and produce a small JSON object with: "
        "capability_tags (list of short lowercase tags), "
        "must_be_local (bool), must_be_free (bool), "
        "preferred_auth (api-key, oauth, none, or any). "
        "Respond with JSON only."
    )
    user_payload = {"user_query": user_query}

    try:
        completion = client.chat.completions.create(
            model=get_model_name(),
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        )
        content = completion.choices[0].message.content or ""
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped.lstrip("`")
            if stripped.startswith("json"):
                stripped = stripped[4:]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
            stripped = stripped.strip()

        parsed = json.loads(stripped)
        return parsed
    except Exception:
        return None


def extract_intent(user_query: str, use_llm: bool = True) -> QueryIntent:
    """
    Primary entry point: heuristics first, optionally refined by the LLM.
    """
    intent = _heuristic_intent(user_query)
    if not use_llm:
        return intent
    llm_data = _call_llm_for_intent(user_query)
    if not llm_data:
        return intent

    capability_tags = llm_data.get("capability_tags") or llm_data.get("capabilities")
    if isinstance(capability_tags, list):
        intent.add_capabilities(capability_tags)

    llm_local = llm_data.get("must_be_local")
    if isinstance(llm_local, bool):
        intent.must_be_local = llm_local

    llm_free = llm_data.get("must_be_free")
    if isinstance(llm_free, bool):
        intent.must_be_free = llm_free

    llm_auth = _normalize_auth(llm_data.get("preferred_auth"))
    if llm_auth:
        intent.preferred_auth = llm_auth

    return intent
