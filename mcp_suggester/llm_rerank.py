from __future__ import annotations

import json
from typing import Any, Dict, List

from openai import OpenAI

from .config import get_model_name, get_openai_api_key, has_openai_api_key
from .models import ServerSuggestion, ToolEntry, ToolSuggestion
from .scoring import basic_score


def _first_nonempty_line(text: str) -> str:
    """
    Take the first non-empty line from a block of text.

    This is handy for picking a single example query from a multiline field.
    """
    for line in (text or "").splitlines():
        stripped_line = line.strip()
        if stripped_line:
            return stripped_line
    return ""


def heuristic_group_only(
    user_query: str,
    candidates: List[ToolEntry],
    top_n: int = 3,
) -> List[ServerSuggestion]:
    """
    Group candidates by server_name and score using basic_score with no embeddings.

    This is the fully local, deterministic path that we fall back to when
    we cannot or do not want to call the LLM.
    """
    if not candidates:
        return []

    server_groups: Dict[str, List[ToolEntry]] = {}
    for entry in candidates:
        # Group all tools by the server that owns them.
        server_groups.setdefault(entry.server_name, []).append(entry)

    server_suggestions: List[ServerSuggestion] = []

    for server_name, group_entries in server_groups.items():
        tool_suggestions: List[ToolSuggestion] = []
        tool_scores: List[float] = []

        # Sort tools by heuristic score descending so the “best” tools are first.
        scored_tools: List[tuple[ToolEntry, float]] = []
        for entry in group_entries:
            score = basic_score(user_query, entry, query_embedding=None, filter_tags=None)
            scored_tools.append((entry, score))
        scored_tools.sort(key=lambda item: item[1], reverse=True)

        for entry, score in scored_tools[:3]:
            example_query = _first_nonempty_line(entry.example_queries)
            tool_suggestions.append(
                {
                    "tool_name": entry.tool_name,
                    "score": float(score),
                    "reason": "Heuristic match based on keywords and tags.",
                    "example_query_to_run": example_query,
                }
            )
            tool_scores.append(score)

        server_score = max(tool_scores) if tool_scores else 0.0
        first_entry = group_entries[0]

        server_suggestions.append(
            {
                "server_name": first_entry.server_name,
                "server_url": first_entry.server_url,
                "auth_type": first_entry.auth_type,
                "maturity": first_entry.maturity,
                "compatability": list(first_entry.compatability),
                "score": float(server_score),
                "reason": "Heuristic grouping based on keyword overlap and capability tags.",
                "tools": tool_suggestions,
            }
        )

    server_suggestions.sort(key=lambda s: s.get("score", 0.0), reverse=True)
    return server_suggestions[:top_n]


def llm_rerank(
    user_query: str,
    candidates: List[ToolEntry],
    top_n: int = 3,
) -> List[ServerSuggestion]:
    """
    Use an LLM to rerank and group candidates into server suggestions.
    Falls back to heuristic_group_only on any failure or when no API key is available.
    """
    if not candidates:
        return []

    if not has_openai_api_key():
        return heuristic_group_only(user_query, candidates, top_n)

    api_key = get_openai_api_key()
    if not api_key:
        return heuristic_group_only(user_query, candidates, top_n)

    # Prepare compact candidate list for the LLM.
    llm_candidates: List[Dict[str, Any]] = []
    for idx, entry in enumerate(candidates):
        llm_candidates.append(
            {
                "id": idx,
                "server_name": entry.server_name,
                "server_url": entry.server_url,
                "auth_type": entry.auth_type,
                "maturity": entry.maturity,
                "compatability": entry.compatability,
                "tool_name": entry.tool_name,
                "tool_description": entry.tool_description,
                "example_queries": entry.example_queries,
                "capability_tags": entry.capability_tags,
            }
        )

    system_msg = {
        "role": "system",
        "content": (
            "You are an MCP routing engine. "
            "You receive a user query and a list of MCP servers and tools. "
            "Your job is to pick and rank the best servers and tools. "
            "Respond with STRICT JSON only, no extra commentary."
        ),
    }

    user_content = {
        "user_query": user_query,
        "candidates": llm_candidates,
        "instructions": (
            "Return JSON of the form:\n"
            "{\n"
            '  \"servers\": [\n'
            "    {\n"
            '      \"server_name\": \"...\",\n'
            '      \"server_url\": \"...\",\n'
            '      \"auth_type\": \"...\",\n'
            '      \"maturity\": \"...\",\n'
            '      \"compatability\": [\"cursor\", \"claude_desktop\"],\n'
            '      \"score\": 0.0,\n'
            '      \"reason\": \"...\",\n'
            '      \"tools\": [\n'
            "        {\n"
            '          \"tool_name\": \"...\",\n'
            '          \"score\": 0.0,\n'
            '          \"reason\": \"...\",\n'
            '          \"example_query_to_run\": \"...\" \n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Pick and rank the best servers and tools for the given user_query. "
            "Only include servers and tools that are plausible matches."
        ),
    }

    try:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=get_model_name(),
            temperature=0.0,
            messages=[
                system_msg,
                {"role": "user", "content": json.dumps(user_content)},
            ],
        )
        content = completion.choices[0].message.content or ""

        # Strip optional Markdown code fences.
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped.lstrip("`")
            if stripped.startswith("json"):
                stripped = stripped[4:]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
            stripped = stripped.strip()

        parsed = json.loads(stripped)
        servers = parsed.get("servers", [])
        if not isinstance(servers, list):
            return heuristic_group_only(user_query, candidates, top_n)

        suggestions: List[ServerSuggestion] = []
        for server in servers[:top_n]:
            if not isinstance(server, dict):
                continue
            suggestion: ServerSuggestion = {
                "server_name": server.get("server_name", ""),
                "server_url": server.get("server_url", ""),
                "auth_type": server.get("auth_type", ""),
                "maturity": server.get("maturity", ""),
                "compatability": server.get("compatability", []) or [],
                "score": float(server.get("score", 0.0)),
                "reason": server.get("reason", ""),
                "tools": [],
            }

            tools = server.get("tools", []) or []
            if isinstance(tools, list):
                for tool in tools:
                    if not isinstance(tool, dict):
                        continue
                    suggestion["tools"].append(
                        {
                            "tool_name": tool.get("tool_name", ""),
                            "score": float(tool.get("score", 0.0)),
                            "reason": tool.get("reason", ""),
                            "example_query_to_run": tool.get(
                                "example_query_to_run", ""
                            ),
                        }
                    )

            suggestions.append(suggestion)

        if not suggestions:
            return heuristic_group_only(user_query, candidates, top_n)

        return suggestions[:top_n]
    except Exception as exc:  # noqa: BLE001
        # If the LLM call fails for any reason, fall back to a simpler but
        # predictable heuristic ranking so the caller still gets a result.
        print(
            f"WARNING: LLM rerank failed, falling back to heuristic scoring: {exc}"
        )
        return heuristic_group_only(user_query, candidates, top_n)
