from __future__ import annotations

from typing import List, Optional

from fastmcp import FastMCP

from .catalog import get_catalog
from .intent import extract_intent
from .llm_rerank import llm_rerank
from .models import ServerSuggestion
from .scoring import select_candidates

mcp = FastMCP(name="MCP Suggestion Engine")


def suggest_mcp_servers_impl(
    user_query: str,
    top_n: int = 3,
    max_candidates: int = 20,
    filter_tags: Optional[List[str]] = None,
    preferred_client: Optional[str] = None,
) -> List[ServerSuggestion]:
    # Load all tools, understand the query, score them, then ask GPT to pick the best
    catalog_entries = get_catalog()
    intent = extract_intent(user_query)
    candidates = select_candidates(
        user_query=user_query,
        entries=catalog_entries,
        intent=intent,
        max_candidates=max_candidates,
        filter_tags=filter_tags,
    )

    # If nothing matches, return empty result
    if not candidates:
        return [
            {
                "server_name": "none",
                "server_url": "",
                "auth_type": "",
                "maturity": "",
                "compatability": [],
                "score": 0.0,
                "reason": "No matching servers found in the catalog for this query.",
                "tools": [],
            }
        ]

    return llm_rerank(user_query, candidates, top_n)


@mcp.tool
def suggest_mcp_servers(
    user_query: str,
    top_n: int = 3,
    max_candidates: int = 20,
    filter_tags: Optional[List[str]] = None,
) -> List[ServerSuggestion]:
    return suggest_mcp_servers_impl(
        user_query=user_query,
        top_n=top_n,
        max_candidates=max_candidates,
        filter_tags=filter_tags,
    )


def get_mcp_app() -> FastMCP:
    return mcp


if __name__ == "__main__":
    mcp.run()
