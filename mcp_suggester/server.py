from __future__ import annotations

from typing import List, Optional

from fastmcp import FastMCP

from .catalog import get_catalog
from .intent import extract_intent
from .llm_rerank import llm_rerank
from .models import ServerSuggestion
from .scoring import select_candidates

# This is the MCP app object that Cursor (or any MCP client) will talk to.
mcp = FastMCP(name="MCP Suggestion Engine")


def suggest_mcp_servers_impl(
    user_query: str,
    top_n: int = 3,
    max_candidates: int = 20,
    filter_tags: Optional[List[str]] = None,
) -> List[ServerSuggestion]:
    """
    Suggest the best MCP servers and tools for the given user query.

    The logic is:
    - Load the catalog from SQLite.
    - Score every tool against the query and pick a shortlist.
    - Ask the reranker (LLM or heuristic) to group and rank servers.
    """
    catalog_entries = get_catalog()
    intent = extract_intent(user_query)
    candidates = select_candidates(
        user_query=user_query,
        entries=catalog_entries,
        intent=intent,
        max_candidates=max_candidates,
        filter_tags=filter_tags,
    )

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

    # Use LLM rerank when possible; it already falls back on failure.
    return llm_rerank(user_query, candidates, top_n)


@mcp.tool
def suggest_mcp_servers(
    user_query: str,
    top_n: int = 3,
    max_candidates: int = 20,
    filter_tags: Optional[List[str]] = None,
) -> List[ServerSuggestion]:
    """
    FastMCP exposed tool proxying to the core implementation.
    """
    return suggest_mcp_servers_impl(
        user_query=user_query,
        top_n=top_n,
        max_candidates=max_candidates,
        filter_tags=filter_tags,
    )


def get_mcp_app() -> FastMCP:
    """
    Return the FastMCP application instance.
    """
    return mcp


if __name__ == "__main__":
    mcp.run()
