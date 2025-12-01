from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from mcp_suggester.server import suggest_mcp_servers_impl

app = FastAPI(title="MCP Finder API")


class RecommendRequest(BaseModel):
    user_query: str
    top_n: int = 3
    max_candidates: int = 20
    filter_tags: Optional[List[str]] = None


class RecommendResponse(BaseModel):
    results: List[dict]


@app.post("/recommend", response_model=RecommendResponse)
async def recommend_tools(payload: RecommendRequest) -> RecommendResponse:
    suggestions = suggest_mcp_servers_impl(
        user_query=payload.user_query,
        top_n=payload.top_n,
        max_candidates=payload.max_candidates,
        filter_tags=payload.filter_tags,
    )
    return RecommendResponse(results=suggestions)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
