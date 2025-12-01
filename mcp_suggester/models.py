from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CatalogEntry(BaseModel):
    """
    Strongly typed representation of a catalog row loaded from SQLite.
    """

    server_name: str
    server_url: str
    server_description: str
    auth_type: Optional[str] = None
    maturity: Optional[str] = None
    compatability: List[str] = Field(default_factory=list)
    tool_name: str
    tool_description: str
    example_queries: List[str] = Field(default_factory=list)
    capability_tags: List[str] = Field(default_factory=list)
    embedded_text: Optional[str] = None
    embedded_vector: Optional[List[float]] = None

    model_config = {"frozen": True}

    @property
    def combined_text(self) -> str:
        """
        Concatenate all searchable text into one string for lexical scoring.
        """
        text_chunks: List[str] = [
            self.server_name,
            self.server_description,
            self.tool_name,
            self.tool_description,
            "\n".join(self.example_queries),
            "\n".join(self.capability_tags),
            "\n".join(self.compatability),
        ]
        return "\n".join(chunk for chunk in text_chunks if chunk)


# These are the shapes we return from the MCP tool. They are intentionally
# loose dictionaries because the MCP client just needs JSON-compatible data.
ToolSuggestion = Dict[str, Any]
ServerSuggestion = Dict[str, Any]
