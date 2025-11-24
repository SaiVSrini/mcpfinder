from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ToolEntry:
    """
    In-memory representation of a single row from the catalog CSV.

    Each instance describes one MCP tool that belongs to one MCP server.
    """

    server_name: str
    server_url: str
    server_description: str
    auth_type: str
    maturity: str
    compatability: List[str]
    tool_name: str
    tool_description: str
    example_queries: str
    actions_supported: List[str]
    capability_tags: List[str]
    embedded_text: str
    embedded_vector: Optional[List[float]]

    @property
    def combined_text(self) -> str:
        """
        Concatenate the key text fields into one blob.

        This is the text we use for simple keyword matching against the
        user query before we involve embeddings or an LLM.
        """
        text_chunks: List[str] = [
            self.server_name,
            self.server_description,
            self.tool_name,
            self.tool_description,
            self.example_queries,
            "\n".join(self.capability_tags),
            "\n".join(self.actions_supported),
        ]
        return "\n".join(chunk for chunk in text_chunks if chunk)


# These are the shapes we return from the MCP tool. They are intentionally
# loose dictionaries because the MCP client just needs JSON-compatible data.
ToolSuggestion = Dict[str, Any]
ServerSuggestion = Dict[str, Any]
