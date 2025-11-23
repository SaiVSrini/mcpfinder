from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ToolEntry:
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
        Concatenate key text fields for keyword-based scoring:
        server_name, server_description, tool_name, tool_description,
        example_queries, capability_tags (joined), actions_supported (joined).
        """
        parts: List[str] = [
            self.server_name,
            self.server_description,
            self.tool_name,
            self.tool_description,
            self.example_queries,
            "\n".join(self.capability_tags),
            "\n".join(self.actions_supported),
        ]
        return "\n".join(p for p in parts if p)


ToolSuggestion = Dict[str, Any]
ServerSuggestion = Dict[str, Any]

