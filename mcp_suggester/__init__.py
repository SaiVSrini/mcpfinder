from __future__ import annotations

from .server import get_mcp_app

# Public surface of this package: the MCP app factory and version.
__all__ = ["get_mcp_app", "__version__"]

__version__ = "1.0.0"
