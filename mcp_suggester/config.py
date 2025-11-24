from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    # If python-dotenv is installed, load a local .env file so I can
    # configure things without exporting environment variables by hand.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # This import is optional. If it fails, I just fall back to the
    # normal environment variables.
    pass


def get_catalog_path() -> str:
    """
    Return the path to the catalog CSV file.

    - Uses MCP_CATALOG_PATH if it is set.
    - Otherwise falls back to ./db.csv relative to the project.
    - Raises a clear error if the file does not exist.
    """
    configured_path = os.environ.get("MCP_CATALOG_PATH") or "./db.csv"
    catalog_path = Path(configured_path).expanduser().resolve()
    if not catalog_path.exists():
        raise RuntimeError(
            f"Catalog file not found at '{catalog_path}'. "
            "Set MCP_CATALOG_PATH to a valid CSV path or create ./db.csv."
        )
    return str(catalog_path)


def get_openai_api_key() -> Optional[str]:
    """
    Return the OpenAI API key, or None if not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    return api_key if api_key else None


def get_model_name() -> str:
    """
    Return the chat model name for reranking.
    """
    return os.environ.get("MCP_SUGGEST_MODEL", "gpt-4o-mini")


def get_embedding_model() -> str:
    """
    Return the embedding model name.
    """
    return os.environ.get("MCP_EMBED_MODEL", "text-embedding-3-small")


def has_openai_api_key() -> bool:
    """
    Return True if an OpenAI API key is available.
    """
    return get_openai_api_key() is not None
