from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    # Load environment variables from a .env file if present.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Optional dependency; ignore failures.
    pass


def get_catalog_path() -> str:
    """
    Return the path to the catalog CSV.

    Defaults to ./db.csv if MCP_CATALOG_PATH is not set.
    Raises RuntimeError if the file does not exist.
    """
    env_value = os.environ.get("MCP_CATALOG_PATH") or "./db.csv"
    path = Path(env_value).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(
            f"Catalog file not found at '{path}'. "
            "Set MCP_CATALOG_PATH to a valid CSV path or create ./db.csv."
        )
    return str(path)


def get_openai_api_key() -> Optional[str]:
    """
    Return the OpenAI API key, or None if not set.
    """
    key = os.environ.get("OPENAI_API_KEY")
    return key if key else None


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

