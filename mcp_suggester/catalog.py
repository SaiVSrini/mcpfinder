from __future__ import annotations

import json
import sqlite3
from typing import List, Optional

from .config import get_catalog_db_path
from .models import CatalogEntry

# Simple in-process cache so we only read and parse the catalog once.
_CACHE: Optional[List[CatalogEntry]] = None


def _split_multi_value(raw_value: Optional[str]) -> List[str]:
    """
    Normalize newline and comma separated fields into a clean list.
    """
    if not raw_value:
        return []
    cleaned = raw_value.replace("\r", "\n")
    parts: List[str] = []
    for chunk in cleaned.split("\n"):
        tokens = chunk.split(",") if "," in chunk else [chunk]
        for token in tokens:
            normalized = token.strip()
            if normalized:
                parts.append(normalized)
    return parts


def _split_examples(raw_value: Optional[str]) -> List[str]:
    """
    Split the example_queries blob into individual examples.
    """
    if not raw_value:
        return []
    cleaned = raw_value.replace("\r", "\n")
    return [line.strip() for line in cleaned.split("\n") if line.strip()]


def _parse_vector(raw_value: Optional[str]) -> Optional[List[float]]:
    """
    Parse a JSON list of floats out of the embedded_vector column.
    """
    cleaned = (raw_value or "").strip()
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, list) and all(
        isinstance(element, (int, float)) for element in parsed
    ):
        return [float(element) for element in parsed]
    return None


def _hydrate_entry(row: sqlite3.Row) -> CatalogEntry:
    """
    Build a CatalogEntry model from the SQLite row.
    """
    capability_tags = _split_multi_value(row["capability_tags"])
    example_queries = _split_examples(row["example_queries"])
    compatability = _split_multi_value(row["compatability"])

    return CatalogEntry(
        server_name=row["server_name"] or "",
        server_url=row["server_url"] or "",
        server_description=row["server_description"] or "",
        auth_type=row["auth_type"] or "",
        maturity=row["maturity"] or "",
        compatability=compatability,
        tool_name=row["tool_name"] or "",
        tool_description=row["tool_description"] or "",
        example_queries=example_queries,
        capability_tags=capability_tags,
        embedded_text=row["embedded_text"],
        embedded_vector=_parse_vector(row["embedded_vector"]),
    )


def load_catalog_once() -> List[CatalogEntry]:
    """
    Load the catalog SQLite DB into CatalogEntry objects once and cache the result.
    """
    global _CACHE

    db_path = get_catalog_db_path()
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(
            """
            SELECT
                server_name,
                server_url,
                server_description,
                auth_type,
                maturity,
                compatability,
                tool_name,
                tool_description,
                example_queries,
                capability_tags,
                embedded_text,
                embedded_vector
            FROM mcp_tools
            """
        )
        rows = cursor.fetchall()
    finally:
        connection.close()

    parsed_entries = [_hydrate_entry(row) for row in rows]
    _CACHE = parsed_entries
    print(f"Loaded {len(parsed_entries)} catalog entries from {db_path}.")
    return parsed_entries


def get_catalog() -> List[CatalogEntry]:
    """
    Return the catalog entries, loading them once if needed.
    """
    global _CACHE
    if _CACHE is None:
        _CACHE = load_catalog_once()
    return _CACHE
