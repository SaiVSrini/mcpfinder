from __future__ import annotations

import csv
import json
from typing import List, Optional

from .config import get_catalog_path
from .models import ToolEntry

# Simple in-process cache so we only read and parse the CSV once.
_CACHE: Optional[List[ToolEntry]] = None


def _parse_str_list(raw_value: str) -> List[str]:
    """
    Parse a CSV field that can either be comma-separated or newline-separated.

    This lets me be a bit sloppy in the CSV formatting while still ending up
    with a clean list of strings.
    """
    cleaned_value = (raw_value or "").strip()
    if not cleaned_value:
        return []
    separators = [",", "\n"]
    parts: List[str] = [cleaned_value]
    for separator in separators:
        next_parts: List[str] = []
        for piece in parts:
            next_parts.extend(piece.split(separator))
        parts = next_parts
    return [piece.strip() for piece in parts if piece.strip()]


def load_catalog_once() -> List[ToolEntry]:
    """
    Load the catalog CSV into ToolEntry objects once and cache the result.
    """
    global _CACHE

    catalog_path = get_catalog_path()
    parsed_entries: List[ToolEntry] = []

    try:
        with open(catalog_path, "r", newline="", encoding="utf-8") as file_handle:
            reader = csv.DictReader(file_handle)
            for row_index, row in enumerate(reader, start=1):
                try:
                    server_name = (row.get("server_name") or "").strip()
                    server_url = (row.get("server_url") or "").strip()
                    server_description = (row.get("server_description") or "").strip()
                    auth_type = (row.get("auth_type") or "").strip()
                    maturity = (row.get("maturity") or "").strip()
                    compatability = _parse_str_list(row.get("compatability") or "")
                    tool_name = (row.get("tool_name") or "").strip()
                    tool_description = (row.get("tool_description") or "").strip()
                    example_queries = (row.get("example_queries") or "").strip()
                    actions_supported = _parse_str_list(
                        row.get("actions_supported") or ""
                    )
                    capability_tags = _parse_str_list(
                        row.get("capability_tags") or ""
                    )
                    embedded_text = (row.get("embedded_text") or "").strip()

                    if not server_name or not tool_name:
                        print(
                            f"WARNING: Skipping row {row_index} due to missing server_name or tool_name."
                        )
                        continue

                    embedded_vector_raw = (row.get("embedded_vector") or "").strip()
                    embedded_vector: Optional[List[float]] = None
                    if embedded_vector_raw:
                        try:
                            parsed_vector = json.loads(embedded_vector_raw)
                            if isinstance(parsed_vector, list) and all(
                                isinstance(element, (int, float))
                                for element in parsed_vector
                            ):
                                embedded_vector = [
                                    float(element) for element in parsed_vector
                                ]
                            else:
                                print(
                                    f"WARNING: Row {row_index} embedded_vector is not a list of numbers; ignoring."
                                )
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"WARNING: Failed to parse embedded_vector in row {row_index}: {exc}"
                            )

                    catalog_entry = ToolEntry(
                        server_name=server_name,
                        server_url=server_url,
                        server_description=server_description,
                        auth_type=auth_type,
                        maturity=maturity,
                        compatability=compatability,
                        tool_name=tool_name,
                        tool_description=tool_description,
                        example_queries=example_queries,
                        actions_supported=actions_supported,
                        capability_tags=capability_tags,
                        embedded_text=embedded_text,
                        embedded_vector=embedded_vector,
                    )
                    parsed_entries.append(catalog_entry)
                except Exception as exc:  # noqa: BLE001
                    print(f"WARNING: Failed to parse row {row_index}: {exc}")
    except FileNotFoundError:
        # This should normally be caught by get_catalog_path, but guard anyway.
        raise RuntimeError(f"Catalog file not found at '{catalog_path}'.")

    _CACHE = parsed_entries
    print(f"Loaded {len(parsed_entries)} catalog entries from {catalog_path}.")
    return parsed_entries


def get_catalog() -> List[ToolEntry]:
    """
    Return the catalog entries, loading them once if needed.
    """
    global _CACHE
    if _CACHE is None:
        _CACHE = load_catalog_once()
    return _CACHE
