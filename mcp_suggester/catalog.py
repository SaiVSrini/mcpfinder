from __future__ import annotations

import csv
import json
from typing import List, Optional

from .config import get_catalog_path
from .models import ToolEntry

_CACHE: Optional[List[ToolEntry]] = None


def _parse_str_list(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    separators = [",", "\n"]
    values: List[str] = [raw]
    for sep in separators:
        temp: List[str] = []
        for item in values:
            temp.extend(item.split(sep))
        values = temp
    return [v.strip() for v in values if v.strip()]


def load_catalog_once() -> List[ToolEntry]:
    """
    Load the catalog CSV into ToolEntry objects once and cache the result.
    """
    global _CACHE

    path = get_catalog_path()
    entries: List[ToolEntry] = []

    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
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
                    actions_supported = _parse_str_list(row.get("actions_supported") or "")
                    capability_tags = _parse_str_list(row.get("capability_tags") or "")
                    embedded_text = (row.get("embedded_text") or "").strip()

                    if not server_name or not tool_name:
                        print(
                            f"WARNING: Skipping row {idx} due to missing server_name or tool_name."
                        )
                        continue

                    embedded_vector_raw = (row.get("embedded_vector") or "").strip()
                    embedded_vector: Optional[List[float]] = None
                    if embedded_vector_raw:
                        try:
                            parsed = json.loads(embedded_vector_raw)
                            if isinstance(parsed, list) and all(
                                isinstance(x, (int, float)) for x in parsed
                            ):
                                embedded_vector = [float(x) for x in parsed]
                            else:
                                print(
                                    f"WARNING: Row {idx} embedded_vector is not a list of numbers; ignoring."
                                )
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"WARNING: Failed to parse embedded_vector in row {idx}: {exc}"
                            )

                    entry = ToolEntry(
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
                    entries.append(entry)
                except Exception as exc:  # noqa: BLE001
                    print(f"WARNING: Failed to parse row {idx}: {exc}")
    except FileNotFoundError:
        # This should normally be caught by get_catalog_path, but guard anyway.
        raise RuntimeError(f"Catalog file not found at '{path}'.")

    _CACHE = entries
    print(f"Loaded {len(entries)} catalog entries from {path}.")
    return entries


def get_catalog() -> List[ToolEntry]:
    """
    Return the catalog entries, loading them once if needed.
    """
    global _CACHE
    if _CACHE is None:
        _CACHE = load_catalog_once()
    return _CACHE

