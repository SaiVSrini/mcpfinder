from __future__ import annotations

"""
Small helper that keeps db_with_embeddings.csv in sync with db.csv.

In my own words:
- I treat db.csv as the source of truth for tool metadata.
- Whenever I add or edit rows there, I run this script.
- It looks at db_with_embeddings.csv (if it exists), reuses any existing
  embeddings when the text has not changed, and only calls OpenAI for
  rows that still need an embedding.

The result is a fresh db_with_embeddings.csv that matches db.csv and
has up-to-date embedded_vector values wherever possible.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from mcp_suggester.config import get_embedding_model


SOURCE_CSV = Path("db.csv")
TARGET_CSV = Path("db_with_embeddings.csv")


def get_openai_client() -> Optional[OpenAI]:
    """
    Create an OpenAI client from the OPENAI_API_KEY environment variable.

    If the key is not set, return None so the script can still run and
    simply skip embedding generation.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "NOTE: OPENAI_API_KEY is not set. "
            "I will reuse existing embeddings but will not generate new ones.",
            file=sys.stderr,
        )
        return None
    return OpenAI(api_key=api_key)


def load_csv_rows(csv_path: Path) -> List[dict]:
    """
    Load all rows from a CSV file as a list of dictionaries.
    """
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def safe_vector_from_string(raw_value: str) -> Optional[List[float]]:
    """
    Try to parse an embedded_vector string into a list of floats.
    """
    cleaned_value = (raw_value or "").strip()
    if not cleaned_value:
        return None
    try:
        parsed = json.loads(cleaned_value)
        if isinstance(parsed, list) and all(
            isinstance(element, (int, float)) for element in parsed
        ):
            return [float(element) for element in parsed]
    except Exception:
        return None
    return None


def embed_text_with_retries(
    client: OpenAI,
    text_to_embed: str,
    model_name: str,
    max_retries: int = 5,
) -> List[float]:
    """
    Call the embeddings API with simple retry logic and backoff.
    """
    cleaned_text = text_to_embed.strip()
    if not cleaned_text:
        return []

    delay_seconds = 1.0
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=model_name,
                input=cleaned_text,
            )
            return response.data[0].embedding
        except Exception as exc:  # noqa: BLE001
            print(
                f"OpenAI error on attempt {attempt + 1}/{max_retries}: {exc}",
                file=sys.stderr,
            )
            if attempt == max_retries - 1:
                raise
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, 30.0)

    # Should not be reached, but keep the type happy.
    return []


def build_existing_embedding_index(
    rows: List[dict],
) -> Dict[Tuple[str, str], dict]:
    """
    Index an existing CSV (db_with_embeddings.csv) by (server_name, tool_name).

    This lets me reuse embedded_vector values when the same tool is present
    in the new db.csv and its embedded_text has not changed.
    """
    index: Dict[Tuple[str, str], dict] = {}
    for row in rows:
        server_name = (row.get("server_name") or "").strip()
        tool_name = (row.get("tool_name") or "").strip()
        if server_name and tool_name:
            index[(server_name, tool_name)] = row
    return index


def main() -> None:
    if not SOURCE_CSV.exists():
        raise SystemExit(f"Source CSV not found at {SOURCE_CSV}")

    source_rows = load_csv_rows(SOURCE_CSV)
    if not source_rows:
        print(f"No rows found in {SOURCE_CSV}. Nothing to do.")
        return

    if TARGET_CSV.exists():
        existing_rows = load_csv_rows(TARGET_CSV)
        existing_index = build_existing_embedding_index(existing_rows)
        print(f"Found existing target CSV with {len(existing_rows)} rows.")
    else:
        existing_index = {}
        print("No existing db_with_embeddings.csv found; starting from scratch.")

    client = get_openai_client()
    embedding_model_name = get_embedding_model()

    # Collect all fieldnames from the source, plus embedded_vector if missing.
    fieldnames = list(source_rows[0].keys())
    if "embedded_vector" not in fieldnames:
        fieldnames.append("embedded_vector")

    total_rows = len(source_rows)
    updated_rows: List[dict] = []
    embedded_count = 0
    reused_count = 0
    skipped_count = 0

    for row_index, source_row in enumerate(source_rows, start=1):
        server_name = (source_row.get("server_name") or "").strip()
        tool_name = (source_row.get("tool_name") or "").strip()

        # Work on a copy so we don't accidentally mutate the original in memory.
        output_row = dict(source_row)

        existing_row = existing_index.get((server_name, tool_name))
        existing_vector_str = (existing_row or {}).get("embedded_vector", "") or ""
        existing_text = (existing_row or {}).get("embedded_text", "") or ""

        embedded_text = (
            output_row.get("embedded_text")
            or output_row.get("embedding_text")
            or ""
        ).strip()

        if not embedded_text:
            # Nothing to embed; just keep whatever vector we might already have.
            output_row["embedded_vector"] = existing_vector_str
            skipped_count += 1
            updated_rows.append(output_row)
            continue

        # If we have an existing row with the same embedded_text and a valid
        # vector, just reuse it.
        existing_vector = safe_vector_from_string(existing_vector_str)
        if (
            existing_row is not None
            and embedded_text == existing_text.strip()
            and existing_vector is not None
        ):
            output_row["embedded_vector"] = existing_vector_str
            reused_count += 1
            updated_rows.append(output_row)
            continue

        if client is None:
            # We cannot generate a new embedding without an API key, so leave
            # the vector field empty (or as-is if it had something).
            output_row["embedded_vector"] = existing_vector_str
            skipped_count += 1
            updated_rows.append(output_row)
            continue

        # Generate a new embedding for this row.
        try:
            embedding = embed_text_with_retries(
                client=client,
                text_to_embed=embedded_text,
                model_name=embedding_model_name,
            )
            output_row["embedded_vector"] = json.dumps(embedding)
            embedded_count += 1
        except Exception as exc:  # noqa: BLE001
            print(
                f"Failed to embed row {row_index} "
                f"(server_name={server_name}, tool_name={tool_name}): {exc}",
                file=sys.stderr,
            )
            # Keep previous value if any, else empty.
            output_row["embedded_vector"] = existing_vector_str
            skipped_count += 1

        updated_rows.append(output_row)

    # Write out the new target CSV.
    with TARGET_CSV.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"Wrote {len(updated_rows)} rows to {TARGET_CSV}.")
    print(f"New embeddings generated: {embedded_count}")
    print(f"Existing embeddings reused: {reused_count}")
    print(f"Rows skipped (no text or no key): {skipped_count}")


if __name__ == "__main__":
    main()

