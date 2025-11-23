#!/usr/bin/env python3
"""
Generate embeddings for each row in a CSV and store them in a new column.

- Reads a CSV with a column like "embedding_text" (manual description).
- Calls OpenAI text-embedding-3-small to embed that text.
- Writes a new CSV with all original columns + a new "embedded_vector" column
  containing the JSON-encoded embedding.

Usage:
    python generate_embeddings.py \
        --input db.csv \
        --output db_with_embeddings.csv \
        --text-column embedding_text \
        --vector-column embedded_vector
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import List, Optional

from openai import OpenAI



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings for a CSV column.")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input CSV file (e.g. db.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output CSV file (e.g. db_with_embeddings.csv)",
    )
    parser.add_argument(
        "--text-column",
        "-t",
        default="embedding_text",
        help="Name of the column containing text to embed (default: embedding_text)",
    )
    parser.add_argument(
        "--vector-column",
        "-v",
        default="embedded_vector",
        help="Name of the output column to store JSON embeddings (default: embedded_vector)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="text-embedding-3-small",
        help="Embedding model to use (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="If set, regenerate embeddings even if vector column is already filled.",
    )
    return parser.parse_args()


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key)


def load_rows(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def safe_float_list(value: str) -> Optional[List[float]]:
    """
    Try to parse an existing JSON list of floats. Returns None if invalid/empty.
    """
    value = (value or "").strip()
    if not value:
        return None
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        return None
    return None


def embed_text(client: OpenAI, text: str, model: str, max_retries: int = 5) -> List[float]:
    """
    Call OpenAI embeddings API with simple retry logic.
    """
    text = text.strip()
    if not text:
        return []

    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(
                model=model,
                input=text,
            )
            return resp.data[0].embedding
        except Exception as e:  # noqa: BLE001
            # Basic backoff on transient errors
            print(f"OpenAI error on attempt {attempt + 1}/{max_retries}: {e}", file=sys.stderr)
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    # Should not reach here
    return []


def main() -> None:
    args = parse_args()
    client = get_openai_client()

    print(f"Loading rows from: {args.input}")
    rows = load_rows(args.input)

    if not rows:
        print("No rows found in input CSV. Exiting.")
        return

    # Ensure vector column exists in header
    fieldnames = list(rows[0].keys())
    if args.vector_column not in fieldnames:
        fieldnames.append(args.vector_column)

    total = len(rows)
    print(f"Total rows: {total}")
    processed = 0
    skipped = 0
    embedded = 0

    for idx, row in enumerate(rows):
        text = (row.get(args.text_column, "") or "").strip()
        if not text:
            # No text to embed
            row[args.vector_column] = row.get(args.vector_column, "") or ""
            skipped += 1
            continue

        existing_vec_str = row.get(args.vector_column, "")
        existing_vec = safe_float_list(existing_vec_str)

        if existing_vec is not None and not args.overwrite_existing:
            # Already has a valid embedding, keep it
            skipped += 1
            continue

        # Generate new embedding
        try:
            vec = embed_text(client, text, args.model)
        except Exception as e:
            print(f"Failed to embed row {idx} (server_name={row.get('server_name', '')}, "
                  f"tool_name={row.get('tool_name', '')}): {e}", file=sys.stderr)
            # Keep previous value if any, else empty
            row[args.vector_column] = existing_vec_str or ""
            skipped += 1
            continue

        row[args.vector_column] = json.dumps(vec)
        embedded += 1

        processed += 1
        if processed % 10 == 0 or processed == total:
            print(f"Processed {processed}/{total} rows "
                  f"(embedded={embedded}, skipped={skipped})")

    print(f"Writing output CSV to: {args.output}")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")
    print(f"Embedded rows: {embedded}")
    print(f"Skipped rows (empty text or already embedded): {skipped}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
