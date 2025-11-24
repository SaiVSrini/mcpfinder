#!/usr/bin/env python3
"""
Small helper script I use to generate embeddings for my catalog CSV.

What it does, in my own words:
- Reads a CSV file that has a column with free-text descriptions.
- Calls the OpenAI embeddings API on that text.
- Writes a new CSV with all original columns plus an "embedded_vector" column
  containing the JSON-encoded embedding.

Example usage:
    python generate_embeddings.py \
        --input db.csv \
        --output db_with_embeddings.csv \
        --text-column embedded_text \
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
    """
    Parse command-line arguments for the embedding generator.
    """
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
    """
    Create an OpenAI client using the OPENAI_API_KEY from the environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key)


def load_rows(csv_path: str) -> List[dict]:
    """
    Load all rows from the CSV into a list of dictionaries.
    """
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    return rows


def safe_float_list(raw_value: str) -> Optional[List[float]]:
    """
    Try to parse an existing JSON list of floats.

    Returns None if the value is empty or not a list.
    """
    cleaned_value = (raw_value or "").strip()
    if not cleaned_value:
        return None
    try:
        parsed = json.loads(cleaned_value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        return None
    return None


def embed_text(
    client: OpenAI,
    text_to_embed: str,
    model_name: str,
    max_retries: int = 5,
) -> List[float]:
    """
    Call the OpenAI embeddings API with simple retry logic and backoff.
    """
    cleaned_text = text_to_embed.strip()
    if not cleaned_text:
        return []

    delay_seconds = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(
                model=model_name,
                input=cleaned_text,
            )
            return resp.data[0].embedding
        except Exception as e:  # noqa: BLE001
            # Basic backoff on transient errors
            print(f"OpenAI error on attempt {attempt + 1}/{max_retries}: {e}", file=sys.stderr)
            if attempt == max_retries - 1:
                raise
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, 30.0)
    # Should not reach here
    return []


def main() -> None:
    """
    Entry point for the embedding generation script.
    """
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

    total_rows = len(rows)
    print(f"Total rows: {total_rows}")
    processed_rows = 0
    skipped_rows = 0
    embedded_rows = 0

    for row_index, row in enumerate(rows):
        text_to_embed = (row.get(args.text_column, "") or "").strip()
        if not text_to_embed:
            # No text to embed for this row; just keep whatever is already there.
            row[args.vector_column] = row.get(args.vector_column, "") or ""
            skipped_rows += 1
            continue

        existing_vector_str = row.get(args.vector_column, "")
        existing_vector = safe_float_list(existing_vector_str)

        if existing_vector is not None and not args.overwrite_existing:
            # Already has a valid embedding, keep it.
            skipped_rows += 1
            continue

        # Generate a fresh embedding.
        try:
            new_embedding = embed_text(client, text_to_embed, args.model)
        except Exception as exc:  # noqa: BLE001
            print(
                f"Failed to embed row {row_index} "
                f"(server_name={row.get('server_name', '')}, "
                f"tool_name={row.get('tool_name', '')}): {exc}",
                file=sys.stderr,
            )
            # Keep previous value if any, else empty.
            row[args.vector_column] = existing_vector_str or ""
            skipped_rows += 1
            continue

        row[args.vector_column] = json.dumps(new_embedding)
        embedded_rows += 1

        processed_rows += 1
        if processed_rows % 10 == 0 or processed_rows == total_rows:
            print(
                f"Processed {processed_rows}/{total_rows} rows "
                f"(embedded={embedded_rows}, skipped={skipped_rows})"
            )

    print(f"Writing output CSV to: {args.output}")
    with open(args.output, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")
    print(f"Embedded rows: {embedded_rows}")
    print(f"Skipped rows (empty text or already embedded): {skipped_rows}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
