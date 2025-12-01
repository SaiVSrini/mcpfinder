import sqlite3
import csv
import json
from pathlib import Path

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

CSV_PATH = Path("db_with_embeddings.csv")   # input CSV
DB_PATH = Path("mcpfinder.sqlite")          # output SQLite DB file

TABLE_NAME = "mcp_tools"                    # you can rename if you like

# ---------------------------------------------------------------------
# Create SQLite connection and table
# ---------------------------------------------------------------------

def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()

    # Note: keeping column names exactly as in the CSV
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            server_name        TEXT,
            server_url         TEXT,
            server_description TEXT,
            auth_type          TEXT,
            maturity           TEXT,
            compatability      TEXT,
            tool_name          TEXT,
            tool_description   TEXT,
            example_queries    TEXT,
            capability_tags    TEXT,
            embedded_text      TEXT,
            embedded_vector    TEXT  -- JSON string of the embedding
        );
    """)

    conn.commit()


# ---------------------------------------------------------------------
# Load CSV and insert into DB
# ---------------------------------------------------------------------

def load_csv_into_db(conn: sqlite3.Connection, csv_path: Path):
    cur = conn.cursor()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        rows_inserted = 0

        for row in reader:
            # Normalize embedding vector:
            # - CSV already stores it as a JSON-like string "[0.1, 0.2, ...]"
            # - We parse it and re-dump it to ensure valid JSON
            raw_vec = row.get("embedded_vector") or "[]"

            try:
                vec_list = json.loads(raw_vec)
            except json.JSONDecodeError:
                # If something is malformed, fall back to empty list
                vec_list = []

            embedding_json = json.dumps(vec_list, separators=(",", ":"))

            # Build values in the same order as the INSERT statement
            values = (
                row.get("server_name"),
                row.get("server_url"),
                row.get("server_description"),
                row.get("auth_type"),
                row.get("maturity"),
                row.get("compatability"),
                row.get("tool_name"),
                row.get("tool_description"),
                row.get("example_queries"),
                row.get("capability_tags"),
                row.get("embedded_text"),
                embedding_json,
            )

            cur.execute(f"""
                INSERT INTO {TABLE_NAME} (
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
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, values)

            rows_inserted += 1

    conn.commit()
    print(f"Inserted {rows_inserted} rows into table `{TABLE_NAME}` in {DB_PATH}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found at {CSV_PATH.resolve()}")

    # Create DB file if it does not exist; connect otherwise
    conn = sqlite3.connect(DB_PATH)
    try:
        init_db(conn)
        load_csv_into_db(conn, CSV_PATH)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
