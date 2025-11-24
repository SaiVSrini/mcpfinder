# MCP Suggester

This project is my small “router” for Model Context Protocol (MCP) servers.

In plain words: I give this server a question in natural language, and it tells me which MCP server and tool are most likely to help, based on a CSV file that I maintain as a catalog.

The goal is to make choosing the right MCP server feel like calling a single “help me pick the right tool” endpoint.

---

## How the project works (in my own words)

### 1. The catalog: `db.csv`

- I keep a CSV file in the project root (usually `db.csv` or `db_with_embeddings.csv`).
- Each row describes **one tool on one MCP server**.
- Some of the fields:
  - `server_name`, `server_url`, `server_description`
  - `auth_type` and `maturity` (for example: api-key, unstable, stable)
  - `compatability` (which clients understand it, like Cursor, Claude Desktop, etc.)
  - `tool_name`, `tool_description`, and `example_queries`
  - `capability_tags` (free‑form tags I can filter by, such as `database`, `docker`, `pdf`)
  - `embedded_text` (a slightly richer free‑form description)
  - `embedded_vector` (a precomputed embedding stored as JSON – a long list of floats)

The MCP server **never** recomputes embeddings for rows. It only reads what I already stored in the CSV.

### 2. One‑time embedding generation

I use `generate_embeddings.py` to fill the `embedded_vector` column from the `embedded_text` column.

I run it like this:

```bash
export OPENAI_API_KEY=your_api_key_here
python generate_embeddings.py --input db.csv --output db_with_embeddings.csv
```

This script:

- Reads the input CSV.
- Looks at the `embedded_text` field.
- For rows that have text and no valid `embedded_vector` yet, it calls the OpenAI embeddings API (`text-embedding-3-small` by default).
- It writes the resulting vector into `embedded_vector` as JSON.

After that, I can either:

- Replace the original file:

  ```bash
  mv db_with_embeddings.csv db.csv
  ```

  and keep using the default `MCP_CATALOG_PATH=./db.csv`

- Or point `MCP_CATALOG_PATH` directly to `db_with_embeddings.csv`.

### 3. How a query is scored

When I call the MCP tool `suggest_mcp_servers`, this is roughly what happens:

1. The server loads the catalog (`db.csv` by default) into memory as a list of tool entries.
2. The server takes my `user_query` string and:
   - Breaks it into tokens (simple lowercase word splitting).
   - Optionally turns it into an **embedding vector** using the OpenAI embeddings API, if the `OPENAI_API_KEY` is set.
3. For every tool entry in the catalog, the code builds a “combined text” string out of:
   - server name and description
   - tool name and description
   - example queries
   - capability tags and actions
4. Each tool gets a **basic score** made from:
   - **Keyword overlap** between my query tokens and that combined text.
   - **Embedding cosine similarity** between the query embedding and the tool’s precomputed `embedded_vector` (only if both exist).
   - A small **tag bonus** if I passed `filter_tags` that overlap with the tool’s `capability_tags`.
5. Tools with a score of zero or less are ignored. The rest are sorted by score, and the top N (based on `max_candidates`) are kept as “candidates”.

If I do not set `OPENAI_API_KEY`, the server simply:

- Skips embeddings entirely.
- Uses only keyword overlap + tag bonus.

The behavior is the same overall, just less “semantic”.

### 4. What the LLM does

After the basic scoring, there is usually still more than one reasonable tool. At this point, the code asks an OpenAI chat model (by default `gpt-4o-mini`) to help.

The LLM receives:

- My original `user_query`.
- A compact JSON list of the candidate servers and tools (server name, URL, auth, tool name, description, tags, etc.).

The prompt asks the model to:

- Group tools by server.
- Decide which servers and tools are most relevant.
- Assign final scores.
- Write short, human‑readable reasons.
- Suggest an example query for each tool (e.g., something I can actually run).

The model must answer with **strict JSON** that looks like:

```json
{
  "servers": [
    {
      "server_name": "...",
      "server_url": "...",
      "auth_type": "...",
      "maturity": "...",
      "compatability": [...],
      "score": 0.0,
      "reason": "...",
      "tools": [
        {
          "tool_name": "...",
          "score": 0.0,
          "reason": "...",
          "example_query_to_run": "..."
        }
      ]
    }
  ]
}
```

The server code parses this JSON and returns a list of `ServerSuggestion` dictionaries to the client (for example, Cursor).

If anything goes wrong with the LLM call (no key, network error, bad JSON, etc.), the code falls back to a purely heuristic version that:

- Groups candidates by server.
- Ranks each server by the best tool score.
- Returns the top N servers with generic, heuristic reasons.

### 5. What my OpenAI API key is actually used for

My OpenAI key is only used for:

1. **Embeddings** (query‑time):
   - Turning the `user_query` text into a vector via `text-embedding-3-small` (or whatever I set as `MCP_EMBED_MODEL`).
2. **LLM reranking**:
   - Calling the chat model (default `gpt-4o-mini`) to group and rerank candidates and to produce the `reason` and `example_query_to_run` strings.

If `OPENAI_API_KEY` is not set:

- The server does not call OpenAI at all.
- The embedding field on the query side is treated as missing.
- The LLM rerank step is skipped and the heuristic grouping is used instead.

---

## How I run and test the MCP server

### 1. Install dependencies

From the project root:

```bash
pip install -r requirements.txt
```

### 2. Set up the catalog

By default the server expects `MCP_CATALOG_PATH` to point to the catalog CSV.

I usually do:

```bash
export MCP_CATALOG_PATH=./db_with_embeddings.csv  # or ./db.csv
```

### 3. Run the MCP server directly

If I just want to test it on the command line:

```bash
export OPENAI_API_KEY=your_api_key_here  # optional but recommended
python -m mcp_suggester.server
```

This starts the FastMCP server and exposes a single tool:

- `suggest_mcp_servers(user_query, top_n, max_candidates, filter_tags)`

### 4. Quick Python test without MCP

Sometimes I just want to see the raw JSON without going through a client:

```bash
python - << 'PY'
from mcp_suggester.server import suggest_mcp_servers
from mcp_suggester.catalog import get_catalog
import json

results = suggest_mcp_servers(
    user_query="Scan my Docker images for vulnerabilities.",
    top_n=3,
    max_candidates=20,
    filter_tags=["security", "docker"]
)

print(json.dumps(results, indent=2))
PY
```

---

## How I plug this into Cursor

In Cursor, I add an MCP server configuration similar to this (the exact key name may differ depending on the version of Cursor):

```json
{
  "mcpServers": {
    "mcp_suggester": {
      "type": "command",
      "command": "/path/to/python",
      "args": ["-m", "mcp_suggester.server"],
      "env": {
        "PYTHONPATH": "/path/to/mcpfinder",
        "MCP_CATALOG_PATH": "/path/to/mcpfinder/db_with_embeddings.csv",
        "OPENAI_API_KEY": "YOUR_OPENAI_API_KEY"
      }
    }
  }
}
```

After I reload MCP servers in Cursor, I see:

- A server called `mcp_suggester`.
- A tool called `suggest_mcp_servers`.

From there I can:

- Use the settings “Test tool” panel with a JSON body like:

  ```json
  {
    "user_query": "Summarize a long PDF from a public URL.",
    "top_n": 3,
    "max_candidates": 20,
    "filter_tags": []
  }
  ```

- Or call the tool from chat using the tools picker, letting Cursor fill in the arguments for me.

---

## Mental model

The way I think about this project:

- The CSV is my **source of truth** for MCP servers and tools.
- The embeddings turn fuzzy natural‑language questions into something I can compare numerically.
- The scoring functions do a first pass of “roughly right” filtering and ranking.
- The LLM then looks at a small shortlist and produces a cleaner, human‑friendly ranking and explanation.

From my point of view, when I ask “What MCP server should I use for X?”, this project answers that question by combining:

1. Simple statistics (keywords, tags, cosine similarity).
2. A small LLM that can reason about the candidates and write explanations.

Everything else (FastMCP, environment variables, CSV plumbing) is there just to make that interaction reliable and easy to reuse.

If the LLM ever fails (for example, rate limits or bad JSON),
the server prints a clear message and automatically falls back to the local
heuristic ranking so I still get a sensible answer instead of an error.
