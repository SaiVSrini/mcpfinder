# MCP Suggester

This project is my small “router” for Model Context Protocol (MCP) servers.

In plain words: I give this server a question in natural language, and it tells me which MCP server and tool are most likely to help, based on a SQLite catalog that I maintain.

The goal is to make choosing the right MCP server feel like calling a single “help me pick the right tool” endpoint.

---

## How the project works (in my own words)

### 1. The catalog: `mcpfinder.sqlite`

- I keep a SQLite database (`mcpfinder.sqlite` by default) in the project root.
- Inside it there is one table, `mcp_tools`, where each row describes **one tool on one MCP server**.
- Some of the columns:
  - `server_name`, `server_url`, `server_description`
  - `auth_type`, `maturity`, and `compatability`
  - `tool_name`, `tool_description`, and `example_queries`
  - `capability_tags`
  - `embedded_text` / `embedded_vector`

If I edit the CSV source of truth, I sync it into the SQLite DB with:

```bash
python -m mcp_suggester.load_mcp_csv_to_sqlite
```

The MCP server **never** recomputes embeddings for rows. It only reads what is already stored in SQLite. The active DB path can be overridden with `MCP_CATALOG_DB`.

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

After that I run `python -m mcp_suggester.load_mcp_csv_to_sqlite` to copy the enriched rows
into `mcpfinder.sqlite`, or point `MCP_CATALOG_DB` at whatever SQLite file I want.

### 3. How a query is scored

When I call the MCP tool `suggest_mcp_servers`, this is roughly what happens:

1. The server loads the catalog (`mcpfinder.sqlite` by default) into memory as a list of `CatalogEntry` Pydantic objects.
2. The server runs `extract_intent(user_query)` to build a `QueryIntent`. It uses heuristics locally and upgrades with OpenAI if a key is available. The intent captures things like desired capabilities, whether the user insists on local/offline tools, and auth preferences (`api-key`, `oauth`, `none`).
3. Every catalog entry builds a `combined_text` string from the server/tool descriptions, capability tags, compatibility strings, and example queries.
4. The hybrid retriever scores each tool using:
   - A TF‑IDF cosine similarity between the query tokens and the tool’s combined text.
   - (Optional) Embedding cosine similarity between the query vector and the stored tool embedding.
   - Capability bonuses when explicit `filter_tags` match tool tags.
   - Intent bonuses/penalties: matching auth methods, boosting tools tagged as `local`/`free` when the user insists on those constraints, and penalizing anything that violates them.
5. Tools with a positive score are sorted, and the best `max_candidates` items pass to the reranker.

If I do not set `OPENAI_API_KEY`, the server:

- Skips embedding generation (TF‑IDF only).
- Keeps intent extraction purely heuristic.
- Falls back to the deterministic grouping instead of the LLM reranker.

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

By default the server expects `MCP_CATALOG_DB` to point to the catalog SQLite file.

I usually do:

```bash
export MCP_CATALOG_DB=./mcpfinder.sqlite
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
from mcp_suggester.server import suggest_mcp_servers_impl
import json

results = suggest_mcp_servers_impl(
    user_query="Scan my Docker images for vulnerabilities.",
    top_n=3,
    max_candidates=20,
    filter_tags=["security", "docker"]
)

print(json.dumps(results, indent=2))
PY
```

### 5. Offline evaluation

The repo includes a small labeled dataset under `evaluation/eval_dataset.csv`. I can score the lexical, hybrid, and hybrid+LLM pipelines with:

```bash
python evaluation/evaluate.py
```

If `OPENAI_API_KEY` is set, the script also runs the final rerank stage; otherwise it reports the first two baselines only.

### 6. HTTP helper API

For tools that expect a plain HTTP endpoint, I run a small FastAPI wrapper:

```bash
uvicorn api_server:app --reload
```

`POST /recommend` accepts the exact same payload as the MCP tool and returns the same JSON, because it simply calls `suggest_mcp_servers_impl` under the hood.

### 7. Optional Streamlit UI

For a lightweight UI during demos, I run:

```bash
streamlit run ui_app.py
```

The app lets me type a query, tweak `top_n`/`max_candidates`, see server reasoning, and copy example tool invocations without leaving the browser.

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
        "MCP_CATALOG_DB": "/path/to/mcpfinder/mcpfinder.sqlite",
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

- The SQLite catalog is my **source of truth** for MCP servers and tools.
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
