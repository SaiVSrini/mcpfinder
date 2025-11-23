# MCP Suggester

MCP Suggester is an MCP server that suggests the best MCP servers and tools for a natural language query.  
It uses a CSV catalog (`db.csv`) of MCP servers/tools plus precomputed embeddings, keyword overlap, and an LLM for reranking.

## Setup

1. Create and activate a virtual environment (optional but recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure you have a `db.csv` file in the project root with the expected columns.

## Generating Embeddings

To precompute embeddings for rows where `embedded_text` is present:

```bash
export OPENAI_API_KEY=your_api_key_here
python generate_embeddings.py --input db.csv --output db_with_embeddings.csv
```

Then replace your catalog with the new file:

```bash
mv db_with_embeddings.csv db.csv
```

Or point `MCP_CATALOG_PATH` to `db_with_embeddings.csv` instead.

## Running the MCP Server

Set environment variables and run:

```bash
export MCP_CATALOG_PATH=./db.csv      # optional, defaults to ./db.csv
export OPENAI_API_KEY=your_api_key   # optional; without this, the server uses heuristic-only scoring
python -m mcp_suggester.server
```

The server exposes a single tool: `suggest_mcp_servers`.

## Cursor Integration Example

In Cursor, add an MCP server configuration similar to:

```json
{
  "mcpServers": {
    "mcp_suggester": {
      "command": ["python", "-m", "mcp_suggester.server"],
      "env": {
        "MCP_CATALOG_PATH": "./db.csv",
        "OPENAI_API_KEY": "YOUR_API_KEY_HERE"
      }
    }
  }
}
```

After configuring, you can call the `suggest_mcp_servers` tool from Cursor and receive structured JSON suggestions of MCP servers and tools for your query.

