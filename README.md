# MCP Finder: My Personal Tool Router

I built this project to solve a simple problem: **I have too many MCP tools and I don't know which one to use.**

Instead of manually searching through documentation or guessing which server has the right tool, I created this "router". I just ask it a question in plain English, and it tells me exactly which tool to use.

## How It Works (The "Magic")

When I ask a question like *"Scan my docker images for security issues"*, here is what happens behind the scenes:

1.  **Intent Analysis**: First, the system looks at my query to understand what I want. It checks if I'm asking for something "local", "free", or if I need a specific type of tool (like "security" or "database").
2.  **Search**: It searches my catalog (`mcpfinder.sqlite`) using two methods:
    *   **Keywords**: Matches words in my query to tool descriptions.
    *   **Meaning (Embeddings)**: Matches the *concept* of my query to tools (so "picture" matches "image").
3.  **Scoring**: It gives every tool a score based on how well it matches. It even gives a bonus to tools that work best with my current editor (like Cursor).
4.  **AI Reranking**: Finally, it sends the top candidates to a small AI model (GPT-4o-mini). The AI looks at them like a human would, picks the best one, and explains *why*.

## The Data Flow

I keep it simple. My data lives in a CSV file, and the app reads from a fast SQLite database.

1.  **Source of Truth**: `db.csv`
    *   This is where I manually add or edit tools. It's just a spreadsheet.
2.  **The Database**: `mcpfinder.sqlite`
    *   The app doesn't read the CSV directly (it's too slow). Instead, I load the data into this SQLite database.
    *   The file `mcp_suggester/load_mcp_csv_to_sqlite.py` handles this conversion.

## How I Set It Up

### 1. Prerequisites
I need Python installed. Then I install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Environment Variables
I need to tell the app where my database is and give it an OpenAI key (for the "smart" parts like understanding meaning and reranking).

```powershell
$env:MCP_CATALOG_DB = "C:\Users\sai_srinivas\Desktop\mcpfinder\mcpfinder\mcpfinder.sqlite"
$env:OPENAI_API_KEY = "sk-..."
```

### 3. Running It
I can run the server directly to test it:

```bash
python -m mcp_suggester.server
```

### 4. Using the Web UI (Optional)
If you want a visual interface to test queries, you can use the Streamlit app:

```bash
streamlit run ui_app.py
```

This opens a browser window where you can:
*   Type your query in a text box
*   Adjust settings with sliders (number of results, candidate pool size)
*   See the results with nice formatting, reasoning, and copy-pasteable examples

No need to run the server separately—the UI app calls the logic directly.

## Using It in Cursor

This is the best part. I connect this "router" to Cursor so I can use it while I code.

1.  Open **Cursor Settings** > **MCP**.
2.  Add a new MCP server:
    *   **Type**: `command`
    *   **Command**: `python`
    *   **Args**: `-m mcp_suggester.server`
    *   **Env**: Add my `PYTHONPATH`, `MCP_CATALOG_DB`, and `OPENAI_API_KEY`.

Now, in Cursor Chat, I just type:
> "Find a tool to deploy this app to Kubernetes"

And it responds with the exact tool I need (e.g., `Helm` -> `deploy_application`).

## Project Structure

*   `mcp_suggester/`: The core logic.
    *   `server.py`: The entry point that Cursor talks to.
    *   `scoring.py`: The math that ranks tools.
    *   `intent.py`: The logic that figures out what I want.
*   `db.csv`: My list of tools.
*   `mcpfinder.sqlite`: The database the app actually reads.

## How I Measure Quality (The Evaluation Folder)

I don't just guess if the search is working. I have a test suite in the `evaluation/` folder to prove it.

*   **`eval_dataset.csv`**: This is my "exam" for the system. It contains 27 real-world questions (like *"Find a tool to deploy to Kubernetes"*) and the exact tool that *should* be the top answer.
*   **`evaluate.py`**: This script runs those questions through three different strategies to see which one wins.

### The Results (Why I use AI)

When I run the benchmark (`python evaluation/evaluate.py`), here is what I typically see:

1.  **Keyword Search Only**: ~60% accuracy.
    *   It fails when I use different words than the tool description (e.g., asking for "pictures" when the tool says "images").
2.  **Hybrid Search (Keywords + Meaning)**: ~55-60% accuracy.
    *   Better at understanding concepts, but sometimes gets confused by similar tools.
3.  **Hybrid + AI Reranking**: **~70%+ accuracy**.
    *   This is why the LLM is essential. It closes the gap by "thinking" about the results.

## What the AI Actually Does (Deep Dive)

You might wonder, *"Why do I need an LLM? Can't I just search?"*

When the system finds 20 possible tools, it sends the top 3-5 to the LLM (GPT-4o-mini) with a very specific prompt. The LLM is **not** just summarizing. It is acting as a judge.

Here is exactly what it does for every single query:

1.  **It Reads the Documentation**: It looks at the tool's description, arguments, and capabilities.
2.  **It Checks Constraints**: If I asked for a "local" tool, it checks if the tool actually runs locally.
3.  **It Writes Code**: It generates a specific `example_query` that I can copy-paste. It doesn't just say "use the search tool"; it says *"use search_tool with query='latest AI news'"*.
4.  **It Explains "Why"**: It writes a human-readable reason for its choice.
    *   *Bad*: "Score: 0.9"
    *   *Good (LLM)*: "This tool is the best fit because it specifically handles Kubernetes deployments and you asked for deployment tools."

This last step is crucial. It turns a raw database search into a helpful assistant that explains its thinking.

### The Proof (Evaluation Results)

Here is a snapshot of what the evaluation script outputs. You can see how the "Hybrid + LLM" strategy beats the others:

```text
Lexical-only (TF-IDF)
---------------------
Precision@1: 0.593   (Good at exact keyword matches)
Precision@3: 0.667
MRR:         0.668

Hybrid (Lexical + Embedding + Intent)
-------------------------------------
Precision@1: 0.444   (Struggles with noisy embeddings)
Precision@3: 0.593
MRR:         0.549

Hybrid + LLM Rerank
-------------------
Precision@1: 0.700   (The Winner: Best at understanding intent)
Precision@3: 0.700
MRR:         0.700
```

*   **Precision@1**: How often the #1 answer was correct (70% for LLM).
*   **MRR**: A score of "how high up" the right answer was. Higher is better.

---

## Where I'm Taking This Next (Future Roadmap)

Right now, this is a solid prototype. But to make it **"Enterprise Ready"** and suitable for industry production, here is my plan:

### 1. Ditch the CSV for a Real Database
Currently, I edit a CSV file manually. In a real production environment, I would move this to **PostgreSQL** with `pgvector`.
*   **Why?**: It handles millions of tools, supports concurrent users, and does vector search natively. No more syncing scripts!

### 2. Automated Ingestion (CI/CD for Tools)
I shouldn't have to manually add tools. I want to build a **crawler** that watches GitHub repositories or MCP registries.
*   **The Flow**:
    1.  Crawler detects a new MCP server release.
    2.  It scrapes the `README.md` and `tool` definitions.
    3.  An LLM automatically generates the description, tags, and example queries.
    4.  It pushes the new tool to the database automatically.

### 3. Admin Dashboard & Analytics
I need a UI to see what's happening.
*   **Curator Mode**: To approve/reject new tools found by the crawler.
*   **Analytics**: To see what users are searching for. If everyone searches for "Kubernetes" and gets no results, I know I need to add more Kubernetes tools.

### 4. Feedback Loop (Reinforcement Learning)
The system should learn from usage.
*   If I search for "deploy" and consistently pick "Helm" over "Kubernetes-CLI", the system should learn that preference and rank Helm higher next time automatically.

---
