# RAG Server - Retrieval-Augmented Generation Toolbox

A minimal, pipe-based RAG (Retrieval-Augmented Generation) system for MLX Knife.

**Philosophy:** Unix pipes + standalone tools + reusable components

**Status:** ✅ Runnable
**Runnable against:** mlxk ≥ 2.0.7 (`mlxk embed` is experimental, alpha-gated via `MLXK2_ENABLE_ALPHA_FEATURES=1`)
**Requires:** Python `numpy fastapi uvicorn httpx pydantic`; `mlxk serve` for the HTTP server. Pull an MLX embedding model first, e.g. `mlxk pull mlx-community/bge-small-en-v1.5-4bit`. The scripts set `MLXK2_ENABLE_ALPHA_FEATURES=1` for their own `mlxk embed` subprocess calls.
**Run:** `./rag-pipeline.sh "how does auth work?" index.jsonl 3`

> Best-effort consumer demo, not part of mlxk core. The standalone tools
> (`cosine-search.py`, `retrieve-files.py`, `index-files.py`) are written and
> reusable; `mlxk embed` ships (experimental) in 2.0.7.

---

## Overview

This toolbox provides:
- **Standalone Python tools** for embedding, search, and retrieval
- **Shell pipeline** for RAG workflows
- **HTTP server** for OpenAI-compatible API with automatic context injection

All tools are **independent and reusable** - use them individually or combine via pipes.

---

## Quick Start

### 1. Setup

```bash
cd examples/rag-server

# mlxk embed is experimental in 2.0.7 — enable the alpha gate
export MLXK2_ENABLE_ALPHA_FEATURES=1

# Pull an MLX embedding model (encoder or decoder embedder)
mlxk pull mlx-community/bge-small-en-v1.5-4bit
```

### 2. Create Index

```bash
# Index your project files
./index-files.py ./your-project -o project-index.jsonl

# Or specific files
./index-files.py file1.py file2.py -o files-index.jsonl

# Or with glob
./index-files.py "**/*.py" -o code-index.jsonl --recursive
```

### 3. Test Pipeline

```bash
# Test RAG pipeline
./rag-pipeline.sh "How does authentication work?" project-index.jsonl 3

# Output: Relevant file contents with scores
```

### 4. Start Server

```bash
# The RAG server is a transparent middleware: it has no model of its own — it
# retrieves context and forwards chat to a real `mlxk serve`. Put it on the port
# your client talks to (many chat UIs, e.g. nChat, are pinned to 8000) and move the
# backend aside with MLXK_PORT so they do not collide. CORS is built in, so browser
# clients can reach it.

# Terminal 1: real LLM backend on any free port (here 8001)
mlxk serve --port 8001

# Terminal 2: transparent RAG server on 8000 — point your client here.
#   MLXK_PORT=8001 tells it where the backend is (default 8000 would self-forward).
MLXK2_ENABLE_ALPHA_FEATURES=1 MLXK_PORT=8001 ./rag-server.py --index project-index.jsonl --port 8000

# Terminal 3: any OpenAI client (curl, SDK, or a chat UI like nChat) -> localhost:8000.
# A vanilla call gets RAG transparently (enable_rag defaults on); override per request if needed.
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<a-chat-model-in-your-cache>",
    "messages": [{"role": "user", "content": "How does login work?"}]
  }'
```

---

## Tools Reference

### 1. `index-files.py` - Create Embeddings Index

**Standalone indexer for files.**

```bash
# Basic usage
./index-files.py ./src -o index.jsonl

# Recursive with pattern
./index-files.py ./docs -o docs-index.jsonl --pattern "*.md" --recursive

# Custom embedding model
./index-files.py ./code -o code-index.jsonl --model bge-small

# Multiple paths
./index-files.py auth.py user.py db.py -o backend-index.jsonl
```

**Output:** JSONL file with embeddings and metadata
```jsonl
{"embedding": [...], "text": "...", "filename": "auth.py", "filepath": "/abs/path/auth.py"}
{"embedding": [...], "text": "...", "filename": "user.py", "filepath": "/abs/path/user.py"}
```

---

### 2. `cosine-search.py` - Vector Similarity Search

**Standalone search tool.**

```bash
# Search from query file
./cosine-search.py index.jsonl query.json --top-k 5

# Search from stdin (pipe)
echo '{"embedding": [...]}' | ./cosine-search.py index.jsonl - --top-k 3

# JSON output for pipes
cat query.json | ./cosine-search.py index.jsonl - --output-json

# Minimum score threshold
./cosine-search.py index.jsonl query.json --min-score 0.7
```

**Output (human-readable):**
```
[0.923] auth.py
  Preview: class AuthHandler:...

[0.871] user.py
  Preview: class User:...
```

**Output (JSON):**
```json
{
  "results": [
    {"score": 0.923, "filename": "auth.py", "filepath": "/path/auth.py", "text": "..."},
    {"score": 0.871, "filename": "user.py", "filepath": "/path/user.py", "text": "..."}
  ]
}
```

---

### 3. `retrieve-files.py` - Load File Contents

**Standalone file loader.**

```bash
# From search results
./cosine-search.py index.jsonl query.json --output-json \
  | ./retrieve-files.py

# From file
./retrieve-files.py search-results.json

# Custom format
./retrieve-files.py - --format markdown --include-score

# Limit files
./retrieve-files.py - --max-files 3
```

**Formats:**
- `text` (default): Plain text with separators
- `markdown`: Markdown code blocks
- `json`: Structured JSON

---

### 4. `rag-pipeline.sh` - Complete RAG Pipeline

**Shell script orchestrating the full pipeline.**

```bash
# Basic usage
./rag-pipeline.sh "query" index.jsonl 3

# Custom embedding model
EMBED_MODEL=bge-small ./rag-pipeline.sh "query" index.jsonl 5

# Use in other scripts
CONTEXT=$(./rag-pipeline.sh "$QUERY" index.jsonl 3)
echo "$CONTEXT" | mlxk run qwen3 - "Summarize this"
```

**Pipeline:**
```
Query → mlxk embed → cosine-search.py → retrieve-files.py → Context
```

---

### 5. `rag-server.py` - HTTP Server

**OpenAI-compatible API server with RAG.**

```bash
# Start server
./rag-server.py --index project-index.jsonl --port 8001

# With custom backend
MLXK_HOST=192.168.1.100 MLXK_PORT=9000 \
  ./rag-server.py --index index.jsonl
```

**API Endpoints:**

#### `POST /v1/chat/completions`

OpenAI-compatible chat with optional RAG:

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [
      {"role": "user", "content": "Fix the login bug"}
    ],
    "enable_rag": true,
    "top_k": 3
  }'
```

**Parameters:**
- `enable_rag` (bool): Enable context retrieval (default: true)
- `top_k` (int): Number of context files (default: 3)

#### `GET /health`

Health check:

```bash
curl http://localhost:8001/health
```

**Response:**
```json
{
  "status": "healthy",
  "mlxk_backend": {
    "url": "http://localhost:8000",
    "status": "healthy"
  },
  "pipeline": "/path/to/rag-pipeline.sh",
  "index": "project-index.jsonl"
}
```

---

## Use Cases

### CLI Workflows

**Simple RAG query:**
```bash
./rag-pipeline.sh "How does OAuth work?" docs-index.jsonl 5
```

**Pipeline composition:**
```bash
# Query → Context → LLM
./rag-pipeline.sh "$QUERY" index.jsonl 3 \
  | mlxk run qwen3 - "Answer based on the code above"
```

**Custom workflows:**
```bash
# Vision → RAG → Summary
mlxk run vision-model --image screenshot.png "Extract text" \
  | mlxk embed bge-small-en-v1.5-4bit - \
  | ./cosine-search.py docs-index.jsonl - --output-json \
  | ./retrieve-files.py --format markdown
```

---

### Web Client (nChat) Integration

Many chat UIs (e.g. nChat) are pinned to a fixed port (often **8000**) and cannot be
re-pointed. Because the RAG server is transparent you don't touch the client — run
`rag-server.py` on that port and move `mlxk serve` aside with `MLXK_PORT` (Quick Start §4).

**Architecture:**
```
nChat (Web, :8000) → rag-server.py (:8000) → mlxk serve (:8001)
                    ↓
              rag-pipeline.sh
```

**nChat Configuration:**

```javascript
// nChat keeps its usual endpoint — no change needed (rag-server runs on this port).
const API_URL = 'http://localhost:8000'

// Standard OpenAI API calls work transparently
fetch(API_URL + '/v1/chat/completions', {
  method: 'POST',
  body: JSON.stringify({
    // RAG happens server-side (enable_rag defaults on) — no client changes.
    model: '<a-chat-model-in-your-cache>',
    messages: [{role: 'user', content: 'Fix the login bug'}]
  })
})
```

**Benefits:**
- Token savings: 50k → 8k (83% reduction)
- Faster responses (less tokens to process)
- Focused context (only relevant files)

---

### One base URL: embeddings + chat via `mlxk serve --embed-backend` (no middleware)

Instead of the `rag-server.py` middleware, a client can point an OpenAI SDK at a
**single** `mlxk serve` for *both* embeddings and chat — the "one base URL"
contract (ADR-015 §4). Retrieval stays client-side; no extra server process.

```bash
# Terminal 1 — embedding backend (alpha-gated, like all embed surfaces in 2.0.7)
MLXK2_ENABLE_ALPHA_FEATURES=1 mlxk embed-serve bge-small-en-v1.5-4bit --port 8002

# Terminal 2 — one serve exposes /v1/embeddings (proxied) AND /v1/chat/completions (local)
MLXK2_ENABLE_ALPHA_FEATURES=1 mlxk serve --embed-backend http://127.0.0.1:8002 --port 8000
```

```python
from openai import OpenAI
import numpy as np

client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")  # one base URL for both
query = "how does auth work?"
docs = ["...doc A...", "...doc B...", "...doc C..."]

# 1) embed query + documents (same client)
qv = client.embeddings.create(model="bge-small-en-v1.5-4bit", input=query).data[0].embedding
dvs = [d.embedding for d in client.embeddings.create(model="bge-small-en-v1.5-4bit", input=docs).data]

# 2) retrieve top-k client-side (embeddings are L2-normalized → cosine == dot product)
top = sorted(range(len(docs)), key=lambda i: float(np.dot(qv, dvs[i])), reverse=True)[:3]
context = "\n\n".join(docs[i] for i in top)

# 3) chat with the retrieved context (same client, same base URL)
answer = client.chat.completions.create(
    model="<a-chat-model-in-your-cache>",
    messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
)
print(answer.choices[0].message.content)
```

`mlxk serve` lazy-loads the chat model named in the request; the OpenAI SDK sends
`encoding_format` for you and returns float vectors in `.data[i].embedding`.

A consumer that does its **own** retrieval (like the snippet above) needs only this
one base URL — RAG lives in the client. A plain chat UI that doesn't retrieve keeps
the `rag-server.py` hop instead, where RAG is **transparent**: the client makes a
normal OpenAI call (`model` + `messages`, no RAG parameters) and the server retrieves
and injects the context underneath — the client never sees that RAG happened.

---

## Advanced Examples

### Ensemble Embeddings

Create custom pipeline with multiple embedding models:

```bash
#!/bin/bash
# rag-pipeline-ensemble.sh

QUERY="$1"
INDEX="$2"

# Parallel embeddings
echo "$QUERY" | mlxk-tee \
  "mlxk embed nomic - > /tmp/embed1.json" \
  "mlxk embed bge - > /tmp/embed2.json"

# Average embeddings
python3 average-embeddings.py /tmp/embed1.json /tmp/embed2.json \
  | ./cosine-search.py "$INDEX" - --output-json \
  | ./retrieve-files.py
```

### Custom Search Logic

```bash
# Combine text search + vector search
grep -r "authentication" src/ \
  | combine-with-rag.py \
  | mlxk run qwen3 - "Explain auth implementation"
```

---

## Requirements

**Python Dependencies:**
```bash
pip install numpy fastapi uvicorn httpx pydantic
```

**MLX Knife:**
- `mlxk embed` (experimental in 2.0.7 — set `MLXK2_ENABLE_ALPHA_FEATURES=1`)
- `mlxk serve` for HTTP server

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Standalone Tools (Reusable)                                 │
├─────────────────────────────────────────────────────────────┤
│ index-files.py      │ Create JSONL index from files         │
│ cosine-search.py    │ Vector similarity search              │
│ retrieve-files.py   │ Load file contents                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Pipeline (Shell)                                            │
├─────────────────────────────────────────────────────────────┤
│ rag-pipeline.sh     │ Query → Embed → Search → Retrieve     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ HTTP Server (Optional)                                      │
├─────────────────────────────────────────────────────────────┤
│ rag-server.py       │ OpenAI API + RAG (Port 8001)          │
└─────────────────────────────────────────────────────────────┘
```

**Design Principles:**
- ✅ Each tool is standalone and testable
- ✅ Pipes for composition
- ✅ No hidden logic (shell scripts are visible)
- ✅ Minimal dependencies
- ✅ Reusable components

---

## Troubleshooting

### "embed is experimental and requires MLXK2_ENABLE_ALPHA_FEATURES=1"

`mlxk embed` ships **experimental** in 2.0.7 behind an alpha gate. Enable it:
```bash
export MLXK2_ENABLE_ALPHA_FEATURES=1
```
The scripts set this for their own `mlxk embed` subprocess calls, but set it in
your shell too if you call `mlxk embed` directly.

### Pipeline fails silently

Check individual components:

```bash
# Test embedding
echo "test" | mlxk embed bge-small-en-v1.5-4bit -

# Test search
echo '{"embedding": [...]}' | ./cosine-search.py index.jsonl -

# Test retrieval
echo '{"results": [...]}' | ./retrieve-files.py
```

### Server returns 503

mlxk serve is not running or unreachable:

```bash
# Check mlxk serve
curl http://localhost:8000/health

# Start mlxk serve
mlxk serve --port 8000
```

---

## Future Enhancements

- Hybrid search (keyword + vector)
- Incremental index updates
- Multi-index support
- Caching layer
- Streaming responses

---

## License

Same as MLX Knife (Apache 2.0)

---

## See Also

- [Multi-Modal Support](../../README.md#multi-modal-support) (Vision + metadata output)
- [Unix Pipe Integration](../../README.md#unix-pipe-integration-beta-204) (stdin/stdout chaining)
