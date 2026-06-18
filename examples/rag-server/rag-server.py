#!/usr/bin/env python3
"""
RAG Server - OpenAI-compatible API with context retrieval.

This server provides an OpenAI-compatible chat endpoint that uses the
rag-pipeline.sh for automatic context retrieval from indexed files.

Usage:
    ./rag-server.py --index codebase.jsonl --port 8001

Environment:
    MLXK_HOST - mlxk serve hostname (default: localhost)
    MLXK_PORT - mlxk serve port (default: 8000)

Example:
    # Start mlxk serve
    mlxk serve --port 8000 &

    # Start RAG server
    ./rag-server.py --index my-index.jsonl --port 8001

    # Query
    curl http://localhost:8001/v1/chat/completions -d '{
      "model": "qwen3",
      "messages": [{"role": "user", "content": "How does auth work?"}],
      "enable_rag": true,
      "top_k": 3
    }'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
import httpx
from typing import Optional

app = FastAPI(
    title="RAG Server",
    description="OpenAI-compatible API with RAG support",
    version="1.0.0"
)

# Configuration
INDEX_FILE = None
PIPELINE = os.path.join(os.path.dirname(__file__), "rag-pipeline.sh")
MLXK_HOST = os.getenv('MLXK_HOST', 'localhost')
MLXK_PORT = os.getenv('MLXK_PORT', '8000')
MLXK_URL = f"http://{MLXK_HOST}:{MLXK_PORT}"

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    top_k: Optional[int] = 3
    enable_rag: Optional[bool] = True
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    OpenAI-compatible chat endpoint with optional RAG.

    Parameters:
        - enable_rag: Enable context retrieval (default: true)
        - top_k: Number of context files to retrieve (default: 3)
    """
    query = request.messages[-1].content

    # Run RAG pipeline if enabled
    context = ""
    if request.enable_rag and INDEX_FILE:
        try:
            result = subprocess.run(
                [PIPELINE, query, INDEX_FILE, str(request.top_k)],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                context = result.stdout
            else:
                # Log error but continue without context
                print(f"RAG pipeline failed: {result.stderr}", flush=True)
        except subprocess.TimeoutExpired:
            print("RAG pipeline timeout", flush=True)
        except Exception as e:
            print(f"RAG pipeline error: {e}", flush=True)

    # Build augmented messages
    messages = request.messages
    if context:
        messages = messages[:-1] + [
            {"role": "system", "content": f"# Relevant Context\n\n{context}"},
            {"role": "user", "content": query}
        ]

    # Forward to mlxk serve
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MLXK_URL}/v1/chat/completions",
                json={
                    "model": request.model,
                    "messages": [{"role": m.role, "content": m.content} for m in messages],
                    "stream": request.stream
                },
                timeout=60.0
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )

        return response.json()

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to mlxk serve at {MLXK_URL}"
        )

@app.get("/health")
async def health():
    """Health check endpoint."""
    # Check mlxk serve connectivity
    mlxk_status = "unknown"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{MLXK_URL}/health", timeout=5.0)
            mlxk_status = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        mlxk_status = "unreachable"

    return {
        "status": "healthy",
        "mlxk_backend": {
            "url": MLXK_URL,
            "status": mlxk_status
        },
        "pipeline": PIPELINE,
        "index": INDEX_FILE
    }

@app.get("/")
async def root():
    """API information."""
    return {
        "name": "RAG Server",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "health": "/health"
        },
        "config": {
            "mlxk_backend": MLXK_URL,
            "index_file": INDEX_FILE
        }
    }

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(
        description="RAG Server with OpenAI-compatible API"
    )
    parser.add_argument(
        "--index", required=True,
        help="JSONL index file for context retrieval"
    )
    parser.add_argument(
        "--port", type=int, default=8001,
        help="Server port (default: 8001)"
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Server host (default: 127.0.0.1)"
    )
    args = parser.parse_args()

    INDEX_FILE = args.index

    print(f"Starting RAG Server on {args.host}:{args.port}")
    print(f"Backend: mlxk serve at {MLXK_URL}")
    print(f"Pipeline: {PIPELINE}")
    print(f"Index: {INDEX_FILE}")
    print()

    uvicorn.run(app, host=args.host, port=args.port)
