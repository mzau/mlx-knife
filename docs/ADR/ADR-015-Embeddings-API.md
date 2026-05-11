# ADR-015 — Embeddings API

- **Status:** Draft (consolidated 2026-04-07; 2.0.7 implementation slot confirmed 2026-05-11)
- **Authors:** mlx-knife maintainers
- **Date:** 2025-11-16
- **Updated:** 2026-05-11 (2.0.7 slot pinned, experimental-gated via `MLXK2_ENABLE_ALPHA_FEATURES=1`, stable-promotion in 2.1); 2026-04-07 (consolidated: workspace-first, memory safety, server architecture)
- **Target:** 2.0.7 experimental (gated), 2.1 stable promotion
- **Related:** Issue #26, ADR-014 (Pipe Integration, fulfilled), ADR-021 (MCP), ADR-022 (Workspace-First)

## Context

Issue #26 requests embeddings support for mlx-knife. Embeddings are the fourth and final stateless inference primitive (after text, vision, audio) — a natural fit for the existing architecture.

**MLX-native implementation:** [mlx-embedding-models](https://github.com/taylorai/mlx_embedding_models) (recommended by Awni Hannun, MLX core developer). GPU-accelerated embeddings with quantization support (4bit models ~150MB).

**Current state (2.0.5):**
- mlx-knife supports text, vision, and audio inference
- Workspace-First paradigm implemented (ADR-022)
- Pipe integration available (ADR-014, since 2.0.4)
- No embedding model support
- RAG workflows require external tools

**Goal:**
Enable local RAG workflows with minimal dependencies. CLI + standalone server, following established mlx-knife patterns.

## Goals

1. **CLI-first:** `mlxk embed` for local workflows
2. **Pipe-compatible:** Full stdin/stdout support (reuses ADR-014 pipe semantics)
3. **GPU-default:** MLX-native acceleration (consistent with all other mlxk primitives)
4. **JSONL output:** Human-readable, jq-queryable, git-versionable
5. **Workspace-aware:** Embedding models work with MLXK_WORKSPACE_HOME and explicit paths
6. **Complete RAG loop:** Ship `examples/cosine-search.py` + Photo Search POC
7. **No framework bloat:** Unix pipes > LangChain/ChromaDB

## Non-Goals

- In-process embedding in main server (see Memory Safety below)
- Vector DB integration (FAISS, ChromaDB) — user choice
- LangChain/LlamaIndex integration — external, composable via pipes
- Production-grade indexing (multi-TB datasets)

## Design

### CLI Command

```bash
# Basic usage
mlxk embed <model> <text>
mlxk embed nomic-embed "machine learning tutorial"

# Workspace path
mlxk embed ./workspace/bge-small "machine learning tutorial"

# Pipe mode (stdin)
echo "text" | mlxk embed nomic-embed -
cat document.txt | mlxk embed bge-small -

# Batch mode (JSONL input → JSONL output)
cat docs.jsonl | mlxk embed nomic-embed - --batch

# CPU override (GPU is default)
mlxk embed nomic-embed "text" --cpu
```

**Arguments:**
- `<model>`: Embedding model name, HuggingFace ID, or workspace path
- `<text>` or `-`: Text to embed (stdin if `-`)
- `--cpu`: Force CPU execution (default: GPU via MLX)
- `--batch`: Process JSONL input (multiple embeddings). Reads `"text"` field from each line, passes through all other fields to output
- `--output <file>`: Write to file (default: stdout)
- `--human`: Human-readable summary instead of JSONL (for inspection, not piping)

**Note on output default:** Unlike other mlxk commands (where human-readable is default and `--json` switches to machine output), `mlxk embed` defaults to JSONL. Rationale: the typical consumer of embedding output is another program (cosine-search, index builder), not a human reading a terminal. This avoids requiring `--json` in every pipe.

### Output Format (JSONL)

**Single embedding:**
```json
{"text": "machine learning tutorial", "embedding": [0.123, -0.456, ...], "metadata": {"model": "nomic-embed", "dimensions": 768}}
```

**Batch mode:**
```jsonl
{"text": "doc1", "embedding": [...], "metadata": {...}}
{"text": "doc2", "embedding": [...], "metadata": {...}}
```

**Why JSONL:** Pipe-friendly (one object per line), jq-queryable, git-diffable, no binary formats needed.

### Model Detection

```python
# Embedding models have different config.json markers (initial list, will be extended)
if model_type in ["bert", "nomic_bert", "xlm-roberta"]:
    return "embedding"
# TODO: BGE, GTE, E5, sentence-transformers — extend during implementation
```

**Model sources (workspace-aware):**
- Workspace: `MLXK_WORKSPACE_HOME/bge-small/` or `./my-embedding-model/`
- HuggingFace: `mlx-community/nomic-embed-text-v1.5-MLX`
- Cache: `~/.cache/huggingface/hub/models--*/`

### GPU Default

Unlike the original ADR draft (which proposed CPU-default), GPU is the default. Rationale:

- `mlx-embedding-models` is MLX-native — CPU-forcing is a hack against the framework
- Consistent with all other mlxk primitives (run, serve all use GPU)
- Embedding computation is 5-50ms — negligible GPU contention risk in CLI mode
- `--cpu` available as explicit override for parallel workloads

## Server Architecture: Memory Safety Constraint

**This is the most critical architectural decision in this ADR.**

### Problem

The main mlx-knife server (`mlxk serve`) uses a singleton ModelManager — one model at a time. Loading an embedding model alongside an LLM/Vision model in the same process undermines the empirically established memory safety margins:

| Component | Memory Budget |
|-----------|---------------|
| Vision Memory Gate | 8 GB free before load (conservative, motivated by Metal OOM crashes) |
| Vision Chunking | `--chunk 1` default (stability over performance) |
| Audio Memory Gate | 4 GB free before load |

These margins were established through real OOM debugging during 2.0.4 development. They are not theoretical.

### Risk Analysis

| RAM | Vision (~7 GB) + Embedding (~300 MB) | Assessment |
|-----|---------------------------------------|------------|
| 64 GB | 7.3 GB / 64 GB = 11% | Safe |
| 32 GB | 7.3 GB / 32 GB = 23% — Vision chunking safety buffer shrinks | Risky |
| 16 GB | Vision alone at limit — any additional model is dangerous | Unsafe |

### Decision: Embedding as Separate Process

The `/v1/embeddings` endpoint runs in its own process, NOT inside the main server:

```bash
# Main server (LLM/Vision/Audio) — unchanged
mlxk serve --port 8080

# Embedding server — separate process, explicit user decision
mlxk embed-serve --port 8082
```

**Rationale:**
- Main server memory gates remain untouched
- Embedding memory footprint visible in Activity Monitor (user control)
- RAM-constrained systems: don't start embed-serve (explicit choice, not implicit degradation)
- Aligns with ADR-021 Option A (standalone services) and MCP architecture
- CLI `mlxk embed` is already a separate process — server follows same pattern

**The main server (`mlxk serve`) does NOT get a `/v1/embeddings` endpoint.**

## Use Cases

### 1. Simple RAG (Codebase Q&A)

```bash
# Index
find . -name "*.py" | while read f; do
    cat "$f" | mlxk embed code-model -
done > codebase.jsonl

# Query
echo "How does stop token detection work?" \
  | mlxk embed code-model - \
  | cosine-search codebase.jsonl - --top-k 5 \
  | mlxk run chat-model - "Explain based on this code"
```

### 2. Photo Search (POC — validates Vision + Embeddings)

```bash
# Step 1: Describe photos (mlxk run already supports multi-image batch)
mlxk run pixtral --image photos/*.jpg "Describe" --json > descriptions.jsonl

# Step 2: Embed descriptions
cat descriptions.jsonl | mlxk embed bge-small - --batch > photo-index.jsonl

# Step 3: Query by natural language
mlxk embed bge-small "photos with bridges" \
  | cosine-search photo-index.jsonl - --top-k 5
```

Note: Vision output already includes EXIF metadata (GPS, date, camera) when available — this enriches the embedding with location and temporal context beyond what the model sees in the image.

**Enhanced mode (`--enrich-geo`):** The POC optionally reverse-geocodes EXIF GPS coordinates via Nominatim/OpenStreetMap (free, no API key) to add human-readable location names to descriptions before embedding. This means "photos from Provence" finds results even when the vision model only described "a stone bridge over a river" — because the geocoded location "Pont du Gard, Remoulins, Gard, Occitanie, France" is part of the embedded text. The geocoding lives entirely in the POC script, not in mlx-knife (no network calls added to the tool).

The exact pipeline (how vision output maps to embedding input) is defined in the POC, not here. Ships as `examples/photo-search/`.

### 3. Vision → Embedding Pipeline

```bash
# Describe image AND index the description (sequential, not parallel)
mlxk run vision-model --image photo.jpg "describe" \
  | tee description.txt \
  | mlxk embed doc-model - >> index.jsonl
```

Note: Vision and embedding run sequentially in this pipeline (separate processes, no Metal contention).

## Reference Tool: cosine-search.py

**Purpose:** Complete the RAG loop without vector DB (~100 LOC)

```python
#!/usr/bin/env python3
"""cosine-search.py - Simple vector search for JSONL embeddings

Usage:
  cosine-search index.jsonl query.json --top-k 5
  echo '{"embedding": [...]}' | cosine-search index.jsonl - --top-k 5
"""
import sys, json
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(index_path, query_embedding, top_k):
    results = []
    with open(index_path) as f:
        for line in f:
            doc = json.loads(line)
            score = cosine_similarity(np.array(query_embedding), np.array(doc['embedding']))
            results.append((score, doc))
    results.sort(reverse=True, key=lambda x: x[0])
    for score, doc in results[:top_k]:
        print(json.dumps({'score': float(score), 'text': doc.get('text', ''), 'metadata': doc.get('metadata', {})}))
```

Ships in `examples/cosine-search.py`. No dependencies beyond numpy.

**Sweet spot:** 1K-100K documents. For larger scales, recommend sqlite-vss or usearch.

## Implementation Plan

### Phase 1: CLI + Standalone Server (Post-2.0.6)

**Prerequisites:**
- 2.0.6 upstream dependency consolidation complete
- ADR-014 pipe semantics: fulfilled (since 2.0.4)

**Deliverables:**

- [ ] **Model detection** — distinguish embeddings from LLMs via config.json
- [ ] **CLI command** (`mlxk embed`) — workspace-aware, pipe-compatible
- [ ] **Standalone server** (`mlxk embed-serve`) — `/v1/embeddings` OpenAI-compatible, separate process
- [ ] **Embedding generation** via mlx-embedding-models (GPU-default)
- [ ] **Reference tool** (`examples/cosine-search.py`)
- [ ] **Photo Search POC** (`examples/photo-search/`)
- [ ] **Tests** — unit, pipe, RAG workflow end-to-end
- [ ] **Documentation** — README section, help text, examples

### Phase 2: Advanced Features (Future)

- [ ] Chunking strategies (configurable)
- [ ] Batch optimization (auto-batching for throughput)
- [ ] Model ensembles (multi-model averaging)
- [ ] Performance benchmarks (document throughput per model)

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Model compatibility | Clear error messages, health check integration, document supported models |
| JSONL file size (768+ dims x many docs) | Document sweet spot (1K-100K), recommend vector DB for larger |
| mlx-embedding-models stability | Evaluate before committing, fallback to direct MLX if needed |
| Memory on small systems (16 GB) | Separate process = explicit user decision, not implicit degradation |

## Alternatives Considered

| Alternative | Rejected Because |
|-------------|------------------|
| Integrate FAISS/ChromaDB | Heavy deps, scope creep, breaks Unix philosophy |
| Embed inside main server | Memory safety risk (Vision gates, see above) |
| Binary embedding format | Not human-readable, poor debugging |
| LangChain integration | Framework lock-in, we provide pipes instead |
| CPU-only default | Against MLX philosophy, slower, `--cpu` available as option |

## Open Questions

1. **Default embedding model:** Recommend `nomic-embed-text-v1.5` (small, good quality)? User downloads explicitly.
2. **mlx-embedding-models status:** Evaluate current version, API stability, model support post-2.0.6.
3. **embed-serve architecture:** Minimal single-endpoint server or reuse full handler/model_manager infrastructure from main server? Probably minimal (one endpoint, one model, no switching logic).

## Success Criteria

- [ ] `mlxk embed model "text"` generates JSONL embedding
- [ ] `echo "text" | mlxk embed model -` works (stdin)
- [ ] `mlxk embed ./workspace/model "text"` works (workspace path)
- [ ] `mlxk embed-serve --port 8082` serves `/v1/embeddings` (separate process)
- [ ] `cosine-search.py` finds similar embeddings (top-k)
- [ ] **Photo Search POC:** Vision describe → embed → search works end-to-end
- [ ] **Full RAG workflow:** embed → cosine-search → mlxk run chain works
- [ ] Main server (`mlxk serve`) memory gates unaffected by embedding workload

---

## Appendix A: RAG Code-Refactoring POC

**Motivation:** Large code refactoring tasks cause exponential slowdown due to O(n^2) attention complexity. Chunking + RAG + focused context solves this.

**Observed:** 6k token input → 25k token output. Performance degrades from ~20 T/s to ~1-3 T/s (KV-cache grows to 31k tokens).

**RAG approach:** Split into 4 semantic chunks (~1.5k tokens each), embed for context retrieval, process each chunk with relevant context (3-4k tokens). Result: constant ~20 T/s.

```
examples/rag-code-refactoring/
├── README.md
├── tools/
│   ├── code-chunk-splitter.py    # Intelligent code splitting (~150 LOC)
│   ├── cosine-search.py          # Shared with examples/cosine-search.py
│   └── refactor-pipeline.sh      # End-to-end orchestrator (uses mlxk embed)
├── input/
│   └── simple_chat.html          # Test input (583 lines)
└── output/                       # Generated modular files
```
