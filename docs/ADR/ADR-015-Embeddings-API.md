# ADR-015 — Embeddings API

- **Status:** Proposed — ready for 2.0.7 implementation (2026-06-14; implementation-library + server-topology + code-structure + 2.0.7-scope + verified-encoder-list + config-first detection all decided — see the §Decision sections below). (Consolidated 2026-04-07; 2.0.7 implementation slot confirmed 2026-05-11)
- **Authors:** mlx-knife maintainers
- **Date:** 2025-11-16
- **Updated:** 2026-06-14 (Open Q #1 resolved — verified-encoder list = one vendored BERT file + zero-vendored `mlx-lm` decoder path; config-first `declared ∩ runnable` detection; showcase `Qwen3-Embedding-0.6B-4bit-DWQ`; see §Decision: Verified-Encoder List & Model Detection); 2026-06-13 (implementation-library resolved — direct/vendored MIT, no turnkey lib; vision/multimodal embeddings deferred to BEYOND; CLI `embed`-verb confirmed; **server topology resolved (Open Q #3): separate processes, `serve` proxies `/v1/embeddings`; three-layer runner/op/handler structure with `EmbeddingRunner` as the 4th runner; CLI + server both minimal in 2.0.7**); 2026-05-11 (2.0.7 slot pinned, experimental-gated via `MLXK2_ENABLE_ALPHA_FEATURES=1`, stable-promotion in 2.1); 2026-04-07 (consolidated: workspace-first, memory safety, server architecture)
- **Target:** 2.0.7 experimental (gated), 2.1 stable promotion
- **Related:** Issue #26, ADR-014 (Pipe Integration, fulfilled), ADR-021 (MCP), ADR-022 (Workspace-First)

## Context

Issue #26 requests embeddings support for mlx-knife. Embeddings add a **fourth runner** alongside text/vision/audio (see §Decision: Code Structure) and fit the existing architecture. **Honest caveat on the "fourth primitive" framing:** text/vision/audio are *input modalities* of one **generate** primitive (all output text); embeddings are distinct by **output contract** (a vector, not a token stream) — the odd one out, not a symmetric peer. This categorical difference is the root of the vendored-code and RAG-wedge tensions (see §Philosophy Fit).

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
- **Vision/multimodal embeddings** (CLIP/SigLIP/ColQwen — image↔text shared vector space) — out of 2.0.7 scope — deferred to a later release. 2.0.7 ships **text** embeddings only. (Note: the photo-search use case below embeds the VLM's *text description*, not the image — it needs no vision embedder.)
- **Standing boundary (permanent, not a Phase-N todo):** retrieval, chunking, indexing, reranking, vector storage stay **consumer-side forever**. Embeddings ships the `text→vector` primitive only; the RAG *orchestration* is the user's (or broke's). This is the line Goal 7 ("Unix pipes > LangChain") draws — and the one embeddings will pressure hardest post-ship (see §Philosophy Fit + the Phase 2 wedge-watch).

## Philosophy Fit & Named Tensions

Embeddings fits most of mlx-knife's philosophy cleanly: stateless `text→vector` (no index, no state); pipe-first JSONL CLI; OpenAI-honest `serve` (the proxy keeps the promise complete); memory-safe separate process; broke-terrain boundary held (node primitive, broke routes). It is the natural completion of the inference surface.

It is also the ADR where mlx-knife walks **closest to its own lines**. Three tensions, named so the ADR owns them rather than hiding them:

1. **Vendored encoder vs. ADR-023 "never more than upstream."** The verified list would include encoder models (bge/e5) that no MLX upstream lib loads → mlxk becomes their upstream. Resolved as a **conscious bounded exception** → §Decision: Implementation Library (Philosophy note).
2. **RAG-framework wedge vs. "no framework bloat."** Embeddings + RAG examples pull toward the LangChain-ification Goal 7 refuses. Held by a **standing** Non-Goal (retrieval/chunking/indexing permanently consumer-side) + the Phase 2 wedge-watch.
3. **"Fourth primitive" taxonomy.** Embed is categorically different (output = vector, not text) — the odd one out, not a symmetric peer — and the root of (1) and (2). Named in Context.

Naming these *is* the philosophy: mlx-knife's deepest principle is honesty, and the only real betrayal would be an ADR that hides its own compromises.

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
# Encoders — config-first (these model_types are never causal LMs → always embedders):
#   model_type in {bert, xlm-roberta, modernbert, nomic_bert, gemma3_text}  => DECLARED embedding
#   corroborated by sentence-transformers sidecar (modules.json w/ a Pooling module) when present
# Decoders — name / known-list (a qwen3 embedder is structurally identical to a qwen3 chat model):
#   architecture *ForCausalLM + name "embed"/"embedding" (or curated list)  => DECLARED embedding
# declared ∩ runnable: DECLARE for all the above; RUNNABLE = the verified list
#   (see §Decision: Verified-Encoder List & Model Detection). Non-runnable types surface honestly,
#   e.g. "embedding (not runnable: modernbert encoder not vendored)".
# NB: replaces the old `"embed" in name` heuristic that mislabels bge-small as `base`.
```

**Model sources (workspace-aware):**
- Workspace: `MLXK_WORKSPACE_HOME/bge-small/` or `./my-embedding-model/`
- HuggingFace: `mlx-community/nomic-embed-text-v1.5-MLX`
- Cache: `~/.cache/huggingface/hub/models--*/`

### GPU Default

GPU is the default for the **standalone CLI** path. Rationale:

- MLX-native acceleration — consistent with all other mlxk primitives (run, serve use GPU)
- Embedding computation is 5–50 ms — negligible GPU contention risk in CLI mode
- `--cpu` available as explicit override

**Context-dependent caveat (reconciled with §Decision: Server Topology):** CPU is *not* a hack. In the **co-resident server** case (`embed-serve` alongside a GPU-bound `mlxk serve`), CPU is the *recommended* device — it keeps the single Metal GPU dedicated to latency-critical chat (embedding runs on otherwise-idle CPU cores; on unified memory this trades GPU-contention, not RAM). So: GPU = standalone-CLI default; CPU = sensible co-residency choice, decided at deploy time via `--cpu`.

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

Embedding **computation** runs in its own process; `serve` **proxies** the endpoint to it (revised 2026-06-13 — see §Decision: Server Topology). The embed model is never loaded in serve's address space:

```bash
# Embedding backend — separate process, owns the EmbeddingRunner, localhost-internal
mlxk embed-serve bge-small --port 8002

# Main server — proxies /v1/embeddings to the backend; one OpenAI surface, one port
mlxk serve --embed-backend http://127.0.0.1:8002
```

**Rationale:**
- Main server memory gates remain untouched
- Embedding memory footprint visible in Activity Monitor (user control)
- RAM-constrained systems: don't start embed-serve (explicit choice, not implicit degradation)
- Aligns with ADR-021 Option A (standalone services) and MCP architecture
- CLI `mlxk embed` is already a separate process — server follows same pattern

**Revised 2026-06-13:** the main server (`mlxk serve`) does not *compute* embeddings in-process — but it *does* expose `/v1/embeddings` as a thin **proxy** to the isolated `embed-serve` backend. Memory-safety spirit preserved (no embed model in serve's process); OpenAI-API completeness preserved (endpoint on serve's port). See §Decision: Server Topology.

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

### Phase 1: CLI + embed-serve + serve-proxy (2.0.7) — **living TODO checklist**

> This is the canonical TODO home for embeddings implementation (CLAUDE.md points here, no duplication). Decision-aligned with the §Decision sections below; check items off as they land.

**Prerequisites:** ✅ 2.0.6 dep-consolidation · ✅ ADR-014 pipe semantics (since 2.0.4) · ✅ decisions frozen 2026-06-13/14 (§Decision: Implementation Library / Server Topology / Code Structure / 2.0.7 Scope).

**Deliverables (decision-aligned):**

- [x] **Open Q #1 — initial verified-encoder list** — **DECIDED 2026-06-14** (§Decision: Verified-Encoder List & Model Detection): v1 = one vendored BERT file (`model_type: bert` — bge-small default, multilingual-e5-small free, mxbai-large optional) + zero-vendored `mlx-lm` decoder (`model_type: qwen3` — `Qwen3-Embedding-0.6B-4bit-DWQ` showcase). `xlm-roberta` = ~30-line fast-follow; modernbert / embeddinggemma / nomic_bert / vision deferred.
- [ ] **`core/embedding_runner.py`** — `EmbeddingRunner` (4th runner). Encoder = vendored minimal MIT BERT/XLM-R impl (from mlx-examples, NOTICE attribution) + pooling (CLS/mean) + L2-normalize + model prompt-prefixes (E5); decoder embedders (qwen3-embed) = directly on `mlx-lm`. **No turnkey lib.**
- [ ] **Model detection** — config-first: declare embedding for `model_type ∈ {bert, xlm-roberta, modernbert, nomic_bert, gemma3_text}` (never causal) + sentence-transformers sidecar; decoders via name/known-list; gate *runnable* on the verified list (`declared ∩ runnable`). Replaces the `"embed" in name` heuristic that mislabels bge-small as `base`. See §Decision: Verified-Encoder List & Model Detection.
- [ ] **`runtime_compatible` / ARCHITECTURE.md §1 gate [5]** — once `embed` exists, replace the blanket *"embeddings not supported by mlxk run"* with the verified-list filter (`bert`/`qwen3` → runnable-via-`embed`; `modernbert` etc. → honest *"encoder not vendored"*, not "not supported by run"); add the complementary embed-side pre-execution reject in `operations/embed.py` (mirrors the run-side ADR-024 Class A reject); promote the ARCHITECTURE.md §1 gate-[5] forward-note + §Capability Presentation Scope line from *forthcoming* → shipped behavior. **Must land with the `embed` verb** (else gate [5] lies once `embed` exists). Verified embedder **classes** (`bert`, `qwen3`) recorded in `docs/MODEL-COVERAGE.md` at release.
- [ ] **`operations/embed.py`** — CLI `mlxk embed <model> [text|-] [--batch] [--cpu]`, JSONL-default (forward-compatible subset of ADR-014 Appendix C: vector = structuredContent, inline locator).
- [ ] **`core/server/handlers/embeddings.py` + `mlxk embed-serve <model>`** — single-model backend process owning the runner; `/v1/embeddings` OpenAI-compatible; localhost-internal.
- [ ] **`serve --embed-backend URL`** — thin proxy of `POST /v1/embeddings` to the backend (no embed model in serve's process).
- [ ] **Reference tool** (`examples/cosine-search.py`) + **RAG examples** (`examples/rag-server` HTTP, `examples/photo-rag`).
- [ ] **Tests** — unit, pipe, RAG workflow end-to-end.
- [ ] **Documentation** — README section, help text, examples.

**Deferred → point-release / 2.1** (see §Decision: 2.0.7 Scope): `GET /v1/models` merge in serve · Variant B (serve spawns embed-serve) · CPU-co-residency device guidance · full typed-JSON envelope (ADR-014 Appendix C) · vision/audio embedders (BEYOND §3).

### Phase 2: Advanced Features (Future) — ⚠ wedge-watch

**Named tension:** several of these lean over the "no framework bloat" line (Goal 7 / standing Non-Goal). Before adopting *any*, ask: does it belong *in mlxk* (the `text→vector` primitive) or *consumer-side* (RAG orchestration)? Don't accrete the RAG framework by treating this as a todo-list.

- [ ] Batch optimization (auto-batching for throughput) — *in-primitive*
- [ ] Performance benchmarks (document throughput per model) — *in-primitive*
- [ ] Chunking strategies — *likely consumer-side (wedge) → probably an example, not a feature*
- [ ] Model ensembles (multi-model averaging) — *likely consumer-side (wedge) → reconsider*

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Model compatibility | Clear error messages, health check integration, document supported models |
| JSONL file size (768+ dims x many docs) | Document sweet spot (1K-100K), recommend vector DB for larger |
| mlx-embedding-models stability | ~~Evaluate before committing~~ — **resolved 2026-06-13: no turnkey lib taken (license + maintenance signal); direct/vendored MIT path chosen, see §Decision: Implementation Library** |
| Memory on small systems (16 GB) | Separate process = explicit user decision, not implicit degradation |

## Alternatives Considered

| Alternative | Rejected Because |
|-------------|------------------|
| Integrate FAISS/ChromaDB | Heavy deps, scope creep, breaks Unix philosophy |
| Embed inside main server | Memory safety risk (Vision gates, see above) |
| Binary embedding format | Not human-readable, poor debugging |
| LangChain integration | Framework lock-in, we provide pipes instead |
| CPU-only default | Against MLX philosophy, slower, `--cpu` available as option |

## Decision: Implementation Library (Open Question #2 — Resolved 2026-06-13)

**Decision:** No turnkey third-party embedding library as a runtime dependency. Build on the MIT-licensed MLX infrastructure already in the stack.

- **Encoder embedders** (BGE, E5, MiniLM, nomic = BERT/XLM-RoBERTa) — `mlx-lm` loads only causal/decoder models, so encoder support needs its own model code. **Vendor a minimal MIT encoder implementation** (e.g. `mlx-examples`/BERT) into the tree, attribution in `mlxk2/NOTICE`. mlx-knife owns the code — no leaf dependency that breaks on the next `mlx-lm` bump.
- **Decoder embedders** (Qwen3-Embedding) — ride directly on `mlx-lm` (forward → last-token/mean pool → L2-normalize). Optional, post-encoder.

**Rejected libraries:**

| Library | License | Why rejected |
|---|---|---|
| `taylorai/mlx_embedding_models` | MIT | Encoder-only; no decoder/multimodal; stale; superseded even among MIT options |
| `Blaizzy/mlx-embeddings` (Prince Canuma) | **GPLv3** | License-incompatible with mlx-knife's **Apache-2.0** as a runtime dependency (copyleft / derivative-work via import). Maintainer signal: still v0.1.0 (Mar 2026), GPLv3 outlier vs. his MIT `mlx-vlm`/`mlx-audio` → side-project, not driven |
| `mzbac/mlx.embeddings` | MIT | Covers encoder + Qwen3, but 0.x single-maintainer, last release Jun 2025 — same staleness/breakage risk as any leaf dep; conflicts with the project's dependency-hygiene + sunset-watchlist discipline |

**Rationale:** mlx-knife pins exact upstream versions and actively maintains a sunset-watchlist; a 0.x single-maintainer embedding dependency reintroduces exactly the breakage risk that discipline exists to avoid. The heavy lifting (MLX model loading, workspace resolution) already exists; embedding extraction is bounded glue plus — for encoders — a vendored MIT model implementation. Consistent with the "No framework bloat" non-goal and the zero-external-state principle.

**Philosophy note — bounded exception to ADR-023 (named tension):** vendoring an encoder is the **first model-architecture code mlx-knife carries**. It strains ADR-023's *"never more than upstream / the verified list follows upstream reality, not the other way around"*: the list would include encoder models (bge/e5) that **no MLX upstream lib loads** — mlxk itself becomes the upstream for that capability. This is a **conscious, bounded exception**, justified by: (a) BERT/XLM-R are ancient, stable architectures (no mlx-vlm-style churn); (b) the source is mlx-examples (Apple/MIT — quasi-upstream reference, not a fragile third-party); (c) it is the only permissive path that serves the example models; (d) ADR-023's *spirit* — honesty, don't overpromise — is preserved (a vendored, verified encoder delivers exactly what it claims). The dependency-hygiene argument that rejected the turnkey libs cuts *toward* this: owning stable code beats a fragile 0.x leaf dep. **Escape hatch:** if an MLX-team library ships encoder embeddings, the vendored code retires and the list returns to following upstream. **Purer alternative considered, rejected for 2.0.7:** decoder-only via mlx-lm (qwen3-embed, zero vendored code) — rejected because it does not serve the encoder example models (bge/e5), which are the point.

**CLI surface (confirmed):** `mlxk embed` stays a **dedicated verb**, not folded into `run`. The output contracts differ fundamentally: `run` streams generated text with generation flags (temperature, max-tokens, chat-template); `embed` returns a fixed vector (JSONL-default, none of those flags). `run`'s polymorphism is over *input modality* (image/audio → text); folding embeddings in would make it polymorphic over *output type*, which is surprising. Embedding is additionally surfaced as a model **capability** on the discovery layer (`health` / `show` / `/v1/models`) — orthogonal to the verb.

## Decision: Server Topology (Open Question #3 — Resolved 2026-06-13)

**Decision:** Separate processes, with `serve` as the node's OpenAI gateway proxying to an isolated embed backend.

- **`embed-serve`** is its own process and owns the `EmbeddingRunner` + the `/v1/embeddings` handler. Single model, no swapping (minimal). It is a **localhost-internal backend** — typically bound to `127.0.0.1`, not network-exposed.
- **`serve`** exposes `/v1/embeddings` as a **thin proxy** to the embed backend (configured via `--embed-backend URL`) and **merges** the backend's model into `GET /v1/models`. The embed model is **never** loaded in serve's address space — serve forwards bytes only.
- This **revises the earlier "no `/v1/embeddings` endpoint" stance** while keeping its *reason*: the original ban was about *memory* (no embed model in the main process). The proxy honors that (computation stays in the isolated process) and additionally keeps the **OpenAI-API promise complete** on serve's port — `/v1/embeddings` is a core OpenAI endpoint, and a `serve` that 404s on it is a broken OpenAI surface.

**Operational shape (2.0.7):**
```bash
mlxk embed-serve <model> [--port 8002] [--cpu]                # localhost-internal backend
mlxk serve [--model …] --embed-backend http://127.0.0.1:8002  # node gateway; proxies /v1/embeddings
```

**`--host` / `--cpu` = mechanism (mlxk), value = broke-config / operator.** mlxk owns the flags + safe defaults (`serve` already defaults `--host 127.0.0.1`; GPU is the device default). The *values* — whether to bind `0.0.0.0`, whether embed runs on CPU — are part of the per-node deployment manifest: **broke's config** in a cluster, the operator directly in standalone. mlxk persists none of it (stateless executor; broke = stateful manifest + routing — node-local execution is mlxk-terrain, inter-node routing is broke-terrain).

**Rationale — `serve : Node :: broke : Cluster`.** A node has **one network face: `serve`** (the node gateway); `embed-serve` is an internal organ. broke unifies the cluster the same way `serve` unifies the node. Standalone mlxk is a functional mirror of broke scoped to one node — same OpenAI surface, only the base URL changes (serve in standalone, broke's gateway in cluster). A RAG consumer points at one base URL for both `/v1/embeddings` and `/v1/chat/completions`; it never talks to `embed-serve` directly.

## Decision: Code Structure (three-layer, 2026-06-13)

Mirrors the existing canonical separation (text/vision/audio):

```
core/embedding_runner.py            → EmbeddingRunner   (engine — the 4th runner after text/vision/audio)
operations/embed.py                 → CLI verb `embed`  (drives the runner in-process, one-shot)
core/server/handlers/embeddings.py  → /v1/embeddings    (HTTP handler — drives the runner; runs inside embed-serve)
```

- `EmbeddingRunner` is the **fourth runner** alongside `MLXRunner` (text), `VisionRunner`, `AudioRunner` — i.e. the "fourth stateless primitive" of this ADR made structurally concrete (a file beside the other three).
- Both surfaces drive the **same** runner: CLI op (`operations/embed.py`) and server handler (`core/server/handlers/embeddings.py`). The handler + runner live in the **`embed-serve`** process; `serve`'s `/v1/embeddings` is a proxy route with **no runner** in its address space.
- **Build clean per the three-layer split** — `run.py` accreted to ~850 LOC by inlining handlers (BEYOND §4 tech-debt); embed is the chance to instantiate the pattern cleanly from the start.
- **Stay open for vision/audio embedders.** The runner / op / handler shapes must **not preclude multimodal embedders** (CLIP/SigLIP vision embeddings, audio embeddings) — those are deferred to a later release, but the structure keeps the door open. Note (ADR-014 Appendix C): a vector has no MCP media type → it rides as typed `structuredContent`, not a media block; keep the artifact shape transport-agnostic so a future vision/audio embedder slots into the same three layers.

## Decision: 2.0.7 Scope (CLI + server, both minimal)

Both surfaces ship in 2.0.7 — they share `embedding_runner.py`, and the CLI is the cheap, identity-defining shell on top (this ADR is CLI-first; the flagship RAG examples are pipe-based). Cutting the CLI saves little; the weight is in the server.

**In 2.0.7:**
- `mlxk embed <model> [text|-] [--batch] [--cpu]` — JSONL output (forward-compatible, see below).
- `mlxk embed-serve <model> [--port 8002] [--cpu]` — single-model backend.
- `mlxk serve … --embed-backend URL` — proxy `POST /v1/embeddings`.

**Deferred → point-release / 2.1:**
- `GET /v1/models` merge in `serve` (embeddings *work* without it; only "list→embed" discovery is incomplete — documented gap).
- Variant B (`serve` spawns/manages `embed-serve` as a child subprocess — one-command UX).
- CPU-co-residency device-placement guidance (docs).
- Full **typed-JSON envelope** (ADR-014 Appendix C).

**Output forward-compatibility:** the `embed` JSONL is a **forward-compatible subset** of the ADR-014 Appendix C typed-JSON envelope — vector as typed structured data (not a media artifact), payload inline (vectors are small) without precluding a future `inline | file-ref | content-addressed handle` locator, metadata field names aligned with the JSON-API (`model`, `dimensions`). The full envelope is deferred to the dedicated JSON-Pipes/exec session; shipping the subset now avoids migration debt while keeping `embed` identical across pipe · HTTP · (future) MCP transports.

## Decision: Verified-Encoder List & Model Detection (Open Question #1 — Resolved 2026-06-14)

**Decision:** v1 ships the smallest list that runs every 2.0.7 example gate — **one vendored BERT encoder + the zero-vendored `mlx-lm` decoder path** — and detects embedders config-first so the *declared* set is honest about the *runnable* subset.

### Empirical basis (mlx-community demand, 2026-06-14)

Download counts summed across quants in the `mlx-community` org, grouped by `model_type` (the axis that decides vendored-code cost):

| Code path | `model_type` | Vendored cost | Representative models (downloads) | Demand |
|---|---|---|---|---|
| **Decoder (causal)** | `qwen3`, `mistral` | **zero** (rides `mlx-lm`) | Qwen3-Embedding-0.6B 11.7k, 8B 2k, 4B 2.7k; e5-mistral-7b 151 | ~18k |
| **Plain BERT** | `bert` | one vendored file | multilingual-e5-small 2.0k, bge-small 886, mxbai-large 414 | ~3.4k |
| **XLM-RoBERTa** | `xlm-roberta` | BERT file + ~30 lines | bge-m3 4.4k, multilingual-e5-large 392 | ~4.8k |
| **ModernBERT** | `modernbert` | heavy separate file | nomic modernbert-embed (not in top-100) | ~0 |
| **EmbeddingGemma** | `gemma3_text` + bidirectional attn | needs bidir handling | embeddinggemma-300m ~3.5k | ~3.5k |
| Vision (→ BEYOND §3) | `qwen3_vl…` | deferred | Qwen3-VL-Embedding-2B ~600 | — |

Two facts drove the cut: (a) the decoder path is both dominant and free — Qwen3-Embedding-0.6B alone is the single most-downloaded embedder and costs no code (`mlx-lm` already loads `qwen3`); (b) `multilingual-e5-small` is itself `model_type: bert`, so the one BERT file is already multilingual-capable — the *only* thing `xlm-roberta` uniquely unlocks is bge-m3.

### v1 verified-runnable list

- **Encoder path — one vendored MIT BERT file** (`model_type: bert`, from mlx-examples): bge-small-en-v1.5 (**default example model**, 20 MB, English), multilingual-e5-small (free — also `bert`), mxbai-embed-large-v1 (optional, larger English). Pooling CLS|mean (bge→CLS, e5/sentence-transformers→mean), L2-normalize default.
- **Decoder path — `mlx-lm`, zero vendored** (`model_type: qwen3`): Qwen3-Embedding-0.6B / 4B / 8B. Last-token pool + L2-normalize + model instruction-prefix. **Quality/multilingual showcase model: `Qwen3-Embedding-0.6B-4bit-DWQ`** (smallest ~400 MB, most-adopted, DWQ recovers 4-bit quality, and — uniquely among the 0.6B variants — its model card explicitly tags `library: mlx-lm`, confirming it loads via the decoder path with no surprise); **`-8bit`** is the production-conservative candidate (near-lossless, no quant-induced retrieval drift — relevant because a vector store is a one-way-door commitment to its model) **but its card does not declare `mlx-lm`, so verify loadability before recommending**; **`-mxfp8` is the wrong runtime** — its model card targets `pip install mlx-embeddings` (the GPLv3 library rejected in §Decision: Implementation Library), not `mlx-lm`, so it does not load via our decoder path (newest format, lowest adoption besides).

This sizes the vendored work to **one BERT file + pooling/normalize/prefix glue + decoder pool-glue on `mlx-lm`** — the minimum that gates 2.0.7. The examples are model-agnostic (one script, `<model>` argument), so a single example demonstrates the encoder↔decoder tradeoff by swapping one argument. (Same-model rule: query and corpus must use the *same* model — same vector space, and conveniently same dimension; the JSONL `metadata.model` stamp lets a consumer detect a mismatch.)

### Deferred — declared-but-not-runnable (honest, additive later)

- **`xlm-roberta`** (bge-m3, multilingual-e5-large) — the ~30-line fast-follow; the production-grade multilingual upgrade. Fold in when an example or a production user pulls bge-m3.
- **`gemma3_text` + bidirectional (embeddinggemma)** — popular (~3.5k) but needs bidirectional-attention handling (neither the plain decoder forward nor BERT runs it correctly); the strongest post-v1 candidate, *not* modernbert.
- **`modernbert`** (nomic modernbert-embed) — heavy code (alternating local/global attention), ~zero mlx-community demand; defer is now evidence-based, not instinct.
- **`nomic_bert`** — no mlx-community presence.
- **Vision embedders** (Qwen3-VL-Embedding) — → BEYOND §3.

### Coupled detection fix (`declared ∩ runnable`)

The current heuristic is `"embed" in name_lower` (`capabilities.py`), which mislabels **bge-small as `base`** (no "embed" substring) and has no robust signal for embedders whose name lacks the token. Replace with:

- **Encoders — config-first:** `model_type ∈ {bert, xlm-roberta, modernbert, nomic_bert, gemma3_text}` ⇒ declared embedding (these are never causal LMs); corroborated by the sentence-transformers sidecar (`modules.json` with a Pooling module) when present.
- **Decoders — name / known-list:** a `qwen3` embedder is structurally identical to a `qwen3` *chat* model (`Qwen3ForCausalLM`, has `chat_template.jinja`, no sentence-transformers sidecars). Only the name ("Embedding"/"embed") or a curated known-list distinguishes it — so the name heuristic stays, *for decoders only*.
- **`declared ∩ runnable` (capability-honesty rule):** detection *declares* "embedding" for all the above `model_type`s; the verified list defines *runnable*. A `modernbert` model therefore surfaces honestly as `embedding (not runnable: modernbert encoder not vendored — use bge/e5 or Qwen3-Embedding)`, never a silent failure. This fixes the bge=`base` mislabel in the same change.

**Pooling note:** mlx-community conversions **strip** `1_Pooling/config.json` (only `modules.json` survives — it does confirm a Normalize step → L2-normalize is the safe default). The CLS-vs-mean pooling mode must therefore be inferred per model (bge→CLS, e5/MiniLM/sentence-transformers→mean), not read from the sidecar.

## Open Questions

1. ~~**Initial verified-encoder list + how much to vendor**~~ **RESOLVED 2026-06-14 → see §Decision: Verified-Encoder List & Model Detection.** v1 = one vendored BERT file (`model_type: bert`) + zero-vendored `mlx-lm` decoder path (`model_type: qwen3`). Default example model bge-small-en-v1.5; quality showcase `Qwen3-Embedding-0.6B-4bit-DWQ` (the only 0.6B variant explicitly `mlx-lm`-tagged). `xlm-roberta` = ~30-line fast-follow; modernbert / embeddinggemma (`gemma3_text`+bidir) / nomic_bert / vision deferred declared-but-not-runnable. Detection becomes config-first (fixes the bge=`base` mislabel).
2. ~~**mlx-embedding-models status:** Evaluate current version, API stability, model support post-2.0.6.~~ **RESOLVED 2026-06-13 → see §Decision: Implementation Library.** No turnkey lib; direct/vendored MIT + `mlx-lm`.
3. ~~**embed-serve architecture:** Minimal single-endpoint server or reuse full handler/model_manager infrastructure from main server?~~ **RESOLVED 2026-06-13 → see §Decision: Server Topology.** Minimal single-model `embed-serve` (own process, owns `EmbeddingRunner`, localhost-internal); `serve` proxies `/v1/embeddings` + merges `/v1/models`. CLI + server both ship minimal in 2.0.7.

## Success Criteria

- [ ] `mlxk embed model "text"` generates JSONL embedding
- [ ] `echo "text" | mlxk embed model -` works (stdin)
- [ ] `mlxk embed ./workspace/model "text"` works (workspace path)
- [ ] `mlxk embed-serve --port 8002` serves `/v1/embeddings` (separate process)
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
