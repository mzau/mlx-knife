# ADR-012 — Vision / Multimodal Support Roadmap

- **Status:** Implemented (Phase 1 complete, 2.0.4-beta.1)
- **Authors:** mlx-knife maintainers
- **Date:** 2024-11-11
- **Updated:** 2025-12-04

## Context

Community feedback (e.g., Reddit comments comparing `mlx_lm.server + llama swap`) highlights the lack of multimodal support in MLX-Knife 2.0. The 2.0 CLI deliberately focused on LLM lifecycle management (cache, health, server, JSON automation). Before publicly committing to a multimodal feature set we need to document what "vision support" actually means for mlx-knife and how it fits with existing abstractions.

## Architecture Principles

This ADR follows the **Core Principles for Backend Selection & Error Handling** defined in `docs/ARCHITECTURE.md`:

- **No silent fallbacks:** Vision models detected without mlx-vlm → explicit error
- **Fail fast, fail clearly:** Missing configs, memory constraints → abort before load
- **Memory gates:** Vision models >70% RAM → block (CLI abort, Server HTTP 507)
- **Explicit error codes:** HTTP 501 (not supported), 507 (memory), 400 (bad input)

See `docs/ARCHITECTURE.md` for full details and rationale.

## Backend Architecture

**Critical clarification (2025-12):**

| Package | Scope | Maintainer |
|---------|-------|------------|
| `mlx-lm` | Text LLMs only | Apple (ml-explore) |
| `mlx-vlm` | Vision Language Models | Blaizzy  |

- **mlx-lm will vision support NOT expected** - it's explicitly for text LLMs only
- **mlx-vlm is the de-facto standard** for Vision in MLX ecosystem (~2k stars, actively maintained)
- `mlx_lm.server` = text only; Vision server requires custom implementation
- Separate packages will remain (Awni contributes to both, keeps them separate)

**Dependency strategy:**
```toml
# pyproject.toml
[project.optional-dependencies]
vision = [
    "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git@v0.3.9"
]
# Note: v0.3.9 (Dec 3 2024) not yet on PyPI, installing from GitHub
# Will switch to PyPI version when available: "mlx-vlm>=0.3.9,<0.4"
```

**Architecture:**
```
mlxk run (text)   → MLXRunner    → mlx-lm
mlxk run --image  → VisionRunner → mlx-vlm (new)

mlxk serve (text)   → mlx_lm.server (existing)
mlxk serve (vision) → mlx_vlm.server (exists! OpenAI-compatible)
```

**mlx_vlm.server endpoints** (already available):
- `GET /models` - list available models
- `POST /chat/completion` - OpenAI-compatible (text + image + audio)
- `POST /responses` - OpenAI-compatible responses endpoint
- `GET /health` - healthcheck
- `POST /unload` - unload model from memory

## Goals

1. Deliver an alpha-level `mlxk run` flow that accepts still images (one-shot prompt or chat) without breaking current UX patterns.
2. Extend JSON API schemas so callers can submit/receive multimodal payloads predictably once the runner core is solid.
3. Keep HF cache semantics identical to text-only models (no bespoke asset folders); use `mlx-vlm` as vision backend.
4. Preserve CLI-first ergonomics—no GUI dependencies; the sample Web client only mirrors server capabilities.
5. **Vision server endpoint is required** (community expectation); may be developed on separate branch.

## Proposed Phases

### Phase 0 — Research & Scoping ✅ Complete
- Inventory MLX models on Hugging Face that expose vision/multimodal configs.
- ~~Validate mlx-lm backend hooks~~ → **Confirmed: mlx-lm has NO vision support**
- **Backend decision: mlx-vlm** (0.3.9+) as vision backend
- Capture UX expectations from comparable tools (Ollama `/api/chat` with `images`, OpenAI `responses` API).
- Supported models via mlx-vlm: LLaVA, Pixtral, Qwen2-VL, Llama-3.2-Vision, Phi-3-Vision, SmolVLM, etc.

### Phase 1 — `mlxk run --image` CLI ✅ Complete (2.0.4-beta.1)

**Phase 1a (Detection):** ✅
- `detect_vision_capability()` in `common.py:207-236`
- `"vision"` capability in `list`/`show` JSON output (JSON API 0.1.5)

**Phase 1b (CLI):** ✅
- `--image <path>` (repeatable) for `mlxk run` one-shot prompts
  - **Multiple images supported:**
    - `--image img1.jpg img2.jpg img3.jpg` (space-separated)
    - `--image vacation/*.jpg` (glob expansion)
    - `--image *.jpg --image *.png` (multiple flags)
  - **Filename mapping:** Output includes position→filename mapping for >1 image (deterministic)
    - Model sees: "Images 1 and 3 show motorboats"
    - Output adds: `Filename mapping: Image 1: vacation1.jpg ...`
  - Use cases: photo search ("Which images show motorboats?"), document comparison, batch OCR
- `VisionRunner` class in `core/vision_runner.py` wrapping mlx-vlm
- Default prompt "Describe the image." when user supplies only `--image`
- Non-streaming (mlx-vlm limitation)

**Tests:**
- 5 unit tests in `tests_2.0/test_run_vision.py` (mocked mlx-vlm):
  - Vision runner routing
  - Default prompt with images
  - Result normalization
  - Filename mapping for multiple images (deterministic)
  - Single image (no mapping footer)
- 1 capability test in `tests_2.0/test_human_output.py`
- **5 live E2E tests** in `tests_2.0/live/test_vision_e2e_live.py` (deterministic queries with real models):
  - Chess position reading (field e6 = black king)
  - OCR text extraction (contract name: John A. Smith)
  - Color recognition (mug color: blue)
  - Chart label reading (Y-axis label: Tokens)
  - Large image support (2.7MB test asset validates 10MB limit)

**Phase 1c (CLI Batch Processing):** Planned for 2.0.5-beta
- **Problem:** Vision models have per-image memory overhead
  - 1 image: ~3-5 GB RAM ✅
  - 24 images: ~181 GB RAM ❌ (Metal OOM crash)
  - Per-image overhead varies by model, resolution, system RAM
- **Solution:** Automatic batch processing in `run.py`
  - **Global numbering:** Assign `Image 1..N` once per `mlxk run` invocation and preserve numbering across internal chunks and OOM-driven chunk-size reduction
  - **Pipe-friendly output:** Prefer a stable, compact per-image index so the result can be piped into a chat-only model for synthesis (see `examples/vision_pipe.sh`)
  - No upper limit - process 1000+ images via batching
  - Default: 5 images per batch, configurable via `MLXK2_VISION_BATCH_SIZE`
  - Automatic chunk-size reduction on OOM
  - Combined output with filename mapping for all images
- **CRITICAL Implementation Detail:**
  - **Model loaded ONCE, reused across all batches**
  - Batching logic in `operations/run.py`, NOT in `VisionRunner.generate()`
  - Model loading: ~10-30 seconds
  - Example: 100 images in 20 batches
    - WITHOUT reuse: 200-600s loading overhead ❌
    - WITH reuse: 10-30s loading (once) ✅
  - Implementation pattern:
    ```python
    with VisionRunner(...) as runner:  # Load once
        for chunk in chunks(images, batch_size):
            results.append(runner.generate(chunk))  # Reuse model
    ```
- **Benefit:** Enables photo collection workflows
  - `mlx run llava "Which show motorboats?" --image photos/*.jpg` (1000 images)
  - `mlx run pixtral "Extract text" --image documents/*.pdf`

**Deferred to future:**
- Chat semantics (image-on-first-turn, follow-up turns)
- Pipe integration: `mlxk run vision_model --image x.jpg "describe" | mlxk run chat_model -`

### Phase 2 — Metadata & Capabilities (Complete ✅)
- ✅ Emit capability metadata (`capabilities: ["text","vision"]`) in `mlxk list --json` and `mlxk show` — **Done in Phase 1**
- ✅ **Vision model detection** via `model_type` + `preprocessor_config.json` — **Done in Phase 1**
- ✅ **Health checks for auxiliary vision assets** (2.0.4-beta.1):
  - Vision models: `preprocessor_config.json` required (validated + JSON parsed)
  - Chat models: If `tokenizer_config.json` exists, `tokenizer.json` must also exist
  - Prevents runtime failures with `local_files_only=True` mode
  - Tests: `tests_2.0/test_health_vision.py` (8 new tests)
- ⏳ Extend `docs/json-api-specification.md` with alpha notes for `inputs.images[]`, the "image-on-first-turn" contract, and explicit output modality indicators.

**Health Check Assumptions (documented 2025-12-06):**

The requirement for `preprocessor_config.json` is based on HuggingFace de-facto conventions, not a formal specification:
- **Source:** HuggingFace `AutoProcessor.from_pretrained()` convention
- **Risk:** May produce false negatives for vision models that store preprocessing info in `config.json` or use alternative mechanisms
- **Philosophy:** Conservative approach - prefer false negatives (working model marked unhealthy) over false positives (broken model marked healthy that crashes at runtime)
- **Rationale:** No formal HuggingFace spec exists; mlx-vlm delegates to `AutoProcessor` which handles file loading internally
- **Future:** Will monitor real-world false negative rate and adjust if necessary

### Phase 3 — Server Integration

**Analysis (2025-12-06):** `mlx_vlm.server` provides Vision endpoints but **lacks base64 image support** (only HTTP URLs), which is **required for OpenAI API compatibility**.

**Selected approach: Option A+ (Thin Wrapper)**

Implement mlx-knife FastAPI server that reuses VisionRunner:
- **Image handling:** Base64 decoding (OpenAI standard) → temp files (VisionRunner reuse)
- **Generation:** `VisionRunner.generate()` (existing, tested)
- **API format:** OpenAI `/chat/completions` with `content: [{type: "text"}, {type: "image_url"}]`
- **Multiple images:** Already supported by VisionRunner (up to 10 images per request)

**Why not delegate to mlx_vlm.server?**
- ❌ No base64 support (OpenAI clients expect `data:image/jpeg;base64,...`)
- ❌ Would require forking/patching mlx_vlm.server
- ✅ VisionRunner already handles image prep correctly

**Implementation plan:**
```python
# New: mlxk2/core/vision_server.py
class VisionHTTPAdapter:
    @staticmethod
    def decode_base64_images(content_array) -> List[Tuple[str, bytes]]

    @staticmethod
    def format_openai_response(text: str) -> dict
```

**UX decisions:**
- `/models` endpoint returns `capabilities: ["vision"]` for vision models
- Clients select from unified model list (no separate vision UI)
- Multiple images per message (standard OpenAI format)
- Model switching in client: vision analysis → text chat workflow

**Image limit handling (Server):**
- No hard limit - fail fast with clear error message
- Metal OOM → HTTP 507 "Too many images for available memory. Try 3-5 images. Per-image overhead varies by model/resolution."
- No batching (each HTTP request is independent; OpenAI API typically sends ≤10-16 images per request)
- Stable image IDs via history-based reconstruction (Session 32): Server scans `messages[]` history and assigns IDs chronologically using content-hash deduplication
- Rationale: Per-image memory overhead varies by model/resolution/system RAM - no universal limit possible

**Deferred:**
- Streaming support (mlx-vlm limitation, return full response)
- Sample Web client updates (Phase 4)

### Phase 4 — Tooling & Docs
- Document workflows in README (new “Vision Models” section) and add regression tests under `tests_2.0/vision/`.
- Provide sample HF models + scripted downloads for contributors.
- Evaluate need for GPU/ANE configuration toggles specific to large pixel encoders.

## Non-Goals

- Shipping a GUI or image annotation tool.
- Supporting arbitrary video/audio inputs in the same release (separate ADR if needed).

## Open Questions

1. ~~Do we gate vision support behind `MLXK2_ENABLE_ALPHA_FEATURES`, or carve out a dedicated `MLXK2_ENABLE_ALPHA_VISION=1` gate so releases can graduate independently?~~ **Resolved (2.0.4-beta.1):** Vision gate removed. Vision support is now a regular beta feature (no gate required).
2. How do we represent mixed-mode outputs (e.g., returning base64 images) in the JSON schema without breaking OpenAI compatibility?
3. Should we expose preprocessing hooks so advanced users can override transforms, or keep the runner opinionated?
4. When (if ever) do we allow mid-chat image updates, or do we enforce "image only on first turn" permanently?

## Implementation Notes

### Python Compatibility
**Vision support requires Python 3.10+** due to mlx-vlm dependency:
- **Python 3.9**: Core mlx-knife supported, vision not available
- **Python 3.10-3.14**: Full support (text + vision)
- **Workaround for Python 3.9 users**: Use Homebrew Python 3.10+
- **Future**: Python 3.9 support tracked (similar to mlx-lm patch)

### Vision Model Detection (Phase 1a prerequisite)
Detection strategy for automatic backend routing (konfigurationsfrei):
- **Primary**: `config.json` → `model_type` in VISION_MODEL_TYPES (`llava`, `llava_next`, `pixtral`, `qwen2_vl`, `phi3_v`, `mllama`, `paligemma`, `idefics`, `smolvlm`)
- **Fallback**: `preprocessor_config.json` exists
- **Exposed via**: `capabilities: ["vision"]` in `mlxk list --json` and `mlxk show --json`
- **Implementation**: Extend `mlxk2/operations/common.py::detect_capabilities()`

### JSON API Schema Changes (v0.1.5 compatible)
Vision support via capabilities extension:
- **New capability**: `"vision"` added to `capabilities` array enum
- **No schema version bump**: Backward-compatible array extension
- **Backend routing**: Internal implementation detail (not exposed in JSON API)

**Example output (mlxk list --json):**
```json
{
  "name": "mlx-community/llava-1.5-7b-hf-4bit-mlx",
  "model_type": "chat",
  "capabilities": ["text-generation", "chat", "vision"],
  "framework": "MLX",
  "health": "healthy",
  "runtime_compatible": true,
  "reason": null
}
```

**Note:** Phase 1a (detection) required BEFORE Phase 1b (`--image` flag) - enables users to query which models support vision.

### Server Routing (Phase 3)
Both `mlxk serve` (mlx-lm) and `mlx_vlm.server` use FastAPI:
- **Option A**: Subprocess delegation (`mlxk serve --vision` → `mlx_vlm.server`)
- **Option B**: Unified router with auto-detection (routes internally based on capabilities)

## Next Steps

1. ~~Complete Phase 0~~ ✅ (mlx-vlm confirmed as backend, mlx_vlm.server available)
2. ~~**Phase 1a**: Vision model detection~~ ✅ (2.0.4-beta.1)
   - `detect_vision_capability()` in `common.py`
   - `"vision"` capability in `list`/`show` JSON output
   - JSON API spec updated (0.1.5 compatible)
3. ~~**Phase 1b**: `mlxk run --image` CLI~~ ✅ (2.0.4-beta.1)
   - VisionRunner class wrapping mlx-vlm
   - `--image <path>` flag (repeatable, gated)
   - Non-streaming implementation
4. ~~**Phase 2**: Health checks for auxiliary assets~~ ✅ (2.0.4-beta.1)
   - Vision models: `preprocessor_config.json` required
   - Chat models: `tokenizer.json` required if `tokenizer_config.json` exists
   - 8 new health check tests
5. ~~**Phase 3**: Server integration~~ ✅ (2.0.4-beta.1, Sessions 23-24)
   - VisionHTTPAdapter with OpenAI-compatible Base64 image support
   - Multimodal history filtering (Vision→Text model switching)
   - SSE graceful degradation (non-streaming mlx-vlm backend)
6. ~~**Phase 4**: Documentation + remove alpha gate~~ ✅ (2.0.4-beta.1, Session 16)

**Completed effort:**
- Phase 0 (backend research): ✅
- Phase 1a (detection): ✅
- Phase 1b (run --image): ✅
- Phase 2 (health checks): ✅
- Phase 3 (server): ✅ (2.0.4-beta.1)
- Phase 4 (docs + gate removal): ✅ (2.0.4-beta.1)

**Remaining effort:**
- Phase 1c (CLI batch processing): Planned for 2.0.5-beta
