# ADR-018: Convert Operation

**Status:** Implemented (Phases 0a-0c + 1 complete in 2.0.4-beta.6)
**Created:** 2025-12-18
**Updated:** 2026-02-01 (Added: Known Model Defects & Repair Strategies survey)
**Context:** Users need to (a) quantize MLX workspaces locally without polluting the HF cache and (b) repair MLX/HF compliance issues (notably safetensors index/shard mismatches) in a deterministic way.

**Phase Status:**
- **Phase 0a:** Workspace infrastructure — ✅ Implemented (2.0.4-beta.5)
- **Phase 0b:** Resumable clone — ✅ Implemented (2.0.4-beta.6)
- **Phase 0c:** Workspace run/show/server support — ✅ Implemented (2.0.4-beta.6)
- **Phase 1:** `--repair-index` — ✅ Implemented (2.0.4-beta.5)

**Feature Gates (2.0.4-beta.7+):**
- `clone`, `push`: **Production** (no gate required)
- `convert`: **Experimental** (requires `MLXK2_ENABLE_ALPHA_FEATURES=1`)
  - Rationale: `--quantize` not yet implemented, only `--repair-index` available

**Note:** Complete workspace infrastructure shipped in 2.0.4-beta.6. Full `clone → convert → run/show/server` workflow with resume support, no HF push requirement.

---

## Problem

1. Pre-quantized models (notably VLM) may ship with a broken `model.safetensors.index.json` (index ↔ shard mismatch), e.g. due to the mlx-vlm overwrite regression (issue #624, fixed by PR #638; stable release pending).
2. Users want custom quantization (specific bit depths, mixed recipes).
3. Quantizing “in cache” pollutes operational runtime models used by `mlxk serve`.
4. Strict validators (mlx-knife health) block models that are “lenient-loader runnable” but formally inconsistent.

---

## Decision

### Convert is a Workspace-to-Workspace Transform

`mlxk convert` operates on **existing MLX workspaces** (or 3rd-party model directories treated as *unmanaged workspaces*) and writes results to a **target workspace**.

**Core modes (one per invocation):**
- `--quantize <bits>`: produce new quantized weights in target workspace and **always** write a correct `model.safetensors.index.json` for the output.
- `--repair-index`: **fix packaging only** (rebuild `model.safetensors.index.json` from existing shards) without changing weights / quantization.
- `--dequantize`: produce dequantized weights (future / optional).

### Good citizenship rule

mlx-knife must be able to **detect, explain, and repair** common MLX/HF compliance breakage where possible.
This does **not** imply silently mutating the HF cache; repair happens in a workspace.

### Cache sanctity & CoW constraint (explicit)

- The HF cache is the **production base** for inference and must be treated as **read-only** by `convert`.
- `convert` must **never** write into the HF cache, even if the workspace is on the same volume.
- Workspaces may live on the **same filesystem/volume** as the cache to benefit from CoW (`cp -c`), but they remain a distinct namespace.
- Safety rule: if `<source>` or `<target>` resolves inside the cache root, `convert` must fail with a hard error.

### Managed vs unmanaged workspaces (sentinel contract)

- A **managed workspace** contains a workspace sentinel/metadata file (e.g. `.mlxk_workspace.json`).
- `mlxk clone` and `mlxk convert` MUST produce **managed** workspaces (sentinel always written).
- 3rd-party directories may be used as a **source** without a sentinel (treated as *unmanaged*), but:
  - they are treated read-only,
  - `convert` always produces a managed target workspace.
- `mlxk health <path>` MUST support both managed and unmanaged directories, but reports `managed=false` when the sentinel is missing.

### Atomic target initialization rule

For any `mlxk convert SRC DST ...` operation:
- DST is created (or validated empty) and the **workspace sentinel is written first** (atomic write + rename) **before** any other processing (copy/CoW, repair, quantize).
- If sentinel creation fails, the operation aborts without modifying SRC or cache.

---

## Non-Goals

- Converting arbitrary PyTorch/Transformers checkpoints to MLX format.
- Implementing architecture-specific conversion pipelines beyond what mlx-lm / mlx-vlm provide.
- Auto-modifying downloaded artifacts in-place (cache) without explicit user action.

---

## CLI

```bash
mlxk convert <source-workspace> <target-workspace> [MODE] [options]
```

**Modes (exactly one required):**
- `--quantize <bits>`      Quantize to N bits (2, 3, 4, 6, 8)
- `--repair-index`         Rebuild `model.safetensors.index.json` from existing `.safetensors` shards
- `--dequantize`           Dequantize weights (optional / later)

**Options:**
- `--q-group-size N`       Group size (default: 64)
- `--mixed-recipe R`       Mixed quantization recipe (future phase)
- `--dtype TYPE`           Output dtype (float16, bfloat16) (where applicable)
- `--skip-health`          Skip health check on output (debug/internal)
- *(compat alias, optional)* `--q-bits N` as alias for `--quantize N` (if desired)

**Mode constraints:**
- `--quantize <bits>` and `--repair-index` are **not** combined in one call.  
  Rationale: `--quantize` already guarantees a correct output index by design; `--repair-index` exists for “no quant change, just fix metadata”.

---

## Workflows

### A) Repair broken index without changing quantization

```bash
# Clone a pre-quantized model to workspace (even if unhealthy)
mlxk clone mlx-community/DeepSeek-OCR-4bit ./ws-ocr

# Repair index/shard metadata only (no weight changes)
mlxk convert ./ws-ocr ./ws-ocr-fixed --repair-index

# Now it should validate and be runnable under strict tooling
mlxk health ./ws-ocr-fixed
```

### B) Quantize from a clean source workspace (bf16/fp16) to multiple outputs

```bash
# Clone bf16 (or fp16) source to workspace
mlxk clone mistralai/Mistral-Small-3.1-24B ./ws-bf16

# Quantize to 4-bit workspace
mlxk convert ./ws-bf16 ./ws-4bit --quantize 4

# Optional: another quantization from the same source
mlxk convert ./ws-bf16 ./ws-6bit --quantize 6

# Upload result (still behind ALPHA gate if desired)
mlxk push ./ws-4bit myorg/Mistral-Small-4bit

# Cleanup source workspace
rm -rf ./ws-bf16
```

### C) Adopt a 3rd-party directory as a managed workspace (without changing weights)

```bash
# Treat 3rd-party dir as unmanaged source; produce a managed target workspace
mlxk convert ./third-party-model ./ws-managed --repair-index

mlxk health ./ws-managed
```

---

## Operation Matrix

| Command    | Source      | Target      | Transform |
|------------|-------------|-------------|----------|
| `pull`     | HF          | cache        | none     |
| `clone`    | HF          | workspace    | none (CoW) |
| `convert`  | workspace   | workspace    | `quantize` / `repair-index` / `dequantize` |
| `push`     | workspace   | HF           | none     |

---

## Design Rationale

### Why workspace → workspace (not cache → workspace)?
- Cache stays clean for operational runtime models (`mlxk serve`).
- Workspace is the explicit “working area” for transformations.
- Aligns with philosophy: clone creates a workspace to “do something” with it.

### Why convert as a single command with modes?
- Keeps command surface small (Unix-friendly).
- Clear semantics: “transform workspace” with explicit mode flags.

### Why `--repair-index` exists separately?
- Many issues are **metadata-only** (index mismatch) and should be fixable without re-quantization.
- Enables strict tooling to work with “lenient-runnable” repos by making them formally consistent.

### Why enforce "sentinel first"?
- Ensures every `convert` result is safely identifiable as a mlxk-managed workspace.
- Supports safe cleanup and stricter gating (e.g. `push`) without relying on conventions.

### Why split Phase 0 into 0a and 0b?

**Rationale:**
- Phase 0a (workspace infrastructure) is **prerequisite** for convert operation
- Phase 0b (resumable clone) is **user convenience**, not blocking
- Splitting allows 2.0.4 to ship community repair tool without waiting for resume feature
- Clean dependency chain: 0a → Phase 1 (repair-index) → 2.0.4 stable
- Phase 0b can mature in 2.0.5-beta with more testing/validation

**Timeline:**
- 2.0.4 stable blocked by mlx-vlm 0.3.10 PyPI release anyway
- Use waiting time to implement 0a + Phase 1 (community value)
- Phase 0b deferred to 2.0.5-beta (larger scope, more testing needed)

---

## Implementation (POC-first)

### Phase 0a: Workspace Infrastructure (2.0.4 stable) ✅

**Foundation for all workspace operations (clone, convert, future push).**

#### 1. Workspace Sentinel Primitives

**File:** `mlxk2/operations/workspace.py` (NEW)

Functions:
- `write_workspace_sentinel(path, metadata)` - Atomic write with rename
- `is_managed_workspace(path) -> bool` - Check for valid sentinel
- `read_workspace_metadata(path) -> dict` - Read sentinel metadata

Sentinel format (`.mlxk_workspace.json`):
```json
{
  "mlxk_version": "2.0.4",
  "created_at": "2025-12-29T10:30:00Z",
  "source_repo": "mlx-community/Llama-3.2-3B",
  "source_revision": "abc123def456",
  "managed": true,
  "operation": "clone"  // or "convert"
}
```

#### 2. Workspace Health Check Extension

**File:** `mlxk2/operations/health.py` (MODIFY)

New function:
- `health_check_workspace(path) -> (bool, str, bool)` - Returns (healthy, reason, is_managed)

Modified function:
- `health_from_cache(model_spec, cache_dir)` - Now detects workspace vs cache structure

Logic:
- If path has `.mlxk_workspace.json` → treat as workspace, read metadata
- Otherwise → treat as cache, use existing logic
- Backward compatible: unmanaged workspaces still work, just flagged as unmanaged

#### 3. Clone Integration

**File:** `mlxk2/operations/clone.py` (MODIFY)

Changes:
- Write workspace sentinel after APFS clone (before declaring success)
- Uses `health_check_workspace()` for final validation (optional enhancement)

Benefits:
- All cloned workspaces are now managed (traceable)
- Foundation for convert, push operations
- Health checks work with workspace paths

---

### Phase 0b: Resumable Clone (Deferred to 2.0.5-beta)

**Not critical for 2.0.4 community repair workflow.**

See PLAN-resumable-pull-clone.md for detailed design.

---

### Phase 0c: Workspace Run/Show/Server Support (2.0.4-beta.6) 🚧

**Goal:** Complete local workflow without HF push requirement.

**Problem:**
Current workflow incomplete for local testing:
```bash
mlxk clone model ./ws
mlxk convert ./ws ./ws-fixed --repair-index
mlxk health ./ws-fixed    # ✅ Works (Session 59)
mlxk run ./ws-fixed "test" # ❌ Fails (needs HF model ID)
mlxk show ./ws-fixed       # ❌ Fails (needs HF model ID)
```

Workaround requires HF push:
```bash
mlxk push ./ws-fixed user/model-fixed  # Phase 2 (not implemented)
mlxk run user/model-fixed "test"       # Now works
```

**Solution:**
Extend `run`, `show`, `server` to accept workspace paths (like `health` already does).

**Design Principles:**
1. **Central implementation** - No operation-specific hacks
2. **Consistent behavior** - All operations use same resolution logic
3. **Zero breaking changes** - HF model IDs still work
4. **OpenAI compatible** - Server remains spec-compliant

**Implementation:**

1. **Core Layer** - `resolve_model_for_operation()` (model_resolution.py):
   ```python
   def resolve_model_for_operation(model_spec: str) -> Tuple[str, Optional[str], Optional[List[str]]]:
       """Resolve model specification to (name, commit_hash, ambiguous_matches).

       New: Workspace path detection (analog to health.py:502-504)
       """
       # NEW: Check if model_spec is workspace path
       workspace_path = Path(model_spec)
       if workspace_path.exists() and (workspace_path / "config.json").exists():
           # Return absolute path as resolved_name, skip cache logic
           return (str(workspace_path.resolve()), None, None)

       # Existing cache resolution logic...
   ```

2. **Runner Layer** - `MLXRunner` + `VisionRunner` (__init__.py, vision_runner.py):
   ```python
   def load_model(self):
       resolved_name, commit_hash, ambiguous = resolve_model_for_operation(self.model_spec)

       # NEW: Path vs cache detection
       if Path(resolved_name).exists():
           # Workspace path - use directly
           model_path = Path(resolved_name)
       else:
           # Cache model - existing logic
           model_cache_dir = cache_root / hf_to_cache_dir(resolved_name)
           # ... existing snapshot resolution ...

       # Load from model_path (works for both)
   ```

3. **Operation Layer** - `show_model_operation()` (show.py):
   ```python
   def show_model_operation(model_pattern: str, ...):
       resolved_name, commit_hash, ambiguous = resolve_model_for_operation(model_pattern)

       # NEW: Direct path handling
       if Path(resolved_name).exists():
           model_path = Path(resolved_name)
           # build_model_object works on any path
           model_obj = build_model_object(resolved_name, model_path.parent, model_path)
       else:
           # Existing cache logic...
   ```

4. **Server Layer** - `/v1/models` endpoint (server_base.py):
   ```python
   @app.get("/v1/models")
   async def list_models():
       model_list = []

       # Existing: Scan cache for models...

       # NEW: Add preloaded workspace if present
       global _preload_model
       if _preload_model and Path(_preload_model).exists():
           model_list.append(ModelInfo(
               id=_preload_model,           # Original path string
               object="model",
               owned_by="workspace",        # ✅ Distinguishable!
               context_length=None
           ))

       # Sort: preloaded first (cache or workspace), then alphabetical
   ```

**Affected Operations:**
- ✅ `mlxk run ./workspace "prompt"` - Direct testing
- ✅ `mlxk show ./workspace` - Metadata inspection
- ✅ `mlxk server --model ./workspace` - Local dev/testing
- ✅ `mlxk health ./workspace` - Already works (Session 59)

**OpenAI API Compatibility:**
- `/v1/models` response includes workspace with `"owned_by": "workspace"`
- Clients use the model ID from `/v1/models` (standard flow)
- Remote clients would see local path - acceptable for local dev servers

**JSON API Schema:**
- ✅ No changes required (`model` field already accepts strings)
- ✅ Backward compatible (cache model IDs still work)

**Files Modified:**
- `mlxk2/core/model_resolution.py` (~20 LOC)
- `mlxk2/core/runner/__init__.py` (~30 LOC)
- `mlxk2/core/vision_runner.py` (~20 LOC)
- `mlxk2/operations/show.py` (~15 LOC)
- `mlxk2/core/server_base.py` (~10 LOC)
- Tests: +10-12 tests (resolve: 3, MLXRunner: 3, show: 2, run E2E: 2, server: 2)

**Effort:** ~1.5-2 sessions (~85 LOC, 10-12 tests)

**Benefits:**
1. Complete local workflow: `clone → convert → run` (no HF push)
2. Faster iteration (test fixes immediately)
3. Privacy (no upload for local testing)
4. Consistent UX (all operations accept paths)

**Limitations:**
- Server `--model ./workspace` only for local development
- Remote clients cannot resolve local file paths
- Production: Push workspace to HF first (`mlxk push` - Phase 2)

**Documentation Note:**
> **Workspace paths:** Supported in `run`, `show`, `server` for local development/testing. For production servers, push workspace to HuggingFace first - remote clients cannot access local file paths.

---

### Phase 1: repair-index (2.0.4 stable) ✅

**Community repair tool for mlx-vlm #624 affected models.**

#### Dependencies
- Requires Phase 0a complete (workspace infrastructure)
- Uses safetensors library for header parsing
- Cache sanctity validation via workspace primitives

#### Community Impact
Affected models can be repaired without reconversion:
- Qwen2.5-VL-7B-Instruct-4bit
- gemma-3-27b-it-4bit
- Mistral-Small-3.1-24B-Instruct-2503-4bit
- DeepSeek-OCR-4bit
- Devstral-Small-2-24B-Instruct-2512-6bit
- (7+ models total, see mlx-vlm issue #624)

Workflow:
```bash
mlxk clone mlx-community/<affected-model> ./ws
mlxk convert ./ws ./ws-fixed --repair-index
mlxk health ./ws-fixed  # Should be healthy
# Optional: push back if maintainer
```

---

### Common prerequisites (Convert Operation)
1. Validate source path exists and contains model files (workspace-like structure).
2. Resolve real paths for source/target and **hard-block cache paths** for both.
3. Ensure target workspace path is new/empty (or explicitly allowed by policy).
4. Create target directory and **write workspace sentinel first** (atomic write + rename).
5. Copy required non-weight assets to target (config/tokenizer/etc.), using CoW where possible (`cp -c`).
6. Run the selected mode.
7. Run health check on output (unless `--skip-health`).

### Primitive: `rebuild_safetensors_index()`
- Scan all `*.safetensors` shard files in the workspace.
- Read safetensors headers to enumerate tensor keys (no full tensor load).
- Build `weight_map: key -> shard_filename`.
- Write `model.safetensors.index.json` deterministically.

### Mode: `--repair-index`
1. (After copy) run `rebuild_safetensors_index()` in target.
2. Run health check on output (unless `--skip-health`).

### Mode: `--quantize <bits>`
1. Load model from source workspace (text: mlx-lm; vision: mlx-vlm later).
2. Quantize into target workspace.
3. Ensure output index is correct by:
   - either using the upstream “save weights” output **and verifying**, or
   - always running `rebuild_safetensors_index()` as the final step (preferred for robustness).
4. Run health check on output (unless `--skip-health`).

**Dependencies:**
- Phase 0/1 (POC): safetensors index rebuild primitive + mlx-lm (text quantize)
- Phase 3 (later): mlx-vlm (vision) once stable release includes the upstream fix

---

## Known Model Defects & Repair Strategies

This section catalogs known defects in mlx-community models and upstream conversion pipelines. Understanding these patterns is essential for:
1. Deciding which `--repair-*` flags to implement
2. Providing actionable error messages to users
3. Contributing upstream fixes

### Defect Categories

#### Category A: Repairable Without Original Model

These defects can be fixed from the MLX model alone, without access to the original HuggingFace model.

| ID | Defect | Affected Models | Detection | Repair | Status |
|----|--------|-----------------|-----------|--------|--------|
| A1 | **Index/Shard Mismatch** | mlx-vlm converted models (7+) | `health` → index mismatch | `--repair-index` | ✅ Phase 1 |
| A2 | **Tokenizer PreTokenizer Regex** | EuroLLM, Mistral (transformers 4.39-4.57.2) | garbled output (Ġ, UTF-8 corruption) | Runtime fix in runner | ✅ Implemented |
| A3 | **weights.npz → safetensors** | Whisper legacy | `health` → .npz detected | `--repair-weights` | ❌ Planned |
| A4 | **eos_token_id=null** | Various | config.json check | `--repair-config` | ❌ Future |
| A5 | **video_processor=null** | Qwen2-VL models | config.json check | `--repair-config` | ❌ Future |
| A6 | **Missing preprocessor_config.json** | mlx-community Whisper models | mlx-audio warning | `convert --add-preprocessor-config` | ❌ Future |

#### Category B: Requires Original Model or Manual Intervention

These defects require access to the original HuggingFace model or manual configuration.

| ID | Defect | Affected Models | Detection | Resolution |
|----|--------|-----------------|-----------|------------|
| B1 | **Missing model_type** | Custom/converted models | config.json check | User must add manually |
| B2 | **Missing tokenizer.json** | Some older models | file existence | Re-convert from original |
| B3 | **chat_template issues** | Various (see upstream) | runtime errors | Manual fix or re-convert |
| B4 | **safetensors missing metadata** | Some converts | header inspection | Re-convert from original |

### Detailed Defect Descriptions

#### A1: Index/Shard Mismatch (mlx-vlm #624)

**Root Cause:** mlx-vlm overwrite regression during quantization, writing same keys to multiple shards.

**Symptoms:**
- `mlxk health` reports "index mismatch"
- Model may load with lenient loaders but fails strict validation

**Affected Models:**
- Qwen2.5-VL-7B-Instruct-4bit
- gemma-3-27b-it-4bit
- Mistral-Small-3.1-24B-Instruct-2503-4bit
- DeepSeek-OCR-4bit
- Devstral-Small-2-24B-Instruct-2512-6bit
- (7+ models total)

**Repair:** `mlxk convert ./ws ./ws-fixed --repair-index`

**Upstream:** Fixed in mlx-vlm PR #638

---

#### A2: Tokenizer PreTokenizer Regex (transformers bug)

**Root Cause:** transformers versions 4.39.0 - 4.57.2 produced broken `tokenizer.json` files with invalid PreTokenizer regex patterns.

**Symptoms:**
- `Ġ` (U+0120) BPE space markers visible in output
- UTF-8 corruption: `ö` → `Ã¶`, `ä` → `Ã¤`
- Words concatenated without spaces

**Affected Models:**
- EuroLLM-22B-Instruct-2512 variants
- DeepHermes-3-Mistral-24B (transformers 4.46.3)
- Mistral-Small-3.2-24B (transformers 4.52.4)
- DeepSeek-R1-Distill-Llama-8B (transformers 4.43.0)

**Repair:** Runtime workaround in `MLXRunner._apply_mistral_regex_fix()` - no file modification needed.

**Upstream References:**
- mlx-lm Issue #49 (Mistral tokenizer)
- transformers issue (version range 4.39-4.57.2)

---

#### A3: Whisper Legacy weights.npz

**Root Cause:** Original mlx-examples whisper convert.py saved weights as `weights.npz` instead of `model.safetensors`.

**Symptoms:**
- `health` reports NPZ format detected
- Works but not compliant with modern MLX conventions

**Affected Models:**
- whisper-large-v3-turbo (early converts)
- Other early Whisper conversions

**Repair (Proposed):** `--repair-weights` to convert npz → safetensors

**Upstream:** Issue #938 - Update whisper/convert.py to save as safetensors

---

#### A6: Missing preprocessor_config.json (Whisper models)

**Root Cause:** mlx-community quantized Whisper models omit `preprocessor_config.json` during conversion, despite it being present in original OpenAI models.

**Symptoms:**
- mlx-audio emits warning: "Could not load WhisperProcessor: Can't load feature extractor..."
- Warning pollutes JSON logs in server mode
- Model works (mlx-audio falls back to tiktoken tokenizer)

**Affected Models:**
- All mlx-community Whisper quantized models (whisper-large-v3-turbo-4bit, etc.)
- Does NOT affect original OpenAI whisper models (they have the file)

**Current Workarounds:**
- Server mode: Warning suppressed via `warnings.filterwarnings()` to keep JSON logs clean
- CLI mode: Warning visible (intentional - users should be aware)

**Repair (Proposed):** `mlxk convert ./ws ./ws-fixed --add-preprocessor-config`
- **Option 1 (Preferred):** Copy from original OpenAI model (e.g., `openai/whisper-large-v3-turbo`)
- **Option 2 (Fallback):** Use standard template (identical across all Whisper variants)

**Why Category A:** Unlike other B-category defects, `preprocessor_config.json` is **identical** across all Whisper models (standard audio parameters: 16kHz sampling, 30s chunks, etc.). No model-specific content, making it safe to use a template if original is unavailable.

**Upstream:** mlx-community should preserve preprocessor_config.json during Whisper conversions

---

### Upstream Issue Survey

#### mlx-lm / mlx-examples Issues

| Issue | Description | Category | Status |
|-------|-------------|----------|--------|
| [#683](https://github.com/ml-explore/mlx-lm/issues/683) | TokenizersBackend class error | B3 | Open |
| [#682](https://github.com/ml-explore/mlx-lm/issues/682) | TokenizersBackend initialization | B3 | Open |
| [#470](https://github.com/ml-explore/mlx-lm/issues/470) | qwen3_next model_type not supported | B1 | Pending |
| [#355](https://github.com/ml-explore/mlx-examples/issues/355) | convert modifies tokenizer_config.json | A2 | Related |
| [#737](https://github.com/ml-explore/mlx-examples/issues/737) | generate doesn't halt at `<\|eot_id\|>` | A4/B3 | Open |
| [#1243](https://github.com/ml-explore/mlx-examples/issues/1243) | chat_template not set | B3 | Open |
| [#1195](https://github.com/ml-explore/mlx-examples/issues/1195) | chat_template issues | B3 | Open |
| [#832](https://github.com/ml-explore/mlx-examples/issues/832) | tokenizer issues | A2/B3 | Related |
| [#938](https://github.com/ml-explore/mlx-examples/issues/938) | whisper saves npz not safetensors | A3 | Open |

#### mlx-vlm Issues

| Issue | Description | Category | Status |
|-------|-------------|----------|--------|
| [#624](https://github.com/Blaizzy/mlx-vlm/issues/624) | Index/shard mismatch | A1 | Fixed PR #638 |
| [#676](https://github.com/Blaizzy/mlx-vlm/issues/676) | MP3 transcription bug | - | Fixed 0.3.10 |

#### mlx Core Issues

| Issue | Description | Category | Status |
|-------|-------------|----------|--------|
| [#743](https://github.com/ml-explore/mlx/issues/743) | safetensors missing metadata | B4 | Open |

### Repair Strategy Matrix

| Defect | Can Detect | Can Repair (No Original) | Repair Method | Priority |
|--------|------------|--------------------------|---------------|----------|
| Index Mismatch | ✅ | ✅ | `--repair-index` | ✅ Done |
| Tokenizer Regex | ⚠️ Runtime only | ✅ | Runtime workaround | ✅ Done |
| weights.npz | ✅ | ✅ | `--repair-weights` | Medium |
| eos_token_id=null | ✅ | ⚠️ Needs heuristics | `--repair-config` | Low |
| video_processor=null | ✅ | ⚠️ Model-specific | `--repair-config` | Low |
| Missing model_type | ✅ | ❌ | User manual | N/A |
| Missing tokenizer.json | ✅ | ❌ | Re-convert | N/A |
| chat_template | ⚠️ Runtime | ⚠️ Complex | Manual | N/A |

### Future `--repair-*` Flags (Proposed)

Based on this survey, future convert modes could include:

```bash
# Phase 1 (Done)
mlxk convert ./ws ./ws-fixed --repair-index

# Phase 2 (Proposed)
mlxk convert ./ws ./ws-fixed --repair-weights    # npz → safetensors
mlxk convert ./ws ./ws-fixed --repair-config     # Fix known config issues

# Combined (Future)
mlxk convert ./ws ./ws-fixed --repair-all        # Apply all safe repairs
```

**Design Principle:** Only implement repairs that are:
1. Deterministic (same input → same output)
2. Safe (no data loss risk)
3. Verifiable (`health` can confirm fix)

---

## Status / Phases

- [x] **Phase 0a (2.0.4-beta.5):** ✅ Workspace infrastructure foundation
  - Workspace sentinel (`.mlxk_workspace.json`) - atomic write, managed/unmanaged detection
  - Health checks support workspace directories
  - Clone produces managed workspaces
  - **Files:** `mlxk2/operations/workspace.py` (NEW), `health.py` (extended), `clone.py` (integrated)
  - **Tests:** 20 new tests, all passing

- [x] **Phase 0b (2.0.4-beta.6):** ✅ Resumable clone
  - Temp cache reuse with user prompt (analog to resumable pull)
  - Conditional cleanup based on workspace health
  - UX parity with pull operation
  - `--force-resume` flag for non-interactive use
  - **Status:** Complete (Sessions 67-70, beta.6)

- [x] **Phase 0c (2.0.4-beta.6):** ✅ Workspace run/show/server support
  - Direct workspace execution: `mlxk run ./workspace "prompt"`
  - Workspace inspection: `mlxk show ./workspace`
  - Local dev server: `mlxk server --model ./workspace`
  - Central implementation in `resolve_model_for_operation()` + runners
  - Server: `/v1/models` shows workspace with `"owned_by": "workspace"`
  - **Files:** `model_resolution.py`, `runner/__init__.py`, `vision_runner.py`, `show.py`, `server_base.py`
  - **Status:** Complete (Sessions 68-69, beta.6)

- [x] **Phase 1 (2.0.4-beta.5):** ✅ `--repair-index` for safetensors index/shard mismatch
  - `rebuild_safetensors_index()` primitive
  - `mlxk convert <src> <dst> --repair-index` command
  - Cache sanctity enforcement (hard block)
  - Fixes mlx-vlm #624 affected models (7+ models)
  - **Files:** `mlxk2/operations/convert.py` (NEW), `cli.py` (convert subparser), `output/human.py` (render_convert)
  - **Tests:** 11 new tests, all passing

- [ ] **Phase 2 (future):** `--quantize <bits>` for text models (mlx-lm)
- [ ] **Phase 3 (future):** Mixed recipes / advanced quant options
- [ ] **Phase 4 (future):** Vision model support (mlx-vlm) once stable and dependency policy allows

---

## References

- mlx-vlm issue #624 (index overwrite regression)
- mlx-vlm PR #638 (fix)
- ADR-007: Clone Implementation (workspace concept)
- mlx-lm Issue #49 (Mistral tokenizer regression)
- mlx-examples Issue #938 (Whisper npz → safetensors)
- transformers versions 4.39.0 - 4.57.2 (tokenizer PreTokenizer bug window)