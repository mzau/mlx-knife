# MLX Knife Architecture

**Stand: 2.0.6 (May 2026)**

## Core Principles

This document defines the architectural principles and design patterns for MLX Knife 2.0+.

---

## Workspace Model

mlx-knife 2.0.5 introduced workspaces as the primary store for managed models; 2.0.6 hardens the integrity story with content_hash v2 (ADR-025). A workspace is any directory whose root contains a `.mlxk_workspace.json` sentinel.

### Sentinel: `.mlxk_workspace.json`

Written atomically (`tmp + rename`) by `clone` and `convert`; the operation that wrote it owns its provenance fields. Schema in 2.0.6:

| Field | Type | Purpose |
|---|---|---|
| `mlxk_version` | string | mlx-knife version that produced the sentinel |
| `created_at` | ISO-8601 | UTC timestamp at write time |
| `source_repo` | string | HF repo of origin (e.g. `mlx-community/whisper-large-v3-mlx`) |
| `source_revision` | string | Git revision pinned at clone time |
| `managed` | bool | Always `true` for mlxk-created workspaces |
| `operation` | string | `"clone"` or `"convert"` |
| `content_hash` | `"sha256:<64-hex>"` | Aggregate digest of the workspace recipe (ADR-025 §3) |
| `hash_algorithm` | `"v2"` | Algorithm tag — readers gate behaviour on this (ADR-025 §9) |
| `file_index` | array | Per-file records (`{path, size, mtime_ns, sha}`) — the portable recipe |
| `exclude_patterns` | array | Exclude list **frozen** at write time (transport-invariant verification) |

ADR-022 makes the workspace the recommended unit of work; ADR-025 makes its clean state trustworthy across repairs, transports, and future algorithm bumps.

### Managed vs. External Paths

mlx-knife distinguishes three on-disk forms:

- **Managed workspace.** Directory + sentinel with `managed=true`. Origin known; integrity tracked; `Clean: ✓ / ✗ / —` rendered in `mlxk show`.
- **External workspace path.** Directory that looks like a model snapshot (has `config.json`) but lacks the sentinel. `mlxk` commands run against it; integrity is not tracked.
- **HF cache entry.** A snapshot under `$HF_HOME/hub/models--…/snapshots/<rev>/`. No `Clean` column applies (rendered as `—`).

The HF-cache path stays a first-class store for `pull` and `run`; the workspace path is preferred for repair workflows, multi-variant convert outputs, and any offline / archive use case.

### HF_HOME Bootstrap (`cli.py:109-140`)

When `mlxk run` or `mlxk serve --model` receives a workspace path, mlx-knife reroutes `HF_HOME` to `<workspace>/.hf_cache` **before any other import**. The bootstrap (`_bootstrap_hf_home()`) inspects `sys.argv` and calls `os.environ["HF_HOME"] = ...` before `huggingface_hub` is imported by any transitive dependency (huggingface_hub reads `HF_HOME` once at import time). Workspace isolation takes priority even over a user-set `HF_HOME` — CoW snapshots and archives stay self-contained. Non-workspace runs respect the user's `HF_HOME` or fall back to the HF default.

The `_resolve_workspace_for_bootstrap()` helper accepts explicit paths (`./`, `../`, `/`) and `MLXK_WORKSPACE_HOME` fuzzy matches; it intentionally performs no imports so the resolution stays inside the pre-import window.

### content_hash: v1 vs. v2

| Aspect | v1 (pre-2.0.6) | v2 (ADR-025, 2.0.6) |
|---|---|---|
| Coverage | `config.json` full + safetensors name/size/4 KB + tokenizer name/size | Include-by-default; every file under the workspace except `exclude_patterns` |
| Safetensors | Filename + size + 4 KB tail sample | JSON header bytes only (parses 8-byte LE length prefix, max 100 MiB; tensor data never read) |
| Size cap | n/a | Non-safetensors files > `CATCHALL_FULL_READ_CAP` (1 GiB) fall back to `sha256(name‖0x00‖size)` |
| Recipe storage | None (algorithm lived only in code) | `file_index` + `exclude_patterns` frozen in sentinel — portable across future readers |
| `mlxk ls` hot path | Recompute every time | `stat()` only with mtime self-heal; ~50 ms over 50 workspaces |
| Issue #52 (`convert --repair-index` invisible) | Affected | Fixed |

The hot-path read flow (`is_workspace_clean()` at `workspace.py:734`): if `hash_algorithm != "v2"` → `clean: None` plus the migration hint *"run `mlxk show <name> --recalc-hash` to upgrade"*. Otherwise stat-walk filtered by the sentinel's stored `exclude_patterns` (not current code defaults), with self-heal on `mtime_ns` drift that silently writes back to the sentinel. Symlinks inside the workspace become path fingerprints; outside or broken targets refuse.

### Clean-State Visibility

- `mlxk list` — `Source` column: `ws` (clean), `ws*` (modified), `ws?` (state unknown — typically v1 sentinels awaiting `--recalc-hash`), `cache` (HF cache entry).
- `mlxk show` — workspace block includes `Clean: ✓ / ✗ / —` and surfaces `content_hash` as `sha256:<7-prefix><16-hex>` (23 chars).
- Migration — one-time `mlxk show <name> --recalc-hash` rewrites the sentinel in-place with `hash_algorithm: "v2"`, fresh `file_index`, frozen `exclude_patterns`.

### Related ADRs

- **ADR-022** — workspace-first paradigm; `.hf_cache/` isolation; clone vs. pull positioning.
- **ADR-025** — content_hash v2 algorithm, sentinel migration, the recipe-storage principle that decouples future readers from current code.

---

## Backend Selection & Error Handling Principles

MLX Knife supports multiple ML backend types (text, vision, embeddings, audio). The following principles govern how backends are selected, loaded, and how errors are handled across all execution paths (CLI, server, utilities).

### 1. Unified Pipeline: Resolve → Probe → Policy → Load → Run

All code paths follow this sequence:

1. **Resolve:** Determine model specification (name, path, repo_id)
2. **Probe:** Detect capabilities, runtime requirements, memory constraints
3. **Policy:** Select appropriate backend (mlx_lm, mlx_vlm, etc.) or block execution
4. **Load:** Initialize the selected backend
5. **Run:** Execute inference

**Rationale:** Consistent probing and policy enforcement prevents silent fallbacks and ensures errors are visible at the earliest possible stage.

#### The Probe Concept

**User Perspective (UX):**

"Probe" is the step where mlx-knife reads a model's metadata files to understand:
- What the model can do (text, vision, audio, embeddings)
- Whether it's healthy (files present, formats correct)
- Whether it can run on this system (backend availability, memory)

This happens automatically during `mlxk list`, `mlxk show`, `mlxk run`, and server model loading.

**Implementation:**

A "probe" is a `Path` object pointing to the model's snapshot directory containing:

```
probe/
├── config.json              # Model architecture (model_type, vision_config, audio_config)
├── tokenizer_config.json    # Chat template, tokenizer settings
├── tokenizer.json           # Vocabulary (optional, may be sentencepiece)
├── preprocessor_config.json # Vision/audio processor info (optional)
└── *.safetensors            # Weight files (checked for naming convention)
```

**Key Probe Functions:**

| Function | Location | Purpose |
|----------|----------|---------|
| `detect_framework()` | `common.py` | MLX vs PyTorch vs GGUF |
| `detect_model_type()` | `common.py` | chat, base, audio, embedding |
| `detect_capabilities()` | `common.py` | text-generation, vision, audio, embeddings |
| `detect_vision_capability()` | `common.py` | vision_config, preprocessor_config |
| `detect_audio_capability()` | `common.py` | audio_config, WhisperFeatureExtractor |
| `detect_audio_backend()` | `common.py` | MLX_AUDIO (STT) vs MLX_VLM (multimodal) |
| `is_workspace_clean()` | `operations/workspace.py` | content_hash v2 clean-check (stat-only hot path) |
| `check_runtime_compatibility()` | `health.py` | Backend supports model_type |

**Probe vs Config:**

- `probe`: The `Path` to the snapshot directory (filesystem location)
- `config`: The parsed `dict` from `config.json` (model metadata)

Both are passed to detection functions because some signals come from config fields, others from file presence.

#### Runtime Compatibility Decision Tree

The `runtime_compatible` field in `mlxk list --json` follows this decision tree:

```
runtime_compatible?
│
├─[1] healthy == False?
│     └─→ False (reason from health check)
│
├─[2] framework != "MLX"?
│     └─→ False ("Incompatible framework: {framework}")
│
├─[3] has_audio AND audio_backend != None?
│     │
│     ├─[3a] audio_backend == MLX_AUDIO?
│     │      │
│     │      ├─ mlx-audio not installed?
│     │      │  └─→ False ("mlx-audio not installed")
│     │      │
│     │      ├─ model_type NOT in [whisper*, voxtral]?
│     │      │  └─→ False ("Model type '{x}' not supported by mlx-audio")
│     │      │
│     │      └─ tekken.json exists WITHOUT tokenizer.json?
│     │         └─→ False ("Voxtral tekken.json tokenizer not supported")
│     │
│     └─[3b] audio_backend == MLX_VLM?
│            │
│            ├─ vision_runtime_compatibility(probe) fails?
│            │  └─→ False (vision reason)
│            │
│            └─ check_runtime_compatibility(probe) fails?
│               └─→ False ("Model type '{x}' not supported")
│
├─[4] has_vision?
│     │
│     ├─[4a] vision_runtime_compatibility(probe):
│     │      │
│     │      ├─ Python < 3.10?
│     │      │  └─→ False ("Vision requires Python 3.10+")
│     │      │
│     │      ├─ mlx-vlm not installed?
│     │      │  └─→ False ("mlx-vlm not installed")
│     │      │
│     │      └─ transformers 5.x + temporal_patch_size?
│     │         └─→ False ("Video processor bug in transformers 5.x")
│     │
│     └─[4b] check_runtime_compatibility(probe):
│            │
│            └─ mlx-lm doesn't support model_type?
│               └─→ False ("Model type '{x}' not supported")
│
├─[5] "embeddings" in capabilities?
│     └─→ False ("Embedding models not supported by mlxk run")
│
└─[6] else (text-only models):
      │
      └─ check_runtime_compatibility(probe):
         │
         ├─ Legacy weight format (weights.*.safetensors)?
         │  └─→ False ("Legacy format not supported by mlx-lm")
         │
         └─ mlx-lm doesn't support model_type?
            └─→ False ("Model type '{x}' not supported")

If all gates pass → True (runtime_compatible)
```

> **Note (2.0.6).** This tree governs the `runtime_compatible` field on the listing side (`build_model_object()`). `mlxk run` adds a further pre-execution capability-mismatch reject (ADR-024 Class A: STT-only and embedding-only models invoked text-only) at `run.py:462-486`. The reject fires before the runner is invoked and returns a typed error with a corrective hint; `runtime_compatible` itself stays unchanged for those models because the listing-side gates (above) do not detect the invocation form.

**Gate Priority:**

| Priority | Gate | Checked For |
|----------|------|-------------|
| 1 | Health | All models |
| 2 | Framework | All models |
| 3 | Audio Backend | Audio-capable models |
| 4 | Vision Backend | Vision-capable models (non-audio) |
| 5 | Embeddings | Embedding models |
| 6 | Text/LLM | Text-only models |

**Implementation:** `build_model_object()` in `common.py:582-706`

#### Capability Presentation — `declared ∩ runnable` (decided 2026-06-10; full form 2.1)

**Decision record:** [ADR-024 §Generalization](ADR/ADR-024-Pre-Execution-Capability-Mismatch-Reject.md). **Bug-class catalog:** [RUNTIME-FEATURES.md §5](RUNTIME-FEATURES.md). **Tracker:** [#53](https://github.com/mzau/mlx-knife/issues/53).

The tree above fails the whole model when any one per-modality gate fails (`AND` over declared modalities) — over-rejecting omni models (vision runnable, audio not → reported wholly not-runnable). The decided direction inverts this to a **filter**:

> A modality is **listed** as a capability iff it is both **declared** (`detect_capabilities`) and **runnable** (its per-modality gate passes, subject to the ADR-023 verified list). The gates **filter** the capability set; `runtime_compatible` is a boolean over the filtered set — `healthy AND effective_capabilities ≠ ∅`.

**Two surfaces.** Diagnostic (`mlxk list --all`, `mlxk show`): declared set + effective set + per-modality drop reason. Consumer (`mlxk list` default, `/v1/models`): effective set only, non-runnable models filtered out (the server already filters `healthy AND runtime_compatible`, `core/server/handlers/models.py:64-68`). The client-facing capability *contract* (`/v1/models` capability emission, client consumption) is **`SERVER-HANDBOOK.md`** terrain and currently **out of scope** — emission + workspace scan unbuilt (#58).

**Invariants.** (1) *No silent fallback* (Principle #2): a dropped modality surfaces its gate reason. (2) *Capability is host-effective, not intrinsic*: the **declared** set is retained in `show`/`--json`. (3) *Integrity precedes capability*: gate [1] health runs before the filter → a false-negative integrity verdict drops a *runnable* model from the consumer surface (see below).

**Open prerequisite (integrity).** The auxiliary-asset check requires `preprocessor_config.json` (Principle #3, Fail Fast), yet the probe layout above marks it *optional*, and mlx-vlm ≥ 0.6 builds the vision processor from an embedded `config.json` (`image_token_id`, `vision_config`). An embedded-processor model is then wrongly `unhealthy` and dropped despite running. ADR-012 already flags this requirement as a "de-facto convention, not formal." Accepting an embedded processor config is a prerequisite for the consumer filter.

**Scope.** The audio-forward-spin reject + the integrity reconciliation land surgically in 2.0.7; the full `MLX_LM_TEXT_LOADER_TYPES`-driven filter is 2.1 (ADR-024 Classes C/D).

### 2. No Silent Fallbacks

If a model requires a specific capability but the corresponding backend is unavailable, the system **must fail explicitly**. Do not degrade to a lower-capability mode.

Examples:
- Vision model + images, but `mlx_vlm` unavailable → fail (do not run text-only)
- Audio model, but `mlx_audio` unavailable → fail (do not skip transcription)
- Vision model + text-only, but `mlx_lm` doesn't support model_type → fail (do not attempt mlx-vlm)

Error handling:
- CLI: Print clear error to stderr with actionable guidance (e.g., "Install mlx-vlm: pip install mlx-vlm")
- Server: Return HTTP 501 (Not Implemented) or HTTP 507 (Insufficient Storage) with error details
- JSON API: Include error details in `error.code` and `error.message`

**Rationale:** Silent fallbacks hide configuration issues and lead to confusing user experiences.

### 3. Fail Fast, Fail Clearly

Capability detection and configuration validation errors **must not be caught silently**.

Examples by modality:
- Vision: `preprocessor_config.json` missing → fail
- Vision: Python < 3.10 and `mlx_vlm` required → fail
- Audio: `mlx_audio` not installed → fail
- Audio: Unsupported audio format → fail
- All: Memory pressure > threshold → fail (CLI abort, Server HTTP 507)

**Error Channels:**
- CLI: stderr (human-readable) + exit code
- Server: HTTP status code + JSON error body
- Logs: warn/error level for gate violations

**Rationale:** Early failures prevent resource exhaustion and provide clear debugging signals.

### 4. Memory Gates: Pre-Load Validation

Memory checks occur **after probing, before loading**.

Thresholds by modality:
- Vision models: Memory pressure > 70% → **abort** (CLI) or HTTP 507 (server)
- Audio models: Memory pressure > 70% → **abort** (unpredictable chunk memory)
- Text models: Memory pressure > 70% → **warning only** (backwards compatible)

Memory is checked via `vm_stat` free+speculative pages (macOS). Future: Add Linux support.

**Rationale:** Vision and audio models have unpredictable per-item memory overhead. Pre-load validation prevents OOM crashes.

### 5. Backend Reuse & Lifecycle Management

Backends (e.g., `VisionRunner`) should be loaded **once per process** and reused across multiple operations.

- Vision batching (ADR-012 Phase 1c): Reuse same `VisionRunner` for all image chunks
- Temporary files: Track and clean up on exit
- Context managers: Use `with` statements for resource safety

**Rationale:** Model loading is expensive (~5-10s). Reuse improves performance for batch operations.

### 6. Explicit Error Codes for Servers

Server endpoints return standardized HTTP status codes:

- **501 Not Implemented:** Feature not supported (e.g., vision models on text-only server)
- **507 Insufficient Storage:** Memory constraints violated
- **400 Bad Request:** Invalid input (e.g., missing images for vision model)
- **404 Not Found:** Model not found in cache
- **500 Internal Server Error:** Unexpected backend failures

**Rationale:** Clear HTTP semantics enable better client-side error handling and debugging.

### 7. Feature Gates (Temporary)

New features may be gated behind environment variables during alpha/beta:

- `MLXK2_ENABLE_PIPES=1` (ADR-014 Phase 1) — required for `-` / stdin input on `mlxk run`. Prevents unexpected stdin blocking in non-piped invocations. Active in 2.0.6 (`cli.py:647`).
- `MLXK2_ENABLE_ALPHA_FEATURES=1` — introduced in 2.0.7 for experimental surfaces (Embeddings per ADR-015, MCP per ADR-021). Not yet active in the 2.0.6 codebase; mentioned here for forward visibility per the ADR-015 2026-05-11 update.
- Gates are **documented** in ADRs and `--help` output.
- Gates are **removed** when features reach stable status.

**Rationale:** Gates allow incremental rollout and protect against breaking changes in production workflows.

### 8. Extensibility for Backend Types

The probe/policy architecture supports multiple backend types without major refactoring.

Current backends:
- **Text:** `mlx_lm` (chat, completion)
- **Vision:** `mlx_vlm` (multimodal with images)
- **Audio:** `mlx_audio` (speech-to-text transcription)

Future backends:
- **Embeddings:** Slated for 2.0.7 experimental, gated by `MLXK2_ENABLE_ALPHA_FEATURES=1` (ADR-015); stable promotion in 2.1.

API:
- `probe_model_capabilities()`: Returns capability dictionary (text, vision, audio, embeddings)
- `select_backend_policy()`: Maps capabilities to backend implementations
- New backends: Add detection logic to probe, add backend class to policy

**Rationale:** Consistent architecture reduces technical debt as new ML capabilities are added.

---

## Implementation

The core probe/policy implementation lives in `mlxk2/core/capabilities.py`:

- `probe_model_capabilities(model_path)` → Capability detection (`capabilities.py:429`)
- `select_backend_policy(capabilities, context)` → Backend selection (`capabilities.py:564`)
- `classify_convert_target(config)` → Single-dispatcher for `convert --quantize` (`capabilities.py:164`)

Workspace Model implementation (ADR-022, ADR-025):

| Concern | Symbol | File |
|---|---|---|
| Sentinel I/O | `write_workspace_sentinel`, `read_workspace_metadata` | `operations/workspace.py:118, 195` |
| HF_HOME bootstrap | `_bootstrap_hf_home`, `_resolve_workspace_for_bootstrap` | `cli.py:109-140` |
| content_hash v2 compute | `compute_workspace_hash_v2` | `operations/workspace.py:550` |
| Clean-check hot path | `is_workspace_clean` | `operations/workspace.py:734` |
| Algorithm constants | `HASH_ALGORITHM_V2`, `CATCHALL_FULL_READ_CAP`, `SAFETENSORS_HEADER_MAX`, `DEFAULT_EXCLUDE_PATTERNS` | `operations/workspace.py:43-76` |

Dependency stack (`pyproject.toml:41-55`):

| Package | Pin (2.0.6) | Note |
|---|---|---|
| `mlx` | `>=0.30.0,<0.32` | Apple Silicon ML framework |
| `mlx-lm` | `==0.31.3` | Text backend; Gemma 4 + KV-cache fixes |
| `mlx-audio` | `==0.4.3` | STT backend; Voxtral tokenizer fixes (drives `transformers >=5.5.0` floor) |
| `mlx-vlm` | `==0.4.4` | VLM backend; adds `gemma4` vision convert support |
| `transformers` | `==5.5.4` | Required by `mlx-audio 0.4.3` |
| `torch>=2.0`, `torchvision>=0.15` | Temporary | Pixtral / Llama-Vision / Mistral-Small-3.1; `sunset-by mlx-vlm#1011` |

Upper bounds are hygiene per ADR-023: every upstream minor bump requires an explicit mlx-knife release with re-verified integration. The `torch` / `torchvision` lines carry a sunset marker (ADR-023 Workaround-Sunset Policy) and drop when `mlx-vlm` ships a `use_fast=False` fallback.

See module docstrings for detailed per-symbol API documentation.

---

## ModelManager State Machine

The server's model lifecycle is managed by `ModelManager` in `mlxk2/core/server/model_manager.py`.

### States

| State | Description |
|-------|-------------|
| IDLE | No model loaded (`_current_model_path = None`) |
| LOADED(X) | Model X loaded and cached |
| SWITCHING | Model switch in progress (lock held, cleanup + load) |
| SHUTTING_DOWN | Server shutting down (all requests → 503) |

### Transitions

```
IDLE ──get_or_load(X)──→ LOADED(X) ──get_or_load(Y)──→ SWITCHING ──→ LOADED(Y)
  ↑                           │                              │
  └───────cleanup()───────────┴──────────────────────────────┘

ANY ──shutdown_event.set()──→ SHUTTING_DOWN
```

### Thread Safety

- `_lock` serializes all state changes
- Double-check pattern: Check shutdown before AND after lock acquire
- Cleanup with `list()` copy to avoid dict mutation during iteration

### Memory Gates (see Principle #4)

- Vision/Text: 8 GB minimum before load
- Audio: 4 GB minimum before load
- `wait_for_memory_release()` polls until threshold reached or timeout

### API

```python
# Thin wrappers in server_base.py delegate to ModelManager singleton
get_or_load_model(model_spec, verbose=False) -> MLXRunner | VisionRunner
get_or_load_audio_model(model_spec, verbose=False) -> AudioRunner
```

---

## References

### Architecture Decision Records

- **ADR-012** — Vision Support Roadmap (backend selection for vision models; batching lifecycle behind §5)
- **ADR-014** — Unix Pipe Integration (`MLXK2_ENABLE_PIPES` gate behind §7)
- **ADR-015** — Embeddings API (2.0.7 experimental gated, 2.1 stable; cited by §7 and §8)
- **ADR-016** — Memory-Aware Model Loading (pre-load memory thresholds behind §4)
- **ADR-018** — Convert Operation (workspace sentinel infrastructure; v1 deprecated by ADR-025)
- **ADR-020** — Audio Backend Architecture (STT routing via `detect_audio_backend`; MLX_AUDIO vs MLX_VLM split behind decision-tree gate [3])
- **ADR-022** — Workspace-First Paradigm (workspace as primary store; `.hf_cache/` isolation; sentinel philosophy behind the Workspace Model section)
- **ADR-023** — Text-First + Verified Multimodal (no-silent-degradation policy reinforces §2; Workaround-Sunset Policy governs the `torch` / `torchvision` temporary dep)
- **ADR-024** — Pre-Execution Capability-Mismatch Reject (Class A shipped 2.0.6 — STT/embedding text-only invocation; Class C + D deferred 2.1; behind the §1 decision-tree note)
- **ADR-025** — content_hash v2 (algorithm + portable-recipe storage in sentinel; behind the Workspace Model content_hash subsection)

### Companion Documents

- `docs/RUNTIME-FEATURES.md` — §5 four-bug-class catalog (A shipped, B detection fix shipped, C+D deferred 2.1); shared vocabulary for ADR-024.
- `docs/MODEL-COVERAGE.md` — per-release operation-vs-model_type verification matrix; living document.
- `docs/TESTING-DETAILS.md` — operational test-execution details and env vars (including `MLXK2_LIVE_CHV2=1` for content_hash v2 live tests).
- `docs/SERVER-HANDBOOK.md` — user-facing server documentation.
- `docs/json-api-specification.md` + `docs/json-api-schema.json` — JSON API contract (0.2.2 documents `content_hash` as `sha256:<64-hex>` per ADR-025).

### Code Anchors

- `mlxk2/core/capabilities.py` — probe/policy implementation
- `mlxk2/operations/common.py` — detection helpers + `build_model_object` (line 582)
- `mlxk2/operations/workspace.py` — sentinel + content_hash v2
- `mlxk2/core/server/model_manager.py` — model lifecycle (line 146)
- `mlxk2/cli.py` — HF_HOME bootstrap (line 109)

### Historical

- `docs/vision_server_leitplanken.md` (German, historical pre-2.0 discussion)

---

## Changelog

- **2026-05-12 (2.0.6 sync):** Added Workspace Model section (sentinel schema, managed-vs-external distinction, HF_HOME bootstrap, content_hash v1↔v2, clean-state visibility). Annotated decision-tree with ADR-024 Class A pre-execution reject. Extended probe-function table with `is_workspace_clean()`. Updated `build_model_object()` line reference (599-634 → 582-706). Extended §7 Feature Gates with `MLXK2_ENABLE_ALPHA_FEATURES` (forthcoming 2.0.7). Updated §8 Embeddings status to slated-2.0.7-experimental. Expanded Implementation with workspace pointer map and dep-pin table (mlx-lm 0.31.3, mlx-audio 0.4.3, mlx-vlm 0.4.4, transformers 5.5.4, torch+torchvision temporary). Rebuilt References (ADR-012/014/015/016/018/020/022/023/024/025, RUNTIME-FEATURES, MODEL-COVERAGE, TESTING-DETAILS, SERVER-HANDBOOK, json-api-specification). ModelManager state machine verified unchanged.
- **2026-02-11:** Added ModelManager State Machine documentation (Phase 2 refactoring)
- **2026-02-03:** Modality-agnostic update (audio backend added, examples generalized)
- **2025-12-07:** Initial version
