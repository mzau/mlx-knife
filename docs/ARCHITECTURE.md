# MLX Knife Architecture

## Core Principles

This document defines the architectural principles and design patterns for MLX Knife 2.0+.

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

**Gate Priority:**

| Priority | Gate | Checked For |
|----------|------|-------------|
| 1 | Health | All models |
| 2 | Framework | All models |
| 3 | Audio Backend | Audio-capable models |
| 4 | Vision Backend | Vision-capable models (non-audio) |
| 5 | Embeddings | Embedding models |
| 6 | Text/LLM | Text-only models |

**Implementation:** `build_model_object()` in `common.py:599-634`

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

- Example: `MLXK2_ENABLE_PIPES=1` (ADR-014 Phase 1) - prevents unexpected stdin blocking
- Gates are **documented** in ADRs and `--help` output
- Gates are **removed** when features reach stable status

**Rationale:** Gates allow incremental rollout and protect against breaking changes in production workflows.

### 8. Extensibility for Backend Types

The probe/policy architecture supports multiple backend types without major refactoring.

Current backends:
- **Text:** `mlx_lm` (chat, completion)
- **Vision:** `mlx_vlm` (multimodal with images)
- **Audio:** `mlx_audio` (speech-to-text transcription)

Future backends:
- **Embeddings:** Planned (ADR-015)

API:
- `probe_model_capabilities()`: Returns capability dictionary (text, vision, audio, embeddings)
- `select_backend_policy()`: Maps capabilities to backend implementations
- New backends: Add detection logic to probe, add backend class to policy

**Rationale:** Consistent architecture reduces technical debt as new ML capabilities are added.

---

## Implementation

The core probe/policy implementation lives in `mlxk2/core/capabilities.py`:

- `probe_model_capabilities(model_path)` → Capability detection
- `select_backend_policy(capabilities, context)` → Backend selection

See module docstring for detailed API documentation.

---

## References

- **ADR-012:** Vision Support (backend selection for vision models)
- **ADR-014:** Unix Pipe Integration (feature gates)
- **ADR-016:** Memory-Aware Model Loading (pre-load memory checks)
- **ADR-020:** Audio Backend Architecture (speech-to-text transcription)
- **Code:** `mlxk2/core/capabilities.py` (implementation)
- **Original Discussion:** `docs/vision_server_leitplanken.md` (German, historical)

---

## Changelog

- **2026-02-03:** Modality-agnostic update (audio backend added, examples generalized)
- **2025-12-07:** Initial version
