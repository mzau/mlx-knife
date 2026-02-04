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
