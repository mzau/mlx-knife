# ADR-016: Memory-Aware Model Loading

**Status:** Accepted (Phase 1-2 Complete)
**Created:** 2025-12-05
**Context:** Vision models crash with Metal OOM without warning

## Problem

Vision models (e.g., Llama-3.2-90B-Vision-4bit) cause hard Metal crashes when they exceed available unified memory. The crash happens deep in the Metal layer with no graceful error handling:

```
[METAL] Command buffer execution failed: Insufficient Memory
Abort trap: 6
```

Additionally, leaked semaphore warnings appear due to ungraceful shutdown:
```
UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
```

### Key Observation

Vision models have **additional overhead** beyond their weight size:
- Vision Encoder (ViT): ~2-5 GB
- Projection Layer: ~0.5-1 GB
- Image Tensors
- Larger KV-Cache (image tokens)

A 49.9GB Vision model on a 68.7GB system (73%) crashed, while similar-sized text-only models run with just a warning.

## Decision

### JSON-API 0.1.6

Add `system.memory_total_bytes` to API responses:

```json
{
  "system": {
    "memory_total_bytes": 68719476736
  },
  "models": [...]
}
```

This is a **hardware fact** (from `sysctl -n hw.memsize`), not a heuristic.

### Memory Thresholds (mlx-knife internal)

| Model Type | >70% of total | Action |
|------------|---------------|--------|
| Vision     | **ERROR + abort** | CLI: stderr error, exit 1<br>Server: HTTP 507 Insufficient Storage |
| Text-only  | **No user-facing action** | Server: `logger.warning()` only (internal) |

**Rationale:**

**Vision (ERROR + abort):**
- **Empirically confirmed:** Llama-3.2-90B-Vision-4bit @ 73% → Metal OOM crash + semaphore leak
- Vision Encoder cannot swap (Metal GPU operations)
- Additional overhead (~2-5GB) beyond weight size
- Hard limit necessary to prevent system crash
- **User-facing error justified:** Prevents crash

**Text (no user-facing action):**
- **Empirically confirmed:** Qwen2.5-Coder-32B-bf16 @ 95-97% → swap (10-20 tok/6h), no crash
- Text inference swaps gracefully (extremely slow but stable)
- No crash risk → no crash prevention needed
- **CLI:** No warning (backwards compatible, user notices slowdown anyway)
- **Server:** Internal log only (operator awareness, no client-facing change)

**70% threshold chosen because:**
- mlx-lm's own WARNING triggers at ~75% ("49152 MB maximum recommended" on 64GB)
- Provides safety margin for OS + KV-Cache + activations
- Vision models crash above this threshold, text models do not

### Implementation Location

- `run.py`: Pre-load check before `VisionRunner` or `MLXRunner` instantiation
- Uses `size_bytes` from `build_model_object()` (already available)
- Compares against `memory_total_bytes * 0.70`

**Messages:**

**CLI:**
- Vision >70%: `[ERROR] Model size (XX.X GB) exceeds 70% of system memory (YY.Y GB). Vision models crash with Metal OOM due to Vision Encoder overhead. Aborting.` → stderr, exit 1
- Text >70%: No user-facing message (backwards compatible)

**Server:**
- Vision >70%: HTTP 507 Insufficient Storage + JSON error response
- Text >70%: `logger.warning("Model size XX.X GB exceeds 70% of YY.Y GB system memory. Expect extreme slowness due to swapping.")` → visible via `--log-level warning` (default) and `--log-json` if enabled

## Status

**Phase 1+2:** ✅ Complete (2.0.4-beta.1) - See CHANGELOG.md

**Phase 3 (Future):** Issue #46
- [ ] Configurable threshold (env var or CLI flag)
- [ ] Vision overhead estimation based on model architecture
- [ ] KV-Cache size estimation based on context length

## Empirical Data

| Model | Size | System | % Used | Result |
|-------|------|--------|--------|--------|
| **Vision Models** |
| Llama-3.2-90B-Vision-4bit | 49.9GB | 68.7GB | 73% | **CRASH** (Metal OOM + semaphore leak) |
| Llama-3.2-11B-Vision-4bit | 6.0GB | 68.7GB | 9% | OK |
| pixtral-12b-8bit | 13.5GB | 68.7GB | 20% | OK |
| **Text Models** |
| Qwen2.5-Coder-32B-Instruct-bf16 | 61-62GB | 64GB | 95-97% | **Swap, no crash** (10-20 tokens in 6h, Ctrl+C) |

**Key Finding:** Text models swap gracefully (extremely slow but no crash), while Vision models hard-crash at ~73% due to additional Vision Encoder overhead.

## References

- [mlx-vlm Issue #100: High Memory Usage](https://github.com/Blaizzy/mlx-vlm/issues/100)
- [LLaMA 3.2 90B VRAM Requirements](https://blogs.novita.ai/llama-3-2-90b-vram/)
- mlx-lm WARNING message: "maximum recommended size of 49152 MB"
