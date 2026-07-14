# ADR-016: Memory-Aware Model Loading

**Status:** Accepted — Phase 1+2 (2.0.4-beta.1) and Phase 2b (2.0.4-beta.9) implemented. Phase 3 deferred → Issue #46.
**Created:** 2025-12-05
**Updated:** 2026-07-14 — audited against the tree. Three decisions had been taken *in code* and never recorded here (switch-gate thresholds, two-indicator polling, `inactive` exclusion); they are written up below. One item goes the other way: the JSON-API surface is narrower than this ADR intended and is now marked `[TARGET]` for 2.0.8 rather than downgraded to match the code.
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

Add `system.memory_total_bytes` to API responses. This is a **hardware fact** (from `sysctl -n hw.memsize`), not a heuristic.

```json
{
  "system": {
    "memory_total_bytes": 68719476736
  },
  "models": [...]
}
```

**`[TARGET]` — not yet true (2.0.8).** As shipped, the field is carried by the **`version` command only** (`mlxk --version --json`). `list` and `health` do not emit it. That makes the node's self-description incomplete in the way that matters: a consumer building a roster gets *model sizes* from `list --json`, but must make a second call to learn the *node's RAM* — so "does this model fit on this node?" cannot be answered from one response. The intent above (system info alongside the model list) is the right target; carrying it on `list`/`health --json` is an additive JSON-API change and belongs in 2.0.8, not in a release under smoke-test.

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

The check ships on **two independent paths**, which the original text (one check in `run.py`) did not anticipate:

- **CLI** — `operations/run.py::check_memory_before_load()`, called immediately before the vision runner loads. Vision-only by construction: it returns `None` for text models, so text is never blocked.
- **Server** — `core/capabilities.py::probe_and_select()`, reached from the server's `ModelManager`. Memory is two of its policy gates: vision >70 % → `BLOCK` with `http_status=507`; text >70 % → `WARN`, and the load proceeds.

The server path is the better home (the policy layer is where every other pre-execution gate lives), so the code is right and this ADR was describing an earlier, simpler world.

**Messages:**

**CLI:**
- Vision >70%: `Model size (XX.X GB) exceeds 70% of system memory (YY.Y GB). Vision models crash with Metal OOM due to Vision Encoder overhead. Aborting.` → prefixed with `Error: ` by the caller, stderr, abort
- Text >70%: No user-facing message (backwards compatible)

**Server:**
- Vision >70%: HTTP 507 Insufficient Storage + ADR-004 JSON error envelope
- Text >70%: `logger.warning("Model size (XX.X GB) exceeds 70% of YY.Y GB system memory. Expect extreme slowness due to swapping.")` → visible via `--log-level warning` (default) and `--log-json` if enabled

## Status

**Phase 1+2:** ✅ Complete (2.0.4-beta.1) - See CHANGELOG.md

**Phase 2b:** ✅ Complete (2.0.4-beta.9) - Model Switching Memory Gate

**Phase 3 (Future):** Issue #46

### Phase 3a: User-Configurable Limits
- [ ] `MLXK_MAX_IMAGES` env var
- [ ] `--max-images N` CLI flag
- [ ] `MLXK_MEMORY_THRESHOLD` env var (override 70%)

### Phase 3b: Benchmark-Based Heuristics

**Approach:** Empirische Daten statt theoretischer Berechnung.

**Daten sammeln:**
```
Für jedes Vision-Modell:
  - Config: image_size, patch_size, hidden_size, vision_config
  - Test: 1, 2, 4, 8, 16 Images (verschiedene Größen)
  - Messen: Peak Memory (memmon.py), OOM-Punkt
  - Hardware: M1/M2/M3, 16/32/64/96 GB
```

**Korrelation finden:**
```
Peak_Memory ≈ f(image_size², patch_size, num_images, model_size)
```

**Heuristik-Formel ableiten:**
```python
def estimate_vision_memory(config, num_images):
    base = model_size_bytes
    image_size = config.get("vision_config", {}).get("image_size", 384)
    patch_size = config.get("vision_config", {}).get("patch_size", 14)
    hidden_size = config.get("hidden_size", 4096)

    # Empirisch kalibrierte Konstanten aus Benchmarks
    patches_per_image = (image_size / patch_size) ** 2
    per_image_overhead = patches_per_image * hidden_size * BYTES_PER_ACTIVATION

    return base + (per_image_overhead * num_images * SAFETY_FACTOR)
```

**Infrastruktur:** `memmon.py`, `memplot.py` aus beta.9 existieren bereits.

**Komplexität bei Multimodal:**
- Dynamic Tiling (Qwen2-VL, MiMo-VL) → variable patches pro Bild
- Audio + Vision (Gemma-3n) → zusätzlicher Audio Encoder
- Mixed Input → Kombinations-Overhead

**Pragmatischer Fallback:** Wenn Heuristik unsicher → User-Limit (Phase 3a) verwenden.

---

## Phase 2b: Model Switching Memory Gate

**Problem:** Metal GPU cache is released asynchronously. During model switching, the new model may start loading before memory from the old model is actually freed → OOM / "Broken pipe" crashes.

**Root Cause Analysis:**
- `mx.clear_cache()` releases the Metal cache, but **asynchronously**
- macOS needs time to return memory to the system
- Pre-load check (Phase 1-2) validates `model_size / total_memory` → looks OK
- But **available memory** is still occupied by the previous model

**Solution: after unloading, poll until memory is actually back — then load. Fail soft.**

The gate lives in `core/server/model_manager.py` (module-level, stateless: `get_available_memory_bytes()`, `get_memory_pressure()`, `wait_for_memory_release()`), and the whole switch — evict → `mx.clear_cache()` → poll → load — runs under the `ModelManager` lock. The live test harness carries its own copy for teardown between tests.

Three decisions below were taken during implementation (beta.9) and are recorded here for the first time.

**Decision: poll two indicators, not one.** A single "is enough free?" reading produced false positives that still OOM'd — macOS reports pages as reclaimable that Metal is in fact still holding. The gate therefore proceeds only when **both** hold: the kernel reports memory pressure NORMAL (`sysctl vm.memory_pressure == 0`) **and** enough pages are available.

**Decision: count free + speculative, exclude `inactive`.** `inactive` is the page class macOS calls reclaimable but the GPU cache may still own. Excluding it is what made the gate honest.

**Decision: 8 GB / 4 GB, not 20 GB / 10 GB.**

| Context | Min available | Timeout | Constant |
|---------|---------------|---------|----------|
| Text **and** vision model switch | **8 GB** | 10 s | `ModelManager.MIN_FREE_BYTES_VISION` |
| Audio model switch | **4 GB** | 10 s | `ModelManager.MIN_FREE_BYTES_AUDIO` |
| Live-test teardown | **8 GB** | 10 s | `tests_2.0/live/server_context.py` |

The 20 GB / 10 GB pair in the original draft was never shipped: on a working desktop it would have timed out on nearly every switch. 8 GB was calibrated against `wet-memmon` measurements (≈10.5 GB free with a browser running) and is the value that has always been in the code. The threshold is a **constant floor, not sized to the incoming model** — it is a proxy for "has the async release completed", not a fit check; the fit check is Phase 1-2's 70 % gate.

> Naming wart: `MIN_FREE_BYTES_VISION` gates the **text and vision** path alike — `get_or_load_model()` does not branch on modality. Only audio has its own, lower gate.

**Key Difference from Phase 1-2:**

| Aspect | Phase 1-2 (Pre-load) | Phase 2b (Model Switch) |
|--------|---------------------|------------------------|
| Measures | `total_memory` | `available_memory` + memory pressure |
| Timing | Before first load | After unload, before new load |
| Method | Static check | Active polling with timeout |
| Failure | HTTP 507 (hard block) | Warning + continue (soft) |

**Behavior on Timeout:**
- Log warning with actual available memory
- Continue anyway (probe/policy check will catch real OOM)
- Prevents indefinite blocking on edge cases

### Consequence: the gate arbitrates one process, not the machine

State and lever are **per-process** — each server process has its own `ModelManager`, its own lock, its own single-model slot; no process can evict another's model. The signal, however, is **machine-wide**: `vm_stat` and `vm.memory_pressure` both report the whole system.

Two mlx-knife processes therefore mean two independent gates reading one shared meter. Both can observe the same free memory, both can pass, both can load — the reading is a check, not a reservation — and because the timeout path continues anyway, the gate degrades to advisory under co-residency.

This is a property of the topology, not an oversight. `embed-serve` (ADR-015) is deliberately a second process holding a second model warm, and it carries **no** ModelManager and no gate at all, on the grounds that an embedder is small next to the multi-GB serve process. The gate is calibrated for *one* multi-GB model-swapping process; it does not arbitrate between processes and was never meant to. Anything that would put a second GB-scale model-swapping process on the same box has to reckon with that itself.

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
