# ADR-024: Pre-Execution Capability-Mismatch Reject

**Status:** Partially Implemented — Class A (STT / Embedding) shipped 2.0.6 (`2de2f21`, smoke §P). Class C (Loader Gap) + Class D (Invocation Gap) deferred 2.1.
**Created:** 2026-04-19
**Updated:** 2026-05-11 — title widened from "Vision-only Pre-Execution Routing" to reflect the scope of the pattern as actually shipped in 2.0.6; status promoted from "Proposed (stub)" to "Partially Implemented"; Class A shipped section added; Decision section retro'd — the pattern-choice was made implicitly when `2de2f21` landed.
**Related:** ADR-020 (Audio Backend Architecture), ADR-022 (Workspace-First Paradigm), ADR-023 (Text-First + Verified Multimodal), [Issue #53](https://github.com/mzau/mlx-knife/issues/53)
**Target:** Class A — 2.0.6 ✅. Class C + D — 2.1.
**Analytical anchor:** [`docs/RUNTIME-FEATURES.md`](../RUNTIME-FEATURES.md) §5 Bug Class Catalog. The four-bug-class model (A: STT misclassification, B: audio/vision key-FP, C: loader gap, D: invocation gap) is the single source of truth; this ADR records the routing decision derived from that catalog.

---

## Context

When `mlxk run` receives an invocation that does not match the model's actual capability, two failure modes have been observed:

1. **Cryptic upstream error.** The request reaches the runner (mlx-lm / mlx-vlm / mlx-audio); the framework fails late with messages like `"Model type mllama not supported."` or `"'model'"` KeyError. The user has no path back to a corrective action.
2. **`show` lies relative to `run`.** The health-aggregator (`common.py:645-647`) requires BOTH vision_load AND text_load for vision-capable models. mllama, gemma3n and gemma-4-e4b fail text_load but succeed vision_load — `show` reports `runtime incompatible`, while `run --image` works clean. Two surfaces, two truths.

The current routing comment in `mlxk2/operations/run.py:502-505` ("Vision-capable models WITHOUT media input fall through to Text LLM path") is a *design* decision that works for the majority of VLMs (mlx-lm has the text-tower loader) but lacks an explicit pre-execution check for the cases where it does not.

## Problem

Need a deterministic, pre-execution capability detection (no try/catch recovery) that:

1. Recognises capability/invocation mismatches before reaching the runner.
2. Routes accordingly with a user-friendly typed error and a concrete correction hint.
3. Reads the same truth as `mlxk show` / `health`, so the two surfaces stop contradicting each other.

## Decision

**Pattern: pre-execution typed reject.** Inside `mlxk run`, before the runner is invoked, classify the model's capability via the existing detection layer (`detect_model_type` in `common.py`) and emit a typed error whenever the requested invocation form is incompatible with that capability. Reject UX form: short error, concrete hint, JSON-mode result-string clean.

This pattern is **chosen, not TBD** — it shipped in 2.0.6 for Class A (STT / embedding) via `2de2f21` and is verified in Smoke-Test §P. Class C + D extend the same pattern to additional mismatch axes.

### Rejected alternatives (preserved for context)

- **`VisionRunner(images=None)`-Fallback.** mlx-vlm `generate()`-defaults are 100–256 tokens (image-captioning-dimensioned; see `mlx_vlm/generate.py:235,820,1030,1195`); MLXRunner default is `calculate_dynamic_max_tokens(server_mode=False)` ≈ full context (~131k on Pixtral). 500–1300× discrepancy → text-only via VisionRunner produces reproducibly truncated outputs the user cannot diagnose. Pre-execution reject is the correct UX.
- **Try/catch recovery at the runner.** Runtime recovery does not solve the show/run divergence — `show` makes no try/catch call, would need its own logic. Pre-execution detection is the only single-source-of-truth variant.

---

## Bug Class Coverage (cf. RUNTIME-FEATURES.md §5)

| Class | Mismatch                  | Invocation                  | Pre-Execution Reject                                                  | Status                           |
|-------|---------------------------|-----------------------------|-----------------------------------------------------------------------|----------------------------------|
| A     | audio-only (STT)          | text-only call              | ✅ `Use \`--audio FILE\` for transcription.`                          | Shipped 2.0.6 (`run.py:462-486`) |
| A     | embedding-only            | text-only call              | ✅ `\`mlxk run\` does not support embeddings.`                        | Shipped 2.0.6 (`run.py:462-486`) |
| B     | audio/vision key-FP       | (detection layer)           | n/a — fixed at `detect_audio_capability` / `detect_vision_capability` | Shipped 2.0.6 (separate fix)     |
| C     | vision-only (loader gap)  | text-only call              | ❌ `Use \`--image\` for vision-only model.`                           | **Pending 2.1**                  |
| D     | base + multimodal         | `--image` / `--audio` call  | ❌ `Base variant has no chat template for media; use \`-it\` sibling.`| **Pending 2.1**                  |

Class B is listed for completeness — it is a detection-layer bugfix (key-existence → truthy-dict predicate) and does not interact with routing. It shipped alongside Class A in `2de2f21` but lives elsewhere in code (`common.py:299,233`).

---

## Class A — Shipped 2.0.6

**Code anchor:** `mlxk2/operations/run.py:462-486` (commit `2de2f21`).

**Mechanism.**

```python
# run.py:468-486 (paraphrased)
if not audio and not images and resolved_name and model_path is not None and cfg is not None:
    tok = read_tokenizer_hints(model_path)
    mt = detect_model_type(resolved_name, cfg, tok, model_path)
    if mt == "audio":
        return f"Error: Model '{model_spec}' is audio-only (STT). Use `--audio FILE` for transcription."
    if mt == "embedding":
        return f"Error: Model '{model_spec}' is an embedding model — `mlxk run` does not support embeddings."
```

**Discriminator.** `detect_model_type` uses substring matching (Class A surgical patch in same commit: `STT_MODEL_TYPES = {"whisper", "vibevoice", "voxtral"}`, substring match in `common.py:202`). Result: `vibevoice_asr` matches `vibevoice`, `whisper-large-v3-turbo-4bit` matches `whisper`, etc.

**JSON-mode behaviour.** The reject is returned as the result string; callers receive a clean payload, no exception propagation. Verified in Smoke-Test §P5.

**Affected models in current portfolio (regression anchors).**
- STT: `whisper-large-v3-turbo-{4,8}bit`, `VibeVoice-ASR-{4,8}bit`
- Embedding: any model classified by `detect_model_type` as `embedding` via the embedding-marker pathway

---

## Class C — Pending 2.1 (Loader Gap, originally "Vision-only Routing")

**Symptom.** `mlxk run <C-class-model> "prompt"` (text-only) fails inside the runner with cryptic errors:

- `'model'` KeyError (gemma3n family — no loader module)
- "Received N parameters not in model" (gemma-4-e4b — loader exists but config shape exceeds it)
- `"Model type mllama not supported."` (mllama — Cross-Attention architecture, no separable text-tower)

**Empirical scope** (RUNTIME-FEATURES.md §5 Class C table):

| Model                    | model_type | text-only loadable? |
|--------------------------|------------|---------------------|
| gemma-4-31b-6bit         | gemma4     | ✅                  |
| gemma-4-26b-a4b-it-4bit  | gemma4     | ✅                  |
| gemma-4-e4b-it-4bit      | gemma4     | ❌ param mismatch   |
| gemma-3n-E2B-4bit        | gemma3n    | ❌ KeyError 'model' |
| gemma-3n-E2B-it-4bit     | gemma3n    | ❌ KeyError 'model' |
| Llama-3.2-11B-Vision     | mllama     | ❌ no loader        |

→ The discriminator is **not** purely `model_type`. The hypothesis (RUNTIME-FEATURES.md §5 Class C): text-only is loadable iff `model_type ∈ MLX_LM_TEXT_LOADER_TYPES` (auto-discovered) AND `audio_config` is not a truthy dict. `vision_config`-truthy alone does not break the loader.

**Implementation sketch (2.1).**

1. **`MLX_LM_TEXT_LOADER_TYPES` discovery** in `mlxk2/core/capabilities.py`:
   - Auto-discover via `pkgutil.iter_modules(mlx_lm.models.__path__)`.
   - Include `mlx_lm.utils.MODEL_REMAPPING` keys (e.g. `llava → mistral3`).
   - Filter out non-loader modules (`base`, `cache`, `activations`, …) — set-based exclude list maintained in capabilities.py.
   - Cached at import time; no runtime cost on `mlxk list` hot path.
2. **Pre-execution check in `run.py:462-486` region.** Extend the existing Class-A gate: if `mt in VISION_MODEL_TYPES` and `model_type not in MLX_LM_TEXT_LOADER_TYPES` and no `--image`, return `Error: Model '...' is vision-only in this mlx-lm build (no text-tower loader). Use \`--image\` for inference.`
3. **Capability label derivation.** Same allowlist drives the `text` capability label exposed by `mlxk show` / `mlxk list --json`.
4. **Health-aggregator alignment** in `common.py:645-647`: when `model_type ∈ VISION_MODEL_TYPES` AND `model_type ∉ MLX_LM_TEXT_LOADER_TYPES` AND vision_load succeeds → `runtime_compatible = True`, `runtime_reason = "healthy, vision-only — use --image"`. Single truth-source; `show` no longer contradicts `run`.

**Open design questions for the 2.1 implementer.**

1. Auto-discovery vs. curated frozenset — auto-discovery is preferred (zero per-mlx-lm-release maintenance), but the exclude-list for non-loader modules has to live somewhere. Curated frozenset is more controllable but ages with mlx-lm.
2. Discovery scope: include `MODEL_REMAPPING` keys? Audit set-difference `VISION_MODEL_TYPES \ MLX_LM_TEXT_LOADER_TYPES` against the live mlx-lm version to surface unknown members beyond `mllama`/`gemma3n`/`gemma-4-e4b`.
3. Reject-message wording must round-trip through JSON-API (already established for Class A, reuse).

**Issue tracker.** [#53](https://github.com/mzau/mlx-knife/issues/53).

---

## Class D — Pending 2.1 (Invocation Gap, base + media)

**Symptom.** A base model with `vision_config`/`audio_config` truthy lets `--image`/`--audio` through to the runner and either errors late ("0 audio tokens in the text and N tokens from audio embeddings") or produces base-continuation off the metadata header — not a useful answer. Cause: multimodal grounded chat presupposes a chat template emitting media-placeholder tokens; base models lack such templates.

**Implementation sketch (2.1).** Reachability layer 3 must probe for chat-template-with-media-placeholder before reporting `vision-in` / `audio-in` as reachable. When the probe fails: pre-execution reject from `mlxk run` with hint to use the `-it` sibling variant; `show`/`list` report media-axes as not reachable. Layer 3 design lives in the Iter 2/3 reachability plan (`[POLICY]`-driven reframe per ADR-023 §4 No-Silent-Degradation).

---

## Acceptance Criteria

For Class C to ship (2.1):

- [ ] `MLX_LM_TEXT_LOADER_TYPES` auto-discovery in `capabilities.py`, with explicit non-loader-module exclude list.
- [ ] `mlxk run` Class C reject in `run.py:462-486` region — typed error, JSON-clean.
- [ ] `mlxk show` no longer reports `runtime incompatible` for mllama / gemma3n / gemma-4-e4b — `runtime_compatible = True` with `"healthy, vision-only — use --image"` reason.
- [ ] Smoke-Test §K transitions from `DEFER 2.1` to `ok`.
- [ ] Set-difference audit of `VISION_MODEL_TYPES \ MLX_LM_TEXT_LOADER_TYPES` in CHANGELOG: explicit list of newly-rejected model_types per release.

For Class D to ship (2.1, gated on Iter 2/3 reachability layer 3):

- [ ] Chat-template-with-media-placeholder probe in `capabilities.py`.
- [ ] `mlxk run --image` / `--audio` reject for base+multimodal in `run.py`.
- [ ] `mlxk show` / `mlxk list --json` report media-axes truthfully.
- [ ] Test fixtures cover `gemma-3n-E2B-4bit` (base+vision+audio), `gemma-4-31b-bf16` (base+vision), at least one verified `-it` sibling for regression.

---

## Notes / Provenance

- The original 2026-04-19 ADR draft framed the problem strictly as "Vision-only routing" with the mllama empirical case. The 2026-05-11 update widens the title to reflect what actually shipped (Class A) and what the pattern actually covers.
- The Class A landing in `2de2f21` was treated at commit time as a "UX-Gate STT/Embedding-Reject" (per CHANGELOG and Smoke-Test §P labelling), not as an ADR-024 instantiation. Both framings describe the same mechanism — this ADR is the architectural record connecting them.
- Class B (audio/vision key-FP) is recorded here for completeness only. Its fix is detection-layer, not routing-layer; the canonical record is RUNTIME-FEATURES.md §5 Class B.
