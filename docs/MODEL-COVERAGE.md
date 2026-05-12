# Verified Model Coverage

> **Status: 2.0.6.** Updated per release.

Per-release report of which model types are verified for which operation
in mlx-knife. The policy behind this matrix lives in
[ADR-023](ADR/ADR-023-Text-First-Verified-Multimodal.md) (Text-First +
Verified Multimodal).

From 2.0.6 onwards, the lists follow upstream reality — entries are
removed or downgraded when upstream breaks them and no acceptable fix
exists. The 2.0.5 grandfather clause has ended.

## Legend

- ✅ **Verified** — integration tested in this release (local wet run and/or explicit unit test)
- 📚 **Listed** — in the runtime/convert whitelist, carried forward; stability depends on upstream
- ⚠ **Partial / Known issue** — listed but with a documented upstream limitation or active workaround
- ❌ **Not supported** — rejected explicitly at a gate or off the verified list

## Matrix

| Model types                                                               | Text | Vision | Audio | `convert --quantize` | Verified in | Notes |
|---------------------------------------------------------------------------|:----:|:------:|:-----:|:--------------------:|-------------|-------|
| `llama`, `mistral`, `mixtral`, `qwen3`, `phi3`, `phi3.5`, `gpt-oss`, `deepseek`, `gemma` (text families) | ✅   | —      | —     | ✅                   | 2.0.6       | Text backend via mlx-lm. Wet-umbrella (Phase 1: 275 passed) covers the in-portfolio set via Portfolio Discovery. **2.0.6 reclassifications via `chat_template.jinja` recognition:** `gpt-oss`, GLM-4.x-Flash variants, and Josiefied-Qwen3 variants now correctly typed as `chat` (previously `base`). **Manually verified** (RAM-gated out of automated suite): `Qwen3-Next-80B-A3B-Instruct-4bit`, `Llama-3.3-70B-Instruct-4bit`, `Llama-3.3-70B-Instruct-4bit-DWQ`. |
| `gemma4`                                                                  | ✅   | ✅     | 📚    | ✅                   | 2.0.6       | New family in 2.0.6 via MLX-stack refresh (mlx-lm 0.31.3, mlx-vlm 0.4.4). **Verified instances:** `gemma-4-e4b-it-4bit` (vision-runtime via `mlxk run --image`), `gemma-4-31b-bf16` / `gemma-4-31b-6bit` (text-only — audio/vision configs null), `gemma-4-26b-a4b-it-4bit` (text + vision). **Convert-quantize:** added to `VISION_QUANTIZE_TYPES` 2026-05-06; `gemma-4-31b-bf16 → 6bit` verified with vision tower preserved (Sentinel v2). **Vision quality caveat:** spatial-positional reading on chess-style tasks unreliable vs Pixtral (verified 2026-04-30 on T2.png). Audio capability listed (📚) — `audio_config` is truthy on some variants but not wet-tested for audio-in inference in this cycle. |
| `pixtral`                                                                 | —    | ✅     | —     | ✅                   | 2.0.6       | Vision-chat verified for `-4bit`, `-8bit`, `-bf16` clones; text-only prompts route cleanly to MLXRunner (ADR-020 three-tier routing). **Multi-image capable** — 5+ images per call on M2 Max 64 GB; larger models typically single-image. **Convert-quantize verified mlxk-internal:** `mlxk convert pixtral-12b-bf16 --quantize 6` → clean 10.8 GB target (6.803 bpw), `mlxk run` text + `mlxk run --image` both work on the quantized output without manual config repair. Currently the vision-test anchor in SMOKE-TEST because of the spatial-positional reading gap observed on Gemma 4. Note: text-only quality is not pixtral's strength — prefer a text-optimized model for text-only prompts. |
| `mllama`                                                                  | —    | ✅     | —     | 📚 (vlm)             | 2.0.6       | Vision-runtime verified mlxk-internal: `Llama-3.2-11B-Vision-Instruct-4bit` with `mlxk run --image` produces coherent image descriptions (output quality below pixtral, but functional). Single-image only. **Caveat:** vision-only model — `mlxk run` *without* `--image` currently routes to mlx-lm (which correctly does not support `mllama`) and fails with misleading `model_type not supported`; `mlxk show` then reports `runtime incompatible` even though the vision path works. Use with `--image`. Principled fix (Class C loader-gap) deferred to 2.1 (Issue #53). Convert-quantize not wet-tested mlxk-internal. |
| `mistral3`                                                                | —    | ✅     | —     | ✅                   | 2.0.6       | Vision + text-only verified with `Mistral-Small-3.1-24B-Instruct-2503-4bit`. **Legacy (pre-2026-03) conversions** need a two-step repair: (1) `mlxk convert --repair-index`, (2) manual `"spatial_merge_size": 2` in `processor_config.json`. **New conversions from mlx-vlm 0.4+ are clean** — mlx-vlm #741 (`spatial_merge_size`) closed upstream 2026-02-16. Un-repaired legacy mlx-community uploads remain broken by design. Unified `--repair` (ADR-018 Phase 4, detection-driven A1+A7 auto-fix) deferred to 2.0.7+ — current CLI offers `--repair-index` and `--quantize` only. |
| `whisper`                                                                 | —    | —      | ✅    | ❌                   | 2.0.6       | `large-v3-turbo` 4bit + 8bit variants verified (capability label, pre-execution-reject for text-only, audio runtime). Quantize not implemented for STT. **2.0.6 sunset:** mlx-audio #479 closed upstream — `_apply_tiktoken_patch` removed. Two `post_load_hook` / `get_tokenizer` patches for mlx-audio #645 remain in `mlxk2/core/audio_runner.py`; sunset on next mlx-audio release. |
| `vibevoice`                                                               | —    | —      | ✅    | ❌                   | 2.0.6       | In `AUDIO_MODEL_TYPES` and STT runtime. `VibeVoice-ASR-4bit` and `-8bit` capability-label classification verified — substring match (`vibevoice` covers `vibevoice_asr` and the `*_asr` family); previously misclassified as `base+audio`. Convert-quantize not implemented for STT. |
| `gemma3`                                                                  | —    | 📚     | —     | 📚 (vlm)             | 2.0.5       | In `VISION_QUANTIZE_TYPES`; carried from ADR-020. Not in 2.0.6 wet-benchmark — rely on upstream. |
| `llava`, `llava_next`, `qwen2_vl`, `phi3_v`, `paligemma`, `idefics`, `smolvlm` | —    | 📚     | —     | 📚 (partial)         | 2.0.5       | In `VISION_MODEL_TYPES` (runtime). `llava`, `qwen2_vl` also in `VISION_QUANTIZE_TYPES`; rest are runtime-only. Not in 2.0.6 wet-benchmark. Likely affected by the same vision-only routing bug as `mllama` for text-only invocation; verify per type before relying on `mlxk show`. |
| `idefics2`, `idefics3`, `mimo`                                            | —    | —      | —     | 📚 (vlm)             | 2.0.5       | `VISION_QUANTIZE_TYPES` members without runtime listing. Convert-only. `mimo` runtime fails with NoneType iteration error in the mlx-vlm processor (observed on `MiMo-VL-7B-RL-bf16`) — not part of the verified runtime set. |
| `gemma3n`                                                                 | ⚠    | ⚠     | ⚠    | ❌                   | 2.0.5       | Multimodal (vision + audio + text). Routes to mlx-vlm for runtime; no verified convert path. `convert --quantize` hard-rejects via `ErrorType.UNSUPPORTED_MULTIMODAL` (ADR-023). Not re-verified in 2.0.6; status carried from 2.0.5 pending dedicated test pass. |
| `voxtral`                                                                 | —    | —      | ❌    | ❌                   | 2.0.6       | Off the verified list. mlx-audio #450 (`tekken.json` tokenizer) still open upstream; transformers 5.5.x `VoxtralProcessor` still hardcodes `return_tensors="pt"` (torch dependency). Re-evaluate when #450 closes upstream. |

## Scope and generation

- **Verified** rows reflect model_types whose instances appeared in the wet-benchmark or SMOKE-TEST of the release named in `Verified in` and passed all test stages. The matrix intentionally names the model types, not every individual model instance.
- **Listed** rows reflect whitelist membership in `mlxk2/core/capabilities.py` without an empirical wet-benchmark entry in the current release. They remain supported via the whitelist but are flagged here so users know what *has not* been re-verified this cycle.
- **Partial / Not supported** rows reflect known upstream bugs, active workarounds with a sunset date, or explicit mlx-knife gates.

The `Verified in` column names the **last release that empirically tested the entry**. Not every release re-tests every entry — test portfolios are bounded by what the maintainer has locally available, and large models get rotated off disk. An entry labelled `Verified in: 2.0.5` in a 2.0.6 matrix simply means no 2.0.6 wet-benchmark covered it, not that it regressed.
