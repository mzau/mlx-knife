# Verified Model Coverage

> **Status: 2.0.7 (unreleased).** Updated per release.

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

| Model types                                                               | Text | Vision | Audio | Embed | `convert --quantize` | Verified in | Notes |
|---------------------------------------------------------------------------|:----:|:------:|:-----:|:-----:|:--------------------:|-------------|-------|
| `llama`, `mistral`, `mixtral`, `qwen3`, `phi3`, `phi3.5`, `gpt-oss`, `deepseek`, `gemma` (text families) | ✅   | —      | —     | — | ✅                   | 2.0.6       | Text backend via mlx-lm. Wet-umbrella (Phase 1: 275 passed) covers the in-portfolio set via Portfolio Discovery. **2.0.6 reclassifications via `chat_template.jinja` recognition:** `gpt-oss`, GLM-4.x-Flash variants, and Josiefied-Qwen3 variants now correctly typed as `chat` (previously `base`). **Manually verified** (RAM-gated out of automated suite): `Qwen3-Next-80B-A3B-Instruct-4bit`, `Llama-3.3-70B-Instruct-4bit`, `Llama-3.3-70B-Instruct-4bit-DWQ`. |
| `gemma4`                                                                  | ✅   | ✅     | 📚    | — | ✅                   | 2.0.6       | New family in 2.0.6 via MLX-stack refresh (mlx-lm 0.31.3, mlx-vlm 0.4.4). **Verified instances:** `gemma-4-e4b-it-4bit` (vision-runtime via `mlxk run --image` — **off-list as of the 2.0.7 dep bump**, see below), `gemma-4-31b-bf16` / `gemma-4-31b-6bit` (text-only — audio/vision configs null), `gemma-4-26b-a4b-it-4bit` (text + vision). **2.0.7 dep bump (mlx-vlm 0.6.2): `gemma-4-e4b-it-4bit` falls off the verified list** — the stale 0.4.3-era community conversion no longer loads under mlx-vlm 0.6.x KV-sharing (mlx-vlm #1301; orphan parameters → `Received N parameters not in model`). Recovery is a fresh post-0.6.x conversion; none re-verified yet. The 31b text-only instances run via mlx-lm and are unaffected by the mlx-vlm bump. **Convert-quantize:** added to `VISION_QUANTIZE_TYPES` 2026-05-06; `gemma-4-31b-bf16 → 6bit` verified with vision tower preserved (Sentinel v2). **Vision quality caveat:** spatial-positional reading on chess-style tasks unreliable vs Pixtral (verified 2026-04-30 on T2.png). Audio capability listed (📚) — `audio_config` is truthy on some variants but not wet-tested for audio-in inference in this cycle. |
| `pixtral`                                                                 | —    | ✅     | —     | — | ✅                   | 2.0.6       | Vision-chat verified for `-4bit`, `-8bit`, `-bf16` clones; text-only prompts route cleanly to MLXRunner (ADR-020 three-tier routing). **Multi-image capable** — 5+ images per call on M2 Max 64 GB; larger models typically single-image. **Convert-quantize verified mlxk-internal:** `mlxk convert pixtral-12b-bf16 --quantize 6` → clean 10.8 GB target (6.803 bpw), `mlxk run` text + `mlxk run --image` both work on the quantized output without manual config repair. Currently the vision-test anchor in SMOKE-TEST because of the spatial-positional reading gap observed on Gemma 4. Note: text-only quality is not pixtral's strength — prefer a text-optimized model for text-only prompts. |
| `mllama`                                                                  | —    | ✅     | —     | — | 📚 (vlm)             | 2.0.6       | Vision-runtime verified mlxk-internal: `Llama-3.2-11B-Vision-Instruct-4bit` with `mlxk run --image` produces coherent image descriptions (output quality below pixtral, but functional). Single-image only. **Caveat:** vision-only model — `mlxk run` *without* `--image` currently routes to mlx-lm (which correctly does not support `mllama`) and fails with misleading `model_type not supported`; `mlxk show` then reports `runtime incompatible` even though the vision path works. Use with `--image`. Principled fix (Class C loader-gap) deferred to 2.1 (Issue #53). Convert-quantize not wet-tested mlxk-internal. |
| `mistral3`                                                                | —    | ✅     | —     | — | ✅                   | 2.0.6       | Vision + text-only verified with `Mistral-Small-3.1-24B-Instruct-2503-4bit`. **Legacy (pre-2026-03) conversions** need a two-step repair: (1) `mlxk convert --repair-index`, (2) manual `"spatial_merge_size": 2` in `processor_config.json`. **New conversions from mlx-vlm 0.4+ are clean** — mlx-vlm #741 (`spatial_merge_size`) closed upstream 2026-02-16. Un-repaired legacy mlx-community uploads remain broken by design. Unified `--repair` (ADR-018 Phase 4, detection-driven A1+A7 auto-fix) deferred to 2.0.7+ — current CLI offers `--repair-index` and `--quantize` only. **2.0.7 dep bump:** repaired `mistral3` conversions re-verified under mlx-vlm 0.6.2 (2026-06-09 bump eval); the interim 0.6.0 pin (never released) had broken them. |
| `whisper`                                                                 | —    | —      | ✅    | — | ❌                   | 2.0.6 · 2.0.7 (translate) | `large-v3-turbo` 4bit + 8bit variants verified (capability label, pre-execution-reject for text-only, audio runtime). Quantize not implemented for STT. **`audio-translate-en` (#54, 2.0.7):** **multilingual non-turbo** variants (verified `whisper-large-v3-4bit`) translate non-English speech → English, via CLI `--translate` and server `POST /v1/audio/translations`. **Turbo variants** (`large-v3-turbo-4bit/8bit`) and **`.en` variants** are *not* translate-capable and are rejected pre-execution (HTTP 422 on the server) — turbo's reduced 4-layer decoder emits garbage instead of English, `.en` checkpoints lack the `<\|translate\|>` token. Non-Whisper STT (`voxtral`, `vibevoice`) is rejected for translate too. **2.0.6 sunset:** mlx-audio #479 closed upstream — `_apply_tiktoken_patch` removed. Two `post_load_hook` / `get_tokenizer` patches for mlx-audio #645 remain in `mlxk2/core/audio_runner.py` (still required at 0.4.4); sunset when mlx-audio ≥ 0.5 via the `_MLX_AUDIO_NEEDS_PATCHES` version gate. |
| `vibevoice`                                                               | —    | —      | ✅    | — | ❌                   | 2.0.6       | In `AUDIO_MODEL_TYPES` and STT runtime. `VibeVoice-ASR-4bit` and `-8bit` capability-label classification verified — substring match (`vibevoice` covers `vibevoice_asr` and the `*_asr` family); previously misclassified as `base+audio`. Convert-quantize not implemented for STT. |
| `gemma3`                                                                  | —    | 📚     | —     | — | 📚 (vlm)             | 2.0.5       | In `VISION_QUANTIZE_TYPES`; carried from ADR-020. Not in 2.0.6 wet-benchmark — rely on upstream. |
| `llava`, `llava_next`, `qwen2_vl`, `phi3_v`, `paligemma`, `idefics`, `smolvlm` | —    | 📚     | —     | — | 📚 (partial)         | 2.0.5       | In `VISION_MODEL_TYPES` (runtime). `llava`, `qwen2_vl` also in `VISION_QUANTIZE_TYPES`; rest are runtime-only. Not in 2.0.6 wet-benchmark. Likely affected by the same vision-only routing bug as `mllama` for text-only invocation; verify per type before relying on `mlxk show`. |
| `idefics2`, `idefics3`, `mimo`                                            | —    | —      | —     | — | 📚 (vlm)             | 2.0.5       | `VISION_QUANTIZE_TYPES` members without runtime listing. Convert-only. `mimo` runtime fails with NoneType iteration error in the mlx-vlm processor (observed on `MiMo-VL-7B-RL-bf16`) — not part of the verified runtime set. |
| `gemma3n`                                                                 | ⚠    | ⚠     | ⚠    | — | ❌                   | 2.0.5       | Multimodal (vision + audio + text). Routes to mlx-vlm for runtime; no verified convert path. `convert --quantize` hard-rejects via `ErrorType.UNSUPPORTED_MULTIMODAL` (ADR-023). Not re-verified in 2.0.6; status carried from 2.0.5 pending dedicated test pass. |
| `voxtral`                                                                 | —    | —      | ❌    | — | ❌                   | 2.0.6       | Off the verified list. mlx-audio #450 (`tekken.json` tokenizer) still open upstream; transformers 5.5.x `VoxtralProcessor` still hardcodes `return_tensors="pt"` (torch dependency). Re-evaluate when #450 closes upstream. |
| `molmo2`                                                                  | —    | ❌     | —     | — | ❌                   | —           | Off the verified list — evaluated, not adopted. 2026-06-02 against **mlx-vlm 0.6.0** (out-of-tree eval venv, not the 2.0.6 pin): runtime-vision (`run --image`, fp16 + 6bit) and convert (`fp16 → 6bit`, vision tower preserved, Sentinel v2) both functional; mlx-vlm 0.6.0 fp16-overflow-fix effective. Text-only (`run "prompt"` without `--image`) correctly errors (late mlx-lm loader gap) — `vision-in` is the only reachable mode, so the `chat+vision` capability label overstates (should be `vision`-only). Three independent gates blocked adoption: (a) the dep bump — landed with 2.0.7 (mlx-vlm 0.6.2); (b) `molmo2` absent from `VISION_MODEL_TYPES` + `VISION_QUANTIZE_TYPES` (also tripped by `video_preprocessor_config.json` → `_detect_vision_from_files` video-exclusion); (c) Class-C runtime-gate ([Issue #53](https://github.com/mzau/mlx-knife/issues/53)) — `mlx_lm._get_classes` rejects `molmo2` like `mllama`, so it shows `Runtime: no` yet still executes via `--image` (gating bug, part of Issue #53). Lower priority than `qwen2_5_vl` / `qwen3_vl` / `internvl_chat`. Re-evaluate when #53 lands. |
| `bert` | — | — | — | ✅ | ❌ | 2.0.7 | Vendored MIT encoder (`encoders/bert.py`); `bge-small-en-v1.5` (CLS) + `multilingual-e5-small` (mean) verified live; `mxbai-embed-large-v1` rides the same path; L2-normalize default. |
| `qwen3` (Qwen3-Embedding) | — | — | — | ✅ | 📚 | 2.0.7 | Decoder embedder via mlx-lm — distinct from the chat `qwen3` row above. Qwen3-Embedding-0.6B family; last-token pool + L2-normalize + instruction prefix. Showcase `Qwen3-Embedding-0.6B-4bit-DWQ`. **Convert-quantize (📚, not the practical path):** `qwen3` is convert-capable via the text path, but Qwen3-Embedding ships only as `-4bit-DWQ` / `-8bit` / `-mxfp8` — no full-precision MLX source to feed `mlxk convert`, and the full-precision original is PyTorch (out of mlxk convert scope). DWQ quality-recovery is an upstream method, **not** an `mlxk convert` output; `-mxfp8` targets the GPLv3 `mlx-embeddings` lib and does not load via the mlx-lm decoder path. |
| `xlm-roberta`, `modernbert`, `nomic_bert` | — | — | — | ❌ | — | — | Declared embedders whose encoder is **not vendored** → honest pre-exec reject, never a silent failure. `xlm-roberta` (bge-m3, multilingual-e5-large) is the ~30-line fast-follow. |
| `gemma3_text` (EmbeddingGemma) | — | — | — | ❌ | — | — | Deferred (BEYOND): shared with plain causal Gemma-3 text LMs, so never reclassified from `model_type` alone; EmbeddingGemma needs bidirectional-attention handling + a sentence-transformers sidecar signal. |

## Scope and generation

- **Verified** rows reflect model_types whose instances appeared in the wet-benchmark or SMOKE-TEST of the release named in `Verified in` and passed all test stages. The matrix intentionally names the model types, not every individual model instance.
- **Listed** rows reflect whitelist membership in `mlxk2/core/capabilities.py` without an empirical wet-benchmark entry in the current release. They remain supported via the whitelist but are flagged here so users know what *has not* been re-verified this cycle.
- **Partial / Not supported** rows reflect known upstream bugs, active workarounds with a sunset date, or explicit mlx-knife gates.

The `Verified in` column names the **last release that empirically tested the entry**. Not every release re-tests every entry — test portfolios are bounded by what the maintainer has locally available, and large models get rotated off disk. An entry labelled `Verified in: 2.0.5` in a 2.0.6 matrix simply means no 2.0.6 wet-benchmark covered it, not that it regressed.

## Embeddings (ADR-015)

The `mlxk embed` verb (experimental, gated by `MLXK2_ENABLE_ALPHA_FEATURES=1`) adds the **Embed**
column to the matrix above. Embeddings is an orthogonal capability **axis**, not a separate model
class — a decoder embedder is a causal architecture repurposed (last-token pooling), and a single
model can carry both `chat` and `embeddings`. Detection is **config-first** (ADR-015 Slice C): a
runnable embedder declares the `embeddings` capability *and* is on the verified-runnable subset
(vendored `bert` encoder, or a `qwen3` decoder riding mlx-lm). Declared-but-not-vendored encoders
(`xlm-roberta`/`modernbert`/`nomic_bert`) surface honestly as not-runnable
(`embedding (not runnable: encoder not vendored)`) — never a silent failure. The `convert
--quantize` column is orthogonal to embed and modest in practice: `qwen3` is convert-capable via
the text path (📚) but has no full-precision MLX source to convert (Qwen3-Embedding ships only as
DWQ/8bit/mxfp8, and DWQ quality-recovery is upstream, not an `mlxk convert` output); the vendored
`bert` encoder has no mlxk convert path (❌).

**Surface note.** `mlxk list`/`show` present runnable embedders as runnable; serve's `/v1/models`
deliberately does **not** advertise them in 2.0.7 (the embed-backend `/v1/models` merge is deferred
to 2.1). `mlxk run <embedder>` rejects pre-exec — the verb for an embedder is `mlxk embed`.

> **Follow-up (post-Slice-C):** the full capability-axis presentation — multimodal embedders
> (vision/audio, BEYOND §3) and the per-axis `declared ∩ runnable` view — is a doc + ADR-015
> revision tracked for a later session, together with the `embed` output-format / `--json`
> consistency question.
