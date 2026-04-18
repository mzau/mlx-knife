# Verified Model Coverage

> **Status: 2.0.5.** Updated per release.

Per-release report of which model types are verified for which operation
in mlx-knife. The policy behind this matrix lives in
[ADR-023](ADR/ADR-023-Text-First-Verified-Multimodal.md) (Text-First +
Verified Multimodal).

2.0.5 is a **grandfather release**: entries that have worked at any point
in the 2.0.x line remain listed. From 2.0.6 onwards, the lists follow
upstream reality — entries are removed when upstream breaks them and no
acceptable fix exists.

## Legend

- ✅ **Verified** — integration tested in this release (local wet run and/or explicit unit test)
- 📚 **Listed** — in the runtime/convert whitelist, carried forward under the grandfather clause; stability depends on upstream
- ⚠ **Partial / Known issue** — listed but with a documented upstream limitation or active workaround
- ❌ **Not supported** — rejected explicitly at a gate

## Matrix

| Model types                                                               | Text | Vision | Audio | `convert --quantize` | Verified in | Notes |
|---------------------------------------------------------------------------|:----:|:------:|:-----:|:--------------------:|-------------|-------|
| `llama`, `mistral`, `mixtral`, `qwen3`, `phi3`, `phi3.5`, `gpt-oss`, `deepseek`, `gemma` (text families) | ✅   | —      | —     | ✅                   | 2.0.5       | Text backend via mlx-lm. Automated 2.0.5 wet-benchmark passed 17 distinct text models: Llama-3.2-3B, DeepSeek-R1-Distill-Llama-8B, Mistral-7B-v0.2, Mistral-Small-Instruct-2409, Mistral-Small-3.2-24B (4bit+8bit), DeepHermes-3-Mistral-24B, Mixtral-8x7B, Qwen2.5-0.5B, Qwen2.5-Coder-7B-8bit, Qwen3-30B/32B/Coder-30B/Coder-30B-DWQ, Phi-3-mini / 3.5-mini, gpt-oss-20b-MXFP4-Q8, OpenCodeInterpreter-DS-33B, Gabliterated-Qwen3-0.6B. **Manually verified (RAM-gated out of the automated suite):** `Qwen3-Next-80B-A3B-Instruct-4bit`, `Llama-3.3-70B-Instruct-4bit`, `Llama-3.3-70B-Instruct-4bit-DWQ`. |
| `pixtral`                                                                 | —    | ✅     | —     | ✅                   | 2.0.5       | Three variants in 2.0.5 wet-benchmark (`-4bit`, `-8bit`, `-bf16`), all passed vision-chat tests and routes text-only prompts cleanly to MLXRunner (ADR-020). Note: text-only quality is not pixtral's strength — prefer a text-optimized model for text-only prompts. |
| `mistral3`                                                                | —    | ✅     | —     | ✅                   | 2.0.5       | Vision + text-only verified with `Mistral-Small-3.1-24B-Instruct-2503-4bit` **after two-step repair**: (1) `mlxk convert --repair-index`, (2) manual `"spatial_merge_size": 2` in `processor_config.json`. Auto-repair planned for 2.0.6 via `--repair-config`. Un-repaired upstream mlx-community upload is broken by design. |
| `whisper`                                                                 | —    | —      | ✅    | ❌                   | 2.0.5       | 4bit + 8bit `large-v3-turbo` variants, 22 audio-chat tests passed. Quantize not implemented for STT. Runtime relies on three `sunset-by 2.0.6` workarounds in `mlxk2/core/audio_runner.py` (mlx-audio #479, #645). |
| `gemma3`                                                                  | —    | 📚     | —     | 📚 (vlm)             | 2.0.5       | In `VISION_QUANTIZE_TYPES`; carried from ADR-020. Not in this release's wet-benchmark — rely on upstream. |
| `llava`, `llava_next`, `qwen2_vl`, `phi3_v`, `mllama`, `paligemma`, `idefics`, `smolvlm` | —    | 📚     | —     | 📚 (partial)         | 2.0.5       | In `VISION_MODEL_TYPES` (runtime). `llava`, `qwen2_vl`, `mllama` also in `VISION_QUANTIZE_TYPES`; rest are runtime-only. Not in this release's wet-benchmark. |
| `idefics2`, `idefics3`, `mimo`                                            | —    | —      | —     | 📚 (vlm)             | 2.0.5       | `VISION_QUANTIZE_TYPES` members without runtime listing. Convert-only. Not in this release's wet-benchmark. |
| `vibevoice`                                                               | —    | —      | ⚠     | ❌                   | 2.0.5       | In `AUDIO_MODEL_TYPES` and STT runtime (new in 2.0.5 via ADR-022 HF-cache-isolation). Not in this release's wet-benchmark; status for 2.0.6 depends on upstream tokenizer situation. |
| `gemma3n`                                                                 | ⚠    | ⚠     | ⚠    | ❌                   | 2.0.5       | Multimodal (vision + audio + text). Routes to mlx-vlm for runtime; no verified convert path. `convert --quantize` hard-rejects via `ErrorType.UNSUPPORTED_MULTIMODAL` (ADR-023). |
| `voxtral`                                                                 | —    | —      | ⚠    | ❌                   | 2.0.5       | `tekken.json` tokenizer bug (mlx-audio #450) **and** transformers 5.0 `VoxtralProcessor` hardcodes `return_tensors="pt"` (PyTorch dependency). Blocked by runtime gate. Carried under the grandfather clause for visibility; evaluate for removal at 2.0.6 freeze. |

## Scope and generation

- **Verified** rows reflect model_types whose instances appeared in the wet-benchmark of the release named in `Verified in` and passed all test stages (`outcome: passed` across all recorded test invocations). The matrix intentionally names the model types, not every individual model instance.
- **Listed** rows reflect whitelist membership in `mlxk2/core/capabilities.py` without an empirical wet-benchmark entry in the current release. They remain supported through the grandfather clause but are flagged here so users know what *has not* been re-verified this cycle.
- **Partial / Not supported** rows reflect known upstream bugs, active workarounds with a sunset date, or explicit mlx-knife gates.

The `Verified in` column names the **last release that empirically tested the entry**. Not every release re-tests every entry — test portfolios are bounded by what the maintainer has locally available, and large models get rotated off disk. An entry labelled `Verified in: 2.0.4` in a 2.0.6 matrix simply means no 2.0.5 or 2.0.6 wet-benchmark covered it, not that it regressed.
