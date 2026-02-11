# Benchmark Report v1.1: 2.0.4

**Date:** 2026-02-11-wet-benchmark-2.0.4-stable
**Generated:** 2026-02-11T13:13:56 UTC
**Generator:** generate_benchmark_report.py v1.1
**Hardware:** Mac14,13, 12 cores

---

## Input Files

- **Primary:** `benchmarks/reports/2026-02-11-wet-benchmark-2.0.4-stable.jsonl`
- **Schema:** v0.2.2
- **Comparison:** `benchmarks/reports/2026-02-06-wet-benchmark-1.jsonl`

---

## Executive Summary

**Tests:** 251 total (179 passed, 72 skipped)
**Duration:** 1157.1s (19.3 min)
**Quality:** 100.0% clean (251/251)
**Models:** 25 tested

### Comparison

**vs:** `2026-02-06-wet-benchmark-1.jsonl`
**Duration:** 19.6 min → 19.3 min (-1.7%) ✅
**Models:** 16/25 slower (64%), 9/25 faster (36%)

✅ **System Health:** All tests clean (RAM >5 GB free, 0 zombies)

---

## Test Summary

```
Total tests:       251
Passed:            179
  With model:      122
  Infrastructure:  57
Skipped:           72
Duration:          1157.1s (19.3 min)
```

---

## System Health

```
Swap (MB):         min=1524, max=1524, avg=1524.0
RAM free (GB):     min=8.5, max=38.8, avg=22.8
Zombies:           min=0, max=0

Quality Flags (Thresholds: RAM <5 GB free, zombies >0):
  Clean:           251/251 (100.0%)
  Degraded (RAM):  0
  Degraded (zombies): 0
```

---

## Per-Model Statistics

| Model | Size | Mode | Tests | Time | Old | Δ | Change | RAM (GB) |
|-------|-----:|:----:|------:|-----:|----:|--:|-------:|---------:|
| Mistral-Small-3.1-24B-Instruct-2503-4bit | 13.2 GB | Vision | 2 | 42.3s | 42.8s | -0.5s | -1.2% ✅ | 18.8-19.1 |
| Mistral-Small-3.1-24B-Instruct-2503-4bit | 13.2 GB | Text | 1 | 5.8s | 6.9s | -1.0s | -15.2% ✅ | 15.1 |
| DeepHermes-3-Mistral-24B-Preview-8bit | 23.3 GB | Text | 7 | 134.0s | 138.4s | -4.5s | -3.2% ✅ | 23.8-28.1 |
| DeepSeek-R1-Distill-Llama-8B-4bit | 4.2 GB | Text | 4 | 30.5s | 30.3s | +0.3s | +0.8%  | 19.9-20.3 |
| EuroLLM-22B-Instruct-2512-mlx-8bit | 22.4 GB | Text | 4 | 55.8s | 55.5s | +0.3s | +0.6%  | 23.6-23.7 |
| Gabliterated-Qwen3-0.6B-float32 | 2.2 GB | Text | 4 | 28.6s | 28.1s | +0.5s | +1.7%  | 21.0-21.4 |
| gpt-oss-20b-MXFP4-Q8 | 11.3 GB | Text | 4 | 51.5s | 50.9s | +0.6s | +1.2%  | 14.4-14.7 |
| Llama-3.2-3B-Instruct-4bit | 1.7 GB | Text | 4 | 15.6s | 14.9s | +0.8s | +5.1% ⚠️ | 19.2-19.7 |
| Mistral-7B-Instruct-v0.2-4bit | 4.0 GB | Text | 4 | 14.3s | 14.2s | +0.2s | +1.2%  | 15.2-15.7 |
| Mistral-Small-3.2-24B-Instruct-2506-4bit | 12.4 GB | Text | 4 | 36.9s | 35.7s | +1.2s | +3.4%  | 13.4-13.5 |
| Mistral-Small-3.2-24B-Instruct-2506-8bit | 23.3 GB | Text | 4 | 62.3s | 66.3s | -4.0s | -6.0% ✅ | 24.4-25.3 |
| Mistral-Small-Instruct-2409-4bit | 11.7 GB | Text | 4 | 28.0s | 27.7s | +0.3s | +1.0%  | 12.6-13.4 |
| Mixtral-8x7B-Instruct-v0.1-4bit | 24.5 GB | Text | 4 | 51.9s | 53.1s | -1.2s | -2.2% ✅ | 27.3-27.4 |
| OpenCodeInterpreter-DS-33B-hf-4bit-mlx | 17.8 GB | Text | 4 | 37.9s | 38.1s | -0.2s | -0.5%  | 18.7-18.8 |
| Phi-3-mini-4k-instruct-4bit | 2.0 GB | Text | 4 | 10.9s | 12.3s | -1.3s | -11.0% ✅ | 16.7-16.8 |
| Phi-3.5-mini-instruct-4bit | 2.0 GB | Text | 4 | 13.2s | 13.1s | +0.1s | +0.4%  | 14.6-14.8 |
| pixtral-12b-4bit | 7.0 GB | Vision | 7 | 79.7s | 25.0s | +54.7s | +218.4% ⚠️ | 12.6-18.8 |
| pixtral-12b-4bit | 7.0 GB | Text | 1 | 3.7s | 4.2s | -0.5s | -11.2% ✅ | 12.8 |
| pixtral-12b-8bit | 12.6 GB | Vision | 2 | 29.5s | 25.9s | +3.6s | +13.9% ⚠️ | 16.4-16.6 |
| pixtral-12b-8bit | 12.6 GB | Text | 1 | 3.9s | 3.4s | +0.5s | +14.6% ⚠️ | 14.3 |
| Qwen2.5-0.5B-Instruct-4bit | 0.3 GB | Text | 4 | 16.3s | 22.9s | -6.6s | -28.7% ✅ | 14.3-14.5 |
| Qwen2.5-Coder-7B-Instruct-8bit | 7.5 GB | Text | 4 | 22.5s | 21.5s | +1.0s | +4.9%  | 8.5-8.7 |
| Qwen3-30B-A3B-Instruct-2507-4bit | 16.0 GB | Text | 4 | 34.5s | 34.4s | +0.2s | +0.5%  | 17.1-17.3 |
| Qwen3-32B-4bit | 17.2 GB | Text | 4 | 63.6s | 62.7s | +0.9s | +1.4%  | 18.4-18.7 |
| Qwen3-Coder-30B-A3B-Instruct-4bit | 16.0 GB | Text | 4 | 36.5s | 37.7s | -1.2s | -3.3% ✅ | 17.1-17.2 |
| Qwen3-Coder-30B-A3B-Instruct-6bit-DWQ-lr9e-8 | 24.9 GB | Text | 4 | 56.6s | 56.7s | -0.2s | -0.3%  | 26.0-26.1 |
| whisper-large-v3-turbo-4bit | 0.4 GB | Audio | 11 | 25.5s | 25.4s | +0.1s | +0.5%  | 33.7-34.9 |
| whisper-large-v3-turbo-8bit | 0.8 GB | Audio | 11 | 24.6s | 24.5s | +0.1s | +0.4%  | 33.7-34.4 |

### Model Categories

```
LARGE MODELS (≥20 GB): 5 models
  Avg size:               23.7 GB
  Text Tests:
    Models tested:        5
    Avg test time:        15.2s
    RAM range:            23.6-28.1 GB

MEDIUM MODELS (10-20 GB): 9 models
  Avg size:               14.2 GB
  Vision Tests:
    Models tested:        2
    Avg test time:        18.0s
    RAM range:            16.4-19.1 GB
  Text Tests:
    Models tested:        9
    Avg test time:        9.1s
    RAM range:            12.6-18.8 GB

SMALL MODELS (<10 GB): 11 models
  Avg size:               2.9 GB
  Vision Tests:
    Models tested:        1
    Avg test time:        11.4s
    RAM range:            12.6-18.8 GB
  Text Tests:
    Models tested:        9
    Avg test time:        4.6s
    RAM range:            8.5-21.4 GB
  Audio Tests:
    Models tested:        2
    Avg test time:        2.3s
    RAM range:            33.7-34.9 GB
```

---

## Per-Test Statistics

Shows performance range across models for each test.

| Test Name | Mode | Models | Fastest | Slowest | Med | Old | Δ Med |
|-----------|:----:|-------:|---------|---------|----:|----:|------:|
| test_audio_output_not_empty | Audio | 2 | whisper (1.9s) | whisper (1.9s) | 1.9s | 1.9s | +0.8% |
| test_chart_axis_label_reading | Vision | 1 | pixtral (12.4s) | pixtral (12.4s) | 12.4s | 12.8s | -2.9% |
| test_chat_completions_batch | Text | 20 | Qwen2.5 (2.1s) | Qwen3 (12.4s) | 7.9s | 7.9s | -0.3% |
| test_chat_completions_streaming | Text | 20 | Phi (4.3s) | Qwen3 (30.7s) | 14.2s | 15.2s | -6.3% |
| test_chess_position_e6 | Vision | 1 | pixtral (10.9s) | pixtral (10.9s) | 10.9s | 13.0s | -16.3% |
| test_contract_name_extraction | Vision | 1 | pixtral (12.4s) | pixtral (12.4s) | 12.4s | 12.6s | -2.3% |
| test_json_interactive_error_path | Text | 1 | DeepHermes (3.8s) | DeepHermes (3.8s) | 3.8s | 5.2s | -25.7% |
| test_large_image_support | Vision | 1 | pixtral (9.3s) | pixtral (9.3s) | 9.3s | 9.9s | -5.7% |
| test_mug_color_identification | Vision | 1 | pixtral (12.4s) | pixtral (12.4s) | 12.4s | 12.7s | -2.6% |
| test_pipe_from_list_json | Text | 1 | DeepHermes (65.3s) | DeepHermes (65.3s) | 65.3s | 66.3s | -1.6% |
| test_run_command | Text | 20 | Qwen2.5 (1.4s) | Mistral (13.9s) | 6.9s | 6.8s | +0.5% |
| test_run_json_output | Text | 20 | Qwen2.5 (1.4s) | Mistral (13.9s) | 6.9s | 6.8s | +0.6% |
| test_segment_metadata_optional | Audio | 2 | whisper (2.0s) | whisper (2.0s) | 2.0s | 1.9s | +1.8% |
| test_single_image_chat_completion | Vision | 3 | pixtral (10.2s) | BrokeC/Mistral (20.8s) | 16.0s | 12.4s | +29.0% |
| test_stdin_dash_appends_trailing_text | Text | 1 | DeepHermes (12.1s) | DeepHermes (12.1s) | 12.1s | 12.2s | -1.1% |
| test_streaming_graceful_degradation | Vision | 3 | pixtral (12.2s) | BrokeC/Mistral (21.5s) | 13.5s | 13.5s | +0.0% |
| test_text_request_still_works_on_vision_model | Text | 3 | pixtral (3.7s) | BrokeC/Mistral (5.8s) | 3.9s | 4.2s | -7.5% |
| test_transcribe_longer_audio_wav | Audio | 2 | whisper (2.0s) | whisper (2.0s) | 2.0s | 2.0s | +1.8% |
| test_transcribe_mp3_format | Audio | 2 | whisper (1.9s) | whisper (2.0s) | 1.9s | 1.9s | +0.5% |
| test_transcribe_short_audio_wav | Audio | 2 | whisper (2.2s) | whisper (3.1s) | 2.7s | 2.7s | -0.6% |
| test_transcription_endpoint_json | Audio | 2 | whisper (2.5s) | whisper (2.5s) | 2.5s | 2.5s | +0.0% |
| test_transcription_endpoint_mp3 | Audio | 2 | whisper (2.5s) | whisper (2.6s) | 2.5s | 2.5s | +0.5% |
| test_transcription_endpoint_rejects_oversized_audio | Audio | 2 | whisper (1.8s) | whisper (1.9s) | 1.9s | 1.8s | +1.1% |
| test_transcription_endpoint_text_format | Audio | 2 | whisper (2.5s) | whisper (2.5s) | 2.5s | 2.5s | -0.3% |
| test_transcription_endpoint_verbose_json | Audio | 2 | whisper (2.5s) | whisper (2.5s) | 2.5s | 2.5s | +0.4% |
| test_transcription_endpoint_with_language | Audio | 2 | whisper (2.5s) | whisper (2.5s) | 2.5s | 2.5s | -0.2% |


---

## GPU Analysis (v1.1)

Per-test GPU utilization correlated from memory monitoring data.

### Cold-Start Tests

First test per model (includes model loading overhead).

| Test | Model | GPU Dev | GPU Rnd | Samples |
|------|-------|--------:|--------:|--------:|
| test_transcribe_short_audio_wav | whisper-large-v3-turbo-4bit | 48.0% | 16.2% | 6 |
| test_transcribe_short_audio_wav | whisper-large-v3-turbo-8bit | 48.5% | 18.8% | 4 |
| test_run_command | DeepHermes-3-Mistral-24B-Preview-8bit | 14.4% | 12.5% | 20 |
| test_run_command | DeepSeek-R1-Distill-Llama-8B-4bit | 14.3% | 14.3% | 7 |
| test_run_command | EuroLLM-22B-Instruct-2512-mlx-8bit | 18.7% | 18.2% | 21 |
| test_run_command | Gabliterated-Qwen3-0.6B-float32 | 38.8% | 38.2% | 5 |
| test_run_command | Llama-3.2-3B-Instruct-4bit | 23.5% | 22.8% | 4 |
| test_run_command | Mistral-7B-Instruct-v0.2-4bit | 20.0% | 20.0% | 5 |
| test_run_command | Mistral-Small-3.2-24B-Instruct-2506-4bit | 0.0% | 0.0% | 12 |
| test_run_command | Mistral-Small-3.2-24B-Instruct-2506-8bit | 16.0% | 15.3% | 24 |
| test_run_command | Mistral-Small-Instruct-2409-4bit | 9.6% | 8.7% | 11 |
| test_run_command | Mixtral-8x7B-Instruct-v0.1-4bit | 22.2% | 12.9% | 22 |
| test_run_command | OpenCodeInterpreter-DS-33B-hf-4bit-mlx | 36.7% | 14.6% | 16 |
| test_run_command | Phi-3-mini-4k-instruct-4bit | 49.0% | 22.2% | 4 |
| test_run_command | Phi-3.5-mini-instruct-4bit | 67.3% | 15.0% | 3 |

*... and 10 more cold-start tests*

### High GPU Utilization Tests

Tests with >50% average GPU device utilization.

| Test | Model | GPU Dev | GPU Rnd | Pressure |
|------|-------|--------:|--------:|:--------:|
| test_chess_position_e6 | pixtral-12b-4bit | 83.6% | 82.5% | OK |
| test_contract_name_extraction | pixtral-12b-4bit | 81.6% | 80.1% | OK |
| test_mug_color_identification | pixtral-12b-4bit | 80.8% | 77.0% | OK |
| test_single_image_chat_completion | pixtral-12b-8bit | 71.1% | 67.1% | OK |
| test_streaming_graceful_degradation | pixtral-12b-4bit | 70.3% | 56.0% | OK |
| test_audio_output_not_empty | whisper-large-v3-turbo-4bit | 68.8% | 29.8% | OK |
| test_transcribe_longer_audio_wav | whisper-large-v3-turbo-8bit | 68.0% | 12.2% | OK |
| test_run_command | Phi-3.5-mini-instruct-4bit | 67.3% | 15.0% | OK |
| test_streaming_graceful_degradation | Mistral-Small-3.1-24B-Instruct-2503-4bit | 62.1% | 50.5% | OK |
| test_chart_axis_label_reading | pixtral-12b-4bit | 62.0% | 55.1% | OK |

### GPU Utilization Summary

```
Sample interval:      200ms
Correlated tests:     134
Cold-start tests:     25
GPU Device avg:        22.7%  (was  20.6%, Δ +2.1%)
GPU Device max:        83.6%  (was  84.0%, Δ -0.4%)
GPU Renderer avg:      15.6%  (was  17.6%, Δ -2.0%)
GPU Renderer max:      82.5%  (was  81.4%, Δ +1.1%)
```


---

## Files

- **Benchmark report:** `benchmarks/reports/2026-02-11-wet-benchmark-2.0.4-stable.jsonl`
- **Schema:** `benchmarks/schemas/report-v0.2.2.schema.json`
- **Memory data:** correlated
