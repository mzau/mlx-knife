# Benchmark Report v1.0: 2.0.4b3

**Date:** 2025-12-20
**Generated:** 2025-12-20T14:43:01.786689+00:00
**Generator:** generate_benchmark_report.py v1.0
**Hardware:** Mac14,13, 12 cores

---

## Input Files

- **Primary:** `benchmarks/reports/2025-12-20-v2.0.4b3-2nd_0.2.0_schema.jsonl`
- **Schema:** v0.2.0

---

## Executive Summary

**Tests:** 162 total (141 passed, 21 skipped)
**Duration:** 1169.3s (19.5 min)
**Quality:** 100.0% clean (162/162)
**Models:** 22 tested

✅ **System Health:** All tests clean (0 MB swap, 0 zombies)

---

## Test Summary

```
Total tests:       162
Passed:            141
  With model:      84
  Infrastructure:  57
Skipped:           21
Duration:          1169.3s (19.5 min)
```

---

## System Health

```
Swap (MB):         min=0, max=0, avg=0.0
RAM free (GB):     min=0.0, max=46.7, avg=19.0
Zombies:           min=0, max=0

Quality Flags:
  Clean:           162/162 (100.0%)
  Degraded (swap): 0
  Degraded (zombies): 0
```

---

## Per-Model Statistics

```
Model                                              Size     Tests  Time       RAM (GB)            
================================================== ======== ====== ========== ====================
Mistral-Small-3.2-24B-Instruct-2506-8bit             23.3GB 4         102.2s  21.4-25.9           
Qwen3-Coder-30B-A3B-Instruct-6bit-DWQ-lr9e-8         24.9GB 4          97.5s  21.6-26.8           
Mixtral-8x7B-Instruct-v0.1-4bit                      24.5GB 4          96.9s  2.7-26.4            
DeepHermes-3-Mistral-24B-Preview-8bit                23.3GB 4          63.0s  0.0-24.6            
OpenCodeInterpreter-DS-33B-hf-4bit-mlx               17.8GB 4          62.9s  17.9-33.0           
Qwen3-32B-4bit                                       17.2GB 4          48.7s  17.1-20.3           
Klear-46B-A2.5B-Instruct-3bit                        18.9GB 4          40.7s  18.9-19.9           
MiMo-VL-7B-RL-bf16                                   15.5GB 4          38.9s  14.6-19.7           
gpt-oss-20b-MXFP4-Q8                                 11.3GB 4          36.6s  14.2-36.4           
Qwen3-30B-A3B-Instruct-2507-4bit                     16.0GB 4          34.2s  16.2-23.7           
Qwen3-Coder-30B-A3B-Instruct-4bit                    16.0GB 4          33.1s  16.3-17.1           
Mistral-Small-3.2-24B-Instruct-2506-4bit             12.4GB 4          32.9s  13.0-16.9           
Mistral-Small-Instruct-2409-4bit                     11.7GB 4          27.6s  12.9-26.2           
Qwen2.5-Coder-7B-Instruct-8bit                        7.5GB 4          19.9s  8.5-31.2            
DeepSeek-R1-Distill-Llama-8B-4bit                     4.2GB 4          19.7s  20.2-37.6           
pixtral-12b-8bit                                     12.6GB 2          15.5s  14.3-14.4           
Mistral-7B-Instruct-v0.2-4bit                         4.0GB 4          14.1s  8.9-26.2            
Gabliterated-Qwen3-0.6B-float32                       2.2GB 4          12.7s  16.1-37.3           
Phi-3-mini-4k-instruct-4bit                           2.0GB 4          11.5s  14.6-46.7           
Phi-3.5-mini-instruct-4bit                            2.0GB 4          10.2s  12.6-44.6           
Qwen2.5-0.5B-Instruct-4bit                            0.3GB 4           9.2s  13.8-46.0           
Llama-3.2-11B-Vision-Instruct-4bit                    5.6GB 2           8.9s  10.3-12.1           
```

### Model Categories

```
LARGE MODELS (≥20 GB):    4 models
  Avg size:               24.0 GB
  Avg test time:          22.5s
  Avg min RAM:            11.5 GB

MEDIUM MODELS (10-20 GB): 10 models
  Avg size:               14.9 GB
  Avg test time:          9.7s
  Avg min RAM:            15.6 GB

SMALL MODELS (<10 GB):    8 models
  Avg size:               3.5 GB
  Avg test time:          3.6s
  Avg min RAM:            13.1 GB
```

---

## Per-Test Statistics

Shows performance range across models for each test.

```
Test Name                                          Models  Fastest                   Slowest                   Med Time
================================================== ======= ========================= ========================= ========
test_run_command                                   22      Qwen2.5 (1.2s)            DeepHermes (22.1s)        7.1s    
test_run_json_output                               22      Qwen2.5 (1.2s)            Mistral (13.3s)           7.1s    
test_chat_completions_batch                        20      Phi (3.3s)                Mixtral (30.9s)           8.7s    
test_chat_completions_streaming                    20      Qwen2.5 (3.4s)            Qwen3 (51.3s)             10.6s   
```


---

## Files

- **Benchmark report:** `benchmarks/reports/2025-12-20-v2.0.4b3-2nd_0.2.0_schema.jsonl`
- **Schema:** `benchmarks/schemas/report-v0.2.0.schema.json`
