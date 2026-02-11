# Benchmark Handbook

Step-by-step guide for running benchmarks and generating reports.

## Quick Start

```bash
# 1. Run E2E tests with report output
pytest -m live_e2e tests_2.0/live/ \
  --report-output benchmarks/reports/$(date +%Y-%m-%d)-v2.0.4b3.jsonl

# 2. Generate analysis report
python benchmarks/generate_benchmark_report.py

# 3. View results
cat benchmarks/reports/BENCHMARK-*.md
```

---

## Running Benchmarks

### Basic Test Run

```bash
# Run all E2E tests, output to JSONL
pytest -m live_e2e tests_2.0/live/ \
  --report-output benchmarks/reports/$(date +%Y-%m-%d)-v2.0.4b3.jsonl
```

### With Custom HuggingFace Cache

```bash
HF_HOME=/path/to/huggingface/cache \
  pytest -m live_e2e tests_2.0/live/ -v \
  --report-output benchmarks/reports/2025-12-20-v2.0.4b3.jsonl
```

### With Memory Monitoring

```bash
# Run memmon in parallel to capture memory profile
python benchmarks/tools/memmon.py \
  --output benchmarks/reports/2025-12-20-memory.jsonl \
  -- pytest -m live_e2e tests_2.0/live/ \
     --report-output benchmarks/reports/2025-12-20-v2.0.4b3.jsonl
```

---

## Generating Reports

### Auto-Detect Latest JSONL

```bash
python benchmarks/generate_benchmark_report.py
# → Finds most recent .jsonl in benchmarks/reports/
# → Outputs: BENCHMARK-<template>-<version>-<date>.md
```

### Explicit Input File

```bash
python benchmarks/generate_benchmark_report.py \
  benchmarks/reports/2025-12-20-v2.0.4b3.jsonl
```

### With Comparison (Regression Detection)

```bash
python benchmarks/generate_benchmark_report.py \
  benchmarks/reports/2025-12-20-new.jsonl \
  --compare benchmarks/reports/2025-12-19-old.jsonl
```

Output includes:
- Duration change (e.g., 20.5 min → 19.7 min, -3.8%)
- Per-model changes with Old/Δ/Change columns
- Per-test median time changes
- Status indicators: ⚠️ (>5% slower), ✅ (>1% faster)

### Custom Output Location

```bash
python benchmarks/generate_benchmark_report.py \
  --output /tmp/my-report.md \
  benchmarks/reports/2025-12-20-v2.0.4b3.jsonl
```

---

## Memory Monitoring

### Standalone Monitor (Fixed Duration)

```bash
python benchmarks/tools/memmon.py \
  --duration 60 \
  --interval 200 \
  --output memory.jsonl
```

### Wrap Any Command

```bash
python benchmarks/tools/memmon.py \
  --output memory.jsonl \
  -- ./my-benchmark-script.sh
```

### Output Format

```jsonl
{"ts": 1734567890.1, "ram_free_gb": 45.2, "swap_used_mb": 0, "elapsed_s": 0.2}
{"ts": 1734567890.3, "ram_free_gb": 42.1, "swap_used_mb": 0, "elapsed_s": 0.4}
...
{"summary": {"ram_free_min_gb": 21.3, "ram_free_max_gb": 45.2, "swap_max_mb": 0}}
```

### Correlating with Test Results

Memory samples can be correlated with test results via timestamps:

```python
# Test entry has: timestamp (end time), duration
# Calculate: started_at = timestamp - duration

test_start = parse_iso(entry["timestamp"]) - entry["duration"]
test_end = parse_iso(entry["timestamp"])

# Find matching memory samples
matching = [s for s in samples if test_start <= s["ts"] <= test_end]
```

---

## Validating Reports

### Validate Against Current Schema

```bash
python benchmarks/validate_reports.py benchmarks/reports/*.jsonl
```

### Validate Specific File

```bash
python benchmarks/validate_reports.py benchmarks/reports/2025-12-20-v2.0.4b3.jsonl
```

---

## Schema Reference

### Current Schema: v0.2.2

Required fields:
```json
{
  "schema_version": "0.2.2",
  "timestamp": "2025-12-20T02:26:10.722510+00:00",
  "mlx_knife_version": "2.0.4-beta.3",
  "test": "tests_2.0/live/test_cli_e2e.py::test_run_command[discovered_00]",
  "outcome": "passed",
  "duration": 12.3
}
```

Optional sections:
```json
{
  "model": {
    "id": "mlx-community/Qwen3-32B-4bit",
    "size_gb": 17.2,
    "family": "qwen3"
  },
  "system": {
    "hardware_profile": {
      "model": "Mac14,13",
      "cores_physical": 12
    }
  },
  "system_health": {
    "swap_used_mb": 0,
    "ram_free_gb": 45.2,
    "zombie_processes": 0,
    "quality_flags": ["clean"]
  }
}
```

### Quality Flags

| Flag | Meaning | Threshold |
|------|---------|-----------|
| `clean` | Test ran without issues | swap=0, zombies=0 |
| `degraded_swap` | Memory pressure detected | swap > 100 MB |
| `degraded_zombies` | Zombie processes present | zombies > 0 |

---

## Best Practices

1. **File Naming:** Use `YYYY-MM-DD-vX.Y.Z.jsonl` format
2. **Append Only:** Never edit existing reports (historical data)
3. **Commit Reports:** Reports are git-tracked for trend analysis
4. **Clean State:** Reboot before important benchmark runs
5. **Close Apps:** Minimize background processes during tests
6. **Multiple Runs:** Run 2-3 times, compare for consistency

---

## Troubleshooting

### "No JSONL files found"

```bash
# Check if reports exist
ls -la benchmarks/reports/*.jsonl

# Run tests with output
pytest -m live_e2e tests_2.0/live/ --report-output benchmarks/reports/test.jsonl
```

### Schema Validation Fails

```bash
# Check schema version in file
head -1 benchmarks/reports/file.jsonl | jq .schema_version

# Validate manually
python -c "
import json
from jsonschema import validate
with open('benchmarks/schemas/report-current.schema.json') as f:
    schema = json.load(f)
with open('benchmarks/reports/file.jsonl') as f:
    for line in f:
        validate(json.loads(line), schema)
print('OK')
"
```

### Comparison Shows "N/A"

Model not found in comparison file. Check:
- Same models tested in both runs?
- Model ID spelling matches exactly?

---

## Planned: Canonical Paths (v0.3.0)

**Status:** Planned for Schema v0.3.0

### Directory Structure

```
benchmarks/
├── runs/              # Raw JSONL data (new)
│   ├── 2026-02-06_wet-benchmark.jsonl
│   └── 2026-02-06_wet-memory.jsonl
├── reports/           # Generated Markdown only
│   └── BENCHMARK-*.md
├── schemas/
└── tools/
```

### Naming Convention

```
runs/<YYYY-MM-DD>_<tool-name>.jsonl
```

Examples:
- `runs/2026-02-06_wet-benchmark.jsonl`
- `runs/2026-02-06_wet-memory.jsonl`

### Tool Defaults

| Tool | Input Default | Output Default |
|------|---------------|----------------|
| `generate_report.py` | `runs/*_benchmark.jsonl` (latest) | `reports/BENCHMARK-*.md` |
| `memmon.py` | — | `runs/<today>_<name>-memory.jsonl` |
| `memplot.py` | `runs/*_memory.jsonl` (latest) | `reports/*-timeline.html` |

### New Flags

```bash
# Compare with previous run (auto-detected)
python benchmarks/tools/generate_report.py --compare-with-previous wet-benchmark

# Explicit paths (unchanged)
python benchmarks/tools/generate_report.py \
  runs/2026-02-06_wet-benchmark.jsonl \
  --compare runs/2026-02-05_wet-benchmark.jsonl
```

### Migration

1. Create `runs/` directory
2. Move existing `reports/*.jsonl` → `runs/`
3. Rename to canonical format
4. Update tool defaults

---

## Future: Phase 1 (mlxk-benchmark)

Phase 1 will introduce a standalone benchmark package:

```bash
pip install mlxk-benchmark
mlx-benchmark --model llama-3.2-3b --contribute
```

No pytest, no fixtures, no conftest.py - just simple CLI for community contributions.

See `schemas/LEARNINGS-FOR-v1.0.md` for design notes.
