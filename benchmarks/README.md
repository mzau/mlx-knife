# MLX Knife Benchmarks

**Status:** Phase 0 - Organic Data Collection (WIP)

## What's Here?

This directory contains benchmark infrastructure for mlx-knife:
- Empirical performance and compatibility data from E2E tests
- Tools for analysis and visualization
- Schema definitions for structured reports

## Directory Structure

```
benchmarks/
├── reports/                    # JSONL test reports + Markdown analyses
│   ├── 2025-12-20-v2.0.4b3.jsonl   # Raw data (one file per test run)
│   └── BENCHMARK-*.md               # Generated analysis reports
├── schemas/                    # JSON Schema definitions
│   ├── report-v0.1.schema.json      # legacy schema
│   ├── report-v0.2.2.schema.json    # Current schema
│   └── report-current.schema.json  # Symlink → current schema
├── tools/                      # Standalone tools
│   ├── memmon.py                   # Memory monitor (background sampling)
│   ├── memplot.py                  # Memory timeline visualizer
│   ├── chronos_gauge.py            # mlxk serve vs. mlx_lm.server (mlx-chronos)
│   └── stream_overhead.py          # Streamed vs. unstreamed cost per token
├── generate_benchmark_report.py    # Report generator (Template v1.1)
├── validate_reports.py             # Schema validation
├── README.md                       # ← You are here
└── TESTING.md                      # Benchmark handbook (How-To)
```

## Tools

| Tool | Purpose |
|------|---------|
| `generate_benchmark_report.py` | JSONL → Markdown report (Template v1.1) |
| `validate_reports.py` | Schema validation of JSONL files |
| `tools/memmon.py` | Memory + CPU + GPU monitoring (200ms sampling) |
| `tools/memplot.py` | Interactive 3-row timeline (Memory/CPU/GPU, HTML) |
| `tools/chronos_gauge.py` | `mlxk serve` against `mlx_lm.server`, measured by mlx-chronos (TTFT, throughput, RAM) — [below](#server-overhead-gauge-chronos_gaugepy) |
| `tools/stream_overhead.py` | Streamed against unstreamed generation on one runner: cost per token, as a ratio — nothing to install |

## Schema

**Current:** v0.2.2 (Phase 0 - Test Infrastructure)

| Version | Release | Content |
|---------|---------|---------|
| v0.1.0 | 2.0.3 | Minimal: test, outcome, duration, model |
| v0.2.0 | 2.0.4-beta.3 | + hardware_profile, system_health, quality_flags |
| v0.2.1 | 2.0.4-beta.7 | + inference_modality (vision/text/audio) |
| v0.2.2 | 2.0.4-beta.9 | + test_start_ts, test_end_ts (precise timing) |
| v1.0.0 | Future | Model benchmarks (mlxk-benchmark package) |

**Schema Strategy:** No v0.3.x planned. v0.2.x → v1.0.0 directly.
- v0.x = Test infrastructure ("Was the test run clean?")
- v1.x = Model benchmarks ("How good is the model?")

## Recent Reports

Latest baseline reports are in `reports/` directory:
- Pattern: `BENCHMARK-<template>-<version>-<date>-*.md`
- Hardware: Mac14,13 (M2 Max, 64 GB)
- Test suite: ~167 tests (Vision + Text + Audio E2E)
- Quality target: 100% clean (0 MB swap, 0 zombies)

## Phase 0 Goals

1. **Collect data organically** from E2E tests
2. **No perfect schema** - schema evolves with data
3. **Git-tracked reports** - historical trends
4. **Foundation for Phase 1** - mlxk-benchmark package

## Memory Timeline Visualization

**Tool:** `tools/memplot.py` - 3-row interactive plot (Memory / CPU / GPU)

### Quick Start

```bash
# Collect data (memmon runs in background)
python benchmarks/tools/memmon.py --output memory.jsonl -- \
  pytest -m live_e2e tests_2.0/live/ --report-output benchmark.jsonl

# Generate interactive HTML with test + model markers
python benchmarks/tools/memplot.py memory.jsonl benchmark.jsonl -o timeline.html
```

**Note:** `benchmark.jsonl` adds test markers showing test name + model name - essential for plot navigation!

### Visual Legend

#### Row 1: Memory (RAM Free GB)

**Blue line:** RAM free over time (GB)

**RAM line marker colors** (per-point coloring based on available RAM):
- 🟢 **Green:** ≥32 GB free - healthy
- 🟠 **Orange:** 16-32 GB free - warning zone
- 🔴 **Red:** <16 GB free - critical

**Memory pressure background** (semi-transparent overlays):
- 🟡 **Yellow `rgba(255, 204, 0, 0.15)`:** WARN level - system preparing to swap
- 🔴 **Red `rgba(255, 59, 48, 0.15)`:** CRITICAL level - system actively swapping

**Red line (right axis):** Swap Used (MB) - only visible when > 0

#### Row 2: CPU Load

**Load Average (purple):** 1-minute load average
**User (green fill):** User space CPU %
**System (red fill):** Kernel CPU % (stacked on User)

#### Row 3: GPU Utilization (Apple Silicon)

**Device (orange solid):** Overall GPU busy %
**Renderer (green fill):** 3D rendering cores %
**Tiler (purple dashed):** Geometry processing %

**Source:** `ioreg` PerformanceStatistics (no sudo required)

#### Background Rectangles: Test Regions (All Rows)

**Gray (rgba(200, 200, 200, 0.3)):**
- Model tests that load an LLM model
- Example: `test_run_command[text_00]`, `test_chat_completion[vision_01]`
- **Meaning:** Model is loaded in RAM during this time

**Light Blue (rgba(173, 216, 230, 0.2)):**
- Infrastructure tests without model
- Example: `test_portfolio_discovery`, `test_health_check`
- **Meaning:** No model loaded, only test infrastructure active

⚠️ **Known limitation (v0.2.2):** Server tests appear as "light blue" even when loading models (LocalServer fixture doesn't record model metadata). Recognizable by: high RAM usage + long duration in blue region. Example: `test_text_request_still_works_on_vision_model` (57 GB used, 16s duration).

#### Labels (All Rows)

**Top (90° rotated, black):**
- Model names at each model switch
- Example: `DeepHermes-3-Mistral`, `pixtral-12b-8bit`
- Position: Left-aligned with test start

**Bottom (90° rotated, gray):**
- Test names for each test (model + infrastructure)
- Example: `test_run_command`, `test_chat_completion`
- Position: Left-aligned with test start

**Vertical helper lines:**
- Thin gray lines at each test start
- Help correlate labels with timeline

#### Secondary Y-Axis: Swap Used (MB)

**Red line (right axis):**
- Only visible when swap > 0 MB
- **Meaning:** System paging RAM to SSD → performance loss
- **Normal:** 0 MB
- **Problematic:** >100 MB

### Interpretation Patterns

**Typical model load:**
```
Pattern: RAM Free drops suddenly (e.g., 52 GB → 28 GB)
Duration: 2-5 seconds
Color: Gray rectangle begins
Label: Model name appears at top
→ Model loaded into RAM (24 GB)
```

**Typical model unload:**
```
Pattern: RAM Free rises suddenly (e.g., 28 GB → 52 GB)
Duration: <1 second
Color: Gray rectangle ends (or switches to next)
Label: New model name (or none)
→ Model removed from RAM
```

**Memory pressure without swap:**
```
Pattern: Yellow/Red background WITHOUT swap line
RAM Free: Still >10 GB
→ macOS preparing to swap, not yet active
→ Often during large model loads (temporary)
```

**Memory pressure with swap:**
```
Pattern: Red background + Red swap line rises
RAM Free: <10 GB
Swap: >100 MB
→ System actually at limit
→ Performance significantly worse
→ Typical: Multiple large models in short time
```

**Infrastructure test with high RAM usage:**
```
Pattern: Light blue rectangle + RAM drops significantly (>20 GB)
Duration: >10 seconds
Example: 57 GB used in test_text_request_still_works_on_vision_model
→ ⚠️ Schema bug: Server test loads model but "model": null
→ Should be gray, not light blue
→ Fix: v1.0 schema with log parsing
```

### Data Sources

**RAM Free:**
- Source: `vm_stat` (macOS native)
- Calculation: `(free + inactive + purgeable + speculative) * page_size / 1e9`
- Sample rate: 200ms (5 samples/second)

**Memory Pressure:**
- Source: `sysctl kern.memorystatus_vm_pressure_level`
- Values: 1=NORMAL, 2=WARN, 4=CRITICAL
- Sample rate: 200ms (synchronized with RAM)

**Swap Used:**
- Source: `sysctl vm.swapusage`
- Unit: MB
- Sample rate: 200ms

**Test Metadata:**
- Source: Benchmark JSONL (pytest-json-report format)
- Fields: `timestamp`, `duration`, `test`, `model` (optional), `outcome`
- Correlation: ISO timestamp → Unix timestamp → elapsed seconds

### Known Limitations (v0.2.0)

1. **Model load/unload events missing**
   - Gray regions show "test with model", not "model is loaded"
   - Pytest runs through ALL models 4x → each model loaded/unloaded 4x
   - Regions overlap visually though sequential
   - **Fix planned:** v1.0 schema with explicit events

2. **Server tests without model attribution**
   - Server tests (LocalServer fixture) load models internally
   - Appear as "infrastructure" (light blue) instead of "model" (gray)
   - Recognizable: High RAM + long duration in blue region
   - **Fix planned:** Log parsing in v0.3.0/v1.0

3. **Dense test sequences**
   - Tests shorter than 200ms sample rate → no coloring
   - Typical: Fast infrastructure tests (<100ms)
   - **Workaround:** Test labels show all tests

4. **Label overlap**
   - Many tests in short time (>10 tests/min)
   - Labels may overlap (90° rotated)
   - **Mitigation:** Zoom for detailed view
   - **Future:** Adaptive label density or collapsing

### Interactive Features

- **Zoom & Pan:** Mouse wheel (vertical), Shift+wheel (horizontal), click+drag
- **Range Slider:** Quick navigation in long (>20 min) timelines
- **Hover:** X-axis unified mode shows all values at same time

### Future Extensions (Ideas)

**For plot:**
- [ ] Embedded legend in plot (not external file)
- [ ] Toggle show/hide infrastructure tests
- [ ] Hover shows full test names (not truncated)
- [ ] Color-blind mode (alternative palette)

**For schema v1.0:**
- [ ] Model load/unload events → precise "in RAM" regions
- [ ] Log parsing for server tests → correct attribution
- [ ] GPU activity (Metal performance)
- [ ] Net T/S (tokens/second, pure inference)

**For analysis:**
- [ ] Automatic anomaly detection (memory leaks, zombies)
- [ ] Per-model memory profiling (min/max/avg RAM)
- [ ] Scheduling optimization (avoid model-switch overlap)

---

## Server Overhead Gauge (`chronos_gauge.py`)

**Tool:** `tools/chronos_gauge.py` - what does mlxk's own server shell cost?

For one text model it starts `mlx_lm.server` and `mlxk serve` in turn on the same port, runs
the same [mlx-chronos](https://github.com/igurss/mlx-chronos) protocol against each and prints
both results side by side with their ratio: time to first token (cold and cached), request and
decode throughput, system RAM peak. The reference is measured in the same session, so **the
ratio is the number that travels** - absolute values depend on machine and thermal state.
Output quality is not measured.

### Two environments, and which one runs what

mlx-chronos needs its own venv and is only ever called over its command line, so it never
enters the development or test environment and installing it changes no test result.

| | lives in | started by |
|---|---|---|
| `mlx-chronos` | `venv-chronos/` | the gauge, by absolute path |
| `mlxk serve`, `mlx_lm.server`, the gauge itself | the development environment | you |

⚠️ **Do not activate `venv-chronos` to run the gauge.** It starts both servers from the venv of
the interpreter it runs in, and `venv-chronos` carries neither `mlxk` nor MLX - the run dies with
`FileNotFoundError: .../venv-chronos/bin/mlxk`.

### Setup (once, with the development environment activated)

```bash
python -m venv venv-chronos          # the development environment's interpreter, see below
venv-chronos/bin/pip install "mlx-chronos[thermal]==0.4.1"
# same mlx-lm version as the development environment - chronos records it but never imports it,
# so --no-deps pulls in no MLX:
venv-chronos/bin/pip install --no-deps \
  "mlx-lm==$(python -c 'import importlib.metadata as m; print(m.version("mlx-lm"))')"
```

**Which interpreter:** mlx-chronos needs Python **>= 3.10**. Build the venv from the development
environment (or name a version explicitly, `python3.11 -m venv venv-chronos`) - a bare `python3`
on macOS is still the system's 3.9, and the pip that ships with it reports the version floor as

```
ERROR: Could not find a version that satisfies the requirement mlx-chronos[thermal]==0.4.1 (from versions: none)
```

which reads as if the package did not exist. It does; every release of it needs 3.10.

The gauge compares both mlx-lm versions before it starts and prints the exact command when they
differ. A venv elsewhere goes into `$MLXK_CHRONOS_BIN` or `--chronos`.

### Run

```bash
source <development environment>/bin/activate
python benchmarks/tools/chronos_gauge.py --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
```

Activated, because macOS has no bare `python` on the PATH and `python3` is the system's 3.9
without mlxk. With `HF_HOME` pointing at the cache that holds the model; both servers run with
`HF_HUB_OFFLINE=1`. Results go to `benchmarks/reports/chronos/<timestamp>-<model>/` - chronos
JSON per server, server and chronos logs, `summary.json`, `summary.md` - and stay local.
Options: `--port` (default 8080), `--profile baseline|sustained`, `--trials`, `--pause`,
`--max-gpu-busy`, `--chronos`.

### How long it takes, and how to tell it apart from a hang

A `baseline` run sends **18 requests per server**: 2 warm-ups, 5 cold and 5 cached TTFT trials at
one token each, one cache priming call, and 5 throughput trials of up to 100 tokens. On a small
model that is a minute or two per server; on a large one it is however long that model needs for
~500 tokens plus 26 prompts, which can be far more than the model's size suggests.

Nothing here is a timeout on the measurement: `--pause` (default 60 s) sits **between** the two
servers and cuts off neither, and `--timeout` (default 600 s) covers only how long a server may
take to become ready. A run that looks stuck is best checked in its session directory rather than
interrupted - `<kind>.chronos.log` names the trial in flight and `<kind>.server.log` shows every
request the server answered.

`Engine PID not found; diagnostic engine RSS will use system RAM peak fallback` in the mlxk log is
expected: chronos looks for a process named `mlx_lm.server`, which the mlxk side does not have.

### What it refuses, and why

- **A model mlxk cannot resolve.** `--model` takes what mlxk takes: a cached `org/name`, or a
  workspace model by its bare name with `MLXK_WORKSPACE_HOME` set. `mlxk serve` gets that spec
  and resolves it itself; `mlx_lm` knows no workspaces, so the reference server is handed the
  directory `mlxk show` resolved it to (`summary.json` keeps it as `resolved`).
- **A model that is not text-only, or one mlxk would not run.** It asks `mlxk show --json` first.
  Vision and audio checkpoints are rejected because chronos measures text generation; a
  checkpoint mlxk reports as not runnable is rejected because `mlx_lm.server` would execute its
  `model_file` unconditionally.
- **A busy GPU.** It samples `Device Utilization %` through `ioreg` — the same counter
  `tools/memmon.py` has sampled since 2.0.4, here as a gate before the run rather than as a
  timeline during it — before each server and stops
  above `--max-gpu-busy` (default 10 %). A game or a video stream in a browser tab slows every
  step that waits on the GPU, and the run would measure that application instead.
- **A port that is already in use.**

### ⚠️ Never submit these results

Both runs carry the `mlx-lm` engine label, so neither belongs on the mlx-chronos leaderboard.

How to read the table - what `token counts` means, why `repetition_penalty` makes the two texts
differ, and why the RAM figure is system-wide - is in
[TESTING-DETAILS.md](../TESTING-DETAILS.md#server-overhead-gauge-mlx-chronos).

---

## Roadmap

| Phase | Release | Description |
|-------|---------|-------------|
| **Phase 0** | 2.0.3-2.0.4 | Organic Data Collection ✅ |
| Phase 1 | 2.1+ | `mlxk-benchmark` package (separate tool) |
| Phase 2 | 2.2+ | Report aggregation, hardware correlation |
| Phase 3 | 2.3+ | Public database, community contributions |

## Further Documentation

- **[TESTING.md](TESTING.md)** - Benchmark handbook (How-To)

Two further documents exist but are not published: `schemas/LEARNINGS-FOR-v1.0.md`
(learnings for Phase 1) and ADR-013 *Community Model Quality Database* (the architecture
vision). Ask if you are interested — that is what would move this forward.
