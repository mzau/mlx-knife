#!/usr/bin/env python3
"""Generate benchmark analysis report from JSONL test data.

Reads JSONL benchmark reports and generates structured Markdown analysis.

Usage:
    # Auto-detect latest JSONL
    python benchmarks/generate_benchmark_report.py

    # Explicit file
    python benchmarks/generate_benchmark_report.py benchmarks/reports/2025-12-20-v2.0.4b3.jsonl

    # With comparison
    python benchmarks/generate_benchmark_report.py new.jsonl --compare old.jsonl

    # With GPU/memory correlation (v1.1)
    python benchmarks/generate_benchmark_report.py benchmark.jsonl --memory memory.jsonl

    # Full comparison with GPU analysis
    python benchmarks/generate_benchmark_report.py new-benchmark.jsonl \\
        --memory new-memory.jsonl \\
        --compare old-benchmark.jsonl

    Note: When using --compare, the memory file for the comparison run is
    auto-detected if it follows the naming convention (benchmark → memory).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    import jsonschema
except ImportError:
    print("Error: jsonschema not installed. Install with: pip install jsonschema")
    sys.exit(1)


# Template version
TEMPLATE_VERSION = "1.1"
REPORTS_DIR = Path("benchmarks/reports")
SCHEMA_PATH = Path("benchmarks/schemas/report-current.schema.json")


def correlate_memory_to_tests(
    benchmark_data: List[dict],
    memory_data: List[dict]
) -> Dict[str, Dict]:
    """Correlate memory samples to individual tests by timestamp.

    For each test, finds all memory samples that fall within [test_start, test_end]
    and aggregates GPU/memory statistics.

    Args:
        benchmark_data: List of benchmark test results (with timestamp, duration)
        memory_data: List of memory samples (with ts Unix timestamp)

    Returns:
        Dict mapping test identifier to aggregated GPU/memory stats
    """
    # Filter out non-test entries (e.g., summary lines)
    tests = [e for e in benchmark_data if "timestamp" in e and "duration" in e]

    # Sort memory samples by timestamp
    sorted_memory = sorted(memory_data, key=lambda m: m.get("ts", 0))

    correlations = {}

    for test in tests:
        # Parse test timestamp to Unix
        try:
            test_dt = datetime.fromisoformat(test["timestamp"])
            test_start = test_dt.timestamp()
            test_end = test_start + test["duration"]
        except (ValueError, KeyError):
            continue

        # Find memory samples within test window
        # Using binary search would be faster, but linear is fine for ~3000 samples
        samples_in_window = [
            m for m in sorted_memory
            if test_start <= m.get("ts", 0) <= test_end
        ]

        if not samples_in_window:
            continue

        # Aggregate GPU stats
        gpu_device = [m.get("gpu_device_util", 0) for m in samples_in_window]
        gpu_renderer = [m.get("gpu_renderer_util", 0) for m in samples_in_window]
        gpu_tiler = [m.get("gpu_tiler_util", 0) for m in samples_in_window]
        cpu_user = [m.get("cpu_user", 0) for m in samples_in_window]
        cpu_sys = [m.get("cpu_sys", 0) for m in samples_in_window]
        mem_pressure = [m.get("memory_pressure", 1) for m in samples_in_window]

        # Create unique test identifier
        test_id = test.get("test", "unknown")

        correlations[test_id] = {
            "sample_count": len(samples_in_window),
            "gpu_device_avg": round(sum(gpu_device) / len(gpu_device), 1) if gpu_device else 0,
            "gpu_device_max": max(gpu_device) if gpu_device else 0,
            "gpu_renderer_avg": round(sum(gpu_renderer) / len(gpu_renderer), 1) if gpu_renderer else 0,
            "gpu_renderer_max": max(gpu_renderer) if gpu_renderer else 0,
            "gpu_tiler_avg": round(sum(gpu_tiler) / len(gpu_tiler), 1) if gpu_tiler else 0,
            "cpu_user_avg": round(sum(cpu_user) / len(cpu_user), 1) if cpu_user else 0,
            "cpu_sys_avg": round(sum(cpu_sys) / len(cpu_sys), 1) if cpu_sys else 0,
            "memory_pressure_max": max(mem_pressure) if mem_pressure else 1,
        }

    return correlations


def find_memory_file_for_benchmark(benchmark_file: Path) -> Optional[Path]:
    """Auto-detect memory JSONL file for a benchmark file.

    Convention: benchmark file "2026-02-06-wet-benchmark-1.jsonl"
                → memory file "2026-02-06-wet-memory-1.jsonl"

    Args:
        benchmark_file: Path to benchmark JSONL file

    Returns:
        Path to memory file if found, None otherwise
    """
    # Replace "benchmark" with "memory" in filename
    memory_name = benchmark_file.name.replace("benchmark", "memory")
    memory_path = benchmark_file.parent / memory_name

    if memory_path.exists():
        return memory_path
    return None


def detect_cold_starts(benchmark_data: List[dict]) -> Dict[str, bool]:
    """Detect which tests are cold starts (first test for a given model).

    Cold starts typically have higher latency due to model loading,
    cache warming, and JIT compilation.

    A cold start is the FIRST test for a specific model, regardless of
    test name or parametrization. Subsequent tests on the same model
    benefit from cached weights and warm GPU state.

    Args:
        benchmark_data: List of benchmark test results

    Returns:
        Dict mapping test identifier to is_cold_start boolean
    """
    # Track first occurrence of each model
    seen_models = set()
    cold_starts = {}

    # Sort by timestamp to ensure correct ordering
    sorted_tests = sorted(
        [e for e in benchmark_data if "timestamp" in e and "model" in e],
        key=lambda e: e.get("timestamp", "")
    )

    for test in sorted_tests:
        test_id = test.get("test", "unknown")
        model_id = test.get("model", {}).get("id", "unknown")

        # Normalize model_id: some tests report "mlx-community/foo", others just "foo"
        # This is a data inconsistency in test fixtures, not different models
        normalized_model_id = model_id.replace("mlx-community/", "")

        if normalized_model_id not in seen_models:
            cold_starts[test_id] = True
            seen_models.add(normalized_model_id)
        else:
            cold_starts[test_id] = False

    return cold_starts


def load_schema() -> dict:
    """Load current JSON schema."""
    if not SCHEMA_PATH.exists():
        print(f"❌ Schema not found: {SCHEMA_PATH}")
        sys.exit(1)

    with open(SCHEMA_PATH) as f:
        return json.load(f)


def is_memmon_jsonl(data: List[dict]) -> bool:
    """Detect if JSONL is memmon output (memory samples) vs benchmark results.

    memmon JSONL has: ram_free_gb, swap_used_mb, elapsed_s (no schema_version)
    benchmark JSONL has: schema_version, outcome, timestamp
    """
    if not data:
        return False

    first_entry = data[0]
    # Check for memmon-specific fields
    has_memmon_fields = "ram_free_gb" in first_entry and "elapsed_s" in first_entry
    # Check for benchmark-specific fields
    has_benchmark_fields = "schema_version" in first_entry or "outcome" in first_entry

    return has_memmon_fields and not has_benchmark_fields


def validate_jsonl(data: List[dict], schema: dict, filepath: Path) -> bool:
    """Validate JSONL data against schema.

    Skips validation for memmon JSONL files (memory monitoring data).
    """
    # Skip validation for memmon data
    if is_memmon_jsonl(data):
        print(f"ℹ️  Skipping validation for memmon data: {filepath}")
        return True

    errors = []
    for i, entry in enumerate(data, 1):
        try:
            jsonschema.validate(instance=entry, schema=schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Line {i}: {e.message}")

    if errors:
        print(f"❌ Validation failed for {filepath}")
        for error in errors[:5]:  # Show first 5 errors
            print(f"   {error}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more errors")
        return False

    return True


def load_jsonl(filepath: Path) -> List[dict]:
    """Load JSONL file."""
    data = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def find_latest_jsonl() -> Optional[Path]:
    """Find the most recent JSONL file in reports directory."""
    if not REPORTS_DIR.exists():
        return None

    jsonl_files = sorted(REPORTS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return jsonl_files[0] if jsonl_files else None


def extract_version_from_filename(filepath: Path) -> Optional[str]:
    """Extract version string from filename like '2025-12-20-v2.0.4b3.jsonl'."""
    parts = filepath.stem.split("-v")
    return parts[1].split("-")[0] if len(parts) > 1 else None


def calculate_statistics(data: List[dict]) -> Dict:
    """Calculate all benchmark statistics from JSONL data.

    Filters out memmon entries (memory samples) if mixed with benchmark data.
    """
    # Filter out memmon entries (memory monitoring samples)
    benchmark_data = [e for e in data if not ("ram_free_gb" in e and "elapsed_s" in e and "outcome" not in e)]

    # Separate by outcome
    passed_tests = [e for e in benchmark_data if e.get("outcome") == "passed"]
    skipped_tests = [e for e in benchmark_data if e.get("outcome") == "skipped"]
    passed_with_model = [e for e in passed_tests if "model" in e]
    passed_without_model = [e for e in passed_tests if "model" not in e]

    # System health metrics (optional for backward compatibility with older schemas)
    swap_values = []
    ram_values = []
    zombie_values = []
    quality_flags = []

    for e in data:
        if "system_health" in e:
            swap_mb = e["system_health"].get("swap_used_mb", 0)
            ram_gb = e["system_health"].get("ram_free_gb", 0)
            zombies = e["system_health"].get("zombie_processes", 0)

            swap_values.append(swap_mb)
            ram_values.append(ram_gb)
            zombie_values.append(zombies)

            # Recalculate quality_flags from raw values (ignore stored flags)
            # Rationale: Thresholds are experimental and OS-specific
            #
            # Session 61 Analysis (Sequoia vs Tahoe):
            #   - Sequoia: RAM free varies 10-27 GB, swap=0
            #   - Tahoe: RAM free drops to 0-0.1 GB during load, recovers to ~24 GB between tests
            #
            # Steady-State Baseline (DeepHermes post-load relaxation):
            #   - Tahoe: ~24 GB free (1.2-1.4 min after first test)
            #   - Sequoia: ~40 GB free (similar pattern)
            #
            # Degraded Threshold: ram_free < 5 GB (extreme memory pressure)
            #   - Marks 0-0.1 GB minima as degraded ✅
            #   - Normal tests (10-20 GB free) stay clean ✅
            flags = []
            if ram_gb < 5.0:  # < 5 GB free = extreme memory pressure
                flags.append("degraded_ram")
            if zombies > 0:
                flags.append("degraded_zombies")
            if not flags:
                flags.append("clean")

            quality_flags.append(flags)

    clean_count = sum(1 for flags in quality_flags if flags == ["clean"])
    degraded_ram = sum(1 for flags in quality_flags if "degraded_ram" in flags)
    degraded_zombies = sum(1 for flags in quality_flags if "degraded_zombies" in flags)

    # Per-model statistics (with inference modality breakdown)
    # Filter: Only count actual inference tests (duration >= 0.5s)
    # This excludes infrastructure tests like test_vision_model_info_fixture_works
    inference_tests = [e for e in passed_with_model if e["duration"] >= 0.5]

    model_stats = {}
    for entry in inference_tests:
        model_id = entry["model"]["id"]
        if model_id not in model_stats:
            model_stats[model_id] = {
                "id": model_id,
                "size_gb": entry["model"].get("size_gb", 0),  # Default to 0 if missing (e.g., pipe tests)
                # Total stats (legacy, always populated)
                "count": 0,
                "total_time": 0,
                # Per-modality breakdown (NEW in v0.2.1, Audio in v0.2.2)
                "vision_count": 0,
                "vision_time": 0.0,
                "vision_ram_min": float("inf"),
                "vision_ram_max": 0,
                "text_count": 0,
                "text_time": 0.0,
                "text_ram_min": float("inf"),
                "text_ram_max": 0,
                "audio_count": 0,
                "audio_time": 0.0,
                "audio_ram_min": float("inf"),
                "audio_ram_max": 0,
                "unknown_count": 0,
                "unknown_time": 0.0,
                "unknown_ram_min": float("inf"),
                "unknown_ram_max": 0,
                # System health (global, for backward compat)
                "ram_min": float("inf"),
                "ram_max": 0,
                "swap_max": 0,
                "zombies_max": 0,
            }

        stats = model_stats[model_id]
        duration = entry["duration"]

        # Update totals (always)
        stats["count"] += 1
        stats["total_time"] += duration

        # Update modality-specific stats (NEW in v0.2.1, Audio in v0.2.2)
        modality = entry.get("metadata", {}).get("inference_modality", "unknown")
        if modality == "vision":
            stats["vision_count"] += 1
            stats["vision_time"] += duration
        elif modality == "text":
            stats["text_count"] += 1
            stats["text_time"] += duration
        elif modality == "audio":
            stats["audio_count"] += 1
            stats["audio_time"] += duration
        else:  # "unknown" or any other value (backward compat)
            stats["unknown_count"] += 1
            stats["unknown_time"] += duration

        # Handle optional system_health (backward compatibility)
        if "system_health" in entry:
            ram_gb = entry["system_health"].get("ram_free_gb", 0)
            # Update per-modality RAM stats
            if modality == "vision":
                stats["vision_ram_min"] = min(stats["vision_ram_min"], ram_gb)
                stats["vision_ram_max"] = max(stats["vision_ram_max"], ram_gb)
            elif modality == "text":
                stats["text_ram_min"] = min(stats["text_ram_min"], ram_gb)
                stats["text_ram_max"] = max(stats["text_ram_max"], ram_gb)
            elif modality == "audio":
                stats["audio_ram_min"] = min(stats["audio_ram_min"], ram_gb)
                stats["audio_ram_max"] = max(stats["audio_ram_max"], ram_gb)
            else:
                stats["unknown_ram_min"] = min(stats["unknown_ram_min"], ram_gb)
                stats["unknown_ram_max"] = max(stats["unknown_ram_max"], ram_gb)

        # Handle optional system_health - global stats (backward compatibility)
        if "system_health" in entry:
            stats["ram_min"] = min(stats["ram_min"], entry["system_health"].get("ram_free_gb", 0))
            stats["ram_max"] = max(stats["ram_max"], entry["system_health"].get("ram_free_gb", 0))
            stats["swap_max"] = max(stats["swap_max"], entry["system_health"].get("swap_used_mb", 0))
            stats["zombies_max"] = max(stats["zombies_max"], entry["system_health"].get("zombie_processes", 0))

    # Per-test statistics (use inference_tests to filter infrastructure tests)
    # Group by (test_name, modality) to differentiate Vision/Text phases of same test
    import statistics
    test_stats = {}
    for entry in inference_tests:
        # Extract test function name and normalize (remove parametrization)
        test_full = entry["test"].split("::")[-1]
        test_name = test_full.split("[")[0]  # Remove [discovered_XX] part

        model_id = entry["model"]["id"]
        model_short = model_id.replace("mlx-community/", "").split("-")[0]  # Short name
        duration = entry["duration"]
        modality = entry.get("metadata", {}).get("inference_modality", "unknown")

        # Key: (test_name, modality) to separate Vision/Text phases
        key = (test_name, modality)

        if key not in test_stats:
            test_stats[key] = {
                "name": test_name,
                "modality": modality,
                "models": set(),
                "runs": [],
            }

        test_stats[key]["models"].add(model_id)
        test_stats[key]["runs"].append({
            "model": model_id,
            "model_short": model_short,
            "duration": duration
        })

    # Calculate aggregates per test (key is now tuple: (test_name, modality))
    for key, test_data in test_stats.items():
        durations = [r["duration"] for r in test_data["runs"]]
        test_data["model_count"] = len(test_data["models"])
        test_data["median_time"] = statistics.median(durations) if durations else 0

        # Find fastest and slowest
        sorted_runs = sorted(test_data["runs"], key=lambda r: r["duration"])
        test_data["fastest"] = sorted_runs[0] if sorted_runs else None
        test_data["slowest"] = sorted_runs[-1] if sorted_runs else None

        # Convert set to list for JSON serialization
        test_data["models"] = list(test_data["models"])

    # Hardware profile (scan for first entry with data, handles manual JSONL entries)
    hw_profile = {}
    for entry in data:
        if "system" in entry and "hardware_profile" in entry["system"]:
            hw_profile = entry["system"]["hardware_profile"]
            break

    return {
        "total_tests": len(benchmark_data),
        "passed": len(passed_tests),
        "passed_with_model": len(passed_with_model),
        "passed_infrastructure": len(passed_without_model),
        "skipped": len(skipped_tests),
        "total_duration": sum(e["duration"] for e in passed_tests),
        "schema_version": benchmark_data[0].get("schema_version", "unknown") if benchmark_data else "unknown",
        "mlx_knife_version": benchmark_data[0].get("mlx_knife_version", "unknown") if benchmark_data else "unknown",
        "swap": {
            "min": min(swap_values) if swap_values else 0,
            "max": max(swap_values) if swap_values else 0,
            "avg": sum(swap_values) / len(swap_values) if swap_values else 0,
        },
        "ram": {
            "min": min(ram_values) if ram_values else 0,
            "max": max(ram_values) if ram_values else 0,
            "avg": sum(ram_values) / len(ram_values) if ram_values else 0,
        },
        "zombies": {
            "min": min(zombie_values) if zombie_values else 0,
            "max": max(zombie_values) if zombie_values else 0,
        },
        "quality": {
            "clean": clean_count,
            "degraded_ram": degraded_ram,
            "degraded_zombies": degraded_zombies,
            "clean_percent": 100 * clean_count / len(data) if data else 0,
        },
        "hardware": hw_profile,
        "models": model_stats,
        "tests": test_stats,
    }


def generate_markdown(stats: Dict, input_file: Path, compare_file: Optional[Path] = None, compare_stats: Optional[Dict] = None) -> str:
    """Generate Markdown report from statistics."""
    version = stats["mlx_knife_version"]
    date = input_file.stem.split("-v")[0]  # Extract date from filename
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC")

    # Header
    md = f"""# Benchmark Report v{TEMPLATE_VERSION}: {version}

**Date:** {date}
**Generated:** {now}
**Generator:** generate_benchmark_report.py v{TEMPLATE_VERSION}
**Hardware:** {stats['hardware'].get('model', 'unknown')}, {stats['hardware'].get('cores_physical', '?')} cores

---

## Input Files

- **Primary:** `{input_file}`
- **Schema:** v{stats['schema_version']}
"""

    if compare_file:
        md += f"- **Comparison:** `{compare_file}`\n"

    md += "\n---\n\n"

    # Executive Summary
    md += "## Executive Summary\n\n"
    md += f"**Tests:** {stats['total_tests']} total ({stats['passed']} passed, {stats['skipped']} skipped)\n"
    md += f"**Duration:** {stats['total_duration']:.1f}s ({stats['total_duration']/60:.1f} min)\n"
    md += f"**Quality:** {stats['quality']['clean_percent']:.1f}% clean ({stats['quality']['clean']}/{stats['total_tests']})\n"
    md += f"**Models:** {len(stats['models'])} tested\n\n"

    # Comparison Summary
    if compare_stats:
        old_duration = compare_stats['total_duration']
        new_duration = stats['total_duration']
        duration_delta = new_duration - old_duration
        duration_pct = (duration_delta / old_duration * 100) if old_duration > 0 else 0

        # Count models by change direction
        compare_models_dict = {m['id']: m for m in compare_stats['models'].values()}
        slower_count = 0
        faster_count = 0
        for model in stats['models'].values():
            old_model = compare_models_dict.get(model['id'])
            if old_model:
                if model['total_time'] > old_model['total_time']:
                    slower_count += 1
                elif model['total_time'] < old_model['total_time']:
                    faster_count += 1

        total_compared = slower_count + faster_count
        change_icon = "⚠️" if duration_pct > 3 else "✅" if duration_pct < -1 else "➡️"

        md += f"### Comparison\n\n"
        md += f"**vs:** `{compare_file.name}`\n"
        md += f"**Duration:** {old_duration/60:.1f} min → {new_duration/60:.1f} min ({duration_pct:+.1f}%) {change_icon}\n"
        if total_compared > 0:
            md += f"**Models:** {slower_count}/{total_compared} slower ({100*slower_count/total_compared:.0f}%), {faster_count}/{total_compared} faster ({100*faster_count/total_compared:.0f}%)\n"
        md += "\n"

    # Validation Status
    quality_icon = "✅" if stats['quality']['clean_percent'] == 100 else "⚠️"
    md += f"{quality_icon} **System Health:** "
    if stats['quality']['clean_percent'] == 100:
        md += "All tests clean (RAM >5 GB free, 0 zombies)\n"
    else:
        md += f"{stats['quality']['degraded_ram']} degraded (RAM <5 GB free), {stats['quality']['degraded_zombies']} degraded (zombies)\n"

    md += "\n---\n\n"

    # Test Summary
    md += "## Test Summary\n\n"
    md += f"""```
Total tests:       {stats['total_tests']}
Passed:            {stats['passed']}
  With model:      {stats['passed_with_model']}
  Infrastructure:  {stats['passed_infrastructure']}
Skipped:           {stats['skipped']}
Duration:          {stats['total_duration']:.1f}s ({stats['total_duration']/60:.1f} min)
```

---

## System Health

"""
    md += f"""```
Swap (MB):         min={stats['swap']['min']}, max={stats['swap']['max']}, avg={stats['swap']['avg']:.1f}
RAM free (GB):     min={stats['ram']['min']:.1f}, max={stats['ram']['max']:.1f}, avg={stats['ram']['avg']:.1f}
Zombies:           min={stats['zombies']['min']}, max={stats['zombies']['max']}

Quality Flags (Thresholds: RAM <5 GB free, zombies >0):
  Clean:           {stats['quality']['clean']}/{stats['total_tests']} ({stats['quality']['clean_percent']:.1f}%)
  Degraded (RAM):  {stats['quality']['degraded_ram']}
  Degraded (zombies): {stats['quality']['degraded_zombies']}
```

---

## Per-Model Statistics

"""

    # Sort models alphabetically (stable ordering across reports)
    sorted_models = sorted(stats['models'].values(), key=lambda m: m['id'].lower())

    # Build comparison lookup if available
    compare_models = {}
    if compare_stats:
        compare_models = {m['id']: m for m in compare_stats['models'].values()}

    if compare_stats:
        md += f"""```
{'Model':<40} {'Size':<7} {'Mode':<6} {'Tests':<5} {'Time':<8} {'Old':<8} {'Δ':<8} {'Change':<10} {'RAM (GB)':<12}
{'='*40} {'='*7} {'='*6} {'='*5} {'='*8} {'='*8} {'='*8} {'='*10} {'='*12}
"""
    else:
        md += f"""```
{'Model':<50} {'Size':<8} {'Mode':<6} {'Tests':<6} {'Time':<10} {'RAM (GB)':<20}
{'='*50} {'='*8} {'='*6} {'='*6} {'='*10} {'='*20}
"""

    for model in sorted_models:
        # Shorten model ID (remove mlx-community/ prefix)
        model_short = model['id'].replace('mlx-community/', '')
        max_len = 38 if compare_stats else 48
        if len(model_short) > max_len:
            model_short = model_short[:max_len-3] + "..."

        # Global RAM range (for backward compat / fallback)
        ram_range = f"{model['ram_min']:.1f}-{model['ram_max']:.1f}"

        if compare_stats:
            old_model = compare_models.get(model['id'])

            # Separate rows per modality (same as non-comparison mode)
            rows_written = 0

            # Vision modality
            if model['vision_count'] > 0:
                v_ram_min = model['vision_ram_min']
                v_ram_max = model['vision_ram_max']
                if v_ram_min == float('inf'):
                    v_ram_range = "-"
                elif v_ram_min == v_ram_max:
                    v_ram_range = f"{v_ram_min:.1f}"
                else:
                    v_ram_range = f"{v_ram_min:.1f}-{v_ram_max:.1f}"

                # Get old vision stats (if available)
                if old_model and old_model.get('vision_count', 0) > 0:
                    old_time = old_model['vision_time']
                    delta = model['vision_time'] - old_time
                    change_pct = (delta / old_time * 100) if old_time > 0 else 0
                    if change_pct > 5:
                        status = "⚠️"
                    elif change_pct < -1:
                        status = "✅"
                    else:
                        status = ""
                    change_str = f"{change_pct:+.1f}% {status}"
                    md += f"{model_short:<40} {model['size_gb']:>5.1f}GB {'Vision':<6} {model['vision_count']:<5} {model['vision_time']:>6.1f}s {old_time:>6.1f}s {delta:>+6.1f}s {change_str:<10} {v_ram_range:<12}\n"
                else:
                    md += f"{model_short:<40} {model['size_gb']:>5.1f}GB {'Vision':<6} {model['vision_count']:<5} {model['vision_time']:>6.1f}s {'N/A':<8} {'N/A':<8} {'NEW':<10} {v_ram_range:<12}\n"
                rows_written += 1

            # Text modality
            if model['text_count'] > 0:
                t_ram_min = model['text_ram_min']
                t_ram_max = model['text_ram_max']
                if t_ram_min == float('inf'):
                    t_ram_range = "-"
                elif t_ram_min == t_ram_max:
                    t_ram_range = f"{t_ram_min:.1f}"
                else:
                    t_ram_range = f"{t_ram_min:.1f}-{t_ram_max:.1f}"

                # Get old text stats (if available)
                if old_model and old_model.get('text_count', 0) > 0:
                    old_time = old_model['text_time']
                    delta = model['text_time'] - old_time
                    change_pct = (delta / old_time * 100) if old_time > 0 else 0
                    if change_pct > 5:
                        status = "⚠️"
                    elif change_pct < -1:
                        status = "✅"
                    else:
                        status = ""
                    change_str = f"{change_pct:+.1f}% {status}"
                    md += f"{model_short:<40} {model['size_gb']:>5.1f}GB {'Text':<6} {model['text_count']:<5} {model['text_time']:>6.1f}s {old_time:>6.1f}s {delta:>+6.1f}s {change_str:<10} {t_ram_range:<12}\n"
                else:
                    md += f"{model_short:<40} {model['size_gb']:>5.1f}GB {'Text':<6} {model['text_count']:<5} {model['text_time']:>6.1f}s {'N/A':<8} {'N/A':<8} {'NEW':<10} {t_ram_range:<12}\n"
                rows_written += 1

            # Audio modality (NEW in v0.2.2)
            if model['audio_count'] > 0:
                a_ram_min = model['audio_ram_min']
                a_ram_max = model['audio_ram_max']
                if a_ram_min == float('inf'):
                    a_ram_range = "-"
                elif a_ram_min == a_ram_max:
                    a_ram_range = f"{a_ram_min:.1f}"
                else:
                    a_ram_range = f"{a_ram_min:.1f}-{a_ram_max:.1f}"

                # Get old audio stats (if available)
                if old_model and old_model.get('audio_count', 0) > 0:
                    old_time = old_model['audio_time']
                    delta = model['audio_time'] - old_time
                    change_pct = (delta / old_time * 100) if old_time > 0 else 0
                    if change_pct > 5:
                        status = "⚠️"
                    elif change_pct < -1:
                        status = "✅"
                    else:
                        status = ""
                    change_str = f"{change_pct:+.1f}% {status}"
                    md += f"{model_short:<40} {model['size_gb']:>5.1f}GB {'Audio':<6} {model['audio_count']:<5} {model['audio_time']:>6.1f}s {old_time:>6.1f}s {delta:>+6.1f}s {change_str:<10} {a_ram_range:<12}\n"
                else:
                    md += f"{model_short:<40} {model['size_gb']:>5.1f}GB {'Audio':<6} {model['audio_count']:<5} {model['audio_time']:>6.1f}s {'N/A':<8} {'N/A':<8} {'NEW':<10} {a_ram_range:<12}\n"
                rows_written += 1

            # Fallback for legacy data (no modality info) - rare in comparison mode
            if rows_written == 0 and old_model:
                old_time = old_model['total_time']
                delta = model['total_time'] - old_time
                change_pct = (delta / old_time * 100) if old_time > 0 else 0
                if change_pct > 5:
                    status = "⚠️"
                elif change_pct < -1:
                    status = "✅"
                else:
                    status = ""
                change_str = f"{change_pct:+.1f}% {status}"
                md += f"{model_short:<40} {model['size_gb']:>5.1f}GB {'-':<6} {model['count']:<5} {model['total_time']:>6.1f}s {old_time:>6.1f}s {delta:>+6.1f}s {change_str:<10} {ram_range:<12}\n"
            elif rows_written == 0:
                # New model with no modality info
                md += f"{model_short:<40} {model['size_gb']:>5.1f}GB {'-':<6} {model['count']:<5} {model['total_time']:>6.1f}s {'N/A':<8} {'N/A':<8} {'NEW':<10} {ram_range:<12}\n"
        else:
            # Separate rows per modality (no "Mixed" ambiguity)
            # Each modality gets its own line with specific stats + RAM
            rows_written = 0

            if model['vision_count'] > 0:
                # Use modality-specific RAM range (single value if min==max)
                v_ram_min = model['vision_ram_min']
                v_ram_max = model['vision_ram_max']
                if v_ram_min == float('inf'):
                    v_ram_range = "-"
                elif v_ram_min == v_ram_max:
                    v_ram_range = f"{v_ram_min:.1f}"
                else:
                    v_ram_range = f"{v_ram_min:.1f}-{v_ram_max:.1f}"
                md += f"{model_short:<50} {model['size_gb']:>6.1f}GB {'Vision':<6} {model['vision_count']:<6} {model['vision_time']:>8.1f}s  {v_ram_range:<20}\n"
                rows_written += 1

            if model['text_count'] > 0:
                # Use modality-specific RAM range (single value if min==max)
                t_ram_min = model['text_ram_min']
                t_ram_max = model['text_ram_max']
                if t_ram_min == float('inf'):
                    t_ram_range = "-"
                elif t_ram_min == t_ram_max:
                    t_ram_range = f"{t_ram_min:.1f}"
                else:
                    t_ram_range = f"{t_ram_min:.1f}-{t_ram_max:.1f}"
                md += f"{model_short:<50} {model['size_gb']:>6.1f}GB {'Text':<6} {model['text_count']:<6} {model['text_time']:>8.1f}s  {t_ram_range:<20}\n"
                rows_written += 1

            if model['audio_count'] > 0:
                # Use modality-specific RAM range (single value if min==max)
                a_ram_min = model['audio_ram_min']
                a_ram_max = model['audio_ram_max']
                if a_ram_min == float('inf'):
                    a_ram_range = "-"
                elif a_ram_min == a_ram_max:
                    a_ram_range = f"{a_ram_min:.1f}"
                else:
                    a_ram_range = f"{a_ram_min:.1f}-{a_ram_max:.1f}"
                md += f"{model_short:<50} {model['size_gb']:>6.1f}GB {'Audio':<6} {model['audio_count']:<6} {model['audio_time']:>8.1f}s  {a_ram_range:<20}\n"
                rows_written += 1

            # Fallback for legacy data (no modality info)
            if rows_written == 0:
                md += f"{model_short:<50} {model['size_gb']:>6.1f}GB {'-':<6} {model['count']:<6} {model['total_time']:>8.1f}s  {ram_range:<20}\n"

    md += "```\n\n"

    # Model Categories (with modality differentiation)
    large_models = [m for m in sorted_models if m['size_gb'] >= 20]
    medium_models = [m for m in sorted_models if 10 <= m['size_gb'] < 20]
    small_models = [m for m in sorted_models if m['size_gb'] < 10]

    def format_category_stats(models_list, category_name):
        """Format category statistics with Vision/Text breakdown."""
        if not models_list:
            return ""

        # Collect Vision, Text, and Audio stats (Audio NEW in v0.2.2)
        vision_models = [m for m in models_list if m.get('vision_count', 0) > 0]
        text_models = [m for m in models_list if m.get('text_count', 0) > 0]
        audio_models = [m for m in models_list if m.get('audio_count', 0) > 0]

        output = f"{category_name}: {len(models_list)} models\n"
        output += f"  Avg size:               {sum(m['size_gb'] for m in models_list) / len(models_list):.1f} GB\n"

        # Vision stats
        if vision_models:
            avg_vision_time = sum(m['vision_time']/m['vision_count'] for m in vision_models) / len(vision_models)

            # Collect RAM values (filter sentinel values)
            vision_ram_mins = [m['vision_ram_min'] for m in vision_models if m['vision_ram_min'] != float('inf')]
            vision_ram_maxs = [m['vision_ram_max'] for m in vision_models if m['vision_ram_max'] > 0]

            output += f"  Vision Tests:\n"
            output += f"    Models tested:        {len(vision_models)}\n"
            output += f"    Avg test time:        {avg_vision_time:.1f}s\n"

            # Only output RAM range if data available
            if vision_ram_mins and vision_ram_maxs:
                all_vision_ram_min = min(vision_ram_mins)
                all_vision_ram_max = max(vision_ram_maxs)
                output += f"    RAM range:            {all_vision_ram_min:.1f}-{all_vision_ram_max:.1f} GB\n"

        # Text stats
        if text_models:
            avg_text_time = sum(m['text_time']/m['text_count'] for m in text_models) / len(text_models)

            # Collect RAM values (filter sentinel values)
            text_ram_mins = [m['text_ram_min'] for m in text_models if m['text_ram_min'] != float('inf')]
            text_ram_maxs = [m['text_ram_max'] for m in text_models if m['text_ram_max'] > 0]

            output += f"  Text Tests:\n"
            output += f"    Models tested:        {len(text_models)}\n"
            output += f"    Avg test time:        {avg_text_time:.1f}s\n"

            # Only output RAM range if data available
            if text_ram_mins and text_ram_maxs:
                all_text_ram_min = min(text_ram_mins)
                all_text_ram_max = max(text_ram_maxs)
                output += f"    RAM range:            {all_text_ram_min:.1f}-{all_text_ram_max:.1f} GB\n"

        # Audio stats (NEW in v0.2.2)
        if audio_models:
            avg_audio_time = sum(m['audio_time']/m['audio_count'] for m in audio_models) / len(audio_models)

            # Collect RAM values (filter sentinel values)
            audio_ram_mins = [m['audio_ram_min'] for m in audio_models if m['audio_ram_min'] != float('inf')]
            audio_ram_maxs = [m['audio_ram_max'] for m in audio_models if m['audio_ram_max'] > 0]

            output += f"  Audio Tests:\n"
            output += f"    Models tested:        {len(audio_models)}\n"
            output += f"    Avg test time:        {avg_audio_time:.1f}s\n"

            # Only output RAM range if data available
            if audio_ram_mins and audio_ram_maxs:
                all_audio_ram_min = min(audio_ram_mins)
                all_audio_ram_max = max(audio_ram_maxs)
                output += f"    RAM range:            {all_audio_ram_min:.1f}-{all_audio_ram_max:.1f} GB\n"

        # Fallback for legacy data (no modality info)
        if not vision_models and not text_models and not audio_models:
            avg_time = sum(m['total_time']/m['count'] for m in models_list) / len(models_list)
            avg_ram = sum(m['ram_min'] for m in models_list) / len(models_list)
            output += f"  Avg test time:          {avg_time:.1f}s\n"
            output += f"  Avg min RAM:            {avg_ram:.1f} GB\n"

        return output

    md += "### Model Categories\n\n"
    if large_models or medium_models or small_models:
        md += "```\n"
        if large_models:
            md += format_category_stats(large_models, "LARGE MODELS (≥20 GB)")
            md += "\n"
        if medium_models:
            md += format_category_stats(medium_models, "MEDIUM MODELS (10-20 GB)")
            md += "\n"
        if small_models:
            md += format_category_stats(small_models, "SMALL MODELS (<10 GB)")
        md += "```\n"

    md += "\n---\n\n"

    # Per-Test Statistics
    md += "## Per-Test Statistics\n\n"
    md += "Shows performance range across models for each test.\n\n"

    # Sort tests alphabetically (stable ordering across reports)
    sorted_tests = sorted(stats['tests'].values(), key=lambda t: t['name'].lower())

    # Build comparison lookup for tests (key: (name, modality))
    compare_tests = {}
    if compare_stats:
        compare_tests = {(t['name'], t.get('modality', 'unknown')): t for t in compare_stats['tests'].values()}

    if compare_stats:
        md += f"""```
{'Test Name':<38} {'Mode':<6} {'Models':<7} {'Fastest':<18} {'Slowest':<18} {'Med':<6} {'Old':<6} {'Δ Med':<8}
{'='*38} {'='*6} {'='*7} {'='*18} {'='*18} {'='*6} {'='*6} {'='*8}
"""
    else:
        md += f"""```
{'Test Name':<44} {'Mode':<6} {'Models':<7} {'Fastest':<22} {'Slowest':<22} {'Med Time'}
{'='*44} {'='*6} {'='*7} {'='*22} {'='*22} {'='*8}
"""

    for test in sorted_tests:
        # Shorten test name if needed
        max_test_len = 36 if compare_stats else 42
        test_short = test['name']
        if len(test_short) > max_test_len:
            test_short = test_short[:max_test_len-3] + "..."

        # Format modality (Vision/Text/Audio/- for unknown)
        modality = test.get('modality', 'unknown')
        if modality == 'vision':
            mode_str = 'Vision'
        elif modality == 'text':
            mode_str = 'Text'
        elif modality == 'audio':
            mode_str = 'Audio'
        else:
            mode_str = '-'

        # Format fastest/slowest
        fastest = test['fastest']
        slowest = test['slowest']

        if fastest and slowest:
            max_model_len = 16 if compare_stats else 20
            fastest_str = f"{fastest['model_short']} ({fastest['duration']:.1f}s)"
            slowest_str = f"{slowest['model_short']} ({slowest['duration']:.1f}s)"
            if len(fastest_str) > max_model_len:
                fastest_str = fastest_str[:max_model_len-3] + "..."
            if len(slowest_str) > max_model_len:
                slowest_str = slowest_str[:max_model_len-3] + "..."

            med_time = test['median_time']

            if compare_stats:
                old_test = compare_tests.get((test['name'], test.get('modality', 'unknown')))
                if old_test:
                    old_med = old_test['median_time']
                    delta_pct = ((med_time - old_med) / old_med * 100) if old_med > 0 else 0
                    delta_str = f"{delta_pct:+.1f}%"
                    md += f"{test_short:<38} {mode_str:<6} {test['model_count']:<7} {fastest_str:<18} {slowest_str:<18} {med_time:<5.1f}s {old_med:<5.1f}s {delta_str:<8}\n"
                else:
                    md += f"{test_short:<38} {mode_str:<6} {test['model_count']:<7} {fastest_str:<18} {slowest_str:<18} {med_time:<5.1f}s {'N/A':<6} {'NEW':<8}\n"
            else:
                md += f"{test_short:<44} {mode_str:<6} {test['model_count']:<7} {fastest_str:<22} {slowest_str:<22} {med_time:.1f}s\n"

    md += "```\n\n"

    # GPU Analysis Section (v1.1 - only if memory data provided)
    gpu_correlations = stats.get("gpu_correlations", {})
    cold_starts = stats.get("cold_starts", {})

    if gpu_correlations:
        md += "\n---\n\n"
        md += "## GPU Analysis (v1.1)\n\n"
        md += "Per-test GPU utilization correlated from memory monitoring data.\n\n"

        # Aggregate GPU stats by model
        model_gpu_stats = {}
        for test_id, gpu_stats in gpu_correlations.items():
            # Find the corresponding test entry to get model info
            test_entry = next((t for t in stats.get("tests", {}).values()
                             if any(r["model"] in test_id for r in t.get("runs", []))), None)
            if not test_entry:
                continue

            for run in test_entry.get("runs", []):
                model_id = run["model"]
                if model_id not in model_gpu_stats:
                    model_gpu_stats[model_id] = {
                        "gpu_device_samples": [],
                        "gpu_renderer_samples": [],
                        "cold_start_count": 0,
                        "total_count": 0,
                    }
                model_gpu_stats[model_id]["gpu_device_samples"].append(gpu_stats["gpu_device_avg"])
                model_gpu_stats[model_id]["gpu_renderer_samples"].append(gpu_stats["gpu_renderer_avg"])
                model_gpu_stats[model_id]["total_count"] += 1
                if cold_starts.get(test_id, False):
                    model_gpu_stats[model_id]["cold_start_count"] += 1

        # Cold-start analysis
        cold_start_tests = [(tid, gpu_correlations.get(tid, {}))
                           for tid, is_cold in cold_starts.items() if is_cold]

        if cold_start_tests:
            md += "### Cold-Start Tests\n\n"
            md += "First test per model (includes model loading overhead).\n\n"
            md += f"```\n"
            md += f"{'Test':<60} {'GPU Dev':<8} {'GPU Rnd':<8} {'Samples':<8}\n"
            md += f"{'='*60} {'='*8} {'='*8} {'='*8}\n"

            for test_id, gpu_stats in cold_start_tests[:15]:  # Limit to 15
                test_short = test_id.split("::")[-1][:58]
                if gpu_stats:
                    md += f"{test_short:<60} {gpu_stats['gpu_device_avg']:>6.1f}% {gpu_stats['gpu_renderer_avg']:>6.1f}% {gpu_stats['sample_count']:>8}\n"
                else:
                    md += f"{test_short:<60} {'N/A':>8} {'N/A':>8} {'N/A':>8}\n"

            if len(cold_start_tests) > 15:
                md += f"... and {len(cold_start_tests) - 15} more cold-start tests\n"
            md += f"```\n\n"

        # High GPU utilization tests
        high_gpu_tests = [(tid, gs) for tid, gs in gpu_correlations.items()
                         if gs.get("gpu_device_avg", 0) > 50]
        high_gpu_tests.sort(key=lambda x: x[1]["gpu_device_avg"], reverse=True)

        if high_gpu_tests:
            md += "### High GPU Utilization Tests\n\n"
            md += "Tests with >50% average GPU device utilization.\n\n"
            md += f"```\n"
            md += f"{'Test':<55} {'GPU Dev':<10} {'GPU Rnd':<10} {'Pressure':<10}\n"
            md += f"{'='*55} {'='*10} {'='*10} {'='*10}\n"

            for test_id, gpu_stats in high_gpu_tests[:10]:
                test_short = test_id.split("::")[-1][:53]
                pressure = gpu_stats.get("memory_pressure_max", 1)
                pressure_str = "WARN" if pressure > 1 else "OK"
                md += f"{test_short:<55} {gpu_stats['gpu_device_avg']:>8.1f}% {gpu_stats['gpu_renderer_avg']:>8.1f}% {pressure_str:>10}\n"

            md += f"```\n\n"

        # GPU utilization summary
        all_gpu_device = [gs["gpu_device_avg"] for gs in gpu_correlations.values()]
        all_gpu_renderer = [gs["gpu_renderer_avg"] for gs in gpu_correlations.values()]

        # Get comparison GPU data if available
        compare_gpu = compare_stats.get("gpu_correlations", {}) if compare_stats else {}
        compare_gpu_device = [gs["gpu_device_avg"] for gs in compare_gpu.values()] if compare_gpu else []
        compare_gpu_renderer = [gs["gpu_renderer_avg"] for gs in compare_gpu.values()] if compare_gpu else []

        if all_gpu_device:
            md += "### GPU Utilization Summary\n\n"
            md += f"```\n"
            sample_interval = stats.get("sample_interval_ms", 200)
            md += f"Sample interval:      {sample_interval}ms\n"
            md += f"Correlated tests:     {len(gpu_correlations)}\n"
            md += f"Cold-start tests:     {len(cold_start_tests)}\n"

            gpu_dev_avg = sum(all_gpu_device)/len(all_gpu_device)
            gpu_dev_max = max(all_gpu_device)
            gpu_rnd_avg = sum(all_gpu_renderer)/len(all_gpu_renderer)
            gpu_rnd_max = max(all_gpu_renderer)

            if compare_gpu_device:
                old_dev_avg = sum(compare_gpu_device)/len(compare_gpu_device)
                old_dev_max = max(compare_gpu_device)
                old_rnd_avg = sum(compare_gpu_renderer)/len(compare_gpu_renderer)
                old_rnd_max = max(compare_gpu_renderer)

                dev_avg_delta = gpu_dev_avg - old_dev_avg
                dev_max_delta = gpu_dev_max - old_dev_max
                rnd_avg_delta = gpu_rnd_avg - old_rnd_avg
                rnd_max_delta = gpu_rnd_max - old_rnd_max

                md += f"GPU Device avg:       {gpu_dev_avg:>5.1f}%  (was {old_dev_avg:>5.1f}%, Δ {dev_avg_delta:+.1f}%)\n"
                md += f"GPU Device max:       {gpu_dev_max:>5.1f}%  (was {old_dev_max:>5.1f}%, Δ {dev_max_delta:+.1f}%)\n"
                md += f"GPU Renderer avg:     {gpu_rnd_avg:>5.1f}%  (was {old_rnd_avg:>5.1f}%, Δ {rnd_avg_delta:+.1f}%)\n"
                md += f"GPU Renderer max:     {gpu_rnd_max:>5.1f}%  (was {old_rnd_max:>5.1f}%, Δ {rnd_max_delta:+.1f}%)\n"
            else:
                md += f"GPU Device avg:       {gpu_dev_avg:.1f}%\n"
                md += f"GPU Device max:       {gpu_dev_max:.1f}%\n"
                md += f"GPU Renderer avg:     {gpu_rnd_avg:.1f}%\n"
                md += f"GPU Renderer max:     {gpu_rnd_max:.1f}%\n"

            md += f"```\n\n"

    md += "\n---\n\n"
    md += "## Files\n\n"
    md += f"- **Benchmark report:** `{input_file}`\n"
    md += f"- **Schema:** `benchmarks/schemas/report-v{stats['schema_version']}.schema.json`\n"
    if gpu_correlations:
        md += f"- **Memory data:** correlated\n"

    return md


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark analysis report from JSONL data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'input',
        nargs='?',
        type=Path,
        help='JSONL benchmark file (default: latest in benchmarks/reports/)'
    )
    parser.add_argument(
        '--compare',
        type=Path,
        help='Compare with this JSONL file (adds Old/Δ/Change columns)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output markdown file (default: auto-generated in benchmarks/reports/)'
    )
    parser.add_argument(
        '--memory',
        type=Path,
        help='Memory monitoring JSONL file for GPU/memory correlation (v1.1 feature)'
    )

    args = parser.parse_args()

    # Determine input file
    if args.input:
        input_file = args.input
    else:
        input_file = find_latest_jsonl()
        if not input_file:
            print("❌ No JSONL files found in benchmarks/reports/")
            sys.exit(1)
        print(f"📊 Auto-detected: {input_file}")

    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        sys.exit(1)

    # Load and validate
    print(f"📋 Loading: {input_file}")
    schema = load_schema()
    data = load_jsonl(input_file)

    print(f"✓ Loaded {len(data)} entries")

    # Validate against schema
    if not validate_jsonl(data, schema, input_file):
        sys.exit(1)

    print(f"✓ Schema validation passed")

    # Calculate statistics
    stats = calculate_statistics(data)

    # Load memory data and correlate (v1.1 feature)
    gpu_correlations = {}
    cold_starts = {}
    if args.memory:
        if not args.memory.exists():
            print(f"❌ Memory file not found: {args.memory}")
            sys.exit(1)
        print(f"🔬 Loading memory data: {args.memory}")
        memory_data = load_jsonl(args.memory)
        # Extract summary entry (contains interval_ms, duration, etc.)
        memory_summary = next((m["summary"] for m in memory_data if "summary" in m), {})
        # Filter out summary entry for sample processing
        memory_samples = [m for m in memory_data if "summary" not in m]
        sample_interval_ms = memory_summary.get("interval_ms", 200)
        print(f"✓ Loaded {len(memory_samples)} memory samples (interval: {sample_interval_ms}ms)")

        # Correlate memory samples to tests
        gpu_correlations = correlate_memory_to_tests(data, memory_samples)
        print(f"✓ Correlated GPU data for {len(gpu_correlations)} tests")

        # Detect cold starts
        cold_starts = detect_cold_starts(data)
        cold_count = sum(1 for v in cold_starts.values() if v)
        print(f"✓ Detected {cold_count} cold-start tests")

        # Store in stats for report generation
        stats["gpu_correlations"] = gpu_correlations
        stats["cold_starts"] = cold_starts
        stats["sample_interval_ms"] = sample_interval_ms

    # Load and calculate comparison statistics if requested
    compare_stats = None
    if args.compare:
        if not args.compare.exists():
            print(f"❌ Comparison file not found: {args.compare}")
            sys.exit(1)
        print(f"📊 Comparing with: {args.compare}")
        compare_data = load_jsonl(args.compare)
        if not validate_jsonl(compare_data, schema, args.compare):
            sys.exit(1)
        compare_stats = calculate_statistics(compare_data)
        print(f"✓ Loaded {len(compare_data)} comparison entries")

        # Auto-detect memory file for comparison run (v1.1)
        compare_memory_file = find_memory_file_for_benchmark(args.compare)
        if compare_memory_file:
            print(f"🔬 Auto-detected comparison memory: {compare_memory_file.name}")
            compare_memory_data = load_jsonl(compare_memory_file)
            compare_memory_samples = [m for m in compare_memory_data if "summary" not in m]
            compare_gpu_correlations = correlate_memory_to_tests(compare_data, compare_memory_samples)
            print(f"✓ Correlated GPU data for {len(compare_gpu_correlations)} comparison tests")
            compare_stats["gpu_correlations"] = compare_gpu_correlations

    # Generate report
    markdown = generate_markdown(stats, input_file, args.compare, compare_stats)

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        # Auto-generate: BENCHMARK-v1-<version>-<date>.md
        version = extract_version_from_filename(input_file) or stats["mlx_knife_version"]
        date = input_file.stem.split("-v")[0]  # Extract date portion
        output_file = REPORTS_DIR / f"BENCHMARK-v{TEMPLATE_VERSION}-{version}-{date}.md"

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(markdown)

    print(f"✅ Generated: {output_file}")
    print()
    print(f"Summary:")
    print(f"  Tests: {stats['passed']}/{stats['total_tests']} passed")
    print(f"  Duration: {stats['total_duration']/60:.1f} min")
    print(f"  Quality: {stats['quality']['clean_percent']:.1f}% clean")
    print(f"  Models: {len(stats['models'])}")


if __name__ == "__main__":
    main()
