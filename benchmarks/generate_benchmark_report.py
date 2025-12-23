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
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import jsonschema
except ImportError:
    print("Error: jsonschema not installed. Install with: pip install jsonschema")
    sys.exit(1)


# Template version
TEMPLATE_VERSION = "1.0"
REPORTS_DIR = Path("benchmarks/reports")
SCHEMA_PATH = Path("benchmarks/schemas/report-current.schema.json")


def load_schema() -> dict:
    """Load current JSON schema."""
    if not SCHEMA_PATH.exists():
        print(f"❌ Schema not found: {SCHEMA_PATH}")
        sys.exit(1)

    with open(SCHEMA_PATH) as f:
        return json.load(f)


def validate_jsonl(data: List[dict], schema: dict, filepath: Path) -> bool:
    """Validate JSONL data against schema."""
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
    """Calculate all benchmark statistics from JSONL data."""
    # Separate by outcome
    passed_tests = [e for e in data if e.get("outcome") == "passed"]
    skipped_tests = [e for e in data if e.get("outcome") == "skipped"]
    passed_with_model = [e for e in passed_tests if "model" in e]
    passed_without_model = [e for e in passed_tests if "model" not in e]

    # System health metrics (optional for backward compatibility with older schemas)
    swap_values = []
    ram_values = []
    zombie_values = []
    quality_flags = []

    for e in data:
        if "system_health" in e:
            swap_values.append(e["system_health"].get("swap_used_mb", 0))
            ram_values.append(e["system_health"].get("ram_free_gb", 0))
            zombie_values.append(e["system_health"].get("zombie_processes", 0))
            quality_flags.append(e["system_health"].get("quality_flags", ["unknown"]))

    clean_count = sum(1 for flags in quality_flags if flags == ["clean"])
    degraded_swap = sum(1 for flags in quality_flags if "degraded_swap" in flags)
    degraded_zombies = sum(1 for flags in quality_flags if "degraded_zombies" in flags)

    # Per-model statistics
    model_stats = {}
    for entry in passed_with_model:
        model_id = entry["model"]["id"]
        if model_id not in model_stats:
            model_stats[model_id] = {
                "id": model_id,
                "size_gb": entry["model"]["size_gb"],
                "count": 0,
                "total_time": 0,
                "ram_min": float("inf"),
                "ram_max": 0,
                "swap_max": 0,
                "zombies_max": 0,
            }

        stats = model_stats[model_id]
        stats["count"] += 1
        stats["total_time"] += entry["duration"]
        # Handle optional system_health (backward compatibility)
        if "system_health" in entry:
            stats["ram_min"] = min(stats["ram_min"], entry["system_health"].get("ram_free_gb", 0))
            stats["ram_max"] = max(stats["ram_max"], entry["system_health"].get("ram_free_gb", 0))
            stats["swap_max"] = max(stats["swap_max"], entry["system_health"].get("swap_used_mb", 0))
            stats["zombies_max"] = max(stats["zombies_max"], entry["system_health"].get("zombie_processes", 0))

    # Per-test statistics
    import statistics
    test_stats = {}
    for entry in passed_with_model:
        # Extract test function name and normalize (remove parametrization)
        test_full = entry["test"].split("::")[-1]
        test_name = test_full.split("[")[0]  # Remove [discovered_XX] part

        model_id = entry["model"]["id"]
        model_short = model_id.replace("mlx-community/", "").split("-")[0]  # Short name
        duration = entry["duration"]

        if test_name not in test_stats:
            test_stats[test_name] = {
                "name": test_name,
                "models": set(),
                "runs": [],
            }

        test_stats[test_name]["models"].add(model_id)
        test_stats[test_name]["runs"].append({
            "model": model_id,
            "model_short": model_short,
            "duration": duration
        })

    # Calculate aggregates per test
    for test_name, stats in test_stats.items():
        durations = [r["duration"] for r in stats["runs"]]
        stats["model_count"] = len(stats["models"])
        stats["median_time"] = statistics.median(durations) if durations else 0

        # Find fastest and slowest
        sorted_runs = sorted(stats["runs"], key=lambda r: r["duration"])
        stats["fastest"] = sorted_runs[0] if sorted_runs else None
        stats["slowest"] = sorted_runs[-1] if sorted_runs else None

        # Convert set to list for JSON serialization
        stats["models"] = list(stats["models"])

    # Hardware profile (from first entry, optional for backward compatibility)
    hw_profile = {}
    if data and "system" in data[0] and "hardware_profile" in data[0]["system"]:
        hw_profile = data[0]["system"]["hardware_profile"]

    return {
        "total_tests": len(data),
        "passed": len(passed_tests),
        "passed_with_model": len(passed_with_model),
        "passed_infrastructure": len(passed_without_model),
        "skipped": len(skipped_tests),
        "total_duration": sum(e["duration"] for e in passed_tests),
        "schema_version": data[0]["schema_version"] if data else "unknown",
        "mlx_knife_version": data[0]["mlx_knife_version"] if data else "unknown",
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
            "degraded_swap": degraded_swap,
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
    now = datetime.now(timezone.utc).isoformat()

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
        md += "All tests clean (0 MB swap, 0 zombies)\n"
    else:
        md += f"{stats['quality']['degraded_swap']} degraded (swap), {stats['quality']['degraded_zombies']} degraded (zombies)\n"

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

Quality Flags:
  Clean:           {stats['quality']['clean']}/{stats['total_tests']} ({stats['quality']['clean_percent']:.1f}%)
  Degraded (swap): {stats['quality']['degraded_swap']}
  Degraded (zombies): {stats['quality']['degraded_zombies']}
```

---

## Per-Model Statistics

"""

    # Sort models by total time (descending), or by change if comparing
    sorted_models = sorted(stats['models'].values(), key=lambda m: m['total_time'], reverse=True)

    # Build comparison lookup if available
    compare_models = {}
    if compare_stats:
        compare_models = {m['id']: m for m in compare_stats['models'].values()}
        # Re-sort by change percentage (biggest regression first)
        def get_change_pct(model):
            old = compare_models.get(model['id'])
            if old and old['total_time'] > 0:
                return (model['total_time'] - old['total_time']) / old['total_time'] * 100
            return 0
        sorted_models = sorted(stats['models'].values(), key=get_change_pct, reverse=True)

    if compare_stats:
        md += f"""```
{'Model':<42} {'Size':<7} {'Tests':<5} {'Time':<8} {'Old':<8} {'Δ':<8} {'Change':<10} {'RAM (GB)':<12}
{'='*42} {'='*7} {'='*5} {'='*8} {'='*8} {'='*8} {'='*10} {'='*12}
"""
    else:
        md += f"""```
{'Model':<50} {'Size':<8} {'Tests':<6} {'Time':<10} {'RAM (GB)':<20}
{'='*50} {'='*8} {'='*6} {'='*10} {'='*20}
"""

    for model in sorted_models:
        # Shorten model ID (remove mlx-community/ prefix)
        model_short = model['id'].replace('mlx-community/', '')
        max_len = 40 if compare_stats else 48
        if len(model_short) > max_len:
            model_short = model_short[:max_len-3] + "..."

        ram_range = f"{model['ram_min']:.1f}-{model['ram_max']:.1f}"

        if compare_stats:
            old_model = compare_models.get(model['id'])
            if old_model:
                old_time = old_model['total_time']
                delta = model['total_time'] - old_time
                change_pct = (delta / old_time * 100) if old_time > 0 else 0
                # Status indicator
                if change_pct > 5:
                    status = "⚠️"
                elif change_pct < -1:
                    status = "✅"
                else:
                    status = ""
                change_str = f"{change_pct:+.1f}% {status}"
                md += f"{model_short:<42} {model['size_gb']:>5.1f}GB {model['count']:<5} {model['total_time']:>6.1f}s {old_time:>6.1f}s {delta:>+6.1f}s {change_str:<10} {ram_range:<12}\n"
            else:
                md += f"{model_short:<42} {model['size_gb']:>5.1f}GB {model['count']:<5} {model['total_time']:>6.1f}s {'N/A':<8} {'N/A':<8} {'NEW':<10} {ram_range:<12}\n"
        else:
            md += f"{model_short:<50} {model['size_gb']:>6.1f}GB {model['count']:<6} {model['total_time']:>8.1f}s  {ram_range:<20}\n"

    md += "```\n\n"

    # Model Categories
    large_models = [m for m in sorted_models if m['size_gb'] >= 20]
    medium_models = [m for m in sorted_models if 10 <= m['size_gb'] < 20]
    small_models = [m for m in sorted_models if m['size_gb'] < 10]

    md += "### Model Categories\n\n"
    md += f"""```
LARGE MODELS (≥20 GB):    {len(large_models)} models
  Avg size:               {sum(m['size_gb'] for m in large_models) / len(large_models):.1f} GB
  Avg test time:          {sum(m['total_time']/m['count'] for m in large_models) / len(large_models):.1f}s
  Avg min RAM:            {sum(m['ram_min'] for m in large_models) / len(large_models):.1f} GB

MEDIUM MODELS (10-20 GB): {len(medium_models)} models
  Avg size:               {sum(m['size_gb'] for m in medium_models) / len(medium_models):.1f} GB
  Avg test time:          {sum(m['total_time']/m['count'] for m in medium_models) / len(medium_models):.1f}s
  Avg min RAM:            {sum(m['ram_min'] for m in medium_models) / len(medium_models):.1f} GB

SMALL MODELS (<10 GB):    {len(small_models)} models
  Avg size:               {sum(m['size_gb'] for m in small_models) / len(small_models):.1f} GB
  Avg test time:          {sum(m['total_time']/m['count'] for m in small_models) / len(small_models):.1f}s
  Avg min RAM:            {sum(m['ram_min'] for m in small_models) / len(small_models):.1f} GB
```
""" if large_models and medium_models and small_models else ""

    md += "\n---\n\n"

    # Per-Test Statistics
    md += "## Per-Test Statistics\n\n"
    md += "Shows performance range across models for each test.\n\n"

    # Sort tests by model count (descending) - most representative tests first
    sorted_tests = sorted(stats['tests'].values(), key=lambda t: t['model_count'], reverse=True)

    # Build comparison lookup for tests
    compare_tests = {}
    if compare_stats:
        compare_tests = {t['name']: t for t in compare_stats['tests'].values()}

    if compare_stats:
        md += f"""```
{'Test Name':<40} {'Models':<7} {'Fastest':<20} {'Slowest':<20} {'Med':<6} {'Old':<6} {'Δ Med':<8}
{'='*40} {'='*7} {'='*20} {'='*20} {'='*6} {'='*6} {'='*8}
"""
    else:
        md += f"""```
{'Test Name':<50} {'Models':<7} {'Fastest':<25} {'Slowest':<25} {'Med Time'}
{'='*50} {'='*7} {'='*25} {'='*25} {'='*8}
"""

    for test in sorted_tests:
        # Shorten test name if needed
        max_test_len = 38 if compare_stats else 48
        test_short = test['name']
        if len(test_short) > max_test_len:
            test_short = test_short[:max_test_len-3] + "..."

        # Format fastest/slowest
        fastest = test['fastest']
        slowest = test['slowest']

        if fastest and slowest:
            max_model_len = 18 if compare_stats else 23
            fastest_str = f"{fastest['model_short']} ({fastest['duration']:.1f}s)"
            slowest_str = f"{slowest['model_short']} ({slowest['duration']:.1f}s)"
            if len(fastest_str) > max_model_len:
                fastest_str = fastest_str[:max_model_len-3] + "..."
            if len(slowest_str) > max_model_len:
                slowest_str = slowest_str[:max_model_len-3] + "..."

            med_time = test['median_time']

            if compare_stats:
                old_test = compare_tests.get(test['name'])
                if old_test:
                    old_med = old_test['median_time']
                    delta_pct = ((med_time - old_med) / old_med * 100) if old_med > 0 else 0
                    delta_str = f"{delta_pct:+.1f}%"
                    md += f"{test_short:<40} {test['model_count']:<7} {fastest_str:<20} {slowest_str:<20} {med_time:<5.1f}s {old_med:<5.1f}s {delta_str:<8}\n"
                else:
                    md += f"{test_short:<40} {test['model_count']:<7} {fastest_str:<20} {slowest_str:<20} {med_time:<5.1f}s {'N/A':<6} {'NEW':<8}\n"
            else:
                md += f"{test_short:<50} {test['model_count']:<7} {fastest_str:<25} {slowest_str:<25} {med_time:.1f}s\n"

    md += "```\n\n"

    md += "\n---\n\n"
    md += "## Files\n\n"
    md += f"- **Benchmark report:** `{input_file}`\n"
    md += f"- **Schema:** `benchmarks/schemas/report-v{stats['schema_version']}.schema.json`\n"

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
