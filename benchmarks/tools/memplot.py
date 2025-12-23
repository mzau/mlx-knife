#!/usr/bin/env python3
"""Memory Timeline Visualization - Generate interactive HTML charts from benchmark data.

Correlates memory samples (memmon.py) with test results to show RAM/swap usage
over time with model markers.

Usage:
    # Basic usage
    python benchmarks/tools/memplot.py memory.jsonl benchmark.jsonl

    # Custom output
    python benchmarks/tools/memplot.py memory.jsonl benchmark.jsonl -o report.html

    # PNG export (requires kaleido)
    python benchmarks/tools/memplot.py memory.jsonl benchmark.jsonl --format png

Requires: plotly (pip install plotly)
Optional: kaleido (pip install kaleido) for PNG export
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def parse_memory_samples(path: Path) -> tuple[list[dict], dict]:
    """Parse memmon JSONL output.

    Returns:
        Tuple of (samples list, summary dict)
    """
    samples = []
    summary = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "summary" in entry:
                summary = entry["summary"]
            else:
                samples.append(entry)

    return samples, summary


def parse_benchmark_results(path: Path) -> tuple[list[dict], list[dict]]:
    """Parse benchmark JSONL output.

    Returns:
        Tuple of (tests with models, tests without models)
    """
    tests_with_model = []
    tests_without_model = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "timestamp" not in entry or "duration" not in entry:
                continue

            if "model" in entry and entry.get("outcome") == "passed":
                tests_with_model.append(entry)
            elif "model" not in entry and entry.get("outcome") == "passed":
                tests_without_model.append(entry)

    return tests_with_model, tests_without_model


def parse_iso_timestamp(ts_str: str) -> float:
    """Convert ISO timestamp to Unix timestamp."""
    # Handle timezone suffix
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts_str)
    return dt.timestamp()


def correlate_tests_with_timeline(
    samples: list[dict],
    tests: list[dict],
    memory_start_ts: float
) -> list[dict]:
    """Calculate test time ranges relative to memory timeline.

    Returns:
        List of dicts with model_id, start_elapsed, end_elapsed
    """
    if not samples or not tests:
        return []

    markers = []

    for test in tests:
        if "timestamp" not in test or "duration" not in test:
            continue

        test_end_ts = parse_iso_timestamp(test["timestamp"])
        test_start_ts = test_end_ts - test["duration"]

        # Convert to elapsed time relative to memory monitoring start
        start_elapsed = test_start_ts - memory_start_ts
        end_elapsed = test_end_ts - memory_start_ts

        # Get model info if available (for model tests)
        model_id = test.get("model", {}).get("id", None)
        model_short = model_id.split("/")[-1][:20] if model_id else None

        markers.append({
            "model_id": model_id,
            "model_short": model_short,
            "start_elapsed": start_elapsed,
            "end_elapsed": end_elapsed,
            "duration": test["duration"],
            "test": test.get("test", ""),
        })

    return markers


def get_ram_color(ram_free_gb: float) -> str:
    """Get color based on RAM availability."""
    if ram_free_gb >= 32:
        return "rgb(52, 199, 89)"   # Green - healthy
    elif ram_free_gb >= 16:
        return "rgb(255, 149, 0)"   # Orange - warning
    else:
        return "rgb(255, 59, 48)"   # Red - critical


def create_timeline_chart(
    samples: list[dict],
    summary: dict,
    model_markers: list[dict],
    infra_markers: list[dict],
    title: str = "Memory Timeline"
) -> "Figure":
    """Create interactive plotly timeline chart."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Error: plotly not installed. Run: pip install plotly")
        sys.exit(1)

    # Extract data series
    elapsed = [s["elapsed_s"] for s in samples]
    ram_free = [s["ram_free_gb"] for s in samples]
    swap_used = [s["swap_used_mb"] for s in samples]
    memory_pressure = [s.get("memory_pressure", 1) for s in samples]  # Default: 1=NORMAL

    # Convert elapsed to minutes for readability
    elapsed_min = [e / 60 for e in elapsed]

    # Create figure with secondary y-axis for swap
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # RAM trace - use marker color based on threshold
    # Color each point based on RAM level
    colors = [get_ram_color(ram) for ram in ram_free]

    fig.add_trace(
        go.Scatter(
            x=elapsed_min,
            y=ram_free,
            mode="lines+markers",
            name="RAM Free (GB)",
            line=dict(color="rgb(52, 150, 235)", width=1.5),  # Blue line
            marker=dict(
                color=colors,
                size=3,
                line=dict(width=0),
            ),
            hovertemplate="Time: %{x:.1f} min<br>RAM Free: %{y:.1f} GB<extra></extra>",
        ),
        secondary_y=False,
    )

    # Threshold lines (assuming 64 GB total RAM)
    max_elapsed_min = max(elapsed_min) if elapsed_min else 20
    total_ram = 64  # GB - could be made configurable later

    fig.add_trace(
        go.Scatter(
            x=[0, max_elapsed_min],
            y=[32, 32],
            mode="lines",
            name=f"32 GB (50% of {total_ram} GB - healthy)",
            line=dict(color="green", width=1, dash="dash"),
            hoverinfo="skip",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=[0, max_elapsed_min],
            y=[16, 16],
            mode="lines",
            name=f"16 GB (25% of {total_ram} GB - warning)",
            line=dict(color="orange", width=1, dash="dash"),
            hoverinfo="skip",
        ),
        secondary_y=False,
    )

    # Swap trace (secondary y-axis)
    if any(s > 0 for s in swap_used):
        fig.add_trace(
            go.Scatter(
                x=elapsed_min,
                y=swap_used,
                mode="lines",
                name="Swap Used (MB)",
                line=dict(color="red", width=2),
                hovertemplate="Time: %{x:.1f} min<br>Swap: %{y:.0f} MB<extra></extra>",
            ),
            secondary_y=True,
        )

    # Model test regions (gray background for each test with model)
    # Sort markers by time
    model_markers_sorted = sorted(model_markers, key=lambda m: m["start_elapsed"])

    test_shapes = []
    prev_model_id = None  # Track previous model for switch detection

    for i, marker in enumerate(model_markers_sorted):
        start_min = marker["start_elapsed"] / 60
        end_min = marker["end_elapsed"] / 60

        if start_min < 0 or start_min > max_elapsed_min:
            continue

        # Add gray rectangle for this individual test
        test_shapes.append(dict(
            type="rect",
            xref="x", yref="y",
            x0=start_min,
            x1=end_min,
            y0=0, y1=70,
            fillcolor="rgba(200, 200, 200, 0.3)",  # Gray for model tests
            layer="below",
            line=dict(width=0),
        ))

        # Add model label when model CHANGES (not just first occurrence)
        model_id = marker["model_id"]
        if model_id != prev_model_id:
            fig.add_annotation(
                x=start_min,
                y=1.0,
                xref="x", yref="paper",
                text=marker["model_short"],
                textangle=-90,
                font=dict(size=9, color="rgba(0, 0, 0, 0.7)"),
                showarrow=False,
                xanchor="left",
                yanchor="top",
                xshift=2,
            )
            prev_model_id = model_id

    # Infrastructure test regions (light blue background)
    infra_markers_sorted = sorted(infra_markers, key=lambda m: m["start_elapsed"])

    for marker in infra_markers_sorted:
        start_min = marker["start_elapsed"] / 60
        end_min = marker["end_elapsed"] / 60

        if start_min < 0 or start_min > max_elapsed_min:
            continue

        # Add very light blue rectangle for infrastructure tests
        test_shapes.append(dict(
            type="rect",
            xref="x", yref="y",
            x0=start_min,
            x1=end_min,
            y0=0, y1=70,
            fillcolor="rgba(173, 216, 230, 0.2)",  # Very light blue for infra tests
            layer="below",
            line=dict(width=0),
        ))

    region_shapes = test_shapes

    # Add test markers (small vertical lines) and labels at bottom for both marker types
    all_markers = model_markers_sorted + infra_markers_sorted
    all_markers_sorted = sorted(all_markers, key=lambda m: m["start_elapsed"])

    for marker in all_markers_sorted:
        start_min = marker["start_elapsed"] / 60

        if start_min < 0 or start_min > max_elapsed_min:
            continue

        # Extract test name (shorten if needed)
        test_name = marker["test"].split("::")[-1].split("[")[0]
        if len(test_name) > 25:
            test_name = test_name[:22] + "..."

        fig.add_vline(
            x=start_min,
            line=dict(color="rgba(128, 128, 128, 0.2)", width=0.5),
        )

        # Add test label at bottom (aligned with start time like model labels)
        fig.add_annotation(
            x=start_min,
            y=0.0,
            xref="x", yref="paper",
            text=test_name,
            textangle=-90,
            font=dict(size=9, color="rgba(0, 0, 0, 0.6)"),  # Same size as model labels
            showarrow=False,
            xanchor="left",  # Same as model labels (aligned at start)
            yanchor="bottom",
            xshift=2,  # Same offset as model labels
        )

    # Add memory pressure backgrounds (1=normal/white, 2=warn/yellow, 4=critical/red)
    pressure_shapes = []
    i = 0
    while i < len(memory_pressure):
        pressure = memory_pressure[i]

        if pressure > 1:  # 2=WARN or 4=CRITICAL
            # Find end of this pressure region
            start_min = elapsed_min[i]
            j = i
            while j < len(memory_pressure) and memory_pressure[j] == pressure:
                j += 1
            end_min = elapsed_min[j - 1] if j > i else start_min

            # Color based on pressure level
            if pressure == 2:
                color = "rgba(255, 204, 0, 0.15)"  # Yellow (WARN)
            else:  # pressure == 4
                color = "rgba(255, 59, 48, 0.15)"  # Red (CRITICAL)

            pressure_shapes.append(dict(
                type="rect",
                xref="x", yref="y",  # Changed from "paper" to "y" for rangeslider compatibility
                x0=start_min, x1=end_min,
                y0=0, y1=70,  # Use actual y-axis values
                fillcolor=color,
                layer="below",
                line=dict(width=0),
            ))
            i = j
        else:
            i += 1

    # Combine all shapes (regions first, then pressure on top)
    shapes = region_shapes + pressure_shapes

    # Debug output
    print(f"  Test shapes (gray): {len(region_shapes)}")
    print(f"  Pressure shapes (yellow/red): {len(pressure_shapes)}")
    print(f"  Total shapes: {len(shapes)}")
    if region_shapes:
        print(f"  Sample test shape: {region_shapes[0]}")

    # Layout (without shapes - we'll add them individually)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Time (minutes)",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            rangeslider=dict(visible=True, yaxis=dict(rangemode="match")),
        ),
        yaxis=dict(
            title="RAM Free (GB)",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            range=[0, 70],  # Typical max for 64GB system
        ),
        yaxis2=dict(
            title="Swap Used (MB)",
            showgrid=False,
            range=[0, max(swap_used) * 1.2] if any(s > 0 for s in swap_used) else [0, 100],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background so shapes show through
        height=500,
        margin=dict(t=80, b=60, l=60, r=60),
    )

    # Add shapes individually using fig.add_shape() method
    # This is more explicit than passing shapes array to update_layout
    for shape in shapes:
        fig.add_shape(**shape)

    # Debug: Check shapes after adding individually
    print(f"  Shapes in fig.layout after add_shape: {len(fig.layout.shapes)}")

    # Add summary annotation
    if summary:
        summary_text = (
            f"Duration: {summary.get('duration_s', 0)/60:.1f} min | "
            f"Samples: {summary.get('samples', 0)} | "
            f"RAM: {summary.get('ram_free_min_gb', 0):.1f}-{summary.get('ram_free_max_gb', 0):.1f} GB | "
            f"Swap peak: {summary.get('swap_max_mb', 0):.0f} MB"
        )
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0, y=-0.12,
            showarrow=False,
            font=dict(size=10, color="gray"),
            align="left",
        )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate memory timeline visualization from benchmark data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "memory_file",
        type=Path,
        help="Memory samples JSONL from memmon.py",
    )
    parser.add_argument(
        "benchmark_file",
        type=Path,
        nargs="?",
        help="Benchmark results JSONL (optional, for model markers)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file (default: memory_timeline.html)",
    )
    parser.add_argument(
        "--format",
        choices=["html", "png", "svg"],
        default="html",
        help="Output format (default: html)",
    )
    parser.add_argument(
        "--title",
        default="Memory Timeline",
        help="Chart title",
    )

    args = parser.parse_args()

    # Default output filename
    if not args.output:
        args.output = Path(f"memory_timeline.{args.format}")

    # Parse inputs
    print(f"Reading memory samples: {args.memory_file}")
    samples, summary = parse_memory_samples(args.memory_file)
    print(f"  Found {len(samples)} samples")

    model_markers = []
    infra_markers = []
    if args.benchmark_file:
        print(f"Reading benchmark results: {args.benchmark_file}")
        tests_with_model, tests_without_model = parse_benchmark_results(args.benchmark_file)
        print(f"  Found {len(tests_with_model)} test entries with models")
        print(f"  Found {len(tests_without_model)} infrastructure test entries")

        # Get memory start timestamp from first sample
        if samples:
            memory_start_ts = samples[0]["ts"]
            model_markers = correlate_tests_with_timeline(samples, tests_with_model, memory_start_ts)
            infra_markers = correlate_tests_with_timeline(samples, tests_without_model, memory_start_ts)
            print(f"  Correlated {len(model_markers)} model test markers")
            print(f"  Correlated {len(infra_markers)} infrastructure test markers")

    # Create chart
    print(f"Generating {args.format.upper()} chart...")
    fig = create_timeline_chart(samples, summary, model_markers, infra_markers, title=args.title)

    # Export
    if args.format == "html":
        fig.write_html(
            args.output,
            include_plotlyjs="cdn",
            full_html=True,
        )
    else:
        try:
            fig.write_image(args.output, scale=2)
        except Exception as e:
            print(f"Error: PNG/SVG export requires kaleido: pip install kaleido")
            print(f"Details: {e}")
            sys.exit(1)

    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
