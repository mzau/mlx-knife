#!/usr/bin/env python3
"""Memory/CPU Timeline Visualization - Generate interactive HTML charts from benchmark data.

Correlates memory samples (memmon.py) with test results to show RAM/swap/CPU usage
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

    # CPU data (may not be present in older samples)
    cpu_load = [s.get("load_1", 0) for s in samples]
    cpu_user = [s.get("cpu_user", 0) for s in samples]
    cpu_sys = [s.get("cpu_sys", 0) for s in samples]
    has_cpu_data = any(c > 0 for c in cpu_load) or any(c > 0 for c in cpu_user)

    # GPU data (new in beta.9 memmon)
    gpu_device = [s.get("gpu_device_util", 0) for s in samples]
    gpu_renderer = [s.get("gpu_renderer_util", 0) for s in samples]
    gpu_tiler = [s.get("gpu_tiler_util", 0) for s in samples]
    has_gpu_data = any(g > 0 for g in gpu_device) or any(g > 0 for g in gpu_renderer)

    # Memory pressure data (kern.memorystatus_vm_pressure_level - official macOS levels)
    # Discrete levels: 1=NORMAL, 2=WARN, 4=CRITICAL
    memory_pressure = [s.get("memory_pressure", 1) for s in samples]

    # Convert elapsed to minutes for readability
    elapsed_min = [e / 60 for e in elapsed]

    # Create figure with subplots: Memory (top), CPU (middle), GPU (bottom)
    subplot_count = 1 + (1 if has_cpu_data else 0) + (1 if has_gpu_data else 0)

    if subplot_count == 3:
        # Memory + CPU + GPU
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.45, 0.30, 0.25],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]],
            subplot_titles=("Memory", "CPU", "GPU")
        )
    elif subplot_count == 2:
        # Memory + CPU (legacy behavior)
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
            subplot_titles=("Memory", "CPU")
        )
    else:
        # Memory only
        fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Calculate max elapsed time for later use
    max_elapsed_min = max(elapsed_min) if elapsed_min else 20

    # RAM trace - use marker color based on threshold
    # Color each point based on RAM level (green >32 GB, orange 16-32 GB, red <16 GB)
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
        row=1, col=1,
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
            row=1, col=1,
            secondary_y=True,
        )

    # CPU traces (row 2) - only if CPU data available
    if has_cpu_data:
        # CPU Load (1-min average)
        fig.add_trace(
            go.Scatter(
                x=elapsed_min,
                y=cpu_load,
                mode="lines",
                name="CPU Load (1m)",
                line=dict(color="rgb(142, 68, 173)", width=2),  # Purple
                hovertemplate="Time: %{x:.1f} min<br>Load: %{y:.2f}<extra></extra>",
            ),
            row=2, col=1,
        )

        # CPU User % (filled area)
        fig.add_trace(
            go.Scatter(
                x=elapsed_min,
                y=cpu_user,
                mode="lines",
                name="CPU User %",
                fill="tozeroy",
                line=dict(color="rgb(46, 204, 113)", width=1),  # Green
                fillcolor="rgba(46, 204, 113, 0.3)",
                hovertemplate="Time: %{x:.1f} min<br>User: %{y:.1f}%<extra></extra>",
            ),
            row=2, col=1,
        )

        # CPU Sys % (stacked on top of user)
        cpu_total = [u + s for u, s in zip(cpu_user, cpu_sys)]
        fig.add_trace(
            go.Scatter(
                x=elapsed_min,
                y=cpu_total,
                mode="lines",
                name="CPU Sys %",
                fill="tonexty",
                line=dict(color="rgb(231, 76, 60)", width=1),  # Red
                fillcolor="rgba(231, 76, 60, 0.3)",
                hovertemplate="Time: %{x:.1f} min<br>Sys: %{y:.1f}%<extra></extra>",
            ),
            row=2, col=1,
        )

    # GPU traces (row 3) - only if GPU data available
    if has_gpu_data:
        gpu_row = 2 if not has_cpu_data else 3

        # GPU Device Utilization % (overall GPU busy)
        fig.add_trace(
            go.Scatter(
                x=elapsed_min,
                y=gpu_device,
                mode="lines",
                name="GPU Device %",
                line=dict(color="rgb(255, 127, 14)", width=2),  # Orange
                hovertemplate="Time: %{x:.1f} min<br>GPU Device: %{y:.0f}%<extra></extra>",
            ),
            row=gpu_row, col=1,
        )

        # GPU Renderer Utilization % (3D cores)
        fig.add_trace(
            go.Scatter(
                x=elapsed_min,
                y=gpu_renderer,
                mode="lines",
                name="GPU Renderer %",
                fill="tozeroy",
                line=dict(color="rgb(44, 160, 44)", width=1),  # Green
                fillcolor="rgba(44, 160, 44, 0.3)",
                hovertemplate="Time: %{x:.1f} min<br>GPU Renderer: %{y:.0f}%<extra></extra>",
            ),
            row=gpu_row, col=1,
        )

        # GPU Tiler Utilization % (geometry processing) - only show if different from Renderer
        # On Apple Silicon, Tiler and Renderer are often identical for compute workloads
        tiler_differs = False
        for t, r in zip(gpu_tiler, gpu_renderer):
            if abs(t - r) > 1.0:  # Allow 1% tolerance for floating point
                tiler_differs = True
                break

        if tiler_differs and any(t > 0 for t in gpu_tiler):
            fig.add_trace(
                go.Scatter(
                    x=elapsed_min,
                    y=gpu_tiler,
                    mode="lines",
                    name="GPU Tiler %",
                    line=dict(color="rgb(148, 103, 189)", width=1, dash="dash"),  # Purple dashed
                    hovertemplate="Time: %{x:.1f} min<br>GPU Tiler: %{y:.0f}%<extra></extra>",
                ),
                row=gpu_row, col=1,
            )

    # Model test regions (gray background for each test with model)
    # Sort markers by time
    model_markers_sorted = sorted(model_markers, key=lambda m: m["start_elapsed"])

    test_shapes = []

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

    # Add test markers (vertical lines) and combined labels (test + model)
    # Process model tests and infrastructure tests separately to create combined labels
    for marker in model_markers_sorted:
        start_min = marker["start_elapsed"] / 60

        if start_min < 0 or start_min > max_elapsed_min:
            continue

        # Extract test name (remove test_ prefix, shorten if needed)
        test_name = marker["test"].split("::")[-1].split("[")[0]
        if test_name.startswith("test_"):
            test_name = test_name[5:]  # Remove "test_" prefix
        if len(test_name) > 30:
            test_name = test_name[:27] + "..."

        # Combine test name (left) and model name (right) with spacing
        # When rotated -90°, left becomes top and right becomes bottom
        model_short = marker.get("model_short", "")
        if model_short:
            # Calculate padding to align model name to the right (when vertical)
            # Use fixed width for consistent alignment
            total_width = 35  # characters
            padding = max(0, total_width - len(test_name))
            label = f"{test_name}{' ' * padding}{model_short}"
        else:
            label = test_name

        fig.add_vline(
            x=start_min,
            line=dict(color="rgba(128, 128, 128, 0.2)", width=0.5),
        )

        # Add combined label at top
        fig.add_annotation(
            x=start_min,
            y=1.0,
            xref="x", yref="paper",
            text=label,
            textangle=-90,
            font=dict(size=9, color="rgba(0, 0, 0, 0.7)", family="monospace"),  # Monospace for alignment
            showarrow=False,
            xanchor="left",
            yanchor="top",
            xshift=2,
        )

    # Add infrastructure test markers (no model name)
    for marker in infra_markers_sorted:
        start_min = marker["start_elapsed"] / 60

        if start_min < 0 or start_min > max_elapsed_min:
            continue

        # Extract test name (remove test_ prefix, shorten if needed)
        test_name = marker["test"].split("::")[-1].split("[")[0]
        if test_name.startswith("test_"):
            test_name = test_name[5:]  # Remove "test_" prefix
        if len(test_name) > 30:
            test_name = test_name[:27] + "..."

        fig.add_vline(
            x=start_min,
            line=dict(color="rgba(128, 128, 128, 0.2)", width=0.5),
        )

        # Add test label (no model)
        fig.add_annotation(
            x=start_min,
            y=1.0,
            xref="x", yref="paper",
            text=test_name,
            textangle=-90,
            font=dict(size=9, color="rgba(0, 0, 0, 0.7)", family="monospace"),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            xshift=2,
        )

    # Add memory pressure background zones based on official macOS levels
    # kern.memorystatus_vm_pressure_level: 1=NORMAL, 2=WARN, 4=CRITICAL
    pressure_shapes = []
    i = 0
    while i < len(memory_pressure):
        level_value = memory_pressure[i]

        # Map to level name
        if level_value == 4:
            level = "CRITICAL"
        elif level_value == 2:
            level = "WARN"
        else:  # 1 or other
            level = "NORMAL"

        if level != "NORMAL":  # Only show overlay for WARN/CRITICAL
            # Find end of this pressure region (same level)
            start_min = elapsed_min[i]
            j = i
            while j < len(memory_pressure) and memory_pressure[j] == level_value:
                j += 1
            end_min = elapsed_min[j - 1] if j > i else start_min

            # Color based on pressure level
            if level == "WARN":
                color = "rgba(255, 204, 0, 0.15)"  # Yellow
            else:  # CRITICAL
                color = "rgba(255, 59, 48, 0.15)"  # Red

            pressure_shapes.append(dict(
                type="rect",
                xref="x", yref="y",
                x0=start_min, x1=end_min,
                y0=0, y1=70,
                fillcolor=color,
                layer="below",
                line=dict(width=0),
            ))
            i = j
        else:
            i += 1

    # Combine all shapes (regions first, then pressure on top)
    shapes = region_shapes + pressure_shapes

    # Layout (without shapes - we'll add them individually)
    chart_height = 500 + (200 if has_cpu_data else 0) + (150 if has_gpu_data else 0)

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16),
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
        height=chart_height,
        margin=dict(t=80, b=60, l=60, r=60),
    )

    # Memory subplot (row 1) y-axis
    fig.update_yaxes(
        title_text="RAM Free (GB)",
        showgrid=True,
        gridcolor="rgba(128,128,128,0.2)",
        range=[0, 70],
        row=1, col=1,
        secondary_y=False,
    )

    # Secondary y-axis: Swap (if present)
    if any(s > 0 for s in swap_used):
        fig.update_yaxes(
            title_text="Swap (MB)",
            showgrid=False,
            range=[0, max(swap_used) * 1.2],
            row=1, col=1,
            secondary_y=True,
        )

    # CPU subplot (row 2) y-axis - only if CPU data available
    if has_cpu_data:
        fig.update_yaxes(
            title_text="CPU %",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            range=[0, 100],
            row=2, col=1,
        )

    # GPU subplot (row 3 or 2) y-axis - only if GPU data available
    if has_gpu_data:
        gpu_row = 2 if not has_cpu_data else 3
        fig.update_yaxes(
            title_text="GPU %",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            range=[0, 100],
            row=gpu_row, col=1,
        )

    # X-axis (on bottom subplot only)
    if has_gpu_data:
        bottom_row = 2 if not has_cpu_data else 3
    elif has_cpu_data:
        bottom_row = 2
    else:
        bottom_row = 1

    fig.update_xaxes(
        title_text="Time (minutes)",
        showgrid=True,
        gridcolor="rgba(128,128,128,0.2)",
        row=bottom_row, col=1,
    )

    # Add rangeslider for zoom navigation on all layouts (horizontal-only zoom)
    # The rangeslider shows a miniature overview and allows horizontal panning/zooming
    fig.update_xaxes(
        rangeslider=dict(
            visible=True,
            thickness=0.05,  # Compact rangeslider (5% of plot height)
        ),
        row=bottom_row, col=1,
    )

    # Disable vertical zoom on all subplots (horizontal zoom only)
    fig.update_yaxes(fixedrange=True)

    # Add shapes individually using fig.add_shape() method
    # This is more explicit than passing shapes array to update_layout
    for shape in shapes:
        fig.add_shape(**shape)

    # Add summary annotation
    if summary:
        summary_text = (
            f"Duration: {summary.get('duration_s', 0)/60:.1f} min | "
            f"Samples: {summary.get('samples', 0)} | "
            f"RAM: {summary.get('ram_free_min_gb', 0):.1f}-{summary.get('ram_free_max_gb', 0):.1f} GB | "
            f"Swap peak: {summary.get('swap_max_mb', 0):.0f} MB"
        )
        # Add CPU summary if available
        if summary.get('load_max', 0) > 0:
            summary_text += (
                f" | CPU load: max {summary.get('load_max', 0):.1f} | "
                f"CPU: max {summary.get('cpu_user_max', 0):.0f}%/{summary.get('cpu_sys_max', 0):.0f}%"
            )
        # Add GPU summary if available
        if summary.get('gpu_device_max', 0) > 0:
            summary_text += (
                f" | GPU: max {summary.get('gpu_device_max', 0):.0f}% (device), "
                f"{summary.get('gpu_renderer_max', 0):.0f}% (renderer)"
            )

        # Calculate y position based on subplot count
        subplot_count = 1 + (1 if has_cpu_data else 0) + (1 if has_gpu_data else 0)
        y_offset = {1: -0.12, 2: -0.08, 3: -0.06}[subplot_count]

        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0, y=y_offset,
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
