#!/usr/bin/env python3
"""Memory Monitor - Standalone tool for tracking memory during subprocess execution.

Samples RAM, swap, and memory pressure while running any command.
Outputs JSONL with per-sample data and final summary.

Usage:
    # Basic usage
    python benchmarks/tools/memmon.py -- pytest -m live_e2e tests_2.0/live/

    # With options
    python benchmarks/tools/memmon.py --interval 200 --output memory.jsonl -- pytest -v

    # Just monitor (no subprocess)
    python benchmarks/tools/memmon.py --duration 60 --output memory.jsonl

Future: Will be part of mlxk-benchmark kit.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def get_memory_sample() -> dict:
    """Get current memory state using psutil."""
    try:
        import psutil
        import subprocess

        # Get memory pressure from sysctl (macOS only)
        # Values: 1=NORMAL (green), 2=WARN (yellow), 4=CRITICAL (red)
        memory_pressure = 1  # Default to NORMAL
        try:
            result = subprocess.run(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                capture_output=True, text=True, timeout=1
            )
            memory_pressure = int(result.stdout.strip())
        except Exception:
            pass

        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "ram_free_gb": round(vm.available / 1e9, 2),
            "ram_used_gb": round(vm.used / 1e9, 2),
            "ram_percent": vm.percent,
            "swap_used_mb": round(swap.used / 1e6, 1),
            "swap_percent": swap.percent,
            "memory_pressure": memory_pressure,
        }
    except ImportError:
        # Fallback without psutil
        return get_memory_sample_native()


def get_memory_sample_native() -> dict:
    """Get memory state using native macOS commands (no psutil)."""
    import subprocess

    # Get memory pressure (1=NORMAL/green, 2=WARN/yellow, 4=CRITICAL/red)
    memory_pressure = 1  # Default to NORMAL
    try:
        result = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True, text=True, timeout=1
        )
        memory_pressure = int(result.stdout.strip())
    except Exception:
        pass

    # Get swap usage
    swap_mb = 0
    try:
        result = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            capture_output=True, text=True, timeout=1
        )
        # Parse: "total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)"
        for part in result.stdout.split():
            if part.endswith("M") and "used" in result.stdout.split()[result.stdout.split().index(part)-2]:
                swap_mb = float(part[:-1])
                break
        # Simpler parsing
        parts = result.stdout.replace("M", "").split()
        for i, p in enumerate(parts):
            if p == "used" and i + 2 < len(parts):
                swap_mb = float(parts[i + 2])
                break
    except Exception:
        pass

    # Get RAM via vm_stat
    ram_free_gb = 0
    try:
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True, text=True, timeout=1
        )
        # Parse page size and available pages
        page_size = 16384  # Default for Apple Silicon
        pages_free = 0
        pages_inactive = 0
        pages_purgeable = 0
        pages_speculative = 0

        for line in result.stdout.splitlines():
            if "page size of" in line:
                page_size = int(line.split()[-2])
            elif "Pages free:" in line:
                pages_free = int(line.split()[-1].rstrip("."))
            elif "Pages inactive:" in line:
                pages_inactive = int(line.split()[-1].rstrip("."))
            elif "Pages purgeable:" in line:
                pages_purgeable = int(line.split()[-1].rstrip("."))
            elif "Pages speculative:" in line:
                pages_speculative = int(line.split()[-1].rstrip("."))

        # Total available = free + inactive + purgeable + speculative
        total_available_pages = pages_free + pages_inactive + pages_purgeable + pages_speculative
        ram_free_gb = round((total_available_pages * page_size) / 1e9, 2)
    except Exception:
        pass

    return {
        "ram_free_gb": ram_free_gb,
        "ram_used_gb": 0,  # Not available without psutil
        "ram_percent": 0,
        "swap_used_mb": swap_mb,
        "swap_percent": 0,
        "memory_pressure": memory_pressure,
    }


class MemoryMonitor:
    """Background memory sampler.

    Usage:
        monitor = MemoryMonitor(interval_ms=200)
        monitor.start()
        # ... do work ...
        summary = monitor.stop()
    """

    def __init__(self, interval_ms: int = 200):
        self.interval = interval_ms / 1000
        self.samples: list[dict] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.start_time: float = 0

    def start(self):
        """Start background sampling."""
        self.running = True
        self.samples = []
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict:
        """Stop sampling and return summary."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

        if not self.samples:
            return {"error": "No samples collected"}

        ram_values = [s["ram_free_gb"] for s in self.samples]
        swap_values = [s["swap_used_mb"] for s in self.samples]

        return {
            "duration_s": round(time.time() - self.start_time, 2),
            "samples": len(self.samples),
            "interval_ms": int(self.interval * 1000),
            "ram_free_min_gb": min(ram_values),
            "ram_free_max_gb": max(ram_values),
            "ram_free_avg_gb": round(sum(ram_values) / len(ram_values), 2),
            "swap_max_mb": max(swap_values),
            "swap_avg_mb": round(sum(swap_values) / len(swap_values), 1),
        }

    def get_samples(self) -> list[dict]:
        """Get all collected samples."""
        return self.samples.copy()

    def _sample_loop(self):
        """Background sampling loop."""
        while self.running:
            sample = get_memory_sample()
            sample["ts"] = round(time.time(), 3)
            sample["elapsed_s"] = round(time.time() - self.start_time, 2)
            self.samples.append(sample)
            time.sleep(self.interval)


def run_with_monitoring(
    command: list[str],
    interval_ms: int = 200,
    output_file: Optional[Path] = None,
    verbose: bool = False
) -> dict:
    """Run a command while monitoring memory.

    Args:
        command: Command and arguments to run
        interval_ms: Sampling interval in milliseconds
        output_file: Optional JSONL output file
        verbose: Print samples as they're collected

    Returns:
        Summary dict with memory statistics
    """
    monitor = MemoryMonitor(interval_ms=interval_ms)

    print(f"Starting memory monitor (interval: {interval_ms}ms)")
    print(f"Running: {' '.join(command)}")
    print("-" * 60)

    monitor.start()

    # Run subprocess
    try:
        result = subprocess.run(command)
        exit_code = result.returncode
    except KeyboardInterrupt:
        exit_code = 130
        print("\nInterrupted")
    except Exception as e:
        exit_code = 1
        print(f"\nError: {e}")

    summary = monitor.stop()
    summary["exit_code"] = exit_code
    summary["command"] = " ".join(command)
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()

    print("-" * 60)
    print(f"Memory Monitor Summary:")
    print(f"  Duration:     {summary['duration_s']:.1f}s ({summary['samples']} samples)")
    print(f"  RAM free:     {summary['ram_free_min_gb']:.1f} - {summary['ram_free_max_gb']:.1f} GB")
    print(f"  Swap peak:    {summary['swap_max_mb']:.1f} MB")
    print(f"  Exit code:    {exit_code}")

    # Write output
    if output_file:
        with open(output_file, "w") as f:
            # Write samples
            for sample in monitor.get_samples():
                f.write(json.dumps(sample) + "\n")
            # Write summary as last line
            f.write(json.dumps({"summary": summary}) + "\n")
        print(f"  Output:       {output_file}")

    return summary


def monitor_only(
    duration_s: float,
    interval_ms: int = 200,
    output_file: Optional[Path] = None
) -> dict:
    """Monitor memory for a fixed duration (no subprocess).

    Args:
        duration_s: How long to monitor
        interval_ms: Sampling interval in milliseconds
        output_file: Optional JSONL output file

    Returns:
        Summary dict with memory statistics
    """
    monitor = MemoryMonitor(interval_ms=interval_ms)

    print(f"Monitoring memory for {duration_s}s (interval: {interval_ms}ms)")
    print("-" * 60)

    monitor.start()

    try:
        time.sleep(duration_s)
    except KeyboardInterrupt:
        print("\nInterrupted")

    summary = monitor.stop()
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()

    print("-" * 60)
    print(f"Memory Monitor Summary:")
    print(f"  Duration:     {summary['duration_s']:.1f}s ({summary['samples']} samples)")
    print(f"  RAM free:     {summary['ram_free_min_gb']:.1f} - {summary['ram_free_max_gb']:.1f} GB")
    print(f"  Swap peak:    {summary['swap_max_mb']:.1f} MB")

    if output_file:
        with open(output_file, "w") as f:
            for sample in monitor.get_samples():
                f.write(json.dumps(sample) + "\n")
            f.write(json.dumps({"summary": summary}) + "\n")
        print(f"  Output:       {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Monitor memory while running a command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=200,
        help="Sampling interval in milliseconds (default: 200)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output JSONL file for samples and summary"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        help="Monitor for fixed duration (seconds), no subprocess"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print samples as they're collected"
    )
    parser.add_argument(
        "command",
        nargs="*",
        help="Command to run (after --)"
    )

    args = parser.parse_args()

    if args.duration:
        # Monitor-only mode
        summary = monitor_only(
            duration_s=args.duration,
            interval_ms=args.interval,
            output_file=args.output
        )
    elif args.command:
        # Run command with monitoring
        summary = run_with_monitoring(
            command=args.command,
            interval_ms=args.interval,
            output_file=args.output,
            verbose=args.verbose
        )
        sys.exit(summary.get("exit_code", 0))
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python benchmarks/tools/memmon.py -- pytest -m live_e2e")
        print("  python benchmarks/tools/memmon.py --duration 10 --output mem.jsonl")
        sys.exit(1)


if __name__ == "__main__":
    main()
