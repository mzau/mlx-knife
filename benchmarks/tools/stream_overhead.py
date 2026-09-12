#!/usr/bin/env python3
"""Streaming overhead gauge - `generate_streaming` against `generate_batch`, same runner.

Generates the same answer twice, streamed and unstreamed, and prints the cost per token
side by side with their ratio. The ratio is the number that travels: absolute throughput
depends on the machine and its thermal state, while streamed and unstreamed are measured
in the same process on the same model, so a ratio far above 1 means streaming is paying
for something the model is not.

Calibration on `Qwen2.5-0.5B-Instruct-4bit`, 300 tokens: **66.5x** against the tree that
carried issue #73 (176.0 ms per streamed token), **1.0x** after the fix (2.7 against 2.6 ms).

Pick a **small model with a large vocabulary** — the sharpest detector, because per-token
overhead scales with the vocabulary while the forward pass scales with the model. On a
large model any such overhead hides inside the forward pass and this gauge goes blind.

Usage (from the development environment; no mlx-chronos, nothing to install):
    python benchmarks/tools/stream_overhead.py --model mlx-community/Qwen2.5-0.5B-Instruct-4bit

The model must be in the Hugging Face cache or the workspace. Occasional tool, deliberately
not part of the test suite: it measures time, and a machine under load would fail it without
anything being wrong with the code.

Platform: macOS + Apple Silicon (MLX requirement)
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEFAULT_PROMPT = "Write four detailed paragraphs about the history and engineering of lighthouses."


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def gpu_busy(samples: int = 5) -> Optional[float]:
    """Mean GPU `Device Utilization %` over a few seconds (ioreg, no sudo)."""
    values = []
    for i in range(samples):
        try:
            out = subprocess.run(
                ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
                capture_output=True, text=True, timeout=20,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        match = re.search(r'"Device Utilization %"=(\d+)', out)
        if match:
            values.append(int(match.group(1)))
        if i < samples - 1:
            time.sleep(1.0)
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Cached model or workspace path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--max-gpu-busy", type=float, default=10.0,
                        help="Refuse to measure above this GPU utilisation (default: 10%%)")
    args = parser.parse_args()

    busy = gpu_busy()
    if busy is not None and busy > args.max_gpu_busy:
        fail(f"GPU is {busy:.0f}% busy before the run (limit {args.max_gpu_busy:.0f}%); close "
             "games and video (browser tabs too) or raise --max-gpu-busy")

    from mlxk2.core.runner import MLXRunner

    gen = dict(max_tokens=args.max_tokens, temperature=0.0)
    with MLXRunner(args.model) as runner:
        # Warm-up: the first generation pays for lazy imports and Metal kernel compilation.
        list(runner.generate_streaming(args.prompt, max_tokens=8, temperature=0.0))

        start = time.perf_counter()
        chunks = list(runner.generate_streaming(args.prompt, **gen))
        streamed_seconds = time.perf_counter() - start
        streamed_text = "".join(chunks)

        start = time.perf_counter()
        batch_text = runner.generate_batch(args.prompt, **gen)
        batch_seconds = time.perf_counter() - start

        tokens = len(runner.tokenizer.encode(batch_text)) or 1

    per_streamed = streamed_seconds * 1000 / tokens
    per_batch = batch_seconds * 1000 / tokens
    ratio = streamed_seconds / batch_seconds if batch_seconds else float("inf")

    print(f"\nmodel   {args.model}")
    print(f"tokens  {tokens}" + (f"   GPU before: {busy:.0f}%" if busy is not None else ""))
    print(f"\n{'':10} {'total':>10} {'per token':>12}")
    print(f"{'streamed':10} {streamed_seconds:9.2f}s {per_streamed:10.1f} ms")
    print(f"{'batch':10} {batch_seconds:9.2f}s {per_batch:10.1f} ms")
    print(f"\nratio   {ratio:.1f}x")

    # Same prompt at temperature 0: a difference is a parity defect, not a timing artefact.
    if streamed_text.strip() != batch_text.strip():
        print("\nwarning: streamed and unstreamed text differ - parity defect, the ratio above "
              "compares two different answers")
    if ratio > 3:
        print("warning: streaming costs more than 3x the unstreamed path; something per token "
              "is dominating the forward pass (see issue #73 for the shape of that bug)")


if __name__ == "__main__":
    main()
