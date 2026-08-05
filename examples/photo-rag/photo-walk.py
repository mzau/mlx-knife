#!/usr/bin/env python3
"""Inventory a photo library: what is there, what will be described, what will not.

No model, no server, no decoding — only names, sizes and 128 KiB per file for the
content identity. Run this before committing days of GPU time: it answers how big the
job is, how many files need conversion, and how many are duplicates, using the same
walker the batch itself uses, so the answer is a measurement rather than an estimate.

Usage:
    # the whole library, written next to the catalog
    photo-walk.py

    # to stdout, for a pipe
    photo-walk.py -o -

    # only what a first pass would touch
    photo-walk.py --include-ext .jpg --include-ext .jpeg

    # keep the raw file and skip its JPEG sibling instead
    photo-walk.py --prefer-raw

Input:  $PHOTO_VAULT (required, no default)
Output: JSONL, one record per file; a summary on stderr
"""

from __future__ import annotations

import json
import os
import sys
import time
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import photo_lib as P  # noqa: E402


def main() -> int:
    os.umask(0o077)
    ap = ArgumentParser(description="Inventory a photo library (no model, no decoding)")
    ap.add_argument("-o", "--output", default=None,
                    help="JSONL output file; '-' for stdout "
                         "(default: $PHOTO_CATALOG/inventory.jsonl)")
    ap.add_argument("--include-ext", action="append", default=None, metavar="EXT",
                    help="repeatable, e.g. --include-ext .jpeg (default: every supported format)")
    ap.add_argument("--exclude-dir", action="append", default=[], metavar="NAME",
                    help="repeatable; matches a directory NAME anywhere, not a path")
    ap.add_argument("--no-pair-dedupe", action="store_true",
                    help="describe both halves of a raw+JPEG pair")
    ap.add_argument("--prefer-raw", action="store_true",
                    help="in a raw+JPEG pair keep the raw file (default: the JPEG)")
    ap.add_argument("--follow-symlinks", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="stop after N candidates")
    ap.add_argument("--quiet", action="store_true", help="summary only, no per-file records")
    ap.add_argument("--progress-every", type=int, default=0, metavar="N",
                    help="print a progress line to stderr every N files (0 = never). "
                         "Worth setting on network storage: the first pass reads nothing "
                         "and prints nothing, so a long silence is the scan and not a stall")
    args = ap.parse_args()

    try:
        vault = P.require_env("PHOTO_VAULT", must_exist=True)
        if args.output is None:
            catalog = P.require_env("PHOTO_CATALOG")
            catalog.mkdir(parents=True, exist_ok=True, mode=0o700)
            out_path = catalog / "inventory.jsonl"
        elif args.output == "-":
            out_path = None
        else:
            out_path = Path(args.output).expanduser()
    except P.Precondition as e:
        P.die(e)

    # Default to a file rather than the terminal on purpose: a full inventory is a
    # complete listing of every path in a private library, which is a far richer
    # disclosure than anything else this tool prints. '-o -' is the explicit opt-in.
    sink = sys.stdout if out_path is None else open(out_path, "w", encoding="utf-8")
    branches, skips, exts = Counter(), Counter(), Counter()
    total_bytes = kept = 0
    seen_files = 0
    t_start = time.time()

    try:
        for c in P.walk(vault,
                        include_ext=args.include_ext,
                        exclude_dir=args.exclude_dir,
                        pair_dedupe=not args.no_pair_dedupe,
                        prefer_raw=args.prefer_raw,
                        follow_symlinks=args.follow_symlinks,
                        limit=args.limit):
            if not args.quiet:
                sink.write(json.dumps(c.record(), ensure_ascii=True) + "\n")
            if c.skip:
                skips[c.skip] += 1
            else:
                kept += 1
                branches[c.branch] += 1
                exts[c.ext] += 1
                total_bytes += c.bytes
            seen_files += 1
            if args.progress_every and seen_files % args.progress_every == 0:
                rate = seen_files / max(time.time() - t_start, 1e-9)
                sys.stderr.write(f"\r  {seen_files} file(s), {kept} candidate(s), "
                                 f"{rate:.0f}/s")
                sys.stderr.flush()
    finally:
        if sink is not sys.stdout:
            sink.close()

    def human_bytes(n: int) -> str:
        for unit, div in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
            if n >= div:
                return f"{n / div:.1f} {unit}"
        return f"{n} B"

    def human_time(sec: float) -> str:
        if sec >= 3600:
            return f"{sec / 3600:.1f} h"
        if sec >= 60:
            return f"{sec / 60:.0f} min"
        return f"{sec:.0f} s"

    e = sys.stderr.write
    e(f"\nVault inventory\n{'-' * 60}\n")
    e(f"  to describe        {kept:>8,}   ({human_bytes(total_bytes)} of originals)\n")
    for br, n in sorted(branches.items()):
        how = "read directly" if br == "direct" else "converted with sips first"
        e(f"    {br:<16} {n:>8,}   {how}\n")
    if exts:
        e(f"  by extension       {', '.join(f'{k} {v:,}' for k, v in sorted(exts.items()))}\n")
    if skips:
        e("  skipped\n")
        for r, n in sorted(skips.items()):
            e(f"    {r:<16} {n:>8,}\n")
    if kept:
        # 4.7 s/photo, measured 2026-08-03 at 512 px against a resident pixtral-12b-4bit
        # over HTTP (range 3.8-5.9). Doubling the edge roughly triples this.
        e(f"  captioning at 512 px would take roughly {human_time(kept * 4.7)} "
          f"at ~4.7 s/photo\n")
    if out_path is not None and not args.quiet:
        e(f"  written to         {out_path}\n")
    e("\n")
    return P.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
