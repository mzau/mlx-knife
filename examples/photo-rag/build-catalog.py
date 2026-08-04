#!/usr/bin/env python3
"""Derive the browsable catalog from the append-only log.

The log is the single source of truth and is only ever appended to. This file is the
view: it can be deleted and rebuilt at any moment, which is exactly what makes the
batch safe to kill — the expensive work is never inside the thing being rewritten.

Safe to run while a batch is still going. It takes no lock, reads with
errors="replace", and tolerates a torn final line, so the worst a live writer can
cost is the one record that was mid-flight.

Usage:
    # rebuild from the default log
    build-catalog.py

    # to stdout, for a pipe
    build-catalog.py -o -

    # look for perceptual duplicates and fill missing metadata from partners
    build-catalog.py --near-dup-distance 4 --inherit-metadata

Input:  $PHOTO_CATALOG/log/captions.jsonl (or --log)
Output: $PHOTO_CATALOG/catalog.jsonl (or -o), one JSON object per described photo

Field names are not free: `examples/rag-server/cosine-search.py` reads `text`,
`filename` and `filepath` by those exact names, so reusing them is what lets the
search stage run against that file unchanged. `filepath` is deliberately
vault-RELATIVE — the absolute root lives in vault.json and nowhere else.
"""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import photo_lib as P  # noqa: E402


def main() -> int:
    os.umask(0o077)
    ap = ArgumentParser(description="Derive the catalog from the caption log")
    ap.add_argument("--log", default=None, help="default: $PHOTO_CATALOG/log/captions.jsonl")
    ap.add_argument("-o", "--output", default=None,
                    help="default: $PHOTO_CATALOG/catalog.jsonl; '-' for stdout")
    ap.add_argument("--include-quarantined", action="store_true")
    ap.add_argument("--min-chars", type=int, default=8,
                    help="drop captions shorter than this (default 8)")
    ap.add_argument("--near-dup-distance", type=int, default=4,
                    help="dHash Hamming distance at which two photos count as the same "
                         "picture (default 4). Reported, never acted on.")
    ap.add_argument("--no-near-dup", action="store_true")
    ap.add_argument("--inherit-metadata", action="store_true",
                    help="fill a missing date/place/camera from a perceptually matched "
                         "partner; recorded separately as exif_inherited, never merged")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    try:
        catalog_dir = None
        if args.log is None or args.output is None:
            catalog_dir = P.require_env("PHOTO_CATALOG")
        log_path = Path(args.log) if args.log else catalog_dir / "log" / "captions.jsonl"
        if args.output == "-":
            out_path = None
        elif args.output:
            out_path = Path(args.output).expanduser()
        else:
            out_path = catalog_dir / "catalog.jsonl"
    except P.Precondition as e:
        P.die(e)

    if not log_path.exists():
        P.die(P.Precondition(P.EXIT_LOG_UNREADABLE, f"no log at {log_path}",
                             "run caption-photos.py first"))

    warn = lambda m: sys.stderr.write(f"  ! {m}\n")  # noqa: E731
    log = P.Log(log_path)

    # Last successful result per photo wins. A failed result never displaces a good
    # one, so a transient error after a success cannot erase the caption; and a
    # re-caption at a different resolution simply supersedes the earlier pass, which
    # is what makes a second higher-resolution run a no-op for everything else.
    best: Dict[str, Dict[str, Any]] = {}
    # Where a photo is NOW is a different question from what was said about it. The
    # caption and its metadata come from the last successful result; the location comes
    # from the latest path event, which may be newer. Keeping them apart is what lets a
    # library be reorganised without re-describing anything.
    latest_rel: Dict[str, str] = {}
    quarantined, torn, results, moved = set(), 0, 0, 0
    for rec in log.read(warn=warn):
        if rec.get("__torn__"):
            torn += 1
            continue
        t = rec.get("type")
        if t == "quarantine":
            quarantined.add(rec.get("photo_id"))
        elif t == "path_update":
            if rec.get("photo_id") and rec.get("rel"):
                latest_rel[rec["photo_id"]] = rec["rel"]
                moved += 1
        elif t == "result" and rec.get("ok"):
            results += 1
            best[rec["photo_id"]] = rec
            if rec.get("rel"):
                latest_rel[rec["photo_id"]] = rec["rel"]

    rows: Dict[str, Dict[str, Any]] = {}
    dropped_short = 0
    for pid, r in best.items():
        if pid in quarantined and not args.include_quarantined:
            continue
        text = (r.get("text") or "").strip()
        if len(text) < args.min_chars:
            dropped_short += 1
            continue
        exif = r.get("exif") or {}
        prepared = r.get("prepared") or {}
        src = r.get("src") or {}
        rel = latest_rel.get(pid, r.get("rel", ""))
        rows[pid] = {
            "photo_id": pid,
            "text": text,
            "filename": Path(rel).name,
            "filepath": rel,                       # vault-relative, never absolute
            "captured": exif.get("dt"),
            "gps": exif.get("gps"),                # a filter input, never embedded
            "camera": exif.get("camera"),
            "w": src.get("w"), "h": src.get("h"),
            "bytes": src.get("bytes"), "branch": src.get("branch"),
            "model": r.get("model"), "prompt_sha256": r.get("prompt_sha256"),
            "captioned_at": r.get("ts"),
            "max_edge": prepared.get("max_edge"),
            "prepared_sha256": prepared.get("sha256"),
            "dhash": r.get("dhash"),
            "truncated": r.get("truncated", False),
        }

    # --- perceptual grouping -------------------------------------------------
    groups: Dict[str, int] = {}
    if not args.no_near_dup:
        items = [(pid, int(row["dhash"], 16)) for pid, row in rows.items() if row.get("dhash")]
        if items:
            groups = P.group_near_duplicates(items, max_distance=args.near_dup_distance)
            for pid, g in groups.items():
                rows[pid]["near_dup_group"] = g

    if args.inherit_metadata and groups:
        inherited = P.inherit_metadata(rows, groups)
        for pid, extra in inherited.items():
            rows[pid]["exif_inherited"] = extra

    # Atomic replace: a reader never sees a half-written catalog, even if this is
    # rebuilt while a search is running.
    lines = [json.dumps(rows[pid], ensure_ascii=True) for pid in sorted(rows)]
    if out_path is None:
        sys.stdout.write("\n".join(lines) + ("\n" if lines else ""))
    else:
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        os.replace(tmp, out_path)

    e = sys.stderr.write
    e(f"\nCatalog\n{'-' * 52}\n")
    e(f"  described photos   {len(rows):>8,}   (from {results:,} successful results)\n")
    if quarantined:
        e(f"  quarantined        {len(quarantined):>8,}   "
          f"{'included' if args.include_quarantined else 'excluded'}\n")
    if dropped_short:
        e(f"  captions too short {dropped_short:>8,}\n")
    if moved:
        e(f"  relocated          {moved:>8,}   photo(s) found at a new path since being described\n")
    if torn:
        e(f"  torn log lines     {torn:>8,}   (their photos will be redone on the next run)\n")
    if groups:
        e(f"  near-duplicate     {len(set(groups.values())):>8,} group(s) covering "
          f"{len(groups)} photos, at distance <= {args.near_dup_distance}\n")
        e("                              reported only — a perceptual hash cannot tell one "
          "shot in two formats from two frames of a burst\n")
    if args.stats:
        with_gps = sum(1 for r in rows.values() if r.get("gps"))
        with_dt = sum(1 for r in rows.values() if r.get("captured"))
        trunc = sum(1 for r in rows.values() if r.get("truncated"))
        e(f"  with coordinates   {with_gps:>8,}\n  with a date        {with_dt:>8,}\n")
        e(f"  possibly truncated {trunc:>8,}\n")
    if out_path is not None:
        e(f"  written to         {out_path}\n")
    e("\n")
    return P.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
