#!/usr/bin/env python3
"""Embed the captions, producing a searchable index.

Feeds the catalog into `mlxk embed --batch`, which embeds each line's `text` field and
passes every other field through untouched. That passthrough is the whole trick: the
index line is the catalog line plus an `embedding` and the `metadata` stamp that
identifies which model produced it, so nothing has to be joined back together later.

Only the caption is embedded. Coordinates, capture dates and camera names travel on the
same line as fields, never inside the embedded text: a coordinate folded into a vector
cannot be removed again without rebuilding the entire index.

Usage:
    index-photos.py
    index-photos.py --model Qwen3-Embedding-0.6B-4bit-DWQ
    build-catalog.py -o - | index-photos.py - -o index.jsonl

Input:  $PHOTO_CATALOG/catalog.jsonl (or a path, or '-')
Output: $PHOTO_CATALOG/index.jsonl (or -o), consumable by photo-search.py and by
        examples/rag-server/cosine-search.py unchanged
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import photo_lib as P  # noqa: E402

DEFAULT_EMBED_MODEL = "bge-small-en-v1.5-4bit"


def embed_chunk(model: str, lines: List[str], cpu: bool) -> List[Dict[str, Any]]:
    """One `mlxk embed --batch` subprocess over a chunk of catalog lines.

    The alpha gate is set per subprocess rather than exported: `embed` is experimental
    in 2.0.7, and a global export would silently arm every other mlxk call in the shell.
    (Same idiom as examples/rag-server/index-files.py.)
    """
    cmd = ["mlxk", "embed", model, "-", "--batch"] + (["--cpu"] if cpu else [])
    r = subprocess.run(cmd, input="\n".join(lines), text=True, capture_output=True,
                       env={**os.environ, "MLXK2_ENABLE_ALPHA_FEATURES": "1"})
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:800])
    out = []
    for n, line in enumerate(r.stdout.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"mlxk embed emitted unparseable JSONL on line {n}: {e}") from e
    return out


def main() -> int:
    os.umask(0o077)
    ap = ArgumentParser(description="Embed captions into a searchable index")
    ap.add_argument("catalog", nargs="?", default=None,
                    help="catalog JSONL, or '-' for stdin (default: $PHOTO_CATALOG/catalog.jsonl)")
    ap.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("-o", "--output", default=None,
                    help="default: $PHOTO_CATALOG/index.jsonl; '-' for stdout")
    ap.add_argument("--chunk", type=int, default=512,
                    help="catalog lines per embed subprocess (default 512). Bounds process "
                         "memory and makes a kill cost one chunk instead of the whole index; "
                         "the embedder itself runs one text at a time either way.")
    ap.add_argument("--cpu", action="store_true",
                    help="CPU vectors differ numerically from GPU ones — never mix them in "
                         "one index; the same-model guard will refuse the mixture anyway")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    try:
        catalog_dir = None
        if args.catalog is None or args.output is None:
            catalog_dir = P.require_env("PHOTO_CATALOG")
        src = None if args.catalog == "-" else Path(args.catalog or catalog_dir / "catalog.jsonl")
        if args.output == "-":
            out_path = None
        elif args.output:
            out_path = Path(args.output).expanduser()
        else:
            out_path = catalog_dir / "index.jsonl"
    except P.Precondition as e:
        P.die(e)

    if src is None:
        raw = sys.stdin.read().splitlines()
    else:
        if not src.exists():
            P.die(P.Precondition(P.EXIT_USAGE, f"no catalog at {src}", "run build-catalog.py first"))
        raw = src.read_text(encoding="utf-8", errors="replace").splitlines()

    lines = [ln for ln in (x.strip() for x in raw) if ln]
    if args.limit:
        lines = lines[: args.limit]
    if not lines:
        P.die(P.Precondition(P.EXIT_USAGE, "the catalog is empty", "nothing to index"))

    records: List[Dict[str, Any]] = []
    identity = None
    for start in range(0, len(lines), args.chunk):
        chunk = lines[start:start + args.chunk]
        sys.stderr.write(f"  embedding {start + 1}-{start + len(chunk)} of {len(lines)}\n")
        try:
            got = embed_chunk(args.model, chunk, args.cpu)
        except RuntimeError as e:
            P.die(P.Precondition(
                P.EXIT_EMBED, f"mlxk embed failed on chunk starting at line {start + 1}",
                str(e)[:500]))
        if len(got) != len(chunk):
            P.die(P.Precondition(
                P.EXIT_EMBED,
                f"mlxk embed returned {len(got)} records for {len(chunk)} inputs",
                "the index would no longer correspond to the catalog; refusing to write it"))
        for rec in got:
            ident = tuple((rec.get("metadata") or {}).get(k)
                          for k in ("model", "content_hash", "device", "dimensions"))
            if identity is None:
                identity = ident
            elif ident != identity:
                # One index, one vector space. Vectors from two models or two devices
                # share no geometry; ranking them together produces confident nonsense.
                P.die(P.Precondition(
                    P.EXIT_EMBED, "the embedding identity changed mid-index",
                    "every line must come from the same model, revision and device"))
        records.extend(got)

    payload = "\n".join(json.dumps(r, ensure_ascii=True) for r in records) + "\n"
    if out_path is None:
        sys.stdout.write(payload)
    else:
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, out_path)

    dims = (records[0].get("metadata") or {}).get("dimensions") if records else None
    sys.stderr.write(f"\nIndex\n{'-' * 52}\n")
    sys.stderr.write(f"  vectors            {len(records):>8,}   {dims}-dimensional\n")
    sys.stderr.write(f"  identity           {identity[0]} @ {identity[2]}\n")
    if out_path is not None:
        sys.stderr.write(f"  written to         {out_path}\n")
    sys.stderr.write("\n")
    return P.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
