#!/usr/bin/env python3
"""Search the catalog by description, by date and by place.

The text half is delegated to `examples/rag-server/cosine-search.py`, unchanged. That
file already refuses to rank vectors whose embedding identity does not match — same
model, revision, device and dimensions — and that guard is worth more than a private
copy of a dot product. This script embeds the query in QUERY mode (the index side stays
in document mode, which is the asymmetry a retrieval embedder needs), applies the
metadata filters, and prints the result.

Coordinates and absolute paths are printed by default: this is your own metadata about
your own photographs, on your own terminal, read from your own local index. What stays
out of the *stored* artefacts is a different question with a different answer — a
coordinate folded into an embedding cannot be removed again without rebuilding
everything, so it never enters the embedded text. Use --no-gps and --relative-paths
when the output is going somewhere you did not write.

Usage:
    photo-search.py "a paddle steamer at a pier"
    photo-search.py "snow" --since 2019-01-01 --until 2019-03-31
    photo-search.py "boats" --near 59.33,18.07 --radius-km 5 --top-k 10
    photo-search.py "boats" --top-k 10 --collapse-duplicates
    photo-search.py "boats" --output-json | jq -r '.results[].filepath'

Input:  $PHOTO_CATALOG/index.jsonl (or --index)
Output: ranked matches; --output-json for a pipe
Exit:   2 if the query and the index do not share one embedding identity (no ranking)
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import photo_lib as P  # noqa: E402

COSINE_SEARCH = Path(__file__).resolve().parent.parent / "rag-server" / "cosine-search.py"
DEFAULT_EMBED_MODEL = "bge-small-en-v1.5-4bit"


def embed_query(model: str, text: str, cpu: bool) -> Dict[str, Any]:
    """Embed in QUERY mode.

    `--query` applies the model's retrieval-query preparation (bge's instruction, e5's
    "query: " prefix). The index side deliberately stays in document mode. Dropping
    this asymmetry is a real, previously-shipped bug: without it a bge index answers
    noticeably worse and nothing anywhere reports a problem.
    """
    cmd = ["mlxk", "embed", model, "-", "--query"] + (["--cpu"] if cpu else [])
    r = subprocess.run(cmd, input=text, text=True, capture_output=True,
                       env={**os.environ, "MLXK2_ENABLE_ALPHA_FEATURES": "1"})
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:800])
    line = next((x for x in r.stdout.splitlines() if x.strip()), "")
    if not line:
        raise RuntimeError("mlxk embed produced no output")
    return json.loads(line)


def haversine_km(a: List[float], b: List[float]) -> float:
    r = 6371.0088
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def passes_filters(row: Dict[str, Any], args) -> bool:
    if args.since or args.until:
        dt = row.get("captured")
        if not dt:
            return False
        if args.since and dt[:10] < args.since:
            return False
        if args.until and dt[:10] > args.until:
            return False
    if args.camera:
        cam = (row.get("camera") or "").lower()
        if args.camera.lower() not in cam:
            return False
    if args.near:
        gps = row.get("gps")
        if not gps:
            return False
        if haversine_km(args.near, gps) > args.radius_km:
            return False
    if args.branch and row.get("branch") != args.branch:
        return False
    return True


def main() -> int:  # noqa: C901
    os.umask(0o077)
    ap = ArgumentParser(description="Search a photo catalog by description, date and place")
    ap.add_argument("query", help="what you are looking for, in plain language")
    ap.add_argument("--index", default=None, help="default: $PHOTO_CATALOG/index.jsonl")
    ap.add_argument("--model", default=DEFAULT_EMBED_MODEL,
                    help="must be the model the index was built with")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--since", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--until", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--near", default=None, metavar="LAT,LON",
                    help="filter to photos within --radius-km of a point")
    ap.add_argument("--radius-km", type=float, default=25.0)
    ap.add_argument("--camera", default=None, help="substring match on the camera name")
    ap.add_argument("--branch", default=None, choices=("direct", "sips"))
    ap.add_argument("--output-json", action="store_true")
    # Off by default, deliberately. Grouping is reported and never acted on when the
    # catalog is built, because a perceptual hash cannot tell one shot in two formats
    # from two frames of a burst. Folding a *result list* is a different matter — it
    # deletes nothing and is undone by omitting the flag — but you should see what is
    # there before you decide to hide part of it, so the tool points the flag out
    # rather than applying it for you.
    ap.add_argument("--collapse-duplicates", action="store_true",
                    help="show one hit per near-duplicate group, keeping the best-scoring "
                         "one; the rest are counted, not dropped from the catalog")
    # These print your own metadata about your own photographs on your own terminal, so
    # the useful default is on. What stays out of the *stored* artefacts — coordinates
    # inside embedded text, the library root inside a record — is a correctness
    # constraint (a coordinate in a vector cannot be removed again), not a reason to
    # withhold anything from the person who owns the library.
    ap.add_argument("--no-gps", action="store_true",
                    help="omit coordinates from the output")
    ap.add_argument("--relative-paths", action="store_true",
                    help="print library-relative paths instead of absolute ones, e.g. when "
                         "the output is going somewhere you did not write")
    args = ap.parse_args()

    if args.near:
        try:
            args.near = [float(x) for x in args.near.split(",")]
            assert len(args.near) == 2
        except (ValueError, AssertionError):
            P.die(P.Precondition(P.EXIT_USAGE, "--near wants LAT,LON", "e.g. --near 59.33,18.07"))

    try:
        catalog_dir = P.require_env("PHOTO_CATALOG") if args.index is None else None
        index_path = Path(args.index) if args.index else catalog_dir / "index.jsonl"
    except P.Precondition as e:
        P.die(e)
    if not index_path.exists():
        P.die(P.Precondition(P.EXIT_USAGE, f"no index at {index_path}", "run index-photos.py first"))

    # --- metadata filters first, so the ranking only sees eligible photos ---------
    rows: Dict[str, Dict[str, Any]] = {}
    kept_lines: List[str] = []
    total = 0
    with open(index_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if passes_filters(row, args):
                rows[row.get("photo_id", "")] = row
                kept_lines.append(line)

    if not kept_lines:
        sys.stderr.write(f"No photo matched the filters ({total} in the index).\n")
        if args.output_json:
            print(json.dumps({"results": []}))
        return P.EXIT_OK

    try:
        query_rec = embed_query(args.model, args.query, args.cpu)
    except (RuntimeError, json.JSONDecodeError) as e:
        P.die(P.Precondition(P.EXIT_EMBED, "could not embed the query", str(e)[:400]))

    with tempfile.TemporaryDirectory(prefix="photo-search-") as td:
        sub_index = Path(td) / "filtered.jsonl"
        sub_index.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        qfile = Path(td) / "query.json"
        qfile.write_text(json.dumps(query_rec), encoding="utf-8")

        # cosine-search.py owns the same-model guard and exits 2 on any mismatch. We
        # pass its exit code straight through rather than interpreting it: refusing to
        # rank is the correct answer, and inventing a friendlier one would hide it.
        # Folding has to happen before the top-k cut, so ask for everything and slice
        # afterwards. cosine-search scores every line either way; the limit is only a
        # slice, so over-fetching costs nothing.
        want = len(kept_lines) if args.collapse_duplicates else args.top_k
        r = subprocess.run(
            [sys.executable, str(COSINE_SEARCH), str(sub_index), str(qfile),
             "--top-k", str(want), "--min-score", str(args.min_score), "--output-json"],
            capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(r.stderr)
            return r.returncode
        try:
            hits = json.loads(r.stdout or "{}").get("results", [])
        except json.JSONDecodeError:
            P.die(P.Precondition(P.EXIT_EMBED, "cosine-search.py returned unparseable JSON",
                                 r.stdout[:300]))

    vault_root = None
    if not args.relative_paths and catalog_dir:
        try:
            vault_root = json.loads((catalog_dir / "vault.json").read_text(encoding="utf-8"))["vault_root"]
        except Exception:  # noqa: BLE001
            sys.stderr.write("  ! could not read vault.json; printing relative paths\n")

    results = []
    for h in hits:
        row = next((r for r in rows.values() if r.get("filepath") == h.get("filepath")), {})
        out = {"score": h["score"], "photo_id": row.get("photo_id"),
               "filename": h.get("filename"), "filepath": h.get("filepath"),
               "captured": row.get("captured"), "camera": row.get("camera"),
               "text": row.get("text", "")}
        if row.get("near_dup_group") is not None:
            out["near_dup_group"] = row["near_dup_group"]
        if not args.no_gps:
            out["gps"] = row.get("gps")
        if vault_root and out["filepath"]:
            out["abspath"] = str(Path(vault_root) / out["filepath"])
        results.append(out)

    if args.collapse_duplicates:
        seen_groups, folded = {}, 0
        kept = []
        for r_ in results:                      # already ordered by score
            g = r_.get("near_dup_group")
            if g is None:
                kept.append(r_)
                continue
            if g in seen_groups:
                seen_groups[g]["folded"] = seen_groups[g].get("folded", 0) + 1
                folded += 1
                continue
            seen_groups[g] = r_
            kept.append(r_)
        results = kept[:args.top_k]
        # Count only what was folded into a row that SURVIVED the cut. Reporting every
        # fold across the whole fetched set would name copies of pictures that are not
        # listed either, which reads as a miscount.
        folded = sum(r_.get("folded", 0) for r_ in results)
        if folded:
            sys.stderr.write(f"  {folded} further copy/copies of the pictures below are not "
                             f"listed; omit --collapse-duplicates to see them\n")
    else:
        # Report, do not decide: say that folding is possible, do not do it unasked.
        counts = {}
        for r_ in results:
            g = r_.get("near_dup_group")
            if g is not None:
                counts[g] = counts.get(g, 0) + 1
        dupes = sum(n - 1 for n in counts.values() if n > 1)
        if dupes:
            sys.stderr.write(f"  {dupes} of these {len(results)} hit(s) are another copy of a "
                             f"picture already listed — --collapse-duplicates hides them\n")

    if args.output_json:
        print(json.dumps({"results": results}, ensure_ascii=False))
        return P.EXIT_OK

    for r_ in results:
        head = f"[{r_['score']:.3f}] {r_['filename']}"
        bits = [b for b in (r_.get("captured", "") or "", r_.get("camera") or "") if b]
        if not args.no_gps and r_.get("gps"):
            lat, lon = r_["gps"]
            bits.append(f"{abs(lat):.4f}{'N' if lat >= 0 else 'S'} {abs(lon):.4f}{'E' if lon >= 0 else 'W'}")
        if r_.get("folded"):
            bits.append(f"+{r_['folded']} more of the same picture")
        elif r_.get("near_dup_group") is not None:
            bits.append(f"near-dup group {r_['near_dup_group']}")
        print(f"{head}  {'  '.join(bits)}")
        if r_.get("abspath"):
            print(f"        {r_['abspath']}")
        text = (r_.get("text") or "").strip().replace("\n", " ")
        if text:
            print(f"        {text[:180]}{'…' if len(text) > 180 else ''}")
        print()
    return P.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
