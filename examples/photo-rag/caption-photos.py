#!/usr/bin/env python3
"""Describe every photo in a library, one photo per request, resumably.

This is a batch, not a session. Tens of thousands of photos take days, so it is built
to be interrupted at any moment — deliberately or by a crash — and picked up again
without redoing finished work. Adding new photos later is the same operation as
resuming: a photo is identified by its content, so a second run does only what is
genuinely new.

Requires a server you started yourself:

    mlxk serve --model pixtral-12b-4bit --port 8000

This script never starts or stops one. A server it started would die with it and take
the run down too; and mlx-knife's own `serve` leaves an orphan behind on SIGTERM
(issue #60), so owning that lifecycle would mean owning that bug as well.

Usage:
    # the whole library; Ctrl-C any time, re-run to continue
    caption-photos.py --model pixtral-12b-4bit

    # check the format branch over 30,000 photos without touching the GPU
    caption-photos.py --model pixtral-12b-4bit --dry-run

    # a second pass at higher resolution over selected photos only
    caption-photos.py --model pixtral-12b-4bit --max-edge 1024 \\
                      --only-ids retry.txt --force-recaption

Input:  $PHOTO_VAULT, $PHOTO_CATALOG (both required, no defaults)
Output: appends to $PHOTO_CATALOG/log/captions.jsonl; the final run_end on stdout

Log record types, in the order a photo produces them:
    run_start   once per run: model, prompt hash, every knob that affects a caption
    canary      a synthetic image of known content; proves the model reads pixels
    attempt     written BEFORE any work on a photo, so a process death leaves a trace
    result      ok:true with the caption, or ok:false with a typed error
    skip        a photo deliberately not described (format, duplicate, pair)
    quarantine  a photo that will not be tried again, and why
    run_end     counts, elapsed, exit code
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import time
import uuid
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import photo_lib as P  # noqa: E402

DEFAULT_PROMPT = (
    "Describe this photograph for a searchable catalog. Say what is shown: the subject, "
    "the setting, notable objects, and any text or signage that is legible in the image. "
    "Be specific and factual. Do not speculate about where or when it was taken. "
    "Write two or three sentences of plain prose, no lists, no preamble."
)

_stop = {"requested": False}


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        if _stop["requested"]:
            # A second signal means the operator is not asking any more.
            sys.stderr.write("\nSecond signal — aborting immediately.\n")
            os._exit(130)
        _stop["requested"] = True
        name = signal.Signals(signum).name
        sys.stderr.write(f"\n{name} — finishing the photo in flight, then stopping cleanly. "
                         f"Re-run to resume; nothing is lost.\n")
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def build_parser() -> ArgumentParser:
    ap = ArgumentParser(description="Describe a photo library, one photo per request")
    ap.add_argument("--model", required=True,
                    help="exactly as /v1/models spells it — this string is stamped into "
                         "every record, so it must be the server's own spelling")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--allow-remote-base-url", action="store_true",
                    help="permit a non-loopback plain-HTTP server. mlx-knife's server is "
                         "unauthenticated and CORS-open, so this sends your photographs in "
                         "the clear to a host that authenticates nobody.")
    ap.add_argument("--max-edge", type=int, default=512,
                    help="longest edge of the uploaded JPEG (default 512, ~4.7 s/photo against "
                         "~14 s at 1024). NOT a text-reading knob: measured on a sign covered in "
                         "small print, 512 px read the headlines and stopped, while 1024 px and "
                         "above read the headlines and INVENTED the body text underneath. Raising "
                         "this buys detail in the picture, not accuracy in the writing.")
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--max-pixels", type=int, default=400_000_000,
                    help="decompression-bomb threshold; the server limits bytes, never pixels")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--canary-every", type=int, default=500)
    ap.add_argument("--canary-shots", type=int, default=2)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include-ext", action="append", default=None)
    ap.add_argument("--exclude-dir", action="append", default=[])
    ap.add_argument("--no-pair-dedupe", action="store_true")
    ap.add_argument("--prefer-raw", action="store_true")
    ap.add_argument("--only-ids", default=None,
                    help="file of photo_ids, one per line; process nothing else")
    ap.add_argument("--force-recaption", action="store_true")
    ap.add_argument("--no-prepared-cache", action="store_true",
                    help="do not keep the uploaded JPEGs; a second pass then re-converts "
                         "every HEIC and raw file from scratch")
    ap.add_argument("--dry-run", action="store_true",
                    help="walk, convert and read metadata, but make no request and write "
                         "no log; one JSONL line per photo on stdout")
    ap.add_argument("--fsync", choices=("none", "fsync", "full"), default="fsync")
    ap.add_argument("--break-lock", action="store_true")
    ap.add_argument("--rebind-vault", action="store_true")
    ap.add_argument("--allow-nonlocal-catalog", action="store_true")
    ap.add_argument("--progress-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None, help="canary permutations (tests only)")
    ap.add_argument("--verbose", action="store_true",
                    help="include the vault-relative path in progress lines")
    return ap


def resolve_prompt(args) -> str:
    if args.prompt and args.prompt_file:
        raise P.Precondition(P.EXIT_USAGE, "--prompt and --prompt-file are mutually exclusive", "")
    if args.prompt_file:
        return Path(args.prompt_file).expanduser().read_text(encoding="utf-8").strip()
    return args.prompt or DEFAULT_PROMPT


def run_canary(client, args, log, run_id, phase, orders, warn) -> bool:
    """Return True if every shot passed. Writes one log line per shot."""
    all_ok = True
    for i, order in enumerate(orders, 1):
        jpeg = P.make_canary(order)
        t0 = time.time()
        rec: Dict[str, Any] = {"type": "canary", "run_id": run_id, "ts": P.utc_now(),
                               "shot": i, "phase": phase, "expect": list(order),
                               "model": args.model}
        try:
            text, meta, _ = P.chat_vision(client, args.base_url, args.model, jpeg,
                                          P.CANARY_PROMPT, max_tokens=64, timeout=args.timeout)
            # The table's column count is the one observable report of the server's
            # EXIF setting: six columns means it is reading EXIF, two means it is not.
            rec["table_columns"] = meta.columns
            rec.update(meta.exif_cells())
            ok, why = P.check_canary(text, order)
            rec["ok"], rec["latency_ms"] = ok, int((time.time() - t0) * 1000)
            if not ok:
                rec["diagnosis"], rec["raw"] = why, text[:2000]
        except P.ServerError as e:
            ok = False
            rec["ok"], rec["error"] = False, e.as_dict()
        if log:
            log.append(rec)
        if not ok:
            all_ok = False
            warn(f"canary shot {i} FAILED: {rec.get('diagnosis') or rec.get('error', {}).get('message', '')}")
    return all_ok


CANARY_HINT = (
    "a model that never receives the pixels still answers HTTP 200, fluently and wrongly: "
    "mlx-knife replaces the images in the prompt with '[N image(s) were attached]' and there "
    "is no error to notice. Check `curl {base}/models` and serve a vision model. "
    "Nothing was captioned; no work was lost.")


def main() -> int:  # noqa: C901 — a batch driver is a sequence, splitting it would hide it
    os.umask(0o077)
    args = build_parser().parse_args()
    warn = lambda m: sys.stderr.write(f"  ! {m}\n")  # noqa: E731

    try:
        prompt = resolve_prompt(args)
        vault = P.require_env("PHOTO_VAULT", must_exist=True)
        catalog = P.require_env("PHOTO_CATALOG")

        # Three separate questions about where the output goes, in rising order of how
        # much a wrong answer costs. All of them BEFORE the first mkdir, so a refused
        # configuration leaves nothing behind.
        #
        # 1. Is the output inside the library? Unrecoverable, no override.
        P.require_separate_catalog(catalog, vault)
        catalog.mkdir(parents=True, exist_ok=True, mode=0o700)

        # 2. Is the output on network storage? Slow and the lock is only advisory.
        fstype, is_local, cat_mount = P.filesystem_of(catalog)
        if not is_local and not args.allow_nonlocal_catalog:
            raise P.Precondition(
                P.EXIT_PRECONDITION, f"the catalog is on network storage ({fstype})",
                "put PHOTO_CATALOG on a local disk. One round trip per log line over SMB is "
                "the bottleneck of the whole run, and the O_EXCL lock is only advisory there. "
                "Pass --allow-nonlocal-catalog to accept both.")

        # 3. Is it on the SAME volume as the library? Not an error — a one-disk machine
        #    has no choice — but on a removable or network share it means the days of
        #    work and the thing they describe share a single point of failure, and
        #    unmounting or filling that share loses both at once. Silent on the boot
        #    volume, where it is simply how a laptop is arranged.
        _, _, vault_mount = P.filesystem_of(vault)
        if cat_mount and cat_mount == vault_mount and cat_mount != "/":
            warn(f"catalog and library are on the same volume ({fstype}). They can only be "
                 f"lost together — unmounting or filling it takes the descriptions with the "
                 f"photographs. A local directory outside it is safer.")

        remote = P.check_base_url(args.base_url, args.allow_remote_base_url)
    except P.Precondition as e:
        P.die(e)

    prompt_sha = P.sha256_hex(prompt.encode("utf-8"))
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:6]}"
    tmpdir = catalog / "tmp"
    prep_root = catalog / "prepared"
    rng = random.Random(args.seed)

    # ---- dry run: no server, no lock, no log ------------------------------------
    if args.dry_run:
        tmpdir.mkdir(parents=True, exist_ok=True, mode=0o700)
        n = 0
        for c in P.walk(vault, include_ext=args.include_ext, exclude_dir=args.exclude_dir,
                        pair_dedupe=not args.no_pair_dedupe, prefer_raw=args.prefer_raw,
                        limit=args.limit):
            rec = c.record()
            if not c.skip:
                try:
                    prep, cached = P.prepare_cached(
                        c.path, c.branch, None if args.no_prepared_cache else prep_root,
                        max_edge=args.max_edge, quality=args.jpeg_quality,
                        max_pixels=args.max_pixels, tmpdir=tmpdir, photo_id_=c.photo_id)
                    rec["prepared"] = {**prep.as_prepared_dict(), "from_cache": cached}
                    rec["exif"] = prep.exif.as_dict()
                    rec["src"] = {"w": prep.src_w, "h": prep.src_h, "converter": prep.converter}
                    # Pixels are already in hand, so this costs a thumbnail — and moves
                    # the duplicate question before the days of GPU time.
                    rec["dhash"] = P.dhash_hex(_dhash_of(prep.data))
                except P.PrepareError as e:
                    rec["skip"], rec["detail"] = e.reason, e.detail[:300]
                    rec["errno"] = e.errno
                    # Otherwise a body that will not convert appears with no camera —
                    # invisible in the one query that would diagnose it.
                    make_model = P.sips_identity(c.path) if c.branch == "sips" else None
                    if make_model:
                        rec["exif"] = {"camera": make_model}
            print(json.dumps(rec, ensure_ascii=True))
            n += 1
        sys.stderr.write(f"\ndry run: {n} records, no request made, no log written\n")
        return P.EXIT_OK

    # ---- the real run -----------------------------------------------------------
    try:
        import httpx
    except ImportError:
        P.die(P.Precondition(P.EXIT_USAGE, "httpx is not installed",
                             "pip install httpx  (it is a core mlx-knife dependency, so a "
                             "normal mlx-knife install already has it)"))

    only_ids = None
    if args.only_ids:
        only_ids = {ln.strip() for ln in Path(args.only_ids).read_text(
            encoding="utf-8", errors="replace").splitlines() if ln.strip()}

    log = P.Log(catalog / "log" / "captions.jsonl", fsync_mode=args.fsync,
                redact=(vault, catalog))
    lock = P.Lock(catalog / "log" / ".lock", run_id)
    try:
        lock.acquire(break_lock=args.break_lock)
    except P.Precondition as e:
        P.die(e)

    # LOCK-01: the vault binding is written only now, with the mutex held. Before, a
    # process that lost the race still rewrote vault.json on its way to being refused —
    # so the catalog could end up naming a vault that the holder is not working on, and
    # the guard would then reject the right library and accept the wrong one.
    try:
        P.bind_vault(catalog, vault, rebind=args.rebind_vault)
    except P.Precondition as e:
        lock.release()
        P.die(e)

    _install_signal_handlers()
    # Two questions, two blocks: what is out there, and what this run did. They were
    # both being answered with the word "skipped".
    walk_stats = {"files": 0, "candidates": 0, "skipped": {}}
    counts = {"considered": 0, "already_done": 0, "moved": 0, "captioned": 0,
              "failed": 0, "permission": 0, "quarantined": 0, "not_reached": 0}
    canaries = {"run": 0, "failed": 0}
    exit_code = P.EXIT_OK
    stopped_by = None
    started = time.time()
    consecutive_canary_failures = 0

    # The lock is ours, so nothing else owns anything in tmp/. Sweep it: the sips
    # intermediates still carry a full EXIF block, and the `finally:` that normally
    # removes them does not run when the interpreter is killed — which is exactly the
    # failure the whole resume design exists for.
    #
    # But sweep ONLY what this tool created. A directory that already existed with
    # anything else in it is not ours to empty, and the cost of being wrong here is
    # deleted files rather than a stale cache. The marker is what distinguishes "my
    # scratch space" from "a directory that happens to be called tmp"; without it the
    # sweep is unconditional deletion of a path the operator chose.
    swept = 0
    tmpdir.mkdir(parents=True, exist_ok=True, mode=0o700)
    marker = tmpdir / ".photo-rag-scratch"
    if marker.exists() or not any(tmpdir.iterdir()):
        marker.touch()
        for stale in tmpdir.iterdir():
            if stale == marker:
                continue
            try:
                stale.unlink()
                swept += 1
            except OSError:
                pass
    else:
        warn(f"{tmpdir.name}/ exists and was not created by this tool — leaving its "
             f"contents alone. Conversion scratch will be written beside them.")

    client = httpx.Client(headers={"Content-Type": "application/json"})
    try:
        with log:
            if log.repaired_tail:
                warn("log did not end in a newline (a previous run was killed mid-write); "
                     "repaired before appending")
                log.append({"type": "torn_tail_repaired", "run_id": run_id, "ts": P.utc_now()})

            try:
                P.server_health(client, args.base_url)
            except P.Precondition as e:
                lock.release()
                P.die(e)

            served = P.list_models(client, args.base_url)
            if served and args.model not in served:
                # Advisory, never fatal: /v1/models filters on runtime_compatible, and that
                # gate has a documented false negative — the server happily loads and answers
                # for a model it declines to list. The canary below is the authority on
                # whether this model actually works.
                warn(f"{args.model!r} is not in /v1/models (served: {', '.join(served) or 'none'}). "
                     f"Continuing — that list is filtered and can hide a working model; "
                     f"the canary decides.")

            log.append({
                "type": "run_start", "run_id": run_id, "ts": P.utc_now(), "schema": P.SCHEMA,
                "tool": P.TOOL, "pid": os.getpid(), "base_url": args.base_url,
                "remote_base_url": remote, "model": args.model, "max_edge": args.max_edge,
                "jpeg_quality": args.jpeg_quality, "max_tokens": args.max_tokens,
                "prompt_sha256": prompt_sha, "upload_exif": "stripped",
                "fsync": args.fsync, "tmp_swept": swept,
                # Quarantine is permanent and the log is its only account; without the
                # budget it was measured against, the record cannot be read.
                "max_attempts": args.max_attempts,
            })

            # --- startup canary: before a single real photo -----------------------
            orders = P.canary_orders(args.canary_shots, rng)
            canaries["run"] += len(orders)
            if not run_canary(client, args, log, run_id, "startup", orders, warn):
                canaries["failed"] += 1
                log.append({"type": "run_end", "run_id": run_id, "ts": P.utc_now(),
                            "schema": P.SCHEMA, "walk": walk_stats, "counts": counts,
                            "canaries": canaries,
                            "elapsed_s": round(time.time() - started, 1),
                            "interrupted": False, "stopped_by": "canary",
                            "exit": P.EXIT_CANARY})
                lock.release()
                P.die(P.Precondition(P.EXIT_CANARY, "canary failed — the model did not see the image",
                                     CANARY_HINT.format(base=args.base_url)))

            idx = log.resume_index(warn=warn)
            if idx.torn:
                warn(f"{idx.torn} unparseable log line(s) skipped; their photos will be redone")

            candidates = list(P.walk(vault, include_ext=args.include_ext,
                                     exclude_dir=args.exclude_dir,
                                     pair_dedupe=not args.no_pair_dedupe,
                                     prefer_raw=args.prefer_raw, limit=args.limit))
            todo = [c for c in candidates if not c.skip]
            total = len(todo)

            # Kept rather than dropped: these were produced and discarded one line
            # later, so an unreadable directory looked like an empty one.
            for c in candidates:
                if c.skip:
                    walk_stats["skipped"][c.skip] = walk_stats["skipped"].get(c.skip, 0) + 1
            walk_stats["candidates"] = total
            walk_stats["files"] = total + sum(walk_stats["skipped"].values())

            sys.stderr.write(f"\n{total} candidate(s); {len(idx.done)} already described\n")
            if walk_stats["skipped"]:
                sys.stderr.write("  set aside: " + ", ".join(
                    f"{v} {k}" for k, v in sorted(walk_stats["skipped"].items())) + "\n")

            # Spread on purpose: a library can span mounts, and three witnesses from
            # the front would all vouch for whichever subtree os.walk reached first.
            witness = P.Witness(vault)
            if todo:
                for i in sorted({0, total // 2, total - 1}):
                    witness.remember(todo[i].path)

            # A share that dies during the scan raises nothing: os.walk just stops
            # yielding. One stat and one byte, against a walk that costs minutes.
            env_ok, env_why = witness.alive()
            if not env_ok:
                log.append({"type": "environment_lost", "run_id": run_id, "ts": P.utc_now(),
                            "phase": "walk", "witness": env_why, "not_reached": total})
                counts["not_reached"] = total
                lock.release()
                P.die(P.Precondition(
                    P.EXIT_ENVIRONMENT,
                    f"the library became unreachable during the scan ({env_why})",
                    "Nothing was written against any photograph. Check the mount and run again."))

            # An already-dead share leaves an empty mountpoint, which passes
            # must_exist. Empty log = first run; non-empty log = contradiction.
            if total == 0 and idx.lines:
                log.append({"type": "run_end", "run_id": run_id, "ts": P.utc_now(),
                            "schema": P.SCHEMA, "walk": walk_stats, "counts": counts,
                            "canaries": canaries, "elapsed_s": round(time.time() - started, 1),
                            "interrupted": False, "stopped_by": "empty_library",
                            "exit": P.EXIT_ENVIRONMENT})
                lock.release()
                P.die(P.Precondition(
                    P.EXIT_ENVIRONMENT,
                    f"no photographs found, but this catalog already knows {len(idx.done)}",
                    "An empty mountpoint looks exactly like this. Check that the library is "
                    "mounted before treating the run as finished."))

            since_canary = 0
            moved_seen: set = set()
            for n, c in enumerate(todo, 1):
                if _stop["requested"]:
                    exit_code = P.EXIT_INTERRUPTED
                    break
                pid = c.photo_id

                if only_ids is not None and pid not in only_ids:
                    continue
                counts["considered"] += 1      # after the filter: what this run decided about

                why = P.resume_decision(pid, idx, args.max_attempts, force=args.force_recaption)
                if why == "already_done":
                    counts["already_done"] += 1
                    # The walker has just told us where this photo is now. Identity is
                    # content-based, so a move costs no re-captioning — but the derived
                    # catalog would go on pointing at the old path, and photo-search.py
                    # hands that straight to the user as an absolute path. Record the new
                    # location; a run with nothing moved writes nothing.
                    if pid not in moved_seen and c.rel != idx.rel.get(pid):
                        log.append({"type": "path_update", "run_id": run_id,
                                    "ts": P.utc_now(), "photo_id": pid, "rel": c.rel})
                        idx.rel[pid] = c.rel
                        counts["moved"] += 1
                    # Two byte-identical copies both resolve to this id in one run; the
                    # first one walked wins, deterministically, rather than whichever the
                    # filesystem happened to return last.
                    moved_seen.add(pid)
                    continue
                if why == "quarantined":
                    counts["quarantined"] += 1
                    continue
                if why in ("poison_pill", "repeated_error"):
                    log.append({"type": "quarantine", "run_id": run_id, "ts": P.utc_now(),
                                "photo_id": pid, "rel": c.rel, "reason": why,
                                "attempts": idx.attempts.get(pid, 0),
                                "results": idx.results.get(pid, 0),
                                "failures": idx.failures.get(pid, 0),
                                "max_attempts": args.max_attempts})
                    idx.quarantined.add(pid)
                    counts["quarantined"] += 1
                    warn(f"{pid[:8]} quarantined ({why})")
                    continue

                # The attempt line goes down BEFORE any work, not before the request:
                # a decoder fault inside Pillow or sips kills the interpreter just as
                # dead as a Metal OOM does, and the outermost marker is the one that
                # has to exist for the restart to see it.
                attempt_no = idx.attempts.get(pid, 0) + 1
                log.append({"type": "attempt", "run_id": run_id, "ts": P.utc_now(),
                            "photo_id": pid, "rel": c.rel})
                idx.attempts[pid] = attempt_no

                t0 = time.time()
                rec: Dict[str, Any] = {"type": "result", "run_id": run_id, "ts": P.utc_now(),
                                       "photo_id": pid, "rel": c.rel, "attempt_no": attempt_no}
                # An errno means the machine refused, not the decoder, so the storage
                # is asked before this file is charged with anything. See Witness for
                # why a passing probe buys a second look rather than a verdict.
                prep, from_cache, prep_error, env_dead = None, False, None, None
                for looked_twice in (False, True):
                    try:
                        prep, from_cache = P.prepare_cached(
                            c.path, c.branch, None if args.no_prepared_cache else prep_root,
                            max_edge=args.max_edge, quality=args.jpeg_quality,
                            max_pixels=args.max_pixels, tmpdir=tmpdir, photo_id_=pid)
                        prep_error = None
                        break
                    except P.PrepareError as e:
                        prep_error = e
                        if looked_twice or e.errno is None or e.reason == "permission":
                            break           # the decoder's verdict, or already asked twice
                        env_ok, env_why = witness.alive(near=c.path.parent)
                        if not env_ok:
                            env_dead = env_why
                            break

                if env_dead is not None:
                    # Stop: every photograph after this fails the same way in ~2 ms,
                    # each against a file that is fine.
                    log.append({"type": "environment_lost", "run_id": run_id,
                                "ts": P.utc_now(), "photo_id": pid, "rel": c.rel,
                                "reason": prep_error.reason, "errno": prep_error.errno,
                                "witness": env_dead, "not_reached": total - n})
                    # Closes the attempt written above, or this photograph looks like a
                    # process death next time (see resume_index).
                    idx.results[pid] = idx.results.get(pid, 0) + 1
                    counts["not_reached"] = total - n
                    exit_code, stopped_by = P.EXIT_ENVIRONMENT, "environment_lost"
                    warn(f"{pid[:8]} {prep_error.reason}: {env_dead}")
                    warn(f"stopping: the library became unreachable, "
                         f"{total - n} photograph(s) not reached and not blamed")
                    break

                if prep_error is not None:
                    environmental = prep_error.reason == "permission"
                    log.append({**rec, "ok": False,
                                "error": {"kind": "prepare_error",
                                          "reason": prep_error.reason,
                                          "errno": prep_error.errno,
                                          "message": prep_error.detail[:300]},
                                "environmental": environmental,
                                "latency_ms": int((time.time() - t0) * 1000)})
                    # Always balances the attempt, or the next run reads a process death.
                    idx.results[pid] = idx.results.get(pid, 0) + 1
                    if environmental:
                        # Counted, but it buys no evidence against the file.
                        counts["permission"] += 1
                    else:
                        idx.failures[pid] = idx.failures.get(pid, 0) + 1
                        counts["failed"] += 1
                    warn(f"{pid[:8]} {prep_error.reason}")
                    continue

                try:
                    text, meta, body = P.chat_vision(
                        client, args.base_url, args.model, prep.data, prompt,
                        max_tokens=args.max_tokens, timeout=args.timeout)
                    # The deterministic half of the media-downgrade defence. Unlike the
                    # canary this fires on EVERY photo, so a model swap at the server
                    # cannot leave a single invented caption in the log.
                    P.assert_pixels_arrived(meta, prep.server_hash)
                    if not text.strip():
                        raise P.ServerError("empty_caption", message="model returned no text")

                    rec.update({
                        "ok": True, "text": text, "model": body.get("model") or args.model,
                        "prompt_sha256": prompt_sha,
                        "src": {"bytes": c.bytes, "ext": c.ext, "branch": c.branch,
                                "converter": prep.converter, "w": prep.src_w, "h": prep.src_h},
                        "prepared": {**prep.as_prepared_dict(), "from_cache": from_cache},
                        "dhash": P.dhash_hex(_dhash_of(prep.data)),
                        "exif": prep.exif.as_dict(),
                        "server": {"table_columns": meta.columns, **meta.exif_cells()},
                        # finish_reason is hardcoded "stop" on every server path and the
                        # token counts are a word-count estimate, so neither wire field can
                        # report truncation. Terminal punctuation is the only signal left.
                        "truncated": not text.rstrip().endswith((".", "!", "?", '"', "'", ")")),
                        "latency_ms": int((time.time() - t0) * 1000),
                    })
                    log.append(rec)
                    idx.results[pid] = idx.results.get(pid, 0) + 1
                    idx.done.add(pid)
                    counts["captioned"] += 1
                    _progress(args, n, total, t0, pid, c.rel, "ok")
                except P.ServerError as e:
                    rec.update({"ok": False, "error": e.as_dict(),
                                "latency_ms": int((time.time() - t0) * 1000)})
                    log.append(rec)
                    idx.results[pid] = idx.results.get(pid, 0) + 1
                    idx.failures[pid] = idx.failures.get(pid, 0) + 1
                    counts["failed"] += 1
                    warn(f"{pid[:8]} {e.kind}: {e.message[:120]}")
                    if e.kind in ("no_image_seen", "hash_mismatch"):
                        # Not a per-photo problem: the pixels are not reaching the model
                        # at all. Every further caption would be fiction.
                        exit_code = P.EXIT_CANARY
                        break
                    if e.status == 503:
                        exit_code = P.EXIT_PRECONDITION
                        warn("server is shutting down; stopping. The run is resumable.")
                        break

                since_canary += 1
                if args.canary_every and since_canary >= args.canary_every:
                    since_canary = 0
                    order = P.canary_orders(1, rng)
                    canaries["run"] += 1
                    if not run_canary(client, args, log, run_id, "interval", order, warn):
                        consecutive_canary_failures += 1
                        canaries["failed"] += 1
                        # One ambiguous colour word must not kill a four-day run; two
                        # failures in a row is a signal rather than an accident.
                        if consecutive_canary_failures >= 2:
                            exit_code = P.EXIT_CANARY
                            warn("two consecutive canary failures — stopping.")
                            break
                        warn("canary failed once; retrying at the next interval.")
                    else:
                        consecutive_canary_failures = 0

            # Separate field, not a new meaning for `interrupted` — that flag is the
            # only thing telling a clean Ctrl-C apart.
            end = {"type": "run_end", "run_id": run_id, "ts": P.utc_now(),
                   "schema": P.SCHEMA, "walk": walk_stats, "counts": counts,
                   "canaries": canaries, "elapsed_s": round(time.time() - started, 1),
                   "interrupted": _stop["requested"], "stopped_by": stopped_by,
                   "exit": exit_code}
            log.append(end)
            print(json.dumps(end, ensure_ascii=True))
    finally:
        client.close()
        lock.release()

    aside = ", ".join(f"{v} {k}" for k, v in sorted(walk_stats["skipped"].items()))
    sys.stderr.write(
        f"\nwalk:  {walk_stats['files']} file(s), {walk_stats['candidates']} candidate(s)"
        + (f"; set aside {aside}" if aside else "") + "\n"
        f"run:   {counts['captioned']} described, {counts['already_done']} already done, "
        f"{counts['failed']} failed, {counts['permission']} unreadable by permission, "
        f"{counts['quarantined']} quarantined"
        + (f", {counts['not_reached']} not reached" if counts["not_reached"] else "")
        + f" in {time.time() - started:.1f}s\n")
    if stopped_by:
        sys.stderr.write(f"       stopped by: {stopped_by}\n")
    return exit_code


def _dhash_of(jpeg: bytes) -> int:
    import io
    from PIL import Image
    with Image.open(io.BytesIO(jpeg)) as im:
        return P.dhash(im)


def _progress(args, n: int, total: int, t0: float, pid: str, rel: str, state: str) -> None:
    if args.progress_every and n % args.progress_every:
        return
    dt = time.time() - t0
    tail = f"  {rel}" if args.verbose else ""
    sys.stderr.write(f"[{n:>5}/{total}] {100 * n // max(total, 1):>3}%  {state} {dt:4.1f}s  "
                     f"{pid[:8]}{tail}\n")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except P.Precondition as e:
        P.die(e)
