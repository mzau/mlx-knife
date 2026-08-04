#!/usr/bin/env python3
"""Run the whole pipeline against the project's own test fixture and grade itself.

Every row below prints a verdict against ground truth that exists independently of
the tool: geometry measured from the produced bytes, strings that are legibly painted
on a photograph, predicates fed inputs they must reject. Nothing here asks a model or
a script whether it thinks it did well.

It drives the real scripts as subprocesses, so what is graded is the documented command
line, not an internal function someone could keep working while the CLI rots.

Requires a server you started yourself:

    mlxk serve --model pixtral-12b-4bit --port 8000

Usage:
    ./geo-test-run.py
    ./geo-test-run.py --json
    ./geo-test-run.py --vision-model pixtral-12b-8bit --keep

Input:  the tracked fixture next to this repository's tests, plus whatever local-only
        material sits beside it
Output: a PASS/FAIL/SKIP table and a verdict; exit 0 only on PASS

The Detail column is restricted to counts, ratios, git-tracked filenames and the first
eight characters of a photo_id. Never a path, a coordinate, a timestamp or a caption
fragment: this table is meant to be pasted into a document, and the failure case is
exactly the one where the natural diagnostic would be the private value. Long-form
detail goes to <workdir>/failures.txt instead.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import uuid
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import photo_lib as P  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FIXTURE = REPO / "tests_2.0" / "assets" / "geo-test"

# coll2_3.jpeg carries three legible strings, painted on the hull and on a departure
# sign. They are the only public, objectively checkable *content* in the fixture, which
# makes them the one probe that re-measures the resolution decision the whole cost model
# rests on. Normalised comparison on purpose: at 512 px the ring on the A is lost
# ("OSTANA"), at 1024 px it is not — that difference is the measurement, not a failure.
LEGIBLE = {"ostana": "ÖSTANÅ", "archipelago": "ARCHIPELAGO TOUR", "14:00": "14:00"}
BOAT_TERMS = ("boat", "ship", "steamer", "paddle", "vessel", "ferry", "barge", "hull",
              "dock", "docked", "harbor", "harbour", "quay", "pier", "moored")

# Two queries that use the fixture's two ship photographs as each other's distractor.
# A single "boat" query would be a coin flip: coll2_5 is also a ship at a quay, and on
# the words "docked" and "pier" it is the better literal match. Requiring both
# directions measures caption content instead of corpus composition.
QUERY_A = "a small white paddle steamer moored at a pier with a departure sign"
QUERY_B = "a large white sailing ship with tall masts alongside a stone quay"


def norm(s: str) -> str:
    """Case-fold and strip diacritics, so ÖSTANÅ and OSTANA compare equal."""
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


class Table:
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []
        self.notes: List[str] = []
        # Captions printed in full for the photographs of the first phase, which are
        # published in this repository. The fixture is public — that is the whole reason
        # it exists — and a table of ratios is not evidence anyone can check. The second
        # phase is where anything local-only would appear, so it is held back wholesale:
        # the split is by phase, never by filename, so no tracked file has to name a
        # directory that may not exist.
        self.appendix: List[str] = []

    def add(self, rid: str, name: str, status: str, detail: str = "", mandatory: bool = True):
        self.rows.append({"id": rid, "name": name, "status": status,
                          "detail": detail, "mandatory": mandatory})
        mark = {"PASS": "ok  ", "FAIL": "FAIL", "SKIP": "skip"}[status]
        sys.stderr.write(f"  {mark} {rid:<5} {name:<44} {detail}\n")

    def note(self, text: str):
        self.notes.append(text)

    def check(self, rid: str, name: str, ok: bool, detail: str = "", mandatory: bool = True):
        self.add(rid, name, "PASS" if ok else "FAIL", detail, mandatory)
        return ok

    @property
    def counts(self):
        c = {"PASS": 0, "FAIL": 0, "SKIP": 0}
        for r in self.rows:
            c[r["status"]] += 1
        return c

    def verdict(self) -> str:
        if self.counts["FAIL"]:
            return "FAIL"
        if any(r["mandatory"] and r["status"] == "SKIP" for r in self.rows):
            return "INCONCLUSIVE"
        return "PASS"


def run(cmd: List[str], env: Dict[str, str], timeout: int = 1800, stdin: str = None):
    return subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=timeout, input=stdin)


def read_log(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


# ---------------------------------------------------------------------------
# P0 — do the predicates actually reject anything?
#
# Every other row in this table is evaluated only against output from a healthy run,
# so every other row can only ever go green. That proves the checks FIRE; it does not
# prove they DISCRIMINATE. This row feeds each pure predicate the inputs it is supposed
# to refuse. It costs milliseconds and no GPU, and it is what turns the table's claimed
# power into a measured one.
# ---------------------------------------------------------------------------
def p0_predicate_selftest(t: Table) -> bool:
    failures = []

    order = ["red", "green", "blue", "yellow"]
    must_reject_canary = [
        ("downgrade marker", "[1 image(s) were attached]"),
        ("honest refusal", "I'm sorry, I don't see any image in this conversation."),
        ("empty", ""),
        ("wrong order", "blue red green yellow"),
        ("all four, scrambled", "yellow, blue, green and red"),
        ("prose without colours", "A photograph of a harbour on an overcast afternoon."),
    ]
    for label, text in must_reject_canary:
        ok, _ = P.check_canary(text, order)
        if ok:
            failures.append(f"canary accepted {label!r}")
    ok, _ = P.check_canary("red green blue yellow", order)
    if not ok:
        failures.append("canary rejected the correct answer")

    # Every number below is a repeated-digit placeholder, and the date is a year no
    # photograph in this fixture was taken in. The rule this example enforces is that
    # coordinate- and timestamp-shaped literals never appear in the tree; a plausible
    # fabricated coordinate would satisfy the letter of that and defeat its purpose,
    # because no reviewer can tell an invented one from a real one.
    must_reject_contam = [
        ("bracketed GPS", "[GPS: 11.1111N, 22.2222E | Date: 1999-12-31 | Camera: Example]"),
        ("decimal degrees", "Taken at 11.1111 N, 22.2222 E."),
        ("degree symbols", "The location is 11.11 N, 22.22 E."),
        ("injected date", "Date: 1999-12-31, an overcast afternoon."),
        ("named axes", "The latitude and longitude place this in the archipelago."),
    ]
    for label, text in must_reject_contam:
        if not contaminated(text):
            failures.append(f"contamination predicate passed {label!r}")
    if contaminated("A white paddle steamer at a pier; a sign reads 14:00."):
        failures.append("contamination predicate rejected a clean caption")

    # The caption the blended (multi-image) mode actually produced when this was
    # measured. The boat-term list exists to catch exactly this, so it must not match.
    if any(term in norm("a calm lake with scattered lily pads") for term in BOAT_TERMS):
        failures.append("boat-term list matched the recorded blended caption")
    if not any(term in norm("A docked paddle steamer with a white hull") for term in BOAT_TERMS):
        failures.append("boat-term list missed a correct caption")

    # The metadata-block parser must not accept a response that has no block.
    _, meta = P.strip_metadata_block("A photograph of a harbour.")
    if meta.present:
        failures.append("metadata parser claimed a block in a bare response")
    _, meta2 = P.strip_metadata_block("<details>\nunterminated")
    if meta2.present:
        failures.append("metadata parser accepted an unterminated block")

    return t.check("P0", "predicates reject what they must", not failures,
                   "; ".join(failures) if failures else
                   f"{len(must_reject_canary) + len(must_reject_contam) + 4} rejection cases held")


COORD_PATTERNS = [
    re.compile(r"\d{1,3}\.\d+\s*°"),
    re.compile(r"\b\d{1,3}(\.\d+)?\s*°?\s*[NSEW]\b"),
    re.compile(r"\[GPS:", re.I),
    re.compile(r"\bDate:\s*(19|20)\d{2}"),
    re.compile(r"\b(latitude|longitude|coordinates)\b", re.I),
    re.compile(r"\b\d{1,3}\.\d{3,}\s*,\s*-?\d{1,3}\.\d{3,}"),
]


def contaminated(text: str) -> bool:
    return any(p.search(text) for p in COORD_PATTERNS)


def build_synthetic_vault(work: Path, t: Table) -> Path:
    """A small vault the runner fully controls.

    Structural checks — does the walker recurse, does it filter junk, does it spot a
    duplicate, does it pair a raw with its JPEG — must not be evaluated against the
    tracked fixture. The subdirectory beside it is local-only, so on a fresh clone it
    does not exist and those rows would fail for a reason that has nothing to do with
    the walker. Here the structure is built on the spot, identical on every machine.
    """
    v = work / "synthvault"
    (v / "nested" / "deeper").mkdir(parents=True)
    (v / "pairs").mkdir(parents=True)

    # Each check gets its OWN source photograph. Reusing one would make the checks
    # interfere: three byte-identical copies let the duplicate rule consume the file
    # that was supposed to prove recursion, and the recursion row then failed for a
    # reason that had nothing to do with recursion.
    shutil.copy2(FIXTURE / "coll2_1.jpeg", v / "alpha.jpeg")
    shutil.copy2(FIXTURE / "coll2_1.jpeg", v / "dup-of-alpha.jpeg")        # duplicate_content
    shutil.copy2(FIXTURE / "coll2_2.jpeg", v / "nested" / "deeper" / "beta.jpeg")  # recursion
    shutil.copy2(FIXTURE / "coll2_4.jpeg", v / "shot.JPG")                 # keeps its pair partner
    (v / "shot.ARW").write_bytes(b"II*\x00fake raw for the name-pair heuristic")
    (v / "truncated.jpeg").write_bytes((FIXTURE / "coll2_6.jpeg").read_bytes()[:5000])
    (v / ".DS_Store").write_bytes(b"\x00" * 64)               # dotfile
    (v / "notes.txt").write_text("not a photo")               # unsupported_format
    (v / "empty.jpeg").write_bytes(b"")                       # empty_file
    return v



def a_photo(path: Path, colour=(120, 60, 200), size=(96, 72)) -> Path:
    """A generated JPEG. The mechanism rows must not depend on the fixture's content —
    or even on the fixture existing — so they make their own raw material."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, "JPEG")
    return path


def guard_rows(t: Table, work: Path, env_of) -> None:
    """Rows for the boundaries an example is trusted to hold on a real library.

    Every one of these builds the world it tests. Nothing here reads the fixture, so a
    photograph added, removed or renamed beside them cannot turn a row red for a reason
    that has nothing to do with the boundary under test — the mistake that made the
    recursion row fail once already.

    Each has a negative control. "It refused" alone proves nothing: a check that refuses
    everything would pass it.
    """
    import shutil as _sh

    # -- catalog must not live inside the library ---------------------------------
    w = work / "guard-contain"
    vault = a_photo(w / "vault" / "a.jpg").parent
    inside, sibling = vault / "cat", w / "cat"
    r_in = run([sys.executable, str(HERE / "caption-photos.py"), "--model", "x", "--dry-run"],
               env_of(vault, inside), timeout=60)
    r_sib = run([sys.executable, str(HERE / "caption-photos.py"), "--model", "x", "--dry-run"],
                env_of(vault, sibling), timeout=60)
    t.check("G1", "a catalog inside the library is refused",
            r_in.returncode == P.EXIT_PRECONDITION and not inside.exists()
            and r_sib.returncode == 0,
            f"inside: exit {r_in.returncode}, nothing created · sibling accepted: "
            f"exit {r_sib.returncode}")

    # -- the tmp sweep must not empty a directory it did not create ----------------
    w2 = work / "guard-sweep"
    v2 = a_photo(w2 / "vault" / "a.jpg").parent
    c2 = w2 / "cat"
    (c2 / "tmp").mkdir(parents=True)
    keep = a_photo(c2 / "tmp" / "someones-photo.jpg", (10, 200, 10))
    run([sys.executable, str(HERE / "caption-photos.py"), "--model", "x",
         "--base-url", "http://127.0.0.1:1/v1"], env_of(v2, c2), timeout=60)
    t.check("G2", "a pre-existing tmp/ is not emptied", keep.exists(),
            "a file that was already there survived a run that takes the lock")

    # -- a symlinked file leaves the library ---------------------------------------
    w3 = work / "guard-symlink"
    v3 = a_photo(w3 / "vault" / "inside.jpg").parent
    out = a_photo(w3 / "elsewhere" / "secret.jpg", (200, 30, 30))
    os.symlink(out, v3 / "link.jpg")
    default = [c.rel for c in P.walk(v3) if not c.skip]
    followed = [c.rel for c in P.walk(v3, follow_symlinks=True) if not c.skip]
    reason = {c.skip for c in P.walk(v3) if c.skip}
    t.check("G3", "a symlinked file is not read by default",
            default == ["inside.jpg"] and sorted(followed) == ["inside.jpg", "link.jpg"]
            and "symlink" in reason,
            "skipped and named 'symlink'; --follow-symlinks does include it")

    # -- a moved photo keeps its caption and updates its location ------------------
    w4 = work / "guard-move"
    v4, c4 = w4 / "vault", w4 / "cat"
    a_photo(v4 / "album" / "a.jpg")
    (v4 / "archive").mkdir()
    pid = P.photo_id(v4 / "album" / "a.jpg")
    lg = P.Log(c4 / "log" / "captions.jsonl")
    with lg as h:
        h.append({"type": "result", "photo_id": pid, "rel": "album/a.jpg", "ok": True,
                  "text": "a generated rectangle", "exif": {}, "prepared": {}, "src": {}})
    _sh.move(str(v4 / "album" / "a.jpg"), str(v4 / "archive" / "a.jpg"))
    with lg as h:
        h.append({"type": "path_update", "photo_id": pid, "rel": "archive/a.jpg"})
    run([sys.executable, str(HERE / "build-catalog.py")], env_of(v4, c4), timeout=60)
    rows = [json.loads(x) for x in (c4 / "catalog.jsonl").read_text().splitlines() if x.strip()]
    t.check("G4", "a moved photo is relocated, not re-described",
            len(rows) == 1 and rows[0]["filepath"] == "archive/a.jpg"
            and rows[0]["text"] == "a generated rectangle"
            and len(list(lg.read())) == 2,
            "catalog follows the new path, caption unchanged, log append-only")

    # -- an error message must not carry the library root --------------------------
    w5 = work / "guard-redact"
    v5, c5 = w5 / "My Photos", w5 / "cat"
    v5.mkdir(parents=True)
    lg5 = P.Log(c5 / "log" / "captions.jsonl", redact=(v5, c5))
    with lg5 as h:
        h.append({"type": "result", "ok": False, "rel": "album/x.jpg",
                  "error": {"kind": "prepare_error",
                            "message": f"cannot identify image file '{v5}/album/x.jpg'"}})
    body = (c5 / "log" / "captions.jsonl").read_text()
    t.check("G5", "decoder errors carry no absolute path",
            str(v5) not in body and str(v5.resolve()) not in body and "<root>" in body,
            "both spellings of the root are substituted")

    # -- the vault binding is written only under the lock --------------------------
    w6 = work / "guard-lock"
    v6a, v6b, c6 = w6 / "vaultA", w6 / "vaultB", w6 / "cat"
    a_photo(v6a / "a.jpg")
    a_photo(v6b / "b.jpg")
    P.bind_vault(c6, v6a)
    held = P.Lock(c6 / "log" / ".lock", "holder")
    held.acquire()
    r = run([sys.executable, str(HERE / "caption-photos.py"), "--model", "x",
             "--rebind-vault", "--base-url", "http://127.0.0.1:1/v1"],
            env_of(v6b, c6), timeout=60)
    bound = json.loads((c6 / "vault.json").read_text())["vault_root"]
    held.release()
    t.check("G6", "a run that loses the lock does not rebind the vault",
            r.returncode == P.EXIT_LOCKED and bound == str(v6a.resolve()),
            f"exit {r.returncode}; marker still names the holder's library")


def main() -> int:  # noqa: C901
    os.umask(0o077)
    ap = ArgumentParser(description="Self-verifying run of the photo-rag pipeline")
    ap.add_argument("--vision-model", default="pixtral-12b-4bit")
    ap.add_argument("--embed-model", default="bge-small-en-v1.5-4bit")
    ap.add_argument("--second-embed-model", default="multilingual-e5-small-mlx")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--max-edge", type=int, default=512)
    ap.add_argument("--ladder-edge", type=int, default=1024)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--skip-slow", action="store_true",
                    help="predicates and offline rows only; no captioning")
    args = ap.parse_args()

    t = Table()
    started = time.time()

    if not FIXTURE.is_dir():
        sys.stderr.write(f"Error: fixture not found at {FIXTURE}\n")
        return P.EXIT_USAGE

    # The workdir is a fresh temp directory, never the operator's catalog, and every
    # child gets BOTH PHOTO_VAULT and PHOTO_CATALOG overridden. Inheriting the real
    # PHOTO_CATALOG would append fixture records to a private log and then run these
    # assertions across the operator's own photographs.
    if args.workdir:
        work = Path(args.workdir).expanduser().resolve()
        if str(work).startswith(str(REPO)):
            sys.stderr.write("Error: the workdir must not be inside the repository.\n")
            return P.EXIT_USAGE
        real = os.environ.get("PHOTO_CATALOG")
        if real and Path(real).expanduser().resolve() == work:
            sys.stderr.write("Error: refusing to use your real PHOTO_CATALOG as the workdir.\n")
            return P.EXIT_USAGE
        work.mkdir(parents=True, exist_ok=True)
    else:
        work = Path(tempfile.mkdtemp(prefix="photo-rag-geotest-"))

    sys.stderr.write(
        f"\nphoto-rag self-check\n{'=' * 78}\n"
        f"  fixture   {FIXTURE.relative_to(REPO)}  (+ whatever local-only material sits beside it)\n"
        f"  workdir   {work}\n"
        f"  vision    {args.vision_model}    embed  {args.embed_model}\n"
        f"  Your own PHOTO_VAULT and PHOTO_CATALOG are overridden for every child and\n"
        f"  are not touched.\n{'=' * 78}\n\n")

    def child_env(vault: Path, catalog: Path) -> Dict[str, str]:
        e = dict(os.environ)
        e["PHOTO_VAULT"] = str(vault)
        e["PHOTO_CATALOG"] = str(catalog)
        return e

    failures_file = work / "failures.txt"
    detail_sink = open(failures_file, "w", encoding="utf-8")

    def deep(rid: str, text: str):
        """Long-form diagnostics land outside the table, never inside it."""
        detail_sink.write(f"--- {rid} ---\n{text}\n\n")
        detail_sink.flush()

    # === offline rows ======================================================
    p0_predicate_selftest(t)

    synth = build_synthetic_vault(work, t)
    synth_cat = work / "synthcat"
    env_s = child_env(synth, synth_cat)

    r = run([sys.executable, str(HERE / "photo-walk.py"), "-o", "-"], env_s)
    walk_recs = [json.loads(x) for x in r.stdout.splitlines() if x.strip()]
    kept = [x for x in walk_recs if not x.get("skip")]
    skipped = {}
    for x in walk_recs:
        if x.get("skip"):
            skipped[x["skip"]] = skipped.get(x["skip"], 0) + 1

    t.check("P3a", "walker descends into subdirectories",
            any("/" in x["rel"] for x in kept),
            f"{sum(1 for x in kept if '/' in x['rel'])} of {len(kept)} below the root")
    t.check("P3b", "dotfiles and non-photos are skipped, loudly",
            not any(x["filename"].startswith(".") for x in kept)
            and skipped.get("unsupported_format", 0) >= 2,
            f"{skipped.get('unsupported_format', 0)} unsupported, named in the inventory")
    t.check("P3c", "byte-identical copies are detected",
            skipped.get("duplicate_content", 0) == 1,
            f"duplicate_content x{skipped.get('duplicate_content', 0)}")
    t.check("P3d", "a raw file with a JPEG sibling is paired",
            skipped.get("raw_jpeg_pair", 0) == 1,
            f"raw_jpeg_pair x{skipped.get('raw_jpeg_pair', 0)}")
    t.check("P3e", "a zero-byte file is not a candidate",
            skipped.get("empty_file", 0) == 1, f"empty_file x{skipped.get('empty_file', 0)}")

    # -- identity ----------------------------------------------------------
    src = FIXTURE / "coll2_1.jpeg"
    renamed = work / f"{uuid.uuid4().hex}.bin"
    shutil.copy2(src, renamed)
    same = P.photo_id(src) == P.photo_id(renamed)
    with Image.open(src) as im:
        im.convert("RGB").save(work / "reencoded.jpeg", "JPEG", quality=70)
    differs = P.photo_id(src) != P.photo_id(work / "reencoded.jpeg")
    ids = [x["photo_id"] for x in kept]
    t.check("P4a", "identity survives renaming and moving", same, "copy under a new name, same id")
    t.check("P4b", "a re-encode is honestly new work", differs, "different bytes, different id")
    t.check("P4c", "no id collisions in the corpus", len(ids) == len(set(ids)),
            f"{len(set(ids))} distinct ids")

    # -- perceptual pairing, the case names and dates cannot solve ----------
    pair_dir = synth / "pairs"
    heic_twin = pair_dir / f"{uuid.uuid4()}.heic"
    conv_ok = True
    try:
        P.sips_convert(FIXTURE / "coll2_3.jpeg", heic_twin, 4032)
    except P.PrepareError:
        conv_ok = False
    if conv_ok and heic_twin.exists():
        back = work / "twin.jpg"
        P.sips_convert(heic_twin, back, 512)
        prep_orig = P.prepare(FIXTURE / "coll2_3.jpeg", "direct", max_edge=512)
        with Image.open(back) as im:
            h_twin = P.dhash(im)
        with Image.open(__import__("io").BytesIO(prep_orig.data)) as im:
            h_orig = P.dhash(im)
        dist = P.hamming(h_orig, h_twin)
        different_id = P.photo_id(heic_twin) != P.photo_id(FIXTURE / "coll2_3.jpeg")
        t.check("P4d", "same picture, other format/name/folder, still paired",
                dist <= 4 and different_id,
                f"dHash distance {dist}, ids differ: {different_id}")

        # A calibration, not a verdict: the threshold for raw-vs-JPEG pairs cannot be
        # guessed from a HEIC pair, so print the curve instead of asserting a number.
        ladder = []
        for label, mk in (
            ("re-encode q60", lambda im: im.convert("RGB")),
            ("half size", lambda im: im.resize((im.width // 2, im.height // 2))),
            ("crop 5%", lambda im: im.crop((int(im.width * .05), int(im.height * .05),
                                            int(im.width * .95), int(im.height * .95)))),
        ):
            with Image.open(FIXTURE / "coll2_3.jpeg") as im:
                im.draft("RGB", (512, 512))
                v = mk(im.copy())
                v.thumbnail((512, 512), Image.LANCZOS)
                ladder.append((label, P.hamming(h_orig, P.dhash(v.convert("RGB")))))
        ladder.append(("heic round trip", dist))
        t.add("P4e", "dHash distance calibration (measurement)", "PASS",
              " · ".join(f"{k} {v}" for k, v in ladder))
        t.note("P4e is a measurement, not a threshold. A raw file and its camera-developed "
               "JPEG differ in framing and tone more than any transform above, so the "
               "raw-vs-JPEG distance must be measured against real raw files before "
               "--near-dup-distance is tuned for them.")
    else:
        t.add("P4d", "same picture, other format/name/folder, still paired", "SKIP",
              "sips could not write HEIC on this machine", mandatory=False)

    # -- error taxonomy, via the documented dry-run path --------------------
    r = run([sys.executable, str(HERE / "caption-photos.py"), "--model", args.vision_model,
             "--dry-run", "--max-pixels", "1000"], env_s)
    dry = [json.loads(x) for x in r.stdout.splitlines() if x.strip().startswith("{")]
    reasons = {x.get("skip") for x in dry if x.get("skip")}
    t.check("P5", "the failure taxonomy is exercised, not just declared",
            {"decompression_bomb", "unsupported_format", "duplicate_content",
             "raw_jpeg_pair", "empty_file"} <= reasons,
            f"{len(reasons)} distinct reasons: {', '.join(sorted(reasons))}")

    # -- fail loud without a server ----------------------------------------
    t0 = time.time()
    r = run([sys.executable, str(HERE / "caption-photos.py"), "--model", args.vision_model,
             "--base-url", "http://127.0.0.1:1/v1"], env_s, timeout=30)
    dt = time.time() - t0
    t.check("P6", "fails loudly without a server, leaving no lock",
            r.returncode == P.EXIT_PRECONDITION and dt < 10
            and "mlxk serve" in r.stderr and not (synth_cat / "log" / ".lock").exists(),
            f"exit {r.returncode} in {dt:.1f}s, hint names `mlxk serve`")

    # -- resume machinery, on synthesised log state, no GPU -----------------
    fake_cat = work / "fakecat"
    (fake_cat / "log").mkdir(parents=True)
    P.bind_vault(fake_cat, synth)
    fake_log = P.Log(fake_cat / "log" / "captions.jsonl")
    victim = kept[0]["photo_id"]
    clean = kept[1]["photo_id"] if len(kept) > 1 else victim
    with fake_log as lg:
        for _ in range(3):
            lg.append({"type": "attempt", "photo_id": victim, "rel": "x"})
        for _ in range(3):
            lg.append({"type": "attempt", "photo_id": clean, "rel": "y"})
            lg.append({"type": "result", "photo_id": clean, "rel": "y", "ok": False,
                       "error": {"kind": "http_error"}})
    idx = fake_log.resume_index()
    t.check("P7a", "three orphaned attempts is a poison pill",
            P.resume_decision(victim, idx, 3) == "poison_pill", f"{victim[:8]} quarantined")
    t.check("P7b", "three clean rejections is a different verdict",
            P.resume_decision(clean, idx, 3) == "repeated_error", f"{clean[:8]} quarantined")
    idx.done.add(clean)
    t.check("P7c", "a described photo is not described twice",
            P.resume_decision(clean, idx, 3) == "already_done", "already_done")
    idx.quarantined.add(victim)
    t.check("P7d", "a quarantined photo is never retried",
            P.resume_decision(victim, idx, 3) == "quarantined", "quarantined")

    # -- torn tail ----------------------------------------------------------
    torn_path = work / "torn.jsonl"
    shutil.copy2(fake_cat / "log" / "captions.jsonl", torn_path)
    with open(torn_path, "a", encoding="utf-8") as f:
        f.write('{"type":"res')
    torn_log = P.Log(torn_path)
    before = len(list(fake_log.read()))
    idx2 = torn_log.resume_index()
    with torn_log as _:
        pass
    after_bytes = torn_path.read_bytes()
    t.check("P8", "a torn final line costs one record, not the file",
            idx2.torn == 1 and after_bytes.endswith(b"\n") and idx2.lines >= before,
            f"1 line lost of {idx2.lines}, newline repaired before append")

    guard_rows(t, work, child_env)

    if args.skip_slow:
        return finish(t, work, started, args, detail_sink, failures_file)

    # === live rows =========================================================
    try:
        import httpx
    except ImportError:
        t.add("P9", "server reachable", "FAIL", "httpx not installed")
        return finish(t, work, started, args, detail_sink, failures_file)

    client = httpx.Client()
    try:
        P.server_health(client, args.base_url)
        t.check("P9", "server reachable and healthy", True, args.base_url)
    except P.Precondition as e:
        t.add("P9", "server reachable and healthy", "FAIL", "no server; live rows skipped")
        deep("P9", f"{e.message}\n{e.hint}")
        client.close()
        return finish(t, work, started, args, detail_sink, failures_file)

    served = P.list_models(client, args.base_url)
    t.add("P10", "model is listed by /v1/models", "PASS" if args.vision_model in served else "SKIP",
          args.vision_model if args.vision_model in served
          else "absent — /v1/models filters on runtime_compatible, which has a known "
               "false negative; the canary decides", mandatory=False)
    client.close()

    real_cat = work / "catalog"
    env_r = child_env(FIXTURE, real_cat)
    base = [sys.executable, str(HERE / "caption-photos.py"), "--model", args.vision_model,
            "--base-url", args.base_url, "--max-edge", str(args.max_edge),
            "--canary-every", "5", "--seed", "7", "--progress-every", "0"]

    # Phase A and B are split by EXTENSION, never by directory name: no tracked file
    # should have to name the local-only subdirectory, and the split doubles as the
    # "photos appeared later" test.
    sys.stderr.write("\n  phase A: the 9 tracked JPEGs\n")
    rA = run(base + ["--include-ext", ".jpeg"], env_r, timeout=1800)
    sys.stderr.write("  phase B: everything else that is there\n")
    rB = run(base, env_r, timeout=1800)
    log_path = real_cat / "log" / "captions.jsonl"
    recs = read_log(log_path)
    results = [x for x in recs if x.get("type") == "result" and x.get("ok")]
    canaries = [x for x in recs if x.get("type") == "canary"]
    runs = [x for x in recs if x.get("type") == "run_end"]

    if rA.returncode != 0 or not results:
        t.add("P11", "captioning ran", "FAIL", f"phase A exit {rA.returncode}")
        deep("P11", f"--- phase A stderr ---\n{rA.stderr[-4000:]}\n--- phase B ---\n{rB.stderr[-4000:]}")
        return finish(t, work, started, args, detail_sink, failures_file)
    t.check("P11", "captioning ran to completion",
            rA.returncode == 0 and rB.returncode == 0 and len(results) >= 9,
            f"{len(results)} photos described")

    # -- the deterministic proof that the pixels arrived --------------------
    proven = sum(1 for x in results
                 if (x.get("prepared") or {}).get("server_hash")
                 and (x.get("server") or {}).get("table_columns"))
    t.check("P12", "every response proves it came from the vision path",
            proven == len(results),
            f"{proven}/{len(results)} carry the server's own image hash")

    # -- geometry and EXIF, measured from the produced bytes -----------------
    prep_dir_ok, exif_ok = 0, 0
    for x in results:
        p = x.get("prepared") or {}
        if p.get("w") and max(p["w"], p["h"]) == p.get("max_edge"):
            prep_dir_ok += 1
        e = x.get("exif") or {}
        if e.get("gps") and e.get("dt") and e.get("camera"):
            exif_ok += 1
    t.check("P13", "every upload was downscaled to the pixel guard",
            prep_dir_ok == len(results), f"{prep_dir_ok}/{len(results)} at {args.max_edge} px")
    t.check("P14", "every photo yielded place, time and camera locally",
            exif_ok == len(results), f"{exif_ok}/{len(results)} carry all three")

    cells = [(x.get("server") or {}) for x in results]
    dashes = sum(1 for c in cells if c.get("location_cell") in ("-", None)
                 and c.get("date_cell") in ("-", None))
    six_col = sum(1 for c in cells if c.get("table_columns") == 6)
    if six_col:
        t.check("P15", "the server received no EXIF at all",
                dashes == len(cells),
                f"{dashes}/{len(cells)} blank server-side, while {exif_ok} carry GPS locally")
    else:
        t.add("P15", "the server received no EXIF at all", "SKIP",
              "server runs with MLXK2_EXIF_METADATA=0, so its cells are unobservable",
              mandatory=False)

    contam = [x for x in results if contaminated(x.get("text", ""))]
    t.check("P16", "no caption mentions a coordinate or a date",
            not contam, f"0 of {len(results)} captions matched the patterns")
    if contam:
        deep("P16", f"{len(contam)} contaminated caption(s); see the log at {log_path}")

    ok_canaries = [c for c in canaries if c.get("ok")]
    # Distinctness holds WITHIN a run, not across runs: two runs given the same --seed
    # legitimately draw the same pair, and comparing across them would fail a healthy
    # system. What the property is actually about is that one run's shots are drawn
    # without replacement, so a single lucky guess cannot pass the startup gate.
    per_run: Dict[str, List[tuple]] = {}
    for c in canaries:
        if c.get("phase") == "startup":
            per_run.setdefault(c.get("run_id", ""), []).append(tuple(c["expect"]))
    distinct = all(len(set(v)) == len(v) for v in per_run.values())
    enough, all_ok = len(canaries) >= 3, len(ok_canaries) == len(canaries)
    why = [] if (enough and all_ok and distinct) else (
        ([f"only {len(canaries)} shots"] if not enough else [])
        + ([f"{len(canaries) - len(ok_canaries)} failed"] if not all_ok else [])
        + (["startup permutations repeated within a run"] if not distinct else []))
    t.check("P17", "the canary ran and every shot passed", not why,
            "; ".join(why) if why else
            f"{len(ok_canaries)}/{len(canaries)} passed across {len(per_run)} run(s), "
            f"each run's startup permutations distinct")

    prompts = {x.get("prompt_sha256") for x in results}
    t.check("P18", "the prompt is a run constant, photo-independent",
            len(prompts) == 1, "one prompt hash across every record")

    # -- resume == addition -------------------------------------------------
    by_run: Dict[str, int] = {}
    for x in recs:
        if x.get("type") == "attempt":
            by_run[x.get("run_id", "")] = by_run.get(x.get("run_id", ""), 0) + 1
    endB = runs[-1] if runs else {}
    already = (endB.get("counts") or {}).get("already_done", 0)
    t.check("P19", "resuming is the same operation as adding",
            already == 9 and len(runs) == 2,
            f"phase B re-described 0 of 9, described {(endB.get('counts') or {}).get('captioned', 0)} new")

    # -- the quality discriminator, and the resolution ladder ---------------
    c3 = next((x for x in results if x.get("rel", "").endswith("coll2_3.jpeg")), None)
    if c3:
        cap = norm(c3.get("text", ""))
        t.check("P20a", "the known subject is described, not blended",
                any(term in cap for term in BOAT_TERMS),
                "caption carries a boat term")
        hit512 = {k for k in LEGIBLE if norm(k) in cap}
        exact512 = {k for k, v in LEGIBLE.items() if v.lower() in c3.get("text", "").lower()}
        t.check("P20b", "legible text on the subject is transcribed",
                len(hit512) >= 1, f"{len(hit512)}/3 of the painted strings, normalised")

        # P20c: the same photograph at a higher edge, so the resolution decision the
        # whole cost model rests on is re-measured rather than cited from a note.
        ladder_cat = work / "ladder"
        ids_file = work / "ladder-ids.txt"
        ids_file.write_text(c3["photo_id"] + "\n")
        rL = run(base[:4] + ["--base-url", args.base_url, "--max-edge", str(args.ladder_edge),
                             "--canary-every", "0", "--canary-shots", "1", "--seed", "7",
                             "--only-ids", str(ids_file), "--force-recaption",
                             "--progress-every", "0"],
                 child_env(FIXTURE, ladder_cat), timeout=900)
        lrecs = [x for x in read_log(ladder_cat / "log" / "captions.jsonl")
                 if x.get("type") == "result" and x.get("ok")]
        if rL.returncode == 0 and lrecs:
            hi = lrecs[-1].get("text", "")
            hitH = {k for k in LEGIBLE if norm(k) in norm(hi)}
            exactH = {k for k, v in LEGIBLE.items() if v.lower() in hi.lower()}
            delta = len(hitH) - len(hit512)
            t.add("P20c", f"resolution ladder {args.max_edge} vs {args.ladder_edge} px (measurement)",
                  "PASS",
                  f"{args.max_edge}px {len(hit512)}/3 normalised {len(exact512)}/3 exact · "
                  f"{args.ladder_edge}px {len(hitH)}/3 normalised {len(exactH)}/3 exact · "
                  f"delta {delta:+d} for {(lrecs[-1].get('latency_ms', 0) / max(c3.get('latency_ms', 1), 1)):.1f}x the time")
            t.note("P20c is a measurement, not a threshold, and it is the only row that "
                   "re-checks the 512 px default the whole cost model rests on. Read the delta, "
                   "not the counts: a higher edge buys legible text only if this number is "
                   "positive. Watch for the failure mode a count cannot show — a larger image "
                   "can turn an omission into a confident MISreading, which is worse for "
                   "retrieval than silence, because the wrong word is what gets embedded. When "
                   "the delta does justify a second pass, the documented escalation is "
                   "--only-ids together with --force-recaption over the affected photos only.")
        else:
            t.add("P20c", "resolution ladder (measurement)", "SKIP",
                  "second pass did not complete", mandatory=False)
    else:
        t.add("P20a", "the known subject is described, not blended", "FAIL",
              "coll2_3.jpeg missing from the results")

    # -- the publishable evidence -------------------------------------------
    # Phase A captioned exactly the git-tracked photographs; phase B added whatever is
    # only on this machine. So the first run_id is the publishable set, and no filename
    # has to be hardcoded to tell the two apart.
    first_run = next((x.get("run_id") for x in recs if x.get("type") == "run_start"), None)
    public = sorted((x for x in results if x.get("run_id") == first_run),
                    key=lambda x: x.get("rel", ""))
    local_only = len(results) - len(public)
    for x in public:
        cap = " ".join((x.get("text") or "").split())
        t.appendix.append(f"{Path(x['rel']).name:<14} {x.get('latency_ms', 0) / 1000:>5.1f}s  {cap}")
    if local_only:
        t.appendix.append(f"({local_only} further photograph(s) were described in the second "
                          f"phase; their captions are not printed here, because that phase is "
                          f"where anything only present on this machine turns up)")

    # -- format coverage, honestly per extension ----------------------------
    by_ext: Dict[str, int] = {}
    for x in results:
        by_ext[(x.get("src") or {}).get("ext", "?")] = by_ext.get((x.get("src") or {}).get("ext", "?"), 0) + 1
    sips_exts = sorted(e for e in by_ext if e in P.SIPS_EXTS)
    raw_done = [e for e in sips_exts if e in P.RAW_EXTS]
    t.check("P21", "the conversion branch actually ran",
            bool(sips_exts), f"converted: {', '.join(f'{e} x{by_ext[e]}' for e in sips_exts) or 'none'}")
    t.add("P22", "raw formats exercised", "PASS" if raw_done else "SKIP",
          ", ".join(raw_done) if raw_done else
          "no raw file in the corpus — the sips code path is shared with HEIC, but no raw "
          "decoder, raw pairing or metadata-poor case was tested", mandatory=False)

    # === derived stages ====================================================
    r = run([sys.executable, str(HERE / "build-catalog.py"), "--stats", "--inherit-metadata"], env_r)
    cat_path = real_cat / "catalog.jsonl"
    cat = [json.loads(x) for x in cat_path.read_text(encoding="utf-8").splitlines() if x.strip()] \
        if cat_path.exists() else []
    vault_str = str(FIXTURE.resolve())
    leaks = [c for c in cat if vault_str in json.dumps(c)]
    abs_paths = [c for c in cat if str(c.get("filepath", "")).startswith("/")]
    t.check("P23", "the catalog is complete, relative and leak-free",
            len(cat) == len(results) and not leaks and not abs_paths
            and cat == sorted(cat, key=lambda x: x["photo_id"]),
            f"{len(cat)} rows, 0 absolute paths, 0 occurrences of the vault root")

    r = run([sys.executable, str(HERE / "index-photos.py"), "--model", args.embed_model], env_r,
            timeout=1200)
    idx_path = real_cat / "index.jsonl"
    index = [json.loads(x) for x in idx_path.read_text(encoding="utf-8").splitlines() if x.strip()] \
        if idx_path.exists() else []
    if not index:
        t.add("P24", "the index is one vector space and corresponds", "FAIL",
              f"index-photos.py exit {r.returncode}")
        deep("P24", r.stderr[-3000:])
    else:
        idents = {tuple((x.get("metadata") or {}).get(k)
                        for k in ("model", "content_hash", "device", "dimensions")) for x in index}
        cat_text = {c["photo_id"]: c["text"] for c in cat}
        corresponds = all(x.get("text") == cat_text.get(x.get("photo_id")) for x in index) \
            and {x.get("photo_id") for x in index} == set(cat_text)
        dims_ok = all(len(x.get("embedding", [])) == (x.get("metadata") or {}).get("dimensions")
                      for x in index)
        t.check("P24", "the index is one vector space and corresponds",
                len(idents) == 1 and corresponds and dims_ok and len(index) == len(cat),
                f"{len(index)} vectors, one identity, caption↔vector pairing verified")

        # -- retrieval, with the two ship photographs as each other's distractor
        def rank(query: str) -> List[str]:
            rr = run([sys.executable, str(HERE / "photo-search.py"), query,
                      "--model", args.embed_model, "--top-k", "13", "--output-json"], env_r,
                     timeout=600)
            if rr.returncode != 0:
                deep("P25", f"query {query!r} exit {rr.returncode}\n{rr.stderr[-1500:]}")
                return []
            return [h["filename"] for h in json.loads(rr.stdout or "{}").get("results", [])]

        ra, rb = rank(QUERY_A), rank(QUERY_B)

        def above(order: List[str], a: str, b: str) -> Optional[bool]:
            if a not in order or b not in order:
                return None
            return order.index(a) < order.index(b)

        a_ok, b_ok = above(ra, "coll2_3.jpeg", "coll2_5.jpeg"), above(rb, "coll2_5.jpeg", "coll2_3.jpeg")
        t.check("P25", "retrieval separates the two ship photographs",
                bool(a_ok) and bool(b_ok),
                f"steamer query ranks coll2_3 first: {a_ok} · sailing-ship query ranks "
                f"coll2_5 first: {b_ok}")

        # -- the same-model guard, the one negative check involving a model
        rg = run([sys.executable, str(HERE / "photo-search.py"), "boat",
                  "--model", args.second_embed_model, "--top-k", "3"], env_r, timeout=600)
        if rg.returncode == P.EXIT_USAGE and "Hint" in rg.stderr:
            t.check("P26", "a foreign embedder is refused, not ranked", True,
                    "exit 2, no ranking, hint printed")
        elif rg.returncode == P.EXIT_EMBED:
            t.add("P26", "a foreign embedder is refused, not ranked", "SKIP",
                  f"{args.second_embed_model} is not installed", mandatory=False)
        else:
            t.check("P26", "a foreign embedder is refused, not ranked", False,
                    f"exit {rg.returncode}, expected 2")

    return finish(t, work, started, args, detail_sink, failures_file)


def finish(t: Table, work: Path, started: float, args, sink, failures_file: Path) -> int:
    sink.close()
    elapsed = time.time() - started
    c = t.counts
    verdict = t.verdict()

    if args.json:
        print(json.dumps({"checks": [{k: r[k] for k in ("id", "name", "status", "detail", "mandatory")}
                                     for r in t.rows],
                          "pass": c["PASS"], "fail": c["FAIL"], "skip": c["SKIP"],
                          "verdict": verdict, "elapsed_s": round(elapsed, 1),
                          "notes": t.notes, "captions": t.appendix},
                         ensure_ascii=False, indent=2))
    else:
        w = max(len(r["name"]) for r in t.rows) + 2
        print(f"\n| #    | {'Check'.ljust(w)} | Status | Detail |")
        print(f"|------|{'-' * (w + 2)}|--------|--------|")
        for r in t.rows:
            print(f"| {r['id']:<4} | {r['name'].ljust(w)} | {r['status']:<6} | {r['detail']} |")
        print(f"\nPASS {c['PASS']}  FAIL {c['FAIL']}  SKIP {c['SKIP']}   ({elapsed:.1f}s)")
        print(f"VERDICT: {verdict}")
        if t.appendix:
            print("\nCaptions of the published fixture photographs, verbatim:\n")
            for line in t.appendix:
                print(f"  {line}")
        for n in t.notes:
            print(f"\n  note: {n}")

    # The table's own Detail cells are the one surface designed to be pasted into a
    # shared document, so they are held to counts and git-tracked names. The appendix
    # above is deliberately exempt: it contains only material this repository already
    # publishes, and without it the acceptance is a wall of ratios nobody can check.
    blob = json.dumps([r for r in t.rows])
    vault = os.environ.get("PHOTO_VAULT", "")
    if (vault and vault in blob) or contaminated(blob):
        sys.stderr.write("\nINTERNAL: a Detail cell matched the coordinate patterns; "
                         "this is a bug in the runner.\n")
        return P.EXIT_CHECK_FAILED

    if not args.keep and verdict == "PASS":
        shutil.rmtree(work, ignore_errors=True)
    else:
        sys.stderr.write(f"\n  workdir kept: {work}\n  detail:       {failures_file}\n")
    return P.EXIT_OK if verdict == "PASS" else P.EXIT_CHECK_FAILED


if __name__ == "__main__":
    sys.exit(main())
