"""Shared machinery for the photo-rag example: identity, walking, preparation, log, HTTP.

This module is imported by the hyphenated scripts next to it (a hyphen is not a legal
module name, hence the underscore). It contains no CLI of its own.

Nothing here knows about git, about the mlx-knife source tree, or about any path other
than the two the operator names: PHOTO_VAULT and PHOTO_CATALOG.

Input:  a photo library directory, an OpenAI-compatible base URL
Output: JSONL records; see the module-level record documentation in caption-photos.py
"""

from __future__ import annotations

import base64
import errno as errno_mod
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageOps

# ---------------------------------------------------------------------------
# Exit codes, shared by every script in this directory so a caller can branch on
# them without parsing text. 2 is argparse's own usage code and is reused for
# missing environment on purpose: both mean "you invoked this wrong".
# ---------------------------------------------------------------------------
EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_USAGE = 2
EXIT_PRECONDITION = 3
EXIT_CANARY = 4
EXIT_VAULT_MISMATCH = 5
EXIT_LOCKED = 6
EXIT_INTERRUPTED = 7
EXIT_LOG_UNREADABLE = 8
EXIT_EMBED = 9
EXIT_ENVIRONMENT = 10       # the library itself became unreachable; nothing is wrong with it

SCHEMA = 1
TOOL = "photo-rag/1.0"

# ---------------------------------------------------------------------------
# Format branches.
#
# "direct" means Pillow opens the original. "sips" means macOS converts it first;
# Pillow cannot open HEIC (verified: UnidentifiedImageError on every camera HEIC
# tested, and pillow-heif is deliberately not a dependency of this example).
#
# Measured 2026-08-03 against the project fixture: sips JPEG->HEIC->JPEG preserves
# 15 GPS IFD keys, DateTimeOriginal and Make/Model — the same counts a camera HEIC
# yields through the same path. That is why the rule is "convert first, then read
# EXIF from the converted file", never "interrogate the original".
# ---------------------------------------------------------------------------
DIRECT_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".tif", ".tiff", ".bmp"})
HEIF_EXTS = frozenset({".heic", ".heif", ".avif"})
RAW_EXTS = frozenset({".arw", ".cr2", ".cr3", ".nef", ".dng", ".orf", ".raf", ".rw2", ".srw", ".pef"})
SIPS_EXTS = HEIF_EXTS | RAW_EXTS
ALL_EXTS = DIRECT_EXTS | SIPS_EXTS

SIPS = "/usr/bin/sips"

# Skip reasons. `ext_filter` is deliberately distinct from `unsupported_format`:
# one means "you asked me not to", the other means "I cannot". Conflating them
# would make a filtered run look like it hit failures.
# `decode_error` vs `read_error` is the same distinction: the photograph, or the
# machine it sits on. Told apart by errno in _prepare_from().
SKIP_REASONS = frozenset({
    "unsupported_format", "ext_filter", "raw_jpeg_pair", "duplicate_content",
    "decompression_bomb", "sips_failed", "no_converter", "read_error", "empty_file",
    "symlink", "decode_error", "permission",
})

ERROR_KINDS = frozenset({
    "http_error", "transport_error", "timeout", "prepare_error", "empty_caption",
    "no_image_seen", "hash_mismatch", "unparsed_metadata_block", "prompt_drift",
})

# ---------------------------------------------------------------------------
# 1. Identity — content, never path
# ---------------------------------------------------------------------------
PHOTO_ID_DOMAIN = b"mlxk-photo-id-v1\0"
_HEAD_TAIL = 65536
_WHOLE_BELOW = 2 * _HEAD_TAIL


def photo_id(path: Path, size: Optional[int] = None) -> str:
    """Identify a photo by its content: size, first 64 KiB, last 64 KiB.

    Fixed 128 KiB of I/O whether the file is a 1 MB JPEG or a 60 MB raw. No path,
    no mtime, no inode — which is the whole point: "adding new photos" and "resuming
    an interrupted run" become literally the same operation, and renaming or
    re-sorting the library causes no rework. mtime is excluded deliberately; it is
    unreliable on the SMB mounts this tool is aimed at (some files carry genuine
    sub-second remainders, others a flat zero).

    Head *and* tail because the head of a JPEG or HEIC is container, EXIF and an
    embedded thumbnail — highly similar between burst frames from one camera — while
    the tail is entropy-coded image data.

    Precedent: ADR-025 content_hash v2 hashes size plus the first 4 KB of each
    safetensors file. Same shape, sized up here because a photo's first 4 KB is
    mostly EXIF. Note this is a *different* hash from the one the mlx-knife server
    computes over an upload (sha256 of the full bytes, truncated to 8 hex); a record
    carries both, because they identify different things — the original on disk and
    the downscaled JPEG that actually reached the model.
    """
    if size is None:
        size = path.stat().st_size
    h = hashlib.sha256()
    h.update(PHOTO_ID_DOMAIN)
    h.update(str(size).encode("ascii") + b"\0")
    with open(path, "rb") as f:
        if size <= _WHOLE_BELOW:
            h.update(f.read())
        else:
            h.update(f.read(_HEAD_TAIL))
            f.seek(-_HEAD_TAIL, os.SEEK_END)
            h.update(f.read(_HEAD_TAIL))
    return h.hexdigest()[:16]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def server_image_hash(data: bytes) -> str:
    """The identity the mlx-knife server assigns to an uploaded image.

    Mirrors mlxk2/tools/vision_adapter.py, which names each decoded image
    `image_<sha256(raw_bytes)[:8]>.<mime>` and renders that name into the response's
    metadata table. Computing it client-side is what lets us prove the server hashed
    exactly the bytes we sent.
    """
    return hashlib.sha256(data).hexdigest()[:8]


# ---------------------------------------------------------------------------
# 2. Walking — assume nothing about the tree
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    path: Path
    rel: str
    ext: str
    bytes: int
    branch: Optional[str]             # "direct" | "sips"; None if it was never a candidate
    photo_id: Optional[str] = None
    skip: Optional[str] = None
    detail: Optional[str] = None

    # No `dup_count`: walk() yields the keeper before any duplicate of it is known, so
    # such a field could never have been written. Group the inventory by `photo_id`
    # instead — it appears on the keeper and on every duplicate alike.

    def record(self) -> Dict[str, Any]:
        rec: Dict[str, Any] = {
            "photo_id": self.photo_id,
            "rel": self.rel,
            "filename": Path(self.rel).name,
            "ext": self.ext,
            "bytes": self.bytes,
            "branch": self.branch,
        }
        if self.skip:
            rec["skip"] = self.skip
            if self.detail:
                rec["detail"] = self.detail
        return rec


def _norm(s: str) -> str:
    """NFC-normalise a path component.

    macOS hands out NFD from some filesystems and NFC from others, so the same
    photo can present two different byte strings for its name. Normalising keeps
    the RAW/JPEG pair heuristic and the exclude list from silently missing matches
    on names with umlauts or accents.
    """
    return unicodedata.normalize("NFC", s)


def branch_for(ext: str) -> Optional[str]:
    if ext in DIRECT_EXTS:
        return "direct"
    if ext in SIPS_EXTS:
        return "sips"
    return None


def walk(
    vault: Path,
    *,
    include_ext: Optional[Sequence[str]] = None,
    exclude_dir: Sequence[str] = (),
    pair_dedupe: bool = True,
    prefer_raw: bool = False,
    follow_symlinks: bool = False,
    limit: Optional[int] = None,
    emit_skips: bool = True,
) -> Iterator[Candidate]:
    """Yield one Candidate per file found under `vault`.

    Deliberately assumes NO directory structure. The library this is aimed at mixes
    `Year/Month/*.jpg` with a photo-app export of `.../originals/<hex>/<UUID>.heic`
    whose names carry no meaning at all — so dates come from EXIF, never from a path,
    and nothing here parses a directory name.

    Three passes, cheapest first:
      1. scan names and sizes (no reads)
      2. RAW/JPEG sibling de-duplication (no reads)
      3. content identity for the survivors (128 KiB per file)

    The expensive pass runs last so a filtered or aborted walk pays for nothing it
    did not need.
    """
    include = frozenset(e.lower() for e in include_ext) if include_ext else None
    excluded = frozenset(_norm(d) for d in exclude_dir)

    # -- pass 1: names and sizes ------------------------------------------------
    found: List[Candidate] = []

    def _dir_unreadable(e: OSError) -> None:
        """os.walk discards directory errors unless you ask for them.

        Without this a share that dies during the scan simply stops yielding, and the
        candidate list comes back short with nothing to show for it — a run over 60% of
        a library that reports success. The failure has to arrive as a record like any
        other, so route it through the same skip channel every other refusal uses.
        """
        where = getattr(e, "filename", None) or str(vault)
        try:
            rel_dir = str(Path(where).relative_to(vault))
        except ValueError:
            rel_dir = where
        found.append(Candidate(Path(where), rel_dir, "", 0, None,
                               skip="permission" if e.errno in (errno_mod.EACCES, errno_mod.EPERM)
                               else "read_error",
                               detail=f"directory unreadable: {e.strerror or e}"[:200]))

    for root, dirs, files in os.walk(vault, followlinks=follow_symlinks,
                                     onerror=_dir_unreadable):
        dirs[:] = [d for d in dirs if not d.startswith(".") and _norm(d) not in excluded]
        for name in sorted(files):
            # Two spellings of the same name, kept apart on purpose. `p` must carry the
            # one the filesystem handed out, because that is the one open() will find;
            # `rel` carries the NFC form, because that is what has to compare equal
            # across mounts. Normalising `p` too works on macOS only because the kernel
            # folds on lookup — on SMB or NFS it names a file that does not exist, and
            # the failure surfaces as read_error rather than as an encoding problem.
            p = Path(root) / name
            rel = _norm(str(p.relative_to(vault)))
            name = _norm(name)
            ext = p.suffix.lower()

            if name.startswith("."):
                # .DS_Store and friends. A skip record, not silence — "skipped
                # loudly, never silently" is a decision of this example.
                if emit_skips:
                    found.append(Candidate(p, rel, ext, 0, None,
                                           skip="unsupported_format", detail="dotfile"))
                continue

            # A symlinked FILE is followed by open() no matter what os.walk was told —
            # `followlinks` governs directory descent only. So a link inside the library
            # pointing anywhere on the machine would be read, hashed, uploaded and
            # captioned, and its record would show an unremarkable library-relative path.
            # The boundary the operator drew by naming PHOTO_VAULT has to hold for files
            # as well as for directories.
            if not follow_symlinks and p.is_symlink():
                if emit_skips:
                    found.append(Candidate(p, rel, ext, 0, None, skip="symlink",
                                           detail="symlinked file; --follow-symlinks to include"))
                continue

            br = branch_for(ext)
            if br is None:
                if emit_skips:
                    found.append(Candidate(p, rel, ext, 0, None,
                                           skip="unsupported_format",
                                           detail=f"extension {ext or '(none)'}"))
                continue
            if include is not None and ext not in include:
                if emit_skips:
                    found.append(Candidate(p, rel, ext, 0, br, skip="ext_filter"))
                continue
            if br == "sips" and not have_sips():
                # One fact about the machine, not a thousand facts about photographs.
                # Left to the batch it spent each file's retry budget instead.
                if emit_skips:
                    found.append(Candidate(p, rel, ext, 0, br, skip="no_converter",
                                           detail=f"{SIPS} not present; this branch needs macOS"))
                continue

            try:
                size = p.stat().st_size
            except OSError as e:
                if emit_skips:
                    found.append(Candidate(p, rel, ext, 0, br,
                                           skip="permission" if e.errno in (errno_mod.EACCES,
                                                                           errno_mod.EPERM)
                                           else "read_error",
                                           detail=str(e)[:200]))
                continue
            if size == 0:
                if emit_skips:
                    found.append(Candidate(p, rel, ext, 0, br, skip="empty_file"))
                continue

            found.append(Candidate(p, rel, ext, size, br))

    # -- pass 2: RAW + JPEG of the same shot -----------------------------------
    # A name heuristic, not an identity question: the two files have different
    # bytes and therefore different ids. Describing the same picture twice costs
    # hours over a large library and adds nothing.
    if pair_dedupe:
        by_stem: Dict[Tuple[str, str], List[Candidate]] = {}
        for c in found:
            if c.skip:
                continue
            # Grouping key, so both halves must be the COMPARISON spelling. `c.path`
            # carries whatever the filesystem said, which is the wrong side of the
            # distinction to key on: two spellings of one stem would fail to pair.
            by_stem.setdefault((_norm(str(c.path.parent)),
                                _norm(c.path.stem).lower()), []).append(c)
        for group in by_stem.values():
            if len(group) < 2:
                continue
            raws = [c for c in group if c.ext in RAW_EXTS]
            jpegs = [c for c in group if c.ext in DIRECT_EXTS]
            if not raws or not jpegs:
                continue
            losers = jpegs if prefer_raw else raws
            keeper = (raws if prefer_raw else jpegs)[0]
            for c in losers:
                c.skip = "raw_jpeg_pair"
                c.detail = f"prefers sibling {keeper.ext}"

    # -- pass 3: content identity ----------------------------------------------
    seen: set = set()          # photo_ids already emitted; membership is all we need
    emitted = 0
    for c in found:
        if c.skip:
            if emit_skips:
                yield c
            continue
        try:
            c.photo_id = photo_id(c.path, c.bytes)
        except OSError as e:
            c.skip, c.detail = "read_error", str(e)[:200]
            if emit_skips:
                yield c
            continue

        if c.photo_id in seen:
            c.skip = "duplicate_content"
            c.detail = "same content as an earlier file"
            if emit_skips:
                yield c
            continue

        seen.add(c.photo_id)
        yield c
        emitted += 1
        if limit is not None and emitted >= limit:
            return


class Witness:
    """A second question to ask the environment when a single file will not open.

    At the point of failure the distinction is unavailable: ENOENT from a deleted
    photograph and ENOENT from a storage volume that went away are the same errno on
    the same call. It cannot be recovered from the exception, only by asking something
    else — and the something else must be sharper than `vault_root.exists()`, because
    an unmounted share commonly leaves its mountpoint behind as an empty directory
    that answers yes.

    Two probes, cheapest first:

      * `st_dev` of the library root, remembered at startup. An unmounted volume takes
        its device number with it — the leftover mountpoint belongs to the PARENT
        filesystem, so the number changes even though the path still resolves. One
        stat, and it also catches a remount, which a file read would call healthy even
        though every handle and every identity taken before it is now suspect.
      * the failing file's own directory, when there is one — the sharpest probe
        available, and the only one that answers the question actually being asked.
      * failing that, a byte from a file that was demonstrably readable earlier in
        this run. One that still reads proves the storage is serving; a single one
        that does not proves nothing, because a photograph may simply have been
        deleted while the run was working, and that is an ordinary Tuesday.

    The verdicts are deliberately NOT symmetric, and this is the whole design:

      * a witness that FAILS is proof that the environment is gone;
      * a witness that PASSES proves nothing about a failure observed a moment ago.
        It does not acquit the environment, it merely fails to convict it — the mount
        may have returned in between. So a caller must treat that as INCONCLUSIVE and
        look again at the file itself, never as a licence to blame the photograph.

    One observation must not have a permanent consequence. That was the whole defect.
    """

    def __init__(self, root: Path, keep: int = 3):
        self.root = root
        self.keep = keep
        self.paths: List[Path] = []
        try:
            self.dev: Optional[int] = root.stat().st_dev
        except OSError:
            self.dev = None

    def remember(self, path: Path) -> None:
        """Offer a file that was just read successfully."""
        if len(self.paths) < self.keep:
            self.paths.append(path)

    def alive(self, near: Optional[Path] = None) -> Tuple[bool, str]:
        """(True, "") if the environment still looks like itself, else (False, why)."""
        try:
            dev = self.root.stat().st_dev
        except OSError as e:
            return False, f"library root unreadable: {e.strerror or e}"
        if self.dev is not None and dev != self.dev:
            return False, "library root is on a different device than at startup"

        # When there is a specific failure, its own directory is the sharpest probe
        # there is, and it is the ONLY one that answers the question being asked. A
        # library can span more than one mount; witnesses collected during the walk all
        # come from wherever it went first, so they would vouch — perfectly honestly —
        # for a share that is not the one that just died.
        if near is not None:
            try:
                near.stat()
            except OSError as e:
                return False, f"{near.name}/: {e.strerror or e}"
            return True, ""     # the storage under that file is serving ⇒ it IS the file

        # No specific failure to stand next to (the post-walk check). Here a single
        # unreadable witness proves nothing — it may simply have been deleted while the
        # run was working, which is ordinary. One witness that still reads is enough to
        # show the storage is serving; only losing all of them is evidence of absence.
        why = []
        for probe in self.paths:
            try:
                with open(probe, "rb") as fh:
                    fh.read(1)
                return True, ""
            except OSError as e:
                why.append(f"{probe.name}: {e.strerror or e}")
        return (True, "") if not why else (False, "; ".join(why))


# ---------------------------------------------------------------------------
# 3. EXIF and preparation
# ---------------------------------------------------------------------------
_EXIF_IFD = 0x8769
_GPS_IFD = 0x8825
_TAG_MAKE, _TAG_MODEL, _TAG_DATETIME = 0x010F, 0x0110, 0x0132
_TAG_DATETIME_ORIGINAL = 0x9003


@dataclass
class Exif:
    gps: Optional[Tuple[float, float]] = None
    dt: Optional[str] = None
    camera: Optional[str] = None
    orientation: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {"gps": list(self.gps) if self.gps else None, "dt": self.dt,
                "camera": self.camera, "orientation": self.orientation}


def _dms_to_deg(dms, ref) -> Optional[float]:
    try:
        d, m, s = (float(x) for x in dms)
    except (TypeError, ValueError):
        return None
    deg = d + m / 60.0 + s / 3600.0
    if str(ref).upper() in ("S", "W"):
        deg = -deg
    return deg


def read_exif(im: Image.Image) -> Exif:
    """Read the three axes that matter, from an image Pillow can already open.

    Called on the *converted* file for the sips branch — see the module header.
    Every failure mode here is non-fatal: a photo without EXIF is still a photo.
    """
    out = Exif()
    try:
        base = im.getexif()
    except Exception:  # noqa: BLE001 — a broken EXIF block must not kill a run
        return out
    if not base:
        return out

    try:
        out.orientation = int(base.get(0x0112)) if base.get(0x0112) else None
    except (TypeError, ValueError):
        pass

    make, model = base.get(_TAG_MAKE), base.get(_TAG_MODEL)
    if model:
        out.camera = f"{str(make).strip()} {str(model).strip()}".strip() if make else str(model).strip()

    try:
        exif_ifd = base.get_ifd(_EXIF_IFD)
    except Exception:  # noqa: BLE001
        exif_ifd = {}
    raw_dt = exif_ifd.get(_TAG_DATETIME_ORIGINAL) or base.get(_TAG_DATETIME)
    if raw_dt:
        # EXIF spells it "2025:05:14 13:59:12"; ISO-8601 is what every consumer wants.
        s = str(raw_dt).strip()
        m = re.match(r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})", s)
        if m:
            out.dt = "{}-{}-{}T{}:{}:{}".format(*m.groups())

    try:
        gps_ifd = base.get_ifd(_GPS_IFD)
    except Exception:  # noqa: BLE001
        gps_ifd = {}
    if gps_ifd:
        lat = _dms_to_deg(gps_ifd.get(2), gps_ifd.get(1))
        lon = _dms_to_deg(gps_ifd.get(4), gps_ifd.get(3))
        if lat is not None and lon is not None and not (lat == 0.0 and lon == 0.0):
            out.gps = (lat, lon)
    return out


class PrepareError(Exception):
    """Raised with a skip/error reason from SKIP_REASONS or ERROR_KINDS.

    `errno` is carried when the operating system supplied one, and its presence is
    the signal that the failure came from the machine rather than from the picture.
    A caller that has to decide whether to blame the photograph reads this, never
    the message text — error strings are the decoder's, and they are not a contract.
    """

    def __init__(self, reason: str, detail: str = "", errno: Optional[int] = None):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail
        self.errno = errno


@dataclass
class Prepared:
    data: bytes                       # the JPEG actually uploaded
    w: int
    h: int
    sha256: str
    server_hash: str
    max_edge: int
    src_w: Optional[int] = None
    src_h: Optional[int] = None
    converter: Optional[str] = None
    exif: Exif = field(default_factory=Exif)

    def as_prepared_dict(self) -> Dict[str, Any]:
        return {"sha256": self.sha256, "server_hash": self.server_hash,
                "bytes": len(self.data), "w": self.w, "h": self.h, "max_edge": self.max_edge}


def sips_geometry(src: Path) -> Tuple[Optional[int], Optional[int]]:
    """Ask sips for the ORIGINAL pixel dimensions.

    Needed because on the sips branch the only file Pillow can open is already the
    downscaled conversion — reading geometry there would silently store the reduced
    size in a field that means "the original" on the direct branch. One field must
    not mean two things.
    """
    if not Path(SIPS).exists():
        return None, None
    try:
        r = subprocess.run([SIPS, "-g", "pixelWidth", "-g", "pixelHeight", str(src)],
                           capture_output=True, text=True, timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        # Geometry is a nicety; not knowing the original size must never cost the photo.
        return None, None
    if r.returncode != 0:
        return None, None
    w = h = None
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("pixelWidth:"):
            w = int(line.split(":", 1)[1])
        elif line.startswith("pixelHeight:"):
            h = int(line.split(":", 1)[1])
    return w, h


def sips_identity(src: Path) -> Optional[str]:
    """Ask sips for Make/Model when Pillow cannot open the file at all.

    Only used on the failure path. A raw container that will not convert would
    otherwise be recorded with no camera against it, which hides it from the one
    query — "which body is this?" — that would explain why it failed. Never worth
    failing over: not knowing the camera must not cost the diagnosis.
    """
    if not have_sips():
        return None
    try:
        r = subprocess.run([SIPS, "-g", "make", "-g", "model", str(src)],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    got = {}
    for line in r.stdout.splitlines():
        k, _, v = line.strip().partition(": ")
        if k in ("make", "model") and v:
            got[k] = v.strip()
    joined = " ".join(x for x in (got.get("make"), got.get("model")) if x)
    return joined or None


def have_sips() -> bool:
    """Is the macOS system converter available on this machine?

    A fact about the machine, so it is asked once per walk rather than once per
    photograph, and never cached across processes — a run is short enough that the
    answer cannot change under it, and long enough that caching would be a lie if it did.
    """
    return Path(SIPS).exists()


def sips_convert(src: Path, dst: Path, max_edge: int, timeout: int = 180) -> None:
    """HEIC/RAW -> JPEG via the macOS system converter.

    `-Z` is resampleHeightWidthMax: the longest edge becomes max_edge, aspect kept.
    No new Python dependency — that is the whole reason this branch exists rather
    than pillow-heif or rawpy.
    """
    if not Path(SIPS).exists():
        raise PrepareError("no_converter", f"{SIPS} not found (this branch needs macOS)")
    try:
        r = subprocess.run([SIPS, "-s", "format", "jpeg", "-Z", str(max_edge),
                            str(src), "--out", str(dst)],
                           capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        # One slow raw file must cost one photo, not the night's work. Without this the
        # exception escapes PrepareError and takes the whole batch down — and a converter
        # that hangs is exactly what a large library eventually produces.
        raise PrepareError("sips_failed", f"converter timed out after {timeout}s") from None
    except OSError as e:
        raise PrepareError("sips_failed", f"{type(e).__name__}: {e}"[:300],
                           errno=e.errno) from None
    if r.returncode != 0 or not dst.exists():
        # A broken HEIC and a vanished volume exit the same way, and sips' message is
        # prose we must not parse. One stat answers it.
        try:
            source_gone = not src.exists()
        except OSError:
            source_gone = True          # cannot even ask ⇒ certainly not the file's fault
        if source_gone:
            raise PrepareError("read_error", f"source unreachable during conversion: {src.name}",
                               errno=errno_mod.ENOENT)
        raise PrepareError("sips_failed", (r.stderr or r.stdout or "").strip()[:500])


def prepare(
    src: Path,
    branch: str,
    *,
    max_edge: int = 512,
    quality: int = 85,
    max_pixels: int = 400_000_000,
    tmpdir: Optional[Path] = None,
) -> Prepared:
    """Produce the exact bytes that will be uploaded, and the EXIF that will not.

    Two guarantees live in this function, and both are load-bearing:

    1. **The upload carries no EXIF.** Pillow writes an APP1 block only when `exif=`
       is passed as a save parameter; `im.info["exif"]` is not propagated on its own.
       Omitting it is therefore a positive guarantee rather than an oversight. This
       is what makes a location-contaminated caption impossible: the server's own
       prompt augmentation has nothing to inject when the bytes carry nothing, so
       the guarantee does not depend on the operator remembering an env var — and
       unlike an env var, the client can verify the result (the response's
       Location/Date/Camera cells come back as "-").
       `exif_transpose` runs first so dropping the orientation tag cannot leave a
       sideways upload.

    2. **The pixel guard.** mlx-knife limits an upload by BYTES (20 MB per image,
       50 MB per request) and never by pixels, so a 3 MB 100-megapixel panorama
       passes every server-side limit and arrives as a Metal allocation. The
       client-side downscale is that missing guard, and MAX_IMAGE_PIXELS with the
       decompression-bomb warning promoted to an error gives it a single explicit
       threshold instead of Pillow's two-tier warn-then-raise.
    """
    converter = None
    src_w = src_h = None
    read_from = src

    if branch == "sips":
        if tmpdir is None:
            raise PrepareError("prepare_error", "sips branch needs a tmpdir")
        src_w, src_h = sips_geometry(src)
        # Named after the content, so two runs cannot collide in the same tmpdir.
        inter = Path(tmpdir) / f"conv-{sha256_hex(str(src).encode())[:16]}.jpg"
        try:
            sips_convert(src, inter, max_edge)
            converter = "sips"
            read_from = inter
            return _prepare_from(read_from, max_edge, quality, max_pixels,
                                 src_w=src_w, src_h=src_h, converter=converter)
        finally:
            # Best-effort. This intermediate still carries the full EXIF block, so it
            # must not linger — but a `finally:` does not run on SIGKILL, which is why
            # the batch also sweeps tmp/ unconditionally when it takes the lock.
            try:
                inter.unlink(missing_ok=True)
            except OSError:
                pass

    return _prepare_from(read_from, max_edge, quality, max_pixels,
                         src_w=None, src_h=None, converter=None)


def _prepare_from(
    path: Path, max_edge: int, quality: int, max_pixels: int,
    *, src_w: Optional[int], src_h: Optional[int], converter: Optional[str],
) -> Prepared:
    old_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            try:
                with Image.open(path) as im:
                    if src_w is None:
                        src_w, src_h = im.size
                    exif = read_exif(im)
                    if im.format == "JPEG":
                        # DCT-domain 1/2, 1/4, 1/8 pre-scale: most of the decode cost
                        # of a large JPEG disappears before a single pixel is resized.
                        im.draft("RGB", (max_edge, max_edge))
                    im = ImageOps.exif_transpose(im) or im
                    im.thumbnail((max_edge, max_edge), Image.LANCZOS)
                    im = im.convert("RGB")
                    buf = io.BytesIO()
                    im.save(buf, "JPEG", quality=quality, optimize=True)  # no exif=
                    w, h = im.size
            except Image.DecompressionBombWarning as e:
                raise PrepareError("decompression_bomb", str(e)[:300]) from e
            except Image.DecompressionBombError as e:
                raise PrepareError("decompression_bomb", str(e)[:300]) from e
            except PrepareError:
                raise
            except OSError as e:
                # No errno = Pillow (UnidentifiedImageError, "truncated"); an errno
                # = the kernel. Only the first is evidence against the photograph.
                if e.errno is None:
                    raise PrepareError("decode_error", str(e)[:300]) from e
                if e.errno in (errno_mod.EACCES, errno_mod.EPERM):
                    raise PrepareError("permission", str(e)[:300], errno=e.errno) from e
                raise PrepareError("read_error", str(e)[:300], errno=e.errno) from e
            except Exception as e:  # noqa: BLE001 — any decoder fault is a skip, not a crash
                raise PrepareError("prepare_error", f"{type(e).__name__}: {e}"[:300]) from e
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit

    data = buf.getvalue()
    return Prepared(data=data, w=w, h=h, sha256=sha256_hex(data),
                    server_hash=server_image_hash(data), max_edge=max_edge,
                    src_w=src_w, src_h=src_h, converter=converter, exif=exif)


# ---------------------------------------------------------------------------
# 4. The server's metadata block
#
# Every vision response whose request carried an image is prefixed with this block.
# It is emitted by VisionRunner._add_filename_mapping(), which has exactly one call
# site: inside VisionRunner.generate(), guarded by `if images:`. The text path never
# reaches it. That single fact is what makes the marker a PROOF rather than a hint —
# see caption_once() below.
#
#   <details>
#   <summary>📸 Image Metadata (1 image)</summary>
#
#   <!-- mlxk:filenames -->
#   | Image | Filename | Original | Location | Date | Camera |
#   |-------|----------|----------|----------|------|--------|
#   | 1 | image_<hash8>.jpeg | image_<hash8>.jpeg | 📍 .. | 📅 .. | .. |
#
#   </details>
#
#   <the model's own text starts here>
#
# With MLXK2_EXIF_METADATA=0 the table shrinks to two columns; the wrapper is
# emitted either way. The column count is therefore an observable report of the
# server's EXIF setting, which is the only one of the two server knobs a client can
# see at all (MLXK2_VISION_METADATA_CONTEXT is invisible from here — which is why
# this example does not rely on it; see prepare()).
# ---------------------------------------------------------------------------
MARKER = "<!-- mlxk:filenames -->"
_DETAILS_OPEN = "<details>\n"
_DETAILS_CLOSE = "\n</details>\n\n"


@dataclass
class MetaBlock:
    present: bool
    columns: int = 0
    rows: List[List[str]] = field(default_factory=list)

    def cell(self, row: int, name_index: int) -> Optional[str]:
        if row < len(self.rows) and name_index < len(self.rows[row]):
            return self.rows[row][name_index]
        return None

    @property
    def filename_cell(self) -> Optional[str]:
        return self.cell(0, 1)

    @property
    def image_hash(self) -> Optional[str]:
        """The hash the SERVER computed over the bytes it received, or None.

        Worth storing next to the one we computed locally: with only ours on record,
        any later check of "did the pixels arrive" can restate the claim but cannot
        re-derive it — the server's half is gone by then.
        """
        m = re.match(r"^image_([0-9a-f]{8})\.", self.filename_cell or "")
        return m.group(1) if m else None

    def exif_cells(self) -> Dict[str, Optional[str]]:
        """Location/Date/Camera as the server rendered them, or None on a 2-column table."""
        if self.columns < 6:
            return {"location_cell": None, "date_cell": None, "camera_cell": None}
        return {"location_cell": self.cell(0, 3), "date_cell": self.cell(0, 4),
                "camera_cell": self.cell(0, 5)}


def _split_row(line: str) -> List[str]:
    parts = line.strip().split("|")
    if parts and parts[0].strip() == "":
        parts = parts[1:]
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    return [p.strip() for p in parts]


def strip_metadata_block(content: str) -> Tuple[str, MetaBlock]:
    """Split a vision response into (the model's text, the server's metadata block).

    Deterministic by construction: the block always opens with "<details>\\n" and the
    body always begins after the first "\\n</details>\\n\\n". No heuristics, no
    guessing where the model's prose starts.
    """
    if not content.startswith(_DETAILS_OPEN):
        return content, MetaBlock(present=False)
    end = content.find(_DETAILS_CLOSE)
    if end < 0:
        # An opening tag with no close: the response is not what the contract says.
        # Report it rather than silently returning half a table as a caption.
        return content, MetaBlock(present=False)

    head = content[: end + len(_DETAILS_CLOSE)]
    body = content[end + len(_DETAILS_CLOSE):]
    if MARKER not in head:
        return body, MetaBlock(present=False)

    columns, rows = 0, []
    for line in head.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = _split_row(s)
        if not cells:
            continue
        if cells[0] == "Image" and columns == 0:
            columns = len(cells)
            continue
        if set("".join(cells)) <= set("-: "):        # the separator row
            continue
        if columns:
            rows.append(cells)
    return body, MetaBlock(present=True, columns=columns, rows=rows)


# ---------------------------------------------------------------------------
# 5. Talking to the server
# ---------------------------------------------------------------------------
class ServerError(Exception):
    def __init__(self, kind: str, status: Optional[int] = None, etype: Optional[str] = None,
                 message: str = "", request_id: Optional[str] = None):
        super().__init__(message or kind)
        self.kind, self.status, self.etype = kind, status, etype
        self.message, self.request_id = message, request_id

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": self.kind, "message": self.message[:500]}
        for k in ("status", "etype", "request_id"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d


def _parse_error(resp) -> ServerError:
    """Read an ADR-004 error envelope, not OpenAI's shape.

    mlx-knife answers {"status":"error","error":{"type","message",...},"request_id"};
    FastAPI's own 422 is rewritten to 400 upstream, so a malformed body arrives here
    as a validation_error like any other.
    """
    etype = message = request_id = None
    try:
        body = resp.json()
        err = body.get("error") or {}
        if isinstance(err, dict):
            etype = err.get("type")
            message = err.get("message") or err.get("detail")
        elif isinstance(err, str):
            message = err
        request_id = body.get("request_id")
        if message is None:
            message = body.get("detail") if isinstance(body.get("detail"), str) else None
    except Exception:  # noqa: BLE001 — a non-JSON error body is still an error
        pass
    return ServerError("http_error", status=resp.status_code, etype=etype,
                       message=message or resp.text[:300], request_id=request_id)


def data_uri(jpeg: bytes) -> str:
    """A base64 data URI, the only image form the server accepts.

    Verified against mlxk2/tools/vision_adapter.py: the regex demands
    `^data:image/(jpeg|jpg|png|gif|webp);base64,` and b64decode runs with
    validate=True, so no external URL, no file path, no HEIC mime, and no embedded
    newlines. b64encode emits one unbroken line, which is exactly what is needed.
    """
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def chat_vision(
    client, base_url: str, model: str, jpeg: bytes, prompt: str,
    *, max_tokens: int = 300, timeout: float = 300.0,
) -> Tuple[str, MetaBlock, Dict[str, Any]]:
    """One image, one request. Returns (caption, metadata block, raw response).

    Everything the model is told goes in the LAST user message: on the vision path
    the server extracts media and prompt from that message alone, so a `system` role
    would be silently discarded. Temperature is not sent because the vision path
    forces 0.0 regardless; sending it would only suggest we had a choice.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_uri(jpeg)}},
        ]}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        resp = client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, timeout=timeout)
    except Exception as e:  # noqa: BLE001 — httpx exception types vary by failure
        name = type(e).__name__
        kind = "timeout" if "Timeout" in name else "transport_error"
        raise ServerError(kind, message=f"{name}: {e}"[:300]) from e

    if resp.status_code != 200:
        raise _parse_error(resp)

    body = resp.json()
    try:
        content = body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise ServerError("http_error", status=200, message=f"unexpected response shape: {e}") from e

    caption, meta = strip_metadata_block(content)
    return caption.strip(), meta, body


def assert_pixels_arrived(meta: MetaBlock, prepared_server_hash: str) -> None:
    """The deterministic defence against a silent media downgrade.

    A text-only model handed an `image_url` answers HTTP 200 with a fluent
    hallucination: mlx-knife replaces the images in the prompt with the literal
    "[N image(s) were attached]" and there is no error, no warning field, nothing in
    the response to notice. Sampling a canary every N photos would leave up to N
    invented captions in an append-only log before anyone found out.

    But there IS a per-request signal, and it is exact. The metadata block is emitted
    from exactly one place — inside VisionRunner.generate(), only when images are
    present — and the text path never touches it. So:

        request carried an image AND response carries the marker
        <=> the request ran through the vision path, with the pixels.

    And the Filename cell is `image_<sha256(uploaded bytes)[:8]>.jpeg`, which we can
    recompute locally, so we also prove the server hashed the bytes WE sent rather
    than something from an earlier turn or another client.
    """
    if not meta.present:
        raise ServerError(
            "no_image_seen",
            message="response carried no '" + MARKER + "' block, so the request did not "
                    "run through the vision path — the model answered without the pixels")
    cell = meta.filename_cell or ""
    m = re.match(r"^image_([0-9a-f]{8})\.", cell)
    if not m:
        raise ServerError("unparsed_metadata_block",
                          message=f"metadata table present but its Filename cell is {cell!r}")
    if m.group(1) != prepared_server_hash:
        raise ServerError(
            "hash_mismatch",
            message=f"server hashed {m.group(1)} but we uploaded {prepared_server_hash} — "
                    f"the response does not belong to this image")


# ---------------------------------------------------------------------------
# 6. The canary — a comprehension probe, not the primary proof
# ---------------------------------------------------------------------------
CANARY_COLORS = {"red": (220, 0, 0), "green": (0, 150, 0), "blue": (0, 60, 220), "yellow": (245, 220, 0)}
CANARY_PROMPT = (
    "This image contains four coloured squares on a grey background.\n"
    "Name the colour of each square in reading order: top-left, top-right,\n"
    "bottom-left, bottom-right. Answer with exactly four words, lower case,\n"
    "separated by spaces. No other text."
)
_DOWNGRADE_MARKER = re.compile(r"\[\s*\d+\s*image\(s\)[^\]]*attached\s*\]", re.I)
_REFUSAL = re.compile(r"(don'?t|do not|cannot|can'?t|unable to)\s+(see|view|access|find)\s+"
                      r"(any\s+)?(an\s+)?(image|picture|photo)|no image (was )?(provided|attached)", re.I)


def make_canary(order: Sequence[str], size: int = 512) -> bytes:
    """Draw four solid squares in a given order. No font, no asset, no dependency."""
    img = Image.new("RGB", (size, size), (128, 128, 128))
    d = ImageDraw.Draw(img)
    box, pad = size * 200 // 512, size * 28 // 512
    corners = [(pad, pad), (size - pad - box, pad), (pad, size - pad - box),
               (size - pad - box, size - pad - box)]
    for (x, y), name in zip(corners, order):
        d.rectangle([x, y, x + box, y + box], fill=CANARY_COLORS[name])
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def check_canary(text: str, expect: Sequence[str]) -> Tuple[bool, str]:
    """Did the model read four unambiguous colour blocks in the right order?

    The answer is one of 24 equiprobable permutations, redrawn per shot, and cannot be
    inferred from the prompt (which never names a colour) or from any prior — so a model
    that did not receive the pixels guesses with p = 1/24. A model that DID receive them
    is reading four large saturated blocks, a task far below the floor of anything able
    to caption a photograph. A failure therefore means the pixels are not arriving, not
    that the model is weak.

    A run aborts on the FIRST failed shot, so the startup gate's false negative is the
    probability of guessing every shot: with the default two, drawn without replacement,
    1/(24*23) ≈ 1/552.

    Observed while testing this against a text-only model: the answer varies wildly in
    form — sometimes a bare colour list, sometimes "The image does not appear to be
    attached. Here's a solution based on the description: ..." — and two shots may or
    may not coincide. None of that is dependable, which is why the only predicate is the
    ORDER. The colour *set* is inferable from the prompt's phrasing and proves nothing;
    the order exists only in the pixels.

    Two specific diagnoses are checked first, so the operator gets a cause instead of
    a symptom.
    """
    if _DOWNGRADE_MARKER.search(text):
        return False, "model echoed the server's '[N image(s) were attached]' placeholder"
    if _REFUSAL.search(text):
        return False, "model states it received no image"
    got, seen = [], set()
    for m in re.finditer(r"\b(red|green|blue|yellow)\b", text.lower()):
        c = m.group(1)
        if c not in seen:
            seen.add(c)
            got.append(c)
    if got == list(expect):
        return True, ""
    if len(got) < len(expect):
        return False, (f"named only {len(got)} of {len(expect)} colours "
                       f"({', '.join(got) or 'none'})")
    if sorted(got) == sorted(expect):
        # The interesting case, and the usual one for a text-only model: it knows the
        # prompt lists four colours and repeats them, but the ORDER is information that
        # only the pixels carry. One permutation in 24.
        return False, (f"named the right four colours in the wrong order "
                       f"(said {' '.join(got)}, drawn {' '.join(expect)})")
    return False, f"named {' '.join(got)}, drawn {' '.join(expect)}"


def canary_orders(shots: int, rng) -> List[List[str]]:
    """Distinct permutations, drawn WITHOUT replacement.

    Independent draws would collide with p = 1/24 per pair, which would make a
    distinctness assertion fail ~4% of runs for reasons unrelated to the system.
    """
    import itertools
    perms = [list(p) for p in itertools.permutations(sorted(CANARY_COLORS))]
    return rng.sample(perms, k=min(shots, len(perms)))


# ---------------------------------------------------------------------------
# 7. Perceptual identity — "the same picture", as opposed to "the same file"
#
# A camera that writes HEIC often also stores a JPEG export of the same shot, and
# the two need not share a name, a directory or even a plausible ordering: a
# photo-app export names its files by UUID. So the RAW/JPEG name heuristic in walk()
# cannot see them, and neither can photo_id, whose whole job is to be byte-exact.
#
# dHash closes that gap for free. Reduce to 9x8 greyscale, compare each pixel with
# its right-hand neighbour, and the 64 resulting bits describe the *structure* of
# the image — which survives re-encoding and rescaling, i.e. exactly the difference
# between a HEIC and its JPEG export.
#
# It is reported, never acted on. A perceptual hash cannot tell "one shot in two
# formats" from "two frames of a burst", and silently discarding the second would be
# a mistake that costs pictures. Grouping is a fact the operator gets; the decision
# stays theirs.
# ---------------------------------------------------------------------------
DHASH_SIDE = 8


def dhash(im: Image.Image, side: int = DHASH_SIDE) -> int:
    small = im.convert("L").resize((side + 1, side), Image.LANCZOS)
    px = small.load()
    bits = 0
    for y in range(side):
        for x in range(side):
            bits = (bits << 1) | (1 if px[x, y] < px[x + 1, y] else 0)
    return bits


def dhash_hex(value: int, side: int = DHASH_SIDE) -> str:
    return f"{value:0{side * side // 4}x}"


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def group_near_duplicates(items: Sequence[Tuple[str, int]], max_distance: int = 4) -> Dict[str, int]:
    """Map photo_id -> group number for perceptually near-identical photos.

    `items` is (photo_id, dhash). Only ids that share a group appear in the result;
    a photo with no near neighbour is simply absent. O(n^2) on the number of photos,
    which at tens of thousands is a few seconds once, in a step that is already
    derived and rebuildable — cheap enough not to warrant a BK-tree.
    """
    groups: Dict[str, int] = {}
    next_group = 0
    for i, (pid_a, ha) in enumerate(items):
        for pid_b, hb in items[i + 1:]:
            if hamming(ha, hb) > max_distance:
                continue
            ga, gb = groups.get(pid_a), groups.get(pid_b)
            if ga is None and gb is None:
                groups[pid_a] = groups[pid_b] = next_group
                next_group += 1
            elif ga is None:
                groups[pid_a] = gb
            elif gb is None:
                groups[pid_b] = ga
            elif ga != gb:
                for k, v in list(groups.items()):      # merge the two groups
                    if v == gb:
                        groups[k] = ga
    return groups


# ---------------------------------------------------------------------------
# 8. The append-only log, and the lock that guards it
# ---------------------------------------------------------------------------
@dataclass
class ResumeIndex:
    attempts: Dict[str, int] = field(default_factory=dict)
    results: Dict[str, int] = field(default_factory=dict)
    failures: Dict[str, int] = field(default_factory=dict)
    done: set = field(default_factory=set)
    quarantined: set = field(default_factory=set)
    # Where each photo was last seen. Identity is content-only, which is what makes a
    # rename free — but it also means a moved photo is recognised and its new location
    # thrown away unless something records it. Without this the derived catalog keeps
    # serving a path that no longer exists.
    rel: Dict[str, str] = field(default_factory=dict)
    torn: int = 0
    lines: int = 0


class Log:
    """An append-only JSONL file that survives being killed mid-write.

    Three decisions, each answering a specific way this file gets damaged:

    * `ensure_ascii=True` — a torn tail can then only truncate an ASCII line. With
      non-ASCII on the wire, a half-written multi-byte sequence read back under
      errors="replace" could yield a line that still PARSES, and a parseable wrong
      record is far worse than an unparseable one.
    * a missing final newline is repaired before appending — otherwise the next run's
      first line concatenates onto the fragment and two records are lost instead of one.
    * reads use errors="replace" and skip unparseable lines with a warning, so one bad
      line costs exactly one photo, which the next run simply captions again.
    """

    def __init__(self, path: Path, fsync_mode: str = "fsync", redact: Sequence[Path] = ()):
        self.path = Path(path)
        self.fsync_mode = fsync_mode
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._fh = None
        self.repaired_tail = False
        # Roots to keep out of the log. Every record this tool composes itself carries a
        # library-RELATIVE path, but text it did not write does not obey that: a decoder
        # or converter error arrives with the absolute filename in it and is stored
        # verbatim. One boundary here is worth more than remembering to sanitise at each
        # of the places that build an error record.
        #
        # Both spellings of each root, because they genuinely differ: require_env does
        # not resolve, bind_vault stores .resolve(), and on macOS /tmp/x and
        # /private/tmp/x are the same directory under two names. Substituting only one
        # of them would redact nothing in the common case.
        seen, self._redact = set(), []
        for root in redact:
            for form in (str(Path(root).expanduser()), str(Path(root).expanduser().resolve())):
                if len(form) > 1 and form not in seen:
                    seen.add(form)
                    self._redact.append(form)
        self._redact.sort(key=len, reverse=True)   # longest first, so nesting is safe

    def _scrub(self, text: str) -> str:
        for root in self._redact:
            text = text.replace(root, "<root>")
        return text

    def open(self) -> "Log":
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    with open(self.path, "ab") as fix:
                        fix.write(b"\n")
                    self.repaired_tail = True
        self._fh = open(self.path, "a", encoding="utf-8")
        return self

    def append(self, rec: Dict[str, Any]) -> None:
        assert self._fh is not None, "Log.open() first"
        line = json.dumps(rec, ensure_ascii=True)
        if self._redact:
            line = self._scrub(line)
        self._fh.write(line + "\n")
        self._fh.flush()
        if self.fsync_mode == "fsync":
            os.fsync(self._fh.fileno())
        elif self.fsync_mode == "full":
            # os.fsync() hands the data to the device and returns; it does NOT force
            # the device to flush its own write cache. Only F_FULLFSYNC does that on
            # macOS. The default is the cheap barrier that survives a *process* death,
            # which is the failure this job actually has. Nothing here is "durable".
            import fcntl
            fcntl.fcntl(self._fh.fileno(), fcntl.F_FULLFSYNC)

    def close(self) -> None:
        if self._fh:
            try:
                self._fh.close()
            finally:
                self._fh = None

    def __enter__(self) -> "Log":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def read(self, warn=None) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    if warn:
                        warn(f"log line {n} unparseable ({e.msg}); skipping")
                    yield {"__torn__": True}

    def resume_index(self, warn=None) -> ResumeIndex:
        """One streaming pass, producing everything the restart decision needs."""
        idx = ResumeIndex()
        for rec in self.read(warn=warn):
            idx.lines += 1
            if rec.get("__torn__"):
                idx.torn += 1
                continue
            t, pid = rec.get("type"), rec.get("photo_id")
            if not pid:
                continue
            if t == "attempt":
                idx.attempts[pid] = idx.attempts.get(pid, 0) + 1
            elif t == "result":
                idx.results[pid] = idx.results.get(pid, 0) + 1
                if rec.get("ok"):
                    idx.done.add(pid)
                    if rec.get("rel"):
                        idx.rel[pid] = rec["rel"]
                elif not rec.get("environmental"):
                    # The run that saw the failure judged it; obey that. Without this
                    # the replay re-blames the file and the fix lasts one process.
                    idx.failures[pid] = idx.failures.get(pid, 0) + 1
            elif t == "path_update":
                # A photo already described, found somewhere else. Append-only: the old
                # record stays, this one supersedes it, and the streaming pass gives
                # last-wins for free.
                if rec.get("rel"):
                    idx.rel[pid] = rec["rel"]
            elif t == "environment_lost":
                # Closes the attempt without charging the photograph. The attempt line
                # goes down before the work, so without this the gap reads as a process
                # death and three aborted nights quarantine the file.
                idx.results[pid] = idx.results.get(pid, 0) + 1
            elif t == "quarantine":
                idx.quarantined.add(pid)
        return idx


def resume_decision(pid: str, idx: ResumeIndex, max_attempts: int, force: bool = False) -> Optional[str]:
    """None => process it. Otherwise the reason to skip.

    Two counters, because they mean different things:

    * `attempts - results` counts PROCESS DEATHS. The attempt line reached the disk
      and no result ever followed, so something killed the interpreter — a Metal OOM,
      a decoder fault, the OOM killer. Three of those on one photo is a poison pill,
      and it must never be tried a fourth time or a single file stalls a multi-day
      run forever.
    * `failures` counts CLEAN REJECTIONS that were about the photograph — an HTTP 400
      on an oversized image, a file the decoder cannot read. Bounded by the same limit,
      because each one is a fresh observation of the same file.

    What is NOT counted here is anything the environment caused: a vanished mount, a
    permission bit. Those are recorded with `environmental: true` and resume_index
    passes over them, because such a failure is one fact about the machine repeated
    once per photograph, not one fact about each photograph. This docstring used to say
    they were "retried across runs because they may be environmental, but bounded by
    the same limit" — that limit is exactly how 15,613 intact photographs came within
    two nights of permanent quarantine.
    """
    if pid in idx.quarantined:
        return "quarantined"
    if pid in idx.done and not force:
        return "already_done"
    orphaned = idx.attempts.get(pid, 0) - idx.results.get(pid, 0)
    if orphaned >= max_attempts:
        return "poison_pill"
    if idx.failures.get(pid, 0) >= max_attempts:
        return "repeated_error"
    return None


class Lock:
    """O_CREAT|O_EXCL, never broken automatically.

    An honest note about its reach: this is a real mutex on APFS and a hopeful one on
    SMB or NFS, and a pid probe means nothing across hosts. That is one of the reasons
    the catalog is required to live on local storage.
    """

    def __init__(self, path: Path, run_id: str):
        self.path, self.run_id, self.fd = Path(path), run_id, None

    def acquire(self, break_lock: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if break_lock and self.path.exists():
            self.path.unlink()
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            holder = "unknown"
            alive = None
            try:
                info = json.loads(self.path.read_text(encoding="utf-8", errors="replace"))
                holder = info.get("pid", "unknown")
                if isinstance(holder, int):
                    try:
                        os.kill(holder, 0)
                        alive = True
                    except ProcessLookupError:
                        alive = False
                    except PermissionError:
                        alive = True
            except Exception:  # noqa: BLE001 — a damaged lock file is still a held lock
                pass
            state = "running" if alive else ("gone" if alive is False else "state unknown")
            raise Precondition(
                EXIT_LOCKED,
                f"another run holds the lock (pid {holder}, {state})",
                "wait for it to finish, or pass --break-lock if you are certain it is dead. "
                "This is never broken automatically: two writers would silently double "
                "days of work.")
        os.write(self.fd, json.dumps({"pid": os.getpid(), "run_id": self.run_id,
                                      "started": utc_now()}).encode())

    def release(self) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 9. Preconditions
# ---------------------------------------------------------------------------
class Precondition(Exception):
    def __init__(self, code: int, message: str, hint: str = ""):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


def die(err: Precondition) -> "None":
    print(f"Error: {err.message}", file=sys.stderr)
    if err.hint:
        print(f"Hint: {err.hint}", file=sys.stderr)
    sys.exit(err.code)


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


NETWORK_FSTYPES = frozenset({"smbfs", "nfs", "afpfs", "cifs", "webdav", "ftp", "osxfuse", "macfuse"})


def filesystem_of(path: Path) -> Tuple[str, bool, str]:
    """Return (fstype, is_local, mount_point) for the mount that carries `path`.

    The mount point comes back as well because "is this local" and "is this the SAME
    volume as the library" are two different questions with two different answers, and
    only the second one notices a catalog placed beside the library on one share.
    """
    target = str(path.resolve())
    try:
        out = subprocess.run(["/sbin/mount"], capture_output=True, text=True, timeout=15).stdout
    except Exception:  # noqa: BLE001 — no mount(8) means we cannot tell; say so
        return "unknown", True, ""
    best, best_len, best_mount = None, -1, ""
    for line in out.splitlines():
        m = re.match(r"^(.*?) on (.*?) \((.*?)\)\s*$", line)
        if not m:
            continue
        mount_point, opts = m.group(2), m.group(3)
        if (target == mount_point or target.startswith(mount_point.rstrip("/") + "/")) \
                and len(mount_point) > best_len:
            best, best_len, best_mount = opts, len(mount_point), mount_point
    if best is None:
        return "unknown", True, ""
    fields = [o.strip() for o in best.split(",")]
    fstype = fields[0] if fields else "unknown"
    return fstype, fstype not in NETWORK_FSTYPES, best_mount


def require_env(name: str, must_exist: bool = False) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise Precondition(
            EXIT_USAGE, f"{name} is not set",
            f"export {name}=/path/to/... — there is no default on purpose. Inputs and "
            f"outputs both live outside this repository, so nothing private is ever in "
            f"the tree to begin with (see the Conventions section of examples/README.md).")
    p = Path(raw).expanduser()
    if must_exist and not p.is_dir():
        # An unquoted `export VAULT=/Vols/My Photos` is split by the shell, so what
        # arrives is "/Vols/My" — no space to look for. The giveaway is the neighbour.
        hint = f"check {name} — it is currently {raw!r}"
        try:
            near = sorted(x.name for x in p.parent.iterdir()
                          if x.is_dir() and x.name.startswith(p.name + " "))
        except OSError:
            near = []
        if near:
            hint = (f'{name} is {raw!r}, but {p.parent} contains "{near[0]}".\n'
                    f'      A path with spaces must be quoted where it is SET: '
                    f'export {name}="{p.parent}/{near[0]}"')
        raise Precondition(EXIT_USAGE, f"{name} is not a readable directory", hint)
    return p


def require_separate_catalog(catalog: Path, vault: Path) -> None:
    """The output tree must not live inside the library. Checked before anything is created.

    "Your collection is never written to" is a guarantee this example makes in MANUAL.md, and
    a guarantee nothing enforces is only a claim. Without this check a catalog placed inside
    the vault causes three separate harms, in rising order of how much they cost:

    1. `vault.json`, `log/`, `tmp/` and `prepared/` are written into the library.
    2. The cached uploads are `.jpg`, which is a supported input format, so the next walk
       discovers them as new photographs — each run adding more, forever.
    3. Worst: the batch sweeps `<catalog>/tmp/` unconditionally when it takes the lock,
       because those are its own EXIF-bearing intermediates. If that path resolves onto a
       real `tmp/` album, the sweep deletes photographs. This is the reason the check exists
       at all, and the reason it runs before the first mkdir rather than after.

    Sharing a *parent* is fine — a sibling directory on the same disk is the normal setup.
    What matters is only that the output root is not somewhere the walker will descend into.
    Both paths are resolved first, so a symlinked parent cannot slip past.
    """
    v, c = vault.resolve(), catalog.expanduser().resolve()
    if c == v or v in c.parents:
        raise Precondition(
            EXIT_PRECONDITION,
            "PHOTO_CATALOG is inside PHOTO_VAULT",
            "point PHOTO_CATALOG somewhere outside your photo library — a sibling directory "
            "is fine. Writing the catalog into the library would break the one guarantee this "
            "example makes about it, the prepared uploads would be re-discovered as new "
            "photographs on the next run, and the batch sweeps its own tmp/ directory on "
            "startup, which would delete anything already there. "
            "(Neither path is printed here: they are the private part.)")


def bind_vault(catalog: Path, vault: Path, rebind: bool = False) -> None:
    """The vault root is written down exactly once, here.

    Every record stores a vault-RELATIVE path, so the absolute root never reaches a
    catalog line, an index line, an embedding, or a search result someone might paste
    into an issue.
    """
    marker = catalog / "vault.json"
    resolved = str(vault.resolve())
    if marker.exists() and not rebind:
        try:
            bound = json.loads(marker.read_text(encoding="utf-8", errors="replace")).get("vault_root")
        except Exception:  # noqa: BLE001
            bound = None
        if bound and bound != resolved:
            raise Precondition(
                EXIT_VAULT_MISMATCH,
                "this catalog is bound to a different vault root",
                "point PHOTO_VAULT back at the library this catalog was built from, use a "
                "different PHOTO_CATALOG, or pass --rebind-vault if you moved the library. "
                "(Neither path is printed here: they are the private part.)")
        if bound == resolved:
            return
    catalog.mkdir(parents=True, exist_ok=True, mode=0o700)
    marker.write_text(json.dumps({"schema": SCHEMA, "vault_root": resolved,
                                  "bound": utc_now()}, ensure_ascii=True) + "\n", encoding="utf-8")
    os.chmod(marker, 0o600)


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "[::1]"})


def check_base_url(base_url: str, allow_remote: bool) -> bool:
    """Refuse to ship a private library over an unauthenticated plain-HTTP hop.

    mlx-knife's server enforces no authentication and mounts CORS wide open. On
    loopback that is fine and deliberate. Pointed at another host it means every
    photograph crosses the network in the clear to something that authenticates
    nobody — and the portability story of this example ("swap the base URL for a
    multi-node backend") makes that a one-flag mistake. Returns True when the
    override is in force, so the caller can stamp it into the run record.
    """
    from urllib.parse import urlparse
    u = urlparse(base_url)
    host = (u.hostname or "").lower()
    if host in LOOPBACK_HOSTS or u.scheme == "https":
        return False
    if allow_remote:
        return True
    raise Precondition(
        EXIT_PRECONDITION,
        f"refusing a non-loopback plain-HTTP base URL (host {host!r})",
        "mlx-knife's server is unauthenticated and CORS-open, so this would send your "
        "photographs in the clear to a host that authenticates nobody. Use https, keep it "
        "on loopback, or pass --allow-remote-base-url if you understand the exposure.")


def server_health(client, base_url: str, timeout: float = 5.0) -> None:
    root = base_url.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    try:
        r = client.get(f"{root}/health", timeout=timeout)
    except Exception as e:  # noqa: BLE001
        raise Precondition(
            EXIT_PRECONDITION, f"cannot reach the server at {base_url} ({type(e).__name__})",
            "start it yourself, in another shell:  mlxk serve --model <vision-model> --port 8000\n"
            "      These scripts never start or stop a server: one they started would die with "
            "them and take a multi-day run with it.") from e
    if r.status_code != 200:
        raise Precondition(EXIT_PRECONDITION, f"server /health returned {r.status_code}",
                           "the server is up but not healthy; check its log")


def list_models(client, base_url: str, timeout: float = 10.0) -> List[str]:
    try:
        r = client.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        return [m.get("id", "") for m in (r.json().get("data") or [])]
    except Exception:  # noqa: BLE001 — advisory only, see the caller
        return []


def inherit_metadata(
    records: Dict[str, Dict[str, Any]], groups: Dict[str, int]
) -> Dict[str, Dict[str, Any]]:
    """Fill missing capture metadata from a perceptually matched partner.

    The case this exists for: a raw file that carries no date, no location and no
    camera, next to a derivative of the same shot that does. Name and directory are
    no help there — a photo-app export names by UUID, and the two files need not even
    live in the same folder — so the only thing that can pair them is the picture.

    What it does NOT do is merge. `exif` keeps meaning "read from this file"; anything
    borrowed lands in `exif_inherited` with the partner's photo_id attached. Three
    months later it must still be answerable whether a date was measured or inferred,
    and a single merged field cannot answer that.

    Note the honest limit: a raw and its camera-developed JPEG usually differ slightly
    in framing and tone, so their distance is larger than a HEIC/JPEG pair's. The
    threshold belongs to the caller, measured — see the calibration row in the
    self-check rather than a number guessed here.
    """
    by_group: Dict[int, List[str]] = {}
    for pid, g in groups.items():
        by_group.setdefault(g, []).append(pid)

    out: Dict[str, Dict[str, Any]] = {}
    for g, members in by_group.items():
        for pid in members:
            rec = records.get(pid)
            if rec is None:
                continue
            missing = [ax for ax in ("gps", "captured", "camera") if not rec.get(ax)]
            if not missing:
                continue
            inherited: Dict[str, Any] = {}
            for ax in missing:
                for other in members:
                    if other == pid:
                        continue
                    donor = records.get(other) or {}
                    if donor.get(ax):
                        inherited[ax] = donor[ax]
                        inherited.setdefault("from", {})[ax] = other
                        break
            if inherited:
                out[pid] = inherited
    return out


def prepare_cached(
    src: Path, branch: str, cache_root: Optional[Path], *,
    max_edge: int = 512, quality: int = 85, max_pixels: int = 400_000_000,
    tmpdir: Optional[Path] = None, photo_id_: str = "",
) -> Tuple["Prepared", bool]:
    """prepare(), but keep the result so the expensive half never runs twice.

    Returns (prepared, from_cache).

    Three reasons the bytes are worth keeping, none of them "to go faster":

    1. **A second pass costs no conversion.** Trying another prompt, or a higher edge
       for the photos where legible text decides the match, otherwise re-runs sips over
       every HEIC and raw file in the library — hours of pure repetition.
    2. **It is the artefact an image embedder will need.** A vector must be computed
       from the *same normalised pixels* the caption came from, or the two describe
       different inputs. Keeping them is what makes that stage additive rather than a
       full re-run.
    3. **"What did the model actually see?" stays answerable**, byte for byte, against
       the sha256 in the record.

    The edge is part of the filename, so a 512 px and a 1024 px pass over the same
    photo coexist instead of overwriting each other — which is exactly what the
    documented second-pass escalation needs. The EXIF sidecar is what makes the cache
    hit complete: the uploaded JPEG carries no EXIF by construction, so without it a
    "cached" run would still have to convert the original just to read a date.
    """
    if cache_root is None or not photo_id_:
        return prepare(src, branch, max_edge=max_edge, quality=quality,
                       max_pixels=max_pixels, tmpdir=tmpdir), False

    d = Path(cache_root) / photo_id_[:2]
    # Every setting that changes the BYTES belongs in the key. max_edge was there from
    # the start; quality was not, so lowering it and re-running served the old file and
    # the model saw pixels the run did not ask for. Old entries are orphaned rather than
    # mis-served, which is the right way round.
    stem = f"{photo_id_}-{max_edge}q{quality}"
    jpg, side = d / f"{stem}.jpg", d / f"{stem}.json"
    if jpg.exists() and side.exists():
        try:
            data = jpg.read_bytes()
            meta = json.loads(side.read_text(encoding="utf-8", errors="replace"))
            e = meta.get("exif") or {}
            gps = e.get("gps")
            return Prepared(
                data=data, w=meta["w"], h=meta["h"], sha256=sha256_hex(data),
                server_hash=server_image_hash(data), max_edge=max_edge,
                src_w=meta.get("src_w"), src_h=meta.get("src_h"),
                converter=meta.get("converter"),
                exif=Exif(gps=tuple(gps) if gps else None, dt=e.get("dt"),
                          camera=e.get("camera"), orientation=e.get("orientation")),
            ), True
        except (OSError, KeyError, ValueError, TypeError):
            pass  # a damaged cache entry is not an error; just redo the work

    prep = prepare(src, branch, max_edge=max_edge, quality=quality,
                   max_pixels=max_pixels, tmpdir=tmpdir)
    try:
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = jpg.with_suffix(".jpg.tmp")
        tmp.write_bytes(prep.data)
        os.replace(tmp, jpg)
        side.write_text(json.dumps({
            "w": prep.w, "h": prep.h, "src_w": prep.src_w, "src_h": prep.src_h,
            "converter": prep.converter, "sha256": prep.sha256,
            "exif": prep.exif.as_dict()}, ensure_ascii=True), encoding="utf-8")
    except OSError:
        pass  # the cache is an optimisation; failing to write it must not fail a photo
    return prep, False
