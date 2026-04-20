"""Workspace sentinel management (ADR-018 Phase 0a).

This module provides primitives for managed workspace detection and metadata tracking.

Managed workspaces contain a `.mlxk_workspace.json` sentinel file that enables:
- Workspace lifecycle tracking (clone, convert, push)
- Source provenance (which HF repo was cloned)
- Operation history (what transformations were applied)
- Safety guarantees (e.g., cache sanctity enforcement in convert)

Sentinel format:
{
  "mlxk_version": "<version>",  // e.g., "2.0.4b6"
  "created_at": "2025-12-29T10:30:00Z",
  "source_repo": "mlx-community/Llama-3.2-3B",
  "source_revision": "abc123def456",
  "managed": true,
  "operation": "clone"  // or "convert"
}

ADR-018 Contract:
- Clone and convert MUST produce managed workspaces
- Sentinel is written FIRST (atomic, before other processing)
- Health checks support both managed and unmanaged workspaces
- Unmanaged workspaces can be converted to managed via convert operation
"""

import fnmatch
import hashlib
import json
import logging
import os
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

SENTINEL_FILENAME = ".mlxk_workspace.json"

# ADR-025: content_hash v2 — algorithm constants.
HASH_ALGORITHM_V2 = "v2"
# Non-safetensors files larger than this fall back to stat-only hashing
# (sha256(name || 0x00 || size)) to keep `mlxk ls` fast. Changing this value
# requires a hash_algorithm bump per ADR-025 §9.
CATCHALL_FULL_READ_CAP = 1024 * 1024 * 1024  # 1 GiB
# Hard cap on the safetensors JSON-header length prefix. A prefix above this
# is treated as corrupt/malformed per ADR-025 §1.
SAFETENSORS_HEADER_MAX = 100 * 1024 * 1024  # 100 MiB

# Default exclude list (ADR-025 §2). Seeds first-compute; readers consult the
# list frozen into the sentinel, not this default. Pattern semantics:
# - trailing "/" marks a directory name (pruned from walk)
# - everything else is fnmatch-style against a file basename
DEFAULT_EXCLUDE_PATTERNS: List[str] = [
    # mlxk runtime
    ".hf_cache/", ".mlxk_workspace.json",
    # Dev / VCS tooling
    ".git/", ".github/", "__pycache__/",
    # IDE / dev-tool cruft
    ".vscode/", ".idea/", ".ipynb_checkpoints/",
    ".mypy_cache/", ".pytest_cache/", ".ruff_cache/", ".tox/",
    "node_modules/", ".venv/", "venv/",
    # Generic temp / logs
    "*.lock", "*.tmp", "*.log", "*.bak",
    # macOS file-manager cruft
    ".DS_Store", "._*",
    ".Spotlight-V100/", ".Trashes/", ".fseventsd/",
    ".TemporaryItems/", ".apdisk", "Icon\r",
    # Windows file-manager cruft
    "Thumbs.db", "desktop.ini",
    "$RECYCLE.BIN/", "System Volume Information/",
    # Editor backups
    "*~", ".*.swp", ".*.swo",
]


class CorruptSentinelError(ValueError):
    """Raised when a workspace sentinel contains invalid/tampered data.

    Callers in the clean-check path MUST catch this before any filesystem
    operation driven by sentinel contents (ADR-025 §3 path constraints).
    """


def get_workspace_home() -> Optional[Path]:
    """Get workspace home directory from MLXK_WORKSPACE_HOME environment variable.

    ADR-022: Workspace-first paradigm. When set, this directory is searched
    before the HuggingFace cache for model resolution.

    Returns:
        Path to workspace home directory if MLXK_WORKSPACE_HOME is set and exists,
        None otherwise.

    Example:
        >>> os.environ["MLXK_WORKSPACE_HOME"] = "/path/to/workspaces"
        >>> get_workspace_home()
        PosixPath('/path/to/workspaces')
    """
    workspace_home = os.environ.get("MLXK_WORKSPACE_HOME")
    if not workspace_home:
        return None

    path = Path(workspace_home).expanduser()
    if not path.exists():
        logger.debug(f"MLXK_WORKSPACE_HOME set but path doesn't exist: {path}")
        return None

    if not path.is_dir():
        logger.warning(f"MLXK_WORKSPACE_HOME is not a directory: {path}")
        return None

    return path.resolve()


def write_workspace_sentinel(workspace_path: Path, metadata: Dict[str, Any]) -> None:
    """Write workspace sentinel with atomic write+rename.

    Sentinel is written atomically to prevent partial writes during crashes.
    Uses tmp file + rename pattern (POSIX atomic on same filesystem).

    Args:
        workspace_path: Root directory of workspace
        metadata: Dictionary with sentinel fields:
            - mlxk_version (str): mlxk version that created workspace
            - created_at (str): ISO8601 timestamp
            - source_repo (str): Source HF repo (e.g., "mlx-community/Llama-3.2-3B")
            - source_revision (str|None): Git revision hash
            - managed (bool): Always True for mlxk-created workspaces
            - operation (str): Creating operation ("clone", "convert")
            Additional fields allowed (forward compatibility)

    Raises:
        OSError: If sentinel write fails
        TypeError: If workspace_path is not a Path object
        ValueError: If metadata is missing required fields
    """
    if not isinstance(workspace_path, Path):
        raise TypeError(f"workspace_path must be Path, got {type(workspace_path)}")

    # Validate required fields
    required_fields = {"mlxk_version", "created_at", "managed", "operation"}
    missing = required_fields - set(metadata.keys())
    if missing:
        raise ValueError(f"Missing required metadata fields: {missing}")

    workspace_path = workspace_path.resolve()
    sentinel_path = workspace_path / SENTINEL_FILENAME
    tmp_path = workspace_path / f"{SENTINEL_FILENAME}.tmp"

    try:
        # Atomic write: tmp file + rename
        tmp_path.write_text(json.dumps(metadata, indent=2) + "\n")
        tmp_path.rename(sentinel_path)

        logger.debug(f"Wrote workspace sentinel to {sentinel_path}")
    except Exception as e:
        # Cleanup tmp file if rename failed
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass  # Best effort cleanup
        raise OSError(f"Failed to write workspace sentinel: {e}") from e


def is_managed_workspace(workspace_path: Path) -> bool:
    """Check if workspace has valid sentinel.

    Args:
        workspace_path: Path to workspace directory

    Returns:
        True if workspace has valid .mlxk_workspace.json with managed=True,
        False otherwise (missing sentinel, invalid JSON, managed=False)
    """
    if not isinstance(workspace_path, Path):
        return False

    sentinel = workspace_path.resolve() / SENTINEL_FILENAME

    if not sentinel.exists():
        return False

    try:
        data = json.loads(sentinel.read_text())
        return data.get("managed", False) is True
    except (json.JSONDecodeError, OSError) as e:
        logger.debug(f"Invalid sentinel in {workspace_path}: {e}")
        return False


def read_workspace_metadata(workspace_path: Path) -> Dict[str, Any]:
    """Read workspace sentinel metadata.

    Args:
        workspace_path: Path to workspace directory

    Returns:
        Dictionary with sentinel metadata, or empty dict if:
        - Sentinel doesn't exist
        - JSON is invalid
        - Read fails

    Note: This function does NOT validate metadata fields.
    Use is_managed_workspace() to check if workspace is managed.
    """
    if not isinstance(workspace_path, Path):
        return {}

    sentinel = workspace_path.resolve() / SENTINEL_FILENAME

    if not sentinel.exists():
        return {}

    try:
        return json.loads(sentinel.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.debug(f"Failed to read sentinel in {workspace_path}: {e}")
        return {}


def is_explicit_path(pattern: str) -> bool:
    """Check if pattern is an explicit filesystem path (not an HF model ID).

    Only paths with explicit path markers are treated as filesystem paths.
    This ensures "model-name" goes through cache resolution even if a local dir exists.

    Args:
        pattern: The pattern string to check

    Returns:
        True if pattern is an explicit path, False otherwise

    Examples:
        >>> is_explicit_path("./gemma-3n")
        True
        >>> is_explicit_path("../parent/model")
        True
        >>> is_explicit_path("/abs/path/model")
        True
        >>> is_explicit_path(".")
        True
        >>> is_explicit_path("mlx-community/Phi-3")
        False  # HF model ID
        >>> is_explicit_path("my-model")
        False  # Ambiguous, treated as HF ID
    """
    if not pattern or not isinstance(pattern, str):
        return False
    return (
        pattern.startswith(('./', '../', '/')) or
        pattern in ('.', '..')
    )


def is_workspace_path(path) -> bool:
    """Check if path points to a workspace directory (managed or unmanaged).

    A workspace is any directory containing a config.json file (MLX model structure).
    This includes both managed workspaces (with .mlxk_workspace.json) and
    unmanaged workspaces (3rd-party model directories).

    Args:
        path: Path-like object (str, Path) to check

    Returns:
        True if path exists and contains config.json, False otherwise

    Examples:
        >>> is_workspace_path("./my-workspace")
        True
        >>> is_workspace_path("/path/to/model")
        True
        >>> is_workspace_path("mlx-community/Phi-3-mini")
        False  # HF model ID, not a path
    """
    try:
        p = Path(path)
        return p.exists() and (p / "config.json").exists()
    except (TypeError, OSError):
        return False


def find_matching_workspaces(pattern: str) -> list:
    """Find all workspace directories matching an explicit path pattern.

    Supports three modes:
    1. Exact match: Pattern points to existing workspace directory
    2. Directory scan: Pattern is existing directory (not workspace) → find all workspaces inside
    3. Prefix match: Pattern is partial path → find directories starting with prefix

    Args:
        pattern: Explicit path pattern (e.g., "./gemma-" or "/path/to/model" or ".")
                 Must start with ./, ../, / or be . or ..

    Returns:
        List of Path objects for matching workspaces (directories with config.json).
        Empty list if pattern is not an explicit path or no matches found.

    Examples:
        >>> find_matching_workspaces("./gemma-3n-E2B-it-4bit")
        [PosixPath('/path/to/gemma-3n-E2B-it-4bit')]  # Exact match

        >>> find_matching_workspaces(".")
        [PosixPath('/path/to/model1'), PosixPath('/path/to/model2')]  # Directory scan

        >>> find_matching_workspaces("./gemma-")
        [PosixPath('/path/to/gemma-3n-E2B-it-4bit'),
         PosixPath('/path/to/gemma-3n-E2B-it-FIXED-4bit')]  # Prefix match

        >>> find_matching_workspaces("mlx-community/Phi-3")
        []  # Not an explicit path
    """
    if not is_explicit_path(pattern):
        return []

    try:
        p = Path(pattern).expanduser()

        # Case 1: Exact match - pattern is already a complete workspace
        if is_workspace_path(p):
            return [p.resolve()]

        # Case 2: Directory scan - pattern is existing directory (not a workspace)
        # Find all workspaces inside this directory
        if p.exists() and p.is_dir():
            matches = []
            for entry in p.iterdir():
                if entry.is_dir() and (entry / "config.json").exists():
                    matches.append(entry.resolve())
            matches.sort(key=lambda x: x.name)
            return matches

        # Case 3: Prefix match - find directories starting with pattern
        parent = p.parent
        prefix = p.name

        if not parent.exists() or not parent.is_dir():
            return []

        # Find all directories in parent that start with prefix
        matches = []
        for entry in parent.iterdir():
            if entry.is_dir() and entry.name.startswith(prefix):
                # Only include if it's a valid workspace (has config.json)
                if (entry / "config.json").exists():
                    matches.append(entry.resolve())

        # Sort by name for consistent output
        matches.sort(key=lambda p: p.name)
        return matches

    except (TypeError, OSError) as e:
        logger.debug(f"Error finding workspaces for pattern '{pattern}': {e}")
        return []


# ---------------------------------------------------------------------------
# ADR-025 content_hash v2 — algorithm implementation
# ---------------------------------------------------------------------------


def _is_dir_excluded(dirname: str, exclude_patterns: List[str]) -> bool:
    for pat in exclude_patterns:
        if pat.endswith("/") and fnmatch.fnmatch(dirname, pat[:-1]):
            return True
    return False


def _is_file_excluded(filename: str, exclude_patterns: List[str]) -> bool:
    for pat in exclude_patterns:
        if pat.endswith("/"):
            continue
        if fnmatch.fnmatch(filename, pat):
            return True
    return False


def _parse_safetensors_header(path: Path) -> Optional[bytes]:
    """Read safetensors JSON-metadata header bytes.

    Returns header bytes (without the 8-byte length prefix) or None for a
    corrupt prefix: zero length, length > SAFETENSORS_HEADER_MAX, short read,
    or I/O failure. Tensor data is never touched.
    """
    try:
        with open(path, "rb") as f:
            prefix = f.read(8)
            if len(prefix) < 8:
                return None
            length = int.from_bytes(prefix, byteorder="little", signed=False)
            if length == 0 or length > SAFETENSORS_HEADER_MAX:
                return None
            header = f.read(length)
            if len(header) < length:
                return None
            return header
    except OSError:
        return None


def _validate_sentinel_path(path: str) -> None:
    """Reject unsafe file_index paths before any filesystem op (ADR-025 §3).

    Raises CorruptSentinelError on: non-str, empty, NUL byte, backslash,
    absolute path, drive prefix (`C:...`), or any `..` component.
    """
    if not isinstance(path, str):
        raise CorruptSentinelError(f"path must be str, got {type(path).__name__}")
    if not path:
        raise CorruptSentinelError("empty path")
    if "\x00" in path:
        raise CorruptSentinelError(f"NUL byte in path: {path!r}")
    if "\\" in path:
        raise CorruptSentinelError(f"non-POSIX separator in path: {path!r}")
    if path.startswith("/"):
        raise CorruptSentinelError(f"absolute path: {path!r}")
    if len(path) >= 2 and path[1] == ":":
        raise CorruptSentinelError(f"drive prefix in path: {path!r}")
    if ".." in path.split("/"):
        raise CorruptSentinelError(f"'..' component in path: {path!r}")


def _classify_symlink(link_path: Path, workspace_root: Path) -> Tuple[str, str]:
    """Classify a symlink per ADR-025 §1.

    Returns ("inside", sha_str) with the path fingerprint when the target is
    inside the workspace root. Returns ("outside", reason) for any target
    that escapes the workspace root or is broken.
    """
    try:
        target = link_path.resolve(strict=False)
    except OSError as e:
        return ("outside", f"resolve failed for {link_path}: {e}")
    try:
        target_rel = target.relative_to(workspace_root).as_posix()
    except ValueError:
        return ("outside", f"{link_path} -> {target}")
    fingerprint = hashlib.sha256(b"SYMLINK\x00" + target_rel.encode("utf-8")).hexdigest()
    return ("inside", f"sha256:{fingerprint}")


def _hash_file_by_class(path: Path) -> Optional[str]:
    """Return "sha256:<hex>" for a file per ADR-025 §1 per-class strategy.

    Returns None only for corrupt safetensors header; corrupt catch-all files
    never degrade to None (falls through to stat-only).
    """
    name = path.name
    if name.endswith(".safetensors"):
        header = _parse_safetensors_header(path)
        if header is None:
            return None
        return "sha256:" + hashlib.sha256(header).hexdigest()
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > CATCHALL_FULL_READ_CAP:
        h = hashlib.sha256()
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(size).encode("ascii"))
        return "sha256:" + h.hexdigest()
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return "sha256:" + h.hexdigest()


def _walk_workspace(workspace_root: Path, exclude_patterns: List[str]):
    """Yield (rel_posix, kind, data) tuples for each non-excluded entry.

    kind values:
    - "file"    — regular file; data is the absolute Path
    - "symlink" — symlink; data is ("inside", sha) or ("outside", reason)

    Walk is depth-first; iteration order is byte-sorted by NFC-normalized
    name within each directory. Symlinks are never followed.
    """
    def _rel(p: Path) -> str:
        return "/".join(
            unicodedata.normalize("NFC", part)
            for part in p.relative_to(workspace_root).parts
        )

    stack: List[Path] = [workspace_root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                entries = sorted(it, key=lambda e: unicodedata.normalize("NFC", e.name))
        except OSError as e:
            logger.debug(f"scandir failed for {current}: {e}")
            continue
        for entry in entries:
            name = unicodedata.normalize("NFC", entry.name)
            path = Path(entry.path)
            # Symlinks first — is_dir/is_file would follow the link otherwise.
            try:
                is_link = entry.is_symlink()
            except OSError:
                is_link = False
            if is_link:
                # Symlink classification is exclude-pattern-independent:
                # the policy (inside fingerprint / outside reject) is a
                # security property, not a cruft filter.
                yield (_rel(path), "symlink", _classify_symlink(path, workspace_root))
                continue
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                is_dir = False
            if is_dir:
                if _is_dir_excluded(name, exclude_patterns):
                    continue
                stack.append(path)
                continue
            try:
                is_file = entry.is_file(follow_symlinks=False)
            except OSError:
                is_file = False
            if is_file:
                if _is_file_excluded(name, exclude_patterns):
                    continue
                yield (_rel(path), "file", path)


def _aggregate_hash(file_index: List[Dict[str, Any]]) -> str:
    """Compute the aggregate content_hash from a file_index (ADR-025 §3).

    Formula: sha256( concat( sorted( path || 0x00 || sha ) ) ).
    """
    ordered = sorted(file_index, key=lambda e: e["path"])
    h = hashlib.sha256()
    for entry in ordered:
        h.update(entry["path"].encode("utf-8"))
        h.update(b"\x00")
        h.update(entry["sha"].encode("ascii"))
    return "sha256:" + h.hexdigest()


def compute_workspace_hash_v2(
    workspace_path: Path,
    exclude_patterns: Optional[List[str]] = None,
) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """First-compute path for content_hash v2 (ADR-025 §7).

    Walks the workspace under `exclude_patterns` (defaults to
    DEFAULT_EXCLUDE_PATTERNS), hashes each surviving file per its class
    strategy, and aggregates per the §3 formula. Returns (content_hash,
    file_index) on success, or None if:
    - workspace_path is not a valid workspace (missing config.json)
    - a safetensors file has a corrupt/oversized header
    - a symlink escapes the workspace root
    - an unexpected I/O failure aborts the walk
    """
    if not isinstance(workspace_path, Path):
        return None
    workspace_root = workspace_path.resolve()
    if not (workspace_root / "config.json").exists():
        logger.debug(f"No config.json in {workspace_root}")
        return None
    effective_excludes: List[str] = (
        list(DEFAULT_EXCLUDE_PATTERNS) if exclude_patterns is None else list(exclude_patterns)
    )

    file_index: List[Dict[str, Any]] = []
    for rel, kind, data in _walk_workspace(workspace_root, effective_excludes):
        if kind == "file":
            path = data
            sha = _hash_file_by_class(path)
            if sha is None:
                logger.warning(
                    f"Corrupt safetensors header in {path}; aborting v2 hash"
                )
                return None
            try:
                st = path.stat()
            except OSError as e:
                logger.debug(f"stat failed for {path}: {e}")
                return None
            file_index.append({
                "path": rel,
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
                "sha": sha,
            })
        elif kind == "symlink":
            outcome_kind, outcome_data = data
            if outcome_kind == "outside":
                logger.warning(
                    f"Symlink outside workspace root refused: {outcome_data}"
                )
                return None
            file_index.append({
                "path": rel,
                "size": 0,
                "mtime_ns": 0,
                "sha": outcome_data,
            })

    file_index.sort(key=lambda e: e["path"])
    return (_aggregate_hash(file_index), file_index)


def _stat_check_with_self_heal(
    workspace_root: Path,
    file_index: List[Dict[str, Any]],
    exclude_patterns: List[str],
) -> Tuple[bool, bool, List[Dict[str, Any]]]:
    """Fast stat-walk against a stored file_index (ADR-025 §6).

    Returns (is_clean, healed_any, updated_index):
    - is_clean: True iff every on-disk file matched its index entry
      (optionally via self-heal) and no paths were added or removed
    - healed_any: True iff at least one entry had its mtime updated
      after a content-match self-heal
    - updated_index: new sorted file_index (healed mtimes applied)
    """
    indexed = {e["path"]: dict(e) for e in file_index}
    seen: set = set()
    dirty = False
    healed_any = False

    for rel, kind, data in _walk_workspace(workspace_root, exclude_patterns):
        seen.add(rel)
        entry = indexed.get(rel)
        if entry is None:
            dirty = True
            continue
        if kind == "symlink":
            outcome_kind, outcome_data = data
            if outcome_kind == "outside":
                dirty = True
                continue
            if outcome_data != entry.get("sha"):
                dirty = True
            continue
        if kind != "file":
            continue
        path: Path = data
        try:
            _validate_sentinel_path(entry["path"])
        except CorruptSentinelError:
            dirty = True
            continue
        try:
            st = path.stat()
        except OSError:
            dirty = True
            continue
        if st.st_size != entry.get("size"):
            dirty = True
            continue
        if st.st_mtime_ns == entry.get("mtime_ns"):
            continue
        new_sha = _hash_file_by_class(path)
        if new_sha is None:
            dirty = True
            continue
        if new_sha == entry.get("sha"):
            entry["mtime_ns"] = st.st_mtime_ns
            healed_any = True
        else:
            dirty = True

    missing = set(indexed.keys()) - seen
    if missing:
        dirty = True

    updated = sorted(indexed.values(), key=lambda e: e["path"])
    return (not dirty, healed_any, updated)


def update_workspace_hash(workspace_path: Path) -> Tuple[bool, Optional[str]]:
    """Recompute v2 content_hash + file_index, persist to sentinel.

    ADR-025 §7 first-compute flow. When the sentinel already has v2
    `exclude_patterns`, those govern (preserves the per-workspace frozen
    list); otherwise the code-default list seeds the compute.

    Returns (success, content_hash). On failure the sentinel is left
    untouched.
    """
    if not isinstance(workspace_path, Path):
        return (False, None)
    workspace_path = workspace_path.resolve()

    if not is_managed_workspace(workspace_path):
        logger.warning(f"Cannot update hash: not a managed workspace: {workspace_path}")
        return (False, None)

    metadata = read_workspace_metadata(workspace_path)
    if not metadata:
        logger.warning(f"Cannot read metadata for {workspace_path}")
        return (False, None)

    stored_excludes = metadata.get("exclude_patterns")
    effective_excludes: List[str] = (
        list(stored_excludes) if isinstance(stored_excludes, list)
        else list(DEFAULT_EXCLUDE_PATTERNS)
    )

    result = compute_workspace_hash_v2(workspace_path, effective_excludes)
    if result is None:
        logger.warning(f"Cannot compute v2 hash for {workspace_path}")
        return (False, None)
    content_hash, file_index = result

    hash_modified = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    metadata["hash_algorithm"] = HASH_ALGORITHM_V2
    metadata["content_hash"] = content_hash
    metadata["hash_modified"] = hash_modified
    metadata["exclude_patterns"] = effective_excludes
    metadata["file_index"] = file_index

    try:
        write_workspace_sentinel(workspace_path, metadata)
        logger.info(f"Updated content hash (v2) for {workspace_path}")
        return (True, content_hash)
    except Exception as e:
        logger.warning(f"Failed to update sentinel: {e}")
        return (False, None)


def is_workspace_clean(workspace_path: Path) -> Tuple[Optional[bool], Optional[str], Optional[str]]:
    """Clean-check against v2 sentinel (ADR-025 §6).

    Hot path for `mlxk ls` / `mlxk show`. Steps:
    1. Read sentinel; if `hash_algorithm != "v2"` → `(None, None, stored_hash)`.
    2. Validate every `file_index` path before any FS op (§3); corruption → `(None, ...)`.
    3. Stat-walk; self-heal mtime-only drift (§6 step 6); persist healed mtimes silently.
    4. Dirty path recomputes the new aggregate so callers see the drift hash.

    Returns (is_clean, current_hash, stored_hash) where:
    - is_clean: True / False / None (unknown — migration needed or corrupt)
    - current_hash: aggregate hash on disk (None on unknown)
    - stored_hash: sentinel's `content_hash` field (may be None for pre-v1 workspaces)
    """
    if not isinstance(workspace_path, Path):
        return (None, None, None)
    workspace_root = workspace_path.resolve()

    metadata = read_workspace_metadata(workspace_root)
    stored_hash = metadata.get("content_hash")
    hash_algo = metadata.get("hash_algorithm")

    if hash_algo != HASH_ALGORITHM_V2:
        return (None, None, stored_hash)

    file_index = metadata.get("file_index")
    if not isinstance(file_index, list):
        return (None, None, stored_hash)

    # §3 path validation — before any filesystem operation driven by sentinel paths.
    try:
        for entry in file_index:
            if not isinstance(entry, dict):
                raise CorruptSentinelError("file_index entry is not a dict")
            _validate_sentinel_path(entry.get("path"))
    except CorruptSentinelError as e:
        logger.warning(f"Corrupt sentinel in {workspace_root}: {e}")
        return (None, None, stored_hash)

    exclude_patterns = metadata.get("exclude_patterns")
    if not isinstance(exclude_patterns, list):
        exclude_patterns = list(DEFAULT_EXCLUDE_PATTERNS)

    is_clean, healed_any, healed_index = _stat_check_with_self_heal(
        workspace_root, file_index, exclude_patterns
    )

    if is_clean:
        if healed_any:
            # Persist updated mtimes; best-effort (§6 step 7).
            try:
                metadata["file_index"] = healed_index
                write_workspace_sentinel(workspace_root, metadata)
            except OSError as e:
                logger.debug(f"Self-heal write-back failed: {e}")
        return (True, stored_hash, stored_hash)

    # Dirty: recompute so callers can report the new aggregate.
    recomputed = compute_workspace_hash_v2(workspace_root, exclude_patterns)
    if recomputed is None:
        return (False, None, stored_hash)
    new_hash, _ = recomputed
    return (False, new_hash, stored_hash)


def get_workspace_clean_hint(workspace_path: Path) -> Optional[str]:
    """Return a one-line hint explaining a non-clean workspace, else None.

    Surfaces the actionable messages called out by ADR-025 §6 (v2 upgrade)
    and §3 (corrupt sentinel). Safe to call on any path; returns None for
    valid v2 sentinels.
    """
    if not isinstance(workspace_path, Path):
        return None
    metadata = read_workspace_metadata(workspace_path.resolve())
    if not metadata:
        return None
    hash_algo = metadata.get("hash_algorithm")
    if hash_algo != HASH_ALGORITHM_V2:
        return "run `mlxk show <name> --recalc-hash` to upgrade content_hash to v2"
    file_index = metadata.get("file_index")
    if not isinstance(file_index, list):
        return "workspace sentinel has no file_index — run `--recalc-hash` to rebuild"
    for entry in file_index:
        try:
            if not isinstance(entry, dict):
                raise CorruptSentinelError("non-dict entry")
            _validate_sentinel_path(entry.get("path"))
        except CorruptSentinelError:
            return "workspace sentinel has invalid file path entry — run `--recalc-hash` to rebuild"
    return None
