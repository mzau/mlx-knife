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

import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

SENTINEL_FILENAME = ".mlxk_workspace.json"


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
