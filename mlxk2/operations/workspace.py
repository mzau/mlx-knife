"""Workspace sentinel management (ADR-018 Phase 0a).

This module provides primitives for managed workspace detection and metadata tracking.

Managed workspaces contain a `.mlxk_workspace.json` sentinel file that enables:
- Workspace lifecycle tracking (clone, convert, push)
- Source provenance (which HF repo was cloned)
- Operation history (what transformations were applied)
- Safety guarantees (e.g., cache sanctity enforcement in convert)

Sentinel format:
{
  "mlxk_version": "2.0.4",
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
