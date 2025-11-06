"""Clone operation for MLX Knife 2.0.

Implements ADR-007 Phase 1: Same-Volume APFS Clone strategy.

This implementation:
1. Validates cache and workspace both on same APFS volume
2. Creates isolated temp cache on same volume as workspace
3. Pulls model to temp cache (isolated from user cache)
4. APFS clones temp cache → workspace (instant, zero space initially)
5. Deletes temp cache (cleanup)

User cache is NEVER touched - only temp cache is used and cleaned up.
"""

import logging
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any

from .pull import pull_to_cache
from ..core.cache import hf_to_cache_dir, get_current_cache_root

logger = logging.getLogger(__name__)


def clone_operation(model_spec: str, target_dir: str, health_check: bool = True) -> Dict[str, Any]:
    """Clone operation following ADR-007 Phase 1: Same-Volume APFS strategy.

    Args:
        model_spec: Model specification (org/repo[@revision])
        target_dir: Target directory for workspace
        health_check: Whether to run health check before copy (default: True)

    Returns:
        JSON response following API 0.1.4 schema
    """
    result = {
        "status": "success",
        "command": "clone",
        "error": None,
        "data": {
            "model": model_spec,
            "clone_status": "unknown",
            "message": "",
            "target_dir": str(Path(target_dir).resolve()),
            "health_check": health_check
        }
    }

    temp_cache = None  # Initialize for cleanup in finally block

    try:
        # Validate target directory
        target_path = Path(target_dir).resolve()
        result["data"]["target_dir"] = str(target_path)

        # Check if target exists and is not empty
        if target_path.exists():
            if not target_path.is_dir():
                result["status"] = "error"
                result["error"] = {
                    "type": "InvalidTargetError",
                    "message": f"Target '{target_dir}' exists but is not a directory"
                }
                result["data"]["clone_status"] = "error"
                return result

            # Check if directory is empty
            if any(target_path.iterdir()):
                result["status"] = "error"
                result["error"] = {
                    "type": "InvalidTargetError",
                    "message": f"Target directory '{target_dir}' is not empty"
                }
                result["data"]["clone_status"] = "error"
                return result

        # Phase 1: Validate APFS requirement (ADR-007)
        try:
            _validate_apfs_filesystem(target_path.parent)
        except FilesystemError as e:
            result["status"] = "error"
            result["error"] = {
                "type": "FilesystemError",
                "message": str(e)
            }
            result["data"]["clone_status"] = "filesystem_error"
            return result

        # Phase 1b: Validate same-volume requirement (ADR-007)
        try:
            _validate_same_volume(target_path.parent)
        except FilesystemError as e:
            result["status"] = "error"
            result["error"] = {
                "type": "FilesystemError",
                "message": str(e)
            }
            result["data"]["clone_status"] = "filesystem_error"
            return result

        # Phase 2: Create temp cache on same volume as workspace
        result["data"]["clone_status"] = "preparing"
        temp_cache = _create_temp_cache_same_volume(target_path)

        try:
            # Phase 3: Pull to isolated temp cache (no HF_HOME patching needed)
            result["data"]["clone_status"] = "pulling"
            pull_result = pull_to_cache(model_spec, temp_cache)

            if pull_result["status"] != "success":
                result["status"] = "error"
                result["error"] = {
                    "type": "PullFailedError",
                    "message": f"Pull operation failed: {pull_result.get('error', {}).get('message', 'Unknown error')}"
                }
                result["data"]["clone_status"] = "pull_failed"
                return result

            # Extract resolved model name from pull result
            resolved_model = pull_result["data"]["model"]
            result["data"]["model"] = resolved_model

            # Phase 4: Resolve temp cache snapshot path
            temp_snapshot = _resolve_latest_snapshot(temp_cache, resolved_model)
            if not temp_snapshot or not temp_snapshot.exists():
                result["status"] = "error"
                result["error"] = {
                    "type": "CacheNotFoundError",
                    "message": f"Temp cache snapshot not found for model '{resolved_model}'"
                }
                result["data"]["clone_status"] = "cache_not_found"
                return result

            # Phase 5: Optional health check on temp cache
            if health_check:
                result["data"]["clone_status"] = "health_checking"
                # Use health_from_cache for proper isolation
                from .health import health_from_cache
                healthy, health_message = health_from_cache(model_spec, temp_cache)
                if not healthy:
                    result["status"] = "error"
                    result["error"] = {
                        "type": "ModelUnhealthyError",
                        "message": f"Model failed health check: {health_message}"
                    }
                    result["data"]["clone_status"] = "health_check_failed"
                    return result

            # Phase 6: APFS clone temp cache → workspace (instant, CoW)
            result["data"]["clone_status"] = "cloning"
            target_path.mkdir(parents=True, exist_ok=True)
            clone_success = _apfs_clone_directory(temp_snapshot, target_path)

            if not clone_success:
                result["status"] = "error"
                result["error"] = {
                    "type": "CloneFailedError",
                    "message": "APFS clone operation failed"
                }
                result["data"]["clone_status"] = "filesystem_error"
                return result

            # Success - temp cache auto-cleanup via finally block
            result["data"]["clone_status"] = "success"
            result["data"]["message"] = f"Cloned to {target_dir}"

        finally:
            # Phase 7: Cleanup temp cache (always) - with safety check
            if temp_cache and temp_cache.exists():
                _cleanup_temp_cache_safe(temp_cache)

    except Exception as e:
        result["status"] = "error"
        result["error"] = {
            "type": "CloneOperationError",
            "message": str(e)
        }
        result["data"]["clone_status"] = "error"

    return result


def _validate_apfs_filesystem(path: Path) -> None:
    """Validate APFS requirement for clone operations.

    Called lazily - only on first clone operation, not at CLI startup.
    """
    if not _is_apfs_filesystem(path):
        raise FilesystemError(
            f"APFS required for clone operations. "
            f"Path: {path}\n"
            f"Solution: Use APFS volume or external APFS SSD."
        )


def _validate_same_volume(workspace_path: Path) -> None:
    """Validate that workspace and HF_HOME cache are on same volume (ADR-007 Phase 1)."""
    cache_root = get_current_cache_root()

    # Get volume mount points for both paths
    workspace_volume = _get_volume_mount_point(workspace_path)
    cache_volume = _get_volume_mount_point(cache_root)

    if workspace_volume != cache_volume:
        raise FilesystemError(
            f"Phase 1 requires workspace and cache on same volume.\n"
            f"Workspace volume: {workspace_volume}\n"
            f"Cache volume (HF_HOME): {cache_volume}\n"
            f"Solution: Set HF_HOME to same volume as workspace:\n"
            f"  export HF_HOME={workspace_volume}/huggingface/cache"
        )


def _is_apfs_filesystem(path: Path) -> bool:
    """Simple APFS check - returns True/False only.

    Used by both clone (validation) and push (conditional warning).
    """
    try:
        # Use mount command to check filesystem type on macOS
        result = subprocess.run(['mount'], capture_output=True, text=True)
        abs_path = str(path.resolve())

        # Regex pattern for mount lines: device on mountpoint (fstype, options...)
        mount_pattern = r'^(.+?) on (.+?) \(([^,]+),'

        for line in result.stdout.strip().split('\n'):
            match = re.match(mount_pattern, line)
            if match:
                device, mountpoint, fstype = match.groups()

                # Check if our path is under this mountpoint
                if abs_path.startswith(mountpoint + '/') or abs_path == mountpoint:
                    return fstype == 'apfs'

        return False  # No matching mount found
    except (subprocess.CalledProcessError, re.error):
        return False  # Safe fallback


def _create_temp_cache_same_volume(target_workspace: Path) -> Path:
    """Create temp cache on same APFS volume as target for CoW optimization."""
    # Get target volume mount point via st_dev
    target_volume = _get_volume_mount_point(target_workspace)

    # Create temp cache on same volume
    temp_cache = target_volume / f".mlxk2_temp_{os.getpid()}_{random.randint(1000,9999)}"
    temp_cache.mkdir(parents=True)

    # SAFETY: Create sentinel file to prevent accidental user cache deletion
    sentinel = temp_cache / ".mlxk2_temp_cache_sentinel"
    sentinel.write_text(f"mlxk2_temp_cache_created_{int(time.time())}")

    return temp_cache


def _get_volume_mount_point(path: Path) -> Path:
    """Find mount point (volume root) for given path via st_dev changes."""
    abs_path = path.resolve()
    current = abs_path

    while current != current.parent:
        try:
            parent_stat = current.parent.stat()
            current_stat = current.stat()

            # Different st_dev = mount boundary
            if parent_stat.st_dev != current_stat.st_dev:
                return current
        except (OSError, PermissionError):
            pass
        current = current.parent

    return current  # Filesystem root




def _resolve_latest_snapshot(temp_cache: Path, model_name: str) -> Optional[Path]:
    """Resolve the latest snapshot directory for a model in temp cache."""
    try:
        cache_dir = temp_cache / hf_to_cache_dir(model_name)

        if not cache_dir.exists():
            return None

        snapshots_dir = cache_dir / "snapshots"
        if not snapshots_dir.exists():
            return None

        snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if not snapshots:
            return None

        # Return latest snapshot by modification time
        latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
        return latest_snapshot

    except Exception:
        return None


def _apfs_clone_directory(source: Path, target: Path) -> bool:
    """Clone directory using APFS copy-on-write via clonefile."""
    try:
        for item in source.rglob("*"):
            if item.is_file():
                relative_path = item.relative_to(source)
                target_file = target / relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)

                # Use cp -c for clonefile (APFS CoW)
                subprocess.run(['cp', '-c', str(item), str(target_file)],
                             check=True, capture_output=True)
        return True

    except subprocess.CalledProcessError:
        return False


def _cleanup_temp_cache_safe(temp_cache: Path) -> bool:
    """Safely delete temp cache only if sentinel exists."""
    # SAFETY: Only delete if sentinel exists
    sentinel = temp_cache / ".mlxk2_temp_cache_sentinel"
    if not sentinel.exists():
        logger.warning(f"Refusing to delete {temp_cache} - no sentinel found")
        return False

    shutil.rmtree(temp_cache, ignore_errors=True)
    return True


class FilesystemError(Exception):
    """Raised when filesystem requirements are not met."""
    pass