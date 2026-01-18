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

import hashlib
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from .pull import pull_to_cache
from .workspace import write_workspace_sentinel
from ..core.cache import hf_to_cache_dir, get_current_cache_root
from mlxk2 import __version__

logger = logging.getLogger(__name__)


def clone_operation(model_spec: str, target_dir: str, health_check: bool = True, force_resume: bool = False) -> Dict[str, Any]:
    """Clone operation following ADR-007 Phase 1: Same-Volume APFS strategy.

    Args:
        model_spec: Model specification (org/repo[@revision])
        target_dir: Target directory for workspace
        health_check: Whether to run health check before copy (default: True)
        force_resume: If True, skip unhealthy check and resume partial download (default: False)

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
    target_created_by_us = False  # Track if we created target dir (for cleanup on failure)

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

        # Phase 2: Create or resume temp cache on same volume as workspace (ADR-018 Phase 0b)
        result["data"]["clone_status"] = "preparing"
        temp_cache, should_download = _create_temp_cache_same_volume(
            target_path, model_spec, force_resume
        )

        # Extract resolved model name early for health check
        resolved_model = model_spec  # Will be updated from pull_result if download happens

        try:
            # Phase 3: Pull to isolated temp cache (conditional, ADR-018 Phase 0b)
            try:
                if should_download:
                    result["data"]["clone_status"] = "pulling"
                    pull_result = pull_to_cache(model_spec, temp_cache)
                else:
                    # Resuming healthy existing download - skip pull
                    result["data"]["clone_status"] = "resuming"
                    logger.info("Skipping download - resuming healthy temp cache")
                    # Create minimal pull_result for continuation
                    pull_result = {
                        "status": "success",
                        "data": {
                            "model": model_spec,
                            "commit_hash": None  # Unknown for resumed cache
                        }
                    }

            except KeyboardInterrupt:
                # User cancelled - set status BEFORE finally block runs
                result["data"]["clone_status"] = "cancelled"
                raise  # Re-raise to outer handler

            if pull_result["status"] != "success":
                result["status"] = "error"
                result["error"] = {
                    "type": "PullFailedError",
                    "message": f"Pull operation failed: {pull_result.get('error', {}).get('message', 'Unknown error')}"
                }
                result["data"]["clone_status"] = "pull_failed"
                return result

            # Phase 3b: Mark download as complete (ADR-018 Phase 0b prep)
            # This marker distinguishes "download incomplete" from "model unhealthy"
            # Critical for resumable clone: know when to skip re-download
            download_marker = temp_cache / ".mlxk2_download_complete"
            download_marker.write_text(f"completed_{int(time.time())}\n")

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
            # Health check is informational only - all HF models must be clonable
            # This allows users to repair broken models (e.g., mlx-vlm #624 affected models)
            if health_check:
                result["data"]["clone_status"] = "health_checking"
                from .health import health_from_cache
                healthy, health_message = health_from_cache(model_spec, temp_cache)
                # Store health status but continue regardless
                result["data"]["health"] = "healthy" if healthy else "unhealthy"
                result["data"]["health_reason"] = health_message
                logger.info(f"Health check: {'healthy' if healthy else 'unhealthy'} - {health_message}")

            # Phase 6: APFS clone temp cache → workspace (instant, CoW)
            result["data"]["clone_status"] = "cloning"
            target_path.mkdir(parents=True, exist_ok=True)
            target_created_by_us = True  # Track for cleanup on failure
            clone_success = _apfs_clone_directory(temp_snapshot, target_path)

            if not clone_success:
                result["status"] = "error"
                result["error"] = {
                    "type": "CloneFailedError",
                    "message": "APFS clone operation failed"
                }
                result["data"]["clone_status"] = "filesystem_error"
                return result

            # Phase 6b: Write workspace sentinel (ADR-018 Phase 0a)
            # Sentinel written AFTER clone success, BEFORE declaring operation complete
            from datetime import datetime, timezone

            # Extract commit hash if available from pull result
            commit_hash = pull_result["data"].get("commit_hash")

            metadata = {
                "mlxk_version": __version__,
                "created_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                "source_repo": resolved_model,
                "source_revision": commit_hash,
                "managed": True,
                "operation": "clone"
            }

            try:
                write_workspace_sentinel(target_path, metadata)
                logger.debug(f"Wrote workspace sentinel to {target_path}")
            except Exception as e:
                # Sentinel write failure is non-fatal - workspace is still usable
                # Log warning but don't fail the entire clone operation
                logger.warning(f"Failed to write workspace sentinel: {e}")
                # Workspace is unmanaged but functional

            # Success - temp cache auto-cleanup via finally block
            result["data"]["clone_status"] = "success"
            result["data"]["message"] = f"Cloned to {target_dir}"

        finally:
            # Phase 7: Conditional cleanup (ADR-018 Phase 0b)
            # Cleanup strategy:
            # 1. Success (clone complete) → always cleanup (workspace created, temp no longer needed)
            # 2. User cancelled (Ctrl-C) → keep for resume (partial download, resumable)
            # 3. Failure with complete download → keep for debugging/repair (user can retry)
            # 4. Failure with incomplete download → cleanup (partial download, unusable - non-resumable error)
            if temp_cache and temp_cache.exists():
                should_cleanup = False

                # Check clone_status FIRST (set by inner except before finally runs)
                if result["data"]["clone_status"] == "cancelled":
                    # User cancelled - preserve for resume
                    should_cleanup = False
                    logger.info(f"Preserving partial download for resume: {temp_cache}")
                    logger.info("Retry with same model+target to resume download")
                elif result["status"] == "success":
                    # Clone succeeded - always cleanup
                    should_cleanup = True
                    cleanup_reason = "clone succeeded"
                else:
                    # Clone failed - check if download was complete
                    download_marker = temp_cache / ".mlxk2_download_complete"
                    if download_marker.exists():
                        # Complete download failed - keep for debugging
                        should_cleanup = False
                        logger.info(f"Keeping temp cache for inspection: {temp_cache}")
                        logger.info("Retry with same target to resume, or use --force-resume")
                    else:
                        # Incomplete download - cleanup
                        should_cleanup = True
                        cleanup_reason = "incomplete download"

                if should_cleanup:
                    logger.debug(f"Cleaning up temp cache ({cleanup_reason}): {temp_cache}")
                    _cleanup_temp_cache_safe(temp_cache)

            # Phase 8: Cleanup partial target directory on failure (defensive)
            # If clone failed after we created target_path but before success,
            # remove it so retries can proceed cleanly
            if target_created_by_us and result["status"] != "success":
                if target_path.exists() and not any(target_path.iterdir()):
                    # Only remove if empty (safety check - don't delete user data)
                    try:
                        target_path.rmdir()
                        logger.debug(f"Cleaned up empty target directory: {target_path}")
                    except OSError as e:
                        logger.warning(f"Failed to cleanup target directory: {e}")

    except KeyboardInterrupt:
        # User cancelled - set status (clone_status may already be set by inner except)
        result["status"] = "error"
        if result["data"]["clone_status"] != "cancelled":
            # KeyboardInterrupt before inner try/except - set clone_status now
            result["data"]["clone_status"] = "cancelled"

        # Prepare error message (check temp_cache state)
        if temp_cache and temp_cache.exists():
            result["error"] = {
                "type": "UserCancelledError",
                "message": (
                    "Operation cancelled by user.\n"
                    f"Partial download preserved at: {temp_cache}\n"
                    f"To resume: Run the same command again (same model + target).\n"
                    f"To delete: rm -rf {temp_cache}"
                )
            }
            logger.info("Operation cancelled - temp cache preserved for resume")
        else:
            result["error"] = {
                "type": "UserCancelledError",
                "message": "Operation cancelled by user."
            }
            logger.info("Operation cancelled - no temp cache to preserve")
        # Don't re-raise - return error result for clean CLI output

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


def _get_deterministic_temp_cache_name(model_spec: str, target_workspace: Path) -> str:
    """Generate deterministic temp cache name for resumable clone (ADR-018 Phase 0b).

    Args:
        model_spec: Model specification (org/repo[@revision])
        target_workspace: Target workspace path

    Returns:
        Deterministic temp cache directory name (e.g., ".mlxk2_temp_a1b2c3d4e5f6g7h8")
    """
    # Deterministic hash: model_spec + target_workspace absolute path
    # Use first 16 hex chars for readability (64-bit hash, collision unlikely)
    hash_input = f"{model_spec}:{target_workspace.resolve()}"
    hash_hex = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    return f".mlxk2_temp_{hash_hex}"


def _check_temp_cache_resume(temp_cache: Path, model_spec: str) -> Tuple[bool, str, bool]:
    """Check if temp cache can be resumed (ADR-018 Phase 0b).

    Args:
        temp_cache: Temp cache directory path
        model_spec: Model specification for health check

    Returns:
        Tuple of (can_resume, reason, is_healthy)
        - can_resume: True if temp cache exists (partial or complete)
        - reason: Human-readable explanation
        - is_healthy: True if download is complete AND passed health check
    """
    # Check 1: Does temp cache exist?
    if not temp_cache.exists():
        return False, "No existing download", False

    # Check 2: Is download complete?
    download_marker = temp_cache / ".mlxk2_download_complete"
    if not download_marker.exists():
        # Partial download (e.g., Ctrl-C) - resumable via HuggingFace
        # No health check needed for partial downloads
        return True, "Partial download (resumable via HuggingFace)", False

    # Check 3: Health check on completed download
    from .health import health_from_cache
    healthy, health_message = health_from_cache(model_spec, temp_cache)

    if healthy:
        return True, "Complete and healthy", True
    else:
        return True, f"Complete but unhealthy: {health_message}", False


def _create_temp_cache_same_volume(target_workspace: Path, model_spec: str, force_resume: bool = False) -> Tuple[Path, bool]:
    """Create or resume temp cache on same APFS volume (ADR-018 Phase 0b).

    Args:
        target_workspace: Target workspace path
        model_spec: Model specification (for deterministic naming)
        force_resume: If True, skip unhealthy check and resume

    Returns:
        Tuple of (temp_cache_path, should_download)
        - temp_cache_path: Path to temp cache directory
        - should_download: True if download needed, False if resuming healthy cache
    """
    # Get target volume mount point via st_dev
    target_volume = _get_volume_mount_point(target_workspace)

    # Deterministic naming (ADR-018 Phase 0b)
    temp_cache_name = _get_deterministic_temp_cache_name(model_spec, target_workspace)
    temp_cache = target_volume / temp_cache_name

    # Check if we can resume existing temp cache
    can_resume, resume_reason, is_healthy = _check_temp_cache_resume(temp_cache, model_spec)

    if can_resume:
        download_marker = temp_cache / ".mlxk2_download_complete"

        if not download_marker.exists():
            # Partial download (Ctrl-C) - resume via HuggingFace snapshot_download
            logger.info(f"Resuming partial download: {resume_reason}")
            return temp_cache, True  # should_download=True to call pull_to_cache (resumes automatically)
        elif is_healthy:
            # Complete + Healthy - skip download entirely, use existing
            logger.info(f"Using existing healthy download: {resume_reason}")
            return temp_cache, False  # should_download=False
        elif force_resume:
            # Complete + Unhealthy + force - use broken model without re-downloading
            logger.warning(f"Force resuming unhealthy download: {resume_reason}")
            return temp_cache, False  # should_download=False
        else:
            # Complete + Unhealthy + no force - delete and restart fresh
            logger.warning(f"Deleting unhealthy temp cache: {resume_reason}")
            shutil.rmtree(temp_cache)
            # Fall through to create new

    # Create new temp cache (either no existing or deleted unhealthy)
    temp_cache.mkdir(parents=True, exist_ok=True)

    # SAFETY: Create sentinel file to prevent accidental user cache deletion
    sentinel = temp_cache / ".mlxk2_temp_cache_sentinel"
    sentinel.write_text(f"mlxk2_temp_cache_created_{int(time.time())}\n")

    return temp_cache, True


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