"""Convert operation for MLX Knife 2.0 (ADR-018 Phase 1).

Workspace-to-workspace transformations:
- repair-index: Rebuild safetensors index from existing shards (fixes mlx-vlm #624)
- quantize: (Phase 2, future)
- dequantize: (Phase 3, future)

ADR-018 Core Rules:
1. Convert operates on workspaces, NEVER on HF cache (hard block)
2. Target workspace gets sentinel written FIRST (atomic)
3. Source can be managed or unmanaged
4. Output is always managed workspace
5. Health check runs on output (unless --skip-health)

Philosophy:
- HF cache is "holy mirror" of remote repos (read-only for convert)
- Workspaces are "working area" for transformations
- User workflow: HF Remote → Clone → Edit/Convert → Push to HF Remote
"""

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from .workspace import write_workspace_sentinel, is_managed_workspace, read_workspace_metadata
from .health import health_check_workspace
from ..core.cache import get_current_cache_root
from mlxk2 import __version__

logger = logging.getLogger(__name__)


def rebuild_safetensors_index(workspace_path: Path) -> bool:
    """Rebuild model.safetensors.index.json from existing shards.

    This fixes models affected by mlx-vlm #624 regression where conversion
    overwrites correct index with source model's outdated index.

    Strategy:
    1. Find all *.safetensors shard files
    2. Read safetensors headers (no full tensor load)
    3. Build weight_map: tensor_key → shard_filename
    4. Write index atomically (tmp + rename)

    Args:
        workspace_path: Path to workspace containing .safetensors shards

    Returns:
        True if index rebuilt successfully
        False if no shards found (single-file model, no index needed)

    Raises:
        ValueError: If shard headers are corrupted or unreadable
        OSError: If index write fails

    Example:
        >>> success = rebuild_safetensors_index(Path("./workspace"))
        >>> if success:
        ...     print("Index rebuilt")
        ... else:
        ...     print("Single-file model, no index needed")
    """
    from safetensors import safe_open

    # Find all safetensors shards (sorted for deterministic output)
    shards = sorted(workspace_path.glob("*.safetensors"))

    if not shards:
        logger.debug(f"No safetensors files in {workspace_path}")
        return False

    if len(shards) == 1:
        # Single-file model (e.g., model.safetensors)
        # No index needed - MLX/HF loaders handle directly
        logger.debug(f"Single safetensors file, no index needed: {shards[0].name}")
        return False

    # Build weight map by reading headers only (no tensor load)
    weight_map = {}
    total_size = 0

    for shard in shards:
        total_size += shard.stat().st_size

        try:
            # Read safetensors header without loading tensors
            # framework="mlx" ensures compatibility with MLX tensor format
            with safe_open(shard, framework="mlx", device="cpu") as f:
                for key in f.keys():
                    if key in weight_map:
                        # Duplicate key across shards - invalid model
                        raise ValueError(
                            f"Duplicate tensor key '{key}' found in multiple shards: "
                            f"{weight_map[key]} and {shard.name}"
                        )
                    weight_map[key] = shard.name

                    logger.debug(f"Mapped {key} → {shard.name}")
        except Exception as e:
            raise ValueError(f"Failed to read shard {shard.name}: {e}") from e

    if not weight_map:
        raise ValueError("No tensors found in shards - corrupted model?")

    # Build index structure (HuggingFace format)
    index = {
        "metadata": {
            "total_size": total_size
        },
        "weight_map": weight_map
    }

    # Atomic write (tmp + rename)
    index_path = workspace_path / "model.safetensors.index.json"
    tmp_path = index_path.with_suffix(".json.tmp")

    try:
        tmp_path.write_text(json.dumps(index, indent=2) + "\n")
        tmp_path.rename(index_path)
        logger.info(f"Rebuilt index with {len(weight_map)} tensors across {len(shards)} shards")
        return True
    except Exception as e:
        # Cleanup tmp file on failure
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass  # Best effort cleanup
        raise OSError(f"Failed to write index: {e}") from e


def convert_operation(
    source_path: str,
    target_path: str,
    mode: str,
    skip_health: bool = False
) -> Dict[str, Any]:
    """Convert operation: workspace → workspace transformation.

    Phase 1 modes:
    - "repair-index": Rebuild safetensors index only (fixes mlx-vlm #624)

    Future modes (Phase 2+):
    - "quantize": Quantize to N bits
    - "dequantize": Dequantize weights

    Workflow:
    1. Validate source exists, target is empty
    2. Cache sanctity check (hard block if source/target in cache)
    3. Create target, write sentinel FIRST (atomic)
    4. Copy non-weight assets (config, tokenizer, etc.) with CoW
    5. Clone safetensors shards with CoW
    6. Apply mode operation (repair-index/quantize/etc.)
    7. Health check on output

    Args:
        source_path: Source workspace path
        target_path: Target workspace path
        mode: Conversion mode ("repair-index" for Phase 1)
        skip_health: Skip health check on output (debug only)

    Returns:
        Result dict with JSON API schema:
        {
          "status": "success" | "error",
          "command": "convert",
          "data": {
            "source": str,
            "target": str,
            "mode": str,
            "health_check": bool,
            "health_status": "healthy" | "unhealthy",
            "health_reason": str,
            "message": str
          },
          "error": {
            "type": str,
            "message": str
          } | None
        }

    Raises:
        None - all errors returned in result dict
    """
    result = {
        "status": "success",
        "command": "convert",
        "error": None,
        "data": {
            "source": source_path,
            "target": target_path,
            "mode": mode,
            "health_check": not skip_health
        }
    }

    try:
        # Phase 1: Validate paths
        src = Path(source_path).resolve()
        dst = Path(target_path).resolve()

        if not src.exists():
            raise ValueError(f"Source path does not exist: {source_path}")

        if not src.is_dir():
            raise ValueError(f"Source must be a directory: {source_path}")

        # Phase 2: Cache sanctity check (hard block)
        # ADR-018: Cache is read-only for convert, use clone first
        cache_root = get_current_cache_root()

        def is_in_cache(path: Path) -> bool:
            """Check if path is inside HF cache."""
            try:
                path.relative_to(cache_root)
                return True
            except ValueError:
                return False

        if is_in_cache(src):
            raise ValueError(
                f"Source path is in HF cache (read-only): {src}\n"
                "Convert operates on workspaces only. Use 'mlxk clone' first to create a workspace:\n"
                f"  mlxk clone <model> ./workspace\n"
                f"  mlxk convert ./workspace ./workspace-fixed --repair-index"
            )

        if is_in_cache(dst):
            raise ValueError(
                f"Target path is in HF cache (read-only): {dst}\n"
                "Convert cannot write to cache. Choose workspace location outside cache:\n"
                f"  mlxk convert {source_path} ./fixed-workspace --repair-index"
            )

        # Phase 3: Validate target is empty or new
        if dst.exists():
            if any(dst.iterdir()):
                raise ValueError(
                    f"Target directory not empty: {target_path}\n"
                    "Convert requires empty target workspace. Use different path or delete contents."
                )
        else:
            dst.mkdir(parents=True)

        # Phase 4: Write workspace sentinel FIRST (atomic, ADR-018 contract)
        # Ensures target is identifiable as managed workspace before any processing
        src_metadata = {}
        if is_managed_workspace(src):
            src_metadata = read_workspace_metadata(src)

        target_metadata = {
            "mlxk_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "source_repo": src_metadata.get("source_repo", str(src)),
            "source_revision": src_metadata.get("source_revision"),
            "managed": True,
            "operation": "convert",
            "mode": mode
        }

        try:
            write_workspace_sentinel(dst, target_metadata)
            logger.debug(f"Wrote workspace sentinel to {dst}")
        except Exception as e:
            raise OSError(f"Failed to write workspace sentinel: {e}") from e

        # Phase 5: Copy non-weight assets with CoW
        # Skip .safetensors (will be cloned), copy config/tokenizer/images/etc.
        # Use APFS CoW (cp -c) for efficient cloning
        logger.info(f"Copying non-weight assets from {src} to {dst}")

        for item in src.iterdir():
            # Skip safetensors files (handled separately)
            # Skip existing sentinel in target
            # Skip model.safetensors.index.json (will be rebuilt)
            if item.name.endswith(".safetensors"):
                continue
            if item.name == ".mlxk_workspace.json":
                continue
            if item.name == "model.safetensors.index.json":
                continue

            if item.is_file():
                try:
                    # Use APFS CoW for large files (instant, zero space initially)
                    subprocess.run(
                        ["cp", "-c", str(item), str(dst / item.name)],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to CoW copy {item.name}, using regular copy: {e}")
                    # Fallback to regular copy
                    import shutil
                    shutil.copy2(item, dst / item.name)

        # Phase 6: Clone safetensors shards (CoW)
        logger.info("Cloning safetensors shards")

        for shard in src.glob("*.safetensors"):
            try:
                subprocess.run(
                    ["cp", "-c", str(shard), str(dst / shard.name)],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to CoW copy {shard.name}, using regular copy: {e}")
                import shutil
                shutil.copy2(shard, dst / shard.name)

        # Phase 7: Apply mode operation
        if mode == "repair-index":
            logger.info("Rebuilding safetensors index")
            success = rebuild_safetensors_index(dst)

            if not success:
                result["data"]["message"] = "No index to rebuild (single-file model or no safetensors found)"
            else:
                result["data"]["message"] = "Index rebuilt successfully"

        else:
            raise ValueError(f"Unsupported conversion mode: {mode}")

        # Phase 8: Health check on output
        if not skip_health:
            logger.info("Running health check on output workspace")
            healthy, reason, managed = health_check_workspace(dst)

            result["data"]["health_status"] = "healthy" if healthy else "unhealthy"
            result["data"]["health_reason"] = reason

            if not healthy:
                result["status"] = "error"
                result["error"] = {
                    "type": "HealthCheckFailed",
                    "message": f"Output workspace failed health check: {reason}"
                }
                return result

        logger.info("Convert operation completed successfully")

    except Exception as e:
        logger.error(f"Convert operation failed: {e}")
        result["status"] = "error"
        result["error"] = {
            "type": type(e).__name__,
            "message": str(e)
        }

    return result
