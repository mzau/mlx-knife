"""Convert operation for MLX Knife 2.0 (ADR-018 Phase 1+2).

Workspace-to-workspace transformations:
- repair-index: Rebuild safetensors index from existing shards (fixes mlx-vlm #624)
- quantize: Quantize text or vision model to N bits (2, 3, 4, 6, 8)
- dequantize: (Phase 3, future)

ADR-018 Core Rules:
1. Convert operates on workspaces, NEVER on HF cache (hard block)
2. Target workspace gets sentinel written FIRST (atomic) - repair-index
   OR sentinel written AFTER mlx-lm completes - quantize
3. Source can be managed or unmanaged
4. Output is always managed workspace
5. Health check runs on output (unless --skip-health)

Quantize Workflow (differs from repair-index):
- mlx-lm convert() requires target to NOT exist
- mlx-lm creates target directory itself
- Sentinel written AFTER successful quantization
- On failure: shutil.rmtree(dst) for cleanup

Philosophy:
- HF cache is "holy mirror" of remote repos (read-only for convert)
- Workspaces are "working area" for transformations
- User workflow: HF Remote → Clone → Edit/Convert → Push to HF Remote
"""

import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from .workspace import write_workspace_sentinel, is_managed_workspace, read_workspace_metadata, update_workspace_hash
from .health import health_check_workspace
from .clone import _check_apfs_and_warn
from ..core.cache import get_current_cache_root
from ..core.capabilities import classify_convert_target
from ..errors import ErrorType, unsupported_multimodal_error
from mlxk2 import __version__

logger = logging.getLogger(__name__)


def _quantize_text_model(source: Path, target: Path, bits: int, group_size: int = 64) -> None:
    """Quantize text model using mlx-lm.

    NOTE: mlx-lm requires target to NOT exist. We validate this upfront,
    then mlx-lm creates the directory. Sentinel is written AFTER success.

    Future: mlx-vlm/mlx-audio may have different requirements.

    Args:
        source: Source workspace/model path
        target: Target workspace path (must not exist)
        bits: Quantization bits (2, 3, 4, 6, 8)
        group_size: Quantization group size (default: 64)

    Raises:
        ValueError: If quantization fails or mlx-lm not available
    """
    try:
        from mlx_lm import convert as mlx_lm_convert
    except ImportError:
        raise ValueError("mlx-lm not installed. Install with: pip install mlx-lm")

    logger.info(f"Quantizing {source} → {target} ({bits}-bit, group_size={group_size})")

    # mlx-lm creates target directory itself
    mlx_lm_convert(
        hf_path=str(source),
        mlx_path=str(target),
        quantize=True,
        q_bits=bits,
        q_group_size=group_size,
    )

    # Safety: Always rebuild index for consistency after quantization
    rebuild_safetensors_index(target)


def _quantize_vision_model(source: Path, target: Path, bits: int, group_size: int = 64) -> None:
    """Quantize vision model using mlx-vlm.

    API signature is identical to mlx-lm convert(). Same workflow:
    target must NOT exist, mlx-vlm creates the directory.

    Args:
        source: Source workspace/model path
        target: Target workspace path (must not exist)
        bits: Quantization bits (2, 3, 4, 6, 8)
        group_size: Quantization group size (default: 64)

    Raises:
        ValueError: If quantization fails or mlx-vlm not available
    """
    try:
        from mlx_vlm import convert as mlx_vlm_convert
    except ImportError:
        raise ValueError("mlx-vlm not installed. Install with: pip install mlx-vlm")

    logger.info(f"Quantizing vision model {source} → {target} ({bits}-bit, group_size={group_size})")

    mlx_vlm_convert(
        hf_path=str(source),
        mlx_path=str(target),
        quantize=True,
        q_bits=bits,
        q_group_size=group_size,
    )

    # Safety: Always rebuild index for consistency after quantization
    rebuild_safetensors_index(target)


# Backend dispatch. Only "text" and "vision" are executable backends;
# "stt_unsupported" and "unsupported_multimodal" are reject classes handled
# at the call site (see convert_operation quantize branch, ADR-023).
QUANTIZE_BACKENDS: Dict[str, Callable[[Path, Path, int, int], None]] = {
    "text": _quantize_text_model,
    "vision": _quantize_vision_model,
    # "audio": _quantize_audio_model,  # Future: mlx-audio
}


def _get_quantize_backend(classification: str) -> Callable[[Path, Path, int, int], None]:
    """Get executable quantization backend for a classification from classify_convert_target.

    Accepts only "text" or "vision". Reject classifications ("stt_unsupported",
    "unsupported_multimodal") are filtered out by the caller and never reach
    this function.
    """
    backend = QUANTIZE_BACKENDS.get(classification)
    if not backend:
        raise ValueError(
            f"No executable quantize backend for classification '{classification}' "
            "(expected 'text' or 'vision')"
        )
    return backend


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
    mode_opts: Optional[Dict[str, Any]] = None,
    skip_health: bool = False
) -> Dict[str, Any]:
    """Convert operation: workspace → workspace transformation.

    Modes:
    - "repair-index": Rebuild safetensors index only (fixes mlx-vlm #624)
    - "quantize": Quantize to N bits (requires mode_opts["bits"])

    Future modes:
    - "dequantize": Dequantize weights

    Workflow (repair-index):
    1. Validate source exists, target is empty
    2. Cache sanctity check (hard block if source/target in cache)
    3. Create target, write sentinel FIRST (atomic)
    4. Copy non-weight assets (config, tokenizer, etc.) with CoW
    5. Clone safetensors shards with CoW
    6. Apply mode operation (repair-index)
    7. Health check on output

    Workflow (quantize):
    1. Validate source exists, target does NOT exist
    2. Cache sanctity check
    3. mlx-lm creates target and quantizes
    4. Write sentinel AFTER success
    5. Health check on output
    6. On failure: cleanup target directory

    Args:
        source_path: Source workspace path
        target_path: Target workspace path
        mode: Conversion mode ("repair-index", "quantize")
        mode_opts: Mode-specific options (e.g., {"bits": 4, "group_size": 64})
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
    if mode_opts is None:
        mode_opts = {}
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

        # Prepare metadata (used for sentinel in both workflows)
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

        # === QUANTIZE WORKFLOW ===
        # Different from repair-index: mlx-lm creates target itself
        if mode == "quantize":
            # Phase 3Q: Validate target does NOT exist (mlx-lm requirement)
            if dst.exists():
                raise ValueError(
                    f"Target directory exists: {target_path}\n"
                    "Quantize requires target to NOT exist (mlx-lm creates it).\n"
                    "Delete target or choose different path."
                )

            bits = mode_opts.get("bits", 4)
            group_size = mode_opts.get("group_size", 64)

            # Read source config once, used for both re-quantize warning and dispatch
            src_config: Dict[str, Any] = {}
            src_config_path = src / "config.json"
            if src_config_path.exists():
                try:
                    src_config = json.loads(src_config_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    src_config = {}

            # Warn if re-quantizing a model that is already quantized at lower precision
            src_quant = src_config.get("quantization", {})
            src_bits = src_quant.get("bits") if isinstance(src_quant, dict) else None
            if src_bits is not None and bits > src_bits:
                result["data"]["warning"] = (
                    f"Source already {src_bits}-bit. "
                    f"Quantizing to {bits}-bit does not increase precision."
                )

            # Phase 4Q: Classify source for dispatch (ADR-023 Text-First + Verified Multimodal).
            # Hard reject BEFORE any filesystem side effect on dst, so an unsupported
            # multimodal model never causes a partial write or silent text downgrade.
            classification = classify_convert_target(src_config)
            src_model_type_raw = src_config.get("model_type", "")
            src_model_type = src_model_type_raw if isinstance(src_model_type_raw, str) and src_model_type_raw else "<unknown>"

            if classification == "unsupported_multimodal":
                err = unsupported_multimodal_error(src_model_type, operation="convert --quantize")
                result["status"] = "error"
                result["error"] = err.to_dict()
                return result

            if classification == "stt_unsupported":
                result["status"] = "error"
                result["error"] = {
                    "type": ErrorType.NOT_IMPLEMENTED.value,
                    "message": (
                        f"Quantization of speech-to-text model type '{src_model_type}' "
                        "is not currently supported."
                    ),
                    "detail": {"model_type": src_model_type, "operation": "convert --quantize"},
                }
                return result

            # Add quantization info to metadata
            target_metadata["quantization"] = {
                "bits": bits,
                "group_size": group_size
            }

            try:
                backend = _get_quantize_backend(classification)
                logger.info(f"Using {classification} quantization backend")
                backend(src, dst, bits, group_size)

                # Phase 5Q: Write sentinel AFTER successful quantization
                write_workspace_sentinel(dst, target_metadata)
                logger.debug(f"Wrote workspace sentinel to {dst}")

                result["data"]["message"] = f"Quantized to {bits}-bit successfully"
                result["data"]["bits"] = bits
                result["data"]["group_size"] = group_size

            except Exception as e:
                # Cleanup on failure
                if dst.exists():
                    logger.warning(f"Quantization failed, cleaning up {dst}")
                    shutil.rmtree(dst)
                raise ValueError(f"Quantization failed: {e}") from e

        # === REPAIR-INDEX WORKFLOW ===
        elif mode == "repair-index":
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
            try:
                write_workspace_sentinel(dst, target_metadata)
                logger.debug(f"Wrote workspace sentinel to {dst}")
            except Exception as e:
                raise OSError(f"Failed to write workspace sentinel: {e}") from e

            # Phase 4b: Check APFS for CoW support (warn if not available)
            _check_apfs_and_warn(dst)

            # Phase 5: Copy non-weight assets with CoW
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
                    except subprocess.CalledProcessError:
                        shutil.copy2(item, dst / item.name)

            # Phase 6: Clone safetensors shards (CoW with fallback)
            for shard in src.glob("*.safetensors"):
                try:
                    subprocess.run(
                        ["cp", "-c", str(shard), str(dst / shard.name)],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                except subprocess.CalledProcessError:
                    shutil.copy2(shard, dst / shard.name)

            # Phase 7: Rebuild index
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

        # ADR-022 Phase 2a: Compute content hash for clean tracking
        try:
            hash_success, hash_value = update_workspace_hash(dst)
            if hash_success and hash_value:
                result["data"]["content_hash"] = hash_value
                logger.debug(f"Content hash computed: {hash_value[:7]}")
        except Exception as e:
            logger.warning(f"Failed to compute content hash: {e}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = {
            "type": type(e).__name__,
            "message": str(e)
        }

    return result
