from __future__ import annotations

"""Test fixtures for MLX-Knife 2.0 isolated testing."""

# Ensure lightweight stubs are used for heavy deps (mlx, mlx_lm) during unit tests
import sys
from pathlib import Path
_stubs_path = Path(__file__).parent / "stubs"
if str(_stubs_path) not in sys.path:
    sys.path.insert(0, str(_stubs_path))

import os
import re
import tempfile
import pytest
from pathlib import Path
from typing import Generator, Dict, Any
from contextlib import contextmanager
import shutil
import random
import json as _json
import subprocess
import uuid
import hashlib

TEST_SENTINEL = "models--TEST-CACHE-SENTINEL--mlxk2-safety-check"


# =============================================================================
# Test Session Cleanup: Kill Zombie Servers Before Tests
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def cleanup_zombie_servers(request):
    """Kill zombie mlxk test servers ONLY when running live E2E tests.

    SAFETY: Only activates when `-m live_e2e` marker is used. This prevents
    accidentally killing production/development servers when running unit tests.

    When active, this fixture:
    - Kills zombie servers from previous interrupted test runs (Ctrl-C, crashes)
    - Cleans up any leaked servers after the test session completes

    Use case: Prevents RAM exhaustion and port conflicts from accumulated zombies.
    """
    # SAFETY: Only cleanup when live_e2e or wet marker is explicitly requested
    # This prevents killing production/dev servers during unit test runs
    # wet marker includes live_e2e tests (Wet Umbrella pattern)
    selected_markers = request.config.getoption("-m") or ""
    should_cleanup = "live_e2e" in selected_markers or "wet" in selected_markers

    if should_cleanup:
        # Pre-test cleanup: Kill zombies from previous runs
        try:
            result = subprocess.run(
                ["pkill", "-9", "-f", "mlxk2.core.server_base"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("\n[Test Setup] Killed zombie mlxk servers from previous runs")
        except FileNotFoundError:
            # pkill not available (e.g., Windows) - skip cleanup
            pass
        except Exception as e:
            # Best-effort cleanup - don't fail tests if this fails
            print(f"\n[Test Setup] Warning: Failed to kill zombie servers: {e}")

    yield  # Run tests

    if should_cleanup:
        # Post-test cleanup: Kill any servers that leaked during tests
        try:
            result = subprocess.run(
                ["pkill", "-9", "-f", "mlxk2.core.server_base"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("\n[Test Teardown] Cleaned up zombie servers after test session")
        except Exception:
            # Best-effort cleanup - don't fail tests if this fails
            pass


# =============================================================================
# CoW (Copy-on-Write) Prerequisite Checking
# =============================================================================

def _get_volume_root(path: Path) -> Path:
    """Get the mount point (volume root) for a given path.

    On macOS: /Volumes/SSD, /System/Volumes/Data, etc.
    On Linux: /, /mnt/data, etc.
    """
    path = path.resolve()
    while not os.path.ismount(str(path)):
        parent = path.parent
        if parent == path:  # Reached filesystem root
            break
        path = parent
    return path


def _is_apfs_volume(path: Path) -> bool:
    """Check if the volume containing path uses APFS filesystem.

    Returns False on non-macOS or if detection fails.
    """
    if sys.platform != "darwin":
        return False

    volume_root = _get_volume_root(path)
    try:
        result = subprocess.run(
            ["diskutil", "info", str(volume_root)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Look for APFS in File System Personality or Type
        return "apfs" in result.stdout.lower()
    except Exception:
        return False


def _can_use_cow(src: Path, dst_volume_root: Path) -> bool:
    """Check if CoW copy is possible between src and destination volume.

    Requirements for CoW (clonefile):
    1. Both must be on the SAME volume (mount point)
    2. Volume must be APFS (macOS only)

    Args:
        src: Source file/directory path
        dst_volume_root: The volume root where destination will be created

    Returns:
        True if CoW is possible, False otherwise.
    """
    src_root = _get_volume_root(src)

    # Must be same volume
    if src_root != dst_volume_root:
        return False

    # Must be APFS
    return _is_apfs_volume(src_root)

# CRITICAL SAFETY: Unique signature file to prevent accidental deletion of user data
# This file MUST exist before any cleanup of the isolated cache directory
SAFETY_SIGNATURE_FILENAME = ".mlxk2_test_cache_signature"
SAFETY_SIGNATURE_MAGIC = "MLXK2_ISOLATED_TEST_CACHE_V1"
TEST_CACHE_MARKER = "mlxk2_test_"  # Marker in temp dir names for test caches


def _create_safety_signature(cache_root: Path) -> str:
    """Create a unique safety signature file in the cache root.

    This signature MUST be verified before any deletion of the cache directory.
    The signature contains:
    - Magic string to identify this as a test cache
    - Unique UUID generated at creation time
    - SHA256 hash of the path for extra verification

    Returns:
        str: The unique signature ID that must be verified before deletion.

    Raises:
        RuntimeError: If signature file cannot be created atomically.
    """
    signature_id = str(uuid.uuid4())
    path_hash = hashlib.sha256(str(cache_root).encode()).hexdigest()[:16]

    signature_content = {
        "magic": SAFETY_SIGNATURE_MAGIC,
        "signature_id": signature_id,
        "path_hash": path_hash,
        "created_path": str(cache_root),
        "created_at": str(Path(cache_root).stat().st_ctime) if cache_root.exists() else "unknown",
    }

    signature_file = cache_root / SAFETY_SIGNATURE_FILENAME
    temp_signature = cache_root / f".{SAFETY_SIGNATURE_FILENAME}.tmp.{signature_id}"

    try:
        # Write to temp file first (atomic on POSIX)
        temp_signature.write_text(_json.dumps(signature_content, indent=2))
        # Atomic rename
        temp_signature.rename(signature_file)
    except Exception as e:
        # Clean up temp file if rename failed
        if temp_signature.exists():
            temp_signature.unlink()
        raise RuntimeError(f"CRITICAL: Failed to create safety signature: {e}")

    return signature_id


def _verify_safety_signature(cache_root: Path, expected_signature_id: str) -> bool:
    """Verify the safety signature before allowing deletion.

    This function MUST return True before any rmtree operation on cache_root.

    Args:
        cache_root: The directory to verify
        expected_signature_id: The signature ID returned by _create_safety_signature()

    Returns:
        bool: True if and only if:
            - Signature file exists
            - Magic string matches
            - Signature ID matches the expected value
            - Path hash matches current path
    """
    signature_file = cache_root / SAFETY_SIGNATURE_FILENAME

    if not signature_file.exists():
        return False

    try:
        content = _json.loads(signature_file.read_text())
    except Exception:
        return False

    # Verify magic string
    if content.get("magic") != SAFETY_SIGNATURE_MAGIC:
        return False

    # Verify signature ID matches
    if content.get("signature_id") != expected_signature_id:
        return False

    # Verify path hash matches (guards against path manipulation)
    expected_hash = hashlib.sha256(str(cache_root).encode()).hexdigest()[:16]
    if content.get("path_hash") != expected_hash:
        return False

    return True


def _safe_rmtree(cache_root: Path, expected_signature_id: str) -> None:
    """Safely remove a test cache directory after signature verification.

    CRITICAL: This function will REFUSE to delete if signature verification fails.

    Args:
        cache_root: The directory to remove
        expected_signature_id: The signature ID from _create_safety_signature()

    Raises:
        RuntimeError: If signature verification fails - deletion is BLOCKED.
    """
    if not _verify_safety_signature(cache_root, expected_signature_id):
        raise RuntimeError(
            f"CRITICAL SAFETY ABORT: Refusing to delete '{cache_root}' - "
            f"signature verification FAILED. This may be user data!"
        )

    # Additional paranoia checks
    path_str = str(cache_root)
    if TEST_CACHE_MARKER not in path_str:
        raise RuntimeError(
            f"CRITICAL SAFETY ABORT: Path '{cache_root}' does not contain '{TEST_CACHE_MARKER}' marker"
        )

    # Only now is it safe to delete
    shutil.rmtree(cache_root)


def _create_isolated_temp_dir(base_dir: str | None) -> tuple[Path, str]:
    """Atomically create a temp directory with safety signature.

    CRITICAL: This function ensures that a temp directory is NEVER created
    without its corresponding safety signature. If signature creation fails,
    the directory is immediately removed.

    Args:
        base_dir: Base directory for temp creation, or None for system default.

    Returns:
        Tuple of (temp_dir_path, signature_id)

    Raises:
        RuntimeError: If atomic creation fails (directory will be cleaned up).
    """
    temp_dir = tempfile.mkdtemp(prefix="mlxk2_test_", dir=base_dir)
    temp_dir_path = Path(temp_dir)

    try:
        # IMMEDIATELY create signature - this MUST succeed
        signature_id = _create_safety_signature(temp_dir_path)
        return temp_dir_path, signature_id
    except Exception as e:
        # Signature creation failed - MUST clean up the directory
        # Use shutil.rmtree directly (not _safe_rmtree) because no signature exists
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass  # Best effort cleanup
        raise RuntimeError(
            f"CRITICAL: Failed to create safety signature for '{temp_dir}'. "
            f"Directory has been removed. Error: {e}"
        )


# =============================================================================
# Test Cache Context Detection (moved from mlxk2/core/cache.py)
# =============================================================================

def _is_likely_test_cache(path: Path) -> bool:
    """Heuristic to detect test caches safely.

    The TEST_CACHE_MARKER ('mlxk2_test_') in the path is the authoritative indicator.
    Test cache location may vary - /var/folders/ (default temp) or on same volume
    as user cache (for CoW support with external drives).
    """
    return TEST_CACHE_MARKER in str(path)


def _verify_cache_context(expected: str, cache_path: Path | None = None):
    """Verify the cache path matches the expected context.

    Args:
        expected: "test" or "user"
        cache_path: Path to verify. If None, uses current HF_HOME model cache.

    Raises:
        RuntimeError: If context doesn't match expectation.
    """
    if cache_path is None:
        from mlxk2.core.cache import get_current_model_cache
        cache_path = get_current_model_cache()

    if expected == "test":
        if not _is_likely_test_cache(cache_path):
            raise RuntimeError(f"Expected test cache, but using: {cache_path}")
    elif expected == "user":
        if _is_likely_test_cache(cache_path):
            raise RuntimeError(f"Expected user cache, but using test cache: {cache_path}")
    else:
        raise ValueError(f"Unknown cache context: {expected}")


def assert_is_test_cache(cache_path: Path):
    """Ensure operations run against the isolated test cache only.

    Note: Test cache location varies - may be in /var/folders (default) or
    on same volume as user cache (for CoW support on external drives).
    The sentinel file is the authoritative safety check.
    """
    path_str = str(cache_path)
    if TEST_CACHE_MARKER not in path_str:
        raise RuntimeError(f"WARNING: Unexpected cache path - should be test cache: {path_str}")
    sentinel_dir = cache_path / TEST_SENTINEL
    if not sentinel_dir.exists():
        raise RuntimeError(f"MISSING CANARY: Test cache sentinel not found in {cache_path}")


def _copy_cow(src: Path, dst: Path) -> bool:
    """Copy file using CoW (Copy-on-Write) if available, else regular copy.

    On macOS/APFS: Uses `cp -c` which calls clonefile(2) - instant, no disk space.
    On other systems: Falls back to shutil.copy2().

    Note: CoW requires src and dst on the SAME filesystem (volume).
    The isolated_cache fixture creates temp dirs on the same volume as the
    user cache to enable CoW for model copies. If volumes differ (e.g., user
    cache on external SSD, temp on system disk), this falls back to regular copy.

    Returns:
        bool: True if CoW was used, False if regular copy fallback.
    """
    # macOS cp -c uses clonefile(2) for CoW on APFS
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["cp", "-c", str(src), str(dst)],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                return True
            # cp -c failed (e.g., cross-filesystem) - fall through to regular copy
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
    # Fallback to regular copy
    shutil.copy2(src, dst)
    return False


@pytest.fixture
def isolated_cache() -> Generator[Path, None, None]:
    """Create isolated cache for MLX-Knife 2.0 tests - NEVER touches user cache.

    CRITICAL SAFETY: This fixture uses manual directory creation + _safe_rmtree()
    instead of tempfile.TemporaryDirectory to ensure signature verification
    before ANY deletion. This prevents accidental deletion of user data.

    CoW OPTIMIZATION: If user cache is on an APFS volume, the temp directory
    is created on the SAME volume (at volume root level) to enable Copy-on-Write.
    This makes model copies instant and disk-free.

    Volume detection uses os.path.ismount() and diskutil to verify APFS.
    """
    # Determine user cache location FIRST (needed for same-volume temp dir)
    old_hf_home = os.environ.get("HF_HOME")
    user_hf_home = os.environ.get("MLXK2_USER_HF_HOME")
    injected_user_hf_home = False

    if not user_hf_home:
        if old_hf_home:
            user_hf_home = old_hf_home
        else:
            default_hf = Path.home() / ".cache" / "huggingface"
            if (default_hf / "hub").exists():
                user_hf_home = str(default_hf)

    # Determine temp directory location for optimal CoW support
    # Strategy: Create temp on SAME APFS volume as user cache for CoW
    temp_base_dir = None
    user_volume_root = None

    if user_hf_home:
        user_hf_path = Path(user_hf_home)
        if user_hf_path.exists():
            user_volume_root = _get_volume_root(user_hf_path)

            # Only use volume root if it's APFS (CoW requirement)
            if _is_apfs_volume(user_volume_root):
                # Try to create temp dir on volume root for CoW support
                # Use a dedicated subdirectory to avoid cluttering volume root
                cow_temp_base = user_volume_root / ".mlxk2_test_isolation"
                try:
                    cow_temp_base.mkdir(exist_ok=True)
                    temp_base_dir = str(cow_temp_base)
                except (PermissionError, OSError):
                    # No write access to volume root (e.g., read-only system volume)
                    # Fall back to system default temp directory
                    temp_base_dir = None

    # CRITICAL: Atomic creation of temp directory WITH safety signature
    # This ensures a directory is NEVER created without its signature
    temp_dir_path, signature_id = _create_isolated_temp_dir(temp_base_dir)

    cache_path = temp_dir_path / "test_cache"
    cache_path.mkdir()

    # Create hub subdirectory (HuggingFace standard structure)
    hub_path = cache_path / "hub"
    hub_path.mkdir()

    # Expose user cache path to copy helpers
    if user_hf_home and not os.environ.get("MLXK2_USER_HF_HOME"):
        os.environ["MLXK2_USER_HF_HOME"] = user_hf_home
        injected_user_hf_home = True

    # Point HF_HOME to the isolated test cache (code under test will use this)
    os.environ["HF_HOME"] = str(cache_path)

    # CRITICAL: Patch MODEL_CACHE to use our isolated cache
    from mlxk2.core import cache
    original_cache = cache.MODEL_CACHE
    cache.MODEL_CACHE = hub_path

    # SAFETY CANARY: Create sentinel model to verify we're in test cache
    sentinel_dir = hub_path / TEST_SENTINEL
    sentinel_snapshot = sentinel_dir / "snapshots" / "test123456789abcdef0123456789abcdef0123"
    sentinel_snapshot.mkdir(parents=True)
    (sentinel_snapshot / "config.json").write_text('{"model_type": "test_sentinel", "test_cache": true}')
    # Enable strict deletion safety inside tests
    old_strict = os.environ.get("MLXK2_STRICT_TEST_DELETE")
    os.environ["MLXK2_STRICT_TEST_DELETE"] = "1"

    try:
        yield hub_path  # Return hub path (where models-- directories go)
    finally:
        # Restore everything
        cache.MODEL_CACHE = original_cache
        if old_hf_home:
            os.environ["HF_HOME"] = old_hf_home
        elif "HF_HOME" in os.environ:
            del os.environ["HF_HOME"]
        # Remove injected MLXK2_USER_HF_HOME if we set it
        if injected_user_hf_home:
            # Only remove if it matches our injected values to avoid
            # deleting a user-provided variable
            injected_vals = set()
            if old_hf_home:
                injected_vals.add(old_hf_home)
            injected_vals.add(str(Path.home() / ".cache" / "huggingface"))
            if os.environ.get("MLXK2_USER_HF_HOME") in injected_vals:
                del os.environ["MLXK2_USER_HF_HOME"]
        # Restore strict delete flag
        if old_strict is not None:
            os.environ["MLXK2_STRICT_TEST_DELETE"] = old_strict
        elif "MLXK2_STRICT_TEST_DELETE" in os.environ:
            del os.environ["MLXK2_STRICT_TEST_DELETE"]

        # CRITICAL SAFETY: Use _safe_rmtree() with signature verification
        # This REFUSES to delete if signature doesn't match
        _safe_rmtree(temp_dir_path, signature_id)


@pytest.fixture 
def mock_models(isolated_cache):
    """Create realistic mock models in isolated cache."""
    
    def create_model(hf_name: str, commit_hash: str = "abcdef123456789", healthy: bool = True):
        """Create a mock model with proper directory structure."""
        from mlxk2.core.cache import hf_to_cache_dir
        
        cache_dir_name = hf_to_cache_dir(hf_name)
        model_base_dir = isolated_cache / cache_dir_name
        
        # Create snapshots directory
        snapshots_dir = model_base_dir / "snapshots"
        snapshot_dir = snapshots_dir / commit_hash
        snapshot_dir.mkdir(parents=True)
        
        if healthy:
            # Create healthy model files
            (snapshot_dir / "config.json").write_text('{"model_type": "test", "hidden_size": 768}')
            (snapshot_dir / "tokenizer.json").write_text('{"version": "1.0"}')
            (snapshot_dir / "model.safetensors").write_bytes(b"fake_model_weights" * 1000)
        else:
            # Create corrupted model (missing files)
            (snapshot_dir / "config.json").write_text('invalid json {')
        
        return model_base_dir, snapshot_dir
    
    # Pre-create diverse test models for framework detection
    models_created = {}
    
    # MLX models (detected by "mlx-community" in name)
    models_created["mlx-community/Phi-3-mini-4k-instruct-4bit"] = create_model(
        "mlx-community/Phi-3-mini-4k-instruct-4bit", 
        "e9675aa3def456789abcdef0123456789abcdef0"
    )
    
    models_created["mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"] = create_model(
        "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
        "e9675aa3def456789abcdef0123456789abcdef0"  # Same short hash for testing
    )
    
    # Second Qwen model for ambiguous matching tests (mock only - different hash)
    models_created["Qwen/Qwen3-Coder-480B-A35B-Instruct"] = create_model(
        "Qwen/Qwen3-Coder-480B-A35B-Instruct", 
        "beef1234567890abcdef1234567890abcdefbeef"  # Different hash from above
    )
    
    # PyTorch models (detected by .safetensors files)
    pytorch_model = create_model(
        "microsoft/DialoGPT-small",
        "fedcba987654321fedcba987654321fedcba98"
    )
    # Add safetensors file for PyTorch detection
    (pytorch_model[1] / "model.safetensors").write_bytes(b"fake_safetensors" * 100)
    models_created["microsoft/DialoGPT-small"] = pytorch_model
    
    # GGUF model (detected by .gguf files) 
    gguf_model = create_model(
        "TheBloke/Llama-2-7B-Chat-GGUF",
        "1234567890abcdef1234567890abcdef12345678"
    )
    # Add GGUF file
    (gguf_model[1] / "q4_0.gguf").write_bytes(b"fake_gguf_model" * 200)
    models_created["TheBloke/Llama-2-7B-Chat-GGUF"] = gguf_model
    
    # Embeddings model (different model_type in config)
    embed_model = create_model(
        "sentence-transformers/all-MiniLM-L6-v2",
        "abcd1234567890abcdef1234567890abcdef12"
    )
    # Override config for embeddings
    (embed_model[1] / "config.json").write_text('{"model_type": "bert", "task": "feature-extraction"}')
    models_created["sentence-transformers/all-MiniLM-L6-v2"] = embed_model
    
    # Corrupted model for testing tolerance
    models_created["corrupted/model"] = create_model(
        "corrupted/model",
        "corrupted123456789abcdef0123456789abcdef0",
        healthy=False
    )
    
    return models_created


@pytest.fixture
def create_corrupted_cache_entry(isolated_cache):
    """Create corrupted cache entries for testing naming tolerance."""
    
    def create_corrupted(cache_name: str):
        """Create a corrupted cache directory name (violates naming rules)."""
        corrupted_dir = isolated_cache / cache_name
        snapshots_dir = corrupted_dir / "snapshots" / "main"  
        snapshots_dir.mkdir(parents=True)
        
        # Create minimal files so it's detected as model
        (snapshots_dir / "config.json").write_text('{"model_type": "corrupted"}')
        
        return corrupted_dir
    
    return create_corrupted


def test_list_models(cache_path):
    """Test-specific list_models that uses exact cache path provided.
    
    This ensures test operations use the same cache consistently.
    """
    from mlxk2.core.cache import cache_dir_to_hf
    
    # Centralized safety check
    assert_is_test_cache(cache_path)
    
    models = []
    
    if not cache_path.exists():
        return {
            "status": "success",
            "command": "list",
            "data": {
                "models": models,
                "count": 0
            },
            "error": None
        }
    
    # Find all model directories in the provided cache path
    for model_dir in cache_path.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
            continue
            
        hf_name = cache_dir_to_hf(model_dir.name)
        
        # Get hashes from snapshots
        hashes = []
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists():
            for snapshot_dir in snapshots_dir.iterdir():
                if snapshot_dir.is_dir() and len(snapshot_dir.name) == 40:
                    hashes.append(snapshot_dir.name)
        
        models.append({
            "name": hf_name,
            "hashes": sorted(hashes),
            "cached": True
        })
    
    # Sort by name for consistent output
    models.sort(key=lambda x: x["name"])
    
    return {
        "status": "success", 
        "command": "list",
        "data": {
            "models": models,
            "count": len(models)
        },
        "error": None
    }


def test_resolve_model_for_operation(cache_path, model_query):
    """Test-specific model resolution that uses exact cache path provided.
    
    This ensures model resolution uses the same cache as other test operations.
    """
    # Centralized safety check
    assert_is_test_cache(cache_path)
    
    from mlxk2.core.cache import cache_dir_to_hf
    
    # Parse @hash syntax if present
    if "@" in model_query:
        model_name, requested_hash = model_query.split("@", 1)
        requested_hash = requested_hash.lower()
    else:
        model_name = model_query
        requested_hash = None
    
    # Find matching models in the provided cache path
    matching_models = []
    
    if not cache_path.exists():
        return None, None, []
    
    for model_dir in cache_path.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
            continue
            
        hf_name = cache_dir_to_hf(model_dir.name)
        
        # Skip sentinel model
        if "TEST-CACHE-SENTINEL" in hf_name:
            continue
        
        # Check for name match (exact, partial, fuzzy)
        name_matches = False
        if model_name.lower() == hf_name.lower():
            name_matches = True  # Exact match
        elif model_name.lower() in hf_name.lower():
            name_matches = True  # Partial match
        elif any(part.lower() in hf_name.lower() for part in model_name.split("-")):
            name_matches = True  # Fuzzy match
        
        if name_matches:
            # Get available hashes
            snapshots_dir = model_dir / "snapshots"
            available_hashes = []
            if snapshots_dir.exists():
                for snapshot_dir in snapshots_dir.iterdir():
                    if snapshot_dir.is_dir() and len(snapshot_dir.name) == 40:
                        available_hashes.append(snapshot_dir.name)
            
            # Check hash match if requested
            if requested_hash:
                hash_match = any(h.lower().startswith(requested_hash) for h in available_hashes)
                if hash_match:
                    matching_models.append(hf_name)
            else:
                matching_models.append(hf_name)
    
    # Return resolution results
    if len(matching_models) == 0:
        return None, requested_hash, []
    elif len(matching_models) == 1:
        return matching_models[0], requested_hash, None
    else:
        # Ambiguous - return choices
        return None, requested_hash, matching_models


def test_health_check_operation(cache_path, model_query=None):
    """Test-specific health check that uses exact cache path provided.
    
    This ensures health check uses the same cache as other test operations.
    """
    # Centralized safety check
    assert_is_test_cache(cache_path)
    
    from mlxk2.core.cache import cache_dir_to_hf
    import json
    
    healthy_models = []
    unhealthy_models = []
    
    if not cache_path.exists():
        return {
            "status": "success",
            "command": "health",
            "data": {
                "healthy": [],
                "unhealthy": [],
                "summary": {"total": 0, "healthy_count": 0, "unhealthy_count": 0}
            },
            "error": None
        }
    
    # Check all models in cache path
    for model_dir in cache_path.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
            continue
            
        hf_name = cache_dir_to_hf(model_dir.name)
        
        # Skip sentinel model
        if "TEST-CACHE-SENTINEL" in hf_name:
            continue
        
        # Filter by model_query if specified (supports @hash syntax)
        if model_query:
            # Parse @hash syntax if present
            if "@" in model_query:
                query_name, requested_hash = model_query.split("@", 1)
                requested_hash = requested_hash.lower()
                
                # Check name match
                name_matches = (query_name.lower() in hf_name.lower())
                if not name_matches:
                    continue
                
                # Check hash match
                snapshots_dir = model_dir / "snapshots"
                hash_matches = False
                if snapshots_dir.exists():
                    for snapshot_dir in snapshots_dir.iterdir():
                        if snapshot_dir.is_dir() and len(snapshot_dir.name) == 40:
                            if snapshot_dir.name.lower().startswith(requested_hash):
                                hash_matches = True
                                break
                
                if not hash_matches:
                    continue
            else:
                # Simple name filtering
                if model_query.lower() not in hf_name.lower():
                    continue
        
        # Check model health
        is_healthy = True
        health_issues = []
        
        # Check snapshots directory
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            is_healthy = False
            health_issues.append("Missing snapshots directory")
        else:
            # Check for at least one valid snapshot
            valid_snapshots = []
            for snapshot_dir in snapshots_dir.iterdir():
                if snapshot_dir.is_dir() and len(snapshot_dir.name) == 40:
                    # Check for config.json
                    config_file = snapshot_dir / "config.json"
                    if config_file.exists():
                        try:
                            with open(config_file, 'r') as f:
                                json.load(f)
                            valid_snapshots.append(snapshot_dir.name)
                        except (json.JSONDecodeError, IOError):
                            health_issues.append(f"Invalid config.json in {snapshot_dir.name}")
                    else:
                        health_issues.append(f"Missing config.json in {snapshot_dir.name}")
            
            if not valid_snapshots:
                is_healthy = False
                health_issues.append("No valid snapshots found")
        
        # Categorize model
        model_info = {
            "name": hf_name,
            "issues": health_issues
        }
        
        if is_healthy:
            healthy_models.append(model_info)
        else:
            unhealthy_models.append(model_info)
    
    return {
        "status": "success",
        "command": "health", 
        "data": {
            "healthy": healthy_models,
            "unhealthy": unhealthy_models,
            "summary": {
                "total": len(healthy_models) + len(unhealthy_models),
                "healthy_count": len(healthy_models),
                "unhealthy_count": len(unhealthy_models)
            }
        },
        "error": None
    }


@contextmanager
def atomic_cache_context(cache_path: Path, expected_context="test"):
    """Atomic cache switching context manager.

    Temporarily switches HF_HOME to use specific cache, with verification.
    """
    # Store original HF_HOME
    original_hf_home = os.environ.get("HF_HOME")

    try:
        # Switch to specified cache
        if cache_path:
            os.environ["HF_HOME"] = str(cache_path.parent)  # cache_path is hub/, we need parent

        # Verify we're in the right context (using local function)
        _verify_cache_context(expected_context)

        yield cache_path

    finally:
        # Restore original HF_HOME
        if original_hf_home:
            os.environ["HF_HOME"] = original_hf_home
        elif "HF_HOME" in os.environ:
            del os.environ["HF_HOME"]


@contextmanager
def user_cache_context():
    """Context manager for user cache operations."""
    from mlxk2.core.cache import get_current_model_cache

    # Just verify we're in user cache context (using local function)
    _verify_cache_context("user")

    yield get_current_model_cache()


@pytest.fixture
def copy_user_model_to_isolated(isolated_cache):
    """Utility to copy a real user-cache model into the isolated test cache.

    Safety:
    - Read-only on user cache.
    - Requires explicit env var MLXK2_USER_HF_HOME pointing to the user HF_HOME.
    - Skips if user cache or model not present.

    Usage:
    >>> copier = copy_user_model_to_isolated
    >>> path = copier('mlx-community/Phi-3-mini-4k-instruct-4bit', mutations=['remove_config'])
    """
    from mlxk2.core.cache import hf_to_cache_dir

    # IMPORTANT: Do NOT use HF_HOME here because the isolated_cache fixture
    # overrides HF_HOME to point to the test cache. We need the real user cache,
    # which must be provided via MLXK2_USER_HF_HOME explicitly.
    user_hf_home = os.environ.get("MLXK2_USER_HF_HOME")
    if not user_hf_home:
        pytest.skip("MLXK2_USER_HF_HOME not set; skip user->isolated copy")

    user_hub = Path(user_hf_home) / "hub"
    if not user_hub.exists():
        pytest.skip(f"User hub path not found: {user_hub}")

    def mutate_model_dir(model_dir: Path, mutations):
        if not mutations:
            return
        # Normalize list
        if isinstance(mutations, str):
            mutations_list = [mutations]
        else:
            mutations_list = list(mutations)

        # Find a snapshot dir (prefer any 40-char hex dir)
        snapshots = model_dir / "snapshots"
        snap_dirs = [d for d in snapshots.iterdir() if d.is_dir() and len(d.name) == 40] if snapshots.exists() else []
        target_snap = snap_dirs[0] if snap_dirs else None

        # Helper: load index
        def _load_index():
            if target_snap is None:
                return None
            sft_idx = target_snap / "model.safetensors.index.json"
            pt_idx = target_snap / "pytorch_model.bin.index.json"
            for idx in (sft_idx, pt_idx):
                if idx.exists():
                    try:
                        return _json.loads(idx.read_text())
                    except Exception:
                        return None
            return None

        # Helper: get referenced shard paths
        def _referenced_shards():
            index = _load_index()
            if not index or not isinstance(index.get("weight_map"), dict) or target_snap is None:
                return []
            files = sorted(set(index["weight_map"].values()))
            return [target_snap / f for f in files]

        for m in mutations_list:
            if m == 'remove_config' and target_snap is not None:
                cfg = target_snap / "config.json"
                if cfg.exists():
                    cfg.unlink()
            elif m == 'truncate_weight' and target_snap is not None:
                # Truncate first weight-like file
                candidates = list(target_snap.glob("**/*.safetensors")) or list(target_snap.glob("**/*.gguf"))
                if candidates:
                    p = candidates[0]
                    p.write_bytes(b"")
            elif m == 'remove_snapshot' and target_snap is not None:
                shutil.rmtree(target_snap, ignore_errors=True)
                target_snap = None
            elif m == 'drop_random_files' and target_snap is not None:
                files = [f for f in target_snap.rglob("*") if f.is_file()]
                for f in random.sample(files, k=min(len(files), max(1, len(files)//4))):
                    try:
                        f.unlink()
                    except Exception:
                        pass
            elif m == 'inject_invalid_config' and target_snap is not None:
                (target_snap / "config.json").write_text('invalid json {')
            elif m == 'add_partial_tmp' and target_snap is not None:
                (target_snap / ".partial.tmp").write_bytes(b"downloading...")
            elif m == 'delete_indexed_shard' and target_snap is not None:
                # Delete one referenced shard (if index exists)
                refs = _referenced_shards()
                if refs:
                    try:
                        refs[0].unlink(missing_ok=True)
                    except Exception:
                        pass
            elif m == 'truncate_indexed_shard' and target_snap is not None:
                refs = _referenced_shards()
                if refs:
                    refs[0].write_bytes(b"")
            elif m == 'lfsify_indexed_shard' and target_snap is not None:
                refs = _referenced_shards()
                if refs:
                    lfs_content = (
                        "version https://git-lfs.github.com/spec/v1\n"
                        "oid sha256:123\nsize 123\n"
                    )
                    refs[0].write_text(lfs_content)
            elif m == 'remove_index' and target_snap is not None:
                idx = target_snap / "model.safetensors.index.json"
                if idx.exists():
                    idx.unlink()
            # ADR-012 Phase 2: Vision model mutations
            elif m == 'remove_preprocessor' and target_snap is not None:
                preprocessor = target_snap / "preprocessor_config.json"
                if preprocessor.exists():
                    preprocessor.unlink()
            elif m == 'inject_invalid_preprocessor' and target_snap is not None:
                (target_snap / "preprocessor_config.json").write_text('invalid json {')
            elif m == 'remove_tokenizer_json' and target_snap is not None:
                tokenizer_json = target_snap / "tokenizer.json"
                if tokenizer_json.exists():
                    tokenizer_json.unlink()
            elif m == 'inject_invalid_tokenizer_config' and target_snap is not None:
                (target_snap / "tokenizer_config.json").write_text('not json')

    def _latest_snapshot_dir(model_dir: Path) -> Path | None:
        snaps = model_dir / "snapshots"
        if not snaps.exists():
            return None
        dirs = [d for d in snaps.iterdir() if d.is_dir()]
        if not dirs:
            return None
        return max(dirs, key=lambda p: p.stat().st_mtime)

    def copier(hf_name: str, *, mutations=None) -> Path:
        src = user_hub / hf_to_cache_dir(hf_name)
        if not src.exists():
            pytest.skip(f"User model not found: {hf_name} -> {src}")

        dst = isolated_cache / hf_to_cache_dir(hf_name)
        if dst.exists():
            shutil.rmtree(dst)

        # Minimal copy strategy (implicit):
        # - If an index exists, copy the index and the N smallest referenced shards (default N=1).
        # - Otherwise, copy shards matching the safetensors pattern and limit to N (default N=1).
        subset_count = int(os.environ.get("MLXK2_SUBSET_COUNT", "1"))
        min_free_mb = int(os.environ.get("MLXK2_MIN_FREE_MB", "512"))

        # Create dst structure minimally
        (dst / "snapshots").mkdir(parents=True, exist_ok=True)
        src_snap = _latest_snapshot_dir(src)
        if src_snap is None:
            pytest.skip("Source model has no snapshots")
        dst_snap = (dst / "snapshots" / src_snap.name)
        dst_snap.mkdir(parents=True, exist_ok=True)

        # Decide which files to copy
        selected: list[Path] = []
        sft_idx = src_snap / "model.safetensors.index.json"
        pt_idx = src_snap / "pytorch_model.bin.index.json"
        idx = sft_idx if sft_idx.exists() else (pt_idx if pt_idx.exists() else None)
        if idx is not None and idx.exists():
            try:
                index = _json.loads(idx.read_text())
                wm = index.get("weight_map") or {}
                shard_names = sorted(set(wm.values()))
            except Exception:
                shard_names = []
            # pick N smallest shards by size to minimize copy volume
            shard_paths = [src_snap / name for name in shard_names]
            shard_paths = [p for p in shard_paths if p.exists()]
            shard_paths.sort(key=lambda p: p.stat().st_size)
            for p in shard_paths[:max(0, subset_count)]:
                selected.append(p)
            selected.append(idx)
        else:
            # pattern subset: pick shards by filename pattern
            import re
            rgx = re.compile(r"model-\d{5}-of-\d{5}\.safetensors$")
            shard_files = [p for p in src_snap.iterdir() if p.is_file() and rgx.search(p.name)]
            shard_files.sort()
            selected.extend(shard_files[:subset_count])
            # include index if present (unlikely in this branch but safe)
            if sft_idx.exists():
                selected.append(sft_idx)
            elif pt_idx.exists():
                selected.append(pt_idx)
        # Always include config.json if present
        cfg = src_snap / "config.json"
        if cfg.exists():
            selected.append(cfg)

        # ADR-012 Phase 2: Include vision/tokenizer auxiliary assets for health checks
        for aux_file in [
            "preprocessor_config.json",  # Vision models
            "tokenizer_config.json",      # Chat/tokenizer support
            "tokenizer.json",             # Required if tokenizer_config.json present
        ]:
            aux_path = src_snap / aux_file
            if aux_path.exists():
                selected.append(aux_path)

        # Disk space check (on the test cache volume)
        total_bytes = 0
        for p in selected:
            try:
                total_bytes += p.stat().st_size
            except FileNotFoundError:
                pass
        free_bytes = shutil.disk_usage(str(isolated_cache)).free
        if free_bytes < total_bytes + (min_free_mb * 1024 * 1024):
            pytest.skip(f"Not enough free space for subset copy: need ~{(total_bytes/1e6):.1f}MB + safety, have {(free_bytes/1e6):.1f}MB")

        # Copy selected files (CoW on macOS/APFS for instant, disk-free clones)
        for p in selected:
            rel = p.relative_to(src_snap)
            dst_file = dst_snap / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                _copy_cow(p, dst_file)

        # Also place index file at model root so tests can detect it without network
        if idx is not None and idx.exists():
            try:
                shutil.copy2(idx, dst / idx.name)
            except Exception:
                pass

        mutate_model_dir(dst, mutations)

        # Optional: bootstrap index files into the ISOLATED cache (never user cache)
        # Enable with MLXK2_BOOTSTRAP_INDEX=1 to reduce SKIPs for Issue #27 when the
        # selected model doesn't ship an index in your user cache.
        try_bootstrap = os.environ.get("MLXK2_BOOTSTRAP_INDEX") == "1"
        if try_bootstrap:
            # Quick existence check at model root (tests look here first)
            root_sft = dst / "model.safetensors.index.json"
            root_pt = dst / "pytorch_model.bin.index.json"
            if not root_sft.exists() and not root_pt.exists():
                try:
                    # Use hf snapshot_download with allow_patterns to fetch ONLY index files
                    # into the isolated HF_HOME (set by isolated_cache fixture).
                    from huggingface_hub import snapshot_download
                    _ = snapshot_download(
                        repo_id=hf_name,
                        allow_patterns=[
                            "**/model.safetensors.index.json",
                            "**/pytorch_model.bin.index.json",
                        ],
                        local_files_only=False,
                        resume_download=True,
                        token=(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")),
                    )
                    # Copy any fetched index up to model root so tests can detect it
                    fetched = list((dst / "snapshots").rglob("*index.json"))
                    for f in fetched:
                        try:
                            shutil.copy2(f, dst / f.name)
                        except Exception:
                            pass
                except Exception:
                    # Ignore bootstrap failures; tests will skip as before
                    pass
        return dst

    return copier


# =============================================================================
# Wet Umbrella: Auto-assign marker to Portfolio-compatible tests
# =============================================================================

def pytest_collection_modifyitems(config, items):
    """Auto-assign wet marker to Portfolio Discovery compatible tests.

    Wet Umbrella groups tests that can run together in one pytest invocation:
    - User Cache READ tests (live_e2e, live_stop_tokens, live_run, live_list)
    - Workspace operations (live_push)
    - Issue reproduction (issue27)

    Excluded from wet (Isolated Cache WRITE - requires clean import state):
    - live_pull (resumable pull tests)
    - live_clone (clone tests with internal pull)
    - Model validation tests (memory-intensive, belongs in separate benchmark suite)

    See: TESTING-DETAILS.md → Extended Truth Table
    """
    # Compatible live markers (User Cache READ + Workspace)
    LIVE_MARKERS_FOR_WET = {
        "live_e2e",           # Portfolio Discovery (User Cache READ)
        "live_stop_tokens",   # Portfolio Discovery (User Cache READ)
        "live_run",           # User Cache READ
        "live_list",          # User Cache READ
        "live_push",          # Workspace (not Cache)
        "issue27",            # User Cache READ
    }

    # Tests excluded from wet (model validation, not code tests)
    EXCLUDED_FROM_WET = {
        "test_empirical_mapping_single_model",  # Model benchmark (ADR-013)
        "test_empirical_mapping_generate_report",  # Report generation
    }

    for item in items:
        test_markers = {m.name for m in item.iter_markers()}
        test_path = str(item.path)
        test_name = item.name
        is_in_live_dir = "/live/" in test_path or "\\live\\" in test_path

        # Skip model validation tests
        if any(excluded in test_name for excluded in EXCLUDED_FROM_WET):
            continue

        # Wet marker for compatible tests
        if (test_markers & LIVE_MARKERS_FOR_WET) or is_in_live_dir:
            # EXCLUDE Isolated Cache WRITE tests (incompatible with Portfolio Discovery!)
            if "live_pull" not in test_markers and "live_clone" not in test_markers:
                item.add_marker(pytest.mark.wet)


# ============================================================================
# Benchmark Reporting (ADR-013 Phase 0.5)
# ============================================================================

def pytest_addoption(parser):
    """Add --report-output option for benchmark reporting."""
    parser.addoption(
        "--report-output",
        action="store",
        default=None,
        metavar="PATH",
        help="Generate benchmark reports to JSONL file (ADR-013 Phase 0.5)"
    )


def pytest_configure(config):
    """Initialize report file if --report-output is specified."""
    from pathlib import Path
    config.report_file = None
    if report_path := config.getoption("--report-output"):
        config.report_file = Path(report_path).open("a", encoding="utf-8")
        print(f"\n📊 Benchmark reporting enabled: {report_path}")


def pytest_unconfigure(config):
    """Close report file at end of session."""
    if config.report_file:
        config.report_file.close()


# ============================================================================
# Benchmark Reporting Helpers (ADR-013 Phase 0.5)
# ============================================================================

def parse_vm_stat_page_size(output: str) -> int:
    """Extract vm_stat page size in bytes, falling back to 4096."""
    match = re.search(r"page size of (\d+) bytes", output)
    if match:
        return int(match.group(1))
    return 4096


def _get_macos_system_health() -> Dict[str, Any]:
    """Collect macOS system health metrics (ADR-013 Phase 0.5 - v0.2.0).

    Uses macOS-native tools (sysctl, vm_stat, ps) - ZERO new dependencies.
    Enables automatic regression quality assessment via quality_flags.

    Returns:
        dict: System health metrics with keys:
            - swap_used_mb: Current swap usage in MB
            - ram_free_gb: Available RAM in GB
            - zombie_processes: Count of zombie processes
            - quality_flags: List of quality indicators
                ["clean"] = healthy system
                ["degraded_swap"] = swap usage detected (memory pressure)
                ["degraded_zombies"] = zombie processes detected

    Quality Thresholds (empirically derived from Session 43 analysis):
        - Swap: >100 MB indicates memory pressure (beta2→beta3: 1.8 GB swap = +3.4% slowdown)
        - Zombies: >0 indicates stuck processes (REGRESSION-2025-12-08: 14 zombies = +90% slowdown)
    """
    # Force C locale for consistent number formatting (avoid locale-specific decimal separators)
    env = os.environ.copy()
    env["LC_ALL"] = "C"

    health = {
        "swap_used_mb": 0,
        "ram_free_gb": 0.0,
        "zombie_processes": 0,
        "quality_flags": []
    }

    try:
        # Get swap usage via sysctl (macOS native)
        # sysctl vm.swapusage returns: "vm.swapusage: total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)"
        result = subprocess.run(
            ["sysctl", "vm.swapusage"],
            capture_output=True,
            text=True,
            timeout=2,
            env=env
        )
        if result.returncode == 0:
            # Parse: "total = X.XXM  used = Y.YYM  free = Z.ZZM"
            # LC_ALL=C ensures consistent dot decimal separator
            for part in result.stdout.split():
                if part.endswith("M") and "used" in result.stdout:
                    # Extract used value (appears after "used = ")
                    parts = result.stdout.split("used = ")
                    if len(parts) > 1:
                        used_str = parts[1].split()[0]
                        # Parse size (can be M or G suffix)
                        if used_str.endswith("G"):
                            health["swap_used_mb"] = int(float(used_str[:-1]) * 1024)
                        elif used_str.endswith("M"):
                            health["swap_used_mb"] = int(float(used_str[:-1]))
                        break
    except Exception:
        pass  # Swap metric is optional (not critical if it fails)

    try:
        # Get free RAM via vm_stat (macOS native)
        # vm_stat reports page size in the header (Apple Silicon uses 16KB pages).
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=2,
            env=env
        )
        if result.returncode == 0:
            page_size = parse_vm_stat_page_size(result.stdout)
            # Parse "Pages free: 12345."
            for line in result.stdout.splitlines():
                if "Pages free:" in line:
                    pages_free = int(line.split(":")[1].strip().rstrip("."))
                    health["ram_free_gb"] = round(pages_free * page_size / (1024**3), 2)
                    break
    except Exception:
        pass  # RAM metric is optional

    try:
        # Get zombie process count via ps aux (macOS native)
        # Zombies show as "<defunct>" in ps output
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=2,
            env=env
        )
        if result.returncode == 0:
            # Count lines containing "<defunct>"
            health["zombie_processes"] = result.stdout.count("<defunct>")
    except Exception:
        pass  # Zombie count is optional

    # Determine quality flags (empirical thresholds from regression analysis)
    flags = []
    if health["swap_used_mb"] > 100:
        flags.append("degraded_swap")
    if health["zombie_processes"] > 0:
        flags.append("degraded_zombies")

    # If no degradation detected, mark as clean
    if not flags:
        flags.append("clean")

    health["quality_flags"] = flags
    return health


def _get_macos_hardware_profile() -> Dict[str, Any]:
    """Collect macOS hardware profile (ADR-013 Phase 0.5 - v0.2.0).

    Uses macOS-native sysctl - ZERO new dependencies.
    Enables hardware-specific performance analysis (M1 vs M2 vs M3 vs M4).

    Returns:
        dict: Hardware profile with keys:
            - model: Mac model identifier (e.g., "Mac14,9" = M3 Max)
            - cores_physical: Physical CPU cores (P-cores only)
            - cores_logical: Logical CPU cores (P+E cores with hyperthreading)
    """
    profile = {
        "model": "unknown",
        "cores_physical": 0,
        "cores_logical": 0,
    }

    try:
        # Get Mac model identifier
        result = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            profile["model"] = result.stdout.strip()
    except Exception:
        pass

    try:
        # Get physical cores (P-cores)
        result = subprocess.run(
            ["sysctl", "-n", "hw.physicalcpu"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            profile["cores_physical"] = int(result.stdout.strip())
    except Exception:
        pass

    try:
        # Get logical cores (P+E cores with hyperthreading)
        result = subprocess.run(
            ["sysctl", "-n", "hw.logicalcpu"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            profile["cores_logical"] = int(result.stdout.strip())
    except Exception:
        pass

    return profile


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Generate benchmark report for each test (if --report-output enabled).

    Reports are written as JSONL (one JSON object per line) to allow
    streaming and easy appending across test runs.

    Schema version: 0.2.0 (Phase 0.5 - System Health + Hardware Profile)
    See: ADR-013 Phase 0.5 implementation

    Changelog from 0.1.0 → 0.2.0:
        - Added: system.hardware_profile (Mac model, cores)
        - Added: system_health (swap, RAM, zombies, quality_flags)
        - Backward compatible: All 0.1.0 fields preserved
    """
    import json
    from datetime import datetime, timezone

    outcome = yield
    report = outcome.get_result()

    # Only report on test call phase (not setup/teardown)
    if call.when == "call" and item.config.report_file:
        try:
            # Import version here to avoid circular imports
            from mlxk2 import __version__
        except ImportError:
            __version__ = "unknown"

        # Build report data (required fields)
        data = {
            "schema_version": "0.2.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mlx_knife_version": __version__,
            "test": item.nodeid,
            "outcome": report.outcome,
        }

        # Add duration if available
        if hasattr(report, "duration"):
            data["duration"] = report.duration

        # Add skip reason for skipped tests
        if report.outcome == "skipped" and hasattr(report, "longrepr"):
            # Extract skip reason from longrepr tuple
            if isinstance(report.longrepr, tuple) and len(report.longrepr) >= 3:
                skip_reason = report.longrepr[2]
                data.setdefault("metadata", {})["skip_reason"] = skip_reason

        # Extract structured data from user_properties
        # Tests can add data via: request.node.user_properties.append(("key", value))
        for key, value in item.user_properties:
            if key in ("model", "performance", "stop_tokens", "system"):
                # Structured sections (top-level keys)
                data[key] = value
            else:
                # Everything else goes to metadata
                data.setdefault("metadata", {})[key] = value

        # ADR-013 Phase 0.5: Collect system health metrics (v0.2.0)
        # Enables automatic regression quality assessment
        system_health = _get_macos_system_health()
        data["system_health"] = system_health

        # ADR-013 Phase 0.5: Collect hardware profile (v0.2.0)
        # Enables hardware-specific performance analysis (M1 vs M2 vs M3 vs M4)
        hardware_profile = _get_macos_hardware_profile()

        # Add hardware_profile to system section (create if not exists)
        if "system" not in data:
            data["system"] = {}
        data["system"]["hardware_profile"] = hardware_profile

        # Write JSONL (one line per report)
        try:
            item.config.report_file.write(json.dumps(data) + "\n")
            item.config.report_file.flush()
        except Exception as e:
            # Don't fail tests if reporting fails
            print(f"\n⚠️  Benchmark report write failed: {e}")
