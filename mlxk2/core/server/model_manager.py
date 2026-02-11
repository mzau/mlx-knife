"""Model management for MLX Knife server.

State Machine:
  IDLE → LOADED(model) → SWITCHING → LOADED(other)
  ANY → SHUTTING_DOWN (on shutdown_event)

Thread Safety:
  - _lock serializes all state changes
  - Double-check pattern: check shutdown before AND after lock acquire
  - Cleanup with list() copy to avoid dict mutation during iteration

Memory Gates (ARCHITECTURE.md Principle #4):
  - Vision/Text: 8 GB minimum before load
  - Audio: 4 GB minimum before load
  - wait_for_memory_release() polls until threshold reached or timeout
"""

import subprocess
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import HTTPException


# =============================================================================
# Memory Gate Functions (stateless)
# =============================================================================


def get_available_memory_bytes() -> Optional[int]:
    """Get available system memory in bytes (macOS).

    Returns available (free + speculative) memory, not total memory.
    This is critical for model switching - we need to know if there's
    enough free memory after the previous model was unloaded.

    Note: macOS Tahoe caches aggressively, so "free" is often minimal.
    IMPORTANT: We do NOT count "inactive" pages because Metal/GPU cache may hold
    them even though macOS reports them as "reclaimable". This was causing false
    positives where Memory Gates reported sufficient memory but models failed
    with OOM/Broken pipe due to actual memory pressure. (Session 136 fix)

    Returns:
        Available memory in bytes, or None if unavailable.
    """
    try:
        # macOS: Use vm_stat to get available memory
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.split("\n")
            page_size = 16384  # Default macOS page size (Apple Silicon)
            # Parse page size from first line if available
            if "page size of" in lines[0]:
                try:
                    page_size = int(lines[0].split("page size of")[1].split()[0])
                except (ValueError, IndexError):
                    pass

            free_pages = 0
            speculative_pages = 0
            for line in lines:
                if "Pages free:" in line:
                    free_pages = int(line.split(":")[1].strip().rstrip("."))
                elif "Pages speculative:" in line:
                    speculative_pages = int(line.split(":")[1].strip().rstrip("."))

            # Available = free + speculative only (NOT inactive - may be held by GPU cache)
            return (free_pages + speculative_pages) * page_size
    except Exception:
        pass
    return None


def get_memory_pressure() -> int:
    """Get macOS memory pressure level via sysctl.

    Returns:
        0 = NORMAL (system relaxed, safe to load models)
        1 = WARN (system under some pressure)
        4 = CRITICAL (system under severe pressure)
        -1 = Unable to determine
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "vm.memory_pressure"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return -1


def wait_for_memory_release(
    required_bytes: int,
    timeout_seconds: float = 30.0,
    poll_interval: float = 0.5,
) -> bool:
    """Wait for memory to be released after model unload.

    Metal GPU cache is released asynchronously. This function waits
    until enough memory is available before loading the next model.

    Uses TWO indicators for robust detection (Session 136 finding):
    1. vm.memory_pressure == 0 (macOS kernel says system is relaxed)
    2. Available memory >= required_bytes (enough free+speculative pages)

    Args:
        required_bytes: Minimum available memory needed
        timeout_seconds: Maximum wait time (default 30s for GPU cache release)
        poll_interval: Time between memory checks (default 0.5s)

    Returns:
        True if memory threshold reached, False if timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        # Check memory pressure first (fast sysctl call)
        pressure = get_memory_pressure()
        if pressure == 0:  # NORMAL - system is relaxed
            available = get_available_memory_bytes()
            if available is not None and available >= required_bytes:
                return True
        time.sleep(poll_interval)

    return False


# =============================================================================
# ModelManager Class
# =============================================================================


class ModelManager:
    """Thread-safe model loading with memory gates.

    Manages the lifecycle of MLX models (text, vision, audio) with:
    - Single model at a time (switches cleanup previous before loading new)
    - Memory gates to prevent OOM during model switching
    - Graceful shutdown handling

    State Machine:
        IDLE ──get_or_load(X)──→ LOADED(X) ──get_or_load(Y)──→ SWITCHING ──→ LOADED(Y)
          ↑                           │                              │
          └───────cleanup()───────────┴──────────────────────────────┘

        ANY ──shutdown_event.set()──→ SHUTTING_DOWN (all requests → 503)
    """

    MIN_FREE_BYTES_VISION = 8 * 1024**3  # 8 GB for text/vision models
    MIN_FREE_BYTES_AUDIO = 4 * 1024**3   # 4 GB for audio models

    def __init__(self, shutdown_event: threading.Event):
        """Initialize ModelManager.

        Args:
            shutdown_event: Event to signal server shutdown
        """
        self._cache: Dict[str, Any] = {}
        self._current_model_path: Optional[str] = None
        self._lock = threading.Lock()
        self._shutdown_event = shutdown_event

        # Lazy imports to avoid circular dependencies
        from ...logging import get_logger
        self._logger = get_logger()

    @property
    def current_model(self) -> Optional[str]:
        """Get the currently loaded model path/name."""
        return self._current_model_path

    @property
    def is_shutting_down(self) -> bool:
        """Check if server is shutting down."""
        return self._shutdown_event.is_set()

    def _check_shutdown(self) -> None:
        """Raise 503 if shutdown is in progress."""
        if self._shutdown_event.is_set():
            raise HTTPException(status_code=503, detail="Server is shutting down")

    def _cleanup_previous_model(self) -> None:
        """Clean up previous model and wait for memory release."""
        if not self._cache:
            return

        try:
            for old_runner in list(self._cache.values()):
                try:
                    # VisionRunner uses _cleanup_temp_files, MLXRunner uses cleanup
                    if hasattr(old_runner, 'cleanup'):
                        old_runner.cleanup()
                    if hasattr(old_runner, '_cleanup_temp_files'):
                        old_runner._cleanup_temp_files()
                except Exception as e:
                    self._logger.warning(f"Warning during cleanup: {e}")
        finally:
            self._cache.clear()
            self._current_model_path = None

        # Force Metal GPU memory release before loading new model
        try:
            import mlx.core as mx
            mx.clear_cache()
        except (ImportError, AttributeError):
            pass  # MLX not installed or API changed

    def _wait_for_memory(self, min_free_bytes: int, model_spec: str) -> None:
        """Wait for memory release with logging."""
        if not wait_for_memory_release(min_free_bytes, timeout_seconds=10.0):
            available = get_available_memory_bytes()
            available_gb = (available / (1024**3)) if available else 0
            wanted_gb = min_free_bytes / (1024**3)
            self._logger.warning(
                f"Memory release timeout: {available_gb:.1f} GB available (wanted {wanted_gb:.0f} GB)",
                model=model_spec
            )
            # Continue anyway - the probe/policy check will catch real OOM situations

    def get_or_load_model(self, model_spec: str, verbose: bool = False) -> Any:
        """Get model from cache or load it if not cached.

        Thread-safe model switching with proper cleanup on interruption.
        Supports both text models (MLXRunner) and vision models (VisionRunner).

        Args:
            model_spec: Model name, path, or HuggingFace ID
            verbose: Enable verbose logging

        Returns:
            MLXRunner for text models, VisionRunner for vision models

        Raises:
            HTTPException: On model not found, policy block, or shutdown
        """
        # Abort early if shutdown requested
        self._check_shutdown()

        # Thread-safe model switching
        with self._lock:
            self._check_shutdown()

            # Return cached model if same
            if self._current_model_path == model_spec and model_spec in self._cache:
                return self._cache[model_spec]

            # Clean up previous model
            if self._cache:
                self._cleanup_previous_model()
                self._wait_for_memory(self.MIN_FREE_BYTES_VISION, model_spec)

            # Load new model
            try:
                runner = self._load_text_or_vision_model(model_spec, verbose)
                self._cache[model_spec] = runner
                self._current_model_path = model_spec
                self._logger.info(f"Switched to model: {model_spec}", model=model_spec)
                return runner

            except HTTPException:
                raise
            except KeyboardInterrupt:
                self._logger.warning("Model loading interrupted")
                self._cache.clear()
                self._current_model_path = None
                raise HTTPException(status_code=503, detail="Server interrupted during model load")
            except Exception as e:
                self._logger.error(
                    f"Model load failed: {model_spec}",
                    error_key=f"model_load_{model_spec}",
                    detail=str(e)
                )
                self._cache.clear()
                self._current_model_path = None
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{model_spec}' not found or failed to load: {str(e)}"
                )

    def _load_text_or_vision_model(self, model_spec: str, verbose: bool) -> Any:
        """Load text or vision model with probe/policy validation.

        Args:
            model_spec: Model name, path, or HuggingFace ID
            verbose: Enable verbose logging

        Returns:
            MLXRunner or VisionRunner instance

        Raises:
            HTTPException: On validation failure or policy block
        """
        # Lazy imports to avoid circular dependencies
        from ..cache import get_current_model_cache, hf_to_cache_dir
        from ..model_resolution import resolve_model_for_operation
        from ..capabilities import probe_and_select, PolicyDecision, Backend
        from ...operations.workspace import is_workspace_path
        from ..runner import MLXRunner

        resolved_name, _, _ = resolve_model_for_operation(model_spec)
        model_path = None

        # Resolve model path
        if resolved_name and is_workspace_path(resolved_name):
            model_path = Path(resolved_name)
            self._logger.debug(
                f"Preload path (workspace): resolved_name={resolved_name}, model_path={model_path}",
                model=model_spec
            )
            if not model_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Workspace not found: {model_spec}"
                )
        elif resolved_name:
            cache_root = get_current_model_cache()
            cache_dir = cache_root / hf_to_cache_dir(resolved_name)
            self._logger.debug(
                f"Preload path (cache): resolved_name={resolved_name}, cache_dir={cache_dir}",
                model=model_spec
            )
            snapshots_dir = cache_dir / "snapshots"
            model_path = cache_dir
            if snapshots_dir.exists():
                snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                if snapshots:
                    model_path = max(snapshots, key=lambda x: x.stat().st_mtime)
        else:
            # Resolution failed - check if local directory exists as fallback
            if is_workspace_path(model_spec):
                model_path = Path(model_spec).resolve()
                resolved_name = str(model_path)
                self._logger.debug(
                    f"Preload path (fallback workspace): model_spec={model_spec}, model_path={model_path}",
                    model=model_spec
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model not found in cache: {model_spec}"
                )

        self._logger.debug(f"Preload path: model_path={model_path}", model=model_spec)

        # Probe and select backend (fail-fast validation)
        caps, policy = probe_and_select(
            model_path,
            resolved_name or model_spec,
            context="server",
            has_images=False,
        )

        self._logger.debug(
            f"Probe/policy results: is_vision={caps.is_vision}, "
            f"backend={policy.backend.value}, decision={policy.decision.value}, "
            f"http_status={policy.http_status}",
            model=model_spec
        )

        # Handle policy decision
        if policy.decision == PolicyDecision.BLOCK:
            status_code = policy.http_status or 400
            self._logger.error(f"Model blocked by policy: {policy.message}", model=model_spec)
            raise HTTPException(status_code=status_code, detail=policy.message)

        if policy.decision == PolicyDecision.WARN:
            self._logger.warning(f"Policy warning: {policy.message}", model=model_spec)

        # Check shutdown before expensive load
        self._check_shutdown()

        # Select runner based on backend
        if policy.backend == Backend.MLX_VLM:
            from ..vision_runner import VisionRunner
            self._logger.info(f"Loading vision model: {model_spec}", model=model_spec, backend="mlx_vlm")
            runner = VisionRunner(model_path, resolved_name or model_spec, verbose=verbose)
        else:
            runner = MLXRunner(resolved_name or model_spec, verbose=verbose, install_signal_handlers=False)

        # Load model
        self._check_shutdown()
        runner.load_model()
        self._check_shutdown()

        return runner

    def get_or_load_audio_model(self, model_spec: str, verbose: bool = False) -> Any:
        """Get audio model from cache or load it if not cached.

        Thread-safe model switching with AudioRunner for STT models (ADR-020).
        Uses the same cache as get_or_load_model() but creates AudioRunner instances.

        Args:
            model_spec: Model name, path, or HuggingFace ID
            verbose: Enable verbose logging

        Returns:
            AudioRunner for STT models

        Raises:
            HTTPException: On model not found or shutdown
        """
        # Abort early if shutdown requested
        self._check_shutdown()

        # Thread-safe model switching
        with self._lock:
            self._check_shutdown()

            # Check if model is already cached and is an AudioRunner
            if self._current_model_path == model_spec:
                from ..audio_runner import AudioRunner
                cached = self._cache.get(model_spec)
                if isinstance(cached, AudioRunner):
                    return cached

            # Clean up previous model
            if self._cache:
                self._cleanup_previous_model()
                self._wait_for_memory(self.MIN_FREE_BYTES_AUDIO, model_spec)

            # Load new audio model
            try:
                runner = self._load_audio_model(model_spec, verbose)
                self._cache[model_spec] = runner
                self._current_model_path = model_spec
                self._logger.info(f"Audio model loaded: {model_spec}", model=model_spec)
                return runner

            except HTTPException:
                raise
            except KeyboardInterrupt:
                self._logger.warning("Audio model loading interrupted")
                self._cache.clear()
                self._current_model_path = None
                raise HTTPException(status_code=503, detail="Server interrupted during model load")
            except Exception as e:
                self._logger.error(
                    f"Audio model load failed: {model_spec}",
                    error_key=f"audio_model_load_{model_spec}",
                    detail=str(e)
                )
                self._cache.clear()
                self._current_model_path = None
                raise HTTPException(
                    status_code=404,
                    detail=f"Audio model '{model_spec}' failed to load: {str(e)}"
                )

    def _load_audio_model(self, model_spec: str, verbose: bool) -> Any:
        """Load audio STT model.

        Args:
            model_spec: Model name, path, or HuggingFace ID
            verbose: Enable verbose logging

        Returns:
            AudioRunner instance

        Raises:
            HTTPException: On model not found
        """
        # Lazy imports
        from ..cache import get_current_model_cache, hf_to_cache_dir
        from ..model_resolution import resolve_model_for_operation
        from ...operations.workspace import is_workspace_path
        from ..audio_runner import AudioRunner

        resolved_name, _, _ = resolve_model_for_operation(model_spec)
        model_path = None

        # Resolve model path
        if resolved_name and is_workspace_path(resolved_name):
            model_path = Path(resolved_name)
        elif resolved_name:
            cache_root = get_current_model_cache()
            cache_dir = cache_root / hf_to_cache_dir(resolved_name)
            snapshots_dir = cache_dir / "snapshots"
            if snapshots_dir.exists():
                snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                if snapshots:
                    model_path = max(snapshots, key=lambda x: x.stat().st_mtime)
        elif is_workspace_path(model_spec):
            model_path = Path(model_spec).resolve()
            resolved_name = str(model_path)

        if model_path is None or not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Audio model not found: {model_spec}")

        # Check shutdown before expensive load
        self._check_shutdown()

        self._logger.info(f"Loading audio model: {model_spec}", model=model_spec, backend="mlx_audio")

        # Suppress mlx-audio WhisperProcessor warnings in server mode
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Could not load WhisperProcessor")
            runner = AudioRunner(model_path, resolved_name or model_spec, verbose=verbose)
            runner.load_model()

        self._check_shutdown()
        return runner

    def cleanup(self) -> None:
        """Clean up all cached models and release memory."""
        with self._lock:
            self._cleanup_previous_model()

    def request_interrupt(self) -> None:
        """Request all running generations to stop."""
        try:
            for runner in list(self._cache.values()):
                try:
                    runner.request_interrupt()
                except Exception:
                    pass
        except Exception:
            pass
