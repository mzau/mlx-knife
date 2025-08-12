"""
Pytest configuration and shared fixtures for MLX Knife tests.
"""
import os
import tempfile
import shutil
import pytest
import subprocess
import signal
import time
from pathlib import Path
from typing import Generator, List
import psutil


@pytest.fixture
def temp_cache_dir() -> Generator[Path, None, None]:
    """Create a temporary cache directory for isolated testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = Path(temp_dir) / "test_cache"
        cache_path.mkdir()
        
        # Set HF_HOME to our temp directory
        old_hf_home = os.environ.get("HF_HOME")
        os.environ["HF_HOME"] = str(cache_path)
        
        try:
            yield cache_path
        finally:
            # Restore original HF_HOME
            if old_hf_home:
                os.environ["HF_HOME"] = old_hf_home
            elif "HF_HOME" in os.environ:
                del os.environ["HF_HOME"]


@pytest.fixture
def mlx_knife_process():
    """Factory fixture to create and manage mlx_knife subprocess."""
    processes: List[subprocess.Popen] = []
    
    def _create_process(args: List[str], **kwargs) -> subprocess.Popen:
        """Create a new mlx_knife process and track it."""
        full_args = ["python", "-m", "mlx_knife.cli"] + args
        proc = subprocess.Popen(
            full_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **kwargs
        )
        processes.append(proc)
        return proc
    
    yield _create_process
    
    # Cleanup: Kill all created processes
    for proc in processes:
        if proc.poll() is None:  # Process still running
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


@pytest.fixture
def process_monitor():
    """Monitor processes for zombie detection."""
    def _get_process_tree(pid: int) -> List[psutil.Process]:
        """Get all child processes of a given PID."""
        try:
            parent = psutil.Process(pid)
            return parent.children(recursive=True)
        except psutil.NoSuchProcess:
            return []
    
    def _wait_for_process_cleanup(pid: int, timeout: float = 5.0) -> bool:
        """Wait for all child processes to terminate."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            children = _get_process_tree(pid)
            if not children:
                return True
            time.sleep(0.1)
        return False
    
    return {
        "get_process_tree": _get_process_tree,
        "wait_for_cleanup": _wait_for_process_cleanup
    }


@pytest.fixture
def mock_model_cache(temp_cache_dir):
    """Create mock model cache structures for testing."""
    def _create_mock_model(
        model_name: str,
        healthy: bool = True,
        corruption_type: str = None
    ) -> Path:
        """Create a mock model in the cache directory."""
        # Convert model name to cache directory format
        cache_name = model_name.replace("/", "--")
        model_dir = temp_cache_dir / f"models--{cache_name}" / "snapshots" / "main"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if healthy and not corruption_type:
            # Create healthy model files
            (model_dir / "config.json").write_text('{"model_type": "test"}')
            (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
            (model_dir / "model.safetensors").write_bytes(b"fake_model_data" * 100)
        elif corruption_type:
            _create_corrupted_model(model_dir, corruption_type)
        
        return model_dir
    
    def _create_corrupted_model(model_dir: Path, corruption_type: str):
        """Create various types of corrupted models."""
        if corruption_type == "missing_snapshot":
            # Remove snapshots directory
            shutil.rmtree(model_dir.parent.parent)
        elif corruption_type == "missing_config":
            (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
            (model_dir / "model.safetensors").write_bytes(b"fake_model_data")
            # config.json is missing
        elif corruption_type == "lfs_pointer":
            (model_dir / "config.json").write_text('{"model_type": "test"}')
            (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
            # Create LFS pointer file instead of actual data
            (model_dir / "model.safetensors").write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:abc123\n"
                "size 1000000\n"
            )
        elif corruption_type == "truncated_safetensors":
            (model_dir / "config.json").write_text('{"model_type": "test"}')
            (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
            # Create truncated/corrupted safetensors
            (model_dir / "model.safetensors").write_bytes(b"corrupted")
        elif corruption_type == "missing_tokenizer":
            (model_dir / "config.json").write_text('{"model_type": "test"}')
            (model_dir / "model.safetensors").write_bytes(b"fake_model_data")
            # tokenizer.json is missing
    
    return _create_mock_model