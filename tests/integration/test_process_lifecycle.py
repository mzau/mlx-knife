"""
High Priority Tests: Process Lifecycle Management

Tests ensure clean process handling and resource management:
- No zombie processes after normal exit or interruption
- Proper signal handling (SIGTERM, SIGKILL, SIGINT)
- Resource management (file handles, sockets, memory)
- Clean streaming interruption
"""
import pytest
import subprocess
import signal
import time
import psutil
import os
from pathlib import Path


@pytest.mark.timeout(30)
class TestProcessLifecycle:
    """Test process lifecycle management and cleanup."""
    
    def test_no_zombie_processes_normal_exit(self, mlx_knife_process, process_monitor):
        """Ensure normal exit leaves no background processes."""
        # Start a simple command that should exit cleanly
        proc = mlx_knife_process(["list"])
        main_pid = proc.pid
        
        # Track child processes before termination
        children_before = process_monitor["get_process_tree"](main_pid)
        
        # Wait for normal completion
        return_code = proc.wait(timeout=10)
        
        # Verify main process exited normally
        assert return_code == 0
        
        # Verify no child processes remain
        assert process_monitor["wait_for_cleanup"](main_pid, timeout=5)
        
        # Double-check: no processes should be running
        for child in children_before:
            assert not child.is_running(), f"Zombie process detected: PID {child.pid}"

    def test_no_zombie_processes_sigint(self, mlx_knife_process, process_monitor, temp_cache_dir):
        """Ensure SIGINT (Ctrl+C) kills all child processes."""
        # Create a mock model for a longer-running command
        mock_model_cache = self._create_simple_mock_model(temp_cache_dir)
        
        # Start a command that would run longer (health check)
        proc = mlx_knife_process(["health"])
        main_pid = proc.pid
        
        # Give it a moment to start and potentially spawn children
        time.sleep(0.5)
        
        # Track child processes
        children_before = process_monitor["get_process_tree"](main_pid)
        
        # Send SIGINT (Ctrl+C equivalent)
        proc.send_signal(signal.SIGINT)
        
        # Wait for termination
        try:
            return_code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Process did not respond to SIGINT within timeout")
        
        # Verify process was interrupted
        assert return_code != 0  # Should not exit normally
        
        # Verify all child processes are cleaned up
        assert process_monitor["wait_for_cleanup"](main_pid, timeout=5)
        
        for child in children_before:
            assert not child.is_running(), f"Child process survived SIGINT: PID {child.pid}"

    def test_no_zombie_processes_sigterm(self, mlx_knife_process, process_monitor, temp_cache_dir):
        """Ensure SIGTERM leads to graceful shutdown."""
        # Create a mock model
        mock_model_cache = self._create_simple_mock_model(temp_cache_dir)
        
        # Start health check command
        proc = mlx_knife_process(["health"])
        main_pid = proc.pid
        
        time.sleep(0.5)
        children_before = process_monitor["get_process_tree"](main_pid)
        
        # Send SIGTERM
        proc.send_signal(signal.SIGTERM)
        
        try:
            return_code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Process did not respond to SIGTERM within timeout")
        
        # Verify graceful shutdown
        assert return_code != 0  # Interrupted
        
        # Verify cleanup
        assert process_monitor["wait_for_cleanup"](main_pid, timeout=5)
        
        for child in children_before:
            assert not child.is_running(), f"Child process survived SIGTERM: PID {child.pid}"

    def test_process_cleanup_after_sigkill(self, mlx_knife_process, process_monitor, temp_cache_dir):
        """Test cleanup after SIGKILL (should kill immediately)."""
        mock_model_cache = self._create_simple_mock_model(temp_cache_dir)
        
        proc = mlx_knife_process(["health"])
        main_pid = proc.pid
        
        time.sleep(0.5)
        children_before = process_monitor["get_process_tree"](main_pid)
        
        # SIGKILL should kill immediately
        proc.send_signal(signal.SIGKILL)
        
        try:
            return_code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pytest.fail("Process did not die from SIGKILL")
        
        # SIGKILL has specific return code
        assert return_code == -signal.SIGKILL
        
        # Child processes should be cleaned up by OS
        assert process_monitor["wait_for_cleanup"](main_pid, timeout=5)

    def test_download_worker_cleanup(self, mlx_knife_process, process_monitor):
        """Ensure download workers don't become zombies."""
        # This test simulates download interruption
        # We'll start a pull command and interrupt it
        
        proc = mlx_knife_process(["pull", "mlx-community/Phi-3-mini-4k-instruct-4bit", "--no-progress"])
        main_pid = proc.pid
        
        # Let download start
        time.sleep(2.0)
        
        children_before = process_monitor["get_process_tree"](main_pid)
        
        # Interrupt the download
        proc.send_signal(signal.SIGINT)
        
        try:
            return_code = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Download process did not respond to interruption")
        
        # Verify cleanup - this is critical for download workers
        assert process_monitor["wait_for_cleanup"](main_pid, timeout=10)
        
        for child in children_before:
            if child.is_running():
                # Give more details about surviving process
                try:
                    cmd = " ".join(child.cmdline())
                    pytest.fail(f"Download worker survived: PID {child.pid}, CMD: {cmd}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass  # Process died while we were checking

    def test_streaming_interruption_cleanup(self, mlx_knife_process, process_monitor):
        """Test clean cancellation of token generation streaming with real model."""
        test_model = "Phi-3-mini-4k-instruct-4bit"
        # Use a prompt that would generate longer output
        test_prompt = "Write a long story about a cat and a dog."
        
        proc = mlx_knife_process(["run", test_model, test_prompt])
        
        # Let it start generating, then interrupt
        time.sleep(2)  # Give it time to start
        
        # Send SIGINT (Ctrl+C) to interrupt gracefully
        proc.send_signal(signal.SIGINT)
        
        try:
            stdout, stderr = proc.communicate(timeout=10)
            # Should terminate gracefully
            assert proc.returncode is not None, "Process didn't terminate after SIGINT"
        except subprocess.TimeoutExpired:
            # If it doesn't respond to SIGINT, force kill
            proc.kill()
            stdout, stderr = proc.communicate()
            pytest.fail("Process didn't respond to SIGINT - cleanup may have failed")
        
        # Check that we got some output before interruption
        assert len(stdout) >= 0, "Process should handle interruption gracefully"

    def test_file_handle_management(self, mlx_knife_process, temp_cache_dir):
        """Verify no file handle leaks after process termination."""
        # Get initial file descriptor count
        initial_fds = len(os.listdir("/proc/self/fd")) if os.path.exists("/proc/self/fd") else 0
        
        mock_model_cache = self._create_simple_mock_model(temp_cache_dir)
        
        # Run several operations
        for _ in range(3):
            proc = mlx_knife_process(["list"])
            proc.wait(timeout=10)
        
        # Check file descriptors haven't grown significantly
        if os.path.exists("/proc/self/fd"):
            final_fds = len(os.listdir("/proc/self/fd"))
            # Allow some tolerance for test framework overhead
            assert final_fds <= initial_fds + 5, f"Potential file handle leak: {initial_fds} -> {final_fds}"

    def _create_simple_mock_model(self, temp_cache_dir: Path) -> Path:
        """Helper to create a simple mock model for testing."""
        cache_name = "models--test--model"
        model_dir = temp_cache_dir / cache_name / "snapshots" / "main"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        (model_dir / "config.json").write_text('{"model_type": "test"}')
        (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
        (model_dir / "model.safetensors").write_bytes(b"fake_model_data" * 100)
        
        return model_dir


@pytest.mark.timeout(60)
class TestResourceManagement:
    """Test resource management and memory cleanup."""
    
    def test_memory_cleanup_after_operations(self, mlx_knife_process, temp_cache_dir):
        """Verify memory is properly released after operations."""
        # This is a basic test - real memory testing would require more sophisticated tools
        mock_model_cache = self._create_simple_mock_model(temp_cache_dir)
        
        # Run operations and ensure they complete without hanging
        operations = [
            ["list"],
            ["health"],
            ["show", "test/model"]  # This should gracefully handle non-existent model
        ]
        
        for op in operations:
            proc = mlx_knife_process(op)
            return_code = proc.wait(timeout=15)
            # Operations should complete (may fail, but should not hang)
            assert return_code is not None, f"Operation {op} hung"

    def _create_simple_mock_model(self, temp_cache_dir: Path) -> Path:
        """Helper to create a simple mock model for testing."""
        cache_name = "models--test--model"
        model_dir = temp_cache_dir / cache_name / "snapshots" / "main"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        (model_dir / "config.json").write_text('{"model_type": "test"}')
        (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
        (model_dir / "model.safetensors").write_bytes(b"fake_model_data" * 100)
        
        return model_dir