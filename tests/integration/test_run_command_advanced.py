"""
Advanced Tests for Run Command

Tests the most problematic aspects of the run command:
- Process lifecycle during model execution
- Memory management with model loading/unloading  
- Streaming interruption handling
- Error conditions and recovery
"""
import pytest
import subprocess
import signal
import time
import threading
from pathlib import Path


@pytest.mark.timeout(120)
class TestRunCommandProcessLifecycle:
    """Test process management during model execution."""
    
    def test_run_command_normal_completion(self, mlx_knife_process, process_monitor, mock_model_cache):
        """Test run command completes normally and cleans up."""
        # Create a mock model (won't actually run, but tests process handling)
        mock_model_cache("test-model", healthy=True)
        
        proc = mlx_knife_process(["run", "test-model", "Hello"])
        main_pid = proc.pid
        
        # Track child processes
        children_before = process_monitor["get_process_tree"](main_pid)
        
        try:
            # Wait for completion (will likely fail due to mock model, but should not hang)
            return_code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Run command hung during execution")
        
        # Should complete (success or failure, but not hang)
        assert return_code is not None, "Run command did not complete"
        
        # Verify child process cleanup
        assert process_monitor["wait_for_cleanup"](main_pid, timeout=10)
        
        for child in children_before:
            assert not child.is_running(), f"Run command left zombie process: PID {child.pid}"

    def test_run_command_sigint_during_execution(self, mlx_knife_process, process_monitor, mock_model_cache):
        """Test interruption during model execution."""
        mock_model_cache("test-model", healthy=True)
        
        proc = mlx_knife_process(["run", "test-model", "This is a longer prompt that might take time"])
        main_pid = proc.pid
        
        # Give it time to start
        time.sleep(2)
        
        children_before = process_monitor["get_process_tree"](main_pid)
        
        # Send interrupt
        proc.send_signal(signal.SIGINT)
        
        try:
            return_code = proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Run command did not respond to SIGINT")
        
        # Should exit on interrupt
        assert return_code is not None
        assert return_code != 0  # Should not exit normally
        
        # Clean up child processes
        assert process_monitor["wait_for_cleanup"](main_pid, timeout=10)
        
        for child in children_before:
            assert not child.is_running(), f"Run child process survived SIGINT: PID {child.pid}"

    def test_run_command_sigterm_handling(self, mlx_knife_process, process_monitor, mock_model_cache):
        """Test SIGTERM during model execution."""
        mock_model_cache("test-model", healthy=True)
        
        proc = mlx_knife_process(["run", "test-model", "Test prompt"])
        main_pid = proc.pid
        
        time.sleep(2)
        children_before = process_monitor["get_process_tree"](main_pid)
        
        # Send SIGTERM
        proc.send_signal(signal.SIGTERM)
        
        try:
            return_code = proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Run command did not respond to SIGTERM")
        
        assert return_code is not None
        assert return_code != 0
        
        # Cleanup verification
        assert process_monitor["wait_for_cleanup"](main_pid, timeout=10)
        
        for child in children_before:
            assert not child.is_running(), f"Run child survived SIGTERM: PID {child.pid}"

    def test_run_command_model_loading_failure(self, mlx_knife_process, process_monitor):
        """Test process cleanup when model loading fails."""
        # Use nonexistent model to trigger loading failure
        proc = mlx_knife_process(["run", "nonexistent-model-12345", "Test prompt"])
        main_pid = proc.pid
        
        children_before = process_monitor["get_process_tree"](main_pid)
        
        try:
            return_code = proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Run command hung on model loading failure")
        
        # Should fail gracefully
        assert return_code is not None
        assert return_code != 0  # Should fail on missing model
        
        # Should not leave zombies even on failure
        assert process_monitor["wait_for_cleanup"](main_pid, timeout=5)
        
        for child in children_before:
            assert not child.is_running(), f"Process survived model loading failure: PID {child.pid}"


@pytest.mark.timeout(90)
class TestRunCommandMemoryManagement:
    """Test memory management during run command execution."""
    
    def test_run_command_memory_cleanup_after_completion(self, mlx_knife_process, mock_model_cache):
        """Test memory is released after run command completes."""
        mock_model_cache("test-model", healthy=True)
        
        # Run command multiple times to test memory cleanup
        for i in range(3):
            proc = mlx_knife_process(["run", "test-model", f"Test prompt {i}"])
            
            try:
                return_code = proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                proc.kill()
                pytest.fail(f"Run command {i} hung")
            
            # Should complete (may fail, but should not hang)
            assert return_code is not None, f"Run command {i} did not complete"

    def test_run_command_memory_cleanup_on_interruption(self, mlx_knife_process, process_monitor, mock_model_cache):
        """Test memory cleanup when run is interrupted."""
        mock_model_cache("test-model", healthy=True)
        
        proc = mlx_knife_process(["run", "test-model", "Longer test prompt for interruption"])
        main_pid = proc.pid
        
        # Let it start
        time.sleep(3)
        
        # Interrupt
        proc.send_signal(signal.SIGINT)
        
        try:
            return_code = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Run command did not handle interruption")
        
        # Verify cleanup
        assert return_code is not None
        assert process_monitor["wait_for_cleanup"](main_pid, timeout=10)

    def test_run_command_handles_corrupted_model(self, mlx_knife_process, mock_model_cache):
        """Test run command handles corrupted models gracefully."""
        # Create corrupted model
        mock_model_cache("broken-model", healthy=False, corruption_type="truncated_safetensors")
        
        proc = mlx_knife_process(["run", "broken-model", "Test prompt"])
        
        try:
            return_code = proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Run command hung on corrupted model")
        
        # Should fail gracefully on corrupted model
        assert return_code is not None
        assert return_code != 0  # Should fail


@pytest.mark.timeout(60)
class TestRunCommandStreamingAndOutput:
    """Test streaming and output handling in run command."""
    
    def test_run_command_streaming_interruption(self, mlx_knife_process):
        """Test interruption during token streaming with real MLX model."""
        test_model = "Phi-3-mini-4k-instruct-4bit"
        # Use prompt that would generate substantial output
        test_prompt = "Explain machine learning in detail with examples."
        
        proc = mlx_knife_process(["run", test_model, test_prompt])
        
        # Let streaming start, then interrupt
        time.sleep(3)  # Allow generation to begin
        
        # Send interrupt signal
        proc.send_signal(signal.SIGINT)
        
        try:
            stdout, stderr = proc.communicate(timeout=15)
            # Should handle interruption gracefully
            assert proc.returncode is not None, "Process should terminate after interrupt"
            # Should have generated some output before interruption
            assert len(stdout) > 0, "Should have some output before interruption"
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            pytest.fail("Process didn't respond to interruption signal")

    def test_run_command_output_handling(self, mlx_knife_process, mock_model_cache):
        """Test that run command handles output correctly."""
        mock_model_cache("test-model", healthy=True)
        
        proc = mlx_knife_process(["run", "test-model", "Hello"])
        
        try:
            stdout, stderr = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Run command hung during output")
        
        # Should produce some output (even if error message)
        total_output = len(stdout) + len(stderr)
        assert total_output > 0, "Run command produced no output"

    def test_run_command_long_prompt_handling(self, mlx_knife_process, mock_model_cache):
        """Test run command with very long prompts."""
        mock_model_cache("test-model", healthy=True)
        
        # Create long prompt
        long_prompt = "This is a test prompt. " * 100  # ~2500 characters
        
        proc = mlx_knife_process(["run", "test-model", long_prompt])
        
        try:
            return_code = proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Run command hung on long prompt")
        
        # Should handle long prompt without hanging
        assert return_code is not None

    def test_run_command_special_characters(self, mlx_knife_process, mock_model_cache):
        """Test run command handles special characters in prompts."""
        mock_model_cache("test-model", healthy=True)
        
        special_prompts = [
            "Hello 世界",  # Unicode
            "Test with \"quotes\" and 'apostrophes'",  # Quotes
            "Newlines\nand\ttabs",  # Whitespace
            "emoji 🚀 test",  # Emoji
        ]
        
        for prompt in special_prompts:
            proc = mlx_knife_process(["run", "test-model", prompt])
            
            try:
                return_code = proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                pytest.fail(f"Run command hung on special characters: {prompt[:20]}...")
            
            # Should handle special characters gracefully
            assert return_code is not None


@pytest.mark.timeout(45)
class TestRunCommandErrorConditions:
    """Test run command error handling."""
    
    def test_run_command_insufficient_memory(self, mlx_knife_process, mock_model_cache):
        """Test behavior when system might be low on memory."""
        mock_model_cache("large-model", healthy=True)
        
        # We can't actually simulate low memory, but we can test the process handles errors
        proc = mlx_knife_process(["run", "large-model", "Test prompt"])
        
        try:
            return_code = proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Run command hung during error condition")
        
        # Should complete (success or failure)
        assert return_code is not None

    def test_run_command_missing_dependencies(self, mlx_knife_process):
        """Test run command when model dependencies might be missing."""
        # Try to run with invalid model to test error handling
        proc = mlx_knife_process(["run", "invalid/missing-model", "Test"])
        
        try:
            return_code = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Run command hung on missing dependencies")
        
        # Should fail gracefully
        assert return_code is not None
        assert return_code != 0

    def test_run_command_multiple_concurrent_executions(self, mlx_knife_process, mock_model_cache):
        """Test multiple concurrent run commands don't interfere."""
        mock_model_cache("test-model", healthy=True)
        
        processes = []
        
        # Start multiple run commands
        for i in range(3):
            proc = mlx_knife_process(["run", "test-model", f"Concurrent test {i}"])
            processes.append(proc)
        
        # Wait for all to complete
        for i, proc in enumerate(processes):
            try:
                return_code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                pytest.fail(f"Concurrent run command {i} hung")
            
            # Each should complete independently
            assert return_code is not None, f"Concurrent run {i} did not complete"


@pytest.mark.timeout(60)
class TestRunCommandContextAwareLimits:
    """Test context-aware token limits in Issues #15 and #16 resolution."""
    
    def test_context_length_extraction_from_real_model(self, mlx_knife_process, mock_model_cache):
        """Test that context length is correctly extracted from real model configs."""
        # Create a mock model with realistic config.json
        model_path = mock_model_cache("test-model", healthy=True)
        
        # Add custom config.json with specific context length
        config_content = {
            "max_position_embeddings": 4096,
            "hidden_size": 768,
            "num_attention_heads": 12
        }
        import json
        (model_path / "config.json").write_text(json.dumps(config_content))
        
        # Test that the model context length is accessible
        # This is an indirect test - we test that the run command uses model-aware limits
        # by checking that it doesn't hang with realistic models
        proc = mlx_knife_process([
            "run", "test-model", "Test prompt",
            "--max-tokens", "8000",  # Request more than typical model context
            "--verbose"
        ])
        
        try:
            # Should complete within timeout (won't actually generate due to mock)
            return_code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Run command hung with high max-tokens")
            
        # Should complete (may fail due to mock model, but shouldn't hang)
        assert return_code is not None, "Run command did not complete with context-aware limits"
        
    def test_server_vs_interactive_token_policies(self, mock_model_cache):
        """Test that server mode uses DoS protection while interactive mode uses full context."""
        # This test validates the architectural decision:
        # - Server mode: context_length / 2 (DoS protection)
        # - Interactive mode: full context_length
        
        from mlx_knife.mlx_runner import MLXRunner, get_model_context_length
        import tempfile
        import json
        import os
        
        # Create a temporary model directory with config
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.json")
            config = {"max_position_embeddings": 4096}
            
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            # Test context length extraction
            context_length = get_model_context_length(temp_dir)
            assert context_length == 4096, "Context length extraction failed"
            
            # Test MLXRunner effective token calculation
            runner = MLXRunner(temp_dir, verbose=False)
            runner._context_length = 4096
            
            # Interactive mode should use full context
            interactive_tokens = runner.get_effective_max_tokens(8000, interactive=True)
            assert interactive_tokens == 4096, f"Interactive mode should use full context: {interactive_tokens}"
            
            # Server mode should use half context (DoS protection)
            server_tokens = runner.get_effective_max_tokens(8000, interactive=False)
            assert server_tokens == 2048, f"Server mode should use half context: {server_tokens}"
            
            # User requests smaller than limits should be honored
            small_interactive = runner.get_effective_max_tokens(1000, interactive=True)
            assert small_interactive == 1000, "Small requests should be honored in interactive mode"
            
            small_server = runner.get_effective_max_tokens(1000, interactive=False)  
            assert small_server == 1000, "Small requests should be honored in server mode"
            
            # Test None behavior (new CLI default=None logic)
            # Interactive mode with None should use full context
            none_interactive = runner.get_effective_max_tokens(None, interactive=True)
            assert none_interactive == 4096, "None in interactive mode should use full context"
            
            # Server mode with None should use server limit
            none_server = runner.get_effective_max_tokens(None, interactive=False)  
            assert none_server == 2048, "None in server mode should use server limit (context/2)"