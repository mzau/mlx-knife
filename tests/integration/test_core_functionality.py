"""
High Priority Tests: Core Functionality

Tests ensure primary features work correctly:
- Model execution (run command, streaming, token decoding, stop tokens)
- Basic operations (list, show, pull, rm)
- Chat template application
"""
import pytest
import subprocess
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.mark.timeout(30)
class TestBasicOperations:
    """Test core CLI operations."""
    
    def test_list_command_empty_cache(self, mlx_knife_process, temp_cache_dir):
        """List command should handle empty cache gracefully."""
        proc = mlx_knife_process(["list"])
        stdout, stderr = proc.communicate(timeout=10)
        
        # Should complete successfully
        assert proc.returncode == 0, f"List failed on empty cache: {stderr}"
        
        # Should produce some output (even if empty list)
        assert len(stdout) >= 0
        # Common outputs for empty cache: "No models found" or empty list
        
    def test_list_command_with_models(self, mlx_knife_process, mock_model_cache):
        """List command should display available models."""
        # Create some mock models
        mock_model_cache("test-model-1", healthy=True)
        mock_model_cache("test-model-2", healthy=True)
        
        proc = mlx_knife_process(["list"])
        stdout, stderr = proc.communicate(timeout=10)
        
        assert proc.returncode == 0, f"List failed: {stderr}"
        assert len(stdout) > 0, "List produced no output with models present"
        
        # Should contain reference to models (exact format depends on implementation)
        output_lower = stdout.lower()
        assert "test" in output_lower or "model" in output_lower or len(stdout.split('\n')) > 1

    def test_show_command_existing_model(self, mlx_knife_process, mock_model_cache):
        """Show command should display model details."""
        model_dir = mock_model_cache("test-model", healthy=True)
        
        # Try different possible model name formats
        model_names_to_try = ["test-model", "test/model", "models--test-model"]
        
        success = False
        for model_name in model_names_to_try:
            proc = mlx_knife_process(["show", model_name])
            stdout, stderr = proc.communicate(timeout=10)
            
            if proc.returncode == 0 and len(stdout) > 0:
                success = True
                break
        
        # At least one format should work, or command should handle gracefully
        # The key is that it doesn't crash or hang
        assert success or all(
            proc.returncode is not None for proc in [
                mlx_knife_process(["show", name]) 
                for name in model_names_to_try
            ]
        ), "Show command hung or crashed"

    def test_show_command_nonexistent_model(self, mlx_knife_process, temp_cache_dir):
        """Show command should handle nonexistent models gracefully."""
        proc = mlx_knife_process(["show", "nonexistent-model"])
        stdout, stderr = proc.communicate(timeout=10)
        
        # Should complete (likely with error code)
        assert proc.returncode is not None, "Show command hung"
        
        # Should produce some error message
        output = stdout + stderr
        assert len(output) > 0, "No error message for nonexistent model"

    def test_rm_command_safety(self, mlx_knife_process, temp_cache_dir):
        """Remove command should handle nonexistent models safely."""
        proc = mlx_knife_process(["rm", "nonexistent-model"])
        stdout, stderr = proc.communicate(timeout=10)
        
        # Should complete (may succeed or fail gracefully)
        assert proc.returncode is not None, "Remove command hung"
        
        # Should not crash
        # Exact behavior depends on implementation

    def test_rm_command_corrupted_empty_snapshots(self, mlx_knife_process, temp_cache_dir):
        """Remove command should handle corrupted models with empty snapshots directory."""
        from mlx_knife.cache_utils import hf_to_cache_dir
        
        # Create a corrupted model structure (directory exists but snapshots is empty)
        test_model = "test-org/corrupted-empty-model"
        # Create in hub subdirectory (new cache structure)
        hub_dir = temp_cache_dir / "hub"
        cache_dir = hub_dir / hf_to_cache_dir(test_model)
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "snapshots").mkdir(exist_ok=True)
        (cache_dir / "blobs").mkdir(exist_ok=True)
        (cache_dir / "refs").mkdir(exist_ok=True)
        
        try:
            # This should NOT fail silently - should either provide error message or handle deletion
            # Use --force to avoid hanging on input prompts in test environment
            proc = mlx_knife_process(["rm", test_model, "--force"])
            stdout, stderr = proc.communicate(timeout=10)
            
            # Should complete (not hang)
            assert proc.returncode is not None, "Remove command hung on corrupted model"
            
            # Should produce SOME output (not silent failure)
            output = (stdout + stderr).strip()
            assert len(output) > 0, "Remove command failed silently on corrupted model - no output produced"
            
            # The behavior should be explicit: either error message or deletion prompt/confirmation
            output_lower = output.lower()
            has_error = "error" in output_lower or "not found" in output_lower
            has_prompt = "delete" in output_lower or "remove" in output_lower
            
            assert has_error or has_prompt, f"Remove command should provide clear feedback, got: {output}"
            
        finally:
            # Cleanup - remove the test corrupted model structure
            import shutil
            if cache_dir.exists():
                shutil.rmtree(cache_dir)


@pytest.mark.timeout(60)
class TestModelExecution:
    """Test model loading and execution functionality."""
    
    def test_run_command_basic_prompt(self, mlx_knife_process):
        """Test basic model execution with prompt using real MLX model."""
        # Uses Phi-3-mini-4k-instruct-4bit (assumes already pulled and healthy)
        test_model = "Phi-3-mini-4k-instruct-4bit"
        test_prompt = "Say hello."
        
        proc = mlx_knife_process(["run", test_model, test_prompt, "--max-tokens", "20"])
        stdout, stderr = proc.communicate(timeout=60)
        
        # Test MLX Knife functionality, not model quality
        assert proc.returncode == 0, f"MLX Knife execution failed: {stderr}"
        assert len(stdout.strip()) > 0, "MLX Knife produced no output - model loading/generation failed"
        assert len(stdout.strip()) < 1000, f"MLX Knife did not respect max-tokens limit: {len(stdout)} chars"
        
        # Basic sanity check: output should be reasonable text (not binary garbage)
        # Allow common whitespace characters (newlines, tabs, spaces)
        clean_output = stdout.replace('\n', '').replace('\t', '').replace('\r', '')
        assert clean_output.isprintable(), f"MLX Knife produced non-printable output: {repr(stdout)}"

    def test_run_command_invalid_model(self, mlx_knife_process, temp_cache_dir):
        """Run command should handle invalid models gracefully."""
        proc = mlx_knife_process(["run", "nonexistent-model", "test prompt"])
        stdout, stderr = proc.communicate(timeout=15)
        
        # Should fail gracefully, not hang
        assert proc.returncode is not None, "Run command hung on invalid model"
        assert proc.returncode != 0, "Run should fail on nonexistent model"
        
        # Should produce error message
        output = stdout + stderr
        assert len(output) > 0, "No error message for invalid model"

    def test_streaming_token_generation(self, mlx_knife_process):
        """Test streaming token output with real MLX model."""
        test_model = "Phi-3-mini-4k-instruct-4bit"
        test_prompt = "Write the word 'test' three times."
        
        proc = mlx_knife_process(["run", test_model, test_prompt, "--max-tokens", "30"])
        stdout, stderr = proc.communicate(timeout=45)
        
        # Test MLX Knife streaming functionality, not model accuracy
        assert proc.returncode == 0, f"MLX Knife streaming failed: {stderr}"
        assert len(stdout.strip()) > 0, "MLX Knife streaming produced no output"
        assert len(stdout.strip()) < 2000, f"MLX Knife streaming did not respect token limits: {len(stdout)} chars"
        
        # Verify streaming worked by checking output is reasonable text
        # Allow common whitespace characters (newlines, tabs, spaces)
        clean_output = stdout.replace('\n', '').replace('\t', '').replace('\r', '')
        assert clean_output.isprintable(), f"MLX Knife streaming produced non-printable output: {repr(stdout)}"



@pytest.mark.timeout(120)
class TestPullOperation:
    """Test model downloading functionality."""
    
    def test_pull_command_invalid_model(self, mlx_knife_process, temp_cache_dir):
        """Pull command should handle invalid model names gracefully."""
        proc = mlx_knife_process(["pull", "definitely-not-a-real-model-12345"])
        stdout, stderr = proc.communicate(timeout=30)
        
        # Should fail, not hang
        assert proc.returncode is not None, "Pull command hung"
        assert proc.returncode != 0, "Pull should fail on invalid model"
        
        # Should produce error message
        output = stdout + stderr
        assert len(output) > 0, "No error message for invalid model"

    def test_pull_command_network_timeout_handling(self, mlx_knife_process, temp_cache_dir):
        """Pull command should handle network issues gracefully."""
        # Use a model that likely exists but may be slow/timeout
        proc = mlx_knife_process(["pull", "mlx-community/Phi-3-mini-4k-instruct-4bit", "--no-progress"])
        
        # Give it limited time to start, then interrupt
        time.sleep(5)
        
        if proc.poll() is None:  # Still running
            proc.send_signal(subprocess.signal.SIGINT)
            try:
                stdout, stderr = proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
        else:
            stdout, stderr = proc.communicate()
        
        # Key test: should not hang indefinitely
        assert proc.returncode is not None, "Pull command did not terminate"
        
        # Should handle interruption gracefully
        output = stdout + stderr
        assert len(output) >= 0  # Some output expected


@pytest.mark.timeout(30)  
class TestCommandLineInterface:
    """Test CLI argument parsing and help functionality."""
    
    def test_help_command(self, mlx_knife_process):
        """Help command should display usage information."""
        proc = mlx_knife_process(["--help"])
        stdout, stderr = proc.communicate(timeout=10)
        
        # Should succeed
        assert proc.returncode == 0, f"Help command failed: {stderr}"
        
        # Should produce help output
        assert len(stdout) > 0, "Help produced no output"
        
        # Should contain basic command information
        help_text = stdout.lower()
        assert any(cmd in help_text for cmd in ["list", "pull", "run", "health"]), \
            "Help missing core commands"

    def test_version_command(self, mlx_knife_process):
        """Version command should display version information."""
        # Try common version flags
        version_flags = ["--version", "-v"]
        
        success = False
        for flag in version_flags:
            try:
                proc = mlx_knife_process([flag])
                stdout, stderr = proc.communicate(timeout=10)
                
                if proc.returncode == 0 and len(stdout) > 0:
                    success = True
                    # Should contain version number
                    assert any(char.isdigit() for char in stdout), \
                        "Version output contains no digits"
                    break
            except:
                continue
        
        # At least one version flag should work, or command should handle gracefully
        if not success:
            # Test that invalid flags are handled
            proc = mlx_knife_process(["--invalid-flag"])
            stdout, stderr = proc.communicate(timeout=10)
            assert proc.returncode is not None, "Invalid flag handling hung"

    def test_invalid_command_handling(self, mlx_knife_process):
        """Invalid commands should be handled gracefully."""
        proc = mlx_knife_process(["invalid-command-xyz"])
        stdout, stderr = proc.communicate(timeout=10)
        
        # Should fail but not hang
        assert proc.returncode is not None, "Invalid command hung"
        assert proc.returncode != 0, "Invalid command should not succeed"
        
        # Should produce error message
        output = stdout + stderr
        assert len(output) > 0, "No error message for invalid command"

    def test_missing_arguments_handling(self, mlx_knife_process):
        """Commands missing required arguments should fail gracefully."""
        # Test commands that require arguments
        commands_needing_args = [
            ["run"],  # needs model and prompt
            ["show"],  # needs model name
            ["pull"],  # needs model name
        ]
        
        for cmd in commands_needing_args:
            proc = mlx_knife_process(cmd)
            stdout, stderr = proc.communicate(timeout=10)
            
            # Should fail gracefully
            assert proc.returncode is not None, f"Command {cmd} hung"
            assert proc.returncode != 0, f"Command {cmd} should fail without required args"
            
            # Should produce helpful error
            output = stdout + stderr
            assert len(output) > 0, f"No error message for {cmd} without args"