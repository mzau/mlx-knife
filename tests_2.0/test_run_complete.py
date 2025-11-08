"""
Complete run command functionality tests for Step 1.1/1.2.
Tests all run command scenarios as specified in 2.0-TEST-SPECIFICATIONS.md.
"""

import pytest
import tempfile
from unittest.mock import Mock, patch, call
from pathlib import Path
from io import StringIO
import sys

from mlxk2.operations.run import run_model, interactive_chat, single_shot_generation
from mlxk2.core.runner import MLXRunner


@pytest.fixture
def mock_runner_complete():
    """Complete mock runner for run command tests."""
    with patch('mlxk2.operations.run.MLXRunner') as mock_runner_class:
        mock_runner = Mock()
        mock_runner_class.return_value.__enter__.return_value = mock_runner
        mock_runner_class.return_value.__exit__.return_value = None
        
        # Mock generation methods
        mock_runner.generate_streaming.return_value = iter(["Hello", " ", "world", "!"])
        mock_runner.generate_batch.return_value = "Hello world!"
        mock_runner._format_conversation.return_value = "Formatted conversation"
        
        yield mock_runner


class TestRunBasic:
    """Basic run command functionality tests."""
    
    def test_run_single_shot_streaming(self, mock_runner_complete):
        """mlxk run model "prompt" - streaming mode"""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            result = run_model(
                model_spec="test-model",
                prompt="test prompt",
                stream=True,
                json_output=False
            )
        
        # Should have called generate_streaming
        mock_runner_complete.generate_streaming.assert_called_once()
        
        # Should print streaming output
        output = fake_out.getvalue()
        assert "Hello world!" in output
        
        # Non-JSON mode returns None
        assert result is None
    
    def test_run_single_shot_batch(self, mock_runner_complete):
        """mlxk run model "prompt" --no-stream - batch mode"""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            result = run_model(
                model_spec="test-model",
                prompt="test prompt",
                stream=False,
                json_output=False
            )
        
        # Should have called generate_batch
        mock_runner_complete.generate_batch.assert_called_once()
        
        # Should print batch output
        output = fake_out.getvalue()
        assert "Hello world!" in output
        
        # Non-JSON mode returns None
        assert result is None
    
    def test_run_single_shot_json_output(self, mock_runner_complete):
        """Test JSON output mode for single-shot"""
        result = run_model(
            model_spec="test-model",
            prompt="test prompt",
            stream=False,
            json_output=True
        )
        
        # Should return the generated text
        assert result == "Hello world!"
    
    def test_run_interactive_streaming(self, mock_runner_complete):
        """mlxk run model (no prompt) - interactive streaming mode"""
        # Mock user input
        with patch('builtins.input', side_effect=["hello", "quit"]):
            with patch('sys.stdout', new=StringIO()) as fake_out:
                result = run_model(
                    model_spec="test-model",
                    prompt=None,  # Interactive mode
                    stream=True,
                    json_output=False
                )
        
        # Should have called format_conversation and generate_streaming
        mock_runner_complete._format_conversation.assert_called()
        mock_runner_complete.generate_streaming.assert_called()
        
        # Should show interactive prompts
        output = fake_out.getvalue()
        assert "Starting interactive chat" in output
        assert "You:" in output or "Assistant:" in output
    
    def test_run_interactive_batch(self, mock_runner_complete):
        """mlxk run model --no-stream (no prompt) - interactive batch mode"""
        # Mock user input
        with patch('builtins.input', side_effect=["hello", "quit"]):
            with patch('sys.stdout', new=StringIO()) as fake_out:
                result = run_model(
                    model_spec="test-model",
                    prompt=None,  # Interactive mode
                    stream=False,
                    json_output=False
                )
        
        # Should have called format_conversation and generate_batch
        mock_runner_complete._format_conversation.assert_called()
        mock_runner_complete.generate_batch.assert_called()
    
    def test_run_interactive_json_incompatible(self, mock_runner_complete):
        """Interactive mode should not work with JSON output"""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            result = run_model(
                model_spec="test-model",
                prompt=None,  # Interactive mode
                json_output=True
            )
        
        output = fake_out.getvalue()
        assert "not compatible with JSON output" in output
        assert result is None


class TestRunParameters:
    """Test parameter passing and configuration."""
    
    def test_run_full_context_tokens(self, mock_runner_complete):
        """Test that run command uses full model context by default"""
        run_model(
            model_spec="test-model",
            prompt="test",
            max_tokens=None  # Should use dynamic (full context)
        )
        
        # Should call with None max_tokens (dynamic calculation)
        call_args = mock_runner_complete.generate_streaming.call_args
        assert call_args[1]['max_tokens'] is None
    
    def test_run_explicit_max_tokens(self, mock_runner_complete):
        """Test that explicit max_tokens is respected"""
        run_model(
            model_spec="test-model",
            prompt="test",
            max_tokens=500
        )
        
        # Should pass through explicit max_tokens
        call_args = mock_runner_complete.generate_streaming.call_args
        assert call_args[1]['max_tokens'] == 500
    
    def test_run_temperature_parameter(self, mock_runner_complete):
        """Test temperature parameter passing"""
        run_model(
            model_spec="test-model",
            prompt="test",
            temperature=0.9
        )
        
        call_args = mock_runner_complete.generate_streaming.call_args
        assert call_args[1]['temperature'] == 0.9
    
    def test_run_top_p_parameter(self, mock_runner_complete):
        """Test top_p parameter passing"""
        run_model(
            model_spec="test-model",
            prompt="test",
            top_p=0.95
        )
        
        call_args = mock_runner_complete.generate_streaming.call_args
        assert call_args[1]['top_p'] == 0.95
    
    def test_run_chat_template_control(self, mock_runner_complete):
        """Test chat template enable/disable"""
        # With chat template (default)
        run_model(
            model_spec="test-model",
            prompt="test",
            use_chat_template=True
        )
        
        call_args = mock_runner_complete.generate_streaming.call_args
        assert call_args[1]['use_chat_template'] is True
        
        # Without chat template
        run_model(
            model_spec="test-model",
            prompt="test",
            use_chat_template=False
        )
        
        call_args = mock_runner_complete.generate_streaming.call_args
        assert call_args[1]['use_chat_template'] is False


class TestConversationHistory:
    """Test conversation history tracking in interactive mode."""
    
    def test_conversation_history_accumulation(self, mock_runner_complete):
        """Test that conversation history accumulates properly"""
        conversation_calls = []
        
        def capture_conversation(messages):
            conversation_calls.append(messages.copy())
            return "Formatted conversation"
        
        mock_runner_complete._format_conversation.side_effect = capture_conversation
        
        # Simulate interactive conversation
        with patch('builtins.input', side_effect=["first message", "second message", "quit"]):
            with patch('sys.stdout', new=StringIO()):
                run_model(
                    model_spec="test-model",
                    prompt=None,  # Interactive mode
                    stream=True
                )
        
        # Should have multiple conversation calls with growing history
        assert len(conversation_calls) >= 2
        
        # First call: one user message
        assert len(conversation_calls[0]) == 1
        assert conversation_calls[0][0]["role"] == "user"
        assert conversation_calls[0][0]["content"] == "first message"
        
        # Second call: user + assistant + user
        assert len(conversation_calls[1]) == 3
        assert conversation_calls[1][0]["role"] == "user"
        assert conversation_calls[1][1]["role"] == "assistant"
        assert conversation_calls[1][2]["role"] == "user"
        assert conversation_calls[1][2]["content"] == "second message"
    
    def test_empty_input_handling(self, mock_runner_complete):
        """Test that empty input is ignored"""
        with patch('builtins.input', side_effect=["", "  ", "actual message", "quit"]):
            with patch('sys.stdout', new=StringIO()):
                run_model(
                    model_spec="test-model",
                    prompt=None,
                    stream=True
                )
        
        # Should only process the non-empty message
        conversation_calls = mock_runner_complete._format_conversation.call_args_list
        assert len(conversation_calls) == 1  # Only one actual message processed
        
        messages = conversation_calls[0][0][0]
        assert len(messages) == 1
        assert messages[0]["content"] == "actual message"


class TestChatTemplate:
    """Test chat template integration."""
    
    def test_chat_template_integration(self, mock_runner_complete):
        """Test that chat template is used for conversation formatting"""
        with patch('builtins.input', side_effect=["test message", "quit"]):
            with patch('sys.stdout', new=StringIO()):
                run_model(
                    model_spec="test-model",
                    prompt=None,
                    stream=True
                )
        
        # Should call _format_conversation with proper message structure
        mock_runner_complete._format_conversation.assert_called()
        call_args = mock_runner_complete._format_conversation.call_args[0][0]
        
        assert isinstance(call_args, list)
        assert len(call_args) == 1
        assert call_args[0]["role"] == "user"
        assert call_args[0]["content"] == "test message"
        
        # Should call generate_streaming with use_chat_template=False
        # (because template already applied in _format_conversation)
        gen_call_args = mock_runner_complete.generate_streaming.call_args
        assert gen_call_args[1]['use_chat_template'] is False


class TestErrorHandling:
    """Test error handling in run command."""
    
    def test_model_loading_error(self):
        """Test handling of model loading failures"""
        with patch('mlxk2.operations.run.MLXRunner') as mock_runner_class:
            mock_runner_class.side_effect = FileNotFoundError("Model not found")
            
            with patch('sys.stdout', new=StringIO()) as fake_out:
                result = run_model(
                    model_spec="nonexistent-model",
                    prompt="test",
                    json_output=False
                )
            
            output = fake_out.getvalue()
            assert "Error:" in output
            # Issue #38: run_model now returns error string in both text and JSON modes
            assert result is not None and result.startswith("Error:")
    
    def test_generation_error_json_mode(self):
        """Test error handling in JSON mode"""
        with patch('mlxk2.operations.run.MLXRunner') as mock_runner_class:
            mock_runner_class.side_effect = RuntimeError("Generation failed")
            
            result = run_model(
                model_spec="test-model",
                prompt="test",
                json_output=True
            )
            
            assert "Error:" in result
    
    def test_keyboard_interrupt_handling(self, mock_runner_complete):
        """Test Ctrl-C handling in interactive mode"""
        def simulate_interrupt(*args, **kwargs):
            raise KeyboardInterrupt()
        
        with patch('builtins.input', side_effect=simulate_interrupt):
            with patch('sys.stdout', new=StringIO()) as fake_out:
                result = run_model(
                    model_spec="test-model",
                    prompt=None,
                    stream=True
                )
            
            output = fake_out.getvalue()
            assert "interrupted" in output.lower() or "goodbye" in output.lower()


class TestStreamingVsBatch:
    """Test consistency between streaming and batch modes."""
    
    def test_streaming_vs_batch_output_consistency(self, mock_runner_complete):
        """Test that streaming and batch produce equivalent output"""
        # Configure mocks to return same content
        mock_runner_complete.generate_streaming.return_value = iter(["Hello", " ", "world"])
        mock_runner_complete.generate_batch.return_value = "Hello world"
        
        # Test streaming mode
        with patch('sys.stdout', new=StringIO()) as stream_out:
            run_model(
                model_spec="test-model",
                prompt="test",
                stream=True,
                json_output=False
            )
        
        # Test batch mode  
        with patch('sys.stdout', new=StringIO()) as batch_out:
            run_model(
                model_spec="test-model",
                prompt="test",
                stream=False,
                json_output=False
            )
        
        # Output should be equivalent (modulo formatting)
        stream_output = stream_out.getvalue().strip()
        batch_output = batch_out.getvalue().strip()

        # Both should contain the core content
        assert "Hello world" in stream_output
        assert "Hello world" in batch_output


class TestPreflightCompatibilityCheck:
    """Test runtime compatibility preflight checks in run command."""

    def test_commit_pinned_incompatible_model_blocked(self, isolated_cache):
        """Commit-pinned models must also pass compatibility check (regression test).

        Regression: Beta.5 introduced preflight compatibility checks, but commit-pinned
        models bypassed the check due to incorrect if/else scoping.

        This test verifies that `mlxk run org/model@commit_hash` properly validates
        framework compatibility before attempting to load the model.
        """
        import json
        from unittest.mock import patch

        # Create a PyTorch model in cache with specific commit hash
        commit_hash = "abc123def456"
        model_name = "test-org/pytorch-model"
        cache_dir = isolated_cache / f"models--{model_name.replace('/', '--')}"
        snapshot_dir = cache_dir / "snapshots" / commit_hash
        snapshot_dir.mkdir(parents=True)

        # Create valid config.json (healthy model)
        config = {"model_type": "bert", "architectures": ["BertForSequenceClassification"]}
        (snapshot_dir / "config.json").write_text(json.dumps(config))

        # Create PyTorch weights (incompatible framework)
        (snapshot_dir / "pytorch_model.bin").write_bytes(b"fake_pytorch_weights" * 100)

        # Mock resolve_model_for_operation to return our commit hash
        with patch('mlxk2.operations.run.resolve_model_for_operation') as mock_resolve:
            mock_resolve.return_value = (model_name, commit_hash, None)

            # Mock get_current_model_cache to use our isolated cache
            with patch('mlxk2.operations.run.get_current_model_cache') as mock_cache:
                mock_cache.return_value = isolated_cache

                # Attempt to run with commit-pinned spec
                result = run_model(
                    model_spec=f"{model_name}@{commit_hash}",
                    prompt="test prompt",
                    json_output=True
                )

        # Should be blocked by preflight check
        assert result is not None
        assert "Error:" in result
        assert "not compatible" in result or "Incompatible" in result

    def test_latest_snapshot_incompatible_model_blocked(self, isolated_cache):
        """Non-pinned models should also be blocked by compatibility check."""
        import json
        from unittest.mock import patch

        # Create a PyTorch model in cache (latest snapshot)
        model_name = "test-org/another-pytorch"
        cache_dir = isolated_cache / f"models--{model_name.replace('/', '--')}"
        snapshot_dir = cache_dir / "snapshots" / "latest_snapshot"
        snapshot_dir.mkdir(parents=True)

        # Create valid config.json (healthy model)
        config = {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}
        (snapshot_dir / "config.json").write_text(json.dumps(config))

        # Create PyTorch weights (incompatible framework)
        (snapshot_dir / "pytorch_model.bin").write_bytes(b"fake_weights" * 100)

        # Mock resolve_model_for_operation (no commit hash)
        with patch('mlxk2.operations.run.resolve_model_for_operation') as mock_resolve:
            mock_resolve.return_value = (model_name, None, None)

            with patch('mlxk2.operations.run.get_current_model_cache') as mock_cache:
                mock_cache.return_value = isolated_cache

                result = run_model(
                    model_spec=model_name,
                    prompt="test prompt",
                    json_output=True
                )

        # Should be blocked by preflight check
        assert result is not None
        assert "Error:" in result
        assert "not compatible" in result or "Incompatible" in result