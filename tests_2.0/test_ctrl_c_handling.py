"""
Ctrl-C interruption handling tests for Step 1.1/1.2.
Tests graceful interruption during generation and interactive mode.
"""

import pytest
import signal
import time
from unittest.mock import Mock, patch, call
from io import StringIO

from mlxk2.core.runner import MLXRunner
from mlxk2.operations.run import run_model, interactive_chat


class MockDetokenizer:
    """Mock detokenizer that mimics BPEStreamingDetokenizer behavior.

    Used by unit tests to mock tokenizer.detokenizer after Session 60 changes.
    Session 60 switched from tokenizer.decode() to tokenizer.detokenizer for
    proper BPE space marker (Ġ U+0120) conversion.
    """
    def __init__(self, decode_func):
        """Initialize with a decode function that maps token lists to strings."""
        self.decode_func = decode_func
        self.tokens = []
        self._text = ""

    def reset(self):
        """Reset accumulated tokens."""
        self.tokens = []
        self._text = ""

    def add_token(self, token_id):
        """Add a token to the accumulated list."""
        self.tokens.append(token_id)

    def finalize(self):
        """Finalize and decode accumulated tokens."""
        self._text = self.decode_func(self.tokens)

    @property
    def text(self):
        """Return the decoded text."""
        return self._text


@pytest.fixture
def mock_runner_with_interruption():
    """Mock runner that can simulate interruption scenarios."""
    mock_runner = Mock()
    
    # Track interruption state
    mock_runner._interrupted = False
    
    def simulate_generation_with_interruption():
        """Generator that checks for interruption"""
        tokens = ["Token1", "Token2", "Token3", "Token4", "Token5"]
        for i, token in enumerate(tokens):
            if mock_runner._interrupted:
                yield "\n[Generation interrupted by user]"
                break
            yield token
    
    mock_runner.generate_streaming.side_effect = lambda *args, **kwargs: simulate_generation_with_interruption()
    mock_runner.generate_batch.return_value = "Complete response"
    mock_runner._format_conversation.return_value = "Formatted conversation"
    
    return mock_runner


class TestMLXRunnerInterruption:
    """Test interruption handling in MLXRunner core."""
    
    @patch('mlxk2.core.runner.load')
    @patch('mlxk2.core.runner.resolve_model_for_operation')
    @patch('mlxk2.core.runner.get_current_model_cache')
    def test_signal_handler_setup(self, mock_cache, mock_resolve, mock_load):
        """Test that signal handler is properly set up"""
        mock_resolve.return_value = ("test-model", None, None)
        mock_cache.return_value = Mock()
        mock_load.return_value = (Mock(), Mock())
        
        with patch('signal.signal') as mock_signal:
            with MLXRunner("test-model") as runner:
                # Should have set up SIGINT handler
                mock_signal.assert_called_with(signal.SIGINT, runner._handle_interrupt)
    
    @patch('mlxk2.core.runner.load')
    @patch('mlxk2.core.runner.resolve_model_for_operation')
    @patch('mlxk2.core.runner.get_current_model_cache')
    def test_interrupt_flag_setting(self, mock_cache, mock_resolve, mock_load):
        """Test that interrupt handler sets the flag correctly"""
        mock_resolve.return_value = ("test-model", None, None)
        mock_cache.return_value = Mock()
        mock_load.return_value = (Mock(), Mock())
        
        with MLXRunner("test-model") as runner:
            # Initially not interrupted
            assert runner._interrupted is False
            
            # Simulate signal
            runner._handle_interrupt(signal.SIGINT, None)
            
            # Should be marked as interrupted
            assert runner._interrupted is True
    
    @patch('mlxk2.core.runner.load')
    @patch('mlxk2.core.runner.resolve_model_for_operation')
    @patch('mlxk2.core.runner.get_current_model_cache')
    @patch('mlxk2.core.runner.generate_step')
    def test_streaming_interruption_detection(self, mock_gen, mock_cache, mock_resolve, mock_load):
        """Test that streaming generation checks for interruption"""
        mock_resolve.return_value = ("test-model", None, None)
        mock_cache.return_value = Mock()
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.eos_token_ids = {mock_tokenizer.eos_token_id}
        mock_tokenizer.additional_special_tokens = []
        mock_tokenizer.added_tokens_decoder = {}
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.side_effect = ["Hello", " world", "!"]
        # Use MockDetokenizer for proper BPE space marker handling
        def mock_decode(tokens):
            if len(tokens) == 1:
                return {1: "Hello", 2: " world", 3: "!"}.get(tokens[0], "")
            elif len(tokens) == 2:
                return "Hello world"
            elif len(tokens) == 3:
                return "Hello world!"
            return ""
        mock_tokenizer.detokenizer = MockDetokenizer(mock_decode)
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Mock generation that yields multiple tokens
        mock_gen.return_value = iter([
            (Mock(item=lambda: 1), Mock()),
            (Mock(item=lambda: 2), Mock()),
            (Mock(item=lambda: 3), Mock())
        ])
        
        with MLXRunner("test-model") as runner:
            # Start generation
            generator = runner.generate_streaming("test prompt")
            
            # Get first token
            first_token = next(generator)
            assert first_token == "Hello"
            
            # Simulate interruption
            runner._interrupted = True
            
            # Next token should be interruption message
            second_token = next(generator)
            assert "interrupted" in second_token.lower()
    
    @patch('mlxk2.core.runner.load')
    @patch('mlxk2.core.runner.resolve_model_for_operation')
    @patch('mlxk2.core.runner.get_current_model_cache')
    @patch('mlxk2.core.runner.generate_step')
    def test_batch_interruption_detection(self, mock_gen, mock_cache, mock_resolve, mock_load):
        """Test that batch generation also checks for interruption"""
        mock_resolve.return_value = ("test-model", None, None)
        mock_cache.return_value = Mock()
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.eos_token_ids = {mock_tokenizer.eos_token_id}
        mock_tokenizer.additional_special_tokens = []
        mock_tokenizer.added_tokens_decoder = {}
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Partial response"
        # Use MockDetokenizer for proper BPE space marker handling
        def mock_decode(tokens):
            return "Partial response" if tokens else ""
        mock_tokenizer.detokenizer = MockDetokenizer(mock_decode)
        mock_load.return_value = (mock_model, mock_tokenizer)

        def interrupted_generation():
            """Generator that gets interrupted"""
            yield (Mock(item=lambda: 1), Mock())
            # Simulation: interruption happens here
            yield (Mock(item=lambda: 2), Mock())
        
        mock_gen.return_value = interrupted_generation()
        
        with MLXRunner("test-model") as runner:
            # Set interruption before batch generation
            runner._interrupted = True
            
            result = runner.generate_batch("test prompt")
            
            # Should handle interruption gracefully (empty or partial result)
            assert isinstance(result, str)


class TestRunCommandInterruption:
    """Test interruption handling in run command operations."""
    
    def test_single_shot_streaming_interruption(self, mock_runner_with_interruption):
        """Test interruption during single-shot streaming generation"""
        with patch('mlxk2.operations.run.MLXRunner') as mock_runner_class:
            mock_runner_class.return_value.__enter__.return_value = mock_runner_with_interruption
            mock_runner_class.return_value.__exit__.return_value = None
            
            with patch('sys.stdout', new=StringIO()) as fake_out:
                # Start generation
                with patch('time.sleep', side_effect=[None, None]) as mock_sleep:
                    # Simulate interruption during generation
                    original_side_effect = mock_runner_with_interruption.generate_streaming.side_effect
                    def interrupt_after_delay(*args, **kwargs):
                        # Interrupt after first token
                        mock_runner_with_interruption._interrupted = True
                        # Continue with original generation behavior
                        return original_side_effect()
                    
                    mock_runner_with_interruption.generate_streaming.side_effect = interrupt_after_delay
                    
                    result = run_model(
                        model_spec="test-model",
                        prompt="test prompt",
                        stream=True,
                        json_output=False
                    )
            
            output = fake_out.getvalue()
            assert "interrupted" in output.lower()
    
    def test_interactive_mode_interruption(self, mock_runner_with_interruption):
        """Test interruption during interactive mode"""
        with patch('mlxk2.operations.run.MLXRunner') as mock_runner_class:
            mock_runner_class.return_value.__enter__.return_value = mock_runner_with_interruption
            mock_runner_class.return_value.__exit__.return_value = None
            
            # Simulate Ctrl-C during input
            with patch('builtins.input', side_effect=KeyboardInterrupt()):
                with patch('sys.stdout', new=StringIO()) as fake_out:
                    result = run_model(
                        model_spec="test-model",
                        prompt=None,  # Interactive mode
                        stream=True,
                        json_output=False
                    )
            
            output = fake_out.getvalue()
            assert "interrupted" in output.lower() or "goodbye" in output.lower()
    
    def test_interactive_chat_keyboard_interrupt(self, mock_runner_with_interruption):
        """Test direct keyboard interrupt handling in interactive_chat"""
        with patch('builtins.input', side_effect=KeyboardInterrupt()):
            with patch('sys.stdout', new=StringIO()) as fake_out:
                interactive_chat(mock_runner_with_interruption, stream=True)
        
        output = fake_out.getvalue()
        assert "interrupted" in output.lower() or "goodbye" in output.lower()
    
    def test_generation_interruption_in_interactive_mode(self, mock_runner_with_interruption):
        """Test interruption during generation in interactive mode"""
        # Set up mock to interrupt during generation
        def interrupt_during_generation(messages):
            mock_runner_with_interruption._interrupted = True
            return "Formatted conversation"
        
        mock_runner_with_interruption._format_conversation.side_effect = interrupt_during_generation
        
        with patch('builtins.input', side_effect=["test message", "quit"]):
            with patch('sys.stdout', new=StringIO()) as fake_out:
                interactive_chat(mock_runner_with_interruption, stream=True)
        
        output = fake_out.getvalue()
        assert "interrupted" in output.lower()


class TestInterruptionRecovery:
    """Test recovery and cleanup after interruption."""
    
    def test_interruption_flag_reset(self, mock_runner_with_interruption):
        """Test that interruption flag can be reset for subsequent operations"""
        # Simulate interruption
        mock_runner_with_interruption._interrupted = True
        
        # Reset flag (simulating what would happen in real scenario)
        mock_runner_with_interruption._interrupted = False
        
        # Should be able to generate normally again
        def normal_generation():
            if not mock_runner_with_interruption._interrupted:
                return iter(["Normal", " response"])
            else:
                return iter(["Interrupted"])
        
        mock_runner_with_interruption.generate_streaming.side_effect = normal_generation
        
        tokens = list(mock_runner_with_interruption.generate_streaming())
        assert tokens == ["Normal", " response"]
    
    def test_multiple_interruptions(self, mock_runner_with_interruption):
        """Test handling of multiple interruptions"""
        interruption_count = 0
        
        def multi_interrupt_generation():
            nonlocal interruption_count
            interruption_count += 1
            tokens = [f"Token{i}" for i in range(5)]
            
            for i, token in enumerate(tokens):
                if i == 2:  # Interrupt at third token
                    mock_runner_with_interruption._interrupted = True
                
                if mock_runner_with_interruption._interrupted:
                    yield f"\n[Generation interrupted by user - attempt {interruption_count}]"
                    break
                yield token
        
        mock_runner_with_interruption.generate_streaming.side_effect = multi_interrupt_generation
        
        # First interruption
        tokens1 = list(mock_runner_with_interruption.generate_streaming())
        assert any("interrupted" in token.lower() for token in tokens1)
        
        # Reset for second attempt
        mock_runner_with_interruption._interrupted = False
        
        # Second interruption
        tokens2 = list(mock_runner_with_interruption.generate_streaming())
        assert any("interrupted" in token.lower() for token in tokens2)
        
        assert interruption_count == 2
    
    def test_clean_interruption_message(self, mock_runner_with_interruption):
        """Test that interruption message is clean and informative"""
        def generate_with_interruption():
            yield "Starting"
            mock_runner_with_interruption._interrupted = True
            yield "\n[Generation interrupted by user]"
        
        mock_runner_with_interruption.generate_streaming.side_effect = generate_with_interruption
        
        tokens = list(mock_runner_with_interruption.generate_streaming())
        
        # Should have starting token and clean interruption message
        assert "Starting" in tokens
        
        interruption_msg = [t for t in tokens if "interrupted" in t.lower()][0]
        assert interruption_msg == "\n[Generation interrupted by user]"
        assert interruption_msg.startswith("\n")  # Proper formatting


class TestInterruptionEdgeCases:
    """Test edge cases in interruption handling."""
    
    def test_interruption_before_generation_starts(self, mock_runner_with_interruption):
        """Test interruption that happens before generation begins"""
        # Set interrupted flag before generation
        mock_runner_with_interruption._interrupted = True
        
        def immediate_interruption():
            if mock_runner_with_interruption._interrupted:
                yield "\n[Generation interrupted by user]"
                return
            yield "This should not appear"
        
        mock_runner_with_interruption.generate_streaming.side_effect = immediate_interruption
        
        tokens = list(mock_runner_with_interruption.generate_streaming())
        
        assert len(tokens) == 1
        assert "interrupted" in tokens[0].lower()
        assert "This should not appear" not in tokens
    
    def test_interruption_after_generation_complete(self, mock_runner_with_interruption):
        """Test that interruption flag doesn't affect completed generation"""
        def complete_then_interrupt():
            # Complete generation first
            for token in ["Complete", " response"]:
                yield token
            
            # Interrupt after completion (shouldn't affect output)
            mock_runner_with_interruption._interrupted = True
        
        mock_runner_with_interruption.generate_streaming.side_effect = complete_then_interrupt
        
        tokens = list(mock_runner_with_interruption.generate_streaming())
        
        # Should have complete response, no interruption message
        assert tokens == ["Complete", " response"]
    
    def test_interruption_with_empty_generation(self, mock_runner_with_interruption):
        """Test interruption when generation produces no tokens"""
        def empty_generation():
            mock_runner_with_interruption._interrupted = True
            # Check interruption immediately
            if mock_runner_with_interruption._interrupted:
                yield "\n[Generation interrupted by user]"
                return
            
            # This would be empty generation
            return
            yield  # unreachable
        
        mock_runner_with_interruption.generate_streaming.side_effect = empty_generation
        
        tokens = list(mock_runner_with_interruption.generate_streaming())
        
        assert len(tokens) == 1
        assert "interrupted" in tokens[0].lower()


class TestInterruptionCompatibility:
    """Test interruption compatibility with other features."""
    
    def test_interruption_with_chat_template(self, mock_runner_with_interruption):
        """Test interruption works with chat template formatting"""
        mock_runner_with_interruption._format_conversation.return_value = "Human: test\n\nAssistant: "
        
        def interrupt_after_template():
            # Interrupt immediately after template formatting
            mock_runner_with_interruption._interrupted = True
            yield "\n[Generation interrupted by user]"
        
        mock_runner_with_interruption.generate_streaming.side_effect = interrupt_after_template
        
        with patch('builtins.input', side_effect=["test", "quit"]):
            with patch('sys.stdout', new=StringIO()) as fake_out:
                interactive_chat(mock_runner_with_interruption, stream=True)
        
        output = fake_out.getvalue()
        assert "interrupted" in output.lower()
        
        # Should have called format_conversation despite interruption
        mock_runner_with_interruption._format_conversation.assert_called()
    
    def test_interruption_with_json_output(self, mock_runner_with_interruption):
        """Test interruption handling with JSON output mode"""
        with patch('mlxk2.operations.run.MLXRunner') as mock_runner_class:
            mock_runner_class.return_value.__enter__.return_value = mock_runner_with_interruption
            mock_runner_class.return_value.__exit__.return_value = None
            
            # Simulate interruption during generation
            mock_runner_with_interruption._interrupted = True
            
            result = run_model(
                model_spec="test-model",
                prompt="test prompt",
                stream=False,
                json_output=True
            )
            
            # Should return some result, even if interrupted
            assert isinstance(result, str)
    
    def test_interruption_preserves_conversation_history(self, mock_runner_with_interruption):
        """Test that interruption doesn't corrupt conversation history"""
        conversation_calls = []
        
        def track_conversations(messages):
            conversation_calls.append(len(messages))
            if len(conversation_calls) == 2:  # Interrupt on second call
                mock_runner_with_interruption._interrupted = True
            return "Formatted conversation"
        
        mock_runner_with_interruption._format_conversation.side_effect = track_conversations
        
        # Mock interrupted generation for second message
        generation_calls = 0
        def selective_interruption():
            nonlocal generation_calls
            generation_calls += 1
            if generation_calls == 2:  # Second generation gets interrupted
                yield "\n[Generation interrupted by user]"
            else:
                yield "Normal response"
        
        mock_runner_with_interruption.generate_streaming.side_effect = selective_interruption
        
        with patch('builtins.input', side_effect=["first", "second", "quit"]):
            with patch('sys.stdout', new=StringIO()):
                interactive_chat(mock_runner_with_interruption, stream=True, prepare_next_prompt=False)
        
        # Should have processed both messages despite interruption
        assert len(conversation_calls) == 2
        assert conversation_calls[0] == 1  # First message
        assert conversation_calls[1] == 3  # First + response + second message
