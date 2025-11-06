"""
Test for interruption recovery bug fix.
Ensures that after Ctrl-C, subsequent generations work normally.
"""

import pytest
from unittest.mock import Mock, patch
from io import StringIO

from mlxk2.core.runner import MLXRunner
from mlxk2.operations.run import interactive_chat


class TestInterruptionRecovery:
    """Test recovery after interruption in interactive mode."""
    
    @patch('mlxk2.core.runner.load')
    @patch('mlxk2.core.runner.resolve_model_for_operation')
    @patch('mlxk2.core.cache.get_current_model_cache')
    def test_interruption_flag_reset_streaming(self, mock_cache, mock_resolve, mock_load):
        """Test that interruption flag is reset for new streaming generation"""
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
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        with patch('mlxk2.core.runner.generate_step') as mock_gen:
            # Mock generation that yields tokens
            mock_gen.return_value = iter([
                (Mock(item=lambda: 1), Mock()),
                (Mock(item=lambda: 2), Mock())
            ])
            mock_tokenizer.decode.side_effect = ["Hello", " world"]
            
            with MLXRunner("test-model") as runner:
                # Simulate interruption
                runner._interrupted = True
                assert runner._interrupted is True
                
                # Start new generation - should reset flag
                tokens = list(runner.generate_streaming("test prompt"))
                
                # Flag should be reset at start of generation
                assert runner._interrupted is False
                assert tokens == ["Hello", " world"]
    
    @patch('mlxk2.core.runner.load')
    @patch('mlxk2.core.runner.resolve_model_for_operation')
    @patch('mlxk2.core.cache.get_current_model_cache')
    def test_interruption_flag_reset_batch(self, mock_cache, mock_resolve, mock_load):
        """Test that interruption flag is reset for new batch generation"""
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
        mock_tokenizer.decode.return_value = "Hello world"
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        with patch('mlxk2.core.runner.generate_step') as mock_gen:
            mock_gen.return_value = iter([
                (Mock(item=lambda: 1), Mock()),
                (Mock(item=lambda: 2), Mock())
            ])
            
            with MLXRunner("test-model") as runner:
                # Simulate interruption
                runner._interrupted = True
                assert runner._interrupted is True
                
                # Start new generation - should reset flag
                result = runner.generate_batch("test prompt")
                
                # Flag should be reset at start of generation
                assert runner._interrupted is False
                assert result == "Hello world"
    
    def test_interactive_mode_recovery_after_interruption(self):
        """Test that interactive mode works after interruption"""
        mock_runner = Mock()
        
        # Track interruption state and generation calls
        generation_calls = []
        
        def mock_generation(prompt, **kwargs):
            generation_calls.append(len(generation_calls))
            if len(generation_calls) == 1:
                # First call: simulate interruption
                mock_runner._interrupted = True
                return iter(["\n[Generation interrupted by user]"])
            else:
                # Subsequent calls: normal generation
                mock_runner._interrupted = False
                return iter(["Normal", " response"])
        
        mock_runner.generate_streaming.side_effect = mock_generation
        mock_runner._format_conversation.return_value = "Formatted conversation"
        
        # Simulate user input: first prompt gets interrupted, second works normally
        inputs = ["first prompt", "second prompt", "quit"]
        
        with patch('builtins.input', side_effect=inputs):
            with patch('sys.stdout', new=StringIO()) as fake_out:
                interactive_chat(mock_runner, stream=True)
        
        output = fake_out.getvalue()
        
        # Should show interruption for first, normal response for second
        assert "interrupted" in output.lower()
        assert "Normal response" in output
        
        # Should have made two generation calls
        assert len(generation_calls) == 2
    
    def test_multiple_interruptions_and_recoveries(self):
        """Test multiple cycles of interruption and recovery"""
        mock_runner = Mock()
        
        generation_calls = []
        
        def mock_generation(prompt, **kwargs):
            call_num = len(generation_calls)
            generation_calls.append(call_num)
            
            # Interrupt every other call
            if call_num % 2 == 0:
                mock_runner._interrupted = True
                return iter(["\n[Generation interrupted by user]"])
            else:
                mock_runner._interrupted = False
                return iter([f"Response {call_num}"])
        
        mock_runner.generate_streaming.side_effect = mock_generation
        mock_runner._format_conversation.return_value = "Formatted conversation"
        
        # Multiple prompts with alternating interruptions
        inputs = ["prompt1", "prompt2", "prompt3", "prompt4", "quit"]
        
        with patch('builtins.input', side_effect=inputs):
            with patch('sys.stdout', new=StringIO()) as fake_out:
                interactive_chat(mock_runner, stream=True)
        
        output = fake_out.getvalue()
        
        # Should show interruptions and normal responses
        assert "interrupted" in output.lower()
        assert "Response 1" in output
        assert "Response 3" in output
        
        # Should have made four generation calls
        assert len(generation_calls) == 4
    
    def test_interruption_does_not_affect_conversation_history(self):
        """Test that interruption doesn't corrupt conversation history"""
        mock_runner = Mock()
        
        conversation_calls = []
        
        def track_conversation(messages):
            conversation_calls.append([msg.copy() for msg in messages])
            return "Formatted conversation"
        
        mock_runner._format_conversation.side_effect = track_conversation
        
        # First generation gets interrupted, second succeeds
        generation_calls = []
        def mock_generation(prompt, **kwargs):
            call_num = len(generation_calls)
            generation_calls.append(call_num)
            
            if call_num == 0:
                # First call: interrupted
                return iter(["\n[Generation interrupted by user]"])
            else:
                # Second call: normal
                return iter(["Normal response"])
        
        mock_runner.generate_streaming.side_effect = mock_generation
        
        inputs = ["first prompt", "second prompt", "quit"]
        
        with patch('builtins.input', side_effect=inputs):
            with patch('sys.stdout', new=StringIO()):
                interactive_chat(mock_runner, stream=True)
        
        # Should have proper conversation progression
        assert len(conversation_calls) == 2
        
        # First conversation: just user message
        assert len(conversation_calls[0]) == 1
        assert conversation_calls[0][0]["content"] == "first prompt"
        
        # Second conversation: user + interrupted response + new user message
        assert len(conversation_calls[1]) == 3
        assert conversation_calls[1][0]["content"] == "first prompt"
        assert conversation_calls[1][1]["content"] == "[Generation interrupted by user]"
        assert conversation_calls[1][2]["content"] == "second prompt"
