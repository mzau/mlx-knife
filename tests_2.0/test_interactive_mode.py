"""
Interactive mode and conversation history tests for Step 1.1/1.2.
Tests conversation tracking and chat template integration.
"""

import pytest
from unittest.mock import Mock, patch
from io import StringIO

from mlxk2.operations.run import interactive_chat
from mlxk2.core.runner import MLXRunner


@pytest.fixture
def mock_runner_interactive():
    """Mock runner specifically for interactive mode tests."""
    mock_runner = Mock()
    
    # Mock conversation formatting
    def format_conversation(messages):
        """Mock chat template application"""
        if not messages:
            return ""
        
        # Simulate actual chat template behavior
        formatted_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted_parts.append(f"Human: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(formatted_parts) + "\n\nAssistant: "
    
    mock_runner._format_conversation.side_effect = format_conversation
    
    # Mock generation methods
    mock_runner.generate_streaming.return_value = iter(["Generated", " response"])
    mock_runner.generate_batch.return_value = "Generated response"
    
    return mock_runner


class TestInteractiveBasic:
    """Basic interactive mode functionality."""
    
    def test_interactive_startup_message(self, mock_runner_interactive):
        """Test that interactive mode shows startup message"""
        with patch('builtins.input', side_effect=["quit"]):
            with patch('sys.stdout', new=StringIO()) as fake_out:
                interactive_chat(mock_runner_interactive)
        
        output = fake_out.getvalue()
        assert "Starting interactive chat" in output
        assert "Type 'exit' or 'quit' to end" in output
    
    def test_interactive_exit_commands(self, mock_runner_interactive):
        """Test various exit commands work"""
        exit_commands = ["exit", "quit", "q"]
        
        for exit_cmd in exit_commands:
            with patch('builtins.input', side_effect=[exit_cmd]):
                with patch('sys.stdout', new=StringIO()) as fake_out:
                    interactive_chat(mock_runner_interactive)
            
            output = fake_out.getvalue()
            assert "Goodbye!" in output
    
    def test_interactive_streaming_mode(self, mock_runner_interactive):
        """Test interactive mode with streaming enabled"""
        with patch('builtins.input', side_effect=["test message", "quit"]):
            with patch('sys.stdout', new=StringIO()) as fake_out:
                interactive_chat(mock_runner_interactive, stream=True)
        
        # Should call generate_streaming
        mock_runner_interactive.generate_streaming.assert_called()
        
        # Should not call generate_batch
        mock_runner_interactive.generate_batch.assert_not_called()
        
        output = fake_out.getvalue()
        assert "Generated response" in output
    
    def test_interactive_batch_mode(self, mock_runner_interactive):
        """Test interactive mode with streaming disabled"""
        with patch('builtins.input', side_effect=["test message", "quit"]):
            with patch('sys.stdout', new=StringIO()) as fake_out:
                interactive_chat(mock_runner_interactive, stream=False)
        
        # Should call generate_batch
        mock_runner_interactive.generate_batch.assert_called()
        
        # Should not call generate_streaming
        mock_runner_interactive.generate_streaming.assert_not_called()
        
        output = fake_out.getvalue()
        assert "Generated response" in output


class TestConversationHistory:
    """Test conversation history tracking and management."""
    
    def test_conversation_history_accumulation(self, mock_runner_interactive):
        """Test that conversation history grows correctly"""
        conversation_history = []
        
        def capture_conversation(messages):
            conversation_history.append(messages.copy())
            return f"Formatted: {len(messages)} messages"
        
        mock_runner_interactive._format_conversation.side_effect = capture_conversation
        
        inputs = ["first message", "second message", "third message", "quit"]
        
        with patch('builtins.input', side_effect=inputs):
            with patch('sys.stdout', new=StringIO()):
                interactive_chat(mock_runner_interactive, stream=True)
        
        # Should have captured multiple conversation states
        assert len(conversation_history) == 3
        
        # First conversation: 1 user message
        assert len(conversation_history[0]) == 1
        assert conversation_history[0][0]["role"] == "user"
        assert conversation_history[0][0]["content"] == "first message"
        
        # Second conversation: user + assistant + user
        assert len(conversation_history[1]) == 3
        assert conversation_history[1][0]["role"] == "user"
        assert conversation_history[1][0]["content"] == "first message"
        assert conversation_history[1][1]["role"] == "assistant"
        assert conversation_history[1][1]["content"] == "Generated response"
        assert conversation_history[1][2]["role"] == "user"
        assert conversation_history[1][2]["content"] == "second message"
        
        # Third conversation: full history
        assert len(conversation_history[2]) == 5
        assert conversation_history[2][4]["content"] == "third message"
    
    def test_conversation_message_roles(self, mock_runner_interactive):
        """Test that message roles are correctly assigned"""
        captured_messages = []
        
        def capture_messages(messages):
            if messages:
                captured_messages.extend(messages)
            return "Formatted conversation"
        
        mock_runner_interactive._format_conversation.side_effect = capture_messages
        
        with patch('builtins.input', side_effect=["user input", "quit"]):
            with patch('sys.stdout', new=StringIO()):
                interactive_chat(mock_runner_interactive, prepare_next_prompt=True)
        
        # Should have user and assistant messages
        user_messages = [msg for msg in captured_messages if msg["role"] == "user"]
        assistant_messages = [msg for msg in captured_messages if msg["role"] == "assistant"]
        
        assert len(user_messages) == 1
        assert len(assistant_messages) == 1
        assert user_messages[0]["content"] == "user input"
        assert assistant_messages[0]["content"] == "Generated response"
    
    def test_empty_input_ignored(self, mock_runner_interactive):
        """Test that empty input doesn't affect conversation history"""
        conversation_calls = []
        
        def capture_conversation(messages):
            conversation_calls.append(len(messages))
            return "Formatted conversation"
        
        mock_runner_interactive._format_conversation.side_effect = capture_conversation
        
        # Include empty strings and whitespace
        inputs = ["", "  ", "\t", "actual message", "quit"]
        
        with patch('builtins.input', side_effect=inputs):
            with patch('sys.stdout', new=StringIO()):
                interactive_chat(mock_runner_interactive)
        
        # Should only process the non-empty message
        assert len(conversation_calls) == 1
        assert conversation_calls[0] == 1  # Only one message in history
    
    def test_response_stripping(self, mock_runner_interactive):
        """Test that assistant responses are properly stripped"""
        captured_responses = []
        
        def capture_history(messages):
            # Capture assistant responses from history
            for msg in messages:
                if msg["role"] == "assistant":
                    captured_responses.append(msg["content"])
            return "Formatted conversation"
        
        mock_runner_interactive._format_conversation.side_effect = capture_history
        
        # Mock streaming with whitespace
        mock_runner_interactive.generate_streaming.return_value = iter([
            "  Response", " with", " whitespace  "
        ])
        
        with patch('builtins.input', side_effect=["test", "quit"]):
            with patch('sys.stdout', new=StringIO()):
                interactive_chat(mock_runner_interactive, stream=True, prepare_next_prompt=True)
        
        # Response should be stripped when added to history
        assert len(captured_responses) == 1
        assert captured_responses[0] == "Response with whitespace"


class TestChatTemplateIntegration:
    """Test chat template usage in interactive mode."""
    
    def test_chat_template_called_with_history(self, mock_runner_interactive):
        """Test that _format_conversation is called with proper history"""
        with patch('builtins.input', side_effect=["hello", "quit"]):
            with patch('sys.stdout', new=StringIO()):
                interactive_chat(mock_runner_interactive)
        
        # Should call _format_conversation
        mock_runner_interactive._format_conversation.assert_called()
        
        # Should be called with list of message dicts
        call_args = mock_runner_interactive._format_conversation.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == 1
        assert isinstance(call_args[0], dict)
        assert "role" in call_args[0]
        assert "content" in call_args[0]
    
    def test_formatted_prompt_used_for_generation(self, mock_runner_interactive):
        """Test that formatted conversation is passed to generation"""
        with patch('builtins.input', side_effect=["test input", "quit"]):
            with patch('sys.stdout', new=StringIO()):
                interactive_chat(mock_runner_interactive, stream=True)
        
        # Should call generate_streaming with formatted prompt
        mock_runner_interactive.generate_streaming.assert_called()
        call_args = mock_runner_interactive.generate_streaming.call_args
        
        # First argument should be the formatted conversation
        assert call_args[0][0] == "Human: test input\n\nAssistant: "
        
        # Should disable chat template (already applied)
        assert call_args[1]['use_chat_template'] is False
    
    def test_chat_template_fallback_behavior(self, mock_runner_interactive):
        """Test behavior when chat template formatting fails"""
        def failing_format(messages):
            raise Exception("Template error")

        mock_runner_interactive._format_conversation.side_effect = failing_format

        with patch('builtins.input', side_effect=["test", "quit"]):
            with patch('sys.stdout', new=StringIO()) as fake_out, \
                 patch('sys.stderr', new=StringIO()) as fake_err:
                # Should handle template errors gracefully
                interactive_chat(mock_runner_interactive)

        stderr_output = fake_err.getvalue()
        # Error should be on stderr
        assert "ERROR" in stderr_output


class TestInteractiveParameters:
    """Test parameter passing in interactive mode."""
    
    def test_parameter_passing_streaming(self, mock_runner_interactive):
        """Test that parameters are passed to streaming generation"""
        with patch('builtins.input', side_effect=["test", "quit"]):
            with patch('sys.stdout', new=StringIO()):
                interactive_chat(
                    mock_runner_interactive,
                    stream=True,
                    max_tokens=100,
                    temperature=0.8,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    hide_reasoning=True,
                )
        
        call_args = mock_runner_interactive.generate_streaming.call_args[1]
        assert call_args['max_tokens'] == 100
        assert call_args['temperature'] == 0.8
        assert call_args['top_p'] == 0.95
        assert call_args['repetition_penalty'] == 1.2
        assert call_args['hide_reasoning'] is True
    
    def test_parameter_passing_batch(self, mock_runner_interactive):
        """Test that parameters are passed to batch generation"""
        with patch('builtins.input', side_effect=["test", "quit"]):
            with patch('sys.stdout', new=StringIO()):
                interactive_chat(
                    mock_runner_interactive,
                    stream=False,
                    max_tokens=200,
                    temperature=0.9,
                    top_p=0.85,
                    repetition_penalty=1.3,
                    hide_reasoning=True,
                )
        
        call_args = mock_runner_interactive.generate_batch.call_args[1]
        assert call_args['max_tokens'] == 200
        assert call_args['temperature'] == 0.9
        assert call_args['top_p'] == 0.85
        assert call_args['repetition_penalty'] == 1.3
        assert call_args['hide_reasoning'] is True
    
    def test_use_chat_template_disabled(self, mock_runner_interactive):
        """Test that use_chat_template is disabled in generation calls"""
        with patch('builtins.input', side_effect=["test", "quit"]):
            with patch('sys.stdout', new=StringIO()):
                interactive_chat(
                    mock_runner_interactive,
                    stream=True,
                    use_chat_template=True  # This should be overridden
                )
        
        # Should disable chat template in generation call
        call_args = mock_runner_interactive.generate_streaming.call_args[1]
        assert call_args['use_chat_template'] is False


class TestInteractiveErrorHandling:
    """Test error handling in interactive mode."""
    
    def test_generation_error_recovery(self, mock_runner_interactive):
        """Test that generation errors don't crash interactive mode"""
        # First call fails, second succeeds
        mock_runner_interactive.generate_streaming.side_effect = [
            RuntimeError("Generation failed"),
            iter(["Success"])
        ]

        with patch('builtins.input', side_effect=["first", "second", "quit"]):
            with patch('sys.stdout', new=StringIO()) as fake_out, \
                 patch('sys.stderr', new=StringIO()) as fake_err:
                interactive_chat(mock_runner_interactive, stream=True)

        stdout_output = fake_out.getvalue()
        stderr_output = fake_err.getvalue()
        # Error should be on stderr
        assert "ERROR" in stderr_output
        # Success should be on stdout
        assert "Success" in stdout_output
    
    def test_keyboard_interrupt_handling(self, mock_runner_interactive):
        """Test Ctrl-C handling in interactive mode"""
        with patch('builtins.input', side_effect=KeyboardInterrupt()):
            with patch('sys.stdout', new=StringIO()) as fake_out:
                interactive_chat(mock_runner_interactive)
        
        output = fake_out.getvalue()
        assert "interrupted" in output.lower() or "goodbye" in output.lower()
    
    def test_input_error_recovery(self, mock_runner_interactive):
        """Test recovery from input errors"""
        def failing_input(prompt):
            if "You:" in prompt:
                if not hasattr(failing_input, 'called'):
                    failing_input.called = True
                    raise EOFError("Input failed")
                else:
                    return "quit"
            return prompt
        
        with patch('builtins.input', side_effect=failing_input):
            with patch('sys.stdout', new=StringIO()) as fake_out:
                interactive_chat(mock_runner_interactive)
        
        # Should handle input errors gracefully
        output = fake_out.getvalue()
        assert "Starting interactive chat" in output


class TestInteractiveUI:
    """Test user interface elements of interactive mode."""
    
    def test_user_prompt_display(self, mock_runner_interactive):
        """Test that user prompt is displayed correctly"""
        with patch('builtins.input', side_effect=["test", "quit"]) as mock_input:
            with patch('sys.stdout', new=StringIO()):
                interactive_chat(mock_runner_interactive)
        
        # Should call input with "You: " prompt
        mock_input.assert_called()
        calls = [call.args[0] for call in mock_input.call_args_list]
        assert "You: " in calls
    
    def test_assistant_prompt_display(self, mock_runner_interactive):
        """Test that assistant prompt is displayed correctly"""
        with patch('builtins.input', side_effect=["test", "quit"]):
            with patch('sys.stdout', new=StringIO()) as fake_out:
                interactive_chat(mock_runner_interactive, stream=True)
        
        output = fake_out.getvalue()
        assert "Assistant: " in output
    
    def test_response_formatting(self, mock_runner_interactive):
        """Test that responses are formatted correctly"""
        mock_runner_interactive.generate_streaming.return_value = iter([
            "Token1", "Token2", "Token3"
        ])
        
        with patch('builtins.input', side_effect=["test", "quit"]):
            with patch('sys.stdout', new=StringIO()) as fake_out:
                interactive_chat(mock_runner_interactive, stream=True)
        
        output = fake_out.getvalue()
        # Should include all tokens in output
        assert "Token1Token2Token3" in output or "Token1 Token2 Token3" in output
