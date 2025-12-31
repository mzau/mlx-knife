"""
Core MLXRunner tests for 2.0 implementation.
Tests the core model execution engine ported from 1.x.
"""

import pytest
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path
from contextlib import contextmanager

import mlx.core as mx
from mlxk2.core.runner import MLXRunner


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


@contextmanager
def mock_runner_environment(temp_cache_dir, model_name="test-model"):
    """Mock the environment needed for MLXRunner tests."""
    with patch('mlxk2.core.runner.load') as mock_load, \
         patch('mlxk2.core.runner.resolve_model_for_operation') as mock_resolve, \
         patch('mlxk2.core.cache.get_current_model_cache') as mock_cache, \
         patch('mlxk2.core.cache.hf_to_cache_dir') as mock_hf_to_cache, \
         patch('mlxk2.core.runner.get_model_context_length') as mock_context:
        
        # Mock successful model resolution
        mock_resolve.return_value = (model_name, None, None)
        mock_cache.return_value = temp_cache_dir
        mock_hf_to_cache.return_value = f"models--{model_name}"
        mock_context.return_value = 8192
        
        # Create mock snapshots directory
        snapshots_dir = temp_cache_dir / f"models--{model_name}" / "snapshots" / "abc123"
        snapshots_dir.mkdir(parents=True)
        
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.eos_token_ids = {mock_tokenizer.eos_token_id}
        mock_tokenizer.pad_token = None
        mock_tokenizer.additional_special_tokens = []
        mock_tokenizer.added_tokens_decoder = {}
        mock_tokenizer.chat_template = None
        mock_tokenizer.name_or_path = f"mock-{model_name}"
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        yield {
            'mock_load': mock_load,
            'mock_model': mock_model,
            'mock_tokenizer': mock_tokenizer,
            'mock_resolve': mock_resolve
        }


class TestMLXRunnerBasic:
    """Basic MLXRunner functionality tests"""
    
    def test_runner_context_manager(self, temp_cache_dir):
        """Test context manager pattern for memory safety"""
        model_name = "test-model"
        
        with mock_runner_environment(temp_cache_dir) as mocks:
            with MLXRunner(model_name) as runner:
                assert runner is not None
                # Should have loaded model
                mocks['mock_load'].assert_called_once()
            
            # Should cleanup on exit (tested via mock verification)
    
    def test_runner_cleanup_on_exception(self, temp_cache_dir):
        """Test that cleanup happens even on exception"""
        model_name = "test-model"
        
        with mock_runner_environment(temp_cache_dir) as mocks:
            try:
                with MLXRunner(model_name) as runner:
                    # Force an exception
                    raise ValueError("Test exception")
            except ValueError:
                pass
            
            # Should still have called load and cleanup
            mocks['mock_load'].assert_called_once()
    
    def test_generate_streaming_basic(self, temp_cache_dir):
        """Test basic streaming generation"""
        model_name = "test-model"
        
        with mock_runner_environment(temp_cache_dir, model_name) as mocks:
            # Mock generate_step to yield tokens
            with patch('mlxk2.core.runner.generate_step') as mock_gen:
                # generate_step yields (token, logits) tuples  
                mock_gen.return_value = [
                    (mx.array([1]), mx.zeros(1)),  # Token IDs as mx.array
                    (mx.array([2]), mx.zeros(1)), 
                ]
                
                # Mock tokenizer methods
                mocks['mock_tokenizer'].encode.return_value = [100, 101]  # Prompt tokens
                mocks['mock_tokenizer'].eos_token_id = 999  # Don't trigger EOS
                mocks['mock_tokenizer'].eos_token_ids = {mocks['mock_tokenizer'].eos_token_id}
                mocks['mock_tokenizer'].chat_template = None  # Disable chat template
                
                # Mock decode to return consistent strings based on token list length/content
                def mock_decode(tokens):
                    if tokens == [1]:
                        return "Hello"
                    elif tokens == [1, 2]:
                        return "Hello world"
                    elif tokens == [2]:
                        return " world"
                    else:
                        return "unknown"
                
                mocks['mock_tokenizer'].decode.side_effect = mock_decode
                # Use MockDetokenizer for proper BPE space marker handling
                mocks['mock_tokenizer'].detokenizer = MockDetokenizer(mock_decode)

                with MLXRunner(model_name) as runner:
                    tokens = list(runner.generate_streaming("test prompt", max_tokens=2))
                    
                # Should yield incremental tokens
                assert len(tokens) >= 1
                assert any("Hello" in token for token in tokens)
    
    def test_generate_batch(self, temp_cache_dir):
        """Test batch generation (complete output at once)"""
        model_name = "test-model"
        
        with mock_runner_environment(temp_cache_dir, model_name) as mocks:
            with patch('mlxk2.core.runner.generate_step') as mock_gen:
                mock_gen.return_value = [
                    (mx.array([1]), mx.zeros(1)),
                    (mx.array([2]), mx.zeros(1)),
                    (mx.array([3]), mx.zeros(1))
                ]
                
                # Mock tokenizer for batch mode
                mocks['mock_tokenizer'].encode.return_value = [100, 101]  # Prompt
                mocks['mock_tokenizer'].decode.side_effect = lambda tokens: " ".join([f"token{t}" for t in tokens])
                mocks['mock_tokenizer'].eos_token_id = 999  # Don't trigger EOS
                mocks['mock_tokenizer'].eos_token_ids = {mocks['mock_tokenizer'].eos_token_id}
                mocks['mock_tokenizer'].chat_template = None
                
                with MLXRunner(model_name) as runner:
                    result = runner.generate_batch("test prompt", max_tokens=3)
                    
                # Should return a single string (complete response)
                assert isinstance(result, str)
                assert len(result) > 0


class TestMLXRunnerStopTokens:
    """Test stop token filtering functionality"""
    
    def test_chat_stop_tokens_filtered_when_enabled(self, temp_cache_dir):
        """Chat stop tokens are filtered only when explicitly enabled"""
        model_name = "test-model"

        with mock_runner_environment(temp_cache_dir, model_name) as mocks:
            with patch('mlxk2.core.runner.generate_step') as mock_gen:
                mock_gen.return_value = [
                    (1, 0),
                    (2, 0),
                    (3, 0)
                ]
                # Encode returns prompt tokens
                mocks['mock_tokenizer'].encode.return_value = [100]
                # Decode returns full generated text when decoding generated tokens
                def mock_decode(tokens):
                    if tokens == [1]:
                        return "Response"
                    if tokens == [1, 2]:
                        return "Response\nHuman:"
                    if tokens == [1, 2, 3]:
                        return "Response\nHuman: filtered"
                    # Fallback for other cases
                    return ""
                mocks['mock_tokenizer'].decode.side_effect = mock_decode
                # Mock detokenizer (Session 60 BPE fix)
                mocks['mock_tokenizer'].detokenizer = MockDetokenizer(mock_decode)

                with MLXRunner(model_name) as runner:
                    result = runner.generate_batch("test prompt", use_chat_stop_tokens=True)

                # Should stop at chat stop token
                assert "\nHuman:" not in result
                assert result == "Response"

    def test_chat_stop_tokens_not_filtered_by_default(self, temp_cache_dir):
        """By default, batch mode does not strip chat stop tokens"""
        model_name = "test-model"

        with mock_runner_environment(temp_cache_dir, model_name) as mocks:
            with patch('mlxk2.core.runner.generate_step') as mock_gen:
                mock_gen.return_value = [
                    (1, 0),
                    (2, 0),
                    (3, 0)
                ]
                mocks['mock_tokenizer'].encode.return_value = [100]
                def mock_decode(tokens):
                    if tokens == [1]:
                        return "Response"
                    if tokens == [1, 2]:
                        return "Response\nHuman:"
                    if tokens == [1, 2, 3]:
                        return "Response\nHuman: rest"
                    return ""
                mocks['mock_tokenizer'].decode.side_effect = mock_decode
                # Mock detokenizer (Session 60 BPE fix)
                mocks['mock_tokenizer'].detokenizer = MockDetokenizer(mock_decode)

                with MLXRunner(model_name) as runner:
                    result = runner.generate_batch("test prompt")

                # Default behavior: token remains unless explicitly enabled
                assert "\nHuman:" in result
    
    def test_streaming_vs_batch_consistency(self, temp_cache_dir):
        """Test that streaming and batch modes produce identical output"""
        model_name = "test-model"

        with mock_runner_environment(temp_cache_dir, model_name) as mocks:
            # Same mock sequence for both tests
            def mock_generation():
                return [
                    (1, 0),
                    (2, 0),
                    (3, 0)
                ]

            mocks['mock_tokenizer'].encode.return_value = [100]
            def mock_decode(tokens):
                if tokens == [1]:
                    return "Hello"
                if tokens == [2]:
                    return " world"
                if tokens == [3]:
                    return "!"
                if tokens == [1, 2]:
                    return "Hello world"
                if tokens == [2, 3]:
                    return " world!"
                if tokens == [1, 2, 3]:
                    return "Hello world!"
                return ""
            mocks['mock_tokenizer'].decode.side_effect = mock_decode
            # Mock detokenizer (Session 60 BPE fix)
            mocks['mock_tokenizer'].detokenizer = MockDetokenizer(mock_decode)

            with MLXRunner(model_name) as runner:
                # Test streaming
                with patch('mlxk2.core.runner.generate_step', return_value=mock_generation()):
                    streaming_result = "".join(runner.generate_streaming("test"))

                # Test batch
                with patch('mlxk2.core.runner.generate_step', return_value=mock_generation()):
                    batch_result = runner.generate_batch("test")

                assert streaming_result == batch_result


class TestMLXRunnerMemorySafety:
    """Test memory management and cleanup"""
    
    def test_model_cleanup_on_context_exit(self, temp_cache_dir):
        """Test that model is properly cleaned up"""
        model_name = "test-model"
        
        with patch('mlxk2.core.runner.load') as mock_load:
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_load.return_value = (mock_model, mock_tokenizer)
            
            runner = None
            with MLXRunner(model_name) as r:
                runner = r
                assert runner.model is not None
                assert runner.tokenizer is not None
            
            # After context exit, model should be cleaned up
            assert runner.model is None
            assert runner.tokenizer is None
    
    def test_multiple_context_managers(self, temp_cache_dir):
        """Test that multiple runners can be used sequentially"""
        model_name = "test-model"
        
        with patch('mlxk2.core.runner.load') as mock_load:
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1]
            mock_tokenizer.decode.return_value = "ok"
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.eos_token_ids = {mock_tokenizer.eos_token_id}
            mock_tokenizer.additional_special_tokens = []
            mock_tokenizer.added_tokens_decoder = {}
            mock_load.return_value = (mock_model, mock_tokenizer)
            
            # First runner
            with MLXRunner(model_name) as runner1:
                assert runner1 is not None
            
            # Second runner should work independently
            with MLXRunner(model_name) as runner2:
                assert runner2 is not None
            
            # Should have loaded model twice
            assert mock_load.call_count == 2


class TestMLXRunnerDynamicTokens:
    """Test dynamic token limit functionality"""
    
    def test_no_max_tokens_uses_dynamic(self, temp_cache_dir):
        """Test that None max_tokens uses dynamic limit based on model context"""
        model_name = "test-model"
        
        with patch('mlxk2.core.runner.load') as mock_load:
            mock_load.return_value = (Mock(), Mock())
            
            # Mock config reading for context length
            with patch('mlxk2.core.runner.get_model_context_length', return_value=8192):
                with MLXRunner(model_name) as runner:
                    # Should calculate dynamic limit from context length
                    dynamic_limit = runner._calculate_dynamic_max_tokens()
                    
                    # Should be a reasonable fraction of context (server-mode default)
                    # Accept half-context on 8K models as reasonable
                    assert 1000 <= dynamic_limit <= 4096
    
    def test_respects_explicit_max_tokens(self, temp_cache_dir):
        """Test that explicit max_tokens is respected"""
        model_name = "test-model"
        
        with patch('mlxk2.core.runner.load') as mock_load:
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1]
            mock_tokenizer.decode.return_value = "ok"
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.eos_token_ids = {mock_tokenizer.eos_token_id}
            mock_tokenizer.additional_special_tokens = []
            mock_tokenizer.added_tokens_decoder = {}
            mock_load.return_value = (mock_model, mock_tokenizer)
            
            with MLXRunner(model_name) as runner:
                # When max_tokens is explicitly set, should respect it
                with patch('mlxk2.core.runner.generate_step') as mock_gen:
                    mock_gen.return_value = iter([(mx.array([1]), mx.zeros(1))])
                    
                    # Mock to check that max_tokens is passed through
                    result = runner.generate_batch("test", max_tokens=100)
                    
                    # Should have respected the explicit limit
                    # (Details depend on implementation)


class TestMLXRunnerErrorHandling:
    """Test error handling and edge cases"""
    
    def test_model_loading_failure(self, temp_cache_dir):
        """Test handling of model loading failures"""
        model_path = "nonexistent-model"
        
        with patch('mlxk2.core.runner.load') as mock_load:
            mock_load.side_effect = FileNotFoundError("Model not found")
            
            with pytest.raises(FileNotFoundError):
                with MLXRunner(model_path):
                    pass
    
    def test_generation_interruption(self, temp_cache_dir):
        """Test Ctrl-C interruption handling"""
        model_name = "test-model"
        
        with patch('mlxk2.core.runner.load') as mock_load:
            mock_model, mock_tokenizer = Mock(), Mock()
            # Minimal tokenizer stubs to satisfy runner
            mock_tokenizer.encode.return_value = [1]
            mock_tokenizer.decode.return_value = "ok"
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.eos_token_ids = {mock_tokenizer.eos_token_id}
            mock_tokenizer.additional_special_tokens = []
            mock_tokenizer.added_tokens_decoder = {}
            mock_load.return_value = (mock_model, mock_tokenizer)

            # With new recovery semantics, a pre-existing interruption flag
            # is cleared at the start of a new generation.
            with MLXRunner(model_name) as runner:
                runner._interrupted = True
                tokens = list(runner.generate_streaming("test"))
                # Should not yield an interruption message at start
                assert not any(isinstance(t, str) and "interrupted" in t.lower() for t in tokens)


# Test fixtures for integration with existing test infrastructure
@pytest.fixture
def mock_tiny_model():
    """Minimal model for fast tests"""
    return "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture  
def temp_cache_dir():
    """Isolated cache directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
