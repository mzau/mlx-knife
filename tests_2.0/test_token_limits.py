"""
Token limit tests for Step 1.1/1.2.
Tests dynamic token calculation and server vs run mode differences.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from mlxk2.core.runner import MLXRunner, get_model_context_length
from conftest_runner import mock_mlx_runner_environment


class TestDynamicTokenLimits:
    """Test dynamic token limit calculation based on model context."""
    
    def test_context_length_detection(self):
        """Test that context length is properly extracted from config"""
        # Test various config key patterns
        configs = [
            {"max_position_embeddings": 8192},
            {"n_positions": 4096},
            {"context_length": 16384},
            {"max_sequence_length": 32768},
            {"seq_len": 2048}
        ]
        
        expected_lengths = [8192, 4096, 16384, 32768, 2048]
        
        for config, expected in zip(configs, expected_lengths):
            with patch('builtins.open') as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = str(config).replace("'", '"')
                
                result = get_model_context_length("/fake/path")
                assert result == expected
    
    def test_context_length_fallback(self):
        """Test fallback to default when config unavailable"""
        # Missing file
        with patch('builtins.open', side_effect=FileNotFoundError()):
            result = get_model_context_length("/nonexistent/path")
            assert result == 4096
        
        # Invalid JSON
        with patch('builtins.open') as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "invalid json"
            result = get_model_context_length("/fake/path")
            assert result == 4096
        
        # Missing keys
        with patch('builtins.open') as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = '{"other_key": 1234}'
            result = get_model_context_length("/fake/path")
            assert result == 4096
    
    @patch('mlxk2.core.runner.get_model_context_length')
    def test_runner_dynamic_calculation_run_mode(self, mock_context_length):
        """Test dynamic token calculation for run command (full context)"""
        mock_context_length.return_value = 8192
        
        with patch('mlxk2.core.runner.load') as mock_load:
            mock_load.return_value = (Mock(), Mock())
            
            with patch('mlxk2.core.runner.resolve_model_for_operation') as mock_resolve:
                mock_resolve.return_value = ("test-model", None, None)
                
                with patch('mlxk2.core.runner.get_current_model_cache') as mock_cache:
                    mock_cache.return_value = Mock()
                    
                    # Create runner and test calculation
                    runner = MLXRunner("test-model")
                    runner._context_length = 8192
                    
                    # Run mode: should use full context
                    limit = runner._calculate_dynamic_max_tokens(server_mode=False)
                    assert limit == 8192
    
    @patch('mlxk2.core.runner.get_model_context_length')
    def test_runner_dynamic_calculation_server_mode(self, mock_context_length):
        """Test dynamic token calculation for server (half context for DoS protection)"""
        mock_context_length.return_value = 8192
        
        with patch('mlxk2.core.runner.load') as mock_load:
            mock_load.return_value = (Mock(), Mock())
            
            with patch('mlxk2.core.runner.resolve_model_for_operation') as mock_resolve:
                mock_resolve.return_value = ("test-model", None, None)
                
                with patch('mlxk2.core.runner.get_current_model_cache') as mock_cache:
                    mock_cache.return_value = Mock()
                    
                    # Create runner and test calculation
                    runner = MLXRunner("test-model")
                    runner._context_length = 8192
                    
                    # Server mode: should use half context
                    limit = runner._calculate_dynamic_max_tokens(server_mode=True)
                    assert limit == 4096
    
    def test_no_context_length_fallback(self):
        """Test behavior when context length is unavailable"""
        with patch('mlxk2.core.runner.load') as mock_load:
            mock_load.return_value = (Mock(), Mock())
            
            with patch('mlxk2.core.runner.resolve_model_for_operation') as mock_resolve:
                mock_resolve.return_value = ("test-model", None, None)
                
                with patch('mlxk2.core.runner.get_current_model_cache') as mock_cache:
                    mock_cache.return_value = Mock()
                    
                    # Create runner with no context length
                    runner = MLXRunner("test-model")
                    runner._context_length = None
                    
                    # Should fallback to default
                    limit = runner._calculate_dynamic_max_tokens(server_mode=False)
                    assert limit == 2048
                    
                    limit = runner._calculate_dynamic_max_tokens(server_mode=True)
                    assert limit == 2048


class TestTokenLimitApplication:
    """Test that token limits are properly applied during generation."""
    
    @patch('mlxk2.core.runner.load')
    @patch('mlxk2.core.runner.resolve_model_for_operation')
    @patch('mlxk2.core.runner.get_current_model_cache')
    @patch('mlxk2.core.runner.get_model_context_length')
    def test_generate_streaming_uses_dynamic_limits(self, mock_context, mock_cache, mock_resolve, mock_load):
        """Test that generate_streaming uses dynamic limits when max_tokens=None"""
        # Setup mocks
        mock_context.return_value = 8192
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
            mock_gen.return_value = iter([])  # Empty generation
            
            with MLXRunner("test-model") as runner:
                # Call with max_tokens=None
                list(runner.generate_streaming("test", max_tokens=None))
                
                # Should call generate_step with dynamic limit (full context for run mode)
                mock_gen.assert_called_once()
                call_kwargs = mock_gen.call_args[1]
                assert call_kwargs['max_tokens'] == 8192  # Full context
    
    @patch('mlxk2.core.runner.load')
    @patch('mlxk2.core.runner.resolve_model_for_operation')
    @patch('mlxk2.core.runner.get_current_model_cache')
    @patch('mlxk2.core.runner.get_model_context_length')
    def test_generate_streaming_respects_explicit_limits(self, mock_context, mock_cache, mock_resolve, mock_load):
        """Test that explicit max_tokens is respected"""
        # Setup mocks
        mock_context.return_value = 8192
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
            mock_gen.return_value = iter([])  # Empty generation
            
            with MLXRunner("test-model") as runner:
                # Call with explicit max_tokens
                list(runner.generate_streaming("test", max_tokens=500))
                
                # Should use explicit limit, not dynamic
                mock_gen.assert_called_once()
                call_kwargs = mock_gen.call_args[1]
                assert call_kwargs['max_tokens'] == 500
    
    @patch('mlxk2.core.runner.load')
    @patch('mlxk2.core.runner.resolve_model_for_operation')
    @patch('mlxk2.core.runner.get_current_model_cache')
    @patch('mlxk2.core.runner.get_model_context_length')
    def test_generate_batch_uses_dynamic_limits(self, mock_context, mock_cache, mock_resolve, mock_load):
        """Test that generate_batch also uses dynamic limits"""
        # Setup mocks
        mock_context.return_value = 16384
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
        mock_tokenizer.decode.return_value = "test response"
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        with patch('mlxk2.core.runner.generate_step') as mock_gen:
            mock_gen.return_value = iter([])  # Empty generation
            
            with MLXRunner("test-model") as runner:
                # Call with max_tokens=None
                runner.generate_batch("test", max_tokens=None)
                
                # Should use dynamic limit
                mock_gen.assert_called_once()
                call_kwargs = mock_gen.call_args[1]
                assert call_kwargs['max_tokens'] == 16384  # Full context


class TestLargeContextModels:
    """Test behavior with large context models."""
    
    @patch('mlxk2.core.runner.get_model_context_length')
    def test_large_context_model_limits(self, mock_context_length):
        """Test dynamic limits for large context models"""
        mock_context_length.return_value = 32768  # 32K context
        
        with patch('mlxk2.core.runner.load') as mock_load:
            mock_load.return_value = (Mock(), Mock())
            
            with patch('mlxk2.core.runner.resolve_model_for_operation') as mock_resolve:
                mock_resolve.return_value = ("large-model", None, None)
                
                with patch('mlxk2.core.runner.get_current_model_cache') as mock_cache:
                    mock_cache.return_value = Mock()
                    
                    runner = MLXRunner("large-model")
                    runner._context_length = 32768
                    
                    # Run mode: full context
                    run_limit = runner._calculate_dynamic_max_tokens(server_mode=False)
                    assert run_limit == 32768
                    
                    # Server mode: half context
                    server_limit = runner._calculate_dynamic_max_tokens(server_mode=True)
                    assert server_limit == 16384
    
    @patch('mlxk2.core.runner.get_model_context_length')
    def test_very_large_context_handling(self, mock_context_length):
        """Test handling of very large context models (128K+)"""
        mock_context_length.return_value = 131072  # 128K context
        
        with patch('mlxk2.core.runner.load') as mock_load:
            mock_load.return_value = (Mock(), Mock())
            
            with patch('mlxk2.core.runner.resolve_model_for_operation') as mock_resolve:
                mock_resolve.return_value = ("huge-model", None, None)
                
                with patch('mlxk2.core.runner.get_current_model_cache') as mock_cache:
                    mock_cache.return_value = Mock()
                    
                    runner = MLXRunner("huge-model")
                    runner._context_length = 131072
                    
                    # Should handle very large contexts
                    run_limit = runner._calculate_dynamic_max_tokens(server_mode=False)
                    assert run_limit == 131072
                    
                    server_limit = runner._calculate_dynamic_max_tokens(server_mode=True)
                    assert server_limit == 65536


class TestTokenLimitEdgeCases:
    """Test edge cases in token limit calculation."""
    
    def test_zero_context_length(self):
        """Test handling of zero context length"""
        with patch('mlxk2.core.runner.load') as mock_load:
            mock_load.return_value = (Mock(), Mock())
            
            with patch('mlxk2.core.runner.resolve_model_for_operation') as mock_resolve:
                mock_resolve.return_value = ("test-model", None, None)
                
                with patch('mlxk2.core.runner.get_current_model_cache') as mock_cache:
                    mock_cache.return_value = Mock()
                    
                    runner = MLXRunner("test-model")
                    runner._context_length = 0
                    
                    # Should fallback to default
                    limit = runner._calculate_dynamic_max_tokens(server_mode=False)
                    assert limit == 2048
    
    def test_negative_context_length(self):
        """Test handling of negative context length"""
        runner = MLXRunner.__new__(MLXRunner)  # Create without __init__
        runner._context_length = -1000
        
        # Should fallback to default for negative values
        limit = runner._calculate_dynamic_max_tokens(server_mode=False)
        assert limit == 2048
    
    def test_odd_context_length_division(self):
        """Test server mode with odd context lengths"""
        with patch('mlxk2.core.runner.load') as mock_load:
            mock_load.return_value = (Mock(), Mock())
            
            with patch('mlxk2.core.runner.resolve_model_for_operation') as mock_resolve:
                mock_resolve.return_value = ("test-model", None, None)
                
                with patch('mlxk2.core.runner.get_current_model_cache') as mock_cache:
                    mock_cache.return_value = Mock()
                    
                    runner = MLXRunner("test-model")
                    runner._context_length = 8193  # Odd number
                    
                    # Server mode should handle integer division
                    limit = runner._calculate_dynamic_max_tokens(server_mode=True)
                    assert limit == 4096  # 8193 // 2


class TestServerVsRunDifferences:
    """Test the key difference between server and run mode token policies."""
    
    def test_run_vs_server_mode_policy_difference(self):
        """Test the fundamental difference: run uses full, server uses half"""
        with patch('mlxk2.core.runner.load') as mock_load:
            mock_load.return_value = (Mock(), Mock())
            
            with patch('mlxk2.core.runner.resolve_model_for_operation') as mock_resolve:
                mock_resolve.return_value = ("test-model", None, None)
                
                with patch('mlxk2.core.runner.get_current_model_cache') as mock_cache:
                    mock_cache.return_value = Mock()
                    
                    runner = MLXRunner("test-model")
                    runner._context_length = 8192
                    
                    # Run command: full context (user's own machine, be generous)
                    run_limit = runner._calculate_dynamic_max_tokens(server_mode=False)
                    
                    # Server: half context (DoS protection)
                    server_limit = runner._calculate_dynamic_max_tokens(server_mode=True)
                    
                    # Should be exactly 2:1 ratio
                    assert run_limit == 8192
                    assert server_limit == 4096
                    assert run_limit == 2 * server_limit
    
    def test_rationale_for_different_policies(self):
        """Document the rationale for different token policies"""
        # This test serves as documentation
        
        # Run command rationale:
        # - User's own machine and models
        # - User has full control over resource usage
        # - No DoS concerns (single user)
        # - Be generous with token limits
        
        # Server rationale:
        # - Potentially multiple concurrent requests
        # - DoS protection needed
        # - Resource sharing concerns
        # - Conservative token limits
        
        with patch('mlxk2.core.runner.load') as mock_load:
            mock_load.return_value = (Mock(), Mock())
            
            with patch('mlxk2.core.runner.resolve_model_for_operation') as mock_resolve:
                mock_resolve.return_value = ("test-model", None, None)
                
                with patch('mlxk2.core.runner.get_current_model_cache') as mock_cache:
                    mock_cache.return_value = Mock()
                    
                    runner = MLXRunner("test-model")
                    runner._context_length = 8192
                    
                    # These policies should be clearly different
                    run_policy = runner._calculate_dynamic_max_tokens(server_mode=False)
                    server_policy = runner._calculate_dynamic_max_tokens(server_mode=True)
                    
                    assert run_policy > server_policy
                    assert run_policy / server_policy == 2.0  # Exactly 2x difference
