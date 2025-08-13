"""
Unit tests for MLXRunner memory management robustness.

Tests context manager implementation, exception handling, 
and cleanup guarantees without requiring actual MLX models.
"""
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import gc


class TestMLXRunnerMemoryManagement(unittest.TestCase):
    """Test MLXRunner memory management robustness."""
    
    @patch('mlx_knife.mlx_runner.mx')
    @patch('mlx_knife.mlx_runner.load')
    def test_context_manager_basic_flow(self, mock_load, mock_mx):
        """Test basic context manager flow with successful execution."""
        from mlx_knife.mlx_runner import MLXRunner
        
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '</s>'
        mock_tokenizer.eos_token_id = 2
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_mx.get_active_memory.return_value = 1024 * 1024 * 1024  # 1GB
        
        # Test successful context manager usage
        with MLXRunner("test_model", verbose=False) as runner:
            self.assertIsNotNone(runner.model)
            self.assertIsNotNone(runner.tokenizer)
            self.assertTrue(runner._model_loaded)
            self.assertTrue(runner._context_entered)
        
        # After exiting context, model should be cleaned up
        self.assertIsNone(runner.model)
        self.assertIsNone(runner.tokenizer)
        self.assertFalse(runner._model_loaded)
        self.assertFalse(runner._context_entered)
        
        # Verify cleanup was called
        mock_mx.clear_cache.assert_called()
    
    @patch('mlx_knife.mlx_runner.mx')
    @patch('mlx_knife.mlx_runner.load')
    def test_context_manager_exception_in_load(self, mock_load, mock_mx):
        """Test cleanup when exception occurs during model loading."""
        from mlx_knife.mlx_runner import MLXRunner
        
        # Setup mock to fail during load
        mock_load.side_effect = RuntimeError("Model loading failed")
        mock_mx.get_active_memory.return_value = 1024 * 1024 * 1024
        
        # Test that exception is propagated and cleanup happens
        with self.assertRaises(RuntimeError) as cm:
            with MLXRunner("test_model", verbose=False) as runner:
                pass  # Should never reach here
        
        self.assertIn("Failed to load model", str(cm.exception))
        
        # Verify cleanup was called even on failure
        mock_mx.clear_cache.assert_called()
    
    @patch('mlx_knife.mlx_runner.mx')
    @patch('mlx_knife.mlx_runner.load')
    def test_context_manager_exception_in_body(self, mock_load, mock_mx):
        """Test cleanup when exception occurs in context body."""
        from mlx_knife.mlx_runner import MLXRunner
        
        # Setup successful mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '</s>'
        mock_tokenizer.eos_token_id = 2
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_mx.get_active_memory.return_value = 1024 * 1024 * 1024
        
        # Test exception in context body
        with self.assertRaises(ValueError):
            with MLXRunner("test_model", verbose=False) as runner:
                self.assertTrue(runner._model_loaded)
                raise ValueError("User error")
        
        # Cleanup should still happen
        self.assertIsNone(runner.model)
        self.assertIsNone(runner.tokenizer)
        self.assertFalse(runner._model_loaded)
        mock_mx.clear_cache.assert_called()
    
    @patch('mlx_knife.mlx_runner.mx')
    @patch('mlx_knife.mlx_runner.load')
    def test_prevent_nested_context_usage(self, mock_load, mock_mx):
        """Test that nested context manager usage is prevented."""
        from mlx_knife.mlx_runner import MLXRunner
        
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '</s>'
        mock_tokenizer.eos_token_id = 2
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_mx.get_active_memory.return_value = 1024 * 1024 * 1024
        
        runner = MLXRunner("test_model", verbose=False)
        
        # First context should work
        with runner:
            self.assertTrue(runner._context_entered)
            
            # Nested context should fail
            with self.assertRaises(RuntimeError) as cm:
                with runner:
                    pass
            
            self.assertIn("cannot be entered multiple times", str(cm.exception))
        
        # After exiting, should be able to use again
        self.assertFalse(runner._context_entered)
        
        # Second usage should work
        with runner:
            self.assertTrue(runner._context_entered)
    
    @patch('mlx_knife.mlx_runner.mx')
    @patch('mlx_knife.mlx_runner.load')
    def test_partial_loading_failure_cleanup(self, mock_load, mock_mx):
        """Test cleanup when loading partially succeeds then fails."""
        from mlx_knife.mlx_runner import MLXRunner
        
        # Setup mock to partially succeed
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Missing required attributes to trigger failure in _extract_stop_tokens
        del mock_tokenizer.eos_token
        del mock_tokenizer.eos_token_id
        mock_tokenizer.encode.side_effect = Exception("Tokenizer error")
        
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_mx.get_active_memory.return_value = 1024 * 1024 * 1024
        
        runner = MLXRunner("test_model", verbose=False)
        
        # Load should succeed even with tokenizer issues
        try:
            runner.load_model()
            # Model should be loaded even if stop token extraction had issues
            self.assertIsNotNone(runner.model)
            self.assertIsNotNone(runner.tokenizer)
        finally:
            # Cleanup should work regardless
            runner.cleanup()
            self.assertIsNone(runner.model)
            self.assertIsNone(runner.tokenizer)
            mock_mx.clear_cache.assert_called()
    
    @patch('mlx_knife.mlx_runner.mx')
    def test_cleanup_idempotency(self, mock_mx):
        """Test that cleanup can be called multiple times safely."""
        from mlx_knife.mlx_runner import MLXRunner
        
        mock_mx.get_active_memory.return_value = 1024 * 1024 * 1024
        
        runner = MLXRunner("test_model", verbose=False)
        runner.model = MagicMock()
        runner.tokenizer = MagicMock()
        runner._model_loaded = True
        
        # Call cleanup multiple times
        for _ in range(3):
            runner.cleanup()
            self.assertIsNone(runner.model)
            self.assertIsNone(runner.tokenizer)
            self.assertFalse(runner._model_loaded)
        
        # Should have been called at least once
        mock_mx.clear_cache.assert_called()
    
    @patch('mlx_knife.mlx_runner.mx')
    @patch('mlx_knife.mlx_runner.load')
    def test_memory_baseline_tracking(self, mock_load, mock_mx):
        """Test memory baseline is properly tracked."""
        from mlx_knife.mlx_runner import MLXRunner
        
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '</s>'
        mock_tokenizer.eos_token_id = 2
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        # Simulate memory growth during loading
        memory_values = [
            1 * 1024**3,  # 1GB baseline
            5 * 1024**3,  # 5GB after loading
            5 * 1024**3,  # 5GB when querying stats
        ]
        mock_mx.get_active_memory.side_effect = memory_values
        
        runner = MLXRunner("test_model", verbose=False)
        runner.load_model()
        
        # Check baseline was captured
        self.assertEqual(runner._memory_baseline, 1.0)  # 1GB
        
        # Check memory usage calculation
        memory_stats = runner.get_memory_usage()
        self.assertEqual(memory_stats["model_gb"], 4.0)  # 5GB - 1GB = 4GB
    
    @patch('mlx_knife.mlx_runner.mx')
    @patch('mlx_knife.mlx_runner.load')
    def test_generate_without_loading(self, mock_load, mock_mx):
        """Test that generate methods fail gracefully without loaded model."""
        from mlx_knife.mlx_runner import MLXRunner
        
        runner = MLXRunner("test_model", verbose=False)
        
        # Try to generate without loading
        with self.assertRaises(RuntimeError) as cm:
            list(runner.generate_streaming("test prompt"))
        self.assertIn("Model not loaded", str(cm.exception))
        
        with self.assertRaises(RuntimeError) as cm:
            runner.generate_batch("test prompt")
        self.assertIn("Model not loaded", str(cm.exception))
    
    @patch('mlx_knife.mlx_runner.mx')
    @patch('mlx_knife.mlx_runner.load')
    def test_server_usage_without_context_manager(self, mock_load, mock_mx):
        """Test server-style usage without context manager."""
        from mlx_knife.mlx_runner import MLXRunner
        
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '</s>'
        mock_tokenizer.eos_token_id = 2
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_mx.get_active_memory.return_value = 1024 * 1024 * 1024
        
        # Server style: manual load and cleanup
        runner = MLXRunner("test_model", verbose=False)
        
        try:
            runner.load_model()
            self.assertTrue(runner._model_loaded)
            self.assertIsNotNone(runner.model)
            
            # Simulate server keeping model loaded
            # and potentially switching models
            runner.cleanup()
            self.assertFalse(runner._model_loaded)
            self.assertIsNone(runner.model)
            
            # Load again (simulating model switch)
            runner.load_model()
            self.assertTrue(runner._model_loaded)
            
        finally:
            # Ensure cleanup happens
            runner.cleanup()
            self.assertFalse(runner._model_loaded)
    
    @patch('mlx_knife.mlx_runner.mx')
    @patch('mlx_knife.mlx_runner.load')
    def test_exception_during_cleanup(self, mock_load, mock_mx):
        """Test that cleanup handles exceptions gracefully."""
        from mlx_knife.mlx_runner import MLXRunner
        
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '</s>'
        mock_tokenizer.eos_token_id = 2
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_mx.get_active_memory.return_value = 1024 * 1024 * 1024
        
        # Make clear_cache raise an exception
        mock_mx.clear_cache.side_effect = Exception("Cache clear failed")
        
        runner = MLXRunner("test_model", verbose=False)
        runner.load_model()
        
        # Cleanup should complete even if mx.clear_cache fails
        runner.cleanup()  # Should not raise
        
        # State should still be cleaned
        self.assertIsNone(runner.model)
        self.assertIsNone(runner.tokenizer)
        self.assertFalse(runner._model_loaded)


if __name__ == '__main__':
    unittest.main()