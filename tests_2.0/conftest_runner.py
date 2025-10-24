"""
Fixtures for MLXRunner testing - solves mock complexity issues.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from contextlib import contextmanager


@pytest.fixture
def temp_cache_dir():
    """Isolated cache directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@contextmanager
def mock_mlx_runner_environment(temp_cache_dir, model_name="test-model", context_length=8192):
    """Complete mock environment for MLXRunner that handles all dependencies."""
    
    # Create proper directory structure
    model_cache_dir = temp_cache_dir / f"models--{model_name}"
    snapshots_dir = model_cache_dir / "snapshots" / "abc123"
    snapshots_dir.mkdir(parents=True)
    
    # Create mock config.json
    config_path = snapshots_dir / "config.json"
    config_path.write_text(f'{{"max_position_embeddings": {context_length}}}')
    
    with patch('mlxk2.core.runner.resolve_model_for_operation') as mock_resolve, \
         patch('mlxk2.core.runner.get_current_model_cache') as mock_cache, \
         patch('mlxk2.core.runner.hf_to_cache_dir') as mock_hf_to_cache, \
         patch('mlxk2.core.runner.load') as mock_load, \
         patch('mlxk2.core.runner.generate_step') as mock_gen_step:
        
        # Setup return values
        mock_resolve.return_value = (model_name, None, None)
        mock_cache.return_value = temp_cache_dir
        mock_hf_to_cache.return_value = f"models--{model_name}"
        
        # Setup model and tokenizer mocks
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.eos_token_ids = {mock_tokenizer.eos_token_id}
        mock_tokenizer.pad_token = None
        mock_tokenizer.additional_special_tokens = []
        mock_tokenizer.added_tokens_decoder = {}
        
        # Common encode/decode behavior
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.side_effect = lambda tokens: " ".join(f"token{t}" for t in tokens)
        
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        # Setup generation step mock
        mock_gen_step.return_value = iter([
            (Mock(item=lambda: 1), Mock()),
            (Mock(item=lambda: 2), Mock()),
            (Mock(item=lambda: 3), Mock())
        ])
        
        yield {
            'mock_resolve': mock_resolve,
            'mock_cache': mock_cache,
            'mock_hf_to_cache': mock_hf_to_cache,
            'mock_load': mock_load,
            'mock_model': mock_model,
            'mock_tokenizer': mock_tokenizer,
            'mock_gen_step': mock_gen_step,
            'temp_cache_dir': temp_cache_dir,
            'model_path': snapshots_dir
        }


@pytest.fixture
def mock_runner_env(temp_cache_dir):
    """Fixture version of mock_mlx_runner_environment."""
    with mock_mlx_runner_environment(temp_cache_dir) as env:
        yield env
