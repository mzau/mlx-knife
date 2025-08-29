#!/usr/bin/env python3
"""
Debug the _cleanup_model_locks function to find the exact bug.
"""

import tempfile
import shutil
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).parent))

def debug_lock_cleanup():
    """Debug the _cleanup_model_locks function step by step."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_cache = Path(temp_dir)
        hub_cache = temp_cache / "hub"
        hub_cache.mkdir()
        
        # Set MODEL_CACHE
        import mlx_knife.cache_utils as cache_utils
        original_cache = cache_utils.MODEL_CACHE
        cache_utils.MODEL_CACHE = hub_cache
        
        try:
            # Create test structure
            model_name = "test-org/broken-model"
            cache_dir_name = "models--test-org--broken-model"
            
            locks_dir = hub_cache / ".locks" / cache_dir_name
            locks_dir.mkdir(parents=True)
            (locks_dir / "test1.lock").touch()
            (locks_dir / "test2.lock").touch()
            
            print(f"Setup complete:")
            print(f"  MODEL_CACHE: {cache_utils.MODEL_CACHE}")
            print(f"  model_name: {model_name}")
            print(f"  locks_dir: {locks_dir}")
            print(f"  locks_dir.exists(): {locks_dir.exists()}")
            print(f"  lock files: {list(locks_dir.iterdir())}")
            
            # Now step through _cleanup_model_locks manually
            print(f"\n=== Manual _cleanup_model_locks debug ===")
            
            from mlx_knife.cache_utils import hf_to_cache_dir
            expected_cache_dir = hf_to_cache_dir(model_name)
            calculated_locks_dir = cache_utils.MODEL_CACHE.parent / ".locks" / expected_cache_dir
            
            print(f"  hf_to_cache_dir('{model_name}'): {expected_cache_dir}")
            print(f"  MODEL_CACHE.parent: {cache_utils.MODEL_CACHE.parent}")
            print(f"  calculated locks_dir: {calculated_locks_dir}")
            print(f"  calculated locks_dir.exists(): {calculated_locks_dir.exists()}")
            
            # The bug is probably here! ^^
            
            if calculated_locks_dir.exists():
                lock_files = list(calculated_locks_dir.iterdir())
                print(f"  lock_files found: {lock_files}")
                
                print(f"\n  Would delete: {calculated_locks_dir}")
                # shutil.rmtree(calculated_locks_dir)  # Don't actually delete for debugging
                
            else:
                print(f"  ❌ BUG: calculated locks_dir does not exist!")
                print(f"     Expected: {calculated_locks_dir}")
                print(f"     Actual:   {locks_dir}")
                
        finally:
            cache_utils.MODEL_CACHE = original_cache

if __name__ == "__main__":
    debug_lock_cleanup()