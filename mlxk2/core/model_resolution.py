"""Model name resolution and expansion for MLX-Knife 2.0."""

from pathlib import Path
from typing import Tuple, Optional, List
from .cache import get_current_model_cache, hf_to_cache_dir, cache_dir_to_hf


def expand_model_name(model_name: str) -> str:
    """Expand short model names, preferring mlx-community if it exists."""
    if "/" in model_name:
        return model_name
    
    # Only try mlx-community if it actually exists
    mlx_candidate = f"mlx-community/{model_name}"
    model_cache = get_current_model_cache()
    mlx_cache_dir = model_cache / hf_to_cache_dir(mlx_candidate)
    if mlx_cache_dir.exists():
        return mlx_candidate
    
    # Otherwise return as-is (no pattern forcing!)
    return model_name


def parse_model_spec(model_spec: str) -> Tuple[str, Optional[str]]:
    """Parse model specification with optional @hash syntax.
    
    Examples:
        'Phi-3-mini' → ('mlx-community/Phi-3-mini-4k-instruct-4bit', None)
        'Qwen3@e96' → ('Qwen/Qwen3-Coder-480B-A35B-Instruct', 'e96')
    """
    if "@" in model_spec:
        model_name, commit_hash = model_spec.rsplit("@", 1)
        expanded_name = expand_model_name(model_name)
        return expanded_name, commit_hash
    
    expanded_name = expand_model_name(model_spec)
    return expanded_name, None


def find_matching_models(pattern: str) -> List[Tuple[Path, str]]:
    """Find models that match a partial pattern (case-insensitive)."""
    model_cache = get_current_model_cache()
    if not model_cache.exists():
        return []
        
    all_models = [d for d in model_cache.iterdir() if d.name.startswith("models--")]
    matches = []
    
    for model_dir in all_models:
        hf_name = cache_dir_to_hf(model_dir.name)
        # Case-insensitive partial matching in full name or short name
        short_name = hf_name.split('/')[-1] if '/' in hf_name else hf_name
        
        if (pattern.lower() in hf_name.lower() or 
            pattern.lower() in short_name.lower()):
            matches.append((model_dir, hf_name))
    
    return matches


def find_model_by_hash(pattern: str, commit_hash: str) -> Optional[Tuple[Path, str, str]]:
    """Find model by pattern and verify hash exists in snapshots.
    
    Returns: (model_dir, hf_name, full_hash) or None
    """
    matches = find_matching_models(pattern)
    
    for model_dir, hf_name in matches:
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            continue
            
        # Check for hash match (short hash support)
        for snapshot_dir in snapshots_dir.iterdir():
            if snapshot_dir.is_dir() and snapshot_dir.name.startswith(commit_hash):
                return model_dir, hf_name, snapshot_dir.name
    
    return None


def resolve_model_for_operation(model_spec: str) -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
    """Resolve model specification for operations.
    
    Returns:
        (resolved_name, commit_hash, ambiguous_matches)
        
    Examples:
        'Phi-3-mini' → ('mlx-community/Phi-3-mini-4k-instruct-4bit', None, None)
        'Qwen3@e96' → ('Qwen/Qwen3-Coder-480B-A35B-Instruct', 'e96', None) 
        'ambig' → (None, None, ['model1', 'model2'])
    """
    model_name, commit_hash = parse_model_spec(model_spec)
    
    # For @hash syntax, find by pattern + hash verification
    if commit_hash:
        base_pattern = model_spec.split('@')[0]
        result = find_model_by_hash(base_pattern, commit_hash)
        if result:
            model_dir, hf_name, full_hash = result
            return hf_name, full_hash, None
        else:
            return None, commit_hash, []
    
    # Try exact match first
    model_cache = get_current_model_cache()
    exact_cache_dir = model_cache / hf_to_cache_dir(model_name)
    if exact_cache_dir.exists():
        return model_name, None, None
    
    # Try fuzzy matching
    base_pattern = model_spec.split('@')[0] if '@' in model_spec else model_spec
    matches = find_matching_models(base_pattern)
    
    if not matches:
        return None, None, []
    elif len(matches) == 1:
        # Unambiguous fuzzy match
        model_dir, hf_name = matches[0]
        return hf_name, commit_hash, None
    else:
        # Ambiguous matches
        match_names = [hf_name for _, hf_name in matches]
        return None, commit_hash, match_names