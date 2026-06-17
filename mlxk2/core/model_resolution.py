"""Model name resolution and expansion for MLX-Knife 2.0."""

from pathlib import Path
from typing import Tuple, Optional, List
from .cache import get_current_model_cache, hf_to_cache_dir, cache_dir_to_hf
from ..operations.workspace import (
    is_workspace_path,
    is_explicit_path,
    get_workspace_home,
    read_workspace_metadata,
)


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


def model_display_identity(
    resolved_name: str, commit_hash: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """Map a resolved model to a ``(display_name, content_hash)`` pair for output stamping.

    ``resolve_model_for_operation`` is the forward rule (caller-spec → location). This is its
    reverse-safe identity companion, used wherever a model identity is written into machine
    output (e.g. ``mlxk embed`` JSONL metadata). It **never returns an absolute path**:

    - **Workspace** models surface the sentinel ``source_repo`` (a portable ``org/name``,
      consistent with cache) + the ADR-025 ``content_hash``. An *unmanaged* workspace (no
      sentinel) falls back to the directory basename and a ``None`` hash.
    - **Cache** models surface the HF ``org/name`` + the snapshot revision (40-char git SHA;
      derived from the latest snapshot when no explicit ``@hash`` was given).

    The hash *shape* (``sha256:…`` vs a 40-char revision) also tells a consumer which source a
    record came from. Used for the same-model rule: a consumer compares ``(model, content_hash)``
    to detect that two embeddings share a vector space.
    """
    if is_workspace_path(resolved_name):
        meta = read_workspace_metadata(Path(resolved_name))
        display = meta.get("source_repo") or Path(resolved_name).name
        return display, meta.get("content_hash")

    # Cache model: resolved_name is already the canonical org/name.
    content_hash = commit_hash
    if not content_hash:
        base = get_current_model_cache() / hf_to_cache_dir(resolved_name)
        snapshots = base / "snapshots"
        if snapshots.exists():
            dirs = [d for d in snapshots.iterdir() if d.is_dir()]
            if dirs:
                content_hash = max(dirs, key=lambda d: d.stat().st_mtime).name
    return resolved_name, content_hash


def _short_content_hash(content_hash: Optional[str], hash_len: int = 8) -> str:
    """Normalize a content hash to a short, stable token for the embed ``system_fingerprint``.

    Drops a ``sha256:`` prefix (workspace ADR-025 hashes) so cache (40-char git revision) and workspace
    (``sha256:…``) hashes share one shape, then keeps the first ``hash_len`` hex chars — enough to
    distinguish revisions of the *same* model for change-detection (not adversarial collision resistance).
    ``None`` (an unmanaged workspace with no sentinel) becomes ``nohash`` so the slot is never empty.
    """
    if not content_hash:
        return "nohash"
    h = content_hash.split(":", 1)[1] if content_hash.startswith("sha256:") else content_hash
    return h[:hash_len] or "nohash"


def embed_system_fingerprint(
    resolved_name: str,
    commit_hash: Optional[str] = None,
    *,
    cpu: bool = False,
    hash_len: int = 8,
) -> str:
    """OpenAI ``system_fingerprint`` token for the embeddings response — a change-detection signal.

    An embedding vector space is fixed by the model, its content fingerprint (revision / quant), and the
    device (CPU vs GPU diverge ~0.98 cosine on a 4-bit model). The model name already rides in the response
    ``model`` field — the clean, re-sendable selector. This token carries the two remaining discriminators —
    ``{short_hash}.{device}`` — so a consumer detects a revision / device swap (and, via the hash, a
    different model) by plain equality on **one** field across an ``embed-serve`` restart.

    Mirrors OpenAI's own ``system_fingerprint`` semantics ("the backend configuration changed"); on the
    embeddings response it is an additive mlxk extension (the field is standard on chat/completions, not
    embeddings). Shape ``a1b2c3d4.gpu`` — short content hash + ``.`` + device (``gpu`` | ``cpu``); the ``.``
    is unambiguous (pure ``hex.device``). Opaque: compare it, don't parse it.
    """
    _, content_hash = model_display_identity(resolved_name, commit_hash)
    device = "cpu" if cpu else "gpu"
    return f"{_short_content_hash(content_hash, hash_len)}.{device}"


def resolve_model_for_operation(model_spec: str) -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
    """Resolve model specification for operations.

    Supports both HuggingFace model IDs and local workspace paths.

    Returns:
        (resolved_name, commit_hash, ambiguous_matches)

    Examples:
        'Phi-3-mini' → ('mlx-community/Phi-3-mini-4k-instruct-4bit', None, None)
        'Qwen3@e96' → ('Qwen/Qwen3-Coder-480B-A35B-Instruct', 'e96', None)
        './workspace' → ('/abs/path/to/workspace', None, None)
        '/abs/path/workspace' → ('/abs/path/workspace', None, None)
        'Mistral-Small' → cache resolution (NOT workspace, even if local dir exists)
        'ambig' → (None, None, ['model1', 'model2'])
    """
    # Check if model_spec is an EXPLICIT workspace path (ADR-018 Phase 0c)
    # Only paths starting with ./ ../ / or being . or .. are treated as workspace paths
    # This ensures "model-name" goes through cache resolution even if a local dir exists
    if is_explicit_path(model_spec) and is_workspace_path(model_spec):
        # Explicit workspace path - return absolute path, skip cache logic
        return (str(Path(model_spec).resolve()), None, None)

    # ADR-022: Check MLXK_WORKSPACE_HOME first (workspace-first paradigm)
    workspace_home = get_workspace_home()
    if workspace_home:
        # Try exact match in workspace home
        ws_path = workspace_home / model_spec
        if is_workspace_path(str(ws_path)):
            return (str(ws_path), None, None)

        # Try fuzzy match in workspace home (case-insensitive substring)
        for subdir in workspace_home.iterdir():
            if subdir.is_dir() and model_spec.lower() in subdir.name.lower():
                if is_workspace_path(str(subdir)):
                    return (str(subdir), None, None)

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