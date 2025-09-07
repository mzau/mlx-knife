from __future__ import annotations

# ruff: noqa: UP045

"""
Lightweight helpers to read model metadata hints from cached Hugging Face models.

No external dependencies; YAML front matter is hand-parsed leniently.

Priority rules (Issue #31):
- Tokenizer config: if tokenizer_config.json has chat_template -> Type = chat
- README.md front matter (YAML):
  - tags contains "mlx" OR library_name == "mlx" -> Framework = MLX
  - pipeline_tag == text-generation OR tags contain chat/instruct -> Type = chat
  - pipeline_tag == sentence-similarity OR tags contain embedding -> Type = embedding
- Fallback for framework/type remains in cache_utils
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _latest_snapshot_dir(model_base_dir: Path) -> Optional[Path]:
    """Return latest snapshot directory for a cached HF model base dir."""
    try:
        snaps = (model_base_dir / "snapshots")
        if not snaps.exists():
            return None
        candidates = [d for d in snaps.iterdir() if d.is_dir()]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)
    except Exception:
        return None


def _lenient_yaml_front_matter(text: str) -> Dict[str, Any]:
    """Very small YAML front matter parser for the fields we need.

    Supports forms:
    ---
    tags: [mlx, chat]
    pipeline_tag: text-generation
    library_name: mlx
    ---

    And list style:
    tags:
      - mlx
      - chat
    """
    start = text.find("\n---\n")
    # Accept files starting directly with '---' too
    if text.startswith('---'):
        start = 0
    elif start >= 0:
        start = start + 1  # move to line start
    else:
        # Try at very beginning without newline
        start = 0 if text[:3] == '---' else -1
    if start != 0:
        return {}

    # Find closing '---' after start
    end = text.find('\n---', 3)
    if end == -1:
        return {}
    header = text[3:end] if text.startswith('---') else text[start + 3:end]

    # Normalize lines
    lines = [ln.strip() for ln in header.splitlines() if ln.strip()]

    data: Dict[str, Any] = {}
    current_key: Optional[str] = None
    list_acc: List[str] = []

    def flush_list():
        nonlocal list_acc, current_key
        if current_key is not None and list_acc:
            data[current_key] = list_acc[:]
        list_acc = []

    for ln in lines:
        if ln.startswith('- '):
            # list item under current_key
            val = ln[2:].strip().strip('"\'')
            if current_key is not None:
                list_acc.append(val)
            continue
        # key: value or key: [a, b]
        if ':' in ln:
            # Close any previous list
            flush_list()
            key, val = ln.split(':', 1)
            key = key.strip()
            val = val.strip()
            current_key = key
            if not val:
                # expect multi-line list next
                data.setdefault(key, [])
                continue
            # Inline list [a, b]
            if val.startswith('[') and val.endswith(']'):
                inner = val[1:-1].strip()
                items = [] if not inner else [it.strip().strip('"\'') for it in inner.split(',')]
                data[key] = [x for x in items if x]
                continue
            # Scalar
            data[key] = val.strip('"\'')
            continue
        # Non key-value, ignore
    # Flush last list
    flush_list()
    return data


def read_readme_front_matter(model_base_dir: Path) -> Tuple[Optional[List[str]], Optional[str], Optional[str]]:
    """Read README.md front matter and extract tags, pipeline_tag, library_name.

    Returns (tags, pipeline_tag, library_name) with lowercase normalization where applicable.
    Any read/parse error results in (None, None, None).
    """
    try:
        snap = _latest_snapshot_dir(model_base_dir)
        if not snap:
            return None, None, None
        readme = snap / 'README.md'
        if not readme.exists():
            return None, None, None
        text = readme.read_text(encoding='utf-8', errors='ignore')
        fm = _lenient_yaml_front_matter(text)
        if not fm:
            return None, None, None
        tags = fm.get('tags')
        if isinstance(tags, list):
            tags = [str(t).strip().lower() for t in tags if str(t).strip()]
        else:
            tags = None
        pipeline = fm.get('pipeline_tag')
        pipeline = str(pipeline).strip().lower() if pipeline else None
        lib = fm.get('library_name')
        lib = str(lib).strip().lower() if lib else None
        return tags, pipeline, lib
    except Exception:
        return None, None, None


def tokenizer_has_chat_template(model_base_dir: Path) -> bool:
    """Check tokenizer_config.json for a non-empty 'chat_template' field in latest snapshot."""
    try:
        snap = _latest_snapshot_dir(model_base_dir)
        if not snap:
            return False
        tk = snap / 'tokenizer_config.json'
        if not tk.exists():
            return False
        with open(tk, encoding='utf-8') as f:
            data = json.load(f)
        tmpl = data.get('chat_template')
        return bool(tmpl and isinstance(tmpl, str) and tmpl.strip())
    except Exception:
        return False

