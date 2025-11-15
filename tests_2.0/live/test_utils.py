"""Shared utilities for live E2E tests (ADR-011).

Provides:
- Portfolio discovery functions (reused from test_stop_tokens_live.py)
- RAM gating utilities
- Common test constants
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Import portfolio discovery infrastructure from test_stop_tokens_live.py
_parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_parent_dir))

try:
    from test_stop_tokens_live import (
        discover_mlx_models_in_user_cache,
        get_safe_ram_budget_gb,
        get_system_ram_gb,
        should_skip_model,
        TEST_MODELS,
    )
finally:
    sys.path.remove(str(_parent_dir))


# Re-export for convenience
__all__ = [
    "discover_mlx_models_in_user_cache",
    "get_safe_ram_budget_gb",
    "get_system_ram_gb",
    "should_skip_model",
    "TEST_MODELS",
    "TEST_PROMPT",
    "MAX_TOKENS",
    "TEST_TEMPERATURE",
]


# Standard test constants (shared across all E2E tests)
TEST_PROMPT = "Write one sentence about cats."
MAX_TOKENS = 50
TEST_TEMPERATURE = 0.0  # Deterministic sampling for reproducible tests
