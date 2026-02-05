"""Audio support module for mlxk2.

Contains workarounds for mlx-audio regressions, specifically:
- Whisper tokenizer (tiktoken-based) for Issue #479
"""

from .whisper_tokenizer import Tokenizer, get_tokenizer

__all__ = ["Tokenizer", "get_tokenizer"]
