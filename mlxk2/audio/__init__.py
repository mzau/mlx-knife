"""Audio support module for mlxk2.

Carries the vendored tiktoken-based Whisper tokenizer that bridges
mlx-audio#645 — see whisper_tokenizer.py for the provenance and the
retirement condition.
"""

from .whisper_tokenizer import Tokenizer, get_tokenizer

__all__ = ["Tokenizer", "get_tokenizer"]
