"""MLX Knife - HuggingFace-style cache management for MLX models.

A lightweight, ollama-like CLI for managing and running MLX models on Apple Silicon.
Provides native MLX execution with streaming output and interactive chat capabilities.
"""

__version__ = "1.0.1"
__author__ = "The BROKE team"
__email__ = "broke@gmx.eu"
__license__ = "MIT"
__description__ = "ollama-style CLI for MLX models on Apple Silicon"
__url__ = "https://github.com/mzau/mlx-knife"

# Version tuple for programmatic access (major, minor, patch)
VERSION = (1, 0, 1)  # Simplified for now

# Core functionality imports
from .cache_utils import (
    check_all_models_health,
    check_model_health,
    list_models,
    rm_model,
    show_model,
)
from .hf_download import pull_model
from .mlx_runner import MLXRunner

__all__ = [
    "__version__",
    "list_models",
    "show_model",
    "check_model_health",
    "check_all_models_health",
    "rm_model",
    "pull_model",
    "MLXRunner",
]
