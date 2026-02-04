"""Stub for mlx_lm.utils - provides minimal _get_classes for runtime checks."""


# Supported model types that would return a valid class
# Mirror the real mlx-lm MODEL_REMAPPING keys
SUPPORTED_MODEL_TYPES = frozenset({
    "llama", "mistral", "phi", "phi3", "qwen", "qwen2", "gemma", "gemma2",
    "llava", "pixtral", "qwen2_vl", "phi3_v", "paligemma", "idefics", "smolvlm",
    "whisper", "starcoder", "starcoder2", "codellama", "deepseek",
    # Add more as needed for tests
})


class _DummyModelClass:
    """Dummy model class returned by _get_classes stub."""
    pass


def _get_classes(config):
    """Stub for mlx_lm.utils._get_classes.

    Returns (model_class, model_args_class) tuple.
    Returns (None, None) for unsupported model_types.
    """
    model_type = config.get("model_type", "").lower() if isinstance(config, dict) else ""

    if model_type in SUPPORTED_MODEL_TYPES:
        return _DummyModelClass, _DummyModelClass

    # Unsupported model type
    return None, None
