from __future__ import annotations

import json
import os
from typing import Optional


def get_model_context_length(model_path: str) -> int:
    """Extract max_position_embeddings from model config with safe fallbacks.

    Supports both flat configs (text-only models) and nested configs (multimodal models
    like Mistral3, Pixtral with text_config/vision_config).

    Returns a sensible default (4096) if the config is missing or malformed.
    """
    config_path = os.path.join(model_path, "config.json")
    try:
        with open(config_path) as f:
            config = json.load(f)

        context_keys = [
            "max_position_embeddings",
            "n_positions",
            "context_length",
            "max_sequence_length",
            "seq_len",
        ]

        # Priority 1: Try top-level keys (text-only models)
        for key in context_keys:
            if key in config:
                value = config[key]
                if isinstance(value, int) and value > 0:
                    return value
                if isinstance(value, str) and value.isdigit():
                    parsed = int(value)
                    if parsed > 0:
                        return parsed

        # Priority 2: Try text_config for multimodal models (Mistral3, Pixtral)
        # These models have separate text_config and vision_config
        if "text_config" in config and isinstance(config["text_config"], dict):
            text_config = config["text_config"]
            for key in context_keys:
                if key in text_config:
                    value = text_config[key]
                    if isinstance(value, int) and value > 0:
                        return value
                    if isinstance(value, str) and value.isdigit():
                        parsed = int(value)
                        if parsed > 0:
                            return parsed

        return 4096
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 4096


def calculate_dynamic_max_tokens(context_length: Optional[int], server_mode: bool = True) -> int:
    """Compute an effective generation limit based on context and mode."""
    if not context_length or context_length <= 0:
        return 2048
    return context_length // 2 if server_mode else context_length

