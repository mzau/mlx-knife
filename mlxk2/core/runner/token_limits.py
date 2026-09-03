"""Generation budget: window, prompt, ceiling — and how a generation ended.

One rule for every surface: ``G = min(X, W − P)``. ``X`` is the documented ceiling
(``DEFAULT_MAX_TOKENS`` or the caller's ``max_tokens``), ``W`` the model's context
window, ``P`` the prompt after the chat template. ``max_tokens`` counts *generated*
tokens (mlx-lm semantics), so without the ``W − P`` term prompt plus output could
exceed the window the model was trained for (#66).
"""

from __future__ import annotations

import json
import os
from typing import Optional

# The tool's generation ceiling per single request, text and vision. A statement
# about mlx-knife, not about any model: "one generation emits at most N tokens;
# above that, pass --max-tokens". Lives here once; both shells import it.
DEFAULT_MAX_TOKENS = 32768
DEFAULT_MAX_TOKENS_VISION = 2048

# How the last generation ended, as the runner records it (``last_finish_reason``).
FINISH_STOP = "stop"                # EOS id or a stop string from the model
FINISH_LENGTH = "length"            # the budget ran out before the model stopped
FINISH_INTERRUPTED = "interrupted"  # Ctrl-C / server shutdown cut it


class ContextLengthExceeded(ValueError):
    """The prompt fills the context window; nothing is left to generate.

    Raised before any token is produced (pre-execution reject, ADR-024 shape).
    Carries the two numbers a client needs to shorten its prompt.
    """

    def __init__(self, prompt_tokens: int, context_length: int):
        self.prompt_tokens = prompt_tokens
        self.context_length = context_length
        super().__init__(
            f"Prompt is {prompt_tokens} tokens, but the model's context window is "
            f"{context_length} tokens; nothing is left to generate. Shorten the prompt."
        )


def _positive_int(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) > 0:
        return int(value)
    return None


def get_model_context_length(model_path: str) -> Optional[int]:
    """Read the context window from ``config.json``; ``None`` when it is not stated.

    Looks at the top level first (text-only models), then in ``text_config``
    (multimodal models such as Mistral3 or Pixtral). An unknown window is
    reported as ``None``, never invented: with ``W − P`` a made-up number would
    become a guard over the prompt and reject on made-up grounds.
    """
    config_path = os.path.join(model_path, "config.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(config, dict):
        return None

    context_keys = (
        "max_position_embeddings",
        "n_positions",
        "context_length",
        "max_sequence_length",
        "seq_len",
    )
    text_config = config.get("text_config")
    for section in (config, text_config if isinstance(text_config, dict) else {}):
        for key in context_keys:
            if key in section:
                parsed = _positive_int(section[key])
                if parsed is not None:
                    return parsed
    return None


def resolve_generation_budget(
    requested: Optional[int],
    context_length: Optional[int],
    prompt_tokens: int,
) -> int:
    """``min(X, W − P)`` — the number of tokens a generation may produce.

    ``requested`` is the caller's ``max_tokens`` (``None`` → ``DEFAULT_MAX_TOKENS``);
    every value is clamped to what the window still holds, an explicit one too.
    With ``context_length`` unknown there is no guard, only the ceiling.

    Raises ``ContextLengthExceeded`` when ``W − P ≤ 0`` and ``ValueError`` for a
    ``requested`` below 1: mlx-lm treats every negative ``max_tokens`` as
    unbounded and ``0`` produces nothing, so neither may reach it.
    """
    if requested is not None and requested < 1:
        raise ValueError(f"max_tokens must be at least 1 (got {requested})")
    ceiling = requested if requested is not None else DEFAULT_MAX_TOKENS
    if not context_length or context_length <= 0:
        return ceiling
    room = context_length - prompt_tokens
    if room <= 0:
        raise ContextLengthExceeded(prompt_tokens, context_length)
    return min(ceiling, room)


def reported_finish_reason(reason: Optional[str]) -> Optional[str]:
    """The runner's stop reason as every surface reports it.

    ``stop`` and ``length`` pass through — the OpenAI enum values. ``interrupted``
    is reported as ``stop``: that is what the streaming interrupt marker has
    always carried, and giving the interrupt its own wire value is a contract
    decision, not a side effect of this mapping. Anything else — a runner that
    ran no generation, a mock without the attribute — stays ``None``: absent,
    not invented.
    """
    if reason in (FINISH_STOP, FINISH_LENGTH):
        return reason
    if reason == FINISH_INTERRUPTED:
        return FINISH_STOP
    return None
