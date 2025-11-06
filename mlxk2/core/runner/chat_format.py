from __future__ import annotations

from typing import Any, Dict, List


def apply_user_prompt(tokenizer: Any, prompt: str, use_chat_template: bool = True) -> str:
    """Format a single user prompt using the tokenizer's chat template if present."""
    template = getattr(tokenizer, 'chat_template', None)
    if use_chat_template and isinstance(template, str) and template:
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fall back to raw prompt if chat template application fails
            pass
    return prompt


def format_conversation(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    """Format conversation history into a prompt using chat template if available.

    Falls back to legacy Human/Assistant formatting when no chat template exists.
    """
    template = getattr(tokenizer, 'chat_template', None)
    if isinstance(template, str) and template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fall back to legacy format if template application fails
            pass

    formatted_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            formatted_parts.append(f"System: {content}")
        elif role == "user":
            formatted_parts.append(f"Human: {content}")
        elif role == "assistant":
            formatted_parts.append(f"Assistant: {content}")
    return "\n\n".join(formatted_parts) + "\n\nAssistant: "

