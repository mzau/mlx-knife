from __future__ import annotations

from typing import Optional


def format_reasoning_response(
    response: str,
    is_reasoning_model: bool,
    reasoning_start: Optional[str],
    reasoning_end: Optional[str],
    final_start: Optional[str],
    hide_reasoning: bool = False,
) -> str:
    """Format response for reasoning-style models.

    Mirrors MLXRunner._format_reasoning_response behavior without changing semantics.

    Args:
        response: Raw model output
        is_reasoning_model: Whether this is a reasoning model
        reasoning_start: Marker for reasoning section start
        reasoning_end: Marker for reasoning section end
        final_start: Marker for final answer section
        hide_reasoning: If True, only return final answer (skip reasoning section)
    """
    if not is_reasoning_model:
        return response

    if reasoning_start and final_start and reasoning_start in response and final_start in response:
        try:
            before_reasoning, after_start = response.split(reasoning_start, 1)
            if reasoning_end and reasoning_end in after_start:
                reasoning_content, after_reasoning = after_start.split(reasoning_end, 1)
                if final_start in after_reasoning:
                    final_parts = after_reasoning.split(final_start, 1)
                    if len(final_parts) > 1:
                        final_answer = final_parts[1].replace('<|channel|>final<|message|>', '', 1)

                        # If hiding reasoning, return only final answer
                        if hide_reasoning:
                            return final_answer.strip()

                        # Otherwise, format with reasoning section
                        formatted = []
                        formatted.append("\n**[Reasoning]**\n")
                        formatted.append(reasoning_content.strip())
                        formatted.append("\n\n---\n\n**[Answer]**\n")
                        formatted.append(final_answer.strip())
                        return '\n'.join(formatted)
        except Exception:
            pass

    # Fallback cleanup
    cleaned = response
    if reasoning_start:
        cleaned = cleaned.replace(reasoning_start, '')
    if reasoning_end:
        cleaned = cleaned.replace(reasoning_end, '')
    if final_start:
        cleaned = cleaned.replace(final_start, '')

    for marker in ['<|start|>assistant', '<|return|>']:
        cleaned = cleaned.replace(marker, '')

    return cleaned.strip()

