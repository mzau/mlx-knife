from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Set

from ..reasoning import ReasoningExtractor


@dataclass
class StopTokenInfo:
    stop_tokens: List[str]
    chat_stop_tokens: List[str]
    is_reasoning_model: bool
    reasoning_start: Optional[str]
    reasoning_end: Optional[str]
    final_start: Optional[str]


def extract_stop_tokens(tokenizer: Any, verbose: bool = False) -> StopTokenInfo:
    """Extract stop tokens and reasoning markers from a tokenizer.

    This mirrors MLXRunner._extract_stop_tokens logic.
    """
    stop_tokens: Set[str] = set()

    eos_token = getattr(tokenizer, 'eos_token', None)
    if eos_token:
        stop_tokens.add(eos_token)

    pad_token = getattr(tokenizer, 'pad_token', None)
    if pad_token and pad_token != eos_token:
        stop_tokens.add(pad_token)

    additional = getattr(tokenizer, 'additional_special_tokens', None)
    if isinstance(additional, (list, tuple)):
        for token in additional:
            if isinstance(token, str) and token:
                tl = token.lower()
                if any(keyword in tl for keyword in ['end', 'stop', 'eot']):
                    stop_tokens.add(token)

    decoder = getattr(tokenizer, 'added_tokens_decoder', None)
    if isinstance(decoder, dict):
        for _token_id, token_info in decoder.items():
            if isinstance(token_info, dict) and 'content' in token_info:
                token_content = token_info['content']
                if isinstance(token_content, str) and token_content:
                    token_lower = token_content.lower()
                    if token_content == '<|end|>':
                        # Always add <|end|> as stop token (fixes Phi-3-mini with eos_token_id=null)
                        stop_tokens.add(token_content)
                        add_eos_token = getattr(tokenizer, 'add_eos_token', None)
                        if callable(add_eos_token):
                            try:
                                add_eos_token(token_content)
                            except Exception:
                                pass
                        continue
                    end_patterns = ['stop', 'eot', 'return', 'finish', 'done', 'im_end']
                    if any(pattern in token_lower for pattern in end_patterns):
                        stop_tokens.add(token_content)
                    elif 'end' in token_lower and token_content != '<|end|>':
                        stop_tokens.add(token_content)

    # Common stop tokens: add if tokenizer encodes them as a single token and decodes faithfully
    common_stop_tokens = {'</s>', '<|endoftext|>', '<|im_end|>', '<|eot_id|>'}
    for token in common_stop_tokens:
        try:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if ids and len(ids) == 1:
                decoded = tokenizer.decode(ids)
                if decoded == token:
                    stop_tokens.add(token)
        except Exception:
            pass

    is_reasoning_model = False
    reasoning_start: Optional[str] = None
    reasoning_end: Optional[str] = None
    final_start: Optional[str] = None

    if hasattr(tokenizer, 'name_or_path'):
        try:
            name_or_path = str(getattr(tokenizer, 'name_or_path', '')).lower()
        except Exception:
            name_or_path = ''
        model_type = ReasoningExtractor.detect_model_type(name_or_path)

        if model_type:
            is_reasoning_model = True
            if model_type in ReasoningExtractor.PATTERNS:
                markers = ReasoningExtractor.PATTERNS[model_type]['markers']
                reasoning_start = markers.get('reasoning_start')
                reasoning_end = markers.get('reasoning_end')
                final_start = markers.get('final_marker')

            if reasoning_end:
                stop_tokens.discard(reasoning_end)

            if model_type == 'gpt-oss':
                stop_tokens.add('<|return|>')

            if verbose:
                # Keep any print semantics consistent with previous behavior
                pass

    # Chat stop tokens to prevent self-conversations in server mode
    # Only use full-form tokens to avoid false positives in code/markdown
    # (Short forms like '\nH:' can match code comments, labels, Q&A format)
    chat_stop_tokens = [
        '\nHuman:', '\nAssistant:', '\nYou:',
        '\n\nHuman:', '\n\nAssistant:', '\n\nYou:',
    ]

    # Remove None values and normalize to list[str]
    stop_tokens.discard(None)  # type: ignore[arg-type]
    stop_tokens_list = [t for t in stop_tokens if isinstance(t, str) and t]

    return StopTokenInfo(
        stop_tokens=stop_tokens_list,
        chat_stop_tokens=chat_stop_tokens,
        is_reasoning_model=is_reasoning_model,
        reasoning_start=reasoning_start,
        reasoning_end=reasoning_end,
        final_start=final_start,
    )
