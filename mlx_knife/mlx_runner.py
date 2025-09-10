# mlx_knife/mlx_runner.py
"""
Enhanced MLX model runner with direct API integration.
Provides ollama-like run experience with streaming and interactive chat.
"""

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_repetition_penalty, make_sampler

from .reasoning_utils import ReasoningExtractor, StreamingReasoningParser


def get_model_context_length(model_path: str) -> int:
    """Extract max_position_embeddings from model config.
    
    Args:
        model_path: Path to the MLX model directory
        
    Returns:
        Maximum context length for the model (defaults to 4096 if not found)
    """
    config_path = os.path.join(model_path, "config.json")
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        # Try various common config keys for context length
        context_keys = [
            "max_position_embeddings",
            "n_positions",
            "context_length",
            "max_sequence_length",
            "seq_len"
        ]
        
        for key in context_keys:
            if key in config:
                return config[key]
                
        # If no context length found, return reasonable default
        return 4096
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        # Return default if config can't be read
        return 4096


class MLXRunner:
    """Direct MLX model runner with streaming and interactive capabilities."""

    def __init__(self, model_path: str, adapter_path: Optional[str] = None, verbose: bool = False):
        """Initialize the runner with a model.
        
        Args:
            model_path: Path to the MLX model directory
            adapter_path: Optional path to LoRA adapter
            verbose: Show detailed output
        """
        self.model_path = Path(model_path)
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self._memory_baseline = None
        self._stop_tokens = None  # Will be populated from tokenizer
        self._message_end_tokens = None  # Message-end tokens (e.g., <|end|> for MXFP4)
        self._chat_stop_tokens = None  # Chat-specific stop tokens
        self._context_length = None  # Will be populated from model config
        self._is_reasoning_model = False  # Whether model uses reasoning (MXFP4)
        self._reasoning_start = None  # Reasoning start marker
        self._reasoning_end = None  # Reasoning end marker
        self._final_start = None  # Final answer start marker
        self.verbose = verbose
        self._model_loaded = False
        self._context_entered = False  # Prevent nested context usage

    def __enter__(self):
        """Context manager entry - loads the model."""
        if self._context_entered:
            raise RuntimeError("MLXRunner context manager cannot be entered multiple times")
        
        self._context_entered = True
        try:
            self.load_model()
            return self
        except Exception:
            # If load_model fails, ensure cleanup happens
            self._context_entered = False
            self.cleanup()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleans up the model."""
        self._context_entered = False
        self.cleanup()
        return False  # Don't suppress exceptions

    def load_model(self):
        """Load the MLX model and tokenizer."""
        if self._model_loaded:
            if self.verbose:
                print("Model already loaded, skipping...")
            return

        if self.verbose:
            print(f"Loading model from {self.model_path}...")
        start_time = time.time()

        # Capture baseline memory before loading
        try:
            mx.clear_cache()
        except Exception:
            pass  # Continue even if cache clear fails
        self._memory_baseline = mx.get_active_memory() / 1024**3

        try:
            # Load model and tokenizer
            self.model, self.tokenizer = load(
                str(self.model_path),
                adapter_path=self.adapter_path
            )

            load_time = time.time() - start_time
            current_memory = mx.get_active_memory() / 1024**3
            model_memory = current_memory - self._memory_baseline

            if self.verbose:
                print(f"Model loaded in {load_time:.1f}s")
                print(f"Memory: {model_memory:.1f}GB model, {current_memory:.1f}GB total")

            # Extract stop tokens from tokenizer
            self._extract_stop_tokens()
            
            # Extract context length from model config
            self._context_length = get_model_context_length(str(self.model_path))
            
            if self.verbose:
                print(f"Model context length: {self._context_length} tokens")
                
            self._model_loaded = True
            
        except Exception as e:
            # Ensure partial state is cleaned up on failure
            self.model = None
            self.tokenizer = None
            self._stop_tokens = None
            self._model_loaded = False
            # Clear any memory that might have been allocated
            mx.clear_cache()
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}") from e

    def _extract_stop_tokens(self):
        """Extract stop tokens from the tokenizer dynamically.
        
        This method identifies ALL tokens that should stop generation:
        1. Official EOS token from tokenizer config
        2. Message-end tokens from training (e.g., <|end|> for MXFP4)
        3. Common stop tokens across models
        """
        self._stop_tokens = set()
        self._message_end_tokens = set()  # Tokens that end messages but not conversations

        # Primary source: eos_token
        eos_token = getattr(self.tokenizer, 'eos_token', None)
        if eos_token:
            self._stop_tokens.add(eos_token)

        # Also check pad_token if it's different from eos_token
        pad_token = getattr(self.tokenizer, 'pad_token', None)
        if pad_token and pad_token != eos_token:
            self._stop_tokens.add(pad_token)

        # Check additional_special_tokens
        if hasattr(self.tokenizer, 'additional_special_tokens'):
            for token in self.tokenizer.additional_special_tokens:
                if token and isinstance(token, str):
                    # Only add tokens that look like stop/end tokens
                    if any(keyword in token.lower() for keyword in ['end', 'stop', 'eot']):
                        self._stop_tokens.add(token)
        
        # MLX-LM 0.27.0+: Extract tokens from added_tokens_decoder (comprehensive source)
        if hasattr(self.tokenizer, 'added_tokens_decoder'):
            for _token_id, token_info in self.tokenizer.added_tokens_decoder.items():
                if isinstance(token_info, dict) and 'content' in token_info:
                    token_content = token_info['content']
                    if token_content and isinstance(token_content, str):
                        token_lower = token_content.lower()
                        
                        # NOTE: <|end|> is NOT a stop token for MXFP4 models!
                        # It's a separator between reasoning and final answer
                        if token_content == '<|end|>':
                            self._message_end_tokens.add(token_content)
                            # Do NOT add as stop token - let model continue to final answer
                        
                        # Look for tokens that could be end/stop tokens
                        # Expanded patterns for MLX-LM 0.27.0 token varieties
                        # EXCLUDE <|end|> for MXFP4 models as it's a reasoning separator
                        end_patterns = ['stop', 'eot', 'return', 'finish', 'done', 'im_end']
                        if any(pattern in token_lower for pattern in end_patterns):
                            # Decide if it's a message-end or conversation-end token
                            if 'im_end' in token_lower:
                                self._message_end_tokens.add(token_content)
                            self._stop_tokens.add(token_content)
                        # Special handling for 'end' pattern - more selective
                        elif 'end' in token_lower and token_content != '<|end|>':
                            # Only add non-<|end|> tokens with 'end' in them
                            self._stop_tokens.add(token_content)
                        
                        # Special case: control tokens in |..| format
                        elif token_content.startswith('<|') and token_content.endswith('|>'):
                            # Be inclusive with control tokens that might stop generation
                            if any(pattern in token_lower for pattern in ['end', 'return', 'stop', 'finish']):
                                self._stop_tokens.add(token_content)

        # Model-specific handling based on known patterns
        # Use reasoning_utils for reasoning model detection and patterns
        from .reasoning_utils import ReasoningExtractor
        
        if hasattr(self.tokenizer, 'name_or_path'):
            name_or_path = str(getattr(self.tokenizer, 'name_or_path', '')).lower()
            model_type = ReasoningExtractor.detect_model_type(name_or_path)
            
            if model_type:
                # This is a reasoning model
                self._is_reasoning_model = True
                
                # Get patterns from reasoning_utils
                if model_type in ReasoningExtractor.PATTERNS:
                    markers = ReasoningExtractor.PATTERNS[model_type]['markers']
                    self._reasoning_start = markers.get('reasoning_start')
                    self._reasoning_end = markers.get('reasoning_end')
                    self._final_start = markers.get('final_marker')
                
                # For reasoning models, remove reasoning_end from stop tokens
                if self._reasoning_end:
                    self._stop_tokens.discard(self._reasoning_end)
                
                # Add proper stop token for this model type
                if model_type == 'gpt-oss':
                    if '<|return|>' not in self._stop_tokens:
                        self._stop_tokens.add('<|return|>')
            else:
                self._is_reasoning_model = False
        else:
            self._is_reasoning_model = False

        # Add common stop tokens that might not be in special tokens
        common_stop_tokens = {'</s>', '<|endoftext|>', '<|im_end|>', '<|eot_id|>'}
        
        # Add chat-specific stop tokens to prevent model self-conversations
        # Based on our _format_conversation() format: "Human:" and "Assistant:"
        # Also include "You:" as models might use UI-visible format
        # Include single-letter variations (H:, A:, Y:) that some models use
        chat_stop_tokens = {
            '\nHuman:', '\nAssistant:', '\nYou:',
            '\n\nHuman:', '\n\nAssistant:', '\n\nYou:',
            '\nH:', '\nA:', '\nY:',  # Single-letter variations
            '\n\nH:', '\n\nA:', '\n\nY:'
        }

        # Add common stop tokens only if they decode to themselves (i.e., they're single tokens)
        for token in common_stop_tokens:
            try:
                # Try to encode and decode to verify it's a real single token
                ids = self.tokenizer.encode(token, add_special_tokens=False)
                if ids and len(ids) == 1:  # Single token ID means it's a special token
                    decoded = self.tokenizer.decode(ids)
                    if decoded == token:
                        self._stop_tokens.add(token)
            except:
                pass
        
        # Store chat stop tokens separately - only used in interactive chat mode
        # This prevents stopping mid-story when user asks for dialogues
        self._chat_stop_tokens = list(chat_stop_tokens)

        # Remove any None values
        self._stop_tokens.discard(None)
        self._message_end_tokens.discard(None)

        # Convert to list for easier use
        self._stop_tokens = list(self._stop_tokens)
        self._message_end_tokens = list(self._message_end_tokens)

        if self.verbose:
            if self._stop_tokens:
                print(f"Stop tokens: {self._stop_tokens}")
            if self._message_end_tokens:
                print(f"Message end tokens: {self._message_end_tokens}")

    def cleanup(self):
        """Clean up model resources and clear GPU memory.
        
        This method is safe to call multiple times and handles partial state cleanup.
        """
        if self.verbose and self._model_loaded:
            memory_before = mx.get_active_memory() / 1024**3
            print(f"Cleaning up model (memory before: {memory_before:.1f}GB)...")

        # Always clean up, even if model wasn't fully loaded
        self.model = None
        self.tokenizer = None
        self._stop_tokens = None
        self._message_end_tokens = None
        self._chat_stop_tokens = None
        self._context_length = None
        self._is_reasoning_model = False
        self._reasoning_start = None
        self._reasoning_end = None
        self._final_start = None
        self._model_loaded = False

        # Force garbage collection and clear MLX cache
        import gc
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass  # Continue cleanup even if cache clear fails

        if self.verbose:
            memory_after = mx.get_active_memory() / 1024**3
            if 'memory_before' in locals():
                memory_freed = memory_before - memory_after
                print(f"Cleanup complete (memory after: {memory_after:.1f}GB, freed: {memory_freed:.1f}GB)")
            else:
                print(f"Cleanup complete (memory after: {memory_after:.1f}GB)")

    def get_effective_max_tokens(self, requested_tokens: Optional[int], interactive: bool = False) -> int:
        """Get effective max tokens based on model context and usage mode.
        
        Args:
            requested_tokens: The requested max tokens (None if user didn't specify --max-tokens)
            interactive: True if this is interactive mode (gets full context length)
            
        Returns:
            Effective max tokens to use
        """
        if not self._context_length:
            # Fallback when context length is unknown
            fallback = 4096 if interactive else 2048
            if self.verbose:
                if requested_tokens is None:
                    print(f"[WARNING] Model context length unknown, using fallback: {fallback} tokens")
                else:
                    print(f"[WARNING] Model context length unknown, using user specified: {requested_tokens} tokens")
            return requested_tokens if requested_tokens is not None else fallback
            
        if interactive:
            if requested_tokens is None:
                # User didn't specify --max-tokens: use full model context
                return self._context_length
            else:
                # User specified --max-tokens explicitly: respect their choice but cap at context
                return min(requested_tokens, self._context_length)
        else:
            # Server/batch mode uses half context length for DoS protection
            server_limit = self._context_length // 2
            return min(requested_tokens or server_limit, server_limit)

    def generate_streaming(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        repetition_context_size: int = 20,
        use_chat_template: bool = True,
        use_chat_stop_tokens: bool = False,
        interactive: bool = False,
        hide_reasoning: bool = False,
    ) -> Iterator[str]:
        """Generate text with streaming output.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeated tokens
            repetition_context_size: Context size for repetition penalty
            use_chat_template: Apply tokenizer's chat template if available
            use_chat_stop_tokens: Include chat turn markers as stop tokens (for interactive mode)
            interactive: True if this is interactive mode (affects token limits)
            
        Yields:
            Generated tokens as they are produced
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Initialize reasoning parser if this is a reasoning model
        reasoning_parser = None
        if self._is_reasoning_model:
            model_type = ReasoningExtractor.detect_model_type(
                getattr(self.tokenizer, 'name_or_path', '') or ''
            )
            reasoning_parser = StreamingReasoningParser(model_type, hide_reasoning=hide_reasoning)

        # Apply context-aware token limits
        effective_max_tokens = self.get_effective_max_tokens(max_tokens, interactive)

        # Apply chat template if available and requested
        if use_chat_template and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Tokenize the prompt
        prompt_tokens = self.tokenizer.encode(formatted_prompt)
        prompt_array = mx.array(prompt_tokens)

        # Track generation metrics
        start_time = time.time()
        tokens_generated = 0

        # Create sampler with our parameters
        sampler = make_sampler(temp=temperature, top_p=top_p)

        # Create repetition penalty processor if needed
        logits_processors = []
        if repetition_penalty > 1.0:
            logits_processors.append(
                make_repetition_penalty(repetition_penalty, repetition_context_size)
            )

        # Generate tokens one by one for streaming
        generator = generate_step(
            prompt=prompt_array,
            model=self.model,
            max_tokens=effective_max_tokens,
            sampler=sampler,
            logits_processors=logits_processors if logits_processors else None,
        )

        # Collect tokens and yield text
        generated_tokens = []
        previous_decoded = ""
        accumulated_response = ""  # Track full response for stop token detection

        # Keep a sliding window of recent tokens for context
        context_window = 10  # Decode last N tokens for proper spacing

        for token, _ in generator:
            # Token might be an array or an int
            token_id = token.item() if hasattr(token, 'item') else token
            generated_tokens.append(token_id)

            # Use a sliding window approach for efficiency
            start_idx = max(0, len(generated_tokens) - context_window)
            window_tokens = generated_tokens[start_idx:]

            # Decode the window
            window_text = self.tokenizer.decode(window_tokens)

            # Figure out what's new
            if start_idx == 0:
                # We're still within the context window
                if window_text.startswith(previous_decoded):
                    new_text = window_text[len(previous_decoded):]
                else:
                    new_text = self.tokenizer.decode([token_id])
                previous_decoded = window_text
            else:
                # We're beyond the context window, just decode the last token with context
                # This is approximate but should preserve spaces
                new_text = self.tokenizer.decode(window_tokens)
                if len(window_tokens) > 1:
                    prefix = self.tokenizer.decode(window_tokens[:-1])
                    if new_text.startswith(prefix):
                        new_text = new_text[len(prefix):]
                    else:
                        new_text = self.tokenizer.decode([token_id])

            if new_text:
                # Update accumulated response for stop token checking
                accumulated_response += new_text
                
                # Filter out stop tokens with priority: native first, then chat fallback
                # Check native stop tokens FIRST in accumulated response (highest priority)
                native_stop_tokens = self._stop_tokens if self._stop_tokens else []
                for stop_token in native_stop_tokens:
                    if stop_token in accumulated_response:
                        # Find the stop token position and yield everything before it
                        stop_pos = accumulated_response.find(stop_token)
                        # Calculate what text came before the stop token
                        text_before_stop = accumulated_response[:stop_pos]
                        # Calculate how much of that is new (not previously yielded)
                        previously_yielded_length = len(accumulated_response) - len(new_text)
                        if len(text_before_stop) > previously_yielded_length:
                            # Yield only the new part before stop token
                            new_part_before_stop = text_before_stop[previously_yielded_length:]
                            if new_part_before_stop:
                                if reasoning_parser:
                                    # Process through reasoning parser for formatting
                                    for formatted_token in reasoning_parser.process_token(new_part_before_stop):
                                        yield formatted_token
                                else:
                                    yield new_part_before_stop
                        return  # Stop generation without yielding stop token
                
                # Only check chat stop tokens if no native stop token found (fallback)
                if use_chat_stop_tokens and self._chat_stop_tokens:
                    for stop_token in self._chat_stop_tokens:
                        if stop_token in accumulated_response:
                            # Find the stop token position and yield everything before it
                            stop_pos = accumulated_response.find(stop_token)
                            # Calculate what text came before the stop token
                            text_before_stop = accumulated_response[:stop_pos]
                            # Calculate how much of that is new (not previously yielded)
                            previously_yielded_length = len(accumulated_response) - len(new_text)
                            if len(text_before_stop) > previously_yielded_length:
                                # Yield only the new part before stop token
                                new_part_before_stop = text_before_stop[previously_yielded_length:]
                                if new_part_before_stop:
                                    if reasoning_parser:
                                        # Process through reasoning parser for formatting
                                        for formatted_token in reasoning_parser.process_token(new_part_before_stop):
                                            yield formatted_token
                                    else:
                                        yield new_part_before_stop
                            return  # Stop generation without yielding stop token

                # No stop token found, process the new text
                if reasoning_parser:
                    # Process through reasoning parser for formatting
                    for formatted_token in reasoning_parser.process_token(new_text):
                        yield formatted_token
                else:
                    # Normal streaming for non-reasoning models
                    yield new_text
                tokens_generated += 1

            # Check for EOS token - don't yield it
            if token_id == self.tokenizer.eos_token_id:
                break

        # Finalize reasoning parser if used
        if reasoning_parser:
            yield from reasoning_parser.finalize()
        
        # Print generation statistics if verbose
        if self.verbose:
            generation_time = time.time() - start_time
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            print(f"\n\nGenerated {tokens_generated} tokens in {generation_time:.1f}s ({tokens_per_second:.1f} tokens/s)")

    def generate_batch(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        repetition_context_size: int = 20,
        use_chat_template: bool = True,
        interactive: bool = False,
    ) -> str:
        """Generate text in batch mode (non-streaming).
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeated tokens
            repetition_context_size: Context size for repetition penalty
            use_chat_template: Apply tokenizer's chat template if available
            interactive: True if this is interactive mode (affects token limits)
            
        Returns:
            Generated text
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Apply context-aware token limits
        effective_max_tokens = self.get_effective_max_tokens(max_tokens, interactive)

        # Apply chat template if available and requested
        if use_chat_template and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        start_time = time.time()

        # Tokenize the prompt
        prompt_tokens = self.tokenizer.encode(formatted_prompt)
        prompt_array = mx.array(prompt_tokens)

        # Create sampler with our parameters
        sampler = make_sampler(temp=temperature, top_p=top_p)

        # Create repetition penalty processor if needed
        logits_processors = []
        if repetition_penalty > 1.0:
            logits_processors.append(
                make_repetition_penalty(repetition_penalty, repetition_context_size)
            )

        # Generate all tokens at once
        generated_tokens = []
        all_tokens = list(prompt_tokens)  # Keep prompt for proper decoding

        generator = generate_step(
            prompt=prompt_array,
            model=self.model,
            max_tokens=effective_max_tokens,
            sampler=sampler,
            logits_processors=logits_processors if logits_processors else None,
        )

        for token, _ in generator:
            # Token might be an array or an int
            token_id = token.item() if hasattr(token, 'item') else token
            generated_tokens.append(token_id)
            all_tokens.append(token_id)

            # Check for EOS token - don't yield it
            if token_id == self.tokenizer.eos_token_id:
                break

        # Decode all tokens together for proper spacing
        full_response = self.tokenizer.decode(all_tokens)

        # Remove the prompt part
        if full_response.startswith(formatted_prompt):
            response = full_response[len(formatted_prompt):]
        else:
            # Fallback: just decode generated tokens
            response = self.tokenizer.decode(generated_tokens)

        # Apply end-token filtering (same logic as streaming mode for Issue #20)
        response = self._filter_end_tokens_from_response(response, use_chat_stop_tokens=False)
        
        # Format reasoning models output
        response = self._format_reasoning_response(response)

        generation_time = time.time() - start_time

        # Count tokens for statistics
        if self.verbose:
            tokens_generated = len(generated_tokens)
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            print(f"\nGenerated {tokens_generated} tokens in {generation_time:.1f}s ({tokens_per_second:.1f} tokens/s)")

        return response

    def interactive_chat(
        self,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        use_chat_template: bool = True,
    ):
        """Run an interactive chat session.
        
        Args:
            system_prompt: Optional system prompt to prepend
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeated tokens
            use_chat_template: Use tokenizer's chat template if available
        """
        print("Starting interactive chat. Type 'exit' or 'quit' to end.\n")

        conversation_history = []
        if system_prompt:
            conversation_history.append({"role": "system", "content": system_prompt})

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye!")
                    break

                if not user_input:
                    continue

                # Add user message to history
                conversation_history.append({"role": "user", "content": user_input})

                # Format conversation for the model using chat template if available
                prompt = self._format_conversation(conversation_history, use_chat_template=use_chat_template)

                # Generate response with streaming
                print("\nAssistant: ", end="", flush=True)

                response_tokens = []
                for token in self.generate_streaming(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_chat_template=False,  # Already applied in _format_conversation
                    use_chat_stop_tokens=True,  # Enable chat stop tokens in interactive mode
                    interactive=True,  # Enable full context length for interactive mode
                ):
                    # Stream all tokens directly (already formatted by generate_streaming)
                    print(token, end="", flush=True)
                    response_tokens.append(token)

                # Add assistant response to history
                assistant_response = "".join(response_tokens).strip()
                conversation_history.append({"role": "assistant", "content": assistant_response})

                print()  # New line after response

            except KeyboardInterrupt:
                print("\n\nChat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                continue

    def _format_conversation(self, messages: list, use_chat_template: bool = True) -> str:
        """Format conversation history into a prompt.
        
        Uses the tokenizer's chat template if available, otherwise falls back
        to the legacy Human:/Assistant: format for compatibility.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            use_chat_template: Whether to use chat template if available
            
        Returns:
            Formatted conversation string
        """
        # Try to use native chat template if available
        if use_chat_template and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            try:
                # Apply the tokenizer's chat template
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted_prompt
            except Exception as e:
                # If chat template fails, fall back to legacy format
                if self.verbose:
                    print(f"[WARNING] Chat template failed, using legacy format: {e}")
        
        # Legacy format fallback for compatibility
        return self._legacy_format_conversation(messages)
    
    def _legacy_format_conversation(self, messages: list) -> str:
        """Legacy conversation formatting for backward compatibility.
        
        This format was used in earlier versions and remains as a fallback
        for models without chat templates.
        """
        formatted = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"Human: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")

        # Add prompt for next assistant response
        formatted.append("Assistant:")

        return "\n\n".join(formatted)

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics.
        
        Returns:
            Dictionary with memory statistics in GB
        """
        try:
            current_memory = mx.get_active_memory() / 1024**3
            peak_memory = mx.get_peak_memory() / 1024**3
        except Exception:
            # Return zeros if memory stats unavailable
            current_memory = 0.0
            peak_memory = 0.0

        return {
            "current_gb": current_memory,
            "peak_gb": peak_memory,
            "model_gb": current_memory - self._memory_baseline if self._memory_baseline else 0,
        }

    def _format_reasoning_response(self, response: str) -> str:
        """Format response from reasoning models for better readability.
        
        For MXFP4 models that generate reasoning followed by final answer,
        format it nicely for display.
        """
        if not self._is_reasoning_model:
            return response
            
        # Check if response contains reasoning markers
        if self._reasoning_start in response and self._final_start in response:
            # Extract reasoning and final parts
            try:
                # Split on the reasoning start
                before_reasoning, after_start = response.split(self._reasoning_start, 1)
                
                # Find the reasoning content (until <|end|>)
                if self._reasoning_end in after_start:
                    reasoning_content, after_reasoning = after_start.split(self._reasoning_end, 1)
                    
                    # Find the final answer
                    if self._final_start in after_reasoning:
                        # Extract everything after final marker
                        final_parts = after_reasoning.split(self._final_start, 1)
                        if len(final_parts) > 1:
                            # Remove the <|channel|>final<|message|> marker
                            final_answer = final_parts[1].replace('<|channel|>final<|message|>', '', 1)
                            
                            # Format with clear markers for parsing but minimal visual impact
                            formatted = []
                            formatted.append("\n**[Reasoning]**\n")
                            formatted.append(reasoning_content.strip())
                            formatted.append("\n\n---\n\n**[Answer]**\n")
                            formatted.append(final_answer.strip())
                            
                            return '\n'.join(formatted)
            except Exception:
                # If parsing fails, return original
                pass
        
        # Fallback: just clean up the control tokens
        cleaned = response
        for marker in ['<|channel|>analysis<|message|>', '<|end|>', '<|start|>assistant',
                      '<|channel|>final<|message|>', '<|return|>']:
            cleaned = cleaned.replace(marker, '')
        
        return cleaned.strip()
    
    def _filter_end_tokens_from_response(self, response: str, use_chat_stop_tokens: bool = False) -> str:
        """Filter end tokens from a complete response (batch mode).
        
        This method applies the same filtering logic as the streaming mode
        to ensure consistent behavior between streaming and non-streaming.
        
        Args:
            response: The complete generated response
            use_chat_stop_tokens: Whether to apply chat stop tokens
            
        Returns:
            Response with end tokens filtered out
        """
        # Apply native stop token filtering FIRST (highest priority)
        native_stop_tokens = self._stop_tokens if self._stop_tokens else []
        for stop_token in native_stop_tokens:
            if stop_token in response:
                # Find the stop token position and return everything before it
                stop_pos = response.find(stop_token)
                filtered_response = response[:stop_pos].rstrip()
                if self.verbose:
                    print(f"[DEBUG] Filtered stop token '{stop_token}' at position {stop_pos}")
                return filtered_response
        
        # Only check chat stop tokens if no native stop token found (fallback)
        if use_chat_stop_tokens and self._chat_stop_tokens:
            for stop_token in self._chat_stop_tokens:
                if stop_token in response:
                    # Find the stop token position and return everything before it
                    stop_pos = response.find(stop_token)
                    return response[:stop_pos]
        
        # No stop tokens found, return original response
        return response


def get_gpu_status() -> Dict[str, float]:
    """Independent GPU status check - usable from anywhere.
    
    Returns:
        Dictionary with GPU memory statistics in GB
    """
    return {
        "active_memory_gb": mx.get_active_memory() / 1024**3,
        "peak_memory_gb": mx.get_peak_memory() / 1024**3,
    }


def check_memory_available(required_gb: float) -> bool:
    """Pre-flight check before model loading.
    
    Args:
        required_gb: Required memory in GB
        
    Returns:
        True if memory is likely available (conservative estimate)
    """
    current_memory = mx.get_active_memory() / 1024**3

    # Conservative estimate: assume system has at least 8GB unified memory
    # and we should leave some headroom (2GB) for system processes
    estimated_total = 8.0  # This could be improved by detecting actual system memory
    available = estimated_total - current_memory - 2.0  # 2GB headroom

    return available >= required_gb


def run_model_enhanced(
    model_path: str,
    prompt: Optional[str] = None,
    interactive: bool = False,
    max_tokens: int = 500,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    stream: bool = True,
    use_chat_template: bool = True,
    hide_reasoning: bool = False,
    verbose: bool = False,
) -> Optional[str]:
    """Enhanced run function with direct MLX integration.
    
    Uses context manager pattern for automatic resource cleanup.
    
    Args:
        model_path: Path to the MLX model
        prompt: Input prompt (if None, enters interactive mode)
        interactive: Force interactive mode
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        repetition_penalty: Penalty for repeated tokens
        stream: Whether to stream output
        
    Returns:
        Generated text (in non-interactive mode)
    """
    try:
        with MLXRunner(model_path, verbose=verbose) as runner:
            # Interactive mode
            if interactive or prompt is None:
                runner.interactive_chat(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_chat_template=use_chat_template,
                )
                return None

            # Single prompt mode
            if verbose:
                print(f"\nPrompt: {prompt}\n")
                print("Response: ", end="", flush=True)

            if stream:
                # Streaming generation
                response_tokens = []
                try:
                    for token in runner.generate_streaming(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        use_chat_template=use_chat_template,
                        hide_reasoning=hide_reasoning,
                    ):
                        # Stream all tokens directly (already formatted by generate_streaming)
                        print(token, end="", flush=True)
                        response_tokens.append(token)
                except KeyboardInterrupt:
                    print("\n[INFO] Generation interrupted by user.")
                response = "".join(response_tokens)
            else:
                # Batch generation
                try:
                    response = runner.generate_batch(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        use_chat_template=use_chat_template,
                    )
                except KeyboardInterrupt:
                    print("\n[INFO] Generation interrupted by user.")
                    response = ""
                print(response)

            # Show memory usage if verbose
            if verbose:
                memory_stats = runner.get_memory_usage()
                print(f"\n\nMemory: {memory_stats['model_gb']:.1f}GB model, {memory_stats['current_gb']:.1f}GB total")

            return response

        # Note: cleanup happens automatically due to context manager

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None
