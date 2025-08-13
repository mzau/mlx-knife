# mlx_knife/mlx_runner.py
"""
Enhanced MLX model runner with direct API integration.
Provides ollama-like run experience with streaming and interactive chat.
"""

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_repetition_penalty, make_sampler


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
        """Extract stop tokens from the tokenizer dynamically."""
        self._stop_tokens = set()

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

        # Add common stop tokens that might not be in special tokens
        # but are frequently used across models
        common_stop_tokens = {'</s>', '<|endoftext|>', '<|im_end|>'}

        # Only add common tokens if they decode to themselves (i.e., they're real tokens)
        for token in common_stop_tokens:
            try:
                # Try to encode and decode to verify it's a real token
                ids = self.tokenizer.encode(token, add_special_tokens=False)
                if ids and len(ids) == 1:  # Single token ID means it's a special token
                    decoded = self.tokenizer.decode(ids)
                    if decoded == token:
                        self._stop_tokens.add(token)
            except:
                pass

        # Remove any None values
        self._stop_tokens.discard(None)

        # Convert to list for easier use
        self._stop_tokens = list(self._stop_tokens)

        if self._stop_tokens and self.verbose:
            print(f"Stop tokens: {self._stop_tokens}")

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

    def generate_streaming(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        repetition_context_size: int = 20,
        use_chat_template: bool = True,
    ) -> Iterator[str]:
        """Generate text with streaming output.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeated tokens
            repetition_context_size: Context size for repetition penalty
            
        Yields:
            Generated tokens as they are produced
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")

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
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors if logits_processors else None,
        )

        # Collect tokens and yield text
        generated_tokens = []
        previous_decoded = ""

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
                # Filter out stop tokens that might appear as text
                # Use dynamically detected stop tokens if available
                stop_tokens = self._stop_tokens if self._stop_tokens else []

                for stop_token in stop_tokens:
                    if stop_token in new_text:
                        # Yield everything before the stop token
                        pre_stop = new_text.split(stop_token)[0]
                        if pre_stop:
                            yield pre_stop
                        return  # Stop generation

                yield new_text
                tokens_generated += 1

            # Check for EOS token - don't yield it
            if token_id == self.tokenizer.eos_token_id:
                break

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
    ) -> str:
        """Generate text in batch mode (non-streaming).
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeated tokens
            repetition_context_size: Context size for repetition penalty
            
        Returns:
            Generated text
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")

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
            max_tokens=max_tokens,
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
    ):
        """Run an interactive chat session.
        
        Args:
            system_prompt: Optional system prompt to prepend
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeated tokens
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

                # Format conversation for the model
                # This is a simple format - models may need specific chat templates
                prompt = self._format_conversation(conversation_history)

                # Generate response with streaming
                print("\nAssistant: ", end="", flush=True)

                response_tokens = []
                for token in self.generate_streaming(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                ):
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

    def _format_conversation(self, messages: list) -> str:
        """Format conversation history into a prompt.
        
        This is a simple format. Different models may need different templates.
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
                )
                return None

            # Single prompt mode
            if verbose:
                print(f"\nPrompt: {prompt}\n")
                print("Response: ", end="", flush=True)

            if stream:
                # Streaming generation
                response_tokens = []
                for token in runner.generate_streaming(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_chat_template=use_chat_template,
                ):
                    print(token, end="", flush=True)
                    response_tokens.append(token)

                response = "".join(response_tokens)
            else:
                # Batch generation
                response = runner.generate_batch(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_chat_template=use_chat_template,
                )
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
