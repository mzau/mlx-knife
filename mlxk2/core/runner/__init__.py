"""
MLX model runner for 2.0 implementation.
Ported from 1.x mlx_knife/mlx_runner.py with 2.0 architecture integration.

Refactor: packaged as mlxk2.core.runner with helper modules for
- token limits, chat formatting, reasoning formatting, and stop tokens.
Behavior is unchanged; public API and patch points are preserved.
"""

import time
import signal
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

from ..cache import get_current_model_cache, hf_to_cache_dir
from ..model_resolution import resolve_model_for_operation
from ..reasoning import ReasoningExtractor, StreamingReasoningParser
from .token_limits import get_model_context_length, calculate_dynamic_max_tokens
from .chat_format import apply_user_prompt, format_conversation as _format_conversation_helper
from .reasoning_format import format_reasoning_response as _format_reasoning_helper
from .stop_tokens import extract_stop_tokens as _extract_stop_tokens_helper

# Defer MLX/MLX-LM imports to runtime to avoid init crashes during test collection
mx = None  # type: ignore[assignment]
# Expose patchable names for tests (set by tests or lazily inside methods)
load = None  # type: ignore[assignment]
generate_step = None  # type: ignore[assignment]
make_repetition_penalty = None  # type: ignore[assignment]
make_sampler = None  # type: ignore[assignment]


# get_model_context_length is re-exported from token_limits


class MLXRunner:
    """Core MLX model execution engine for 2.0."""

    def __init__(self, model_name_or_path: str, adapter_path: Optional[str] = None, verbose: bool = False,
                 install_signal_handlers: bool = True):
        """Initialize the runner with a model.
        
        Args:
            model_name_or_path: Model specification or path
            adapter_path: Optional path to LoRA adapter
            verbose: Show detailed output
            install_signal_handlers: Whether to install SIGINT handler (disable for server mode)
        """
        self.model_spec = model_name_or_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self._memory_baseline = None
        self._stop_tokens = None
        self._chat_stop_tokens = None
        self._context_length = None
        self._is_reasoning_model = False
        self._reasoning_start = None
        self._reasoning_end = None
        self._final_start = None
        self.verbose = verbose
        self._model_loaded = False
        self._context_entered = False
        self._interrupted = False
        self._current_generator = None  # Handle to in-flight generation (for early cancellation)
        
        # Lazy-loaded MLX/MLX-LM refs (set in load_model / generation)
        self._mx = None
        self._load = None
        self._generate_step = None
        self._make_repetition_penalty = None
        self._make_sampler = None
        
        # Set up signal handler for Ctrl-C (only for run/interactive mode)
        if install_signal_handlers:
            signal.signal(signal.SIGINT, self._handle_interrupt)

    def __enter__(self):
        """Context manager entry - loads the model."""
        if self._context_entered:
            raise RuntimeError("MLXRunner context manager cannot be entered multiple times")
        
        self._context_entered = True
        try:
            self.load_model()
            return self
        except Exception:
            self._context_entered = False
            self.cleanup()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleans up the model."""
        self._context_entered = False
        self.cleanup()
        return False

    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl-C interruption during generation."""
        self._interrupted = True

    def request_interrupt(self) -> None:
        """Request an interruption from external controller (e.g., server signal).

        This sets the internal interruption flag so that ongoing generation loops
        will stop promptly at the next safe check point. Intended for server mode
        where per-runner OS signal handlers are disabled.
        """
        self._interrupted = True
        # Attempt to close any in-flight generator immediately to stop compute
        gen = getattr(self, "_current_generator", None)
        if gen is not None:
            try:
                close = getattr(gen, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass

    def load_model(self):
        """Load the MLX model and tokenizer."""
        if self._model_loaded:
            if self.verbose:
                print("Model already loaded, skipping...")
            return

        # Lazy import MLX and MLX-LM here
        try:
            import mlx.core as _mx  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to import MLX core: {e}") from e
        # Prefer test-patched load if available
        _load = globals().get('load')
        if _load is None:
            try:
                from mlx_lm import load as _load  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Failed to import MLX-LM load(): {e}") from e

        # Resolve model path using 2.0 resolution
        resolved_name, commit_hash, ambiguous = resolve_model_for_operation(self.model_spec)
        
        if ambiguous:
            raise ValueError(f"Ambiguous model specification '{self.model_spec}'. Could be: {ambiguous}")
        
        if not resolved_name:
            # In tests, resolution may be bypassed; fall back to provided spec
            resolved_name = str(self.model_spec)
        
        model_cache = get_current_model_cache()
        # Support tests that patch cache to a Mock by avoiding Path ops
        is_path_like = isinstance(model_cache, (str, Path)) or all(
            hasattr(model_cache, attr) for attr in ("__truediv__",)
        )

        if not resolved_name:
            # Fallback to provided spec (tests may patch load() to accept any path)
            resolved_name = str(self.model_spec)

        if is_path_like:
            model_cache_dir = (Path(model_cache) if not isinstance(model_cache, Path) else model_cache) / hf_to_cache_dir(resolved_name)
            if commit_hash:
                model_path = model_cache_dir / "snapshots" / commit_hash
            else:
                # Try to find a snapshot directory; tolerate missing during tests
                snapshots_dir = model_cache_dir / "snapshots"
                if snapshots_dir.exists():
                    snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                    model_path = snapshots[0] if snapshots else snapshots_dir / "mock"
                else:
                    model_path = snapshots_dir / "mock"
        else:
            # Non path-like cache (likely a Mock in unit tests) → pass a synthetic path to load()
            model_path = Path("/mock") / hf_to_cache_dir(resolved_name) / "snapshots" / (commit_hash or "mock")

        if self.verbose:
            print(f"Loading model from {model_path}...")
        start_time = time.time()

        # Capture baseline memory before loading
        try:
            _mx.clear_cache()
        except Exception:
            pass
        self._memory_baseline = _mx.get_active_memory() / 1024**3

        try:
            # Load model and tokenizer
            self.model, self.tokenizer = _load(
                str(model_path),
                adapter_path=self.adapter_path
            )

            load_time = time.time() - start_time
            current_memory = _mx.get_active_memory() / 1024**3
            model_memory = current_memory - self._memory_baseline

            if self.verbose:
                print(f"Model loaded in {load_time:.1f}s")
                print(f"Memory: {model_memory:.1f}GB model, {current_memory:.1f}GB total")

            # Extract stop tokens and other properties
            self._extract_stop_tokens()
            self._context_length = get_model_context_length(str(model_path))
            
            if self.verbose:
                print(f"Model context length: {self._context_length} tokens")
                
            self._model_loaded = True
            # Store MLX refs for later use
            self._mx = _mx
            self._load = _load  # type: ignore
            
        except Exception as e:
            self.model = None
            self.tokenizer = None
            self._stop_tokens = None
            self._model_loaded = False
            try:
                _mx.clear_cache()
            except Exception:
                pass
            # Preserve FileNotFoundError (used by tests) and propagate
            if isinstance(e, FileNotFoundError):
                raise e
            raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e

    def _extract_stop_tokens(self):
        """Extract stop tokens from the tokenizer dynamically (delegated)."""
        info = _extract_stop_tokens_helper(self.tokenizer, verbose=self.verbose)
        self._stop_tokens = info.stop_tokens
        self._chat_stop_tokens = info.chat_stop_tokens
        self._is_reasoning_model = info.is_reasoning_model
        self._reasoning_start = info.reasoning_start
        self._reasoning_end = info.reasoning_end
        self._final_start = info.final_start
        if self.verbose and self._stop_tokens:
            print(f"Stop tokens: {self._stop_tokens}")
        if self.verbose and self._is_reasoning_model:
            print("Reasoning model detected - special handling enabled")

    def cleanup(self):
        """Clean up model resources and clear GPU memory."""
        mx_core = self._mx
        if self.verbose and self._model_loaded and mx_core is not None:
            memory_before = mx_core.get_active_memory() / 1024**3
            print(f"Cleaning up model (memory before: {memory_before:.1f}GB)...")

        self.model = None
        self.tokenizer = None
        self._stop_tokens = None
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
            pass

        if self.verbose and mx_core is not None:
            memory_after = mx_core.get_active_memory() / 1024**3
            if 'memory_before' in locals():
                memory_freed = memory_before - memory_after
                print(f"Cleanup complete (memory after: {memory_after:.1f}GB, freed: {memory_freed:.1f}GB)")
            else:
                print(f"Cleanup complete (memory after: {memory_after:.1f}GB)")

    def _calculate_dynamic_max_tokens(self, server_mode: bool = True) -> int:
        """Calculate dynamic max tokens based on model context and usage mode."""
        return calculate_dynamic_max_tokens(self._context_length, server_mode=server_mode)

    def generate_streaming(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        repetition_context_size: int = 20,
        use_chat_template: bool = True,
        use_chat_stop_tokens: bool = False,
        hide_reasoning: bool = False,
    ) -> Iterator[str]:
        """Generate text with streaming output.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (None for dynamic)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeated tokens
            repetition_context_size: Context size for repetition penalty
            use_chat_template: Apply tokenizer's chat template if available
            use_chat_stop_tokens: Include chat turn markers as stop tokens
            hide_reasoning: Hide reasoning section for reasoning models
            
        Yields:
            Generated tokens as they are produced
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Reset any prior interruption at the start of a new generation
        # so that a previous Ctrl-C does not affect the next run
        self._interrupted = False

        # Initialize reasoning parser if this is a reasoning model
        reasoning_parser = None
        if self._is_reasoning_model:
            model_type = ReasoningExtractor.detect_model_type(
                getattr(self.tokenizer, 'name_or_path', '') or ''
            )
            reasoning_parser = StreamingReasoningParser(model_type, hide_reasoning=hide_reasoning)

        # Use dynamic max tokens if not specified (run command uses full context)
        effective_max_tokens = max_tokens if max_tokens is not None else self._calculate_dynamic_max_tokens(server_mode=False)

        # Apply chat template if available and requested
        formatted_prompt = apply_user_prompt(self.tokenizer, prompt, use_chat_template=use_chat_template)

        # Tokenize the prompt (tolerate mocks)
        prompt_tokens = self.tokenizer.encode(formatted_prompt)
        if not isinstance(prompt_tokens, (list, tuple)):
            prompt_tokens = [0]
        # Ensure MLX core is available
        mx_core = self._mx
        if mx_core is None:
            try:
                import mlx.core as mx_core  # type: ignore
                self._mx = mx_core
            except Exception as e:
                raise RuntimeError(f"Failed to import mlx.core for generation: {e}") from e
        prompt_array = mx_core.array(prompt_tokens)

        # Track generation metrics
        start_time = time.time()
        tokens_generated = 0

        # Create sampler and logits processors
        # Lazy import generation utilities
        if self._make_sampler is None or self._make_repetition_penalty is None or self._generate_step is None:
            # Prefer test-patched functions if present
            _ms = globals().get('make_sampler')
            _mrp = globals().get('make_repetition_penalty')
            _gs = globals().get('generate_step')
            if _ms is None or _mrp is None or _gs is None:
                try:
                    from mlx_lm.sample_utils import make_repetition_penalty as _mrp2, make_sampler as _ms2  # type: ignore
                    from mlx_lm.generate import generate_step as _gs2  # type: ignore
                    _mrp = _mrp or _mrp2
                    _ms = _ms or _ms2
                    _gs = _gs or _gs2
                except Exception as e:
                    raise RuntimeError(f"Failed to import MLX-LM generation utils: {e}") from e
            self._make_repetition_penalty = _mrp
            self._make_sampler = _ms
            self._generate_step = _gs

        sampler = self._make_sampler(temp=temperature, top_p=top_p)
        logits_processors = []
        if repetition_penalty > 1.0:
            logits_processors.append(
                self._make_repetition_penalty(repetition_penalty, repetition_context_size)
            )

        # Generate tokens one by one for streaming
        ret = self._generate_step(
            prompt=prompt_array,
            model=self.model,
            max_tokens=effective_max_tokens,
            sampler=sampler,
            logits_processors=logits_processors if logits_processors else None,
        )
        generator = ret
        if isinstance(ret, tuple) and len(ret) == 2:
            # Normalize tuple return into a single-step iterator
            generator = iter([ret])
        self._current_generator = generator

        # Collect and yield tokens
        generated_tokens = []
        previous_decoded = ""
        accumulated_response = ""
        context_window = 10

        for token, _ in generator:
            # Check for interruption
            if self._interrupted:
                # Close underlying generator to stop backend compute quickly
                try:
                    if hasattr(generator, "close"):
                        generator.close()
                except Exception:
                    pass
                yield "\n[Generation interrupted by user]"
                break

            token_id = token.item() if hasattr(token, 'item') else token
            generated_tokens.append(token_id)

            # Use sliding window for proper decoding
            start_idx = max(0, len(generated_tokens) - context_window)
            window_tokens = generated_tokens[start_idx:]
            window_text = self.tokenizer.decode(window_tokens)

            # Extract new text
            if start_idx == 0:
                # Prefer using the decoded window and diff vs previous text
                if previous_decoded and window_text.startswith(previous_decoded):
                    new_text = window_text[len(previous_decoded):]
                else:
                    # Fallback: take the window_text directly (robust to minimal mocks)
                    new_text = window_text
                previous_decoded = window_text
            else:
                new_text = self.tokenizer.decode(window_tokens)
                if len(window_tokens) > 1:
                    prefix = self.tokenizer.decode(window_tokens[:-1])
                    if new_text.startswith(prefix):
                        new_text = new_text[len(prefix):]
                    else:
                        new_text = self.tokenizer.decode([token_id])

            if new_text:
                accumulated_response += new_text
                
                # Check for stop tokens (strings only)
                stop_tokens_to_check = self._stop_tokens if self._stop_tokens else []
                stop_tokens_to_check = [t for t in stop_tokens_to_check if isinstance(t, str) and t]
                if use_chat_stop_tokens:
                    stop_tokens_to_check.extend(self._chat_stop_tokens)

                # Find earliest stop token in accumulated response (ADR-011: multiple EOS token handling)
                if stop_tokens_to_check:
                    earliest_pos = len(accumulated_response)
                    earliest_token = None

                    for stop_token in stop_tokens_to_check:
                        if stop_token in accumulated_response:
                            pos = accumulated_response.find(stop_token)
                            if pos < earliest_pos:
                                earliest_pos = pos
                                earliest_token = stop_token

                    if earliest_token:
                        # Found stop token - yield remaining text before it and stop
                        text_before_stop = accumulated_response[:earliest_pos]
                        previously_yielded_length = len(accumulated_response) - len(new_text)
                        if len(text_before_stop) > previously_yielded_length:
                            new_part_before_stop = text_before_stop[previously_yielded_length:]
                            if new_part_before_stop:
                                if reasoning_parser:
                                    # Process through reasoning parser for formatting
                                    for formatted_token in reasoning_parser.process_token(new_part_before_stop):
                                        yield formatted_token
                                else:
                                    yield new_part_before_stop
                        return

                # No stop token found, process the new text
                if reasoning_parser:
                    # Process through reasoning parser for formatting
                    for formatted_token in reasoning_parser.process_token(new_text):
                        yield formatted_token
                else:
                    # Normal streaming for non-reasoning models
                    yield new_text
                tokens_generated += 1

            # Check for EOS token (ADR-009: use eos_token_ids Set for multi-EOS models)
            if token_id in self.tokenizer.eos_token_ids:
                break

        # Finalize reasoning parser if used
        if reasoning_parser:
            yield from reasoning_parser.finalize()

        # Clear current generator handle
        self._current_generator = None

        if self.verbose:
            generation_time = time.time() - start_time
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            print(f"\n\nGenerated {tokens_generated} tokens in {generation_time:.1f}s ({tokens_per_second:.1f} tokens/s)")

    def generate_batch(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        repetition_context_size: int = 20,
        use_chat_template: bool = True,
        use_chat_stop_tokens: bool = False,
        hide_reasoning: bool = False,
    ) -> str:
        """Generate text in batch mode (non-streaming).

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (None for dynamic)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeated tokens
            repetition_context_size: Context size for repetition penalty
            use_chat_template: Apply tokenizer's chat template if available
            use_chat_stop_tokens: Include chat turn markers as stop tokens (e.g., "\nHuman:")
            hide_reasoning: Hide reasoning output for reasoning models (DeepSeek-R1, QwQ, etc.)

        Returns:
            Generated text
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Reset any prior interruption at the start of a new generation
        self._interrupted = False

        # Use dynamic max tokens if not specified (run command uses full context)
        effective_max_tokens = max_tokens if max_tokens is not None else self._calculate_dynamic_max_tokens(server_mode=False)

        # Apply chat template if available and requested
        formatted_prompt = apply_user_prompt(self.tokenizer, prompt, use_chat_template=use_chat_template)

        start_time = time.time()

        # Tokenize and generate (tolerate mocks)
        prompt_tokens = self.tokenizer.encode(formatted_prompt)
        if not isinstance(prompt_tokens, (list, tuple)):
            prompt_tokens = [0]
        # Ensure MLX core is available
        mx_core = self._mx
        if mx_core is None:
            try:
                import mlx.core as mx_core  # type: ignore
                self._mx = mx_core
            except Exception as e:
                raise RuntimeError(f"Failed to import mlx.core for generation: {e}") from e
        prompt_array = mx_core.array(prompt_tokens)

        if self._make_sampler is None or self._make_repetition_penalty is None or self._generate_step is None:
            _ms = globals().get('make_sampler')
            _mrp = globals().get('make_repetition_penalty')
            _gs = globals().get('generate_step')
            if _ms is None or _mrp is None or _gs is None:
                try:
                    from mlx_lm.sample_utils import make_repetition_penalty as _mrp2, make_sampler as _ms2  # type: ignore
                    from mlx_lm.generate import generate_step as _gs2  # type: ignore
                    _mrp = _mrp or _mrp2
                    _ms = _ms or _ms2
                    _gs = _gs or _gs2
                except Exception as e:
                    raise RuntimeError(f"Failed to import MLX-LM generation utils: {e}") from e
            self._make_repetition_penalty = _mrp
            self._make_sampler = _ms
            self._generate_step = _gs
        sampler = self._make_sampler(temp=temperature, top_p=top_p)
        logits_processors = []
        if repetition_penalty > 1.0:
            logits_processors.append(
                self._make_repetition_penalty(repetition_penalty, repetition_context_size)
            )

        # Generate all tokens
        generated_tokens = []
        all_tokens = list(prompt_tokens)

        ret = self._generate_step(
            prompt=prompt_array,
            model=self.model,
            max_tokens=effective_max_tokens,
            sampler=sampler,
            logits_processors=logits_processors if logits_processors else None,
        )
        generator = ret
        if isinstance(ret, tuple) and len(ret) == 2:
            generator = iter([ret])
        self._current_generator = generator

        for token, _ in generator:
            if self._interrupted:
                try:
                    if hasattr(generator, "close"):
                        generator.close()
                except Exception:
                    pass
                break
                
            token_id = token.item() if hasattr(token, 'item') else token
            generated_tokens.append(token_id)
            all_tokens.append(token_id)

            # Check for EOS token (ADR-009: use eos_token_ids Set for multi-EOS models)
            if token_id in self.tokenizer.eos_token_ids:
                break

        # Decode full response
        full_response = self.tokenizer.decode(all_tokens)

        # Debug: Show raw generated tokens for quality analysis (enabled via --verbose)
        if self.verbose:
            print("\n[DEBUG] Token generation analysis:")
            print(f"[DEBUG]   Generated {len(generated_tokens)} tokens")
            if len(generated_tokens) >= 3:
                last_3_ids = generated_tokens[-3:]
                last_3_decoded = []
                for tid in last_3_ids:
                    try:
                        decoded = self.tokenizer.decode([tid])
                        last_3_decoded.append(f"{tid}={decoded!r}")
                    except Exception:
                        last_3_decoded.append(f"{tid}=<error>")
                print(f"[DEBUG]   Last 3 tokens: {last_3_decoded}")

                # Check for multiple EOS tokens (quality issue indicator)
                eos_count = sum(1 for tid in last_3_ids if tid in self.tokenizer.eos_token_ids)
                if eos_count > 1:
                    print(f"[DEBUG]   ⚠️ WARNING: Multiple EOS tokens detected ({eos_count}) - model quality issue")

        # Remove prompt part (guard types to tolerate mocks)
        if isinstance(full_response, str) and isinstance(formatted_prompt, str) and full_response.startswith(formatted_prompt):
            response = full_response[len(formatted_prompt):]
        else:
            decoded = self.tokenizer.decode(generated_tokens)
            response = decoded if isinstance(decoded, str) else str(decoded)

        # Filter stop tokens (strings only)
        # Find the EARLIEST stop token in the response (not first in list)
        if self._stop_tokens:
            stop_tokens_filtered = [t for t in self._stop_tokens if isinstance(t, str) and t]
            earliest_pos = len(response)
            earliest_token = None

            for stop_token in stop_tokens_filtered:
                if stop_token in response:
                    pos = response.find(stop_token)
                    if pos < earliest_pos:
                        earliest_pos = pos
                        earliest_token = stop_token

            if earliest_token:
                response = response[:earliest_pos]

        # Optionally filter chat stop tokens to prevent self-conversations in batch mode
        # Find the EARLIEST chat stop token (same logic as above)
        if use_chat_stop_tokens and self._chat_stop_tokens:
            earliest_pos = len(response)
            for stop_token in self._chat_stop_tokens:
                if stop_token and stop_token in response:
                    pos = response.find(stop_token)
                    if pos < earliest_pos:
                        earliest_pos = pos
            if earliest_pos < len(response):
                response = response[:earliest_pos]

        # Format reasoning models output
        response = self._format_reasoning_response(response, hide_reasoning=hide_reasoning)

        generation_time = time.time() - start_time

        if self.verbose:
            tokens_generated = len(generated_tokens)
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            print(f"\nGenerated {tokens_generated} tokens in {generation_time:.1f}s ({tokens_per_second:.1f} tokens/s)")

        # Clear current generator handle
        self._current_generator = None

        return response

    def _format_conversation(self, messages):
        """Format conversation history into a prompt using chat template."""
        return _format_conversation_helper(self.tokenizer, messages)

    def _format_reasoning_response(self, response: str, hide_reasoning: bool = False) -> str:
        """Format response from reasoning models for better readability."""
        return _format_reasoning_helper(
            response,
            self._is_reasoning_model,
            self._reasoning_start,
            self._reasoning_end,
            self._final_start,
            hide_reasoning=hide_reasoning,
        )
