"""
Run operation for 2.0 implementation.
Ported from 1.x with 2.0 architecture integration.
"""

import json
import subprocess
import sys
from typing import Optional, Sequence, Tuple

from ..core.runner import MLXRunner
from ..core.cache import get_current_model_cache, hf_to_cache_dir
from ..core.model_resolution import resolve_model_for_operation
from ..operations.health import check_runtime_compatibility
from ..operations.common import (
    _total_size_bytes,
    detect_framework,
    detect_vision_capability,
    read_front_matter,
    vision_runtime_compatibility,
)


# Memory threshold for pre-load checks (ADR-016)
# Vision models crash above ~70% due to Vision Encoder overhead
MEMORY_THRESHOLD_PERCENT = 0.70


def _get_system_memory_bytes() -> Optional[int]:
    """Get total system memory in bytes via sysctl (macOS only).

    Returns:
        Total memory in bytes, or None if unavailable.
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        pass
    return None


def _format_bytes_gb(size_bytes: int) -> str:
    """Format bytes as human-readable GB string."""
    return f"{size_bytes / (1024**3):.1f} GB"


def check_memory_before_load(
    model_path,
    is_vision_model: bool,
    json_output: bool = False,
) -> Optional[str]:
    """Check if model size exceeds safe memory threshold (ADR-016).

    Vision models: ERROR + abort if >70% (Metal OOM crash prevention)
    Text models: No user-facing action (swaps gracefully, no crash)

    Args:
        model_path: Path to model snapshot directory
        is_vision_model: Whether the model has vision capability
        json_output: Whether JSON output mode is active

    Returns:
        Error message string if should abort, None if safe to proceed.
        For text models, always returns None (no abort).
    """
    # Only abort for vision models (text models swap gracefully)
    if not is_vision_model:
        return None

    system_memory = _get_system_memory_bytes()
    if system_memory is None:
        # Cannot determine system memory - proceed (backwards compatible)
        return None

    model_size = _total_size_bytes(model_path)
    if model_size == 0:
        # Cannot determine model size - proceed
        return None

    threshold = int(system_memory * MEMORY_THRESHOLD_PERCENT)
    if model_size > threshold:
        # Vision model exceeds 70% - abort to prevent Metal OOM crash
        return (
            f"Model size ({_format_bytes_gb(model_size)}) exceeds 70% of system memory "
            f"({_format_bytes_gb(system_memory)}). Vision models crash with Metal OOM "
            f"due to Vision Encoder overhead. Aborting."
        )

    return None


def check_memory_for_server(
    model_path,
    is_vision_model: bool,
    model_name: str,
    logger=None,
) -> Optional[str]:
    """Check memory threshold for server mode (ADR-016).

    Vision models: Return error message for HTTP 507 if >70%
    Text models: Log warning only (swaps gracefully, no abort)

    Args:
        model_path: Path to model snapshot directory
        is_vision_model: Whether the model has vision capability
        model_name: Model name for logging
        logger: Logger instance for text model warnings

    Returns:
        Error message string for vision models if should abort (for HTTP 507),
        None otherwise. Text models only log warning, never return error.
    """
    system_memory = _get_system_memory_bytes()
    if system_memory is None:
        return None

    model_size = _total_size_bytes(model_path)
    if model_size == 0:
        return None

    threshold = int(system_memory * MEMORY_THRESHOLD_PERCENT)
    if model_size > threshold:
        if is_vision_model:
            # Vision model exceeds 70% - abort to prevent Metal OOM crash
            return (
                f"Model size ({_format_bytes_gb(model_size)}) exceeds 70% of system memory "
                f"({_format_bytes_gb(system_memory)}). Vision models crash with Metal OOM "
                f"due to Vision Encoder overhead."
            )
        else:
            # Text model exceeds 70% - log warning only (swaps gracefully)
            if logger:
                logger.warning(
                    f"Model size {_format_bytes_gb(model_size)} exceeds 70% of "
                    f"{_format_bytes_gb(system_memory)} system memory. "
                    f"Expect extreme slowness due to swapping.",
                    model=model_name,
                )

    return None


def run_model(
    model_spec: str,
    prompt: Optional[str] = None,
    images: Optional[Sequence[Tuple[str, bytes]]] = None,
    stream: bool = True,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    use_chat_template: bool = True,
    json_output: bool = False,
    verbose: bool = False,
    hide_reasoning: bool = False
) -> Optional[str]:
    """Execute model with prompt - supports both single-shot and interactive modes.

    Args:
        model_spec: Model specification or path
        prompt: Input prompt (None = interactive mode)
        stream: Enable streaming output (default True)
        max_tokens: Maximum tokens to generate (None for dynamic)
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        repetition_penalty: Penalty for repeated tokens
        use_chat_template: Apply tokenizer's chat template if available
        json_output: Return JSON format instead of printing
        verbose: Show detailed output
        hide_reasoning: Hide reasoning output for reasoning models (DeepSeek-R1, QwQ, etc.)

    Returns:
        Generated text on success, "Error: ..." string on failure (both modes)
    """
    json_mode = json_output
    # Pre-flight check: Verify runtime compatibility before attempting to load
    # This is a "best effort" check - if the model is in cache, verify it's compatible
    # If not in cache or check fails, let the runner handle it (for tests and edge cases)
    try:
        resolved_name, commit_hash, ambiguous = resolve_model_for_operation(model_spec)

        if ambiguous:
            error_msg = f"Ambiguous model specification '{model_spec}'. Could be: {ambiguous}"
            error_result = f"Error: {error_msg}"
            if not json_output:
                print(error_result, file=sys.stderr)
            return error_result

        # Only perform compatibility check if model is actually in cache
        is_vision_model = False
        model_path = None
        cfg = None
        if resolved_name:
            model_cache = get_current_model_cache()
            model_cache_dir = model_cache / hf_to_cache_dir(resolved_name)

            if model_cache_dir.exists():
                snapshots_dir = model_cache_dir / "snapshots"
                if snapshots_dir.exists():
                    # Resolve snapshot path (commit-pinned or latest)
                    if commit_hash:
                        model_path = snapshots_dir / commit_hash
                    else:
                        snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                        if snapshots:
                            model_path = max(snapshots, key=lambda x: x.stat().st_mtime)

                    # Detect vision capability to select backend
                    cfg_path = model_path / "config.json" if model_path else None
                    if cfg_path and cfg_path.exists():
                        try:
                            cfg = json.loads(cfg_path.read_text(encoding="utf-8", errors="ignore"))
                        except Exception:
                            cfg = None

                    if model_path is not None:
                        is_vision_model = detect_vision_capability(model_path, cfg)

                    # If images are provided but model is not vision-capable, fail fast
                    if images and not is_vision_model:
                        error_msg = f"Model '{resolved_name}' does not support vision inputs (no vision capability detected)."
                        error_result = f"Error: {error_msg}"
                        if not json_output:
                            print(error_result, file=sys.stderr)
                        return error_result

                    if is_vision_model:
                        compat, reason = vision_runtime_compatibility()
                        if not compat:
                            error_msg = f"Model '{resolved_name}' is vision-capable but not runnable: {reason}"
                            error_result = f"Error: {error_msg}"
                            if not json_output:
                                print(error_result, file=sys.stderr)
                            return error_result

                        # ADR-016: Pre-load memory check for vision models
                        # Vision models crash with Metal OOM above ~70% system memory
                        if model_path:
                            memory_error = check_memory_before_load(model_path, is_vision_model=True)
                            if memory_error:
                                error_result = f"Error: {memory_error}"
                                if not json_output:
                                    print(error_result, file=sys.stderr)
                                return error_result
                    else:
                        # Check runtime compatibility for both pinned and unpinned models (text/LLM path)
                        if model_path and model_path.exists():
                            # Read README front-matter for framework hints (e.g., private MLX models)
                            fm = read_front_matter(model_path)
                            framework = detect_framework(resolved_name, model_cache_dir, selected_path=model_path, fm=fm)
                            compatible, reason = check_runtime_compatibility(model_path, framework)

                            if not compatible:
                                error_msg = f"Model '{resolved_name}' is not compatible: {reason}"
                                error_result = f"Error: {error_msg}"
                                if not json_output:
                                    print(error_result, file=sys.stderr)
                                return error_result

    except Exception:
        # Pre-flight check failed - let the runner handle it
        # This preserves backward compatibility with tests and edge cases
        pass

    if images and not is_vision_model:
        error_result = "Error: Vision inputs require a vision-capable model in cache (config not found)"
        if not json_output:
            print(error_result, file=sys.stderr)
        return error_result

    # Runtime compatibility verified, proceed with model loading
    try:
        if is_vision_model:
            # Vision path uses mlx-vlm backend (non-streaming)
            if model_path is None or not model_path.exists():
                error_result = "Error: Vision model not found in cache"
                if not json_output:
                    print(error_result, file=sys.stderr)
                return error_result

            if prompt is None:
                if images:
                    prompt = "Describe the image."
                else:
                    error_result = "Error: Vision run requires a prompt"
                    if not json_output:
                        print(error_result, file=sys.stderr)
                    return error_result

            # Vision support requires Python 3.10+ (mlx-vlm requirement)
            if sys.version_info < (3, 10):
                error_result = "Error: Vision models require Python 3.10 or newer (mlx-vlm dependency)"
                if not json_output:
                    print(error_result, file=sys.stderr)
                return error_result

            try:
                # Lazy import: only load VisionRunner when needed (Python 3.10+ required)
                from ..core.vision_runner import VisionRunner

                with VisionRunner(model_path, resolved_name or model_spec, verbose=verbose) as runner:
                    result = runner.generate(
                        prompt=prompt,
                        images=list(images or []),
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
            except Exception as e:
                error_result = f"Error: {e}"
                if not json_output:
                    print(error_result, file=sys.stderr)
                return error_result

            if json_output:
                return result
            try:
                print(result)
            except BrokenPipeError:
                sys.stderr.close()
            return None

        # Text LLM path (existing behavior)
        with MLXRunner(model_spec, verbose=verbose) as runner:
            # Interactive mode: no prompt provided
            if prompt is None:
                if json_mode:
                    return "Error: Interactive mode not compatible with JSON output"
                return interactive_chat(
                    runner,
                    stream=stream,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_chat_template=use_chat_template,
                    prepare_next_prompt=False,
                    hide_reasoning=hide_reasoning,
                )
            else:
                # Single-shot mode: prompt provided
                return single_shot_generation(
                    runner,
                    prompt,
                    stream=stream,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_chat_template=use_chat_template,
                    json_output=json_output,
                    hide_reasoning=hide_reasoning
                )
    except Exception as e:
        error_result = f"Error: {e}"
        if not json_output:
            print(error_result, file=sys.stderr)
        return error_result


def interactive_chat(
    runner,
    stream: bool = True,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    use_chat_template: bool = True,
    prepare_next_prompt: bool = False,
    hide_reasoning: bool = False,
):
    """Interactive conversation mode with history tracking."""
    print("Starting interactive chat. Type 'exit' or 'quit' to end.\n")
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
                
            if not user_input:
                continue
                
            # Add user message to conversation history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Format conversation using chat template
            # Pass a shallow copy to avoid later mutations affecting captured args in tests
            formatted_prompt = runner._format_conversation(conversation_history.copy())
            
            # Generate response
            print("\nAssistant: ", end="", flush=True)
            
            if stream:
                # Streaming mode
                response_tokens = []
                # Build standard params but be robust to mocks that don't accept them
                params = dict(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_chat_template=False,
                    use_chat_stop_tokens=True,
                    hide_reasoning=hide_reasoning,
                )
                try:
                    iterator = runner.generate_streaming(formatted_prompt, **params)
                except TypeError:
                    try:
                        iterator = runner.generate_streaming(formatted_prompt)
                    except TypeError:
                        iterator = runner.generate_streaming()
                for token in iterator:
                    print(token, end="", flush=True)
                    response_tokens.append(token)
                response = "".join(response_tokens).strip()
            else:
                # Batch mode
                params = dict(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_chat_template=False,
                    use_chat_stop_tokens=True,
                    hide_reasoning=hide_reasoning,
                )
                try:
                    response = runner.generate_batch(formatted_prompt, **params)
                except TypeError:
                    try:
                        response = runner.generate_batch(formatted_prompt)
                    except TypeError:
                        response = runner.generate_batch()
                print(response)
            
            # Add assistant response to history
            conversation_history.append({"role": "assistant", "content": response})
            print()  # Newline after response
            
            # Optionally expose assistant message to template users without duplicating user entries
            if prepare_next_prompt:
                try:
                    _ = runner._format_conversation([{"role": "assistant", "content": response}])
                except Exception:
                    pass
            
        except KeyboardInterrupt:
            print("\n\nChat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}", file=sys.stderr)
            continue


def single_shot_generation(
    runner,
    prompt: str,
    stream: bool = True,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    use_chat_template: bool = True,
    json_output: bool = False,
    hide_reasoning: bool = False
) -> Optional[str]:
    """Single prompt generation."""
    if stream and not json_output:
        # Streaming mode - print tokens as they arrive
        generated_text = ""
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
                print(token, end="", flush=True)
                generated_text += token

            if not json_output:
                print()  # Final newline
        except BrokenPipeError:
            # Downstream closed the pipe (e.g., `mlxk run model | head -1`)
            # This is expected Unix behavior - exit silently without error
            # Flush stderr to ensure any buffered errors are visible, then close
            sys.stderr.close()
            return None

        return generated_text if json_output else None
    else:
        # Batch mode - generate complete response
        result = runner.generate_batch(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_chat_template=use_chat_template,
            hide_reasoning=hide_reasoning,
        )

        if json_output:
            return result
        else:
            try:
                print(result)
            except BrokenPipeError:
                # Downstream closed the pipe - exit silently
                sys.stderr.close()
            return None


def run_model_enhanced(
    model_spec: str,
    prompt: Optional[str],
    images: Optional[Sequence[Tuple[str, bytes]]] = None,
    stream: bool = True,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    repetition_context_size: int = 20,
    use_chat_template: bool = True,
    json_output: bool = False,
    verbose: bool = False,
    system_prompt: Optional[str] = None,
    hide_reasoning: bool = False
) -> Optional[str]:
    """Enhanced run with additional parameters for future features.
    
    This function signature matches what will be needed for 2.0.0-beta.2
    when system prompts and reasoning features are added.
    
    Args:
        model_spec: Model specification or path
        prompt: Input prompt
        stream: Enable streaming output
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        repetition_penalty: Penalty for repeated tokens
        repetition_context_size: Context size for repetition penalty
        use_chat_template: Apply tokenizer's chat template
        json_output: Return JSON format
        verbose: Show detailed output
        system_prompt: System prompt (future feature)
        hide_reasoning: Hide reasoning output (future feature)
        
    Returns:
        Generated text on success, "Error: ..." string on failure (both modes)
    """
    # For now, forward to basic run_model
    # TODO: Add system_prompt support in future version
    if system_prompt:
        print("Warning: System prompts not yet implemented")

    return run_model(
        model_spec=model_spec,
        prompt=prompt,
        images=images,
        stream=stream,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        use_chat_template=use_chat_template,
        json_output=json_output,
        verbose=verbose,
        hide_reasoning=hide_reasoning
    )
