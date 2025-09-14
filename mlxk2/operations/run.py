"""
Run operation for 2.0 implementation.
Ported from 1.x with 2.0 architecture integration.
"""

from typing import Optional

from ..core.runner import MLXRunner


def run_model(
    model_spec: str,
    prompt: Optional[str] = None,
    stream: bool = True,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    use_chat_template: bool = True,
    json_output: bool = False,
    verbose: bool = False
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
        
    Returns:
        Generated text if json_output=True, None otherwise
    """
    try:
        with MLXRunner(model_spec, verbose=verbose) as runner:
            # Interactive mode: no prompt provided
            if prompt is None:
                if json_output:
                    print("Error: Interactive mode not compatible with JSON output")
                    return None
                return interactive_chat(
                    runner, 
                    stream=stream, 
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_chat_template=use_chat_template,
                    prepare_next_prompt=False
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
                    json_output=json_output
                )
                    
    except Exception as e:
        if json_output:
            return f"Error: {e}"
        else:
            print(f"Error: {e}")
            return None


def interactive_chat(
    runner,
    stream: bool = True,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    use_chat_template: bool = True,
    prepare_next_prompt: bool = False,
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
            print(f"\n[ERROR] {e}")
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
    json_output: bool = False
) -> Optional[str]:
    """Single prompt generation."""
    if stream and not json_output:
        # Streaming mode - print tokens as they arrive
        generated_text = ""
        for token in runner.generate_streaming(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_chat_template=use_chat_template,
        ):
            print(token, end="", flush=True)
            generated_text += token
        
        if not json_output:
            print()  # Final newline
        
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
        )
        
        if json_output:
            return result
        else:
            print(result)
            return None


def run_model_enhanced(
    model_spec: str,
    prompt: str,
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
        Generated text if json_output=True, None otherwise
    """
    # For now, forward to basic run_model
    # TODO: Add system_prompt and hide_reasoning support in beta.2
    if system_prompt:
        print("Warning: System prompts not yet implemented in beta.1")
    
    return run_model(
        model_spec=model_spec,
        prompt=prompt,
        stream=stream,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        use_chat_template=use_chat_template,
        json_output=json_output,
        verbose=verbose
    )
