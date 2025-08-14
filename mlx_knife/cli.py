# mlx_knife/cli.py

import argparse
import sys

from . import __version__
from .cache_utils import (
    check_all_models_health,
    check_model_health,
    list_models,
    rm_model,
    run_model,
    show_model,
)
from .hf_download import pull_model
from .server import run_server


def main():
    parser = argparse.ArgumentParser(
        description="MLX Knife CLI (HuggingFace-style cache management for MLX models)"
    )
    parser.add_argument('--version', action='version', version=f'MLX Knife {__version__}')
    subparsers = parser.add_subparsers(dest="cmd")

    # list
    list_p = subparsers.add_parser("list", help="List available models in cache")
    list_p.add_argument("model", nargs="?", help="Specific model to list (optional)")
    list_p.add_argument("--all", action="store_true", help="Show all models (not just MLX)")
    list_p.add_argument("--framework", choices=["mlx", "pytorch", "tokenizer"], help="Filter by framework")
    list_p.add_argument("--health", action="store_true", help="Show health status")
    list_p.add_argument("--verbose", action="store_true", help="Show detailed information (requires model argument)")

    # pull
    pull_p = subparsers.add_parser("pull", help="Download a model from HuggingFace")
    pull_p.add_argument("model_spec", help="Model[@hash] (e.g. mlx-community/Qwen2.5-0.5B-Instruct-4bit@a5339a41)")

    # run
    run_p = subparsers.add_parser("run", help="Run a model with prompt")
    run_p.add_argument("model_spec", help="Model[@hash] (e.g. mlx-community/Qwen2.5-0.5B-Instruct-4bit@a5339a41)")
    run_p.add_argument("prompt", nargs="?", default=None, help="Prompt text (if not provided, enters interactive mode)")
    run_p.add_argument("--interactive", "-i", action="store_true", help="Force interactive dialog mode")
    run_p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    run_p.add_argument("--max-tokens", type=int, default=500, help="Maximum tokens to generate (default: 500)")
    run_p.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter (default: 0.9)")
    run_p.add_argument("--repetition-penalty", type=float, default=1.1, help="Penalty for repeated tokens (default: 1.1)")
    run_p.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    run_p.add_argument("--no-chat-template", action="store_true", help="Disable chat template formatting (use raw prompt)")
    run_p.add_argument("--verbose", "-v", action="store_true", help="Show detailed output (model loading, memory usage, token stats)")

    # rm
    rm_p = subparsers.add_parser("rm", help="Delete a model from cache")
    rm_p.add_argument("model_spec", help="Model[@hash] (e.g. mlx-community/Qwen2.5-0.5B-Instruct-4bit@a5339a41)")

    # health
    health_p = subparsers.add_parser("health", help="Check model integrity")
    health_p.add_argument("model_spec", nargs="?", help="Model[@hash] (optional)")
    health_p.add_argument("--all", action="store_true", help="Check all models in cache")

    # show
    show_p = subparsers.add_parser("show", help="Show detailed information about a specific model")
    show_p.add_argument("model_spec", help="Model[@hash] (e.g. mlx-community/Qwen2.5-0.5B-Instruct-4bit@a5339a41)")
    show_p.add_argument("--files", action="store_true", help="List all files and sizes under the model path")
    show_p.add_argument("--config", action="store_true", help="Print pretty-formatted config.json")

    # server
    server_p = subparsers.add_parser("server", help="Start OpenAI-compatible API server")
    server_p.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    server_p.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    server_p.add_argument("--max-tokens", type=int, default=2000, help="Default max tokens for completions (default: 2000)")
    server_p.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    server_p.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="Log level (default: info)")

    args = parser.parse_args()

    if args.cmd == "list":
        if args.model:
            if args.verbose and not args.all and not args.framework and not args.health:
                # Show detailed info for a specific model (same as show command)
                show_model(args.model)
            else:
                # Show just the single model row
                list_models(show_all=args.all, framework_filter=args.framework, show_health=args.health, single_model=args.model, verbose=args.verbose)
        else:
            # Normal list behavior - verbose works with MLX models too
            list_models(show_all=args.all, framework_filter=args.framework, show_health=args.health, verbose=args.verbose)
    elif args.cmd == "pull":
        pull_model(args.model_spec)
    elif args.cmd == "run":
        run_model(
            args.model_spec,
            prompt=args.prompt,
            interactive=args.interactive,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stream=not args.no_stream,
            use_chat_template=not args.no_chat_template,
            verbose=args.verbose
        )
    elif args.cmd == "rm":
        rm_model(args.model_spec)
    elif args.cmd == "health":
        if args.model_spec:
            check_model_health(args.model_spec)
        else:
            # Default to checking all models if no specific model is provided
            check_all_models_health()
    elif args.cmd == "show":
        show_model(args.model_spec, show_files=args.files, show_config=args.config)
    elif args.cmd == "server":
        # Validate server arguments
        if args.max_tokens <= 0:
            print(f"Error: --max-tokens must be positive, got: {args.max_tokens}")
            sys.exit(1)
        if args.port <= 0 or args.port > 65535:
            print(f"Error: --port must be between 1-65535, got: {args.port}")
            sys.exit(1)

        run_server(
            host=args.host,
            port=args.port,
            max_tokens=args.max_tokens,
            reload=args.reload,
            log_level=args.log_level
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
