#!/usr/bin/env python3
"""MLX-Knife CLI - HuggingFace model management for MLX."""

import argparse
import json
import os
import sys
from typing import Dict, Any

from . import __version__
from .operations.list import list_models
from .operations.health import health_check_operation
from .operations.pull import pull_operation
from .operations.rm import rm_operation
from .operations.push import push_operation
from .operations.show import show_model_operation
from .operations.run import run_model_enhanced
from .spec import JSON_API_SPEC_VERSION
from .output.human import (
    render_list,
    render_health,
    render_show,
    render_pull,
    render_clone,
    render_rm,
)


def format_json_output(data: Dict[str, Any]) -> str:
    """Format output as JSON."""
    return json.dumps(data, indent=2)


def print_result(result: Dict[str, Any], render_func=None, json_mode=False, **render_kwargs):
    """Print command result to stdout (JSON, success) or stderr (human errors).

    Args:
        result: Command result dict with 'status' field
        render_func: Human-mode rendering function (if json_mode=False)
        json_mode: If True, output JSON format (always to stdout)
        **render_kwargs: Additional arguments for render_func
    """
    is_error = result.get("status") == "error"

    if json_mode:
        # JSON mode: Always stdout (for scripting/jq)
        print(format_json_output(result), file=sys.stdout)
    elif is_error:
        # Human-mode error: stderr (for pipes)
        error_info = result.get("error", {})
        message = error_info.get("message", "Unknown error")
        command = result.get("command", "command")
        print(f"{command}: Error: {message}", file=sys.stderr)
    elif render_func:
        # Human-mode success: stdout
        print(render_func(result, **render_kwargs), file=sys.stdout)
    else:
        # Fallback: print JSON to stdout
        print(format_json_output(result), file=sys.stdout)


def handle_error(error_type: str, message: str) -> Dict[str, Any]:
    """Format error as JSON response."""
    return {
        "status": "error",
        "command": None,
        "data": None,
        "error": {
            "type": error_type,
            "message": message
        }
    }


class MLXKArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that prints JSON errors when --json is present.

    This ensures invocations like `mlxk2 push --json --private` (missing args)
    emit a JSON error instead of argparse usage text.
    """

    def error(self, message):  # type: ignore[override]
        want_json = "--json" in sys.argv
        if want_json:
            err = handle_error("CommandError", message)
            print(format_json_output(err), file=sys.stdout)
            self.exit(2)
        super().error(message)


def main():
    """Main CLI entry point."""
    parser = MLXKArgumentParser(
        prog="mlxk2",
        description="MLX-Knife - HuggingFace model management for MLX"
    )
    
    # Add version argument (supports --json)
    parser.add_argument("--version", action="store_true", help="Show version information and exit")
    parser.add_argument("--json", action="store_true", help="Output in JSON format (with --version or per command)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands", parser_class=MLXKArgumentParser)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all cached models")
    list_parser.add_argument("pattern", nargs="?", help="Filter models by pattern (optional)")
    # Human-output modifiers (JSON output remains unchanged)
    list_parser.add_argument("--all", action="store_true", dest="show_all", help="Show all details (human output)")
    list_parser.add_argument("--health", action="store_true", dest="show_health", help="Include health column (human output)")
    list_parser.add_argument("--verbose", action="store_true", help="Verbose details (human output)")
    list_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Check model health")
    health_parser.add_argument("model", nargs="?", help="Model pattern to check (optional)")
    health_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show detailed model information")
    show_parser.add_argument("model", help="Model name to show")
    show_parser.add_argument("--files", action="store_true", help="Include file listing")
    show_parser.add_argument("--config", action="store_true", help="Include config.json content")
    show_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Download a model")
    pull_parser.add_argument("model", help="Model name to download")
    pull_parser.add_argument("--json", action="store_true", help="Output in JSON format")

    # Clone command (alpha) - only show if alpha features enabled
    if os.getenv("MLXK2_ENABLE_ALPHA_FEATURES"):
        clone_parser = subparsers.add_parser("clone", help="ALPHA: Clone a model to a local workspace")
        clone_parser.add_argument("model", help="Model name to clone (org/repo[@revision])")
        clone_parser.add_argument("target_dir", help="Target directory for workspace")
        clone_parser.add_argument("--branch", help="Specific branch/revision to clone")
        clone_parser.add_argument("--no-health-check", action="store_true", help="Skip health validation before copy")
        clone_parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
        clone_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    # Remove command
    rm_parser = subparsers.add_parser("rm", help="Delete a model")
    rm_parser.add_argument("model", help="Model name to delete")
    rm_parser.add_argument("-f", "--force", action="store_true", help="Delete without confirmation")
    rm_parser.add_argument("--json", action="store_true", help="Output in JSON format")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run model with prompt")
    run_parser.add_argument("model", help="Model name to run")
    run_parser.add_argument("prompt", nargs="?", help="Input prompt (optional - triggers interactive mode if omitted)")
    run_parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    run_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    run_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter (default: 0.9)")
    run_parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty (default: 1.1)")
    run_parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    run_parser.add_argument("--no-chat-template", action="store_true", help="Disable chat template")
    run_parser.add_argument("--no-reasoning", action="store_true", help="Hide reasoning output for reasoning models (show only final answer)")
    run_parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    run_parser.add_argument("--json", action="store_true", help="Output in JSON format")

    # Serve command (primary, ollama-compatible)
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible API server")
    serve_parser.add_argument("--model", help="Specific model to pre-load (optional)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind server to (default: 8000)")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to (default: 127.0.0.1)")
    serve_parser.add_argument("--max-tokens", type=int, help="Default maximum tokens for generation")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    serve_parser.add_argument("--log-level", default="info", help="Logging level (debug/info/warning/error, default: info)")
    serve_parser.add_argument("--log-json", action="store_true", help="Output logs in JSON format (for log aggregation)")
    serve_parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    serve_parser.add_argument("--json", action="store_true", help="Output startup info in JSON format")

    # Server command (alias for backward compatibility with 1.x)
    _ = subparsers.add_parser(
        "server",
        help="Start OpenAI-compatible API server (alias for serve)",
        parents=[serve_parser],
        add_help=False,
    )

    # Push command (alpha) - only show if alpha features enabled
    if os.getenv("MLXK2_ENABLE_ALPHA_FEATURES"):
        push_parser = subparsers.add_parser("push", help="ALPHA: Upload a local folder to Hugging Face")
        push_parser.add_argument("local_dir", help="Local folder to upload")
        push_parser.add_argument("repo_id", help="Target repo as org/model")
        push_parser.add_argument("--create", action="store_true", help="Create repository/branch if missing")
        # Alpha.1 safety: require --private to avoid accidental public uploads
        push_parser.add_argument(
            "--private",
            action="store_true",
            required=True,
            help="REQUIRED (alpha.1): Proceed only when targeting a private repo",
        )
        push_parser.add_argument("--branch", default="main", help="Target branch (default: main)")
        push_parser.add_argument("--commit", dest="commit_message", default="mlx-knife push", help="Commit message")
        push_parser.add_argument("--verbose", action="store_true", help="Verbose details (human output)")
        push_parser.add_argument("--check-only", action="store_true", help="Analyze workspace content; do not upload")
        push_parser.add_argument("--dry-run", action="store_true", help="Compute changes against remote; do not upload")
        push_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    args = parser.parse_args()
    
    try:
        # Handle top-level version first
        if args.version:
            if args.json:
                result = {
                    "status": "success",
                    "command": "version",
                    "data": {
                        "cli_version": __version__,
                        "json_api_spec_version": JSON_API_SPEC_VERSION,
                    },
                    "error": None,
                }
                print(format_json_output(result))
            else:
                # Use the actual command name invoked by the user
                cmd_name = os.path.basename(sys.argv[0])
                print(f"{cmd_name} {__version__}")
            sys.exit(0)

        # Initialize result for all paths
        result = None
        
        # Execute command and render per mode
        if args.command == "list":
            result = list_models(pattern=args.pattern)
            show_health = getattr(args, "show_health", False)
            show_all = getattr(args, "show_all", False)
            verbose = getattr(args, "verbose", False)
            print_result(result, render_list, args.json,
                        show_health=show_health, show_all=show_all, verbose=verbose)
        elif args.command == "health":
            result = health_check_operation(args.model)
            print_result(result, render_health, args.json)
        elif args.command == "show":
            result = show_model_operation(args.model, args.files, args.config)
            print_result(result, render_show, args.json)
        elif args.command == "pull":
            result = pull_operation(args.model)
            print_result(result, render_pull, args.json)
        elif args.command == "clone":
            # Check if alpha features are enabled (should not reach here if not, but double-check)
            if not os.getenv("MLXK2_ENABLE_ALPHA_FEATURES"):
                result = handle_error("CommandError", "Clone command requires MLXK2_ENABLE_ALPHA_FEATURES=1")
                print_result(result, None, True)  # Always JSON for this error
                sys.exit(1)

            # Handle branch parameter by modifying model spec
            model_spec = args.model
            if getattr(args, "branch", None):
                # If --branch is provided, append it to model spec
                model_spec = f"{args.model}@{args.branch}"

            from .operations.clone import clone_operation
            result = clone_operation(
                model_spec=model_spec,
                target_dir=args.target_dir,
                health_check=not getattr(args, "no_health_check", False)
            )
            print_result(result, render_clone, args.json,
                        quiet=getattr(args, "quiet", False))
        elif args.command == "rm":
            result = rm_operation(args.model, args.force)
            print_result(result, render_rm, args.json)
        elif args.command == "run":
            # Handle run command with proper parameter mapping
            result_text = run_model_enhanced(
                model_spec=args.model,
                prompt=args.prompt,  # Can be None for interactive mode
                stream=not args.no_stream,
                max_tokens=getattr(args, "max_tokens", None),
                temperature=args.temperature,
                top_p=getattr(args, "top_p", 0.9),
                repetition_penalty=getattr(args, "repetition_penalty", 1.1),
                use_chat_template=not getattr(args, "no_chat_template", False),
                json_output=args.json,
                verbose=getattr(args, "verbose", False),
                system_prompt=None,  # Not yet implemented
                hide_reasoning=getattr(args, "no_reasoning", False)
            )

            # Detect errors from run_model_enhanced (returns "Error: ..." string on failure)
            # This check must happen BEFORE the JSON/text mode split
            if result_text and isinstance(result_text, str) and result_text.startswith("Error: "):
                error_message = result_text[7:]  # Strip "Error: " prefix
                result = {
                    "status": "error",
                    "command": "run",
                    "data": None,
                    "error": {
                        "type": "execution_error",
                        "message": error_message
                    }
                }
                # Note: run_model() already printed error to stderr in text mode
                if args.json:
                    print_result(result, None, True)
                # Exit code will be 1 (handled by line 369)
            elif args.json and result_text is not None and args.prompt is not None:
                # Success case: wrap result in standard format (only for single-shot mode)
                result = {
                    "status": "success",
                    "command": "run",
                    "data": {
                        "model": args.model,
                        "prompt": args.prompt,
                        "response": result_text
                    },
                    "error": None
                }
                print(format_json_output(result))
            else:
                # For non-JSON or interactive mode, set success result
                result = {"status": "success"}
        elif args.command in ["serve", "server"]:  # Handle both serve and server aliases
            # Handle serve command
            if args.json:
                # JSON startup info
                server_info = {
                    "status": "starting",
                    "command": "serve",
                    "data": {
                        "host": args.host,
                        "port": args.port,
                        "model": getattr(args, "model", None),
                        "max_tokens": getattr(args, "max_tokens", None),
                    },
                    "error": None
                }
                print(format_json_output(server_info))
            
            # Set MLXK2_LOG_JSON if --log-json flag is present
            if getattr(args, "log_json", False):
                os.environ["MLXK2_LOG_JSON"] = "1"

            # Start server (this will run indefinitely)
            # Lazy import to avoid hard dependency on FastAPI/uvicorn at import time
            from .operations.serve import start_server
            start_server(
                model=getattr(args, "model", None),
                port=args.port,
                host=args.host,
                max_tokens=getattr(args, "max_tokens", None),
                reload=getattr(args, "reload", False),
                log_level=getattr(args, "log_level", "info"),
                verbose=getattr(args, "verbose", False),
                supervise=True
            )
            
            # Should never reach here (server runs indefinitely)
            result = {"status": "success"}
        elif args.command == "push":
            # Check if alpha features are enabled (should not reach here if not, but double-check)
            if not os.getenv("MLXK2_ENABLE_ALPHA_FEATURES"):
                result = handle_error("CommandError", "Push command requires MLXK2_ENABLE_ALPHA_FEATURES=1")
                print_result(result, None, True)  # Always JSON for this error
                sys.exit(1)
            result = push_operation(
                local_dir=args.local_dir,
                repo_id=args.repo_id,
                create=getattr(args, "create", False),
                private=getattr(args, "private", False),
                branch=getattr(args, "branch", None),
                commit_message=getattr(args, "commit_message", None),
                check_only=getattr(args, "check_only", False),
                dry_run=getattr(args, "dry_run", False),
                # Quiet mode: when emitting JSON without --verbose, suppress hub progress/log noise
                quiet=(getattr(args, "json", False) and not getattr(args, "verbose", False)),
            )
            from .output.human import render_push
            print_result(result, render_push, args.json,
                        verbose=getattr(args, "verbose", False))
        elif args.command is None:
            # No command specified - show help or JSON error depending on --json flag
            if args.json:
                result = handle_error("CommandError", "No command specified")
                print(format_json_output(result), file=sys.stdout)
                sys.exit(1)
            else:
                parser.print_help()
                sys.exit(2)
        else:
            # Unknown command - show help or JSON error depending on --json flag
            if args.json:
                result = handle_error("CommandError", f"Unknown command: {args.command}")
                print(format_json_output(result), file=sys.stdout)
                sys.exit(1)
            else:
                parser.print_help()
                sys.exit(2)

        # Exit with appropriate code (only reached for successful commands)
        sys.exit(0 if result.get("status") == "success" else 1)
            
    except Exception as e:
        # Check if --json flag was requested
        want_json = "--json" in sys.argv
        if want_json:
            error_result = handle_error("InternalError", str(e))
            print(format_json_output(error_result), file=sys.stdout)
        else:
            # Human-mode error
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
