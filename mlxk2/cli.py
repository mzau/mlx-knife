#!/usr/bin/env python3
"""MLX-Knife 2.0 CLI - JSON-first architecture."""

import argparse
import json
import sys
from typing import Dict, Any

from . import __version__
from .operations.list import list_models
from .operations.health import health_check_operation
from .operations.pull import pull_operation
from .operations.rm import rm_operation
from .operations.push import push_operation
from .operations.show import show_model_operation
from .spec import JSON_API_SPEC_VERSION
from .output.human import (
    render_list,
    render_health,
    render_show,
    render_pull,
    render_rm,
)


def format_json_output(data: Dict[str, Any]) -> str:
    """Format output as JSON."""
    return json.dumps(data, indent=2)


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
            print(format_json_output(err))
            self.exit(2)
        super().error(message)


def main():
    """Main CLI entry point."""
    parser = MLXKArgumentParser(
        prog="mlxk2",
        description="MLX-Knife 2.0 - JSON-first model management"
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
    
    # Remove command
    rm_parser = subparsers.add_parser("rm", help="Delete a model")
    rm_parser.add_argument("model", help="Model name to delete")
    rm_parser.add_argument("-f", "--force", action="store_true", help="Delete without confirmation")
    rm_parser.add_argument("--json", action="store_true", help="Output in JSON format")

    # Push command (experimental)
    push_parser = subparsers.add_parser("push", help="EXPERIMENTAL: Upload a local folder to Hugging Face")
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
                print(f"mlxk2 {__version__}")
            sys.exit(0)

        # Execute command and render per mode
        if args.command == "list":
            result = list_models(pattern=args.pattern)
            if args.json:
                print(format_json_output(result))
            else:
                show_health = getattr(args, "show_health", False)
                show_all = getattr(args, "show_all", False)
                verbose = getattr(args, "verbose", False)
                print(render_list(result, show_health=show_health, show_all=show_all, verbose=verbose))
        elif args.command == "health":
            result = health_check_operation(args.model)
            if args.json:
                print(format_json_output(result))
            else:
                print(render_health(result))
        elif args.command == "show":
            result = show_model_operation(args.model, args.files, args.config)
            if args.json:
                print(format_json_output(result))
            else:
                print(render_show(result))
        elif args.command == "pull":
            result = pull_operation(args.model)
            if args.json:
                print(format_json_output(result))
            else:
                print(render_pull(result))
        elif args.command == "rm":
            result = rm_operation(args.model, args.force)
            if args.json:
                print(format_json_output(result))
            else:
                print(render_rm(result))
        elif args.command == "push":
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
            if args.json:
                print(format_json_output(result))
            else:
                from .output.human import render_push
                print(render_push(result, verbose=getattr(args, "verbose", False)))
        elif args.command is None:
            result = handle_error("CommandError", "No command specified")
            print(format_json_output(result))
        else:
            result = handle_error("CommandError", f"Unknown command: {args.command}")
            print(format_json_output(result))

        # Exit with appropriate code
        sys.exit(0 if result.get("status") == "success" else 1)
            
    except Exception as e:
        error_result = handle_error("InternalError", str(e))
        print(format_json_output(error_result))
        sys.exit(1)


if __name__ == "__main__":
    main()
