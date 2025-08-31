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
from .operations.show import show_model_operation
from .spec import JSON_API_SPEC_VERSION


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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mlxk2",
        description="MLX-Knife 2.0 - JSON-first model management"
    )
    
    # Add version argument (supports --json)
    parser.add_argument("--version", action="store_true", help="Show version information and exit")
    parser.add_argument("--json", action="store_true", help="Output in JSON format (with --version or per command)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all cached models")
    list_parser.add_argument("pattern", nargs="?", help="Filter models by pattern (optional)")
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

        # In alpha version, --json flag is required for broke-cluster compatibility
        if args.command and not hasattr(args, 'json'):
            result = handle_error("CommandError", "Internal error: --json flag not found")
        elif args.command and not args.json:
            result = handle_error("JsonRequired", "MLX-Knife 2.0-alpha requires --json flag. Use: mlxk2 " + args.command + " --json")
        elif args.command == "list":
            result = list_models(pattern=args.pattern)
        elif args.command == "health":
            result = health_check_operation(args.model)
        elif args.command == "show":
            result = show_model_operation(args.model, args.files, args.config)
        elif args.command == "pull":
            result = pull_operation(args.model)
        elif args.command == "rm":
            result = rm_operation(args.model, args.force)
        elif args.command is None:
            result = handle_error("CommandError", "No command specified")
        else:
            result = handle_error("CommandError", f"Unknown command: {args.command}")
            
        print(format_json_output(result))
        
        # Exit with appropriate code
        if result["status"] == "error":
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        error_result = handle_error("InternalError", str(e))
        print(format_json_output(error_result))
        sys.exit(1)


if __name__ == "__main__":
    main()
