#!/usr/bin/env python3
"""MLX-Knife 2.0 CLI - JSON-first architecture."""

import argparse
import json
import sys
from typing import Dict, Any

from .operations.list import list_models


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
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all cached models")
    
    args = parser.parse_args()
    
    try:
        if args.command == "list":
            result = list_models()
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