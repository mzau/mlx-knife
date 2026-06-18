#!/usr/bin/env python3
"""
Load file contents from search results.

Usage:
    # From search results
    cosine-search.py index.jsonl query.json --output-json \
      | retrieve-files.py

    # From file
    retrieve-files.py results.json

    # Custom format
    retrieve-files.py - --format markdown

Input: JSON with "results" array (from cosine-search.py)
Output: File contents with metadata
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

def main():
    parser = ArgumentParser(description="Retrieve file contents")
    parser.add_argument('input', nargs='?', default='-',
                       help='Search results JSON (- for stdin)')
    parser.add_argument('--format', choices=['text', 'markdown', 'json'],
                       default='text', help='Output format')
    parser.add_argument('--max-files', type=int, default=10,
                       help='Maximum files to retrieve')
    parser.add_argument('--include-score', action='store_true',
                       help='Include relevance scores')
    args = parser.parse_args()

    # Read input
    if args.input == '-':
        data = json.load(sys.stdin)
    else:
        with open(args.input) as f:
            data = json.load(f)

    results = data.get('results', [])[:args.max_files]

    # Load files
    files_content = []
    for result in results:
        filepath = result.get('filepath')
        if not filepath or not Path(filepath).exists():
            print(f"Warning: File not found: {filepath}", file=sys.stderr)
            continue

        try:
            with open(filepath) as f:
                content = f.read()

            files_content.append({
                'filename': result['filename'],
                'filepath': filepath,
                'score': result['score'],
                'content': content
            })
        except Exception as e:
            print(f"Error reading {filepath}: {e}", file=sys.stderr)

    # Output
    if args.format == 'json':
        # JSON output
        print(json.dumps({'files': files_content}))

    elif args.format == 'markdown':
        # Markdown format
        for f in files_content:
            score_info = f" (relevance: {f['score']:.2f})" if args.include_score else ""
            print(f"## {f['filename']}{score_info}\n")
            print(f"```\n{f['content']}\n```\n")

    else:
        # Plain text (default)
        for f in files_content:
            score_info = f" (relevance: {f['score']:.2f})" if args.include_score else ""
            print(f"--- {f['filename']}{score_info} ---")
            print(f"{f['content']}\n")

if __name__ == '__main__':
    main()
