#!/usr/bin/env python3
"""
Create embedding index from files.

Usage:
    # Index directory
    index-files.py ./src --output index.jsonl

    # Specific files
    index-files.py file1.py file2.py --output index.jsonl

    # From glob
    index-files.py "*.py" --output code-index.jsonl

    # Custom model
    index-files.py ./docs --model bge-small --output docs-index.jsonl

Output: JSONL file with embeddings
"""

import json
import sys
import subprocess
import os
from pathlib import Path
from argparse import ArgumentParser
import glob

def embed_text(text, model="bge-small-en-v1.5-4bit"):
    """Embed text using mlxk (embed is experimental in 2.0.7 → alpha gate)."""
    result = subprocess.run(
        ["mlxk", "embed", model, "-"],
        input=text.encode(),
        capture_output=True,
        env={**os.environ, "MLXK2_ENABLE_ALPHA_FEATURES": "1"}
    )

    if result.returncode != 0:
        raise RuntimeError(f"Embedding failed: {result.stderr.decode()}")

    return json.loads(result.stdout)

def main():
    parser = ArgumentParser(description="Create embedding index")
    parser.add_argument('paths', nargs='+', help='Files or directories')
    parser.add_argument('--output', '-o', required=True,
                       help='Output JSONL file')
    parser.add_argument('--model', default='bge-small-en-v1.5-4bit',
                       help='Embedding model (MLX encoder or decoder embedder)')
    parser.add_argument('--pattern', default='*',
                       help='File pattern for directories')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Recursive directory search')
    args = parser.parse_args()

    # Collect files
    files = []
    for path_pattern in args.paths:
        # Glob expansion
        for path in glob.glob(path_pattern, recursive=args.recursive):
            p = Path(path)
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                pattern = f"**/{args.pattern}" if args.recursive else args.pattern
                files.extend(p.glob(pattern))

    files = [f for f in files if f.is_file()]

    print(f"Indexing {len(files)} files...", file=sys.stderr)

    # Index files
    with open(args.output, 'w') as out:
        for i, filepath in enumerate(files, 1):
            try:
                # Read file
                content = filepath.read_text()

                # Embed
                print(f"[{i}/{len(files)}] {filepath.name}", file=sys.stderr)
                embedding_data = embed_text(content, args.model)

                # Add metadata
                entry = {
                    **embedding_data,
                    'filename': filepath.name,
                    'filepath': str(filepath.absolute())
                }

                # Write JSONL
                out.write(json.dumps(entry) + '\n')

            except Exception as e:
                print(f"Error processing {filepath}: {e}", file=sys.stderr)

    print(f"Index created: {args.output}", file=sys.stderr)

if __name__ == '__main__':
    main()
