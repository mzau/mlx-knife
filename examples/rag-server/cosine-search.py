#!/usr/bin/env python3
"""
Standalone cosine similarity search.

Usage:
    # From file
    cosine-search.py index.jsonl query.json --top-k 3

    # From stdin (pipe)
    echo '{"embedding": [...]}' | cosine-search.py index.jsonl - --top-k 3

    # JSON output (for pipes)
    cat query.json | cosine-search.py index.jsonl - --output-json

Input: Query embedding (JSON with "embedding" field)
Output: Top-K similar documents
"""

import json
import sys
import numpy as np
from argparse import ArgumentParser

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    parser = ArgumentParser(description="Cosine similarity search")
    parser.add_argument('index', help='JSONL index file')
    parser.add_argument('query', help='Query embedding JSON (- for stdin)')
    parser.add_argument('--top-k', type=int, default=3, help='Number of results')
    parser.add_argument('--output-json', action='store_true',
                       help='Output JSON (for pipes)')
    parser.add_argument('--min-score', type=float, default=0.0,
                       help='Minimum similarity score')
    args = parser.parse_args()

    # Read query
    if args.query == '-':
        query_data = json.load(sys.stdin)
    else:
        with open(args.query) as f:
            query_data = json.load(f)

    query_vec = np.array(query_data['embedding'])

    # Search index
    results = []
    with open(args.index) as f:
        for line_num, line in enumerate(f, 1):
            try:
                doc = json.loads(line)
                doc_vec = np.array(doc['embedding'])
                score = cosine_similarity(query_vec, doc_vec)

                if score >= args.min_score:
                    results.append({
                        'score': float(score),
                        'filename': doc.get('filename', f'doc_{line_num}'),
                        'filepath': doc.get('filepath', ''),
                        'text': doc.get('text', '')[:200]  # Preview
                    })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping line {line_num}: {e}",
                     file=sys.stderr)

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    top_results = results[:args.top_k]

    # Output
    if args.output_json:
        # JSON for piping
        print(json.dumps({'results': top_results}))
    else:
        # Human-readable
        for r in top_results:
            print(f"[{r['score']:.3f}] {r['filename']}")
            if r['text']:
                print(f"  Preview: {r['text']}...")
            print()

if __name__ == '__main__':
    main()
