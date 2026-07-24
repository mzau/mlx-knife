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

Input: Query embedding as emitted by `mlxk embed` (JSON with "embedding" and
       "metadata"); index is a JSONL of the same records (see index-files.py)
Output: Top-K similar documents

Same-model guard: the query and every index line must carry the identity that
`mlxk embed` stamps into "metadata" — (model, content_hash, device, dimensions)
— and those identities must match. Vectors from different models, revisions or
devices share no vector space; ranking them silently produces garbage, so any
mismatch (or missing metadata) aborts with exit code 2 instead.
"""

import json
import sys
import numpy as np
from argparse import ArgumentParser

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embedding_identity(record):
    """Extract the same-model identity stamped by `mlxk embed`.

    Returns (model, content_hash, device, dimensions), or None if the record
    carries no metadata (outdated index format or foreign source).
    """
    meta = record.get('metadata')
    if not isinstance(meta, dict):
        return None
    return (meta.get('model'), meta.get('content_hash'),
            meta.get('device'), meta.get('dimensions'))

def reject(message):
    """Refuse to rank: clear error, non-zero exit (same-model contract violation)."""
    print(f"Error: {message}", file=sys.stderr)
    print("Hint: query and index must come from the same `mlxk embed` model "
          "(same model, content_hash, device, dimensions). "
          "Re-index or re-embed with the matching model.", file=sys.stderr)
    sys.exit(2)

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

    # Same-model guard: never compare vectors across embedding identities.
    query_ident = embedding_identity(query_data)
    if query_ident is None:
        reject("query has no 'metadata' (expected the output of `mlxk embed`)")
    if query_ident[3] != len(query_vec):
        reject(f"query vector length {len(query_vec)} does not match "
               f"metadata.dimensions {query_ident[3]}")

    # Search index
    results = []
    with open(args.index) as f:
        for line_num, line in enumerate(f, 1):
            try:
                doc = json.loads(line)
                doc_ident = embedding_identity(doc)
                if doc_ident is None:
                    reject(f"index line {line_num} has no 'metadata' — "
                           f"outdated or foreign index")
                if doc_ident != query_ident:
                    reject(f"index line {line_num} identity {doc_ident} "
                           f"does not match query identity {query_ident}")
                doc_vec = np.array(doc['embedding'])
                if len(doc_vec) != len(query_vec):
                    reject(f"index line {line_num} vector length {len(doc_vec)} "
                           f"!= query length {len(query_vec)}")
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
