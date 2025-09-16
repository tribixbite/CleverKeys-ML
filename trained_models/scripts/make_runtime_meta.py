#!/usr/bin/env python3
"""
make_runtime_meta.py

Generates runtime metadata from vocabulary file to keep Web and Android deployments
perfectly synchronized. This ensures blank_id, unk_id, and character mappings are
derived programmatically rather than hardcoded.

Usage:
    python scripts/make_runtime_meta.py trained_models/data/vocab.txt > exports/runtime_meta.json
"""

import json
import sys
import argparse
from pathlib import Path


def load_vocab(path: str):
    """
    Load vocabulary from file and generate runtime metadata.

    Expected vocab format:
        <blank>
        '
        a
        b
        ...
        z
        <unk>

    Returns dict with:
        - tokens: list of all tokens
        - blank_id: ID for <blank> token
        - unk_id: ID for <unk> token
        - char_to_id: mapping from valid chars to IDs
        - id_to_char: inverse mapping
    """
    with open(path, "r", encoding="utf-8") as f:
        tokens = [line.strip() for line in f if line.strip()]

    # Create token-to-index mapping
    token_to_idx = {tok: i for i, tok in enumerate(tokens)}

    # Find special token IDs
    blank_id = token_to_idx.get("<blank>")
    unk_id = token_to_idx.get("<unk>")

    # Valid character set: a-z and apostrophe (no blank/unk in expansions)
    allowed_chars = set(["'"] + [chr(c) for c in range(ord('a'), ord('z') + 1)])

    # Create character mappings for trie building and beam search
    char_to_id = {ch: token_to_idx[ch] for ch in allowed_chars if ch in token_to_idx}
    id_to_char = {v: k for k, v in char_to_id.items()}

    meta = {
        "tokens": tokens,
        "blank_id": blank_id,
        "unk_id": unk_id,
        "char_to_id": char_to_id,   # e.g. {"'": 1, "a": 2, ..., "z": 27}
        "id_to_char": id_to_char,   # inverse mapping
        "vocab_size": len(tokens),
        "allowed_chars": sorted(list(allowed_chars))  # for validation
    }

    return meta


def main():
    parser = argparse.ArgumentParser(description="Generate runtime metadata from vocabulary file")
    parser.add_argument("vocab_file", help="Path to vocabulary text file")
    parser.add_argument("--output", "-o", help="Output JSON file (default: stdout)")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON output")

    args = parser.parse_args()

    if not Path(args.vocab_file).exists():
        print(f"Error: Vocabulary file {args.vocab_file} not found", file=sys.stderr)
        sys.exit(1)

    try:
        meta = load_vocab(args.vocab_file)

        # Validate that we found required tokens
        if meta["blank_id"] is None:
            print("Warning: <blank> token not found in vocabulary", file=sys.stderr)
        if meta["unk_id"] is None:
            print("Warning: <unk> token not found in vocabulary", file=sys.stderr)

        # Format JSON output
        json_kwargs = {
            "ensure_ascii": False,
            "indent": 2 if args.pretty else None
        }

        json_output = json.dumps(meta, **json_kwargs)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_output)
            print(f"Runtime metadata written to {args.output}", file=sys.stderr)
        else:
            print(json_output)

    except Exception as e:
        print(f"Error processing vocabulary file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()