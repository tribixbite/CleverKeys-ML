#!/usr/bin/env python3
"""
make_runtime_meta.py

Generates runtime metadata from vocabulary file to keep Web and Android deployments
perfectly synchronized. This ensures blank_id, unk_id, and character mappings are
derived programmatically rather than hardcoded.

Usage:
    python scripts/make_runtime_meta.py trained_models/data/vocab.txt > exports/runtime_meta.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]


def _maybe_load_model(checkpoint: str):
    sys.path.append(str(REPO_ROOT / "trained_models" / "nema1"))
    try:
        from export_common import load_trained_model  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Unable to import trained_models.nema1.export_common. "
            "Ensure CleverKeys repository dependencies are available."
        ) from exc

    model = load_trained_model(checkpoint)
    return model


def load_vocab(path: str) -> Dict[str, object]:
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


def derive_meta_from_model(tokens: List[str], checkpoint: str) -> Dict[str, object]:
    model = _maybe_load_model(checkpoint)

    blank_id = int(getattr(model.decoder, "blank_idx", len(tokens) - 1))

    # Build token list by decoding each index through NeMo utilities to
    # guarantee the exported order matches runtime logits.
    decode = getattr(model, "decoding", None)
    derived_tokens: List[str] = []
    if decode and hasattr(decode, "decode_tokens_to_str"):
        for idx in range(blank_id + 1):
            label = decode.decode_tokens_to_str([idx])  # type: ignore[no-untyped-call]
            if label == "":
                label = "<blank>" if idx == blank_id else f"<id_{idx}>"
            derived_tokens.append(label)
    else:
        derived_tokens = list(tokens)
        if blank_id >= len(derived_tokens):
            derived_tokens.extend([f"<id_{i}>" for i in range(len(derived_tokens), blank_id + 1)])

    # Ensure tokens cover at least the provided vocabulary order for character IDs
    vocab_tokens = {tok: i for i, tok in enumerate(tokens)}

    allowed_chars = ["'"] + [chr(c) for c in range(ord('a'), ord('z') + 1)]
    char_to_id: Dict[str, int] = {}
    for ch in allowed_chars:
        if ch in derived_tokens:
            char_to_id[ch] = derived_tokens.index(ch)
        elif ch in vocab_tokens:
            char_to_id[ch] = vocab_tokens[ch]

    unk_id = derived_tokens.index("<unk>") if "<unk>" in derived_tokens else vocab_tokens.get("<unk>", -1)

    id_to_char = {str(idx): ch for ch, idx in char_to_id.items()}

    meta = {
        "tokens": derived_tokens,
        "blank_id": blank_id,
        "unk_id": unk_id,
        "char_to_id": char_to_id,
        "id_to_char": id_to_char,
        "vocab_size": len(derived_tokens),
        "allowed_chars": sorted(set(allowed_chars)),
    }

    return meta


def main():
    parser = argparse.ArgumentParser(description="Generate runtime metadata from vocabulary file")
    parser.add_argument("vocab_file", help="Path to vocabulary text file")
    parser.add_argument("--output", "-o", help="Output JSON file (default: stdout)")
    parser.add_argument(
        "--checkpoint",
        help="Optional RNNT checkpoint (.ckpt or .nemo) to derive true runtime ordering",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON output")

    args = parser.parse_args()

    if not Path(args.vocab_file).exists():
        print(f"Error: Vocabulary file {args.vocab_file} not found", file=sys.stderr)
        sys.exit(1)

    try:
        base_meta = load_vocab(args.vocab_file)
        vocab_tokens = list(base_meta.get("tokens", []))
        if args.checkpoint:
            meta = derive_meta_from_model(vocab_tokens, args.checkpoint)
        else:
            meta = base_meta

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
