#!/usr/bin/env python3
"""
vocab_utils.py

Vocabulary filtering and normalization utilities to ensure consistent
vocabulary handling across Web and Android deployments.

Key functions:
- normalize_word: Handles apostrophe normalization and lowercasing
- filter_dictionary: Filters word list to only include valid characters
- load_runtime_meta: Loads runtime metadata from JSON
"""

import json
import re
from typing import List, Dict, Set, Any
from pathlib import Path


def normalize_word(word: str) -> str:
    """
    Normalize a word for vocabulary consistency.

    Transformations:
    - Convert to lowercase
    - Replace curly apostrophes (''') with straight apostrophes (')
    - Remove any extra whitespace

    Args:
        word: Input word to normalize

    Returns:
        Normalized word
    """
    if not word:
        return word

    # Convert to lowercase
    normalized = word.lower()

    # Replace curly apostrophes with straight apostrophes
    # U+2019 (''') -> U+0027 (')
    normalized = normalized.replace('\u2019', "'")

    # Strip whitespace
    normalized = normalized.strip()

    return normalized


def is_valid_word(word: str, allowed_chars: Set[str]) -> bool:
    """
    Check if a word contains only allowed characters.

    Args:
        word: Word to validate
        allowed_chars: Set of allowed characters

    Returns:
        True if word contains only allowed characters
    """
    if not word:
        return False

    return all(ch in allowed_chars for ch in word)


def filter_dictionary(words: List[str], allowed_chars: Set[str], normalize: bool = True) -> List[str]:
    """
    Filter dictionary words to only include those with valid characters.

    Args:
        words: List of words to filter
        allowed_chars: Set of allowed characters (typically a-z and ')
        normalize: Whether to normalize words before filtering

    Returns:
        Filtered list of valid words
    """
    valid_words = []

    for word in words:
        if normalize:
            normalized = normalize_word(word)
        else:
            normalized = word

        if normalized and is_valid_word(normalized, allowed_chars):
            valid_words.append(normalized)

    return valid_words


def load_runtime_meta(meta_path: str) -> Dict[str, Any]:
    """
    Load runtime metadata from JSON file.

    Args:
        meta_path: Path to runtime_meta.json file

    Returns:
        Dictionary containing runtime metadata

    Raises:
        FileNotFoundError: If meta file doesn't exist
        json.JSONDecodeError: If meta file is invalid JSON
    """
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_allowed_chars(meta: Dict[str, Any]) -> Set[str]:
    """
    Extract allowed characters from runtime metadata.

    Args:
        meta: Runtime metadata dictionary

    Returns:
        Set of allowed characters
    """
    return set(meta.get('allowed_chars', []))


def build_char_mappings(meta: Dict[str, Any]) -> tuple[Dict[str, int], Dict[int, str]]:
    """
    Build character-to-ID and ID-to-character mappings from metadata.

    Args:
        meta: Runtime metadata dictionary

    Returns:
        Tuple of (char_to_id, id_to_char) mappings
    """
    char_to_id = meta.get('char_to_id', {})
    id_to_char = {int(k): v for k, v in meta.get('id_to_char', {}).items()}

    return char_to_id, id_to_char


def validate_vocabulary_coverage(dictionary_path: str, meta_path: str) -> Dict[str, Any]:
    """
    Validate vocabulary coverage and provide statistics.

    Args:
        dictionary_path: Path to dictionary word list file
        meta_path: Path to runtime metadata JSON

    Returns:
        Dictionary with coverage statistics
    """
    # Load dictionary
    with open(dictionary_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]

    # Load metadata
    meta = load_runtime_meta(meta_path)
    allowed_chars = get_allowed_chars(meta)

    # Filter words
    valid_words = filter_dictionary(words, allowed_chars, normalize=True)

    # Calculate statistics
    total_words = len(words)
    valid_count = len(valid_words)
    coverage_pct = (valid_count / total_words * 100) if total_words > 0 else 0

    # Find invalid characters
    invalid_chars = set()
    for word in words:
        normalized = normalize_word(word)
        for char in normalized:
            if char not in allowed_chars:
                invalid_chars.add(char)

    return {
        'total_words': total_words,
        'valid_words': valid_count,
        'coverage_percentage': coverage_pct,
        'invalid_characters': sorted(list(invalid_chars)),
        'allowed_characters': sorted(list(allowed_chars)),
        'sample_valid_words': valid_words[:10],  # First 10 valid words
        'sample_invalid_words': [w for w in words if normalize_word(w) not in valid_words][:10]
    }


def main():
    """
    CLI interface for vocabulary utilities.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Vocabulary filtering and validation utilities")
    parser.add_argument("command", choices=["filter", "validate", "normalize"],
                       help="Command to execute")
    parser.add_argument("--dictionary", "-d", help="Path to dictionary file")
    parser.add_argument("--meta", "-m", help="Path to runtime metadata JSON")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument("--word", "-w", help="Single word to normalize (for normalize command)")

    args = parser.parse_args()

    if args.command == "normalize":
        if not args.word:
            print("Error: --word required for normalize command")
            return 1
        print(normalize_word(args.word))
        return 0

    if args.command == "validate":
        if not args.dictionary or not args.meta:
            print("Error: --dictionary and --meta required for validate command")
            return 1

        stats = validate_vocabulary_coverage(args.dictionary, args.meta)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0

    if args.command == "filter":
        if not args.dictionary or not args.meta:
            print("Error: --dictionary and --meta required for filter command")
            return 1

        # Load dictionary
        with open(args.dictionary, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]

        # Load metadata and filter
        meta = load_runtime_meta(args.meta)
        allowed_chars = get_allowed_chars(meta)
        valid_words = filter_dictionary(words, allowed_chars, normalize=True)

        # Output
        output_text = '\n'.join(valid_words)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_text)
        else:
            print(output_text)

        return 0


if __name__ == "__main__":
    exit(main() or 0)