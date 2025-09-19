#!/usr/bin/env python3
"""
validate_vocab_system.py

Quick validation script to ensure the vocabulary system is working correctly.
Performs sanity checks on the runtime metadata and vocabulary filtering.
"""

import json
import sys
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent))

from vocab_utils import load_runtime_meta, normalize_word, filter_dictionary, get_allowed_chars


def validate_runtime_meta(meta_path):
    """Validate runtime metadata structure and content"""
    print(f"📋 Validating runtime metadata: {meta_path}")

    try:
        meta = load_runtime_meta(meta_path)
    except Exception as e:
        print(f"❌ Failed to load metadata: {e}")
        return False

    # Check required fields
    required_fields = ['tokens', 'blank_id', 'unk_id', 'char_to_id', 'id_to_char', 'vocab_size', 'allowed_chars']
    for field in required_fields:
        if field not in meta:
            print(f"❌ Missing required field: {field}")
            return False

    # Validate token structure
    tokens = meta['tokens']
    if not isinstance(tokens, list) or len(tokens) == 0:
        print(f"❌ Invalid tokens structure")
        return False

    # Check special tokens
    blank_id = meta['blank_id']
    unk_id = meta['unk_id']

    if blank_id is None or blank_id >= len(tokens):
        print(f"❌ Invalid blank_id: {blank_id}")
        return False

    if unk_id is None or unk_id >= len(tokens):
        print(f"❌ Invalid unk_id: {unk_id}")
        return False

    # NeMo models use empty string as the functional blank (at index 29)
    # while keeping "<blank>" label at index 0 for compatibility
    if tokens[blank_id] != "":
        print(f"❌ Blank token mismatch: expected empty string '', got '{tokens[blank_id]}'")
        print(f"   Note: NeMo models use blank_id={blank_id} with empty string as functional blank")
        return False

    if tokens[unk_id] != "<unk>":
        print(f"❌ UNK token mismatch: expected '<unk>', got '{tokens[unk_id]}'")
        return False

    # Validate character mappings
    char_to_id = meta['char_to_id']
    expected_chars = set(["'"] + [chr(c) for c in range(ord('a'), ord('z') + 1)])

    missing_chars = expected_chars - set(char_to_id.keys())
    if missing_chars:
        print(f"❌ Missing characters in char_to_id: {sorted(missing_chars)}")
        return False

    # Check apostrophe mapping
    if "'" not in char_to_id:
        print(f"❌ Apostrophe not found in char_to_id")
        return False

    print(f"✅ Runtime metadata validation passed")
    print(f"   Vocabulary size: {meta['vocab_size']}")
    print(f"   Blank ID: {blank_id} ('{tokens[blank_id]}')")
    print(f"   UNK ID: {unk_id} ('{tokens[unk_id]}')")
    print(f"   Character range: {len(char_to_id)} characters")
    return True


def test_normalization():
    """Test word normalization functions"""
    print(f"\n🔤 Testing word normalization")

    test_cases = [
        ("Hello", "hello"),
        ("DON'T", "don't"),  # Curly apostrophe
        ("don't", "don't"),  # Straight apostrophe
        ("  spaced  ", "spaced"),
        ("CamelCase", "camelcase"),
        ("'quoted'", "'quoted'"),
    ]

    all_passed = True
    for input_word, expected in test_cases:
        result = normalize_word(input_word)
        if result == expected:
            print(f"   ✅ '{input_word}' → '{result}'")
        else:
            print(f"   ❌ '{input_word}' → '{result}' (expected '{expected}')")
            all_passed = False

    return all_passed


def test_filtering(meta_path):
    """Test vocabulary filtering"""
    print(f"\n🗂️  Testing vocabulary filtering")

    try:
        meta = load_runtime_meta(meta_path)
        allowed_chars = get_allowed_chars(meta)
    except Exception as e:
        print(f"❌ Failed to load metadata: {e}")
        return False

    test_words = [
        "hello",      # Valid
        "don't",      # Valid with apostrophe
        "café",       # Invalid (accented character)
        "hello123",   # Invalid (numbers)
        "hello-world", # Invalid (hyphen)
        "it's",       # Valid
        "",           # Empty
        "a",          # Single character
    ]

    expected_valid = {"hello", "don't", "it's", "a"}

    filtered = filter_dictionary(test_words, allowed_chars, normalize=True)
    filtered_set = set(filtered)

    if filtered_set == expected_valid:
        print(f"✅ Filtering test passed")
        print(f"   Valid words: {sorted(filtered)}")
        return True
    else:
        print(f"❌ Filtering test failed")
        print(f"   Expected: {sorted(expected_valid)}")
        print(f"   Got: {sorted(filtered)}")
        return False


def test_character_coverage(meta_path):
    """Test that all expected characters are covered"""
    print(f"\n🔍 Testing character coverage")

    try:
        meta = load_runtime_meta(meta_path)
    except Exception as e:
        print(f"❌ Failed to load metadata: {e}")
        return False

    char_to_id = meta['char_to_id']
    expected_chars = {"'"} | {chr(c) for c in range(ord('a'), ord('z') + 1)}

    covered_chars = set(char_to_id.keys())
    missing = expected_chars - covered_chars
    extra = covered_chars - expected_chars

    if missing:
        print(f"❌ Missing characters: {sorted(missing)}")
        return False

    if extra:
        print(f"❌ Unexpected characters: {sorted(extra)}")
        return False

    print(f"✅ Character coverage complete")
    print(f"   Characters: {len(covered_chars)} (' + a-z)")
    return True


def main():
    """Run all validation tests"""
    print("CleverKeys Vocabulary System Validation")
    print("=" * 50)

    # File paths
    meta_path = "exports/runtime_meta.json"

    # Check if files exist
    if not Path(meta_path).exists():
        print(f"❌ Runtime metadata not found: {meta_path}")
        print("Run: python scripts/make_runtime_meta.py trained_models/data/vocab.txt --output exports/runtime_meta.json")
        return 1

    # Run tests
    tests = [
        ("Runtime Metadata", lambda: validate_runtime_meta(meta_path)),
        ("Word Normalization", test_normalization),
        ("Vocabulary Filtering", lambda: test_filtering(meta_path)),
        ("Character Coverage", lambda: test_character_coverage(meta_path)),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {name} test failed")
        except Exception as e:
            print(f"❌ {name} test error: {e}")

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All vocabulary system tests passed!")
        print("\nNext steps:")
        print("1. Copy runtime_meta.json to web-demo/ and android/assets/")
        print("2. Update your web/Android code to use the new utilities")
        print("3. Test beam search with filtered vocabulary")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())