#!/usr/bin/env python3
"""
vocab_integration_example.py

Example demonstrating how to use the new vocabulary metadata system
for consistent vocabulary handling across Web and Android deployments.
"""

import sys
import json
from pathlib import Path

# Add scripts to path for imports
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from vocab_utils import (
    load_runtime_meta,
    filter_dictionary,
    validate_vocabulary_coverage,
    normalize_word
)


def example_generate_runtime_meta():
    """Example: Generate runtime metadata from vocabulary file"""
    print("=== Generating Runtime Metadata ===")

    # This would typically be run once during build/export process
    vocab_file = "trained_models/data/vocab.txt"
    meta_file = "exports/runtime_meta.json"

    print(f"Generating runtime metadata:")
    print(f"  Input: {vocab_file}")
    print(f"  Output: {meta_file}")

    # In practice, you'd run:
    # python scripts/make_runtime_meta.py trained_models/data/vocab.txt --output exports/runtime_meta.json --pretty

    print("Command: python scripts/make_runtime_meta.py trained_models/data/vocab.txt --output exports/runtime_meta.json --pretty")
    print()


def example_load_and_validate():
    """Example: Load metadata and validate dictionary coverage"""
    print("=== Loading Metadata and Validating Dictionary ===")

    try:
        # Load runtime metadata
        meta = load_runtime_meta("exports/runtime_meta.json")
        print(f"Loaded metadata:")
        print(f"  Vocabulary size: {meta['vocab_size']}")
        print(f"  Blank ID: {meta['blank_id']}")
        print(f"  UNK ID: {meta['unk_id']}")
        print(f"  Allowed characters: {len(meta['allowed_chars'])}")
        print(f"  Character range: {min(meta['allowed_chars'])} to {max(meta['allowed_chars'])}")
        print()

        # Load a sample of dictionary words
        with open("vocab/final_vocab.txt", "r", encoding="utf-8") as f:
            # Use first 10000 words for faster processing
            sample_words = [line.strip() for line in f][:10000]

        print(f"Testing with {len(sample_words)} dictionary words...")

        # Validate coverage
        stats = validate_vocabulary_coverage("vocab/final_vocab.txt", "exports/runtime_meta.json")

        print(f"Coverage Statistics:")
        print(f"  Total words: {stats['total_words']:,}")
        print(f"  Valid words: {stats['valid_words']:,}")
        print(f"  Coverage: {stats['coverage_percentage']:.1f}%")

        if stats['invalid_characters']:
            print(f"  Invalid characters found: {stats['invalid_characters']}")
        else:
            print("  ✓ All characters in dictionary are valid!")

        print(f"  Sample valid words: {', '.join(stats['sample_valid_words'][:5])}")

        if stats['sample_invalid_words']:
            print(f"  Sample invalid words: {', '.join(stats['sample_invalid_words'][:3])}")

        print()

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Make sure to run the runtime meta generation first.")
        print()


def example_normalization():
    """Example: Word normalization"""
    print("=== Word Normalization Examples ===")

    test_words = [
        "Hello",          # Uppercase
        "don't",          # Straight apostrophe
        "don't",          # Curly apostrophe (U+2019)
        "  spaced  ",     # Extra whitespace
        "CamelCase",      # Mixed case
        "'quoted'",       # Starting/ending apostrophes
    ]

    for word in test_words:
        normalized = normalize_word(word)
        print(f"  '{word}' → '{normalized}'")

    print()


def example_export_script_usage():
    """Example: Using updated export script with vocabulary"""
    print("=== Updated Export Script Usage ===")

    print("The export_rnnt_step.py script now supports automatic blank_id derivation:")
    print()

    example_commands = [
        # Basic usage (uses default blank_id=0)
        "python trained_models/nema1/export_rnnt_step.py --nemo_model model.nemo",

        # With vocabulary file (derives blank_id automatically)
        "python trained_models/nema1/export_rnnt_step.py --nemo_model model.nemo --vocab trained_models/data/vocab.txt",

        # Full export with custom paths
        "python trained_models/nema1/export_rnnt_step.py --nemo_model model.nemo --vocab trained_models/data/vocab.txt --onnx_out exports/step.onnx --pte_out exports/step.pte"
    ]

    for i, cmd in enumerate(example_commands, 1):
        print(f"  {i}. {cmd}")

    print()
    print("Key improvements:")
    print("  • Automatic blank_id derivation from vocabulary file")
    print("  • No more hardcoded blank_id=0 assumptions")
    print("  • Consistent behavior across Web and Android")
    print()


def example_web_integration():
    """Example: Web integration pattern"""
    print("=== Web Integration Example ===")

    web_code = '''
// Load runtime metadata
const meta = await VocabMetaUtils.loadVocabMeta("runtime_meta.json");

// Load dictionary words
const words = await fetch("dictionary.txt").then(r => r.text()).then(t => t.split("\\n"));

// Create filtered lexicon
const lexicon = VocabMetaUtils.createLexicon(words, meta);

// Use in beam search
const decodedWords = await rnntBeamSearchWord(
    encoder, step, featuresBFT, F, T, L, H, D, lexicon,
    {
        blankId: meta.blankId,  // ← Derived, not hardcoded
        beamSize: 16,
        prunePerBeam: 6,
        maxSymbols: 20,
        lmLambda: 0.4,
        returnTopK: 5
    }
);
'''

    print("JavaScript/TypeScript pattern:")
    print(web_code)


def example_android_integration():
    """Example: Android integration pattern"""
    print("=== Android Integration Example ===")

    kotlin_code = '''
// Load runtime metadata from assets
val meta = VocabMetaUtils.loadVocabMetaFromAssets(context, "runtime_meta.json")

// Load dictionary words
val words = loadWordsFromAssets(context, "dictionary.txt")

// Create filtered lexicon
val lexicon = VocabMetaUtils.createLexicon(words, meta)

// Use in beam search decoder
val decoder = RNNTBeamDecoder(
    encProg, stepProg, L, H, D,
    blankId = meta.blank_id  // ← Derived, not hardcoded
)
val results = decoder.decode(features, lexicon)
'''

    print("Kotlin pattern:")
    print(kotlin_code)


def main():
    """Run all examples"""
    print("CleverKeys Vocabulary Integration Examples")
    print("=" * 50)
    print()

    example_generate_runtime_meta()
    example_load_and_validate()
    example_normalization()
    example_export_script_usage()
    example_web_integration()
    example_android_integration()

    print("Summary:")
    print("• Runtime metadata ensures Web/Android sync")
    print("• Vocabulary filtering prevents <unk> expansions")
    print("• Apostrophe normalization handles typography")
    print("• Export scripts derive blank_id automatically")
    print("• High vocabulary coverage with clean character set")


if __name__ == "__main__":
    main()