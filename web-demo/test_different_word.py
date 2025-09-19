#!/usr/bin/env python3
"""Test with a different word from validation set"""

import json
import numpy as np
from test_with_resampling import ResamplingDecoder

# Load decoder
decoder = ResamplingDecoder(
    encoder_path='encoder_fresh.onnx',
    decoder_path='rnnt_step_fresh.onnx',
    runtime_meta_path='../trained_models/nema1/runtime_meta.json',
    words_path='../trained_models/nema1/words.txt'
)

# Try different validation samples
val_file = '../trained_models/nema1/personalized_tuning/20250918_105357/val_short_common.jsonl'

print("Testing first 5 validation samples:")
print("=" * 60)

with open(val_file, 'r') as f:
    for idx, line in enumerate(f):
        if idx >= 5:
            break

        data = json.loads(line)
        word = data['word']
        points = data['points']

        print(f"\nSample {idx + 1}: Expected word = '{word}'")
        print(f"  Points: {len(points)}")

        # Run decoder
        try:
            results = decoder.decode(points, beam_size=16, top_k=3)

            if results:
                print(f"  Top predictions:")
                for i, (pred_word, score) in enumerate(results, 1):
                    match = "✓" if pred_word == word else "✗"
                    print(f"    {i}. {pred_word:15} (score={score:.2f}) {match}")
            else:
                print(f"  No valid predictions!")

            # Check if correct word was in top 3
            if results:
                top3_words = [w for w, _ in results[:3]]
                if word in top3_words:
                    print(f"  ✓ Correct word in top 3!")
                else:
                    print(f"  ✗ Correct word NOT in top 3")
        except Exception as e:
            print(f"  Error: {e}")

print("\n" + "=" * 60)