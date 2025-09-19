#!/usr/bin/env python3
"""
Prepare a subset of real swipe traces for testing the model.
Extracts features and saves them in a format ready for TypeScript consumption.
"""

import json
import numpy as np
import random
from pathlib import Path
from typing import Dict, List
import sys
sys.path.append('.')

from train_transducer_personalized import PersonalizedSwipeFeaturizer, resample_points

# Configuration
NUM_SAMPLES = 50  # Get 50 random samples
OUTPUT_FILE = "test_traces.json"

def load_samples(n: int = 50) -> List[Dict]:
    """Load n random samples from validation set."""
    samples = []

    # Load from validation set
    with open("../../data/train_final_val.jsonl", "r") as f:
        all_samples = [json.loads(line) for line in f]

    # Get random subset, ensuring mix of common and rare words
    random.seed(42)  # For reproducibility

    # Try to get a balanced mix
    common_words = ['the', 'and', 'you', 'that', 'this', 'with', 'have', 'from', 'they', 'will']
    rare_words = ['kubernetes', 'cryptocurrency', 'blockchain', 'algorithm', 'tensorflow', 'pytorch']
    medium_words = ['hello', 'world', 'keyboard', 'phone', 'gesture', 'swipe', 'typing']

    selected = []

    # Get some from each category
    for word_list, count in [(common_words, 15), (medium_words, 15), (rare_words, 10)]:
        for word in word_list:
            matching = [s for s in all_samples if s['word'] == word]
            if matching:
                selected.extend(random.sample(matching, min(count // len(word_list), len(matching))))

    # Fill remainder with random samples
    remaining = n - len(selected)
    if remaining > 0:
        other_samples = [s for s in all_samples if s not in selected]
        selected.extend(random.sample(other_samples, min(remaining, len(other_samples))))

    return selected[:n]

def extract_features(sample: Dict) -> np.ndarray:
    """Extract features from a swipe trace."""
    featurizer = PersonalizedSwipeFeaturizer()

    # Get raw points
    raw_points = sample['points']

    # Normalize points (convert to [-1, 1] range with center at 0,0)
    normalized = []
    for idx, pt in enumerate(raw_points):
        x = float(pt.get('x', 0.5)) * 2.0 - 1.0  # Convert from [0,1] to [-1,1]
        y = float(pt.get('y', 0.5)) * 2.0 - 1.0
        t = float(pt.get('t', idx * 10.0))
        normalized.append({'x': x, 'y': y, 't': t})

    # Resample to target length (56-96 frames)
    target_len = min(96, max(56, len(normalized) * 2))
    resampled = resample_points(normalized, target_len)

    # Extract features
    features = featurizer(resampled)

    return features

def prepare_test_data():
    """Main function to prepare test data."""
    print("Loading samples...")
    samples = load_samples(NUM_SAMPLES)

    print(f"Loaded {len(samples)} samples")

    # Count word frequencies
    word_counts = {}
    for s in samples:
        word = s['word']
        word_counts[word] = word_counts.get(word, 0) + 1

    print("\nWord distribution:")
    for word, count in sorted(word_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {word}: {count}")

    print("\nExtracting features...")
    test_data = []

    for i, sample in enumerate(samples):
        if i % 10 == 0:
            print(f"  Processing {i}/{len(samples)}...")

        try:
            features = extract_features(sample)

            # Convert to list for JSON serialization
            features_list = features.tolist()

            test_data.append({
                'word': sample['word'],
                'points': sample['points'][:100],  # Limit points for file size
                'features': features_list,
                'feature_shape': list(features.shape)
            })
        except Exception as e:
            print(f"  Error processing sample {i}: {e}")
            continue

    print(f"\nSuccessfully processed {len(test_data)} samples")

    # Save to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({
            'samples': test_data,
            'metadata': {
                'num_samples': len(test_data),
                'feature_dim': 37,
                'vocab_size': 30,
                'blank_id': 29
            }
        }, f, indent=2)

    print(f"Test data saved to {OUTPUT_FILE}")

    # Print summary
    print("\n" + "="*60)
    print("TEST DATA SUMMARY")
    print("="*60)
    print(f"Total samples: {len(test_data)}")
    print(f"Feature dimensions: {test_data[0]['feature_shape'] if test_data else 'N/A'}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"File size: {Path(OUTPUT_FILE).stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    prepare_test_data()