#!/usr/bin/env python3
"""
Generate test data for companion word to verify JavaScript implementation
"""

import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'new'))
from train_transducer_personalized import PersonalizedSwipeFeaturizer, determine_resample_target, resample_points


def get_companion_data():
    """Get companion swipe data from line 22440"""
    data_path = '../../data/train_final_train.jsonl'
    with open(data_path, 'r') as f:
        for i, line in enumerate(f, 1):
            if i == 22440:
                data = json.loads(line)
                return data['points']
    return None


def main():
    print("="*70)
    print("COMPANION TEST DATA GENERATION")
    print("="*70)

    # Get companion data
    points = get_companion_data()
    if points is None:
        print("ERROR: Could not load companion data")
        return

    print(f"\nLoaded {len(points)} points for 'companion' from line 22440")

    # Show coordinate ranges
    x_vals = [p['x'] for p in points]
    y_vals = [p['y'] for p in points]
    print(f"Original X range: [{min(x_vals):.3f}, {max(x_vals):.3f}]")
    print(f"Original Y range: [{min(y_vals):.3f}, {max(y_vals):.3f}]")

    # Transform to [-1,1]
    transformed = []
    for pt in points:
        transformed.append({
            'x': pt['x'] * 2.0 - 1.0,
            'y': pt['y'] * 2.0 - 1.0,
            't': pt['t']
        })

    tx_vals = [p['x'] for p in transformed]
    ty_vals = [p['y'] for p in transformed]
    print(f"Transformed X range: [{min(tx_vals):.3f}, {max(tx_vals):.3f}]")
    print(f"Transformed Y range: [{min(ty_vals):.3f}, {max(ty_vals):.3f}]")

    # Preprocessing config
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }

    # Resample
    target_len = determine_resample_target(len(transformed), preprocess_cfg)
    resampled = resample_points(transformed, target_len)
    print(f"\nResampled: {len(points)} → {len(resampled)} points")

    # Featurize
    featurizer = PersonalizedSwipeFeaturizer()
    features = featurizer(resampled)
    print(f"Features shape: {features.shape}")
    print(f"First 5 features: {features[0, :5]}")

    # Generate JavaScript-friendly output
    print("\n" + "="*70)
    print("JAVASCRIPT TEST DATA")
    print("="*70)

    print("\n// Original points array (first 5):")
    print("const companionPoints = [")
    for i, pt in enumerate(points[:5]):
        print(f"  {{ x: {pt['x']:.6f}, y: {pt['y']:.6f}, t: {pt['t']} }},")
    print("  // ... " + str(len(points) - 5) + " more points")
    print("];")

    print(f"\n// Expected feature shape: [{features.shape[0]}, {features.shape[1]}]")
    print("// First frame features:")
    print(f"const expectedFirstFrame = [{', '.join([f'{v:.6f}' for v in features[0, :5]])}];")

    print("\n// Full feature matrix (first 3 frames):")
    print("const expectedFeatures = [")
    for i in range(min(3, features.shape[0])):
        print(f"  [{', '.join([f'{v:.6f}' for v in features[i, :5]])}...], // frame {i}")
    print("];")

    # Save test data for JavaScript
    test_data = {
        'word': 'companion',
        'line_number': 22440,
        'points': points,
        'resampled_count': len(resampled),
        'feature_shape': [features.shape[0], features.shape[1]],
        'first_5_features': features[0, :5].tolist(),
        'expected_tokens': [4, 16, 14, 17, 2, 15, 10, 16, 15]  # c-o-m-p-a-n-i-o-n
    }

    output_path = 'companion_test_data.json'
    with open(output_path, 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"\n✅ Test data saved to: {output_path}")
    print("\nExpected word: 'companion'")
    print("Expected tokens: [4, 16, 14, 17, 2, 15, 10, 16, 15]")


if __name__ == '__main__':
    main()