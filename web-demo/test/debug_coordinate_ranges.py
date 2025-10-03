#!/usr/bin/env python3
"""
Debug coordinate ranges and transformations
"""

import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'new'))
from train_transducer_personalized import PersonalizedSwipeFeaturizer, determine_resample_target, resample_points, load_key_centers


def check_word(word, line_num):
    """Check coordinate transformations for a specific word"""
    data_path = '../../data/train_final_train.jsonl'

    with open(data_path, 'r') as f:
        for i, line_text in enumerate(f, 1):
            if i == line_num:
                data = json.loads(line_text)
                break

    points = data['points']

    print(f"\n{'='*60}")
    print(f"Word: '{word}' (line {line_num})")
    print(f"{'='*60}")

    # 1. Original coordinates
    print("\n1. ORIGINAL COORDINATES [0, 1]:")
    x_vals = [p['x'] for p in points]
    y_vals = [p['y'] for p in points]
    print(f"   X range: [{min(x_vals):.3f}, {max(x_vals):.3f}]")
    print(f"   Y range: [{min(y_vals):.3f}, {max(y_vals):.3f}]")
    print(f"   First 3 points: {points[:3]}")

    # 2. After transformation
    transformed = []
    for pt in points:
        transformed.append({
            'x': pt['x'] * 2.0 - 1.0,
            'y': pt['y'] * 2.0 - 1.0,
            't': pt['t']
        })

    x_trans = [p['x'] for p in transformed]
    y_trans = [p['y'] for p in transformed]
    print("\n2. AFTER TRANSFORMATION [-1, 1]:")
    print(f"   X range: [{min(x_trans):.3f}, {max(x_trans):.3f}]")
    print(f"   Y range: [{min(y_trans):.3f}, {max(y_trans):.3f}]")
    print(f"   First 3 points: {transformed[:3]}")

    # 3. After resampling
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }
    target_len = determine_resample_target(len(transformed), preprocess_cfg)
    resampled = resample_points(transformed, target_len)

    x_resamp = [p['x'] for p in resampled]
    y_resamp = [p['y'] for p in resampled]
    print(f"\n3. AFTER RESAMPLING ({len(points)} → {len(resampled)} points):")
    print(f"   X range: [{min(x_resamp):.3f}, {max(x_resamp):.3f}]")
    print(f"   Y range: [{min(y_resamp):.3f}, {max(y_resamp):.3f}]")

    # 4. Map to nearest keys
    key_centers = load_key_centers(None)
    nearest_keys = []

    for pt in resampled[:10]:  # Check first 10 points
        x, y = pt['x'], pt['y']
        best_key = None
        best_dist = float('inf')

        for char, kx, ky in key_centers:
            dist = np.sqrt((x - kx)**2 + (y - ky)**2)
            if dist < best_dist:
                best_dist = dist
                best_key = char

        nearest_keys.append(best_key)

    print(f"\n4. NEAREST KEYS (first 10 points):")
    print(f"   {''.join(nearest_keys)}")

    # Expected key sequence (approximate)
    expected_keys = []
    for char in word:
        for kc, kx, ky in key_centers:
            if kc == char:
                expected_keys.append(f"{char}({kx:.2f},{ky:.2f})")
                break

    print(f"\n5. EXPECTED KEY POSITIONS:")
    print(f"   {' '.join(expected_keys)}")

    # Extract features
    featurizer = PersonalizedSwipeFeaturizer()
    features = featurizer(resampled)
    print(f"\n6. FEATURES:")
    print(f"   Shape: {features.shape}")
    print(f"   First 5: {features[0, :5]}")

    return features


def main():
    # Test specific words with known line numbers
    test_cases = [
        ('hello', 431621),
        ('person', 4666),
        ('the', 9993),
        ('you', 4063),
    ]

    print("KEY CENTERS REFERENCE:")
    key_centers = load_key_centers(None)
    for char in 'qwertyasdfghzxcvbn':
        for kc, kx, ky in key_centers:
            if kc == char:
                print(f"  {char}: ({kx:+.2f}, {ky:+.2f})")
                break

    for word, line_num in test_cases:
        features = check_word(word, line_num)


if __name__ == '__main__':
    main()