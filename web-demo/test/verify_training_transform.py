#!/usr/bin/env python3
"""
Verify what coordinates training ACTUALLY uses by tracing through the dataset class
"""

import json
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'new'))

# Import the actual dataset class used in training
from train_transducer_personalized import PersonalizedSwipeDataset, PersonalizedSwipeFeaturizer, clamp


def trace_training_data():
    """Trace exactly what the training dataset produces"""

    # Create dataset exactly as training does
    manifest_path = "../../data/train_final_train.jsonl"
    vocab_path = "../../data/vocab.txt"

    # Load vocab
    with open(vocab_path, "r") as f:
        vocab_list = [line.strip() for line in f if line.strip()]

    # Create vocab dict as training does
    vocab = {char: idx for idx, char in enumerate(vocab_list)}

    # Create featurizer and preprocess config as training does
    featurizer = PersonalizedSwipeFeaturizer(mobile_features=False)
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }

    # Create dataset
    dataset = PersonalizedSwipeDataset(
        manifest_path=manifest_path,
        vocab=vocab,
        max_trace_len=200,
        preprocess_cfg=preprocess_cfg,
        featurizer=featurizer,
        augmenter=None,
        is_training=False
    )

    print("Tracing training dataset processing")
    print("="*60)

    # Find the 'hello' sample (line 431621 - 1 for 0-indexing)
    target_idx = 431620

    # Get the sample
    features, feature_len, labels, label_len = dataset[target_idx]

    print(f"Sample {target_idx}:")
    print(f"  Features shape: {features.shape}")
    print(f"  Feature length: {feature_len}")
    print(f"  Labels: {labels[:label_len]}")

    # Decode labels to text
    label_text = ''.join([vocab_list[l] if l < len(vocab_list) else '?' for l in labels[:label_len]])
    print(f"  Label text: '{label_text}'")

    # Check feature values
    print(f"\nFeature statistics:")
    print(f"  X (col 0): [{features[:, 0].min():.3f}, {features[:, 0].max():.3f}]")
    print(f"  Y (col 1): [{features[:, 1].min():.3f}, {features[:, 1].max():.3f}]")

    # Also manually process the same line to compare
    print(f"\nManual processing of same line:")

    with open(manifest_path, 'r') as f:
        for i, line in enumerate(f):
            if i == target_idx:
                data = json.loads(line)
                break

    word = data['word']
    points = data['points']
    print(f"  Word: '{word}'")
    print(f"  Raw points: {len(points)}")

    # Check raw coordinate ranges
    raw_xs = [p['x'] for p in points]
    raw_ys = [p['y'] for p in points]
    print(f"  Raw X: [{min(raw_xs):.3f}, {max(raw_xs):.3f}]")
    print(f"  Raw Y: [{min(raw_ys):.3f}, {max(raw_ys):.3f}]")

    # Now trace through what _load_sample does
    print(f"\nTracing _load_sample:")

    # This is what SwipeDataset._load_sample does:
    from train_transducer_personalized import resample_points, determine_resample_target

    # Step 1: _prepare_points
    prepared = []
    for idx, pt in enumerate(points):
        raw_x = float(pt.get("x", 0.0))
        raw_y = float(pt.get("y", 0.0))

        # The actual code in _prepare_points
        centered_x = raw_x * 2.0 - 1.0
        centered_y = raw_y * 2.0 - 1.0
        centered_x = clamp(centered_x, -1.5, 1.5)
        centered_y = clamp(centered_y, -1.5, 1.5)

        raw_t = float(pt.get("t", idx * 10.0))
        prepared.append({
            "x": centered_x,
            "y": centered_y,
            "t": raw_t,
        })

    prep_xs = [p['x'] for p in prepared]
    prep_ys = [p['y'] for p in prepared]
    print(f"  After _prepare_points: X[{min(prep_xs):.3f}, {max(prep_xs):.3f}], Y[{min(prep_ys):.3f}, {max(prep_ys):.3f}]")

    # Step 2: Resample
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }
    target_len = determine_resample_target(len(prepared), preprocess_cfg)
    resampled = resample_points(prepared, target_len)
    print(f"  Resampled to {len(resampled)} points")

    # Step 3: Feature extraction (use same featurizer as dataset)
    features_manual = featurizer(resampled)

    print(f"  Manual features shape: {features_manual.shape}")
    print(f"  Manual X: [{features_manual[:, 0].min():.3f}, {features_manual[:, 0].max():.3f}]")
    print(f"  Manual Y: [{features_manual[:, 1].min():.3f}, {features_manual[:, 1].max():.3f}]")

    # Compare with dataset features
    print(f"\nComparison:")
    print(f"  Dataset X: [{features[:, 0].min():.3f}, {features[:, 0].max():.3f}]")
    print(f"  Manual X:  [{features_manual[:, 0].min():.3f}, {features_manual[:, 0].max():.3f}]")
    print(f"  Dataset Y: [{features[:, 1].min():.3f}, {features[:, 1].max():.3f}]")
    print(f"  Manual Y:  [{features_manual[:, 1].min():.3f}, {features_manual[:, 1].max():.3f}]")

    # Check if they match
    if features.shape[0] == features_manual.shape[0]:
        x_diff = np.abs(features[:, 0] - features_manual[:, 0]).mean()
        y_diff = np.abs(features[:, 1] - features_manual[:, 1]).mean()
        print(f"  Mean X difference: {x_diff:.6f}")
        print(f"  Mean Y difference: {y_diff:.6f}")
    else:
        print(f"  Different lengths! Dataset: {features.shape[0]}, Manual: {features_manual.shape[0]}")


if __name__ == '__main__':
    trace_training_data()