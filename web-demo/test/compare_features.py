#!/usr/bin/env python3
"""
Compare Python and JS feature extraction for debugging
"""

import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'new'))

from train_transducer_personalized import (
    PersonalizedSwipeFeaturizer,
    resample_points,
    clamp
)

def get_hello_data():
    """Get hello swipe data from line 431621"""
    import linecache
    data_path = '../../data/train_final_train.jsonl'
    line = linecache.getline(data_path, 431621)
    if line:
        data = json.loads(line)
        return data['points'], data['word']
    return None, None

# Get hello data
points, word = get_hello_data()
print(f"Testing: '{word}' with {len(points)} points")

# Process exactly as training does
prepared = []
for idx, pt in enumerate(points):
    raw_x = float(pt.get("x", 0.0))
    raw_y = float(pt.get("y", 0.0))
    centered_x = raw_x * 2.0 - 1.0
    centered_y = raw_y * 2.0 - 1.0
    centered_x = clamp(centered_x, -1.5, 1.5)
    centered_y = clamp(centered_y, -1.5, 1.5)
    raw_t = float(pt.get("t", idx * 10.0))
    prepared.append({"x": centered_x, "y": centered_y, "t": raw_t})

# Resample to 82
resampled = resample_points(prepared, 82)

# Extract features
featurizer = PersonalizedSwipeFeaturizer(mobile_features=False)
features = featurizer(resampled)

# Pad to 37
if features.shape[1] < 37:
    padding = np.zeros((features.shape[0], 37 - features.shape[1]), dtype=np.float32)
    features = np.concatenate([features, padding], axis=1)

print(f"\nPython features shape: {features.shape}")
print(f"First frame (first 10 features):")
for i in range(10):
    print(f"  Feature {i}: {features[0, i]:.6f}")

print(f"\nLast frame (first 10 features):")
for i in range(10):
    print(f"  Feature {i}: {features[-1, i]:.6f}")

# Save features for comparison
np.save('python_features.npy', features)
print("\nSaved features to python_features.npy")