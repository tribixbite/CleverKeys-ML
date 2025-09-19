#!/usr/bin/env python3
"""Create test features for beam_decode_onnx_cli.py"""

import json
import numpy as np
from test_with_resampling import ResamplingDecoder

# Load decoder for preprocessing
decoder = ResamplingDecoder(
    encoder_path='encoder_fresh.onnx',
    decoder_path='rnnt_step_fresh.onnx',
    runtime_meta_path='../trained_models/nema1/runtime_meta.json',
    words_path='../trained_models/nema1/words.txt'
)

# Load first validation sample
val_file = '../trained_models/nema1/personalized_tuning/20250918_105357/val_short_common.jsonl'
with open(val_file, 'r') as f:
    data = json.loads(f.readline())

word = data['word']
points = data['points']

print(f"Processing word: '{word}'")

# Preprocess
processed = decoder.preprocess_points(points)
print(f"Processed to {len(processed)} points")

# Compute features
features = decoder.compute_features_batch(processed)
print(f"Features shape: {features.shape}")

# Save in format expected by CLI: (1, F, T) where F=37
features_bft = features.T[np.newaxis, :, :]
print(f"Saving features with shape: {features_bft.shape}")

np.save('test_features.npy', features_bft)
print(f"Saved to test_features.npy")
print(f"Expected word: '{word}'")