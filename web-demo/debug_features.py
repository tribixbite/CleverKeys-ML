#!/usr/bin/env python3
"""
Debug feature extraction to compare with training
"""

import json
import numpy as np
from test_with_resampling import ResamplingDecoder

# Load decoder
decoder = ResamplingDecoder(
    encoder_path='personalized/encoder_int8_qdq.onnx',
    decoder_path='personalized/rnnt_step_fp32.onnx',
    runtime_meta_path='../trained_models/nema1/runtime_meta.json',
    words_path='../trained_models/nema1/words.txt'
)

# Load first validation sample
val_file = '../trained_models/nema1/personalized_tuning/20250918_105357/val_short_common.jsonl'
with open(val_file, 'r') as f:
    data = json.loads(f.readline())

word = data['word']
points = data['points']

print(f"Word: '{word}'")
print(f"Original points: {len(points)}")
print(f"First point: {points[0]}")
print(f"Last point: {points[-1]}")

# Process points
processed = decoder.preprocess_points(points)
print(f"\nAfter preprocessing: {len(processed)} points")
print(f"First processed: {processed[0]}")
print(f"Last processed: {processed[-1]}")

# Compute features
features = decoder.compute_features_batch(processed)
print(f"\nFeatures shape: {features.shape}")
print(f"Feature stats:")
for i, name in enumerate(['x', 'y', 't', 'vx', 'vy', 'speed']):
    vals = features[:, i]
    print(f"  {name:6s}: min={vals.min():7.3f}, max={vals.max():7.3f}, mean={vals.mean():7.3f}")

# Check key distances
print(f"\nKey distance features (cols 13-17):")
for i in range(13, 18):
    vals = features[:, i]
    print(f"  feat[{i:2d}]: min={vals.min():7.3f}, max={vals.max():7.3f}, mean={vals.mean():7.3f}")

# Run encoder
features_bft = features.T[np.newaxis, :, :]
encoder_outputs = decoder.encoder_session.run(None, {
    'features_bft': features_bft.astype(np.float32),
    'lengths': np.array([features.shape[0]], dtype=np.int32)
})

enc_out = encoder_outputs[0]
print(f"\nEncoder output shape: {enc_out.shape}")
print(f"Encoder output stats:")
print(f"  min={enc_out.min():.3f}, max={enc_out.max():.3f}, mean={enc_out.mean():.3f}")