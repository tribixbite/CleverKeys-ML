#!/usr/bin/env python3
"""
Sanity check for the ONNX models - test if they produce reasonable outputs
"""

import json
import numpy as np
import onnxruntime as ort

# Load models
encoder = ort.InferenceSession('personalized/encoder_int8_qdq.onnx', providers=['CPUExecutionProvider'])
decoder = ort.InferenceSession('personalized/rnnt_step_fp32.onnx', providers=['CPUExecutionProvider'])

# Load metadata
with open('../trained_models/nema1/runtime_meta.json', 'r') as f:
    meta = json.load(f)
    blank_id = meta['blank_id']
    char_to_id = meta['char_to_id']
    id_to_char = {int(k): v for k, v in meta['id_to_char'].items()}

print(f"Blank ID: {blank_id}")
print(f"Vocab size: {len(char_to_id)}")
print(f"Sample chars: {list(char_to_id.keys())[:10]}")

# Create simple test input - just zeros
T = 50
features = np.zeros((1, 37, T), dtype=np.float32)
lengths = np.array([T], dtype=np.int32)

# Run encoder
encoder_out = encoder.run(None, {
    'features_bft': features,
    'lengths': lengths
})

encoded_btf = encoder_out[0]
print(f"\nEncoder output shape: {encoded_btf.shape}")
print(f"Encoder output range: [{encoded_btf.min():.3f}, {encoded_btf.max():.3f}]")

# Check if encoder produces reasonable values
if np.isnan(encoded_btf).any():
    print("WARNING: Encoder produced NaN values!")
if np.isinf(encoded_btf).any():
    print("WARNING: Encoder produced Inf values!")

# Run decoder with blank input
y_prev = np.array([blank_id], dtype=np.int64)
h0 = np.zeros((2, 1, 320), dtype=np.float32)
c0 = np.zeros((2, 1, 320), dtype=np.float32)
enc_t = encoded_btf[0, :, 0:1].T  # Take first time step (1, 256)

decoder_out = decoder.run(None, {
    'y_prev': y_prev,
    'h0': h0,
    'c0': c0,
    'enc_t': enc_t
})

logits = decoder_out[0]
print(f"\nDecoder logits shape: {logits.shape}")
print(f"Decoder logits range: [{logits.min():.3f}, {logits.max():.3f}]")

# Apply softmax and check distribution
logits_flat = logits.squeeze()
exp_logits = np.exp(logits_flat - np.max(logits_flat))
probs = exp_logits / np.sum(exp_logits)

print(f"\nProbability distribution:")
top_k = 5
top_indices = np.argsort(probs)[-top_k:][::-1]
for idx in top_indices:
    char = id_to_char.get(idx, f"[{idx}]")
    print(f"  {char}: {probs[idx]:.4f}")

# Test with a character input
test_char = 'h'
if test_char in char_to_id:
    y_prev = np.array([char_to_id[test_char]], dtype=np.int64)

    decoder_out = decoder.run(None, {
        'y_prev': y_prev,
        'h0': h0,
        'c0': c0,
        'enc_t': enc_t
    })

    logits = decoder_out[0].squeeze()
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    print(f"\nAfter inputting '{test_char}':")
    top_indices = np.argsort(probs)[-top_k:][::-1]
    for idx in top_indices:
        char = id_to_char.get(idx, f"[{idx}]")
        print(f"  {char}: {probs[idx]:.4f}")