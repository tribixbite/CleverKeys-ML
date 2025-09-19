#!/usr/bin/env python3
"""Investigate if blank_id configuration is wrong"""

import json
import numpy as np
import onnxruntime as ort

# Load metadata
with open("../trained_models/nema1/runtime_meta.json", "r") as f:
    meta = json.load(f)

print("Metadata says:")
print(f"  blank_id: {meta['blank_id']}")
print(f"  tokens[0]: {meta['tokens'][0]}")
print(f"  vocab_size: {meta['vocab_size']}")
print()

# But the model outputs 30 tokens, and token 29 behaves strangely
# Let's test if maybe blank_id is actually 29

encoder_sess = ort.InferenceSession("encoder_fresh.onnx", providers=["CPUExecutionProvider"])
step_sess = ort.InferenceSession("rnnt_step_fresh.onnx", providers=["CPUExecutionProvider"])

# Load test features
features = np.load("test_features.npy")
enc_out = encoder_sess.run(None, {
    "features_bft": features.astype(np.float32),
    "lengths": np.array([features.shape[2]], np.int32)
})
enc_bdt = enc_out[0]
enc_btf = enc_bdt[0].T

print("Testing different blank configurations:")
print("=" * 60)

# Test different blank IDs
for blank_test in [0, 29]:
    print(f"\nTesting with blank_id = {blank_test}")

    L, H = 2, 320
    h = np.zeros((L, 1, H), np.float32)
    c = np.zeros((L, 1, H), np.float32)
    y_prev = blank_test

    # Run a few steps
    for t in range(3):
        enc_t = enc_btf[t:t+1]

        outputs = step_sess.run(None, {
            "y_prev": np.array([y_prev], np.int64),
            "h0": h,
            "c0": c,
            "enc_t": enc_t
        })

        logits = outputs[0].squeeze()
        if len(logits.shape) == 0:
            logits = logits.reshape(-1)

        # Get top predictions
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()

        top5_idx = np.argsort(logits)[-5:][::-1]
        print(f"  t={t}: top5 tokens: {top5_idx.tolist()}")
        print(f"        probs: [{', '.join(f'{probs[i]:.3f}' for i in top5_idx)}]")

        # Follow most likely
        y_prev = np.argmax(logits)
        h = outputs[1]
        c = outputs[2]

# Also test if the model expects a different input format
print("\n" + "=" * 60)
print("\nTesting if y_prev should start from different values:")

for start_token in [0, 28, 29]:
    L, H = 2, 320
    h = np.zeros((L, 1, H), np.float32)
    c = np.zeros((L, 1, H), np.float32)

    # Just test first frame
    enc_t = enc_btf[0:1]

    outputs = step_sess.run(None, {
        "y_prev": np.array([start_token], np.int64),
        "h0": h,
        "c0": c,
        "enc_t": enc_t
    })

    logits = outputs[0].squeeze()
    top_pred = np.argmax(logits)
    top_prob = np.exp(logits[top_pred] - np.max(logits))

    print(f"  start with y_prev={start_token:2d} -> predicts {top_pred:2d} (p={top_prob:.3f})")