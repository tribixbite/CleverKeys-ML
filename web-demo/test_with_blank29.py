#!/usr/bin/env python3
"""Test treating token 29 as a second blank"""

import json
import numpy as np
import onnxruntime as ort

# Load models
encoder_sess = ort.InferenceSession("encoder_fresh.onnx", providers=["CPUExecutionProvider"])
step_sess = ort.InferenceSession("rnnt_step_fresh.onnx", providers=["CPUExecutionProvider"])

# Load metadata
with open("../trained_models/nema1/runtime_meta.json", "r") as f:
    meta = json.load(f)
    blank_id = meta["blank_id"]
    char_to_id = meta["char_to_id"]
    id_to_char = {int(k): v for k, v in meta["id_to_char"].items()}

print(f"Blank ID from metadata: {blank_id}")
SECOND_BLANK = 29  # Hypothesis: token 29 is a second blank

# Load test features
features = np.load("test_features.npy")
B, F, T = features.shape
print(f"Features shape: {features.shape}")

# Run encoder
enc_out = encoder_sess.run(None, {
    "features_bft": features.astype(np.float32),
    "lengths": np.array([T], np.int32)
})
enc_bdt = enc_out[0]  # (1, 256, T_out)
enc_btf = enc_bdt[0].T  # (T_out, D)
T_out, D = enc_btf.shape
print(f"Encoder output: {T_out} frames")

# Greedy decode treating 29 as blank
L, H = 2, 320
h = np.zeros((L, 1, H), np.float32)
c = np.zeros((L, 1, H), np.float32)
y_prev = blank_id
decoded = []

print("\nDecoding (treating 0 and 29 as blanks):")
for t in range(T_out):
    enc_t = enc_btf[t:t+1]  # (1, D)

    # Run step
    outputs = step_sess.run(None, {
        "y_prev": np.array([y_prev], np.int64),
        "h0": h,
        "c0": c,
        "enc_t": enc_t
    })

    logits = outputs[0]
    h = outputs[1]
    c = outputs[2]

    # Handle logits shape
    if len(logits.shape) > 2:
        logits = logits.squeeze()
    if len(logits.shape) == 0:
        logits = logits.reshape(-1)

    # Get prediction
    y_pred = np.argmax(logits)

    # Treat both 0 and 29 as blanks
    if y_pred == blank_id or y_pred == SECOND_BLANK:
        print(f"  t={t:2d}: blank ({y_pred})")
        # Stay in blank state
        y_prev = blank_id
    else:
        char = id_to_char.get(y_pred, f"?{y_pred}")
        decoded.append(char)
        print(f"  t={t:2d}: '{char}' ({y_pred})")
        y_prev = y_pred

print(f"\nDecoded: '{''.join(decoded)}'")
print(f"Expected: 'is'")

# Also try a different approach: use token 29 as the blank
print("\n--- Alternative: Use token 29 as primary blank ---")

h = np.zeros((L, 1, H), np.float32)
c = np.zeros((L, 1, H), np.float32)
y_prev = SECOND_BLANK  # Start with token 29
decoded = []

print("\nDecoding (using 29 as primary blank):")
for t in range(min(10, T_out)):  # Just first 10 frames
    enc_t = enc_btf[t:t+1]

    outputs = step_sess.run(None, {
        "y_prev": np.array([y_prev], np.int64),
        "h0": h,
        "c0": c,
        "enc_t": enc_t
    })

    logits = outputs[0]
    h = outputs[1]
    c = outputs[2]

    if len(logits.shape) > 2:
        logits = logits.squeeze()
    if len(logits.shape) == 0:
        logits = logits.reshape(-1)

    y_pred = np.argmax(logits)

    if y_pred == SECOND_BLANK or y_pred == blank_id:
        print(f"  t={t:2d}: blank ({y_pred})")
        y_prev = SECOND_BLANK
    else:
        char = id_to_char.get(y_pred, f"?{y_pred}")
        decoded.append(char)
        print(f"  t={t:2d}: '{char}' ({y_pred})")
        y_prev = y_pred

print(f"\nDecoded: '{''.join(decoded)}'")