#!/usr/bin/env python3
"""Test with blank_id=29 as confirmed by NeMo export"""

import json
import numpy as np
import onnxruntime as ort

# CRITICAL FINDING: NeMo puts blank at index 29, not 0!
BLANK_ID = 29

# Load models
encoder_sess = ort.InferenceSession("encoder_correct.onnx", providers=["CPUExecutionProvider"])
step_sess = ort.InferenceSession("rnnt_step_fresh.onnx", providers=["CPUExecutionProvider"])

# Load metadata
with open("../trained_models/nema1/runtime_meta.json", "r") as f:
    meta = json.load(f)
    char_to_id = meta["char_to_id"]
    # Build corrected id_to_char (shift everything up by 1)
    id_to_char = {}
    for char, old_id in char_to_id.items():
        # Characters get shifted: what was 2 becomes 1, etc.
        new_id = old_id - 1 if old_id > 0 else old_id
        if new_id >= 0:
            id_to_char[new_id] = char

print("Corrected mapping (blank at 29):")
print(f"  Blank: {BLANK_ID}")
print(f"  Characters: {list(id_to_char.items())[:10]}...")

# Load test features
features = np.load("test_features.npy")
T = features.shape[2]

# Run encoder
enc_out = encoder_sess.run(None, {
    "features": features.astype(np.float32),
    "features_length": np.array([T], np.int64)
})
enc_bdt = enc_out[0]
# Encoder output shape varies between exports
if len(enc_bdt.shape) == 3:
    enc_btf = enc_bdt[0].T if enc_bdt.shape[1] > enc_bdt.shape[2] else enc_bdt[0]
else:
    enc_btf = enc_bdt
T_out = enc_btf.shape[0]

print(f"Encoder: {T} -> {T_out} frames")

# Greedy decode with blank=29
L, H = 2, 320
h = np.zeros((L, 1, H), np.float32)
c = np.zeros((L, 1, H), np.float32)
y_prev = BLANK_ID
decoded = []

print("\nGreedy decoding (blank=29):")
for t in range(T_out):
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

    if y_pred != BLANK_ID:
        if y_pred in id_to_char:
            char = id_to_char[y_pred]
            decoded.append(char)
            if len(decoded) <= 10:  # Only print first 10
                print(f"  t={t:2d}: '{char}' (id={y_pred})")
        else:
            if len(decoded) <= 10:
                print(f"  t={t:2d}: [unmapped:{y_pred}]")
        y_prev = y_pred
    else:
        if t == 0 or y_prev != BLANK_ID:
            print(f"  t={t:2d}: <blank>")
        y_prev = BLANK_ID

print(f"\nDecoded: '{''.join(decoded)}'")
print(f"Expected: 'is'")