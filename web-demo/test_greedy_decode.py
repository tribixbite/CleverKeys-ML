#!/usr/bin/env python3
"""Test greedy decoding to see what the model actually predicts"""

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

print(f"Blank ID: {blank_id}")
print(f"Vocab size from metadata: {len(meta['tokens'])}")

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
print(f"Encoder output: {T_out} frames, {D} dims")

# Greedy decode
L, H = 2, 320
h = np.zeros((L, 1, H), np.float32)
c = np.zeros((L, 1, H), np.float32)
y_prev = blank_id
decoded = []
decoded_ids = []

print("\nGreedy decoding:")
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

    # Get probabilities
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()

    # Greedy select
    y_pred = np.argmax(logits)
    prob = probs[y_pred]

    print(f"  t={t:2d}: pred={y_pred:2d} (p={prob:.3f})", end="")

    if y_pred != blank_id:
        if y_pred < len(id_to_char):
            char = id_to_char.get(y_pred, f"?{y_pred}")
            decoded.append(char)
            print(f" -> '{char}'")
        else:
            print(f" -> [OUT_OF_VOCAB:{y_pred}]")
        decoded_ids.append(y_pred)
        y_prev = y_pred
    else:
        print(f" -> <blank>")

    # Show top 3 predictions
    top3_idx = np.argsort(logits)[-3:][::-1]
    top3_chars = []
    for idx in top3_idx:
        if idx == blank_id:
            top3_chars.append("<blank>")
        elif idx < len(id_to_char):
            top3_chars.append(f"'{id_to_char.get(idx, f'?{idx}')}'")
        else:
            top3_chars.append(f"[{idx}]")
    print(f"       (top3: {', '.join(top3_chars)})")

print(f"\nDecoded: '{''.join(decoded)}'")
print(f"Decoded IDs: {decoded_ids}")
print(f"Expected: 'is'")