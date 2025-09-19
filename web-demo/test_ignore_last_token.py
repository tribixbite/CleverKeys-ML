#!/usr/bin/env python3
"""Test ignoring the last token (29) that ONNX export may have added"""

import json
import numpy as np
import onnxruntime as ort

# The hypothesis: ONNX export adds an extra dimension/token
# Solution: Simply ignore token 29 completely in predictions

# Load models and metadata
encoder_sess = ort.InferenceSession("encoder_fresh.onnx", providers=["CPUExecutionProvider"])
step_sess = ort.InferenceSession("rnnt_step_fresh.onnx", providers=["CPUExecutionProvider"])

with open("../trained_models/nema1/runtime_meta.json", "r") as f:
    meta = json.load(f)
    blank_id = meta["blank_id"]
    char_to_id = meta["char_to_id"]
    id_to_char = {int(k): v for k, v in meta["id_to_char"].items()}

# Load test features
features = np.load("test_features.npy")
T = features.shape[2]

# Run encoder
enc_out = encoder_sess.run(None, {
    "features_bft": features.astype(np.float32),
    "lengths": np.array([T], np.int32)
})
enc_bdt = enc_out[0]
enc_btf = enc_bdt[0].T
T_out = enc_btf.shape[0]

print(f"Testing with masking token 29...")
print(f"Encoder: {T} frames -> {T_out} frames")

# Greedy decode with token 29 masked
L, H = 2, 320
h = np.zeros((L, 1, H), np.float32)
c = np.zeros((L, 1, H), np.float32)
y_prev = blank_id
decoded = []

print("\nGreedy decoding (masking token 29):")
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

    # CRITICAL: Mask token 29 by setting to very negative
    if len(logits) > 29:
        logits[29] = -1e10

    # Get prediction
    y_pred = np.argmax(logits)

    if y_pred != blank_id:
        if y_pred < 29:  # Only decode valid tokens
            char = id_to_char.get(y_pred, f"?{y_pred}")
            decoded.append(char)
            print(f"  t={t:2d}: '{char}'")
            y_prev = y_pred
        else:
            print(f"  t={t:2d}: [INVALID:{y_pred}]")
    else:
        # For blank, don't print every frame
        if t == 0 or y_prev != blank_id:
            print(f"  t={t:2d}: <blank>")
        y_prev = blank_id

print(f"\nDecoded: '{''.join(decoded)}'")
print(f"Expected: 'is'")

# Also test with forced alignment to "is"
print("\n--- Testing forced alignment to 'is' ---")

i_id = char_to_id['i']
s_id = char_to_id['s']

print(f"Character IDs: i={i_id}, s={s_id}")

# Reset states
h = np.zeros((L, 1, H), np.float32)
c = np.zeros((L, 1, H), np.float32)

# Try to decode "is" by forcing those characters
forced_seq = [i_id, s_id]
forced_idx = 0
y_prev = blank_id

print("\nForced alignment scores:")
for t in range(min(10, T_out)):
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

    # Mask token 29
    if len(logits) > 29:
        logits[29] = -1e10

    # Check scores for blank and target characters
    blank_score = float(logits[blank_id])
    i_score = float(logits[i_id])
    s_score = float(logits[s_id])

    print(f"  t={t:2d}: blank={blank_score:6.2f}, i={i_score:6.2f}, s={s_score:6.2f}")

    # Greedily pick best
    y_pred = np.argmax(logits)
    if y_pred == i_id:
        print(f"       -> picked 'i'")
        y_prev = i_id
    elif y_pred == s_id:
        print(f"       -> picked 's'")
        y_prev = s_id
    elif y_pred == blank_id:
        print(f"       -> picked blank")
        y_prev = blank_id
    else:
        char = id_to_char.get(y_pred, f"?{y_pred}")
        print(f"       -> picked '{char}'")
        y_prev = y_pred