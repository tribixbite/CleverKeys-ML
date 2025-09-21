#!/usr/bin/env python3
"""Test with correct character mapping when blank=29"""

import json
import numpy as np
import onnxruntime as ort

# NeMo uses blank at 29
BLANK_ID = 29

# The vocab file has this order:
# Line 1: <blank> -> but this goes to index 29
# Line 2: '       -> index 0
# Line 3: a       -> index 1
# ...
# Line 29: <unk>  -> index 28

# So the mapping is:
# 0: '
# 1-26: a-z
# 27: unused?
# 28: <unk>
# 29: <blank>

print("Understanding the character mapping:")
print("=" * 60)

# Load the original vocab file
with open("../data/vocab.txt", "r") as f:
    vocab_lines = [line.strip() for line in f]

print("Vocab file contents:")
for i, token in enumerate(vocab_lines):
    print(f"  Line {i+1}: '{token}'")

print("\nNeMo's ID mapping (blank at 29):")
# Skip first line (<blank>) as it goes to 29
# The rest map to 0-27
id_to_char = {}
char_to_id = {}

for i, token in enumerate(vocab_lines[1:]):  # Skip <blank>
    if token == "<unk>":
        id_to_char[28] = token
        char_to_id[token] = 28
    else:
        id_to_char[i] = token
        char_to_id[token] = i

id_to_char[29] = "<blank>"
char_to_id["<blank>"] = 29

print("Character mappings:")
for i in range(30):
    char = id_to_char.get(i, "?")
    print(f"  {i:2d}: '{char}'")

# Test with this mapping
encoder_sess = ort.InferenceSession("encoder_correct.onnx", providers=["CPUExecutionProvider"])
step_sess = ort.InferenceSession("rnnt_step_fresh.onnx", providers=["CPUExecutionProvider"])

features = np.load("test_features.npy")
T = features.shape[2]

enc_out = encoder_sess.run(None, {
    "features": features.astype(np.float32),
    "features_length": np.array([T], np.int64)
})
enc_bdt = enc_out[0]
enc_btf = enc_bdt[0].T if len(enc_bdt.shape) == 3 else enc_bdt
T_out = enc_btf.shape[0]

print(f"\nEncoder: {T} -> {T_out} frames")

# Greedy decode
L, H = 2, 320
h = np.zeros((L, 1, H), np.float32)
c = np.zeros((L, 1, H), np.float32)
y_prev = BLANK_ID
decoded = []

print("\nGreedy decoding:")
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
    char = id_to_char.get(y_pred, f"?{y_pred}")

    if y_pred != BLANK_ID:
        decoded.append(char)
        print(f"  t={t}: pred={y_pred:2d} -> '{char}'")
        y_prev = y_pred
    else:
        print(f"  t={t}: pred={y_pred:2d} -> <blank>")
        y_prev = BLANK_ID

    # Show what 'i' and 's' scores are
    i_id = char_to_id.get('i', -1)
    s_id = char_to_id.get('s', -1)
    if i_id >= 0 and s_id >= 0:
        i_score = float(logits[i_id])
        s_score = float(logits[s_id])
        print(f"       scores: i({i_id})={i_score:.2f}, s({s_id})={s_score:.2f}")

print(f"\nDecoded: '{''.join(decoded)}'")
print(f"Expected: 'is'")