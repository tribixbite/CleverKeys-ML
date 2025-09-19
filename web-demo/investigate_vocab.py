#!/usr/bin/env python3
"""Investigate the vocabulary mismatch"""

import json
import numpy as np
import onnxruntime as ort

# Check what the model actually outputs
step_sess = ort.InferenceSession("rnnt_step_fresh.onnx", providers=["CPUExecutionProvider"])

print("Step model outputs:")
for out in step_sess.get_outputs():
    print(f"  {out.name}: {out.shape}")

# Check the actual output dimensions
L, H, D = 2, 320, 256
test_inputs = {
    "y_prev": np.array([0], np.int64),
    "h0": np.zeros((L, 1, H), np.float32),
    "c0": np.zeros((L, 1, H), np.float32),
    "enc_t": np.zeros((1, D), np.float32)
}

outputs = step_sess.run(None, test_inputs)
logits = outputs[0]

print(f"\nActual logits shape: {logits.shape}")
print(f"Vocab size from model: {logits.shape[-1]}")

# Load metadata
with open("../trained_models/nema1/runtime_meta.json", "r") as f:
    meta = json.load(f)

print(f"\nMetadata tokens ({len(meta['tokens'])} total):")
for i, token in enumerate(meta["tokens"]):
    print(f"  {i:2d}: {repr(token)}")

print(f"\nVocab size in metadata: {meta['vocab_size']}")
print(f"Blank ID: {meta['blank_id']}")
print(f"UNK ID: {meta['unk_id']}")

# Check if there might be a space token
print(f"\nChecking for space in char_to_id:")
for char, id in meta["char_to_id"].items():
    if char == " " or ord(char) == 32:
        print(f"  Found space: '{char}' -> {id}")

# Check training code for vocab size
print("\n--- Checking what the training expects ---")
print("The model outputs 30 tokens but metadata only defines 29.")
print("Possibilities:")
print("1. Token 29 could be a padding token added during model creation")
print("2. Token 29 could be end-of-sequence (EOS) token")
print("3. There's a mismatch between training vocab and inference vocab")

# Test what happens with different token IDs
print("\nTesting model response to different y_prev values:")
for y_prev_val in [0, 10, 20, 28, 29]:
    test_inputs["y_prev"] = np.array([y_prev_val], np.int64)
    outputs = step_sess.run(None, test_inputs)
    logits = outputs[0].squeeze()

    # Get top prediction
    top_pred = np.argmax(logits)
    top_prob = np.exp(logits[top_pred] - np.max(logits))

    print(f"  y_prev={y_prev_val:2d} -> top_pred={top_pred:2d} (p={top_prob:.3f})")

    # Check if it strongly predicts token 29
    if logits.shape[0] > 29:
        prob_29 = np.exp(logits[29] - np.max(logits))
        print(f"            -> prob[29]={prob_29:.3f}")