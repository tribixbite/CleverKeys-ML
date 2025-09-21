#!/usr/bin/env python3
"""Test decoder with properly extracted features from validation data"""

import sys
sys.path.append('../scripts')

import json
import numpy as np
import onnxruntime as ort
from swipe_data_utils import SwipeFeaturizer, KeyboardGrid

# Load validation data
with open("../data/train_final_val.jsonl", "r") as f:
    lines = f.readlines()

# Create featurizer
grid = KeyboardGrid()
featurizer = SwipeFeaturizer(grid)

# Process first word (should be "is")
data = json.loads(lines[0])
word = data["word"]
points = data["points"]

print(f"Testing with word: '{word}'")
print(f"Number of points: {len(points)}")

# Extract features
features = featurizer(points)  # Shape: (num_points, 37)
print(f"Raw features shape: {features.shape}")

# Resample to match training (56 or 96 points)
from scipy.interpolate import interp1d

def resample_features(features, target_length):
    """Resample features to target length."""
    if len(features) == target_length:
        return features

    # Create interpolation for each feature dimension
    x_old = np.linspace(0, 1, len(features))
    x_new = np.linspace(0, 1, target_length)

    resampled = []
    for dim in range(features.shape[1]):
        f = interp1d(x_old, features[:, dim], kind='linear', fill_value='extrapolate')
        resampled.append(f(x_new))

    return np.array(resampled).T

# Decide on target length based on original length
target_len = 56 if len(features) < 70 else 96
features_resampled = resample_features(features, target_len)
print(f"Resampled features shape: {features_resampled.shape}")

# Prepare for model input: (batch, features, time)
features_input = features_resampled.T[np.newaxis, :, :]  # (1, 37, T)
print(f"Model input shape: {features_input.shape}")

# Load models
print("\nLoading models...")
encoder_sess = ort.InferenceSession(
    "../trained_models/nema1/onnx_rare_words_epoch80/encoder.onnx",
    providers=["CPUExecutionProvider"]
)
decoder_sess = ort.InferenceSession(
    "../trained_models/nema1/rnnt_step_final.onnx",
    providers=["CPUExecutionProvider"]
)

# Load metadata
with open("../trained_models/nema1/runtime_meta_final.json", "r") as f:
    meta = json.load(f)
    blank_id = meta["blank_id"]

# Character mapping (corrected)
id_to_char = {
    29: "<blank>", 28: "<unk>", 0: "'", 27: "<?27>"
}
for i in range(26):
    id_to_char[i + 1] = chr(ord('a') + i)

print(f"\nBlank ID: {blank_id}")

# Run encoder
T = features_input.shape[2]
enc_out = encoder_sess.run(None, {
    "features": features_input.astype(np.float32),
    "features_length": np.array([T], np.int64)
})
enc_bdt = enc_out[0]
enc_btf = enc_bdt.T if len(enc_bdt.shape) == 2 else enc_bdt[0].T
T_out = enc_btf.shape[0]

print(f"\nEncoder: {T} frames -> {T_out} frames")

# Greedy decode
L, H = 2, 320
h = np.zeros((L, 1, H), np.float32)
c = np.zeros((L, 1, H), np.float32)
y_prev = blank_id
decoded = []

print("\nGreedy decoding:")
for t in range(T_out):
    enc_t = enc_btf[t:t+1]

    outputs = decoder_sess.run(None, {
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

    if y_pred != blank_id:
        char = id_to_char.get(y_pred, f"?{y_pred}")
        if char not in ["<blank>", "<unk>", "<?27>"]:
            decoded.append(char)
            if len(decoded) <= 10:
                print(f"  t={t:2d}: pred={y_pred:2d} -> '{char}'")
        y_prev = y_pred
    else:
        y_prev = blank_id

print(f"\nDecoded: '{''.join(decoded)}'")
print(f"Expected: '{word}'")

# Test a few more words
print("\n" + "=" * 60)
print("Testing more words from validation set:\n")

for i in range(1, min(5, len(lines))):
    data = json.loads(lines[i])
    word = data["word"]
    points = data["points"]

    # Extract and resample features
    features = featurizer(points)
    target_len = 56 if len(features) < 70 else 96
    features_resampled = resample_features(features, target_len)
    features_input = features_resampled.T[np.newaxis, :, :]

    # Run encoder
    T = features_input.shape[2]
    enc_out = encoder_sess.run(None, {
        "features": features_input.astype(np.float32),
        "features_length": np.array([T], np.int64)
    })
    enc_bdt = enc_out[0]
    enc_btf = enc_bdt.T if len(enc_bdt.shape) == 2 else enc_bdt[0].T

    # Quick greedy decode
    h = np.zeros((L, 1, H), np.float32)
    c = np.zeros((L, 1, H), np.float32)
    y_prev = blank_id
    decoded = []

    for t in range(enc_btf.shape[0]):
        enc_t = enc_btf[t:t+1]
        outputs = decoder_sess.run(None, {
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
        if y_pred != blank_id:
            char = id_to_char.get(y_pred, f"?{y_pred}")
            if char not in ["<blank>", "<unk>", "<?27>"]:
                decoded.append(char)
            y_prev = y_pred
        else:
            y_prev = blank_id

    result = ''.join(decoded)
    print(f"  Word {i+1}: '{word}' -> '{result}' {'✓' if result == word else '✗'}")