#!/usr/bin/env python3
"""Test the model directly in PyTorch"""

import sys
sys.path.append('../trained_models/nema1')
sys.path.append('../scripts')

import torch
import json
import numpy as np
from export_common import load_trained_model
from swipe_data_utils import SwipeFeaturizer, KeyboardGrid
from scipy.interpolate import interp1d

def resample_features(features, target_length):
    """Resample features to target length."""
    if len(features) == target_length:
        return features

    x_old = np.linspace(0, 1, len(features))
    x_new = np.linspace(0, 1, target_length)

    resampled = []
    for dim in range(features.shape[1]):
        f = interp1d(x_old, features[:, dim], kind='linear', fill_value='extrapolate')
        resampled.append(f(x_new))

    return np.array(resampled).T

# Load model
print("Loading model...")
model = load_trained_model("../trained_models/nema1/last.ckpt")
model.eval()

# Get vocabulary and blank index
vocab = list(model.joint.vocabulary)
blank_idx = model.decoder.blank_idx
print(f"Vocabulary size: {len(vocab)}")
print(f"Blank index from decoder: {blank_idx}")
print(f"First 10 vocab items: {vocab[:10]}")
print(f"Last 5 vocab items: {vocab[-5:]}")

# Create proper ID mapping based on how NeMo structures it
# NeMo adds blank at the end (position 29)
id_to_char = {}
char_to_id = {}

# Regular tokens (non-blank) are at positions 0-27
for i, token in enumerate(vocab):
    if token == "<blank>":
        continue  # Skip, will add at position 29
    elif token == "'":
        id_to_char[0] = token
        char_to_id[token] = 0
    elif token == "<unk>":
        id_to_char[28] = token
        char_to_id[token] = 28
    else:
        # a-z at positions 1-26
        pos = ord(token) - ord('a') + 1
        id_to_char[pos] = token
        char_to_id[token] = pos

# Blank at position 29
id_to_char[29] = "<blank>"
char_to_id["<blank>"] = 29

print("\nID to character mapping:")
for i in range(30):
    print(f"  {i:2d}: '{id_to_char.get(i, '?')}'")

# Load validation data
with open("../data/train_final_val.jsonl", "r") as f:
    lines = f.readlines()

# Test with first few words
print("\n" + "=" * 60)
print("Testing with validation data:\n")

grid = KeyboardGrid()
featurizer = SwipeFeaturizer(grid)

for idx in range(min(5, len(lines))):
    data = json.loads(lines[idx])
    word = data["word"]
    points = data["points"]

    # Extract features
    features = featurizer(points)
    target_len = 56 if len(features) < 70 else 96
    features_resampled = resample_features(features, target_len)

    # Convert to torch tensor (batch, features, time)
    audio_signal = torch.from_numpy(features_resampled.T).float().unsqueeze(0)
    audio_len = torch.tensor([audio_signal.shape[2]])

    print(f"Word {idx+1}: '{word}'")
    print(f"  Input shape: {audio_signal.shape}")

    # Run model inference
    with torch.no_grad():
        # Encode
        enc_out, enc_len = model.encoder(audio_signal=audio_signal, length=audio_len)
        print(f"  Encoder output: {enc_out.shape}")

        # Greedy decode using model's decoder
        T_out = enc_out.shape[2]
        decoded_ids = []

        # Initialize decoder state
        states = model.decoder.initialize_state(enc_out)
        y_prev = torch.tensor([blank_idx])

        for t in range(T_out):
            # Get encoder frame
            enc_frame = enc_out[:, :, t:t+1]

            # Run decoder prediction
            pred_out, states = model.decoder.predict(
                y_prev.unsqueeze(0),  # Add batch dim
                state=states,
                add_sos=False,
                batch_size=1
            )

            # Run joint
            joint_out = model.joint(
                encoder_outputs=enc_frame,
                decoder_outputs=pred_out.unsqueeze(2)
            )

            # Get prediction
            logits = joint_out[0, :, 0]  # Remove batch and time dims
            y_pred = torch.argmax(logits).item()

            if y_pred != blank_idx:
                decoded_ids.append(y_pred)
                y_prev = torch.tensor([y_pred])
            else:
                y_prev = torch.tensor([blank_idx])

            # Only decode first few non-blanks
            if len(decoded_ids) >= 15:
                break

    # Convert IDs to characters
    decoded_chars = [id_to_char.get(i, f"?{i}") for i in decoded_ids]
    decoded_str = ''.join([c for c in decoded_chars if c not in ["<blank>", "<unk>", "?27"]])

    print(f"  Decoded IDs: {decoded_ids[:10]}...")
    print(f"  Decoded: '{decoded_str}'")
    print(f"  Expected: '{word}'")
    print(f"  {'✓' if decoded_str == word else '✗'}")
    print()