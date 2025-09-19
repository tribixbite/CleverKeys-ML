#!/usr/bin/env python3
"""Check if the model was configured/exported incorrectly"""

import torch
import numpy as np

# Load the checkpoint to inspect the actual model configuration
ckpt_path = "../rnnt_checkpoints_rare_words_20250919_140007/lightning_logs/version_0/checkpoints/epoch=epoch=80-wer=val_wer=0.152.ckpt"
checkpoint = torch.load(ckpt_path, map_location="cpu")

print("Checking model configuration in checkpoint...")
print()

# Check if there's info about vocab size in the model state
state_dict = checkpoint["state_dict"]

# Look for joint network output layer
for key in state_dict.keys():
    if "joint" in key and ("proj" in key or "output" in key or "linear" in key):
        if "weight" in key:
            shape = state_dict[key].shape
            print(f"{key}: shape={shape}")
            if len(shape) == 2:
                out_features = shape[0]
                print(f"  -> Output features: {out_features}")

print()

# Check decoder output
for key in state_dict.keys():
    if "decoder" in key and "output" in key and "weight" in key:
        shape = state_dict[key].shape
        print(f"{key}: shape={shape}")

print()
print("The joint network should output vocab_size logits.")
print("If it outputs 30, but vocab only has 29 tokens,")
print("then the model was likely configured with vocab_size=30")
print("during training, possibly due to automatic +1 for padding.")