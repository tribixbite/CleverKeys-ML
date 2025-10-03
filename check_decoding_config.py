#!/usr/bin/env python3
"""
Check the decoding configuration in the checkpoint
"""

import torch
import json

CHECKPOINT_PATH = "./9292025script/20251002/rnnt_checkpoints_medium_balanced_20251003_013434/lightning_logs/version_0/checkpoints/epoch=epoch=71-wer=val_wer=0.457.ckpt"

print("Loading checkpoint...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

cfg = checkpoint['hyper_parameters']['cfg']

print("\nDecoding config from checkpoint:")
print("="*50)
if 'decoding' in cfg:
    decoding_cfg = cfg['decoding']
    for key, value in decoding_cfg.items():
        print(f"  {key}: {value}")

print("\nModel config keys:")
for key in cfg.keys():
    print(f"  {key}")

# Check if there are any beam search or greedy decode parameters
print("\nSearching for decoder-related configs:")
for key, value in cfg.items():
    if 'decode' in key.lower() or 'beam' in key.lower() or 'greedy' in key.lower():
        print(f"  {key}: {value}")

# Check the loss config too
if 'loss' in cfg:
    print("\nLoss config:")
    for key, value in cfg['loss'].items():
        print(f"  {key}: {value}")