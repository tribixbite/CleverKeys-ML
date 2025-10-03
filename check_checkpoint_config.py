#!/usr/bin/env python3
"""
Check the config structure in the checkpoint
"""

import torch
import json

CHECKPOINT_PATH = "./9292025script/20251002/rnnt_checkpoints_medium_balanced_20251003_013434/lightning_logs/version_0/checkpoints/epoch=epoch=71-wer=val_wer=0.457.ckpt"

print("Loading checkpoint...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

print("\nHyper parameters keys:")
for key in checkpoint['hyper_parameters'].keys():
    print(f"  {key}")

print("\nConfig keys:")
cfg = checkpoint['hyper_parameters']['cfg']
for key in cfg.keys():
    print(f"  {key}")

# Check for preprocessing config location
if 'train_ds' in cfg:
    print("\nTrain dataset config keys:")
    for key in cfg['train_ds'].keys():
        print(f"  {key}")

    if 'preprocessing' in cfg['train_ds']:
        print("\nPreprocessing config:")
        print(json.dumps(cfg['train_ds']['preprocessing'], indent=2))

# Also check model config
if 'model' in cfg:
    print("\nModel config keys:")
    for key in cfg['model'].keys():
        print(f"  {key}")