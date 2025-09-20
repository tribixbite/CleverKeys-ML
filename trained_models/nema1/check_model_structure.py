#!/usr/bin/env python3
"""Check the internal structure of the RNNT model."""

import torch
import nemo.collections.asr as nemo_asr

CHECKPOINT_PATH = "/home/will/git/swype/cleverkeys/rnnt_checkpoints_rare_words_20250919_140007/lightning_logs/version_0/checkpoints/epoch=epoch=80-wer=val_wer=0.152.ckpt"

# Load model
model = nemo_asr.models.EncDecRNNTModel.load_from_checkpoint(CHECKPOINT_PATH)

print("Decoder structure:")
print(f"  Type: {type(model.decoder)}")
print(f"  Attributes: {dir(model.decoder)}")

print("\nDecoder.prediction structure:")
print(f"  Type: {type(model.decoder.prediction)}")
if hasattr(model.decoder.prediction, 'keys'):
    print(f"  Keys: {model.decoder.prediction.keys()}")

print("\nTrying to access actual prediction network...")
if 'rnnt_pred' in model.decoder.prediction:
    pred = model.decoder.prediction['rnnt_pred']
    print(f"  Found rnnt_pred: {type(pred)}")
    print(f"  Has forward: {hasattr(pred, 'forward')}")

# Test actual prediction
dummy_targets = torch.tensor([[1]], dtype=torch.long)
try:
    # Try direct call
    out = model.decoder.prediction['rnnt_pred'](dummy_targets)
    print(f"\nDirect call works! Output shape: {out.shape if hasattr(out, 'shape') else type(out)}")
except Exception as e:
    print(f"\nDirect call failed: {e}")

# Check joint network
print("\nJoint network:")
print(f"  Type: {type(model.joint)}")
print(f"  Joint net: {type(model.joint.joint_net)}")