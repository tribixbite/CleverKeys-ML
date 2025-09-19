#!/usr/bin/env python3
"""Check actual encoder dimensions."""

import torch
import nemo.collections.asr as nemo_asr

# Checkpoint path
CHECKPOINT_PATH = "/home/will/git/swype/cleverkeys/rnnt_checkpoints_rare_words_20250919_140007/lightning_logs/version_0/checkpoints/epoch=epoch=80-wer=val_wer=0.152.ckpt"

# Load model
model = nemo_asr.models.EncDecRNNTModel.load_from_checkpoint(CHECKPOINT_PATH)
model.eval()
model = model.cpu()

# Test encoder
dummy_features = torch.randn(1, 37, 96)
dummy_length = torch.tensor([96], dtype=torch.long)

with torch.no_grad():
    encoded, encoded_len = model.encoder(audio_signal=dummy_features, length=dummy_length)

print(f"Encoder input shape: {dummy_features.shape}")
print(f"Encoder output shape: {encoded.shape}")
print(f"Encoder output dimension: {encoded.shape[-1]}")
print(f"model.encoder._feat_out: {model.encoder._feat_out}")

# Check joint network
print(f"\nJoint network layers:")
for name, module in model.joint.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(f"  {name}: Linear({module.in_features} → {module.out_features})")

# Check decoder
print(f"\nDecoder output dim: {model.decoder.prediction.hidden_size}")