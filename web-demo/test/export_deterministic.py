#!/usr/bin/env python3
"""
Re-export models with deterministic settings for JS compatibility
"""

import torch
import json
import os
from pathlib import Path

# Set deterministic mode
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

checkpoint_path = '/home/will/git/swype/cleverkeys/9292025script/20251002/rnnt_checkpoints_medium_balanced_20251003_013434/lightning_logs/version_0/checkpoints/epoch=epoch=71-wer=val_wer=0.457.ckpt'
output_dir = '../models/deterministic_export'
os.makedirs(output_dir, exist_ok=True)

print(f"Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Extract state dict and model config
state_dict = checkpoint['state_dict']
print(f"State dict keys: {len(state_dict)} parameters")

# Get model config from checkpoint
model_cfg = checkpoint['hyper_parameters']['cfg']['model']
print(f"Model config loaded")

# Import NeMo
from nemo.collections.asr.models import EncDecRNNTModel

# Create model with exact config
model = EncDecRNNTModel.restore_from(checkpoint_path, map_location='cpu')
model.eval()

print("Model loaded successfully")

# Export to ONNX with explicit settings
print("\nExporting encoder...")
encoder_path = os.path.join(output_dir, 'encoder.onnx')
model.encoder.export(
    encoder_path,
    input_example=torch.randn(1, 37, 100),  # [B, feat_dim, T]
    verbose=False,
    do_constant_folding=True,
    export_params=True,
    opset_version=14,  # Use older opset for better compatibility
    input_names=['audio_signal', 'length'],
    output_names=['outputs', 'encoded_lengths'],
    dynamic_axes={
        'audio_signal': {2: 'time'},
        'outputs': {2: 'time_out'}
    }
)

print(f"Encoder exported to: {encoder_path}")

# Export decoder/joint network with stateful LSTM
print("\nExporting decoder/joint...")
decoder_joint_path = os.path.join(output_dir, 'decoder_joint.onnx')

# Get decoder config
decoder_cfg = model_cfg['decoder']
predictor_cfg = model_cfg['joint']['predictor']

# Create export wrapper for stateful decoder
import sys
sys.path.insert(0, '../../new')
from export_stateful_pair import StatefulDecoderExport

stateful_export = StatefulDecoderExport(
    model.decoder,
    model.joint,
    decoder_cfg,
    predictor_cfg
)

# Export with explicit settings
dummy_targets = torch.zeros(1, 1, dtype=torch.long)
dummy_target_len = torch.tensor([1], dtype=torch.long)
dummy_encoder = torch.randn(1, model_cfg['encoder']['d_model'], 1)
dummy_h = torch.zeros(decoder_cfg['num_layers'], 1, decoder_cfg['hidden_size'])
dummy_c = torch.zeros(decoder_cfg['num_layers'], 1, decoder_cfg['hidden_size'])

torch.onnx.export(
    stateful_export,
    (dummy_encoder, dummy_targets, dummy_target_len, dummy_h, dummy_c),
    decoder_joint_path,
    input_names=['encoder_outputs', 'targets', 'target_length', 'input_states_1', 'input_states_2'],
    output_names=['outputs', 'prednet_lengths', 'output_states_1', 'output_states_2'],
    dynamic_axes={
        'encoder_outputs': {2: 'enc_time'},
        'targets': {1: 'target_time'},
        'target_length': {0: 'batch'},
        'outputs': {2: 'joint_time'},
        'prednet_lengths': {0: 'batch'}
    },
    opset_version=14,  # Match encoder opset
    export_params=True,
    do_constant_folding=True,
    verbose=False
)

print(f"Decoder/Joint exported to: {decoder_joint_path}")

# Save runtime metadata
runtime_meta = {
    "vocab_size": len(model.tokenizer.vocab),
    "blank_id": model.tokenizer.vocab.size,
    "tokens": list(model.tokenizer.vocab),
    "char_to_id": {char: idx for idx, char in enumerate(model.tokenizer.vocab)},
    "id_to_char": {str(idx): char for idx, char in enumerate(model.tokenizer.vocab)},
    "encoder_config": {
        "feat_in": model_cfg['encoder']['feat_in'],
        "d_model": model_cfg['encoder']['d_model'],
        "encoder_dim": model_cfg['encoder']['d_model']
    },
    "decoder_config": {
        "num_layers": decoder_cfg['num_layers'],
        "hidden_size": decoder_cfg['hidden_size']
    },
    "predictor": {
        "label_map": {
            "joint2pred": list(range(len(model.tokenizer.vocab))) + [-1]
        }
    }
}

meta_path = os.path.join(output_dir, 'runtime_meta.json')
with open(meta_path, 'w') as f:
    json.dump(runtime_meta, f, indent=2)

print(f"\nRuntime metadata saved to: {meta_path}")
print("\nExport complete with deterministic settings!")