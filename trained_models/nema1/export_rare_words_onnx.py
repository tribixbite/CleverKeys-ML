#!/usr/bin/env python3
"""
Export the rare_words checkpoint to ONNX with INT8 quantization.
This script exports both encoder and decoder as separate ONNX models.
"""

import torch
import numpy as np
from pathlib import Path
import json
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import nemo.collections.asr as nemo_asr

# Checkpoint path
CHECKPOINT_PATH = "/home/will/git/swype/cleverkeys/rnnt_checkpoints_rare_words_20250919_140007/lightning_logs/version_0/checkpoints/epoch=epoch=80-wer=val_wer=0.152.ckpt"
OUTPUT_DIR = Path("onnx_rare_words_epoch80")
OUTPUT_DIR.mkdir(exist_ok=True)

def export_model():
    """Export NeMo model to ONNX format."""

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    # Load the model from checkpoint
    model = nemo_asr.models.EncDecRNNTModel.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()
    model = model.cpu()

    # Get vocabulary info - CRITICAL: NeMo adds blank at index 29!
    vocab = model.joint.vocabulary
    blank_idx = model.decoder.blank_idx
    vocab_size = len(vocab) + 1  # NeMo adds an extra dimension

    print(f"Model loaded. Vocab size: {vocab_size}, Blank index: {blank_idx}")

    # Export encoder
    print("\n=== Exporting Encoder ===")
    encoder_path = OUTPUT_DIR / "encoder.onnx"

    # Create dummy input for encoder (batch=1, features=37, time=96)
    # NeMo expects (batch, features, time) not (batch, time, features)
    dummy_encoder_input = torch.randn(1, 37, 96)
    dummy_encoder_length = torch.tensor([96], dtype=torch.long)

    # Create wrapper for encoder to handle NeMo's kwargs requirement
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, audio_signal, length):
            # NeMo requires kwargs
            encoded, encoded_len = self.encoder(audio_signal=audio_signal, length=length)
            return encoded, encoded_len

    encoder_wrapper = EncoderWrapper(model.encoder)
    encoder_wrapper.eval()

    # Export encoder
    torch.onnx.export(
        encoder_wrapper,
        (dummy_encoder_input, dummy_encoder_length),
        encoder_path,
        input_names=['features', 'features_length'],
        output_names=['encoder_output', 'encoder_output_length'],
        dynamic_axes={
            'features': {0: 'batch', 2: 'time'},
            'features_length': {0: 'batch'},
            'encoder_output': {0: 'batch', 1: 'time'},
            'encoder_output_length': {0: 'batch'}
        },
        opset_version=13,
        do_constant_folding=True
    )
    print(f"Encoder exported to {encoder_path}")

    # Export decoder (prediction network)
    print("\n=== Exporting Decoder ===")
    decoder_path = OUTPUT_DIR / "decoder.onnx"

    # Decoder expects (batch, targets) and states
    dummy_targets = torch.tensor([[0]], dtype=torch.long)  # Start with blank
    batch_size = 1

    # Initialize hidden states
    states = model.decoder.prediction.init_states(
        batch_size=batch_size,
        dtype=torch.float32,
        device='cpu'
    )

    # For ONNX export, we need to handle states differently
    # Create a wrapper that takes states as inputs
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, targets, state_0, state_1):
            states = [state_0, state_1] if self.decoder.prediction._pred_rnn_type == 'lstm' else [state_0]
            output, out_states = self.decoder.prediction(targets, states)
            # Flatten states for ONNX
            if len(out_states) == 2:
                return output, out_states[0], out_states[1]
            else:
                return output, out_states[0], torch.zeros_like(out_states[0])

    wrapper = DecoderWrapper(model.decoder)
    wrapper.eval()

    # Create dummy states
    if model.decoder.prediction._pred_rnn_type == 'lstm':
        dummy_state_0 = torch.zeros(2, batch_size, model.decoder.prediction.hidden_size)
        dummy_state_1 = torch.zeros(2, batch_size, model.decoder.prediction.hidden_size)
    else:
        dummy_state_0 = torch.zeros(2, batch_size, model.decoder.prediction.hidden_size)
        dummy_state_1 = torch.zeros(2, batch_size, model.decoder.prediction.hidden_size)

    torch.onnx.export(
        wrapper,
        (dummy_targets, dummy_state_0, dummy_state_1),
        decoder_path,
        input_names=['targets', 'state_0', 'state_1'],
        output_names=['decoder_output', 'new_state_0', 'new_state_1'],
        dynamic_axes={
            'targets': {0: 'batch', 1: 'length'},
            'decoder_output': {0: 'batch', 1: 'length'}
        },
        opset_version=13,
        do_constant_folding=True
    )
    print(f"Decoder exported to {decoder_path}")

    # Export joint network
    print("\n=== Exporting Joint Network ===")
    joint_path = OUTPUT_DIR / "joint.onnx"

    # Joint expects encoder output and decoder output
    dummy_enc_out = torch.randn(1, 1, model.encoder._feat_out)
    dummy_dec_out = torch.randn(1, 1, model.decoder.prediction.hidden_size)

    torch.onnx.export(
        model.joint.joint_net,
        (dummy_enc_out, dummy_dec_out),
        joint_path,
        input_names=['encoder_output', 'decoder_output'],
        output_names=['logits'],
        dynamic_axes={
            'encoder_output': {0: 'batch'},
            'decoder_output': {0: 'batch'},
            'logits': {0: 'batch'}
        },
        opset_version=13,
        do_constant_folding=True
    )
    print(f"Joint network exported to {joint_path}")

    # Quantize models to INT8
    print("\n=== Quantizing to INT8 ===")
    for model_name in ['encoder', 'decoder', 'joint']:
        fp32_path = OUTPUT_DIR / f"{model_name}.onnx"
        int8_path = OUTPUT_DIR / f"{model_name}_int8.onnx"

        quantize_dynamic(
            str(fp32_path),
            str(int8_path),
            weight_type=QuantType.QInt8
        )
        print(f"Quantized {model_name} saved to {int8_path}")

    # Save metadata
    metadata = {
        "vocab": vocab,
        "vocab_size": vocab_size,
        "blank_idx": blank_idx,
        "checkpoint": CHECKPOINT_PATH,
        "epoch": 80,
        "val_wer": 0.152,
        "profile": "rare_words",
        "encoder_subsampling": 2,
        "hidden_size": model.decoder.prediction.hidden_size,
        "encoder_hidden": model.encoder._feat_out,
        "note": "CRITICAL: blank token is at index 29, not 0!"
    }

    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to {metadata_path}")
    print(f"Blank index: {blank_idx}")
    print(f"Vocab size: {vocab_size}")
    print("\nExport complete!")

    return metadata

if __name__ == "__main__":
    metadata = export_model()
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)
    print(f"Models exported to: {OUTPUT_DIR}/")
    print(f"Blank token index: {metadata['blank_idx']}")
    print(f"Vocabulary size: {metadata['vocab_size']}")
    print("\nFiles created:")
    for file in OUTPUT_DIR.glob("*.onnx"):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name}: {size_mb:.1f} MB")