#!/usr/bin/env python3
"""
Export the Oct 3 checkpoint to ONNX for web inference.
"""

import torch
import torch.onnx
import numpy as np
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'new'))

from train_transducer_personalized import PersonalizedRNNTModel

def export_models(checkpoint_path, output_dir):
    """Export encoder and decoder to separate ONNX models"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Create and load model
    model = PersonalizedRNNTModel(checkpoint['hyper_parameters']['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print("Model loaded successfully")

    # Export encoder
    print("\nExporting encoder...")
    batch_size = 1
    time_steps = 82
    n_features = 37

    dummy_input = torch.randn(batch_size, n_features, time_steps)
    dummy_length = torch.tensor([time_steps], dtype=torch.long)

    # Wrap encoder for export
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, audio_signal, length):
            encoded, encoded_len = self.encoder(audio_signal=audio_signal, length=length)
            return encoded

    encoder_wrapper = EncoderWrapper(model.encoder)
    encoder_wrapper.eval()

    torch.onnx.export(
        encoder_wrapper,
        (dummy_input, dummy_length),
        os.path.join(output_dir, "encoder.onnx"),
        input_names=['audio_signal', 'length'],
        output_names=['outputs'],
        dynamic_axes={
            'audio_signal': {2: 'time'},
            'length': {0: 'batch'},
            'outputs': {2: 'time_out'}
        },
        opset_version=14,
        verbose=False
    )
    print(f"Encoder exported to {output_dir}/encoder.onnx")

    # Export decoder with joint
    print("\nExporting decoder_joint...")

    # Get dimensions
    encoder_dim = 256  # From model config
    pred_hidden = 320
    num_layers = 2
    vocab_size = 30

    # Create stateful decoder wrapper
    class StatefulDecoderJoint(torch.nn.Module):
        def __init__(self, decoder, joint):
            super().__init__()
            self.decoder = decoder
            self.joint = joint

        def forward(self, encoder_outputs, targets, target_length, input_states_1, input_states_2):
            # Decoder expects (targets, target_length, initial_states)
            # Returns (decoder_output, new_states)

            # The decoder's predict method handles the LSTM internally
            decoder_output, (h_new, c_new) = self.decoder.prediction.predict(
                targets, target_length, (input_states_1, input_states_2)
            )

            # Joint network combines encoder and decoder outputs
            # encoder_outputs: [B, D, 1]
            # decoder_output: [B, 1, D]
            joint_output = self.joint.joint(encoder_outputs, decoder_output)

            # Return joint output and new states
            return joint_output, target_length, h_new, c_new

    decoder_joint = StatefulDecoderJoint(model.decoder, model.joint)
    decoder_joint.eval()

    # Dummy inputs for decoder
    dummy_enc = torch.randn(1, encoder_dim, 1)
    dummy_targets = torch.tensor([[0]], dtype=torch.long)
    dummy_target_len = torch.tensor([1], dtype=torch.long)
    dummy_h = torch.zeros(num_layers, 1, pred_hidden)
    dummy_c = torch.zeros(num_layers, 1, pred_hidden)

    torch.onnx.export(
        decoder_joint,
        (dummy_enc, dummy_targets, dummy_target_len, dummy_h, dummy_c),
        os.path.join(output_dir, "decoder_joint.onnx"),
        input_names=['encoder_outputs', 'targets', 'target_length', 'input_states_1', 'input_states_2'],
        output_names=['outputs', 'prednet_lengths', 'output_states_1', 'output_states_2'],
        dynamic_axes={
            'encoder_outputs': {0: 'batch', 2: 'time'},
            'targets': {0: 'batch', 1: 'time'},
            'target_length': {0: 'batch'},
            'input_states_1': {1: 'batch'},
            'input_states_2': {1: 'batch'},
            'outputs': {0: 'batch', 1: 'time_enc', 2: 'time_dec'},
            'output_states_1': {1: 'batch'},
            'output_states_2': {1: 'batch'}
        },
        opset_version=14,
        verbose=False
    )
    print(f"Decoder+Joint exported to {output_dir}/decoder_joint.onnx")

    # Save runtime metadata
    vocab = checkpoint['hyper_parameters']['cfg']['labels']
    runtime_meta = {
        "vocab_size": len(vocab),
        "blank_id": len(vocab),  # Blank is at the end
        "tokens": vocab + ['<blank>'],  # Add blank token
        "char_to_id": {char: i for i, char in enumerate(vocab)},
        "id_to_char": {str(i): char for i, char in enumerate(vocab)},
        "decoder_config": {
            "num_layers": num_layers,
            "hidden_size": pred_hidden,
            "encoder_dim": encoder_dim
        }
    }

    # Fix blank_id (it should be 29, not 30)
    runtime_meta["blank_id"] = 29
    runtime_meta["char_to_id"]["<blank>"] = 29
    runtime_meta["id_to_char"]["29"] = "<blank>"

    with open(os.path.join(output_dir, "runtime_meta.json"), "w") as f:
        json.dump(runtime_meta, f, indent=2)

    print(f"\nRuntime metadata saved to {output_dir}/runtime_meta.json")
    print(f"Vocab size: {runtime_meta['vocab_size']}")
    print(f"Blank ID: {runtime_meta['blank_id']}")

    return True

def main():
    checkpoint_path = '../../9292025script/20251002/rnnt_checkpoints_medium_balanced_20251003_013434/lightning_logs/version_0/checkpoints/epoch=epoch=71-wer=val_wer=0.457.ckpt'
    output_dir = '../models/oct3_correct'

    success = export_models(checkpoint_path, output_dir)

    if success:
        print(f"\n✓ Export complete! Models saved to {output_dir}")
        print("\nTo test:")
        print(f"  cd {os.path.dirname(output_dir)}")
        print(f"  python test/test_correct_rnnt_decode.py")

if __name__ == '__main__':
    main()