#!/usr/bin/env python3
"""
Export complete RNNT model for TypeScript inference.
Creates a simplified greedy decoder model that can run end-to-end.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import onnx
import nemo.collections.asr as nemo_asr

# Checkpoint path
CHECKPOINT_PATH = "/home/will/git/swype/cleverkeys/rnnt_checkpoints_rare_words_20250919_140007/lightning_logs/version_0/checkpoints/epoch=epoch=80-wer=val_wer=0.152.ckpt"
OUTPUT_DIR = Path("onnx_rare_words_epoch80")
OUTPUT_DIR.mkdir(exist_ok=True)

class GreedyRNNTInfer(nn.Module):
    """Greedy RNNT decoder for ONNX export."""

    def __init__(self, encoder, decoder, joint, blank_idx=29, max_symbols_per_step=10):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.blank_idx = blank_idx
        self.max_symbols = max_symbols_per_step
        self.vocab_size = 30

    def forward(self, features, features_length):
        """
        End-to-end greedy decoding.
        Returns logits for each timestep.
        """
        # Encode audio
        encoded, encoded_len = self.encoder(audio_signal=features, length=features_length)
        batch_size = encoded.shape[0]
        max_time = encoded.shape[1]

        # Initialize decoder state
        targets = torch.zeros((batch_size, 1), dtype=torch.long, device=features.device)
        # Access the prediction network properly
        pred_net = self.decoder.prediction
        if hasattr(pred_net, 'dec_rnn'):
            # NeMo's StatelessTransducerDecoder structure
            states = None  # Will be initialized inside the network
        else:
            states = None

        # Collect all logits for the sequence
        all_logits = []

        for t in range(max_time):
            # Get encoder features for this timestep
            encoder_t = encoded[:, t:t+1, :]  # [B, 1, D_enc]

            # Inner loop for symbols at this timestep
            for s in range(self.max_symbols):
                # Decoder prediction - NeMo's decoder expects just targets
                decoder_output = self.decoder.prediction(targets, states=states)
                if isinstance(decoder_output, tuple):
                    decoder_output, states = decoder_output
                decoder_t = decoder_output[:, -1:, :]  # [B, 1, D_dec]

                # Joint network
                logits = self.joint.joint_net(encoder_t, decoder_t)  # [B, 1, V]
                all_logits.append(logits)

                # Get prediction
                preds = torch.argmax(logits, dim=-1)  # [B, 1]

                # If blank, move to next time step
                if preds.squeeze(-1).item() == self.blank_idx:
                    break

                # Otherwise, update targets for next decoder step
                targets = torch.cat([targets, preds], dim=1)

        # Stack all logits
        if all_logits:
            final_logits = torch.cat(all_logits, dim=1)  # [B, total_steps, V]
        else:
            # Fallback if no predictions
            final_logits = torch.zeros((batch_size, 1, self.vocab_size), device=features.device)

        return final_logits

def export_greedy_model():
    """Export greedy RNNT model for end-to-end inference."""

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    # Load the model
    model = nemo_asr.models.EncDecRNNTModel.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()
    model = model.cpu()

    # Get model info
    vocab = model.joint.vocabulary
    blank_idx = model.decoder.blank_idx
    vocab_size = len(vocab) + 1

    print(f"Model loaded. Vocab size: {vocab_size}, Blank index: {blank_idx}")

    # Create greedy inference wrapper
    greedy_model = GreedyRNNTInfer(
        encoder=model.encoder,
        decoder=model.decoder,
        joint=model.joint,
        blank_idx=blank_idx,
        max_symbols_per_step=5
    )
    greedy_model.eval()

    # Export to ONNX
    output_path = OUTPUT_DIR / "model_greedy.onnx"

    # Create dummy input
    dummy_features = torch.randn(1, 37, 96)
    dummy_length = torch.tensor([96], dtype=torch.long)

    print("Exporting to ONNX...")
    torch.onnx.export(
        greedy_model,
        (dummy_features, dummy_length),
        output_path,
        input_names=['features', 'features_length'],
        output_names=['logits'],
        dynamic_axes={
            'features': {0: 'batch', 2: 'time'},
            'features_length': {0: 'batch'},
            'logits': {0: 'batch', 1: 'steps'}
        },
        opset_version=13,
        do_constant_folding=True,
        export_params=True
    )

    print(f"Model exported to {output_path}")

    # Save updated metadata
    metadata = {
        "vocab": vocab,
        "vocab_size": vocab_size,
        "blank_idx": blank_idx,
        "checkpoint": CHECKPOINT_PATH,
        "epoch": 80,
        "val_wer": 0.152,
        "profile": "rare_words",
        "model_type": "greedy_rnnt",
        "max_symbols_per_step": 5,
        "note": "Greedy RNNT model for end-to-end inference"
    }

    metadata_path = OUTPUT_DIR / "metadata_greedy.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}")
    print("Export complete!")

    return metadata

if __name__ == "__main__":
    metadata = export_greedy_model()
    print(f"\nBlank token index: {metadata['blank_idx']}")
    print(f"Vocabulary size: {metadata['vocab_size']}")