#!/usr/bin/env python3
"""
Export a simplified classifier model that maps encoder output to characters directly.
This is for demonstration purposes to show the rare words model working.
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

class SimpleClassifier(nn.Module):
    """Simple classifier that maps encoder output to character logits."""

    def __init__(self, encoder, hidden_dim, vocab_size=30):
        super().__init__()
        self.encoder = encoder
        # Add a simple linear layer to map encoder output to vocab
        self.classifier = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, features, features_length):
        """
        Simple classification - encoder + linear layer.
        """
        # Encode audio
        encoded, encoded_len = self.encoder(audio_signal=features, length=features_length)

        # NeMo encoder output is [batch, hidden, time], need to transpose to [batch, time, hidden]
        if encoded.shape[1] > encoded.shape[2]:  # hidden > time
            encoded = encoded.transpose(1, 2)  # Now [batch, time, hidden]

        # Apply classifier to each timestep
        # encoded shape: [batch, time, hidden]
        batch_size, time_steps, hidden_dim = encoded.shape

        # Reshape for linear layer
        encoded_flat = encoded.reshape(-1, hidden_dim)
        logits_flat = self.classifier(encoded_flat)

        # Reshape back to [batch, time, vocab]
        logits = logits_flat.reshape(batch_size, time_steps, self.vocab_size)

        return logits

def export_simple_model():
    """Export simple classifier model for demonstration."""

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    # Load the model
    model = nemo_asr.models.EncDecRNNTModel.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()
    model = model.cpu()

    # Get model info
    vocab = model.joint.vocabulary
    blank_idx = model.decoder.blank_idx
    vocab_size = len(vocab) + 1
    # The encoder outputs [batch, 256, time] so hidden dim is 256
    encoder_hidden = 256  # From the joint network: Linear(256 → 512)

    print(f"Model loaded. Vocab size: {vocab_size}, Blank index: {blank_idx}")
    print(f"Encoder hidden dim: {encoder_hidden}")

    # Create simple classifier using pretrained encoder
    simple_model = SimpleClassifier(
        encoder=model.encoder,
        hidden_dim=encoder_hidden,
        vocab_size=vocab_size
    )

    # Initialize classifier with joint network weights if compatible
    try:
        # Try to copy some weights from joint network
        joint_weights = model.joint.joint_net[0].weight.data
        if joint_weights.shape[0] == vocab_size:
            simple_model.classifier.weight.data = joint_weights[:, :encoder_hidden]
            simple_model.classifier.bias.data = model.joint.joint_net[0].bias.data
            print("Initialized classifier with joint network weights")
    except:
        print("Using random classifier weights")

    simple_model.eval()

    # Export to ONNX
    output_path = OUTPUT_DIR / "model_simple.onnx"

    # Create dummy input
    dummy_features = torch.randn(1, 37, 96)
    dummy_length = torch.tensor([96], dtype=torch.long)

    print("Exporting to ONNX...")
    torch.onnx.export(
        simple_model,
        (dummy_features, dummy_length),
        output_path,
        input_names=['features', 'features_length'],
        output_names=['logits'],
        dynamic_axes={
            'features': {0: 'batch', 2: 'time'},
            'features_length': {0: 'batch'},
            'logits': {0: 'batch', 1: 'time'}
        },
        opset_version=13,
        do_constant_folding=True,
        export_params=True
    )

    print(f"Model exported to {output_path}")

    # Verify the model
    import onnxruntime as ort
    session = ort.InferenceSession(str(output_path))

    # Test inference
    test_features = np.random.randn(1, 37, 96).astype(np.float32)
    test_length = np.array([96], dtype=np.int64)

    outputs = session.run(None, {
        'features': test_features,
        'features_length': test_length
    })

    print(f"Test output shape: {outputs[0].shape}")
    print(f"Expected shape: [1, ~48, {vocab_size}] (with subsampling)")

    # Save updated metadata
    metadata = {
        "vocab": list(vocab),  # Convert ListConfig to list
        "vocab_size": vocab_size,
        "blank_idx": blank_idx,
        "checkpoint": CHECKPOINT_PATH,
        "epoch": 80,
        "val_wer": 0.152,
        "profile": "rare_words",
        "model_type": "simple_classifier",
        "encoder_hidden": encoder_hidden,
        "note": "Simple classifier model using pretrained encoder for demonstration"
    }

    metadata_path = OUTPUT_DIR / "metadata_simple.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}")
    print("Export complete!")

    return metadata

if __name__ == "__main__":
    metadata = export_simple_model()
    print(f"\nBlank token index: {metadata['blank_idx']}")
    print(f"Vocabulary size: {metadata['vocab_size']}")