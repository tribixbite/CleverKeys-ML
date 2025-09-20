#!/usr/bin/env python3
"""
Export end-to-end RNNT model that includes greedy decoding.
This wraps the entire inference pipeline for easier TypeScript integration.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import nemo.collections.asr as nemo_asr
import onnxruntime as ort

CHECKPOINT_PATH = "/home/will/git/swype/cleverkeys/rnnt_checkpoints_rare_words_20250919_140007/lightning_logs/version_0/checkpoints/epoch=epoch=80-wer=val_wer=0.152.ckpt"
OUTPUT_DIR = Path("onnx_rare_words_epoch80")
OUTPUT_DIR.mkdir(exist_ok=True)

class GreedyRNNTInfer(nn.Module):
    """End-to-end RNNT model with greedy decoding built in."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.blank_idx = model.decoder.blank_idx
        self.vocab_size = model.decoder.vocab_size
        self.max_symbols_per_step = 5

    def forward(self, audio_signal, length):
        """
        End-to-end forward with greedy decoding.
        Returns token indices for the best hypothesis.
        """
        # Get encoder output
        encoded, encoded_len = self.model.encoder(audio_signal=audio_signal, length=length)

        batch_size = encoded.shape[0]

        # Run greedy RNNT decoding
        with torch.no_grad():
            hypotheses = self.model.decoding.rnnt_decoder_predictions_tensor(
                encoded, encoded_len,
                return_hypotheses=True
            )

        # Extract token sequences
        if hypotheses and len(hypotheses) > 0:
            hyp = hypotheses[0]
            if hasattr(hyp, 'y_sequence'):
                tokens = hyp.y_sequence
                # Pad to fixed length for ONNX
                max_len = 20
                if len(tokens) < max_len:
                    tokens = torch.cat([
                        tokens,
                        torch.full((max_len - len(tokens),), self.blank_idx, dtype=torch.long)
                    ])
                else:
                    tokens = tokens[:max_len]
                return tokens.unsqueeze(0).float()  # [1, max_len]

        # Return blank sequence if no hypothesis
        return torch.full((batch_size, 20), self.blank_idx, dtype=torch.float32)

def export_e2e_model():
    """Export end-to-end RNNT model."""

    print("Loading checkpoint...")
    model = nemo_asr.models.EncDecRNNTModel.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()
    model = model.cpu()

    # Get model info
    vocab = model.joint.vocabulary
    blank_idx = model.decoder.blank_idx
    vocab_size = len(vocab) + 1

    print(f"Model info:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Blank index: {blank_idx}")

    # Create end-to-end wrapper
    e2e_model = GreedyRNNTInfer(model)
    e2e_model.eval()

    # Export to ONNX
    output_path = OUTPUT_DIR / "rnnt_e2e.onnx"

    dummy_audio = torch.randn(1, 37, 96)
    dummy_length = torch.tensor([96], dtype=torch.long)

    print("Exporting to ONNX...")
    try:
        torch.onnx.export(
            e2e_model,
            (dummy_audio, dummy_length),
            output_path,
            input_names=['audio_signal', 'length'],
            output_names=['tokens'],
            dynamic_axes={
                'audio_signal': {2: 'time'},
            },
            opset_version=13,
            do_constant_folding=False,  # Keep it simple
            export_params=True
        )
        print(f"Model exported to {output_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("\nFalling back to TorchScript export...")

        # Try TorchScript instead
        script_model = torch.jit.script(e2e_model)
        script_path = OUTPUT_DIR / "rnnt_e2e.pt"
        script_model.save(str(script_path))
        print(f"TorchScript model saved to {script_path}")

    # Save metadata
    metadata = {
        "vocab": list(vocab),
        "vocab_size": vocab_size,
        "blank_idx": blank_idx,
        "checkpoint": CHECKPOINT_PATH,
        "epoch": 80,
        "val_wer": 0.152,
        "profile": "rare_words",
        "model_type": "e2e_greedy",
        "max_output_len": 20,
        "note": "End-to-end RNNT with greedy decoding"
    }

    metadata_path = OUTPUT_DIR / "metadata_e2e.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata

def test_accuracy():
    """Test the model accuracy on our test data."""
    print("\n" + "=" * 80)
    print("TESTING MODEL ACCURACY")
    print("=" * 80)

    model = nemo_asr.models.EncDecRNNTModel.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
    else:
        model = model.cpu()
        device = 'cpu'

    # Load all test data
    with open("test_traces.json", "r") as f:
        data = json.load(f)
        test_samples = data['samples']

    print(f"Testing on {len(test_samples)} samples...")

    correct = 0
    predictions = []

    for i, sample in enumerate(test_samples):
        word = sample['word']
        features = np.array(sample['features'], dtype=np.float32)

        # Convert to tensor
        features_tensor = torch.from_numpy(features).unsqueeze(0).transpose(1, 2).to(device)
        length_tensor = torch.tensor([features.shape[0]], dtype=torch.long).to(device)

        with torch.no_grad():
            # Encode
            encoded, encoded_len = model.encoder(
                audio_signal=features_tensor,
                length=length_tensor
            )

            # Decode
            hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
                encoded, encoded_len,
                return_hypotheses=True
            )

        if hypotheses and len(hypotheses) > 0:
            hyp = hypotheses[0]
            prediction = hyp.text if hasattr(hyp, 'text') else ""
        else:
            prediction = ""

        is_correct = prediction == word
        if is_correct:
            correct += 1

        predictions.append({
            'true': word,
            'pred': prediction,
            'correct': is_correct
        })

        if i < 10:  # Show first 10
            print(f"  {i+1}. '{word}' -> '{prediction}' {'✓' if is_correct else '✗'}")

    accuracy = correct / len(test_samples) * 100
    print(f"\nOverall Accuracy: {correct}/{len(test_samples)} = {accuracy:.1f}%")

    # Analyze by word frequency
    common_words = ['the', 'and', 'you', 'that', 'this', 'with', 'have', 'from', 'they', 'will']
    rare_words = ['kubernetes', 'cryptocurrency', 'blockchain', 'algorithm']

    common_correct = sum(1 for p in predictions if p['true'] in common_words and p['correct'])
    common_total = sum(1 for p in predictions if p['true'] in common_words)

    rare_correct = sum(1 for p in predictions if p['true'] in rare_words and p['correct'])
    rare_total = sum(1 for p in predictions if p['true'] in rare_words)

    if common_total > 0:
        print(f"Common words: {common_correct}/{common_total} = {common_correct/common_total*100:.1f}%")
    if rare_total > 0:
        print(f"Rare words: {rare_correct}/{rare_total} = {rare_correct/rare_total*100:.1f}%")

    return accuracy

if __name__ == "__main__":
    # Test accuracy first
    accuracy = test_accuracy()

    if accuracy < 30:
        print("\nWARNING: Accuracy is lower than expected.")
        print("The model may need different preprocessing or the data may be incorrect.")

    # Export model
    print("\n" + "=" * 80)
    print("EXPORTING MODEL")
    print("=" * 80)
    metadata = export_e2e_model()

    print("\nExport complete!")