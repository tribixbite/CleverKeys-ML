#!/usr/bin/env python3
"""
Verify PyTorch checkpoint predictions directly without ONNX.
This tests if the model checkpoint actually works correctly.
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path
import linecache

sys.path.insert(0, 'new')
from train_transducer_personalized import (
    PersonalizedRNNTModel,
    PersonalizedSwipeFeaturizer,
    resample_points,
    clamp
)

def get_test_data(line_number=431621):
    """Get hello swipe data from specific line"""
    data_path = 'data/train_final_train.jsonl'
    line = linecache.getline(data_path, line_number)
    if line:
        data = json.loads(line)
        return data['points'], data['word']
    return None, None

def prepare_features(points):
    """Prepare features exactly as training does"""
    prepared = []
    for idx, pt in enumerate(points):
        raw_x = float(pt.get("x", 0.0))
        raw_y = float(pt.get("y", 0.0))
        centered_x = raw_x * 2.0 - 1.0  # Transform [0,1] → [-1,1]
        centered_y = raw_y * 2.0 - 1.0
        centered_x = clamp(centered_x, -1.5, 1.5)
        centered_y = clamp(centered_y, -1.5, 1.5)
        raw_t = float(pt.get("t", idx * 10.0))
        prepared.append({"x": centered_x, "y": centered_y, "t": raw_t})

    # Resample to 82
    resampled = resample_points(prepared, 82)

    # Extract features
    featurizer = PersonalizedSwipeFeaturizer(mobile_features=False)
    features = featurizer(resampled)

    # Pad to 37
    if features.shape[1] < 37:
        padding = np.zeros((features.shape[0], 37 - features.shape[1]), dtype=np.float32)
        features = np.concatenate([features, padding], axis=1)

    return features

def main():
    CHECKPOINT_PATH = "./9292025script/20251002/rnnt_checkpoints_medium_balanced_20251003_013434/lightning_logs/version_0/checkpoints/epoch=epoch=71-wer=val_wer=0.457.ckpt"

    # Load the PyTorch model
    print(f"Loading PyTorch model from checkpoint...")
    print(f"  {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    model = PersonalizedRNNTModel(checkpoint['hyper_parameters']['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.freeze()

    print("Model loaded successfully")
    print(f"  Encoder: {model.encoder.__class__.__name__}")
    print(f"  Decoder: {model.decoder.__class__.__name__}")

    # Test multiple words
    test_cases = [
        431621,  # hello
        100,     # the
        1000,    # the
        10000,   # market
    ]

    print("\n" + "="*60)
    print("Testing PyTorch model predictions:")
    print("="*60)

    for line_num in test_cases:
        # Get test data
        points, word = get_test_data(line_num)
        if not points:
            continue

        # Prepare features
        features = prepare_features(points)

        # Save features for comparison
        if line_num == 431621:  # Save hello features
            np.save("pytorch_features.npy", features)
            print(f"\nSaved 'hello' features to pytorch_features.npy")

        # Convert to tensor format expected by model
        # Shape should be [batch, features, time]
        features_tensor = torch.from_numpy(features.T).float().unsqueeze(0)  # [1, 37, 82]
        feature_lengths = torch.tensor([features.shape[0]], dtype=torch.long)

        print(f"\nLine {line_num}: Word = '{word}'")
        print(f"  Features shape: {features_tensor.shape}")

        # Run the model's forward pass
        with torch.no_grad():
            # Try different decoding methods

            # Method 1: Direct forward pass
            try:
                log_probs, encoded_len = model(
                    input_signal=None,
                    input_signal_length=None,
                    processed_signal=features_tensor,
                    processed_signal_length=feature_lengths
                )
                print(f"  Encoder output shape: {log_probs.shape}")
                print(f"  Encoded length: {encoded_len}")
            except Exception as e:
                print(f"  Forward pass failed: {e}")

            # Method 2: Use transcribe if available
            try:
                if hasattr(model, 'transcribe'):
                    # NeMo's transcribe expects list of numpy arrays
                    hypotheses = model.transcribe([features], batch_size=1)
                    print(f"  Transcribe result: '{hypotheses[0] if hypotheses else 'NONE'}'")
            except Exception as e:
                print(f"  Transcribe method not available or failed: {e}")

            # Method 3: Use decoding module if available
            try:
                if hasattr(model, 'decoding'):
                    # Use the decoding module
                    best_hyp, all_hyp = model.decoding.rnnt_decoder_predictions_tensor(
                        log_probs,
                        encoded_len,
                        return_hypotheses=True
                    )

                    if best_hyp and len(best_hyp) > 0:
                        text = best_hyp[0].text if hasattr(best_hyp[0], 'text') else str(best_hyp[0])
                        print(f"  Decoder result: '{text}'")
                    else:
                        print(f"  Decoder returned no hypothesis")
            except Exception as e:
                print(f"  Decoding failed: {e}")

    print("\n" + "="*60)
    print("IMPORTANT: If predictions are wrong here, the checkpoint itself is bad.")
    print("If predictions are correct here but wrong in ONNX, the export is broken.")
    print("="*60)

if __name__ == '__main__':
    main()