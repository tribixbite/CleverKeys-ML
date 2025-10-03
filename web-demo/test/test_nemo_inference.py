#!/usr/bin/env python3
"""
Test ONNX models using NeMo's actual inference pipeline
This mimics exactly what happens during training validation
"""

import json
import numpy as np
import torch
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'new'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'trained_models', 'nema1'))

from train_transducer_personalized import PersonalizedSwipeFeaturizer, determine_resample_target, resample_points
from export_common import load_trained_model


def get_companion_data():
    """Get companion swipe data from line 22440"""
    data_path = '../../data/train_final_train.jsonl'
    with open(data_path, 'r') as f:
        for i, line in enumerate(f, 1):
            if i == 22440:
                data = json.loads(line)
                return data['points'], data['word']
    return None, None


def process_swipe(points, featurizer, preprocess_cfg):
    """Process swipe points through full pipeline - exactly as in training"""
    # 1. Transform coordinates from [0, 1] to [-1, 1]
    transformed_points = []
    for pt in points:
        transformed_points.append({
            'x': pt['x'] * 2.0 - 1.0,
            'y': pt['y'] * 2.0 - 1.0,
            't': pt['t']
        })

    # 2. Determine resample target
    target_len = determine_resample_target(len(transformed_points), preprocess_cfg)

    # 3. Resample points
    resampled_points = resample_points(transformed_points, target_len)

    # 4. Extract features
    features = featurizer(resampled_points)

    return features


def test_with_nemo_model():
    """Test using the actual NeMo model that was exported"""
    print("="*70)
    print("TESTING WITH NEMO MODEL (EXACT TRAINING INFERENCE)")
    print("="*70)

    # Load the checkpoint that was exported
    checkpoint_path = "9292025script/20251002/rnnt_checkpoints_short_common_20251002_233024/conformer_rnnt_final.nemo"

    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Looking for alternatives...")
        # Try to find any .nemo file
        base = Path("9292025script/20251002")
        nemo_files = list(base.glob("**/*.nemo"))
        if nemo_files:
            checkpoint_path = str(nemo_files[0])
            print(f"Using: {checkpoint_path}")
        else:
            print("No .nemo files found!")
            return

    print(f"\nLoading NeMo model from: {checkpoint_path}")
    model = load_trained_model(checkpoint_path)
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Get companion data
    points, expected_word = get_companion_data()
    if points is None:
        print("ERROR: Could not load companion data")
        return

    print(f"\nTesting word: '{expected_word}' ({len(points)} points)")

    # Process swipe
    featurizer = PersonalizedSwipeFeaturizer()
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }

    features = process_swipe(points, featurizer, preprocess_cfg)
    print(f"Features shape: {features.shape}")
    print(f"First 5 features: {features[0, :5]}")

    # Convert to torch tensor with batch dimension
    # Shape should be [batch, time, features]
    signal = torch.from_numpy(features).float().unsqueeze(0).to(device)
    signal_len = torch.tensor([features.shape[0]], dtype=torch.long).to(device)

    print(f"\nInput tensor shape: {signal.shape}")
    print(f"Input length: {signal_len}")

    # Run encoder
    with torch.no_grad():
        encoded, encoded_len = model.forward(
            input_signal=signal,
            input_signal_length=signal_len
        )
        print(f"Encoded shape: {encoded.shape}")
        print(f"Encoded length: {encoded_len}")

        # Run greedy decoding - this is the EXACT method used in training
        predictions = model.decoding.rnnt_decoder_predictions_tensor(
            encoded, encoded_len
        )

        print(f"\nPredictions type: {type(predictions)}")
        if predictions:
            if isinstance(predictions[0], list) and predictions[0]:
                pred_text = predictions[0][0].text
            else:
                pred_text = str(predictions[0])
        else:
            pred_text = ""

    print(f"\nPredicted: '{pred_text}'")
    print(f"Expected: '{expected_word}'")

    if pred_text == expected_word:
        print("\n✅ SUCCESS!")
    else:
        print("\n❌ FAILED")

    # Also try to get the actual token IDs
    try:
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'blank_idx'):
            print(f"\nBlank index: {model.decoder.blank_idx}")
        if hasattr(model, 'joint') and hasattr(model.joint, 'vocabulary'):
            vocab = model.joint.vocabulary
            print(f"Vocabulary size: {len(vocab)}")
            print(f"First 10 vocab items: {vocab[:10]}")
    except Exception as e:
        print(f"Could not get vocab info: {e}")


def main():
    try:
        test_with_nemo_model()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()