#!/usr/bin/env python3
"""
Test the checkpoint directly using NeMo's inference (not ONNX export)
"""

import torch
from nemo.collections.asr.models import EncDecRNNTModel
import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'new'))

from train_transducer_personalized import (
    PersonalizedSwipeFeaturizer,
    resample_points,
    clamp
)

def get_hello_data():
    """Get hello swipe data from line 431621"""
    import linecache
    data_path = '../../data/train_final_train.jsonl'
    line = linecache.getline(data_path, 431621)
    if line:
        data = json.loads(line)
        return data['points'], data['word']
    return None, None

def main():
    # Load the actual checkpoint (not ONNX)
    checkpoint_path = '/home/will/git/swype/cleverkeys/9292025script/20251002/rnnt_checkpoints_medium_balanced_20251003_013434/lightning_logs/version_0/checkpoints/epoch=epoch=71-wer=val_wer=0.457.ckpt'

    print(f"Loading NeMo model from checkpoint...")
    model = EncDecRNNTModel.restore_from(checkpoint_path, map_location='cpu')
    model.eval()

    # Get test data
    points, word = get_hello_data()
    print(f"Testing: '{word}' with {len(points)} points")

    # Process swipe exactly as training does
    prepared = []
    for idx, pt in enumerate(points):
        raw_x = float(pt.get("x", 0.0))
        raw_y = float(pt.get("y", 0.0))
        centered_x = raw_x * 2.0 - 1.0
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

    # Convert to tensor and add batch dimension
    # NeMo expects [batch, time, features] for processed_signal
    features_tensor = torch.from_numpy(features).float().unsqueeze(0)
    feature_lengths = torch.tensor([features.shape[0]], dtype=torch.long)

    print(f"Features shape: {features_tensor.shape}")

    # Use NeMo's transcribe method (which handles all the decoding internally)
    with torch.no_grad():
        # NeMo expects the feature tensor directly as processed_signal
        # We need to transpose to [batch, features, time] for the encoder
        features_for_encoder = features_tensor.transpose(1, 2)  # [1, 37, 82]

        # Call the model's forward pass directly
        log_probs, encoded_len, greedy_predictions = model(
            input_signal=None,
            input_signal_length=None,
            processed_signal=features_for_encoder,
            processed_signal_length=feature_lengths
        )

        # Decode the predictions
        hypotheses = model.decoding.decode_predictions_tensor(
            greedy_predictions[0], feature_lengths, return_hypotheses=True
        )

        if hypotheses:
            pred_text = hypotheses[0].text if hasattr(hypotheses[0], 'text') else str(hypotheses[0])
            print(f"Predicted: '{pred_text}'")
        else:
            print(f"Predicted: (no output)")

        print(f"Expected:  '{word}'")

    # Also try the transcribe_generator method if available
    try:
        # Create a simple batch
        batch = (features_for_encoder, feature_lengths)

        # Get predictions using the model's built-in greedy decoder
        print("\nUsing model's greedy decoder:")
        model.decoding.decoding.max_symbols_per_step = 8

        best_hyp, all_hyp = model.decoding.rnnt_decoder_predictions_tensor(
            features_for_encoder,
            decoder_lengths=feature_lengths,
            return_hypotheses=True
        )

        if best_hyp and len(best_hyp) > 0:
            print(f"Greedy prediction: '{best_hyp[0].text}'")

    except Exception as e:
        print(f"Could not use transcribe_generator: {e}")

if __name__ == '__main__':
    main()