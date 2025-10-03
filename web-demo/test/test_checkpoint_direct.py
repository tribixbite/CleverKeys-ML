#!/usr/bin/env python3
"""
Test the checkpoint directly without ONNX export
"""

import torch
import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'new'))

# Import the actual model class used in training
from train_transducer_personalized import PersonalizedRNNTModel

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
    checkpoint_path = '/home/will/git/swype/cleverkeys/9292025script/20251002/rnnt_checkpoints_medium_balanced_20251003_013434/lightning_logs/version_0/checkpoints/epoch=epoch=71-wer=val_wer=0.457.ckpt'

    print(f"Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Load the model
    print("Creating model from checkpoint...")
    model = PersonalizedRNNTModel(checkpoint['hyper_parameters']['cfg'])

    # Load the state dict
    model.load_state_dict(checkpoint['state_dict'])
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
    # Model expects [batch, features, time] for encoder
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).transpose(1, 2)
    feature_lengths = torch.tensor([features.shape[0]], dtype=torch.long)

    print(f"Features shape for encoder: {features_tensor.shape}")

    with torch.no_grad():
        # Use the model's forward pass
        try:
            # Call the model - it returns (log_probs, encoded_len)
            log_probs, encoded_len = model(
                input_signal=None,
                input_signal_length=None,
                processed_signal=features_tensor,
                processed_signal_length=feature_lengths
            )

            print(f"Log probs shape: {log_probs.shape}")
            print(f"Encoded length: {encoded_len}")

            # Use the model's greedy decoder
            if hasattr(model, 'decoding'):
                # Use the decoding module to get predictions
                hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=log_probs,
                    encoded_lengths=encoded_len,
                    return_hypotheses=True
                )
                if hypotheses and len(hypotheses) > 0:
                    pred_text = hypotheses[0].text if hasattr(hypotheses[0], 'text') else str(hypotheses[0])
                    print(f"Predicted: '{pred_text}'")
                    print(f"Expected:  '{word}'")
            else:
                # Manual decode
                vocab = checkpoint['hyper_parameters']['cfg']['labels']
                pred_indices = greedy_predictions[0].cpu().numpy()
                pred_text = ''.join([vocab[idx] for idx in pred_indices if idx < len(vocab)])
                print(f"Predicted (manual): '{pred_text}'")
                print(f"Expected:  '{word}'")

        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()