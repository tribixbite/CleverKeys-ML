#!/usr/bin/env python3
"""
Test Python inference to verify model outputs correct predictions
"""

import json
import numpy as np
import onnxruntime as ort
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'new'))
from train_transducer_personalized import PersonalizedSwipeFeaturizer, determine_resample_target, resample_points


def load_test_data(line_num=431621):
    """Load test data from training set"""
    data_path = '../../data/train_final_train.jsonl'
    with open(data_path, 'r') as f:
        for i, line in enumerate(f, 1):
            if i == line_num:
                return json.loads(line)
    raise ValueError(f"Line {line_num} not found")


def main():
    # Load test data
    test_data = load_test_data(431621)  # "hello"
    print(f"Testing with word: '{test_data['word']}'")
    print(f"Points: {len(test_data['points'])} samples")

    # Extract features using same featurizer as training
    featurizer = PersonalizedSwipeFeaturizer()

    # Transform coordinates from [0, 1] to [-1, 1] to match training
    transformed_points = []
    for pt in test_data['points']:
        transformed_points.append({
            'x': pt['x'] * 2.0 - 1.0,
            'y': pt['y'] * 2.0 - 1.0,
            't': pt['t']
        })

    # Determine resample target based on length
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }
    target_len = determine_resample_target(len(transformed_points), preprocess_cfg)
    print(f"Resampling from {len(transformed_points)} to {target_len} points")

    # Resample points
    resampled_points = resample_points(transformed_points, target_len)

    # Extract features - featurizer expects list of dicts
    features = featurizer(resampled_points)
    print(f"Features shape: {features.shape}")
    print(f"First frame features: {features[0, :5]}")

    # Load ONNX models
    encoder = ort.InferenceSession('../models/best_latest/encoder.onnx')
    decoder = ort.InferenceSession('../models/best_latest/decoder_joint.onnx')

    # Prepare encoder input - transpose to [batch, feat_dim, seq_len]
    encoder_input = features.T[np.newaxis, :, :]  # [1, 37, 96]
    length = np.array([features.shape[0]], dtype=np.int64)

    # Run encoder
    encoder_outputs = encoder.run(
        None,
        {
            'audio_signal': encoder_input.astype(np.float32),
            'length': length
        }
    )

    encoded = encoder_outputs[0]  # [1, encoder_dim, time]
    print(f"Encoded shape: {encoded.shape}")

    # Simple greedy decode
    vocab_size = 30
    blank_id = 29
    max_symbols = 24

    decoded_tokens = []

    # Initialize decoder states
    pred_layers = 1
    pred_hidden = 192
    state_h = np.zeros((pred_layers, 1, pred_hidden), dtype=np.float32)
    state_c = np.zeros((pred_layers, 1, pred_hidden), dtype=np.float32)

    last_token = 0  # Start with <blank>

    # Process each encoder frame
    for t in range(encoded.shape[2]):
        encoder_frame = encoded[:, :, t:t+1]  # [1, encoder_dim, 1]

        symbols_emitted = 0
        max_symbols_per_frame = 6

        while symbols_emitted < max_symbols_per_frame and len(decoded_tokens) < max_symbols:
            # Run decoder
            decoder_outputs = decoder.run(
                None,
                {
                    'encoder_outputs': encoder_frame,
                    'targets': np.array([[last_token]], dtype=np.int32),
                    'target_length': np.array([1], dtype=np.int32),
                    'input_states_1': state_h,
                    'input_states_2': state_c
                }
            )

            logits = decoder_outputs[0]  # [batch, time, decoder_time, vocab]
            state_h = decoder_outputs[2]
            state_c = decoder_outputs[3]

            # Get prediction
            if len(logits.shape) == 4:
                probs = logits[0, 0, 0, :]  # Get first time step
            else:
                probs = logits[0, 0, :]

            predicted_token = np.argmax(probs)

            if predicted_token == blank_id:
                # Blank - move to next frame
                break
            else:
                decoded_tokens.append(predicted_token)
                last_token = predicted_token
                symbols_emitted += 1

    # Convert tokens to text
    with open('../models/best_latest/runtime_meta.json', 'r') as f:
        meta = json.load(f)

    decoded_text = ''.join([meta['id_to_char'][str(t)] for t in decoded_tokens if str(t) in meta['id_to_char']])

    print(f"\nDecoded tokens: {decoded_tokens}")
    print(f"Decoded text: '{decoded_text}'")
    print(f"Expected: 'hello'")
    print(f"Match: {decoded_text == 'hello'}")


if __name__ == '__main__':
    main()