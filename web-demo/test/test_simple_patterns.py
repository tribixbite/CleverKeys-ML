#!/usr/bin/env python3
"""
Test if model can predict anything correctly - try simple patterns
"""

import json
import numpy as np
import onnxruntime as ort
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'new'))
from train_transducer_personalized import PersonalizedSwipeFeaturizer, determine_resample_target, resample_points


def create_synthetic_swipe(word, key_layout):
    """Create a perfect straight-line swipe for a word"""
    points = []
    t = 0

    for char in word:
        # Find key position
        for key_char, x, y in key_layout:
            if key_char == char:
                # Convert from [-1,1] to [0,1]
                x01 = (x + 1.0) / 2.0
                y01 = (y + 1.0) / 2.0
                points.append({'x': x01, 'y': y01, 't': t})
                t += 50
                break

    return points


def test_model_basic():
    """Test if model produces ANY reasonable output"""

    # Load key layout
    from train_transducer_personalized import load_key_centers
    key_layout = load_key_centers(None)

    # Load models
    encoder = ort.InferenceSession('../models/best_latest/encoder.onnx')
    decoder = ort.InferenceSession('../models/best_latest/decoder_joint.onnx')

    # Load runtime meta
    with open('../models/best_latest/runtime_meta.json', 'r') as f:
        meta = json.load(f)

    print("="*70)
    print("TESTING SYNTHETIC PERFECT SWIPES")
    print("="*70)

    # Test very simple words with straight paths
    test_words = ['a', 'aa', 'hi', 'the', 'cat']

    featurizer = PersonalizedSwipeFeaturizer()
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }

    for word in test_words:
        print(f"\nTesting synthetic swipe for: '{word}'")
        print("-"*40)

        # Create perfect swipe
        points = create_synthetic_swipe(word, key_layout)
        print(f"Created {len(points)} points")

        # Transform and process
        transformed = []
        for pt in points:
            transformed.append({
                'x': pt['x'] * 2.0 - 1.0,
                'y': pt['y'] * 2.0 - 1.0,
                't': pt['t']
            })

        target_len = determine_resample_target(len(transformed), preprocess_cfg)
        resampled = resample_points(transformed, target_len)
        features = featurizer(resampled)

        print(f"Features shape: {features.shape}")

        # Run inference
        encoder_input = features.T[np.newaxis, :, :]
        length = np.array([features.shape[0]], dtype=np.int64)

        encoder_outputs = encoder.run(
            None,
            {'audio_signal': encoder_input.astype(np.float32), 'length': length}
        )

        encoded = encoder_outputs[0]

        # Greedy decode
        decoded_tokens = []
        pred_layers = 1
        pred_hidden = 192
        state_h = np.zeros((pred_layers, 1, pred_hidden), dtype=np.float32)
        state_c = np.zeros((pred_layers, 1, pred_hidden), dtype=np.float32)
        last_token = 0
        blank_id = 29
        max_symbols = 10

        for t in range(encoded.shape[2]):
            encoder_frame = encoded[:, :, t:t+1]
            symbols_emitted = 0

            while symbols_emitted < 3 and len(decoded_tokens) < max_symbols:
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

                logits = decoder_outputs[0]
                state_h = decoder_outputs[2]
                state_c = decoder_outputs[3]

                if len(logits.shape) == 4:
                    probs = logits[0, 0, 0, :]
                else:
                    probs = logits[0, 0, :]

                # Get top 3 predictions
                top3_idx = np.argsort(probs)[-3:][::-1]
                top3_probs = probs[top3_idx]

                predicted_token = top3_idx[0]

                if predicted_token == blank_id:
                    break
                else:
                    decoded_tokens.append(predicted_token)
                    last_token = predicted_token
                    symbols_emitted += 1

        # Convert to text
        predicted = ''.join([meta['id_to_char'][str(t)] for t in decoded_tokens if str(t) in meta['id_to_char']])

        print(f"Expected: '{word}'")
        print(f"Predicted: '{predicted}'")
        print(f"Tokens: {decoded_tokens[:10]}")
        print(f"Match: {'✅' if predicted == word else '❌'}")

    print("\n" + "="*70)
    print("TESTING RAW LOGIT DISTRIBUTION")
    print("="*70)

    # Check what the model outputs for a single frame
    dummy_features = np.zeros((56, 37), dtype=np.float32)
    dummy_features[0, 0] = 0.5  # Set some non-zero value

    encoder_input = dummy_features.T[np.newaxis, :, :]
    length = np.array([dummy_features.shape[0]], dtype=np.int64)

    encoder_outputs = encoder.run(
        None,
        {'audio_signal': encoder_input.astype(np.float32), 'length': length}
    )

    encoded = encoder_outputs[0]
    encoder_frame = encoded[:, :, 0:1]

    state_h = np.zeros((1, 1, 192), dtype=np.float32)
    state_c = np.zeros((1, 1, 192), dtype=np.float32)

    decoder_outputs = decoder.run(
        None,
        {
            'encoder_outputs': encoder_frame,
            'targets': np.array([[0]], dtype=np.int32),
            'target_length': np.array([1], dtype=np.int32),
            'input_states_1': state_h,
            'input_states_2': state_c
        }
    )

    logits = decoder_outputs[0]
    if len(logits.shape) == 4:
        probs = logits[0, 0, 0, :]
    else:
        probs = logits[0, 0, :]

    # Apply softmax
    probs_exp = np.exp(probs - np.max(probs))
    probs_softmax = probs_exp / np.sum(probs_exp)

    print("\nLogit statistics for dummy input:")
    print(f"Shape: {probs.shape}")
    print(f"Min logit: {np.min(probs):.3f}")
    print(f"Max logit: {np.max(probs):.3f}")
    print(f"Mean logit: {np.mean(probs):.3f}")
    print(f"Std logit: {np.std(probs):.3f}")

    print("\nTop 5 predictions (with softmax):")
    top5_idx = np.argsort(probs_softmax)[-5:][::-1]
    for idx in top5_idx:
        char = meta['id_to_char'].get(str(idx), f"?{idx}")
        prob = probs_softmax[idx]
        print(f"  {idx:2d}: '{char}' = {prob:.4f}")


if __name__ == '__main__':
    test_model_basic()