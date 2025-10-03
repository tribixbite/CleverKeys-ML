#!/usr/bin/env python3
"""
Test ONNX models by exactly mimicking the training pipeline from train_transducer_personalized.py
"""

import json
import numpy as np
import onnxruntime as ort
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'new'))

from train_transducer_personalized import PersonalizedSwipeFeaturizer, determine_resample_target, resample_points


def get_companion_data():
    """Get companion swipe data from line 22440 efficiently"""
    import linecache
    data_path = '../../data/train_final_train.jsonl'
    line = linecache.getline(data_path, 22440)
    if line:
        data = json.loads(line)
        return data['points'], data['word']
    return None, None


def process_swipe_exact_training(points, featurizer, preprocess_cfg):
    """Process swipe points EXACTLY as in training"""

    # 1. Transform coordinates from [0, 1] to [-1, 1] - EXACTLY as training does
    transformed_points = []
    for pt in points:
        transformed_points.append({
            'x': pt['x'] * 2.0 - 1.0,
            'y': pt['y'] * 2.0 - 1.0,
            't': pt['t']
        })

    print(f"Original first point: x={points[0]['x']:.4f}, y={points[0]['y']:.4f}")
    print(f"Transformed first point: x={transformed_points[0]['x']:.4f}, y={transformed_points[0]['y']:.4f}")

    # 2. Determine resample target - EXACTLY as training does
    target_len = determine_resample_target(len(transformed_points), preprocess_cfg)
    print(f"Original length: {len(transformed_points)}, Target length: {target_len}")

    # 3. Resample points - EXACTLY as training does
    resampled_points = resample_points(transformed_points, target_len)
    print(f"Resampled to {len(resampled_points)} points")

    # 4. Extract features - EXACTLY as training does
    features = featurizer(resampled_points)
    print(f"Features shape: {features.shape}")
    print(f"Feature ranges - min: {features.min():.4f}, max: {features.max():.4f}")
    print(f"First 5 features of first frame: {features[0, :5]}")

    return features


def test_with_onnx():
    """Test using ONNX models with exact training pipeline"""
    print("="*70)
    print("TESTING WITH ONNX (EXACT TRAINING PIPELINE)")
    print("="*70)

    # Load models
    model_dir = '../models/best_latest'
    encoder_path = os.path.join(model_dir, 'encoder.onnx')
    decoder_path = os.path.join(model_dir, 'decoder_joint.onnx')
    meta_path = os.path.join(model_dir, 'runtime_meta.json')

    print(f"\nLoading models from: {model_dir}")
    encoder_session = ort.InferenceSession(encoder_path)
    decoder_session = ort.InferenceSession(decoder_path)

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    blank_id = meta['blank_id']
    vocab = meta['tokens']
    decoder_config = meta.get('decoder_config', {})
    num_layers = decoder_config.get('num_layers', 1)
    hidden_size = decoder_config.get('hidden_size', 192)

    print(f"Vocab size: {len(vocab)}, Blank ID: {blank_id}")
    print(f"Decoder config: {num_layers} layers, {hidden_size} hidden")

    # Get companion data
    points, expected_word = get_companion_data()
    if points is None:
        print("ERROR: Could not load companion data")
        return

    print(f"\nTesting word: '{expected_word}' ({len(points)} points)")

    # Process swipe EXACTLY as training does
    featurizer = PersonalizedSwipeFeaturizer()
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }

    features = process_swipe_exact_training(points, featurizer, preprocess_cfg)

    # Run encoder
    print("\n--- Running Encoder ---")
    signal = features.astype(np.float32).reshape(1, -1, 37)
    signal_len = np.array([features.shape[0]], dtype=np.int64)

    encoder_outputs = encoder_session.run(None, {
        'audio_signal': signal,
        'length': signal_len
    })
    encoded = encoder_outputs[0]
    encoded_len = encoder_outputs[1]

    print(f"Encoded shape: {encoded.shape}")
    print(f"Encoded length: {encoded_len}")
    print(f"Encoded stats - min: {encoded.min():.4f}, max: {encoded.max():.4f}, mean: {encoded.mean():.4f}")

    # Run greedy decoding
    print("\n--- Running Greedy Decoding ---")

    # Initialize decoder states
    state_h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    state_c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)

    # Start with blank/BOS
    y = np.array([[0]], dtype=np.int64)  # BOS token

    predictions = []
    max_symbols = 24

    for t in range(encoded_len[0]):
        enc_frame = encoded[:, t:t+1, :]

        for _ in range(max_symbols):
            # Run decoder
            decoder_outputs = decoder_session.run(None, {
                'targets': y,
                'states_0': state_h,
                'states_1': state_c,
                'encoder_outputs': enc_frame,
                'target_length': np.array([1], dtype=np.int64),
                'encoder_outputs_length': np.array([1], dtype=np.int64)
            })

            # Get logits and next states
            logits = decoder_outputs[0][0, 0, :]  # Shape: [vocab_size]
            state_h = decoder_outputs[1]
            state_c = decoder_outputs[2]

            # Get prediction
            pred_idx = np.argmax(logits)

            if pred_idx == blank_id:
                # Emit blank and move to next frame
                break
            else:
                # Emit character
                predictions.append(pred_idx)
                y = np.array([[pred_idx]], dtype=np.int64)

    # Convert predictions to text
    pred_text = ''.join([vocab[idx] if idx < len(vocab) else '?' for idx in predictions])

    print(f"\nPredictions: {predictions}")
    print(f"Predicted text: '{pred_text}'")
    print(f"Expected: '{expected_word}'")

    if pred_text == expected_word:
        print("\n✅ SUCCESS!")
    else:
        print("\n❌ FAILED")

        # Try alternative decoding - take top non-blank at each frame
        print("\n--- Alternative: Top Non-Blank Per Frame ---")
        alt_predictions = []

        # Reset states
        state_h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
        state_c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
        y = np.array([[0]], dtype=np.int64)

        for t in range(encoded_len[0]):
            enc_frame = encoded[:, t:t+1, :]

            # Run decoder once per frame
            decoder_outputs = decoder_session.run(None, {
                'targets': y,
                'states_0': state_h,
                'states_1': state_c,
                'encoder_outputs': enc_frame,
                'target_length': np.array([1], dtype=np.int64),
                'encoder_outputs_length': np.array([1], dtype=np.int64)
            })

            logits = decoder_outputs[0][0, 0, :]
            state_h = decoder_outputs[1]
            state_c = decoder_outputs[2]

            # Get top 2 predictions
            top2 = np.argsort(logits)[-2:][::-1]

            # Take top non-blank
            for idx in top2:
                if idx != blank_id:
                    alt_predictions.append(idx)
                    y = np.array([[idx]], dtype=np.int64)
                    break

        alt_text = ''.join([vocab[idx] if idx < len(vocab) else '?' for idx in alt_predictions])
        print(f"Alternative predictions: {alt_predictions}")
        print(f"Alternative text: '{alt_text}'")


def main():
    try:
        test_with_onnx()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()