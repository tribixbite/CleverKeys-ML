#!/usr/bin/env python3
"""
Test new ONNX models specifically with 'companion' word
"""

import json
import numpy as np
import onnxruntime as ort
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'new'))
from train_transducer_personalized import PersonalizedSwipeFeaturizer, determine_resample_target, resample_points


def get_companion_data():
    """Get companion swipe data from line 22440"""
    data_path = '../../data/train_final_train.jsonl'
    with open(data_path, 'r') as f:
        for i, line in enumerate(f, 1):
            if i == 22440:
                data = json.loads(line)
                return data['points']
    return None


def process_swipe(points, featurizer, preprocess_cfg):
    """Process swipe points through full pipeline"""
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

    return features, len(transformed_points), target_len


def test_inference(features, encoder, decoder, meta):
    """Test inference with better debugging"""
    # Prepare encoder input
    encoder_input = features.T[np.newaxis, :, :]
    length = np.array([features.shape[0]], dtype=np.int64)

    print("\n--- Running Encoder ---")
    print(f"Encoder input shape: {encoder_input.shape}")

    # Run encoder
    encoder_outputs = encoder.run(
        None,
        {'audio_signal': encoder_input.astype(np.float32), 'length': length}
    )

    encoded = encoder_outputs[0]
    print(f"Encoded shape: {encoded.shape}")

    # Check decoder input/output names
    print("\n--- Decoder Info ---")
    print(f"Decoder inputs: {[i.name for i in decoder.get_inputs()]}")
    print(f"Decoder outputs: {[o.name for o in decoder.get_outputs()]}")

    # Get actual vocab size from decoder output shape
    dummy_frame = encoded[:, :, 0:1]
    # Based on IMPLEMENTATION_GUIDE.md: mobile preset has 1 layer, 192 hidden
    state_h = np.zeros((1, 1, 192), dtype=np.float32)
    state_c = np.zeros((1, 1, 192), dtype=np.float32)

    try:
        test_outputs = decoder.run(
            None,
            {
                'encoder_outputs': dummy_frame,
                'targets': np.array([[0]], dtype=np.int32),
                'target_length': np.array([1], dtype=np.int32),
                'input_states_1': state_h,
                'input_states_2': state_c
            }
        )
        logits = test_outputs[0]
        print(f"Logits shape: {logits.shape}")
        actual_vocab_size = logits.shape[-1]
        print(f"Actual vocab size from model: {actual_vocab_size}")
    except Exception as e:
        print(f"Error testing decoder: {e}")
        return []

    # Now do actual greedy decoding
    print("\n--- Greedy Decoding ---")
    blank_id = actual_vocab_size - 1  # Use last index as blank
    print(f"Using blank_id: {blank_id}")

    # Use same state dimensions as test
    state_h = np.zeros((1, 1, 192), dtype=np.float32)
    state_c = np.zeros((1, 1, 192), dtype=np.float32)

    decoded_tokens = []
    last_token = 0
    max_symbols = 24

    for t in range(encoded.shape[2]):
        encoder_frame = encoded[:, :, t:t+1]
        symbols_emitted = 0
        max_symbols_per_frame = 6

        while symbols_emitted < max_symbols_per_frame and len(decoded_tokens) < max_symbols:
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

            # Get prediction
            if len(logits.shape) == 4:
                probs = logits[0, 0, 0, :]
            else:
                probs = logits[0, 0, :]

            predicted_token = np.argmax(probs)

            if predicted_token == blank_id:
                break
            else:
                decoded_tokens.append(predicted_token)
                last_token = predicted_token
                symbols_emitted += 1

    return decoded_tokens


def main():
    print("="*70)
    print("TESTING 'COMPANION' WITH NEW ONNX MODELS")
    print("="*70)

    # Get companion data
    print("\nLoading companion swipe from dataset...")
    points = get_companion_data()
    if points is None:
        print("ERROR: Could not load companion data")
        return

    print(f"Loaded {len(points)} points for 'companion'")

    # Check coordinate ranges
    x_vals = [p['x'] for p in points]
    y_vals = [p['y'] for p in points]
    print(f"X range: [{min(x_vals):.3f}, {max(x_vals):.3f}]")
    print(f"Y range: [{min(y_vals):.3f}, {max(y_vals):.3f}]")

    # Initialize featurizer
    featurizer = PersonalizedSwipeFeaturizer()

    # Preprocessing config
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }

    # Process swipe
    features, orig_len, resample_len = process_swipe(points, featurizer, preprocess_cfg)
    print(f"\nProcessed: {orig_len} → {resample_len} points")
    print(f"Features shape: {features.shape}")
    print(f"First 5 features: {features[0, :5]}")

    # Load ONNX models - try auto_best which was created today
    model_dir = '../models/auto_best'
    print(f"\nLoading models from: {model_dir}")
    encoder = ort.InferenceSession(os.path.join(model_dir, 'encoder.onnx'))
    decoder = ort.InferenceSession(os.path.join(model_dir, 'decoder_joint.onnx'))

    # Load runtime meta
    with open(os.path.join(model_dir, 'runtime_meta.json'), 'r') as f:
        meta = json.load(f)

    # Run inference
    tokens = test_inference(features, encoder, decoder, meta)

    # Convert to text
    predicted = ''.join([meta['id_to_char'][str(t)] for t in tokens if str(t) in meta['id_to_char']])

    print(f"\nPredicted tokens: {tokens}")
    print(f"Predicted text: '{predicted}'")

    if predicted == 'companion':
        print("\n✅ SUCCESS: 'companion' correctly predicted!")
    else:
        print(f"\n❌ FAILED: Expected 'companion', got '{predicted}'")

        # Show what the expected tokens should be
        expected_tokens = []
        for char in 'companion':
            if char in meta['char_to_id']:
                expected_tokens.append(meta['char_to_id'][char])
        print(f"Expected tokens would be: {expected_tokens}")


if __name__ == '__main__':
    main()