#!/usr/bin/env python3
"""
Test ONNX models using the CORRECT architecture from trained_models/nema1
"""

import json
import numpy as np
import onnxruntime as ort
import sys
import os
from pathlib import Path

# Import from the CORRECT location
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'trained_models', 'nema1'))

from train_transducer_personalized import (
    PersonalizedSwipeFeaturizer,
    resample_points,
    determine_resample_target
)


def get_companion_data():
    """Get companion swipe data from line 22440 efficiently"""
    import linecache
    data_path = '../../data/train_final_train.jsonl'
    line = linecache.getline(data_path, 22440)
    if line:
        data = json.loads(line)
        return data['points'], data['word']
    return None, None


def clamp(x, min_val, max_val):
    return max(min_val, min(max_val, x))


def process_swipe_correctly(points):
    """Process swipe EXACTLY as training does"""

    # 1. Transform coordinates from [0, 1] to [-1, 1] - EXACTLY as training does
    start_t = float(points[0].get("t", 0.0))
    normalized = []
    for idx, pt in enumerate(points):
        raw_x = float(pt.get("x", 0.5))
        raw_y = float(pt.get("y", 0.5))
        centered_x = clamp(raw_x * 2.0 - 1.0, -1.0, 1.0)
        centered_y = clamp(raw_y * 2.0 - 1.0, -1.0, 1.0)
        raw_t = float(pt.get("t", idx * 10.0))
        normalized.append({
            "x": centered_x,
            "y": centered_y,
            "t": max(0.0, raw_t - start_t),
        })

    print(f"Original first point: x={points[0]['x']:.4f}, y={points[0]['y']:.4f}")
    print(f"Normalized first point: x={normalized[0]['x']:.4f}, y={normalized[0]['y']:.4f}")

    # 2. Determine resample target
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }
    target_len = determine_resample_target(len(normalized), preprocess_cfg)
    print(f"Original length: {len(normalized)}, Target length: {target_len}")

    # 3. Resample points
    resampled = resample_points(normalized, target_len)
    print(f"Resampled to {len(resampled)} points")

    # 4. Extract features using the ACTUAL featurizer from training
    featurizer = PersonalizedSwipeFeaturizer()
    features = featurizer(resampled)

    return features


def test_with_correct_architecture():
    """Test using ONNX models with correct architecture"""
    print("="*70)
    print("TESTING WITH CORRECT ARCHITECTURE (trained_models/nema1)")
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

    # Process swipe with correct architecture
    features = process_swipe_correctly(points)
    print(f"Features shape: {features.shape}")
    print(f"Feature ranges - min: {features.min():.4f}, max: {features.max():.4f}")
    print(f"First 5 features of first frame: {features[0, :5]}")

    # Run encoder
    print("\n--- Running Encoder ---")
    # Encoder expects [batch, features, time]
    signal = features.astype(np.float32).T.reshape(1, 37, -1)
    signal_len = np.array([features.shape[0]], dtype=np.int64)

    encoder_outputs = encoder_session.run(None, {
        'audio_signal': signal,
        'length': signal_len
    })
    encoded = encoder_outputs[0]
    encoded_len = encoder_outputs[1][0]  # Extract scalar from array

    print(f"Encoded shape: {encoded.shape}")
    print(f"Encoded length: {encoded_len}")
    print(f"Encoded stats - min: {encoded.min():.4f}, max: {encoded.max():.4f}, mean: {encoded.mean():.4f}")

    # Run greedy decoding
    print("\n--- Running Greedy Decoding ---")

    # Initialize decoder states
    state_h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    state_c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)

    # Start with BOS (0 in predictor space)
    y = np.array([[0]], dtype=np.int32)

    predictions = []
    max_symbols_per_frame = 6

    for t in range(encoded_len):  # Process all frames
        enc_frame = encoded[:, :, t:t+1]

        symbols_this_frame = 0
        while symbols_this_frame < max_symbols_per_frame:
            # Run decoder
            decoder_outputs = decoder_session.run(None, {
                'targets': y,
                'input_states_1': state_h,
                'input_states_2': state_c,
                'encoder_outputs': enc_frame,
                'target_length': np.array([1], dtype=np.int32)
            })

            # Get logits and next states
            logits_raw = decoder_outputs[0]
            # Extract the actual logits vector
            if len(logits_raw.shape) == 4:
                logits = logits_raw[0, 0, 0, :]  # [batch, 1, 1, vocab_size]
            else:
                logits = logits_raw.flatten()

            state_h = decoder_outputs[2]
            state_c = decoder_outputs[3]

            # Get prediction
            pred_idx = int(np.argmax(logits))

            # Debug: show top predictions for first 2 frames
            if t < 2 and symbols_this_frame == 0:
                # Get the actual indices as integers
                sorted_indices = np.argsort(logits)
                top5_idx = sorted_indices[-5:].tolist()[::-1]

                print(f"Frame {t}: Top 5:")
                for rank, idx in enumerate(top5_idx):
                    char = vocab[idx] if idx < len(vocab) else '?'
                    score = logits[idx]
                    print(f"  {rank+1}. '{char}' (idx={idx}, score={score:.3f})")

            if pred_idx == blank_id:
                # Emit blank and move to next frame
                break
            else:
                # Emit character
                predictions.append(int(pred_idx))
                symbols_this_frame += 1

                # Map from joint vocab to predictor vocab for next input
                # The predictor uses blankless labels
                if pred_idx < blank_id:
                    next_y = pred_idx
                else:
                    next_y = pred_idx - 1
                y = np.array([[next_y]], dtype=np.int32)

    # Convert predictions to text
    pred_text = ''.join([vocab[idx] if idx < len(vocab) else '?' for idx in predictions])

    print(f"\nPredictions (token IDs): {predictions}")
    print(f"Predicted text: '{pred_text}'")
    print(f"Expected: '{expected_word}'")

    if pred_text == expected_word:
        print("\n✅ SUCCESS! Model correctly predicted the word!")
    else:
        print("\n❌ FAILED - but let's check if it's close")

        # Check character-level accuracy
        correct = sum(1 for c1, c2 in zip(pred_text, expected_word) if c1 == c2)
        print(f"Character accuracy: {correct}/{len(expected_word)} = {correct/len(expected_word)*100:.1f}%")


def main():
    try:
        test_with_correct_architecture()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()