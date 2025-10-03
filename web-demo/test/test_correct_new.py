#!/usr/bin/env python3
"""
Test ONNX models using the CORRECT architecture from new/train_transducer_personalized.py
"""

import json
import numpy as np
import onnxruntime as ort
import sys
import os
from pathlib import Path

# Add the correct path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'new'))

# Import from the CORRECT new/train_transducer_personalized.py
from train_transducer_personalized import (
    PersonalizedSwipeFeaturizer,
    resample_points,
    determine_resample_target,
    clamp,
    load_key_centers
)


def get_companion_data():
    """Get companion swipe data from line 22440"""
    import linecache
    data_path = '../../data/train_final_train.jsonl'
    line = linecache.getline(data_path, 22440)
    if line:
        data = json.loads(line)
        return data['points'], data['word']
    return None, None


def prepare_points(points):
    """Prepare points EXACTLY as training does (from new/train_transducer_personalized.py)"""
    prepared = []
    for idx, pt in enumerate(points):
        raw_x = float(pt.get("x", 0.0))
        raw_y = float(pt.get("y", 0.0))

        # Transform coordinates from [0, 1] to [-1, 1]
        # Dataset has (0,0) at top-left Q key, we need (0,0) at keyboard center
        centered_x = raw_x * 2.0 - 1.0
        centered_y = raw_y * 2.0 - 1.0

        # Allow for out-of-bounds gestures but cap at reasonable limits
        centered_x = clamp(centered_x, -1.5, 1.5)
        centered_y = clamp(centered_y, -1.5, 1.5)

        raw_t = float(pt.get("t", idx * 10.0))
        prepared.append({
            "x": centered_x,
            "y": centered_y,
            "t": raw_t,
        })

    return prepared


def test_with_correct_new():
    """Test using the correct new/train_transducer_personalized.py architecture"""
    print("="*70)
    print("TESTING WITH CORRECT NEW ARCHITECTURE")
    print("="*70)

    # Load models - use better_trained (epoch 156, WER 0.176)
    model_dir = '../models/better_trained'
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

    # Process swipe EXACTLY as new/train_transducer_personalized.py does

    # 1. Prepare points (coordinate transformation)
    prepared_points = prepare_points(points)
    print(f"Original first point: x={points[0]['x']:.4f}, y={points[0]['y']:.4f}")
    print(f"Prepared first point: x={prepared_points[0]['x']:.4f}, y={prepared_points[0]['y']:.4f}")

    # 2. Determine resample target
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }
    target_len = determine_resample_target(len(prepared_points), preprocess_cfg)
    print(f"Original length: {len(prepared_points)}, Target length: {target_len}")

    # 3. Resample points
    resampled = resample_points(prepared_points, target_len)
    print(f"Resampled to {len(resampled)} points")

    # 4. Extract features using the actual PersonalizedSwipeFeaturizer
    featurizer = PersonalizedSwipeFeaturizer(mobile_features=False)
    features = featurizer(resampled)

    print(f"Features shape: {features.shape}")
    print(f"Feature ranges - min: {features.min():.4f}, max: {features.max():.4f}")
    print(f"First 5 values of first frame: {features[0, :5]}")

    # Pad features to 37 dimensions if needed
    if features.shape[1] < 37:
        padding = np.zeros((features.shape[0], 37 - features.shape[1]), dtype=np.float32)
        features = np.concatenate([features, padding], axis=1)
        print(f"Padded features to shape: {features.shape}")

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
    encoded_len = encoder_outputs[1][0]  # Extract scalar

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
    total_symbols = 0
    max_total_symbols = 24

    for t in range(encoded_len):
        if total_symbols >= max_total_symbols:
            break

        enc_frame = encoded[:, :, t:t+1]

        symbols_this_frame = 0
        while symbols_this_frame < max_symbols_per_frame and total_symbols < max_total_symbols:
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
            if len(logits_raw.shape) == 4:
                logits = logits_raw[0, 0, 0, :]
            else:
                logits = logits_raw.flatten()

            state_h = decoder_outputs[2]
            state_c = decoder_outputs[3]

            # Get prediction
            pred_idx = int(np.argmax(logits))

            # Debug first few frames
            if t < 3 and symbols_this_frame == 0:
                top5_idx = np.argsort(logits)[-5:].tolist()[::-1]
                top5_chars = [vocab[i] if i < len(vocab) else '?' for i in top5_idx]
                top5_scores = [float(logits[i]) for i in top5_idx]
                print(f"  Frame {t}: Top 5 = {list(zip(top5_chars, top5_scores[:3]))}")

            if pred_idx == blank_id:
                # Emit blank and move to next frame
                break
            else:
                # Emit character
                predictions.append(pred_idx)
                symbols_this_frame += 1
                total_symbols += 1

                # Map from joint vocab to predictor vocab for next input
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
        print("\n❌ FAILED")
        if pred_text:
            # Check character-level accuracy
            correct = sum(1 for c1, c2 in zip(pred_text, expected_word) if c1 == c2)
            print(f"Character accuracy: {correct}/{len(expected_word)} = {correct/len(expected_word)*100:.1f}%")


def main():
    try:
        test_with_correct_new()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()