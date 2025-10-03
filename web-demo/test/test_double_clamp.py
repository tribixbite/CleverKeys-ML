#!/usr/bin/env python3
"""
Test if the double clamping is causing issues
Training does: [0,1] -> [-1,1] -> clamp[-1.5,1.5] -> featurizer clamps[-1,1]
"""

import json
import numpy as np
import onnxruntime as ort
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'new'))

from train_transducer_personalized import (
    PersonalizedSwipeFeaturizer,
    resample_points,
    determine_resample_target,
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


def test_exact_training_pipeline():
    """Test EXACTLY what training does"""
    print("Testing EXACT training pipeline")
    print("="*60)

    # Get hello data
    points, word = get_hello_data()
    print(f"Testing: '{word}' with {len(points)} points")

    # Step 1: _prepare_points (from training)
    print("\nStep 1: _prepare_points")
    start_t = float(points[0].get("t", 0.0))
    prepared = []
    for idx, pt in enumerate(points):
        raw_x = float(pt.get("x", 0.0))
        raw_y = float(pt.get("y", 0.0))

        # Training does this:
        centered_x = raw_x * 2.0 - 1.0
        centered_y = raw_y * 2.0 - 1.0
        centered_x = clamp(centered_x, -1.5, 1.5)  # Clamp to [-1.5, 1.5]
        centered_y = clamp(centered_y, -1.5, 1.5)

        raw_t = float(pt.get("t", idx * 10.0))
        prepared.append({
            "x": centered_x,
            "y": centered_y,
            "t": raw_t,
        })

    xs = [p['x'] for p in prepared]
    ys = [p['y'] for p in prepared]
    print(f"  After transform: X∈[{min(xs):.3f}, {max(xs):.3f}], Y∈[{min(ys):.3f}, {max(ys):.3f}]")

    # Step 2: Resample
    print("\nStep 2: Resample")
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }
    target_len = determine_resample_target(len(prepared), preprocess_cfg)
    print(f"  Target length: {target_len}")
    resampled = resample_points(prepared, target_len)

    # Step 3: Feature extraction (featurizer will clamp to [-1, 1])
    print("\nStep 3: Feature extraction")
    featurizer = PersonalizedSwipeFeaturizer(mobile_features=False)

    # Let's see what happens to coordinates
    print(f"  Featurizer will clamp each x,y to [-1, 1]")

    # Count how many points get clamped
    clamped_count = 0
    for p in resampled:
        if abs(p['x']) > 1.0 or abs(p['y']) > 1.0:
            clamped_count += 1

    print(f"  Points that will be clamped by featurizer: {clamped_count}/{len(resampled)}")

    features = featurizer(resampled)
    print(f"  Features shape: {features.shape}")

    # Check the actual x,y values in features (first two columns)
    feature_xs = features[:, 0]
    feature_ys = features[:, 1]
    print(f"  Feature X∈[{feature_xs.min():.3f}, {feature_xs.max():.3f}]")
    print(f"  Feature Y∈[{feature_ys.min():.3f}, {feature_ys.max():.3f}]")

    # Pad to 37
    if features.shape[1] < 37:
        padding = np.zeros((features.shape[0], 37 - features.shape[1]), dtype=np.float32)
        features = np.concatenate([features, padding], axis=1)

    # Now run inference
    print("\nStep 4: Inference")
    model_dir = '../models/correct_9292025'

    encoder_session = ort.InferenceSession(os.path.join(model_dir, 'encoder.onnx'))
    decoder_session = ort.InferenceSession(os.path.join(model_dir, 'decoder_joint.onnx'))

    with open(os.path.join(model_dir, 'runtime_meta.json'), 'r') as f:
        meta = json.load(f)

    blank_id = meta['blank_id']
    vocab = meta['tokens']
    num_layers = meta['decoder_config']['num_layers']
    hidden_size = meta['decoder_config']['hidden_size']
    joint2pred = meta['predictor']['label_map']['joint2pred']

    # Run encoder
    signal = features.astype(np.float32).T.reshape(1, 37, -1)
    signal_len = np.array([features.shape[0]], dtype=np.int64)

    encoder_outputs = encoder_session.run(None, {
        'audio_signal': signal,
        'length': signal_len
    })
    encoded = encoder_outputs[0]
    encoded_len = encoder_outputs[1][0]

    # Greedy decode
    state_h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    state_c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    y = np.array([[0]], dtype=np.int32)

    predictions = []
    chars_per_frame = []

    for t in range(encoded_len):
        enc_frame = encoded[:, :, t:t+1]
        frame_chars = []

        for _ in range(8):
            decoder_outputs = decoder_session.run(None, {
                'targets': y,
                'input_states_1': state_h,
                'input_states_2': state_c,
                'encoder_outputs': enc_frame,
                'target_length': np.array([1], dtype=np.int32)
            })

            logits = decoder_outputs[0]
            if len(logits.shape) == 4:
                logits = logits[0, 0, 0, :]
            else:
                logits = logits.flatten()

            state_h = decoder_outputs[2]
            state_c = decoder_outputs[3]

            joint_pred_idx = int(np.argmax(logits))

            if joint_pred_idx == blank_id:
                break
            else:
                char = vocab[joint_pred_idx] if joint_pred_idx < len(vocab) else '?'
                frame_chars.append(char)
                predictions.append(joint_pred_idx)

                pred_idx = joint2pred[joint_pred_idx]
                if pred_idx == -1:
                    pred_idx = 0
                y = np.array([[pred_idx]], dtype=np.int32)

        chars_per_frame.append(frame_chars)

        if len(predictions) >= 50:
            break

    pred_text = ''.join([vocab[idx] if idx < len(vocab) else '?' for idx in predictions])

    print(f"  Encoded frames: {encoded_len}")
    print(f"  Frames with output: {sum(1 for f in chars_per_frame if f)}")
    print(f"  Predicted: '{pred_text}'")
    print(f"  Expected:  '{word}'")

    # Try without any clamping in prepare_points
    print("\n" + "="*60)
    print("Testing WITHOUT clamping in prepare_points")

    prepared_unclamped = []
    for idx, pt in enumerate(points):
        raw_x = float(pt.get("x", 0.0))
        raw_y = float(pt.get("y", 0.0))

        # Transform but DON'T clamp
        centered_x = raw_x * 2.0 - 1.0
        centered_y = raw_y * 2.0 - 1.0
        # NO CLAMPING HERE

        raw_t = float(pt.get("t", idx * 10.0))
        prepared_unclamped.append({
            "x": centered_x,
            "y": centered_y,
            "t": raw_t,
        })

    resampled_unclamped = resample_points(prepared_unclamped, target_len)
    features_unclamped = featurizer(resampled_unclamped)

    if features_unclamped.shape[1] < 37:
        padding = np.zeros((features_unclamped.shape[0], 37 - features_unclamped.shape[1]), dtype=np.float32)
        features_unclamped = np.concatenate([features_unclamped, padding], axis=1)

    # Quick test
    signal_unclamped = features_unclamped.astype(np.float32).T.reshape(1, 37, -1)
    signal_len_unclamped = np.array([features_unclamped.shape[0]], dtype=np.int64)

    encoder_outputs_unclamped = encoder_session.run(None, {
        'audio_signal': signal_unclamped,
        'length': signal_len_unclamped
    })
    encoded_unclamped = encoder_outputs_unclamped[0]

    # Check if encoder outputs are different
    if np.allclose(encoded, encoded_unclamped, rtol=1e-5):
        print("  Encoder outputs are identical (clamping made no difference)")
    else:
        diff = np.abs(encoded - encoded_unclamped).mean()
        print(f"  Encoder outputs differ by {diff:.6f} on average")


if __name__ == '__main__':
    test_exact_training_pipeline()