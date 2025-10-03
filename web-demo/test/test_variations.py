#!/usr/bin/env python3
"""
Test multiple variations of preprocessing to find what works
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


def prepare_points_transformed(points):
    """Transform [0,1] to [-1,1] as training does"""
    prepared = []
    for idx, pt in enumerate(points):
        raw_x = float(pt.get("x", 0.0))
        raw_y = float(pt.get("y", 0.0))
        centered_x = raw_x * 2.0 - 1.0
        centered_y = raw_y * 2.0 - 1.0
        centered_x = clamp(centered_x, -1.5, 1.5)
        centered_y = clamp(centered_y, -1.5, 1.5)
        raw_t = float(pt.get("t", idx * 10.0))
        prepared.append({
            "x": centered_x,
            "y": centered_y,
            "t": raw_t,
        })
    return prepared


def prepare_points_untransformed(points):
    """Keep coordinates in [0,1] - NO transformation"""
    prepared = []
    for idx, pt in enumerate(points):
        raw_x = float(pt.get("x", 0.0))
        raw_y = float(pt.get("y", 0.0))
        raw_t = float(pt.get("t", idx * 10.0))
        prepared.append({
            "x": raw_x,
            "y": raw_y,
            "t": raw_t,
        })
    return prepared


def prepare_points_wrong_transform(points):
    """Try a different transform - maybe training had a bug?"""
    prepared = []
    for idx, pt in enumerate(points):
        raw_x = float(pt.get("x", 0.0))
        raw_y = float(pt.get("y", 0.0))
        # Try assuming data is already in [-1,1] and clamp it
        centered_x = clamp(raw_x, -1.0, 1.0)
        centered_y = clamp(raw_y, -1.0, 1.0)
        raw_t = float(pt.get("t", idx * 10.0))
        prepared.append({
            "x": centered_x,
            "y": centered_y,
            "t": raw_t,
        })
    return prepared


def test_variation(points, word, prepare_func, variation_name, model_dir='../models/correct_9292025'):
    """Test a specific variation of preprocessing"""

    # Load models
    encoder_session = ort.InferenceSession(os.path.join(model_dir, 'encoder.onnx'))
    decoder_session = ort.InferenceSession(os.path.join(model_dir, 'decoder_joint.onnx'))

    with open(os.path.join(model_dir, 'runtime_meta.json'), 'r') as f:
        meta = json.load(f)

    blank_id = meta['blank_id']
    vocab = meta['tokens']
    num_layers = meta['decoder_config']['num_layers']
    hidden_size = meta['decoder_config']['hidden_size']
    joint2pred = meta['predictor']['label_map']['joint2pred']

    # Process swipe with given preparation function
    prepared = prepare_func(points)

    # Show coordinate range
    xs = [p['x'] for p in prepared]
    ys = [p['y'] for p in prepared]
    print(f"\n{variation_name}:")
    print(f"  X range: [{min(xs):.3f}, {max(xs):.3f}]")
    print(f"  Y range: [{min(ys):.3f}, {max(ys):.3f}]")

    # Try different resample targets
    resample_targets = [56, 70, 82, 96]

    for target_len in resample_targets:
        # Resample
        resampled = resample_points(prepared, target_len)

        # Extract features
        featurizer = PersonalizedSwipeFeaturizer(mobile_features=False)
        features = featurizer(resampled)

        # Pad to 37 dims
        if features.shape[1] < 37:
            padding = np.zeros((features.shape[0], 37 - features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, padding], axis=1)

        # Run encoder
        signal = features.astype(np.float32).T.reshape(1, 37, -1)
        signal_len = np.array([features.shape[0]], dtype=np.int64)

        encoder_outputs = encoder_session.run(None, {
            'audio_signal': signal,
            'length': signal_len
        })
        encoded = encoder_outputs[0]
        encoded_len = encoder_outputs[1][0]

        # Greedy decode through ALL frames
        state_h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
        state_c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
        y = np.array([[0]], dtype=np.int32)

        predictions = []

        for t in range(encoded_len):
            enc_frame = encoded[:, :, t:t+1]

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
                    predictions.append(joint_pred_idx)
                    pred_idx = joint2pred[joint_pred_idx]
                    if pred_idx == -1:
                        pred_idx = 0
                    y = np.array([[pred_idx]], dtype=np.int32)

            if len(predictions) >= 50:
                break

        pred_text = ''.join([vocab[idx] if idx < len(vocab) else '?' for idx in predictions])
        print(f"  Resample {target_len:2d}: '{pred_text}'")

    return predictions


def test_no_resampling(points, word, model_dir='../models/correct_9292025'):
    """Test without resampling - use raw points"""

    # Load models
    encoder_session = ort.InferenceSession(os.path.join(model_dir, 'encoder.onnx'))
    decoder_session = ort.InferenceSession(os.path.join(model_dir, 'decoder_joint.onnx'))

    with open(os.path.join(model_dir, 'runtime_meta.json'), 'r') as f:
        meta = json.load(f)

    blank_id = meta['blank_id']
    vocab = meta['tokens']
    num_layers = meta['decoder_config']['num_layers']
    hidden_size = meta['decoder_config']['hidden_size']
    joint2pred = meta['predictor']['label_map']['joint2pred']

    print(f"\nNo resampling (raw {len(points)} points):")

    # Try with and without transformation
    for transform_name, transform_func in [
        ("transformed [-1,1]", prepare_points_transformed),
        ("untransformed [0,1]", prepare_points_untransformed)
    ]:
        prepared = transform_func(points)

        # Extract features WITHOUT resampling
        featurizer = PersonalizedSwipeFeaturizer(mobile_features=False)
        features = featurizer(prepared)

        # Pad to 37 dims
        if features.shape[1] < 37:
            padding = np.zeros((features.shape[0], 37 - features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, padding], axis=1)

        # Truncate if too long
        if features.shape[0] > 200:
            features = features[:200, :]

        # Run encoder
        signal = features.astype(np.float32).T.reshape(1, 37, -1)
        signal_len = np.array([features.shape[0]], dtype=np.int64)

        encoder_outputs = encoder_session.run(None, {
            'audio_signal': signal,
            'length': signal_len
        })
        encoded = encoder_outputs[0]
        encoded_len = encoder_outputs[1][0]

        # Decode
        state_h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
        state_c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
        y = np.array([[0]], dtype=np.int32)

        predictions = []

        for t in range(min(encoded_len, 100)):  # Limit frames
            enc_frame = encoded[:, :, t:t+1]

            for _ in range(8):
                decoder_outputs = decoder_session.run(None, {
                    'targets': y,
                    'input_states_1': state_h,
                    'input_states_2': state_c,
                    'encoder_outputs': enc_frame,
                    'target_length': np.array([1], dtype=np.int32)
                })

                logits = decoder_outputs[0].flatten() if len(decoder_outputs[0].shape) > 2 else decoder_outputs[0].flatten()
                state_h = decoder_outputs[2]
                state_c = decoder_outputs[3]

                joint_pred_idx = int(np.argmax(logits))

                if joint_pred_idx == blank_id:
                    break
                else:
                    predictions.append(joint_pred_idx)
                    pred_idx = joint2pred[joint_pred_idx]
                    if pred_idx == -1:
                        pred_idx = 0
                    y = np.array([[pred_idx]], dtype=np.int32)

            if len(predictions) >= 50:
                break

        pred_text = ''.join([vocab[idx] if idx < len(vocab) else '?' for idx in predictions])
        print(f"  {transform_name}: '{pred_text}'")


def main():
    print("Testing multiple preprocessing variations")
    print("="*60)

    # Get hello data
    points, word = get_hello_data()
    print(f"Target word: '{word}' with {len(points)} points")

    # Test coordinate transformations
    test_variation(points, word, prepare_points_transformed, "Transform [0,1] -> [-1,1] (STANDARD)")
    test_variation(points, word, prepare_points_untransformed, "NO transform (keep [0,1])")
    test_variation(points, word, prepare_points_wrong_transform, "Wrong transform (assume already [-1,1])")

    # Test without resampling
    test_no_resampling(points, word)

    print("\n" + "="*60)
    print("Summary: If all variations produce similar wrong output,")
    print("the issue is likely with the model training, not preprocessing.")


if __name__ == '__main__':
    main()