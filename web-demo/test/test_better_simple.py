#!/usr/bin/env python3
"""
Test better trained models with simple words
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


def get_test_data(line_num):
    """Get data from specific line"""
    import linecache
    data_path = '../../data/train_final_train.jsonl'
    line = linecache.getline(data_path, line_num)
    if line:
        data = json.loads(line)
        return data['points'], data['word']
    return None, None


def prepare_points(points):
    """Prepare points as training does"""
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


def test_word(line_num, model_dir='../models/correct_9292025'):
    """Test a specific word"""

    # Get data
    points, word = get_test_data(line_num)
    if points is None:
        print(f"Could not load data from line {line_num}")
        return False

    # Load models
    encoder_session = ort.InferenceSession(os.path.join(model_dir, 'encoder.onnx'))
    decoder_session = ort.InferenceSession(os.path.join(model_dir, 'decoder_joint.onnx'))

    with open(os.path.join(model_dir, 'runtime_meta.json'), 'r') as f:
        meta = json.load(f)

    blank_id = meta['blank_id']
    vocab = meta['tokens']

    # Process swipe
    prepared = prepare_points(points)
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }
    target_len = determine_resample_target(len(prepared), preprocess_cfg)
    resampled = resample_points(prepared, target_len)

    featurizer = PersonalizedSwipeFeaturizer(mobile_features=False)
    features = featurizer(resampled)

    # Pad to 37 dims if needed
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

    # Greedy decode - get decoder dimensions from meta
    num_layers = meta.get('decoder_config', {}).get('num_layers', 1)
    hidden_size = meta.get('decoder_config', {}).get('hidden_size', 192)

    state_h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    state_c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    y = np.array([[0]], dtype=np.int32)

    predictions = []
    for t in range(encoded_len):
        enc_frame = encoded[:, :, t:t+1]

        for _ in range(6):  # max symbols per frame
            decoder_outputs = decoder_session.run(None, {
                'targets': y,
                'input_states_1': state_h,
                'input_states_2': state_c,
                'encoder_outputs': enc_frame,
                'target_length': np.array([1], dtype=np.int32)
            })

            logits_raw = decoder_outputs[0]
            if len(logits_raw.shape) == 4:
                logits = logits_raw[0, 0, 0, :]
            else:
                logits = logits_raw.flatten()

            state_h = decoder_outputs[2]
            state_c = decoder_outputs[3]

            pred_idx = int(np.argmax(logits))

            if pred_idx == blank_id:
                break
            else:
                predictions.append(pred_idx)
                if pred_idx < blank_id:
                    next_y = pred_idx
                else:
                    next_y = pred_idx - 1
                y = np.array([[next_y]], dtype=np.int32)

        if len(predictions) >= 24:  # max total
            break

    pred_text = ''.join([vocab[idx] if idx < len(vocab) else '?' for idx in predictions])

    success = pred_text == word
    print(f"Line {line_num}: '{word}' -> '{pred_text}' {'✓' if success else '✗'}")

    return success


def main():
    print("Testing correct 9292025script models (epoch 74, WER 0.192)")
    print("="*50)

    # Test some specific words
    test_cases = [
        (1, "raped"),  # First line
        (431621, "hello"),  # Known hello line
        (4666, "person"),  # Known person line
        (22440, "companion"),  # Known companion line
        (100, None),  # Unknown word at line 100
        (1000, None),  # Unknown word at line 1000
    ]

    successes = 0
    total = 0

    for line_num, expected_word in test_cases:
        if test_word(line_num):
            successes += 1
        total += 1

    print("="*50)
    print(f"Success rate: {successes}/{total} = {successes/total*100:.1f}%")


if __name__ == '__main__':
    main()