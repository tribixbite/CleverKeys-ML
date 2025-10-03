#!/usr/bin/env python3
"""
Test with a simple word to debug model
"""

import json
import numpy as np
import onnxruntime as ort
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'trained_models', 'nema1'))

from train_transducer_personalized import (
    PersonalizedSwipeFeaturizer,
    resample_points,
    determine_resample_target
)


def clamp(x, min_val, max_val):
    return max(min_val, min(max_val, x))


def get_test_data(word_to_find):
    """Get first instance of word from dataset"""
    import linecache
    data_path = '../../data/train_final_train.jsonl'

    # Quick search for common words
    test_lines = {
        'the': 1,
        'and': 5,
        'hello': 431621,
        'person': 4666,
        'companion': 22440
    }

    if word_to_find in test_lines:
        line_num = test_lines[word_to_find]
        line = linecache.getline(data_path, line_num)
        if line:
            data = json.loads(line)
            return data['points'], data['word']

    # Otherwise search
    with open(data_path, 'r') as f:
        for i, line in enumerate(f, 1):
            if i > 10000:  # Don't search too far
                break
            data = json.loads(line)
            if data['word'] == word_to_find:
                print(f"Found '{word_to_find}' at line {i}")
                return data['points'], data['word']

    return None, None


def test_simple():
    """Test with a simple word"""

    # Test with "the" - a very common word
    test_word = "the"
    points, word = get_test_data(test_word)

    if points is None:
        print(f"Could not find word '{test_word}'")
        return

    print(f"\nTesting word: '{word}' ({len(points)} points)")

    # Load models - try auto_best instead
    model_dir = '../models/auto_best'
    encoder_path = os.path.join(model_dir, 'encoder.onnx')
    decoder_path = os.path.join(model_dir, 'decoder_joint.onnx')
    meta_path = os.path.join(model_dir, 'runtime_meta.json')

    encoder_session = ort.InferenceSession(encoder_path)
    decoder_session = ort.InferenceSession(decoder_path)

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    blank_id = meta['blank_id']
    vocab = meta['tokens']

    # Process swipe
    # 1. Transform coordinates
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

    # 2. Resample
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }
    target_len = determine_resample_target(len(normalized), preprocess_cfg)
    resampled = resample_points(normalized, target_len)

    # 3. Extract features
    featurizer = PersonalizedSwipeFeaturizer()
    features = featurizer(resampled)

    print(f"Features: shape={features.shape}, min={features.min():.2f}, max={features.max():.2f}")

    # 4. Run encoder
    signal = features.astype(np.float32).T.reshape(1, 37, -1)
    signal_len = np.array([features.shape[0]], dtype=np.int64)

    encoder_outputs = encoder_session.run(None, {
        'audio_signal': signal,
        'length': signal_len
    })
    encoded = encoder_outputs[0]
    encoded_len = encoder_outputs[1][0]

    print(f"Encoded: shape={encoded.shape}, frames={encoded_len}")

    # 5. Simple greedy decode - just take top prediction at each frame
    print("\nFrame-by-frame predictions:")

    state_h = np.zeros((1, 1, 192), dtype=np.float32)
    state_c = np.zeros((1, 1, 192), dtype=np.float32)
    y = np.array([[0]], dtype=np.int32)

    predictions = []
    for t in range(min(10, encoded_len)):
        enc_frame = encoded[:, :, t:t+1]

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

        # Get top 3 predictions
        top3_idx = np.argsort(logits)[-3:].tolist()[::-1]
        top3_chars = [vocab[i] if i < len(vocab) else '?' for i in top3_idx]
        top3_scores = [logits[i] for i in top3_idx]

        pred_idx = top3_idx[0]
        pred_char = top3_chars[0]

        print(f"  Frame {t}: Top 3 = {list(zip(top3_chars, top3_scores))}")

        if pred_idx != blank_id:
            predictions.append(pred_idx)
            # Update y for next step
            if pred_idx < blank_id:
                next_y = pred_idx
            else:
                next_y = pred_idx - 1
            y = np.array([[next_y]], dtype=np.int32)

    pred_text = ''.join([vocab[i] if i < len(vocab) else '?' for i in predictions])
    print(f"\nPredicted: '{pred_text}'")
    print(f"Expected: '{word}'")


if __name__ == '__main__':
    test_simple()