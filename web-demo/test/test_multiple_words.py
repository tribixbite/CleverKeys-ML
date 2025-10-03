#!/usr/bin/env python3
"""Test multiple words to understand prediction patterns."""

import numpy as np
import onnxruntime as ort
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'new'))

from train_transducer_personalized import (
    PersonalizedSwipeFeaturizer,
    resample_points,
    clamp
)

def get_sample_data(line_number):
    """Get swipe data from specific line"""
    import linecache
    data_path = '../../data/train_final_train.jsonl'
    line = linecache.getline(data_path, line_number)
    if line:
        data = json.loads(line)
        return data['points'], data['word']
    return None, None

def prepare_features(points):
    """Prepare features exactly as training does"""
    prepared = []
    for idx, pt in enumerate(points):
        raw_x = float(pt.get("x", 0.0))
        raw_y = float(pt.get("y", 0.0))
        centered_x = raw_x * 2.0 - 1.0
        centered_y = raw_y * 2.0 - 1.0
        centered_x = clamp(centered_x, -1.5, 1.5)
        centered_y = clamp(centered_y, -1.5, 1.5)
        raw_t = float(pt.get("t", idx * 10.0))
        prepared.append({"x": centered_x, "y": centered_y, "t": raw_t})

    resampled = resample_points(prepared, 82)
    featurizer = PersonalizedSwipeFeaturizer(mobile_features=False)
    features = featurizer(resampled)

    if features.shape[1] < 37:
        padding = np.zeros((features.shape[0], 37 - features.shape[1]), dtype=np.float32)
        features = np.concatenate([features, padding], axis=1)

    return features

def stateful_rnnt_decode(encoder_session, decoder_session, features, blank_id=29):
    """Perform stateful RNN-T decoding"""
    encoder_input = features.T.reshape(1, 37, -1).astype(np.float32)
    encoder_outputs = encoder_session.run(None, {
        'audio_signal': encoder_input,
        'length': np.array([features.shape[0]], dtype=np.int64)
    })[0]

    T = encoder_outputs.shape[2]

    num_layers = 2
    batch_size = 1
    hidden_size = 320
    h_state = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)
    c_state = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)

    hypothesis = []
    prev_token = np.array([[blank_id]], dtype=np.int32)

    for t in range(T):
        enc_frame = encoder_outputs[:, :, t:t+1]

        for step in range(8):
            decoder_outputs = decoder_session.run(None, {
                'encoder_outputs': enc_frame,
                'targets': prev_token,
                'target_length': np.array([1], dtype=np.int32),
                'input_states_1': h_state,
                'input_states_2': c_state
            })

            logits = decoder_outputs[0]
            h_state = decoder_outputs[2]
            c_state = decoder_outputs[3]

            pred_id = np.argmax(logits[0, 0, 0, :])

            if pred_id == blank_id:
                break
            else:
                hypothesis.append(int(pred_id))
                prev_token = np.array([[pred_id]], dtype=np.int32)

        prev_token = np.array([[blank_id]], dtype=np.int32)

    return hypothesis

def decode_hypothesis(hypothesis, vocab):
    """Convert token IDs to string"""
    result = []
    for token_id in hypothesis:
        if 0 <= token_id < len(vocab):
            char = vocab[token_id]
            if char not in ['<blank>', '<unk>']:
                result.append(char)
    return ''.join(result)

def main():
    model_dir = '../models/rnnt_new_latest'
    encoder_session = ort.InferenceSession(f'{model_dir}/encoder.onnx')
    decoder_session = ort.InferenceSession(f'{model_dir}/decoder_joint.onnx')

    with open('../runtime_meta.json', 'r') as f:
        meta = json.load(f)
    vocab = meta['tokens']
    blank_id = meta.get('blank_id', 29)

    test_lines = [
        431621,  # hello
        100, 1000, 10000, 50000, 100000, 200000, 300000, 400000, 500000
    ]

    print("Testing multiple words:")
    print("="*60)

    correct = 0
    total = 0

    for line_num in test_lines:
        points, word = get_sample_data(line_num)
        if not points:
            continue

        features = prepare_features(points)
        hypothesis = stateful_rnnt_decode(encoder_session, decoder_session, features, blank_id)
        predicted = decode_hypothesis(hypothesis, vocab)

        match = '✓' if predicted == word else '✗'
        total += 1
        if predicted == word:
            correct += 1

        print(f"Line {line_num:6d}: '{word:15s}' → '{predicted:15s}' {match}")

    print("="*60)
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")

if __name__ == '__main__':
    main()
