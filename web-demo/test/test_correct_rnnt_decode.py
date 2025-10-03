#!/usr/bin/env python3
"""
Test CORRECT RNN-T decoding based on training behavior.
Key insight: The model may start predicting from the FIRST character, not blank.
"""

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

def get_hello_data():
    """Get hello swipe data from line 431621"""
    import linecache
    data_path = '../../data/train_final_train.jsonl'
    line = linecache.getline(data_path, 431621)
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

def correct_rnnt_decode(encoder_session, decoder_session, features, vocab_size=30, blank_id=29):
    """
    Correct RNN-T decoding:
    1. Start with blank token
    2. If model predicts non-blank, emit it and continue with same frame
    3. If model predicts blank, move to next frame
    4. Key: Don't continue infinitely after blank - just move to next frame
    """

    # Run encoder
    encoder_input = features.T.reshape(1, 37, -1).astype(np.float32)
    encoder_outputs = encoder_session.run(None, {
        'audio_signal': encoder_input,
        'length': np.array([features.shape[0]], dtype=np.int64)
    })[0]

    T = encoder_outputs.shape[2]
    print(f"Encoder frames: {T}")

    # Initialize decoder states
    num_layers = 2
    batch_size = 1
    hidden_size = 320

    h_state = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)
    c_state = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)

    hypothesis = []

    # Process each encoder frame
    for t in range(T):
        enc_frame = encoder_outputs[:, :, t:t+1]

        # Start each frame with blank token
        prev_token = np.array([[blank_id]], dtype=np.int32)

        # Allow multiple emissions per frame
        for step in range(8):  # Max symbols per frame
            decoder_outputs = decoder_session.run(None, {
                'encoder_outputs': enc_frame,
                'targets': prev_token,
                'target_length': np.array([1], dtype=np.int32),
                'input_states_1': h_state,
                'input_states_2': c_state
            })

            logits = decoder_outputs[0][0, 0, 0, :]
            h_state = decoder_outputs[2]
            c_state = decoder_outputs[3]

            pred_id = np.argmax(logits)

            if pred_id == blank_id:
                # Blank means done with this frame
                break
            else:
                # Emit character and continue
                hypothesis.append(int(pred_id))
                prev_token = np.array([[pred_id]], dtype=np.int32)

                # Stop if we have enough characters
                if len(hypothesis) >= 10:
                    return hypothesis

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
    # Load models
    model_dir = '../models/rnnt_new_latest'
    encoder_session = ort.InferenceSession(f'{model_dir}/encoder.onnx')
    decoder_session = ort.InferenceSession(f'{model_dir}/decoder_joint.onnx')

    # Load vocabulary
    with open('../runtime_meta.json', 'r') as f:
        meta = json.load(f)
    vocab = meta['tokens']
    blank_id = meta.get('blank_id', 29)

    # Get test data
    points, word = get_hello_data()
    print(f"Testing: '{word}' with {len(points)} points")

    # Prepare features
    features = prepare_features(points)

    # Perform correct RNN-T decoding
    hypothesis = correct_rnnt_decode(
        encoder_session,
        decoder_session,
        features,
        vocab_size=len(vocab),
        blank_id=blank_id
    )

    # Decode result
    predicted = decode_hypothesis(hypothesis, vocab)

    print(f"\nResults:")
    print(f"Expected:  '{word}'")
    print(f"Predicted: '{predicted}'")
    print(f"Token IDs: {hypothesis}")

    # Expected IDs for 'hello': [9, 6, 13, 13, 16]
    print(f"\nExpected IDs for 'hello': [9, 6, 13, 13, 16]")

if __name__ == '__main__':
    main()