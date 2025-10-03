#!/usr/bin/env python3
"""
Test the ORIGINAL inference approach - single step, no state management.
This is to verify what the model actually outputs without my "improvements".
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
        centered_x = raw_x * 2.0 - 1.0  # Transform [0,1] → [-1,1]
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

def simple_decode(encoder_session, decoder_session, features, vocab):
    """Simple single-step decode to see what happens"""

    # Run encoder
    encoder_input = features.T.reshape(1, 37, -1).astype(np.float32)
    encoder_outputs = encoder_session.run(None, {
        'audio_signal': encoder_input,
        'length': np.array([features.shape[0]], dtype=np.int64)
    })[0]

    print(f"Encoder output shape: {encoder_outputs.shape}")

    # Try decoding just the first frame with blank input
    num_layers = 2
    batch_size = 1
    hidden_size = 320

    h_state = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)
    c_state = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)

    # Test frame 0 with blank input
    enc_frame = encoder_outputs[:, :, 0:1]  # First frame
    blank_id = 29
    prev_token = np.array([[blank_id]], dtype=np.int32)

    print("\nTesting first encoder frame with blank input:")
    decoder_outputs = decoder_session.run(None, {
        'encoder_outputs': enc_frame,
        'targets': prev_token,
        'target_length': np.array([1], dtype=np.int32),
        'input_states_1': h_state,
        'input_states_2': c_state
    })

    logits = decoder_outputs[0][0, 0, 0, :]  # [30]
    pred_id = np.argmax(logits)

    print(f"Logits shape: {logits.shape}")
    print(f"Top 5 predictions:")
    top5_ids = np.argsort(logits)[-5:][::-1]
    for i, idx in enumerate(top5_ids):
        char = vocab[idx] if idx < len(vocab) else f"[{idx}]"
        print(f"  {i+1}. ID {idx:2d} ('{char:6s}'): {logits[idx]:.3f}")

    print(f"\nPredicted token ID: {pred_id}")
    if pred_id < len(vocab):
        print(f"Predicted character: '{vocab[pred_id]}'")

    # Now test what happens if we feed 'h' (ID 9) as input
    print("\n" + "="*50)
    print("Testing with 'h' (ID 9) as input:")
    h_token = np.array([[9]], dtype=np.int32)

    # Use the updated states from previous call
    h_state_new = decoder_outputs[2]
    c_state_new = decoder_outputs[3]

    decoder_outputs2 = decoder_session.run(None, {
        'encoder_outputs': enc_frame,  # Still first frame
        'targets': h_token,
        'target_length': np.array([1], dtype=np.int32),
        'input_states_1': h_state_new,
        'input_states_2': c_state_new
    })

    logits2 = decoder_outputs2[0][0, 0, 0, :]
    pred_id2 = np.argmax(logits2)

    print(f"Top 5 predictions after 'h':")
    top5_ids2 = np.argsort(logits2)[-5:][::-1]
    for i, idx in enumerate(top5_ids2):
        char = vocab[idx] if idx < len(vocab) else f"[{idx}]"
        print(f"  {i+1}. ID {idx:2d} ('{char:6s}'): {logits2[idx]:.3f}")

    if pred_id2 < len(vocab):
        print(f"\nNext predicted: '{vocab[pred_id2]}' (should be 'e' for hello)")

def main():
    # Load models
    model_dir = '../models/rnnt_new_latest'

    print("Loading ONNX models...")
    encoder_session = ort.InferenceSession(f'{model_dir}/encoder.onnx')
    decoder_session = ort.InferenceSession(f'{model_dir}/decoder_joint.onnx')

    # Load vocabulary
    with open('../runtime_meta.json', 'r') as f:
        meta = json.load(f)
    vocab = meta['tokens']

    # Get test data
    points, word = get_hello_data()
    print(f"Testing word: '{word}' with {len(points)} points")

    # Prepare features
    features = prepare_features(points)
    print(f"Features shape: {features.shape}")

    # Expected IDs for 'hello'
    print(f"\nExpected token IDs for 'hello': [9, 6, 13, 13, 16]")
    print(f"Which maps to: h={vocab[9]}, e={vocab[6]}, l={vocab[13]}, l={vocab[13]}, o={vocab[16]}")

    # Run simple decode
    simple_decode(encoder_session, decoder_session, features, vocab)

if __name__ == '__main__':
    main()