#!/usr/bin/env python
"""Test ONNX inference with the exported stateful models."""

import json
import numpy as np
import onnxruntime as ort
import sys
sys.path.append('../new')

from swipe_data_utils import KeyboardGrid, SwipeFeaturizer

def load_test_trace():
    """Load a test trace from validation data."""
    import random

    # Load from validation data
    samples = []
    with open('../data/train_final_val.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:  # Just load first 10 for testing
                break
            samples.append(json.loads(line))

    # Pick a sample
    test_sample = samples[0]  # Use first sample
    print(f"Testing with word: '{test_sample['word']}'")
    return test_sample

def preprocess_trace(trace_data):
    """Apply feature extraction pipeline to trace."""
    grid = KeyboardGrid()
    featurizer = SwipeFeaturizer(grid)

    # Extract features
    features = featurizer(trace_data['points'])
    print(f"Feature shape: {features.shape}")

    return features

def run_inference(features):
    """Run inference with ONNX models."""
    # Load models
    print("Loading ONNX models...")
    encoder_session = ort.InferenceSession('models/encoder.onnx')
    decoder_session = ort.InferenceSession('models/decoder.onnx')
    joint_session = ort.InferenceSession('models/joint.onnx')

    # Load metadata
    with open('models/runtime_meta.json', 'r') as f:
        meta = json.load(f)

    blank_id = meta['blank_id']
    num_layers = meta['decoder_config']['num_layers']
    hidden_size = meta['decoder_config']['hidden_size']

    # Prepare encoder input
    batch_size = 1
    time_steps = features.shape[0]
    feat_dim = features.shape[1]

    # Reshape to [batch, features, time]
    audio_signal = features.T.reshape(1, feat_dim, time_steps).astype(np.float32)
    length = np.array([time_steps], dtype=np.int64)

    # Run encoder
    print("Running encoder...")
    encoder_outputs = encoder_session.run(
        None,
        {'audio_signal': audio_signal, 'length': length}
    )
    encoded = encoder_outputs[0]  # Shape: [batch, time, features]
    encoded_len = encoder_outputs[1][0]

    # Transpose if needed - encoder output is [batch, features, time] from Conformer
    # but we need [batch, time, features] for frame-by-frame processing
    if encoded.shape[1] > encoded.shape[2]:
        encoded = np.transpose(encoded, (0, 2, 1))

    print(f"Encoder output shape: {encoded.shape}, length: {encoded_len}")

    # Initialize decoder states
    h_state = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)
    c_state = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)

    # Greedy decoding
    tokens = []
    current_token = blank_id
    max_steps = min(encoded_len, 20)

    print("Running decoder...")
    for t in range(max_steps):
        # Extract encoder frame
        encoder_frame = encoded[:, t:t+1, :]

        # Run decoder
        input_tokens = np.array([[current_token]], dtype=np.int64)
        decoder_outputs = decoder_session.run(
            None,
            {
                'input_tokens': input_tokens,
                'h_in': h_state,
                'c_in': c_state
            }
        )
        decoder_out = decoder_outputs[0]
        h_state = decoder_outputs[1]
        c_state = decoder_outputs[2]

        # Run joint network
        joint_outputs = joint_session.run(
            None,
            {
                'encoder_output': encoder_frame,
                'decoder_output': decoder_out
            }
        )
        logits = joint_outputs[0]

        # Get argmax prediction
        predicted = np.argmax(logits[0, 0, :])

        # Update token if not blank
        if predicted != blank_id:
            tokens.append(predicted)
            current_token = predicted

    # Convert tokens to text
    text = ''.join([meta['tokens'][t] for t in tokens if t < len(meta['tokens'])])
    return text, tokens

def main():
    """Main test function."""
    # Test multiple samples
    samples = []
    with open('../data/train_final_val.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:  # Test first 5
                break
            samples.append(json.loads(line))

    correct = 0
    for i, test_sample in enumerate(samples):
        print(f"\n=== Sample {i+1} ===")
        print(f"Word: '{test_sample['word']}'")

        # Preprocess
        features = preprocess_trace(test_sample)

        # Run inference
        predicted_text, tokens = run_inference(features)

        # Results
        print(f"Expected: '{test_sample['word']}'")
        print(f"Predicted: '{predicted_text}'")

        # Check accuracy
        if predicted_text == test_sample['word']:
            print("✓ Correct!")
            correct += 1
        else:
            print("✗ Incorrect")

    print(f"\n=== Overall Accuracy ===")
    print(f"Correct: {correct}/{len(samples)} ({100*correct/len(samples):.1f}%)")

if __name__ == "__main__":
    main()