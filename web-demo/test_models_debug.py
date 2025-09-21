#!/usr/bin/env python
"""Debug ONNX models to understand accuracy issue."""

import json
import numpy as np
import onnxruntime as ort
import sys
sys.path.append('../new')

from swipe_data_utils import KeyboardGrid, SwipeFeaturizer

def test_model_outputs():
    """Test what the models are actually outputting."""

    # Load models
    encoder_session = ort.InferenceSession('models/encoder.onnx')
    decoder_session = ort.InferenceSession('models/decoder.onnx')
    joint_session = ort.InferenceSession('models/joint.onnx')

    # Load metadata
    with open('models/runtime_meta.json', 'r') as f:
        meta = json.load(f)

    print(f"Vocab size: {meta['vocab_size']}")
    print(f"Blank ID: {meta['blank_id']}")
    print(f"Tokens: {meta['tokens'][:10]}...")

    # Create simple test input
    batch_size = 1
    time_steps = 10
    feat_dim = 37

    # Random features
    features = np.random.randn(time_steps, feat_dim).astype(np.float32)

    # Prepare encoder input
    audio_signal = features.T.reshape(1, feat_dim, time_steps).astype(np.float32)
    length = np.array([time_steps], dtype=np.int64)

    print(f"\n=== Testing Encoder ===")
    print(f"Input shape: audio_signal={audio_signal.shape}, length={length}")

    # Run encoder
    encoder_outputs = encoder_session.run(None, {
        'audio_signal': audio_signal,
        'length': length
    })
    encoded = encoder_outputs[0]
    encoded_len = encoder_outputs[1]

    print(f"Encoder output: shape={encoded.shape}, length={encoded_len}")
    print(f"Encoder output range: [{encoded.min():.3f}, {encoded.max():.3f}]")

    # Test decoder
    print(f"\n=== Testing Decoder ===")
    num_layers = meta['decoder_config']['num_layers']
    hidden_size = meta['decoder_config']['hidden_size']

    h_state = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)
    c_state = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)
    input_token = np.array([[0]], dtype=np.int64)  # Start with blank

    print(f"Input shapes: token={input_token.shape}, h={h_state.shape}, c={c_state.shape}")

    decoder_outputs = decoder_session.run(None, {
        'input_tokens': input_token,
        'h_in': h_state,
        'c_in': c_state
    })

    decoder_out = decoder_outputs[0]
    h_next = decoder_outputs[1]
    c_next = decoder_outputs[2]

    print(f"Decoder output: shape={decoder_out.shape}")
    print(f"Decoder output range: [{decoder_out.min():.3f}, {decoder_out.max():.3f}]")

    # Test joint
    print(f"\n=== Testing Joint ===")

    # Need to transpose encoder output if needed
    if encoded.shape[1] > encoded.shape[2]:
        encoded = np.transpose(encoded, (0, 2, 1))
        print(f"Transposed encoder to: {encoded.shape}")

    encoder_frame = encoded[:, 0:1, :]  # Take first frame
    print(f"Encoder frame shape: {encoder_frame.shape}")
    print(f"Decoder out shape: {decoder_out.shape}")

    joint_outputs = joint_session.run(None, {
        'encoder_output': encoder_frame,
        'decoder_output': decoder_out
    })

    logits = joint_outputs[0]
    print(f"Joint output shape: {logits.shape}")
    print(f"Joint output range: [{logits.min():.3f}, {logits.max():.3f}]")

    # Apply softmax
    logits_flat = logits[0, 0, :]
    exp_logits = np.exp(logits_flat - np.max(logits_flat))
    probs = exp_logits / np.sum(exp_logits)

    print(f"\n=== Probability Distribution ===")
    top5_indices = np.argsort(probs)[-5:][::-1]
    for idx in top5_indices:
        token = meta['tokens'][idx] if idx < len(meta['tokens']) else f"[{idx}]"
        print(f"  Token {idx:2d} ('{token}'): {probs[idx]:.4f}")

    # Check if model is biased towards certain tokens
    print(f"\nBlank token (ID={meta['blank_id']}) probability: {probs[meta['blank_id']]:.4f}")

    # Test with actual data
    print(f"\n=== Testing with Real Data ===")
    with open('../data/train_final_val.jsonl', 'r') as f:
        sample = json.loads(f.readline())

    print(f"Word: '{sample['word']}'")
    print(f"Points: {len(sample['points'])} points")

    # Extract features
    grid = KeyboardGrid()
    featurizer = SwipeFeaturizer(grid)
    features = featurizer(sample['points'])

    print(f"Features shape: {features.shape}")

    # Run through encoder
    audio_signal = features.T.reshape(1, feat_dim, -1).astype(np.float32)
    length = np.array([features.shape[0]], dtype=np.int64)

    encoder_outputs = encoder_session.run(None, {
        'audio_signal': audio_signal,
        'length': length
    })
    encoded = encoder_outputs[0]

    # Transpose if needed
    if encoded.shape[1] > encoded.shape[2]:
        encoded = np.transpose(encoded, (0, 2, 1))

    print(f"Encoded shape: {encoded.shape}")

    # Simple greedy decode
    tokens = []
    h = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)
    c = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)
    last_token = meta['blank_id']

    for t in range(min(encoded.shape[1], 20)):
        # Run decoder
        input_token = np.array([[last_token]], dtype=np.int64)
        decoder_outputs = decoder_session.run(None, {
            'input_tokens': input_token,
            'h_in': h,
            'c_in': c
        })
        decoder_out = decoder_outputs[0]
        h = decoder_outputs[1]
        c = decoder_outputs[2]

        # Run joint
        encoder_frame = encoded[:, t:t+1, :]
        joint_outputs = joint_session.run(None, {
            'encoder_output': encoder_frame,
            'decoder_output': decoder_out
        })
        logits = joint_outputs[0][0, 0, :]

        # Get prediction
        pred = np.argmax(logits)

        if pred != meta['blank_id']:
            tokens.append(pred)
            last_token = pred

    predicted = ''.join([meta['tokens'][t] for t in tokens if t < len(meta['tokens'])])
    print(f"Predicted: '{predicted}'")
    print(f"Token IDs: {tokens}")

if __name__ == "__main__":
    test_model_outputs()