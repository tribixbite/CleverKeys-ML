#!/usr/bin/env python3
"""
Test stateful ONNX inference with proper state management.
This demonstrates the correct way to maintain decoder state between timesteps.
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
    # Process swipe exactly as training does
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

    # Resample to 82
    resampled = resample_points(prepared, 82)

    # Extract features
    featurizer = PersonalizedSwipeFeaturizer(mobile_features=False)
    features = featurizer(resampled)

    # Pad to 37
    if features.shape[1] < 37:
        padding = np.zeros((features.shape[0], 37 - features.shape[1]), dtype=np.float32)
        features = np.concatenate([features, padding], axis=1)

    return features

def stateful_rnnt_decode(encoder_session, decoder_session, features, vocab_size=30, blank_id=29):
    """
    Perform stateful RNN-T decoding with proper state management.

    Key insight: The decoder state must be maintained between timesteps.
    Each call to the decoder produces new states that become inputs for the next call.
    """

    # Step 1: Run encoder (stateless, run once)
    print("Running encoder...")
    encoder_input = features.T.reshape(1, 37, -1).astype(np.float32)  # [1, features, time]
    encoder_outputs = encoder_session.run(None, {
        'audio_signal': encoder_input,
        'length': np.array([features.shape[0]], dtype=np.int64)
    })[0]

    T = encoder_outputs.shape[2]  # Number of encoder frames
    print(f"Encoder output shape: {encoder_outputs.shape} (T={T})")

    # Step 2: Initialize decoder states
    # From ONNX inspection: states have shape [2, batch, 320]
    num_layers = 2
    batch_size = 1
    hidden_size = 320

    h_state = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)
    c_state = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)

    # Step 3: Initialize decoding
    hypothesis = []
    prev_token = np.array([[blank_id]], dtype=np.int32)  # Start with blank (int32 for ONNX)

    # Step 4: Decode each encoder frame
    print(f"\nDecoding {T} frames with stateful RNN-T...")
    for t in range(T):
        # Get current encoder frame
        enc_frame = encoder_outputs[:, :, t:t+1]  # [1, 256, 1]

        # Inner loop: emit multiple characters per frame if needed
        max_symbols_per_step = 8
        for step in range(max_symbols_per_step):
            # Run decoder with current state
            decoder_outputs = decoder_session.run(None, {
                'encoder_outputs': enc_frame,
                'targets': prev_token,
                'target_length': np.array([1], dtype=np.int32),  # int32 for ONNX
                'input_states_1': h_state,
                'input_states_2': c_state
            })

            # Unpack outputs
            logits = decoder_outputs[0]  # [1, 1, 1, 30]
            # prednet_lengths = decoder_outputs[1]  # Not needed
            h_state = decoder_outputs[2]  # New hidden state [2, 1, 320]
            c_state = decoder_outputs[3]  # New cell state [2, 1, 320]

            # Get prediction
            pred_id = np.argmax(logits[0, 0, 0, :])

            if pred_id == blank_id:
                # Blank means "no more output for this frame"
                # Keep the state but move to next frame
                break
            else:
                # Emit character and continue with same frame
                hypothesis.append(int(pred_id))
                prev_token = np.array([[pred_id]], dtype=np.int32)  # int32 for ONNX

                # Debug: show emissions
                if t < 5 or t % 10 == 0:
                    char = chr(ord('a') + pred_id - 2) if 2 <= pred_id <= 27 else f"[{pred_id}]"
                    print(f"  Frame {t}, step {step}: emitted '{char}' (id={pred_id})")

        # Always update prev_token to blank after processing a frame
        # This ensures the decoder knows we're at a frame boundary
        prev_token = np.array([[blank_id]], dtype=np.int32)  # int32 for ONNX

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

    print("Loading ONNX models...")
    encoder_session = ort.InferenceSession(f'{model_dir}/encoder.onnx')
    decoder_session = ort.InferenceSession(f'{model_dir}/decoder_joint.onnx')

    # Load vocabulary
    with open('../runtime_meta.json', 'r') as f:
        meta = json.load(f)
    vocab = meta['tokens']
    blank_id = meta.get('blank_id', 29)

    # Get test data
    points, word = get_hello_data()
    print(f"\nTesting word: '{word}' with {len(points)} points")

    # Prepare features
    features = prepare_features(points)
    print(f"Features shape: {features.shape}")

    # Perform stateful decoding
    hypothesis = stateful_rnnt_decode(
        encoder_session,
        decoder_session,
        features,
        vocab_size=len(vocab),
        blank_id=blank_id
    )

    # Decode result
    predicted = decode_hypothesis(hypothesis, vocab)

    print(f"\n{'='*50}")
    print(f"Expected:  '{word}'")
    print(f"Predicted: '{predicted}'")
    print(f"Match: {predicted == word}")

    # Show character-by-character comparison
    print(f"\nCharacter comparison:")
    for i, (e, p) in enumerate(zip(word.ljust(len(predicted)), predicted.ljust(len(word)))):
        match = '✓' if e == p else '✗'
        print(f"  Position {i}: '{e}' vs '{p}' {match}")

if __name__ == '__main__':
    main()