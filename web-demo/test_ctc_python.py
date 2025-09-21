#!/usr/bin/env python
"""Test CTC models with Python to verify they work."""

import json
import numpy as np
import onnxruntime as ort

def test_ctc_models():
    """Test CTC models."""

    # Load models
    encoder = ort.InferenceSession('models/ctc/swipe_model_character_quant.onnx')
    decoder = ort.InferenceSession('models/ctc/swipe_decoder_character_quant.onnx')

    # Load tokenizer
    with open('models/ctc/tokenizer_config.json', 'r') as f:
        tokenizer = json.load(f)

    print(f"Tokenizer loaded: vocab_size={tokenizer['vocab_size']}")

    # Create test input - 10 points
    num_points = 10

    # Trajectory features: [x, y, vx, vy, t, pressure]
    trajectory_features = np.random.randn(1, num_points, 6).astype(np.float32)

    # Nearest keys - random character indices
    nearest_keys = np.random.randint(4, 30, (1, num_points), dtype=np.int64)

    # Source mask - all true
    src_mask = np.ones((1, num_points), dtype=bool)

    print(f"\nInputs:")
    print(f"  trajectory_features: {trajectory_features.shape}")
    print(f"  nearest_keys: {nearest_keys.shape}")
    print(f"  src_mask: {src_mask.shape} dtype={src_mask.dtype}")

    # Run encoder
    print("\nRunning encoder...")
    encoder_outputs = encoder.run(None, {
        'trajectory_features': trajectory_features,
        'nearest_keys': nearest_keys,
        'src_mask': src_mask
    })

    memory = encoder_outputs[0]
    print(f"Encoder output shape: {memory.shape}")

    # Test decoder with autoregressive decoding
    print("\nRunning decoder...")

    # Start with SOS token
    target_tokens = np.array([[tokenizer['special_tokens']['sos_idx']]], dtype=np.int64)
    target_mask = np.ones((1, 1), dtype=bool)

    decoder_outputs = decoder.run(None, {
        'memory': memory,
        'target_tokens': target_tokens,
        'src_mask': src_mask,
        'target_mask': target_mask
    })

    logits = decoder_outputs[0]
    print(f"Decoder output shape: {logits.shape}")

    # Get prediction
    probs = np.exp(logits[0, -1, :])
    probs = probs / np.sum(probs)
    pred_token = np.argmax(probs)

    if pred_token in tokenizer['idx_to_char']:
        pred_char = tokenizer['idx_to_char'][str(pred_token)]
        print(f"Predicted token: {pred_token} ('{pred_char}')")
    else:
        print(f"Predicted token: {pred_token}")

    print("\n✓ CTC models are working!")

if __name__ == "__main__":
    test_ctc_models()