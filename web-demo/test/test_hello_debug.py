#!/usr/bin/env python3
"""
Debug test for 'hello' with correct models
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


def prepare_points(points):
    """Prepare points as training does"""
    prepared = []
    for idx, pt in enumerate(points):
        raw_x = float(pt.get("x", 0.0))
        raw_y = float(pt.get("y", 0.0))

        # Transform from [0,1] to [-1,1]
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


def main():
    print("Testing 'hello' with correct 9292025script models")
    print("="*60)

    model_dir = '../models/correct_9292025'

    # Load models
    encoder_session = ort.InferenceSession(os.path.join(model_dir, 'encoder.onnx'))
    decoder_session = ort.InferenceSession(os.path.join(model_dir, 'decoder_joint.onnx'))

    with open(os.path.join(model_dir, 'runtime_meta.json'), 'r') as f:
        meta = json.load(f)

    blank_id = meta['blank_id']
    vocab = meta['tokens']
    num_layers = meta['decoder_config']['num_layers']
    hidden_size = meta['decoder_config']['hidden_size']

    print(f"Model config: {num_layers} layers, {hidden_size} hidden")
    print(f"Vocab size: {len(vocab)}, Blank ID: {blank_id}")
    print()

    # Get hello data
    points, word = get_hello_data()
    print(f"Testing word: '{word}' with {len(points)} points")

    # Process swipe
    prepared = prepare_points(points)

    # Resample
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }
    target_len = determine_resample_target(len(prepared), preprocess_cfg)
    resampled = resample_points(prepared, target_len)
    print(f"Resampled to {len(resampled)} points")

    # Extract features
    featurizer = PersonalizedSwipeFeaturizer(mobile_features=False)
    features = featurizer(resampled)
    print(f"Features shape: {features.shape}")

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

    print(f"Encoded shape: {encoded.shape}, length: {encoded_len}")
    print(f"Encoded stats - min: {encoded.min():.4f}, max: {encoded.max():.4f}, mean: {encoded.mean():.4f}")
    print()

    # Greedy decode with extensive debug
    state_h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    state_c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    y = np.array([[0]], dtype=np.int32)

    predictions = []
    print("Frame-by-frame decoding:")

    for t in range(min(encoded_len, 10)):  # Debug first 10 frames
        enc_frame = encoded[:, :, t:t+1]

        print(f"\nFrame {t}:")
        print(f"  Encoder frame stats - min: {enc_frame.min():.4f}, max: {enc_frame.max():.4f}")

        symbols_this_frame = []
        for symbol_idx in range(6):  # max symbols per frame
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

            # Show top predictions
            top5_idx = np.argsort(logits)[-5:][::-1]
            top5_chars = [vocab[i] if i < len(vocab) else '?' for i in top5_idx]
            top5_scores = [float(logits[i]) for i in top5_idx]

            if symbol_idx == 0:  # Show details for first symbol of frame
                print(f"  Symbol {symbol_idx}: Top 5 predictions:")
                for ch, score in zip(top5_chars, top5_scores):
                    print(f"    '{ch}': {score:.3f}")

            if pred_idx == blank_id:
                if symbol_idx > 0:
                    print(f"  Emitted {symbol_idx} symbols: {symbols_this_frame}")
                break
            else:
                symbols_this_frame.append(vocab[pred_idx] if pred_idx < len(vocab) else '?')
                predictions.append(pred_idx)

                # Update y for next iteration
                if pred_idx < blank_id:
                    next_y = pred_idx
                else:
                    next_y = pred_idx - 1
                y = np.array([[next_y]], dtype=np.int32)

        if len(predictions) >= 24:
            break

    pred_text = ''.join([vocab[idx] if idx < len(vocab) else '?' for idx in predictions])

    print(f"\n{'='*60}")
    print(f"Predicted: '{pred_text}'")
    print(f"Expected:  '{word}'")
    print(f"Success:   {'✅' if pred_text == word else '❌'}")


if __name__ == '__main__':
    main()