#!/usr/bin/env python3
"""
Test with CORRECT RNN-T decoding - continue through ALL frames, don't stop at blanks!
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
    print("Testing with CORRECT RNN-T decoding (continue through blanks)")
    print("="*70)

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
    joint2pred = meta['predictor']['label_map']['joint2pred']

    print(f"Model: {num_layers} layers, {hidden_size} hidden, blank_id={blank_id}")

    # Get hello data
    points, word = get_hello_data()
    print(f"Testing: '{word}' with {len(points)} points\n")

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

    # Extract features
    featurizer = PersonalizedSwipeFeaturizer(mobile_features=False)
    features = featurizer(resampled)

    # Pad to 37 dims
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

    print(f"Encoded: {encoded_len} frames from {features.shape[0]} input frames")

    # Initialize decoder states
    state_h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    state_c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)

    # Start with BOS (0 in predictor space)
    y = np.array([[0]], dtype=np.int32)

    all_predictions = []
    frame_outputs = []

    print("\nFrame-by-frame decoding:")
    print("-" * 50)

    # Process ALL frames, don't stop at blanks!
    for t in range(encoded_len):
        enc_frame = encoded[:, :, t:t+1]
        frame_chars = []

        # Keep emitting until blank
        for symbol_idx in range(8):  # max symbols per frame
            decoder_outputs = decoder_session.run(None, {
                'targets': y,
                'input_states_1': state_h,
                'input_states_2': state_c,
                'encoder_outputs': enc_frame,
                'target_length': np.array([1], dtype=np.int32)
            })

            logits = decoder_outputs[0]
            if len(logits.shape) == 4:
                logits = logits[0, 0, 0, :]
            else:
                logits = logits.flatten()

            # Always update states!
            state_h = decoder_outputs[2]
            state_c = decoder_outputs[3]

            # Get prediction
            joint_pred_idx = int(np.argmax(logits))

            if joint_pred_idx == blank_id:
                # Blank means done with this frame, move to next
                # Keep the states AND keep the last y (don't reset to BOS)
                # y stays as is from last non-blank emission
                break
            else:
                # Non-blank: emit character and continue
                char = vocab[joint_pred_idx] if joint_pred_idx < len(vocab) else '?'
                frame_chars.append(char)
                all_predictions.append(joint_pred_idx)

                # Map to predictor space for next input
                pred_idx = joint2pred[joint_pred_idx]
                if pred_idx == -1:
                    pred_idx = 0  # shouldn't happen
                y = np.array([[pred_idx]], dtype=np.int32)

        # Log frame output
        if t < 10 or frame_chars:  # Show first 10 frames or any with output
            frame_str = ''.join(frame_chars) if frame_chars else '<blank>'
            if t < 10:
                # Also show top predictions for debugging
                if not frame_chars and t < 5:  # Show why it's blank
                    # Get first symbol logits for this frame
                    test_out = decoder_session.run(None, {
                        'targets': y,
                        'input_states_1': state_h,
                        'input_states_2': state_c,
                        'encoder_outputs': enc_frame,
                        'target_length': np.array([1], dtype=np.int32)
                    })
                    test_logits = test_out[0].flatten() if len(test_out[0].shape) > 2 else test_out[0].flatten()
                    top3_idx = np.argsort(test_logits)[-3:][::-1]
                    top3_chars = [vocab[i] if i < len(vocab) else '?' for i in top3_idx]
                    top3_scores = [test_logits[i] for i in top3_idx]
                    print(f"Frame {t:2d}: {frame_str} (top3: {top3_chars[0]}={top3_scores[0]:.1f}, {top3_chars[1]}={top3_scores[1]:.1f}, {top3_chars[2]}={top3_scores[2]:.1f})")
                else:
                    print(f"Frame {t:2d}: {frame_str}")

        frame_outputs.append(frame_chars)

        # Safety limit
        if len(all_predictions) >= 50:
            print(f"(Stopped at {len(all_predictions)} chars)")
            break

    print("-" * 50)

    # Build final prediction
    pred_text = ''.join([vocab[idx] if idx < len(vocab) else '?' for idx in all_predictions])

    print(f"\nResults:")
    print(f"  Predicted: '{pred_text}'")
    print(f"  Expected:  '{word}'")
    print(f"  Success:   {'✅' if pred_text == word else '❌'}")

    # Show distribution of outputs across frames
    non_blank_frames = [i for i, f in enumerate(frame_outputs) if f]
    print(f"\nOutput distribution:")
    print(f"  Total frames: {encoded_len}")
    print(f"  Frames with output: {len(non_blank_frames)}")
    if non_blank_frames:
        print(f"  Output frame indices: {non_blank_frames[:20]}{'...' if len(non_blank_frames) > 20 else ''}")


if __name__ == '__main__':
    main()