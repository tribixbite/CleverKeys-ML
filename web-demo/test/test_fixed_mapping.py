#!/usr/bin/env python3
"""
Test with FIXED predictor label mapping
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
    print("Testing 'hello' with FIXED predictor label mapping")
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

    # Get the label mappings
    joint2pred = meta['predictor']['label_map']['joint2pred']
    pred2joint = meta['predictor']['label_map']['pred2joint']
    bos_id = meta['predictor']['bos_id']

    print(f"Model config: {num_layers} layers, {hidden_size} hidden")
    print(f"Vocab size: {len(vocab)}, Blank ID: {blank_id}")
    print(f"BOS ID (predictor space): {bos_id}")
    print()

    # Get hello data
    points, word = get_hello_data()
    print(f"Testing word: '{word}' with {len(points)} points")

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

    print(f"Encoded shape: {encoded.shape}, length: {encoded_len}")

    # Greedy decode with FIXED mapping
    state_h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    state_c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)

    # Start with BOS in predictor space
    y = np.array([[bos_id]], dtype=np.int32)

    predictions = []
    print("\nDecoding with FIXED label mapping:")

    for t in range(encoded_len):
        enc_frame = encoded[:, :, t:t+1]

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

            # Get prediction in joint space
            joint_pred_idx = int(np.argmax(logits))

            # Debug first frame
            if t == 0 and symbol_idx == 0:
                top5_idx = np.argsort(logits)[-5:][::-1]
                print(f"  Frame 0: Top predictions in joint space:")
                for idx in top5_idx[:5]:
                    token = vocab[idx] if idx < len(vocab) else '?'
                    score = float(logits[idx])
                    pred_space = joint2pred[idx] if idx < len(joint2pred) else -1
                    print(f"    Joint {idx} ('{token}') -> Pred {pred_space}, score: {score:.3f}")

            if joint_pred_idx == blank_id:
                # Emit blank and move to next frame
                break
            else:
                # Emit character
                predictions.append(joint_pred_idx)

                # Map from joint space to predictor space for next input
                predictor_idx = joint2pred[joint_pred_idx]

                if predictor_idx == -1:
                    # This shouldn't happen for non-blank
                    print(f"  WARNING: Got -1 mapping for joint index {joint_pred_idx}")
                    predictor_idx = 0  # Fallback to BOS

                y = np.array([[predictor_idx]], dtype=np.int32)

        if len(predictions) >= 24:
            break

    pred_text = ''.join([vocab[idx] if idx < len(vocab) else '?' for idx in predictions])

    print(f"\n{'='*60}")
    print(f"Predicted: '{pred_text}'")
    print(f"Expected:  '{word}'")
    print(f"Success:   {'✅' if pred_text == word else '❌'}")

    # Also test the original (wrong) approach for comparison
    print("\n" + "="*60)
    print("For comparison, original (WRONG) mapping approach:")

    # Reset states
    state_h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    state_c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    y = np.array([[0]], dtype=np.int32)
    predictions_wrong = []

    for t in range(min(encoded_len, 10)):
        enc_frame = encoded[:, :, t:t+1]
        for symbol_idx in range(6):
            decoder_outputs = decoder_session.run(None, {
                'targets': y,
                'input_states_1': state_h,
                'input_states_2': state_c,
                'encoder_outputs': enc_frame,
                'target_length': np.array([1], dtype=np.int32)
            })
            logits = decoder_outputs[0].flatten() if len(decoder_outputs[0].shape) > 2 else decoder_outputs[0].flatten()
            state_h = decoder_outputs[2]
            state_c = decoder_outputs[3]

            pred_idx = int(np.argmax(logits))
            if pred_idx == blank_id:
                break
            else:
                predictions_wrong.append(pred_idx)
                # WRONG mapping:
                if pred_idx < blank_id:
                    next_y = pred_idx
                else:
                    next_y = pred_idx - 1
                y = np.array([[next_y]], dtype=np.int32)
        if len(predictions_wrong) >= 24:
            break

    pred_text_wrong = ''.join([vocab[idx] if idx < len(vocab) else '?' for idx in predictions_wrong])
    print(f"Predicted (wrong mapping): '{pred_text_wrong}'")


if __name__ == '__main__':
    main()