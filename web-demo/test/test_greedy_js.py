#!/usr/bin/env python3
"""
Simple greedy decode to compare with JS
"""

import json
import numpy as np
import onnxruntime as ort
import os
from pathlib import Path

def get_hello_data():
    """Get hello swipe data from line 431621"""
    import linecache
    data_path = '../../data/train_final_train.jsonl'
    line = linecache.getline(data_path, 431621)
    if line:
        data = json.loads(line)
        return data['points'], data['word']
    return None, None

# Get hello data
points, word = get_hello_data()
print(f"Testing: '{word}' with {len(points)} points")

# Load models
model_dir = '../models/correct_9292025'
encoder_session = ort.InferenceSession(os.path.join(model_dir, 'encoder.onnx'))
decoder_session = ort.InferenceSession(os.path.join(model_dir, 'decoder_joint.onnx'))

with open(os.path.join(model_dir, 'runtime_meta.json'), 'r') as f:
    meta = json.load(f)

blank_id = meta['blank_id']
vocab = meta['tokens']
num_layers = meta['decoder_config']['num_layers']
hidden_size = meta['decoder_config']['hidden_size']
joint2pred = meta['predictor']['label_map']['joint2pred']

# Process swipe (simplified - using fixed 82 frames)
import sys
sys.path.insert(0, '../../new')
from train_transducer_personalized import (
    PersonalizedSwipeFeaturizer,
    resample_points,
    clamp
)

# Prepare points
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

# Resample to 82 (as training does)
resampled = resample_points(prepared, 82)

# Extract features
featurizer = PersonalizedSwipeFeaturizer(mobile_features=False)
features = featurizer(resampled)

# Pad to 37
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

print(f"Encoded frames: {encoded_len}")

# Pure greedy decode - no beam search
state_h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
state_c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
y = np.array([[0]], dtype=np.int32)  # Start with BOS

predictions = []
chars_per_frame = []

for t in range(encoded_len):
    enc_frame = encoded[:, :, t:t+1]
    frame_chars = []
    
    # Try to emit up to 8 characters per frame
    for _ in range(8):
        decoder_outputs = decoder_session.run(None, {
            'targets': y,
            'input_states_1': state_h,
            'input_states_2': state_c,
            'encoder_outputs': enc_frame,
            'target_length': np.array([1], dtype=np.int32)
        })
        
        logits = decoder_outputs[0].flatten()
        state_h = decoder_outputs[2]
        state_c = decoder_outputs[3]
        
        joint_pred_idx = int(np.argmax(logits))
        
        if joint_pred_idx == blank_id:
            # Blank - stop emitting for this frame
            break
        else:
            # Emit character
            char = vocab[joint_pred_idx] if joint_pred_idx < len(vocab) else '?'
            frame_chars.append(char)
            predictions.append(joint_pred_idx)
            
            # Update y for next prediction
            pred_idx = joint2pred[joint_pred_idx]
            if pred_idx == -1:
                pred_idx = 0  # Map blank to BOS
            y = np.array([[pred_idx]], dtype=np.int32)
    
    chars_per_frame.append(frame_chars)
    
    # Safety: stop if we've predicted too many characters
    if len(predictions) >= 50:
        break

pred_text = ''.join([vocab[idx] if idx < len(vocab) else '?' for idx in predictions])

print(f"Predicted: '{pred_text}'")
print(f"Expected:  '{word}'")

# Show which frames emitted characters
print(f"\nFrames with output:")
for i, chars in enumerate(chars_per_frame[:20]):  # Show first 20 frames
    if chars:
        print(f"  Frame {i}: {chars}")
