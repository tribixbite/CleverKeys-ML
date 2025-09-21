#!/usr/bin/env python
"""Test with correct feature extraction matching the training."""

import json
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any, Tuple

def clamp(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, val))

def build_default_key_centers() -> List[Tuple[str, float, float]]:
    """Build keyboard key centers in [-1, 1] coordinates matching training."""
    centers: List[Tuple[str, float, float]] = []
    layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    for row_idx, row in enumerate(layout):
        for col_idx, char in enumerate(row):
            x01 = (col_idx + 0.5) / 10.0
            y01 = (row_idx + 0.5) / 3.0
            # Convert to [-1, 1]
            centers.append((char, x01 * 2.0 - 1.0, y01 * 2.0 - 1.0))
    return centers

KEY_CENTERS_CENTERED = build_default_key_centers()

def normalize_points(points: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Normalize points from [0,1] to [-1,1] matching training."""
    if not points:
        return []

    start_t = float(points[0].get("t", 0.0))
    normalized: List[Dict[str, float]] = []

    for idx, pt in enumerate(points):
        raw_x = float(pt.get("x", 0.5))
        raw_y = float(pt.get("y", 0.5))
        # Convert [0,1] to [-1,1]
        centered_x = clamp(raw_x * 2.0 - 1.0, -1.0, 1.0)
        centered_y = clamp(raw_y * 2.0 - 1.0, -1.0, 1.0)
        raw_t = float(pt.get("t", idx * 10.0))

        normalized.append({
            "x": centered_x,
            "y": centered_y,
            "t": raw_t - start_t
        })

    return normalized

def resample_points(points: List[Dict[str, float]], target_count: int):
    """Resample points using linear interpolation."""
    if not points or target_count <= 0:
        return []
    if len(points) == target_count:
        return points[:]

    resampled = []
    duration = points[-1]["t"] - points[0]["t"] if len(points) > 1 else 0.0
    step = duration / max(target_count - 1, 1)

    src_idx = 0
    for i in range(target_count):
        target_t = points[0]["t"] + step * i

        while src_idx < len(points) - 2 and points[src_idx + 1]["t"] < target_t:
            src_idx += 1

        p1 = points[src_idx]
        p2 = points[min(src_idx + 1, len(points) - 1)]

        dt = max(p2["t"] - p1["t"], 1.0)
        alpha = clamp((target_t - p1["t"]) / dt, 0.0, 1.0)

        resampled.append({
            "x": p1["x"] + (p2["x"] - p1["x"]) * alpha,
            "y": p1["y"] + (p2["y"] - p1["y"]) * alpha,
            "t": target_t
        })

    return resampled

def extract_features(points: List[Dict[str, float]]) -> np.ndarray:
    """Extract 37-dimensional features matching training."""
    if len(points) < 2:
        return np.zeros((1, 37), dtype=np.float32)

    features = []

    for i in range(len(points)):
        curr = points[i]

        # Position and time
        x, y = curr["x"], curr["y"]
        t = curr["t"] / 1000.0  # Convert to seconds

        # Velocity
        if i > 0:
            prev = points[i - 1]
            dt = max((curr["t"] - prev["t"]) / 1000.0, 0.001)
            vx = (x - prev["x"]) / dt
            vy = (y - prev["y"]) / dt
            speed = np.sqrt(vx**2 + vy**2)
        else:
            vx = vy = speed = 0.0

        # Acceleration
        if i > 1:
            prev = points[i - 1]
            prev2 = points[i - 2]
            dt1 = max((curr["t"] - prev["t"]) / 1000.0, 0.001)
            dt2 = max((prev["t"] - prev2["t"]) / 1000.0, 0.001)

            vx_prev = (prev["x"] - prev2["x"]) / dt2
            vy_prev = (prev["y"] - prev2["y"]) / dt2

            ax = (vx - vx_prev) / dt1
            ay = (vy - vy_prev) / dt1
            acc = np.sqrt(ax**2 + ay**2)
        else:
            ax = ay = acc = 0.0

        # Angle
        angle = np.arctan2(vy, vx) if i > 0 else 0.0
        angle_sin = np.sin(angle)
        angle_cos = np.cos(angle)

        # Curvature
        curvature = 0.0
        if i > 1:
            prev = points[i - 1]
            prev2 = points[i - 2]
            angle_prev = np.arctan2(
                prev["y"] - prev2["y"],
                prev["x"] - prev2["x"]
            )
            curvature = angle - angle_prev
            while curvature > np.pi:
                curvature -= 2 * np.pi
            while curvature < -np.pi:
                curvature += 2 * np.pi

        # Distance to nearest keys
        key_dists = []
        for _, kx, ky in KEY_CENTERS_CENTERED:
            dist = np.sqrt((x - kx)**2 + (y - ky)**2)
            key_dists.append(dist)
        key_dists.sort()
        nearest_5 = key_dists[:5]
        while len(nearest_5) < 5:
            nearest_5.append(1.0)

        # Progress
        progress = i / max(len(points) - 1, 1)
        is_start = 1.0 if i == 0 else 0.0
        is_end = 1.0 if i == len(points) - 1 else 0.0

        # Window features
        win_start = max(0, i - 2)
        win_end = min(len(points), i + 3)
        win_pts = points[win_start:win_end]

        win_mean_x = np.mean([p["x"] for p in win_pts])
        win_mean_y = np.mean([p["y"] for p in win_pts])

        if len(win_pts) > 1:
            win_std_x = np.std([p["x"] for p in win_pts])
            win_std_y = np.std([p["y"] for p in win_pts])
            win_range_x = max(p["x"] for p in win_pts) - min(p["x"] for p in win_pts)
            win_range_y = max(p["y"] for p in win_pts) - min(p["y"] for p in win_pts)
        else:
            win_std_x = win_std_y = 0.0
            win_range_x = win_range_y = 0.0

        # Assemble features
        feat = [
            x, y, t,  # 3
            vx, vy, speed,  # 6
            ax, ay, acc,  # 9
            angle, angle_sin, angle_cos, curvature,  # 13
            *nearest_5,  # 18
            progress, is_start, is_end,  # 21
            win_mean_x, win_std_x, win_mean_y, win_std_y,  # 25
            win_range_x, win_range_y  # 27
        ]

        # Pad to 37
        while len(feat) < 37:
            feat.append(0.0)

        features.append(feat[:37])

    return np.array(features, dtype=np.float32)

def test_with_correct_features():
    """Test with feature extraction matching training."""

    # Load models
    encoder_session = ort.InferenceSession('models/encoder.onnx')
    decoder_session = ort.InferenceSession('models/decoder.onnx')
    joint_session = ort.InferenceSession('models/joint.onnx')

    # Load metadata
    with open('models/runtime_meta.json', 'r') as f:
        meta = json.load(f)

    blank_id = meta['blank_id']
    num_layers = meta['decoder_config']['num_layers']
    hidden_size = meta['decoder_config']['hidden_size']

    # Test samples
    samples = []
    with open('../data/train_final_val.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            samples.append(json.loads(line))

    correct = 0
    for i, sample in enumerate(samples):
        # Process points
        normalized = normalize_points(sample['points'])

        # Determine resample target based on length
        length = len(normalized)
        if length <= 48:
            target = 56
        elif length >= 112:
            target = 96
        else:
            progress = (length - 48) / (112 - 48)
            target = round(56 + progress * (96 - 56))

        resampled = resample_points(normalized, target)
        features = extract_features(resampled)

        # Run encoder
        audio_signal = features.T.reshape(1, 37, -1).astype(np.float32)
        length_tensor = np.array([features.shape[0]], dtype=np.int64)

        encoder_outputs = encoder_session.run(None, {
            'audio_signal': audio_signal,
            'length': length_tensor
        })
        encoded = encoder_outputs[0]
        encoded_len = encoder_outputs[1][0]

        # Transpose if needed
        if encoded.shape[1] > encoded.shape[2]:
            encoded = np.transpose(encoded, (0, 2, 1))

        # Greedy decode
        tokens = []
        h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
        c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
        last_token = blank_id

        for t in range(min(int(encoded_len), 30)):
            # Decoder
            input_token = np.array([[last_token]], dtype=np.int64)
            decoder_outputs = decoder_session.run(None, {
                'input_tokens': input_token,
                'h_in': h,
                'c_in': c
            })
            decoder_out = decoder_outputs[0]
            h = decoder_outputs[1]
            c = decoder_outputs[2]

            # Joint
            encoder_frame = encoded[:, t:t+1, :]
            joint_outputs = joint_session.run(None, {
                'encoder_output': encoder_frame,
                'decoder_output': decoder_out
            })
            logits = joint_outputs[0][0, 0, :]

            # Get prediction
            pred = np.argmax(logits)

            if pred != blank_id:
                tokens.append(pred)
                last_token = pred

        # Convert to text
        predicted = ''.join([meta['tokens'][t] for t in tokens if t < len(meta['tokens'])])

        print(f"Sample {i+1}: '{sample['word']}'")
        print(f"  Predicted: '{predicted}'")

        if predicted == sample['word']:
            print("  ✓ Correct!")
            correct += 1
        else:
            print("  ✗ Incorrect")

    print(f"\n=== Results ===")
    print(f"Accuracy: {correct}/{len(samples)} ({100*correct/len(samples):.1f}%)")

if __name__ == "__main__":
    test_with_correct_features()