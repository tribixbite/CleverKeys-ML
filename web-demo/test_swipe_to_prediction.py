#!/usr/bin/env python3
"""
Full test from swipe input to prediction using the exported ONNX models.
Tests the complete pipeline with real swipe data from training set.
"""

import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
import random
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

def load_models(model_dir="models"):
    """Load the exported ONNX models."""
    model_path = Path(model_dir)

    # Load metadata
    with open(model_path / "runtime_meta.json", "r") as f:
        meta = json.load(f)

    # Use FP32 models (INT8 has compatibility issues with ConvInteger)
    encoder_path = model_path / "encoder.onnx"
    decoder_path = model_path / "decoder.onnx"
    joint_path = model_path / "joint.onnx"

    print(f"Loading models from {model_path}:")
    print(f"  Encoder: {encoder_path.name}")
    print(f"  Decoder: {decoder_path.name}")
    print(f"  Joint: {joint_path.name}")

    # Create sessions
    providers = ['CPUExecutionProvider']
    encoder = ort.InferenceSession(str(encoder_path), providers=providers)
    decoder = ort.InferenceSession(str(decoder_path), providers=providers)
    joint = ort.InferenceSession(str(joint_path), providers=providers)

    return encoder, decoder, joint, meta

def load_test_swipes(data_path="../data/train_final_val.jsonl", num_samples=5):
    """Load some test swipes from training data."""
    samples = []
    with open(data_path, "r") as f:
        lines = f.readlines()
        # Get random samples
        selected = random.sample(lines, min(num_samples, len(lines)))
        for line in selected:
            sample = json.loads(line)
            samples.append(sample)
    return samples

def extract_features(trace, target_length=64):
    """Extract 37-dimensional features from trace."""
    # Adaptive resampling based on trace length
    original_length = len(trace)
    if original_length < 30:
        target_length = 56
    elif original_length < 50:
        target_length = 64
    elif original_length < 80:
        target_length = 80
    else:
        target_length = 96

    # Resample to target length
    if len(trace) != target_length:
        old_indices = np.arange(len(trace))
        new_indices = np.linspace(0, len(trace) - 1, target_length)

        f_x = interpolate.interp1d(old_indices, trace[:, 0], kind='linear')
        f_y = interpolate.interp1d(old_indices, trace[:, 1], kind='linear')
        f_t = interpolate.interp1d(old_indices, trace[:, 2], kind='linear')

        trace = np.column_stack([
            f_x(new_indices),
            f_y(new_indices),
            f_t(new_indices)
        ])

    # Apply smoothing
    trace[:, 0] = gaussian_filter1d(trace[:, 0], sigma=1.0)
    trace[:, 1] = gaussian_filter1d(trace[:, 1], sigma=1.0)

    # Compute velocities
    dt = np.diff(trace[:, 2])
    dt[dt == 0] = 1  # Prevent division by zero

    vx = np.diff(trace[:, 0]) / (dt / 1000.0)  # Convert ms to s
    vy = np.diff(trace[:, 1]) / (dt / 1000.0)
    v_mag = np.sqrt(vx**2 + vy**2)

    # Pad velocities
    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])
    v_mag = np.concatenate([[v_mag[0]], v_mag])

    # Compute accelerations
    dt_v = dt
    ax = np.diff(vx) / (dt_v / 1000.0)
    ay = np.diff(vy) / (dt_v / 1000.0)
    a_mag = np.sqrt(ax**2 + ay**2)

    # Pad accelerations
    ax = np.concatenate([[ax[0]], ax])
    ay = np.concatenate([[ay[0]], ay])
    a_mag = np.concatenate([[a_mag[0]], a_mag])

    # Compute angles
    angles = np.arctan2(vy, vx)

    # Compute angular velocity
    angle_diff = np.diff(angles)
    angle_diff = np.where(angle_diff > np.pi, angle_diff - 2*np.pi, angle_diff)
    angle_diff = np.where(angle_diff < -np.pi, angle_diff + 2*np.pi, angle_diff)
    angular_velocity = angle_diff / (dt / 1000.0)
    angular_velocity = np.concatenate([[0], angular_velocity])

    # Compute path features
    cumulative_dist = np.zeros(len(trace))
    for i in range(1, len(trace)):
        dx = trace[i, 0] - trace[i-1, 0]
        dy = trace[i, 1] - trace[i-1, 1]
        cumulative_dist[i] = cumulative_dist[i-1] + np.sqrt(dx**2 + dy**2)

    # Relative position from start
    rel_x_start = trace[:, 0] - trace[0, 0]
    rel_y_start = trace[:, 1] - trace[0, 1]

    # Relative position from end
    rel_x_end = trace[:, 0] - trace[-1, 0]
    rel_y_end = trace[:, 1] - trace[-1, 1]

    # Curvature (simplified)
    curvature = np.zeros(len(trace))
    for i in range(1, len(trace) - 1):
        v1 = trace[i] - trace[i-1]
        v2 = trace[i+1] - trace[i]
        angle = np.arccos(np.clip(np.dot(v1[:2], v2[:2]) /
                          (np.linalg.norm(v1[:2]) * np.linalg.norm(v2[:2]) + 1e-8), -1, 1))
        curvature[i] = angle

    # Pressure simulation (based on speed)
    pressure = 1.0 - np.clip(v_mag / np.max(v_mag + 1e-8), 0, 0.5)

    # Direction changes
    direction_change = np.zeros(len(trace))
    direction_change[1:] = np.abs(angle_diff)

    # Time features
    time_norm = trace[:, 2] / (trace[-1, 2] + 1e-8)
    time_since_start = trace[:, 2] / 1000.0  # Convert to seconds

    # Segment features
    segment_progress = np.arange(len(trace)) / (len(trace) - 1)

    # Distance features
    dist_from_center = np.sqrt(trace[:, 0]**2 + trace[:, 1]**2)

    # Statistical features (repeated for each point)
    mean_x = np.full(len(trace), np.mean(trace[:, 0]))
    mean_y = np.full(len(trace), np.mean(trace[:, 1]))
    std_x = np.full(len(trace), np.std(trace[:, 0]))
    std_y = np.full(len(trace), np.std(trace[:, 1]))

    # Combine all features
    features = np.column_stack([
        trace[:, 0],  # x
        trace[:, 1],  # y
        vx,  # velocity x
        vy,  # velocity y
        v_mag,  # velocity magnitude
        ax,  # acceleration x
        ay,  # acceleration y
        a_mag,  # acceleration magnitude
        angles,  # angle
        angular_velocity,  # angular velocity
        cumulative_dist,  # cumulative distance
        rel_x_start,  # relative x from start
        rel_y_start,  # relative y from start
        rel_x_end,  # relative x from end
        rel_y_end,  # relative y from end
        curvature,  # curvature
        pressure,  # pressure
        direction_change,  # direction change
        time_norm,  # normalized time
        time_since_start,  # time since start
        segment_progress,  # segment progress
        dist_from_center,  # distance from center
        mean_x,  # mean x
        mean_y,  # mean y
        std_x,  # std x
        std_y,  # std y
    ])

    # Add placeholder features to reach 37 dimensions
    while features.shape[1] < 37:
        features = np.column_stack([features, np.zeros(len(trace))])

    # Normalize features
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features.astype(np.float32)

def preprocess_swipe(swipe_data):
    """Preprocess swipe data into model input features."""
    points = swipe_data["points"]
    word = swipe_data["word"]

    # Convert to numpy array
    trace = np.array([[p["x"], p["y"], p["t"]] for p in points])

    # Extract features (37D features)
    features = extract_features(trace)

    # Add batch dimension
    features = features[np.newaxis, :, :]  # [1, T, 37]

    # Create length tensor
    lengths = np.array([features.shape[1]], dtype=np.int64)

    return features.astype(np.float32), lengths, word

def greedy_decode(encoder, decoder, joint, features, lengths, meta, max_symbols=50):
    """Perform greedy decoding using the ONNX models."""
    blank_id = meta["blank_id"]
    tokens = meta["tokens"]

    # Transpose features to match expected shape [B, F, T] instead of [B, T, F]
    features = np.transpose(features, (0, 2, 1))

    # Encode audio
    encoded, encoded_lengths = encoder.run(None, {
        "audio_signal": features,
        "length": lengths
    })

    # Initialize decoder state (2 layers)
    batch_size = 1
    decoder_hidden = np.zeros((2, batch_size, 320), dtype=np.float32)
    decoder_cell = np.zeros((2, batch_size, 320), dtype=np.float32)

    # Start with blank token
    decoder_input = np.array([[blank_id]], dtype=np.int64)

    # Decode
    hypothesis = []
    time_idx = 0
    max_time = encoded_lengths[0]

    while time_idx < max_time and len(hypothesis) < max_symbols:
        # Get encoder slice for current time
        encoder_slice = encoded[:, time_idx:time_idx+1, :]  # [B, 1, D]

        # Run decoder
        decoder_out, decoder_hidden, decoder_cell = decoder.run(None, {
            "input_tokens": decoder_input,
            "h_in": decoder_hidden,
            "c_in": decoder_cell
        })

        # Run joint network
        logits = joint.run(None, {
            "encoder_output": encoder_slice,
            "decoder_output": decoder_out
        })[0]

        # Get prediction
        y = np.argmax(logits[0, 0, :])

        if y == blank_id:
            # Blank token - advance time
            time_idx += 1
        else:
            # Non-blank token - emit and update decoder input
            hypothesis.append(y)
            decoder_input = np.array([[y]], dtype=np.int64)

    # Convert token IDs to characters
    predicted_chars = []
    for token_id in hypothesis:
        if token_id < len(tokens) and token_id != blank_id:
            predicted_chars.append(tokens[token_id])

    return ''.join(predicted_chars)

def test_pipeline():
    """Test the full pipeline from swipe to prediction."""
    print("=" * 60)
    print("Testing Swipe-to-Prediction Pipeline")
    print("=" * 60)

    # Load models
    encoder, decoder, joint, meta = load_models()
    print(f"✓ Models loaded successfully")
    print(f"  Vocabulary size: {meta['vocab_size']}")
    print(f"  Blank ID: {meta['blank_id']}")
    print()

    # Load test swipes
    test_samples = load_test_swipes(num_samples=10)
    print(f"✓ Loaded {len(test_samples)} test swipes")
    print()

    # Test each sample
    correct = 0
    results = []

    for i, sample in enumerate(test_samples):
        features, lengths, true_word = preprocess_swipe(sample)
        predicted = greedy_decode(encoder, decoder, joint, features, lengths, meta)

        is_correct = predicted == true_word
        correct += is_correct

        result = {
            "true": true_word,
            "predicted": predicted,
            "correct": is_correct,
            "num_points": len(sample["points"])
        }
        results.append(result)

        # Print result
        status = "✓" if is_correct else "✗"
        print(f"{status} Sample {i+1}:")
        print(f"  True word: '{true_word}'")
        print(f"  Predicted: '{predicted}'")
        print(f"  Points: {result['num_points']}")
        print()

    # Print summary
    accuracy = correct / len(test_samples) * 100
    print("=" * 60)
    print(f"Summary:")
    print(f"  Correct: {correct}/{len(test_samples)} ({accuracy:.1f}%)")
    print("=" * 60)

    # Save results
    with open("test_results.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_samples),
            "results": results
        }, f, indent=2)
    print(f"✓ Results saved to test_results.json")

if __name__ == "__main__":
    test_pipeline()