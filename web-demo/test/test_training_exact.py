#!/usr/bin/env python3
"""
Test ONNX models by exactly mimicking the training pipeline
"""

import json
import numpy as np
import onnxruntime as ort
import math


def clamp(x, min_val, max_val):
    return max(min_val, min(max_val, x))


def determine_resample_target(original_len, cfg):
    """Determine resampling target based on original length - from training"""
    short_target = cfg.get("resample_short_target", 56)
    long_target = cfg.get("resample_long_target", 96)
    short_threshold = cfg.get("resample_short_threshold", 48)
    long_threshold = cfg.get("resample_long_threshold", 112)

    if original_len <= short_threshold:
        return short_target
    elif original_len >= long_threshold:
        return long_target
    else:
        ratio = (original_len - short_threshold) / (long_threshold - short_threshold)
        target = short_target + (long_target - short_target) * ratio
        return int(round(target))


def resample_points(points, target_length):
    """Resample points to target length - from training"""
    if len(points) == target_length:
        return points

    if len(points) == 1:
        # If we have only one point, duplicate it
        return [points[0]] * target_length

    # Linear interpolation
    result = []
    for i in range(target_length):
        # Map i to the original points
        pos = i * (len(points) - 1) / (target_length - 1)
        idx = int(pos)
        frac = pos - idx

        if idx >= len(points) - 1:
            result.append(points[-1])
        else:
            # Interpolate between points[idx] and points[idx+1]
            p1 = points[idx]
            p2 = points[idx + 1]
            interpolated = {
                'x': p1['x'] * (1 - frac) + p2['x'] * frac,
                'y': p1['y'] * (1 - frac) + p2['y'] * frac,
                't': p1['t'] * (1 - frac) + p2['t'] * frac
            }
            result.append(interpolated)

    return result


def extract_features(points):
    """Extract features EXACTLY as PersonalizedSwipeFeaturizer does"""
    n = len(points)
    features = np.zeros((n, 37), dtype=np.float32)

    # Key centers from training (in [-1, 1] space)
    key_centers = [
        ('q', -0.8, -0.6), ('w', -0.6, -0.6), ('e', -0.4, -0.6), ('r', -0.2, -0.6), ('t', 0.0, -0.6),
        ('y', 0.2, -0.6), ('u', 0.4, -0.6), ('i', 0.6, -0.6), ('o', 0.8, -0.6), ('p', 1.0, -0.6),
        ('a', -0.7, 0.0), ('s', -0.5, 0.0), ('d', -0.3, 0.0), ('f', -0.1, 0.0), ('g', 0.1, 0.0),
        ('h', 0.3, 0.0), ('j', 0.5, 0.0), ('k', 0.7, 0.0), ('l', 0.9, 0.0),
        ('z', -0.5, 0.6), ('x', -0.3, 0.6), ('c', -0.1, 0.6), ('v', 0.1, 0.6), ('b', 0.3, 0.6),
        ('n', 0.5, 0.6), ('m', 0.7, 0.6)
    ]

    for i in range(n):
        pt = points[i]
        x, y, t = pt['x'], pt['y'], pt['t']

        # Position features (already in [-1, 1])
        features[i, 0] = x
        features[i, 1] = y

        # Velocity features
        if i > 0:
            prev = points[i-1]
            dt = max(t - prev['t'], 1.0) / 1000.0  # Convert to seconds
            vx = (x - prev['x']) / dt
            vy = (y - prev['y']) / dt
            v_mag = math.sqrt(vx**2 + vy**2)
            features[i, 2] = clamp(vx, -10.0, 10.0)
            features[i, 3] = clamp(vy, -10.0, 10.0)
            features[i, 4] = clamp(v_mag, 0.0, 20.0)

        # Acceleration features
        if i > 1:
            prev = points[i-1]
            prev2 = points[i-2]
            dt = max(t - prev['t'], 1.0) / 1000.0
            dt_prev = max(prev['t'] - prev2['t'], 1.0) / 1000.0

            vx = (x - prev['x']) / dt
            vy = (y - prev['y']) / dt
            vx_prev = (prev['x'] - prev2['x']) / dt_prev
            vy_prev = (prev['y'] - prev2['y']) / dt_prev

            ax = (vx - vx_prev) / dt
            ay = (vy - vy_prev) / dt
            a_mag = math.sqrt(ax**2 + ay**2)

            features[i, 5] = clamp(ax, -50.0, 50.0)
            features[i, 6] = clamp(ay, -50.0, 50.0)
            features[i, 7] = clamp(a_mag, 0.0, 100.0)

        # Direction changes
        if i > 0 and i < n - 1:
            prev = points[i-1]
            next_pt = points[i+1]

            dx1 = x - prev['x']
            dy1 = y - prev['y']
            dx2 = next_pt['x'] - x
            dy2 = next_pt['y'] - y

            len1 = math.sqrt(dx1**2 + dy1**2)
            len2 = math.sqrt(dx2**2 + dy2**2)

            if len1 > 1e-6 and len2 > 1e-6:
                cos_angle = (dx1*dx2 + dy1*dy2) / (len1 * len2)
                cos_angle = clamp(cos_angle, -1.0, 1.0)
                angle = math.acos(cos_angle)
                features[i, 8] = angle

                # Curvature approximation
                cross = dx1 * dy2 - dy1 * dx2
                curvature = 2.0 * cross / (len1 * len2 + 1e-6)
                features[i, 9] = clamp(curvature, -10.0, 10.0)

        # Progress along path
        features[i, 10] = i / (n - 1) if n > 1 else 0.5

        # Distance to nearest key centers (top 5)
        distances = []
        for char, kx, ky in key_centers:
            dist = math.sqrt((x - kx)**2 + (y - ky)**2)
            distances.append((dist, char, kx, ky))
        distances.sort()

        for j, (dist, char, kx, ky) in enumerate(distances[:5]):
            features[i, 11 + j*2] = dist
            features[i, 12 + j*2] = 1.0 / (dist + 0.1)  # Inverse distance

        # Segment features
        if i > 0:
            segment_len = math.sqrt((x - points[i-1]['x'])**2 + (y - points[i-1]['y'])**2)
            features[i, 21] = clamp(segment_len, 0.0, 2.0)

        # Total distance traveled
        if i > 0:
            total_dist = features[i-1, 22] + features[i, 21]
            features[i, 22] = total_dist

        # Time features
        features[i, 23] = t / 1000.0  # Convert to seconds
        if i > 0:
            features[i, 24] = (t - points[i-1]['t']) / 1000.0

        # Start/end indicators
        features[i, 25] = 1.0 if i == 0 else 0.0
        features[i, 26] = 1.0 if i == n - 1 else 0.0

        # Speed consistency
        if i > 2:
            speeds = []
            for j in range(max(0, i-3), i):
                if j > 0:
                    dt = max(points[j]['t'] - points[j-1]['t'], 1.0) / 1000.0
                    dx = points[j]['x'] - points[j-1]['x']
                    dy = points[j]['y'] - points[j-1]['y']
                    speed = math.sqrt(dx**2 + dy**2) / dt
                    speeds.append(speed)
            if speeds:
                mean_speed = sum(speeds) / len(speeds)
                variance = sum((s - mean_speed)**2 for s in speeds) / len(speeds)
                features[i, 27] = math.sqrt(variance)

        # Local density (points within radius)
        radius = 0.15
        nearby_count = 0
        for j in range(n):
            if i != j:
                dist = math.sqrt((x - points[j]['x'])**2 + (y - points[j]['y'])**2)
                if dist < radius:
                    nearby_count += 1
        features[i, 28] = nearby_count

        # Directional features
        if i > 0:
            dx = x - points[i-1]['x']
            dy = y - points[i-1]['y']
            features[i, 29] = 1.0 if dx > 0.01 else 0.0  # Moving right
            features[i, 30] = 1.0 if dx < -0.01 else 0.0  # Moving left
            features[i, 31] = 1.0 if dy > 0.01 else 0.0  # Moving down
            features[i, 32] = 1.0 if dy < -0.01 else 0.0  # Moving up

        # Quadrant features
        features[i, 33] = 1.0 if x < 0 and y < 0 else 0.0  # Top-left
        features[i, 34] = 1.0 if x >= 0 and y < 0 else 0.0  # Top-right
        features[i, 35] = 1.0 if x < 0 and y >= 0 else 0.0  # Bottom-left
        features[i, 36] = 1.0 if x >= 0 and y >= 0 else 0.0  # Bottom-right

    return features


def get_companion_data():
    """Get companion swipe data from line 22440 efficiently"""
    import linecache
    data_path = '../../data/train_final_train.jsonl'
    line = linecache.getline(data_path, 22440)
    if line:
        data = json.loads(line)
        return data['points'], data['word']
    return None, None


def test_with_onnx():
    """Test using ONNX models with exact training pipeline"""
    print("="*70)
    print("TESTING WITH ONNX (EXACT TRAINING PIPELINE)")
    print("="*70)

    # Load models
    model_dir = '../models/best_latest'
    encoder_path = model_dir + '/encoder.onnx'
    decoder_path = model_dir + '/decoder_joint.onnx'
    meta_path = model_dir + '/runtime_meta.json'

    print(f"\nLoading models from: {model_dir}")
    encoder_session = ort.InferenceSession(encoder_path)
    decoder_session = ort.InferenceSession(decoder_path)

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    blank_id = meta['blank_id']
    vocab = meta['tokens']
    decoder_config = meta.get('decoder_config', {})
    num_layers = decoder_config.get('num_layers', 1)
    hidden_size = decoder_config.get('hidden_size', 192)

    print(f"Vocab size: {len(vocab)}, Blank ID: {blank_id}")
    print(f"Decoder config: {num_layers} layers, {hidden_size} hidden")

    # Get companion data
    points, expected_word = get_companion_data()
    if points is None:
        print("ERROR: Could not load companion data")
        return

    print(f"\nTesting word: '{expected_word}' ({len(points)} points)")

    # Process swipe EXACTLY as training does
    # 1. Transform coordinates from [0, 1] to [-1, 1]
    transformed_points = []
    for pt in points:
        transformed_points.append({
            'x': pt['x'] * 2.0 - 1.0,
            'y': pt['y'] * 2.0 - 1.0,
            't': pt['t']
        })

    print(f"Original first point: x={points[0]['x']:.4f}, y={points[0]['y']:.4f}")
    print(f"Transformed first point: x={transformed_points[0]['x']:.4f}, y={transformed_points[0]['y']:.4f}")

    # 2. Determine resample target
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }
    target_len = determine_resample_target(len(transformed_points), preprocess_cfg)
    print(f"Original length: {len(transformed_points)}, Target length: {target_len}")

    # 3. Resample points
    resampled_points = resample_points(transformed_points, target_len)
    print(f"Resampled to {len(resampled_points)} points")

    # 4. Extract features
    features = extract_features(resampled_points)
    print(f"Features shape: {features.shape}")
    print(f"Feature ranges - min: {features.min():.4f}, max: {features.max():.4f}")
    print(f"First 5 features of first frame: {features[0, :5]}")

    # Run encoder
    print("\n--- Running Encoder ---")
    # Encoder expects [batch, features, time] not [batch, time, features]
    signal = features.astype(np.float32).T.reshape(1, 37, -1)
    signal_len = np.array([features.shape[0]], dtype=np.int64)

    encoder_outputs = encoder_session.run(None, {
        'audio_signal': signal,
        'length': signal_len
    })
    encoded = encoder_outputs[0]
    encoded_len = encoder_outputs[1]

    print(f"Encoded shape: {encoded.shape}")
    print(f"Encoded length: {encoded_len}")
    print(f"Encoded stats - min: {encoded.min():.4f}, max: {encoded.max():.4f}, mean: {encoded.mean():.4f}")

    # Run greedy decoding
    print("\n--- Running Greedy Decoding ---")

    # Initialize decoder states
    state_h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    state_c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)

    # Start with BOS (use index 0 which maps to <blank> in predictor space)
    y = np.array([[0]], dtype=np.int32)

    predictions = []
    max_symbols = 24

    # Encoder output is already [batch, encoder_dim, time]

    for t in range(min(5, encoded_len[0])):  # Just first 5 frames for debugging
        enc_frame = encoded[:, :, t:t+1]

        print(f"\nFrame {t}:")
        for symbol_idx in range(min(3, max_symbols)):  # Just first 3 symbols
            # Run decoder
            decoder_outputs = decoder_session.run(None, {
                'targets': y,
                'input_states_1': state_h,
                'input_states_2': state_c,
                'encoder_outputs': enc_frame,
                'target_length': np.array([1], dtype=np.int32)
            })

            # Get logits and next states
            logits = decoder_outputs[0][0, 0, :]  # Shape: [vocab_size]
            state_h = decoder_outputs[2]
            state_c = decoder_outputs[3]

            # Get top 5 predictions
            top5_idx = np.argsort(logits)[-5:][::-1]
            top5_scores = logits[top5_idx]
            top5_chars = []
            for i, s in zip(top5_idx, top5_scores):
                char = vocab[int(i)] if int(i) < len(vocab) else '?'
                top5_chars.append((char, f'{s:.2f}'))
            print(f"  Symbol {symbol_idx}: y={y[0,0]}, Top 5: {top5_chars}")

            # Get prediction
            pred_idx = np.argmax(logits)

            if pred_idx == blank_id:
                # Emit blank and move to next frame
                print(f"  -> Blank emitted, moving to next frame")
                break
            else:
                # Emit character
                predictions.append(pred_idx)
                print(f"  -> Emitting '{vocab[pred_idx]}'")
                # Map from joint vocab to predictor vocab for next input
                if pred_idx < blank_id:
                    next_y = pred_idx
                else:
                    next_y = pred_idx - 1
                y = np.array([[next_y]], dtype=np.int32)

    # Convert predictions to text
    pred_text = ''.join([vocab[idx] if idx < len(vocab) else '?' for idx in predictions])

    print(f"\nPredictions: {predictions}")
    print(f"Predicted text: '{pred_text}'")
    print(f"Expected: '{expected_word}'")

    if pred_text == expected_word:
        print("\n✅ SUCCESS!")
    else:
        print("\n❌ FAILED")


def main():
    try:
        test_with_onnx()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()