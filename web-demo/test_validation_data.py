#!/usr/bin/env python3
"""
Test decoder with actual validation data from training.
"""

import json
import math
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clamp(x: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, x))


def build_correct_key_positions():
    """Build the EXACT key positions used in training."""
    layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    key_centers = []

    for row_idx, row in enumerate(layout):
        for col_idx, char in enumerate(row):
            # Training computation
            x01 = (col_idx + 0.5) / 10.0  # In [0,1]
            y01 = (row_idx + 0.5) / 3.0   # In [0,1]
            # Convert to [-1,1] for distance computation
            x = x01 * 2.0 - 1.0
            y = y01 * 2.0 - 1.0
            key_centers.append((char, x, y))

    return key_centers


class ValidationDataDecoder:
    """RNN-T decoder for testing with validation data"""

    def __init__(self, encoder_path: str, decoder_path: str, runtime_meta_path: str, words_path: str):
        """Initialize decoder with ONNX models"""

        # Load models
        print(f"Loading encoder: {encoder_path}")
        self.encoder_session = ort.InferenceSession(
            encoder_path,
            providers=['CPUExecutionProvider']
        )

        print(f"Loading decoder: {decoder_path}")
        self.decoder_session = ort.InferenceSession(
            decoder_path,
            providers=['CPUExecutionProvider']
        )

        # Load runtime metadata
        with open(runtime_meta_path, 'r') as f:
            meta = json.load(f)
            self.blank_id = meta['blank_id']
            self.unk_id = meta['unk_id']
            self.char_to_id = meta['char_to_id']
            self.id_to_char = {int(k): v for k, v in meta['id_to_char'].items()}
            self.vocab_size = meta['vocab_size']

        # Load words
        with open(words_path, 'r') as f:
            self.words = [line.strip() for line in f]

        # Build trie
        self.trie_root = self._build_trie()

        # Get key centers for distance features
        self.key_centers = build_correct_key_positions()

        print(f"Loaded {len(self.words)} words, vocab size: {self.vocab_size}")

        # Model dimensions
        self.L = 2  # LSTM layers
        self.H = 320  # Hidden size
        self.D = 256  # Encoder dimension
        self.V_output = 30  # Actual output size

    def _build_trie(self) -> dict:
        """Build trie from word list"""
        root = {"children": {}, "is_word": False, "word_id": -1}
        kept = 0

        for word_id, word in enumerate(self.words):
            word = word.lower().replace("'", "'")

            if not all(ch in self.char_to_id for ch in word):
                continue

            cur = root
            for ch in word:
                cid = self.char_to_id[ch]
                if cid not in cur["children"]:
                    cur["children"][cid] = {"children": {}, "is_word": False, "word_id": -1}
                cur = cur["children"][cid]

            cur["is_word"] = True
            cur["word_id"] = word_id
            kept += 1

        print(f"Trie built: {kept}/{len(self.words)} words")
        return root

    def compute_feature_exact(self, points: List[dict], idx: int) -> np.ndarray:
        """
        Compute EXACT 37D features as done in training.
        CRITICAL: Points are in [0,1] range from validation data,
        must convert to [-1,1] for feature computation!
        """
        total = len(points)
        point = points[idx]
        prev = points[idx - 1] if idx > 0 else None
        prev2 = points[idx - 2] if idx > 1 else None

        # Convert from [0,1] to [-1,1] range
        x01 = float(point.get("x", 0.5))
        y01 = float(point.get("y", 0.5))
        x = clamp(x01 * 2.0 - 1.0, -1.0, 1.0)
        y = clamp(y01 * 2.0 - 1.0, -1.0, 1.0)
        t_ms = float(point.get("t", idx * 10.0))
        t_seconds = t_ms / 1000.0

        # Velocity
        vx = vy = 0.0
        if prev is not None:
            prev_t = float(prev.get("t", (idx - 1) * 10.0))
            dt = max((t_ms - prev_t) / 1000.0, 0.001)
            prev_x01 = float(prev.get("x", x01))
            prev_y01 = float(prev.get("y", y01))
            prev_x = clamp(prev_x01 * 2.0 - 1.0, -1.0, 1.0)
            prev_y = clamp(prev_y01 * 2.0 - 1.0, -1.0, 1.0)
            vx = (x - prev_x) / dt
            vy = (y - prev_y) / dt

        speed = math.hypot(vx, vy)

        # Acceleration
        ax = ay = acc = 0.0
        if prev is not None and prev2 is not None:
            prev_t = float(prev.get("t", (idx - 1) * 10.0))
            prev2_t = float(prev2.get("t", (idx - 2) * 10.0))
            dt1 = max((t_ms - prev_t) / 1000.0, 0.001)
            dt2 = max((prev_t - prev2_t) / 1000.0, 0.001)

            prev_x01 = float(prev.get("x", x01))
            prev_y01 = float(prev.get("y", y01))
            prev_x = clamp(prev_x01 * 2.0 - 1.0, -1.0, 1.0)
            prev_y = clamp(prev_y01 * 2.0 - 1.0, -1.0, 1.0)

            prev2_x01 = float(prev2.get("x", prev_x01))
            prev2_y01 = float(prev2.get("y", prev_y01))
            prev2_x = clamp(prev2_x01 * 2.0 - 1.0, -1.0, 1.0)
            prev2_y = clamp(prev2_y01 * 2.0 - 1.0, -1.0, 1.0)

            vx_prev = (prev_x - prev2_x) / dt2
            vy_prev = (prev_y - prev2_y) / dt2
            ax = (vx - vx_prev) / dt1
            ay = (vy - vy_prev) / dt1
            acc = math.hypot(ax, ay)

        # Angle features
        angle = math.atan2(vy, vx) if prev is not None else 0.0
        angle_sin = math.sin(angle)
        angle_cos = math.cos(angle)

        # Curvature
        curvature = 0.0
        if prev is not None and prev2 is not None:
            prev_x01 = float(prev.get("x", x01))
            prev_y01 = float(prev.get("y", y01))
            prev_x = clamp(prev_x01 * 2.0 - 1.0, -1.0, 1.0)
            prev_y = clamp(prev_y01 * 2.0 - 1.0, -1.0, 1.0)

            prev2_x01 = float(prev2.get("x", prev_x01))
            prev2_y01 = float(prev2.get("y", prev_y01))
            prev2_x = clamp(prev2_x01 * 2.0 - 1.0, -1.0, 1.0)
            prev2_y = clamp(prev2_y01 * 2.0 - 1.0, -1.0, 1.0)

            prev_angle = math.atan2(prev_y - prev2_y, prev_x - prev2_x)
            curvature = angle - prev_angle
            while curvature > math.pi:
                curvature -= 2 * math.pi
            while curvature < -math.pi:
                curvature += 2 * math.pi

        # Distances to nearest keys (top 5)
        distances = []
        for _, kx, ky in self.key_centers:
            distances.append(math.hypot(x - kx, y - ky))
        distances.sort()
        key_distances = distances[:5]
        while len(key_distances) < 5:
            key_distances.append(1.0)

        # Progress and position flags
        progress = idx / max(total - 1, 1)
        is_start = 1.0 if idx == 0 else 0.0
        is_end = 1.0 if idx == total - 1 else 0.0

        # Window statistics
        window_size = 5
        half = window_size // 2
        win_pts = points[max(0, idx - half): min(total, idx + half + 1)]
        if len(win_pts) > 1:
            xs = []
            ys = []
            for p in win_pts:
                px01 = float(p.get("x", x01))
                py01 = float(p.get("y", y01))
                xs.append(clamp(px01 * 2.0 - 1.0, -1.0, 1.0))
                ys.append(clamp(py01 * 2.0 - 1.0, -1.0, 1.0))
            mean_x = float(np.mean(xs))
            std_x = float(np.std(xs))
            mean_y = float(np.mean(ys))
            std_y = float(np.std(ys))
            range_x = max(xs) - min(xs)
            range_y = max(ys) - min(ys)
        else:
            mean_x = x
            std_x = 0.0
            mean_y = y
            std_y = 0.0
            range_x = 0.0
            range_y = 0.0

        features = [
            x,
            y,
            t_seconds,
            vx,
            vy,
            speed,
            ax,
            ay,
            acc,
            angle,
            angle_sin,
            angle_cos,
            curvature,
            *key_distances,
            progress,
            is_start,
            is_end,
            mean_x,
            std_x,
            mean_y,
            std_y,
            range_x,
            range_y,
        ]

        while len(features) < 37:
            features.append(0.0)

        return np.array(features[:37], dtype=np.float32)

    def compute_features_batch(self, points: List[dict]) -> np.ndarray:
        """Compute features for all points using exact method"""
        T = len(points)
        features = np.zeros((T, 37), dtype=np.float32)

        for i in range(T):
            features[i] = self.compute_feature_exact(points, i)

        return features

    def decode_greedy(self, points: List[dict]) -> str:
        """Simple greedy decoding for testing"""
        features = self.compute_features_batch(points)
        features_bft = features.T[np.newaxis, :, :]

        # Run encoder
        T = features_bft.shape[2]
        encoder_outputs = self.encoder_session.run(None, {
            'features_bft': features_bft.astype(np.float32),
            'lengths': np.array([T], dtype=np.int32)
        })

        encoded_btf = encoder_outputs[0]

        # Handle dimension transpose if needed
        if len(encoded_btf.shape) == 3 and encoded_btf.shape[1] == self.D:
            encoded_btf = np.transpose(encoded_btf, (0, 2, 1))

        T_out = encoded_btf.shape[1]

        # Greedy decoding
        h = np.zeros((self.L, 1, self.H), dtype=np.float32)
        c = np.zeros((self.L, 1, self.H), dtype=np.float32)
        y_prev = self.blank_id

        decoded_chars = []
        for t in range(T_out):
            enc_t = encoded_btf[0, t:t+1, :]

            outputs = self.decoder_session.run(None, {
                'y_prev': np.array([y_prev], dtype=np.int64),
                'h0': h,
                'c0': c,
                'enc_t': enc_t
            })

            logits = outputs[0]
            h = outputs[1]
            c = outputs[2]

            # Reshape logits
            while len(logits.shape) > 1 and logits.shape[0] == 1:
                logits = logits.squeeze(0)
            if len(logits.shape) > 1:
                logits = logits[0]

            # Get top prediction
            valid_vocab_size = min(self.vocab_size, logits.shape[0])
            probs = logits[:valid_vocab_size]

            # Apply softmax
            probs_exp = np.exp(probs - np.max(probs))
            probs_softmax = probs_exp / np.sum(probs_exp)

            # Get top char
            char_id = np.argmax(probs_softmax)

            if char_id != self.blank_id and char_id in self.id_to_char:
                decoded_chars.append(self.id_to_char[char_id])
                y_prev = char_id
            else:
                y_prev = self.blank_id

        return ''.join(decoded_chars)


def test_validation_data():
    """Test with actual validation data from training"""

    # Try personalized models
    print("Testing with personalized models...")
    decoder = ValidationDataDecoder(
        encoder_path='personalized/encoder_int8_qdq.onnx',
        decoder_path='personalized/rnnt_step_fp32.onnx',
        runtime_meta_path='../trained_models/nema1/runtime_meta.json',
        words_path='../trained_models/nema1/words.txt'
    )

    print("\n" + "="*70)
    print("TESTING WITH ACTUAL VALIDATION DATA")
    print("="*70)

    # Load validation data
    val_file = '../trained_models/nema1/personalized_tuning/20250918_105357/val_short_common.jsonl'

    results = []
    with open(val_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:  # Test first 10 samples
                break

            data = json.loads(line)
            word = data['word']
            points = data['points']

            # Decode
            predicted = decoder.decode_greedy(points)

            is_correct = predicted == word
            results.append((word, predicted, is_correct))

            marker = "✓" if is_correct else "✗"
            print(f"{marker} Word: '{word:10s}' -> Predicted: '{predicted:10s}'")

    # Calculate accuracy
    correct = sum(1 for _, _, is_correct in results if is_correct)
    total = len(results)

    print(f"\n" + "="*70)
    print(f"ACCURACY: {correct}/{total} = {correct/total*100:.1f}%")
    print("="*70)


if __name__ == "__main__":
    test_validation_data()