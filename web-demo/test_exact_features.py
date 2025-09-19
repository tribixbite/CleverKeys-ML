#!/usr/bin/env python3
"""
Test decoder with EXACT feature extraction from training code.
"""

import json
import math
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clamp(x: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, x))


def build_correct_key_positions():
    """Build the EXACT key positions used in training."""
    layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    positions = {}
    key_centers = []

    for row_idx, row in enumerate(layout):
        for col_idx, char in enumerate(row):
            # This is EXACTLY how training computes positions
            x01 = (col_idx + 0.5) / 10.0  # In [0,1]
            y01 = (row_idx + 0.5) / 3.0   # In [0,1]
            # Convert to [-1,1] for training
            x = x01 * 2.0 - 1.0
            y = y01 * 2.0 - 1.0
            positions[char] = (x, y)
            key_centers.append((char, x, y))

    return positions, key_centers


class ExactFeatureRNNTDecoder:
    """RNN-T decoder with exact feature extraction from training"""

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
        _, self.key_centers = build_correct_key_positions()

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
        This replicates the exact feature extraction from train_transducer_personalized.py
        """
        total = len(points)
        point = points[idx]
        prev = points[idx - 1] if idx > 0 else None
        prev2 = points[idx - 2] if idx > 1 else None

        # Get current position (already in [-1,1] from correct keyboard layout)
        x = clamp(float(point.get("x", 0.0)), -1.0, 1.0)
        y = clamp(float(point.get("y", 0.0)), -1.0, 1.0)
        t_ms = float(point.get("t", idx * 10.0))
        t_seconds = t_ms / 1000.0

        # Velocity
        vx = vy = 0.0
        if prev is not None:
            prev_t = float(prev.get("t", (idx - 1) * 10.0))
            dt = max((t_ms - prev_t) / 1000.0, 0.001)
            prev_x = clamp(float(prev.get("x", x)), -1.0, 1.0)
            prev_y = clamp(float(prev.get("y", y)), -1.0, 1.0)
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
            prev_x = clamp(float(prev.get("x", x)), -1.0, 1.0)
            prev_y = clamp(float(prev.get("y", y)), -1.0, 1.0)
            prev2_x = clamp(float(prev2.get("x", prev_x)), -1.0, 1.0)
            prev2_y = clamp(float(prev2.get("y", prev_y)), -1.0, 1.0)
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
            prev_x = clamp(float(prev.get("x", x)), -1.0, 1.0)
            prev_y = clamp(float(prev.get("y", y)), -1.0, 1.0)
            prev2_x = clamp(float(prev2.get("x", prev_x)), -1.0, 1.0)
            prev2_y = clamp(float(prev2.get("y", prev_y)), -1.0, 1.0)
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
            xs = [clamp(float(p.get("x", x)), -1.0, 1.0) for p in win_pts]
            ys = [clamp(float(p.get("y", y)), -1.0, 1.0) for p in win_pts]
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

    def decode_beam_search(self, points: List[dict], beam_size: int = 8) -> List[Tuple[str, float]]:
        """Full beam search decoding with exact features"""
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

        # Initialize beams
        beams = [{
            'y_prev': self.blank_id,
            'h': np.zeros((self.L, 1, self.H), dtype=np.float32),
            'c': np.zeros((self.L, 1, self.H), dtype=np.float32),
            'trie': self.trie_root,
            'logp': 0.0,
            'chars': []
        }]

        # Time-synchronous beam search
        for t in range(T_out):
            for s in range(20):  # Max symbols per frame
                # Sort and prune beams
                beams.sort(key=lambda b: b['logp'], reverse=True)
                active = beams[:beam_size]
                N = len(active)

                # Prepare batch
                y_prev = np.array([b['y_prev'] for b in active], dtype=np.int64)
                h0 = np.concatenate([b['h'] for b in active], axis=1)
                c0 = np.concatenate([b['c'] for b in active], axis=1)
                enc_t = np.repeat(encoded_btf[0, t:t+1, :], N, axis=0)

                # Run decoder
                outputs = self.decoder_session.run(None, {
                    'y_prev': y_prev,
                    'h0': h0,
                    'c0': c0,
                    'enc_t': enc_t
                })

                logits = outputs[0]
                h1 = outputs[1]
                c1 = outputs[2]

                # Reshape logits if needed
                if len(logits.shape) == 4:  # (1, 1, 1, vocab_size)
                    logits = logits.squeeze()
                elif len(logits.shape) == 3:  # (1, 1, vocab_size) or (1, N, vocab_size)
                    if logits.shape[0] == 1 and logits.shape[1] == 1:
                        logits = logits.squeeze()
                    elif logits.shape[0] == 1:
                        logits = logits.squeeze(0)
                elif len(logits.shape) == 2 and logits.shape[0] == 1:  # (1, vocab_size)
                    logits = logits.squeeze(0)

                # Ensure logits is 2D: (N, vocab_size)
                if len(logits.shape) == 1:
                    logits = logits[np.newaxis, :]

                # Apply log softmax
                logits_max = np.max(logits, axis=-1, keepdims=True)
                logits = logits - logits_max
                exp_logits = np.exp(logits)
                logits = np.log(exp_logits / np.sum(exp_logits, axis=-1, keepdims=True) + 1e-10)

                # Expand beams
                next_beams = []
                for i, beam in enumerate(active):
                    # Get logits for this beam
                    if logits.shape[0] == 1:
                        beam_logits = logits[0]
                    else:
                        beam_logits = logits[i] if i < logits.shape[0] else logits[0]

                    # Blank transition
                    lp_blank = beam_logits[self.blank_id]
                    next_beams.append({
                        'y_prev': self.blank_id,
                        'h': h1[:, i:i+1, :],
                        'c': c1[:, i:i+1, :],
                        'trie': beam['trie'],
                        'logp': beam['logp'] + lp_blank,
                        'chars': beam['chars']
                    })

                    # Character transitions
                    allowed = list(beam['trie']['children'].keys())
                    if allowed:
                        char_scores = [(cid, beam_logits[cid]) for cid in allowed]
                        char_scores.sort(key=lambda x: x[1], reverse=True)

                        for cid, score in char_scores[:4]:  # Prune per beam
                            child = beam['trie']['children'][cid]
                            next_beams.append({
                                'y_prev': cid,
                                'h': h1[:, i:i+1, :],
                                'c': c1[:, i:i+1, :],
                                'trie': child,
                                'logp': beam['logp'] + score,
                                'chars': beam['chars'] + [cid]
                            })

                # Update beams
                next_beams.sort(key=lambda b: b['logp'], reverse=True)
                beams = next_beams[:beam_size]

                # Early stop if best beam is blank
                if beams[0]['y_prev'] == self.blank_id:
                    break

        # Collect completed words
        results = []
        seen = set()

        for beam in beams:
            if beam['trie']['is_word']:
                word_id = beam['trie']['word_id']
                if word_id not in seen:
                    seen.add(word_id)
                    word = self.words[word_id]
                    results.append((word, beam['logp']))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:5]


def test_exact_features():
    """Test with exact feature extraction from training"""

    # Try personalized models first
    print("Testing with personalized models...")
    decoder = ExactFeatureRNNTDecoder(
        encoder_path='personalized/encoder_int8_qdq.onnx',
        decoder_path='personalized/rnnt_step_fp32.onnx',
        runtime_meta_path='../trained_models/nema1/runtime_meta.json',
        words_path='../trained_models/nema1/words.txt'
    )

    # Use CORRECT key positions from training
    key_positions, _ = build_correct_key_positions()

    print("\n" + "="*70)
    print("TESTING WITH EXACT FEATURE EXTRACTION FROM TRAINING")
    print("="*70)

    def generate_swipe(word: str) -> List[dict]:
        """Generate swipe with correct key positions in [-1,1] range"""
        points = []
        t = 0
        for i, char in enumerate(word.lower()):
            if char in key_positions:
                x, y = key_positions[char]  # Already in [-1,1] range
                # Add multiple points with slight noise
                num_points = 5 if i in [0, len(word)-1] else 3
                for j in range(num_points):
                    points.append({
                        'x': x + np.random.uniform(-0.01, 0.01),
                        'y': y + np.random.uniform(-0.01, 0.01),
                        't': t
                    })
                    t += 10
        return points

    # Test words
    test_words = ['hello', 'world', 'test', 'the', 'and', 'good', 'time']

    success_count = 0
    for word in test_words:
        print(f"\nTesting: '{word}'")
        points = generate_swipe(word)

        # Show first point coordinates
        if points:
            x, y = points[0]['x'], points[0]['y']
            print(f"  First key: {word[0]} at ({x:.3f}, {y:.3f}) in [-1,1] range")

        # Decode
        predictions = decoder.decode_beam_search(points)

        print(f"  Predictions:")
        found = False
        for i, (pred_word, score) in enumerate(predictions[:5]):
            marker = "✓" if pred_word == word else " "
            if pred_word == word:
                found = True
                success_count += 1
            print(f"    {marker} {i+1}. {pred_word:15s} (score: {score:.2f})")

        if not found and predictions:
            print(f"    ✗ Expected '{word}' not in top 5")

    print(f"\n" + "="*70)
    print(f"ACCURACY: {success_count}/{len(test_words)} = {success_count/len(test_words)*100:.1f}%")
    print("="*70)


if __name__ == "__main__":
    test_exact_features()