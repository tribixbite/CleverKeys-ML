#!/usr/bin/env python3
"""
Test decoder with proper beam search
"""

import json
import math
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import preprocessing functions from test_with_resampling
from test_with_resampling import (
    clamp, determine_resample_target, resample_points,
    normalize_points, build_correct_key_positions
)


class BeamSearchDecoder:
    """RNN-T decoder with proper beam search"""

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
            # Use original blank_id from metadata
            self.blank_id = meta['blank_id']  # Should be 0
            self.unk_id = meta['unk_id']
            self.char_to_id = meta['char_to_id']
            self.id_to_char = {int(k): v for k, v in meta['id_to_char'].items()}
            self.vocab_size = 30  # Actual vocab size

        # Load words
        with open(words_path, 'r') as f:
            self.words = [line.strip() for line in f]

        # Build trie
        self.trie_root = self._build_trie()

        # Get key centers for distance features
        self.key_centers = build_correct_key_positions()

        print(f"Loaded {len(self.words)} words")

        # Model dimensions
        self.L = 2  # LSTM layers
        self.H = 320  # Hidden size
        self.D = 256  # Encoder dimension

    def _build_trie(self) -> dict:
        """Build trie from word list"""
        root = {"ch": {}, "is": False, "wid": -1}
        kept = 0

        for wid, word in enumerate(self.words):
            word = word.lower().replace("'", "'")

            if not all(ch in self.char_to_id for ch in word):
                continue

            cur = root
            for ch in word:
                cid = self.char_to_id[ch]
                if cid not in cur["ch"]:
                    cur["ch"][cid] = {"ch": {}, "is": False, "wid": -1}
                cur = cur["ch"][cid]

            cur["is"] = True
            cur["wid"] = wid
            kept += 1

        print(f"Trie built: {kept}/{len(self.words)} words")
        return root

    def compute_feature_exact(self, points: List[dict], idx: int) -> np.ndarray:
        """Compute EXACT 37D features as done in training"""
        total = len(points)
        point = points[idx]
        prev = points[idx - 1] if idx > 0 else None
        prev2 = points[idx - 2] if idx > 1 else None

        # Points are already in [-1,1] after normalization
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
            x, y, t_seconds, vx, vy, speed, ax, ay, acc,
            angle, angle_sin, angle_cos, curvature,
            *key_distances,
            progress, is_start, is_end,
            mean_x, std_x, mean_y, std_y, range_x, range_y,
        ]

        while len(features) < 37:
            features.append(0.0)

        return np.array(features[:37], dtype=np.float32)

    def compute_features_batch(self, points: List[dict]) -> np.ndarray:
        """Compute features for all points"""
        T = len(points)
        features = np.zeros((T, 37), dtype=np.float32)
        for i in range(T):
            features[i] = self.compute_feature_exact(points, i)
        return features

    def preprocess_points(self, raw_points: List[dict]) -> List[dict]:
        """Apply full preprocessing pipeline"""
        # 1. Normalize from [0,1] to [-1,1] and adjust time
        normalized = normalize_points(raw_points)
        # 2. Determine resample target
        target_len = determine_resample_target(len(normalized))
        # 3. Resample points
        resampled = resample_points(normalized, target_len)
        return resampled

    def beam_search(self, raw_points: List[dict], beam_size: int = 16, max_sym: int = 20) -> List[Tuple[str, float]]:
        """Beam search with lexicon constraints"""
        # Preprocess
        processed_points = self.preprocess_points(raw_points)
        features = self.compute_features_batch(processed_points)
        features_bft = features.T[np.newaxis, :, :]

        # Run encoder
        T = features_bft.shape[2]
        encoder_outputs = self.encoder_session.run(None, {
            'features_bft': features_bft.astype(np.float32),
            'lengths': np.array([T], dtype=np.int32)
        })

        # Extract encoder output - shape is (B, D, T_out)
        encoded_bdt = encoder_outputs[0]
        # Get first batch element and transpose to (T_out, D)
        enc_btf = encoded_bdt[0].T
        T_out, _ = enc_btf.shape

        # Initialize beam
        beams = [{
            "y": self.blank_id,
            "h": np.zeros((self.L, 1, self.H), dtype=np.float32),
            "c": np.zeros((self.L, 1, self.H), dtype=np.float32),
            "tr": self.trie_root,
            "lp": 0.0,
            "chars": []
        }]

        # Main beam search loop
        for t in range(T_out):
            for s in range(max_sym):
                # Sort and prune beams
                beams.sort(key=lambda b: b["lp"], reverse=True)
                active = beams[:beam_size]
                N = len(active)

                # Prepare batch
                y_prev = np.array([b["y"] for b in active], dtype=np.int64)
                h0 = np.concatenate([b["h"] for b in active], axis=1)
                c0 = np.concatenate([b["c"] for b in active], axis=1)
                enc_t = np.repeat(enc_btf[t:t+1, :], N, axis=0)

                # Run decoder
                outputs = self.decoder_session.run(None, {
                    'y_prev': y_prev,
                    'h0': h0,
                    'c0': c0,
                    'enc_t': enc_t
                })

                logits = outputs[0]  # Shape: (N, 1, 1, 30)
                h1 = outputs[1]
                c1 = outputs[2]

                # Process logits
                logits = logits.squeeze()  # Remove extra dims
                if len(logits.shape) == 1:
                    logits = logits[np.newaxis, :]

                # Apply log softmax
                logits = logits - np.max(logits, axis=-1, keepdims=True)
                logp = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))

                # Expand beams
                next_beams = []
                for i, beam in enumerate(active):
                    # Blank transition
                    lp_blank = logp[i, self.blank_id] if logp.ndim > 1 else logp[self.blank_id]
                    next_beams.append({
                        "y": self.blank_id,
                        "h": h1[:, i:i+1, :],
                        "c": c1[:, i:i+1, :],
                        "tr": beam["tr"],
                        "lp": beam["lp"] + lp_blank,
                        "chars": beam["chars"]
                    })

                    # Character transitions (only valid ones from trie)
                    for cid in beam["tr"]["ch"].keys():
                        lp_char = logp[i, cid] if logp.ndim > 1 else logp[cid]
                        next_beams.append({
                            "y": cid,
                            "h": h1[:, i:i+1, :],
                            "c": c1[:, i:i+1, :],
                            "tr": beam["tr"]["ch"][cid],
                            "lp": beam["lp"] + lp_char,
                            "chars": beam["chars"] + [cid]
                        })

                # Update beams
                beams = next_beams

                # Early stop if best beam is blank
                if beams and beams[0]["y"] == self.blank_id:
                    break

        # Collect completed words
        results = []
        seen = set()

        for beam in beams:
            if beam["tr"]["is"]:
                wid = beam["tr"]["wid"]
                if wid not in seen:
                    seen.add(wid)
                    word = self.words[wid]
                    results.append((word, beam["lp"]))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:10]


def test_beam_search():
    """Test with proper beam search"""

    print("Testing with FRESH models and BEAM SEARCH...")
    decoder = BeamSearchDecoder(
        encoder_path='encoder_fresh.onnx',
        decoder_path='rnnt_step_fresh.onnx',
        runtime_meta_path='../trained_models/nema1/runtime_meta.json',
        words_path='../trained_models/nema1/words.txt'
    )

    print("\n" + "="*70)
    print("TESTING WITH BEAM SEARCH")
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

            print(f"\nWord: '{word}'")

            # Decode with beam search
            predictions = decoder.beam_search(points, beam_size=16)

            # Check if correct word is in predictions
            is_correct = False
            for j, (pred_word, score) in enumerate(predictions[:5]):
                marker = "✓" if pred_word == word else " "
                if pred_word == word:
                    is_correct = True
                print(f"  {marker} {j+1}. {pred_word:10s} (score: {score:.2f})")

            results.append((word, is_correct))

    # Calculate accuracy
    correct = sum(1 for _, is_correct in results if is_correct)
    total = len(results)

    print(f"\n" + "="*70)
    print(f"ACCURACY: {correct}/{total} = {correct/total*100:.1f}%")
    print("="*70)


if __name__ == "__main__":
    test_beam_search()