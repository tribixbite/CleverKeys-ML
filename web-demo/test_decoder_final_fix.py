#!/usr/bin/env python3
"""
Test RNN-T decoder with FULLY FIXED coordinate mapping and keyboard layout.
This version uses the EXACT keyboard layout from training.
"""

import json
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_correct_key_positions():
    """
    Build the EXACT key positions used in training.
    Training uses a simplified grid layout with evenly spaced keys.
    """
    layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    positions = {}

    for row_idx, row in enumerate(layout):
        for col_idx, char in enumerate(row):
            # This is EXACTLY how training computes positions
            x01 = (col_idx + 0.5) / 10.0  # In [0,1]
            y01 = (row_idx + 0.5) / 3.0   # In [0,1]
            positions[char] = (x01, y01)

    return positions


class FinalFixedRNNTDecoder:
    """RNN-T decoder with fully corrected coordinate mapping and keyboard layout"""

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

        print(f"Loaded {len(self.words)} words, vocab size: {self.vocab_size}")

        # Model dimensions
        self.L = 2  # LSTM layers
        self.H = 320  # Hidden size
        self.D = 256  # Encoder dimension
        self.V_output = 30  # Actual output size (has extra token)

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

    def _compute_features_correct(self, points: List[dict]) -> np.ndarray:
        """
        Compute features with correct coordinate transformation.
        Points are expected to be in [0,1] range from the correct keyboard layout.
        """
        T = len(points)
        features = np.zeros((T, 37), dtype=np.float32)

        for i, pt in enumerate(points):
            # Transform from [0,1] to [-1,1] as expected by model
            x = pt['x'] * 2.0 - 1.0
            y = pt['y'] * 2.0 - 1.0
            x = np.clip(x, -1.0, 1.0)
            y = np.clip(y, -1.0, 1.0)

            # Basic features
            features[i, 0] = x
            features[i, 1] = y
            features[i, 2] = pt.get('t', i * 10) / 1000.0

            # Velocity
            if i > 0:
                prev = points[i-1]
                x_prev = prev['x'] * 2.0 - 1.0
                y_prev = prev['y'] * 2.0 - 1.0
                x_prev = np.clip(x_prev, -1.0, 1.0)
                y_prev = np.clip(y_prev, -1.0, 1.0)

                dt = max((pt.get('t', i*10) - prev.get('t', (i-1)*10)) / 1000.0, 0.001)
                features[i, 3] = (x - x_prev) / dt
                features[i, 4] = (y - y_prev) / dt

            # Acceleration
            if i > 1:
                dt = max((pt.get('t', i*10) - points[i-1].get('t', (i-1)*10)) / 1000.0, 0.001)
                features[i, 5] = (features[i, 3] - features[i-1, 3]) / dt
                features[i, 6] = (features[i, 4] - features[i-1, 4]) / dt

        return features

    def decode_beam_search(self, points: List[dict], beam_size: int = 8) -> List[Tuple[str, float]]:
        """Full beam search decoding"""
        features = self._compute_features_correct(points)
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
                logits = np.log(exp_logits / np.sum(exp_logits, axis=-1, keepdims=True))

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


def test_final_fix():
    """Test with the fully fixed decoder"""

    decoder = FinalFixedRNNTDecoder(
        encoder_path='encoder_web_ultra.onnx',
        decoder_path='rnnt_step_fp32.onnx',
        runtime_meta_path='../trained_models/nema1/runtime_meta.json',
        words_path='../trained_models/nema1/words.txt'
    )

    # Use CORRECT key positions from training
    key_positions = build_correct_key_positions()

    print("\n" + "="*70)
    print("TESTING WITH FULLY FIXED KEYBOARD LAYOUT AND COORDINATES")
    print("="*70)

    def generate_swipe(word: str) -> List[dict]:
        """Generate swipe with correct key positions"""
        points = []
        t = 0
        for i, char in enumerate(word.lower()):
            if char in key_positions:
                x, y = key_positions[char]
                # Add multiple points with slight noise
                num_points = 5 if i in [0, len(word)-1] else 3
                for j in range(num_points):
                    points.append({
                        'x': x + np.random.uniform(-0.005, 0.005),
                        'y': y + np.random.uniform(-0.005, 0.005),
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
            x_train = x * 2.0 - 1.0
            y_train = y * 2.0 - 1.0
            print(f"  First key: {word[0]} at ({x:.3f}, {y:.3f}) -> ({x_train:.3f}, {y_train:.3f})")

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
    test_final_fix()