#!/usr/bin/env python3
"""
Test RNN-T decoder with FIXED coordinate mapping.
Converts from [0,1] web coordinates to [-1,1] training coordinates.
"""

import json
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FixedRNNTDecoder:
    """RNN-T decoder with correct coordinate mapping"""

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

        # Build trie for lexicon-constrained decoding
        self.trie_root = self._build_trie()

        print(f"Loaded {len(self.words)} words, vocab size: {self.vocab_size}")
        print(f"Blank ID: {self.blank_id}, UNK ID: {self.unk_id}")

        # Get model dimensions
        self._infer_dimensions()

    def _infer_dimensions(self):
        """Infer LSTM dimensions from decoder model"""
        # Get input/output info
        dec_inputs = self.decoder_session.get_inputs()
        dec_outputs = self.decoder_session.get_outputs()

        print(f"Decoder input shapes: {[(i.name, i.shape) for i in dec_inputs]}")
        print(f"Decoder output shapes: {[(o.name, o.shape) for o in dec_outputs]}")

        # Run a dummy inference to get actual output dimensions
        dummy_y = np.array([self.blank_id], dtype=np.int64)
        dummy_h = np.zeros((2, 1, 320), dtype=np.float32)
        dummy_c = np.zeros((2, 1, 320), dtype=np.float32)
        dummy_enc = np.zeros((1, 256), dtype=np.float32)

        outputs = self.decoder_session.run(None, {
            'y_prev': dummy_y,
            'h0': dummy_h,
            'c0': dummy_c,
            'enc_t': dummy_enc
        })

        self.L = 2  # Number of LSTM layers
        self.H = 320  # Hidden size
        self.D = 256  # Encoder output dimension

        # Check actual vocab size from logits output
        logits_shape = outputs[0].shape
        print(f"Logits shape: {logits_shape}")

        # Handle the extra dimension - decoder might output 30 instead of 29
        if len(logits_shape) == 4:
            self.V_output = logits_shape[-1]  # Actual output vocab size
        else:
            self.V_output = logits_shape[-1]

        print(f"Model dimensions - L: {self.L}, H: {self.H}, D: {self.D}")
        print(f"Vocab size from meta: {self.vocab_size}, from model output: {self.V_output}")

    def _build_trie(self) -> dict:
        """Build trie from word list"""
        root = {"children": {}, "is_word": False, "word_id": -1}
        kept = 0

        for word_id, word in enumerate(self.words):
            word = word.lower().replace("'", "'")

            # Skip words with unknown characters
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

        print(f"Trie built: {kept}/{len(self.words)} words kept")
        return root

    def _compute_features_with_correct_coords(self, points: List[dict]) -> np.ndarray:
        """
        Compute 37D features with CORRECT coordinate mapping.
        Converts from [0,1] web coordinates to [-1,1] training coordinates.
        """
        T = len(points)
        features = np.zeros((T, 37), dtype=np.float32)

        for i, pt in enumerate(points):
            # CRITICAL FIX: Convert from [0,1] to [-1,1] range
            x_web = pt['x']  # In [0,1] range
            y_web = pt['y']  # In [0,1] range

            # Transform to [-1,1] with (0,0) as keyboard center
            x_train = x_web * 2.0 - 1.0
            y_train = y_web * 2.0 - 1.0

            # Clamp to [-1,1] as done in training
            x_train = np.clip(x_train, -1.0, 1.0)
            y_train = np.clip(y_train, -1.0, 1.0)

            # Position
            features[i, 0] = x_train
            features[i, 1] = y_train
            features[i, 2] = pt.get('t', i * 10) / 1000.0

            # Velocity
            if i > 0:
                prev = points[i-1]
                x_prev_web = prev['x']
                y_prev_web = prev['y']

                # Transform previous point too
                x_prev_train = x_prev_web * 2.0 - 1.0
                y_prev_train = y_prev_web * 2.0 - 1.0
                x_prev_train = np.clip(x_prev_train, -1.0, 1.0)
                y_prev_train = np.clip(y_prev_train, -1.0, 1.0)

                dt = max((pt.get('t', i*10) - prev.get('t', (i-1)*10)) / 1000.0, 0.001)
                features[i, 3] = (x_train - x_prev_train) / dt
                features[i, 4] = (y_train - y_prev_train) / dt

            # Acceleration
            if i > 1:
                prev2 = points[i-2]
                dt2 = max((pt.get('t', i*10) - prev2.get('t', (i-2)*10)) / 1000.0, 0.001)
                features[i, 5] = (features[i, 3] - features[i-1, 3]) / dt
                features[i, 6] = (features[i, 4] - features[i-1, 4]) / dt

        return features

    def decode_from_points(self, points: List[dict]) -> List[Tuple[str, float]]:
        """Decode from raw gesture points with corrected coordinates"""
        features = self._compute_features_with_correct_coords(points)
        features_bft = features.T[np.newaxis, :, :]  # Shape: (1, 37, T)

        # Run encoder
        T = features_bft.shape[2]
        encoder_outputs = self.encoder_session.run(None, {
            'features_bft': features_bft.astype(np.float32),
            'lengths': np.array([T], dtype=np.int32)
        })

        encoded_btf = encoder_outputs[0]

        # Handle dimension transpose if needed
        if len(encoded_btf.shape) == 3:
            if encoded_btf.shape[1] == self.D and encoded_btf.shape[2] != self.D:
                # Shape is (1, D, T_out), need to transpose to (1, T_out, D)
                encoded_btf = np.transpose(encoded_btf, (0, 2, 1))

        T_out = encoded_btf.shape[1]
        print(f"Encoder output shape: {encoded_btf.shape}")

        # Simple greedy decoding for testing
        predictions = []

        # Initialize beam
        h = np.zeros((self.L, 1, self.H), dtype=np.float32)
        c = np.zeros((self.L, 1, self.H), dtype=np.float32)
        y_prev = self.blank_id

        decoded_chars = []

        for t in range(T_out):
            enc_t = encoded_btf[0, t:t+1, :]  # (1, D)

            outputs = self.decoder_session.run(None, {
                'y_prev': np.array([y_prev], dtype=np.int64),
                'h0': h,
                'c0': c,
                'enc_t': enc_t
            })

            logits = outputs[0]
            h = outputs[1]
            c = outputs[2]

            # Reshape logits if needed
            while len(logits.shape) > 2 and logits.shape[0] == 1:
                logits = logits.squeeze(0)

            # Get top prediction (ignore extra token if present)
            valid_vocab_size = min(self.vocab_size, logits.shape[-1])
            probs = logits[0, :valid_vocab_size]

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

        # Convert chars to word
        decoded_word = ''.join(decoded_chars)
        predictions.append((decoded_word, 1.0))

        return predictions


def test_with_fixed_coordinates():
    """Test decoder with fixed coordinate mapping"""

    # Initialize decoder
    decoder = FixedRNNTDecoder(
        encoder_path='encoder_web_ultra.onnx',
        decoder_path='rnnt_step_fp32.onnx',
        runtime_meta_path='../trained_models/nema1/runtime_meta.json',
        words_path='../trained_models/nema1/words.txt'
    )

    # QWERTY layout in [0,1] coordinates (web format)
    key_positions = {
        'q': (0.05, 0.25), 'w': (0.15, 0.25), 'e': (0.25, 0.25),
        'r': (0.35, 0.25), 't': (0.45, 0.25), 'y': (0.55, 0.25),
        'u': (0.65, 0.25), 'i': (0.75, 0.25), 'o': (0.85, 0.25),
        'p': (0.95, 0.25),
        'a': (0.10, 0.50), 's': (0.20, 0.50), 'd': (0.30, 0.50),
        'f': (0.40, 0.50), 'g': (0.50, 0.50), 'h': (0.60, 0.50),
        'j': (0.70, 0.50), 'k': (0.80, 0.50), 'l': (0.90, 0.50),
        'z': (0.20, 0.75), 'x': (0.30, 0.75), 'c': (0.40, 0.75),
        'v': (0.50, 0.75), 'b': (0.60, 0.75), 'n': (0.70, 0.75),
        'm': (0.80, 0.75)
    }

    def generate_swipe(word: str) -> List[dict]:
        """Generate swipe points for a word"""
        points = []
        t = 0
        for i, char in enumerate(word.lower()):
            if char in key_positions:
                x, y = key_positions[char]
                # Add multiple points per key
                num_points = 5 if i in [0, len(word)-1] else 3
                for j in range(num_points):
                    points.append({
                        'x': x + np.random.uniform(-0.01, 0.01),
                        'y': y + np.random.uniform(-0.01, 0.01),
                        't': t
                    })
                    t += 10
        return points

    print("\n" + "="*60)
    print("TESTING WITH FIXED COORDINATE MAPPING")
    print("="*60)

    test_words = ['hello', 'world', 'test', 'the', 'and']

    for word in test_words:
        print(f"\nTesting word: '{word}'")
        points = generate_swipe(word)

        # Show coordinate transformation
        sample_point = points[0]
        x_web = sample_point['x']
        y_web = sample_point['y']
        x_train = x_web * 2.0 - 1.0
        y_train = y_web * 2.0 - 1.0

        print(f"  Web coords: ({x_web:.3f}, {y_web:.3f})")
        print(f"  Train coords: ({x_train:.3f}, {y_train:.3f})")

        # Decode
        predictions = decoder.decode_from_points(points)

        print(f"  Predictions:")
        for i, (pred_word, score) in enumerate(predictions):
            marker = "✓" if pred_word == word else "✗"
            print(f"    {marker} {pred_word} (score: {score:.2f})")


if __name__ == "__main__":
    test_with_fixed_coordinates()