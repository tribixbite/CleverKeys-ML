#!/usr/bin/env python3
"""
Test RNN-T decoder with actual gesture traces from validation data.
This verifies that the full encoder + decoder pipeline produces accurate predictions.
"""

import json
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RNNTDecoder:
    """Full RNN-T decoder with beam search"""

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

        # Check encoder outputs
        enc_input_names = [i.name for i in self.encoder_session.get_inputs()]
        enc_output_names = [o.name for o in self.encoder_session.get_outputs()]
        print(f"Encoder inputs: {enc_input_names}")
        print(f"Encoder outputs: {enc_output_names}")

    def _infer_dimensions(self):
        """Infer LSTM dimensions from decoder model"""
        # Get input/output names
        input_names = [i.name for i in self.decoder_session.get_inputs()]
        output_names = [o.name for o in self.decoder_session.get_outputs()]
        print(f"Decoder inputs: {input_names}")
        print(f"Decoder outputs: {output_names}")

        # Run a dummy inference to get dimensions
        dummy_y = np.array([self.blank_id], dtype=np.int64)
        dummy_h = np.zeros((2, 1, 320), dtype=np.float32)  # L=2, B=1, H=320
        dummy_c = np.zeros((2, 1, 320), dtype=np.float32)
        dummy_enc = np.zeros((1, 256), dtype=np.float32)  # D=256

        outputs = self.decoder_session.run(None, {
            'y_prev': dummy_y,
            'h0': dummy_h,
            'c0': dummy_c,
            'enc_t': dummy_enc
        })

        self.L = 2  # Number of LSTM layers
        self.H = 320  # Hidden size
        self.D = 256  # Encoder output dimension
        self.V = self.vocab_size  # Use the vocab size from metadata

        print(f"Model dimensions - L: {self.L}, H: {self.H}, D: {self.D}, V: {self.V}")
        print(f"Decoder output shapes: {[o.shape for o in outputs]}")

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

    def decode(self, features_bft: np.ndarray, beam_size: int = 16, max_symbols: int = 20) -> List[Tuple[str, float]]:
        """
        Decode gesture features using RNN-T beam search.

        Args:
            features_bft: Feature tensor of shape (1, 37, T)
            beam_size: Number of beams to maintain
            max_symbols: Maximum symbols per time frame

        Returns:
            List of (word, score) tuples
        """

        # Run encoder
        T = features_bft.shape[2]
        encoder_outputs = self.encoder_session.run(None, {
            'features_bft': features_bft,
            'lengths': np.array([T], dtype=np.int32)
        })

        # Get the encoded output (might be named 'encoded_btf')
        encoded_btf = encoder_outputs[0]  # Shape might be (1, D, T_out) or (1, T_out, D)

        # Check if we need to transpose
        if len(encoded_btf.shape) == 3:
            if encoded_btf.shape[1] == self.D and encoded_btf.shape[2] != self.D:
                # Shape is (1, D, T_out), need to transpose to (1, T_out, D)
                encoded_btf = np.transpose(encoded_btf, (0, 2, 1))

        T_out = encoded_btf.shape[1]

        print(f"Encoder output shape: {encoded_btf.shape}")

        # Initialize beam
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
            # Inner label loop
            for s in range(max_symbols):
                # Sort beams by score
                beams.sort(key=lambda b: b['logp'], reverse=True)
                active = beams[:beam_size]
                N = len(active)

                # Prepare batch inputs
                y_prev = np.array([b['y_prev'] for b in active], dtype=np.int64)
                h0 = np.concatenate([b['h'] for b in active], axis=1)
                c0 = np.concatenate([b['c'] for b in active], axis=1)
                enc_t = np.repeat(encoded_btf[0, t:t+1, :], N, axis=0)

                # Run decoder step
                outputs = self.decoder_session.run(None, {
                    'y_prev': y_prev,
                    'h0': h0,
                    'c0': c0,
                    'enc_t': enc_t
                })

                logits = outputs[0]  # Might have extra dimensions
                h1 = outputs[1]  # (L, N, H)
                c1 = outputs[2]  # (L, N, H)

                # Reshape logits to (N, V) if it has extra dimensions
                # Expected shape is (N, V) where N is batch size and V is vocab size
                if logits.shape[-1] == self.V + 1:  # Vocab size plus one for some reason
                    logits = logits.reshape(N, -1)
                elif len(logits.shape) > 2:
                    # Squeeze only dimensions with size 1
                    logits = np.squeeze(logits)
                    if len(logits.shape) == 1:
                        logits = logits.reshape(1, -1)

                # Apply log softmax if logits are raw scores
                # Check if values are very large (raw scores) or already normalized
                if np.max(np.abs(logits)) > 10:
                    # Apply log softmax
                    logits_max = np.max(logits, axis=-1, keepdims=True)
                    logits = logits - logits_max
                    exp_logits = np.exp(logits)
                    logits = np.log(exp_logits / np.sum(exp_logits, axis=-1, keepdims=True))

                # Expand beams
                next_beams = []
                for i, beam in enumerate(active):
                    # Blank transition
                    lp_blank = float(logits[i, self.blank_id])
                    next_beams.append({
                        'y_prev': self.blank_id,
                        'h': h1[:, i:i+1, :],
                        'c': c1[:, i:i+1, :],
                        'trie': beam['trie'],
                        'logp': beam['logp'] + lp_blank,
                        'chars': beam['chars']
                    })

                    # Character transitions (trie-constrained)
                    allowed = list(beam['trie']['children'].keys())
                    if allowed:
                        # Sort by logit score and take top k
                        char_scores = [(cid, float(logits[i, cid])) for cid in allowed]
                        char_scores.sort(key=lambda x: x[1], reverse=True)

                        for cid, score in char_scores[:6]:  # prune_per_beam = 6
                            child = beam['trie']['children'][cid]
                            next_beams.append({
                                'y_prev': cid,
                                'h': h1[:, i:i+1, :],
                                'c': c1[:, i:i+1, :],
                                'trie': child,
                                'logp': beam['logp'] + score,
                                'chars': beam['chars'] + [cid]
                            })

                # Prune to beam size
                next_beams.sort(key=lambda b: b['logp'], reverse=True)
                beams = next_beams[:beam_size]

                # Early stop if best beam picked blank
                if beams[0]['y_prev'] == self.blank_id:
                    break

        # Collect completed words
        candidates = []
        for beam in beams:
            if beam['trie']['is_word']:
                word_id = beam['trie']['word_id']
                word = self.words[word_id]
                candidates.append((word, beam['logp']))

        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Remove duplicates
        seen = set()
        results = []
        for word, score in candidates:
            if word not in seen:
                seen.add(word)
                results.append((word, score))
                if len(results) >= 5:  # Return top 5
                    break

        return results

    def decode_from_points(self, points: List[dict]) -> List[Tuple[str, float]]:
        """Decode from raw gesture points"""
        features = self._compute_features(points)
        features_bft = features.T[np.newaxis, :, :]  # Shape: (1, 37, T)
        return self.decode(features_bft.astype(np.float32))

    def _compute_features(self, points: List[dict]) -> np.ndarray:
        """Compute 37D features from gesture points"""
        T = len(points)
        features = np.zeros((T, 37), dtype=np.float32)

        for i, pt in enumerate(points):
            # Position
            features[i, 0] = pt['x']
            features[i, 1] = pt['y']
            features[i, 2] = pt.get('t', i * 10) / 1000.0

            # Velocity
            if i > 0:
                prev = points[i-1]
                dt = max((pt.get('t', i*10) - prev.get('t', (i-1)*10)) / 1000.0, 0.001)
                features[i, 3] = (pt['x'] - prev['x']) / dt
                features[i, 4] = (pt['y'] - prev['y']) / dt

            # Acceleration
            if i > 1:
                prev2 = points[i-2]
                dt2 = max((pt.get('t', i*10) - prev2.get('t', (i-2)*10)) / 1000.0, 0.001)
                features[i, 5] = (features[i, 3] - features[i-1, 3]) / dt
                features[i, 6] = (features[i, 4] - features[i-1, 4]) / dt

        return features


def load_validation_sample(jsonl_path: str, sample_idx: int = 0) -> dict:
    """Load a sample from validation data"""
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i == sample_idx:
                return json.loads(line)
    raise ValueError(f"Sample {sample_idx} not found")


def test_with_validation_data():
    """Test decoder with actual validation samples"""

    # Initialize decoder
    decoder = RNNTDecoder(
        encoder_path='encoder_web_ultra.onnx',
        decoder_path='rnnt_step_fp32.onnx',
        runtime_meta_path='../trained_models/nema1/runtime_meta.json',
        words_path='../trained_models/nema1/words.txt'
    )

    # Load validation samples
    val_path = '../data/train_final_val.jsonl'

    print("\n" + "="*60)
    print("TESTING RNN-T DECODER WITH VALIDATION DATA")
    print("="*60)

    # Test multiple samples
    test_samples = [0, 1, 2, 3, 4]  # First 5 samples

    for idx in test_samples:
        sample = load_validation_sample(val_path, idx)
        # Field is 'word'
        true_word = sample.get('word', '')

        # Check if we have precomputed features or need to compute from points
        if 'features' in sample:
            features = np.array(sample['features'], dtype=np.float32)  # Shape: (T, 37)
            print(f"\nSample {idx}: True word = '{true_word}'")
            print(f"Feature shape: {features.shape}")
            # Reshape to (1, 37, T)
            features_bft = features.T[np.newaxis, :, :]
            # Decode
            predictions = decoder.decode(features_bft)
        else:
            # Compute features from points
            points = sample['points']
            print(f"\nSample {idx}: True word = '{true_word}'")
            print(f"Points: {len(points)}")
            # Decode from points
            predictions = decoder.decode_from_points(points)

        print(f"Predictions:")
        for i, (word, score) in enumerate(predictions):
            marker = "✓" if word == true_word else " "
            print(f"  {marker} {i+1}. {word:15s} (score: {score:.2f})")

        # Check if correct word is in top predictions
        pred_words = [w for w, _ in predictions]
        if true_word in pred_words:
            rank = pred_words.index(true_word) + 1
            print(f"  → Correct! Rank: {rank}")
        else:
            print(f"  → Missed! True word '{true_word}' not in top 5")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


def test_with_simulated_gesture():
    """Test with a simulated gesture for a known word"""

    # Initialize decoder
    decoder = RNNTDecoder(
        encoder_path='encoder_web_ultra.onnx',
        decoder_path='rnnt_step_fp32.onnx',
        runtime_meta_path='../trained_models/nema1/runtime_meta.json',
        words_path='../trained_models/nema1/words.txt'
    )

    # QWERTY layout
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
    print("TESTING WITH SIMULATED GESTURES")
    print("="*60)

    test_words = ['hello', 'world', 'test', 'good', 'time']

    for word in test_words:
        print(f"\nSimulating swipe for: '{word}'")
        points = generate_swipe(word)

        # Decode
        predictions = decoder.decode_from_points(points)

        print(f"Predictions:")
        for i, (pred_word, score) in enumerate(predictions):
            marker = "✓" if pred_word == word else " "
            print(f"  {marker} {i+1}. {pred_word:15s} (score: {score:.2f})")


if __name__ == "__main__":
    print("Testing RNN-T decoder...")

    # First test with simulated gestures
    test_with_simulated_gesture()

    # Then test with real validation data
    test_with_validation_data()