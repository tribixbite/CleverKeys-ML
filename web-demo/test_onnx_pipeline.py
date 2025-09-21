#!/usr/bin/env python3
"""
Comprehensive ONNX Pipeline Test
Tests encoder, decoder, and beam search using real training data
"""

import json
import numpy as np
import onnxruntime as ort
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys
from rnnt_beam_search import RNNTBeamSearch

class SwipeFeatureExtractor:
    """Feature extraction matching the training pipeline"""

    def __init__(self):
        self.key_centers = self._get_default_qwerty_layout()
        self.feature_dim = 37

    def _get_default_qwerty_layout(self):
        layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        centers = []

        for row_idx, row in enumerate(layout):
            for col_idx, char in enumerate(row):
                x01 = (col_idx + 0.5) / 10.0
                y01 = (row_idx + 0.5) / 3.0
                centers.append({
                    'char': char,
                    'x': x01 * 2.0 - 1.0,
                    'y': y01 * 2.0 - 1.0
                })
        return centers

    def prepare_points(self, points):
        """Normalize points to [-1, 1] with relative time"""
        if not points:
            return []

        start_t = points[0].get('t', 0.0)
        prepared = []

        for idx, pt in enumerate(points):
            prepared.append({
                'x': np.clip(float(pt.get('x', 0.0)), -1.0, 1.0),
                'y': np.clip(float(pt.get('y', 0.0)), -1.0, 1.0),
                't': float(pt.get('t', idx * 10.0)) - start_t
            })
        return prepared

    def get_resample_target(self, length):
        """Adaptive resampling target"""
        if length <= 48:
            return 56
        if length >= 112:
            return 96
        progress = (length - 48) / 64.0
        return int(56 + progress * 40)

    def resample_points(self, points, target_count):
        """Resample points using linear interpolation"""
        if not points or target_count <= 0:
            return []
        if len(points) == target_count:
            return points[:]

        resampled = []
        duration = max(points[-1]['t'] - points[0]['t'], 1.0)
        step = duration / max(target_count - 1, 1)

        for i in range(target_count):
            target_time = points[0]['t'] + step * i if i < target_count - 1 else points[-1]['t']

            # Find surrounding points
            idx = 0
            while idx < len(points) - 2 and points[idx + 1]['t'] < target_time:
                idx += 1

            p1 = points[idx]
            p2 = points[min(idx + 1, len(points) - 1)]

            # Interpolate
            span = max(p2['t'] - p1['t'], 1.0)
            alpha = np.clip((target_time - p1['t']) / span, 0.0, 1.0)

            resampled.append({
                'x': p1['x'] + (p2['x'] - p1['x']) * alpha,
                'y': p1['y'] + (p2['y'] - p1['y']) * alpha,
                't': target_time
            })

        return resampled

    def extract_point_features(self, points, idx):
        """Extract 37-dimensional feature vector for a point"""
        curr = points[idx]
        prev = points[idx - 1] if idx > 0 else None
        prev2 = points[idx - 2] if idx > 1 else None
        total = len(points)

        features = []

        # Position and time (3)
        x, y = curr['x'], curr['y']
        t_seconds = curr['t'] / 1000.0
        features.extend([x, y, t_seconds])

        # Velocity (3)
        vx = vy = speed = 0.0
        if prev:
            dt = max((curr['t'] - prev['t']) / 1000.0, 1e-6)
            vx = (x - prev['x']) / dt
            vy = (y - prev['y']) / dt
            speed = np.hypot(vx, vy)
        features.extend([vx, vy, speed])

        # Acceleration (3)
        ax = ay = acc = 0.0
        if prev and prev2:
            dt1 = max((curr['t'] - prev['t']) / 1000.0, 1e-6)
            dt2 = max((prev['t'] - prev2['t']) / 1000.0, 1e-6)
            vx_prev = (prev['x'] - prev2['x']) / dt2
            vy_prev = (prev['y'] - prev2['y']) / dt2
            ax = (vx - vx_prev) / dt1
            ay = (vy - vy_prev) / dt1
            acc = np.hypot(ax, ay)
        features.extend([ax, ay, acc])

        # Angle features (4)
        angle = np.arctan2(vy, vx) if prev else 0.0
        features.extend([angle, np.sin(angle), np.cos(angle)])

        # Curvature
        curvature = 0.0
        if prev and prev2:
            prev_angle = np.arctan2(prev['y'] - prev2['y'], prev['x'] - prev2['x'])
            curvature = angle - prev_angle
            while curvature > np.pi: curvature -= 2 * np.pi
            while curvature < -np.pi: curvature += 2 * np.pi
        features.append(curvature)

        # Distance to 5 nearest keys (5)
        key_dists = sorted([np.hypot(x - k['x'], y - k['y']) for k in self.key_centers])[:5]
        while len(key_dists) < 5:
            key_dists.append(1.0)
        features.extend(key_dists)

        # Progress markers (3)
        progress = idx / max(total - 1, 1)
        is_start = 1.0 if idx == 0 else 0.0
        is_end = 1.0 if idx == total - 1 else 0.0
        features.extend([progress, is_start, is_end])

        # Window statistics (6)
        win_start = max(0, idx - 2)
        win_end = min(total, idx + 3)
        window = points[win_start:win_end]

        if len(window) > 1:
            xs = [p['x'] for p in window]
            ys = [p['y'] for p in window]
            mean_x = np.mean(xs)
            mean_y = np.mean(ys)
            std_x = np.std(xs)
            std_y = np.std(ys)
            range_x = max(xs) - min(xs)
            range_y = max(ys) - min(ys)
        else:
            mean_x, std_x = x, 0.0
            mean_y, std_y = y, 0.0
            range_x = range_y = 0.0

        features.extend([mean_x, std_x, mean_y, std_y, range_x, range_y])

        # Pad to 37 features
        while len(features) < self.feature_dim:
            features.append(0.0)

        return np.array(features[:self.feature_dim], dtype=np.float32)

    def extract_features(self, raw_points):
        """Extract feature matrix from raw swipe points"""
        # Prepare points
        points = self.prepare_points(raw_points)

        # Resample
        target_len = self.get_resample_target(len(points))
        resampled = self.resample_points(points, target_len)

        # Extract features
        feature_matrix = []
        for i in range(len(resampled)):
            feature_matrix.append(self.extract_point_features(resampled, i))

        return np.stack(feature_matrix, axis=0)


class ONNXPipelineTest:
    def __init__(self, encoder_path, decoder_path, vocab_path=None):
        self.encoder_session = None
        self.decoder_session = None
        self.vocab = []
        self.blank_id = 29
        self.extractor = SwipeFeatureExtractor()
        self.decoder = None

        # Load models
        self._load_models(encoder_path, decoder_path)

        # Initialize RNN-T beam search decoder
        self.decoder = RNNTBeamSearch(
            self.decoder_session,
            vocab_size=30,
            blank_id=29,
            beam_size=5
        )

        # Load vocabulary
        if vocab_path and Path(vocab_path).exists():
            with open(vocab_path, 'r') as f:
                self.vocab = [line.strip() for line in f if line.strip()]
        else:
            self.vocab = ['<blank>'] + ["'"] + list('abcdefghijklmnopqrstuvwxyz') + ['<unk>']

        print(f"✓ Models loaded, vocabulary size: {len(self.vocab)}")

    def _load_models(self, encoder_path, decoder_path):
        """Load ONNX models"""
        providers = ['CPUExecutionProvider']

        print(f"Loading encoder from: {encoder_path}")
        self.encoder_session = ort.InferenceSession(encoder_path, providers=providers)
        print(f"  Inputs: {[i.name for i in self.encoder_session.get_inputs()]}")
        print(f"  Outputs: {[o.name for o in self.encoder_session.get_outputs()]}")

        print(f"Loading decoder from: {decoder_path}")
        self.decoder_session = ort.InferenceSession(decoder_path, providers=providers)
        print(f"  Inputs: {[i.name for i in self.decoder_session.get_inputs()]}")
        print(f"  Outputs: {[o.name for o in self.decoder_session.get_outputs()]}")

    def run_encoder(self, features):
        """Run encoder inference"""
        T, F = features.shape

        # Reshape to [batch, features, time]
        features_bft = np.zeros((1, F, T), dtype=np.float32)
        for t in range(T):
            for f in range(F):
                features_bft[0, f, t] = features[t, f]

        inputs = {
            'audio_signal': features_bft,
            'length': np.array([T], dtype=np.int64)
        }

        outputs = self.encoder_session.run(None, inputs)
        return outputs

    def greedy_decode(self, encoder_outputs):
        """RNN-T beam search decoding"""
        encoded = encoder_outputs[0]  # Shape: [batch, time, features]
        encoded_len = int(encoder_outputs[1][0]) if len(encoder_outputs) > 1 else encoded.shape[1]

        # Use RNN-T beam search
        results = self.decoder.decode(encoded, encoded_len)

        # Get top prediction
        if results:
            prediction = results[0][0]  # Get text from top hypothesis
        else:
            # Fallback
            prediction = "unknown"

        return prediction if prediction else "unknown"

    def test_sample(self, sample):
        """Test a single sample"""
        word = sample['word']
        points = sample['points']

        # Extract features
        features = self.extractor.extract_features(points)

        # Run encoder
        start_time = time.time()
        encoder_outputs = self.run_encoder(features)

        # Decode
        prediction = self.greedy_decode(encoder_outputs)

        latency = (time.time() - start_time) * 1000

        return {
            'word': word,
            'prediction': prediction,
            'correct': prediction == word,
            'latency': latency,
            'num_points': len(points),
            'num_frames': len(features)
        }

    def test_dataset(self, data_path, max_samples=100):
        """Test on a dataset"""
        print(f"\n📊 Testing on: {data_path}")

        samples = []
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                if line.strip():
                    samples.append(json.loads(line))

        print(f"Testing {len(samples)} samples...")

        results = []
        correct = 0
        total_latency = 0

        for i, sample in enumerate(samples):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(samples)}", end='\r')

            result = self.test_sample(sample)
            results.append(result)

            if result['correct']:
                correct += 1
            total_latency += result['latency']

            # Log first few results
            if i < 5:
                status = "✓" if result['correct'] else "✗"
                print(f"{status} '{result['word']}' → '{result['prediction']}' "
                      f"({result['num_points']} pts → {result['num_frames']} frames, "
                      f"{result['latency']:.1f}ms)")

        # Summary
        print(f"\n\n📈 Results:")
        print(f"Accuracy: {correct}/{len(samples)} ({100*correct/len(samples):.1f}%)")
        print(f"Avg latency: {total_latency/len(samples):.1f}ms")

        # Save results
        results_file = f"test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_file}")

        return results


def main():
    # Parse arguments
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    data_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('../data/train_final_val.jsonl')
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    # Paths
    encoder_path = model_dir / 'encoder-model.onnx'
    decoder_path = model_dir / 'decoder_joint-model.onnx'
    vocab_path = Path('../data/vocab.txt')

    print(f"🚀 ONNX Pipeline Test")
    print(f"Encoder: {encoder_path}")
    print(f"Decoder: {decoder_path}")
    print(f"Dataset: {data_path}")
    print(f"Max samples: {max_samples}")

    # Run tests
    tester = ONNXPipelineTest(encoder_path, decoder_path, vocab_path)
    results = tester.test_dataset(data_path, max_samples)

    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()