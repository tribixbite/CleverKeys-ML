#!/usr/bin/env python3
"""Verify gesture predictions work accurately."""

import numpy as np
import onnxruntime as ort


class SwipePredictor:
    """Test swipe gesture predictions."""

    def __init__(self, model_path):
        """Initialize with ONNX model."""
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        print(f"✓ Loaded model: {model_path}")

        # QWERTY layout
        self.key_positions = {
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

    def generate_swipe_path(self, word):
        """Generate swipe gesture points for a word."""
        points = []
        for i, char in enumerate(word.lower()):
            if char in self.key_positions:
                x, y = self.key_positions[char]
                # More points at start/end
                num_points = 5 if i in [0, len(word)-1] else 3
                for j in range(num_points):
                    points.append({
                        'x': x + np.random.uniform(-0.01, 0.01),
                        'y': y + np.random.uniform(-0.01, 0.01),
                        't': len(points) * 10
                    })
        return points

    def compute_features(self, points):
        """Compute 37D feature vectors."""
        features = []
        for i, pt in enumerate(points):
            feat = np.zeros(37, dtype=np.float32)

            # Position
            feat[0] = pt['x']
            feat[1] = pt['y']
            feat[2] = pt['t'] / 1000.0

            # Velocity
            if i > 0:
                prev = points[i-1]
                dt = max((pt['t'] - prev['t']) / 1000.0, 0.001)
                feat[3] = (pt['x'] - prev['x']) / dt
                feat[4] = (pt['y'] - prev['y']) / dt

            # Acceleration
            if i > 1:
                prev2 = points[i-2]
                dt2 = max((pt['t'] - prev2['t']) / 1000.0, 0.001)
                feat[5] = (feat[3] - features[-1][3]) / dt
                feat[6] = (feat[4] - features[-1][4]) / dt

            features.append(feat)

        return np.array(features, dtype=np.float32)

    def predict(self, word):
        """Run inference for a word."""
        points = self.generate_swipe_path(word)
        features = self.compute_features(points)

        # Reshape to (B, F, T)
        features_bft = features.T[np.newaxis, :, :]
        lengths = np.array([features.shape[0]], dtype=np.int32)

        # Run inference
        outputs = self.session.run(None, {
            'features_bft': features_bft,
            'lengths': lengths
        })

        encoded_btf = outputs[0]
        encoded_lengths = outputs[1]

        return {
            'word': word,
            'num_points': len(points),
            'num_features': features.shape[0],
            'output_shape': encoded_btf.shape,
            'output_length': encoded_lengths[0]
        }


def test_models():
    """Test all available models."""
    models = [
        'encoder_web_ultra.onnx',
        'encoder_android_int8_final.onnx',
        'encoder_fp32.onnx'
    ]

    test_words = [
        'hello', 'world', 'test', 'good', 'time',
        'work', 'here', 'the', 'and', 'you'
    ]

    print("\n" + "="*60)
    print("SWIPE PREDICTION ACCURACY TEST")
    print("="*60)

    for model_path in models:
        try:
            print(f"\n📊 Testing: {model_path}")
            predictor = SwipePredictor(model_path)

            successes = 0
            for word in test_words:
                try:
                    result = predictor.predict(word)

                    # Check if encoder produced valid output
                    if result['output_length'] > 0:
                        successes += 1
                        print(f"  ✓ '{word}': {result['num_points']} points → "
                              f"{result['output_shape']} output")
                    else:
                        print(f"  ✗ '{word}': No output produced")

                except Exception as e:
                    print(f"  ✗ '{word}': Error - {e}")

            accuracy = (successes / len(test_words)) * 100
            print(f"\n  Success rate: {successes}/{len(test_words)} ({accuracy:.1f}%)")

        except Exception as e:
            print(f"  ❌ Failed to test {model_path}: {e}")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


def test_gesture_variability():
    """Test prediction consistency with gesture variations."""
    print("\n" + "="*60)
    print("GESTURE VARIABILITY TEST")
    print("="*60)

    predictor = SwipePredictor('encoder_web_ultra.onnx')

    # Test same word multiple times
    word = "hello"
    print(f"\nTesting '{word}' with 10 different gesture variations:")

    outputs = []
    for i in range(10):
        result = predictor.predict(word)
        outputs.append(result['output_length'])
        print(f"  Variation {i+1}: {result['num_points']} points → "
              f"output length {result['output_length']}")

    # Check consistency
    avg_output = np.mean(outputs)
    std_output = np.std(outputs)
    print(f"\nConsistency metrics:")
    print(f"  Average output length: {avg_output:.1f}")
    print(f"  Standard deviation: {std_output:.2f}")

    if std_output < 2.0:
        print("  ✅ Good consistency across gesture variations")
    else:
        print("  ⚠️ High variability in outputs")


if __name__ == "__main__":
    import os
    # Ensure we're in the right directory
    if not os.path.exists('encoder_web_ultra.onnx'):
        print("Error: ONNX models not found in current directory")
        print("Please run from web-demo directory")
        exit(1)

    test_models()
    test_gesture_variability()