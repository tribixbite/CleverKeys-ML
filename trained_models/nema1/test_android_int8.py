#!/usr/bin/env python3
"""Test Android INT8 ONNX model with validation samples."""

import json
import numpy as np
import onnxruntime as ort
import time


def load_validation_sample(jsonl_path: str, max_samples: int = 3):
    """Load a few validation samples for testing."""
    samples = []
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num >= max_samples:
                break
            try:
                sample = json.loads(line)
                if sample.get('word') and sample.get('points'):
                    samples.append(sample)
            except json.JSONDecodeError:
                continue
    return samples


def simple_featurizer(points):
    """Simplified featurizer for testing - just basic kinematic features."""
    if len(points) < 2:
        return np.zeros((1, 37), dtype=np.float32)

    features = []
    for i, pt in enumerate(points):
        x = float(pt.get('x', 0))
        y = float(pt.get('y', 0))
        t = float(pt.get('t', i * 10)) / 1000.0  # Convert to seconds

        # Basic velocity/acceleration
        vx = vy = 0.0
        if i > 0:
            prev = points[i-1]
            dt = max((pt.get('t', i*10) - prev.get('t', (i-1)*10)) / 1000.0, 0.001)
            vx = (x - float(prev.get('x', x))) / dt
            vy = (y - float(prev.get('y', y))) / dt

        # Pad to 37 features
        feat_vec = [x, y, t, vx, vy, 0, 0, 0, 0, 0, 0, 0, 0] + [0.0] * 24
        features.append(feat_vec[:37])

    return np.array(features, dtype=np.float32)


def test_int8_model(model_path: str, samples):
    """Test INT8 ONNX model performance."""
    print(f"\n🧪 Testing INT8 model: {model_path}")

    # Load model
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers, sess_options=session_options)

    print(f"✓ Model loaded successfully")
    print(f"  Inputs: {[inp.name for inp in session.get_inputs()]}")
    print(f"  Outputs: {[out.name for out in session.get_outputs()]}")

    # Check model size
    import os
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Model size: {size_mb:.1f} MB")

    # Warmup
    print("\n🔥 Warming up...")
    warmup_features = np.random.randn(1, 37, 50).astype(np.float32)
    warmup_lengths = np.array([50], dtype=np.int32)
    for _ in range(3):
        session.run(None, {
            'features_bft': warmup_features,
            'lengths': warmup_lengths
        })

    # Test samples
    print("\n📊 Running inference tests...")
    inference_times = []

    for i, sample in enumerate(samples):
        word = sample['word']
        points = sample['points']

        # Convert to features
        features = simple_featurizer(points)
        features_bft = features.T[np.newaxis, :, :]  # (1, 37, T)
        lengths = np.array([features.shape[0]], dtype=np.int32)

        print(f"\n  Sample {i+1}: '{word}' ({len(points)} points → {features.shape[0]} frames)")

        # Run inference multiple times for timing
        times = []
        for _ in range(5):
            start = time.perf_counter()
            outputs = session.run(None, {
                'features_bft': features_bft,
                'lengths': lengths
            })
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)
        inference_times.append(avg_time)

        encoded_btf = outputs[0]  # (B, T_out, D)
        encoded_lengths = outputs[1]  # (B,)

        print(f"    Input shape: {features_bft.shape}")
        print(f"    Output shape: {encoded_btf.shape}")
        print(f"    Output length: {encoded_lengths[0]}")
        print(f"    Inference time: {avg_time:.2f}ms ± {std_time:.2f}ms")

    # Summary
    print("\n📈 Performance Summary:")
    print(f"  Model size: {size_mb:.1f} MB")
    print(f"  Average inference: {np.mean(inference_times):.2f}ms")
    print(f"  Median inference: {np.median(inference_times):.2f}ms")
    print(f"  Min inference: {np.min(inference_times):.2f}ms")
    print(f"  Max inference: {np.max(inference_times):.2f}ms")

    # Check if INT8 quantization is effective
    if size_mb < 30:
        print(f"  ✅ Model successfully quantized (INT8)")
    else:
        print(f"  ⚠️ Model may not be properly quantized")

    return True


def main():
    """Test Android INT8 model."""

    # Load validation samples
    val_file = "personalized_tuning/20250918_105357/val_short_common.jsonl"
    try:
        samples = load_validation_sample(val_file, max_samples=5)
        print(f"📂 Loaded {len(samples)} validation samples")
    except FileNotFoundError:
        # Use dummy samples
        print("⚠️ Validation file not found, using dummy samples")
        samples = [
            {"word": "hello", "points": [{"x": 0.2, "y": 0.5, "t": 0}] * 20},
            {"word": "world", "points": [{"x": 0.8, "y": 0.3, "t": 0}] * 25},
        ]

    # Test INT8 model
    test_int8_model("encoder_android_int8_final.onnx", samples)

    # Compare with FP32 if available
    if os.path.exists("encoder_android_fp32.onnx"):
        print("\n" + "="*60)
        print("Comparing with FP32 model...")
        test_int8_model("encoder_android_fp32.onnx", samples[:2])


if __name__ == "__main__":
    import os
    os.environ['OMP_NUM_THREADS'] = '4'  # Optimize for mobile-like performance
    main()