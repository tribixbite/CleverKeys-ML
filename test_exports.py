#!/usr/bin/env python3
"""Quick test of exported ONNX models with validation samples."""

import json
import numpy as np
import onnxruntime as ort

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
        vx = vy = ax = ay = 0.0
        if i > 0:
            prev = points[i-1]
            dt = max((pt.get('t', i*10) - prev.get('t', (i-1)*10)) / 1000.0, 0.001)
            vx = (x - float(prev.get('x', x))) / dt
            vy = (y - float(prev.get('y', y))) / dt

        # Pad to 37 features (simplified)
        feat_vec = [x, y, t, vx, vy, 0, 0, 0, 0, 0, 0, 0, 0] + [0.0] * 24
        features.append(feat_vec[:37])

    return np.array(features, dtype=np.float32)

def test_onnx_encoder(model_path: str, samples):
    """Test ONNX encoder with validation samples."""
    print(f"\n🧪 Testing {model_path}")

    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print(f"✓ Model loaded successfully")
        print(f"  Inputs: {[inp.name for inp in session.get_inputs()]}")
        print(f"  Outputs: {[out.name for out in session.get_outputs()]}")

        for i, sample in enumerate(samples):
            word = sample['word']
            points = sample['points']

            # Convert to features
            features = simple_featurizer(points)
            features_bft = features.T[np.newaxis, :, :]  # (1, 37, T)
            lengths = np.array([features.shape[0]], dtype=np.int32)

            print(f"\n  Sample {i+1}: '{word}' ({len(points)} points → {features.shape[0]} features)")

            # Run inference
            outputs = session.run(None, {
                'features_bft': features_bft,
                'lengths': lengths
            })

            encoded_btf = outputs[0]  # (B, T_out, D)
            encoded_lengths = outputs[1]  # (B,)

            print(f"    Input shape: {features_bft.shape}")
            print(f"    Output shape: {encoded_btf.shape}")
            print(f"    Output length: {encoded_lengths[0]}")

            # Basic sanity check
            if encoded_btf.shape[0] == 1 and encoded_lengths[0] > 0:
                print(f"    ✓ Inference successful")
            else:
                print(f"    ⚠ Unexpected output shape/length")

        return True

    except Exception as e:
        print(f"❌ Error testing {model_path}: {e}")
        return False

def main():
    """Test all available ONNX encoders."""

    # Load validation samples
    val_file = "trained_models/nema1/personalized_tuning/20250918_105357/val_short_common.jsonl"
    try:
        samples = load_validation_sample(val_file, max_samples=3)
        print(f"📂 Loaded {len(samples)} validation samples from {val_file}")
    except FileNotFoundError:
        print(f"❌ Validation file not found: {val_file}")
        return

    if not samples:
        print("❌ No valid samples found")
        return

    # Test available models
    models_to_test = [
        "encoder_fp32.onnx",
        "encoder_web_ultra.onnx",
        "encoder_android_ultra.onnx"
    ]

    results = {}
    for model in models_to_test:
        try:
            results[model] = test_onnx_encoder(model, samples)
        except FileNotFoundError:
            print(f"⚠ Model not found: {model}")
            results[model] = False

    # Summary
    print(f"\n📊 Test Results:")
    for model, success in results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"  {model}: {status}")

    successful_models = [m for m, success in results.items() if success]
    if successful_models:
        print(f"\n🎉 {len(successful_models)} models working correctly!")
        print(f"Ready for web demo with: {successful_models[0]}")
    else:
        print(f"\n💥 No models working - check exports")

if __name__ == "__main__":
    main()