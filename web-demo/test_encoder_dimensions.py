#!/usr/bin/env python3
"""Test encoder output dimensions with different input lengths"""

import numpy as np
import onnxruntime as ort

# Load encoder
encoder_sess = ort.InferenceSession(
    "encoder_fresh.onnx",
    providers=["CPUExecutionProvider"]
)

# Print input/output info
print("Encoder inputs:")
for inp in encoder_sess.get_inputs():
    print(f"  {inp.name}: {inp.shape} {inp.type}")

print("\nEncoder outputs:")
for out in encoder_sess.get_outputs():
    print(f"  {out.name}: {out.shape} {out.type}")

# Test different input lengths
test_lengths = [28, 56, 84, 112]

for T in test_lengths:
    # Create test features (1, 37, T)
    features = np.random.randn(1, 37, T).astype(np.float32)
    lengths = np.array([T], dtype=np.int32)

    # Run encoder
    outputs = encoder_sess.run(None, {
        "features_bft": features,
        "lengths": lengths
    })

    encoded = outputs[0]
    encoded_len = outputs[1] if len(outputs) > 1 else None

    print(f"\nInput T={T}:")
    print(f"  Encoded shape: {encoded.shape}")
    if encoded_len is not None:
        print(f"  Encoded length: {encoded_len}")

    # Calculate downsampling factor
    if len(encoded.shape) == 3:
        T_out = encoded.shape[2] if encoded.shape[1] != T else encoded.shape[1]
    else:
        T_out = encoded.shape[1] if len(encoded.shape) > 1 else None

    if T_out:
        factor = T / T_out
        print(f"  Downsampling factor: {factor:.2f} ({T} → {T_out})")