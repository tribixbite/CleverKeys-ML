#!/usr/bin/env python3
"""
Minimal encoder test - save input and output for comparison
"""

import numpy as np
import onnxruntime as ort
import json

# Load features
features = np.load('python_features.npy')
print(f'Features shape: {features.shape}')

# Transpose for encoder
signal = features.T.reshape(1, 37, -1).astype(np.float32)
signal_len = np.array([82], dtype=np.int64)

# Save input for comparison
np.save('encoder_input.npy', signal)
print(f'Saved encoder input to encoder_input.npy')

# Load and run encoder
encoder = ort.InferenceSession('../models/correct_9292025/encoder.onnx')
enc_out = encoder.run(None, {'audio_signal': signal, 'length': signal_len})

encoded = enc_out[0]
encoded_len = enc_out[1]

print(f'Encoder output shape: {encoded.shape}')
print(f'Encoded length: {encoded_len}')
print(f'First 10 values: {encoded.flat[:10]}')

# Save output
np.save('encoder_output_python.npy', encoded)
print(f'Saved encoder output to encoder_output_python.npy')