#!/bin/bash
# Test the decoder using the existing CLI script

cd /home/will/git/swype/cleverkeys/trained_models/nema1

# Create a test features file from validation data
python3 -c "
import json
import numpy as np

# Load first sample from validation data
with open('../../data/train_final_val.jsonl', 'r') as f:
    sample = json.loads(f.readline())

# Compute features from points
points = sample['points']
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

# Save features in BFT format (1, 37, T)
features_bft = features.T[np.newaxis, :, :]
np.save('test_features.npy', features_bft)

print(f'Saved features for word: {sample[\"word\"]}')
print(f'Shape: {features_bft.shape}')
"

# Run the CLI decoder
echo "Testing beam_decode_onnx_cli.py..."
python beam_decode_onnx_cli.py \
    --encoder ../../web-demo/encoder_web_ultra.onnx \
    --step ../../web-demo/rnnt_step_fp32.onnx \
    --features test_features.npy \
    --words words.txt \
    --meta runtime_meta.json \
    --D 256 \
    --beam 16 \
    --prune 6 \
    --topk 5