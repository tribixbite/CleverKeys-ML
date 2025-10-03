import numpy as np
import json
import sys
sys.path.insert(0, '../new')
from train_transducer_personalized import PersonalizedSwipeFeaturizer, resample_points, clamp

# Get hello data
import linecache
data_path = '../data/train_final_train.jsonl'
line = linecache.getline(data_path, 431621)
data = json.loads(line)
points, word = data['points'], data['word']

# Prepare features EXACTLY as in verify_pytorch.py
prepared = []
for idx, pt in enumerate(points):
    raw_x = float(pt.get('x', 0.0))
    raw_y = float(pt.get('y', 0.0))
    centered_x = raw_x * 2.0 - 1.0
    centered_y = raw_y * 2.0 - 1.0
    centered_x = clamp(centered_x, -1.5, 1.5)
    centered_y = clamp(centered_y, -1.5, 1.5)
    raw_t = float(pt.get('t', idx * 10.0))
    prepared.append({'x': centered_x, 'y': centered_y, 't': raw_t})

resampled = resample_points(prepared, 82)
featurizer = PersonalizedSwipeFeaturizer(mobile_features=False)
features = featurizer(resampled)

if features.shape[1] < 37:
    padding = np.zeros((features.shape[0], 37 - features.shape[1]), dtype=np.float32)
    features = np.concatenate([features, padding], axis=1)

# Save for comparison
np.save("../onnx_features.npy", features)
print(f"Saved ONNX features with shape {features.shape}")
