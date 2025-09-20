#!/usr/bin/env python3
"""Extract features from validation data for testing"""

import json
import numpy as np

# Read validation data
with open("../data/train_final_val.jsonl", "r") as f:
    lines = f.readlines()

# Find "is" in the validation set
for i, line in enumerate(lines[:100]):  # Check first 100
    data = json.loads(line)
    word = data.get("word", "")
    if word == "is":
        print(f"Found 'is' at line {i+1}")
        features = np.array(data["features"])
        print(f"Features shape: {features.shape}")

        # Save the features
        # Shape should be (37, T) but model expects (1, 37, T)
        if features.ndim == 2:
            features = features[np.newaxis, :, :]

        np.save("val_features_is.npy", features)
        print(f"Saved features to val_features_is.npy with shape {features.shape}")

        # Also check a few other words
        print("\nFirst 10 words in validation:")
        for j in range(min(10, len(lines))):
            d = json.loads(lines[j])
            print(f"  {j+1}. {d.get('word', 'N/A')}")
        break
else:
    print("'is' not found in first 100 lines")