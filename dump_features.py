import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / 'new'))

import json
import numpy as np
from new.train_transducer_personalized import PersonalizedSwipeFeaturizer, determine_resample_target, resample_points, CONFIG

def main():
    if len(sys.argv) < 2:
        print("Usage: python dump_features.py <line_number>")
        sys.exit(1)
    line_num = int(sys.argv[1])

    # Read the specific line from the training data
    line_content = None
    with open('/home/will/git/swype/cleverkeys/data/train_final_train.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i == line_num - 1:
                line_content = line
                break
    
    if not line_content:
        print(f"Error: Line {line_num} not found in file.")
        sys.exit(1)

    item = json.loads(line_content)
    word = item.get('word', 'unknown')

    featurizer = PersonalizedSwipeFeaturizer(mobile_features=True) # Match mobile preset

    # Mimic the dataset processing
    raw_points = item['points']
    
    from new.train_transducer_personalized import PersonalizedSwipeDataset
    normalized = PersonalizedSwipeDataset._prepare_points(raw_points)
    target_len = determine_resample_target(len(normalized), CONFIG['preprocess'])
    processed = resample_points(normalized, target_len)
    features = featurizer(processed)

    print(f"--- Python Features for '{word}' (line {line_num}) ---")
    print(f"// Shape: {features.shape}")
    print("const goldenFeatures = [")
    for i, row in enumerate(features):
        print(f"    [{', '.join(f'{x:.6f}' for x in row)}]{',' if i < len(features) - 1 else ''}")
    print("];")

if __name__ == '__main__':
    main()