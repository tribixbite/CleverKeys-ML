import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / 'new'))

import json
import numpy as np
from new.train_transducer_personalized import PersonalizedSwipeFeaturizer, determine_resample_target, resample_points, CONFIG

def main():
    with open('/home/will/git/swype/cleverkeys/data/train_final_train.jsonl', 'r') as f:
        test_data_line = f.readline()
    item = json.loads(test_data_line)
    featurizer = PersonalizedSwipeFeaturizer()

    # Mimic the dataset processing
    raw_points = item['points']
    start_t = float(raw_points[0].get("t", 0.0))
    normalized = [{'x': p.get('x', 0.0), 'y': p.get('y', 0.0), 't': p.get('t', 0.0) - start_t} for p in raw_points]

    target_len = determine_resample_target(len(normalized), CONFIG['preprocess'])
    processed = resample_points(normalized, target_len)
    features = featurizer(processed)

    print("--- Python Features ---")
    print("const features = new Float32Array([")
    for row in features:
        print(f"    {str(list(row))}, ")
    print("]);")

if __name__ == '__main__':
    main()