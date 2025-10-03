#!/usr/bin/env python3
"""
Test the PyTorch checkpoint on various words to find patterns.
"""

import torch
import numpy as np
import json
import sys
import random
from pathlib import Path
import linecache

sys.path.insert(0, 'new')
from train_transducer_personalized import (
    PersonalizedRNNTModel,
    PersonalizedSwipeFeaturizer,
    resample_points,
    clamp
)

def get_test_data(line_number):
    """Get swipe data from specific line"""
    data_path = 'data/train_final_train.jsonl'
    line = linecache.getline(data_path, line_number)
    if line:
        data = json.loads(line)
        return data['points'], data['word']
    return None, None

def prepare_features(points):
    """Prepare features exactly as training does"""
    prepared = []
    for idx, pt in enumerate(points):
        raw_x = float(pt.get("x", 0.0))
        raw_y = float(pt.get("y", 0.0))
        centered_x = raw_x * 2.0 - 1.0
        centered_y = raw_y * 2.0 - 1.0
        centered_x = clamp(centered_x, -1.5, 1.5)
        centered_y = clamp(centered_y, -1.5, 1.5)
        raw_t = float(pt.get("t", idx * 10.0))
        prepared.append({"x": centered_x, "y": centered_y, "t": raw_t})

    resampled = resample_points(prepared, 82)
    featurizer = PersonalizedSwipeFeaturizer(mobile_features=False)
    features = featurizer(resampled)

    if features.shape[1] < 37:
        padding = np.zeros((features.shape[0], 37 - features.shape[1]), dtype=np.float32)
        features = np.concatenate([features, padding], axis=1)

    return features

def main():
    CHECKPOINT_PATH = "./9292025script/20251002/rnnt_checkpoints_medium_balanced_20251003_013434/lightning_logs/version_0/checkpoints/epoch=epoch=71-wer=val_wer=0.457.ckpt"

    # Load the PyTorch model
    print("Loading PyTorch model...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    model = PersonalizedRNNTModel(checkpoint['hyper_parameters']['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.freeze()

    # Test a variety of words - different lengths and patterns
    # Include some that might be in validation set
    test_lines = [
        # Original test cases
        431621,  # hello

        # Try to find 'fatal' and 'nearby' that were shown in logs
        # We'll sample randomly and look for these words
        *random.sample(range(1, 642909), 100)  # Random sample
    ]

    print("\nSearching for words that the model might predict correctly...")
    print("="*70)

    results = []

    for line_num in test_lines:
        points, word = get_test_data(line_num)
        if not points:
            continue

        # Prepare features
        features = prepare_features(points)
        features_tensor = torch.from_numpy(features.T).float().unsqueeze(0)
        feature_lengths = torch.tensor([features.shape[0]], dtype=torch.long)

        # Get prediction
        with torch.no_grad():
            try:
                if hasattr(model, 'transcribe'):
                    hypotheses = model.transcribe([features], batch_size=1)
                    if hypotheses and hypotheses[0]:
                        pred_obj = hypotheses[0]
                        if hasattr(pred_obj, 'text'):
                            predicted = pred_obj.text
                        else:
                            predicted = str(pred_obj).split("text='")[1].split("'")[0] if "text='" in str(pred_obj) else ""
                    else:
                        predicted = ""
                else:
                    predicted = ""
            except Exception as e:
                predicted = f"ERROR: {e}"

        # Check for interesting results
        is_correct = predicted == word
        is_partial = word.startswith(predicted) if predicted else False

        # Store result
        results.append({
            'line': line_num,
            'word': word,
            'predicted': predicted,
            'correct': is_correct,
            'partial': is_partial
        })

        # Print interesting cases
        if is_correct:
            print(f"✓ CORRECT! Line {line_num:6d}: '{word:12s}' → '{predicted:12s}'")
        elif word in ['fatal', 'nearby', 'the', 'and', 'for', 'that', 'with']:
            # Common words or words mentioned in logs
            print(f"  Line {line_num:6d}: '{word:12s}' → '{predicted:12s}' {'(partial)' if is_partial else ''}")

    print("\n" + "="*70)
    print("Summary Statistics:")
    print("="*70)

    # Calculate statistics
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    partial = sum(1 for r in results if r['partial'])

    print(f"Total words tested: {total}")
    print(f"Completely correct: {correct} ({correct/total*100:.1f}%)")
    print(f"Partial matches:    {partial} ({partial/total*100:.1f}%)")

    # Group by word length
    print("\nAccuracy by word length:")
    for length in range(2, 10):
        length_results = [r for r in results if len(r['word']) == length]
        if length_results:
            length_correct = sum(1 for r in length_results if r['correct'])
            print(f"  {length} chars: {length_correct}/{len(length_results)} = {length_correct/len(length_results)*100:.1f}%")

    # Show all correct predictions
    correct_results = [r for r in results if r['correct']]
    if correct_results:
        print(f"\nWords predicted correctly ({len(correct_results)} total):")
        for r in correct_results[:20]:  # Show first 20
            print(f"  '{r['word']}'")

    # Look for specific words
    print("\nSearching for specific words mentioned in validation logs...")
    for target_word in ['fatal', 'nearby']:
        matching = [r for r in results if r['word'] == target_word]
        if matching:
            for r in matching:
                print(f"  Found '{target_word}' at line {r['line']}: predicted '{r['predicted']}'")

if __name__ == '__main__':
    main()