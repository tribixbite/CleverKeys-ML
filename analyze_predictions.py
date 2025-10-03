#!/usr/bin/env python3
"""
Analyze prediction patterns to understand why the model is failing.
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, 'new')
from train_transducer_personalized import (
    PersonalizedRNNTModel,
    PersonalizedSwipeDataset,
    PersonalizedSwipeFeaturizer,
)

def analyze_predictions():
    """Analyze patterns in model predictions"""

    # Test with curriculum stage 1 checkpoint
    checkpoint_path = "./9292025script/rnnt_checkpoints_curriculum_stage1_20250929_193613/lightning_logs/version_0/checkpoints/epoch=epoch=74-wer=val_wer=0.192.ckpt"

    print(f"Loading checkpoint: {Path(checkpoint_path).name}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = PersonalizedRNNTModel(checkpoint['hyper_parameters']['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.freeze()

    # Setup preprocessing
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }

    featurizer = PersonalizedSwipeFeaturizer(
        key_centers_path=None,
        mobile_features=False
    )

    vocab = {}
    with open('data/vocab.txt', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            token = line.strip()
            vocab[token] = i

    # Test on validation dataset
    val_dataset = PersonalizedSwipeDataset(
        manifest_path='data/train_final_val.jsonl',
        vocab=vocab,
        max_trace_len=256,
        preprocess_cfg=preprocess_cfg,
        featurizer=featurizer,
        augmenter=None,
        is_training=False
    )

    print("\n" + "="*70)
    print("Analyzing prediction patterns:")
    print("="*70)

    predictions = []
    word_lengths = []
    pred_lengths = []

    # Test 200 samples
    test_count = 200
    for idx in range(test_count):
        sample = val_dataset.samples[idx]
        word = sample['word']
        word_lengths.append(len(word))

        features_tensor, feat_len, tokens_tensor, token_len = val_dataset[idx]

        with torch.no_grad():
            try:
                features_numpy = features_tensor.numpy()
                hypotheses = model.transcribe([features_numpy], batch_size=1)

                if hypotheses and hypotheses[0]:
                    pred_obj = hypotheses[0]
                    if hasattr(pred_obj, 'text'):
                        predicted = pred_obj.text
                    else:
                        pred_str = str(pred_obj)
                        if "text='" in pred_str:
                            predicted = pred_str.split("text='")[1].split("'")[0]
                        else:
                            predicted = ""
                else:
                    predicted = ""

                predictions.append(predicted)
                pred_lengths.append(len(predicted))

            except Exception as e:
                predictions.append("")
                pred_lengths.append(0)

    # Analyze patterns
    print(f"\nStatistics over {test_count} validation samples:")
    print("-" * 50)

    # Length analysis
    avg_word_len = np.mean(word_lengths)
    avg_pred_len = np.mean(pred_lengths)
    print(f"Average word length: {avg_word_len:.1f} characters")
    print(f"Average prediction length: {avg_pred_len:.1f} characters")
    print(f"Length ratio: {avg_pred_len/avg_word_len:.2f}")

    # Distribution of prediction lengths
    print("\nPrediction length distribution:")
    length_counter = Counter(pred_lengths)
    for length in sorted(length_counter.keys())[:10]:
        count = length_counter[length]
        pct = count / test_count * 100
        print(f"  {length} chars: {count} ({pct:.1f}%)")

    # Most common predictions
    print("\nMost common predictions:")
    pred_counter = Counter(predictions)
    for pred, count in pred_counter.most_common(10):
        pct = count / test_count * 100
        if pred == "":
            print(f"  '<empty>': {count} ({pct:.1f}%)")
        else:
            print(f"  '{pred}': {count} ({pct:.1f}%)")

    # Character frequency in predictions
    all_pred_chars = ''.join(predictions)
    if all_pred_chars:
        char_counter = Counter(all_pred_chars)
        print("\nMost common predicted characters:")
        for char, count in char_counter.most_common(10):
            print(f"  '{char}': {count}")

    # Check if model is stuck in certain patterns
    print("\nChecking for repeated patterns:")
    single_char_preds = [p for p in predictions if len(p) == 1]
    if single_char_preds:
        print(f"  Single character predictions: {len(single_char_preds)}/{test_count} ({len(single_char_preds)/test_count*100:.1f}%)")
        single_char_counter = Counter(single_char_preds)
        print(f"  Most common single chars: {dict(single_char_counter.most_common(5))}")

    empty_preds = [p for p in predictions if len(p) == 0]
    print(f"  Empty predictions: {len(empty_preds)}/{test_count} ({len(empty_preds)/test_count*100:.1f}%)")

    # Test if model responds differently to different input lengths
    print("\nPrediction length by input word length:")
    for target_len in range(2, 8):
        indices = [i for i, wl in enumerate(word_lengths) if wl == target_len]
        if indices:
            avg_pred_for_len = np.mean([pred_lengths[i] for i in indices])
            print(f"  Words with {target_len} chars → avg prediction: {avg_pred_for_len:.1f} chars")

if __name__ == '__main__':
    analyze_predictions()