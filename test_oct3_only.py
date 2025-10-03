#!/usr/bin/env python3
"""
Test ONLY the Oct 3 checkpoint - the correct architecture.
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

def test_oct3_checkpoint():
    """Test ONLY the Oct 3 checkpoint"""

    # ONLY Oct 3 checkpoint - correct architecture
    CHECKPOINT_PATH = "./9292025script/20251002/rnnt_checkpoints_medium_balanced_20251003_013434/lightning_logs/version_0/checkpoints/epoch=epoch=71-wer=val_wer=0.457.ckpt"

    print(f"Loading Oct 3 checkpoint ONLY")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
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
    print("Oct 3 Checkpoint Analysis (epoch 71, WER=0.457):")
    print("="*70)

    predictions = []
    word_lengths = []
    pred_lengths = []
    correct = 0

    # Test 100 samples
    test_count = 100
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

                if predicted == word:
                    correct += 1
                    print(f"  ✓ CORRECT: '{word}' → '{predicted}'")

                # Show first 10 predictions
                if idx < 10:
                    result = "✓" if predicted == word else "✗"
                    print(f"  {result} Val {idx:3d}: '{word:12s}' → '{predicted:12s}'")

            except Exception as e:
                predictions.append("")
                pred_lengths.append(0)

    # Analyze patterns
    print(f"\n\nStatistics:")
    print("-" * 50)

    print(f"Accuracy: {correct}/{test_count} = {correct/test_count*100:.1f}%")

    avg_word_len = np.mean(word_lengths)
    avg_pred_len = np.mean(pred_lengths)
    print(f"Average word length: {avg_word_len:.1f} characters")
    print(f"Average prediction length: {avg_pred_len:.1f} characters")
    print(f"Length ratio: {avg_pred_len/avg_word_len:.2f}")

    # Most common predictions
    print("\nMost common predictions:")
    pred_counter = Counter(predictions)
    for pred, count in pred_counter.most_common(10):
        pct = count / test_count * 100
        if pred == "":
            print(f"  '<empty>': {count} ({pct:.1f}%)")
        else:
            print(f"  '{pred}': {count} ({pct:.1f}%)")

if __name__ == '__main__':
    test_oct3_checkpoint()