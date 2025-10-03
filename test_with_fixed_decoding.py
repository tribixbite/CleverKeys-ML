#!/usr/bin/env python3
"""
Test the Oct 3 checkpoint with FIXED decoding parameters.
The issue: max_symbols is set to 13-15, limiting predictions to 1-2 chars.
The fix: Override max_symbols to a reasonable value like 100.
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path
from omegaconf import open_dict

sys.path.insert(0, 'new')
from train_transducer_personalized import (
    PersonalizedRNNTModel,
    PersonalizedSwipeDataset,
    PersonalizedSwipeFeaturizer,
)

def test_with_fixed_decoding():
    """Test with corrected max_symbols parameter"""

    CHECKPOINT_PATH = "./9292025script/20251002/rnnt_checkpoints_medium_balanced_20251003_013434/lightning_logs/version_0/checkpoints/epoch=epoch=71-wer=val_wer=0.457.ckpt"

    print("Loading Oct 3 checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    model = PersonalizedRNNTModel(checkpoint['hyper_parameters']['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.freeze()

    # FIX THE DECODING CONFIG!
    print("\nFixing decoding config...")
    print(f"  Original max_symbols (greedy): {model.cfg.decoding.greedy.max_symbols}")
    print(f"  Original max_symbols (greedy_batch): {model.cfg.decoding.greedy_batch.max_symbols}")

    with open_dict(model.cfg.decoding):
        model.cfg.decoding.greedy.max_symbols = 100
        model.cfg.decoding.greedy_batch.max_symbols = 100

    print(f"  Fixed max_symbols (greedy): {model.cfg.decoding.greedy.max_symbols}")
    print(f"  Fixed max_symbols (greedy_batch): {model.cfg.decoding.greedy_batch.max_symbols}")

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
    print("Testing with FIXED decoding (max_symbols=100):")
    print("="*70)

    correct_count = 0
    test_count = 50

    for idx in range(test_count):
        sample = val_dataset.samples[idx]
        word = sample['word']

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

                is_correct = predicted == word
                if is_correct:
                    correct_count += 1

                # Show first 20 predictions
                if idx < 20:
                    result = "✓" if is_correct else "✗"
                    print(f"  {result} Val {idx:3d}: '{word:12s}' → '{predicted:12s}'")

            except Exception as e:
                print(f"  Val {idx}: Error - {e}")

    print(f"\nValidation accuracy: {correct_count}/{test_count} = {correct_count/test_count*100:.1f}%")

    # Test training samples including hello
    print("\n" + "="*70)
    print("Testing training samples including 'hello':")
    print("="*70)

    train_dataset = PersonalizedSwipeDataset(
        manifest_path='data/train_final_train.jsonl',
        vocab=vocab,
        max_trace_len=256,
        preprocess_cfg=preprocess_cfg,
        featurizer=featurizer,
        augmenter=None,
        is_training=False
    )

    test_indices = [431620, 0, 100, 1000, 10000]  # Including hello

    for idx in test_indices:
        sample = train_dataset.samples[idx]
        word = sample['word']

        features_tensor, feat_len, tokens_tensor, token_len = train_dataset[idx]

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

                is_correct = predicted == word
                result = "✓" if is_correct else "✗"
                print(f"  {result} Train {idx:6d}: '{word:12s}' → '{predicted:12s}'")

            except Exception as e:
                print(f"  Train {idx}: Error - {e}")

if __name__ == '__main__':
    test_with_fixed_decoding()