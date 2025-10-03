#!/usr/bin/env python3
"""
Test the PyTorch checkpoint using correct preprocessing config from training.
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, 'new')
from train_transducer_personalized import (
    PersonalizedRNNTModel,
    PersonalizedSwipeDataset,
    PersonalizedSwipeFeaturizer,
)

def test_with_correct_preprocessing():
    """Test using the correct preprocessing defaults from training"""
    CHECKPOINT_PATH = "./9292025script/20251002/rnnt_checkpoints_medium_balanced_20251003_013434/lightning_logs/version_0/checkpoints/epoch=epoch=71-wer=val_wer=0.457.ckpt"

    # Load the PyTorch model
    print("Loading PyTorch model...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    model = PersonalizedRNNTModel(checkpoint['hyper_parameters']['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.freeze()

    # Define preprocessing config with defaults from training script
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }

    max_trace_len = 256

    # Create featurizer
    featurizer = PersonalizedSwipeFeaturizer(
        key_centers_path=None,
        mobile_features=False  # Desktop mode
    )

    # Load vocab
    vocab = {}
    with open('data/vocab.txt', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            token = line.strip()
            vocab[token] = i

    print(f"Vocab size: {len(vocab)}")

    # Create training dataset
    print("\nCreating training dataset...")
    train_dataset = PersonalizedSwipeDataset(
        manifest_path='data/train_final_train.jsonl',
        vocab=vocab,
        max_trace_len=max_trace_len,
        preprocess_cfg=preprocess_cfg,
        featurizer=featurizer,
        augmenter=None,
        is_training=False  # No augmentation for testing
    )

    # Test specific samples (including hello)
    test_indices = [
        431620,  # Index for hello (line 431621)
        99,      # Random early sample
        999,     # Random sample
    ]

    print("\n" + "="*70)
    print("Testing TRAINING samples with correct preprocessing:")
    print("="*70)

    for idx in test_indices:
        sample = train_dataset.samples[idx]
        word = sample['word']

        # Get preprocessed features
        features_tensor, feat_len, tokens_tensor, token_len = train_dataset[idx]

        print(f"\nSample {idx}: Word = '{word}'")
        print(f"  Features shape: {features_tensor.shape}")
        print(f"  Feature length: {feat_len.item()}")

        # Run transcription
        with torch.no_grad():
            try:
                # Use model's transcribe method
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

                print(f"  Prediction: '{predicted}'")
                if predicted == word:
                    print(f"  ✓ CORRECT!")
                else:
                    print(f"  ✗ WRONG (expected '{word}')")

            except Exception as e:
                print(f"  Error: {e}")

    # Test validation dataset
    print("\n" + "="*70)
    print("Testing VALIDATION samples:")
    print("="*70)

    val_dataset = PersonalizedSwipeDataset(
        manifest_path='data/train_final_val.jsonl',
        vocab=vocab,
        max_trace_len=max_trace_len,
        preprocess_cfg=preprocess_cfg,
        featurizer=featurizer,
        augmenter=None,
        is_training=False
    )

    # Test first 20 validation samples
    correct_count = 0
    for idx in range(20):
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

                result = "✓" if is_correct else "✗"
                print(f"  {result} Val {idx:2d}: '{word:12s}' → '{predicted:12s}'")

            except Exception as e:
                print(f"  Val {idx}: Error - {e}")

    print(f"\nValidation accuracy: {correct_count}/20 = {correct_count/20*100:.1f}%")

    # Look for specific words mentioned in validation logs
    print("\n" + "="*70)
    print("Searching for words seen in validation logs:")
    print("="*70)

    target_words = ['fatal', 'nearby', 'the', 'and', 'for', 'with', 'that']

    for target in target_words:
        # Find first occurrence of target word in validation
        found = False
        for idx, sample in enumerate(val_dataset.samples[:1000]):  # Check first 1000
            if sample['word'] == target:
                features_tensor, feat_len, tokens_tensor, token_len = val_dataset[idx]

                with torch.no_grad():
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

                    result = "✓" if predicted == target else "✗"
                    print(f"  {result} Found '{target}' at val index {idx}: predicted '{predicted}'")
                    found = True
                    break

        if not found:
            print(f"  '{target}' not found in first 1000 validation samples")

if __name__ == '__main__':
    test_with_correct_preprocessing()