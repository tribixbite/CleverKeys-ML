#!/usr/bin/env python3
"""
Test the PyTorch checkpoint using EXACT training preprocessing.
This ensures we're using the same feature extraction as training.
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
    PersonalizedSwipeDataset,
    PersonalizedSwipeFeaturizer,
    resample_points,
    determine_resample_target,
    clamp
)

def test_with_dataset_preprocessing():
    """Use the actual dataset class to ensure exact preprocessing"""
    CHECKPOINT_PATH = "./9292025script/20251002/rnnt_checkpoints_medium_balanced_20251003_013434/lightning_logs/version_0/checkpoints/epoch=epoch=71-wer=val_wer=0.457.ckpt"

    # Load the PyTorch model
    print("Loading PyTorch model...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    model = PersonalizedRNNTModel(checkpoint['hyper_parameters']['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.freeze()

    # Get the config from checkpoint
    cfg = checkpoint['hyper_parameters']['cfg']

    # Create dataset with EXACT same config as training
    print("\nCreating dataset with training config...")
    dataset = PersonalizedSwipeDataset(
        manifest_path='data/train_final_train.jsonl',
        vocab_path='data/vocab.txt',
        is_training=False,  # No augmentation for testing
        preprocess_cfg=cfg['preprocessing'],
        max_trace_len=cfg['preprocessing']['max_trace_len']
    )

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Vocab size: {len(dataset.vocab)} tokens")

    # Test specific samples
    test_indices = [
        431620,  # Index for hello (line 431621)
        99,      # Random early sample
        999,     # Random sample
        9999,    # Random sample
    ]

    print("\n" + "="*70)
    print("Testing with EXACT training preprocessing:")
    print("="*70)

    for idx in test_indices:
        # Get the sample directly
        sample = dataset.samples[idx]
        word = sample['word']

        # Process through dataset (exact training preprocessing)
        features_tensor, feat_len, tokens_tensor, token_len = dataset[idx]

        print(f"\nSample {idx}: Word = '{word}'")
        print(f"  Features shape: {features_tensor.shape}")
        print(f"  Feature length: {feat_len.item()}")
        print(f"  Token length: {token_len.item()}")

        # Prepare for model (batch dimension)
        features_batch = features_tensor.unsqueeze(0)  # [1, time, features]
        feat_len_batch = feat_len.unsqueeze(0)  # [1]

        # Run transcription
        with torch.no_grad():
            try:
                # Use model's transcribe method with numpy input
                features_numpy = features_tensor.numpy()
                hypotheses = model.transcribe([features_numpy], batch_size=1)

                if hypotheses and hypotheses[0]:
                    pred_obj = hypotheses[0]
                    if hasattr(pred_obj, 'text'):
                        predicted = pred_obj.text
                    else:
                        # Parse from string representation
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

            except Exception as e:
                print(f"  Error: {e}")

    # Now test random validation samples
    print("\n" + "="*70)
    print("Testing random VALIDATION samples:")
    print("="*70)

    val_dataset = PersonalizedSwipeDataset(
        manifest_path='data/train_final_val.jsonl',
        vocab_path='data/vocab.txt',
        is_training=False,
        preprocess_cfg=cfg['preprocessing'],
        max_trace_len=cfg['preprocessing']['max_trace_len']
    )

    # Test first 10 validation samples
    for idx in range(10):
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

                result = "✓" if predicted == word else "✗"
                print(f"  {result} Val {idx}: '{word:12s}' → '{predicted:12s}'")

            except Exception as e:
                print(f"  Val {idx}: Error - {e}")

if __name__ == '__main__':
    test_with_dataset_preprocessing()