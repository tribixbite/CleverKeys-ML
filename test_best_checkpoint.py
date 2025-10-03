#!/usr/bin/env python3
"""
Test the BEST PyTorch checkpoint (epoch 99, WER=0.201)
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

def test_best_checkpoint():
    """Test the best available checkpoint"""
    # Try to find the best checkpoint - using curriculum stage 1 which has same architecture
    checkpoint_paths = [
        "./9292025script/rnnt_checkpoints_curriculum_stage1_20250929_193613/lightning_logs/version_0/checkpoints/epoch=epoch=74-wer=val_wer=0.192.ckpt",
        "./9292025script/rnnt_checkpoints_curriculum_stage1_20250929_193613/lightning_logs/version_0/checkpoints/epoch=epoch=99-wer=val_wer=0.201.ckpt",
        "./9292025script/20251002/rnnt_checkpoints_medium_balanced_20251003_013434/lightning_logs/version_0/checkpoints/epoch=epoch=71-wer=val_wer=0.457.ckpt",
    ]

    checkpoint_path = None
    for path in checkpoint_paths:
        if Path(path).exists():
            checkpoint_path = path
            break

    if not checkpoint_path:
        print("No better checkpoint found. Looking for any checkpoint with lower WER...")
        # List all checkpoints
        checkpoint_dir = Path("./9292025script/20251002/rnnt_checkpoints_medium_balanced_20251003_013434/lightning_logs/version_0/checkpoints/")
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            # Sort by WER value in filename
            checkpoints_with_wer = []
            for ckpt in checkpoints:
                if "wer=" in ckpt.name:
                    wer_str = ckpt.name.split("wer=")[-1].replace(".ckpt", "")
                    try:
                        wer = float(wer_str.split("=")[-1])
                        checkpoints_with_wer.append((wer, str(ckpt)))
                    except:
                        pass

            if checkpoints_with_wer:
                checkpoints_with_wer.sort()  # Sort by WER
                best_wer, checkpoint_path = checkpoints_with_wer[0]
                print(f"Found best checkpoint with WER={best_wer}: {Path(checkpoint_path).name}")

    if not checkpoint_path:
        print("No checkpoints found!")
        return

    # Load the PyTorch model
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

    max_trace_len = 256

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
        max_trace_len=max_trace_len,
        preprocess_cfg=preprocess_cfg,
        featurizer=featurizer,
        augmenter=None,
        is_training=False
    )

    print("\n" + "="*70)
    print("Testing validation samples with BEST checkpoint:")
    print("="*70)

    correct_count = 0
    total_count = 50  # Test more samples

    for idx in range(total_count):
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

                # Print first 20 and any correct predictions
                if idx < 20 or is_correct:
                    result = "✓" if is_correct else "✗"
                    print(f"  {result} Val {idx:3d}: '{word:12s}' → '{predicted:12s}'")

            except Exception as e:
                print(f"  Val {idx}: Error - {e}")

    print(f"\nValidation accuracy: {correct_count}/{total_count} = {correct_count/total_count*100:.1f}%")

    # Also test training samples
    print("\n" + "="*70)
    print("Testing training samples including 'hello':")
    print("="*70)

    train_dataset = PersonalizedSwipeDataset(
        manifest_path='data/train_final_train.jsonl',
        vocab=vocab,
        max_trace_len=max_trace_len,
        preprocess_cfg=preprocess_cfg,
        featurizer=featurizer,
        augmenter=None,
        is_training=False
    )

    test_indices = [
        431620,  # hello
        0,       # First sample
        1,
        2,
        100,
        1000,
        10000,
    ]

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
    test_best_checkpoint()