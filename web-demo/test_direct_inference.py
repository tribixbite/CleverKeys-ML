#!/usr/bin/env python
"""Test inference directly with the checkpoint to verify model works."""

import torch
import json
import sys
sys.path.append('../trained_models/nema1')
from train_transducer_personalized import (
    SwipeRNNTModule,
    PersonalizedSwipeFeaturizer,
    SwipeDataModule
)

def test_direct_inference():
    """Test with the actual checkpoint and training code."""

    # Load checkpoint
    checkpoint_path = '../rnnt_checkpoints_default_20250921_043756/lightning_logs/version_0/checkpoints/epoch=epoch=64-wer=val_wer=0.232.ckpt'

    print(f"Loading checkpoint: {checkpoint_path}")
    model = SwipeRNNTModule.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu'
    )
    model.eval()

    # Create data module to get samples
    data_module = SwipeDataModule(
        train_path='../data/train_final_train.jsonl',
        val_path='../data/train_final_val.jsonl',
        batch_size=1,
        num_workers=0
    )
    data_module.setup()

    # Get validation dataloader
    val_loader = data_module.val_dataloader()

    # Test first 10 samples
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 10:
                break

            # Get inputs
            audio_signal = batch['audio_signal']
            audio_lengths = batch['audio_signal_length']
            targets = batch['targets']
            target_lengths = batch['target_length']

            # Run model decoding
            hypotheses = model.model.decoding.rnnt_decoder_predictions_tensor(
                audio_signal,
                audio_lengths,
                return_hypotheses=True
            )

            # Get prediction
            if hypotheses and len(hypotheses[0]) > 0:
                best_hyp = hypotheses[0][0]  # First hypothesis
                predicted = best_hyp.text if hasattr(best_hyp, 'text') else ''
            else:
                predicted = ''

            # Get target text
            target_ids = targets[0][:target_lengths[0]].cpu().numpy()
            vocab = data_module.val_dataset.vocab
            inv_vocab = {v: k for k, v in vocab.items()}
            target_text = ''.join([inv_vocab.get(int(tid), '') for tid in target_ids])

            print(f"\nSample {batch_idx + 1}:")
            print(f"  Target: '{target_text}'")
            print(f"  Predicted: '{predicted}'")

            if predicted == target_text:
                print("  ✓ Correct!")
                correct += 1
            else:
                print("  ✗ Incorrect")

            total += 1

    print(f"\n=== Results ===")
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

if __name__ == "__main__":
    test_direct_inference()