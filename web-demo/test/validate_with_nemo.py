#!/usr/bin/env python3
"""
Validate using NeMo's actual validation pipeline to see what the expected behavior is
"""

import json
import torch
import nemo.collections.asr as nemo_asr
from pathlib import Path


def main():
    checkpoint_path = '/home/will/git/swype/cleverkeys/9292025script/rnnt_checkpoints_curriculum_stage1_20250929_193613/lightning_logs/version_0/checkpoints/epoch=epoch=74-wer=val_wer=0.192.ckpt'

    print(f"Loading checkpoint: {checkpoint_path}")
    model = nemo_asr.models.EncDecRNNTModel.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model = model.eval()

    # Check model configuration
    print(f"Model config:")
    print(f"  Encoder: {model.cfg.encoder._target_ if hasattr(model.cfg.encoder, '_target_') else 'unknown'}")
    print(f"  Decoder layers: {model.cfg.decoder.prednet.pred_rnn_layers}")
    print(f"  Decoder hidden: {model.cfg.decoder.prednet.pred_hidden}")
    print(f"  Vocab size: {len(model.joint.vocabulary)}")
    print(f"  Blank ID: {model.decoder.blank_idx}")

    # Let's check what the model expects for features
    print(f"\nPreprocessor config:")
    print(f"  Type: {model.cfg.preprocessor._target_ if hasattr(model.cfg.preprocessor, '_target_') else 'unknown'}")
    if hasattr(model.cfg.preprocessor, 'features'):
        print(f"  Features: {model.cfg.preprocessor.features}")
    if hasattr(model.cfg.preprocessor, 'n_fft'):
        print(f"  n_fft: {model.cfg.preprocessor.n_fft}")

    # Print full preprocessor config
    print(f"\nFull preprocessor config:")
    for key in dir(model.cfg.preprocessor):
        if not key.startswith('_'):
            print(f"  {key}: {getattr(model.cfg.preprocessor, key, 'N/A')}")

    # Check what feature dimension the model expects
    print(f"\nEncoder input dimension: {model.cfg.encoder.feat_in}")

    # Test with a sample from validation data
    print("\n" + "="*60)
    print("Testing with actual validation sample...")

    # Read a validation sample
    import linecache
    data_path = '/home/will/git/swype/cleverkeys/data/train_final_val.jsonl'

    # Get first few samples
    for line_num in [1, 2, 3]:
        line = linecache.getline(data_path, line_num)
        if line:
            data = json.loads(line)
            word = data['word']
            points = data['points']
            print(f"\nSample {line_num}: '{word}' with {len(points)} points")

            # Try to process with model's preprocessor
            # Note: NeMo models expect audio signal input, but we have swipe data
            # This is the fundamental issue - the model was trained on swipe features
            # but NeMo's preprocessor expects audio


if __name__ == '__main__':
    main()