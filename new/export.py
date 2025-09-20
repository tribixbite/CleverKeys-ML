#!/usr/bin/env python3
"""
Export a trained NeMo RNN-T model to ONNX format.

This script first loads a model architecture, then loads the weights from a
PyTorch Lightning .ckpt file, saves the combined model to a .nemo file, and
finally exports the .nemo file to ONNX.

Usage:
    python new/export.py --checkpoint <path_to_ckpt_file> --output <path_to_output_onnx_file>
"""

import argparse
from pathlib import Path
import sys
import torch
from omegaconf import DictConfig

# Add the script's directory to the Python path
sys.path.append(str(Path(__file__).parent.absolute()))

from train_transducer_personalized import PersonalizedRNNTModel, CONFIG, build_model_config, load_vocab

def main():
    parser = argparse.ArgumentParser(description="Export NeMo model to ONNX.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to the input .ckpt checkpoint file.")
    parser.add_argument("--output", required=True, type=str, help="Path for the output ONNX file.")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    output_path = Path(args.output)
    nemo_path = output_path.with_suffix('.nemo')

    if not ckpt_path.exists():
        print(f"Error: Input checkpoint file not found at {ckpt_path}")
        return

    print("Step 1: Building model from configuration...")
    cfg = DictConfig(CONFIG)
    vocab = load_vocab(cfg.data.vocab_path)
    nemo_cfg = build_model_config(cfg, list(vocab.keys()))
    model = PersonalizedRNNTModel(cfg=nemo_cfg)

    print(f"Step 2: Loading weights from checkpoint: {ckpt_path}")
    # PyTorch 2.6+ requires explicit trust for pickled classes
    # We trust OmegaConf as it's part of our model's configuration
    import omegaconf
    with torch.serialization.safe_globals([omegaconf.dictconfig.DictConfig]):
        model.load_state_dict(torch.load(str(ckpt_path), weights_only=False)['state_dict'])

    print(f"Step 3: Saving temporary .nemo file to: {nemo_path}")
    model.save_to(str(nemo_path))

    print(f"Step 4: Exporting encoder from .nemo file to ONNX at: {output_path}")
    try:
        # Restore from the .nemo file to ensure all metadata is correct
        final_model = PersonalizedRNNTModel.restore_from(str(nemo_path))
        final_model.eval()
        final_model.export(str(output_path))
        print("Export successful!")
    except Exception as e:
        print(f"Error during ONNX export: {e}")
    finally:
        # Clean up the temporary .nemo file
        if nemo_path.exists():
            nemo_path.unlink()

if __name__ == "__main__":
    main()