# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CleverKeys is a privacy-first gesture typing system for local-only modern keyboards. It's a production-ready swipe gesture recognition system using Transformer models with CTC loss for keyboard input prediction. No data leaves the user's device.

## Architecture

The system consists of three core components:

1. **Feature Engineering (`SwipeFeaturizer`)**: Transforms raw (x, y, t) gesture points into rich feature vectors including kinematics (velocity, acceleration) and spatial context (nearest keyboard keys)

2. **Neural Model (`GestureCTCModel`)**: Transformer Encoder that outputs character probability distributions using CTC loss - chosen for inference speed (non-autoregressive), simplicity (encoder-only), and alignment-free training

3. **Decoder (`pyctcdecode`)**: Beam search decoder with vocabulary trie and KenLM language model for high accuracy without complex neural models

## Common Commands

### Environment Setup
```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync project dependencies
uv sync
```

### Training
```bash
# Run main training script
uv run python train.py

# Monitor training with TensorBoard
tensorboard --logdir logs
```

### Data Processing
```bash
# Split data into train/validation sets
uv run python scripts/split_data.py

# Generate vocabulary wordlists
cd vocab/wordlist_gen && ./generate_wordlist.sh
```

### Model Export
```bash
# Export to ONNX format for web deployment
uv run python scripts/onnx_export.py

# Export to PTE format for Android deployment  
uv run python scripts/pte_export.py
```

### Code Formatting
```bash
# Format Python code
uv format
```

## Project Structure

- `train.py` - Main training script with complete pipeline (featurizer, model, decoder)
- `data/` - Training/validation data in JSONL format (word + gesture points)
- `vocab/` - Vocabulary management (153k+ word list, generation scripts)
- `scripts/` - Export utilities for ONNX and PTE formats
- `web-demo/` - Browser-based demo with ONNX runtime
- `checkpoints/` - Saved model checkpoints during training
- `exports/` - Exported models (TorchScript, ONNX, PTE)
- `logs/` - TensorBoard training logs

## Key Configuration

Training configuration is centralized in the `CONFIG` dict at the top of `train.py`:
- Model: 256d embeddings, 4 attention heads, 6 encoder layers
- Training: batch_size=256, lr=3e-4, 50 epochs, mixed precision
- Decoding: beam_width=100, KenLM language model with alpha=0.5, beta=1.5

## Data Format

Training data uses JSONL format:
```json
{
  "word": "example",
  "points": [
    {"x": 0.1, "y": 0.2, "t": 0.0},
    {"x": 0.15, "y": 0.25, "t": 0.1}
  ]
}
```

## Deployment Notes

- **Android**: Convert `exports/model.pt` to `.pte` using ExecuTorch toolchain
- **Web**: Use `exports/model.onnx` with ONNX.js runtime (demo in `web-demo/`)
- **Dynamic Vocabulary**: The decoder supports adding user-specific words at runtime without retraining

## Development Tips

- The model and decoder are decoupled - neural model stays static while vocabulary can be dynamic
- CTC architecture enables fast single-pass inference critical for real-time keyboard response
- Strong decoding with vocabulary trie + language model achieves high accuracy without large models