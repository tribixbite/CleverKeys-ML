# CleverKeys - privacy-first gesture typing system for local-only modern keyboards.

no data leaves your device.

A production-ready swipe gesture recognition system using Transformer models with CTC loss for keyboard input prediction.

## Setup

This project uses `uv` for dependency management. Make sure you have `uv` installed:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

All dependencies are already configured in `pyproject.toml`. To install them:

```bash
# Dependencies are automatically installed when running with uv
uv sync
```

## Training

To run the training script:

```bash
uv run python train.py
```

### Configuration

Before training, update the paths in the `CONFIG` dictionary at the top of `train.py`:

- `train_data_path`: Path to your training data (JSONL format)
- `val_data_path`: Path to your validation data (JSONL format) 
- `vocab_path`: Path to your vocabulary file (text file with one word per line)

### Data Format

The training data should be in JSONL format with each line containing:
```json
{
  "word": "example",
  "points": [
    {"x": 0.1, "y": 0.2, "t": 0.0},
    {"x": 0.15, "y": 0.25, "t": 0.1},
    ...
  ]
}
```

## Model Architecture

- **Base Model**: Transformer Encoder with CTC loss
- **Features**: Kinematic features (position, velocity, acceleration) + spatial features (nearest keys)
- **Decoding**: pyctcdecode with KenLM language model for improved accuracy

## Output

The training script will:
1. Save checkpoints to `checkpoints/`
2. Log training metrics to `logs/` (viewable with TensorBoard)
3. Export models to `exports/` in both TorchScript and ONNX formats

## Dependencies

- PyTorch 2.8.0+ with CUDA support
- TensorBoard for training visualization
- Hugging Face Hub for language model downloads
- pyctcdecode + kenlm for CTC decoding
- NumPy, tqdm for utilities
