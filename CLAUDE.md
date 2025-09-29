# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CleverKeys is a privacy-first gesture typing system for local-only modern keyboards - a high-performance Gboard alternative that never sends data off-device. The project implements RNN-Transducer (RNN-T) models using NeMo for gesture swipe recognition on Android keyboards, with support for personalized on-device learning.

## Architecture & Key Components

### Model Architecture
- **Base Model**: RNN-Transducer (RNN-T) with Conformer encoder for superior accuracy over CTC models
- **Key Advantage**: Models output dependencies P(y_i | y_1...y_{i-1}, x) unlike CTC, providing 40-50% WER reduction
- **Encoder**: Conformer with multi-head attention, convolutional modules, and feed-forward layers
- **Decoder**: LSTM-based prediction network for character sequence modeling
- **Joint Network**: Combines encoder and decoder outputs for final character predictions

### Training Script: `new/train_transducer_personalized.py`
- **Latest Architecture**: Personalized RNN-T with end-to-end preprocessing pipeline
- **Key Features**:
  - Adaptive resampling (56-96 frames depending on trace length)
  - 37-dimensional features: kinematic (position, velocity, acceleration) + spatial (nearest keys)
  - Knowledge distillation support for smaller deployment models
  - Configurable character-hypothesis budget for downstream decoders
  - GPU/CPU auto-fallback optimized for RTX 4090M (16GB VRAM)
  - Frequency-aware weighted sampling to handle word frequency imbalance

### Data Format & Coordinate System

**CRITICAL BUG**: The current `_prepare_points` method in `new/train_transducer_personalized.py` (lines 609-613) incorrectly assumes coordinates are already in [-1,1] range. The actual dataset uses:

**Dataset Format**:
```json
{
  "word": "example",
  "points": [
    {"x": 0.784, "y": 0.214, "t": 0},
    {"x": 0.522, "y": 0.193, "t": 37}
  ]
}
```
- **Coordinates**: x,y ∈ [0,1] where (0,0) is top-left corner of Q key
- **Coordinate mapping**: Top-right P is (1.0, 0.0), Bottom-left Z is (0.15, 1.0)
- **Required transformation**: `centered_x = raw_x * 2.0 - 1.0` to convert [0,1] → [-1,1]
- **Timing**: t in milliseconds from gesture start

**Fix Required**:
```python
# Current (INCORRECT):
centered_x = clamp(raw_x, -1.0, 1.0)  # Assumes already in [-1,1]

# Should be:
centered_x = raw_x * 2.0 - 1.0  # Transform [0,1] → [-1,1]
centered_y = raw_y * 2.0 - 1.0
```

### Hardware & Performance Optimization
- **Target Hardware**: RTX 4090M with 16GB VRAM
- **Precision**: bf16-mixed for training (avoids CUDA graph dtype conflicts)
- **Batch Size**: 320-400 optimized for 16GB memory
- **Optimizations**: TF32, cuDNN benchmarking, torch.compile when available

## Frequency-Aware Training

### The Problem
Natural language has massive frequency imbalance - words like "the", "and", "to" appear thousands of times more frequently than other words. Without intervention, training plateaus as the model overfits to common words.

### Solution: Sampling Profiles
The project uses weighted random sampling with different profiles defined in `new/sampling_profiles.py`:

**Key Profiles**:
- `ultra_common_suppressed`: Heavily suppress top 100 most common words
- `rare_focused`: Boost words appearing <1000 times
- `curriculum_stage[1-4]`: Progressive learning from common to rare
- `validation_balanced`: Consistent validation metric

### Comprehensive Training Script
```bash
# Run curriculum learning (recommended)
./train_comprehensive.sh curriculum

# Run frequency band training
./train_comprehensive.sh frequency

# Run all strategies
./train_comprehensive.sh all

# Quick test
./train_comprehensive.sh test
```

## Development Commands

### Training
```bash
# Run personalized training with specific profile
uv run python new/train_transducer_personalized.py --profile rare_focused

# Fast development run (single batch smoke test)
FAST_DEV_RUN=1 uv run python new/train_transducer_personalized.py

# Comprehensive training
./train_comprehensive.sh curriculum
```

### Dependencies
```bash
# Install all dependencies
uv sync

# Python version
uv run python --version  # Should be 3.12.x
```

### Model Export & Deployment
```bash
# Export to ONNX for web inference
uv run python trained_models/nema1/export_onnx.py

# Export to PyTorch Mobile (.pte) for Android
uv run python trained_models/nema1/export_pte_ultra.py

# Beam search CLI with ONNX
uv run python trained_models/nema1/beam_decode_onnx_cli.py
```

## Data Paths & Structure
- **Training Data**: `data/train_final_train.jsonl` (642,909 samples)
- **Validation Data**: `data/train_final_val.jsonl` (33,838 samples)
- **Vocabulary**: `data/vocab.txt` (29 tokens: `<blank>`, `'`, `a-z`, `<unk>`)
- **Checkpoints**: Auto-generated `rnnt_checkpoints_PROFILE_YYYYMMDD_HHMMSS/`

## Critical Notes

### Vocabulary & Token Handling
NeMo's `blank_as_pad=True` setting modifies vocabulary handling:
- **Model output**: 30 logits (29 vocab tokens + functional blank at index 29)
- **Blank token**: Index 29 is the functional blank for RNN-T decoding
- **Character mappings**: `'a' → 2`, `'i' → 10`, `'s' → 20` etc.

### Known Issues
1. **Coordinate System Bug**: Training script assumes [-1,1] but data is [0,1] - needs fix
2. **Checkpoint Resume**: Script creates new checkpoint dirs per profile, breaking continuity
3. **Max Epochs Control**: Training script doesn't accept --max-epochs parameter yet

### Export/Deployment Parameters
Multiple export scripts exist with different optimization levels:
- `export_pte_ultra.py`: Maximum optimization for Android
- `export_pte_fp32.py`: Full precision for accuracy-critical applications
- `export_onnx.py`: Web deployment

### Hardware Requirements
- **Training**: RTX 4090M or equivalent (16GB+ VRAM recommended)
- **Inference**: Mid to high-end smartphones for Android deployment
- **Dependencies**: Requires CUDA-capable PyTorch with NeMo toolkit