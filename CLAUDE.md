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

### Training Scripts (Primary Focus)

#### Current Production Script: `trained_models/nema1/train_transducer_personalized.py`
- **Latest Architecture**: Personalized RNN-T with end-to-end preprocessing pipeline
- **Key Features**:
  - Adaptive resampling (56-96 frames depending on trace length)
  - 37-dimensional features: kinematic (position, velocity, acceleration) + spatial (nearest keys)
  - Knowledge distillation support for smaller deployment models
  - Configurable character-hypothesis budget for downstream decoders
  - GPU/CPU auto-fallback optimized for RTX 4090M (16GB VRAM)

#### Legacy Script: `archive/train_transducer.py`
- **Earlier Implementation**: Basic RNN-T without personalization features
- **Differences**: Simpler feature extraction, no adaptive resampling, no distillation
- **Note**: May have validation WER metric inconsistencies vs. personalized version

### Data Format & Features
```json
{
  "word": "example",
  "points": [
    {"x": -0.784, "y": 0.214, "t": 0},
    {"x": 0.522, "y": -0.193, "t": 37}
  ]
}
```
- **Coordinates**: x,y ∈ [-1,1] with (0,0) at keyboard center
- **Timing**: t in milliseconds from gesture start
- **Vocabulary**: Lowercase letters + apostrophe only

### Hardware & Performance Optimization
- **Target Hardware**: RTX 4090M with 16GB VRAM
- **Precision**: bf16-mixed for training (avoids CUDA graph dtype conflicts)
- **Batch Size**: 256-320 optimized for memory usage
- **Optimizations**: TF32, cuDNN benchmarking, torch.compile when available

## Development Commands

### Training
```bash
# Run current personalized training
uv run python trained_models/nema1/train_transducer_personalized.py

# Run legacy training (for comparison)
uv run python archive/train_transducer.py

# Fast development run (single batch smoke test)
FAST_DEV_RUN=1 uv run python trained_models/nema1/train_transducer_personalized.py
```

### Dependencies
```bash
# Install all dependencies
uv sync

# Python version
uv run python --version  # Should be 3.12.x
```

### Data Validation
```bash
# Validate vocabulary system
uv run python trained_models/scripts/validate_vocab_system.py

# Split data for training
uv run python trained_models/scripts/split_data.py
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

## Key Configuration Differences

### Personalized vs Legacy Training
| Feature | Personalized | Legacy |
|---------|-------------|--------|
| Resampling | Adaptive 56-96 frames | Fixed length |
| Features | 37D with spatial awareness | 37D basic kinematic |
| Distillation | Teacher-student support | None |
| Validation | Configurable subset sampling | Full dataset |
| Precision | bf16-mixed optimized | bf16-mixed basic |

### Model Export Formats
- **ONNX**: Web inference via JavaScript/TypeScript
- **PTE**: Android on-device inference (PyTorch Mobile)
- **NeMo**: Full model checkpoints for continued training

## Data Paths & Structure
- **Training Data**: `data/train_final_train.jsonl` (642,909 samples)
- **Validation Data**: `data/train_final_val.jsonl` (33,838 samples)
- **Vocabulary**: `data/vocab.txt` (150k words from wordfreq)
- **Checkpoints**: Auto-generated `rnnt_checkpoints_YYYYMMDD_HHMMSS/`

## Important Notes

### Critical: Vocabulary & Token Handling

**CRITICAL UNDERSTANDING**: NeMo's `blank_as_pad=True` setting modifies vocabulary handling:

**Training Script Vocabulary** (`data/vocab.txt`):
- 29 tokens: `<blank>`, `'`, `a-z`, `<unk>`
- Loaded sequentially with `<blank>` at index 0

**NeMo Model Architecture with `blank_as_pad=True`**:
- **Purpose**: Enables efficient batch processing and RNNT model export
- **Effect**: Adds extra embedding dimension for blank token as padding
- **Vocab Size**: Parameter excludes blank token (29), but embedding has 30 dimensions
- **Blank Position**: `model.decoder.blank_idx = 29` (moved to end)
- **Index 0**: Still contains `<blank>` label but NOT the functional blank
- **Index 29**: Empty string `''` serves as the actual blank token

**Why This Architecture**:
- `blank_as_pad=True` is required for:
  - Efficient batched beam search
  - Proper ONNX export support
  - Zero tensor returns for padding optimization
- This is standard NeMo RNNT practice, not a bug

**Runtime Metadata Requirements**:
```json
{
  "vocab_size": 30,
  "blank_id": 29,  // CRITICAL: Functional blank at end
  "tokens": ["<blank>", "'", "a", ..., "z", "<unk>", ""]  // 30 tokens total
}
```

**ONNX Export Behavior**:
- ONNX models correctly output 30 logits
- The 30th dimension (index 29) is the functional blank
- Must use `blank_id: 29` for decoding
- Character mappings remain: `'a' → 2`, `'i' → 10`, `'s' → 20`

**Script Consistency Requirements**:
- All scripts must recognize 30-token output
- `make_runtime_meta.py`: Derive from model checkpoint (gets blank_idx=29)
- Export scripts: Preserve 30-token architecture
- Beam decoders: Use `blank_id: 29` consistently

### Validation WER Concerns
The user has concerns about validation WER metric consistency between training scripts. The personalized version may restrict validation datasets differently based on config, potentially affecting metric comparability. Monitor validation subset sampling configurations when comparing models.

### Model Selection Guidance
- **Use Personalized**: For production deployments requiring on-device personalization
- **Use Legacy**: For baseline comparisons or simpler deployment scenarios
- **Architecture**: Both use RNN-T but personalized has superior feature engineering

### Export/Deployment Parameters
Multiple export scripts exist with different optimization levels. Choose based on target platform:
- `export_pte_ultra.py`: Maximum optimization for Android
- `export_pte_fp32.py`: Full precision for accuracy-critical applications
- `export_onnx.py`: Web deployment

### Hardware Requirements
- **Training**: RTX 4090M or equivalent (16GB+ VRAM recommended)
- **Inference**: Mid to high-end smartphones for Android deployment
- **Dependencies**: Requires CUDA-capable PyTorch with NeMo toolkit