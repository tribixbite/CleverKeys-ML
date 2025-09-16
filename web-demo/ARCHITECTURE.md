# CleverKeys RNNT Architecture - Complete End-to-End Documentation

**Version**: 2025-09-16
**Model**: RNNT Transducer (Conformer + LSTM + Joint Network)
**Purpose**: Privacy-first gesture typing for local keyboards

## Overview

CleverKeys uses an RNN Transducer (RNNT) architecture for gesture typing - converting swipe traces into text predictions. The system consists of training, export, and inference phases with specific file formats and data flows.

---

## 1. Training Architecture

### 1.1 Training Script: `train_transducer.py`
**Location**: `trained_models/nema1/train_transducer.py`
**Last Modified**: 2025-09-16 04:40:10 -0400

**Model Configuration**:
```python
CONFIG = {
    "model": {
        "encoder": {
            "feat_in": 37,           # 9 kinematic + 28 RBF key features
            "d_model": 256,          # Hidden dimension
            "n_heads": 4,            # Attention heads
            "num_layers": 8,         # Conformer layers
            "conv_kernel_size": 31,  # Conv layer kernel
            "subsampling_factor": 2, # Time reduction factor
        },
        "decoder": {
            "pred_hidden": 320,      # LSTM hidden size
            "pred_rnn_layers": 2,    # LSTM layers
        },
        "joint": {
            "joint_hidden": 512,     # Joint network hidden size
        }
    }
}
```

**Loss Function**: RNNT Loss (alignment-free sequence-to-sequence)
- Automatically handles variable-length inputs and outputs
- No forced alignment required (unlike CTC)
- Supports blank token for streaming inference

**Training Data Format** (JSONL):
```json
{
    "word": "example",
    "points": [
        {"x": 0.1, "y": 0.2, "t": 0.0},
        {"x": 0.15, "y": 0.25, "t": 0.1}
    ]
}
```

**Coordinate Space**: [-1, 1] normalized from QWERTY keyboard layout

### 1.2 Model Class: `GestureRNNTModel`
**Location**: `trained_models/nema1/model_class.py` (referenced)

**Components**:
1. **Encoder**: Conformer (Transformer + CNN hybrid)
2. **Decoder**: LSTM prediction network
3. **Joint**: Feed-forward network combining encoder/decoder

### 1.3 Feature Engineering: `SwipeFeaturizer`
**Location**: `trained_models/nema1/swipe_data_utils.py`
**Last Modified**: 2025-09-16 05:20:36 -0400

**37D Feature Vector Structure**:

**9 Kinematic Features**:
1. `x` - X coordinate [-1, 1]
2. `y` - Y coordinate [-1, 1]
3. `t` - Relative time from trace start (seconds)
4. `vx` - X velocity (units/second)
5. `vy` - Y velocity (units/second)
6. `speed` - Combined velocity magnitude
7. `ax` - X acceleration (units/second²)
8. `ay` - Y acceleration (units/second²)
9. `acc` - Combined acceleration magnitude

**28 RBF Key Features** (Gaussian RBF for each character):
```python
for char in "abcdefghijklmnopqrstuvwxyz'":
    key_center = key_centers[char]  # [kx, ky] in [-1,1] space
    distance_sq = (x - kx)² + (y - ky)²
    rbf_value = exp(-distance_sq / (2 * σ²))  # σ = 0.2
```

**QWERTY Key Centers** ([-1, 1] coordinate space):
```python
key_centers = {
    'q': [-1.0, 1.0], 'w': [-0.8, 1.0], 'e': [-0.4, 1.0], ...
    'a': [-0.8, 0.33], 's': [-0.6, 0.33], 'd': [-0.4, 0.33], ...
    'z': [-0.8, -0.33], 'x': [-0.6, -0.33], 'c': [-0.4, -0.33], ...
}
```

### 1.4 Vocabulary System
**Vocabulary Size**: 29 tokens
**Tokens**: `["<blank>", "'", "a", "b", ..., "z", "<unk>"]`
- `blank_id = 0` - Used for RNNT blank transitions
- `unk_id = 28` - Out-of-vocabulary token

---

## 2. Training Process

### 2.1 Data Flow
```
Raw Swipe Points → SwipeFeaturizer → [B, T, 37] Features → Model → RNNT Loss
```

### 2.2 Model Forward Pass
1. **Encoder**: `[B, 37, T] → [B, D, T']` (where T' < T due to subsampling)
2. **Decoder**: Takes previous token + LSTM states → `[B, H, 1]` predictions
3. **Joint**: Combines encoder frame + decoder prediction → `[B, V, 1]` logits

### 2.3 RNNT Loss Computation
- Computes all valid alignment paths between input sequence and target
- Learns to emit characters or blanks at each time step
- Enables streaming inference (can decode partial sequences)

### 2.4 Training Metrics
- **Loss**: RNNT Loss (lower is better)
- **WER**: Word Error Rate on validation set
- **CER**: Character Error Rate on validation set

---

## 3. Model Export Pipeline

### 3.1 Encoder Export: `export_onnx.py`
**Location**: `trained_models/nema1/export_onnx.py`
**Last Modified**: 2025-09-16 05:20:36 -0400

**Process**:
1. Load trained `.nemo` or `.ckpt` model
2. Extract encoder (Conformer) component
3. Export to ONNX FP32 format
4. Quantize to INT8 using ONNX Runtime QDQ

**Input/Output**:
- **Input**: `features_bft [B, F=37, T]`, `lengths [B]`
- **Output**: `encoded_btf [B, D, T']`, `encoded_lengths [B]`
- **Files**: `encoder_fp32.onnx` → `encoder_int8_qdq.onnx`

### 3.2 Step Model Export: `export_rnnt_step.py`
**Location**: `trained_models/nema1/export_rnnt_step.py`
**Last Modified**: 2025-09-16 05:20:36 -0400

**Process**:
1. Extract decoder LSTM + joint network
2. Create single-step wrapper for beam search
3. Export to ONNX FP32 and ExecuTorch PTE

**Input/Output**:
- **Input**: `y_prev [B]`, `h0 [L, B, H]`, `c0 [L, B, H]`, `enc_t [B, D]`
- **Output**: `logits [B, V]`, `h1 [L, B, H]`, `c1 [L, B, H]`
- **Files**: `rnnt_step_fp32.onnx`, `rnnt_step_fp32.pte`

### 3.3 Runtime Metadata: `runtime_meta.json`
**Location**: `web-demo/runtime_meta.json`
**Last Modified**: 2025-09-16 09:10:32 -0400

**Contents**:
```json
{
    "tokens": ["<blank>", "'", "a", "b", ...],
    "blank_id": 0,
    "unk_id": 28,
    "char_to_id": {"a": 2, "b": 3, ...},
    "id_to_char": {"2": "a", "3": "b", ...},
    "vocab_size": 29,
    "feat_in": 37,
    "encoder": {"d_model": 256, "n_heads": 4, ...},
    "decoder": {"pred_hidden": 320, "pred_rnn_layers": 2},
    "joint": {"joint_hidden": 512},
    "key_order": "abcdefghijklmnopqrstuvwxyz'",
    "key_centers": {"a": [-0.8, 0.33], ...},
    "key_rbf_sigma": 0.2
}
```

---

## 4. Inference Architecture

### 4.1 Word List & Trie: `words.txt`
**Location**: `web-demo/words.txt`
**Last Modified**: 2025-09-16 05:20:36 -0400
**Size**: ~153,000+ English words

**Trie Structure**:
```javascript
class TrieNode {
    children: Map<number, TrieNode>  // char_id → child node
    isWord: boolean                  // Terminal word flag
    wid: number                     // Word index (-1 if not terminal)
}
```

### 4.2 Word Frequency Priors: `swipe_vocabulary.json`
**Format**: `{"word": frequency_score, ...}`
**Usage**: Bias beam search toward common words

### 4.3 Beam Search Inference
**Location**: `web-demo/swipe-onnx.html` (rnntWordBeam function)

**Process**:
1. **Encoder Forward**: `features [1, 37, T] → encoded [1, D, T']`
2. **Beam Search Loop**: For each encoder time step:
   - Expand active beams with blank transitions
   - Expand with vocabulary-constrained character transitions
   - Prune to top-K beams by score
3. **Scoring**: `score = rnnt_logp + λ * lm_prior`
   - `λ = 0.4` (language model weight)
   - `rnnt_logp`: Neural model log probability
   - `lm_prior`: Word frequency log prior

### 4.4 Feature Extraction (Browser)
**Process**: Raw touch points → 37D feature vectors
1. **Coordinate Normalization**: `[0,1] → [-1,1]` (canvas to model space)
2. **Kinematic Computation**: Velocity, acceleration from finite differences
3. **RBF Key Features**: Gaussian distance to each QWERTY key
4. **Tensor Format**: `[B=1, F=37, T]` for ONNX input

---

## 5. File Dependencies & Versions

### 5.1 Training Files (Archive: `web-demo/archive/`)
- `train_transducer.py` - Main training script
- `model_class.py` - RNNT model definition (if available)
- `swipe_data_utils.py` - SwipeFeaturizer implementation
- `export_onnx.py` - Encoder export script
- `export_rnnt_step.py` - Step model export script
- `runtime_meta.json` - Original metadata (29 tokens only)

### 5.2 Inference Files (web-demo/)
- `swipe-onnx.html` - Complete inference demo
- `runtime_meta.json` - Enhanced metadata (feat_in, key_centers, etc.)
- `words.txt` - English vocabulary list
- `swipe_vocabulary.json` - Word frequency priors
- `encoder_int8_qdq.onnx` - Quantized encoder model
- `rnnt_step_fp32.onnx` - Step model for beam search

---

## 6. Tensor Specifications

### 6.1 Data Types
- **Features**: `float32` - Input gesture features
- **Lengths**: `int32` - Sequence lengths
- **Tokens**: `int64` - Character IDs
- **Hidden States**: `float32` - LSTM h/c states
- **Logits**: `float32` - Output probabilities

### 6.2 Tensor Shapes
```python
# Encoder
features_bft: [Batch=1, Features=37, Time=T]        # Input gesture
lengths:      [Batch=1]                             # Sequence length
encoded_btf:  [Batch=1, D, Time'=T//2]             # Encoded features
enc_lengths:  [Batch=1]                             # Encoded length

# Step Model
y_prev:       [Batch=N]                             # Previous tokens
h0, c0:       [Layers=2, Batch=N, Hidden=320]       # LSTM states
enc_t:        [Batch=N, D]                          # Encoder frame
logits:       [Batch=N, Vocab=29]                   # Output probabilities
h1, c1:       [Layers=2, Batch=N, Hidden=320]       # Updated states
```

---

## 7. Performance Characteristics

### 7.1 Model Size
- **Encoder**: ~15MB (quantized INT8)
- **Step Model**: ~2MB (FP32)
- **Total**: ~17MB for complete system

### 7.2 Latency Targets
- **Excellent**: <500ms end-to-end
- **Good**: <1000ms end-to-end
- **Optimization Needed**: >1000ms

### 7.3 Accuracy Metrics
- **Domain Match**: Critical - features must exactly match training
- **Coordinate Space**: [-1, 1] normalization essential
- **Feature Structure**: 9 kinematic + 28 RBF keys required

---

## 8. Critical Implementation Notes

### 8.1 Coordinate Normalization
```javascript
// CORRECT: Training uses [-1, 1] space
const x = ((canvas_x / canvas_width) * 2) - 1;
const y = ((canvas_y / canvas_height) * 2) - 1;

// INCORRECT: [0, 1] causes massive domain shift
const x = canvas_x / canvas_width;  // WRONG!
```

### 8.2 Feature Extraction Order
**Must match training exactly**:
1. 9 kinematic features: `[x, y, t, vx, vy, speed, ax, ay, acc]`
2. 28 RBF features in key_order: `"abcdefghijklmnopqrstuvwxyz'"`

### 8.3 Tensor I/O Names
**Use runtime detection** - never hardcode:
```javascript
const inputNames = session.inputNames;
const featureInput = inputNames.find(name =>
    name.includes('feature') || name.includes('audio'));
```

### 8.4 Beam Search Parameters
- `beamSize: 16` - Number of active hypotheses
- `lmLambda: 0.4` - Language model weight
- `prunePerBeam: 8` - Vocabulary pruning per beam
- `maxSymbols: 15` - Maximum output length

---

## 9. Reproducibility Checklist

✅ **Training Script**: `train_transducer.py` archived
✅ **Feature Extraction**: `SwipeFeaturizer` implementation copied
✅ **Export Scripts**: Both encoder and step model exports archived
✅ **Model Config**: Complete hyperparameters documented
✅ **Tensor Specs**: All input/output shapes and types specified
✅ **Vocabulary**: Word list and frequency priors included
✅ **Runtime Metadata**: Enhanced metadata with all parameters
✅ **Version Control**: Last commit dates for all files recorded
✅ **Architecture Diagram**: See `architecture_diagram.js` for visual flow

This documentation ensures complete reproducibility of the CleverKeys RNNT gesture typing system from training to deployment.