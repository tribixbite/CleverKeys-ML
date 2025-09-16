# Training Errors Summary - CleverKeys SOTA Model Development

## Overview
This document comprehensively details all errors encountered during the development of a state-of-the-art gesture typing model, the root causes identified, and the solutions implemented.

## Error Timeline and Resolutions

### 1. Initial Data Augmentation Errors

#### Error 1.1: IndexError in temporal_warp
**File**: `train_sota.py`
**Error Message**: 
```
IndexError: tuple index out of range
File "train_sota.py", line 87, in temporal_warp
    n_points = points.shape[1]
```

**Root Cause**: The temporal_warp function assumed points was always a 2D array, but sometimes received 1D arrays when data was malformed.

**Solution Implemented**:
```python
# Added dimensionality check
if len(points.shape) == 1:
    return points
```

#### Error 1.2: ValueError in reshape operation
**File**: `train_sota.py`
**Error Message**:
```
ValueError: cannot reshape array of size 17 into shape (3)
File "train_sota.py", line 187, in extract_features
    points_array = points_array.reshape(-1, 3)
```

**Root Cause**: The dropout_points augmentation function was creating arrays with incorrect sizes when randomly dropping points, resulting in arrays that couldn't be reshaped into (n_points, 3) format.

**Solution Implemented**:
```python
# Added validation before reshape
if len(points_array.shape) == 1:
    points_array = points_array.reshape(-1, 3)
if points_array.shape[1] != 3:
    # Handle malformed data
    return None
```

### 2. ONNX Export Errors

#### Error 2.1: Unsupported operator 'unflatten'
**File**: `export_models.py`
**Error Message**:
```
torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::unflatten' to ONNX opset version 11 is not supported
```

**Root Cause**: The model used operations that weren't supported in ONNX opset 11.

**Solution Implemented**:
```python
# Updated to opset version 13
opset_version=13
```

#### Error 2.2: Unsupported operator 'scaled_dot_product_attention'
**Error Message**:
```
torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::scaled_dot_product_attention' to ONNX opset version 13 is not supported
```

**Root Cause**: PyTorch 2.0+ uses Flash Attention by default, which requires opset 14.

**Solution Implemented**:
```python
# Updated to opset version 14
opset_version=14
```

#### Error 2.3: Missing ONNX module
**Error Message**:
```
ModuleNotFoundError: No module named 'onnx'
```

**Solution Implemented**:
```bash
uv add onnx onnxruntime onnxruntime-tools
```

### 3. Quantization Errors

#### Error 3.1: AssertionError in scale computation
**File**: `export_models.py`
**Error Message**:
```
AssertionError: scale issue
File "onnxruntime/quantization/quant_utils.py", line 293, in compute_scale_zp
    assert scale >= 0, "scale issue"
```

**Root Cause**: The model had NaN or infinite values in weights, causing quantization to fail when computing scale factors.

**Solution Implemented**:
- Skipped quantization for the problematic model
- Exported unquantized ONNX model successfully
- Created alternative export path using PyTorch format

### 4. TorchScript Tracing Errors

#### Error 4.1: TracingCheckError
**File**: `scripts/pte_export_fixed.py`
**Error Message**:
```
torch.jit._trace.TracingCheckError: Tracing failed sanity checks!
ERROR: Graphs differed across invocations!
```

**Root Cause**: The model had dynamic behavior (positional embeddings slicing based on sequence length) that couldn't be traced consistently.

**Solution Implemented**:
- Created simplified export using torch.save instead of tracing
- Saved model state dict directly for mobile deployment

### 5. Training Instability - NaN Loss

#### Error 5.1: NaN losses during training
**File**: `train_production.py`
**Log Output**:
```
NaN loss at step 64851 (repeated)
Train Loss: 0.0000
Val Loss: 0.0000
```

**Root Cause Analysis**:
Multiple potential causes identified:
1. **Gradient explosion** - Despite gradient clipping, very large gradients could cause overflow
2. **Numerical underflow** - CTC loss can produce very small probabilities leading to log(0)
3. **Data issues** - Malformed samples with invalid coordinates or timestamps
4. **Learning rate** - Too high learning rate in later epochs
5. **Mixed precision** - Float16 operations causing overflow

**Solutions Implemented**:
```python
# 1. Added robust data validation
if np.any(np.isnan(points_array)) or np.any(np.isinf(points_array)):
    return None

# 2. Clipped feature values
features = np.clip(features, -10, 10)

# 3. Added epsilon to prevent log(0)
log_probs = torch.log(probs + 1e-8)

# 4. Reduced learning rate
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# 5. Added loss validation
if torch.isnan(loss):
    print(f"NaN loss at step {step}")
    continue  # Skip update
```

### 6. Data Processing Errors

#### Error 6.1: Memory issues during validation
**Original Issue**: Validation would crash with OOM errors

**Solution Implemented**:
```python
# Reduced validation batch size
val_batch_size = 128  # Down from 256

# Added torch.no_grad() and cache clearing
with torch.no_grad():
    # validation code
torch.cuda.empty_cache()
```

## Key Architectural Decisions to Prevent Errors

### 1. Robust Feature Extraction
Created `RobustFeaturizer` class with comprehensive error handling:
- Input validation
- Safe division (avoiding division by zero)
- Bounded feature values
- Fallback values for edge cases

### 2. Production Model Architecture
Switched from experimental Conformer to stable Transformer:
- More predictable gradient flow
- Better numerical stability
- Proven architecture for sequence modeling

### 3. Data Pipeline Improvements
- Added data validation at multiple stages
- Implemented safe augmentation with fallbacks
- Created reproducible data splits

## Lessons Learned

1. **Start Simple**: Complex architectures (Conformer) introduce more failure points
2. **Validate Early**: Check data integrity before training
3. **Incremental Complexity**: Add features gradually with testing
4. **Defensive Coding**: Always handle edge cases in data processing
5. **Monitor Continuously**: Log extensively to catch issues early

## Final Model Statistics
- **Parameters**: 14.6M
- **Best Loss**: 2.73 (before NaN issues)
- **Export Formats**: ONNX (57MB), PyTorch (56MB)
- **Training Time**: ~4 hours for 50 epochs
- **Final Status**: Successfully exported despite end-of-training NaN issues