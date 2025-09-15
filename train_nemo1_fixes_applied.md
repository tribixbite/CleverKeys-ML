# Fixes Applied to train_nemo1.py

## ✅ All Critical Errors Fixed

### 1. **Feature Dimension (Line 47)**
- **Fixed**: Changed from 36 to 37 features
- **Reason**: 9 kinematic features + 28 keys (a-z + apostrophe) = 37

### 2. **Type Hint Error (Line 83)**
- **Fixed**: Changed `List]` to `List`
- **Reason**: Syntax error in type hint

### 3. **List Append Bug (Line 85)**
- **Fixed**: Changed `points.append(points)` to `points = points + [points[-1]]`
- **Reason**: Original would append entire list to itself, causing infinite recursion

### 4. **Missing Initialization (Line 117)**
- **Fixed**: Changed `self.data =` to `self.data = []`
- **Reason**: Incomplete statement causing syntax error

### 5. **Shape Tensor Bug (Line 142)**
- **Fixed**: Changed `torch.tensor(features_tensor.shape)` to `torch.tensor(features_tensor.shape[0])`
- **Reason**: Should pass length as scalar, not shape tuple

### 6. **Missing Loss Configuration (Lines 281-284)**
- **Added**: RNNTLoss configuration to model config
- **Reason**: Required for RNN-T training

### 7. **Tokenizer/Vocab in Logger (Lines 165-170, 196, 300)**
- **Fixed**: Added vocab parameter to PredictionLogger
- **Added**: idx_to_char mapping for decoding
- **Fixed**: References to use self.idx_to_char instead of non-existent tokenizer

### 8. **Missing Main Call (Line 319)**
- **Fixed**: Ensured single `main()` call in `if __name__ == '__main__':`
- **Reason**: File was incomplete

## ✨ Current Status

All syntax errors have been fixed and the file now:
- ✅ Parses without syntax errors
- ✅ Has correct feature dimensions
- ✅ Properly handles data structures
- ✅ Includes loss configuration
- ✅ Has working tokenizer/vocab references

## ⚠️ Remaining Considerations

While the syntax is fixed, for production use you still need:

1. **Install NeMo**: 
   ```bash
   uv add nemo-toolkit pytorch-lightning omegaconf
   ```

2. **Consider using optimized train.py instead**: 
   - Already has RTX 4090M optimizations
   - No dependency issues
   - Proven to work with your data

3. **Potential Runtime Issues**:
   - Direct DataLoader assignment (lines 296-297) may not work properly with NeMo
   - Model class might need to be EncDecRNNTBPEModel instead of EncDecRNNTModel
   - Decoding method calls may need adjustment based on NeMo version

## 🎯 Recommendation

While the fixes make train_nemo1.py syntactically correct, **train.py remains the better choice** because:
- It's already optimized for your RTX 4090M
- Has all dependencies installed
- Includes comprehensive performance optimizations
- Is tested and working

Use the fixed train_nemo1.py only if you specifically need NeMo's features.
