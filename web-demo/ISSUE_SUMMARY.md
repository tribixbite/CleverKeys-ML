# Web Demo Prediction Pipeline Issues - Summary

## Issues Found and Fixed

### 1. ✅ Coordinate Transformation Bug
**Problem**: Dataset uses [0,1] coordinates, but feature extractor expected [-1,1]
**Solution**: Added transformation `x * 2.0 - 1.0` in JavaScript feature extractor
**Files Fixed**:
- `web-demo/js/feature-extractor-corrected.js`
- `web-demo/test/test-beam-best.js`

### 2. ✅ Vocabulary Size Mismatch
**Problem**: `runtime_meta.json` had vocab_size=29 but model outputs 30 logits
**Solution**: Updated to use `best_latest/runtime_meta.json` with vocab_size=30
**Files**: `web-demo/models/best_latest/runtime_meta.json`

### 3. ✅ Blank Token Representation Issue
**Problem**: Blank token (index 29) mapped to empty string "", causing beam search failures
**Solution**: Created `runtime_meta_fixed.json` with proper blank token representation
**Files**: `web-demo/models/best_latest/runtime_meta_fixed.json`

## Critical Issues Still Present

### 4. ⚠️ Adaptive Resampling Not Implemented
**Problem**: Training uses adaptive resampling (91 points → 82 frames), but JS uses fixed 96 frames
**Impact**: Feature mismatch between training and inference
**Required Fix**: Implement adaptive resampling in JavaScript:
```javascript
function determineResampleTarget(length) {
    if (length < 20) return length;

    const shortTarget = 56;
    const longTarget = 96;
    const shortThresh = 48;
    const longThresh = 112;

    if (length <= shortThresh) return shortTarget;
    if (length >= longThresh) return longTarget;

    // Linear interpolation
    const frac = (length - shortThresh) / (longThresh - shortThresh);
    return Math.floor(shortTarget + frac * (longTarget - shortTarget));
}
```

### 5. ⚠️ Model Predictions Still Wrong
**Current State**: Even with correct features and coordinate transformation:
- Python greedy decode: "hello" → "am" (wrong)
- JavaScript beam search: "hello" → "sentient" (wrong)
- Expected: "hello" → "hello"

**Possible Causes**:
1. Model is undertrained (ONNX models from Oct 2, 09:36 AM)
2. Model was exported from wrong checkpoint
3. Additional preprocessing steps missing

## Next Steps

1. **Verify Model Training**: Check which checkpoint was exported to ONNX
2. **Implement Adaptive Resampling**: Add to JavaScript pipeline
3. **Test with Better Model**: Export from a well-trained checkpoint
4. **Verify Feature Pipeline**: Ensure complete match between training and inference

## Test Results

### After Coordinate Fix
```
Features match: ✅
First frame: [0.165550, 0.049296, 0.000000, 0.000000, 0.000000]
```

### With Proper Resampling (Python)
```
Input: "hello" (91 points → 82 frames)
Output: "am" ❌
```

### Without Resampling (JavaScript)
```
Input: "hello" (91 points → 96 frames)
Output: "sentient" ❌
```

## Files to Update

1. `web-demo/js/feature-extractor-corrected.js` - Add adaptive resampling
2. `web-demo/swipe-onnx-modular.html` - Use adaptive resampling
3. `web-demo/models/best_latest/` - Export from better checkpoint

## Commands to Re-export Model

```bash
# Find best checkpoint
ls -la rnnt_checkpoints_*/conformer_rnnt_final.nemo

# Export to ONNX
python new/export_stateful_pair.py \
    --nemo-path [BEST_CHECKPOINT].nemo \
    --out-dir web-demo/models/best_latest
```