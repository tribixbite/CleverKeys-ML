# Complete Prediction Pipeline Analysis

## Step-by-Step Process and Potential Issues

### 1. INPUT: Raw swipe data from dataset
- **Format**: `{"x": 0.784, "y": 0.214, "t": 0}` where x,y ∈ [0,1]
- **Coordinate system**: (0,0) is top-left of Q key, (1,0) is top-right P
- ✅ **Verified**: Dataset format confirmed

### 2. COORDINATE TRANSFORMATION
**Training (train_transducer_personalized.py lines 609-613)**:
```python
centered_x = raw_x * 2.0 - 1.0  # [0,1] → [-1,1]
centered_y = raw_y * 2.0 - 1.0
centered_x = clamp(centered_x, -1.5, 1.5)
centered_y = clamp(centered_y, -1.5, 1.5)
```
**Python test**: ✅ Doing this correctly
**JS test**: ❓ Need to verify

### 3. RESAMPLING
**Training (resample_points function)**:
- Uses temporal interpolation based on timestamps
- Target length: 56-96 frames based on thresholds
- Linear interpolation between points
**Python test**: ✅ Using exact same function
**JS test**: ❓ Need to verify implementation matches

### 4. FEATURE EXTRACTION (PersonalizedSwipeFeaturizer)
**Training features (27 base + 10 key features = 37 total)**:
- Positions: x, y
- Velocities: dx/dt, dy/dt
- Accelerations: d²x/dt², d²y/dt²
- Path features: speed, angle, angle_change, etc.
- Key features: nearest 5 keys with distances
**Python test**: ✅ Using exact same PersonalizedSwipeFeaturizer
**JS test**: ❓ Need to verify feature calculation

### 5. KEY CENTERS
**Training**:
```python
layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
x01 = (col_idx + 0.5) / 10.0  # e.g., 'h' at col 6: (6.5)/10 = 0.65
y01 = (row_idx + 0.5) / 3.0   # row 1: (1.5)/3 = 0.5
x = x01 * 2.0 - 1.0  # 0.65*2-1 = 0.3 ❌ WRONG!
y = y01 * 2.0 - 1.0  # 0.5*2-1 = 0.0
```
**Actual 'h' position**: x=0.1, y=0.0
**Issue**: The calculation is wrong for 'h'!

Let me recalculate...

### 6. ENCODER
- Input: [1, 37, time_steps] tensor
- Output: [1, 144, encoded_steps]
- Encoded length typically ~half of input

### 7. DECODER (RNN-T)
**Training vocabulary**:
- 29 tokens: `['<blank>', "'", 'a', ..., 'z', '<unk>']`
- Functional blank at index 29 (not in vocab)
- `blank_as_pad: True` setting

**Decoder process**:
1. Start with y=0 (BOS)
2. For each encoder frame:
   - Run decoder up to 6 times
   - If pred_idx == 29 (blank), move to next frame
   - Otherwise, emit character and continue
3. Predictor uses "blankless" labels (0-28)
4. Joint network outputs 30 logits (0-29)

### 8. POTENTIAL ISSUES IDENTIFIED

#### Issue 1: Blank Token Handling
- **Training**: Uses functional blank at index 29
- **Inference**: We're checking `if pred_idx == blank_id` correctly
- ✅ This appears correct

#### Issue 2: Predictor Label Mapping
- **Training**: Predictor gets blankless labels
- **Our code**:
  ```python
  if pred_idx < blank_id:
      next_y = pred_idx
  else:
      next_y = pred_idx - 1
  ```
- ❌ **WRONG!** We should map from joint space to predictor space using the label map!

#### Issue 3: BOS Token
- **Training**: Likely uses index 0 as BOS in predictor space
- **Our code**: Starting with y=0
- ✅ This appears correct

#### Issue 4: Feature Normalization
- **Training**: Features might be normalized
- **Our code**: No normalization after feature extraction
- ❓ Need to check if training normalizes

#### Issue 5: Key Center Calculation
Let me verify the exact calculation...

```python
# For 'h' (row 1, col 6 in "asdfghjkl")
col_idx = 6
row_idx = 1
x01 = (6 + 0.5) / 10.0 = 0.65
y01 = (1 + 0.5) / 3.0 = 0.5
x = 0.65 * 2.0 - 1.0 = 0.3
y = 0.5 * 2.0 - 1.0 = 0.0
```
But we see x=0.1 in key-centers.json!

Wait, let me recount...
"asdfghjkl" - 'h' is at position 5 (0-indexed), not 6!
```python
col_idx = 5
x01 = (5 + 0.5) / 10.0 = 0.55
x = 0.55 * 2.0 - 1.0 = 0.1 ✅
```

### 9. THE MAIN ISSUE: Predictor Label Mapping

The critical bug is in how we handle the predictor input after getting a non-blank prediction:

**Current (WRONG)**:
```python
if pred_idx < blank_id:
    next_y = pred_idx
else:
    next_y = pred_idx - 1
```

**Should be**:
```python
# Use the joint2pred mapping from runtime_meta.json
joint2pred = meta['predictor']['label_map']['joint2pred']
next_y = joint2pred[pred_idx]  # Maps joint vocab to predictor vocab
```

This is likely why predictions are wrong - we're feeding incorrect indices to the predictor!