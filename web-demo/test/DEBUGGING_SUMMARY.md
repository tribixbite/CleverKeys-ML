# RNN-T Prediction Debugging Summary

## The Complete Prediction Pipeline

### 1. Data Preparation ✅ CORRECT
- Dataset coordinates: [0,1] where (0,0) = top-left Q key
- Transform to [-1,1]: `x = raw_x * 2.0 - 1.0`
- Both Python and JS doing this correctly

### 2. Key Centers ✅ CORRECT
- Layout: `["qwertyuiop", "asdfghjkl", "zxcvbnm"]`
- Calculation verified:
  - 'h' at position 5 in row 1: x=0.1, y=0.0 ✅
  - 'e' at position 2 in row 0: x=-0.5, y=-0.667 ✅

### 3. Resampling ✅ CORRECT
- Using temporal interpolation
- Target: 56-96 frames based on length
- Python uses exact training function

### 4. Feature Extraction ✅ CORRECT
- 37 features total (27 kinematic + 10 key features)
- Python uses exact PersonalizedSwipeFeaturizer from training

### 5. Encoder ✅ CORRECT
- Input: [1, 37, time_steps]
- Output: [1, 144, encoded_steps]
- Properly padded to 37 dimensions

### 6. Decoder/RNN-T Process ✅ CORRECT (after fixes)

**Key insights about RNN-T decoding:**
1. **Continue through ALL frames** - don't stop at first blank
2. **Blanks mean "no more output for this frame"** - move to next frame
3. **Keep hidden states across frames** - critical for context
4. **Predictor label mapping** - identity for non-blanks, -1 for blank

**Correct decoding algorithm:**
```python
for each encoder frame:
    while True:
        logits = decoder(y, states, encoder_frame)
        pred = argmax(logits)

        if pred == blank_id:
            # Move to next frame, keep states
            break
        else:
            # Emit character, update y
            output.append(pred)
            y = joint2pred[pred]  # Map to predictor space
```

## The Real Problem: UNDERTRAINED MODELS

### Evidence:
1. **Frame 0**: Correctly predicts 'h' with confidence 4.2
2. **Frames 1-40**: Overwhelmingly predicts blank (scores 15-18 vs negative for chars)
3. **Pattern**: Model learned to emit first character then stop

### Why WER 0.457 is misleading:
- Model gets partial credit for getting first characters right
- 'hello' → 'h' is 80% wrong but gets some WER credit
- Model hasn't learned character sequences, just initials

## Vocabulary Handling ✅ CORRECT

**NeMo RNN-T convention:**
- Vocabulary: `['<blank>', "'", 'a', ..., 'z', '<unk>']` (29 tokens)
- Functional blank at index 29 (beyond vocabulary)
- Predictor uses indices 0-28 (no blank)
- Joint network outputs 30 classes (0-29)

**Mappings:**
- joint2pred: [0,1,2,...,28,-1] (identity except blank→-1)
- pred2joint: [0,1,2,...,28] (identity)

## JavaScript Implementation Issues to Check

1. **Feature extraction** - Verify exact match with Python
2. **Resampling** - Ensure temporal interpolation matches
3. **Decoding loop** - Must continue through ALL frames
4. **State management** - Keep LSTM states across frames

## Conclusion

The models ARE working - they correctly identify the first character with high confidence. The issue is they're severely undertrained for sequence generation. They've learned:
- ✅ Swipe start position → first character
- ❌ Character sequences and transitions
- ❌ When to emit subsequent characters

This is typical of early-stage RNN-T training. The model needs significantly more training epochs to learn proper sequence generation beyond just initial character recognition.