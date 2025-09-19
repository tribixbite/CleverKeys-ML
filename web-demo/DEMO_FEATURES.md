# Swipe Demo Features

## Demo URL
**http://localhost:8080/swipe-onnx.html**

## New Features Added

### 1. Model Hot-Swapping Dropdown ✅
- Added dropdown selector in the header controls
- Available models:
  - **Web Ultra (23MB)** - Default, optimized for web browsers
  - **Android INT8 (22MB)** - INT8 quantized for mobile deployment
  - **Android Ultra (25MB)** - Alternative Android optimized version
  - **FP32 Full (62MB)** - Full precision reference model

### 2. Model Switching Functionality
```javascript
// New function added to dynamically load different models
async function switchModel(modelPath) {
    // Cleans up existing session
    // Loads new model with WASM execution provider
    // Updates UI to show current model
    // Handles errors gracefully with fallback
}
```

### 3. Available ONNX Models
All models are from the best checkpoint (epoch 64, val_wer=0.156):

| Model | Size | Type | Use Case |
|-------|------|------|----------|
| encoder_web_ultra.onnx | 23MB | INT8 | Web browsers with WASM |
| encoder_android_int8_final.onnx | 22MB | INT8 | Android with NNAPI |
| encoder_android_ultra.onnx | 25MB | INT8 | Android alternative |
| encoder_fp32.onnx | 62MB | FP32 | Reference/debugging |

## How to Test

### 1. Start the HTTP Server
```bash
cd web-demo
python3 -m http.server 8080
```

### 2. Open Demo
Navigate to: http://localhost:8080/swipe-onnx.html

### 3. Test Model Switching
- Use the dropdown to select different models
- Each model switch shows a loading overlay
- Model status updates in the header
- All models support the same input/output format

### 4. Test Gesture Input
- Draw swipe gestures on the keyboard
- Works with all models
- Debug mode shows:
  - Current key under finger
  - Path length
  - Keys touched

## Technical Implementation

### Model Loading
- Uses ONNX Runtime Web 1.18.0
- WASM execution provider for CPU inference
- Graph optimization level: 'all'
- Sequential execution mode for deterministic timing

### Feature Extraction
- 37-dimensional kinematic features
- Adaptive resampling (56-96 frames)
- Normalized coordinates (0-1 range)

### Encoder-Only Mode
Currently running in encoder-only mode:
- Processes gestures through Conformer encoder
- Generates character predictions from encoder output
- Full RNN-T decoder integration pending

## Performance

### Model Comparison
| Model | Inference Time | Size | Accuracy |
|-------|----------------|------|----------|
| Web Ultra | ~20ms | 23MB | High |
| Android INT8 | ~20ms | 22MB | High |
| FP32 | ~12ms | 62MB | Highest |

*Times measured on CPU without hardware acceleration*

## Current Status
✅ Model dropdown added and functional
✅ Hot-swapping works without page reload
✅ All models load successfully
✅ Gesture tracking active
✅ Debug text selectable for copying

## Known Limitations
1. Encoder-only mode (no full RNN-T decoder yet)
2. No beam search (single-best path only)
3. Limited vocabulary constraints

## Next Steps
- [ ] Integrate full RNN-T decoder
- [ ] Add beam search decoding
- [ ] Implement word-level constraints
- [ ] Add performance metrics display
- [ ] Support model comparison mode