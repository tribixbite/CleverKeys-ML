# Neural Swipe Typing Web Demo

A production-ready browser demonstration of the **nema1** RNNT gesture typing model with real-time lexicon-constrained beam search.

## 🚀 Live Demo

The demo is automatically deployed to GitHub Pages on every commit: **[Live Demo](https://username.github.io/cleverkeys/swipe-onnx.html)**

## ⚡ Performance

- **Sub-1s latency** for 6-10 letter words on modern devices
- **25MB encoder** (INT8 quantized) + **8MB step model** (FP32)
- **150k vocabulary** with lexicon-constrained beam search
- **Real-time inference** with WebGPU acceleration (fallback to WASM)

## 🏗️ Architecture

### Complete RNNT Pipeline
1. **Feature Extraction**: 37D comprehensive features (position, velocity, acceleration, curvature, temporal)
2. **Encoder**: Ultra-optimized `encoder_web_ultra_web_final.onnx` (256D conformer)
3. **Step Model**: `rnnt_step_fp32.onnx` (LSTM decoder + joint network)
4. **Beam Search**: Lexicon-constrained with 150k word vocabulary trie
5. **UI Integration**: Gorgeous neon keyboard with smooth swipe trails

### Files Overview
- `swipe-onnx.html` - **Main demo entry point**
- `encoder_web_ultra_web_final.onnx` - Quantized encoder model (25MB)
- `rnnt_step_fp32.onnx` - Step model for beam search (8MB)
- `runtime_meta.json` - Character vocabulary and mappings (29 tokens)
- `words.txt` - Complete vocabulary wordlist (150k words)
- `swipe_vocabulary.json` - Legacy vocabulary with frequency data
- Supporting JS modules for dictionary management and UI

## 🔥 Key Features

### Real Neural Predictions
- **No mocks or fallbacks** - 100% production RNNT beam search
- **Lexicon constraints** using vocabulary trie for valid words only
- **Character-level decoding** with proper blank token handling
- **Multi-beam exploration** with pruning for efficiency

### Ultra-Fast Performance
- **Optimized beam search** (8 beams, 4 prune-per-beam, 15 max symbols)
- **Comprehensive feature extraction** with kinematic analysis
- **WebGPU acceleration** with WASM fallback
- **Performance monitoring** with real-time latency tracking

### Gorgeous UI
- **Neon keyboard styling** with hover effects and animations
- **Multi-layer swipe trails** with gradient coloring and glow effects
- **Real-time gesture visualization** with smooth path rendering
- **Responsive design** for desktop and mobile

## 🧪 Testing

### Manual Testing
1. Start local server: `python3 -m http.server 8080`
2. Open `http://localhost:8080/swipe-onnx.html`
3. Swipe across letters to spell words (e.g., "hello", "world", "testing")
4. Check browser console for latency metrics

### Expected Performance
- **6-10 letter words**: <500ms typical, <1s target
- **Console logging**: Shows inference timing and beam search results
- **Prediction quality**: Real words from 150k vocabulary

## 🔧 Development

### Model Files
- Models are tracked with Git LFS due to size
- `encoder_web_ultra_web_final.onnx` - 62% size reduction via INT8 quantization
- `rnnt_step_fp32.onnx` - Exported from `9_15_val_09.ckpt` checkpoint

### Feature Engineering
The demo extracts 37D features per time step:
- Position (2D): Normalized x, y coordinates
- Velocity (2D): First derivatives with proper time scaling
- Acceleration (2D): Second derivatives
- Key features (6D): Nearest key position and distance
- Temporal (2D): Time since start, start marker
- Direction (4D): Stroke angle and curvature (sin/cos)
- Context (15D): Progress, speed, acceleration magnitude, positional encoding

### Optimization Notes
- Beam size reduced to 8 for speed (from 16)
- Pruning limited to 4 candidates per beam
- Maximum symbol length capped at 15
- WebGPU preferred over CPU/WASM for inference

## 🚀 Deployment

Automatic deployment via GitHub Actions:

1. **Push to main branch** triggers deployment
2. **Files copied** from `web-demo/` to root
3. **GitHub Pages** serves the complete demo
4. **Verification** ensures all models and assets are present

### Manual Deployment
```bash
# Copy web-demo contents to a web server
cp -r web-demo/* /var/www/html/

# Ensure proper MIME types for .onnx files
# Add to your web server config:
# .onnx -> application/octet-stream
```

## 📊 Technical Specifications

- **Model**: nema1 (9_15_val_09.ckpt)
- **Architecture**: RNNT (Conformer encoder + LSTM decoder)
- **Vocabulary**: 29 character tokens + 150k wordlist
- **Quantization**: INT8 QDQ format (Dynamic quantization)
- **Inference**: ONNX Runtime Web with WebGPU/WASM
- **Framework**: Vanilla JavaScript (no dependencies)

---

*Generated from nema1 checkpoint - Production-ready RNNT gesture typing system*