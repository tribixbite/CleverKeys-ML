# CleverKeys RNN-T Implementation Guide

Last Updated: 2025-10-02
Status: Active Development (resumable multi-profile training online)

## Overview

This document contains implementation details for the CleverKeys RNN-T swipe gesture model, including architecture analysis, training insights, and deployment guidance. It documents both confirmed findings and areas of uncertainty discovered during development.

## Architecture Summary

### Core Model: Conformer-RNNT
- Presets via `--model-size {mobile,tablet,server}`
  - Mobile (default): 4 layers, d_model=144, 4 heads, joint=256
  - Tablet: 5 layers, d_model=192, 4 heads, joint=384
  - Server: 6 layers, d_model=256, 8 heads, joint=512
- Decoder: LSTM prednet (`pred_rnn_layers` per preset), joint hidden per preset
- Vocabulary: provided by vocab file (blank-as-pad); use runtime_meta for IDs
- Features: 37-D swipe features (kinematics + spatial + temporal)

### Training Configuration (Current)
- Script: `new/train_transducer_personalized.py`
- Batch size: hardware dependent (auto via runners), overrideable
- LR: 2e-4 with CosineAnnealing (warmup + computed max_steps)
- Precision: bf16-mixed (Ampere+), compile/cudagraphs disabled by default in runners
- Sampling: profile-driven weighted sampling; profile aliases supported for orchestration

## Confirmed Findings

### ✅ What We Know Works

1. **Vocabulary System**:
   - `blank_as_pad=True` (blank treated as pad); derive IDs from runtime metadata
   - Export preserves all token IDs; do not hardcode blank or unk IDs in decoders

2. **Feature Engineering**:
   - 37D features (x/y/kinematics/spatial windows) and adaptive resampling (≈56–96 frames)
   - Dataset coordinates are normalized to keyboard size in [0,1], with slight out‑of‑bounds permitted.
   - JS featurizer mirrors Python exactly; do not clamp.

3. **Training Process**:
   - Resumable both per-profile (comprehensive runner) and per-strategy (curriculum runner)
   - WER embedded in checkpoint filenames; per-profile WER tracked to CSV
   - Default config stable; batch/num_workers should match hardware

4. **ONNX Export**:
   - Successfully exports encoder, decoder, and joint networks
   - Stateful RNN-T architecture maintains decoder state between frames
   - Web inference feasible with onnxruntime-web

### ⚠️ Areas of Concern

1. **Sampling/Validation Balance**:
   - Profiles can bias WER by focusing on specific slices (rare-vs-common)
   - Compare WER across apples-to-apples subsets; rely on CSV metrics per profile

2. **Augmentation Defaults**:
   - Augmentation is optional/off by default; enable with `--augment` when needed

3. **Profile Selection**:
   - Use `--profile/--val-profile` or orchestration runners; do not hardcode sampling in code

## Uncertainties & Open Questions

### 🤔 Things We're Not Sure About

1. **ONNX Export Naming**:
   - Encoder/decoder/joint may be separate ONNX files; verify filenames before web wiring

2. **Decoder State Management**:
   - Exact tensor shapes for LSTM states in ONNX unclear
   - Initial state initialization strategy not verified
   - State propagation between beam search hypotheses needs testing

3. **Web Performance**:
   - Unknown if 37D features × 96 frames runs smoothly in browser
   - WASM vs WebGL backend performance not benchmarked
   - Mobile browser compatibility untested

4. **Training Optimizations**:
   - `torch.compile()` can be enabled later; currently disabled by runners for NeMo stability
   - Consider profiling dataloading and GPU utilization after stability

## Next Steps for Implementation

### 🎯 Immediate Priorities

1. **Tune Sampling/Validation**:
   - Adjust per-profile validation settings when comparing WER across profiles

2. **Use CLI + Runners**:
   - `--augment`, `--profile`, `--val-profile`, `--checkpoint`, and dataset path overrides are available
   - Prefer `train_comprehensive.sh` / `run_comprehensive_training.sh` for long jobs + resumption

3. **Enable Augmentation for Rare Words**:
   - `uv run python new/train_transducer_personalized.py --augment --profile rare_words`

### 🚀 Performance Improvements

1. **Enable compile later**:
   - Remove env guards from runner and selectively compile encoder/joint if stable

2. **Profile GPU Utilization**:
```python
# Add to trainer:
profiler = "simple"  # Shows where time is spent
trainer = pl.Trainer(..., profiler=profiler)
```

### 🌐 Web Deployment Path

1. **Verify ONNX Export Structure**:
```bash
# After export, check what files exist:
ls onnx_models/
# Should see: encoder.onnx, decoder.onnx, joint.onnx (or decoder_joint.onnx)
```

2. **Test Stateful Decoding**:
   - Implement JavaScript preprocessor (port `PersonalizedSwipeFeaturizer`)
   - Handle decoder state initialization and propagation
   - Implement greedy decoding first, then beam search

3. **Optimize for Browser**:
   - Consider quantization (int8) for smaller model size
   - Test WebGL backend for GPU acceleration
   - Implement progressive model loading

## Critical Implementation Notes

### ⚠️ Must Remember

1. **Blank Token Handling**: Use runtime metadata; don’t hardcode indices (blank is typically last)
2. **Feature Order**: JavaScript features MUST match Python FEATURE_NAMES order exactly (parity tests present)
3. **Coordinate System**: Use dataset’s [0,1] coordinate frame (slight OOB allowed). Key centers defined in [0,1].
4. **Checkpoint Resume**: Use `.ckpt` for resumption; `.nemo` for export only

### 🐛 Known Issues

1. **NeMo warnings**: Cosine scheduler messages in FAST_DEV_RUN are expected
2. **CUDA Graphs**: Disabled by default for stability; can enable with `cuda-python`
3. **Export Scripts**: Use `trained_models/nema1/export_stateful_pair.py` for web (pair ONNX + runtime_meta)

## ONNX Export + Web Use (definitive)

### Export (stateful pair)
```bash
python trained_models/nema1/export_stateful_pair.py \\
  --checkpoint rnnt_checkpoints_<profile>_<date>/conformer_rnnt_final.nemo \\
  --outdir web-demo/models/rnnt_new_latest

# Outputs:
#  - encoder.onnx  (inputs: audio_signal[B,F,T], length[B]; outputs: outputs[B,256,T], encoded_lengths[B])
#  - decoder_joint.onnx (inputs: encoder_outputs[B,256,1], targets[B,1], target_length[B], input_states_1[2,B,320], input_states_2[2,B,320]; outputs: outputs[B,V], prednet_lengths, output_states_1, output_states_2)
#  - runtime_meta.json (tokens, blank_id, char_to_id, id_to_char)
```

### Web decoding quick start
```js
// 1) Load sessions
const enc = await ort.InferenceSession.create('models/rnnt_new_latest/encoder.onnx');
const dec = await ort.InferenceSession.create('models/rnnt_new_latest/decoder_joint.onnx');
const meta = await (await fetch('models/rnnt_new_latest/runtime_meta.json')).json();

// 2) Featurize (features: Float32Array[T*37]) and transpose to [B,F,T]
const T = numFrames, F = 37;
const bft = new Float32Array(F*T);
for (let t=0;t<T;t++) for (let f=0;f<F;f++) bft[f*T+t] = features[t*F+f];

// 3) Run encoder
const encOut = await enc.run({
  audio_signal: new ort.Tensor('float32', bft, [1,F,T]),
  length: new ort.Tensor('int64', BigInt64Array.from([BigInt(T)]), [1])
});
const encoded = encOut.outputs; // dims [1,256,T'] (or [1,T',256])
const Tprime = Number(encOut.encoded_lengths.data[0]);

// 4) RNNT greedy/beam inner loop
let h = new ort.Tensor('float32', new Float32Array(2*1*320), [2,1,320]);
let c = new ort.Tensor('float32', new Float32Array(2*1*320), [2,1,320]);
let last = meta.blank_id; // typical start
for (let t=0;t<Tprime;t++){
  // slice enc frame to [1,256,1]
  const frame = new Float32Array(256);
  if (encoded.dims[1]===256){ for (let i=0;i<256;i++) frame[i]=encoded.data[i*Tprime+t]; }
  else { const s=t*256; for (let i=0;i<256;i++) frame[i]=encoded.data[s+i]; }
  const out = await dec.run({
    encoder_outputs: new ort.Tensor('float32', frame, [1,256,1]),
    targets: new ort.Tensor('int32', Int32Array.from([last]), [1,1]),
    target_length: new ort.Tensor('int32', Int32Array.from([1]), [1]),
    input_states_1: h, input_states_2: c
  });
  const logits = out.outputs.data; h = out.output_states_1; c = out.output_states_2;
  // argmax non-blank or beam expansion...
}
```

## Architecture Decisions & Rationale

### Why These Choices Work

1. **Conformer > LSTM**: Better at capturing both local and global patterns in swipes
2. **RNN-T > CTC**: Models output dependencies, 40-50% WER reduction
3. **Adaptive Resampling**: Handles varying swipe speeds naturally
4. **Weighted Sampling**: Critical for rare word performance

### Trade-offs Made

1. **Batch Size 1000**: Maximizes GPU usage but might cause OOM on smaller GPUs
2. **bf16 Precision**: Faster training but slightly less accurate than fp32
3. **37D Features**: Rich representation but increases inference compute

## For the Next Developer

### Quick Start
```bash
# Full curriculum (resumable, date-based run base)
./train_comprehensive.sh curriculum

# Multi-profile cycles (metrics CSV, resumable)
./run_comprehensive_training.sh

# One-off trainer with overrides (fresh base)
CKS_RUN_BASE=./9292025script/20251002 uv run python new/train_transducer_personalized.py \
  --profile sqrt_balanced --val-profile validation_balanced \
  --batch-size 320 --num-workers 8 --max-epochs 100

# TensorBoard
uv run tensorboard --logdir 9292025script
```

### Key Files to Understand
1. `new/train_transducer_personalized.py` - Main training logic (CLI + scheduler)
2. `new/sampling_profiles.py` - Sampling strategies + aliases for runners
3. `new/data_augmentation.py` - Optional augmentation pipeline
4. `new/export_onnx_stateful.py` / `new/export_advanced.py` - ONNX export
5. `train_comprehensive.sh` / `run_comprehensive_training.sh` - Orchestration (resumable)

### What Needs Work
1. Per-profile validation balance and cross-profile WER comparison methodology
2. Optional compile/cudagraph enablement and guard-rails
3. Web featurizer parity tests + benchmark across backends (WASM/WebGL)
4. Documentation stays current with pipeline changes

## Beam Search Decoder (Web)

This demo now includes a full lexicon‑constrained RNNT beam search in JavaScript:

- Loads `web-demo/words.txt` and `web-demo/word_frequencies_aligned.json` (log frequencies aligned with words.txt order) and builds a trie of valid words.
- Filters the word list client‑side to remove unsuitable entries for gesture prediction:
  - Allowed chars: `[a-z']`, length 2–20.
  - No triple repeated characters (regex `(.)\1\1`).
  - Minimum frequency per length (e.g., 2‑char ≥1e‑5, 3‑char ≥1e‑6, 4‑char ≥1e‑7, 5‑char ≥5e‑8, 6–7‑char ≥1e‑8, 8‑char ≥5e‑9, ≥9‑char ≥1e‑9).
- Uses RNNT inner‑loop decoding (multiple symbols per encoder frame until blank) and constrains expansions to trie children.
- Adds a log‑prior bonus for completed words to favor lexicon endpoints.

The modular page (`swipe-onnx-modular.html`) now calls `rnntDecoder.loadLexicon(...)` at init and prefers `beamSearch()` over greedy when a lexicon is available.

## Wordlist Generation and Filtering

For high‑quality suggestions, regenerate the word list with stricter filtering:

1. Update `vocab/wordlist_gen/gen_words.py` (already patched):
   - Accept only `[a-z']{2,20}`.
   - Reject words with 3+ repeated chars.
   - Apply length‑dependent frequency thresholds (2‑char ≥1e‑5, 3‑char ≥1e‑6, 4‑char ≥1e‑7, 5‑char ≥5e‑8, 6–7‑char ≥1e‑8, 8‑char ≥5e‑9, ≥9‑char ≥1e‑9).
   - If not in NLTK `words` corpus and `freq < 1e‑8`, drop.

2. Generate and combine:
```bash
cd vocab
uv run --with wordfreq --with nltk wordlist_gen/gen_words.py
python combine_wordlists.py
```

3. Copy `vocab/wordlist_gen/combined_wordlist.txt` into `web-demo/words.txt` (or adjust the URL in `loadLexicon`).

This substantially reduces low‑value entries (e.g., `aaaaaa`, `khuzestan`, `khwaja`) and improves beam search quality for gesture input.

## RNNT Implementation Notes

- Use `meta.blank_id` (blank at end, typically 29). Initial predictor token can be blank; some exports accept `0` for the very first step.
- Encoder input: `[B, F, T]` with `F=37`. Encoder output may be `[B, 256, T]` or `[B, T, 256]`.
- Slice encoder frames to `[1, 256, 1]` and run decoder‑joint per hypothesis.
- RNNT inner loop per frame is essential; continue emitting symbols until a blank is predicted.

### Questions to Investigate
1. Does torch.compile() actually speed up Conformer models?
2. What's the real WER when including all word lengths?
3. Can we reduce features from 37D without losing accuracy?
4. How much does augmentation help on rare words?
5. Is beam search necessary or does greedy work well enough?

---

*This document reflects the current understanding as of January 2025. Update as new findings emerge.*

-----

## Part 2: Drastically Speeding Up Training

Your training configuration is already well-optimized for a 4090M (using `bf16`, TF32, and good `DataLoader` settings). The following will provide the most significant speed boosts.

### 🚀 Use `torch.compile` (Easiest & Most Impactful)

This is the number one recommendation. `torch.compile()` is PyTorch's JIT compiler that fuses operations, significantly reducing Python overhead and optimizing GPU kernel launches. It often yields speedups of 30-200% with a single line of code.

**How to implement:**
Just before creating the `pl.Trainer`, add this line:

```python
# --- Trainer ---
# ... (previous code)

print("Compiling the model with torch.compile() for a significant speedup...")
model = torch.compile(model)

# Find the latest checkpoint to resume from
resume_from = find_latest_checkpoint()
# ... (rest of the trainer code)
```

### ⚡️ Offload Preprocessing with NVIDIA DALI (Advanced)

Your bottleneck might be the data preprocessing, which currently runs on the CPU. If you monitor `nvidia-smi` and see GPU utilization is frequently below 95%, it means your GPU is waiting for data.

**NVIDIA DALI** can move the entire preprocessing pipeline (resampling, featurization) onto the GPU, running it in parallel with training. This is a more involved change as it requires rewriting your `PersonalizedSwipeFeaturizer` using DALI's operators, but it can fully saturate your GPU.

### 📊 Profile for Bottlenecks

Before making complex changes like DALI, confirm where the bottleneck is. Use the **PyTorch Profiler** (which integrates with PyTorch Lightning) to get a detailed breakdown of time spent in data loading vs. model computation.

```python
# In your pl.Trainer constructor:
profiler = "simple" # or "advanced" for more detail

trainer = pl.Trainer(
    # ... other args ...
    profiler=profiler,
)
```

This will print a performance summary after training, showing you exactly where to focus your optimization efforts.

-----

## Part 3: Web App Architecture Document

Here is a comprehensive guide to running your exported `encoder.onnx` and `decoder_joint.onnx` models in a web application using `onnxruntime-web`.

### Overview

The process involves three main stages:

1.  **Frontend Data Capture:** Capture raw `(x, y, t)` swipe points from the user.
2.  **Frontend Preprocessing:** Re-implement your Python preprocessing (resampling and featurization) **identically** in JavaScript. This is critical to avoid train-serve skew.
3.  **Inference with ONNX Runtime:** Feed the preprocessed features into the ONNX models and run a greedy decoding loop to generate the final text.

### Step 1: Frontend Preprocessing in JavaScript

You'll need `onnxruntime-web`. Install it via npm: `npm install onnxruntime-web`.

First, you need to port your Python preprocessing functions to JavaScript. This code should be part of your web app's logic.

```javascript
// Filename: swipe_preprocessor.js

/**
 * Clamps a value to a specified min/max range.
 */
function clamp(value, min, max) {
    return Math.max(min, Math.min(value, max));
}

/**
 * Determines the target number of frames for adaptive resampling (JS version).
 */
function determineResampleTarget(length, cfg) {
    if (length <= 1) return length;
    const { resample_short_target, resample_long_target, resample_short_threshold, resample_long_threshold } = cfg;
    if (length <= resample_short_threshold) return resample_short_target;
    if (length >= resample_long_threshold) return resample_long_target;

    const progress = (length - resample_short_threshold) / (resample_long_threshold - resample_short_threshold);
    return Math.round(resample_short_target + progress * (resample_long_target - resample_short_target));
}


/**
 * Resamples a sequence of points to a target count (JS version).
 */
function resamplePoints(points, targetCount) {
    if (targetCount <= 0 || points.length === 0) return [];
    if (points.length === targetCount) return points.map(p => ({...p}));

    const resampled = [];
    const firstTime = points[0].t;
    const lastTime = points[points.length - 1].t;
    const duration = Math.max(lastTime - firstTime, 1.0);
    const step = duration / Math.max(targetCount - 1, 1);
    let srcIdx = 0;

    for (let i = 0; i < targetCount; i++) {
        const targetTime = (i === targetCount - 1) ? lastTime : firstTime + step * i;
        while (srcIdx < points.length - 2 && points[srcIdx + 1].t < targetTime) {
            srcIdx++;
        }
        const p1 = points[srcIdx];
        const p2 = points[Math.min(srcIdx + 1, points.length - 1)];
        const span = Math.max(p2.t - p1.t, 1.0);
        const alpha = clamp((targetTime - p1.t) / span, 0.0, 1.0);
        const x = p1.x + (p2.x - p1.x) * alpha;
        const y = p1.y + (p2.y - p1.y) * alpha;
        resampled.push({ x, y, t: targetTime });
    }
    return resampled;
}


/**
 * The core feature extractor, ported to JavaScript.
 * NOTE: `keyCenters` must be loaded and passed in.
 */
function featurize(resampledPoints, keyCenters) {
    if (!resampledPoints || resampledPoints.length === 0) {
        return new Float32Array(37).fill(0);
    }
    const featureVectors = resampledPoints.map((_, idx) => 
        computeFeatureVector(resampledPoints, idx, keyCenters)
    );
    // Flatten the array of arrays into a single Float32Array
    const flatFeatures = new Float32Array(resampledPoints.length * 37);
    for (let i = 0; i < featureVectors.length; i++) {
        flatFeatures.set(featureVectors[i], i * 37);
    }
    return flatFeatures;
}

function computeFeatureVector(points, idx, keyCenters) {
    const total = points.length;
    const curr = points[idx];
    const prev = idx > 0 ? points[idx - 1] : null;
    const prev2 = idx > 1 ? points[idx - 2] : null;

    const x = clamp(curr.x || 0.0, -1.0, 1.0);
    const y = clamp(curr.y || 0.0, -1.0, 1.0);
    const t_ms = curr.t || (idx * 10.0);
    const t_seconds = t_ms / 1000.0;
    
    let vx = 0, vy = 0, speed = 0;
    if (prev) {
        const dt = Math.max((t_ms - (prev.t || 0.0)) / 1000.0, 1e-6);
        vx = (x - (prev.x || x)) / dt;
        vy = (y - (prev.y || y)) / dt;
        speed = Math.hypot(vx, vy);
    }

    let ax = 0, ay = 0, acc = 0;
    if (prev && prev2) {
        const dt1 = Math.max((t_ms - (prev.t || 0.0)) / 1000.0, 1e-6);
        const dt2 = Math.max(((prev.t || 0.0) - (prev2.t || 0.0)) / 1000.0, 1e-6);
        const vx_prev = ((prev.x || 0.0) - (prev2.x || 0.0)) / dt2;
        const vy_prev = ((prev.y || 0.0) - (prev2.y || 0.0)) / dt2;
        ax = (vx - vx_prev) / dt1;
        ay = (vy - vy_prev) / dt1;
        acc = Math.hypot(ax, ay);
    }

    const angle = prev ? Math.atan2(vy, vx) : 0.0;
    let curvature = 0.0;
    if (prev && prev2) {
        const prev_angle = Math.atan2((prev.y || 0.0) - (prev2.y || 0.0), (prev.x || 0.0) - (prev2.x || 0.0));
        curvature = angle - prev_angle;
        while (curvature > Math.PI) curvature -= 2 * Math.PI;
        while (curvature < -Math.PI) curvature += 2 * Math.PI;
    }

    const keyDistances = keyCenters.map(kc => Math.hypot(x - kc[1], y - kc[2])).sort((a, b) => a - b).slice(0, 5);
    while (keyDistances.length < 5) keyDistances.push(1.0);

    const progress = idx / Math.max(total - 1, 1);
    const is_start = idx === 0 ? 1.0 : 0.0;
    const is_end = idx === total - 1 ? 1.0 : 0.0;

    const win_pts = points.slice(Math.max(0, idx - 2), Math.min(total, idx + 3));
    let win_mean_x = x, win_std_x = 0, win_mean_y = y, win_std_y = 0, win_range_x = 0, win_range_y = 0;
    if (win_pts.length > 1) {
        const xs = win_pts.map(p => p.x);
        const ys = win_pts.map(p => p.y);
        const mean = arr => arr.reduce((a, b) => a + b) / arr.length;
        const std = (arr, m) => Math.sqrt(arr.reduce((sq, n) => sq + Math.pow(n - m, 2), 0) / (arr.length - 1));
        win_mean_x = mean(xs);
        win_mean_y = mean(ys);
        win_std_x = std(xs, win_mean_x);
        win_std_y = std(ys, win_mean_y);
        win_range_x = Math.max(...xs) - Math.min(...xs);
        win_range_y = Math.max(...ys) - Math.min(...ys);
    }
    
    // IMPORTANT: The order must EXACTLY match `FEATURE_NAMES` in your Python script.
    const features = [
        x, y, t_seconds, vx, vy, speed, ax, ay, acc, 
        angle, Math.sin(angle), Math.cos(angle), curvature, 
        ...keyDistances,
        progress, is_start, is_end,
        win_mean_x, win_std_x, win_mean_y, win_std_y,
        win_range_x, win_range_y
    ];
    
    // Pad to 37 features
    const finalFeatures = new Float32Array(37).fill(0.0);
    finalFeatures.set(features);
    return finalFeatures;
}
```

### Step 2: Inference with ONNX Runtime

This part orchestrates the ONNX models. The core is the **greedy decoding loop**.

```javascript
// Filename: swipe_decoder.js
import * as ort from 'onnxruntime-web';

export class SwipeDecoder {
    constructor() {
        this.encoderSession = null;
        this.decoderJointSession = null;
        this.meta = null;
        this.isInitialized = false;
    }

    /**
     * Load models and metadata. Call this once.
     * @param {string} encoderPath - URL to encoder.onnx
     * @param {string} decoderJointPath - URL to decoder_joint.onnx
     * @param {string} metaPath - URL to runtime_meta.json
     */
    async initialize(encoderPath, decoderJointPath, metaPath) {
        // Use 'wasm' for broad compatibility, or 'webgl' for GPU acceleration
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        
        const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
        
        [this.encoderSession, this.decoderJointSession, this.meta] = await Promise.all([
            ort.InferenceSession.create(encoderPath, options),
            ort.InferenceSession.create(decoderJointPath, options),
            fetch(metaPath).then(res => res.json())
        ]);
        
        this.isInitialized = true;
        console.log("Swipe decoder initialized successfully.");
    }

    /**
     * Performs the full recognition on a feature tensor.
     * @param {Float32Array} features - The flattened feature array from featurize().
     * @param {number} numFrames - The number of time steps in the swipe.
     * @returns {string} The predicted word.
     */
    async predict(features, numFrames) {
        if (!this.isInitialized) throw new Error("Decoder not initialized.");

        // 1. Run the Encoder (once)
        const featureTensor = new ort.Tensor('float32', features, [1, numFrames, 37]);
        const encoderFeeds = { 'audio_signal': featureTensor };
        const encoderResults = await this.encoderSession.run(encoderFeeds);
        const encoded = encoderResults.encoded; // Shape: [1, T, D_encoder]

        // 2. Greedy Decoding Loop
        let decodedTokens = [];
        const maxSymbols = 15; // Safety break
        
        // Initial state for the decoder (prediction network)
        // The first token fed to the decoder is always the blank token.
        let prevToken = new ort.Tensor('int64', [BigInt(this.meta.blank_id)], [1, 1]);
        
        // You must know the initial state shapes. NeMo's LSTM decoders typically
        // have two states (h and c) per layer. For 2 layers, hidden=320:
        // Shape: [num_layers, batch_size, hidden_dim] -> [2, 1, 320]
        const predHiddenDim = 320; 
        const predRnnLayers = 2;
        let predState = new ort.Tensor('float32', new Float32Array(predRnnLayers * 1 * predHiddenDim).fill(0), [predRnnLayers, 1, predHiddenDim]);
        let predCell = new ort.Tensor('float32', new Float32Array(predRnnLayers * 1 * predHiddenDim).fill(0), [predRnnLayers, 1, predHiddenDim]);

        for (let t = 0; t < encoded.dims[1]; ++t) {
            // Get the encoder output for the current time step `t`
            const encoded_t = this.sliceTensor(encoded, t);

            // This loop simulates the "while (pred != blank)" part of RNN-T
            // It allows multiple characters to be predicted for a single audio frame.
            while (decodedTokens.length < maxSymbols) {
                // Prepare inputs for the decoder/joint model
                const decoderFeeds = {
                    'encoder_output': encoded_t,
                    'previous_token': prevToken,
                    'pred_hidden_state': predState,
                    'pred_cell_state': predCell
                };

                const decoderResults = await this.decoderJointSession.run(decoderFeeds);
                
                // Get the most likely token (argmax)
                const logits = decoderResults.logits.data;
                const nextTokenId = logits.indexOf(Math.max(...logits));
                
                // Update decoder state for the *next* iteration
                predState = decoderResults.next_pred_hidden_state;
                predCell = decoderResults.next_pred_cell_state;
                
                if (nextTokenId === this.meta.blank_id) {
                    // If blank is predicted, consume the audio frame and move to the next one.
                    break;
                } else {
                    // If a character is predicted, append it, and feed it back
                    // into the decoder for the SAME audio frame `t`.
                    decodedTokens.push(nextTokenId);
                    prevToken = new ort.Tensor('int64', [BigInt(nextTokenId)], [1, 1]);
                }
            }
        }
        
        // 3. Convert token IDs to a string
        return decodedTokens.map(id => this.meta.id_to_char[id]).join('');
    }

    /** Helper to slice a tensor along a specific dimension */
    sliceTensor(tensor, index) {
        const [batch, time, dims] = tensor.dims;
        const offset = index * dims;
        const slicedData = tensor.data.slice(offset, offset + dims);
        return new ort.Tensor(tensor.type, slicedData, [1, 1, dims]);
    }
}
```

### Step 3: Putting It All Together

In your main application logic:

1.  Instantiate the `SwipeDecoder`.
2.  Call `initialize()` once with the paths to your model files.
3.  In your gesture handling logic (e.g., on `touchend` or `mouseup`):
    a. Collect the raw swipe points `[{x, y, t}, ...]`.
    b. Normalize coordinates to `[-1, 1]`.
    c. Perform resampling and featurization using the JavaScript functions from Step 1.
    d. Call `decoder.predict(features, numFrames)` to get the result.
    e. Display the result to the user.

**Example Usage:**

```javascript
// Main app file

import { SwipeDecoder } from './swipe_decoder.js';
import { determineResampleTarget, resamplePoints, featurize } from './swipe_preprocessor.js';

// --- Assume these are loaded from your config/server ---
const PREPROCESS_CFG = {
    resample_short_target: 56,
    resample_long_target: 96,
    resample_short_threshold: 48,
    resample_long_threshold: 112,
};
let KEY_CENTERS = []; // Load this from a JSON file, e.g., [['q', -0.9, -0.8], ...]


async function main() {
    const decoder = new SwipeDecoder();
    await decoder.initialize('./models/encoder.onnx', './models/decoder_joint.onnx', './models/runtime_meta.json');
    
    // --- In your event handler for a completed swipe ---
    // const rawSwipePoints = [{x: 230, y: 450, t: 1668...}, ...];
    // const keyboardLayout = {width: 1080, height: 600, x: 0, y: 400}; // Example
    
    // 1. Normalize points
    // const normalizedPoints = rawSwipePoints.map(p => ({
    //     x: (p.x - keyboardLayout.x) / keyboardLayout.width * 2 - 1,
    //     y: (p.y - keyboardLayout.y) / keyboardLayout.height * 2 - 1,
    //     t: p.t
    // }));

    // For this example, let's use dummy data:
    const dummyPoints = Array.from({length: 80}, (_, i) => ({
        x: Math.sin(i/10) * 0.5,
        y: (i/80) * 1.6 - 0.8,
        t: i * 16
    }));

    // 2. Preprocess
    const targetLen = determineResampleTarget(dummyPoints.length, PREPROCESS_CFG);
    const resampled = resamplePoints(dummyPoints, targetLen);
    const features = featurize(resampled, KEY_CENTERS);
    
    // 3. Predict
    const result = await decoder.predict(features, resampled.length);
    console.log("Prediction:", result);
    // document.getElementById('output').innerText = result;
}

main();
```


-----

### Key Concepts for ONNX Beam Search

1.  **Hypothesis State:** You'll need a data structure (like a class or object) for each hypothesis in your beam to track:

      * `tokens`: The array of predicted token IDs so far.
      * `score`: The cumulative log-probability of this sequence.
      * `decoderState`: The last hidden and cell states from the `decoder_joint.onnx` model for *this specific hypothesis*.
      * `lastToken`: The last non-blank token ID predicted, used as the next input to the decoder.

2.  **Log Probabilities:** Always work with log probabilities. Your model outputs logits, which should be converted to log probabilities using a LogSoftmax function. This prevents numerical underflow (multiplying many small probabilities results in zero) by turning multiplication into addition.

3.  **The Loop:** The core logic is to iterate through each time step (`t`) of the encoder's output. At each step, you expand every hypothesis currently in your beam with all possible next tokens, creating a large pool of new candidates. You then prune this pool down to the `k` best candidates, which become your new beam for the next time step.

-----

### Step-by-Step Implementation

Here is how you would modify the `predict` function from the previous answer to perform beam search.

```javascript
// In swipe_decoder.js

// A helper class to manage the state of each hypothesis in the beam
class Hypothesis {
    constructor(tokens, score, decoderState, decoderCell, lastToken) {
        this.tokens = tokens;         // Array of token IDs
        this.score = score;           // Cumulative log-probability
        this.decoderState = decoderState; // Hidden state tensor for this path
        this.decoderCell = decoderCell;   // Cell state tensor for this path
        this.lastToken = lastToken;     // The last predicted non-blank token tensor
    }
    
    // Creates a new hypothesis by extending the current one
    extend(token, logProb, newState, newCell) {
        const newTokens = [...this.tokens, token.data[0]];
        const newScore = this.score + logProb;
        return new Hypothesis(newTokens, newScore, newState, newCell, token);
    }
}


export class SwipeDecoder {
    // ... constructor and initialize methods are the same ...

    /**
     * Performs beam search recognition on a feature tensor.
     * @param {Float32Array} features - The flattened feature array from featurize().
     * @param {number} numFrames - The number of time steps in the swipe.
     * @param {number} beamWidth - The number of hypotheses to keep (e.g., 5).
     * @returns {string} The best predicted word.
     */
    async predictBeamSearch(features, numFrames, beamWidth = 5) {
        if (!this.isInitialized) throw new Error("Decoder not initialized.");

        // 1. Run the Encoder (once)
        const featureTensor = new ort.Tensor('float32', features, [1, numFrames, 37]);
        const encoderResults = await this.encoderSession.run({ 'audio_signal': featureTensor });
        const encoded = encoderResults.encoded;

        // 2. Initialize the Beam
        // The initial state for the decoder (prediction network)
        const predHiddenDim = 320; 
        const predRnnLayers = 2;
        const initialPredState = new ort.Tensor('float32', new Float32Array(predRnnLayers * 1 * predHiddenDim).fill(0), [predRnnLayers, 1, predHiddenDim]);
        const initialPredCell = new ort.Tensor('float32', new Float32Array(predRnnLayers * 1 * predHiddenDim).fill(0), [predRnnLayers, 1, predHiddenDim]);
        const initialToken = new ort.Tensor('int64', [BigInt(this.meta.blank_id)], [1, 1]);

        // The beam starts with a single empty hypothesis
        let beam = [new Hypothesis([], 0.0, initialPredState, initialPredCell, initialToken)];

        // 3. Beam Search Loop (Iterate through encoder time steps)
        for (let t = 0; t < encoded.dims[1]; ++t) {
            const encoded_t = this.sliceTensor(encoded, t);
            const allCandidates = [];

            // --- EXPANSION STEP ---
            // For every hypothesis currently in the beam...
            for (const hypo of beam) {
                // Run the decoder/joint model for this specific hypothesis's state
                const decoderFeeds = {
                    'encoder_output': encoded_t,
                    'previous_token': hypo.lastToken,
                    'pred_hidden_state': hypo.decoderState,
                    'pred_cell_state': hypo.decoderCell
                };
                const decoderResults = await this.decoderJointSession.run(decoderFeeds);
                
                // Convert logits to log probabilities
                const logProbs = this.logSoftmax(decoderResults.logits.data);

                // ...expand it with all possible next tokens (from the vocabulary)
                for (let tokenId = 0; tokenId < logProbs.length; tokenId++) {
                    const logProb = logProbs[tokenId];
                    let nextHypo;

                    if (tokenId === this.meta.blank_id) {
                        // If blank is predicted, we don't change the text,
                        // just update the score and carry over the state.
                        nextHypo = new Hypothesis(hypo.tokens, hypo.score + logProb, hypo.decoderState, hypo.decoderCell, hypo.lastToken);
                    } else {
                        // If a character is predicted, create a new extended hypothesis
                        const newToken = new ort.Tensor('int64', [BigInt(tokenId)], [1, 1]);
                        nextHypo = hypo.extend(
                            newToken, 
                            logProb, 
                            decoderResults.next_pred_hidden_state, 
                            decoderResults.next_pred_cell_state
                        );
                    }
                    allCandidates.push(nextHypo);
                }
            }
            
            // --- PRUNING STEP ---
            // Sort all generated candidates by their score
            allCandidates.sort((a, b) => b.score - a.score); // descending order
            
            // Keep only the top `beamWidth` candidates
            beam = allCandidates.slice(0, beamWidth);
        }

        // 4. Finalize and select the best hypothesis
        // The beam is already sorted, so the best one is the first element.
        // Optional: Apply a length penalty here to avoid bias for shorter sentences.
        // bestHypo.score /= Math.pow(bestHypo.tokens.length, alpha);
        const bestHypothesis = beam[0];
        
        // Convert token IDs to a string
        return bestHypothesis.tokens.map(id => this.meta.id_to_char[id]).join('');
    }

    /** Helper to apply log-softmax to logits */
    logSoftmax(logits) {
        const maxLogit = Math.max(...logits);
        const exps = logits.map(l => Math.exp(l - maxLogit));
        const sumExps = exps.reduce((a, b) => a + b);
        const logSumExps = Math.log(sumExps);
        return logits.map((l, i) => l - maxLogit - logSumExps);
    }
    
    // ... sliceTensor helper method is the same ...
    sliceTensor(tensor, index) {
        const [batch, time, dims] = tensor.dims;
        const offset = index * dims;
        const slicedData = tensor.data.slice(offset, offset + dims);
        return new ort.Tensor(tensor.type, slicedData, [1, 1, dims]);
    }
}

```
