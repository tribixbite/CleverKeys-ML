Of course. Your model architecture and training script are very well-structured and employ advanced techniques like weighted sampling and knowledge distillation. The primary flaws are in the experimental ExecuTorch (`.pte`) export logic. For training speed, the easiest and most effective improvement is to use `torch.compile`.

Here's a detailed breakdown of potential improvements, training speed optimizations, and a full architecture document for a web demo.

-----

## Part 1: Architecture and Exporting Flaws & Improvements

Your core training script is robust. The Conformer-RNNT architecture, data preprocessing (adaptive resampling), and sampling strategies are excellent choices. The issues are minor or concentrated in the export script.

### 📉 Flaws

1.  **Major Flaw in `.pte` Export:** The ExecuTorch export function `export_pte` is incomplete and will produce a non-functional model.

      * **Missing Joint Network:** The script exports the `encoder` and `decoder`, but **it completely omits the `joint` network**. The joint network is essential for combining the encoder and decoder outputs to produce probabilities. It must be exported as well.
      * **Static Input Shapes:** The export uses `torch.randn(1, 37, 96)` as a sample input. This bakes a static sequence length of 96 into the exported graph. The model will fail on any input that isn't exactly 96 frames long. You must use `torch.export`'s `dynamic_shapes` argument to specify that the time dimension is variable.

2.  **Minor Flaw in ONNX Export:** The script renames `encoder-model.onnx` to `encoder.onnx`, but for the decoder, it renames `decoder_joint-model.onnx` to `decoder_joint.onnx`. NeMo's export actually produces three files: `encoder-model.onnx`, `decoder-model.onnx`, and `joint-model.onnx`. Your script seems to assume a combined `decoder_joint-model.onnx`, which might not be what NeMo produces by default. It's better to export them separately and combine them in the runtime logic or ensure NeMo is configured to export them as a single graph.

### ✨ Improvements

1.  **Feature Vector Padding:** In `PersonalizedSwipeFeaturizer`, `FINAL_FEATURE_COUNT` is hardcoded to 37, and the generated vector is padded with zeros to match. This works, but it's brittle. It's better to make the model's `feat_in` parameter match the actual number of features you generate (`len(self.FEATURE_NAMES)`). This eliminates unused zero inputs and makes the feature engineering process easier to modify.

2.  **Explicit Transpose:** Your `PersonalizedRNNTModel.forward` method transposes the input from `(Batch, Time, Features)` to `(Batch, Features, Time)` to match the Conformer's expectation. This is fine, but for clarity, you could add a `torch.nn.Permute(0, 2, 1)` layer as the very first layer in your `encoder` module to make this data shape transformation an explicit part of the model architecture.

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