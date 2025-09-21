# CleverKeys Web Demo Implementation Guide

This guide explains how to run the exported RNN-T models in a web application using `onnxruntime-web`.

## Architecture Overview

The RNN-T (Recurrent Neural Network Transducer) model consists of three separate components that must be exported and run as stateful models:

1. **Encoder (`encoder.onnx`)**: Conformer-based acoustic model that processes the entire input gesture
2. **Decoder (`decoder.onnx`)**: Stateful LSTM prediction network with explicit state management
3. **Joint Network (`joint.onnx`)**: Combines encoder and decoder outputs to produce character probabilities

## Part 1: Exporting the Models

Use the stateful export script to properly export all three components:

```bash
uv run python new/export_onnx_stateful.py \
  --checkpoint path/to/model.ckpt \
  --output_dir web-demo/models/
```

This creates:
- `encoder.onnx`: Processes features [batch, features, time] → [batch, time, encoder_dim]
- `decoder.onnx`: Stateful LSTM with inputs (token, h_in, c_in) → (decoder_out, h_out, c_out)
- `joint.onnx`: Combines (encoder_frame, decoder_out) → logits
- `runtime_meta.json`: Vocabulary and model configuration

## Part 2: JavaScript Feature Extraction

The feature extraction must exactly match the Python training pipeline. The actual features used are:

```javascript
// Feature order (37 total dimensions):
const FEATURE_NAMES = [
    'x', 'y', 't_seconds',           // Position and time (3)
    'vx', 'vy', 'speed',              // Velocity (3)
    'ax', 'ay', 'acc',                // Acceleration (3)
    'angle', 'angle_sin', 'angle_cos', 'curvature', // Trajectory (4)
    'dist_key1', 'dist_key2', 'dist_key3', 'dist_key4', 'dist_key5', // Key distances (5)
    'progress', 'is_start', 'is_end', // Temporal markers (3)
    'win_mean_x', 'win_std_x', 'win_mean_y', 'win_std_y', // Window stats (4)
    'win_range_x', 'win_range_y',     // Window range (2)
    // Plus 9 padding zeros to reach 37 dimensions
];
```

### Preprocessing Pipeline

```javascript
class SwipePreprocessor {
    constructor() {
        // Adaptive resampling parameters (must match Python)
        this.config = {
            resample_short_target: 56,
            resample_long_target: 96,
            resample_short_threshold: 48,
            resample_long_threshold: 112
        };

        this.keyCenters = this.getQWERTYLayout();
    }

    getQWERTYLayout() {
        const layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"];
        const centers = [];

        for (let row = 0; row < layout.length; row++) {
            const rowStr = layout[row];
            const rowOffset = row === 2 ? 0.05 : 0; // Bottom row offset

            for (let col = 0; col < rowStr.length; col++) {
                const char = rowStr[col];
                // Convert to [-1, 1] coordinate system
                const x01 = rowOffset + (col + 0.5) / 10.0;
                const y01 = (row + 0.5) / 3.0;
                const x = x01 * 2.0 - 1.0;
                const y = y01 * 2.0 - 1.0;
                centers.push([char, x, y]);
            }
        }
        return centers;
    }

    determineResampleTarget(length) {
        if (length <= this.config.resample_short_threshold) {
            return this.config.resample_short_target;
        }
        if (length >= this.config.resample_long_threshold) {
            return this.config.resample_long_target;
        }
        // Linear interpolation
        const progress = (length - this.config.resample_short_threshold) /
                        (this.config.resample_long_threshold - this.config.resample_short_threshold);
        return Math.round(this.config.resample_short_target +
                         progress * (this.config.resample_long_target - this.config.resample_short_target));
    }

    process(rawPoints) {
        // 1. Normalize coordinates to [-1, 1]
        const normalized = this.normalizePoints(rawPoints);

        // 2. Adaptive resampling
        const targetLen = this.determineResampleTarget(normalized.length);
        const resampled = this.resamplePoints(normalized, targetLen);

        // 3. Feature extraction
        const features = this.extractFeatures(resampled);

        return { features, numFrames: resampled.length };
    }
}
```

## Part 3: Stateful RNN-T Decoder

The decoder must properly manage LSTM states between time steps:

```javascript
class RNNTDecoder {
    constructor() {
        this.encoderSession = null;
        this.decoderSession = null;
        this.jointSession = null;
        this.meta = null;
    }

    async initialize(modelPaths) {
        const options = {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        };

        // Load all three models
        this.encoderSession = await ort.InferenceSession.create(modelPaths.encoder, options);
        this.decoderSession = await ort.InferenceSession.create(modelPaths.decoder, options);
        this.jointSession = await ort.InferenceSession.create(modelPaths.joint, options);

        // Load metadata
        const response = await fetch(modelPaths.meta);
        this.meta = await response.json();

        // Extract configuration
        this.blankId = this.meta.blank_id; // Should be 29
        this.numLayers = this.meta.decoder_config.num_layers; // 2
        this.hiddenSize = this.meta.decoder_config.hidden_size; // 320
    }

    async decode(features, numFrames) {
        // 1. Run encoder once on entire sequence
        const encoderInputs = {
            'audio_signal': new ort.Tensor('float32', features, [1, 37, numFrames]),
            'length': new ort.Tensor('int64', [BigInt(numFrames)], [1])
        };

        const encoderOutputs = await this.encoderSession.run(encoderInputs);
        const encoded = encoderOutputs.encoded;
        const encodedLen = encoderOutputs.encoded_lengths.data[0];

        // 2. Initialize LSTM states
        const batchSize = 1;
        let hState = new Float32Array(this.numLayers * batchSize * this.hiddenSize);
        let cState = new Float32Array(this.numLayers * batchSize * this.hiddenSize);

        // 3. Greedy decoding loop
        const tokens = [];
        let currentToken = this.blankId;

        for (let t = 0; t < encodedLen; t++) {
            // Extract encoder frame at time t
            const frameData = new Float32Array(this.meta.decoder_config.encoder_dim);
            const offset = t * this.meta.decoder_config.encoder_dim;
            for (let i = 0; i < frameData.length; i++) {
                frameData[i] = encoded.data[offset + i];
            }
            const encoderFrame = new ort.Tensor('float32', frameData,
                                                [1, 1, this.meta.decoder_config.encoder_dim]);

            // Run decoder with current token and state
            const decoderInputs = {
                'input_tokens': new ort.Tensor('int64', [BigInt(currentToken)], [1, 1]),
                'h_in': new ort.Tensor('float32', hState, [this.numLayers, batchSize, this.hiddenSize]),
                'c_in': new ort.Tensor('float32', cState, [this.numLayers, batchSize, this.hiddenSize])
            };

            const decoderOutputs = await this.decoderSession.run(decoderInputs);

            // Update states for next iteration
            hState = decoderOutputs.h_out.data;
            cState = decoderOutputs.c_out.data;

            // Run joint network
            const jointInputs = {
                'encoder_output': encoderFrame,
                'decoder_output': decoderOutputs.decoder_output
            };

            const jointOutputs = await this.jointSession.run(jointInputs);
            const logits = jointOutputs.logits.data;

            // Get argmax prediction
            let maxIdx = 0;
            let maxVal = logits[0];
            for (let i = 1; i < logits.length; i++) {
                if (logits[i] > maxVal) {
                    maxVal = logits[i];
                    maxIdx = i;
                }
            }

            // Update token if not blank
            if (maxIdx !== this.blankId) {
                tokens.push(maxIdx);
                currentToken = maxIdx;
            }

            // Early stopping
            if (tokens.length >= 20) break;
        }

        // Convert tokens to text
        return tokens.map(t => this.meta.tokens[t] || '').join('');
    }
}
```

## Part 4: Complete Web Application

```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    <script src="js/feature-extractor.js"></script>
    <script src="js/onnx-rnnt-decoder.js"></script>
</head>
<body>
    <canvas id="keyboard"></canvas>
    <div id="output"></div>

    <script>
        const decoder = new RNNTDecoder();
        const preprocessor = new SwipePreprocessor();

        async function init() {
            await decoder.initialize({
                encoder: 'models/encoder.onnx',
                decoder: 'models/decoder.onnx',
                joint: 'models/joint.onnx',
                meta: 'models/runtime_meta.json'
            });
            console.log('Models loaded');
        }

        async function processSwipe(swipePoints) {
            // Preprocess
            const { features, numFrames } = preprocessor.process(swipePoints);

            // Decode
            const text = await decoder.decode(features, numFrames);

            document.getElementById('output').textContent = text;
        }

        init();
    </script>
</body>
</html>
```

## Key Differences from Original Guide

1. **Three Separate Models**: The stateful architecture requires encoder, decoder, and joint as separate ONNX files, not a combined decoder_joint
2. **Explicit State Management**: The decoder now explicitly manages LSTM hidden and cell states as tensors
3. **Proper Input Names**: Uses the actual tensor names from the exported models
4. **Correct Feature Order**: Matches the exact 37-dimensional feature vector from training

## Testing

To test the implementation:

1. Export models using `export_onnx_stateful.py`
2. Copy the generated files to `web-demo/models/`
3. Open `swipe-rnnt-stateful.html` in a browser
4. Draw a swipe gesture on the keyboard

The models should produce text predictions matching the training performance.