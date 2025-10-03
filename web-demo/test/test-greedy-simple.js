#!/usr/bin/env node
/*
 * Simple greedy decoder test - no beam search
 */

const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');

async function main() {
    // Load models
    const modelDir = path.join(__dirname, '..', 'models', 'correct_9292025');
    const encoderSession = await ort.InferenceSession.create(path.join(modelDir, 'encoder.onnx'));
    const decoderSession = await ort.InferenceSession.create(path.join(modelDir, 'decoder_joint.onnx'));

    // Load metadata
    const meta = JSON.parse(fs.readFileSync(path.join(modelDir, 'runtime_meta.json'), 'utf-8'));
    const blankId = meta.blank_id;
    const vocab = meta.tokens;
    const numLayers = meta.decoder_config.num_layers;
    const hiddenSize = meta.decoder_config.hidden_size;
    const joint2pred = meta.predictor.label_map.joint2pred;

    // Get hello data (line 431621)
    const readline = require('readline');
    const dataPath = path.join(__dirname, '..', '..', 'data', 'train_final_train.jsonl');
    const fileStream = fs.createReadStream(dataPath);
    const rl = readline.createInterface({ input: fileStream });

    let currentLine = 0;
    let points, word;

    for await (const line of rl) {
        currentLine++;
        if (currentLine === 431621) {
            const lineData = JSON.parse(line);
            points = lineData.points;
            word = lineData.word;
            break;
        }
    }
    rl.close();

    console.log(`Testing: '${word}' with ${points.length} points`);

    // Process swipe using JS feature extractor
    const FeatureExtractor = require('../js/feature-extractor-corrected.js');
    const featureExtractor = new FeatureExtractor();
    const featureData = featureExtractor.process(points);

    console.log(`Features: ${featureData.numFrames} frames`);

    // Prepare for encoder
    const T = featureData.numFrames;
    const featDim = 37;
    const transposed = new Float32Array(featDim * T);
    for (let t = 0; t < T; t++) {
        for (let f = 0; f < featDim; f++) {
            transposed[f * T + t] = featureData.features[t * featDim + f];
        }
    }

    // Verify the transposed values
    console.log(`Transposed first 5 values: [${transposed.slice(0, 5).map(x => x.toFixed(6)).join(', ')}]`);

    // Run encoder
    const encInputs = {
        'audio_signal': new ort.Tensor('float32', transposed, [1, featDim, T]),
        'length': new ort.Tensor('int64', BigInt64Array.from([BigInt(T)]), [1])
    };

    const encOut = await encoderSession.run(encInputs);
    const encoded = encOut['outputs'];
    const encodedLen = Number(encOut['encoded_lengths'].data[0]);

    console.log(`Encoded frames: ${encodedLen}`);
    console.log(`Encoder output shape: [${encoded.dims}]`);
    console.log(`Frame 0 first 5 values: [${encoded.data.slice(0, 5).map(x => x.toFixed(3)).join(', ')}]`);

    // Initialize decoder states
    let stateH = new ort.Tensor('float32', new Float32Array(numLayers * hiddenSize).fill(0), [numLayers, 1, hiddenSize]);
    let stateC = new ort.Tensor('float32', new Float32Array(numLayers * hiddenSize).fill(0), [numLayers, 1, hiddenSize]);
    let y = new ort.Tensor('int32', Int32Array.from([0]), [1, 1]); // BOS

    const predictions = [];
    const charsPerFrame = [];

    // Greedy decode through all frames
    for (let t = 0; t < encodedLen; t++) {
        // Extract encoder frame - ONNX format is [batch, encoder_dim, time]
        const encoderDim = encoded.dims[1];
        const frameVec = new Float32Array(encoderDim);

        // Correct indexing for [1, encoder_dim, time] layout
        for (let d = 0; d < encoderDim; d++) {
            frameVec[d] = encoded.data[d * encodedLen + t];
        }

        const encoderFrame = new ort.Tensor('float32', frameVec, [1, frameVec.length, 1]);

        const frameChars = [];

        // Try to emit up to 8 characters per frame
        for (let s = 0; s < 8; s++) {
            const decInputs = {
                'targets': y,
                'input_states_1': stateH,
                'input_states_2': stateC,
                'encoder_outputs': encoderFrame,
                'target_length': new ort.Tensor('int32', Int32Array.from([1]), [1])
            };

            const decOut = await decoderSession.run(decInputs);

            // Get logits and states
            const logits = decOut['outputs'].data;
            stateH = decOut['output_states_1'];
            stateC = decOut['output_states_2'];

            // Find argmax
            let maxVal = -Infinity;
            let predIdx = -1;
            for (let i = 0; i < logits.length; i++) {
                if (logits[i] > maxVal) {
                    maxVal = logits[i];
                    predIdx = i;
                }
            }

            if (t === 0 && s === 0) {
                console.log(`Frame 0 logits: blank=${logits[blankId]?.toFixed(2)}, h=${logits[9]?.toFixed(2)}, e=${logits[6]?.toFixed(2)}`);
                console.log(`Frame 0 prediction: ${predIdx} (${vocab[predIdx]}) with score ${maxVal.toFixed(2)}`);
            }

            if (predIdx === blankId) {
                // Blank - stop emitting for this frame
                break;
            } else {
                // Emit character
                const char = vocab[predIdx] || '?';
                frameChars.push(char);
                predictions.push(predIdx);

                // Update y for next prediction
                const mappedIdx = joint2pred[predIdx];
                const nextIdx = mappedIdx === -1 ? 0 : mappedIdx;
                y = new ort.Tensor('int32', Int32Array.from([nextIdx]), [1, 1]);
            }
        }

        charsPerFrame.push(frameChars);

        // Safety: stop if we've predicted too many
        if (predictions.length >= 50) break;
    }

    const predText = predictions.map(idx => vocab[idx] || '?').join('');

    console.log(`Predicted: '${predText}'`);
    console.log(`Expected:  '${word}'`);

    // Show which frames emitted characters
    console.log('\nFrames with output:');
    for (let i = 0; i < Math.min(20, charsPerFrame.length); i++) {
        if (charsPerFrame[i].length > 0) {
            console.log(`  Frame ${i}: [${charsPerFrame[i].join(', ')}]`);
        }
    }
}

main().catch(console.error);