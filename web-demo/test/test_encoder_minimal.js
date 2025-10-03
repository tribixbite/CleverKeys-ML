#!/usr/bin/env node
/*
 * Minimal encoder test - load the same input as Python and compare
 */

const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');

async function main() {
    // Load the transposed features from JS
    const jsFeatures = JSON.parse(fs.readFileSync('js_features.json', 'utf-8'));

    // Create transposed array like Python
    const T = jsFeatures.shape[0];  // 82
    const featDim = jsFeatures.shape[1];  // 37
    const transposed = new Float32Array(featDim * T);

    for (let t = 0; t < T; t++) {
        for (let f = 0; f < featDim; f++) {
            transposed[f * T + t] = jsFeatures.data[t * featDim + f];
        }
    }

    console.log(`Input shape: [1, ${featDim}, ${T}]`);
    console.log(`First 10 values: [${transposed.slice(0, 10).map(x => x.toFixed(6)).join(', ')}]`);

    // Load and run encoder with explicit CPU provider
    const encoderPath = path.join(__dirname, '..', 'models', 'correct_9292025', 'encoder.onnx');
    const encoderSession = await ort.InferenceSession.create(encoderPath, {
        executionProviders: ['cpu']
    });

    const encInputs = {
        'audio_signal': new ort.Tensor('float32', transposed, [1, featDim, T]),
        'length': new ort.Tensor('int64', BigInt64Array.from([BigInt(T)]), [1])
    };

    const encOut = await encoderSession.run(encInputs);
    const encoded = encOut['outputs'];
    const encodedLen = Number(encOut['encoded_lengths'].data[0]);

    console.log(`Encoder output shape: [${encoded.dims}]`);
    console.log(`Encoded length: ${encodedLen}`);
    console.log(`First 10 values: [${encoded.data.slice(0, 10).map(x => x.toFixed(6)).join(', ')}]`);

    // Save output for comparison
    const outputData = {
        shape: encoded.dims,
        data: Array.from(encoded.data)
    };
    fs.writeFileSync('encoder_output_js.json', JSON.stringify(outputData, null, 2));
    console.log('Saved encoder output to encoder_output_js.json');
}

main().catch(console.error);