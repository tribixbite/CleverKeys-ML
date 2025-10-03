#!/usr/bin/env node
/**
 * Debug vocabulary and model output issues
 */

const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');
const readline = require('readline');

async function loadRealSwipe(targetLine) {
    const dataPath = path.join(__dirname, '../../data/train_final_train.jsonl');
    const fileStream = fs.createReadStream(dataPath);
    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity
    });

    let lineCount = 0;
    for await (const line of rl) {
        lineCount++;
        if (lineCount === targetLine) {
            rl.close();
            return JSON.parse(line);
        }
    }

    throw new Error(`Line ${targetLine} not found`);
}

async function main() {
    const baseDir = path.resolve(__dirname, '..');
    const modelDir = path.join(baseDir, 'models', 'best_latest');

    // Load runtime meta
    const metaPath = path.join(modelDir, 'runtime_meta.json');
    const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));

    console.log('Runtime Meta Analysis:');
    console.log('----------------------');
    console.log(`vocab_size: ${meta.vocab_size}`);
    console.log(`blank_id: ${meta.blank_id}`);
    console.log(`tokens length: ${meta.tokens.length}`);
    console.log(`tokens[29]: "${meta.tokens[29]}" (empty string)`);
    console.log();

    // Check vocab.txt
    const vocabPath = path.join(baseDir, '../data/vocab.txt');
    const vocabLines = fs.readFileSync(vocabPath, 'utf-8').split('\n').filter(l => l);
    console.log('vocab.txt Analysis:');
    console.log('-------------------');
    console.log(`Lines in vocab.txt: ${vocabLines.length}`);
    console.log(`First: "${vocabLines[0]}"`);
    console.log(`Last: "${vocabLines[vocabLines.length - 1]}"`);
    console.log();

    // Load test data
    const helloData = await loadRealSwipe(431621);
    console.log('Test Data:');
    console.log('----------');
    console.log(`Word: "${helloData.word}"`);
    console.log(`Expected indices for "hello": h=9, e=6, l=13, l=13, o=16`);
    console.log();

    // Test encoder output shape
    const encoderPath = path.join(modelDir, 'encoder.onnx');
    const decoderPath = path.join(modelDir, 'decoder_joint.onnx');

    const sessionOptions = {
        executionProviders: ['cpu'],
        graphOptimizationLevel: 'all'
    };

    console.log('Loading models...');
    const encoder = await ort.InferenceSession.create(encoderPath, sessionOptions);
    const decoder = await ort.InferenceSession.create(decoderPath, sessionOptions);

    console.log('\nEncoder Analysis:');
    console.log('-----------------');
    console.log('Inputs:', encoder.inputNames);
    console.log('Outputs:', encoder.outputNames);

    console.log('\nDecoder/Joint Analysis:');
    console.log('-----------------------');
    console.log('Inputs:', decoder.inputNames);
    console.log('Outputs:', decoder.outputNames);

    // Create dummy input to check output shape
    const dummyFeatures = new Float32Array(37 * 96).fill(0.0);
    const featureTensor = new ort.Tensor('float32', dummyFeatures, [1, 37, 96]);
    const lengthTensor = new ort.Tensor('int64', new BigInt64Array([96n]), [1]);

    const encoderOutputs = await encoder.run({
        'audio_signal': featureTensor,
        'length': lengthTensor
    });
    const encoded = encoderOutputs['outputs'];
    console.log('\nEncoder output shape:', encoded.dims);

    // Test decoder with dummy input - use proper names
    const predStates1 = new ort.Tensor('float32', new Float32Array(1 * 1 * 192).fill(0), [1, 1, 192]);
    const predStates2 = new ort.Tensor('float32', new Float32Array(1 * 1 * 192).fill(0), [1, 1, 192]);
    const targetIds = new ort.Tensor('int32', new Int32Array([0]), [1, 1]);
    const targetLengths = new ort.Tensor('int32', new Int32Array([1]), [1]);

    const decoderOutputs = await decoder.run({
        'encoder_outputs': encoded,
        'targets': targetIds,
        'target_length': targetLengths,
        'input_states_1': predStates1,
        'input_states_2': predStates2
    });

    const logits = decoderOutputs['outputs'];
    console.log('\nDecoder output (logits) shape:', logits.dims);
    console.log(`Expected: [1, seq_len, ${meta.vocab_size}]`);

    if (logits.dims[2] !== meta.vocab_size) {
        console.error(`\n⚠️ MISMATCH: Logits dimension ${logits.dims[2]} != vocab_size ${meta.vocab_size}`);
    }

    // Check what the model thinks about blank
    console.log('\nBlank Token Analysis:');
    console.log('---------------------');
    console.log(`Model blank_id: ${meta.blank_id}`);
    console.log(`Token at blank_id: "${meta.tokens[meta.blank_id]}" (should be empty string for RNN-T)`);
    console.log(`NeMo typically uses index ${meta.vocab_size - 1} as functional blank`);

    // Test actual predictions
    console.log('\n\nChecking character mapping:');
    console.log('---------------------------');
    for (const char of 'hello') {
        const id = meta.char_to_id[char];
        console.log(`'${char}' -> id ${id} -> '${meta.id_to_char[id]}'`);
    }
}

main().catch(console.error);