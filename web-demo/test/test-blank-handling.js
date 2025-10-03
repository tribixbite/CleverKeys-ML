#!/usr/bin/env node
/**
 * Test how the decoder handles blank tokens and vocabulary mapping
 */

const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');
const RNNTDecoder = require('../js/onnx-rnnt-decoder-fixed.js');

async function main() {
    const baseDir = path.resolve(__dirname, '..');
    const modelDir = path.join(baseDir, 'models', 'best_latest');

    // Load runtime meta
    const metaPath = path.join(modelDir, 'runtime_meta.json');
    const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));

    console.log('Blank Token Analysis:');
    console.log('---------------------');
    console.log(`blank_id: ${meta.blank_id}`);
    console.log(`tokens[${meta.blank_id}]: "${meta.tokens[meta.blank_id]}"`);
    console.log();

    // Initialize decoder
    const decoder = new RNNTDecoder();
    decoder.verbose = true;

    await decoder.initialize(
        ort,
        path.join(modelDir, 'encoder.onnx'),
        path.join(modelDir, 'decoder_joint.onnx'),
        metaPath
    );

    // Check how toChar handles each token
    console.log('Token to Character Mapping:');
    console.log('---------------------------');
    for (let i = 0; i < meta.vocab_size; i++) {
        const token = meta.tokens[i];
        const mapped = meta.id_to_char[String(i)];
        console.log(`${i}: token="${token}" id_to_char="${mapped}" (${token === '' ? 'EMPTY STRING!' : 'ok'})`);
    }
    console.log();

    // Test with simple features that should produce "hello"
    const numFrames = 48;
    const features = new Float32Array(numFrames * 37);

    // Initialize with small values
    for (let t = 0; t < numFrames; t++) {
        for (let f = 0; f < 37; f++) {
            features[t * 37 + f] = 0.01 * Math.random();
        }
    }

    console.log('Testing greedy decode with dummy features...');
    try {
        const results = await decoder.greedyDecode(features);
        console.log('Greedy decode result:', results);
    } catch (error) {
        console.error('Greedy decode error:', error);
    }

    // Check lexicon loading
    console.log('\nTrying to load lexicon...');
    try {
        await decoder.loadLexicon(
            path.join(baseDir, 'words.txt'),
            path.join(baseDir, 'word_frequencies_aligned.json')
        );
        console.log('Lexicon loaded successfully');

        // Check charToId mapping
        if (decoder.lexicon && decoder.lexicon.charToId) {
            console.log('\nLexicon charToId mapping:');
            console.log('-------------------------');
            const chars = Object.keys(decoder.lexicon.charToId).slice(0, 10);
            for (const ch of chars) {
                console.log(`"${ch}" -> ${decoder.lexicon.charToId[ch]}`);
            }
            console.log(`Has empty string mapping: ${decoder.lexicon.charToId.hasOwnProperty('')}`);
        }
    } catch (error) {
        console.error('Lexicon load error:', error);
    }
}

main().catch(console.error);