#!/usr/bin/env node
/*
 * CLI test runner for the modular RNN-T web demo.
 * This script loads the exact same modules used by the browser demo
 * to run a command-line prediction test.
 *
 * Usage:
 *  node web-demo/test/test-beam-best.js --word hello --debug
 */

const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');
const FeatureExtractor = require('../js/feature-extractor-corrected.js');
const RNNTDecoder = require('../js/onnx-rnnt-decoder-fixed.js');

// --- Argument Parser ---
function parseArgs() {
    const cfg = {};
    const args = process.argv.slice(2);
    for (let i = 0; i < args.length; i++) {
        const k = args[i];
        const v = args[i + 1];
        if (k && k.startsWith('--')) {
            const key = k.slice(2);
            if (v && !v.startsWith('--')) {
                cfg[key] = v;
                i++;
            } else {
                cfg[key] = true;
            }
        }
    }
    return cfg;
}

// --- Main Test Function ---
async function main() {
    const cfg = parseArgs();
    const debug = !!cfg.debug;
    const wordToTest = (cfg.word || 'hello').toLowerCase();
    const lineNum = cfg.line ? parseInt(cfg.line, 10) : null;

    console.log(`🚀 Initializing CLI test for "${lineNum ? `line ${lineNum}` : wordToTest}"`);

    // --- 1. Set up paths ---
    const baseDir = path.resolve(__dirname, '..');
    const modelDir = path.join(baseDir, 'models', 'best_latest');
    const encoderPath = path.join(modelDir, 'encoder.onnx');
    const decoderJointPath = path.join(modelDir, 'decoder_joint.onnx');
    const metaPath = path.join(modelDir, 'runtime_meta.json');
    const keyCentersPath = path.join(baseDir, 'js', 'key-centers.json');
    const wordsPath = path.join(baseDir, 'words.txt');
    const freqsPath = path.join(baseDir, 'word_frequencies_aligned.json');

    // --- 2. Instantiate and configure modules ---
    const featureExtractor = new FeatureExtractor();
    const rnntDecoder = new RNNTDecoder();

    if (debug) {
        featureExtractor.verbose = true;
        rnntDecoder.verbose = true;
    }

    // Load shared keyboard geometry and inject it
    const keyCenters = JSON.parse(fs.readFileSync(keyCentersPath, 'utf-8'));
    featureExtractor.keyCenters = keyCenters;
    rnntDecoder.keyCenters = keyCenters;

    // --- 3. Initialize decoder (loads models and metadata) ---
    await rnntDecoder.initialize(ort, encoderPath, decoderJointPath, metaPath);
    await rnntDecoder.loadLexicon(wordsPath, freqsPath);

    console.log('✅ Modules initialized successfully.');

    // --- 4. Get swipe path ---
    let swipePath;
    let testWord;

    if (lineNum) {
        console.log(`📝 Reading line ${lineNum} from training data...`);
        const readline = require('readline');
        const fileStream = fs.createReadStream(path.join(baseDir, '..', 'data', 'train_final_train.jsonl'));
        const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });
        
        let currentLine = 0;
        let lineFound = false;
        for await (const line of rl) {
            currentLine++;
            if (currentLine === lineNum) {
                const lineData = JSON.parse(line);
                swipePath = lineData.points;
                testWord = lineData.word;
                console.log(`   Found word "${testWord}" with ${swipePath.length} points.`);
                lineFound = true;
                break;
            }
        }
        if (!lineFound) {
            console.error(`Error: Line number ${lineNum} not found.`);
            return;
        }
    } else {
        // Fallback to synthetic path generation if no line number is given
        console.log(`📝 Generating synthetic test path for "${wordToTest}"...`);
        const makeTestPath = (word, centers) => {
            const pts = [];
            let t = 0;
            for (const ch of word) {
                const kc = centers.find(k => k.char === ch);
                if (!kc) {
                    console.error(`Failed to find key center for '${ch}'`);
                    return [];
                }
                // Transform from [-1, 1] key centers to [0, 1] dataset format
                pts.push({ x: (kc.x + 1) / 2, y: (kc.y + 1) / 2, t });
                t += 40; // ms
            }
            return pts;
        };
        swipePath = makeTestPath(wordToTest, keyCenters);
        testWord = wordToTest;
    }

    if (!swipePath || swipePath.length === 0) {
        console.error('Failed to get a swipe path.');
        return;
    }

    // --- 5. Run inference ---
    console.log('🧠 Running inference...');
    const startTime = performance.now();

    // a) Extract features
    const featureData = featureExtractor.process(swipePath);

    if (debug) {
        console.log("--- JavaScript Features ---");
        console.log(`// Shape: [${featureData.numFrames}, ${featureData.featureMatrix[0].length}]`);
        console.log("const jsFeatures = [");
        for (let i = 0; i < featureData.featureMatrix.length; i++) {
            const row = featureData.featureMatrix[i];
            console.log(`    [${row.map(x => x.toFixed(6)).join(', ')}]${i < featureData.featureMatrix.length - 1 ? ',' : ''}`);
        }
        console.log("];");
    }


    // b) Run beam search
    const results = await rnntDecoder.beamSearch(featureData, {
        beamSize: 32,
        topK: 12,
        symbolsPerStep: 3,
        maxSymbols: 24,
        lengthPenalty: 0.6
    });

    const processingTime = performance.now() - startTime;
    console.log(`✅ Inference complete in ${processingTime.toFixed(1)}ms`);

    // --- 6. Display results ---
    const topPrediction = results[0] || { text: 'N/A', score: -Infinity };

    console.log('\n---------- RESULTS ----------');
    console.log(`🥇 Top Prediction: "${topPrediction.text}" (Score: ${topPrediction.score.toFixed(3)})`);
    console.log('----------\n');

    if (topPrediction.text === testWord) {
        console.log(`\x1b[32m✓ SUCCESS: Prediction matches expected output ("${testWord}").\x1b[0m`);
    } else {
        console.log(`\x1b[31m✗ FAILURE: Prediction does not match. Expected "${testWord}".\x1b[0m`);
    }

    console.log('\nTop 10 Hypotheses:');
    results.forEach((res, i) => {
        console.log(`  ${i + 1}. "${res.text}" (Score: ${res.score.toFixed(3)})`);
    });
}


main().catch(e => {
    console.error(e);
    process.exit(1);
});