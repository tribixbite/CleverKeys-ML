#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const readline = require('readline');
const FeatureExtractor = require('./web-demo/js/feature-extractor-corrected.js');

async function main() {
    // Read line 431621 (hello) efficiently
    const fileStream = fs.createReadStream('data/train_final_train.jsonl');
    const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

    let lineNum = 0;
    let item = null;
    for await (const line of rl) {
        lineNum++;
        if (lineNum === 431621) {
            item = JSON.parse(line);
            break;
        }
    }

    console.log(`Testing with word: "${item.word}"`);
    console.log(`Number of points: ${item.points.length}`);

    // Show first few raw points
    console.log("\nFirst 3 raw points from dataset:");
    for (let i = 0; i < Math.min(3, item.points.length); i++) {
        const p = item.points[i];
        console.log(`  Point ${i}: x=${p.x.toFixed(4)}, y=${p.y.toFixed(4)}, t=${p.t}`);
    }

    // Initialize feature extractor
    const featureExtractor = new FeatureExtractor();
    featureExtractor.featureDim = 37;  // Match training
    featureExtractor.keyCenters = JSON.parse(fs.readFileSync('web-demo/js/key-centers.json', 'utf-8'));

    // Extract features
    const featureData = featureExtractor.process(item.points);
    console.log(`\nFeature shape: [${featureData.numFrames}, ${featureData.featureMatrix[0].length}]`);

    // Print first few frames for comparison
    console.log("\nFirst 3 frames of JavaScript features:");
    for (let i = 0; i < Math.min(3, featureData.featureMatrix.length); i++) {
        const frame = featureData.featureMatrix[i];
        console.log(`Frame ${i}: [${frame.slice(0, 5).map(f => f.toFixed(6)).join(', ')}, ...]`);
    }

    // Compare with expected Python features
    const expectedFirstFrame = [0.165550, 0.049296, 0.000000, 0.000000, 0.000000];
    const actualFirstFrame = featureData.featureMatrix[0].slice(0, 5);

    console.log("\nComparison with Python features:");
    console.log(`Expected first 5 features: [${expectedFirstFrame.map(f => f.toFixed(6)).join(', ')}]`);
    console.log(`Actual first 5 features:   [${actualFirstFrame.map(f => f.toFixed(6)).join(', ')}]`);

    // Check if they match (within tolerance)
    const tolerance = 0.0001;
    let matches = true;
    for (let i = 0; i < 5; i++) {
        if (Math.abs(expectedFirstFrame[i] - actualFirstFrame[i]) > tolerance) {
            matches = false;
            console.log(`  ❌ Feature ${i}: expected ${expectedFirstFrame[i]}, got ${actualFirstFrame[i]}`);
        }
    }

    if (matches) {
        console.log("✅ Features match Python implementation!");
    } else {
        console.log("❌ Features DO NOT match Python implementation");
    }
}

main().catch(console.error);