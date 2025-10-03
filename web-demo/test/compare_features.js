#!/usr/bin/env node
/*
 * Compare JS feature extraction for debugging
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

async function main() {
    // Get hello data (line 431621)
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

    // Process using JS feature extractor
    const FeatureExtractor = require('../js/feature-extractor-corrected.js');
    const featureExtractor = new FeatureExtractor();
    const featureData = featureExtractor.process(points);

    console.log(`\nJS features shape: [${featureData.numFrames}, 37]`);

    console.log('First frame (first 10 features):');
    for (let i = 0; i < 10; i++) {
        console.log(`  Feature ${i}: ${featureData.features[i].toFixed(6)}`);
    }

    console.log('\nLast frame (first 10 features):');
    const lastFrameStart = (featureData.numFrames - 1) * 37;
    for (let i = 0; i < 10; i++) {
        console.log(`  Feature ${i}: ${featureData.features[lastFrameStart + i].toFixed(6)}`);
    }

    // Save features for comparison
    const featuresArray = Array.from(featureData.features);
    fs.writeFileSync('js_features.json', JSON.stringify({
        shape: [featureData.numFrames, 37],
        data: featuresArray
    }, null, 2));
    console.log('\nSaved features to js_features.json');
}

main().catch(console.error);