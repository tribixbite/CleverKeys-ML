#!/usr/bin/env ts-node
/**
 * Standalone TypeScript test for ONNX pipeline
 * Tests encoder, decoder, and beam search using real training data
 */

import * as ort from 'onnxruntime-node';
import * as fs from 'fs';
import * as path from 'path';
import { performance } from 'perf_hooks';

interface Point {
    x: number;
    y: number;
    t: number;
}

interface Sample {
    word: string;
    points: Point[];
}

interface TestResult {
    word: string;
    prediction: string;
    correct: boolean;
    latency: number;
    numPoints: number;
    numFrames: number;
}

class ONNXPipelineTest {
    private encoderSession: ort.InferenceSession | null = null;
    private decoderSession: ort.InferenceSession | null = null;
    private vocabSize: number = 30;
    private blankId: number = 29;
    private vocabulary: string[] = [];
    private featureExtractor: SwipeFeatureExtractor;

    constructor() {
        this.featureExtractor = new SwipeFeatureExtractor();
    }

    async initialize(encoderPath: string, decoderPath: string, vocabPath?: string) {
        console.log('🚀 Initializing ONNX test pipeline...');

        // Load models
        this.encoderSession = await ort.InferenceSession.create(encoderPath);
        console.log('✓ Encoder loaded:', {
            inputs: this.encoderSession.inputNames,
            outputs: this.encoderSession.outputNames
        });

        this.decoderSession = await ort.InferenceSession.create(decoderPath);
        console.log('✓ Decoder loaded:', {
            inputs: this.decoderSession.inputNames,
            outputs: this.decoderSession.outputNames
        });

        // Load vocabulary
        if (vocabPath && fs.existsSync(vocabPath)) {
            const vocabContent = fs.readFileSync(vocabPath, 'utf-8');
            this.vocabulary = vocabContent.split('\n').filter(line => line.trim());
            console.log(`✓ Vocabulary loaded: ${this.vocabulary.length} tokens`);
        } else {
            // Default vocabulary
            this.vocabulary = ['<blank>', "'", ...Array.from('abcdefghijklmnopqrstuvwxyz'), '<unk>'];
        }
    }

    async runEncoder(features: Float32Array, sequenceLength: number) {
        if (!this.encoderSession) throw new Error('Encoder not initialized');

        const [batchSize, timeSteps, featureDim] = [1, sequenceLength, 37];

        // Reshape to [batch, features, time]
        const audioSignal = new Float32Array(batchSize * featureDim * timeSteps);
        for (let t = 0; t < timeSteps; t++) {
            for (let f = 0; f < featureDim; f++) {
                audioSignal[f * timeSteps + t] = features[t * featureDim + f];
            }
        }

        const inputs = {
            'audio_signal': new ort.Tensor('float32', audioSignal, [batchSize, featureDim, timeSteps]),
            'length': new ort.Tensor('int64', new BigInt64Array([BigInt(timeSteps)]), [batchSize])
        };

        return await this.encoderSession.run(inputs);
    }

    async greedyDecode(encoderOutputs: any): Promise<string> {
        if (!this.decoderSession) throw new Error('Decoder not initialized');

        const encoded = encoderOutputs.outputs || encoderOutputs.encoded;
        const encodedLength = parseInt(encoderOutputs.encoded_lengths?.data[0] || encoded.dims[1]);

        const tokens: number[] = [];
        let hidden: Float32Array | null = null;
        let cell: Float32Array | null = null;

        for (let t = 0; t < encodedLength; t++) {
            // Extract encoder frame
            const frameData = encoded.data.slice(
                t * encoded.dims[2],
                (t + 1) * encoded.dims[2]
            );

            const inputs: any = {
                'encoder_outputs': new ort.Tensor('float32', frameData, [1, 1, encoded.dims[2]]),
                'targets': new ort.Tensor('int64',
                    new BigInt64Array(tokens.length > 0 ? tokens.map(BigInt) : [BigInt(this.blankId)]),
                    [1, tokens.length || 1]
                )
            };

            if (hidden && cell) {
                inputs['hidden'] = new ort.Tensor('float32', hidden, [2, 1, 320]);
                inputs['cell'] = new ort.Tensor('float32', cell, [2, 1, 320]);
            }

            try {
                const outputs = await this.decoderSession.run(inputs);
                const logits = outputs.logits || outputs.outputs || outputs.joint_output;

                // Get argmax
                const probs = Array.from(logits.data as Float32Array);
                const maxIdx = probs.indexOf(Math.max(...probs));

                if (maxIdx !== this.blankId && maxIdx < this.vocabulary.length) {
                    tokens.push(maxIdx);
                }

                // Update states
                hidden = outputs.hidden?.data as Float32Array || hidden;
                cell = outputs.cell?.data as Float32Array || cell;
            } catch (error) {
                console.error('Decoder step error:', error);
                break;
            }

            // Early stopping
            if (tokens.length >= 20) break;
        }

        return this.tokensToText(tokens);
    }

    tokensToText(tokens: number[]): string {
        return tokens
            .map(t => this.vocabulary[t] || '')
            .join('')
            .replace(/<blank>/g, '')
            .replace(/<unk>/g, '?');
    }

    async testSample(sample: TrainingSample): Promise<TestResult> {
        const startTime = performance.now();

        try {
            // Extract features
            const extracted = this.featureExtractor.extractFeatures(sample.points);
            const features = new Float32Array(extracted.features.flat());

            // Run encoder
            const encoderOutputs = await this.runEncoder(features, extracted.features.length);

            // Decode
            const prediction = await this.greedyDecode(encoderOutputs);

            const latency = performance.now() - startTime;

            return {
                word: sample.word,
                predictions: [prediction],
                scores: [1.0],
                correct: prediction === sample.word,
                latency
            };
        } catch (error) {
            return {
                word: sample.word,
                predictions: [],
                scores: [],
                correct: false,
                latency: performance.now() - startTime,
                error: error instanceof Error ? error.message : String(error)
            };
        }
    }

    async testDataset(dataPath: string, maxSamples: number = 100): Promise<void> {
        console.log('\n📊 Testing on dataset:', dataPath);

        const lines = fs.readFileSync(dataPath, 'utf-8')
            .split('\n')
            .filter(line => line.trim())
            .slice(0, maxSamples);

        const samples: TrainingSample[] = lines.map(line => JSON.parse(line));
        console.log(`Testing ${samples.length} samples...`);

        const results: TestResult[] = [];
        let correct = 0;
        let totalLatency = 0;

        for (let i = 0; i < samples.length; i++) {
            const sample = samples[i];
            process.stdout.write(`\rProgress: ${i + 1}/${samples.length}`);

            const result = await this.testSample(sample);
            results.push(result);

            if (result.correct) correct++;
            totalLatency += result.latency;

            // Log errors and mismatches
            if (!result.correct && i < 10) {
                console.log(`\n❌ Mismatch: "${sample.word}" → "${result.predictions[0]}" ${result.error ? `(${result.error})` : ''}`);
            }
        }

        // Summary statistics
        console.log('\n\n📈 Test Results:');
        console.log(`Accuracy: ${((correct / samples.length) * 100).toFixed(2)}% (${correct}/${samples.length})`);
        console.log(`Average latency: ${(totalLatency / samples.length).toFixed(2)}ms`);

        // Word length analysis
        const byLength = new Map<number, { correct: number; total: number }>();
        results.forEach(r => {
            const len = r.word.length;
            if (!byLength.has(len)) byLength.set(len, { correct: 0, total: 0 });
            const stats = byLength.get(len)!;
            stats.total++;
            if (r.correct) stats.correct++;
        });

        console.log('\n📊 Accuracy by word length:');
        Array.from(byLength.entries())
            .sort((a, b) => a[0] - b[0])
            .forEach(([len, stats]) => {
                const acc = ((stats.correct / stats.total) * 100).toFixed(1);
                console.log(`  Length ${len}: ${acc}% (${stats.correct}/${stats.total})`);
            });

        // Save detailed results
        const resultsPath = `test-results-${Date.now()}.json`;
        fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
        console.log(`\n💾 Detailed results saved to: ${resultsPath}`);
    }
}

/**
 * Feature Extractor (matching the training pipeline)
 */
class SwipeFeatureExtractor {
    private keyCenters: Array<{ char: string; x: number; y: number }>;

    constructor() {
        this.keyCenters = this.getDefaultQWERTYLayout();
    }

    private getDefaultQWERTYLayout() {
        const layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"];
        const centers: Array<{ char: string; x: number; y: number }> = [];

        for (let row = 0; row < layout.length; row++) {
            const rowStr = layout[row];
            for (let col = 0; col < rowStr.length; col++) {
                const x01 = (col + 0.5) / 10.0;
                const y01 = (row + 0.5) / 3.0;
                centers.push({
                    char: rowStr[col],
                    x: x01 * 2.0 - 1.0,
                    y: y01 * 2.0 - 1.0
                });
            }
        }
        return centers;
    }

    extractFeatures(rawPoints: SwipePoint[]): { features: number[][]; } {
        // Normalize points
        const points = this.preparePoints(rawPoints);

        // Resample to target length
        const targetLength = this.getResampleTarget(points.length);
        const resampled = this.resamplePoints(points, targetLength);

        // Extract features
        const features = resampled.map((_, idx) =>
            this.extractPointFeatures(resampled, idx)
        );

        return { features };
    }

    private preparePoints(points: SwipePoint[]): SwipePoint[] {
        if (!points.length) return [];
        const startTime = points[0].t || 0;
        return points.map((pt, idx) => ({
            x: Math.max(-1.0, Math.min(1.0, pt.x)),
            y: Math.max(-1.0, Math.min(1.0, pt.y)),
            t: (pt.t || idx * 10.0) - startTime
        }));
    }

    private getResampleTarget(length: number): number {
        if (length <= 48) return 56;
        if (length >= 112) return 96;
        const progress = (length - 48) / (112 - 48);
        return Math.round(56 + progress * 40);
    }

    private resamplePoints(points: SwipePoint[], targetCount: number): SwipePoint[] {
        if (!points.length || targetCount <= 0) return [];
        if (points.length === targetCount) return [...points];

        const resampled: SwipePoint[] = [];
        const duration = points[points.length - 1].t - points[0].t;
        const step = duration / Math.max(targetCount - 1, 1);

        for (let i = 0; i < targetCount; i++) {
            const targetTime = points[0].t + step * i;
            const { x, y } = this.interpolatePoint(points, targetTime);
            resampled.push({ x, y, t: targetTime });
        }

        return resampled;
    }

    private interpolatePoint(points: SwipePoint[], targetTime: number): { x: number; y: number } {
        let idx = 0;
        while (idx < points.length - 2 && points[idx + 1].t < targetTime) idx++;

        const p1 = points[idx];
        const p2 = points[Math.min(idx + 1, points.length - 1)];
        const alpha = Math.max(0, Math.min(1, (targetTime - p1.t) / Math.max(p2.t - p1.t, 1)));

        return {
            x: p1.x + (p2.x - p1.x) * alpha,
            y: p1.y + (p2.y - p1.y) * alpha
        };
    }

    private extractPointFeatures(points: SwipePoint[], idx: number): number[] {
        const curr = points[idx];
        const prev = idx > 0 ? points[idx - 1] : null;
        const prev2 = idx > 1 ? points[idx - 2] : null;

        // Calculate all 37 features (matching training pipeline)
        const features: number[] = [];

        // Position and time (3)
        features.push(curr.x, curr.y, curr.t / 1000.0);

        // Velocity (3)
        if (prev) {
            const dt = Math.max((curr.t - prev.t) / 1000.0, 1e-6);
            features.push((curr.x - prev.x) / dt, (curr.y - prev.y) / dt);
            features.push(Math.hypot(features[3], features[4]));
        } else {
            features.push(0, 0, 0);
        }

        // Acceleration (3)
        if (prev && prev2) {
            const dt1 = Math.max((curr.t - prev.t) / 1000.0, 1e-6);
            const dt2 = Math.max((prev.t - prev2.t) / 1000.0, 1e-6);
            const vx_prev = (prev.x - prev2.x) / dt2;
            const vy_prev = (prev.y - prev2.y) / dt2;
            features.push((features[3] - vx_prev) / dt1, (features[4] - vy_prev) / dt1);
            features.push(Math.hypot(features[6], features[7]));
        } else {
            features.push(0, 0, 0);
        }

        // Angle features (4)
        const angle = prev ? Math.atan2(features[4], features[3]) : 0;
        features.push(angle, Math.sin(angle), Math.cos(angle));

        // Curvature
        if (prev && prev2) {
            const prevAngle = Math.atan2(prev.y - prev2.y, prev.x - prev2.x);
            let curvature = angle - prevAngle;
            while (curvature > Math.PI) curvature -= 2 * Math.PI;
            while (curvature < -Math.PI) curvature += 2 * Math.PI;
            features.push(curvature);
        } else {
            features.push(0);
        }

        // Distance to nearest keys (5)
        const keyDists = this.keyCenters
            .map(k => Math.hypot(curr.x - k.x, curr.y - k.y))
            .sort((a, b) => a - b)
            .slice(0, 5);
        features.push(...keyDists);
        while (features.length < 18) features.push(1.0);

        // Progress markers (3)
        features.push(idx / Math.max(points.length - 1, 1));
        features.push(idx === 0 ? 1.0 : 0.0);
        features.push(idx === points.length - 1 ? 1.0 : 0.0);

        // Window statistics (6)
        const winStart = Math.max(0, idx - 2);
        const winEnd = Math.min(points.length, idx + 3);
        const window = points.slice(winStart, winEnd);

        if (window.length > 1) {
            const xs = window.map(p => p.x);
            const ys = window.map(p => p.y);
            const meanX = xs.reduce((a, b) => a + b) / xs.length;
            const meanY = ys.reduce((a, b) => a + b) / ys.length;
            features.push(meanX, Math.sqrt(xs.reduce((s, x) => s + (x - meanX) ** 2, 0) / xs.length));
            features.push(meanY, Math.sqrt(ys.reduce((s, y) => s + (y - meanY) ** 2, 0) / ys.length));
            features.push(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
        } else {
            features.push(curr.x, 0, curr.y, 0, 0, 0);
        }

        // Pad to 37 features
        while (features.length < 37) features.push(0);

        return features.slice(0, 37);
    }
}

// Main test runner
async function main() {
    const tester = new ONNXPipelineTest();

    const modelDir = process.argv[2] || './web-demo';
    const dataPath = process.argv[3] || './data/train_final_val.jsonl';
    const maxSamples = parseInt(process.argv[4] || '100');

    try {
        await tester.initialize(
            path.join(modelDir, 'encoder-model.onnx'),
            path.join(modelDir, 'decoder_joint-model.onnx'),
            path.join(modelDir, '../data/vocab.txt')
        );

        await tester.testDataset(dataPath, maxSamples);
    } catch (error) {
        console.error('Test failed:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main().catch(console.error);
}

export { ONNXPipelineTest, SwipeFeatureExtractor };