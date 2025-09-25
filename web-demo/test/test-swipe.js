const ort = require('onnxruntime-node');
const fs = require('fs');
const path = require('path');

// This is the corrected feature extractor logic, adapted for Node.js.
// It is self-contained here to act as a reliable reference.
class SwipeFeatureExtractorForTest {
    constructor() {
        this.featureDim = 37;
        this.keyCenters = this.buildKeyCenters();
    }

    buildKeyCenters() {
        const layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"];
        const centers = [];
        for (let row = 0; row < layout.length; row++) {
            const rowStr = layout[row];
            for (let col = 0; col < rowStr.length; col++) {
                const char = rowStr[col];
                const x01 = (col + 0.5) / 10.0;
                const y01 = (row + 0.5) / 3.0;
                const x = x01 * 2.0 - 1.0;
                const y = y01 * 2.0 - 1.0;
                centers.push({ char, x, y });
            }
        }
        return centers;
    }

    // CORRECTED: This function now correctly handles coordinates that are already in [-1, 1]
    normalizePoints(points) {
        if (!points || points.length === 0) {
            return [];
        }
        const startTime = points[0].t || 0;
        return points.map((pt, idx) => {
            // The data is already in [-1, 1], so we just clamp it.
            const centeredX = Math.max(-1.0, Math.min(1.0, pt.x || 0.0));
            const centeredY = Math.max(-1.0, Math.min(1.0, pt.y || 0.0));
            const t = (pt.t || idx * 10.0) - startTime;
            return { x: centeredX, y: centeredY, t };
        });
    }

    getResampleTarget(length) {
        const shortTarget = 56, longTarget = 96, shortThresh = 48, longThresh = 112;
        if (length <= shortThresh) return shortTarget;
        if (length >= longThresh) return longTarget;
        const progress = (length - shortThresh) / (longThresh - shortThresh);
        return Math.floor(shortTarget + progress * (longTarget - shortTarget));
    }

    resamplePoints(points, targetCount) {
        if (targetCount <= 0 || points.length === 0) return [];
        if (points.length === targetCount) return [...points];
        const resampled = [];
        const duration = points.length > 1 ? points[points.length - 1].t - points[0].t : 0;
        const step = duration / Math.max(targetCount - 1, 1);
        let srcIdx = 0;
        for (let i = 0; i < targetCount; i++) {
            const targetTime = (points[0].t || 0) + step * i;
            while (srcIdx < points.length - 2 && points[srcIdx + 1].t < targetTime) {
                srcIdx++;
            }
            const p1 = points[srcIdx];
            const p2 = points[Math.min(srcIdx + 1, points.length - 1)];
            const span = Math.max(p2.t - p1.t, 1e-6);
            const alpha = Math.max(0, Math.min(1, (targetTime - p1.t) / span));
            resampled.push({
                x: p1.x + (p2.x - p1.x) * alpha,
                y: p1.y + (p2.y - p1.y) * alpha,
                t: targetTime
            });
        }
        return resampled;
    }

    extractPointFeatures(points, idx) {
        const total = points.length;
        const curr = points[idx];
        const prev = idx > 0 ? points[idx - 1] : null;
        const prev2 = idx > 1 ? points[idx - 2] : null;

        const x = curr.x, y = curr.y, t_seconds = curr.t / 1000.0;
        let vx = 0, vy = 0, speed = 0;
        if (prev) {
            const dt = Math.max((curr.t - prev.t) / 1000.0, 1e-6);
            vx = (x - prev.x) / dt;
            vy = (y - prev.y) / dt;
            speed = Math.sqrt(vx * vx + vy * vy);
        }

        let ax = 0, ay = 0, acc = 0;
        if (prev && prev2) {
            const dt1 = Math.max((curr.t - prev.t) / 1000.0, 1e-6);
            const dt2 = Math.max((prev.t - prev2.t) / 1000.0, 1e-6);
            const vx_prev = (prev.x - prev2.x) / dt2;
            const vy_prev = (prev.y - prev2.y) / dt2;
            ax = (vx - vx_prev) / dt1;
            ay = (vy - vy_prev) / dt1;
            acc = Math.sqrt(ax * ax + ay * ay);
        }

        const angle = prev ? Math.atan2(vy, vx) : 0.0;
        let curvature = 0;
        if (prev && prev2) {
            const prev_angle = Math.atan2(prev.y - prev2.y, prev.x - prev2.x);
            curvature = angle - prev_angle;
            while (curvature > Math.PI) curvature -= 2 * Math.PI;
            while (curvature < -Math.PI) curvature += 2 * Math.PI;
        }

        const keyDistances = this.keyCenters
            .map(key => Math.sqrt((x - key.x) ** 2 + (y - key.y) ** 2))
            .sort((a, b) => a - b)
            .slice(0, 5);
        while (keyDistances.length < 5) keyDistances.push(1.0);

        const progress = idx / Math.max(total - 1, 1);
        const is_start = idx === 0 ? 1.0 : 0.0;
        const is_end = idx === total - 1 ? 1.0 : 0.0;

        const winStart = Math.max(0, idx - 2);
        const winEnd = Math.min(total, idx + 3);
        const winPts = points.slice(winStart, winEnd);
        let win_mean_x = x, win_std_x = 0, win_mean_y = y, win_std_y = 0, win_range_x = 0, win_range_y = 0;
        if (winPts.length > 1) {
            const xs = winPts.map(p => p.x);
            const ys = winPts.map(p => p.y);
            win_mean_x = xs.reduce((a, b) => a + b, 0) / xs.length;
            win_mean_y = ys.reduce((a, b) => a + b, 0) / ys.length;
            win_std_x = Math.sqrt(xs.reduce((sum, xi) => sum + (xi - win_mean_x) ** 2, 0) / xs.length);
            win_std_y = Math.sqrt(ys.reduce((sum, yi) => sum + (yi - win_mean_y) ** 2, 0) / ys.length);
            win_range_x = Math.max(...xs) - Math.min(...xs);
            win_range_y = Math.max(...ys) - Math.min(...ys);
        }

        const features = [
            x, y, t_seconds, vx, vy, speed, ax, ay, acc, angle, Math.sin(angle), Math.cos(angle), curvature,
            ...keyDistances, progress, is_start, is_end,
            win_mean_x, win_std_x, win_mean_y, win_std_y, win_range_x, win_range_y
        ];
        while (features.length < this.featureDim) features.push(0.0);
        return features.slice(0, this.featureDim);
    }

    process(rawPoints) {
        const normalizedPoints = this.normalizePoints(rawPoints);
        const targetLength = this.getResampleTarget(normalizedPoints.length);
        const resampledPoints = this.resamplePoints(normalizedPoints, targetLength);
        const featureMatrix = resampledPoints.map((_, idx) => this.extractPointFeatures(resampledPoints, idx));
        const numFrames = featureMatrix.length;
        const flatFeatures = new Float32Array(numFrames * this.featureDim);
        for (let t = 0; t < numFrames; t++) {
            flatFeatures.set(featureMatrix[t], t * this.featureDim);
        }
        return { features: flatFeatures, numFrames };
    }
}

async function main() {
    try {
        console.log('Starting test...');
        const modelDir = path.join(__dirname, '../models/rnnt_new');
        const encoderPath = path.join(modelDir, 'encoder.onnx');
        const decoderJointPath = path.join(modelDir, 'decoder_joint.onnx');
        const metaPath = path.join(modelDir, 'runtime_meta.json');

        // 1. Load Models and Meta
        const [encoder, decoderJoint, meta] = await Promise.all([
            ort.InferenceSession.create(encoderPath),
            ort.InferenceSession.create(decoderJointPath),
            fs.promises.readFile(metaPath, 'utf-8').then(JSON.parse)
        ]);
        console.log('Models and metadata loaded.');

        // 2. Prepare Test Data
        const testData = {"word": "raped", "points": [{"x": 0.377898441745, "y": 0.309550308126, "t": 0}, {"x": 0.377898441745, "y": 0.309550308126, "t": 20}, {"x": 0.374141490448, "y": 0.313618343637, "t": 37}, {"x": 0.361931347031, "y": 0.32582209188, "t": 53}, {"x": 0.307455243013, "y": 0.360400393726, "t": 70}, {"x": 0.252979159676, "y": 0.407182443815, "t": 86}, {"x": 0.193806876878, "y": 0.427522621371, "t": 103}, {"x": 0.161872635748, "y": 0.439726727905, "t": 120}, {"x": 0.156237157101, "y": 0.443794763416, "t": 136}, {"x": 0.160933377243, "y": 0.447862798928, "t": 153}, {"x": 0.184414436593, "y": 0.451930834439, "t": 169}, {"x": 0.237951271766, "y": 0.453964493904, "t": 186}, {"x": 0.311212235672, "y": 0.47430467146, "t": 203}, {"x": 0.39574407415, "y": 0.500746902284, "t": 219}, {"x": 0.481215160792, "y": 0.525155115351, "t": 236}, {"x": 0.55541531082, "y": 0.523120739305, "t": 252}, {"x": 0.617405358794, "y": 0.514984668282, "t": 269}, {"x": 0.664367477494, "y": 0.510916632771, "t": 285}, {"x": 0.707572644898, "y": 0.504814937795, "t": 303}, {"x": 0.750777895025, "y": 0.490577171796, "t": 319}, {"x": 0.784590580782, "y": 0.468202618193, "t": 335}, {"x": 0.808071702175, "y": 0.443794763416, "t": 352}, {"x": 0.82873499635, "y": 0.425488603616, "t": 369}, {"x": 0.844702132426, "y": 0.413284855373, "t": 385}, {"x": 0.851276869579, "y": 0.411250479326, "t": 402}, {"x": 0.851276869579, "y": 0.409216819861, "t": 418}, {"x": 0.846580628755, "y": 0.403114408304, "t": 435}, {"x": 0.829674244515, "y": 0.380740571282, "t": 452}, {"x": 0.778955091794, "y": 0.335992180658, "t": 468}, {"x": 0.695362501481, "y": 0.283108077302, "t": 485}, {"x": 0.593924278763, "y": 0.252598169259, "t": 502}, {"x": 0.515027887911, "y": 0.252598169259, "t": 519}, {"x": 0.467126521046, "y": 0.272938346815, "t": 535}, {"x": 0.430496090795, "y": 0.299380219348, "t": 551}, {"x": 0.396683322314, "y": 0.32175441466, "t": 568}, {"x": 0.367566794657, "y": 0.340060216169, "t": 585}, {"x": 0.345024962821, "y": 0.356278211345, "t": 601}, {"x": 0.331686319254, "y": 0.368481959588, "t": 618}, {"x": 0.3289048159, "y": 0.380685707831, "t": 635}, {"x": 0.3289048159, "y": 0.39492347383, "t": 651}, {"x": 0.3289048159, "y": 0.41322963358, "t": 668}, {"x": 0.3289048159, "y": 0.435603470597, "t": 685}, {"x": 0.3289048159, "y": 0.46204534313, "t": 701}, {"x": 0.3289048159, "y": 0.490521233418, "t": 718}, {"x": 0.3289048159, "y": 0.525099535254, "t": 734}, {"x": 0.3289048159, "y": 0.561729965505, "t": 751}, {"x": 0.33078330868, "y": 0.596308267341, "t": 768}, {"x": 0.335379549506, "y": 0.632938697592, "t": 784}, {"x": 0.338097744945, "y": 0.657346910659, "t": 801}]};
        const featurizer = new SwipeFeatureExtractorForTest();

        // 3. Feature Extraction
        console.log('--- DEBUG: Feature Extraction ---');
        const normalizedPoints = featurizer.normalizePoints(testData.points);
        const targetLength = featurizer.getResampleTarget(normalizedPoints.length);
        const resampledPoints = featurizer.resamplePoints(normalizedPoints, targetLength);
        const { features, numFrames } = featurizer.process(testData.points);

        console.log('Normalized Points (first 5):', JSON.stringify(normalizedPoints.slice(0, 5), null, 2));
        console.log('Target Length:', targetLength);
        console.log('Resampled Points (first 5):', JSON.stringify(resampledPoints.slice(0, 5), null, 2));
        
        // Log the first 2 full feature vectors for comparison
        const featureMatrixForLog = [];
        for (let i = 0; i < 2; i++) {
            featureMatrixForLog.push(Array.from(features.slice(i * featurizer.featureDim, (i + 1) * featurizer.featureDim)));
        }
        console.log('Feature Matrix (first 2 rows):', JSON.stringify(featureMatrixForLog, null, 2));
        console.log('-----------------------------------');


        console.log(`Features extracted. Shape: [${numFrames}, ${featurizer.featureDim}]`);

        // 4. Encoder Pass
        // The model expects [Batch, Features, Time], but our features are [Time, Features].
        // We need to transpose it.
        const transposedFeatures = new Float32Array(1 * featurizer.featureDim * numFrames);
        for (let t = 0; t < numFrames; t++) {
            for (let f = 0; f < featurizer.featureDim; f++) {
                // Source: features[t * featureDim + f]
                // Destination: transposedFeatures[f * numFrames + t]
                transposedFeatures[f * numFrames + t] = features[t * featurizer.featureDim + f];
            }
        }

        const featuresTensor = new ort.Tensor('float32', transposedFeatures, [1, featurizer.featureDim, numFrames]);
        const lengthsTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(numFrames)]), [1]);
        const encoderFeeds = { 'audio_signal': featuresTensor, 'length': lengthsTensor };
        const encoderResults = await encoder.run(encoderFeeds);
        const encoded = encoderResults.outputs;
        const encodedLength = Number(encoderResults.encoded_lengths.data[0]);
        console.log(`Encoder pass complete. Output shape: ${encoded.dims}, Length: ${encodedLength}`);

        // 5. Greedy Decode with Decoder-Joint Model
        let decodedTokens = [];
        let decoderState = {
            h: new ort.Tensor('float32', new Float32Array(2 * 1 * 320).fill(0), [2, 1, 320]),
            c: new ort.Tensor('float32', new Float32Array(2 * 1 * 320).fill(0), [2, 1, 320]),
        };
        let lastToken = meta.blank_id;

        for (let t = 0; t < encodedLength; t++) {
            const frameData = new Float32Array(256);
            for (let f = 0; f < 256; f++) {
                frameData[f] = encoded.data[f * encodedLength + t];
            }
            const encoderFrame = new ort.Tensor('float32', frameData, [1, 256, 1]);
            const decoderInput = new ort.Tensor('int32', Int32Array.from([lastToken]), [1, 1]);
            const targetLength = new ort.Tensor('int32', Int32Array.from([1]), [1]);


            console.log(`\n--- Time Step ${t} ---`);
            console.log('Encoder Frame (first 10):', encoderFrame.data.slice(0, 10));
            console.log('Decoder Input Token:', lastToken);

            const jointFeeds = {
                'encoder_outputs': encoderFrame,
                'targets': decoderInput,
                'target_length': targetLength,
                'input_states_1': decoderState.h,
                'input_states_2': decoderState.c,
            };

            const jointResults = await decoderJoint.run(jointFeeds);
            console.log('Joint Results:', jointResults);
            const logits = jointResults.outputs.data;
            console.log('Logits (first 10):', logits.slice(0, 10));
            
            // Argmax
            let maxVal = -Infinity;
            let predictedToken = -1;
            for (let i = 0; i < meta.vocab_size; i++) {
                if (logits[i] > maxVal) {
                    maxVal = logits[i];
                    predictedToken = i;
                }
            }
            console.log(`Predicted Token: ${predictedToken} (${meta.id_to_char[predictedToken] || ''})`);

            // The state is ALWAYS updated
            decoderState = {
                h: jointResults.output_states_1,
                c: jointResults.output_states_2,
            };

            if (predictedToken !== meta.blank_id && predictedToken !== 0) {
                decodedTokens.push(predictedToken);
                lastToken = predictedToken;
            }
        }

        const decodedText = decodedTokens.map(token => meta.id_to_char[token]).join('');
        
        // 6. Verify Result
        console.log('-------------------');
        console.log(`Ground Truth: ${testData.word}`);
        console.log(`Predicted Text: ${decodedText}`);
        console.log(`Test ${decodedText === testData.word ? 'PASSED' : 'FAILED'}`);
        console.log('-------------------');

    } catch (e) {
        console.error('Test failed with error:', e);
    }
}

main();
