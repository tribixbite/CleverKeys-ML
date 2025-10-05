/**
 * CTC Model Decoder
 * Handles inference for CTC-based swipe recognition models
 */

class CTCDecoder {
    constructor() {
        this.encoder = null;
        this.decoder = null;
        this.tokenizer = null;
        this.initialized = false;
    }

    async initialize(encoderPath, decoderPath, tokenizerPath) {
        try {
            console.log('Loading CTC models...');

            // Load models
            this.encoder = await ort.InferenceSession.create(encoderPath);
            console.log('CTC Encoder loaded');

            this.decoder = await ort.InferenceSession.create(decoderPath);
            console.log('CTC Decoder loaded');

            // Load tokenizer config
            const response = await fetch(tokenizerPath);
            this.tokenizer = await response.json();
            console.log('Tokenizer loaded:', {
                vocabSize: this.tokenizer.vocab_size,
                specialTokens: this.tokenizer.special_tokens
            });

            this.initialized = true;
            return true;
        } catch (error) {
            console.error('Failed to initialize CTC models:', error);
            throw error;
        }
    }

    /**
     * Resample points to target count
     */
    resamplePoints(points, targetCount) {
        if (points.length === 0 || targetCount <= 0) return [];
        if (points.length === targetCount) return [...points];

        const resampled = [];
        const step = (points.length - 1) / (targetCount - 1);

        for (let i = 0; i < targetCount; i++) {
            const sourceIdx = i * step;
            const idx1 = Math.floor(sourceIdx);
            const idx2 = Math.min(idx1 + 1, points.length - 1);
            const alpha = sourceIdx - idx1;

            const p1 = points[idx1];
            const p2 = points[idx2];

            resampled.push({
                x: p1.x + (p2.x - p1.x) * alpha,
                y: p1.y + (p2.y - p1.y) * alpha,
                t: p1.t + (p2.t - p1.t) * alpha
            });
        }

        return resampled;
    }

    /**
     * Extract CTC-specific features from points
     */
    extractCTCFeatures(points) {
        // CTC model expects exactly 150 points
        const targetPoints = 150;

        // Resample points to exactly 150
        const resampledPoints = this.resamplePoints(points, targetPoints);
        const numPoints = resampledPoints.length;
        const trajectoryFeatures = new Float32Array(numPoints * 6);

        for (let i = 0; i < numPoints; i++) {
            const curr = resampledPoints[i];
            const prev = i > 0 ? resampledPoints[i - 1] : curr;

            // Calculate velocity - matching Python implementation
            const dt = i > 0 ? Math.max((curr.t - prev.t) / 1000.0, 0.001) : 0.001;
            const vx = i > 0 ? (curr.x - prev.x) / dt : 0;
            const vy = i > 0 ? (curr.y - prev.y) / dt : 0;

            // 6D features: [x, y, vx, vy, t, pressure] - matching test_ctc_python.py
            trajectoryFeatures[i * 6 + 0] = curr.x;      // x position (already in [-1,1])
            trajectoryFeatures[i * 6 + 1] = curr.y;      // y position (already in [-1,1])
            trajectoryFeatures[i * 6 + 2] = vx;          // x velocity
            trajectoryFeatures[i * 6 + 3] = vy;          // y velocity
            trajectoryFeatures[i * 6 + 4] = curr.t / 1000.0; // time in seconds
            trajectoryFeatures[i * 6 + 5] = 1.0;         // pressure (default to 1)
        }

        // Find nearest keys for each point
        const nearestKeys = new BigInt64Array(numPoints);
        const keyboardLayout = this.getKeyboardLayout();

        for (let i = 0; i < numPoints; i++) {
            const point = resampledPoints[i];
            let minDist = Infinity;
            let nearestKey = 4; // Default to 'a'

            for (const [char, pos] of Object.entries(keyboardLayout)) {
                const dist = Math.sqrt((point.x - pos.x) ** 2 + (point.y - pos.y) ** 2);
                if (dist < minDist) {
                    minDist = dist;
                    nearestKey = this.tokenizer.char_to_idx[char] || 4;
                }
            }

            nearestKeys[i] = BigInt(nearestKey);
        }

        // Create source mask (boolean array for valid positions)
        const srcMask = new Uint8Array(numPoints).fill(1);

        return {
            trajectoryFeatures,
            nearestKeys,
            srcMask,
            sequenceLength: numPoints
        };
    }

    /**
     * Get keyboard layout for nearest key calculation
     */
    getKeyboardLayout() {
        const layout = {};
        const rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"];

        for (let row = 0; row < rows.length; row++) {
            const rowStr = rows[row];
            for (let col = 0; col < rowStr.length; col++) {
                const char = rowStr[col];
                // Position in [-1, 1] coordinates (matching training data format)
                const x = ((col + 0.5) / 10.0) * 2.0 - 1.0;  // Convert [0,1] to [-1,1]
                const y = ((row + 0.5) / 3.0) * 2.0 - 1.0;   // Convert [0,1] to [-1,1]
                layout[char] = { x, y };
            }
        }

        return layout;
    }

    /**
     * Run encoder on extracted features
     */
    async runEncoder(features) {
        console.log('Encoder input features:', {
            trajectoryShape: [1, features.sequenceLength, 6],
            nearestKeysShape: [1, features.sequenceLength],
            srcMaskShape: [1, features.sequenceLength]
        });

        const inputs = {
            'trajectory_features': new ort.Tensor('float32',
                features.trajectoryFeatures, [1, features.sequenceLength, 6]),
            'nearest_keys': new ort.Tensor('int64',
                features.nearestKeys, [1, features.sequenceLength]),
            'src_mask': new ort.Tensor('bool',
                features.srcMask, [1, features.sequenceLength])
        };

        try {
            const outputs = await this.encoder.run(inputs);
            console.log('Encoder output received:', outputs['encoder_output'].dims);
            return {
                memory: outputs['encoder_output'],
                sequenceLength: features.sequenceLength
            };
        } catch (error) {
            console.error('Encoder error:', error);
            throw error;
        }
    }

    /**
     * Autoregressive decoding with CTC decoder
     */
    async decode(encoderOutput, maxLength = 20) {
        const tokens = [this.tokenizer.special_tokens.sos_idx]; // Start with SOS token
        const memory = encoderOutput.memory;
        const srcMask = new Uint8Array(encoderOutput.sequenceLength).fill(1);

        for (let step = 0; step < maxLength; step++) {
            // Create target mask for current sequence
            const targetMask = new Uint8Array(tokens.length).fill(1);

            // Convert tokens to BigInt array
            const targetTokens = new BigInt64Array(tokens.map(t => BigInt(t)));

            // Run decoder
            const inputs = {
                'memory': memory,
                'target_tokens': new ort.Tensor('int64', targetTokens, [1, tokens.length]),
                'src_mask': new ort.Tensor('bool', srcMask, [1, encoderOutput.sequenceLength]),
                'target_mask': new ort.Tensor('bool', targetMask, [1, tokens.length])
            };

            console.log(`Decoder step ${step}, tokens so far:`, tokens);
            const outputs = await this.decoder.run(inputs);
            const logits = outputs['logits'];
            console.log('Decoder logits shape:', logits.dims);

            // Get prediction for last position
            const lastLogits = new Float32Array(this.tokenizer.vocab_size);
            for (let i = 0; i < this.tokenizer.vocab_size; i++) {
                lastLogits[i] = logits.data[(tokens.length - 1) * this.tokenizer.vocab_size + i];
            }

            // Apply softmax and get argmax
            const maxLogit = Math.max(...lastLogits);
            const expLogits = lastLogits.map(x => Math.exp(x - maxLogit));
            const sumExp = expLogits.reduce((a, b) => a + b, 0);
            const probs = expLogits.map(x => x / sumExp);

            let nextToken = 0;
            let maxProb = probs[0];
            for (let i = 1; i < probs.length; i++) {
                if (probs[i] > maxProb) {
                    maxProb = probs[i];
                    nextToken = i;
                }
            }

            // Stop if EOS token
            if (nextToken === this.tokenizer.special_tokens.eos_idx) {
                break;
            }

            // Add token if not special
            if (nextToken > 3) { // Skip special tokens
                tokens.push(nextToken);
            }
        }

        // Convert tokens to text (skip SOS token)
        const chars = [];
        for (let i = 1; i < tokens.length; i++) {
            const token = tokens[i];
            if (token >= 4 && token <= 29) { // Valid character range
                chars.push(this.tokenizer.idx_to_char[token]);
            }
        }

        return {
            text: chars.join(''),
            tokens: tokens.slice(1) // Remove SOS token
        };
    }

    /**
     * Greedy decoding for CTC (non-autoregressive)
     */
    async greedyDecodeCTC(points) {
        if (!this.initialized) {
            throw new Error('CTC decoder not initialized');
        }

        // Extract features
        const features = this.extractCTCFeatures(points);

        // Run encoder
        console.log('Running CTC encoder...');
        const encoderOutput = await this.runEncoder(features);
        console.log(`Encoder output shape: [${encoderOutput.memory.dims}]`);

        // Run autoregressive decoding
        console.log('Running CTC decoder...');
        const result = await this.decode(encoderOutput);

        return result;
    }

    /**
     * Simple CTC decoding (if encoder directly outputs character probabilities)
     */
    async simpleCTCDecode(points) {
        if (!this.initialized) {
            throw new Error('CTC decoder not initialized');
        }

        // Extract features
        const features = this.extractCTCFeatures(points);

        // Run encoder only
        const encoderOutput = await this.runEncoder(features);
        const memory = encoderOutput.memory;

        // If encoder outputs character probabilities directly
        // Apply CTC decoding: collapse repeated characters and remove blanks
        const sequence = [];
        let prevChar = -1;

        for (let t = 0; t < encoderOutput.sequenceLength; t++) {
            // Get logits for this time step
            const logits = new Float32Array(30);
            for (let c = 0; c < 30; c++) {
                logits[c] = memory.data[t * 256 + c]; // Assuming first 30 dims are logits
            }

            // Get argmax
            let maxIdx = 0;
            let maxVal = logits[0];
            for (let c = 1; c < 30; c++) {
                if (logits[c] > maxVal) {
                    maxVal = logits[c];
                    maxIdx = c;
                }
            }

            // CTC collapse: only add if different from previous and not blank/special
            if (maxIdx >= 4 && maxIdx !== prevChar) {
                sequence.push(this.tokenizer.idx_to_char[maxIdx]);
                prevChar = maxIdx;
            }
        }

        return {
            text: sequence.join(''),
            tokens: []
        };
    }

    /**
     * Basic prefix beam search over encoder per-frame logits (simple CTC mode)
     */
    async beamSearchSimpleCTC(points, beamSize = 10, blankId = 0) {
        if (!this.initialized) throw new Error('CTC decoder not initialized');
        const features = this.extractCTCFeatures(points);
        const encOut = await this.runEncoder(features);
        const memory = encOut.memory; // assume [T, D]
        const T = encOut.sequenceLength;
        const V = this.tokenizer.vocab_size || 30;
        const getLogits = (t) => {
            // Assumes logits are in first V dims (adjust if needed)
            const arr = new Float32Array(V);
            for (let c = 0; c < V; c++) arr[c] = memory.data[t * memory.dims[1] + c];
            // Convert to log-probs
            const maxv = Math.max(...arr);
            let s = 0.0; const exps = new Float32Array(V);
            for (let i=0;i<V;i++){const e=Math.exp(arr[i]-maxv); exps[i]=e; s+=e;}
            for (let i=0;i<V;i++) exps[i] = Math.log(exps[i]/s);
            return exps;
        };

        let beam = new Map(); // key: seq string, val: {p_b, p_nb, seq}
        const key = (seq) => seq.join(',');
        const add = (map, seq, pb, pnb) => {
            const k = key(seq);
            const prev = map.get(k) || {p_b:-Infinity,p_nb:-Infinity,seq};
            prev.p_b = Math.max(prev.p_b, pb);
            prev.p_nb = Math.max(prev.p_nb, pnb);
            map.set(k, prev);
        };
        add(beam, [], 0.0, -Infinity);

        for (let t=0; t<T; t++) {
            const lps = getLogits(t);
            // prune current beam
            const top = Array.from(beam.values()).sort((a,b)=> (Math.max(a.p_b,a.p_nb) > Math.max(b.p_b,b.p_nb) ? -1:1)).slice(0, beamSize);
            const next = new Map();
            for (const hyp of top) {
                const seq = hyp.seq; const p_b = hyp.p_b; const p_nb = hyp.p_nb;
                // extend blank
                add(next, seq, p_b + lps[blankId], -Infinity);
                // extend tokens
                for (let c=0;c<V;c++){
                    if (c===blankId) continue;
                    const last = seq.length ? seq[seq.length-1] : -1;
                    const lp = lps[c];
                    if (c===last){
                        add(next, seq, -Infinity, p_nb + lp);
                        add(next, seq.concat([c]), p_b + lp, -Infinity);
                    } else {
                        add(next, seq.concat([c]), Math.max(p_b,p_nb) + lp, -Infinity);
                    }
                }
            }
            beam = next;
        }
        const finals = Array.from(beam.values()).map(h=>({seq:h.seq, score: Math.max(h.p_b,h.p_nb)}));
        finals.sort((a,b)=>b.score - a.score);
        const toChar = (id) => this.tokenizer.idx_to_char ? this.tokenizer.idx_to_char[id] || '' : '';
        const top = finals.slice(0,5).map(h=>({ text: h.seq.map(toChar).join(''), score: h.score }));
        return top;
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CTCDecoder;
}
