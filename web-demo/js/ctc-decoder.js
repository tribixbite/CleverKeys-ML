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
        this.featureExtractor = null; // Optional: use shared 37D extractor
    }

    async initialize(encoderPath, decoderPath, tokenizerPath) {
        try {
            console.log('Loading CTC models...');

            // Load models
            this.encoder = await ort.InferenceSession.create(encoderPath);
            console.log('CTC Encoder loaded');

            this.decoder = await ort.InferenceSession.create(decoderPath);
            console.log('CTC Decoder loaded');

            // Load tokenizer config (Node vs Browser)
            if (typeof window === 'undefined') {
                const fs = require('fs');
                const raw = fs.readFileSync(tokenizerPath, 'utf-8');
                this.tokenizer = JSON.parse(raw);
            } else {
                const response = await fetch(tokenizerPath);
                this.tokenizer = await response.json();
            }
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

    setFeatureExtractor(extractor) {
        this.featureExtractor = extractor;
    }

    // Normalize to [-1,1] and zero-start time, matching training
    _normalizePoints(points) {
        if (!points || points.length === 0) return [];
        const t0 = points[0].t || 0;
        return points.map(p => ({
            x: (p.x ?? 0.5) * 2 - 1,
            y: (p.y ?? 0.5) * 2 - 1,
            t: Math.max(0, (p.t || 0) - t0)
        }));
    }

    _determineResampleTarget(n) {
        const shortTarget = 56, longTarget = 96, shortThresh = 48, longThresh = 112;
        if (n <= shortThresh) return shortTarget;
        if (n >= longThresh) return longTarget;
        const p = (n - shortThresh) / (longThresh - shortThresh);
        return Math.floor(shortTarget + p * (longTarget - shortTarget));
    }

    _resample(points, target) {
        if (!points || !points.length) return [];
        if (points.length === target) return points;
        if (points.length === 1) return Array.from({length: target}, () => ({...points[0]}));
        const duration = Math.max(points[points.length-1].t - points[0].t, 1.0);
        const step = duration / Math.max(target - 1, 1);
        const out = [];
        let i = 0;
        for (let k=0;k<target;k++){
            const tt = points[0].t + step*k;
            while (i < points.length-2 && points[i+1].t < tt) i++;
            const p1 = points[i], p2 = points[Math.min(i+1, points.length-1)];
            const alpha = p2.t > p1.t ? Math.min(1, Math.max(0, (tt - p1.t) / (p2.t - p1.t))) : 0;
            out.push({ x: p1.x + (p2.x - p1.x)*alpha, y: p1.y + (p2.y - p1.y)*alpha, t: tt });
        }
        return out;
    }

    _ctcKeyLayout01() {
        // Matches train_squeezeformer_ctc PersonalizedSwipeFeaturizer._get_keyboard_layout()
        const rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]; const layout=[];
        for (let r=0;r<rows.length;r++){
            const y = r * 0.33;
            for (let c=0;c<rows[r].length;c++){
                const ch = rows[r][c];
                const x = (c / 10.0) + (r>0 ? 0.05*r : 0);
                layout.push({ ch, x, y });
            }
        }
        layout.push({ ch: "'", x: 0.95, y: 0.33 });
        return layout; // 28 entries
    }

    _featurize37Exact(points) {
        const norm = this._normalizePoints(points);
        const target = this._determineResampleTarget(norm.length);
        const pts = this._resample(norm, target);
        const keys = this._ctcKeyLayout01();
        const T = pts.length; const F = 37;
        const out = new Float32Array(T*F);
        for (let i=0;i<T;i++){
            const p = pts[i]; const base = i*F;
            out[base+0]=p.x; out[base+1]=p.y;
            // velocity
            if (i>0){ const prev=pts[i-1]; const dt=Math.max((p.t - prev.t)/1000.0,1e-6); out[base+2]=(p.x-prev.x)/dt; out[base+3]=(p.y-prev.y)/dt; }
            // acceleration via central difference
            if (i>0 && i<T-1){ const prev=pts[i-1], next=pts[i+1]; const dt_next=Math.max((next.t - p.t)/1000.0,1e-6); const vx_next=(next.x - p.x)/dt_next; const vy_next=(next.y - p.y)/dt_next; const dt_total=Math.max((next.t - prev.t)/1000.0,1e-6); out[base+4]=(vx_next - out[base+2])/dt_total; out[base+5]=(vy_next - out[base+3])/dt_total; }
            const speed=Math.hypot(out[base+2]||0, out[base+3]||0); out[base+6]=speed; out[base+7]=speed>1e-6 ? Math.atan2(out[base+3]||0,out[base+2]||0) : 0;
            // distances to 28 keys (in [0,1] layout; training used this despite [-1,1] coords)
            for (let k=0;k<keys.length;k++){
                const dx=p.x - keys[k].x, dy=p.y - keys[k].y; const dist=Math.hypot(dx,dy); out[base+8+k]=Math.exp(-dist*5);
            }
            // curvature
            if (i>1 && i<T-1){ const prev=pts[i-1], next=pts[i+1]; const v1=[p.x - prev.x, p.y - prev.y]; const v2=[next.x - p.x, next.y - p.y]; const n1=Math.hypot(v1[0],v1[1]), n2=Math.hypot(v2[0],v2[1]); if (n1>1e-6 && n2>1e-6){ const cos=(v1[0]*v2[0]+v1[1]*v2[1])/(n1*n2); out[base+36]=Math.acos(Math.max(-1,Math.min(1,cos))); } }
        }
        return { flat: out, T };
    }

    /**
     * Resample points to target count (fallback simple path)
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
     * Extract features from points
     * If a shared 37D feature extractor is provided, prefer it to match training.
     */
    extractCTCFeatures(points) {
        // Always use exact 37D CTC featurizer to match training
        const exact = this._featurize37Exact(points);
        return { flatFeatures: exact.flat, sequenceLength: exact.T, featureDim: 37 };

        // Fallback minimal 6D features (not used when exact is available)
        const targetPoints = 150; // fixed length fallback
        const resampledPoints = this.resamplePoints(points, targetPoints);
        const numPoints = resampledPoints.length;
        const trajectoryFeatures = new Float32Array(numPoints * 6);
        for (let i = 0; i < numPoints; i++) {
            const curr = resampledPoints[i];
            const prev = i > 0 ? resampledPoints[i - 1] : curr;
            const dt = i > 0 ? Math.max((curr.t - prev.t) / 1000.0, 0.001) : 0.001;
            const vx = i > 0 ? (curr.x - prev.x) / dt : 0;
            const vy = i > 0 ? (curr.y - prev.y) / dt : 0;
            trajectoryFeatures[i * 6 + 0] = curr.x;
            trajectoryFeatures[i * 6 + 1] = curr.y;
            trajectoryFeatures[i * 6 + 2] = vx;
            trajectoryFeatures[i * 6 + 3] = vy;
            trajectoryFeatures[i * 6 + 4] = curr.t / 1000.0;
            trajectoryFeatures[i * 6 + 5] = 1.0;
        }
        const srcMask = new Uint8Array(numPoints).fill(1);
        const keyboardLayout = this.getKeyboardLayout();
        const nearestKeys = new BigInt64Array(numPoints);
        for (let i = 0; i < numPoints; i++) {
            const point = resampledPoints[i];
            let minDist = Infinity;
            let nearestKey = 0;
            for (const [char, pos] of Object.entries(keyboardLayout)) {
                const dist = Math.hypot(point.x - pos.x, point.y - pos.y);
                if (dist < minDist) { minDist = dist; nearestKey = this.tokenizer.char_to_idx[char] || 0; }
            }
            nearestKeys[i] = BigInt(nearestKey);
        }
        return { trajectoryFeatures, nearestKeys, srcMask, sequenceLength: numPoints };
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
        if (features.flatFeatures) {
            console.log('Encoder input features:', {
                featuresShape: [1, features.sequenceLength, features.featureDim || 37]
            });
        } else {
            console.log('Encoder input features:', {
                trajectoryShape: [1, features.sequenceLength, 6],
                nearestKeysShape: [1, features.sequenceLength],
                srcMaskShape: [1, features.sequenceLength]
            });
        }

        // Build inputs based on model signature
        const inputNames = this.encoder.inputNames || Object.keys(this.encoder.inputMetadata || {});
        let inputs;
        if (inputNames.includes('features')) {
            const T = features.sequenceLength;
            const F = features.featureDim || 37;
            if (!features.flatFeatures) throw new Error('Expected 37D features. Provide featureExtractor to CTCDecoder.');
            inputs = {
                'features': new ort.Tensor('float32', features.flatFeatures, [1, T, F]),
                'feature_lengths': new ort.Tensor('int64', BigInt64Array.from([BigInt(T)]), [1])
            };
        } else {
            // Legacy triple-input path (not typical for CTC export)
            inputs = {
                'trajectory_features': new ort.Tensor('float32',
                    features.trajectoryFeatures, [1, features.sequenceLength, 6]),
                'nearest_keys': new ort.Tensor('int64',
                    features.nearestKeys, [1, features.sequenceLength]),
                'src_mask': new ort.Tensor('bool',
                    features.srcMask, [1, features.sequenceLength])
            };
        }

        try {
            const outputs = await this.encoder.run(inputs);
            // Be flexible about output name: prefer 'log_probs' (CTC), then 'encoder_output', then 'logits', else first tensor
            let memory = outputs['log_probs'] || outputs['encoder_output'] || outputs['logits'] || null;
            if (!memory) {
                const first = Object.values(outputs)[0];
                if (!first) throw new Error('No outputs from encoder session');
                memory = first;
            }
            console.log('Encoder output received:', memory.dims);
            return {
                memory,
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
        const dims = memory.dims || [];
        const T = (dims.length === 3) ? dims[1] : encoderOutput.sequenceLength;
        const D = (dims.length === 3) ? dims[2] : (dims[1] || 0);
        const V = this.tokenizer?.vocab_size || D || 28;
        const blankId = (this.tokenizer?.special_tokens && typeof this.tokenizer.special_tokens.blank_id === 'number')
            ? this.tokenizer.special_tokens.blank_id : (V - 1);

        // If encoder outputs character probabilities directly
        // Apply CTC decoding: collapse repeated characters and remove blanks
        const sequence = [];
        let prevId = -1;

        for (let t = 0; t < T; t++) {
            // Get logits for this time step
            const logits = new Float32Array(V);
            for (let c = 0; c < V; c++) logits[c] = memory.data[t * D + c];

            // Get argmax
            let maxIdx = 0;
            let maxVal = logits[0];
            for (let c = 1; c < V; c++) {
                if (logits[c] > maxVal) {
                    maxVal = logits[c];
                    maxIdx = c;
                }
            }

            // CTC collapse: only add if different from previous and not blank/special
            if (maxIdx !== blankId && maxIdx !== prevId) {
                const ch = this.tokenizer?.idx_to_char ? this.tokenizer.idx_to_char[String(maxIdx)] : '';
                if (ch) sequence.push(ch);
            }
            prevId = maxIdx;
        }

        return {
            text: sequence.join(''),
            tokens: []
        };
    }

    /**
     * Basic prefix beam search over encoder per-frame logits (simple CTC mode)
     */
    async beamSearchSimpleCTC(points, beamSize = 10, blankId) {
        if (!this.initialized) throw new Error('CTC decoder not initialized');
        const features = this.extractCTCFeatures(points);
        const encOut = await this.runEncoder(features);
        const memory = encOut.memory;
        const dims = memory.dims || [];
        const T = (dims.length === 3) ? dims[1] : encOut.sequenceLength;
        const D = (dims.length === 3) ? dims[2] : (dims[1] || 0);
        const V = this.tokenizer.vocab_size || D || 28;
        const resolvedBlank = (blankId !== undefined && blankId !== null)
            ? blankId
            : ((this.tokenizer.special_tokens && typeof this.tokenizer.special_tokens.blank_id === 'number')
                ? this.tokenizer.special_tokens.blank_id : (V - 1));
        const getLogits = (t) => {
            // Assumes logits are in first V dims (adjust if needed)
            const arr = new Float32Array(V);
            for (let c = 0; c < V; c++) arr[c] = memory.data[t * D + c];
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
                add(next, seq, p_b + lps[resolvedBlank], -Infinity);
                // extend tokens
                for (let c=0;c<V;c++){
                    if (c===resolvedBlank) continue;
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
