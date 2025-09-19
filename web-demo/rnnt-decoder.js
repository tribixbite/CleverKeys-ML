/**
 * RNN-T Beam Search Decoder for Web Demo
 * Implements lexicon-constrained beam search with ONNX Runtime Web
 */

class RNNTDecoder {
    constructor() {
        this.encoderSession = null;
        this.decoderSession = null;
        this.trie = null;
        this.words = [];
        this.charToId = {};
        this.idToChar = {};
        this.blankId = 0;
        this.vocabSize = 29;
        this.L = 2;  // LSTM layers
        this.H = 320;  // Hidden size
        this.D = 256;  // Encoder output dimension
        this.isReady = false;
    }

    async initialize(encoderPath, decoderPath, wordsPath, runtimeMetaPath) {
        console.log('Initializing RNN-T Decoder...');

        // Load runtime metadata
        const metaResponse = await fetch(runtimeMetaPath);
        const meta = await metaResponse.json();
        this.blankId = meta.blank_id;
        this.charToId = meta.char_to_id;
        this.idToChar = {};
        for (const [k, v] of Object.entries(meta.id_to_char)) {
            this.idToChar[parseInt(k)] = v;
        }
        this.vocabSize = meta.vocab_size;

        // Load words
        const wordsResponse = await fetch(wordsPath);
        const wordsText = await wordsResponse.text();
        this.words = wordsText.trim().split('\n');

        // Build trie
        this.trie = this.buildTrie();

        // Load ONNX models
        const sessionOptions = {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        };

        this.encoderSession = await ort.InferenceSession.create(encoderPath, sessionOptions);
        this.decoderSession = await ort.InferenceSession.create(decoderPath, sessionOptions);

        this.isReady = true;
        console.log(`✓ Decoder initialized with ${this.words.length} words`);
    }

    buildTrie() {
        const root = { children: new Map(), isWord: false, wordId: -1 };
        let kept = 0;

        for (let wordId = 0; wordId < this.words.length; wordId++) {
            const word = this.words[wordId].toLowerCase().replace(/'/g, "'");

            // Skip words with unknown characters
            if (![...word].every(ch => ch in this.charToId)) continue;

            let cur = root;
            for (const ch of word) {
                const cid = this.charToId[ch];
                if (!cur.children.has(cid)) {
                    cur.children.set(cid, { children: new Map(), isWord: false, wordId: -1 });
                }
                cur = cur.children.get(cid);
            }
            cur.isWord = true;
            cur.wordId = wordId;
            kept++;
        }

        console.log(`Trie built: ${kept}/${this.words.length} words kept`);
        return root;
    }

    async decode(featuresBFT, beamSize = 16, maxSymbols = 20) {
        if (!this.isReady) {
            throw new Error('Decoder not initialized');
        }

        // Run encoder
        const T = featuresBFT.dims[2];
        const encoderFeeds = {
            'features_bft': featuresBFT,
            'lengths': new ort.Tensor('int32', new Int32Array([T]), [1])
        };

        const encoderOutputs = await this.encoderSession.run(encoderFeeds);
        let encodedBTF = encoderOutputs['encoded_btf'];

        // Handle dimension issues - might be (1, D, T_out) instead of (1, T_out, D)
        let encodedData = encodedBTF.data;
        let shape = encodedBTF.dims;

        // Check if we need to transpose
        if (shape[1] === this.D && shape[2] !== this.D) {
            // Shape is (1, D, T_out), need to transpose to (1, T_out, D)
            const [B, D, T_out] = shape;
            const transposed = new Float32Array(B * T_out * D);
            for (let t = 0; t < T_out; t++) {
                for (let d = 0; d < D; d++) {
                    transposed[t * D + d] = encodedData[d * T_out + t];
                }
            }
            encodedData = transposed;
            shape = [B, T_out, D];
        }

        const T_out = shape[1];
        console.log(`Encoder output shape: [${shape}]`);

        // Initialize beam
        let beams = [{
            yPrev: this.blankId,
            h: this.zeros([this.L, 1, this.H]),
            c: this.zeros([this.L, 1, this.H]),
            trie: this.trie,
            logp: 0,
            chars: []
        }];

        // Time-synchronous beam search
        for (let t = 0; t < T_out; t++) {
            // Inner label loop
            for (let s = 0; s < maxSymbols; s++) {
                // Sort beams by score
                beams.sort((a, b) => b.logp - a.logp);
                const active = beams.slice(0, Math.min(beamSize, beams.length));
                const N = active.length;

                // Prepare batch inputs
                const yPrevData = new BigInt64Array(N);
                const h0Data = new Float32Array(this.L * N * this.H);
                const c0Data = new Float32Array(this.L * N * this.H);
                const encTData = new Float32Array(N * this.D);

                for (let i = 0; i < N; i++) {
                    yPrevData[i] = BigInt(active[i].yPrev);

                    // Copy h and c states
                    const hSrc = active[i].h.data;
                    const cSrc = active[i].c.data;
                    for (let j = 0; j < this.L * this.H; j++) {
                        h0Data[j * N + i] = hSrc[j];
                        c0Data[j * N + i] = cSrc[j];
                    }

                    // Copy encoder frame
                    for (let d = 0; d < this.D; d++) {
                        encTData[i * this.D + d] = encodedData[t * this.D + d];
                    }
                }

                // Run decoder step
                const decoderFeeds = {
                    'y_prev': new ort.Tensor('int64', yPrevData, [N]),
                    'h0': new ort.Tensor('float32', h0Data, [this.L, N, this.H]),
                    'c0': new ort.Tensor('float32', c0Data, [this.L, N, this.H]),
                    'enc_t': new ort.Tensor('float32', encTData, [N, this.D])
                };

                const decoderOutputs = await this.decoderSession.run(decoderFeeds);
                let logits = decoderOutputs['logits'];
                const h1 = decoderOutputs['h1'];
                const c1 = decoderOutputs['c1'];

                // Handle logits dimension issues
                let logitsData = logits.data;
                let logitsShape = logits.dims;

                // Reshape if necessary (might have extra dimensions)
                while (logitsShape.length > 2 && logitsShape[0] === 1) {
                    logitsShape = logitsShape.slice(1);
                }

                const V = logitsShape[logitsShape.length - 1];

                // Apply log softmax if needed
                logitsData = this.applyLogSoftmax(logitsData, N, V);

                // Expand beams
                const nextBeams = [];
                for (let i = 0; i < N; i++) {
                    const beam = active[i];
                    const row = logitsData.slice(i * V, (i + 1) * V);

                    // Blank transition
                    const lpBlank = row[this.blankId];
                    nextBeams.push({
                        yPrev: this.blankId,
                        h: this.sliceState(h1, i, this.L, this.H),
                        c: this.sliceState(c1, i, this.L, this.H),
                        trie: beam.trie,
                        logp: beam.logp + lpBlank,
                        chars: beam.chars.slice()
                    });

                    // Character transitions (trie-constrained)
                    const allowed = Array.from(beam.trie.children.keys());
                    if (allowed.length > 0) {
                        // Sort by score and take top k
                        const charScores = allowed.map(cid => ({ cid, score: row[cid] }));
                        charScores.sort((a, b) => b.score - a.score);

                        for (const { cid, score } of charScores.slice(0, 6)) {
                            const child = beam.trie.children.get(cid);
                            nextBeams.push({
                                yPrev: cid,
                                h: this.sliceState(h1, i, this.L, this.H),
                                c: this.sliceState(c1, i, this.L, this.H),
                                trie: child,
                                logp: beam.logp + score,
                                chars: beam.chars.concat(cid)
                            });
                        }
                    }
                }

                // Prune to beam size
                nextBeams.sort((a, b) => b.logp - a.logp);
                beams = nextBeams.slice(0, beamSize);

                // Early stop if best beam picked blank
                if (beams[0].yPrev === this.blankId) break;
            }
        }

        // Collect completed words
        const candidates = [];
        for (const beam of beams) {
            if (beam.trie.isWord) {
                const wordId = beam.trie.wordId;
                const word = this.words[wordId];
                candidates.push({ word, score: beam.logp, wordId });
            }
        }

        // Sort by score and remove duplicates
        candidates.sort((a, b) => b.score - a.score);
        const seen = new Set();
        const results = [];
        for (const cand of candidates) {
            if (!seen.has(cand.wordId)) {
                seen.add(cand.wordId);
                results.push(cand);
                if (results.length >= 5) break;
            }
        }

        return results;
    }

    zeros(shape) {
        const size = shape.reduce((a, b) => a * b, 1);
        return new ort.Tensor('float32', new Float32Array(size), shape);
    }

    sliceState(state, i, L, H) {
        // Slice beam i from stacked (L, N, H) → (L, 1, H)
        const N = state.dims[1];
        const result = new Float32Array(L * H);
        const src = state.data;

        for (let l = 0; l < L; l++) {
            for (let h = 0; h < H; h++) {
                result[l * H + h] = src[(l * N + i) * H + h];
            }
        }

        return new ort.Tensor('float32', result, [L, 1, H]);
    }

    applyLogSoftmax(logits, N, V) {
        const result = new Float32Array(N * V);

        for (let i = 0; i < N; i++) {
            const offset = i * V;
            const row = logits.slice(offset, offset + V);

            // Check if already log probabilities
            const maxVal = Math.max(...row);
            if (maxVal < 0 && maxVal > -10) {
                // Already log probabilities
                result.set(row, offset);
            } else {
                // Apply log softmax
                const maxLogit = Math.max(...row);
                const expSum = row.reduce((sum, val) => sum + Math.exp(val - maxLogit), 0);
                const logSumExp = Math.log(expSum) + maxLogit;

                for (let j = 0; j < V; j++) {
                    result[offset + j] = row[j] - logSumExp;
                }
            }
        }

        return result;
    }

    async decodeFromGesture(points) {
        // Compute features from points
        const features = this.computeFeatures(points);

        // Convert to BFT format
        const T = features.length;
        const featureData = new Float32Array(1 * 37 * T);
        for (let t = 0; t < T; t++) {
            for (let f = 0; f < 37; f++) {
                featureData[f * T + t] = features[t][f];
            }
        }

        const featuresBFT = new ort.Tensor('float32', featureData, [1, 37, T]);
        return this.decode(featuresBFT);
    }

    computeFeatures(points) {
        const features = [];

        for (let i = 0; i < points.length; i++) {
            const feat = new Float32Array(37);
            const pt = points[i];

            // Position
            feat[0] = pt.x;
            feat[1] = pt.y;
            feat[2] = (pt.t || i * 10) / 1000.0;

            // Velocity
            if (i > 0) {
                const prev = points[i - 1];
                const dt = Math.max((pt.t - prev.t) / 1000.0, 0.001);
                feat[3] = (pt.x - prev.x) / dt;
                feat[4] = (pt.y - prev.y) / dt;
            }

            // Acceleration
            if (i > 1 && features.length > 0) {
                const dt = Math.max((pt.t - points[i - 1].t) / 1000.0, 0.001);
                feat[5] = (feat[3] - features[i - 1][3]) / dt;
                feat[6] = (feat[4] - features[i - 1][4]) / dt;
            }

            features.push(feat);
        }

        return features;
    }
}

// Export for use in HTML
window.RNNTDecoder = RNNTDecoder;