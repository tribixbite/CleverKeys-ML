/**
 * RNN-T Decoder with a combined Decoder/Joint model.
 */

class RNNTDecoder {
    constructor() {
        this.encoderSession = null;
        this.decoderJointSession = null;
        this.runtimeMeta = null;
        this.vocabSize = 30;
        this.blankId = 29;
        this.lexicon = null; // { trie, words, logFreqs, charToId, idToChar }
    }

    /**
     * Initialize ONNX sessions with the stateful models
     */
    async initialize(encoderPath, decoderJointPath, metaPath) {
        const sessionOptions = {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        };

        console.log('Loading RNN-T models (2-model setup)...');

        [this.encoderSession, this.decoderJointSession, this.runtimeMeta] = await Promise.all([
            ort.InferenceSession.create(encoderPath, sessionOptions),
            ort.InferenceSession.create(decoderJointPath, sessionOptions),
            fetch(metaPath).then(r => r.json())
        ]);

        this.blankId = this.runtimeMeta.blank_id;
        this.vocabSize = this.runtimeMeta.vocab_size;

        console.log('Encoder loaded. Inputs:', this.encoderSession.inputNames, 'Outputs:', this.encoderSession.outputNames);
        console.log('Decoder/Joint loaded. Inputs:', this.decoderJointSession.inputNames, 'Outputs:', this.decoderJointSession.outputNames);
        console.log('Runtime meta loaded:', this.runtimeMeta);
    }

    /**
     * Load and build lexicon trie from words.txt and aligned log frequencies JSON.
     * Applies filtering to remove unsuitable entries for gesture prediction.
     */
    async loadLexicon(wordListUrl = 'words.txt', freqUrl = 'word_frequencies_aligned.json') {
        const [wordsText, freqJson] = await Promise.all([
            fetch(wordListUrl).then(r => r.text()),
            fetch(freqUrl).then(r => r.json())
        ]);

        const rawWords = wordsText.split(/\r?\n/).map(w => w.trim()).filter(Boolean);
        const logFreqs = freqJson.log_frequencies || [];
        const words = [];
        const priors = [];

        const allow = /^[a-z']{2,20}$/;
        const triple = /(.)\1\1/; // no triple repeats
        const minFreqByLen = (L) => {
            if (L <= 2) return 1e-5;
            if (L === 3) return 1e-6;
            if (L === 4) return 1e-7;
            if (L === 5) return 5e-8;
            if (L <= 7) return 1e-8;
            if (L === 8) return 5e-9;
            if (L === 9) return 1e-9;
            return 5e-10;
        };

        for (let i = 0; i < rawWords.length; i++) {
            const w = rawWords[i];
            if (!allow.test(w)) continue;
            if (triple.test(w)) continue;
            const logp = logFreqs[i] ?? -30.0;
            const p = Math.exp(logp);
            if (p < minFreqByLen(w.length)) continue;
            words.push(w);
            priors.push(logp);
        }

        const charToId = this.runtimeMeta?.char_to_id || {};
        const idToChar = this.runtimeMeta?.id_to_char || {};

        // Build trie with integer character IDs
        const root = { children: new Map(), isWordEnd: false, wid: -1, logp: 0.0 };
        const toId = (ch) => (charToId[ch] !== undefined ? charToId[ch] : null);
        for (let i = 0; i < words.length; i++) {
            const w = words[i];
            let node = root;
            let ok = true;
            for (const ch of w) {
                const cid = toId(ch);
                if (cid == null) { ok = false; break; }
                if (!node.children.has(cid)) node.children.set(cid, { children: new Map(), isWordEnd: false, wid: -1, logp: 0.0 });
                node = node.children.get(cid);
            }
            if (ok) { node.isWordEnd = true; node.wid = i; node.logp = priors[i]; }
        }

        this.lexicon = { trie: root, words, logFreqs: priors, charToId, idToChar };
        console.log(`Lexicon loaded: kept ${words.length} words from ${rawWords.length}`);
    }

    /**
     * Greedy decoding (beam size = 1)
     */
    async greedyDecode(featureData, maxSymbols = 20) {
        // Accept either {features, numFrames} or a raw Float32Array
        let features; let numFrames;
        if (featureData && featureData.features && featureData.numFrames != null) {
            features = featureData.features;
            numFrames = featureData.numFrames;
        } else if (featureData instanceof Float32Array) {
            features = featureData;
            if (features.length % 37 !== 0) throw new Error('Feature length not divisible by 37');
            numFrames = features.length / 37;
        } else {
            throw new Error('greedyDecode expected featureData object or Float32Array');
        }

        // 1. Encoder Pass: transpose to [B, F, T]
        const featDim = 37;
        const timeSteps = numFrames;
        const transposed = new Float32Array(featDim * timeSteps);
        for (let t = 0; t < timeSteps; t++) {
            for (let f = 0; f < featDim; f++) {
                transposed[f * timeSteps + t] = features[t * featDim + f];
            }
        }

        // Flexible input names
        const encInputs = {};
        const encInputNames = this.encoderSession.inputNames || [];
        const audioName = encInputNames.includes('audio_signal') ? 'audio_signal' : (encInputNames.includes('features_bft') ? 'features_bft' : encInputNames[0]);
        const lenName = encInputNames.includes('length') ? 'length' : (encInputNames.includes('lengths') ? 'lengths' : (encInputNames[1] || 'length'));
        encInputs[audioName] = new ort.Tensor('float32', transposed, [1, featDim, timeSteps]);
        encInputs[lenName] = new ort.Tensor('int64', BigInt64Array.from([BigInt(timeSteps)]), [1]);

        const encOut = await this.encoderSession.run(encInputs);
        const encoded = encOut.outputs || encOut.encoded_btf || encOut.encoded || encOut.encoder_output;
        const encodedLenTensor = encOut.encoded_lengths || encOut.length || encOut.lengths;
        const encodedLength = Number(encodedLenTensor.data[0]);

        // 2. Decode Loop
        let decodedTokens = [];
        let lastToken = this.blankId;
        let state_h = new ort.Tensor('float32', new Float32Array(2 * 1 * 320).fill(0), [2, 1, 320]);
        let state_c = new ort.Tensor('float32', new Float32Array(2 * 1 * 320).fill(0), [2, 1, 320]);

        for (let t = 0; t < encodedLength; t++) {
            // Frame extraction robust to [B, 256, T] or [B, T, 256]
            const dims = encoded.dims;
            let frameVec;
            if (dims[1] === 256 && dims[2] === encodedLength) {
                frameVec = new Float32Array(256);
                for (let f = 0; f < 256; f++) frameVec[f] = encoded.data[f * encodedLength + t];
            } else if (dims[1] === encodedLength && dims[2] === 256) {
                const start = t * 256;
                frameVec = encoded.data.slice(start, start + 256);
            } else {
                throw new Error(`Unexpected encoder output dims: ${JSON.stringify(dims)}`);
            }
            const encoderFrame = new ort.Tensor('float32', frameVec, [1, 256, 1]);
            const decoderInput = new ort.Tensor('int32', Int32Array.from([lastToken]), [1, 1]);
            const targetLength = new ort.Tensor('int32', Int32Array.from([1]), [1]);

            const jointFeeds = {
                'encoder_outputs': encoderFrame,
                'targets': decoderInput,
                'target_length': targetLength,
                'input_states_1': state_h,
                'input_states_2': state_c,
            };

            // RNNT greedy: emit multiple symbols per frame until blank
            const maxSymbolsPerFrame = 6;
            let symbolsEmitted = 0;
            while (symbolsEmitted < maxSymbolsPerFrame && decodedTokens.length < maxSymbols) {
                const jointResults = await this.decoderJointSession.run(jointFeeds);
                const logitsTensor = jointResults.outputs || jointResults.logits || jointResults.joint_output;
                const logits = logitsTensor.data;

                // Argmax
                let maxVal = -Infinity;
                let predictedToken = -1;
                for (let i = 0; i < this.vocabSize; i++) {
                    if (logits[i] > maxVal) { maxVal = logits[i]; predictedToken = i; }
                }

                // Update recurrent states
                state_h = jointResults.output_states_1;
                state_c = jointResults.output_states_2;

                if (predictedToken === this.blankId || predictedToken === 0) {
                    // blank: advance to next time step
                    break;
                } else {
                    decodedTokens.push(predictedToken);
                    lastToken = predictedToken;
                    symbolsEmitted += 1;
                    // Prepare next symbol prediction for same time frame
                    jointFeeds.targets = new ort.Tensor('int32', Int32Array.from([lastToken]), [1, 1]);
                    jointFeeds.input_states_1 = state_h;
                    jointFeeds.input_states_2 = state_c;
                }
            }
        }

        return [{ text: this.tokensToText(decodedTokens), tokens: decodedTokens }];
    }

    tokensToText(tokens) {
        if (this.runtimeMeta && Array.isArray(this.runtimeMeta.tokens)) {
            return tokens.map(t => this.runtimeMeta.tokens[t] || '').join('');
        }
        if (this.runtimeMeta && this.runtimeMeta.id_to_char) {
            return tokens.map(t => this.runtimeMeta.id_to_char[String(t)] || '').join('');
        }
        const fallback = " 'abcdefghijklmnopqrstuvwxyz";
        return tokens.map(t => fallback[t] || '').join('');
    }

    // Beam search is more complex and left as a future exercise.
    // The greedy decoder is sufficient for verifying the flow.
    async beamSearch(featureData, config = {}) {
        if (!this.lexicon) {
            console.warn('Lexicon not loaded; falling back to greedy decode');
            return this.greedyDecode(featureData, config.maxSymbols);
        }

        const {
            beamSize = 16,
            topK = 8,
            symbolsPerStep = 8,
            maxSymbols = 24,
            lengthPenalty = 0.0
        } = config;

        // Prepare encoder as in greedy
        let features; let numFrames;
        if (featureData && featureData.features && featureData.numFrames != null) {
            features = featureData.features; numFrames = featureData.numFrames;
        } else if (featureData instanceof Float32Array) {
            features = featureData; numFrames = features.length / 37;
        } else { throw new Error('beamSearch expected featureData object or Float32Array'); }

        const featDim = 37, T = numFrames;
        const transposed = new Float32Array(featDim * T);
        for (let t = 0; t < T; t++) for (let f = 0; f < featDim; f++) transposed[f * T + t] = features[t * featDim + f];
        const encInputs = {};
        const encInputNames = this.encoderSession.inputNames || [];
        const audioName = encInputNames.includes('audio_signal') ? 'audio_signal' : (encInputNames.includes('features_bft') ? 'features_bft' : encInputNames[0]);
        const lenName = encInputNames.includes('length') ? 'length' : (encInputNames.includes('lengths') ? 'lengths' : (encInputNames[1] || 'length'));
        encInputs[audioName] = new ort.Tensor('float32', transposed, [1, featDim, T]);
        encInputs[lenName] = new ort.Tensor('int64', BigInt64Array.from([BigInt(T)]), [1]);
        const encOut = await this.encoderSession.run(encInputs);
        const encoded = encOut.outputs || encOut.encoded_btf || encOut.encoded || encOut.encoder_output;
        const encodedLen = Number((encOut.encoded_lengths || encOut.length || encOut.lengths).data[0]);

        const softmax = (arr) => {
            let max = -Infinity; for (const x of arr) if (x > max) max = x;
            const exps = arr.map(x => Math.exp(x - max));
            const sum = exps.reduce((a, b) => a + b, 0);
            return exps.map(x => x / sum);
        };

        const toChar = (tid) => {
            if (Array.isArray(this.runtimeMeta.tokens)) return this.runtimeMeta.tokens[tid] || '';
            if (this.runtimeMeta.id_to_char) return this.runtimeMeta.id_to_char[String(tid)] || '';
            return '';
        };

        // Hypothesis structure
        const initState = {
            tokens: [],
            score: 0.0,
            h: new ort.Tensor('float32', new Float32Array(2 * 1 * 320).fill(0), [2, 1, 320]),
            c: new ort.Tensor('float32', new Float32Array(2 * 1 * 320).fill(0), [2, 1, 320]),
            lastToken: this.blankId,
            node: this.lexicon.trie,
            text: ''
        };

        let beam = [initState];

        for (let t = 0; t < encodedLen; t++) {
            const dims = encoded.dims;
            let frameVec;
            if (dims[1] === 256 && dims[2] === encodedLen) {
                frameVec = new Float32Array(256);
                for (let f = 0; f < 256; f++) frameVec[f] = encoded.data[f * encodedLen + t];
            } else if (dims[1] === encodedLen && dims[2] === 256) {
                const start = t * 256; frameVec = encoded.data.slice(start, start + 256);
            } else { throw new Error(`Unexpected encoder output dims: ${JSON.stringify(dims)}`); }
            const encoderFrame = new ort.Tensor('float32', frameVec, [1, 256, 1]);

            let nextBeam = [];
            for (const hyp of beam) {
                // Copy per-hypothesis states
                let h = hyp.h, c = hyp.c, last = hyp.lastToken, node = hyp.node;
                let emitted = 0;
                while (emitted < symbolsPerStep && nextBeam.length < beamSize * (topK + 1)) {
                    const feeds = {
                        'encoder_outputs': encoderFrame,
                        'targets': new ort.Tensor('int32', Int32Array.from([last]), [1, 1]),
                        'target_length': new ort.Tensor('int32', Int32Array.from([1]), [1]),
                        'input_states_1': h,
                        'input_states_2': c,
                    };
                    const out = await this.decoderJointSession.run(feeds);
                    const logitsT = out.outputs || out.logits || out.joint_output;
                    h = out.output_states_1; c = out.output_states_2;
                    const probs = softmax(Array.from(logitsT.data));

                    // Expand topK non-blank tokens constrained by trie
                    const indexed = probs.map((p, i) => [i, p]);
                    indexed.sort((a, b) => b[1] - a[1]);

                    let expanded = 0;
                    for (let k = 0; k < indexed.length && expanded < topK; k++) {
                        const [tid, p] = indexed[k];
                        if (tid === this.blankId || tid === 0) continue; // handled separately
                        const ch = toChar(tid);
                        if (!ch) continue;
                        const cid = this.lexicon.charToId[ch];
                        if (cid == null) continue;
                        if (!node.children.has(cid)) continue; // lexicon constraint
                        const child = node.children.get(cid);
                        nextBeam.push({
                            tokens: hyp.tokens.concat([tid]),
                            score: hyp.score + Math.log(p + 1e-12),
                            h, c,
                            lastToken: tid,
                            node: child,
                            text: hyp.text + ch
                        });
                        expanded++;
                    }

                    // Handle blank transition: advance time without adding symbol
                    const pBlank = probs[this.blankId] || 1e-12;
                    nextBeam.push({
                        tokens: hyp.tokens,
                        score: hyp.score + Math.log(pBlank + 1e-12),
                        h, c,
                        lastToken: this.blankId,
                        node,
                        text: hyp.text
                    });

                    // If we emitted a token, continue; otherwise break for next time step
                    if (expanded === 0) break; else emitted++;
                }
            }

            // Prune
            nextBeam.sort((a, b) => b.score - a.score);
            beam = nextBeam.slice(0, beamSize);
        }

        // Score completed words with priors
        const scored = beam.map(h => {
            let bonus = 0;
            if (h.node && h.node.isWordEnd && h.node.wid >= 0) bonus = h.node.logp || 0;
            const lp = h.score + bonus;
            const lenNorm = lp / Math.pow((h.tokens.length || 1), 1.0 - lengthPenalty);
            return { text: h.text, tokens: h.tokens, score: lenNorm, rawScore: lp, isComplete: !!(h.node && h.node.isWordEnd) };
        }).sort((a, b) => b.score - a.score);
        const complete = scored.filter(x => x.isComplete);
        return (complete.length ? complete : scored).slice(0, 10);
    }
}
