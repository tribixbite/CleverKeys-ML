/**
 * RNN-T Decoder with a combined Decoder/Joint model.
 */

class RNNTDecoder {
    constructor() {
        this.encoderSession = null;
        this.decoderJointSession = null;
        this.runtimeMeta = null;
        this.vocabSize = 0;
        this.blankId = 0;
        this.predHidden = 320;
        this.predLayers = 2;
        this.encoderDim = 256;
        this.lexicon = null; // { trie, words, logFreqs, charToId, idToChar }
        this.verbose = false;
        this.keyCenters = [];
        this.ort = null;
    }

    /**
     * Initialize ONNX sessions with the stateful models
     */
    async initialize(ort, encoderPath, decoderJointPath, metaPath) {
        this.ort = ort;
        if (typeof window === 'undefined') {
            // Node.js environment
            const fs = require('fs');
            const sessionOptions = { executionProviders: ['cpu'], graphOptimizationLevel: 'all' };
            console.log('Loading RNN-T models for Node.js...');

            const [encoderBuffer, decoderJointBuffer, metaContents] = await Promise.all([
                fs.promises.readFile(encoderPath),
                fs.promises.readFile(decoderJointPath),
                fs.promises.readFile(metaPath, 'utf-8').then(JSON.parse)
            ]);

            this.runtimeMeta = metaContents;
            [this.encoderSession, this.decoderJointSession] = await Promise.all([
                ort.InferenceSession.create(encoderBuffer, sessionOptions),
                ort.InferenceSession.create(decoderJointBuffer, sessionOptions)
            ]);

        } else {
            // Browser environment
            const sessionOptions = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
            console.log('Loading RNN-T models for browser...');

            const [encoder, decoder, meta] = await Promise.all([
                ort.InferenceSession.create(encoderPath, sessionOptions),
                ort.InferenceSession.create(decoderJointPath, sessionOptions),
                fetch(metaPath).then(r => r.json())
            ]);
            this.encoderSession = encoder;
            this.decoderJointSession = decoder;
            this.runtimeMeta = meta;
        }

        this.blankId = this.runtimeMeta.blank_id;
        this.vocabSize = this.runtimeMeta.vocab_size;
        // BOS/start token: functional RNNT start is usually the '<blank>' label index
        const tokens = this.runtimeMeta.tokens || [];
        const cid = tokens.indexOf('<blank>');
        this.bosId = cid >= 0 ? cid : this.blankId;

        // Strictly derive predictor state sizes from decoder_joint input metadata first
        const djMeta = this.decoderJointSession.inputMetadata;
        if (!djMeta || !djMeta['input_states_1']) {
            if (this.runtimeMeta.decoder_config) {
                this.predLayers = this.runtimeMeta.decoder_config.num_layers;
                this.predHidden = this.runtimeMeta.decoder_config.hidden_size;
                this.encoderDim = this.runtimeMeta.decoder_config.encoder_dim ?? this.encoderDim;
            } else {
                throw new Error('Decoder input metadata missing and decoder_config absent');
            }
        } else {
            const dims = djMeta['input_states_1'].dimensions;
            if (!Array.isArray(dims) || dims.length !== 3 || typeof dims[0] !== 'number' || typeof dims[2] !== 'number') {
                if (this.runtimeMeta.decoder_config) {
                    this.predLayers = this.runtimeMeta.decoder_config.num_layers;
                    this.predHidden = this.runtimeMeta.decoder_config.hidden_size;
                    this.encoderDim = this.runtimeMeta.decoder_config.encoder_dim ?? this.encoderDim;
                } else {
                    throw new Error('Decoder input dims not concrete and decoder_config absent');
                }
            } else {
                this.predLayers = dims[0];
                this.predHidden = dims[2];
            }
        }
        console.log('Decoder state dims:', { predLayers: this.predLayers, predHidden: this.predHidden });

        console.log('Encoder loaded. Inputs:', this.encoderSession.inputNames, 'Outputs:', this.encoderSession.outputNames);
        console.log('Decoder/Joint loaded. Inputs:', this.decoderJointSession.inputNames, 'Outputs:', this.decoderJointSession.outputNames);
        console.log('Runtime meta loaded:', this.runtimeMeta);
    }

    /**
     * Load and build lexicon trie from words.txt and aligned log frequencies JSON.
     * Applies filtering to remove unsuitable entries for gesture prediction.
     */
    async loadLexicon(wordListUrl = 'words.txt', freqUrl = 'word_frequencies_aligned.json') {
        let wordsText, freqJson;
        if (typeof window === 'undefined') {
            const fs = require('fs');
            wordsText = fs.readFileSync(wordListUrl, 'utf-8');
            freqJson = JSON.parse(fs.readFileSync(freqUrl, 'utf-8'));
        } else {
            const [words, freqs] = await Promise.all([
                fetch(wordListUrl).then(r => r.text()),
                fetch(freqUrl).then(r => r.json())
            ]);
            wordsText = words;
            freqJson = freqs;
        }

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

        // Deterministic input/output names
        const encInputs = {};
        const encInputNames = this.encoderSession.inputNames;
        if (!encInputNames || encInputNames.length < 2) throw new Error('Encoder input names not found');
        encInputs[encInputNames[0]] = new this.ort.Tensor('float32', transposed, [1, featDim, timeSteps]);
        encInputs[encInputNames[1]] = new this.ort.Tensor('int64', BigInt64Array.from([BigInt(timeSteps)]), [1]);
        if (this.verbose) {
            console.log('[RNNT-greedy] enc feed:', Object.fromEntries(Object.entries(encInputs).map(([k,v])=>[k, v.dims])));
        }
        const encOut = await this.encoderSession.run(encInputs);
        const encOutNames = this.encoderSession.outputNames;
        if (!encOutNames || encOutNames.length < 2) throw new Error('Encoder output names not found');
        const encoded = encOut[encOutNames[0]];
        const encodedLenTensor = encOut[encOutNames[1]];
        const encodedLength = Number(encodedLenTensor.data[0]);

        if (this.verbose) {
            console.log('[RNNT-greedy] enc out dims:', encoded.dims, 'Tprime=', encodedLength, 'encoderDim=', this.encoderDim);
        }

        // 2. Decode Loop
        let decodedTokens = [];
        let lastToken = this.bosId;
        let state_h = new this.ort.Tensor('float32', new Float32Array(this.predLayers * 1 * this.predHidden).fill(0), [this.predLayers, 1, this.predHidden]);
        let state_c = new this.ort.Tensor('float32', new Float32Array(this.predLayers * 1 * this.predHidden).fill(0), [this.predLayers, 1, this.predHidden]);

        for (let t = 0; t < encodedLength; t++) {
            const start = t * this.encoderDim;
            const frameVec = encoded.data.slice(start, start + this.encoderDim);
            const encoderFrame = new this.ort.Tensor('float32', frameVec, [1, this.encoderDim, 1]);
            const decoderInput = new this.ort.Tensor('int32', Int32Array.from([lastToken]), [1, 1]);
            const targetLength = new this.ort.Tensor('int32', Int32Array.from([1]), [1]);

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
                const logits = jointResults.outputs.data;

                // Argmax
                let maxVal = -Infinity;
                let predictedToken = -1;
                for (let i = 0; i < this.vocabSize; i++) {
                    if (logits[i] > maxVal) { maxVal = logits[i]; predictedToken = i; }
                }

                // Update recurrent states
                state_h = jointResults.output_states_1;
                state_c = jointResults.output_states_2;

                if (predictedToken === this.blankId) {
                    // blank: advance to next time step
                    break;
                } else {
                    decodedTokens.push(predictedToken);
                    lastToken = predictedToken;
                    symbolsEmitted += 1;
                    // Prepare next symbol prediction for same time frame
                    jointFeeds.targets = new this.ort.Tensor('int32', Int32Array.from([lastToken]), [1, 1]);
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
        throw new Error('Runtime meta missing token mappings');
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
            symbolsPerStep = 3,
            maxSymbols = 24,
            lengthPenalty = 0.6,
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
        const encInputNames = this.encoderSession.inputNames;
        if (!encInputNames || encInputNames.length < 2) throw new Error('Encoder input names not found');
        encInputs[encInputNames[0]] = new this.ort.Tensor('float32', transposed, [1, featDim, T]);
        encInputs[encInputNames[1]] = new this.ort.Tensor('int64', BigInt64Array.from([BigInt(T)]), [1]);
        if (this.verbose) {
            console.log('[RNNT] enc feed:', Object.fromEntries(Object.entries(encInputs).map(([k,v])=>[k, v.dims])));
        }
        const encOut = await this.encoderSession.run(encInputs);
        const encOutNames = this.encoderSession.outputNames;
        if (!encOutNames || encOutNames.length < 2) throw new Error('Encoder output names not found');
        const encoded = encOut[encOutNames[0]];
        const encodedLen = Number(encOut[encOutNames[1]].data[0]);

        if (this.verbose) {
            console.log('[RNNT] enc out dims:', encoded.dims, 'Tprime=', encodedLen, 'encoderDim=', this.encoderDim);
        }

        const softmax = (arr) => {
            let max = -Infinity; for (const x of arr) if (x > max) max = x;
            const exps = arr.map(x => Math.exp(x - max));
            const sum = exps.reduce((a, b) => a + b, 0);
            return exps.map(x => x / sum);
        };

        const unkId = Array.isArray(this.runtimeMeta.tokens) ? this.runtimeMeta.tokens.indexOf('<unk>') : (this.runtimeMeta.char_to_id && this.runtimeMeta.char_to_id['<unk>'] !== undefined ? this.runtimeMeta.char_to_id['<unk>'] : -1);
        const toChar = (tid) => {
            if (Array.isArray(this.runtimeMeta.tokens)) return this.runtimeMeta.tokens[tid] || '';
            if (this.runtimeMeta.id_to_char) return this.runtimeMeta.id_to_char[String(tid)] || '';
            return '';
        };

        // Hypothesis structure
        const initState = {
            tokens: [],
            score: 0.0,
            h: new this.ort.Tensor('float32', new Float32Array(this.predLayers * 1 * this.predHidden).fill(0), [this.predLayers, 1, this.predHidden]),
            c: new this.ort.Tensor('float32', new Float32Array(this.predLayers * 1 * this.predHidden).fill(0), [this.predLayers, 1, this.predHidden]),
            lastToken: this.bosId,
            node: this.lexicon.trie,
            text: ''
        };

        let beam = [initState];

        const F = (featureData && featureData.featureMatrix) ? featureData.featureMatrix.length : 0;
        const stepToFrame = (t) => {
            if (!F) return -1;
            const ratio = F / encodedLen;
            let idx = Math.floor(t * ratio);
            if (idx < 0) idx = 0; if (idx >= F) idx = F-1;
            return idx;
        };

        for (let t = 0; t < encodedLen; t++) {
            const start = t * this.encoderDim;
            const frameVec = encoded.data.slice(start, start + this.encoderDim);
            const encoderFrame = new this.ort.Tensor('float32', frameVec, [1, this.encoderDim, 1]);

            let nextBeam = [];
            for (const hyp of beam) {
                // Copy per-hypothesis states
                let h = hyp.h, c = hyp.c, last = hyp.lastToken, node = hyp.node;
                let emitted = 0;
                while (emitted < symbolsPerStep && nextBeam.length < beamSize * (topK + 1)) {
                    const feeds = {
                        'encoder_outputs': encoderFrame,
                        'targets': new this.ort.Tensor('int32', Int32Array.from([last]), [1, 1]),
                        'target_length': new this.ort.Tensor('int32', Int32Array.from([1]), [1]),
                        'input_states_1': h,
                        'input_states_2': c,
                    };
                    if (this.verbose) {
                        const fd = Object.fromEntries(Object.entries(feeds).map(([k,v])=>[k, v.dims]));
                        console.log('[RNNT] joint feed:', fd);
                    }
                    const out = await this.decoderJointSession.run(feeds);
                    const logitsT = out.outputs; // expected standard name
                    h = out.output_states_1; c = out.output_states_2;
                    const probs = softmax(Array.from(logitsT.data));
                    // Geometry bias warm-up: strong early, decays to zero by 30% of the sequence
                    let allowedSet = null;
                    if (false && F > 0 && typeof featureData.featureMatrix?.[0]?.[0] === 'number') {
                        const warmFrac = 0.3;
                        const warmSteps = Math.max(1, Math.floor(encodedLen * warmFrac));
                        const decay = t < warmSteps ? 1 - (t / warmSteps) : 0;
                        if (decay > 0) {
                            const fi = stepToFrame(t);
                            if (fi >= 0) {
                                const fm = featureData.featureMatrix[fi];
                                const x = fm[0], y = fm[1];
                                const dists = this.keyCenters.map(k => ({ ch: k.char, d: Math.hypot(x - k.x, y - k.y) }));
                                dists.sort((a,b)=>a.d-b.d);
                                const nearest = dists[0]?.ch; const second = dists[1]?.ch;
                                const cid1 = nearest!=null ? this.lexicon.charToId[nearest] : null;
                                const cid2 = second!=null ? this.lexicon.charToId[second] : null;
                                const b1 = 1.8 * decay; const b2 = 0.9 * decay;
                                if (cid1 != null) probs[cid1] += b1;
                                if (cid2 != null) probs[cid2] += b2;
                                allowedSet = new Set([cid1, cid2].filter(v=>v!=null));
                            }
                        }
                    }
                    if (this.verbose) {
                        const tops = Array.from(probs).map((p,i)=>({i,p}))
                          .sort((a,b)=>b.p-a.p).slice(0,8)
                          .map(x=>({ id:x.i, ch: (this.runtimeMeta.tokens?this.runtimeMeta.tokens[x.i]:(this.runtimeMeta.id_to_char?this.runtimeMeta.id_to_char[String(x.i)]:'')), p:+x.p.toFixed(4)}));
                        console.log('[RNNT] top logits:', tops);
                    }

                    // Expand topK non-blank tokens constrained by trie
                    const indexed = probs.map((p, i) => [i, p]);
                    indexed.sort((a, b) => b[1] - a[1]);

                    let expanded = 0;
                    for (let k = 0; k < indexed.length && expanded < topK; k++) {
                        const [tid, p] = indexed[k];
                        if (tid === this.blankId) continue; // handled separately
                        if (unkId >= 0 && tid === unkId) continue; // disallow <unk>
                        if (allowedSet && !allowedSet.has(tid)) continue;
                        const ch = toChar(tid);
                        if (!ch) continue;
                        const cid = this.lexicon.charToId[ch];
                        if (cid == null) continue;
                        if (!node.children.has(cid)) continue; // lexicon constraint
                        // Avoid unintended double letters unless current nearest-key segment matches
                        const prevChar = (hyp.tokens.length>0 && this.runtimeMeta) ? (this.runtimeMeta.tokens ? this.runtimeMeta.tokens[hyp.tokens[hyp.tokens.length-1]] : (this.runtimeMeta.id_to_char ? this.runtimeMeta.id_to_char[String(hyp.tokens[hyp.tokens.length-1])] : '')) : '';
                        const fi = stepToFrame(t);
                        const segChar = (()=>{
                            if (fi < 0 || !featureData || !featureData.featureMatrix) return null;
                            const fm = featureData.featureMatrix[fi];
                            const x = fm[0], y = fm[1];
                            const dists = this.keyCenters.map(k => ({ ch: k.char, d: Math.hypot(x - k.x, y - k.y) }));
                            dists.sort((a,b)=>a.d-b.d);
                            return dists[0]?.ch || null;
                        })();
                        if (prevChar === ch && segChar !== ch) continue;
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

            // Merge duplicates by text + lastToken + node id, then prune
            const uniq = new Map();
            for (const c of nextBeam) {
                const key = c.text + '|' + c.lastToken + '|' + (c.node?.wid ?? -1);
                const prev = uniq.get(key);
                if (!prev || c.score > prev.score) uniq.set(key, c);
            }
            const merged = Array.from(uniq.values());
            merged.sort((a, b) => b.score - a.score);
            beam = merged.slice(0, beamSize);
        }

        // Score completed words with priors
        const priorAlpha = 0.0;
        const completeBonus = 3.0;
        const scored = beam.map(h => {
            let bonus = 0;
            if (h.node && h.node.isWordEnd && h.node.wid >= 0) bonus = h.node.logp || 0;
            const lp = h.score + bonus;
            const lenNorm = lp / Math.pow((h.tokens.length || 1), 1.0 - lengthPenalty);
            const isComplete = !!(h.node && h.node.isWordEnd);
            const endBias = isComplete ? completeBonus : -2.0;
            const priorTerm = priorAlpha * (h.node && h.node.wid>=0 ? (h.node.logp||0) : 0);
            return { text: h.text, tokens: h.tokens, score: lenNorm + endBias + priorTerm, rawScore: lp, isComplete };
        }).sort((a, b) => b.score - a.score);
        const complete = scored.filter(x => x.isComplete);
        return (complete.length ? complete : scored).slice(0, 10);
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = RNNTDecoder;
}
