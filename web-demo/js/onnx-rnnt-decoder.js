/**
 * RNN-T Decoder with Stateful LSTM Management
 * Implements proper RNN-Transducer decoding with explicit state handling
 */

class RNNTDecoder {
    constructor() {
        this.encoderSession = null;
        this.decoderSession = null;
        this.jointSession = null;
        this.runtimeMeta = null;
        this.vocabSize = 30;
        this.blankId = 29;
        this.decoderConfig = null;
    }

    /**
     * Initialize ONNX sessions with the stateful models
     */
    async initialize(encoderPath = 'models/encoder.onnx', decoderPath = 'models/decoder.onnx', jointPath = 'models/joint.onnx') {
        const sessionOptions = {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
            enableCpuMemArena: true,
            enableMemPattern: true,
            executionMode: 'sequential',
            logSeverityLevel: 2
        };

        console.log('Loading RNN-T models...');

        // Load encoder
        console.log('Loading encoder from:', encoderPath);
        this.encoderSession = await ort.InferenceSession.create(encoderPath, sessionOptions);
        console.log('Encoder loaded. Inputs:', this.encoderSession.inputNames, 'Outputs:', this.encoderSession.outputNames);

        // Load stateful decoder
        console.log('Loading decoder from:', decoderPath);
        this.decoderSession = await ort.InferenceSession.create(decoderPath, sessionOptions);
        console.log('Decoder loaded. Inputs:', this.decoderSession.inputNames, 'Outputs:', this.decoderSession.outputNames);

        // Load joint network
        console.log('Loading joint network from:', jointPath);
        this.jointSession = await ort.InferenceSession.create(jointPath, sessionOptions);
        console.log('Joint loaded. Inputs:', this.jointSession.inputNames, 'Outputs:', this.jointSession.outputNames);

        // Load runtime metadata
        await this.loadRuntimeMeta();
    }

    /**
     * Load runtime metadata with decoder configuration
     */
    async loadRuntimeMeta(metaPath = 'models/runtime_meta.json') {
        try {
            const response = await fetch(metaPath);
            this.runtimeMeta = await response.json();
            this.vocabSize = this.runtimeMeta.vocab_size || 30;
            this.blankId = this.runtimeMeta.blank_id || 29;
            this.decoderConfig = this.runtimeMeta.decoder_config || {
                num_layers: 2,
                hidden_size: 320,
                encoder_dim: 256,
                decoder_dim: 320
            };
            console.log('Runtime meta loaded:', {
                vocabSize: this.vocabSize,
                blankId: this.blankId,
                decoderConfig: this.decoderConfig
            });
        } catch (error) {
            console.warn('Could not load runtime meta, using defaults:', error);
            this.decoderConfig = {
                num_layers: 2,
                hidden_size: 320,
                encoder_dim: 256,
                decoder_dim: 320
            };
        }
    }

    /**
     * Initialize LSTM states to zeros
     */
    initializeStates(batchSize = 1) {
        const { num_layers, hidden_size } = this.decoderConfig;
        return {
            h: new Float32Array(num_layers * batchSize * hidden_size),
            c: new Float32Array(num_layers * batchSize * hidden_size)
        };
    }

    /**
     * Run encoder on input features
     */
    async runEncoder(features, sequenceLength) {
        const [batchSize, timeSteps, featureDim] = [1, sequenceLength, 37];

        // Reshape features to [batch, features, time] as expected by Conformer
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

        const outputs = await this.encoderSession.run(inputs);
        return {
            encoded: outputs.encoded,
            encodedLength: outputs.encoded_lengths.data[0]
        };
    }

    /**
     * Run one step of the stateful decoder
     */
    async runDecoderStep(inputToken, hState, cState) {
        const { num_layers, hidden_size } = this.decoderConfig;
        const batchSize = 1;

        const inputs = {
            'input_tokens': new ort.Tensor('int64', new BigInt64Array([BigInt(inputToken)]), [batchSize, 1]),
            'h_in': new ort.Tensor('float32', hState, [num_layers, batchSize, hidden_size]),
            'c_in': new ort.Tensor('float32', cState, [num_layers, batchSize, hidden_size])
        };

        const outputs = await this.decoderSession.run(inputs);
        return {
            decoderOutput: outputs.decoder_output,
            hOut: outputs.h_out.data,
            cOut: outputs.c_out.data
        };
    }

    /**
     * Run joint network to combine encoder and decoder outputs
     */
    async runJoint(encoderFrame, decoderOutput) {
        const inputs = {
            'encoder_output': encoderFrame,
            'decoder_output': decoderOutput
        };

        const outputs = await this.jointSession.run(inputs);
        return outputs.logits;
    }

    /**
     * Greedy decoding (beam size = 1)
     */
    async greedyDecode(featureMatrix, maxSymbols = 20) {
        const T = featureMatrix.length;
        const F = featureMatrix[0].length;

        // Flatten feature matrix for encoder
        const features = new Float32Array(T * F);
        for (let t = 0; t < T; t++) {
            for (let f = 0; f < F; f++) {
                features[t * F + f] = featureMatrix[t][f];
            }
        }

        // Run encoder once
        console.log('Running encoder...');
        const encoderOut = await this.runEncoder(features, T);
        const encoded = encoderOut.encoded;
        const encodedLen = encoderOut.encodedLength;
        console.log(`Encoder output shape: [${encoded.dims}], length: ${encodedLen}`);

        // Initialize decoder state
        const states = this.initializeStates();
        const tokens = [];
        let currentToken = this.blankId; // Start with blank token

        // Process each encoder frame
        const seqLength = typeof encodedLen === 'bigint' ? Number(encodedLen) : encodedLen;
        for (let t = 0; t < Math.min(seqLength, maxSymbols); t++) {
            // Extract encoder frame at time t
            const frameData = new Float32Array(this.decoderConfig.encoder_dim);
            for (let i = 0; i < this.decoderConfig.encoder_dim; i++) {
                frameData[i] = encoded.data[t * this.decoderConfig.encoder_dim + i];
            }
            const encoderFrame = new ort.Tensor('float32', frameData, [1, 1, this.decoderConfig.encoder_dim]);

            // Run decoder step
            const decoderOut = await this.runDecoderStep(currentToken, states.h, states.c);
            states.h = decoderOut.hOut;
            states.c = decoderOut.cOut;

            // Run joint network
            const logits = await this.runJoint(encoderFrame, decoderOut.decoderOutput);

            // Get argmax prediction
            const probs = this.softmax(logits.data);
            const nextToken = this.argmax(probs);

            // Update current token and accumulate if not blank
            if (nextToken !== this.blankId) {
                tokens.push(nextToken);
                currentToken = nextToken;
            }

            // Early stopping if we've generated enough tokens
            if (tokens.length >= maxSymbols) break;
        }

        return {
            tokens: tokens,
            text: this.tokensToText(tokens)
        };
    }

    /**
     * Beam search decoding with proper state management
     */
    async beamSearch(featureMatrix, config = {}) {
        const {
            beamSize = 5,
            maxSymbols = 20,
            topK = 5,
            lengthPenalty = 0.0,
            temperature = 1.0
        } = config;

        const T = featureMatrix.length;
        const F = featureMatrix[0].length;

        // Flatten feature matrix
        const features = new Float32Array(T * F);
        for (let t = 0; t < T; t++) {
            for (let f = 0; f < F; f++) {
                features[t * F + f] = featureMatrix[t][f];
            }
        }

        // Run encoder once
        const encoderOut = await this.runEncoder(features, T);
        const encoded = encoderOut.encoded;
        const encodedLen = encoderOut.encodedLength;

        // Initialize beam with single empty hypothesis
        let beam = [{
            tokens: [],
            score: 0.0,
            states: this.initializeStates(),
            lastToken: this.blankId,
            text: ''
        }];

        // Process each encoder frame
        const seqLength = typeof encodedLen === 'bigint' ? Number(encodedLen) : encodedLen;
        for (let t = 0; t < Math.min(seqLength, maxSymbols); t++) {
            let nextBeam = [];

            // Extract encoder frame
            const frameData = new Float32Array(this.decoderConfig.encoder_dim);
            for (let i = 0; i < this.decoderConfig.encoder_dim; i++) {
                frameData[i] = encoded.data[t * this.decoderConfig.encoder_dim + i];
            }
            const encoderFrame = new ort.Tensor('float32', frameData, [1, 1, this.decoderConfig.encoder_dim]);

            // Expand each hypothesis in the beam
            for (const hyp of beam) {
                // Run decoder with hypothesis state
                const decoderOut = await this.runDecoderStep(hyp.lastToken, hyp.states.h, hyp.states.c);

                // Run joint network
                const logits = await this.runJoint(encoderFrame, decoderOut.decoderOutput);

                // Get top K tokens
                const probs = this.softmax(logits.data, temperature);
                const topKTokens = this.getTopK(probs, Math.min(topK, this.vocabSize));

                // Expand beam with non-blank tokens
                for (const [tokenId, prob] of topKTokens) {
                    if (tokenId === this.blankId) {
                        // Keep hypothesis unchanged with blank emission
                        nextBeam.push({
                            tokens: hyp.tokens,
                            score: hyp.score + Math.log(prob + 1e-10),
                            states: { h: decoderOut.hOut, c: decoderOut.cOut },
                            lastToken: hyp.lastToken,
                            text: hyp.text
                        });
                    } else {
                        // Emit new token
                        const newTokens = [...hyp.tokens, tokenId];
                        nextBeam.push({
                            tokens: newTokens,
                            score: hyp.score + Math.log(prob + 1e-10),
                            states: { h: decoderOut.hOut, c: decoderOut.cOut },
                            lastToken: tokenId,
                            text: this.tokensToText(newTokens)
                        });
                    }
                }
            }

            // Prune beam to top beamSize hypotheses
            nextBeam.sort((a, b) => b.score - a.score);
            beam = nextBeam.slice(0, beamSize);

            // Early stopping if all hypotheses have enough tokens
            if (beam.every(h => h.tokens.length >= maxSymbols)) break;
        }

        // Apply length penalty and return final hypotheses
        return beam.map(hyp => ({
            text: hyp.text,
            tokens: hyp.tokens,
            score: hyp.score / Math.pow(Math.max(hyp.tokens.length, 1), lengthPenalty),
            rawScore: hyp.score
        })).sort((a, b) => b.score - a.score);
    }

    /**
     * Helper functions
     */

    softmax(logits, temperature = 1.0) {
        const scaled = Array.from(logits).map(x => x / temperature);
        const maxLogit = Math.max(...scaled);
        const exp = scaled.map(x => Math.exp(x - maxLogit));
        const sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(x => x / sum);
    }

    argmax(arr) {
        let maxIdx = 0;
        let maxVal = arr[0];
        for (let i = 1; i < arr.length; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    getTopK(arr, k) {
        const indexed = arr.map((val, idx) => [idx, val]);
        indexed.sort((a, b) => b[1] - a[1]);
        return indexed.slice(0, k);
    }

    tokensToText(tokens) {
        if (!this.runtimeMeta || !this.runtimeMeta.tokens) {
            // Fallback mapping
            const chars = " 'abcdefghijklmnopqrstuvwxyz";
            return tokens.map(t => t < chars.length ? chars[t] : '').join('');
        }
        return tokens.map(t => this.runtimeMeta.tokens[t] || '').join('');
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RNNTDecoder;
}