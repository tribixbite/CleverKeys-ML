/**
 * ONNX Model Interface - Decoupled model-specific logic
 * Handles all ONNX model interactions including encoder, decoder, and beam search
 */

class ONNXModelInterface {
    constructor() {
        this.encoderSession = null;
        this.decoderSession = null;
        this.runtimeMeta = null;
        this.vocabSize = 30;
        this.blankId = 29;
    }

    /**
     * Initialize ONNX sessions with the model files
     */
    async initialize(encoderPath = 'encoder-model.onnx', decoderPath = 'decoder_joint-model.onnx') {
        const sessionOptions = {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
            enableCpuMemArena: true,
            enableMemPattern: true,
            executionMode: 'sequential',
            logSeverityLevel: 2
        };

        console.log('Loading encoder model from:', encoderPath);
        this.encoderSession = await ort.InferenceSession.create(encoderPath, sessionOptions);
        console.log('Encoder inputs:', this.encoderSession.inputNames);
        console.log('Encoder outputs:', this.encoderSession.outputNames);

        console.log('Loading decoder/joint model from:', decoderPath);
        this.decoderSession = await ort.InferenceSession.create(decoderPath, sessionOptions);
        console.log('Decoder inputs:', this.decoderSession.inputNames);
        console.log('Decoder outputs:', this.decoderSession.outputNames);

        // Load runtime metadata
        await this.loadRuntimeMeta();
    }

    /**
     * Load runtime metadata for vocabulary
     */
    async loadRuntimeMeta(metaPath = 'runtime_meta.json') {
        try {
            const response = await fetch(metaPath);
            this.runtimeMeta = await response.json();
            this.vocabSize = this.runtimeMeta.vocab_size || 30;
            this.blankId = this.runtimeMeta.blank_id || 29;
            console.log(`Runtime meta loaded: vocab_size=${this.vocabSize}, blank_id=${this.blankId}`);
        } catch (error) {
            console.warn('Could not load runtime meta, using defaults:', error);
        }
    }

    /**
     * Run encoder inference
     * @param {Float32Array} features - Feature matrix [batch, time, features]
     * @param {number} sequenceLength - Length of the sequence
     * @returns {Object} Encoder outputs
     */
    async runEncoder(features, sequenceLength) {
        const [batchSize, timeSteps, featureDim] = [1, sequenceLength, 37];

        // Reshape features to [batch, features, time] as expected by the model
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
            encoded: outputs.outputs || outputs.encoded || outputs.encoder_output,
            encodedLength: outputs.encoded_lengths || outputs.length
        };
    }

    /**
     * Run decoder/joint network step
     * @param {ort.Tensor} encoderOutput - Encoder output tensor
     * @param {Array} tokens - Previous tokens
     * @param {Array} hiddenState - LSTM hidden state
     * @param {Array} cellState - LSTM cell state
     * @returns {Object} Decoder outputs with logits and states
     */
    async runDecoderStep(encoderOutput, tokens, hiddenState, cellState) {
        const inputs = {
            'encoder_outputs': encoderOutput,
            'targets': new ort.Tensor('int64', new BigInt64Array(tokens.map(t => BigInt(t))), [1, tokens.length])
        };

        // Add hidden/cell states if provided
        if (hiddenState) {
            inputs['hidden'] = new ort.Tensor('float32', hiddenState, [2, 1, 320]);
        }
        if (cellState) {
            inputs['cell'] = new ort.Tensor('float32', cellState, [2, 1, 320]);
        }

        const outputs = await this.decoderSession.run(inputs);
        return {
            logits: outputs.logits || outputs.outputs || outputs.joint_output,
            hidden: outputs.hidden || hiddenState,
            cell: outputs.cell || cellState
        };
    }

    /**
     * Beam search decoding
     * @param {Array} featureMatrix - 2D array of features [time][features]
     * @param {Object} config - Beam search configuration
     * @returns {Array} Array of hypotheses with scores
     */
    async beamSearch(featureMatrix, config = {}) {
        const {
            beamSize = 10,
            maxSymbols = 20,
            symbolsPerStep = 3,
            topK = 5,
            lengthPenalty = 0.0,
            temperature = 1.0
        } = config;

        const T = featureMatrix.length;
        const F = featureMatrix[0].length;

        // Flatten feature matrix for encoder
        const features = new Float32Array(T * F);
        for (let t = 0; t < T; t++) {
            for (let f = 0; f < F; f++) {
                features[t * F + f] = featureMatrix[t][f];
            }
        }

        // Run encoder
        const encoderOut = await this.runEncoder(features, T);
        const encoded = encoderOut.encoded;
        const encodedLen = encoderOut.encodedLength.data[0];

        // Initialize beam with blank hypothesis
        let beam = [{
            tokens: [],
            score: 0.0,
            hidden: null,
            cell: null,
            text: ''
        }];

        // Beam search main loop
        for (let step = 0; step < Math.min(encodedLen, maxSymbols); step++) {
            let nextBeam = [];

            for (const hyp of beam) {
                // Get encoder frame for this step
                const frameStart = step * encoded.dims[2];
                const frameEnd = (step + 1) * encoded.dims[2];
                const encoderFrame = new ort.Tensor(
                    'float32',
                    encoded.data.slice(frameStart, frameEnd),
                    [1, 1, encoded.dims[2]]
                );

                // Run decoder step
                const decoderOut = await this.runDecoderStep(
                    encoderFrame,
                    hyp.tokens.length > 0 ? hyp.tokens : [this.blankId],
                    hyp.hidden,
                    hyp.cell
                );

                // Get top K tokens
                const logits = decoderOut.logits.data;
                const probs = this.softmax(logits, temperature);
                const topKTokens = this.getTopK(probs, Math.min(topK, this.vocabSize));

                // Expand beam with top K tokens
                for (const [tokenId, prob] of topKTokens) {
                    if (tokenId === this.blankId) continue; // Skip blank token

                    const newTokens = [...hyp.tokens, tokenId];
                    const newScore = hyp.score + Math.log(prob + 1e-10);
                    const newText = this.tokensToText(newTokens);

                    nextBeam.push({
                        tokens: newTokens,
                        score: newScore,
                        hidden: decoderOut.hidden,
                        cell: decoderOut.cell,
                        text: newText
                    });
                }

                // Also consider continuing with blank
                nextBeam.push({
                    tokens: hyp.tokens,
                    score: hyp.score + Math.log(probs[this.blankId] + 1e-10),
                    hidden: decoderOut.hidden,
                    cell: decoderOut.cell,
                    text: hyp.text
                });
            }

            // Prune beam to top beamSize hypotheses
            nextBeam.sort((a, b) => b.score - a.score);
            beam = nextBeam.slice(0, beamSize);

            // Early stopping if all hypotheses end with EOS or reach max length
            if (beam.every(h => h.tokens.length >= maxSymbols)) break;
        }

        // Apply length penalty and return final hypotheses
        return beam.map(hyp => ({
            text: hyp.text,
            tokens: hyp.tokens,
            score: hyp.score / Math.pow(hyp.tokens.length + 1, lengthPenalty),
            rawScore: hyp.score
        })).sort((a, b) => b.score - a.score);
    }

    /**
     * Softmax with temperature
     */
    softmax(logits, temperature = 1.0) {
        const scaled = logits.map(x => x / temperature);
        const maxLogit = Math.max(...scaled);
        const exp = scaled.map(x => Math.exp(x - maxLogit));
        const sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(x => x / sum);
    }

    /**
     * Get top K elements from array
     */
    getTopK(arr, k) {
        const indexed = arr.map((val, idx) => [idx, val]);
        indexed.sort((a, b) => b[1] - a[1]);
        return indexed.slice(0, k);
    }

    /**
     * Convert token IDs to text
     */
    tokensToText(tokens) {
        if (!this.runtimeMeta || !this.runtimeMeta.tokens) {
            // Fallback mapping
            const chars = " 'abcdefghijklmnopqrstuvwxyz";
            return tokens.map(t => chars[t] || '').join('');
        }
        return tokens.map(t => this.runtimeMeta.tokens[t] || '').join('');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ONNXModelInterface;
}