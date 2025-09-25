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
        console.warn('Beam search is not implemented in this version, falling back to greedy decode.');
        return this.greedyDecode(featureData, config.maxSymbols);
    }
}
