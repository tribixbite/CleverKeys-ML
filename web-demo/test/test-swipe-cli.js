#!/usr/bin/env node
/*
 Node CLI to test RNN-T encode/decode against JSONL swipes
 Usage:
   node web-demo/test/test-swipe-cli.js \
     --jsonl data/train_final_val.jsonl \
     --n 3 \
     --encoder web-demo/models/rnnt_new/encoder.onnx \
     --decoder_joint web-demo/models/rnnt_new/decoder_joint.onnx \
     --meta web-demo/models/runtime_meta.json
*/

const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');

const FeatureExtractor = require('../js/feature-extractor-corrected.js');

async function loadJSONL(filePath, n = 5) {
  const lines = fs.readFileSync(filePath, 'utf-8').split(/\r?\n/).filter(Boolean);
  const out = [];
  for (let i = 0; i < Math.min(n, lines.length); i++) {
    try { out.push(JSON.parse(lines[i])); } catch {}
  }
  return out;
}

function parseArgs() {
  const args = process.argv.slice(2);
  const cfg = {};
  for (let i = 0; i < args.length; i += 2) {
    const k = args[i]; const v = args[i + 1];
    if (!k) break;
    if (k.startsWith('--')) cfg[k.slice(2)] = v;
  }
  return cfg;
}

async function main() {
  const cwd = process.cwd();
  const cfg = parseArgs();
  const jsonl = cfg.jsonl || path.join(cwd, 'data/train_final_val.jsonl');
  const count = parseInt(cfg.n || '3', 10);

  const encoderPath = cfg.encoder || path.join(cwd, 'web-demo/models/rnnt_new/encoder.onnx');
  const decoderJointPath = cfg.decoder_joint || path.join(cwd, 'web-demo/models/rnnt_new/decoder_joint.onnx');
  const metaPathPreferred = cfg.meta || path.join(cwd, 'web-demo/models/rnnt_new/runtime_meta.json');
  const metaPathFallback = path.join(cwd, 'web-demo/models/runtime_meta.json');

  // Load metadata (fallback if truncated)
  let meta;
  try { meta = JSON.parse(fs.readFileSync(metaPathPreferred, 'utf-8')); }
  catch { meta = JSON.parse(fs.readFileSync(metaPathFallback, 'utf-8')); }

  const featurizer = new FeatureExtractor();

  // Prepare ONNX sessions
  const sessionOptions = { executionProviders: ['cpu'], graphOptimizationLevel: 'all' };
  const [encoder, decoderJoint] = await Promise.all([
    ort.InferenceSession.create(encoderPath, sessionOptions),
    ort.InferenceSession.create(decoderJointPath, sessionOptions),
  ]);

  // Flexible encoder I/O names
  const encIn = encoder.inputNames || [];
  const audioName = encIn.includes('audio_signal') ? 'audio_signal' : (encIn.includes('features_bft') ? 'features_bft' : encIn[0]);
  const lenName = encIn.includes('length') ? 'length' : (encIn.includes('lengths') ? 'lengths' : (encIn[1] || 'length'));

  const samples = await loadJSONL(jsonl, count);
  let correct = 0;
  for (let i = 0; i < samples.length; i++) {
    const ex = samples[i];
    const word = ex.word || '';
    const { features, numFrames } = featurizer.process(ex.points || []);

    // Transpose to [B, F, T]
    const featDim = 37, T = numFrames;
    const transposed = new Float32Array(featDim * T);
    for (let t = 0; t < T; t++) for (let f = 0; f < featDim; f++) transposed[f * T + t] = features[t * featDim + f];

    const encFeeds = {};
    encFeeds[audioName] = new ort.Tensor('float32', transposed, [1, featDim, T]);
    encFeeds[lenName] = new ort.Tensor('int64', BigInt64Array.from([BigInt(T)]), [1]);
    const encOut = await encoder.run(encFeeds);
    const encoded = encOut.outputs || encOut.encoded_btf || encOut.encoded || encOut.encoder_output;
    const encLenTensor = encOut.encoded_lengths || encOut.length || encOut.lengths;
    const encLen = Number(encLenTensor.data[0]);

    // Greedy decode using decoder_joint
    const blankId = meta.blank_id ?? 29;
    const vocabSize = meta.vocab_size ?? 30;
    // Some RNNT exports expect first predictor input to be 0 ('<blank>' label) rather than blankId
    let lastToken = 0;
    let tokens = [];
    let h = new ort.Tensor('float32', new Float32Array(2 * 1 * 320).fill(0), [2, 1, 320]);
    let c = new ort.Tensor('float32', new Float32Array(2 * 1 * 320).fill(0), [2, 1, 320]);

    for (let t = 0; t < encLen; t++) {
      // Extract frame [1, 256, 1] from either [1, 256, T] or [1, T, 256]
      const dims = encoded.dims;
      let frameVec;
      if (dims[1] === 256 && dims[2] === encLen) {
        frameVec = new Float32Array(256);
        for (let f = 0; f < 256; f++) frameVec[f] = encoded.data[f * encLen + t];
      } else if (dims[1] === encLen && dims[2] === 256) {
        const start = t * 256;
        frameVec = encoded.data.slice(start, start + 256);
      } else {
        throw new Error(`Unexpected encoder dims: ${JSON.stringify(dims)}`);
      }
      const encoderFrame = new ort.Tensor('float32', frameVec, [1, 256, 1]);
      const decInToken = new ort.Tensor('int32', Int32Array.from([lastToken]), [1, 1]);
      const targetLen = new ort.Tensor('int32', Int32Array.from([1]), [1]);

      let feeds = {
        'encoder_outputs': encoderFrame,
        'targets': decInToken,
        'target_length': targetLen,
        'input_states_1': h,
        'input_states_2': c,
      };
      // RNNT Greedy: emit multiple symbols per frame until blank
      const maxSymbolsPerFrame = 6;
      let emitted = 0;
      while (emitted < maxSymbolsPerFrame) {
        const out = await decoderJoint.run(feeds);
        const logitsT = out.outputs || out.logits || out.joint_output;
        h = out.output_states_1; c = out.output_states_2;

        // Argmax
        let best = 0; let bestVal = -Infinity;
        const logits = logitsT.data;
        for (let j = 0; j < vocabSize; j++) if (logits[j] > bestVal) { bestVal = logits[j]; best = j; }
        if (best === blankId || best === 0) break;
        tokens.push(best); lastToken = best; emitted++;
        // Prepare next symbol prediction at same time step
        feeds.targets = new ort.Tensor('int32', Int32Array.from([lastToken]), [1, 1]);
        feeds.input_states_1 = h; feeds.input_states_2 = c;
      }
    }

    const id2char = (id) => (meta.tokens ? meta.tokens[id] : (meta.id_to_char ? meta.id_to_char[String(id)] : '')) || '';
    const pred = tokens.map(id2char).join('');
    const ok = pred === word;
    if (ok) correct++;
    console.log(`[${i+1}/${samples.length}] word='${word}' pred='${pred}' ${ok ? '✓' : '✗'}`);
  }

  console.log(`\nAccuracy: ${correct}/${samples.length}`);
}

main().catch(e => { console.error(e); process.exit(1); });
