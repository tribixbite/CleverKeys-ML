#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');
const CTCDecoder = require('../js/ctc-decoder.js');
const FeatureExtractor = require('../js/feature-extractor-corrected.js');

async function readWord(datasetPath, word) {
  const fd = fs.openSync(datasetPath, 'r');
  const rl = require('readline').createInterface({ input: fs.createReadStream('', { fd }) });
  for await (const line of rl) {
    if (!line) continue;
    try { const rec = JSON.parse(line); if (rec.word && rec.word.toLowerCase()===word) { fs.closeSync(fd); return rec; } } catch {}
  }
  fs.closeSync(fd);
  return null;
}

async function main(){
  const base = path.resolve(__dirname, '..');
  const modelDir = path.join(base, 'models', 'ctc');
  const dataset = path.join(base, '..', 'data', 'train_final_train.jsonl');
  const word = (process.argv[2] || 'hello').toLowerCase();

  const rec = await readWord(dataset, word);
  if (!rec) { console.error('Word not found in dataset:', word); process.exit(2); }

  global.ort = ort; // make available to decoder
  const dec = new CTCDecoder();
  const fe = new FeatureExtractor();
  // Load keyboard geometry for proper 37D features
  const keyCenters = JSON.parse(fs.readFileSync(path.join(base, 'js', 'key-centers.json'), 'utf-8'));
  fe.keyCenters = keyCenters;
  dec.setFeatureExtractor(fe);
  await dec.initialize(
    path.join(modelDir, 'gesture_model.onnx'),
    path.join(modelDir, 'gesture_model.onnx'),
    path.join(modelDir, 'tokenizer.json')
  );
  const res = await dec.simpleCTCDecode(rec.points);
  console.log('CTC greedy:', res);
  if (dec.beamSearchSimpleCTC) {
    const blank = (dec.tokenizer && dec.tokenizer.special_tokens && typeof dec.tokenizer.special_tokens.blank_id === 'number')
      ? dec.tokenizer.special_tokens.blank_id : undefined;
    const top = await dec.beamSearchSimpleCTC(rec.points, 10, blank);
    console.log('CTC beam:', top);
  }
}
main().catch(e=>{console.error(e); process.exit(1);});
