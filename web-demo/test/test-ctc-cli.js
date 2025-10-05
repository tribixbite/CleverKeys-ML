#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');
const CTCDecoder = require('../js/ctc-decoder.js');

async function readWord(datasetPath, word) {
  const lines = fs.readFileSync(datasetPath, 'utf-8').trim().split(/\n+/);
  for (const line of lines) {
    try { const rec = JSON.parse(line); if (rec.word && rec.word.toLowerCase()===word) return rec; } catch {}
  }
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
  await dec.initialize(
    path.join(modelDir, 'gesture_model.onnx'),
    path.join(modelDir, 'gesture_model.onnx'),
    path.join(modelDir, 'tokenizer.json')
  );
  const res = await dec.simpleCTCDecode(rec.points);
  console.log('CTC greedy:', res);
  if (dec.beamSearchSimpleCTC) {
    const top = await dec.beamSearchSimpleCTC(rec.points, 10, 0);
    console.log('CTC beam:', top);
  }
}
main().catch(e=>{console.error(e); process.exit(1);});

