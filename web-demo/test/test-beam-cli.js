#!/usr/bin/env node
/*
 RNNT Beam Search CLI with detailed logging.
 Usage:
  node web-demo/test/test-beam-cli.js \
    --jsonl data/train_final_val.jsonl \
    --n 3 \
    --encoder web-demo/models/rnnt_new_latest/encoder.onnx \
    --decoder_joint web-demo/models/rnnt_new_latest/decoder_joint.onnx \
    --meta web-demo/models/rnnt_new_latest/runtime_meta.json \
    --words web-demo/words.txt \
    --freqs web-demo/word_frequencies_aligned.json \
    --beam 24 --topK 8 --sps 8 --maxSymbols 24 --frames 64 --debug
*/

const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');

const FeatureExtractor = require('../js/feature-extractor-corrected.js');

function parseArgs() {
  const cfg = {};
  const a = process.argv.slice(2);
  for (let i = 0; i < a.length; i++) {
    const k = a[i];
    const v = a[i + 1];
    if (k.startsWith('--')) {
      const key = k.slice(2);
      if (v && !v.startsWith('--')) { cfg[key] = v; i++; }
      else { cfg[key] = true; }
    }
  }
  return cfg;
}

async function loadJSONL(filePath, n) {
  const lines = fs.readFileSync(filePath, 'utf-8').split(/\r?\n/).filter(Boolean);
  const out = [];
  for (let i = 0; i < Math.min(n, lines.length); i++) {
    try { out.push(JSON.parse(lines[i])); } catch {}
  }
  return out;
}

function logSoftmax(arr) {
  let max = -Infinity; for (const x of arr) if (x > max) max = x;
  let sumExp = 0; for (const x of arr) sumExp += Math.exp(x - max);
  const logZ = Math.log(sumExp) + max;
  return arr.map(x => x - logZ);
}

function buildLexicon(wordsTxt, freqsJson, runtimeMeta, log) {
  const raw = wordsTxt.split(/\r?\n/).map(w => w.trim()).filter(Boolean);
  const logf = freqsJson.log_frequencies || [];
  const words = []; const priors = [];
  const allow = /^[a-z']{2,20}$/;
  const triple = /(.)\1\1/;
  const thr = (L) => L<=2?1e-5: L===3?1e-6: L===4?1e-7: L===5?5e-8: (L<=7?1e-8: L===8?5e-9:1e-9);
  for (let i=0; i<raw.length; i++){
    const w = raw[i]; if (!allow.test(w)) continue; if (triple.test(w)) continue;
    const logp = logf[i] ?? -30; const p = Math.exp(logp); if (p < thr(w.length)) continue;
    words.push(w); priors.push(logp);
  }
  const charToId = runtimeMeta.char_to_id || {};
  const root = { children:new Map(), isWordEnd:false, wid:-1, logp:0 };
  for (let i=0;i<words.length;i++){
    let node = root; let ok=true;
    for (const ch of words[i]){
      const cid = charToId[ch]; if (cid==null){ ok=false; break; }
      if (!node.children.has(cid)) node.children.set(cid, {children:new Map(), isWordEnd:false, wid:-1, logp:0});
      node = node.children.get(cid);
    }
    if (ok){ node.isWordEnd = true; node.wid=i; node.logp=priors[i]; }
  }
  log && console.log(`Lexicon: kept ${words.length}/${raw.length}`);
  return { trie: root, words, priors, charToId };
}

async function main(){
  const cfg = parseArgs();
  const jsonl = cfg.jsonl || 'data/train_final_val.jsonl';
  const n = parseInt(cfg.n || '3',10);
  const encoderPath = cfg.encoder || 'web-demo/models/rnnt_new_latest/encoder.onnx';
  const decoderPath = cfg.decoder_joint || 'web-demo/models/rnnt_new_latest/decoder_joint.onnx';
  const metaPath = cfg.meta || 'web-demo/models/rnnt_new_latest/runtime_meta.json';
  const wordsPath = cfg.words || 'web-demo/words.txt';
  const freqsPath = cfg.freqs || 'web-demo/word_frequencies_aligned.json';
  const beamSize = parseInt(cfg.beam || '32',10);
  const topK = parseInt(cfg.topK || '12',10);
  const sps = parseInt(cfg.sps || '10',10);
  const maxSymbols = parseInt(cfg.maxSymbols || '28',10);
  const lenPenalty = parseFloat(cfg.lenPenalty || '0.1');
  const completeBonus = parseFloat(cfg.completeBonus || '2.0');
  const frameLimit = cfg.frames ? parseInt(cfg.frames,10) : null;
  const debug = !!cfg.debug;

  const runtimeMeta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
  const wordsTxt = fs.readFileSync(wordsPath, 'utf-8');
  const freqsJson = JSON.parse(fs.readFileSync(freqsPath, 'utf-8'));
  const lexicon = buildLexicon(wordsTxt, freqsJson, runtimeMeta, true);

  const so = { executionProviders:['cpu'], graphOptimizationLevel:'all' };
  const [encoder, decoder] = await Promise.all([
    ort.InferenceSession.create(encoderPath, so),
    ort.InferenceSession.create(decoderPath, so),
  ]);
  const encIn = encoder.inputNames || [];
  const audioName = encIn.includes('audio_signal')?'audio_signal':(encIn.includes('features_bft')?'features_bft':encIn[0]);
  const lenName = encIn.includes('length')?'length':(encIn.includes('lengths')?'lengths':(encIn[1]||'length'));
  const featurizer = new FeatureExtractor();
  const samples = await loadJSONL(jsonl, n);

  for (let idx=0; idx<samples.length; idx++){
    const ex = samples[idx];
    console.log(`\n=== Sample ${idx+1}/${samples.length} word='${ex.word}' ===`);
    const {features,numFrames} = featurizer.process(ex.points||[]);
    console.log(`Features: T=${numFrames}, F=37, bytes=${features.byteLength}`);
    const T = frameLimit? Math.min(numFrames, frameLimit) : numFrames; const F=37;
    const trans = new Float32Array(F*T); for(let t=0;t<T;t++) for(let f=0;f<F;f++) trans[f*T+t]=features[t*F+f];
    const feeds = {}; feeds[audioName]=new ort.Tensor('float32',trans,[1,F,T]); feeds[lenName]=new ort.Tensor('int64',BigInt64Array.from([BigInt(T)]),[1]);
    const encOut = await encoder.run(feeds);
    const encoded = encOut.outputs || encOut.encoded_btf || encOut.encoded || encOut.encoder_output;
    const encLen = Number((encOut.encoded_lengths||encOut.length||encOut.lengths).data[0]);
    console.log(`Encoded dims=${encoded.dims} length=${encLen}`);

    const initH = new ort.Tensor('float32', new Float32Array(2*1*320).fill(0), [2,1,320]);
    const initC = new ort.Tensor('float32', new Float32Array(2*1*320).fill(0), [2,1,320]);

    const toChar = (id)=> (runtimeMeta.tokens? runtimeMeta.tokens[id] : (runtimeMeta.id_to_char? runtimeMeta.id_to_char[String(id)] : ''))||'';
    // Start predictor with blank_id to match runtime_meta
    let beam=[{tokens:[], score:0, h:initH, c:initC, last: runtimeMeta.blank_id, node: lexicon.trie, text:''}];

    for (let t=0;t<encLen && t<T;t++){
      const dims = encoded.dims; let frameVec;
      if (dims[1]===256 && dims[2]===encLen){ frameVec = new Float32Array(256); for(let f=0;f<256;f++) frameVec[f]=encoded.data[f*encLen+t]; }
      else if (dims[1]===encLen && dims[2]===256){ const s=t*256; frameVec=encoded.data.slice(s,s+256); }
      else throw new Error('Unexpected encoder dims '+JSON.stringify(dims));
      const encT = new ort.Tensor('float32', frameVec, [1,256,1]);

      let next=[];
      for (const hyp of beam){
        let h=hyp.h, c=hyp.c, last=hyp.last, node=hyp.node;
        let emitted=0;
        while (emitted < sps && next.length < beamSize*(topK+1)){
          const stepFeeds={ 'encoder_outputs':encT, 'targets': new ort.Tensor('int32', Int32Array.from([last]), [1,1]), 'target_length': new ort.Tensor('int32', Int32Array.from([1]), [1]), 'input_states_1':h, 'input_states_2':c };
          const out=await decoder.run(stepFeeds);
          h=out.output_states_1; c=out.output_states_2; const logitsT = out.outputs || out.logits || out.joint_output;
          const ls = logSoftmax(Array.from(logitsT.data));
          if (debug) console.log(`t=${t} hyp='${hyp.text}' blankLogP=${ls[runtimeMeta.blank_id].toFixed(2)} topLogP=${Math.max(...ls).toFixed(2)}`);
          // Expand topK
          const idxp = ls.map((lp,i)=>[i,lp]).sort((a,b)=>b[1]-a[1]);
          let expanded=0; for (let k=0;k<idxp.length && expanded<topK;k++){
            const [tid,lp]=idxp[k]; if (tid===runtimeMeta.blank_id) continue; const ch=toChar(tid); if (!ch) continue;
            const cid = lexicon.charToId[ch]; if (cid==null) continue; if (!node.children.has(cid)) continue;
            const child = node.children.get(cid);
            next.push({ tokens: hyp.tokens.concat([tid]), score: hyp.score+lp, h, c, last:tid, node:child, text: hyp.text+ch });
            expanded++;
          }
          // blank transition
          next.push({ tokens:hyp.tokens, score:hyp.score+ls[runtimeMeta.blank_id], h, c, last: runtimeMeta.blank_id, node, text:hyp.text });
          if (expanded===0) break; else emitted++;
        }
      }
      next.sort((a,b)=>b.score-a.score); beam = next.slice(0, beamSize);
      if (debug){
        const preview = beam.slice(0,5).map(h=>`${h.text}:${h.score.toFixed(2)}`);
        console.log(`t=${t} beam: ${preview.join(' | ')}`);
      }
    }

    const results = beam.map(h=>{
      const isComplete = !!(h.node && h.node.isWordEnd);
      const prior = (isComplete && h.node.wid>=0) ? (lexicon.priors[h.node.wid]||0) : 0;
      const lengthNorm = h.tokens.length>0 ? h.score / Math.pow(h.tokens.length, lenPenalty) : h.score;
      const finalScore = lengthNorm + prior + (isComplete ? completeBonus : 0);
      return { text:h.text, score: finalScore, rawScore: h.score, isComplete };
    }).sort((a,b)=>b.score-a.score);
    const finals = results.filter(r=>r.isComplete);
    const best = (finals.length? finals: results)[0];
    console.log(`Prediction: '${best.text}'  (complete=${best.isComplete})`);
    console.log(`Top candidates: ${(finals.length? finals: results).slice(0,5).map(r=>r.text).join(', ')}`);
  }
}

main().catch(e=>{ console.error(e); process.exit(1); });
