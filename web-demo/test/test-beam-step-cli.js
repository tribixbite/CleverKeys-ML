#!/usr/bin/env node
const fs=require('fs');
const ort=require('onnxruntime-node');
const FeatureExtractor=require('../js/feature-extractor-corrected.js');

function parseArgs(){ const cfg={}; const a=process.argv.slice(2); for(let i=0;i<a.length;i++){const k=a[i],v=a[i+1]; if(k.startsWith('--')){ const key=k.slice(2); if(v && !v.startsWith('--')){cfg[key]=v; i++;} else cfg[key]=true; } } return cfg; }
function softmax(arr){ let m=-Infinity; for(const x of arr) if(x>m) m=x; const ex=arr.map(x=>Math.exp(x-m)); const s=ex.reduce((a,b)=>a+b,0); return ex.map(x=>x/s); }
function buildLex(wordsTxt,freqsJson,charToId){ const raw=wordsTxt.split(/\r?\n/).map(w=>w.trim()).filter(Boolean); const logs=freqsJson.log_frequencies||[]; const words=[],priors=[]; const allow=/^[a-z']{2,20}$/; const triple=/(.)\1\1/; const thr=L=>L<=2?1e-5:L===3?1e-6:L===4?1e-7:L===5?5e-8:(L<=7?1e-8:L===8?5e-9:1e-9); const root={ch:new Map(),is:false,wid:-1,logp:0}; for(let i=0;i<raw.length;i++){ const w=raw[i]; if(!allow.test(w)) continue; if(triple.test(w)) continue; const p=Math.exp(logs[i]??-30); if(p<thr(w.length)) continue; let node=root,ok=true; for(const ch of w){ const cid=charToId[ch]; if(cid==null){ ok=false; break;} if(!node.ch.has(cid)) node.ch.set(cid,{ch:new Map(),is:false,wid:-1,logp:0}); node=node.ch.get(cid);} if(ok){ node.is=true; node.wid=words.length; node.logp=logs[i]??-30; words.push(w); priors.push(node.logp); } } return {root,words,priors}; }

async function main(){
 const cfg=parseArgs();
 const jsonl=cfg.jsonl||'data/train_final_val.jsonl';
 const n=parseInt(cfg.n||'10',10);
 const encoderPath=cfg.encoder||'web-demo/models/rnnt_new_latest/encoder.onnx';
 const stepPath=cfg.step||'web-demo/models/rnnt_new_latest/rnnt_step_fp32.onnx';
 const metaPath=cfg.meta||'web-demo/models/rnnt_new_latest/runtime_meta.json';
 const wordsPath=cfg.words||'web-demo/words.txt';
 const freqsPath=cfg.freqs||'web-demo/word_frequencies_aligned.json';
 const beamSize=parseInt(cfg.beam||'24',10), topK=parseInt(cfg.topK||'8',10), sps=parseInt(cfg.sps||'8',10), maxSymbols=parseInt(cfg.maxSymbols||'24',10);
 const debug=!!cfg.debug;

 const runtimeMeta=JSON.parse(fs.readFileSync(metaPath,'utf-8'));
 const wordsTxt=fs.readFileSync(wordsPath,'utf-8');
 const freqsJson=JSON.parse(fs.readFileSync(freqsPath,'utf-8'));
 const lex=buildLex(wordsTxt,freqsJson,runtimeMeta.char_to_id||{});
 console.log(`Lexicon: kept ${lex.words.length}`);

 const enc=await ort.InferenceSession.create(encoderPath);
 const step=await ort.InferenceSession.create(stepPath);
 const fe=new FeatureExtractor();
 const samples=fs.readFileSync(jsonl,'utf-8').trim().split(/\n/).slice(0,n).map(l=>JSON.parse(l));

 for(let si=0; si<samples.length; si++){
  const s=samples[si]; const J=fe.process(s.points); const F=37, T=J.numFrames; const trans=new Float32Array(F*T); for(let t=0;t<T;t++) for(let f=0;f<F;f++) trans[f*T+t]=J.features[t*F+f];
  const encOut=await enc.run({'audio_signal': new ort.Tensor('float32',trans,[1,F,T]), 'length': new ort.Tensor('int64', BigInt64Array.from([BigInt(T)]), [1])});
  const outTensor = encOut.outputs || encOut.encoded_btf || encOut.encoded || encOut.encoder_output;
  const bdt=outTensor;
  let D, Tout;
  if (bdt.dims[1]===256){ D=256; Tout=bdt.dims[2]; }
  else if (bdt.dims[2]===256){ D=256; Tout=bdt.dims[1]; }
  else { throw new Error(`Unexpected encoder dims ${bdt.dims}`); }
  console.log(`\n=== ${si+1}/${samples.length} '${s.word}' T=${T} -> T_out=${Tout}, D=${D}`);
  let beam=[{y: runtimeMeta.blank_id, h: new ort.Tensor('float32', new Float32Array(2*1*320), [2,1,320]), c: new ort.Tensor('float32', new Float32Array(2*1*320), [2,1,320]), tr: lex.root, lp:0, text:''}];
  for(let t=0;t<Tout;t++){
    let next=[];
    for(const hyp of beam){
      let h=hyp.h, c=hyp.c, last=hyp.y, node=hyp.tr; let emitted=0;
      while(emitted<sps && next.length<beamSize*(topK+1)){
        // Slice enc_t for all N? Here we do one-by-one for simplicity
        const frame = new Float32Array(D);
        if (bdt.dims[1]===256){ for(let d=0; d<D; d++) frame[d]=bdt.data[ d*Tout + t ]; }
        else { const start=t*D; for(let d=0; d<D; d++) frame[d]=bdt.data[start+d]; }
        const out = await step.run({ 'y_prev': new ort.Tensor('int64', BigInt64Array.from([BigInt(last)]), [1]), 'h0': h, 'c0': c, 'enc_t': new ort.Tensor('float32', frame, [1,D])});
        const logits= Array.from(out.logits.data);
        h=out.h1; c=out.c1;
        // Blank transition
        next.push({ y: runtimeMeta.blank_id, h, c, tr: node, lp: hyp.lp + logits[runtimeMeta.blank_id], text: hyp.text });
        // Trie-constrained expansions
        const allowed=[...node.ch.keys()];
        if(allowed.length){
          allowed.sort((a,b)=>logits[b]-logits[a]);
          for(const cid of allowed.slice(0,topK)){
            const ch = runtimeMeta.id_to_char[String(cid)] || '';
            next.push({ y: cid, h, c, tr: node.ch.get(cid), lp: hyp.lp + logits[cid], text: hyp.text + ch });
          }
        }
        emitted++;
      }
    }
    next.sort((a,b)=>b.lp-a.lp); beam = next.slice(0,beamSize);
    if(debug){ console.log(`t=${t} beam: ${beam.slice(0,5).map(b=>b.text+':'+b.lp.toFixed(2)).join(' | ')}`); }
  }
  // Collect completes
  const fin = beam.filter(b=>b.tr && b.tr.is);
  const top = (fin.length? fin: beam).slice(0,5);
  console.log(`Prediction: '${top[0].text}' complete=${!!(fin.length)}`);
 }
}

main().catch(e=>{ console.error(e); process.exit(1); });
