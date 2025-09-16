import * as ort from "onnxruntime-web";

type ND = ort.Tensor;

export type TrieNode = {
  children: Map<number, TrieNode>;  // charId -> node
  isWord: boolean;
  wordId: number | -1;             // index into words[] if terminal
};
export type Trie = { root: TrieNode; };
export type Lexicon = {
  words: string[];                 // id -> word
  charToId: Map<string, number>;   // 'a'->0 ... apostrophe->?
  idToChar: string[];
  trie: Trie;
  // Optional: word log priors (natural log, already normalized)
  wordLogPrior?: Float32Array;     // length = words.length
};

// Build a trie from a word list and char vocab mapping
export function buildTrie(words: string[], charToId: Map<string, number>): Trie {
  const node = (): TrieNode => ({ children: new Map(), isWord: false, wordId: -1 });
  const root = node();
  for (let wid = 0; wid < words.length; wid++) {
    let cur = root;
    for (const ch of words[wid]) {
      const cid = charToId.get(ch);
      if (cid === undefined) { cur = null as any; break; } // skip OOV word
      if (!cur.children.has(cid)) cur.children.set(cid, node());
      cur = cur.children.get(cid)!;
    }
    if (cur) { cur.isWord = true; cur.wordId = wid; }
  }
  return { root };
}

type Beam = {
  // RNNT state
  yPrev: bigint;             // last emitted char id (int64)
  h: ND;                     // (L,B=1,H)
  c: ND;                     // (L,B=1,H)
  // Lexical state
  trie: TrieNode;            // current trie node
  // Scoring
  logp: number;              // running RNNT log-prob
  // Output tokens (optional; we can reconstruct by trie path too)
  chars: number[];
};

// Utility: new zero tensor
function zeros(shape: number[], dtype: "float32"|"int32"|"int64") {
  const size = shape.reduce((a,b)=>a*b,1);
  let data;
  if (dtype === "float32") data = new Float32Array(size);
  else if (dtype === "int32") data = new Int32Array(size);
  else data = new BigInt64Array(size);
  return new ort.Tensor(dtype as any, data, shape);
}

// Argmax topK indices on a row-major 2D buffer
function topKRow(row: Float32Array, K: number): { idx: number; val: number }[] {
  const heap: { idx: number; val: number }[] = [];
  for (let i = 0; i < row.length; i++) {
    const v = row[i];
    if (heap.length < K) { heap.push({ idx: i, val: v }); heap.sort((a,b)=>a.val-b.val); }
    else if (v > heap[0].val) { heap[0] = { idx: i, val: v }; heap.sort((a,b)=>a.val-b.val); }
  }
  heap.sort((a,b)=>b.val-a.val);
  return heap;
}

/**
 * Prefix-constrained RNNT beam search for a single swipe (B=1).
 * - Encoder: features_bft (1,F,T), lengths (1)
 * - Step: batched over beams: y_prev (N), h0/c0 (L,N,H), enc_t (N,D)
 * Returns top-K candidate words with scores.
 */
export async function rnntBeamSearchWord(
  encoder: ort.InferenceSession,
  step: ort.InferenceSession,
  featuresBFT: Float32Array, F: number, T: number,
  L: number, H: number, D: number,
  lex: Lexicon,
  {
    blankId = 0,
    beamSize = 16,
    prunePerBeam = 6,       // max child chars expanded per beam per inner step
    maxSymbols = 20,        // cap on inner symbols per frame
    lmLambda = 0.4,         // weight for word log prior at end
    returnTopK = 5
  }: {
    blankId?: number; beamSize?: number; prunePerBeam?: number;
    maxSymbols?: number; lmLambda?: number; returnTopK?: number;
  } = {}
) {
  // 1) Run encoder
  const encOut = await encoder.run({
    "features_bft": new ort.Tensor("float32", featuresBFT, [1, F, T]),
    "lengths":      new ort.Tensor("int32",   new Int32Array([T]), [1]),
  });
  const encBTF = encOut["encoded_btf"] as ND;  // (1,T_out,D)
  const T_out = encBTF.dims[1];
  const encData = encBTF.data as Float32Array;

  // 2) init beam state (B=1 -> beams dimension only)
  let beams: Beam[] = [{
    yPrev: BigInt(blankId),
    h: zeros([L, 1, H], "float32"),
    c: zeros([L, 1, H], "float32"),
    trie: lex.trie.root,
    logp: 0,
    chars: []
  }];

  // Temporary buffers reused across loop
  const enc_t_batch = new Float32Array(beamSize * D);

  // 3) time-synchronous beam search
  for (let t = 0; t < T_out; t++) {
    // (a) inner label loop with prefix constraint
    for (let s = 0; s < maxSymbols; s++) {
      // Batch current beams into one step call
      const N = Math.min(beamSize, beams.length);
      // expand only top N beams by score
      beams.sort((a,b)=>b.logp - a.logp);
      const active = beams.slice(0, N);

      // Prepare step inputs
      const yPrevArr = new BigInt64Array(N);
      const hStack = zeros([L, N, H], "float32");
      const cStack = zeros([L, N, H], "float32");
      // stack h,c
      for (let i = 0; i < N; i++) {
        yPrevArr[i] = active[i].yPrev;
        // copy h/c data into stacked tensors
        (hStack.data as Float32Array).set(active[i].h.data as Float32Array, i*L*H);
        (cStack.data as Float32Array).set(active[i].c.data as Float32Array, i*L*H);
      }
      // enc_t for each beam = same frame t (we “tile” it)
      // Slice enc_t from (1,T_out,D)
      const baseFrame = t * D;
      for (let i = 0; i < N; i++) {
        enc_t_batch.set(encData.subarray(baseFrame, baseFrame + D), i * D);
      }

      const stepOut = await step.run({
        y_prev: new ort.Tensor("int64", yPrevArr, [N]),
        h0: hStack, c0: cStack,
        enc_t: new ort.Tensor("float32", enc_t_batch, [N, D])
      });

      const logits = stepOut["logits"] as ND; // (N,V) logits (log-softmax or raw; treat as scores)
      const V = logits.dims[1];
      const logBuf = logits.data as Float32Array;
      const hNextAll = stepOut["h1"] as ND;
      const cNextAll = stepOut["c1"] as ND;

      // Collect next beams under trie constraint
      const nextBeams: Beam[] = [];
      for (let i = 0; i < N; i++) {
        const row = logBuf.subarray(i*V, (i+1)*V);

        // 1) Consider "blank": end label loop for this path
        const lpBlank = row[blankId];
        nextBeams.push({
          yPrev: BigInt(blankId),
          h: sliceLC(hNextAll, i, L, H),
          c: sliceLC(cNextAll, i, L, H),
          trie: active[i].trie,
          logp: active[i].logp + lpBlank,
          chars: active[i].chars.slice(),
        });

        // 2) Expand only characters that exist in trie children (prefix constraint)
        // Get top few candidates to prune compute
        const allowed: number[] = [];
        for (const cid of active[i].trie.children.keys()) allowed.push(cid);
        // Fast top-K on allowed set
        const scores: { cid: number; val: number }[] = [];
        for (const cid of allowed) scores.push({ cid, val: row[cid] });
        scores.sort((a,b)=>b.val-a.val);
        const toExpand = scores.slice(0, Math.min(prunePerBeam, scores.length));

        for (const { cid, val } of toExpand) {
          const child = active[i].trie.children.get(cid)!;
          const h1 = sliceLC(hNextAll, i, L, H);
          const c1 = sliceLC(cNextAll, i, L, H);
          nextBeams.push({
            yPrev: BigInt(cid),
            h: h1, c: c1,
            trie: child,
            logp: active[i].logp + val,
            chars: active[i].chars.concat(cid),
          });
        }
      }

      // Prune to beamSize
      nextBeams.sort((a,b)=>b.logp - a.logp);
      beams = nextBeams.slice(0, beamSize);

      // Early stop inner loop if best path picked blank (common heuristic)
      // If top beam ended with blank and stays top, break
      if (Number(beams[0].yPrev) === blankId) break;
    }

    // proceed to next time frame (RNNT time-synchronous decoding)
    // No extra action needed; beams already updated with states.
  }

  // 4) Gather completed words from beams (terminal trie nodes)
  type Cand = { word: string; wid: number; score: number; logpModel: number; logpLM: number; };
  const cands: Cand[] = [];
  for (const b of beams) {
    let bestWid = -1;
    let node = b.trie;
    // If exactly at a word
    if (node.isWord) bestWid = node.wordId;
    // If not terminal, you can choose to ignore or “snap” to nearest terminal (optional).
    if (bestWid >= 0) {
      const logpLM = lex.wordLogPrior ? lex.wordLogPrior[bestWid] : 0;
      const score = b.logp + lmLambda * logpLM;
      cands.push({
        word: lex.words[bestWid],
        wid: bestWid,
        score, logpModel: b.logp, logpLM
      });
    }
  }
  // Rank and unique by word
  cands.sort((a,b)=>b.score - a.score);
  const seen = new Set<number>();
  const top: Cand[] = [];
  for (const c of cands) {
    if (seen.has(c.wid)) continue;
    seen.add(c.wid);
    top.push(c);
    if (top.length >= returnTopK) break;
  }
  return top;
}

// Slice beam i from stacked (L,N,H) → (L,1,H)
function sliceLC(all: ND, i: number, L: number, H: number): ND {
  const out = zeros([L, 1, H], "float32");
  const src = all.data as Float32Array;
  const dst = out.data as Float32Array;
  // contiguous layout expected: (L,N,H)
  // Copy each layer’s H chunk for beam i
  for (let l = 0; l < L; l++) {
    const srcBase = (l * all.dims[1] + i) * H;
    const dstBase = (l * 1 + 0) * H;
    dst.set(src.subarray(srcBase, srcBase + H), dstBase);
  }
  return out;
}



/* usage
const { encoder, step } = await createSessions("encoder_int8_qdq.onnx", "rnnt_step_fp32.onnx");
const lex = /* { words, charToId, idToChar, trie, wordLogPrior? } */;
const F = 37; const T = traceLen; const L = 2; const H = 320; const D = /* encoder out dim */;
const featuresBFT = featurizeSwipeToBFT(swTrace); // Float32Array of [1,F,T]
const top = await rnntBeamSearchWord(encoder, step, featuresBFT, F, T, L, H, D, lex, {
  beamSize: 16, prunePerBeam: 6, maxSymbols: 20, lmLambda: 0.4, returnTopK: 5
});
*/ top -> [{word, score, logpModel, logpLM}, ...]

