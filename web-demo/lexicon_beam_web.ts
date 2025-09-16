/**
 * lexicon_beam_web.ts
 *
 * Web TypeScript implementation for ONNX-based lexicon beam search.
 * Loads quantized encoder (INT8) + fp32 step model + runtime metadata + wordlist.
 * Runs prefix-constrained RNNT beam search and returns top-K words.
 */

import * as ort from "onnxruntime-web";

/** ---------- I/O helpers ---------- */
async function fetchText(url: string): Promise<string> {
    return await (await fetch(url)).text();
}

async function fetchJSON<T = any>(url: string): Promise<T> {
    return await (await fetch(url)).json() as T;
}

/** ---------- Meta & lexicon ---------- */
type Meta = {
    blank_id: number;
    unk_id: number;
    char_to_id: Record<string, number>;
    // You can also stash L,H,D here at export time if handy
};

type TrieNode = {
    children: Map<number, TrieNode>;
    isWord: boolean;
    wid: number
};

function normalizeWord(w: string): string {
    return w.toLowerCase().replace(/\u2019/g, "'");
}

function buildTrie(words: string[], charToId: Map<string, number>): { root: TrieNode; kept: number } {
    const node = (): TrieNode => ({ children: new Map(), isWord: false, wid: -1 });
    const root = node();
    let kept = 0;

    words.forEach((w, wid) => {
        w = normalizeWord(w);
        if (![...w].every(ch => charToId.has(ch))) return;
        let cur = root;
        for (const ch of w) {
            const cid = charToId.get(ch)!;
            if (!cur.children.has(cid)) cur.children.set(cid, node());
            cur = cur.children.get(cid)!;
        }
        cur.isWord = true;
        cur.wid = wid;
        kept++;
    });

    return { root, kept };
}

/** ---------- Beam search ---------- */
type ND = ort.Tensor;

type Beam = {
    yPrev: bigint;
    h: ND;
    c: ND;
    trie: TrieNode;
    logp: number;
    chars: number[];
};

function zeros(shape: number[], dtype: "float32" | "int32" | "int64"): ND {
    const n = shape.reduce((a, b) => a * b, 1);
    const buf = dtype === "float32" ? new Float32Array(n) :
                dtype === "int32" ? new Int32Array(n) :
                new BigInt64Array(n);
    return new ort.Tensor(dtype as any, buf, shape);
}

function sliceLC(all: ND, i: number, L: number, H: number): ND {
    const out = zeros([L, 1, H], "float32");
    const src = all.data as Float32Array;
    const dst = out.data as Float32Array;
    const N = all.dims[1];

    for (let l = 0; l < L; l++) {
        const srcBase = (l * N + i) * H;
        const dstBase = l * H;
        dst.set(src.subarray(srcBase, srcBase + H), dstBase);
    }
    return out;
}

/** Core word-beam RNNT (B=1 trace) */
export async function rnntWordBeam(
    encoder: ort.InferenceSession,
    step: ort.InferenceSession,
    featuresBFT: Float32Array,
    F: number,
    T: number,
    L: number,
    H: number,
    D: number,
    meta: Meta,
    words: string[],
    trieRoot: TrieNode,
    wordLogPrior?: Float32Array,
    opts = {
        beamSize: 16,
        prunePerBeam: 6,
        maxSymbols: 20,
        lmLambda: 0.4,
        topK: 5
    }
): Promise<{ word: string; score: number; rnnt: number }[]> {
    const encOut = await encoder.run({
        features_bft: new ort.Tensor("float32", featuresBFT, [1, F, T]),
        lengths: new ort.Tensor("int32", new Int32Array([T]), [1]),
    });

    const encBTF = encOut["encoded_btf"] as ND; // (1,T_out,D)
    const T_out = encBTF.dims[1];
    const enc = encBTF.data as Float32Array;

    let beams: Beam[] = [{
        yPrev: BigInt(meta.blank_id),
        h: zeros([L, 1, H], "float32"),
        c: zeros([L, 1, H], "float32"),
        trie: trieRoot,
        logp: 0,
        chars: []
    }];

    const enc_t_batch = new Float32Array(opts.beamSize * D);

    for (let t = 0; t < T_out; t++) {
        for (let s = 0; s < opts.maxSymbols; s++) {
            beams.sort((a, b) => b.logp - a.logp);
            const N = Math.min(opts.beamSize, beams.length);
            const act = beams.slice(0, N);

            const yPrev = new BigInt64Array(N);
            const h0 = zeros([L, N, H], "float32");
            const c0 = zeros([L, N, H], "float32");

            for (let i = 0; i < N; i++) {
                yPrev[i] = act[i].yPrev;
                (h0.data as Float32Array).set(act[i].h.data as Float32Array, i * L * H);
                (c0.data as Float32Array).set(act[i].c.data as Float32Array, i * L * H);
            }

            const base = t * D;
            for (let i = 0; i < N; i++) {
                enc_t_batch.set(enc.subarray(base, base + D), i * D);
            }

            const out = await step.run({
                y_prev: new ort.Tensor("int64", yPrev, [N]),
                h0,
                c0,
                enc_t: new ort.Tensor("float32", enc_t_batch, [N, D]),
            });

            const logits = out["logits"] as ND;
            const V = logits.dims[1];
            const logBuf = logits.data as Float32Array;
            const h1 = out["h1"] as ND;
            const c1 = out["c1"] as ND;

            const next: Beam[] = [];
            for (let i = 0; i < N; i++) {
                // blank transition
                const lpBlank = logBuf[i * V + meta.blank_id];
                next.push({
                    yPrev: BigInt(meta.blank_id),
                    h: sliceLC(h1, i, L, H),
                    c: sliceLC(c1, i, L, H),
                    trie: act[i].trie,
                    logp: act[i].logp + lpBlank,
                    chars: act[i].chars.slice()
                });

                // allowed children (trie constraint)
                const allowed = Array.from(act[i].trie.children.keys());
                if (allowed.length) {
                    allowed.sort((a, b) => logBuf[i * V + b] - logBuf[i * V + a]);
                    for (const cid of allowed.slice(0, Math.min(opts.prunePerBeam, allowed.length))) {
                        next.push({
                            yPrev: BigInt(cid),
                            h: sliceLC(h1, i, L, H),
                            c: sliceLC(c1, i, L, H),
                            trie: act[i].trie.children.get(cid)!,
                            logp: act[i].logp + logBuf[i * V + cid],
                            chars: act[i].chars.concat(cid)
                        });
                    }
                }
            }
            next.sort((a, b) => b.logp - a.logp);
            beams = next.slice(0, opts.beamSize);
            if (Number(beams[0].yPrev) === meta.blank_id) break;
        }
    }

    // collect terminals
    const cands: { wid: number; score: number; rnnt: number }[] = [];
    for (const b of beams) {
        if (b.trie.isWord && b.trie.wid >= 0) {
            const wid = b.trie.wid;
            const lm = wordLogPrior ? wordLogPrior[wid] : 0;
            cands.push({ wid, score: b.logp + opts.lmLambda * lm, rnnt: b.logp });
        }
    }

    cands.sort((a, b) => b.score - a.score);
    const seen = new Set<number>();
    const out: { word: string; score: number; rnnt: number }[] = [];

    for (const c of cands) {
        if (seen.has(c.wid)) continue;
        seen.add(c.wid);
        out.push({ word: words[c.wid], score: c.score, rnnt: c.rnnt });
        if (out.length >= opts.topK) break;
    }

    return out;
}

/** ---------- End-to-end helper ---------- */
export async function loadAndDecode(
    encoderUrl: string,
    stepUrl: string,
    metaUrl: string,
    wordsUrl: string,
    priorsUrl?: string,
    featuresBFT?: Float32Array,  // precomputed; else supply your own featurizer
    F = 37,
    T?: number,
    L = 2,
    H = 320,
    D?: number,
    opts?: Partial<Parameters<typeof rnntWordBeam>[12]>
): Promise<{ word: string; score: number; rnnt: number }[]> {
    const ep = ort.env.webgpu?.enabled ? "webgpu" : "wasm";
    const encoder = await ort.InferenceSession.create(encoderUrl, { executionProviders: [ep] });
    const step = await ort.InferenceSession.create(stepUrl, { executionProviders: [ep] });

    const meta = await fetchJSON<Meta>(metaUrl);
    const words = (await fetchText(wordsUrl)).split(/\r?\n/).filter(s => s.length > 0);
    const priors = priorsUrl ? new Float32Array(await (await fetch(priorsUrl)).arrayBuffer()) : undefined;

    // Build trie
    const charToId = new Map<string, number>(Object.entries(meta.char_to_id));
    const { root } = buildTrie(words, charToId);

    if (!featuresBFT || !T) {
        throw new Error("Provide featuresBFT (Float32Array[B*F*T]) and T (frames).");
    }

    // If you don't know D, run encoder once to read output shape; here we require caller to pass D.
    if (!D) {
        throw new Error("Set encoder output dim D.");
    }

    return rnntWordBeam(encoder, step, featuresBFT, F, T, L, H, D, meta, words, root, priors, {
        beamSize: 16,
        prunePerBeam: 6,
        maxSymbols: 20,
        lmLambda: 0.4,
        topK: 5,
        ...(opts || {})
    });
}

// Usage example:
// const top = await loadAndDecode(
//     "/models/encoder_int8_qdq.onnx",
//     "/models/rnnt_step_fp32.onnx",
//     "/models/runtime_meta.json",
//     "/lexicon/words.txt",
//     "/lexicon/word_priors.f32",   // optional (Float32Array with log priors per word)
//     featuresBFT, 37, T, 2, 320, D /* encoder out dim */
// );
// => [{word, score, rnnt}, ...]