import * as ort from "onnxruntime-web";  // use WebGPU if available, else WASM-SIMD

type ND = ort.Tensor;

// Helper to create zero state
function zeros(shape: number[], dtype: "float32"|"int64"|"int32" = "float32"): ND {
  const size = shape.reduce((a,b)=>a*b,1);
  const data =
    dtype === "float32" ? new Float32Array(size) :
    dtype === "int32"   ? new Int32Array(size)   :
                          new BigInt64Array(size);
  return new ort.Tensor(dtype as any, data, shape);
}

export async function createSessions(
  encoderModelUrl: string,   // "encoder_int8_qdq.onnx"
  stepModelUrl: string       // "rnnt_step_fp32.onnx"
) {
  const ep = ort.env.webgpu?.enabled ? "webgpu" : "wasm";
  const encoder = await ort.InferenceSession.create(encoderModelUrl, { executionProviders: [ep] });
  const step    = await ort.InferenceSession.create(stepModelUrl,    { executionProviders: [ep] });
  return { encoder, step };
}

/**
 * Greedy RNNT decode
 * @param encoder session for encoder_int8_qdq.onnx (inputs: features_bft(float32 B,F,T), lengths(int32 B))
 * @param step session for rnnt_step_fp32.onnx (inputs: y_prev(int64 B), h0/c0(float32 L,B,H), enc_t(float32 B,D))
 * @param featuresBFT Float32Array of shape [B,F,T]
 * @param lengths Int32Array [B]
 * @param L number of LSTM layers
 * @param H hidden size
 * @param blankId usually 0
 * @param maxSymbols max emitted tokens per time frame
 */
export async function greedyDecode(
  encoder: ort.InferenceSession,
  step: ort.InferenceSession,
  featuresBFT: Float32Array,
  B: number, F: number, T: number,
  lengths: Int32Array,
  L: number, H: number,
  blankId = 0,
  maxSymbols = 15
) {
  // Run encoder
  const encOut = await encoder.run({
    "features_bft": new ort.Tensor("float32", featuresBFT, [B, F, T]),
    "lengths":      new ort.Tensor("int32",   lengths,     [B]),
  });
  // encoded_btf: (B,T_out,D), encoded_lengths: (B)
  const encodedBTF = encOut["encoded_btf"] as ND;   // float32
  const encLens    = encOut["encoded_lengths"] as ND; // int32
  const [b, T_out, D] = encodedBTF.dims;

  // Init decoder state
  let h = zeros([L, B, H], "float32");
  let c = zeros([L, B, H], "float32");
  // Start token: use blank or BOS (your choice; blank is fine for RNNT greedy)
  let yPrev = new ort.Tensor("int64", BigInt64Array.from({length: B}, () => BigInt(blankId)), [B]);

  const encData = encodedBTF.data as Float32Array;
  const outputs: number[][] = Array.from({length: B}, () => []);

  // Loop over time frames
  for (let t = 0; t < T_out; t++) {
    // Slice enc_t: (B,D)
    const enc_t = new Float32Array(B * D);
    for (let bIdx = 0; bIdx < B; bIdx++) {
      const base = (bIdx * T_out + t) * D;
      enc_t.set(encData.subarray(base, base + D), bIdx * D);
    }
    const encTensor = new ort.Tensor("float32", enc_t, [B, D]);

    // RNNT greedy inner loop (emit up to maxSymbols until blank)
    for (let s = 0; s < maxSymbols; s++) {
      const stepOut = await step.run({ y_prev: yPrev, h0: h, c0: c, enc_t: encTensor });
      const logits = stepOut["logits"] as ND;  // (B,V)
      const V = logits.dims[1];
      const probs = logits.data as Float32Array;

      // Argmax per batch
      const nextIds = new BigInt64Array(B);
      let anyNonBlank = false;
      for (let bIdx = 0; bIdx < B; bIdx++) {
        let best = 0, bestVal = -Infinity;
        const rowBase = bIdx * V;
        for (let v = 0; v < V; v++) {
          const val = probs[rowBase + v];
          if (val > bestVal) { bestVal = val; best = v; }
        }
        nextIds[bIdx] = BigInt(best);
        if (best !== blankId) {
          outputs[bIdx].push(best);
          anyNonBlank = true;
        }
      }

      // Update state and token
      h = stepOut["h1"] as ND;  // (L,B,H)
      c = stepOut["c1"] as ND;
      yPrev = new ort.Tensor("int64", nextIds, [B]);

      // Stop if all blank
      if (!anyNonBlank) break;
    }
  }
  return outputs; // array of token id sequences per batch
}



// For your UI, map token ids → chars using your vocab.txt.
