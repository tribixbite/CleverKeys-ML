# CleverKeys RNNT Architecture – Training to Inference

**Version**: 2025-09-16  
**Model**: NeMo EncDecRNNTModel (Conformer encoder + LSTM prediction network + joint)

This document reflects the `trained_models/nema1` pipeline that produced the currently exported ONNX/ExecuTorch artifacts. Source of truth lives under `trained_models/nema1/` unless otherwise noted.

---

## 1. Training Stack

### 1.1 Entry Point – `trained_models/nema1/train_transducer.py`
- Wraps NVIDIA NeMo's `EncDecRNNTModel` with a swipe-specific dataset/featurizer.
- Key configuration (`CONFIG` block) mirrors the exported model:
  - **Encoder**: `feat_in=37`, `d_model=256`, `num_layers=8`, `n_heads=4`, `conv_kernel_size=31`, `subsampling_factor=2`.
  - **Prediction network**: LSTM with `pred_hidden=320`, `pred_rnn_layers=2`.
  - **Joint network**: `joint_hidden=512`, `activation='relu'`, `dropout=0.1`.
  - **Loss**: `RNNTLoss` with `loss_name='warprnnt_numba'`, `blank_idx=0`, `fastemit_lambda=0.001`.
  - **Training**: batch size 256, AdamW (lr 4e-4, cosine schedule with 2k warmup), BF16 mixed precision on 1 GPU.
- Resumes the most recent checkpoint automatically via `find_latest_checkpoint()`.

### 1.2 Data Interface – `SwipeDataset` & `collate_fn`
- Manifests (`data/train_final_train.jsonl`, `data/train_final_val.jsonl`) contain per-word swipe traces:
  ```json
  {"word": "example", "points": [{"x": 0.13, "y": -0.92, "t": 0}, ...]}
  ```
- Coordinates are already normalized into a keyboard-centric space that typically lies in `[-1, 1]` but can extend to approximately `[-1.5, 1.5]`; values are consumed verbatim.
- `SwipeDataset` applies the featurizer on each trace, returns `(features, feature_len, target_tokens, token_len)`.
- `collate_fn` pads variable-length gestures and token targets, with optional frame stacking disabled by default.

### 1.3 Gesture Features – `trained_models/nema1/swipe_data_utils.py`
Each time step emits a 37-D vector built from 27 engineered statistics plus zero padding:
1. **Position & time (3)**: `x`, `y`, elapsed seconds `t`.
2. **Velocity (3)**: instantaneous `vx`, `vy`, `speed`.
3. **Acceleration (3)**: `ax`, `ay`, magnitude `acc`.
4. **Heading (3)**: absolute angle, `sin(angle)`, `cos(angle)`.
5. **Curvature (1)**: delta angle to previous segment, wrapped to `[-π, π]`.
6. **Key proximity (5)**: Euclidean distances to the five closest QWERTY key centres (grid stored in `[0,1]×[0,1]`).
7. **Trace context (1)**: normalized progress `idx / (T-1)`.
8. **Boundary flags (2)**: `is_start`, `is_end`.
9. **Window statistics (6)**: local mean/std/range for `x` and `y` over a 5-point window.
10. **Padding (up to 37)**: zeros to maintain the expected feature width.

The featurizer does **not** compute radial-basis activations; any runtime code must replicate the quantities above to stay on-distribution.

### 1.4 Vocabulary
- `data/vocab.txt` defines 29 tokens: `<blank>`, `'`, `a`–`z`, `<unk>`.
- `SwipeDataset` maps characters via this vocab; `<unk>` handles OOV symbols.

---

## 2. Export Pipeline

### 2.1 Encoder Export – `trained_models/nema1/export_onnx.py`
- Accepts either a `.ckpt` or `.nemo` checkpoint and instantiates `GestureRNNTModel`.
- Wraps the Conformer encoder and exports:
  - `encoder_fp32.onnx` (opset 17), input names `features_bft` `[B,37,T]` and `lengths` `[B]`.
  - Runs ONNX Runtime static quantization to emit `encoder_int8_qdq.onnx` (per-channel int8 QDQ).
- Calibration data come from `SwipeDataset` on the validation manifest.

### 2.2 Decoder+Joint Export – `trained_models/nema1/export_rnnt_step.py`
- Builds a single-step module around the NeMo prediction network + joint.
- Outputs:
  - `rnnt_step_fp32.onnx` (inputs: `y_prev` `[N]` int64, `h0` `[L,N,H]`, `c0` `[L,N,H]`, `enc_t` `[N,D]`; outputs logits `[N,V]`, `h1`, `c1`).
  - `rnnt_step_fp32.pte` compiled via ExecuTorch/XNNPACK for mobile targets.
- Blank token ID is auto-derived from the vocab when provided.

### 2.3 Runtime Metadata – `trained_models/nema1/runtime_meta.json`
- Captures the deployed vocabulary (`tokens`, `char_to_id`, `id_to_char`, `blank_id=0`, `unk_id=28`).
- Downstream tooling (validators, demos) should use this file as the canonical mapping. Any extended metadata in web builds must preserve these IDs.

---

## 3. Inference Building Blocks

### 3.1 Word Assets
- `trained_models/nema1/words.txt`: ~154k lowercase words used to build decoding tries.
- `trained_models/nema1/swipe_vocabulary.json`: enriched metadata for post-processing – contains `metadata`, a dense `word_frequencies` map, length buckets, adjacency hints, and curated sets such as `common_words` and `top_5000`.

### 3.2 Lexicon Construction – `trained_models/nema1/vocab-meta-utils.js`
- `buildTrieFromWords` constructs an ID-based trie compatible with the exported ONNX step model.
- Filters vocabulary through `normalizeWord`/`isValidWord` to enforce the 29-token alphabet.

### 3.3 Beam Search Reference – `trained_models/nema1/beam_decode_web.ts`
- Implements word-level beam search over the encoder/step ONNX pair.
- Pseudocode of the deployed algorithm:
  1. Encode gesture: `features_bft` `[1,37,T]` → `encoded_btf` `[1,T_out,D]`.
  2. Iterate time frames and expand beams with blank + trie-constrained symbol transitions using `rnnt_step_fp32.onnx`.
  3. Score hypotheses as `rnnt_logp + λ · word_log_prior`; default `beamSize=16`, `prunePerBeam=6`, `maxSymbols=20`, `lmLambda=0.4`.
- Exposes helpers (`buildTrie`, `zeros`, `topKRow`) that web/Android surfaces should reuse.

### 3.4 Required Runtime Inputs
- **Features**: float32 `[1,37,T]` exactly matching the featurizer described in §1.3.
- **Lengths**: int32 `[1]` real trace length before padding.
- **Decoder state**: int64 tokens, float32 LSTM states sized `L=2`, `H=320`.
- **Vocabulary prior**: optional `Float32Array` aligned with `words.txt` indices (often inverse-rank or log frequency).

Any deviation—such as feeding RBF features or different coordinate normalization—constitutes domain shift and will degrade accuracy.

---

## 4. Tensor & File Summary

| Component | Path | Format | Key I/O |
|-----------|------|--------|---------|
| Encoder | `trained_models/nema1/encoder_int8_qdq.onnx` | ONNX INT8 QDQ | `features_bft [B,37,T]`, `lengths [B]` → `encoded_btf [B,T_out,D]`, `encoded_lengths [B]` |
| Step model | `trained_models/nema1/rnnt_step_fp32.onnx` | ONNX FP32 | `y_prev [N]`, `h0/c0 [L,N,H]`, `enc_t [N,D]` → `logits [N,V]`, `h1`, `c1` |
| Step mobile | `trained_models/nema1/rnnt_step_fp32.pte` | ExecuTorch | Same as above |
| Runtime vocab | `trained_models/nema1/runtime_meta.json` | JSON | Tokens + ID mappings |
| Word list | `trained_models/nema1/words.txt` | Text | Vocabulary for trie |
| Extended vocab | `trained_models/nema1/swipe_vocabulary.json` | JSON | Frequencies, adjacency, curated sets |

---

## 5. Implementation Notes & Checks

1. **Coordinate space** – ingest points exactly as recorded. Typical range is `[-1.5, 1.5]`; do not re-normalize to `[0,1]` or alter origin unless the dataset changes.
2. **Feature parity** – reproduce the 27 engineered statistics + zero padding verbatim. Differences in ordering or scaling will mislead the encoder.
3. **Token discipline** – ensure `<blank>` stays at index 0 and `<unk>` at 28 when building tries, priors, or metadata.
4. **Quantized encoder** – verify ONNX input names by introspecting the model (`session.inputNames`) rather than hard-coding; training export guarantees `features_bft`/`lengths` but tooling can add suffixes.
5. **Word priors** – the shipped beam search expects log priors aligned with `words.txt`. When using `swipe_vocabulary.json`, derive priors from its `word_frequencies` and preserve ordering.
6. **Validation tooling** – `trained_models/scripts/validate_vocab_system.py` accepts the runtime metadata JSON to check trie/vocab consistency before packaging demos.

---

## 6. Reproducibility Checklist

- ✅ `train_transducer.py`, `swipe_data_utils.py`, and supporting scripts archived under `trained_models/nema1/`.
- ✅ Quantized encoder (`encoder_int8_qdq.onnx`) and step model (`rnnt_step_fp32.onnx` / `.pte`) checked into the same directory.
- ✅ Runtime metadata and vocab assets provided (`runtime_meta.json`, `words.txt`, `swipe_vocabulary.json`).
- ✅ Beam search reference implementation available (`beam_decode_web.ts`).
- ✅ Manifests (`data/train_final_*.jsonl`) and vocab (`data/vocab.txt`) remain unchanged from training.

Maintain this document as the authoritative description of the NEMA1 export; any demo (web, Android, etc.) must align with it or be updated alongside functional changes.
