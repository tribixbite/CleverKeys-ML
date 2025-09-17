# Neural Swipe Typing Architecture (Character-Level Stack)

This document captures how the character-level swipe typing model is trained, exported, and served in the web demo. It consolidates the most relevant source files and highlights expected inputs/outputs, vocabulary handling, and deployment considerations. Copies of the referenced code live alongside this file in `docs_and_assets/architecture_snapshot/` for portability.

## Snapshot Contents

The snapshot folder contains:

- `train_character_model.py` – shared data pipeline, tokenizer, and `CharacterLevelSwipeModel` definition.
- `train_full_model.py` – full-data training entry point using the above components.
- `export_character_model.py` – ONNX and ExecuTorch exporters plus packaging utilities.
- `web-demo/` – browser demo assets (`swipe-onnx.html`, `swipe-vocabulary.js`, `custom-dictionary.js`, `niche-words-loader.js`).
- `ARCHITECTURE.md` – this documentation.

---

## Shared Building Blocks (`train_character_model.py`)

### Keyboard Geometry (`KeyboardGrid`, lines 22-60)
- Loads `data/data_preprocessed/gridname_to_grid.json` and caches QWERTY key centroids.
- Provides `get_nearest_key(x, y)` to map raw coordinates onto character labels.
- Adds `<unk>` and `<pad>` pseudo-keys; exposes full keyboard width/height for normalization.

### Tokenizer (`CharTokenizer`, lines 63-103)
- Vocabulary: 26 lowercase letters + special tokens `<pad>`, `<unk>`, `<sos>`, `<eos>`.
- `encode_word(word)` prepends `<sos>`, appends `<eos>`, and maps unknown glyphs to `<unk>`.
- `decode(indices)` stops at `<pad>` and strips special tokens when forming output words.

### Dataset (`SwipeDataset`, lines 105-226)
- Accepts JSONL input with per-swipe trajectories. Supported schemas:
  - Combined dataset records: `{ "curve": {"x": [...], "y": [...], "t": [...]}, "word": "..." }`.
  - Synthetic traces: `{ "word_seq": {"x": [...], "y": [...], "time": [...]}, "word": "..." }`.
  - Legacy format with `grid_name == "qwerty_english"`.
- Normalization pipeline per swipe:
  - Scale `x`/`y` to `[0,1]` using keyboard width/height (lines 165-167).
  - Compute velocity via first-order differences of normalized coordinates divided by time deltas (lines 169-177). Very small `dt` are clamped to `1e-6` (line 171).
  - Compute acceleration via second-order differences and clip all dynamics to `[-10, 10]` for stability (lines 178-187).
  - Stack `[x, y, vx, vy, ax, ay]` into a 6-D feature vector per point (line 196).
  - Determine nearest key index using `KeyboardGrid` (lines 189-194).
  - Pad/truncate trajectories to 150 points and targets to 20 tokens using `<pad>` (lines 200-218).
- Returns a dictionary suitable for PyTorch DataLoader:
  - `traj_features`: `FloatTensor [seq_len, 6]` (padded to `[150, 6]`).
  - `nearest_keys`: `LongTensor [seq_len]` (character indices, padded with `<pad>`).
  - `target`: `LongTensor [20]` (tokenized word incl. SOS/EOS).
  - `seq_len`: original trajectory length before padding.
  - `word`: ground-truth string (for evaluation and debugging).

### Model (`CharacterLevelSwipeModel`, lines 229-339)
- Encoder path:
  - Projects 6-D trajectory features and nearest-key embeddings to `d_model/2` each, concatenates, and layer-norms (lines 247-303).
  - Adds sinusoidal positional encoding fixed at 150 steps (lines 252-260).
  - Runs a Transformer encoder stack (`num_encoder_layers`, default 4 but bumped to 6 in full training).
- Decoder path:
  - Embeds target tokens, scales by `sqrt(d_model)`, adds positional encoding, and applies a Transformer decoder (`num_decoder_layers`, default 3 but 4 in full training) with causal masking (lines 317-334).
  - Projects decoder output to vocabulary logits via a linear head (lines 336-337).
- `generate_beam(...)` (lines 341-339) performs beam search entirely in PyTorch, mirroring the JavaScript runtime logic while reusing the encoder/decoder weights.

**Input/Output Signature (training time):**
- Inputs:
  - `traj_features`: `FloatTensor [batch, 150, 6]`.
  - `nearest_keys`: `LongTensor [batch, 150]` containing tokenizer indices for the nearest key at each swipe sample.
  - `targets`: `LongTensor [batch, 20]` of character tokens (SOS-prefixed, EOS-suffixed, PAD-padded).
  - `src_mask`: `BoolTensor [batch, 150]` with `True` for padded encoder positions.
  - `tgt_mask`: `BoolTensor [batch, 19]` for padded decoder input positions.
- Output: logits `FloatTensor [batch, 19, vocab_size]` (one step shorter than target due to teacher forcing shift).

---

## Full Training Loop (`train_full_model.py`)

### Configuration & Data Loading
- Uses the production combined dataset at `data/combined_dataset/cleaned_english_swipes_{train,val,test}.jsonl` (lines 47-63).
- Hyperparameters: batch size 64, learning rate `5e-4`, OneCycle LR scheduler across 50 epochs with 2-epoch warmup (lines 33-125).
- Instantiates the large-capacity variant of `CharacterLevelSwipeModel`: `d_model=256`, `nhead=8`, encoder layers 6, decoder layers 4, FFN 1024 (lines 92-104).

### Training Loop
- Builds boolean source masks by flagging padded time steps using `seq_len` metadata (lines 149-154).
- Creates target padding mask by comparing against tokenizer pad index (line 155).
- Loss: cross-entropy ignoring PAD tokens (line 113), gradient clipped at 1.0 (line 166).
- Tracks character-level accuracy during training (lines 170-175).

### Validation & Early Stopping
- Switches to `model.generate_beam(...)` with beam size 5 for word-level accuracy on the validation loader (lines 190-224).
- Saves checkpoints to `checkpoints/full_character_model/full-model-{epoch}-{val_acc}.ckpt` whenever the validation score improves, with target 70% word accuracy (lines 232-274).
- Optional early stop with patience 15 if accuracy stalls (lines 275-284).
- Once validation hits ≥ 0.99, evaluates on the test loader with the same beam search (lines 241-271).

### Expected Inputs at Training Time
- JSONL records must provide synchronized arrays `x`, `y`, and `t` of equal length representing sampled swipe coordinates and timestamps.
- Vocabulary is implicitly managed by `CharTokenizer`; no external word list is required during training.

---

## Export Pipeline (`export_character_model.py`)

### Checkpoint Loading (`load_best_checkpoint`, lines 22-63)
- Searches `checkpoints/full_character_model` for a named best checkpoint (`full-model-14-0.701.ckpt`) or the highest-accuracy fallback.
- Reconstructs the full `CharacterLevelSwipeModel` with dropout disabled (line 51) for deterministic inference.

### ONNX Export (`export_to_onnx`, lines 66-199)
- Splits encoder and decoder into separate graphs tailored for web inference.
- **Encoder ONNX**
  - Wrapper exposes `forward(trajectory_features, nearest_keys, src_mask)` and returns encoder memory (lines 72-85).
  - Input dtypes: `float32` for trajectories `[batch, seq, 6]`, `int64` for nearest keys `[batch, seq]`, `bool` for masks `[batch, seq]`.
  - Dynamic axes mark batch/sequence dimensions to accept arbitrary swipe lengths up to 150.
- **Decoder ONNX**
  - Accepts encoder memory `[batch, seq, 256]`, target tokens `[batch, dec_seq]`, source mask `[batch, seq]`, and target mask `[batch, dec_seq]` (lines 138-193).
  - Outputs logits `[batch, dec_seq, vocab]` (vocab size 30).

### ExecuTorch Export (`export_to_executorch`, lines 201-275)
- Wraps the model in a traced `MobileModel` that runs full autoregressive decoding inside a single forward pass (lines 211-252).
- Generates tokens iteratively for a fixed `max_len=20` with greedy argmax at each step.
- Traces with example inputs `[1, 150, 6]` and `[1, 150]`, converts to ExecuTorch, and applies XNNPACK partitioning before saving `swipe_model_character.pte` (lines 256-269).

### Deployment Artifacts (`create_tokenizer_config`, `create_model_config`, `create_deployment_guide`, lines 277-430)
- `tokenizer_config.json`: serializes character mappings and special token indices for client runtimes.
- `model_config.json`: documents architecture sizes, feature preprocessing assumptions, beam search settings, and training stats.
- `DEPLOYMENT_GUIDE.md`: describes browser and mobile integration steps with code snippets (lines 331-430).

**Key Input Contracts:**
- Both ONNX and ExecuTorch paths assume normalized trajectories in `[0,1]`, **not** raw keyboard pixels.
- Nearest key IDs must use the tokenizer’s character indices.
- Masks use `bool` tensors where `True` denotes padding for PyTorch and ONNX.

---

## Browser Demo (`web-demo/swipe-onnx.html`)

### Runtime Setup (lines 363-460)
- Defines normalization constants `NORMALIZED_WIDTH=360`, `NORMALIZED_HEIGHT=215`, `MAX_SEQUENCE_LENGTH=150` to mirror training.
- Loads ONNX Runtime Web (WASM backend), `tokenizer_config.json`, and quantized ONNX graphs: `swipe_model_character_quant.onnx` (encoder) and `swipe_decoder_character_quant.onnx` (decoder).
- Fetches vocabulary resources via `SwipeVocabulary` and augments them with the custom dictionary manager once loaded.

### Gesture Capture & Feature Extraction (lines 1086-1255)
- Collects pointer/touch events on a canvas overlay to track swipe paths and key contacts.
- `prepareSwipeFeatures` normalizes the gesture to 150 points by dividing raw screen coordinates by the computed keyboard bounds (lines 1161-1188).
- `runInference` replicates the training-time feature construction:
  - Populates a `Float32Array` with x/y positions plus velocity/acceleration approximated via discrete differences (lines 1196-1239). Note: velocities ignore actual timestamp spacing and assume uniform sampling, which works empirically but differs from training’s time-based scaling.
  - Maps detected key labels to tokenizer indices (lines 1201-1207).
  - Builds Boolean source masks marking padded positions (line 1248).
  - Creates ONNX tensors: `trajectory_features` (float32), `nearest_keys` (int64), `src_mask` (bool) (lines 1252-1255).

### Encoder + Decoder Invocation (lines 1263-1415)
- Runs the encoder session and feeds `encoder_output` into a JavaScript beam-search loop that repeatedly calls the decoder ONNX graph.
- Beam search parameters:
  - Beam width 8, maximum length 35 tokens (`decodeLogits`, lines 1289-1292).
  - Decoder input tokens stored as `BigInt64Array` to satisfy ONNX Runtime’s int64 requirement (lines 1331-1347).
  - Target mask marks padded slots with `true` (implemented via `Uint8Array` values `1`, lines 1348-1352).
  - Source mask reuses zero (non-padded) flags for all encoder positions (lines 1354-1356).
- After each decoder call, the code softmaxes logits in JavaScript, extends partial beams, and checks for EOS to terminate early (lines 1375-1414).

### Post-processing and Display (lines 1418 onwards)
- Converts token sequences back into strings using `idx_to_char` from the tokenizer config, skipping special tokens (lines 1418-1428).
- Passes raw predictions to `SwipeVocabulary.filterPredictions` for frequency-based ranking and merges with custom dictionaries; the final suggestions update the UI.

### Vocabulary Tooling
- `swipe-vocabulary.js` loads word frequency data, organizes words by length, and scores predictions using a combination of neural confidence and log-frequency heuristics.
- `custom-dictionary.js` allows importing Android dictionaries and personal word lists, syncing them to `localStorage`, and merging frequencies into `SwipeVocabulary`.
- `niche-words-loader.js` (not detailed here) handles optional niche word packs.

---

## Data & Vocabulary Expectations

| Stage        | Trajectory Format                           | Nearest Key Encoding                          | Target Tokens |
|--------------|---------------------------------------------|-----------------------------------------------|---------------|
| Training     | Raw keyboard pixels (`x`, `y`), timestamps `t` per sample. Normalized to `[0,1]`, velocities/accelerations from `Δx/Δt`, `Δv/Δt`. | Lookup from `KeyboardGrid` -> `CharTokenizer` index, padded with `<pad>`. | `CharTokenizer` outputs `<sos> word <eos>` padded to 20. |
| ONNX Export  | Assumes pre-normalized `[0,1]` trajectories and token indices exactly as produced by `SwipeDataset`. Masks must be boolean with `True` marking padding. | Same as training. | Input length for decoder chosen by caller (web JS pads to 20). |
| Web Demo     | Normalized screen coordinates scaled to `[0,1]`; velocities/accelerations inferred from discrete diffs (no timing). | Uses tokenizer mapping from `tokenizer_config.json`; defaults cover `a-z`. | Beam search uses SOS/EOS indices from config; BigInt arrays satisfy int64. |

Vocabulary is consistent end-to-end via `CharTokenizer`. The web demo augments model outputs with `SwipeVocabulary` frequency info and user dictionaries, but the neural model itself only ever sees character indices.

---

## Android & ONNX Considerations

The Android issue stems from deploying the ONNX pair outside the browser. Critical aspects to double-check:

1. **Tensor dtypes:** ONNX Runtime Mobile for Android historically lacks full `int64` support; feeding int64 nearest-key tensors may fail. Consider exporting an int32-friendly model (cast embeddings and weights) or inserting a preprocessing step that casts to int64 inside the graph.
2. **Boolean masks:** Some mobile inference stacks prefer `uint8` masks over `bool`. The JavaScript client intentionally builds `Uint8Array` masks; ensure your Android tensors match the exporter’s expectation (`bool` in ONNX) or adjust the graph to accept `uint8` with explicit casts.
3. **Decoder loop:** Mobile integrations must replicate the JavaScript beam search. Running two ONNX sessions (encoder, decoder) in Java/Kotlin mirrors the browser logic but introduces latency. An alternative is to export a single-step greedy decoder (like the ExecuTorch wrapper) or to script the beam search in Python and re-export a fused graph via TorchScript.
4. **Quantized artifacts:** The browser loads `*_quant.onnx` models. Verify that Android uses the same quantized weights or, if using the FP32 versions, align tensor dtypes accordingly.
5. **Feature extraction parity:** Ensure Android’s gesture normalization reproduces the training pipeline (including velocity/acceleration scaling). Sampling without timestamps may degrade accuracy; consider recording time deltas or mirroring the JS approximation deliberately.

---

## Recommendations for Improvement

1. **Consolidate Preprocessing:** Extract the feature-construction code (normalization, velocity/acceleration, nearest key lookup) into a shared library used by both Python and client runtimes to eliminate drift between training and inference.
2. **Mobile-Friendly ONNX Variant:** Generate a single encoder+decoder ONNX graph that performs greedy or beam decoding internally, using only `float32`/`int32`/`uint8` tensors. This will simplify Android integration and avoid `BigInt64Array` patterns that are awkward outside JS.
3. **Time-Aware Web Features:** The JS demo currently ignores timestamps. If swipe sampling varies on mobile, incorporate estimated `Δt` values or velocity scaling factors similar to training to improve robustness.
4. **Quantization Pipeline Documentation:** Capture the steps that produced the `*_quant.onnx` files (likely via `optimize_onnx_models.py`) inside this snapshot so future ports can reproduce the exact weights.
5. **Android Reference Implementation:** Provide a minimal Kotlin sample that mirrors the JS beam search, including tokenizer/vocabulary loading, to accelerate debugging of the mobile ONNX path.
6. **Unit Tests for Export Shapes:** Add automated checks ensuring ONNX graph inputs/outputs align with web/mobile expectations, catching dtype or shape regressions early.

---

## Quick Start Checklist

1. **Training:** Ensure `data/combined_dataset/*.jsonl` is available, then run `python train_full_model.py`. Checkpoints appear under `checkpoints/full_character_model/`.
2. **Export:** Run `python export_character_model.py --output <dir>` (script uses current working directory for outputs). Collect ONNX/ExecuTorch artifacts plus configs.
3. **Web Demo:** Serve `web-demo/` along with exported ONNX files and `tokenizer_config.json`. Browser must support WASM with `BigInt64Array` (modern Chrome/Firefox/Safari).
4. **Android ONNX:** Replicate feature extraction (normalize to 360×215 space, compute velocity/acceleration, nearest keys) and feed tensors into ONNX Runtime Mobile. Consider the dtype recommendations above if you encounter crashes.

---

*Prepared for repository extraction: copy `docs_and_assets/architecture_snapshot/` into a new project to carry both source code and this write-up.*
