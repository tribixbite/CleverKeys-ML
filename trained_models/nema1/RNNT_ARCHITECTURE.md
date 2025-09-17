# RNNT Swipe Architecture (Android-Focused)

This document captures the end-to-end pipeline used by the **personalized RNN-Transducer** training run (`trained_models/nema1/train_transducer_personalized.py`) and how it differs from the legacy transformer snapshot. It is the source of truth for Android work; the transformer docs in `trained_models/architecture_snapshot/` remain available as a web/demo reference but are not the production path for mobile.

---

## 1. Data & Feature Pipeline

### 1.1 Manifests
- Training and validation JSONL files live in `data/train_final_{train,val}.jsonl`.
- Each record provides a `word` label and `points` array with `(x, y, t)` in touchscreen-normalised units.
- `max_trace_len` (configurable, default 256) guards against pathological traces.

### 1.2 Normalisation & Resampling
- `_normalize_points` (line 356) maps raw `[0,1]` coordinates into `[-1,1]` space and time-shifts so the gesture starts at `t=0`.
- `determine_resample_target` adapts sequence length into a 56–96 frame band, so short swipes get densified while long traces are decimated smoothly.
- `resample_points` interpolates in time to obtain a fixed-length gesture while preserving high-frequency motion for longer words.

### 1.3 Feature Vector (37 dims)
Implemented by `PersonalizedSwipeFeaturizer` (lines 193-269):
1. Position/time: `(x, y, t_seconds)`
2. Velocity + speed: `(vx, vy, ||v||)`
3. Acceleration + magnitude: `(ax, ay, ||a||)`
4. Direction: `(angle, sin(angle), cos(angle))`
5. Curvature delta between successive directions.
6. Distances to the five nearest QWERTY key centroids in `[-1,1]` space.
7. Progress and start/end flags.
8. Local window statistics (mean/std/range) over a 5-point neighbourhood.

These 37 features are what Android must reproduce (see `BeamDecode.kt`/`GreedyDecode.kt`). Any client implementation should call the Kotlin featurizer helper or port this exact logic.

---

## 2. RNNT Model & Training

- Model definition inherits from NeMo’s `EncDecRNNTModel` with conformer encoder (8 layers, `d_model=256`) and two-layer prediction network (`pred_hidden=320`).
- `PersonalizedRNNTModel.forward` bypasses NeMo’s audio preprocessor and consumes the 37-dim feature tensor directly (`train_transducer_personalized.py:388-404`).
- Knowledge distillation hooks (`kd_lambda`, `kd_temperature`) allow bootstrapping from a teacher checkpoint when available.
- Precision defaults to `bf16-mixed`; WER updates run inside `with torch.cuda.amp.autocast(enabled=False)` to keep accumulators in FP32 (fixing the earlier runtime mismatch).
- Training configuration (batch size 320, 4090M optimisations) lives in the top-level `CONFIG` dict.

Lightning callbacks and NeMo logging mirror the current run; checkpoints land under `rnnt_checkpoints_*` with timestamps.

---

## 3. Export & Android Runtime

- Encoder exports: use `export_optimized_onnx.py` or `export_pte_ultra.py` to produce INT8 ONNX / ExecuTorch binaries optimised for mobile. These scripts pull calibration batches from the validation manifest, ensuring quantisation stats match real gestures.
- Decoder exports: `export_rnnt_step.py` emits both ONNX and `.pte` single-step decoder graphs used by the Kotlin beam search.
- Kotlin runtime (`BeamDecode.kt`, `LexiconLoader.kt`) expects the RNNT encoder/decoder pair plus cached lexicon bundles. The 37-dim featuriser is embedded in Kotlin to guarantee parity with training.

---

## 4. Relation to Transformer Snapshot

| Aspect | RNNT Pipeline (this doc) | Transformer Snapshot (`trained_models/architecture_snapshot/`) |
| --- | --- | --- |
| Features | 37 dims with adaptive resampling | 6 dims `[x,y,vx,vy,ax,ay]` with pad/truncate to 150 |
| Model | RNNT (Conformer encoder + prediction network) | Transformer encoder/decoder with beam search |
| Exports | RNNT encoder + step decoder (ONNX/PTE) | Encoder/decoder ONNX pair + ExecuTorch greedy wrapper |
| Runtime | Kotlin RNNT decoders, optimised lexicons | JS beam search + web vocabulary | 
| Use case | Android production | Web demo / legacy references |

Keep the transformer snapshot for experimentation, but treat it as **legacy for Android**. Any new Android feature work should reference this RNNT document and the personalized training script.

---

## 5. Validation Checklist

Once training finishes:
1. Run `trained_models/nema1/beam_decode_onnx_cli.py --encoder encoder_android_*.onnx --decoder decoder.onnx --manifest ../../data/train_final_val.jsonl` to verify on-host beam search quality.
2. Push the new encoder/decoder into the Android demo and run instrumented swipe tests (short, long, tap-heavy words) to confirm latency and accuracy.
3. Capture WER/latency metrics and, if acceptable, package with `export_pte_ultra.py --package-dir release/android_<date>` for APK integration.

Document any deviations (e.g., different featuriser variants or teacher checkpoints) in `runtime_meta.json` so future exports remain traceable.

---

**Next actions when the current training run completes:** evaluate the checkpoint with the steps above, promote the best-performing export, and archive the manifest + settings next to the `.ckpt` for reproducibility.
