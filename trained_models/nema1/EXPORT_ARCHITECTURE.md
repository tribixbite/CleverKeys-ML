# Export Architecture Documentation

## Overview

CleverKeys ships a single RNNT training graph and a focused export toolchain tailored for two deployment targets:

- **Web & Desktop** – ONNX Runtime (WASM / WebGPU / CPU / CUDA)
- **Android** – ExecuTorch (XNNPACK) with optional INT8 quantization

All encoder exports now share `export_common.py`, which centralises model loading, featurizer creation, and calibration data management. Calibration defaults to the validation manifest so quantization statistics always come from real swipe traces.

## Lexicon Preparation & Packaging

The swipe vocabulary remains decoupled from the exported graphs so Android builds can graft in OS/user dictionaries at runtime. When you need curated lexicons (for demos or locked-down deployments), run:

```bash
python trained_models/nema1/prepare_lexicons.py \
  --small-size 12000 \
  --full-size 70000 \
  --output-dir vocab/lexicons
```

Outputs:
- `vocab/lexicons/lexicon_small.txt` – compact list for dynamic builds (fast on-device personalisation)
- `vocab/lexicons/lexicon_full.txt` – pruned ~70k dictionary for showcase builds
- `vocab/lexicons/lexicon_stats.json` – summary of thresholds and retained frequency mass

Every export script now accepts `--lexicon`, `--package-dir`, and `--bundle-assets` flags. Use them to publish both flavours:

- **Dynamic builds**: omit `--lexicon`; package only encoder/decoder artefacts so runtime code can append user vocabularies.
- **Bundled builds**: supply `--lexicon` (usually the full lexicon) and any trie/LM metadata via `--bundle-assets` to produce a self-contained folder.

```
Training Checkpoint (.ckpt/.nemo)
│
├── Encoder → ONNX Exports
│   ├── export_onnx.py              (FP32 baseline + optional FP16 + INT8 QDQ)
│   └── export_optimized_onnx.py    (Multi-target INT8 + ORT graph optimisation)
│
└── Encoder → ExecuTorch PTE Exports
    ├── export_pte_fp32.py          (FP32 baseline)
    ├── export_pte_optimized.py     (Optimised FP32)
    ├── export_pte.py               (PT2E + XNNPACK INT8)
    └── export_pte_ultra.py         (Aggressively optimised INT8 with validation)

Decoder → export_rnnt_step.py      (Single-step decoder/joint, FP32 ONNX + PTE)
```

---

## Encoder Exports

### `export_onnx.py`
Baseline ONNX export with optional FP16 conversion and INT8 QDQ quantisation.

```bash
python export_onnx.py \
  --checkpoint trained.ckpt \
  --vocab vocab/final_vocab.txt \
  --fp32-output encoder_fp32.onnx \
  --fp16-output encoder_fp16.onnx \
  --int8-output encoder_int8_qdq.onnx
```

Key details:
- Uses `torch.onnx.export` with dynamic axes for batch/sequence length.
- INT8 quantisation pulls batches from `data/train_final_val.jsonl` (or `--calib-manifest`).
- FP16 conversion is handled via `onnxruntime.tools.convert_float_to_float16` when available.
- `--package-dir` plus `--lexicon/--bundle-assets` copies the models and optional vocabulary metadata into a ready-to-ship folder.

### `export_optimized_onnx.py`
Creates production-ready INT8 ONNX binaries for Web and Android plus optional FP16 baseline.

```bash
python export_optimized_onnx.py \
  --checkpoint trained.ckpt \
  --vocab vocab/final_vocab.txt \
  --web_onnx encoder_web_int8.onnx \
  --android_onnx encoder_android_int8.onnx \
  --fp32_base encoder_base_fp32.onnx \
  --fp16_base encoder_base_fp16.onnx \
  --create_ort_optimized \
  --compression_test
```

Highlights:
- Shared calibration loader with configurable batch count (`--calibration_batches`, `--calibration_batch_size`).
- Optional ORT graph optimisation emits `_ort.onnx` variants tuned for parallel (web) or sequential (android) execution.
- Size and dtype audits confirm INT8 tensors and report expected APK footprint.
- Set `--skip_quantization` when only the FP32/FP16 baselines are required.
- Packaging mirrors the baseline script: point `--package-dir` at an output folder and list lexicons/LMS under `--bundle-assets` to build dictionary-inclusive releases.

---

## ExecuTorch PTE Exports

### `export_pte_fp32.py`
Minimal FP32 ExecuTorch export for debugging or accuracy baselines.

```bash
python export_pte_fp32.py --checkpoint trained.ckpt --output encoder_fp32.pte
```

Add `--package-dir` (and optional `--lexicon`/`--bundle-assets`) to emit a deployment folder containing the `.pte` file plus accompanying metadata.

### `export_pte_optimized.py`
Applies XNNPACK graph partitioning to a non-quantised model for faster FP32 inference.

```bash
python export_pte_optimized.py --checkpoint trained.ckpt --output encoder_android_optimized.pte
```

### `export_pte.py`
PT2E + XNNPACK quantisation flow using real gesture batches.

```bash
python export_pte.py \
  --checkpoint trained.ckpt \
  --vocab vocab/final_vocab.txt \
  --output encoder_quant_xnnpack.pte \
  --calib-batches 64 --calib-batch-size 16
```

Pipeline:
1. `torch.export` generates an ExportedProgram for the encoder wrapper.
2. `prepare_pt2e` inserts observers using the XNNPACK quantiser (per-channel symmetric INT8).
3. Calibration iterates over dataloader batches (`data/train_final_val.jsonl` by default).
4. `convert_pt2e` freezes quantisation parameters.
5. ExecuTorch lowering partitions kernels for XNNPACK and emits `.pte`.

Package the resulting encoder with `--package-dir`, pointing to either the small (dynamic) or full (bundled) lexicon depending on the deployment.

### `export_pte_ultra.py`
Aggressive INT8 export with diagnostic checks, optional synthetic calibration fallback, and detailed size reporting.

```bash
python export_pte_ultra.py \
  --checkpoint trained.ckpt \
  --vocab vocab/final_vocab.txt \
  --output encoder_ultra_quant.pte \
  --calib-batches 32 --calib-batch-size 8
```

Features:
- Dataset-backed calibration by default; `--synthetic-calibration` retains the older random generator for quick smoke tests.
- Verifies presence of quantize/dequantize nodes and warns if XNNPACK partition ratio falls below 50%.
- Offers `--fallback_to_fp32` when quantisation fails but an optimised FP32 build is still desired.
- Reports raw and estimated compressed (APK) sizes to guide packaging decisions.
- `--package-dir` gathers the `.pte`, chosen lexicon, and any supplied metadata into a deployment-ready directory.

---

## Decoder Export

### `export_rnnt_step.py`
Exports the RNNT prediction + joint network for single-step inference in ONNX and ExecuTorch formats.

```bash
python export_rnnt_step.py \
  --checkpoint trained.ckpt \
  --onnx_out rnnt_step_fp32.onnx \
  --pte_out rnnt_step_fp32.pte \
  --vocab trained_models/nema1/words.txt
```

- Automatically detects embedding/LSTM layers and infers encoder dimensionality via a short forward pass.
- Warns when `<blank>` is absent from the supplied vocabulary (defaults to 0 in that case).
- ExecuTorch export benefits from XNNPACK partitioning even in FP32 for consistent mobile performance.
- Packaging support mirrors the encoder scripts; include decoder assets and (optionally) lexicons via `--package-dir`/`--bundle-assets`.

---

## Recommended Workflow

1. **Train / fine-tune** with `train_transducer.py` or `train_transducer_experimental.py`.
2. **Run validation** to log WER and confirm checkpoints in `trained_models/nema1`.
3. **Curate lexicons** (when needed) via `prepare_lexicons.py`, choosing between the small and full lists.
4. **Export ONNX** using `export_optimized_onnx.py`, optionally packaging models + lexicons with `--package-dir`.
5. **Generate ExecuTorch artefacts** with `export_pte.py` / `export_pte_ultra.py` (and FP32 fallbacks) tailored to the target deployment.
6. **Bundle decoder assets** via `export_rnnt_step.py` alongside vocabulary metadata (`runtime_meta.json`, tries, language models).
7. **Smoke test** exports using the provided Kotlin/TypeScript decoders (`BeamDecode.kt`, `beam_decode_web.ts`).

This pipeline keeps calibration tied to real swipe traces, exposes optional FP16 paths for ONNX, and surfaces diagnostics whenever quantisation quality could regress. Adjust batch counts or manifests as new datasets become available, and regenerate this documentation after adding new export targets.
