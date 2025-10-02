# CleverKeys — Privacy‑First Gesture Typing

Local‑only swipe recognition. No data leaves your device.

Conformer‑RNNT model (NeMo) with personalized swipe featurization, resumable multi‑profile training, and date‑scoped artifact management.

## Setup

This project uses `uv` for Python dependency management.

- Install uv (if needed):
  - `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Install project deps:
  - `uv sync`

## Data Format

Training data is JSONL with normalized points in [0,1] and timestamps (ms):

```
{"word":"example","points":[{"x":0.12,"y":0.44,"t":0},{"x":0.15,"y":0.47,"t":12} ...]}
```

## Architecture

- Conformer‑RNNT (NeMo): presets via `--model-size {mobile,tablet,server}`
- Personalized 37‑D swipe features and adaptive resampling (~56–96 frames)
- CosineAnnealing scheduler with computed `max_steps` and warmup
- Knowledge distillation optional (teacher model)

## Training Options

We provide two runners for long‑running, resumable workflows. Artifacts are scoped by date under `./9292025script/<yyyymmdd>/`.

- Comprehensive cycles (multi‑profile):
  - `./run_comprehensive_training.sh`
  - Force a single validation profile for apples‑to‑apples WER: `FORCE_VAL_PROFILE=validation_balanced ./run_comprehensive_training.sh`
  - Enable compile experiments later: `ENABLE_COMPILE=1 ./run_comprehensive_training.sh`

- Curriculum strategy (4×100 epochs = 400 total):
  - `./train_comprehensive.sh curriculum`

All artifacts for today’s date (2025‑10‑02) are under `./9292025script/20251002`:

- Checkpoints: `rnnt_checkpoints_<profile>_<timestamp>/lightning_logs/.../checkpoints/*.ckpt`
- Periodic NeMo exports: `rnnt_checkpoints_<profile>_<timestamp>/*.nemo`
- Logs/metrics: `training_logs/` and `metrics_*.csv`
- Resume state: `training_state.json` (curriculum runner)

## Direct Trainer

Use when you need one‑off experiments (debugging, small runs):

```
CKS_RUN_BASE=./9292025script/20251002 \
uv run python new/train_transducer_personalized.py \
  --profile sqrt_balanced --val-profile validation_balanced \
  --batch-size 320 --num-workers 8 --max-epochs 100 \
  --train-manifest data/train_final_train.jsonl \
  --val-manifest data/train_final_val.jsonl \
  --vocab-path data/vocab.txt
```

Common env toggles (the runners set these for stability):

- `DISABLE_COMPILE=1 TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_CUDAGRAPHS=0`

## Profiles & Aliases

Sampling profiles are defined in `new/sampling_profiles.py`. The runner supports aliases:

- `short_common` → `short_words` (+ high‑frequency bias)
- `medium_balanced` → `medium_words`
- `base_random` → `uniform`
- `rare_words` → `rare_focused`
- `very_rare` → `ultra_rare_boost`
- `high_confusion` → `production_balanced`
- `production_current` → `production_balanced`
- `validation_current` → `validation_balanced`

## Monitoring & Metrics

- TensorBoard: `uv run tensorboard --logdir 9292025script`
- Per‑profile CSV metrics written by the comprehensive runner:
  - `scripts/metrics_aggregate.py --base ./9292025script/20251002`
    - Prints per‑profile best WER, latest WER, and overall best.
  - Set `FORCE_VAL_PROFILE=validation_balanced` during training to compare profiles fairly.

## Export & Runtime Metadata

- Vocabulary metadata: `scripts/make_runtime_meta.py` → `exports/runtime_meta.json`
- Validate vocabulary metadata: `uv run python trained_models/scripts/validate_vocab_system.py exports/runtime_meta.json`
- Canonical ONNX exporter (stateful pair): `new/export_stateful_pair.py`
  - Example (auto‑discover best .ckpt under date base):
    - `CKS_RUN_BASE=9292025script uv run python new/export_stateful_pair.py --outdir web-demo/models/best_latest --force-cpu`
  - Example (explicit checkpoint):
    - `uv run python new/export_stateful_pair.py --checkpoint 9292025script/.../epoch=74-wer=0.192.ckpt --outdir web-demo/models/best_latest --force-cpu`
  - Outputs: `encoder.onnx`, `decoder_joint.onnx`, `runtime_meta.json`
  - Note: Exporter pads tokens if `blank_id >= len(tokens)` to keep `runtime_meta.json` consistent.

## Notes

- The runners default to disabling `torch.compile` and CUDA graphs for NeMo stability. You can opt‑in with `ENABLE_COMPILE=1`.
- To start fresh any day, set a new date folder under `./9292025script/<yyyymmdd>`.
