# Repository Guidelines (Oct 2025)

## Project Structure & Artifacts
- Training code centers on `new/train_transducer_personalized.py` (Conformer‑RNNT + NeMo) and orchestration runners:
  - `train_comprehensive.sh` (curriculum strategies, resumable)
  - `run_comprehensive_training.sh` (multi‑profile cycles, metrics CSV)
- Data in `data/` (JSONL swipe traces, `vocab.txt`).
- Sampling profiles in `new/sampling_profiles.py` (with aliases for runners).
- Export/ONNX in `new/export_*.py`.
- Artifacts (logs, checkpoints, metrics) are date‑scoped under `./9292025script/<yyyymmdd>/`.
  - Do not commit these unless intentionally versioned.

## Build, Test, and Development Commands
- `uv sync` — install Python deps.
- Run curriculum (400 total):
  - `./train_comprehensive.sh curriculum`
- Run multi‑profile cycles (resumable, metrics CSV):
  - `./run_comprehensive_training.sh`
- Fresh date base (today):
  - Uses `./9292025script/20251002` by default; change the date folder to start fresh.
- Force a single validation profile to compare across profiles:
  - `FORCE_VAL_PROFILE=validation_balanced ./run_comprehensive_training.sh`
- Toggle compile experiments later:
  - `ENABLE_COMPILE=1 ./run_comprehensive_training.sh`
- Direct trainer with overrides:
  - `CKS_RUN_BASE=./9292025script/20251002 uv run python new/train_transducer_personalized.py --profile sqrt_balanced --val-profile validation_balanced --batch-size 320 --num-workers 8 --max-epochs 100`
- Metrics aggregation:
  - `uv run python scripts/metrics_aggregate.py --base ./9292025script/20251002`
- TensorBoard:
  - `uv run tensorboard --logdir 9292025script`

## Coding Style & Naming
- Python 3.12, 4‑space indentation, type hints, use `logging`.
- Keep components small; separate loaders/models/decoding logic.
- snake_case for functions/variables; PascalCase for classes.
- Avoid committing large artifacts; keep them under the date‑scoped base.

## Testing Guidelines
- Prefer fast functional checks and instrumented dry runs (e.g., `FAST_DEV_RUN=1`).
- Validate vocabulary metadata before shipping: `uv run python trained_models/scripts/validate_vocab_system.py exports/runtime_meta.json`.
- For WER comparisons, fix a validation profile (env `FORCE_VAL_PROFILE`) and use the metrics aggregator.

## Commit & PR Guidelines
- Conventional commits (`feat:`, `fix:`, `docs:`). Focus on observable behavior.
- PRs should detail the training/decoding scenario, required data/config updates, and include WER/metrics or TensorBoard screenshots. Note any manual steps or artifact uploads.
