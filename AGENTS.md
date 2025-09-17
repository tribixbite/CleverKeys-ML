# Repository Guidelines

## Project Structure & Module Organization
CleverKeys centers on `train.py`, which packs the end-to-end Conformer + CTC pipeline and the `CONFIG` block contributors should tune for new datasets. Gesture corpora live under `data/` (JSONL) and generated vocabularies in `vocab/`, while export artifacts are staged in `exports/` and archived variants in `exports_sota/` and `archive/`. Supporting references and experiment notes are housed in `docs/`, platform assets in `android/` and `web-demo/`, and reusable helpers in `scripts/` and `trained_models/scripts/`. Logs, checkpoints, and released models stay in `logs*/`, `checkpoints*/`, and `trained_models/`—avoid committing heavyweight outputs unless they are deliberately versioned.

## Build, Test, and Development Commands
- `uv sync` – install and lock Python dependencies declared in `pyproject.toml`.
- `uv run python train.py` – launch the main training loop; ensure the `CONFIG` paths reference local data.
- `uv run python trained_models/scripts/validate_vocab_system.py path/to/runtime_meta.json` – sanity check vocabulary metadata before shipping exports.
- `uv run tensorboard --logdir logs_final` – inspect training runs (adjust to the log directory used).

## Coding Style & Naming Conventions
Target Python 3.12 with 4-space indentation, descriptive type-hinted APIs, and `logging` over bare prints. Keep modules small and data loaders, models, and decoding utilities in separate files when adding new components. Favor snake_case for functions and variables, PascalCase for classes, and mirror existing filenames such as `swipe_data_utils.py`. Large JSON or binary assets belong outside the core source tree unless required at runtime.

## Testing Guidelines
There is no automated suite today, so add targeted validation alongside features. Prefer fast functional checks (e.g., `validate_vocab_system.py`) and instrumented dry runs that exercise new featurizers or decoders on a trimmed JSONL sample. Document expected word error rate deltas in PRs and commit any helpful analysis notebooks under `docs/` rather than leaving them ad hoc.

## Commit & Pull Request Guidelines
Follow the conventional prefix style seen in history (`feat:`, `fix:`, `docs:`) and keep commit bodies focused on observable behavior. Pull requests should describe the training or decoding scenario affected, list required data/config updates, and attach before/after metrics or TensorBoard screenshots when model quality changes. Link issues or experiments that motivated the change and call out any artifact uploads or manual steps reviewers must perform.
