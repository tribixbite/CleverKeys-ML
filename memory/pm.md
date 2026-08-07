Project memory - training script update

Date: 2025-10-07

- Added new training entrypoint `train_distilhubert_ctc.py` implementing a DistilHuBERT + CTC pipeline using Hugging Face Trainer.
- Mirrors dataset paths from `train_squeeze2.py`:
  - `data/train_final_train.jsonl`
  - `data/train_final_val.jsonl`
- Uses a 37D gesture featurizer (position, velocity, acceleration, proximity, curvature) and adapts the first conv to accept 37 channels.
- Tokenizer: character-level `abcdefghijklmnopqrstuvwxyz'` with `[PAD]`/`[UNK]`. Processor is configured with `feature_size=37`.
- Auto-resume from latest checkpoint in `--output-dir` and saves best to `<output-dir>/best_checkpoint`.
- Default hyperparameters tuned for 16GB VRAM (batch_size=128, fp16 enabled when CUDA available). Adjust with CLI flags if needed.

2025-10-07 (update)

- Critical fixes applied per review:
  - Fail-fast conv adaptation: removed try/except around conv layer replacement in both HuBERT/W2V2 adapters. Script will error early if adaptation fails.
  - Removed feature upsampling; preserve true gesture length. Increased `min_points` from 3 -> 10 to filter ultra-short swipes.
  - Simplified collator to use `processor.feature_extractor.pad` and `tokenizer.pad`, with a small post-step to normalize to `(B, T, 37)` and build `attention_mask` when absent.
- Metrics now use jiwer CER with lowercase + empty-string sanitization; select best by `cer`.

2025-10-07 (update 2)

- Removed apostrophe from vocabulary and keyboard layout due to training keyboard lacking the key.
- Cleaned labels to only a-z; samples that reduce to empty after cleaning are filtered out.
- Switched collator to strict manual padding to `(B, T, 37)` with assertions and clear attention masks; added input_length for `group_by_length` sampler.
- Enabled `group_by_length=True` and provided `length_column_name="input_length"`.
- Added `empty_pred_rate` metric for visibility into empty decodes.
2025-10-10

- Upgraded `train_transformer_1.py` (lightweight Transformer + CTC) for mobile-feasible training:
  - Added CLI args: `--lr`, `--warmup-steps`, `--subset-train`, `--subset-val`, `--fast-test`, `--dry-run-first-batch`.
  - Switched to manual collator padding to strict `(B, T, 37)` with `attention_mask`.
- Enabled length grouping via `length_column_name="input_length"`.
- Replaced deprecated `evaluation_strategy` with `eval_strategy`.
- Added optional subset-before-map to speed fast tests.
- Preserved dataset paths: `data/train_final_train.jsonl`, `data/train_final_val.jsonl`.

2025-10-14

- Improved CUDA probing in `new/train_transducer_personalized.py`:
  - `_has_usable_cuda()` now returns `torch.cuda.device_count() > 0 and torch.cuda.is_initialized()` to avoid lazy-init quirks.
- Optional mixed precision with autocast:
  - Added CLI flag `--autocast-bf16`. When provided and BF16 is supported, wraps training/validation/test forward paths in `torch.cuda.amp.autocast(dtype=torch.bfloat16)`.
  - When enabled, Trainer precision is forced to `"32-true"` to avoid double autocast with Lightning; otherwise, existing precision config remains (default `bf16-mixed`).
  - Does not alter data logging/dry-run helpers beyond core forward path.
- Decoupled feature dimension from model config and vectorized featurization:
   - Removed hardcoded `FINAL_FEATURE_COUNT=37` and padding in `PersonalizedSwipeFeaturizer`.
   - `PersonalizedSwipeFeaturizer.feature_dim` now reports the active feature count dynamically based on `FEATURE_NAMES` (mobile vs full).
   - `build_dataloaders()` now returns `(train_loader, val_loader, feature_dim)` and `build_model_config(..., feature_dim)` wires `encoder.feat_in` to the dynamic value.
   - Rewrote featurizer to use NumPy vectorized ops for velocity/acceleration, angles/curvature, nearest-key distances, and 5-frame window stats (with a safe fallback). This significantly reduces per-batch CPU overhead.
 - Small structure refactor:
  - Extracted `try_compile_model(model, resume_from)` and `build_callbacks(cfg, args, root_dir)` to simplify `main()`.
  - Added basic logging via Python `logging` and converted several key prints to `log.info` for cleaner long-run logs.
  - Improved checkpoint selection semantics: prefer explicit `--checkpoint` if it exists; otherwise fall back to latest discovered checkpoint.
  - Added `DISABLE_COMPILE` env var guard with explicit skip message.
  - Fast test now reduces validation load (`limit_batches=0.1`).
  - `load_key_centers` is now memoized with `lru_cache(maxsize=1)` to avoid redundant loads across workers.
  - Robust resume from compiled checkpoints: `PersonalizedRNNTModel.on_load_checkpoint` normalizes `state_dict` keys by stripping `._orig_mod.` wrappers (from torch.compile), preventing mismatch on resume.
  - ONNX exporter now performs shape inference:
    - Infers `feature_dim` from checkpoint weights (prefers `pre_encode.out.weight`), builds matching config, and loads weights with normalized keys.
    - Exports a `runtime_helper.md` alongside ONNX files documenting dims, feature columns, and decoding steps.
  - Faster, more frequent validation logging:
    - Default validation reduced to `limit_batches=0.1` and `check_interval=0.25` (4×/epoch, lighter each time).
    - Added CLI overrides: `--val-limit-batches`, `--val-check-interval`.
    - Added `QuickValStats` callback to print `val_wer`, `val_loss`, average input length (`avg_T`) and average target length per validation run.

2025-10-14 (update)

- Added conditional coordinate normalization to `new/train_transducer_personalized.py`:
  - Added `--normalize` CLI flag to control coordinate normalization behavior.
  - When `--normalize` is passed, coordinates are transformed from [0,1] to [-1,1] (original behavior).
  - When `--normalize` is NOT passed, assumes data is already in [-1,1] coordinate system and skips normalization.
  - Updated `PersonalizedSwipeDataset` to accept `normalize_coords` parameter and pass it to `_prepare_points()`.
  - Modified `_prepare_points()` to conditionally apply normalization based on flag.
  - Still applies clamping to [-1.5, 1.5] in both cases for safety.

2026-08-07 — new `ctc/` pipeline (from-scratch CTC swipe encoder)

- Added `ctc/`: prepare_data / model / train / eval_beam / export_onnx / make_golden,
  plus vendored `futo_decoder_{eval,ceiling}.py` + `en_qwerty.json` from the
  CleverKeys app repo @ 79ddfb0f. Implements `docs/guides/train-ctc-swipe-model.md`
  with 15 pre-run audit fixes (see `ctc/README.md` for the numbered table).
- Headline fix: the recipe's fixed 8x8 cosine key basis is rank 23 of 26 at the
  canonical en_qwerty centers, so three emission directions were structurally
  unreachable. Replaced with a learned per-key embedding + matmul scoring; the
  trained embedding is rank 26 (cond 33) and the export contains no Einsum.
- Splits were contaminated: 298 train rows duplicate val/test bit-exactly and 977
  are train self-duplicates; prepare_data now drops them (109,600 of 110,876 kept).
- Runtime artifacts live in `~/ctc-train/` (data/cache/ckpt), never committed.
- Smoke run (6+2 epochs, ~4 s/epoch, RTX 5080): val greedy 50.31 %, beam top-1
  80.83 % on 120 val rows. Full run + G2/G4 gates still TODO.
- TODO: full 300-epoch run; match the baseline 131,544-word lexicon before making a
  formal G2 claim (staged wordlist yields 146,964); phase-2 refinement head (§11).
