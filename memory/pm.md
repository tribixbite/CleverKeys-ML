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

2026-08-07 (later) — phase-2 refinement head: G4 MISSED

- Added `ctc/train_refine.py`, `ctc/export_refine_onnx.py`, `CtcRefineHead` in
  `ctc/model.py`, and `eval_beam.py --refine-ckpt/--refine-onnx/--scoring`.
- Head (15.6K params) trained 60 epochs on frozen r2, 2.7 s/epoch. Best val greedy
  58.91 % @ epoch 40 vs r2's 58.00 % — only +0.9 pt (FUTO's lever was +25 pt greedy).
- G4 probe on val[0:2000]: enc-only 81.55 t1; refined 81.55 (dec preset) / 81.35
  (enc preset). Delta +0.00 pt against a >= +4 pt bar. 28 rows fixed, 28 broken.
  Short words +1.00, long words -0.54.
- Read: our base emissions are already far sharper than FUTO's 43.96 % greedy base,
  so the per-frame refiner has almost no headroom. Both scoring presets bracket the
  result, so it is not a preset artifact.
- Next options: temporal-context head (FUTO's magic_macaw is DFSMN, ours is strictly
  per-frame), --unfreeze-after end-to-end fine-tune, or drop phase 2 and sweep
  (gamma, beta, lambda) on val for the enc-only emissions.

2026-08-07 (close-out) — scoring sweep + unfreeze probe: both null, phase 2 closed

- Added `ctc/sweep_scoring.py`: caches sliced [N,32,27] emissions from the r2 ONNX
  once, then grid-sweeps beam scoring params. Key trick: (gamma,beta,lambda) only
  affect FINAL scoring, so the vendored beam runs once per PRUNE setting with
  gamma=beta=lambda=0 / top_k=beam_width and the grid is re-scored analytically.
  Verified it reproduces eval_beam's full-val numbers exactly (81.57/89.84/91.37)
  in seconds instead of 45 min.
- Sweep result: tuned preset (g 0.275, l 0.026, b 0.84, gp 0.3734, bp 0.9882) gave
  +0.45 t1 on the 2000 rows it was fitted to, +0.05 on untouched rows 2000-4000,
  and -0.01 on full val-9918. SE at n=2000 is ~0.87 pt, so the gain was noise.
  VERDICT: keep CtcScoringParams.encoderOnly unchanged; no free win exists.
- Headroom bound: a wider grid (405 coarse + 9x125 refinement points) run directly
  on all 9918 val rows -- selecting on the rows it is scored on, an optimistic upper
  bound -- tops out at 81.78 t1 vs 81.57 baseline, i.e. +0.21 pt MAXIMUM, while
  costing -0.01 on both t3 and t5 (trades 4+ 79.12->78.89 for <=3 86.28->87.34).
  The two sweeps also pick different winners, confirming a flat objective surface.
- Unfreeze probe (--unfreeze-after 10 --epochs 40): train loss 0.316 -> 0.276 but
  val greedy only 58.91 -> 59.16 % (+0.25, best @ epoch 22). Below the +0.5 pt bar
  so no beam probe was run. Phase 2 closed for good.
- Only untried structural idea if ever revisited: temporal context in the refine
  head (FUTO's magic_macaw is a DFSMN; ours is strictly per-frame).
- Ship candidate remains enc-only r2 with the published scoring preset.

2026-08-07 (phase D) — TODO plan: beam-selected checkpoints + T3 benchmark tier

- [ ] 1. train.py: beam-t1 checkpoint selection on a fixed 2,000-row val prefix
      (in-process, vendored futo_viterbi_beam + 146,964-word trie, fork pool).
      Select best.pt on beam2000 t1; keep greedy logged.
- [ ] 2. build_tiers.py: T3 = full FUTO swipe-1 train (potentially_invalid_sentence
      only, NO session exclusion, exact-trace dedup vs val/test in both hash forms)
      + the FULL How-We-Swipe release (1,338 users). Featurize -> cache/train_t3.npz.
- [ ] 3. Phase D arms on T3 at 94,000 steps, batch 256, lr 3e-3, beam-selected:
      D0 ch96 | D1 ch128/embed_hid128 | D2 ConvNeXt trunk | D3 winner + EMA 0.999.
      3 seeds (1234/4321/7777) for the top-2 by beam-selected val t1.
- [ ] 4. Bridge arm: D0 recipe on T1 tier, seed 1234.
- [ ] 5. eval_arms.py full-val beam t1/t3/t5 + per-source for every run.
- [ ] 6. PHASE_D.md with per-seed + mean tables, the T3 contamination disclosure,
      and the milestone-gate recommendation. test-2400 stays SEALED.
