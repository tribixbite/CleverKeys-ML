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
