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

