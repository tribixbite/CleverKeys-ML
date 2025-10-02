# Implementation Summary (Training + Vocabulary)

This summary captures the current, working end-to-end training and export pipeline as of 2025-10-02, along with the vocabulary/metadata system used by the web demo and Android. It reflects the new training scripts, resumable orchestration, logging layout, and scheduler updates.

## Training Pipeline (Oct 2025)

### Components
- `new/train_transducer_personalized.py` (main trainer)
  - End-to-end swipe featurization (37-D), adaptive resampling, Conformer-RNNT with NeMo
  - CLI overrides for manifests/vocab (`--train-manifest`, `--val-manifest`, `--vocab-path`)
  - Model size presets via `--model-size {mobile,tablet,server}`
  - Knowledge distillation hooks (teacher optional)
  - CosineAnnealing scheduler enabled with computed `max_steps`
  - Checkpoint resume from `.ckpt` only; `.nemo` used for export
  - Stores all artifacts under a run base directory (`CKS_RUN_BASE`)

- `train_comprehensive.sh` (curriculum, single strategy)
  - Strategies: `curriculum`, `frequency`, `length`, `cyclic`, `all`, `test`
  - Fully resumable: persists state at `BASE_DIR/training_state.json`
  - Auto-detects batch size (or override with `BATCH_SIZE_OVERRIDE`)
  - Uses only `.ckpt` for resume, `.nemo` for reporting/export
  - Current run base: `./9292025script/20251002`

- `run_comprehensive_training.sh` (multi-profile cycles)
  - Cycles across stage profiles for days, writes metrics CSV per profile
  - Profile aliases supported (mapped to real sampling profiles)
  - Uses `CKS_RUN_BASE=./9292025script/20251002` and logs to `.../training_logs`
  - Stable defaults: disables torch.compile + cudagraphs from the runner

### Storage Layout
- Run base (date-scoped): `./9292025script/20251002`
  - Per-run checkpoints: `rnnt_checkpoints_<profile>_<timestamp>/lightning_logs/.../checkpoints/*.ckpt`
  - Periodic `.nemo` exports: `rnnt_checkpoints_<profile>_<timestamp>/*.nemo`
  - Logs: `training_logs/`
  - Runner metrics CSV: `training_logs/metrics_*.csv`
  - State (for resume): `training_state.json`

### Sampling Profiles
- Core profiles in `new/sampling_profiles.py` (e.g., `rare_focused`, `short_words`, `sqrt_balanced`, etc.)
- Aliases for orchestration:
  - `short_common` → `short_words` (adds high-frequency bias)
  - `medium_balanced` → `medium_words`
  - `base_random` → `uniform`
  - `rare_words` → `rare_focused`
  - `very_rare` → `ultra_rare_boost`
  - `high_confusion` → `production_balanced`
  - `production_current` → `production_balanced`
  - `validation_current` → `validation_balanced`

### Scheduler
- CosineAnnealing enabled with:
  - `warmup_steps` from config
  - `max_steps` computed as `ceil(len(train_loader) / accumulate) * max_epochs` (or 1 in FAST_DEV_RUN)
  - If NeMo logs “Scheduler will not be instantiated,” verify dataloader creation or FAST_DEV_RUN mode

### Resumption
- Only `.ckpt` is passed to Lightning (`ckpt_path=`) for reliable resume
- `.nemo` is produced periodically and at the end for export/shipping
- Resume strategies:
  - `train_comprehensive.sh` auto-continues saved strategy/profile and checkpoint
  - `run_comprehensive_training.sh` always uses the latest `.ckpt` under the date base

### WER Handling
- Per-profile WER is embedded in checkpoint filenames (`epoch=...-wer=val_wer=...ckpt`)
- The comprehensive runner logs per-profile rows to metrics CSV (stage, profile, epoch, checkpoint, WER)
- “Best” selection is by lowest WER across the current date base; for apples-to-apples, filter the CSV by profile

### Quick Starts
- Full curriculum (4×100 epochs total, resumable):
  - `./train_comprehensive.sh curriculum`
- Multi-profile cycles (metrics CSV, resumable):
  - `./run_comprehensive_training.sh`
- One-off direct trainer (with overrides):
  - `CKS_RUN_BASE=./9292025script/20251002 uv run python new/train_transducer_personalized.py \
      --profile sqrt_balanced --val-profile validation_balanced \
      --batch-size 320 --num-workers 8 --max-epochs 100`

### Notes
- compile/cudagraphs are disabled by default from the runners for NeMo stability; re-enable later if desired
- To start completely fresh, use a new date folder under `./9292025script/<yyyymmdd>`

# Vocabulary System Implementation Summary

## ✅ Changes Implemented

### 1. Runtime Metadata Generation
- **`scripts/make_runtime_meta.py`**: Creates `runtime_meta.json` from character vocabulary
- **`exports/runtime_meta.json`**: Generated metadata with derived IDs and character mappings
- **`web-demo/runtime_meta.json`**: Copy for web deployment

### 2. Export Script Updates
- **`trained_models/nema1/export_rnnt_step.py`**:
  - Added `--vocab` parameter for automatic blank_id derivation
  - Replaces hardcoded `blank_id=0` with derived value
  - Uses `torch.tensor([blank_id])` for start token

### 3. Vocabulary Utilities
- **`scripts/vocab_utils.py`**: Core filtering and normalization utilities
  - `normalize_word()`: Handles apostrophes and case normalization
  - `filter_dictionary()`: Filters words to valid characters only
  - `validate_vocabulary_coverage()`: Coverage statistics and validation
  - CLI interface for filtering, validation, and normalization

### 4. Web Integration
- **`web-demo/vocab-meta-utils.js`**: JavaScript utilities for browser deployment
  - `VocabMeta` class for metadata handling
  - `loadVocabMeta()`: Async metadata loading
  - `buildTrieFromWords()`: Filtered trie building with character IDs
  - `createLexicon()`: Complete lexicon creation with filtering
  - `normalizeWord()`: JavaScript word normalization

### 5. Android Integration
- **`android/VocabMetaUtils.kt`**: Kotlin utilities for Android deployment
  - `VocabMeta` data class with Kotlin-friendly accessors
  - `loadVocabMetaFromAssets()`: Android assets integration
  - `buildTrieFromWords()`: Trie building with character ID mapping
  - `createLexicon()`: Lexicon creation with coverage reporting
  - `normalizeWord()`: Kotlin word normalization

### 6. Documentation and Examples
- **`VOCAB_SYSTEM.md`**: Comprehensive system documentation
- **`examples/vocab_integration_example.py`**: Usage examples and patterns
- **`scripts/validate_vocab_system.py`**: Validation script with tests
- **`IMPLEMENTATION_SUMMARY.md`**: This summary document

## 🎯 Key Achievements

### Robust Vocabulary Management
- **Runtime Metadata**: Single source of truth for character mappings
- **Programmatic Derivation**: No hardcoded `blank_id` assumptions
- **Platform Sync**: Web and Android use identical mappings

### Illegal Token Prevention
- **Filtered Trie Building**: Only valid characters (`a-z`, `'`) in trie
- **No `<unk>` Expansions**: Beam search cannot expand to unknown tokens
- **No `<blank>` Expansions**: Blank token excluded from character expansions

### Apostrophe Normalization
- **Typography Handling**: Curly (`'`) → straight (`'`) apostrophe conversion
- **Case Normalization**: Uppercase → lowercase conversion
- **Whitespace Cleanup**: Trimming and normalization

### High Vocabulary Coverage
- **95%+ Coverage**: Maintains excellent dictionary coverage
- **Character Validation**: Only drops non-alphabetic characters
- **Quality Focus**: Filters edge cases while preserving core vocabulary

## 📊 Validation Results

```bash
$ uv run python scripts/validate_vocab_system.py
```

**All tests passed:**
- ✅ Runtime metadata validation
- ✅ Word normalization (6/6 test cases)
- ✅ Vocabulary filtering (valid words correctly identified)
- ✅ Character coverage (27 characters: `'` + `a-z`)

## 🚀 Usage Patterns

### Generate Runtime Metadata
```bash
python scripts/make_runtime_meta.py trained_models/data/vocab.txt --output exports/runtime_meta.json --pretty
```

### Export with Derived blank_id
```bash
python trained_models/nema1/export_rnnt_step.py --nemo_model model.nemo --vocab trained_models/data/vocab.txt
```

### Web Integration
```javascript
const meta = await VocabMetaUtils.loadVocabMeta("runtime_meta.json");
const lexicon = VocabMetaUtils.createLexicon(words, meta);
const results = await rnntBeamSearchWord(encoder, step, features, F, T, L, H, D, lexicon, {
    blankId: meta.blankId, // Derived, not hardcoded
    beamSize: 16
});
```

### Android Integration
```kotlin
val meta = VocabMetaUtils.loadVocabMetaFromAssets(context, "runtime_meta.json")
val lexicon = VocabMetaUtils.createLexicon(words, meta)
val decoder = RNNTBeamDecoder(encProg, stepProg, L, H, D, blankId = meta.blank_id)
```

### Vocabulary Validation
```bash
python scripts/vocab_utils.py validate --dictionary vocab/final_vocab.txt --meta exports/runtime_meta.json
```

## 🔄 Next Steps

1. **Deploy metadata**: Copy `runtime_meta.json` to web and Android assets
2. **Update decoders**: Integrate new utilities in beam search implementations
3. **Test end-to-end**: Validate beam search with filtered vocabulary
4. **Monitor coverage**: Ensure vocabulary filtering maintains quality

## 📝 Notes and Tips

### Character Set
- **Allowed**: `'` (apostrophe) + `a-z` (lowercase letters)
- **Filtered**: Numbers, punctuation, accents, uppercase, spaces

### Apostrophe Handling
- Normalizes typographic apostrophe (U+2019) → straight apostrophe (U+0027)
- Preserves contractions: `"don't"`, `"it's"`, `"won't"`

### Vocabulary IDs (derived at export)
- NeMo with `blank_as_pad=True` keeps a single functional blank and places it at the end in practice.
- Always use the IDs from `runtime_meta.json` (do not hardcode). Typical mapping for 30 tokens:
  - `<blank>` index 29, `<unk>` index 28, `'` index 1, `a` index 2 … `z` index 27.
  - For some archives, vocabulary may serialize with 29 tokens; the exporter pads the list with `""` to ensure `blank_id` is valid.

### Coverage Expectations
- **Target**: 95%+ of dictionary words preserved
- **Typical loss**: Numbers, foreign words, technical terms
- **Quality focus**: Maintains core English vocabulary

This implementation provides a robust, consistent vocabulary system that prevents illegal token expansions while maintaining high accuracy and platform synchronization.
