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

- [x] 1. train.py: beam-t1 checkpoint selection on a fixed 2,000-row val prefix
      (in-process, vendored futo_viterbi_beam + 146,964-word trie, fork pool).
      Select best.pt on beam2000 t1; keep greedy logged.
- [x] 2. build_tiers.py: T3 = full FUTO swipe-1 train (potentially_invalid_sentence
      only, NO session exclusion, exact-trace dedup vs val/test in both hash forms)
      + the FULL How-We-Swipe release (1,338 users). Featurize -> cache/train_t3.npz.
- [x] 3. Phase D arms on T3 at 94,000 steps, batch 256, lr 3e-3, beam-selected:
      D0 ch96 | D1 ch128/embed_hid128 | D2 ConvNeXt trunk | D3 winner + EMA 0.999.
      3 seeds (1234/4321/7777) for the top-2 by beam-selected val t1.
- [x] 4. Bridge arm: D0 recipe on T1 tier, seed 1234.
- [x] 5. eval_arms.py full-val beam t1/t3/t5 + per-source for every run.
- [x] 6. PHASE_D.md with per-seed + mean tables, the T3 contamination disclosure,
      and the milestone-gate recommendation. test-2400 stays SEALED.

2026-08-07 (phase D) — beam selection adopted, ch128 adopted, T3 rejected, gate NOT spent

- train.py selects best.pt on beam top-1 over a fixed 2,000-row val prefix (pool
  forked once pre-CUDA, trie copy-on-write, shared RawArray emissions, ~2 s/val
  point). Golden-checked: reproduces r2's committed val[0:2000] 81.55/89.85/91.65
  exactly. eval_arms now reports the <=3 / 4+ strata too.
- T3 built: full FUTO swipe-1 (929,568 kept) + FULL How-We-Swipe release (1,338
  users, 78,155 kept) = 1,007,723 rows -> 1,005,336 cached. NO session exclusion
  (documented in PHASE_D.md §2). parse_hws_log reproduces all 60,303 unique
  canonical HWS traces bit-exactly; prepare_data re-dedup found 0 extra leaks.
- Seed-mean val t1 (3 seeds, paired): D1 ch128 on T3 = 84.81 (sd 0.56);
  ch128 on T1 = 84.38 (sd 0.15). Paired t(2) = 1.31 -> tiers indistinguishable.
  T3 buys FUTO (+1.35) and 4+ (+0.87), costs HWS (-0.48) and <=3 (-0.41).
- ch96 -> ch128 = +1.07 pt on T3, the only arch lever that has ever gained here.
  Costs 0.31 -> 0.49 ms single-thread CPU (+60 %).
- ConvNeXt regresses again UNDER beam selection (-0.87 vs D1), so the Phase-B
  mis-selection hypothesis is dead. EMA is a null (-0.13).
- vs FUTO ceiling: t1 84.81 vs 84.83, t5 +0.25, 4+ +0.75, but <=3 -1.56. Since
  test ran 0.61 BELOW val on r2, expected test ~84.2 -> gate NOT recommended yet.
  Next levers: close <=3 (length-conditioned score / refine head, which gave
  +1.00 on <=3), scale ch further, widen the selection prefix to 5,000 rows.

2026-08-07 (phase E) — TODO plan: close the gap to the FUTO ceiling on val-9918

THE BAR (FUTO ceiling measured on OUR val-9918, from the app repo's committed eval):
overall t1/t3/t5 = 85.52 / 91.54 / 92.80, <=3 t1 89.29 (n=3389), 4+ t1 83.57 (n=6529).
D1 seed-mean deficits: overall -0.71, t3 -0.53, t5 -0.47, <=3 -1.28, 4+ -0.42.
Gate to unseal test-2400: 3-seed seed-mean full-val beats ALL FIVE. test-2400 stays
SEALED this phase regardless.

- [x] E1 scoring re-tune on D1. ADOPTED (gamma 1.05, lambda 1.1, beta 0.2, gp 0.3734,
      bp 0.9882): +3.04 t1 on the untouched holdout half, +2.74 full val. Four grids
      hit a boundary before the fifth converged -- the published preset was ~0.6 in
      gamma and ~60x in lambda away from the optimum for our emissions. RETRACTS the
      README "no free win" verdict and its +0.21 pt headroom bound (grid-width
      artifact; re-swept r2 gains +4.25 on untouched rows).
- [x] E2 refinement head on the D1 seed-1234 base. NEGATIVE: -0.58 vs its own base on
      the same val[0:5000] at the E1 preset. Phase-2's null reproduces on a strong base.
- [x] E3a T4 tier (curated FUTO at benchmark scale): -0.26 t1, negative on all five.
      Third independent negative for the user's quality cascade. Rejected.
- [x] E3b T3 with HWS oversampled 3x (npz concatenation, not row duplication):
      +0.83 t1 at seed 1234, HWS half +1.45 vs FUTO +0.20. ADOPTED.
- [x] E4 capacity ch=192 embed_hid=192 on T3: +0.48 t1 vs ch128 paired at seed 1234,
      inside the ~1 pt single-seed floor. Latency needs an idle re-measure (1.54 ms
      mean / 1.90 ms p90 taken under load, vs 0.49 ms for ch128).
- [x] E5 beam-selection prefix 2000 -> 5000: +0.23 t1 paired at seed 1234. Adopted.
- [x] FINAL stack (ch192 + T3-3xHWS + 5,000-row selection + E1 preset), 3 seeds.

2026-08-08 (phase E) — GATE PASSES: all five FUTO-ceiling numbers beaten on val-9918

- Seed-mean full val-9918 (seeds 1234/4321/7777, E1 preset):
  t1 88.06 (sd 0.23) | t3 92.32 | t5 93.08 | <=3 90.86 | 4+ 86.62
  bar 85.52 / 91.54 / 92.80 / 89.29 / 83.57  ->  +2.54 +0.78 +0.28 +1.57 +3.05. ALL PASS.
- Also passes on the 4,959 val rows used by NEITHER the preset sweep nor checkpoint
  selection: 87.58 / 92.03 / 92.85 / 90.67 / 85.98 (+2.06 +0.49 +0.05 +1.38 +2.41).
  t5 is the narrow one there -- level with the ceiling rather than ahead of it.
- test-2400 NOT decoded. eval_arms.py/train.py guards untouched. Orchestrator owns
  the milestone + audit.
- Biggest lever by far was the SCORING PRESET, not the model: the published
  encoderOnly preset is ~0.6 off in gamma and 60x off in lambda for our emissions.
  Four grids hit a boundary before the fifth converged.
- ch192 vs ch128 at the final tier, 3 PAIRED seeds: +0.19 t1 (not the +0.48 one seed
  showed), -0.12 on <=3, 1.9x latency. ch128 clears the same gate at 0.470 ms and is
  the better shipping trade.
- Machine rebooted mid-phase; E3a/E3b were resumed from last.pt with the cosine
  horizon intact (step-budget mode makes --resume exact).

2026-08-08 (post-decode) — SEAL SPENT, claim verified, hygiene landed

- test-2400 decoded ONCE per checkpoint exactly as pre-registered (AUDIT_PREDECODE.md
  §E), audited post-hoc (AUDIT_FINAL.md, 937e112). Verdict: claim holds AS REGISTERED.
  ch192 seed-mean 88.36/92.65/93.50, <=3 91.37, 4+ 86.81
  ch128 seed-mean 87.92/92.33/93.00, <=3 91.08, 4+ 86.29
  bar 84.83/91.04/92.08/89.57/82.40 -> all five, and on all six individual runs.
- BUT only 2 of 5 bars are statistically resolved (t1 z3.6, 4+ z3.4). t3 ~2sigma,
  t5 1.9, <=3 1.2 -> positive point estimate, NOT resolved. Never quote "beats FUTO"
  without the preset asymmetry: at the published preset the same model clears 3 of 5
  on val (85.78/91.66/92.67/88.10/84.58). The tuning is worth +2.29 pt t1.
- THE SEAL IS SPENT. No further test-2400 decode is legitimate, including a "fair
  rematch" at the published preset -- argue that from the val control instead.

- [x] HYGIENE 1: dedup key now normalize_word(word) in build_tiers.hash_row (via
      scan_futo_sessions.trace_hash) and prepare_data.trace_hash. Tiers deliberately
      NOT rebuilt per AUDIT_PREDECODE §E; audit bounded the defect at <0.05 val /
      0.20 test pt, all bars still clearing on every seed, and the leaked rows score
      4.34 pt BELOW comparable non-leaked ones (no memorization signal).
- [x] HYGIENE 2: seal.py -- content guard replacing the filename substring. Hashes the
      rows a run actually LOADS against 2,400 committed fingerprints; refuses >1%
      overlap unless --unseal-test. Threshold non-zero because val and test genuinely
      share 7 traces. Verified it refuses the split, a RENAMED copy and a 120-row
      slice (both of which the old guard passed) and lets val through.
      DISCLOSED: verifying the override branch decoded 3 test rows. No number used.
- [x] Corrected PHASE_D.md §2 "caught every match on both sides" and PHASE_E.md §5
      "removed bit-exactly" -- both false as written (AUDIT_FINAL §6.2).
- [x] RESULTS.md Campaign 2 section; Campaign 1 retained + marked superseded.
- [x] artifacts/: 6 ONNX (ch128/ch192 x 3 seeds, sha256 verified byte-identical to the
      decoded checkpoints) + golden fixture regenerated from ch128_s1234 AT THE E1
      PRESET (make_golden.py gained --preset + provenance).

- [ ] APP-SIDE (not this repo): G3 wiring of ch128_s1234.onnx; CtcScoringParams ->
      (1.05, 1.1, 0.2, alpha 0.0, 0.3734, 0.9882) -- REQUIRED, not optional; land
      ctc_golden.json so CtcParityTest actually runs (it currently fails its own
      file-existence assertion); NOTICE attribution for the FUTO corpus (MIT) and
      How-We-Swipe / OSF sj67f (MIT); re-measure latency on a phone little core.

2026-08-08 — PHASE F (latency): target <=0.15 ms single-thread batch-1 CPU

Goal: an ONNX encoder at <=0.15 ms (half the campaign-1 r2 artifact's 0.306 ms,
~3x the ch128 ship candidate's 0.455 ms) that still clears all five FUTO-ceiling
VAL bars (85.52/91.54/92.80, <=3 89.29, 4+ 83.57) at the E1 preset, 3-seed mean.
test-2400 is SEALED-SPENT — Phase F evidence is val-only, by construction.

- [x] Instruments: bench_latency.py (AUDIT_PREDECODE §7 protocol + ORT op profiling
      + optimized-graph serialization), quantize_onnx.py (dynamic/static QDQ int8
      with exclusion lists), arch_latency.py (price an architecture at random init
      before training it).
- [x] PROFILE: ch128 is 66 % Conv (8 nodes, ~46 us each), ~9 % GroupNorm
      (Reshape/InstanceNorm/Reshape/Mul/Add = 5 nodes x 9 norms), rest is
      per-node dispatch on a 32-frame graph. Node COUNT matters as much as MACs.
- [x] F1 quantization measured (see PHASE_F.md).
- [ ] F2 students: resbn (dense + foldable BatchNorm) and dwsep blocks, KD from
      our own ch192 teacher.
- [ ] F3 = F2 + static int8, if it pays.
- [ ] Final candidate at 3 seeds + PHASE_F.md.

Phase F round 1 (seed 1234, T3+2xT3hws, 94k steps, KD w1.0 T2 from ch192-s1234),
full val-9918 at the E1 preset — bar 85.52/91.54/92.80/89.29/83.57:

  A resbn:64:1,2,4    143.7k  0.135 ms  85.89/91.48/92.50  <=3 88.76  4+ 84.41  2/5
  B resbn:48:1,2,4,8  111.3k  0.120 ms  86.39/91.41/92.38  <=3 89.82  4+ 84.61  3/5
  C dwsep:128:1,2,4,8  97.8k  0.141 ms  85.78/91.39/92.27  <=3 88.40  4+ 84.42  2/5

- t1, 4+ and (for B) <=3 clear comfortably; **t3 and t5 are the binding bars**.
- Depth beats width at equal latency (B > A) and dense beats depthwise-separable
  (B > C) — the brief's dwsep hypothesis is measurably the wrong shape here, because
  dense and separable have the SAME MAC/param ratio at T=32 and the dense kernel
  vectorizes ~1.8x better (68 vs 38 GMAC/s measured).
- Re-tuning the scoring preset on B buys +0.12 t1 / +0.05 t3 / +0.03 t5 over the
  transferred E1 preset. The preset transfers; t3/t5 must come from the model.

Phase F round 2 (same recipe, seed 1234):
  D resbn:64:1,2,4,8  185.1k  0.160 ms  86.70/91.84/92.78  <=3 89.44  4+ 85.28  4/5
    -> misses ONLY t5, by 0.02 pt. Same knife-edge Phase E hit at E1 (92.79 vs 92.80).
  G resbn:56:1,2,4,8  145.6k  0.149 ms  (running)
  H = G's arch at 188k steps (the free lever: these students underfit badly,
      ctc_loss 0.42-0.47 against the teacher's 0.30)
  I resbn:80:1,2,4,8  279.3k  0.210 ms  (the pareto point above the target)
Killed as dominated/too slow at 5-6 concurrent: E resbn:48:1,2,4,8,16,
F resbn:40:1,2,4,8,1,2 (both trailing G at step 21000). Disclose as killed, not as arms.

Phase F, seed 1234, full val-9918 at E1 (bar 85.52/91.54/92.80/89.29/83.57):
  G resbn:56:1,2,4,8  145.6k  0.149 ms  86.25/91.67/92.61  <=3 89.52  4+ 84.55  4/5 (t5 -0.19)
  D resbn:64:1,2,4,8  185.1k  0.160 ms  86.70/91.84/92.78  <=3 89.44  4+ 85.28  4/5 (t5 -0.02)
  I resbn:80:1,2,4,8  279.3k  0.210 ms  87.41/92.18/92.85  <=3 90.38  4+ 85.86  **5/5**
F3 (student + static int8) REJECTED: -1.03 (I) / -1.48 (G) t1, and int8 barely helps
latency at this size (QDQ adds Q/DQ nodes; the graph is dispatch-bound, not
arithmetic-bound). I int8 falls 5/5 -> 3/5.
Killed for GPU budget, disclose as killed not as arms: E resbn:48x5, F resbn:40x6,
H resbn:56x4 at 188k steps (the step-budget lever, untested), J the no-KD ablation.
Seeding: FINAL = resbn:80:1,2,4,8 (5/5 at 0.210), FAST = resbn:64:1,2,4,8 (0.160,
t5 knife-edge) -- seeds 4321/7777 each, 94k steps.

PHASE F COMPLETE (PHASE_F.md). Answer: <=0.15 ms is NOT reachable with all five
val bars intact. Measured boundary, 3 seeds each:
  FAST resbn:64:1,2,4,8  185.1k  0.162 ms  86.82/91.85/92.67/89.86/85.24  4/5 (t5 -0.13)
  FINAL resbn:80:1,2,4,8 279.3k  0.213 ms  87.47/92.13/92.89/90.35/85.98  **5/5 on the
        seed mean AND on every individual seed**; 2.23x faster + 2.45x smaller than
        the Phase-E ch128 candidate for -0.55 t1.
Artifacts: artifacts/fast_resbn{80,64,56}_*.onnx, sha256 + parity in PHASE_F.md §9.
VAL-ONLY: the seal is spent; ch128/ch192 remain the only test-validated anchors.
Untested levers cut for GPU budget (PHASE_F §7.1): 188k-step schedule (the students
underfit, train CTC 0.42-0.47 vs teacher 0.30 -- most promising remaining lever and
free at inference), 5-/6-block narrow trunks, and the no-KD ablation.

- [ ] NEXT (optional): re-run resbn:56/64 at 188k steps x 3 seeds to see whether the
      free step budget closes the 0.13-0.19 pt on t5 at <=0.162 ms.
- [ ] NEXT (optional): retrain ch128 itself with the resbn trunk -- folding BatchNorm
      is a strict improvement (50 fewer ONNX nodes, no accuracy cost) and would make
      the test-validated anchor faster without changing its capacity.

Phase F round 3 — EXTENDED SCHEDULE (coordinator directive: test the underfit lever).
188k steps, same recipe+KD, seed 1234, full val at E1:
  resbn:48:1,2,4,8,16  134.6k  0.139 ms  86.64/91.80/92.53/89.73/85.04  4/5 (t5 -0.27)
  resbn:56:1,2,4,8     145.6k  0.144 ms  86.79/91.83/92.65/90.26/84.99  4/5 (t5 -0.15)
  resbn:64:1,2,4,8     185.1k  0.161 ms  87.19/92.09/92.76/90.29/85.59  4/5 (t5 -0.04)
Delta vs 94k: +0.5 t1, +0.2 t3, +0.8 <=3, +0.4 4+ ... and +0.04 / -0.02 on t5.
Train CTC 0.4425->0.4284 (ch56) and 0.4178->0.4039 (ch64): doubling the schedule buys
0.014, while ch56->ch80 buys 0.061 and ch128 buys 0.141. The models are UNDER-CAPACITY,
not undertrained -- the underfit reading in PHASE_F 7.1 was wrong in its mechanism.
No arm cleared all five at seed 1234, so per the decision rule none earned a seed round.

Round 4 running: M56-280k (the coordinator's 280k contingency on the best <=0.153 arm),
N72-188k (resbn:72:1,2,4,8, 229.6k, 0.185 ms) and O56x5-188k (resbn:56:1,2,4,8,16,
177.3k, 0.166 ms) -- the last two target criterion (2): clear all five below 0.213 ms.

BREAKTHROUGH (round 4): resbn:72:1,2,4,8 @ 188k steps, 229.6k params, **0.185 ms**,
seed 1234 full val at E1: 87.25/92.24/**92.96**/90.44/85.59 -> **ALL FIVE CLEAR**,
and its t5 margin (+0.16) beats resbn80@94k's (+0.09 at 3 seeds, 0.213 ms).
=> the bar-clearing frontier moves 0.213 -> 0.185 ms. Seeds 4321/7777 launched.
Still running: M56-280k (<=0.153 exhaustion test), O56x5-188k (resbn:56:1,2,4,8,16,
177.3k, 0.166 ms -- if it clears, the frontier moves again), P56-188k-T4 (KD
temperature ablation; clearly behind T=2 at equal step, likely a negative).

Round 4 results (seed 1234, full val at E1):
  M56-280k  resbn:56:1,2,4,8 @280k  0.144 ms  86.83/91.85/92.67/90.23/85.07  4/5
      -> the schedule ladder on the SAME arch: 94k 92.61 t5, 188k 92.65, 280k 92.67.
         TRIPLING the schedule moves t5 +0.06 total; the bar needs +0.19. Criterion (1)
         is answered to exhaustion: <=0.153 ms cannot clear t5 by training longer.
         Train CTC 0.4425 -> 0.4284 -> 0.4192 (still far above ch128's 0.3017).
  N72-188k  resbn:72:1,2,4,8       0.185 ms  87.25/92.24/**92.96**/90.44/85.59  **5/5**
  O56x5-188k resbn:56:1,2,4,8,16   0.166 ms  87.07/91.92/92.74/90.20/85.45  4/5 (t5 -0.06)
  P56-188k-T4 KD temperature 4     0.144 ms  86.20/91.66/92.45/90.38/84.03  NEGATIVE
      -> vs T=2 at the same 188k: -0.59 t1, -0.20 t5. The temperature lever aimed at
         t5 moves t5 the wrong way. (T^2 scaling also 4x's the effective KD weight.)
Running: N72 seeds 4321/7777 (188k), Q68-188k (resbn:68, ~207k params, ~0.173 ms)
probing whether the clear point drops below 0.185.

PHASE F FINAL (extended rounds complete). Two success criteria answered:
 (1) all five bars at <=0.153 ms: **NOT REACHABLE**, tested to exhaustion.
     resbn:56:1,2,4,8 at 94k/188k/280k -> t5 92.61/92.65/92.67 against a 92.80 bar.
     TRIPLING the schedule = +0.06 t5. KD temperature 2->4 = -0.20 t5 (negative).
     Train CTC 0.4425->0.4192 over 3x the steps, still far above ch128's 0.3017:
     these models are UNDER-CAPACITY, not undertrained.
 (2) lowest latency clearing all five: **0.186 ms**, resbn:72:1,2,4,8 @188k, 229,642
     params, 3 seeds -- mean 87.27/92.09/92.87/90.49/85.60, and EVERY seed clears.
     2.55x faster + 2.96x smaller than the ch128 Phase-E candidate for -0.61 t1.
     Frontier improved from 0.215 -> 0.186 ms. Probes pin the crossing between
     206.7k params (0.176 ms, t5 92.74, fail) and 229.6k (0.186 ms, pass).
     resbn:80 @0.215 stays the conservative pick: worst-seed t5 margin +0.05 vs +0.01.
 t5 vs params is the whole story: 92.53 @134.6k -> 92.96 @229.6k -> 93.03 @689.3k,
 flat in everything else varied. Crossing at 210-230k params.
Artifacts: fast_resbn72_s{1234,4321,7777}, fast_resbn80_s{1234,4321,7777},
 fast_resbn{56,64}_188k_s1234 (frontier evidence, under the bar). sha256 + parity in
 PHASE_F.md 9. STILL VAL-ONLY: seal spent, ch128/ch192 remain the test anchors.
- [ ] Only unmeasured lever left: the no-KD ablation (PHASE_F 11.3). KD weight also
      never swept. Everything else in the phase brief has been run.

2026-08-08 — FUTO WEIGHTS VERIFICATION (independent re-run of the bar on this box)

- [x] Download FUTO's real weights (HF `futo-org/futo-swipe`) and re-run the app repo's
      harness on val-9918 + test-2400 to verify the committed bar numbers on x86_64.
      sha256 of both .pte VERIFIED against the documented values (encoder 725242ba…,
      decoder 01eaf16a…). ExecuTorch 1.2.0 cp310 manylinux x86_64 + torch 2.11.0+cpu
      runs both XNNPACK-delegated .pte natively — no proot/aarch64 needed.
- [x] test-2400 at the pre-fix DROP trie (131,544, the config of the committed table):
      ceiling 84.83/91.04/92.08, <=3 89.57, 4+ 82.40, in-vocab 88.48/94.96/96.05,
      greedy 69.12, macro 91.39 — EXACT on every published digit. Floor 79.25/87.71,
      strata 82.45/77.60, macro 83.28 exact; only t5 +0.04 and greedy -0.13.
- [x] val-9918 at the STRIP trie (146,964, the config the val row used):
      ceiling 85.54/91.52/92.78, <=3 89.29, 4+ 83.60 vs the bar 85.52/91.54/92.80/
      89.29/83.57 -> +0.02/-0.02/-0.02/0.00/+0.03. All five bar metrics inside 0.04 pt.
- VERDICT: the bar is CONFIRMED on this hardware; no bar metric moves enough to change
      any Phase-E/F gate outcome (smallest margin was t5 +0.28, drift is 0.02).
- LICENSE: FUTO weights were RUN for benchmarking only. No FUTO output entered any
      training loop, was saved as training data, or influenced model/preset selection.
- Report: `ctc/FUTO_WEIGHTS_VERIFICATION.md`. Scratch run tree (not committed):
      `~/ctc-train/futo_verify/` (venv, artifacts, verbatim harness copies, per-trace out).

2026-08-08 — O3 APP-LEXICON VALIDATION + RESBN80 TEST-VALIDATION (user-ordered)

Two tasks, both scoped and pre-registered before any decode.

- [x] T1. O3 lexicon validation (val-9918, NOT sealed). The E1 preset was fitted
      against the 146,964-word AOSP-frequency STRIP trie; the app ships the 98k
      `en_enhanced.json` trie (byte frequencies floored at 134..255). Verify the
      `--vocab-json` loader semantics (frequency scale vs the lambda term), then
      decode full val-9918 for `ch128_s1234` AND `fast_resbn80_s1234` at the E1
      preset with the app trie. Compare against the five val bars
      (85.52 / 91.54 / 92.80, <=3 89.29, 4+ 83.57).
- [x] T1b. If any bar fails: re-sweep LAMBDA ONLY (the frequency-scale knob) on
      val[0:4959], confirm on val[4959:9918], report the adjusted preset as the
      app-ship preset candidate.
- [x] T1c. Quantify the OOV impact of the 98k trie vs the 147k one.
- [x] T2. resbn80 test-validation — SECOND UNSEALING OF test-2400, ORDERED BY THE
      USER (the benchmark owner). fast_resbn80 was frozen on val-only evidence in
      Phase F; its training/selection never saw test; the first unsealing decoded
      only ch128/ch192. Decode test-2400 ONCE per seed for
      `fast_resbn80_s{1234,4321,7777}` at the frozen E1 preset + STRIP-146,964
      trie, identical protocol to the Phase-E decode (`eval_beam.py --preset
      1.05,1.1,0.2,0.3734,0.9882 --beam-width 100 --top-k 8 --unseal-test`).
      Pre-state: val seed-mean 87.47/92.13/92.89/90.35/85.98 and the observed
      val->test shifts of +0.30 (ch192) / +0.04 (ch128). Bars: 84.83/91.04/92.08,
      <=3 89.57, 4+ 82.40. Also at the en_enhanced trie IFF T1 makes that the ship
      configuration. HARD CAP: max 2 configs x 3 seeds, one decode each, no
      iteration.
- [x] T3. Document: RESULTS.md (resbn80 evidence tier val-only -> test-validated,
      with the second-unsealing disclosure paragraph), PHASE_F.md footnote, seal
      ledger. Commit.

2026-08-08 — ALT-LAYOUT / CROSS-LANGUAGE EMPIRICAL EVAL (user-ordered)

Independent of the O3/resbn80 block above; touches NO sealed split (test-2400 is
not read by any step here) and NO file that block owns.

Question: the encoders take layout geometry as an INPUT and were trained with
slot-permutation + affine-jitter augmentation on en_qwerty ONLY. Layout-agnostic
by design, validated on nothing. Does it transfer? Answer with real corpora.

- [x] A1. Regenerate the five FUTO `swipe-5` per-layout corpora (dvorak/en,
      azerty/fr, qwertz/de, german/de, spanish/es) — real single-finger human
      swipes. Vendored fetcher `ctc/fetch_futo_multilayout.mjs`; official
      geometries to `ctc/layouts/` (verified byte-identical to the app repo's
      committed `src/test/resources/layouts/futo_*.json`).
- [x] A2. Establish the frame mapping. Corpus x,y and the layout key centers are
      BOTH already normalized over the [0,1] letter area, so the mapping is the
      identity — proven, not assumed, by an endpoint-key proximity metric plus a
      wrong-geometry falsification control.
- [x] A3. Alphabet + lexicon policy: a-z emissions as trained; NFD-fold accents,
      STRIP `'`/`-`, count untypeable (ss/oe/ae) separately. fr/de/es tries from
      the app's bundled CKDT-v2 dictionaries; en = the campaign's 146,964 STRIP
      trie, so dvorak varies GEOMETRY ONLY.
- [x] A4. Harness sanity control: reproduce ch128_s1234 on en_qwerty val-9918
      at E1 (published 88.02 / 92.27 / 93.03) BEFORE trusting any alt-layout number.
- [x] A5. Decode all five layouts, both key-slot arms, vs the geometric-engine
      anchors. Lexicon-frequency-scale confound bounded by a lambda ablation.
- [x] A6. Report: `ctc/ALT_LAYOUT_EVAL.md` + `eval_altlayout.py`. Commit.

RESULT (both tasks complete, 2026-08-08).

T1 -- O3 CLEARS AT THE UNCHANGED PRESET. The frequency-scale risk is real
(log_freq spread 5.403 -> 0.643, sd 1.354 -> 0.089, Spearman 0.844 on the 83,113
overlap) but coverage moves the OTHER way: the 98k app trie has FEWER OOV targets
than the 147k AOSP one (val 2.52% vs 3.39%, test 2.67% vs 3.58%) -- only 12 val /
2 test rows are in AOSP and not in the app trie, while it adds 14,820 words AOSP
lacks. DROP vs STRIP is a null (en_enhanced has no apostrophes; 148 junk
skeletons; every number bit-identical).
The bar was RE-MEASURED on the app trie from FUTO's cached ceiling emissions so
the comparison is trie-matched (validated: reproduces FUTO_WEIGHTS_VERIFICATION
4b/4c to the digit). App-trie val bar 85.59/91.82/93.20/89.05/83.80 (t5 is +0.42
HIGHER than the AOSP bar); app-trie test bar 84.92/91.54/92.96/89.57/82.52.
val-9918, E1 preset, app trie, 3 seeds each -- ALL FIVE CLEAR, EVERY SEED:
  ch128    87.96/92.77/93.67/91.49/86.12  (worst-seed margins +2.15..+2.16)
  resbn80  86.93/92.39/93.51/90.51/85.07  (worst t5 93.45, +0.25)
Controls at the AOSP trie reproduce the committed numbers exactly (ch128
88.02/92.27/93.03/91.12/86.41; resbn80 87.41/92.18/92.85/90.38/85.86).
lambda-only re-sweep (not required, ran anyway): optimum 2.0 (ch128) / 2.5
(resbn80), +0.58 / +1.01 t1 on the untouched holdout half. NOT recommended --
diverges the shipped preset from the fixture and every published number.

T2 -- SECOND UNSEALING (user-ordered). Pre-registered in PHASE_F 16 and COMMITTED
(50c303a) before the decode. 6 decodes, one each, no retry. fast_resbn80 test-2400:
  config A (AOSP 146,964) 87.29/91.89/92.82/91.17/85.30 vs bar 84.83/91.04/92.08/
    89.57/82.40 -> +2.46/+0.85/+0.74/+1.60/+2.90, ALL FIVE, EVERY SEED (z 3.4/1.5/
    1.3/1.5/3.0)
  config B (app 98,081)   86.51/92.28/93.25/90.76/84.33 vs bar 84.92/91.54/92.96/
    89.57/82.52 -> +1.59/+0.74/+0.29/+1.19/+1.81, ALL FIVE, EVERY SEED; worst-seed
    t5 margin +0.08 = 2 rows of 2400
The pre-registered prediction was WRONG IN THE UNFAVOURABLE DIRECTION on 4 of 5
(-0.35/-0.46/-0.30/+0.51/-0.79): resbn80's val->test shift is -0.18/-0.24/-0.07/
+0.82/-0.68, opposite in sign to both Phase-E anchors. The val->test offset is not
a per-split constant that extrapolates across architectures.
Tier: fast_resbn80 val-only -> TEST-VALIDATED. fast_resbn72 and everything else in
Phase F remain val-only and were NOT decoded. test-2400 has now been read twice;
ledger in test2400_seal.json["test-2400"]["unsealings"].

2026-08-08 — FAIR REMATCH (both engines val-tuned): the equal-footing question is answered

- [x] Swept FUTO's scoring preset on val-9918 with Phase E's own machinery
      (sweep_scoring.py imported, not reimplemented; tune val[0:4959], confirm
      val[4959:9918], reject boundary winners). Ceiling converged interior after 4
      grids: gamma 1.15 lambda 1.3 beta 0.2 gp 0.3734 bp 0.7. Floor after 4 grids:
      0.35 / 4.8 / 1.6 / 0.05 / 1.4.
- [x] Tuning buys FUTO +1.94 t1 on val (85.54 -> 87.48) and +2.20 on test
      (84.92 -> 87.12), vs the +2.29 it bought us. THE ASYMMETRY WAS MATERIAL:
      ~2/3 of the published test margin was the untuned-vs-tuned comparison.
- [x] One test-2400 decode of FUTO's engine (ours NOT re-decoded; frozen
      test2400_e1.jsonl dumps re-read only). Equal footing, STRIP trie:
      bar 87.12/92.29/92.96/89.94/85.68
      ch192 +1.24/+0.36/+0.54/+1.43/+1.14 -- all five still win
      ch128 +0.79/+0.04/+0.04/+1.15/+0.61 -- t3/t5 are ties (1 trace)
- [x] Exact paired McNemar (now possible): ch192 resolves on 2 of 3 seeds,
      ch128 on 0 of 3. The ranking survives the rematch; the size of the lead
      does not. Equal-footing claim allowed for ch192 (qualified), still
      forbidden for ch128 -- the ship candidate.
- SIDE FINDING: the val-tuned FLOOR (encoder only, 85.97) beats the PUBLISHED
      ceiling (85.54). magic_macaw is worth +1.51 val / +1.33 test once both
      configs are tuned, not +5.88 -- futo-decoder-eval-notes' "the decoder is
      the whole lever" is preset-conditional.
- OPEN: hungry_jellyfish (FUTO's context LM) is still not in the bar -> the bar
      remains a floor on FUTO's full published stack. FUTO floor sweep may not be
      fully exhausted (was still creeping when it went interior).
- Report: `ctc/FAIR_REMATCH.md`; RESULTS.md asymmetry section marked SUPERSEDED.
- [x] ADDENDUM: fast_resbn80 (test-validated by a concurrent session same day) added
      to the equal-footing table at its config A (AOSP 146,964, matched footing).
      It FAILS 3 of 5 against the val-tuned bar: t1 +0.17, t3 -0.40, t5 -0.14,
      <=3 +1.23, 4+ -0.38. McNemar unresolved on all three seeds, one net NEGATIVE.
      Its five-of-five pass holds only against the published preset. This bears on
      the shipping choice -- the 0.215 ms variant's accuracy case rests entirely on
      the untuned comparison.
- [ ] NOT verified by me: the resbn80 config-B bar (app 98,081 trie) from PHASE_F
      15.2. Out of scope here; only config A was re-checked.

2026-08-09 — Phase G (affine-sampler fix + upgraded resbn student recipe)

User directive (2026-08-09): fix the latent affine-sampler truncation, upgrade the
recipe, retrain the resbn80-class ship candidate to push BOTH accuracy and latency,
per-model FULL preset sweep, export+validate winner, and a PRE-AUTHORIZED third
unsealing of test-2400 ("retrain and reexport and re-run tests on new onnx
(resbn80)") gated on the 3-seed val seed-mean clearing all five val bars.

- [x] G0 fix train.py affine sampler (coupled; acceptance 1.0; legacy kept; affine_stats.py verifies: legacy sx mean 0.9554/31.4% rejects -> coupled uniform [0.85,1.1111] mean 0.9807): couple translate to sampled scale, sample
      uniformly over the per-axis feasible region (acceptance 1.0 by construction);
      keep legacy path behind --affine-sampler legacy; verify before/after realized
      scale distribution + acceptance rate; document.
- [x] G0b ensemble --kd-teacher implemented (logsumexp - log n).
- [x] G1 lever decomposition DONE (full val, E1): A legacy+KD 87.46; B coupled+KD 87.52; C coupled+noKD 88.04; D ensemble-KD 87.07; E legacy+noKD 87.94. KD is -0.5 t1 (first-ever ablation); affine fix +0.06/+0.10; 188k at ch80+KD only +0.05. at ch80/188k, seed 1234, paired arms:
      A legacy sampler (schedule lever vs phaseF-I-resbn80x4@94k),
      B coupled sampler (affine lever vs A),
      C no-KD (KD ablation vs B — never measured in F),
      D 3-seed ch192 ensemble teacher (teacher lever vs B).
- [x] G2 winner (coupled+noKD+188k, resbn80g) 3 seeds: val seed-mean 87.72/92.25/92.97/90.78/86.14 -- all five bars, every seed, margins >= incumbent on all five. Equal-footing val bar NOT cleared on seed mean (t3 -0.06, t5 -0.06, 4+ -0.15).
- [x] G3 latency push: resbn72g 3 seeds val 87.62/92.22/93.02/90.48/86.14 --
      all five bars every seed at 0.184ms, exceeds fast_resbn80's seed-mean on
      all five (val-only tier). resbn64g (0.161ms) stays 4/5 (t5 92.70): no-KD
      does not transfer to ch64 -> phase F <=0.15ms verdict unchanged.
- [x] G4 preset sweep: AOSP converges to E1 exactly (keep); app trie interior winner 0.9/4.0/0.25/0.25/0.9882 (+1.39 t1 holdout-confirmed) -- ADOPTED as app preset; golden fixture regenerated from resbn80g_s1234 at that preset (sha ce3b5456...).
- [x] G5 exports: resbn80g x3 + resbn72g x3 in artifacts/ with sha256; parity
      argmax 100/100 all; idle latency 0.213/0.184/0.161 ms; resbn72g_s4321
      parity margin thin (2.1e-4 worst draw, argmax clean) -- disclosed.
- [x] G6 third unsealing: pre-registered (commit 46aecb1) then decoded 6 runs.
      Config A (AOSP/E1) test seed-mean 87.68/92.18/92.82/90.80/86.08 -- ALL FIVE
      published bars, every seed -> resbn80g TEST-VALIDATED. Config B (app trie,
      app preset) 88.14/93.22/93.90/91.86/86.23 -- all five, every seed, worst-seed
      t5 margin +0.75. Equal-footing: 3 of 5, McNemar unresolved every seed (+17
      p.17/+23 p.052/+0 p1.0) -- level, no claim. Ledger appended (unsealings[2]).
- [x] G7 PHASE_G.md complete (levers, factorial, preset, unsealing, frontier);
      RESULTS.md phase-g section + frontier addendum; pm.md closed. Phase G DONE.

2026-08-09 — Phase H (layout-resampling augmentation: close the dvorak gap)

Directive: build the geometry-sampling augmentation the recipe named and skipped
(train-ctc-swipe-model.md §6 item 3; evidence ALT_LAYOUT_EVAL.md — dvorak t1
63.04 vs geo engine 76.8; cause = key re-arrangement, not slot permutation).
Recipe otherwise Phase-G (resbn80 class, 188k, coupled sampler, no KD). Dvorak
HELD OUT of training-geometry sampling (true transfer probe). NO test-2400.

- [x] H0 warp design: residual re-anchoring of cached [2,64] QWERTY paths onto a
      sampled geometry via the word's ideal polyline (monotone-DP point-to-segment
      correspondence, tangent/normal residual transfer); ctc/layout_aug.py with
      validation CLI (identity round-trip, ideal-path exactness, endpoint-proximity
      stats vs ALT_LAYOUT_EVAL §2 real-corpus band).
- [x] H1 train.py: --layout-alt-p / --layout-synth-frac / real-layout pool
      (azerty,qwertz,german,spanish; dvorak+qwerty excluded); per-geometry affine
      feasible bounds; commit atomically (concurrent phaseG runs unaffected).
- [x] H2 p sweep (winner p=0.5: val 87.66, dvorak 88.85 aosp / 88.20 app-trie vs 76.8 anchor) 0.15/0.3/0.5 at seed 1234 (phaseH-p15/p30/p50), Phase-G recipe.
- [x] H3 eval per candidate (all six layouts beat geo anchors at p50, single seed): eval_altlayout.py all six real corpora at E1 (az26)
      + dvorak app-trie arm + full val-9918 eval_beam (AOSP/E1). Gate: dvorak t1
      >= 76.8 with en_qwerty val within 0.3 of 87.72 seed-mean; else map pareto.
- [x] H4 winner 3 seeds (resbn80h): val seed-mean 87.69/92.22/93.00/90.79/86.08 (all bars, every seed; -0.03 t1 vs resbn80g); dvorak held-out 90.01 aosp / 89.51 app vs geo 76.8; all six layouts beat geo; latency 0.216ms (same graph); artifacts resbn80h_s* + sha256.
- [x] H5 PHASE_H.md complete (design, warp validation, p ablation, 3-seed tables, routing update: displacement gate obsolete); RESULTS.md phase-h section at top. NOT test-validated (no unsealing requested).
      tables, routing/gate update); RESULTS.md Phase-H section appended at top.

2026-08-09 — MODEL_COMPARISON.md (cross-phase reference, standalone)

- [x] ctc/MODEL_COMPARISON.md — model cards (resbn80g / fast_resbn80 superseded /
      ch128 / ch192 / old shipped transformer / FUTO published + val-tuned),
      accuracy tables on all three footings kept in separate columns, the latency
      ladder incl. per-stage web-demo + desktop-JVM + Python figures with the
      non-comparability caveat, the four deltas that matter, a ship matrix with
      the fixture+preset move-together rule, and the caveat register (incl.
      resbn80g cross-layout UNMEASURED). Every number cited to a named doc
      section; no new measurement run. Phase-G F72/H64 latency probes noted as
      in progress.

2026-08-09 — Phase I-A (capacity under the new ~10ms budget; ship = highest
accuracy + max versatility, <=5MB preferred smaller; NO test-2400 anywhere)

User directive: latency is no longer the constraint (old 2x target was vs the
~178ms transformer; users can't feel <10ms). Unlimited training compute. A
concurrent Phase I-B data agent may share the GPU; coordination via commits.
This agent (I-A) owns train.py.

- [x] I1 capacity ladder UP with Phase-H layout aug (p=0.5) + Phase-G recipe
      (no KD, coupled sampler, 188k, T3+3xHWS, 5000-row beam-t1 selection):
      resbn ch128 / ch192 / ch256 (+ deeper probe if a rung says so), seed 1234
      per rung; measure val-9918 (AOSP/E1) + all six alt-layout corpora per
      rung (does layout aug invert ch128's memorization-vs-transfer trade?).
- [x] I2 size levers so capacity fits <=5MB: fp16 weight storage (parity +
      argmax stability + CPU-EP latency), weight-only int8 (fp32 compute —
      sidesteps the PHASE_F MASK_NEG activation catastrophe); bytes/accuracy
      per variant on the capacity winner.
- [x] I3 training-code headroom: multi-layout checkpoint selection (PHASE_H
      names QWERTY-only selection a known gap); T_OUT=64 emission-resolution
      probe (contract-breaking — measure, report as app decision, don't adopt);
      aug interactions + lr/schedule at capacity if underfit/instability shows.
- [x] I4 export-code: BN-fold drift at bigger widths; ORT graph profiling at
      ch256; ORT offline-optimized serialization size/latency.
- [x] I5 preset sweep for the final candidate on both footings (AOSP + app
      trie), holdout-confirmed; 3 seeds for the final pick.
- [x] I6 PHASE_I.md: findings ranked, capacity/size/transfer table, <=5MB ship
      recommendation, artifacts + sha256 + parity.

2026-08-09 — Phase I-B (data quality + language versatility; owns data-prep /
corpus tooling in NEW files; NO test-2400 anywhere; app repo read-only)

Concurrent with Phase I-A (owns train.py + capacity runs). I-B arms use the
frozen Phase-G/H recipe (resbn80-class, layout-alt p=0.5, no KD, 188k) with only
the training-npz composition varying; control (d) = phaseH-p50 s1234 already
trained + evaled.

- [x] B1 HWS filtering arms (the never-applied native/quality filtering):
      build_hws_arms.py — per-uid englishLevel from hws_full .json sidecars;
      arms: (a) native+advanced, (b) native only, (c) all levels + HWS-derived
      duration/speed/point-count gates (thresholds from measured HWS
      distributions, NOT the FUTO cascade that measured negative), (d) control.
      Rebuild T3-FUTO-only npz + per-arm HWS npz; verify control reproduces
      t3hws 78,155.
- [x] B2 train arms a/b/c at the frozen recipe, seed 1234 (share GPU politely);
      judge on val-9918 per-source (esp. HWS half) + per-englishLevel val
      breakdown + contributor-overlap disclosure (T3 is contributor-dirty; level
      filters change the leak asymmetrically) + all-six alt-layout suite.
      Winner -> feed I-A capacity runs.
- [x] B3 Cyrillic feasibility (a): Yandex Cup 2023 corpus — LIVE at
      disk.yandex.ru/d/IYiSpLob-zAxqg (data.zip 1.63GB, sha256 2e65d7a2…,
      downloading to ~/ctc-train/data/yandex_cup). Inventory + license check +
      format/geometry mapping.
- [x] B4 Cyrillic (b): cyrillic_synth.py — residual-transplant generator
      (layout_aug warp machinery) English residuals -> Russian ideal polylines
      on ЙЦУКЕН (33 letters, 64-slot contract); ru lexicon from app langpacks;
      PHASE_H-style endpoint validation + real-data probe from a held-out
      Yandex slice if B3 lands.
- [x] B5 Cyrillic (c): if train+eval data usable — multi-script prototype (ru
      alphabet in the data pipeline + ru trie; model unchanged), honest first
      Cyrillic decode measurement.
- [x] B6 PHASE_I_DATA.md: HWS arm table (per-source val + alt-layout), Cyrillic
      verdict + measured numbers, data-asset inventory + licenses, commits.

2026-08-09 — Phase I-B COMPLETE (ctc/PHASE_I_DATA.md)

- HWS arms (frozen H recipe, seed 1234, val-9918 + alt-layout suite):
  control 87.66 t1 (hws-half 81.09) | quality 87.71 (80.89) | nativeadv
  87.30 (80.25) | native 87.33 (80.14). englishLevel filtering NEGATIVE on
  every slice including the leak-matched native rows (81.97 -> 81.43/81.34);
  hws-derived motion gates a statistical tie (mild + point estimate).
  -> I-A: keep T3+3xHWS as-is; hws_quality.npz acceptable drop-in; NO level
  filter. (fourth exclusion-curation negative in the campaign.)
- Cyrillic: Yandex Cup 2023 corpus LIVE + verified (6.0M jcuken swipes,
  license unstated - research-use caution, no ship without owner call).
  ru-real prototype (94k, committed train.py, greedy sel): valid-10k in-dict
  t1 89.64 / 95.82 / 96.97 app-ru trie, greedy 75.2 — English-class.
  ru-synth (residual transplant, ZERO real rows): 76.21 in-dict t1 —
  geometric-engine class from synthesis alone. Next rung (not run): synth
  pretrain + small real finetune; joint multi-script needs per-row layout
  batching in train.py (I-A's file).

2026-08-10 — Yandex corpus licence question CLOSED (ctc/YANDEX_LICENSE_RESEARCH.md)

- [x] Web research: contest task, Yandex Cup 2023 Положение + Правила (2023
      archive snapshots), Общие условия конкурса, Yandex services user
      agreement, Yandex Disk ToU, ГК РФ ст.1333-1335.1, Kaggle mirror, both
      solution repos, HF, arXiv. Verdict: NO licence grant anywhere; the
      "open unless stated otherwise" assumption is refuted.
- Binding constraints found: Yandex ToS authorises only *personal
  non-commercial* use of content reached via Yandex services (rules cl.
  6.2 / 2.8.1); corpus is a protected database under ГК РФ ст.1334 (6 M rows
  vs the 10 k presumption; term to ~2039); ст.1335.1 permits research /
  education / insubstantial parts, not a shipped product. Every available
  permission theory is non-commercial → structurally incompatible with
  GPL-3.0 freedom 0.
- DECISION (pending owner confirmation): Yandex data = held-out RU eval set
  only (the FUTO pattern, arXiv 2606.25247 §4.1 — they trained EN-only on
  MIT swipe.futo.org and used Yandex purely for RU validation). Anything
  that ships is synth-only, accepting the measured 89.64 -> 76.21 in-dict t1
  cost. Never fetch the accepted/suggestion_accepted archive (real Yandex
  Keyboard user input) or the Kaggle mirror that bundles it.
- [ ] Owner call: ask Yandex for written GPL-compatible permission? (no
      contact made; needs explicit per-instance approval). Durable fix is
      our own RU data — HWS has zero Russian rows (en 815 / es 40 / ar 9 …),
      so the options are RU prompts into swipe.futo.org, our own donation
      flow, or more synth-generator work.

2026-08-10 — Yandex licence RE-REVIEW: the proshian precedent (§10 of
ctc/YANDEX_LICENSE_RESEARCH.md). Verdict: **CONFIRMED** (eval-only stands).

- [x] Read proshian's full trail: local fork, upstream README/report/thesis,
      GitHub API (licences, releases, issues), Google Drive re-hosts, HF,
      kbrodt + 7 other solution repos, Kaggle uploader identity.
- Findings: he goes FURTHER than this memo allows — re-hosts the preprocessed
      corpus on two personal Drive folders + a DVC gdrive remote, publishes
      competition weights, runs a live demo, and ships an MIT-licensed Android
      library whose tree contains a byte-identical copy of the Yandex voc.txt
      (sha256 b85623d0…, 503,598 lines) plus .pte weights, released as an APK.
      Three in-tree copies of voc.txt across his repos. kbrodt (1st) publishes
      weights, no data, no licence. Everyone else is code-only.
- Not a permission: nemo dat (his MIT cannot license Yandex's database), no
      Yandex counterparty ever spoke anywhere, and conduct ≠ construction.
      Visibility is tiny (6-18 stars, 5 model downloads) so non-enforcement
      proves ~nothing.
- [x] NEW primary sources §§2-3 missed, both cutting AGAINST us: (i) Яндекс.
      Контест ToU yandex.ru/legal/contest_termsofuse cl.2.4/4.1 — explicit ban
      on распространение + любое использование в коммерческих целях, service
      granted for личное некоммерческое использование; (ii) yandex.ru/cup/ml
      2023 snapshot — «Все задачи построены на основе реальных обезличенных
      данных Яндекса» (ownership assertion, not a release).
- [x] Enforcement: github/dmca has ~2 dozen Yandex notices incl. takedowns of
      participants' own Yandex.Lyceum solutions (2022-03-30, 2024-03-11).
      None touch NeuroSwipe, but Yandex IS an active DMCA enforcer in the
      competition/educational space — worse, not better, than assumed.
- [x] Thesis checked (ITMO MSc, 11.06.2024, sup. Nikolenko, 61pp): one sentence
      of provenance, zero hits for лиценз/соглашени/этик/персональн/авторск/
      правообладат/GDPR. No arXiv/DOI version. He never wrote about the
      question at all.
- Ops deltas added to the memo: never pull data/weights/voc.txt from proshian
      (false licence + unauditable provenance); never lift voc.txt as a
      wordlist (it arrives disguised as trie.ser / an app asset / MIT file);
      keep github.com/tribixbite/neural-swipe-typing clean (public, currently
      only .dvc POINTERS for the Yandex splits — verified, no Yandex bytes);
      attribute the corpus in any published RU numbers (ст.1335.1 last sent.).
- [ ] Unrelated hygiene spotted: our neural-swipe-typing fork is public with
      NO LICENSE, inherited from an unlicensed upstream. Decide + add one.

2026-08-10 — research scan + on-device personalization design study (ctc/RESEARCH_SCAN.md)

- [x] Part 1: web scan (arXiv/ICASSP/Interspeech/ICLR 2024-26) for higher CTC accuracy
      at our scale: CR-CTC (never run), InterCTC/self-conditioned, Bayes-risk/MWER
      sequence training vs the beam metric, Zipformer-tiny, SAM/ASAM, stochastic depth,
      label priors / peakiness (FUTO emission-count regularizer), blank-penalty decode,
      augmentation theory. Rank + top 2-3 concrete experiment specs. Do NOT re-recommend
      refuted levers (KD, EMA, >188k schedule, int8, dwsep, refinement head, curation).
- [x] Part 2: LOCAL-ONLY on-device personal fine-tuning design: (a) ORT on-device
      training / ExecuTorch status, (b) parameter-efficient variants (BitFit/LoRA/
      adapter/per-user input calibration), (c) non-gradient (lexicon freq adaptation,
      emission recalibration, residual-bank synth + head retrain). Rank, v1 + v2
      proposal, cite app-side hooks (MLDataCollector, personalization/, swipe_ml prefs).
- [x] Deliver ctc/RESEARCH_SCAN.md, lowercase conventional commit, no push.
      Outcome: top picks CR-CTC (spec'd) + FUTO-parity aug (shear/rotation/
      time-reversal, spec'd) + ASAM secondary; blank-penalty sweep axis +
      beam-selected checkpoint soup as free riders; MWER/differentiable-beam
      ruled out (2026 oracle-gap negative). Personalization: v1 = CTC-side
      personal lexicon/rerank + gradient-free input calibration; v2 = Kotlin
      head-only CTC fine-tune (ORT on-device training is deprecated @1.19.2,
      ExecuTorch training experimental — no framework dependency taken).

2026-08-10 — dataset scout: every additional swipe corpus we could train on (ctc/DATASET_SCOUT.md)

- [x] "leon" dataset identified + measured: it is HF `leonweber/swipe` = futo-org
      swipe-1 with every row duplicated EXACTLY 10x (9,395,500/542,690/499,700 vs
      939,550/54,269/49,970). Local 1,734,660 rows collapse to 173,463 unique traces
      (multiplicity {10: 173,460; 20: 3}); 95.53% are bit-exact members of our own
      FUTO pools; the 7,746 residual are rare-word rows our MIN_WORD_FREQ gate
      dropped. **1,352 of our 12,299 holdout traces (11.0%) are inside it.**
      REJECTED - not even as a measured arm.
- [x] Two corrections to DATA_TIERS.md §5: the file is NOT 30-point standardised
      (min 8 / median 57 / max 470, real irregular device timestamps - the 30-point
      column is `trajectory_sampled`, which we never used), and the licence is not
      unknown in substance (unlicensed redistribution of MIT FUTO data).
- [x] FUTO remainders: swipe-1 UNCHANGED at 939,550 (LFS oids identical to the
      2025-03-11 upload). swipe-2/3/4/5 added 2026-06-15 = 28,095 + 38,228 + 50,300
      + 59,247 = 175,870 new MIT rows, same schema/frame, session-disjoint from
      swipe-1 (68 sampled sessions, 0 hits vs the raw 5.1 GB train.jsonl).
      swipe-4 = confusable words; swipe-3 = max unique words + deliberate
      misspellings; swipe-2 = informal; swipe-5 = 11 layouts / 8 languages incl.
      11,805 clearflow + 1,058 kasroz on layouts no arm has trained on.
      HAZARD: swipe-5 dvorak/azerty/qwertz/german/spanish IS our alt-layout eval set.
- [x] WordGesture-GAN local pull (49,228 rows) assessed + REJECTED: GAN outputs
      scraped from a now-dead demo endpoint (no licence, model-output rule), exactly
      1 trace/word, fixed 128 points, frame needs a fitted affine (x s=1.086
      o=-0.045, y s=1.221 o=+0.008) after which endpoints are still 0.512/0.520 vs
      the 0.79-0.91 real-corpus band. Our residual-transplant synth dominates it.
- [x] Systematic sweep (HF 50+ hits, Kaggle, Zenodo, figshare, Dryad, Dataverse,
      IEEE DataPort, OSF, GitHub, arXiv/ACM): the complete list of real human
      corpora is FUTO + How-We-Swipe + Yandex + a 3,129-trace CC0 TU Delft VR set.
      NO cleanly-licensed real non-Latin swipe corpus exists anywhere. SHARK2/
      Kristensson-Zhai lineage never deposited a corpus; How-We-Type is tap-only;
      WordGesture-GAN / Gesture2Text / AdaptiKeyboard data never released.
      FUTO is the ONLY OSS keyboard that collects and publishes traces, and the
      collection is still live and accepting new languages/layouts.
- [x] Deliver ctc/DATASET_SCOUT.md, lowercase conventional commit, no push.
- [ ] NEXT (ranked trial arms, none run): #1 `swipe2345q` = T3 + swipe-2/3/4 +
      swipe-5 qwerty (~128k new rows, one fetch ~0.6 GB, no eval put at risk);
      #2 HWS frame correction (~0.064 Y offset, zero data cost - the swipetest
      geometry in §4.1 gives an independent re-derivation); #3 `realalt` =
      swipe-5 clearflow/kasroz/toki_pona with a fresh 20% holdout, testing real
      alt-layout data vs synthetic layout_aug.
- [ ] Highest-leverage non-engineering item: seeding a Cyrillic/other-script
      collection run at swipe.futo.org is the ONLY route to clean non-Latin real
      data (they shipped Shavian on request).
      RESULT: winner resbn192i (ch192 + layout-alt p0.65) 3-seed val
      88.30/92.60/93.26/91.27/86.77 (all bars every seed; +0.61 t1 vs resbn80h);
      dvorak held-out 89.13 (+12.3 vs geo); ship bytes fp16w 3,052,318 B @ 0.831ms;
      app preset 0.975/3.0/0.35/0.25/0.9882; ch256 frontier 88.65 recorded
      (dose-unscaled transfer volatile); t64 probe = app decision (+0.33 4+,
      +2.5-2.8 transfer, 2x beam); NOT test-validated (no unsealing). Phase J
      handoff in PHASE_I.md §9 (ch256+p-scaled dose, T64 bundle, per-row layout
      batching for ru, CR-CTC/aug/blank-penalty levers on the resbn192i base).

2026-08-10 — Phase J (FINAL convergence campaign; orchestrator directive of
2026-08-09/10: unlimited compute; end only at high-confidence SOTA on existing
usable data/research — ≤5MB, <50ms, beat resbn192i on ALL spreads + all six
alt-layouts + shippable-Cyrillic bar; test-2400 unsealing only after ALL bars
beaten on val, pre-registered)

Bars to beat (seed-mean, 3 seeds): en val-9918 E1/AOSP 88.30/92.60/93.26/
91.27/86.77 (resbn192i); dvorak-heldout 89.13 / dvorak-app 88.20 / azerty
83.60 / qwertz 82.50 / german 79.64 / spanish 88.28; ru real-val probe
in-dict t1 > 76.21 with NO Yandex training rows. Size ≤5MB (fp16w free;
int8-trunk free at ch256), latency <50ms (trivially met).

- [x] J1 GPU round 1 (DONE 2026-08-10: all four 188k arms finished; sel-beam
      p0.65 plateau-optimal at ch192 AND ch256, CR-CTC a0.2 negative at ch80;
      full battery running — PHASE_J.md §5)
      spec: GPU round 1 — dose×capacity (PHASE_I §9 highest-value follow-ups),
      one variable at a time, seed 1234, 188k, resbn192i recipe otherwise:
      phaseJ-ch256-p65 | phaseJ-ch256-p80 (coarse dose sweep at ch256) |
      phaseJ-ch192-p80 (is 0.65 already optimal at ch192?).
- [x] J2 free lever (DONE: REFUTED, PHASE_J.md §2 — 0 is a sharp optimum)
      spec: free lever (0 GPU): blankOffset {0,±0.5,±1,−2} 6th axis in
      sweep_scoring.py on existing resbn192i s1234 val emissions — informs
      CR-CTC peakiness prior before GPU is spent.
- [x] J3 data (DONE: PHASE_J.md §3 — pools built + verified disjoint)
      spec: data: fetch FUTO swipe-2/3/4 + swipe-5 (~0.6GB, MIT); verify
      session-disjointness + holdout-trace overlap MYSELF (bit-exact);
      build swipe2345q additions (distance-gated, dual_finger=0, swipe-5
      qwerty only — NEVER eval-set layouts); realalt npz (clearflow/kasroz/
      toki_pona, fresh 20% session-disjoint holdout); HWS Y-frame-correction
      arm (swipetest-geometry derivation, DATASET_SCOUT §4.1).
- [x] J4 train.py levers (DONE: committed d29d648, bit-identical regression
      guard; per-source layouts smoke-verified on en+ru 2026-08-10)
      spec: train.py levers (RESEARCH_SCAN specs): CR-CTC (dual views SHARE
      layout draw + slot permutation, independent affine/noise/temporal
      frame-hold masking; α=0.2, stop-grad KL both ways); FUTO-parity aug
      (shear k~U(±0.1), rotation ±8°, time-reversal p0.25 w/ reversed
      targets, frame-hold masking); per-row layout batching for joint
      multi-script (PHASE_I §9 contained change).
- [~] J5 GPU rounds 2-4, all detached and self-queuing (no orchestrator needed):
      r2 phaseJ-sw234 / phaseJ-yfix / phaseJ-realalt / phaseJ-ch256-280k;
      r3 (auto) phaseJ-cr192 / phaseJ-cr256 / phaseJ-futoaug;
      r4 (auto) phaseJ-joint / phaseJ-ru192. Batteries auto-run per arm via
      phaseJ_eval_round2.sh + phaseJ_eval_round34.sh. Round-1 verdicts in
      PHASE_J.md 5.1: p0.65 plateau-optimal (dose axis closed), ch256-p65 wins
      val but loses the euro layouts, CR-CTC is the big transfer lever.
      spec: GPU round 2 — lever probes at ch80 paired vs phaseH-p50 (cheap,
      2-3h): CR-CTC | shear+rot | time-reversal | frame-hold; data arms:
      swipe2345q @ ch192-p65 paired vs phaseI-ch192-p65; HWS-Y @ ch80;
      realalt @ ch80 (+ its own holdout eval).
- [x] J6 checkpoint soup — DONE, POSITIVE: +0.50 selection t1 / +0.38 full-val
      t1 (4 members, BN re-estimated) but -0.12 t3 / -0.15 t5. Candidate for
      the final artifact, must be re-measured on the winner. Stack seeds carry
      --snapshot-every 4.
- [x] J4b CR-CTC — RETRACTED after round 3: the +3.13 dvorak at ch80 becomes
      -1.63 at ch192 and wrecks the euro corpora at ch256. Lever dropped.
- [x] J7a ru-only rung — NEGATIVE: ch192/188k = 73.53 in-dict t1 vs the
      ch80/94k bar's 76.21, while greedy improves +3.11. Synthetic train AND
      synthetic selection => generator overfitting; last.pt refutes the
      selector explanation. 76.21 stands.
- [x] PHASE J CLOSED 2026-08-11 — TERMINAL CONDITION **NOT MET** (10/11 bars
      on the SEED-MEAN footing; 5/11 on the stricter every-seed reading).
      Finalist sw2345 = resbn192i + tier_sw234 + tier_sw5q, 3 seeds:
      val 88.51/92.67/93.37/91.20/87.11 (bars +0.21/+0.07/+0.11/**-0.07**/+0.34)
      and ALL SIX alt-layouts beaten (+0.17..+1.00). The <=3 stratum misses by
      0.07 (two rows of 3,389). Cyrillic bar 76.21 NOT beaten.
      test-2400 NOT unsealed, no pre-registration filed, no seal entry —
      the gate precondition never came true. resbn80g keeps the test tier.
      Ship bytes: sw2345_s1234_fp16w.onnx 3,052,318 B (2.91 MiB), 0.842 ms.
      FREE WIN FOR THE APP: ru decode lambda 1.1 -> 2.0 is worth ~+1.2 in-dict
      t1 on the ALREADY-SHIPPED Cyrillic model (correct number ~77.4, not
      76.21). Model-independent; costs nothing.
      Negatives, all recorded with evidence: CR-CTC (ch80-only, reverses at
      capacity), FUTO-parity augs, HWS y-fix, ch256 280k schedule, ru capacity
      rung, checkpoint soup (does not generalise), layout-alt p0.8, joint en+ru
      (-0.42 en), stratum-aware decode preset (+0.03).
      Remaining routes for the <=3 stone are candidate-generation-side:
      T'=64 (contract-breaking), length-conditioned beam width, <=3-weighted
      training signal. See PHASE_J.md sections 8-9.
- [x] J8a-done 3 seeds of the leading candidate sw234 (s4321/s7777 running with
      snapshots; s1234 already measured) + sw234-p80 dose repair, + futoaug
      and joint en+ru still in flight.
- [ ] J6-old checkpoint soup (beam-t1-selected greedy soup + BN re-estimation
      before export fold) on the best completed run — offline script.
- [ ] J7 multi-script: joint en+ru-synth arm (per-row layout batching);
      eval en full suite + ru real-val (EVAL-ONLY per YANDEX_LICENSE). If
      joint costs en >0.3pt → ship separate ru model, say so. Also ru-only
      upgrade rung: ch192/188k/beam-sel synth-only vs the 76.21 bar.
- [ ] J8 convergence: stack sign-consistent winners → frontier candidate
      (ch256-class, int8-trunk ≤5MB) + efficient candidate (ch192-class,
      fp16w ≤3.1MB); 3 seeds each; full battery (val-9918 + per-source
      halves, six alt-layouts, ru); preset sweeps both footings + app trie;
      export parity; latency.
- [ ] J9 IFF all bars beaten: pre-register (commit BEFORE decode: models,
      seeds, presets, footings, numeric expectations, quote user directive
      2026-08-09/10 authorizing final verification) → decode test-2400 once
      per registered config via --unseal-test → seal ledger append.
- [ ] J10 docs: PHASE_J.md full record; RESULTS.md top section;
      MODEL_COMPARISON.md; APP_INTEGRATION_PLAN.md (new model/preset/
      fixture + RESEARCH_SCAN v1 user-dictionary fix + drop layout-routing
      gate + multi-script notes); regenerate golden fixture at ship preset;
      artifacts + sha256s. Adversarial audit commissioned by orchestrator
      AFTER report — not run here.

2026-08-11 — Phase K (candidate-generation campaign; orchestrator directive of
2026-08-11). Target: the two standing stones — val <=3 (-0.07 vs bar) via
candidate generation, plus best-configuration convergence. test-2400 STAYS
SEALED (ledger stays at 3 entries; any unsealing is the parent orchestrator's
call). Incumbent resbn192i (all-bars-every-seed footing); finalist sw2345
(10/11 seed-mean, <=3 -0.07). Bars = PHASE_J.md §8 footing, report BOTH
seed-mean and every-seed.

- [ ] K1 seed-ensemble emission averaging (eval-only, Tier 1): average
      log-emissions AND probs across sw2345's 3 seeds before the beam; also
      resbn192i's 3 seeds and 2-model mixes. Full battery (val + 6 layouts)
      for the winner. Report N×-encoder latency honestly; check 3×fp16w <=5MB.
      Discuss what "every-seed" means for a deterministic ensemble-of-3.
- [ ] K2 T'=64 contract-v2 retrain (Tier 1): sw2345 recipe (ch192, p0.65,
      sw2345 tier, 188k, no KD, coupled sampler) with --t-out 64. Seed 1234
      first -> full battery incl. per-stratum; if <=3 and/or transfer confirm
      the PHASE_I §6.1 probe, 3 seeds. Export contract-v2 ONNX ([1,64,65])
      + document ALL app-side implications (CtcEmissions slice, beam frames,
      fixture format, ~2x beam latency — measure). Pre-authorized contract
      change per user directive.
- [ ] K3 discriminative candidate rescorer (Tier 1): mine (trace, gold,
      confusables) triples by running the sw2345 beam over TRAIN-set
      emissions (self-mined, license-clean — never FUTO decoder outputs);
      features = beam score components, forced-alignment score, length,
      per-letter emission mass, rank; tiny pairwise ranker (<100k params)
      -> second ONNX; top-k rerank after beam; blend weight swept on
      val[0:half], confirmed on holdout half. Target rank-2 confusions
      (2/3 of errors, worst on <=3). SYMMETRIC: offer to incumbent too.
- [ ] K4 Tier 2 riders: beam width 300 recheck on current models;
      length-conditioned decode knobs (short-trace width / blank handling);
      <=3-weighted CTC loss arm; sw2345 at 280k + soup; lr/wd micro-sweep.
- [ ] K5 convergence: stack survivors (T64 + rescorer + ensemble are
      composable — measure the stack), full battery both footings,
      size/latency, PHASE_K.md + RESULTS.md + MODEL_COMPARISON.md +
      APP_INTEGRATION_PLAN.md, artifacts + sha256s, fixtures regenerated
      (contract-v2 needs its own fixture format — flag it). Final report:
      lever table, best-single-model card, best-configuration card, bars
      table both footings vs incumbent+finalist, honest verdict. Do NOT
      unseal; if bars fall, the unsealing decision goes up.

2026-08-12 — Phase K close-out state (pre-final): K1 DONE (seed-ensembles
refuted; mix2-i8f16 configuration = sw2345_s1234 int8w + resbn192i_s1234
fp16w, prob-averaged, ALL 11 en bars on single-config footing, 4.45MB/1.79ms;
per-frame agreement >=95% is the label-free pair gate, derived post-hoc).
K2 DONE (t64: all 6 layout bars, val t3/<=3 miss + 4+ sign-flip vs probe;
contract-v2 artifact + fixture committed; ~2.1x decode cost; not promoted).
K3 DONE (rescorer: sign-consistent t1/t5/4+ ~ +0.1, NOT <=3; symmetric —
incumbent gains same; flat stacked on ensemble; 21.8KB onnx committed).
K4: 280k+soup wash; slw2 (<=3-weight 2.0) = +0.56 <=3, all 5 val bars on
s1234, azerty/spanish miss; s4321+s7777 PAIRED SEEDS RUNNING (last open
measurement). Box rebooted 4x; all runs recovered from last.pt, --workers 0.
Docs: PHASE_K.md + RESULTS + MODEL_COMPARISON + APP_INTEGRATION_PLAN updated;
artifacts + fixtures staged with sha256s. test-2400 SEALED throughout.

2026-08-12 — PHASE K CLOSED. slw2 3-seed verdict: <=3 bar CLEARED seed-mean
(91.39, +0.12) and EVERY seed — first ever; cost t1 -0.03 / t3 -0.01 /
4+ -0.13 / spanish -0.66 => 7/11; sw2345 stays single-model finalist (10/11).
No single model clears all 11. mix2-i8f16 configuration (sw2345_s1234 int8w +
resbn192i_s1234 fp16w, prob-averaged emissions) clears ALL 11 en bars on the
single-configuration footing, 4.45MB / 1.79ms, disclosures in PHASE_K.md §8.2
(post-hoc gate, seed-mean-vs-deterministic footing, thin t3/<=3 margins).
Cyrillic stone stands. test-2400 SEALED (3 ledger entries, untouched).
Unsealing decision handed to orchestrator/user with §8.2 disclosures.
Registered not run: blind gate confirmation on fresh pairs, W∈(1,2)
interpolation arm, slw2-as-mix-member, Tier-2 decode knobs.

2026-08-12 — BLIND GATE CONFIRMATION (final measurement): pre-registered
protocol executed in order (register 8f0c4fb -> gate 97.0% PASS + prediction
committed 3156080 -> decode). Prediction PASSED prospectively: s5555 mix
88.72 t1 / greedy 68.40 (bands were >=88.30 / >=55). Fresh pair = 10/11
(<=3 91.18, -0.09); all six layout bars clear. Mechanism confirmed; all-11
stays a property of the s1234 mix2-i8f16 configuration. PHASE K FULLY CLOSED.

2026-08-12 — FINAL ARCHITECT REVIEW (in progress): full-campaign retrospective
(PHASE_A..K, RESULTS, MODEL_COMPARISON, FAIR_REMATCH, ALT_LAYOUT_EVAL,
DATASET_SCOUT, RESEARCH_SCAN, PHASE_I_DATA, audits, code) -> extract general
insights; if justified, write PIPELINE_V2_PROPOSAL.md (+ train_v2.py skeleton)
with losses, aug structure, data mix, selection protocol, expected gains,
cost, success bars (headline: beat mix2-i8f16's numbers with a SINGLE model).
Design/proposal only — NO training runs, NO test-2400 contact.
- [x] Part 1 retrospective review (docs + code read — full A..K record,
      audits, scout/scan docs, model/train/layout_aug/sweep/K3 code)
- [x] Part 2 decision: v2 IS proposed — ctc/PIPELINE_V2_PROPOSAL.md +
      ctc/train_v2.py (coupled-pair trainer E1+E3+E6, syntax/import-verified,
      NOT executed). Core elements: alignment-coupled pair training (mutual
      per-frame KL, trained-in >=95% agreement gate), targeted en
      residual-transplant synthesis (short/tail words), slw W=1.5 as member
      asymmetry, explicit 3-way aug mixture, layout-probe selection,
      optional geo-alignment prior, pair->single distillation contingency.
      Key new readings: Phase-G ensemble-teacher KD refutation is confounded
      by K1 alignment incompatibility; T64 4+ flip-flop is single-seed noise
      (transfer signal is the reproducible part); dose law = mixture
      allocation, not a law. Success bars pre-stated (pair >= mix2-i8f16 on
      2/3 seeds; single model 11/11 campaign bars; single-beats-mix2 judged
      unlikely without E7). ~40 GPU-h estimate. test-2400 untouched.
- [x] Commit + push

2026-08-12 — PHASE L OPENED (execute pipeline v2). Plan of record:
ctc/PIPELINE_V2_PROPOSAL.md; log: ctc/PHASE_L.md; workdir ~/ctc-train.
- [x] L0 verify train_v2.py vs proposal + train.py APIs (no code change
      needed; MASK_NEG finite => KL pad-column safe; fork-before-CUDA ok;
      member ckpts export-compatible)
- [x] L0 smoke 800 steps + paired pw0 control: coupling raises per-frame
      agreement +5..+9.5pt (96.5% vs 91.4%), CTC not harmed, gate logic
      correct, resume works, export + eval_beam --ens-avg prob verified.
      PREMISE NOT REFUTED.
- [~] FIVE ARMS IN FLIGHT (all detached, --workers 0, 188k, ch192 resbn,
      slw A/B 1.0/1.5, pw 0.3 ramp 5000+15000, E5 probes on, val-every 4000):
      v2pair-s1234 (L1 reference), v2pair-e2-s1234 (L2 = +synth pools),
      v2pair-pw0-s1234 (L3 = pair-weight 0 attribution control),
      v2pair-e2-s4321, v2pair-e2-s7777 (speculative 3-seed stage on the E2
      recipe). Resume = identical args + --resume ckpt/<run>/last.pt.
      Eval after each: ctc/phaseL_eval.sh <run>  (gate BEFORE decode).
- [x] S0 targeted en synthesis DONE: english_synth.py; synth_en_short.npz
      (150k, len<=4, 7919 uniq) + synth_en_tail.npz (150k, <3 real traces,
      74314 uniq) = 18.9% of the v2 mix; 3 gates PASS (displacement magnitude
      matches real to 0.001; wrong-geometry dvorak control 0.04 vs 0.76;
      hit-gap better than the ru precedent). Original band gate FAILED and was
      revised with the reasoning recorded in PHASE_L.md 4.1 (not silent).
- [ ] L2 pair-level element ablation (E2 synth) per L1 verdict
- [ ] L3 winner pair x 3 seeds, blind gate applied before decode
- [ ] L4 batteries, export/quantize, fixtures, RESULTS/MODEL_COMPARISON
- [ ] E7 distillation contingency only if bars 1-2 fall
test-2400 SEALED throughout (ledger stays at 3).

2026-08-13 — PHASE L CLOSED (pipeline v2 executed). Five 188k arms, all
pre-registered before launch; gates measured and predictions committed before
every decode; test-2400 SEALED (ledger 3).
- [x] L0 verify + smoke (+ paired pw0 control) — premise not refuted
- [x] S0 english_synth.py: synth_en_short/tail, 3 gates pass (one gate revised
      with the reasoning disclosed, PHASE_L 4.1)
- [x] L1 v2pair-s1234 / L2 v2pair-e2-s1234 / L3 v2pair-pw0-s1234 /
      v2pair-e2-s4321 / v2pair-e2-s7777 — all 188k, all decoded full battery
- [x] E1 CONFIRMED: coupled 98.18-98.34% agreement 4/4 over the gate vs the
      pw0 control 92.09% (2/47 evals); control mix greedy 29.10 vs coupled
      72.92. Coupling, not batch sharing, pins the gauge.
- [x] E2 NOT PROMOTED: misses its pre-registered gate by 0.01 (t5 -0.16 vs
      0.15 limit); measured effect = val wash + euro-layout gain (azerty
      +0.86, qwertz +0.93, spanish +0.57) outside the gate's scope.
- [x] BAR 1 (pair >= mix2-i8f16 card on 2/3 seeds): NOT MET (E2 recipe 1/3).
      BAR 2 (single model 11/11): met at ONE seed by two members (campaign
      first), not the seed-mean footing, not promoted. BAR 3 / E7: trigger not
      met, no distillation run.
- [x] Candidate staged: v2pair-s1234 int8w+fp16w 4.39MB, 11/11 campaign bars,
      10/11 vs card (azerty -0.82), val 88.86/92.82/93.59/91.56/87.46;
      artifacts + golden fixture + sha256s in PHASE_L 12.1; RESULTS +
      MODEL_COMPARISON updated.
- [x] PHASE_K 8.5 QUALIFIED (not retracted): first exercise of its broken-band
      arm — greedy prediction exact (29.10<=30), t1 87.64 misses <=87.5; a
      marginal 95.32% pass missed both working-band thresholds.
REGISTERED NOT RUN: 3 seeds of the L1 recipe (the measurement that would
settle bar 1, ~10 GPU-h), E2 at 3 seeds, --pair-weight sweep, E6 geo prior,
E4 w_real sweep.

2026-08-13 — PHASE L SETTLED at three seeds (v2pair-s4321 + v2pair-s7777, L1
recipe verbatim; gate-first blind order as always).
- [x] BAR 2 **MET**: L1 member A clears ALL 11 campaign bars SEED-MEAN
      (88.54/92.60/93.33/91.35/87.08 + 6 layouts) — campaign first; supersedes
      sw2345 as single-model finalist. Ships fp16w 2.91MB. Disclosures: t3 is
      an exact tie (+0.000), qwertz +0.007, every-seed footing 8/11.
- [x] BAR 1 NOT MET: per-seed vs mix2 card 10/11, 8/11, 6/11. BUT all three L1
      pairs clear all 11 CAMPAIGN bars EVERY SEED (nothing in A..K did this;
      Phase J was 5/11 every-seed). Ship standing unchanged per directive;
      supersede evidence handed up.
- [x] Gate now 6/6 pairs (98.05-98.33%); working-band prediction 6/6.
      My pre-stated azerty-failure forecast was WRONG (azerty passed both new
      seeds 85.02/85.45); bar 1 failed on dvorak/dv-app/spanish instead.
- [x] E2 REFUTED at 3 paired seeds: sign-consistent -0.21 t1 / -0.12 t5 /
      -0.22 4+; its s1234 <=3 (+0.06) and euro gains did NOT reproduce.
- [x] int8w NOT free for a single model (<=3 91.24 < 91.27 bar, dvorak -0.78);
      free only for the averaged pair. Corrects a generalization of K 4.6.
- [x] Artifacts + fixtures + sha256s (PHASE_L 16.1); RESULTS, MODEL_COMPARISON,
      APP_INTEGRATION_PLAN 9 updated; pushed.
E7 not triggered (bar 1 stands). test-2400 SEALED (ledger 3, untouched).
REGISTERED NOT RUN: pair-weight sweep {0.1,1.0}, E6 geo prior, E4 w_real,
E7 distillation, a 4th/5th seed to firm the two tie margins.
PHASE L CLOSED. Nothing else launches without new instruction.

2026-08-14 — PHASE M CLOSED (final training phase). 12 arms, all pre-registered.
- [x] NEW SINGLE-MODEL FINALIST: v2kd-fresh-w1 (E7, distilled from the coupled
      pair) — ALL 11 campaign bars on ALL 3 SEEDS + seed-mean (88.750/92.773/
      93.473/91.373/87.387 + 6 layouts, margins +0.10..+2.90). Supersedes
      sw2345. Ships fp16w 2.91MB. Teacher gauge-consistency is what mattered;
      student init did NOT (fresh beat warm-started initA).
- [x] RETRACTED: Phase L's member-A all-eleven claim — 9/11 at five seeds
      (t3 -0.024, qwertz -0.156). Propagated to PHASE_L/RESULTS/
      MODEL_COMPARISON/APP_INTEGRATION.
- [x] PAIR at 5 seeds: 11/11 EVERY SEED (5/5), mean margins +0.12..+2.76.
- [x] Coupling sweep 0/0.1/0.3/1.0: 0.3 interior-optimal; agreement monotone
      92.09->98.58% while transfer collapses at 1.0 (dvorak -1.95).
- [x] E4 DROPPED (dvorak -2.81); E6 DROPPED by kill criterion (4 val bars past
      -0.15); E2 refuted earlier at 3 paired seeds. None retried after failing.
- [x] Crown NOT won (best single beats all 5 card VAL numbers seed-mean, misses
      4 transfer axes). Bar 1 NOT met.
- [x] Gate band predictions 12/12 correct. Artifacts+fixtures+sha256s staged.
LEDGER EMPTY — every registered-not-run item from Phase L was run.
test-2400 SEALED (ledger 3, never opened in L or M). Unsealing + audit are the
orchestrator's acts. NO FURTHER TRAINING.

2026-08-14 — UNSEALING 4 (final): test-2400 opened for the SHIPPED model only.
Authority: user directive of 2026-08-13/14 (one final pre-registered unsealing
+ adversarial audit for whichever model ships) + PHASE_M 11.2 option B.
Subject: v2kd-fresh-w1 (distilled single, fp16w 2.91MB), seeds 1234/4321/7777.
- [x] Ledger verified at exactly 3 entries before starting.
- [x] Config-B footing measured on VAL first (unsealed, disclosed UNSEALING_4
      1.2): app trie + app preset, 89.377/93.680/94.467/92.563/87.727;
      fp16w == fp32 to 0.00 on all five.
- [x] PRE-REGISTERED in ctc/UNSEALING_4.md 1-7 and COMMITTED+PUSHED BEFORE any
      decode: configs, sha256s, beam 100 / top-k 8 / OOV=miss, bands per
      metric, hard cap 6 decodes no retries, tier + equal-footing + McNemar
      rules, fixture chore.
- [x] Decoded 6 (2 configs x 3 seeds), no retry, no crash. Ledger entry 4
      appended (now 4 entries; no fifth is authorised).
- [x] TEST-VALIDATED both footings, every seed. config A (aosp/e1)
      88.931/92.681/93.361/92.597/87.045 vs published bar +4.10/+1.64/+1.28/
      +3.03/+4.64; config B (app trie + app preset) 89.306/93.792/94.500/
      93.701/87.045 vs trie-matched bar +4.39/+2.25/+1.54/+4.13/+4.53,
      worst-seed t5 +1.50.
- [x] QUALIFIED EQUAL-FOOTING WIN (the registered ceiling): all five vs the
      val-tuned FUTO bar every seed, McNemar 3/3 p<0.001 (+45/+46/+39) — 2nd
      in the campaign after ch192, at 2.91MB not 6.14MB. LIMITATIONS: the
      whole lead is the HWS half (FUTO is +0.38 ahead on its OWN half); ch192
      keeps t5 by 0.14.
- [x] Expectations 7/7 verdicts right, band coverage 9/10. Only miss: config A
      <=3, 92.597 vs band top 92.593 (+0.004 = 1/30 of a row) — reported as a
      miss. <=3 val->test shift +1.22/+1.14, largest ever; a <=3 band wants
      +-1.3 not +-0.8.
- [x] Golden fixture regenerated from the fp16w ship artifact AT THE SHIP
      PRESET (was E1): 140,462 B, sha 2a449c4f...; RESULTS/PHASE_M 11.1+12/
      MODEL_COMPARISON updated; pushed.
NOT DECODED and val-only permanently: v2pair-s1234 (option A) and every other
val-only artifact. CAMPAIGN CLOSED.
- [x] AUDIT_FINAL2 (47e5b4c) corrections applied: (1) McNemar p-values fixed to
      the exact two-sided 3.87e-05 / 7.69e-05 / 4.99e-04 with an inline erratum
      (two of the three had been hand-transcribed at precision never computed;
      counts and verdict unchanged); (2) APP_INTEGRATION_PLAN 9.5 "no number is
      test-validated" replaced with the unsealing result; (3) the two stale
      "ch 192 is the only qualified equal-footing win" lines struck and
      superseded (MODEL_COMPARISON 4.3, RESULTS Phase-G); (4) rounding
      reconciled (<=3 lift +1.22/+1.14, hws 83.23, memberA qwertz 82.344 /
      -0.156 propagated); plus the HWS-half limitation added to the
      MODEL_COMPARISON 5 recommendation row.

2026-08-15 — MODELS_TABLE.md (the definitive model registry)

Task: build `ctc/MODELS_TABLE.md`, one row per trained model/configuration for
the whole A→M campaign, superseding the scattered per-phase records. Every
number must trace to a named doc/section; unrecorded values are written
"not recorded" rather than reconstructed.

- [x] extracted every model/arm from PHASE_A..M, RESULTS, MODEL_COMPARISON,
      UNSEALING_4, ALT_LAYOUT_EVAL, FAIR_REMATCH, FUTO_WEIGHTS_VERIFICATION,
      THREEWAY_AUDIT, AUDIT_PHASEJ, AUDIT_FINAL2 + all 52 artifact sha256s
      (recomputed) and the 147 ~/ctc-train/ckpt run dirs (export sizes measured
      with stat where a phase doc prints none; nothing re-run)
- [x] structure: ship menu / test-validated tier (5 models, 4 unsealings) /
      val-only finalists / full ladder by phase / configurations with member
      composition / FUTO opponent rows / footings legend / run+hash appendix
- [x] committed d3f3293
- [x] adversarial number-by-number audit, two independent passes over the whole
      file. part 2 (4.6-5): 7 provenance corrections, zero accuracy errors.
      part 1 (1-4.5, 6, 7): 11 value corrections, 20 attribution corrections,
      11 over-claims narrowed. all 37 derived euro-means recomputed correct;
      all 52 artifact sha256s verified; all 32 tables shape-validated.
      commits d3f3293, bd05a84, 6e76a75, e8f8634, 947a45e, 5bdd82a.
NOTE: phase N (futo-test-49970) is IN FLIGHT in a concurrent session and is
explicitly out of MODELS_TABLE scope; it needs a 4.14 + its own footing entry
when it closes.

2026-08-15 — PHASE N OPENED (beat FUTO on EVERY metric on the FUTO dataset
itself). Plan of record: ctc/PHASE_N.md, committed BEFORE execution.
- [x] Plan: primary benchmark = FUTO official test split (49,970 rows), both
      engines through our harness, symmetric dev-tuned presets (dev 54,269);
      secondary = FUTO half of val-9918. Seal futo-test-49970, hard cap 3
      milestone reads for our models (M0 ship baseline / M1 optional / M2
      final); FUTO engine 2 reads (published anchor + dev-tuned bar).
      Bars B1 (all five metrics, every seed, McNemar) / B2 (val FUTO-half)
      / B3 (11 campaign bars every seed + ship-card noise floor) / B4 (app
      footing). Levers: source-loss-weight, HWS x2 rebalance, domain aug,
      ch256 — screened via G-N2, then pair+KD stack. test-2400 NEVER touched.
- [x] Contamination measured: official dev/test are SESSION-DISJOINT from
      swipe-1 train (0/692, 0/697) => no tier shares a contributor with the
      benchmark. futo_verify ExecuTorch env re-verified running.
- [x] N0 DONE: futo_dev 53,373 / futo_test 49,208 converted (drops accounted),
      futo-test-49970 sealed + guard verified, splits fully contributor-
      disjoint, only leak = 3 cross-contributor 'a'-tap bit-collisions
      (0.006%, symmetric, documented in PHASE_N.md 10.3). dev-8k prefix frozen.
- [x] N1 DONE: symmetric dev-tuned presets adopted interior (ours
      0.725,1.75,0.35,0.05,1.2; futo 0.65,2.2,0.55,0.3734,0.7). ERRATUM
      caught before any test read: sweep target-matching auto-missed the
      14.11% punctuated raw words for both engines; corrected from per-row
      dumps (normalized convention). B1 bar frozen from futo's two spent
      reads: 90.42/94.31/95.01/93.73/88.52. Paper anchor = in-vocab
      convention (92.25 measured vs 92.94). B2 bar 95.65 (ship s1234 95.08,
      gap -0.57). Decomposition: our lead = short/slow (+2.15 <=3, p~1e-49);
      futo a nose ahead mid-length; 4+ = -0.10. PHASE_N.md 11-12.
- [x] M0 DONE (read 1 of 3): ship model 3 seeds vs the bar — t1/t3/t5/<=3
      clear EVERY SEED (t1 +0.87, mcnemar p<=1.7e-18 all seeds); 4+ misses
      by 0.010 seed-mean [miss,pass,miss] — the registered coin-flip fell
      as registered, bands 10/10. B1 NOT met; branch M0-3 binds. PHASE_N 14.
- [x] N2 DONE — both reweighting arms FAIL G-N2 on every prong (dev-8k
      +0.07/-0.41 vs +0.10; hws floor 81.61/81.53 vs 82.05; 9/11+10/11).
      Source reweighting refuted: mixture already optimal (mirrors e4).
      Pair lift on dev-8k 4+ = +0.04 (noise). PHASE_N 14.4.
- [x] N2e DONE — FAIL: minmargin dev preset tops out 4+ 88.63 vs bar 88.65
      (full-dev -0.02, holdout-half -0.22); emissions ceiling, not scoring.
      Root fix landed: sweep_scoring now scores a-z-normalized targets,
      validated digit-exact vs dump recomputation. PHASE_N 15.6.
- [x] N2d DONE — CAPACITY REAL: ch256 pair dev-8k 4+ 89.09 (+0.29,
      in-band), t1 +0.15, val improved (89.02/87.69 4+), hws floor clear
      (82.54). G-N2 spanish prong failed at one seed (-0.17, sd~1.4) —
      deferred by registered amendment to the students' every-seed B3 bar.
      N2e-b FAIL: pair hits the identical 88.63 dev-4+ ceiling (ch192
      emissions family property). PHASE_N 16.1-16.3.
- [x] N3 DONE — both students FAIL the gate (t1 -0.39 / -0.01 vs the
      stronger pair member; rule applied as written). Capacity is NOT
      student-distillable (kept +0.10 / +0.02 of the +0.29 edge).
- [x] N3b DONE — FAIL as registered: ch256 pair is the FIRST config with
      all-five-positive full-dev margins (+0.16 on 4+) but the 4+ margin
      flips sign across dev halves (+0.43/-0.10) => holdout-unconfirmed.
- [x] PHASE N TERMINAL STANDING written (PHASE_N.md 19, committed):
      B1 = 4/5 outright EVERY seed on futo's official test (t1 +0.87,
      mcnemar p<=1.7e-18 all seeds), 4+ statistical tie (-0.010). B2 not
      closed (-0.365 seed-mean, mirrors the original -0.38). B3/B4 floors
      untouched. All levers closed with evidence. M2 decision handed up
      with 4 priced options (recommend: bank the read). 2 of 3 milestone
      reads unspent; test-2400 sealed at ledger 4 throughout.
- [ ] M0 pre-registration + baseline read (ship model, 3 seeds)
- [ ] N2 lever screening -> N3 pair+KD -> M2 final read -> close-out

## 2026-08-18 — ru export + multi-script guide (requested)

- [x] Re-verify APP_INTEGRATION_AUDIT's 23 findings against the app's NEW head
      (`9a6ffdd2`, post neural-engine removal) and write a re-verification
      addendum into `ctc/APP_INTEGRATION_AUDIT.md`.
- [x] Export `phaseIB-ru-synth/best.pt` -> `ctc/artifacts/ru_synth_ch80.onnx`
      (fp32) + `_fp16w.onnx` (ship bytes) at the campaign export conventions
      (sliced-view parity + argmax gate).
- [x] Golden fixture at the app's `tunedRuCkdt` preset
      (1.05/2.0/0.2/0.3734/0.9882) on the CKDT frequency scale + the vendored
      `ru_jcuken_default` geometry -> `ru_synth_ch80_fp16w_golden.json`.
      `make_golden.py` grew a `--vocab` switch; the `en` path is asserted
      byte-identical to the shipped fixture.
- [x] Validate the EXPORTED artifact on the ru real-val probe (Yandex
      eval-only, 9,416 rows): reproduce the 77.92-class confirm-half number
      at lambda = 2.0.
- [x] Write `docs/specs/ctc-architecture-and-multiscript-guide.md` (app repo,
      the one authorised app commit) and mirror it to `ctc/`.

## 2026-08-18 — PHASE O: per-script models for every non-Latin script the app can serve

- [x] O1 INVENTORY committed before any training (`ctc/PHASE_O.md` §1).
      app_layout.py replicates KeyboardGeometry.computeKeyRects + KeyboardData
      parse defaults + buildMappedLayout's letter-box normalization, and
      reproduces en_qwerty.json from the app's own latn_qwerty_us.xml to
      4.7e-4 — app frame == training frame. Free retro-validation: app
      cyrl_jcuken_ru sits 3.4e-3 from the Yandex grid the ru model trained on.
      Verdict: exactly TWO non-Latin scripts have layout AND lexicon in-repo
      (ru done, el new); uk/he/bg/mk are one wordfreq list away on the app's
      own rank formula; sr blocked (wordfreq 'sh' has 0 Cyrillic in 80k);
      hy/ka blocked-on-dictionary; Indic/Hangul structurally blocked (7-20
      centre keys, rest on corner slots a swipe cannot reach);
      Arabic/Persian/Urdu priced not attempted (hamza carriers corner-only).
      Two app defects measured: shipped grek_qwerty.xml says script="latin",
      and langpack-el has NO final sigma (25.7% of the lexicon).
- [x] O2 tooling: script_synth.py (generic residual transplant, 90/10 donor
      split for an honest holdout), eval_script.py (any alphabet, npz or
      jsonl probe, --dump for paired tests, --permute-layout falsification).
      eval_script cross-checked digit-exact against eval_cyrillic on the real
      ru confirm half (77.92/89.50/92.00, allrows 70.18, greedy 37.71).
- [x] CALIBRATION (the phase's central result): the synthesis holdout gets the
      model comparison BACKWARDS. ru-synth vs shipped-EN zero-shot is -2.28 on
      the holdout (p=7.1e-12) and +1.09 on REAL swipes (p=0.0099); vs the
      capacity-matched ch80 EN it is +1.62 real (p=1.4e-4). English capacity
      buys nothing cross-script (+0.53, n.s.). Price list for a new script:
      layout+trie wiring ~76, per-script synthesis ~+1.6, real data ~+13.
      Third probe defect: holdout is 3.3% short words vs real usage's 38.7%.
- [x] Synthesis for ru/el/uk/he/bg/mk (1M train + 5k val + 10k holdout each),
      endpoint stats in the ru reference band, wrong-geo controls collapse.
- [x] Five trainings at the ru-synth recipe VERBATIM (94k steps, resbn ch80,
      greedy selection) — all reached 94,000.
- [x] O2 eval battery DONE. holdout t1 at the adopted preset: el 82.54 /
      uk 79.27 / bg 71.80 / mk 71.69 / he 65.36. four pass the registered >=70
      gate; he fails at lambda 2.0 (70.28 at 1.1) and is exported flagged.
      vs capacity-matched ch80 EN zero-shot: +4.9..+7.3 every script.
      vs the 3x-capacity ship model: -0.6..-3.8 every script (the probe's
      capacity bias, see the calibration).
- [x] LAMBDA SWEEP run as registered and shown INVALID: all five scripts pick
      1.1 monotonically; the ru control has the holdout preferring 1.1 by
      +4.70 while REAL data prefers 2.0 by +1.20. lambda 2.0 adopted on the
      only real evidence; all fixtures frozen there.
- [x] phaseO-ru-initH REFUTED: warm start from the en ch80 is +0.88 on the
      holdout and -0.14 on real (p=0.69). not promoted.
- [x] FALSIFICATION: permuted key centres -> 0.00 t1 / 0.00 greedy on every
      model and every script. geometry entirely load-bearing.
- [x] O2(e) artifacts + goldens committed (5 x fp32 1,142,727 B + fp16w
      589,406 B + golden at 1.05/2.0/0.2/0.3734/0.9882), sha256s in PHASE_O
      2.6. fp32 argmax 100/100 all five; fp16w free at the decode (<=0.02).
- [x] O3 CLOSE: PHASE_O.md complete (inventory, calibration, per-script
      results, controls, export gates, artifact registry, evidence tiers,
      app-integration notes + termux hand-off list, phase p order of work);
      MODELS_TABLE 4.15; RESULTS head entry. pushed.
- [ ] regenerate cache_ru/val.npz (clobbered and disclosed; see f0c7a66).
      NOTE prepare_yandex.py also rewrites the vendored
      layouts/ru_jcuken_default.json — check git status before committing.
- [ ] PHASE P, in order: (1) fix the generator's word draw to corpus token
      frequency (the length mix is 3.3% short vs real 38.7%); (2) re-run the
      ru calibration — if rank-preservation is restored the holdout becomes a
      usable probe, if not, stop reporting holdout numbers entirely;
      (3) cause real non-latin data collection. Do NOT re-run capacity,
      warm-starting or lambda sweeps against a synthesis holdout — all three
      are now measured to be probe artefacts.

## 2026-08-19 — SYNTH V2 DESIGN (foundation before the user's rework requirements arrive)

Concurrent Phase O agent owns its runs/files — do not touch; its v1 results are
baselines. Deliverable: ctc/SYNTH_V2_DESIGN.md + small measurement scripts.
NO training beyond a tiny real-vs-synth classifier (GPU idle, use spare only).

- [x] read: cyrillic_synth.py, script_synth.py, layout_aug.py, PHASE_H.md,
      PHASE_I_DATA.md, PHASE_O (§2.1 calibration, short-word defect, probe
      inversion, zero-shot control), PHASE_J §6.5 (ru192), DATASET_SCOUT §3
- [x] Part 1a distributional gaps (synth_gap_audit.py, 9,416 word-matched
      pairs): top 3 = speed profile (step_cv KS 0.60, step_max 3.2x real —
      mechanism pinned: vertex-count-only donor match leaves 25% of segments
      >2x compressed / 12% >2x stretched and the arc remap scales spacing by
      exactly that ratio, never re-timed), length mix (le3 3.3% vs 35.6%),
      transit jaggedness (sharp turns 3.3x, incl. a 60Hz up/down-sampling
      asymmetry between fast ru traces and slower EN donors)
- [x] Part 1b classifier: word-matched + word-disjoint MLP = 90.4% on the
      speed profile ALONE (coords 77.2, angles 75.9, endpoints 66.3 —
      endpoint stats, v1's only gate, are the LEAST discriminative view);
      unmatched (train-draw) footing 90.0
- [x] Part 1c downstream chaining (13-pt gap, short-word inversion, ru192
      artifact overfit, lambda/probe inversion) + design-assumption audit
      (EN-only bank, i.i.d. residuals lose user coherence, no bigram
      conditioning, no time axis, 255-rank draw, endpoint-only gating)
- [x] Part 2 ranked options A-I with expected gains chained to measurements
      (A corpus-freq draw / B kinematic re-timing / C geometry-matched donors
      / D session coherence / E start-side / F segment bank / G learned
      corrector / H VAE-diffusion / I self-training — I mostly blocked:
      Yandex license + no real data elsewhere + ru192 hazard)
- [x] Part 3 recommended spec: v2 transplant = A+B+C+D, gates G0-G5
      pre-registered (classifier speed view <=0.70 from 0.904; ru real probe
      >=78.9 floor, band +2..+5 over 77.41), cost ~10 GPU-h + 3 days;
      EXPLICITLY awaiting user requirements before any build
- [x] commit + push (ctc/SYNTH_V2_DESIGN.md + ctc/synth_gap_audit.py)
