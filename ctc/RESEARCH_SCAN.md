# Research scan — higher CTC accuracy + on-device personal fine-tuning

**Date:** 2026-08-10 · **Status:** design study only — nothing here was run,
no training arm was launched, no benchmark was decoded. Part 1 scans the
2024–2026 literature (arXiv/ICASSP/Interspeech/ICLR) for levers that could
move the campaign models' beam accuracy, ranks them against the campaign's
own evidence base, and specifies the top candidates as runnable experiments.
Part 2 is a design study for LOCAL-ONLY personalization of the shipped CTC
decoder from an individual user's own swipes on Android (GPL app,
onnxruntime-android). Sources: the phase docs (`PHASE_E/F/G/H/I.md`,
`PHASE_I_DATA.md`, `MODEL_COMPARISON.md`), `model.py`, `train.py`, the app
repo (read-only), and web research cited inline.

## 0. Our setting, and the refuted-lever registry

The model family every recommendation is judged against: `resbn:{80..256}:
1,2,4,8` dilated-conv TCN, embed_hid 96, 0.28–1.5 M params, T′=32 emission
frames over 65 classes (64 key slots + blank), in-graph log-softmax, lexicon-
trie Viterbi beam (width 100) at a val-tuned preset, 1.16 M training rows
(T3 + 3× HWS), 188 k steps AdamW lr 3e-3 cosine, coupled shared affine +
mirroring + slot permutation p 0.5 + path/center noise + layout-resampling
warp p 0.5 (Phase H). Current bests: `resbn80g` test-validated
87.68/92.18/92.82 (footing A), `resbn80h` val 87.66 + dvorak 90.01;
Phase I capacity ladder (ch 128/192/256 with layout aug) in flight.

Two structural facts that shape what can still pay:

* **The beam absorbs emission sloppiness.** Phase H: val greedy fell 7 pt
  (71.4 → 64.4) under layout aug while beam t1 was unchanged; greedy has
  anti-correlated with beam t1 since Phase B (checkpoint selection had to
  move to beam-t1 for exactly this reason). A lever that only sharpens
  emissions where the beam already recovers them buys nothing; the levers
  that matter either (a) fix rows where the right word is *not in the beam's
  top-5* (the t5-capacity story of Phase F §13–14), or (b) re-order the
  top-5 (the t1−t5 gap is ~5.3 pt at ch 80).
* **The single-seed noise floor is ~1 pt t1** (Phase C/D); anything smaller
  needs paired arms + sign-consistency across the five metrics, and adoption
  has required 3-seed confirmation all campaign.

Refuted or exhausted — do not re-propose (measured, with sources):

| lever | verdict | source |
|---|---|---|
| KD from our ch 192 (weight 1.0, T2; also 3-seed ensemble teacher) | **−0.5 t1 at ch 80**; ensemble worse still; ~null at ch 64 | `PHASE_G.md` §3 |
| KD temperature 4 | −0.59 t1, −0.20 t5 | `PHASE_F.md` §13.2 |
| EMA (decay 0.999) | null at one seed, not confirmed at the second | `PHASE_C.md` §4 |
| schedule past 188 k (280 k) | +0.04 t1, +0.02 t5 — exhausted | `PHASE_F.md` §13.1 |
| post-training int8 (static/dynamic, any exclusion set) | loses t5 at every size; MASK_NEG head structurally unquantizable | `PHASE_F.md` §2/§5 |
| depthwise-separable trunk | −0.61 t1 at higher latency | `PHASE_F.md` §3/§4 |
| 5-block depth at these widths | does not pay past 4 blocks | `PHASE_F.md` §14 |
| per-frame refinement head (magic_macaw analogue) | negative twice, incl. on a strong base with its own re-tuned preset | `PHASE_E.md` §2 |
| feature v2, path-only jitter (σ 0.02/0.05) | null / clean negative | `PHASE_B.md`, `PHASE_C.md` §3 |
| data curation (FUTO cascade, T4, englishLevel filtering) | negative four independent times | `PHASE_A/E.md`, `PHASE_I_DATA.md` §3 |

---

## 1. Part 1 — research scan for higher CTC accuracy

### 1.1 Ranked candidate table

Rank 1 = run next round; 5 = do not run. "Gain" is an honest bound on val
t1 for our beam-decoded, lexicon-constrained setting, not the paper's
headline. GPU = per-step cost multiplier.

| # | candidate | evidence at our scale | bounded gain | GPU | cost | rank |
|---|---|---|---|---|---|---|
| 1 | **CR-CTC** (consistency-regularized CTC, ICLR 2025) | extrapolated (smallest tested 22 M, gain largest there, monotone) | +0.2–0.7 t1, **+0.1–0.3 t5** (the bar nothing else moved); could be ~0 under our already-strong aug | ~2× | ~60–100 lines | **1** |
| 2 | **FUTO-parity augmentation**: shear, in-bounds rotation, time-reversal, frame-hold masking | **direct, in-domain** (FUTO paper, same task/size/decode) | +0.0–0.3 aggregate, +0.1–0.4 HWS-half; transfer-side most plausible | 1× | ~50 lines | **1** |
| 3 | **ASAM** (ρ≈0.5) | direct for param class (ResNet-20 0.27 M), none for CTC | +0.0–0.4 t1 (center ~+0.15; literature: gains shrink under strong aug) | ~2× | ~30 lines | 2 |
| 4 | **Greedy within-run checkpoint soup**, beam-t1-selected, + BN-recalibration before export fold | extrapolated (ASR 10⁷–10⁸ params) | +0.0–0.3 t1; cosine-to-zero blunts it (Phase E: 0.02 between late ckpts) | ~1× | offline script | 2 |
| 5 | **Blank-penalty decode axis** (constant added to blank log-emission, 6th sweep axis) | ASR literature says ≈0 (decode-only priors bought nothing) | +0.0–0.15; cheapest possible test of the peakiness→t5 hypothesis | 0 | ~5 lines in `sweep_scoring.py` | 2 (free rider) |
| 6 | Frame-level label priors in training (α·log-prior inside CTC) | closest arch (5 M TDNN) but objective was alignment, not WER | −0.2–+0.3, direction uncertain; mechanism subsumed by CR-CTC | 1× | ~40 lines | 3 |
| 7 | Stochastic depth / DropPath | against it at ch 80 (model under-capacity, train CTC 0.42); standard fix if a Phase-I ch 192/256 rung memorizes (ALT_LAYOUT ch-128 pattern) | 0 at ch 80; +0.0–0.3 *transfer* at ch 256 | 1× | ~15 lines, export-invariant | 3 (pocket) |
| 8 | InterCTC / self-conditioned CTC | none for a 4-block TCN; all gains are 12–18-layer depth phenomena | 0–+0.2, plausibly negative on ≤3 | 1.15× | ~30 lines | 4 |
| 9 | Muon / Schedule-Free AdamW | small-convnet evidence is sample-efficiency (CIFAR speedrun), not converged-budget accuracy | ~0 at a converged 188 k budget | 1× | small | 4 |
| 10 | Mixup / manifold-mixup for CTC | 2024 revisit: gains concentrate in low-resource; we are not | +0.0–0.2, coherence problems with slot permutation + per-sample geometry | 1× | moderate | 4 |
| 11 | Zipformer-style blocks / BiasNorm / SwooshR at tiny scale | none below 1 M; our own ConvNeXt-proxy block lost every measured comparison; T′=32 has no room for U-Net multirate | unknown, likely ~0 | — | high | 5 |
| 12 | **MWER / Bayes-risk / differentiable-beam sequence training** | **direct negative**: 2026 "CTC oracle gap" paper — MWER fine-tune collapsed in all 4 configs (oracle gap ≈0.007 pp → no signal; +3.4 % rel drift in 3 k steps); 11 CTC-internal rescoring strategies null; only an external LM helped | ≤+0.2, real risk of negative; 3–5× fine-tune cost | 3–5× | high | 5 |

Key citations (verified by the scan): CR-CTC [arXiv 2410.05101, ICLR 2025;
k2-fsa/icefall] · FUTO Swipe [arXiv 2606.25247] · CTC oracle gap
[arXiv 2606.23306, 2026] · AWP [arXiv 2307.01715, ICLR 2024] · BRCTC
[arXiv 2210.07499] · MWER-for-CTC/FDT [arXiv 2408.13008] · label-prior CTC
[arXiv 2406.02560, ICASSP 2024] · blank-regularized CTC [Interspeech 2023
yang23l] · InterCTC [arXiv 2102.03216] · self-conditioned [arXiv 2202.08474]
· ASAM [arXiv 2102.11600, incl. ResNet-20 0.27 M] · SAM×augmentation
substitutability [arXiv 2106.01548; DART CVPR 2023 2302.14685] · model soups
[ICML 2022] · weight-averaging survey [arXiv 2502.06761] · SpecAugment
[1904.08779] · OnHWR pen-trajectory CTC augmentation [arXiv 2202.07036] ·
Zipformer [arXiv 2310.11230, ICLR 2024] · Muon [kellerjordan.github.io/muon].

### 1.2 Reads worth keeping (beyond the table)

* **The sequence-training idea is answered, negatively, by the closest
  paper to our question.** The intuition ("train against the BEAM metric —
  our greedy anti-correlates with beam") is tempting, but the 2026
  oracle-gap paper is nearly a direct test and a rout: on strong CTC models
  the N-best training-oracle gap is too small to carry reward signal, and
  fine-tuning against it degrades (sharp-basin drift). Its wider finding —
  *no reweighting of acoustic signal closes the oracle gap; only an external
  LM does* — matches this campaign's own history exactly: the two largest
  levers ever measured here were λ/preset levers (E1 +2.7–4.6 t1, app-trie
  λ 4.0 +1.39), i.e. the linguistic side. Our greedy↔beam anti-correlation
  is a *checkpoint-selection* phenomenon and is already solved by beam-t1
  selection in `train.py`; it is not evidence of exploitable sequence-level
  training signal. Consequence: remaining headroom on the emission side is
  distribution *shape* (CR-CTC's mechanism), and the big remaining headroom
  is context/lexicon (FUTO's unused `hungry_jellyfish` analogue; Part 2's
  personalized lexicon) — not sequence-discriminative training.
* **CR-CTC's decomposed mechanisms map onto our gaps.** Its ablation
  attributes gains to (i) self-distillation between augmented views,
  (ii) masked-region prediction, (iii) suppression of peaky CTC (blank mass
  99.64 % → 94.19 %). (iii) is aimed at exactly what t5 reads — the shape of
  the whole emission distribution, the one axis Phase F proved
  capacity-bound and training-lever-immune. Risks, stated: gains may
  compress under our already-heavy augmentation, and at 279 k params the
  capacity slack CR exploits at 22 M may be absent (mirror of the KD
  capacity-dependence result).
* **FUTO's own augmentation set is the strongest-evidence item in the whole
  scan** (same task, same size class, same decode): we ship affine
  scale/translate/mirror + slot permutation + layout warp, but **no shear,
  no rotation, no time reversal**. Time reversal (reverse frames AND target)
  is genuinely new signal. All three are ×1.0 GPU.
* **Augmentation/sharpness/capacity are partially substitutable
  regularizers** (SAM-vs-aug literature + our own Phase-H dose-response).
  Guidance for the Phase-I ladder: as width rises to ch 192/256, raise
  augmentation diversity first, add DropPath second, and expect ASAM's
  margin to shrink.
* **Checkpoint soup ≠ EMA.** The refuted EMA averaged a blind exponential
  window; a greedy soup selects checkpoints post-hoc on the shipping metric
  (the existing 5,000-row beam-t1 validator), which sidesteps the
  selection-metric divergence. Trap: `resbn` running BN stats must be
  re-estimated after averaging, before the export-time fold. Cross-seed
  soups: refuse (different basins; the app ships one seed anyway).

### 1.3 Recommended experiment specs (next training phase — NOT run here)

**Spec A — `phaseJ-cr80`: CR-CTC on the Phase-H winner recipe. Primary.**

* Base: exact `resbn80h` recipe (`resbn:80:1,2,4,8`, embed_hid 96, T3+3×HWS,
  188 k, batch 256, lr 3e-3, wd 0.01, warmup 1 k, coupled sampler,
  layout-alt p 0.5, no KD, 5 k-row beam-t1 selection, seed 1234).
* Loss: `L = ½(L_CTC(z_a, y) + L_CTC(z_b, y)) + α · L_CR`,
  `L_CR = ½ Σ_t [KL(sg(z_b,t) ‖ z_a,t) + KL(sg(z_a,t) ‖ z_b,t)]`,
  frame-level KL over all 65 columns (pad columns contribute exactly 0, same
  argument as the KD term in `train.py`), stop-grad on the target side,
  **α = 0.2** (paper's optimum; 0.1/0.3 both worse there).
* View construction (the correctness-critical part): per sample, draw the
  layout re-target AND the slot permutation **once, shared by both views**
  (emission columns must mean the same key in both), then draw affine,
  path/center noise, and temporal masking **independently per view**.
  Plumbing note: the KD code path already forwards two models on one
  augmented batch; CR needs the dual — one model on two augmentations —
  so `SwipeDataset.__getitem__` returns both views (shared
  layout/permutation state), and the loop runs two grad-carrying forwards.
* Add temporal masking as the analogue of the paper's 2.5× time-masking
  (their most sensitive knob): 2–3 random spans, total ≤ 25 % of the 64
  input frames, masked by **holding the last unmasked coordinate** (a zero
  is a legal position; a hold reads as a stall). Probe masking budget
  {1×, 2.5×} in two short arms before the full run.
* Cost ~2× GPU/step at 188 k. Gate: paired single seed vs `resbn80h`;
  promote to 3 seeds iff sign-consistent on ≥4/5 metrics; watch t5 and
  dvorak (if CR works, transfer should improve too). Val-9918 + alt-layout
  corpora only; test-2400 untouched.

**Spec B — `phaseJ-aug`: FUTO-parity augmentation set. Co-primary (×1.0 GPU).**

Three paired single-seed arms on the Phase-H recipe, Phase-G-factorial
style, each ~40 lines in `SwipeDataset.__getitem__`, applied to path AND
centers together (shear/rotation are shared-frame transforms, like the
existing affine) with the coupled-sampler containment discipline:

1. **Shear**: `x += k·(y − 0.5)`, `k ~ U(−0.1, 0.1)`, containment-tested
   per draw (reject→resample k, or precompute the feasible k range the way
   `affine_axis_bounds` does for scale).
2. **Rotation**: θ ~ U(−8°, +8°) about (0.5, 0.5), same containment test.
3. **Time reversal**, p = 0.25: `feats = feats[:, ::-1]` + reversed CTC
   target, applied **before** the layout warp (the warp is
   direction-agnostic; the reversed word's polyline is the reversed
   polyline).

Optionally a fourth arm: the frame-hold masking of Spec A alone (isolates
its contribution from CR). Promotion rule: sign-consistency across the five
metrics; expect the gain (if any) on the HWS half and the alt-layout suite
more than on aggregate val — report per-source, per Phase-A discipline.

**Spec C — ASAM ρ = 0.5. Secondary (only with the compute to burn).**

Wrap the optimizer step: ε = ρ · |θ| ⊙ g / ‖|θ| ⊙ g‖ (adaptive/elementwise
scaling), ascend, recompute grad at θ+ε, descend from θ. Grad-clip applies
to the second (descent) gradient only. Everything else the Phase-H recipe;
~2× GPU. Expectation honestly ~+0.15 t1 — this needs the 3-seed protocol to
resolve at all, so run it only as a piggyback when a seed round is
scheduled anyway.

**Free riders (no training):** (1) add `blankOffset ∈ {0, ±0.5, ±1, −2}` as
a sixth axis to the next `sweep_scoring.py` run on existing emissions — if
it moves t5 > +0.1 that raises CR-CTC's prior before any GPU is spent;
(2) greedy checkpoint soup over the retained `--val-every` checkpoints of
any completed run, beam-t1-selected, with BN-stat re-estimation (one forward
pass over ~10 k training rows) before the export fold.

**Explicitly not recommended:** MWER/differentiable-beam fine-tuning (direct
2026 negative, two documented failure mechanisms, 3–5× cost); Zipformer
blocks at this scale (no evidence below 1 M, our closest proxy lost);
InterCTC on a 4-block trunk (depth phenomenon we don't have); the FUTO
emission-count regularizer as-is (it stabilizes their `p(blank)=1−λ_t` gate
factorization — an architecture problem our plain 65-way softmax head does
not have); training-time label priors as a standalone arm (weaker-evidence
subset of CR-CTC's mechanism (iii)).

---

## 2. Part 2 — on-device personal fine-tuning, LOCAL-ONLY (design study)

Goal: learn from the individual user's own swipes on their device; nothing
is pooled or uploaded, ever. Labeled pairs exist by construction: a
committed swipe (raw path + timestamps) + the word the user accepted = one
training row; a suggestion-list pick or an immediate correction is a
higher-value label than a silent accept.

### 2.1 The framework reality check (web-verified 2026-08-10)

* **ORT On-Device Training is deprecated.** ORT v1.20.0 release notes
  (2024-11-01): "All ONNX Runtime Training packages have been deprecated.
  ORT 1.19.2 was the last release for which … onnxruntime-training-android
  (Maven Central) were published." The artifacts pipeline
  (`generate_artifacts` → training/eval/optimizer graphs + checkpoint,
  `OrtTrainingSession` on device) still *functions* at 1.19.2 — the
  gradient registry covers every op in our graph (Conv/Gemm/MatMul/
  LogSoftmax/Where); `requires_grad`/`frozen_params` name lists give
  bias-only or head-only subsets for free; AdamW and SGD exist; CPU EP
  only — but adopting it pins the app's entire runtime to a frozen 2024
  release (the training AAR, 29.8 MB vs 27.8 MB inference, *replaces*
  onnxruntime-android; the two collide), and its built-in losses are
  MSE/CE/BCE/L1 — **no CTC**.
* **ExecuTorch training is experimental** (`extension/training`,
  `_export_forward_backward` traces autograd at export; Kotlin
  `TrainingModule`), SGD-only, and `aten::_ctc_loss` has no portable
  decomposition/kernel — plus it would add a second ML runtime and a
  PyTorch re-export pipeline beside ORT. Not viable for v1/v2.
* **LiteRT (ex-TFLite) signatures** are the only first-party-maintained
  on-device-training path in 2026, but require porting the model to TF, and
  `tf.nn.ctc_loss` is not a TFLite builtin (Flex delegate = tens of MB).
  Ruled out on integration cost.
* **CTC loss is not exportable to ONNX at all** (no ONNX CTC op through
  opset 23; torch.onnx has no `aten::ctc_loss` symbolic — verified
  unchanged 2024–2026). Any on-device gradient path therefore needs one of:
  (a) CTC forward (log-α recursion) hand-authored as ONNX ops over our
  static shapes (T′=32, bounded label length — feasible, all constituent
  ops have ORT gradients); (b) a per-frame CE proxy from a forced alignment
  computed on device (Viterbi over T×(2L+1) is a trivial Kotlin DP); or
  (c) **pure-Kotlin exact gradients for a tiny trainable subset**, with the
  frozen encoder run through the existing inference runtime — the CTC α–β
  DP is ~200 lines and gives exact ∂loss/∂logits, from which any linear
  head/bias/affine gradient is closed-form.

Consequence: **v1/v2 should not depend on a training framework.** The
frozen-ONNX + Kotlin-gradient (or gradient-free) route has zero new
dependencies, no deprecated-runtime pin, and is GPL-clean (everything cited
is MIT/BSD; nothing proprietary).

### 2.2 What the app already has (read-only survey, paths verified)

All paths under `src/main/kotlin/tribixbite/cleverkeys/` in the app repo.

* **Swipe-ML data collection exists**: `MLDataCollector.kt` →
  `ml/SwipeMLData.kt` / `ml/SwipeMLDataStore.kt` (SQLite
  `swipe_ml_data.db`, JSON blob per trace). Per swipe: normalized
  `TracePoint(x, y, tDeltaMs)` list, committed `targetWord`, screen/keyboard
  geometry, source tag. Gated through `PrivacyManager.canCollectSwipeData()`
  → `LearningGate.canCollectSwipeMl` (prefs `on_device_learning_enabled`,
  `privacy_collect_swipe`; UI in `ui/settings/sections/PrivacySection.kt`).
  **Two gaps that matter for personalization:** (1) the candidate slate,
  chosen rank, and accepted-vs-corrected flag are NOT recorded; (2) the two
  call sites are asymmetric — suggestion-picks
  (`SuggestionBridge.onSuggestionSelected`) are collected, but silent
  top-1 accepts (`SuggestionHandler`, ~line 539) only under
  `swipe_debug_detailed_logging`, so normal builds collect corrections
  only. A learner fed from this store as-is would train on negatives.
  Also: **no retention cap or auto-pruning** on the DB
  (`PrivacyManager.getDataRetentionCutoff()` exists but nothing enforces it
  on this table) — a privacy gap to close regardless.
* **Per-user lexicon adaptation ships, but only for the tap path.**
  `personalization/UserVocabulary.kt` (singleton, cap
  `personalization_max_words` default 5,000, prefs-persisted) +
  `UserWordUsage.getPersonalizationBoost()` (0..~4) +
  `PersonalizationEngine.kt`; consumed by `WordPredictor` (via
  `UnifiedScore.combine`) and `NextWordPredictor`. **The swipe engines
  consume none of it** — zero `personaliz*` references in
  `NeuralSwipeTypingEngine`, `PredictionCoordinator`, `SuggestionRanker`,
  or `swipe/`. Today it covers 0 % of swipe decoding.
* **The CTC module is wired for exactly the hooks we need** (currently dead
  code awaiting the model drop — `swipe/ctc/`):
  * per-user **input calibration** slot: `CtcFeaturizer.normalizeRawX/
    normalizeRawY(raw, dim, s, o)` already take scale/offset params
    (identity today); a post-resample warp would go at the end of
    `CtcFeaturizer.featurize`;
  * per-user **emission recalibration** slot: a decorator implementing
    `CtcEmissionModel.emit()` (or inside `CtcEmissions.sliceFromHead`)
    applying per-class bias/temperature to log-emissions before
    `CtcBeamDecoder.decode`;
  * **lexicon**: `CtcLexiconTrie.insert(word, freq)` (AOSP 1..255 scale) —
    note there is **no user-dictionary merge in the CTC trie yet**; the
    model to copy is `swipe/GeometricEngineAdapter.mergeUserWords` (custom
    words at freq 1000, disabled-word removal);
  * **rerank**: the final-score block of `CtcBeamDecoder.decode`
    (`ctc/len^γ + β·len + λ·logFreq`; the unused `alpha` in
    `CtcScoringParams` is reserved for a context/rerank term), or a
    post-decode stage reusing `WordPredictor.getPersonalizationBoostFor`.
* **Learning/privacy infrastructure to reuse**: `LearningGate.kt` (master
  funnel; incognito `IME_FLAG_NO_PERSONALIZED_LEARNING`, password-field
  suppression), `contextaware/` per-user bigram/trigram stores **with
  `recordCommit`/`rollbackCommit`** (autocorrect-undo unlearns — the
  anti-typo pattern to copy), `UserAdaptationManager.kt`/
  `SelectionHistory.kt` (per-word selection counts),
  `persist/DebouncedPersister.kt` + `LearnedDataStorage.kt` (storage
  abstraction), `UserDictionaryObserver.kt` (system + custom dictionary
  change feed).

### 2.3 What our own data says the value is (bounds)

* **A global per-user offset is worth little for the median user.** The
  known ~0.064 systematic y-offset between the HWS and FUTO halves was the
  motivating case, and the campaign tested its augmentation twin: C1
  (path-only offset/scale jitter) moved the HWS half by **+0.06 pt** — the
  20 pt per-source gap is a distribution difference in *how* people swipe,
  not a registration error (`PHASE_C.md` §3). The model absorbed the offset
  on its own.
* **The tolerance envelope is wide but has edges.** The shared-affine probe
  (`ALT_LAYOUT_EVAL.md` §7.2): transforms inside/near the trained envelope
  cost 0.3–1.5 t1; sy=0.5 costs 7 t1; sx=0.7 costs 5.8 t1 (and 51 pt of
  greedy). So per-user *input calibration* has a skewed value profile:
  ~0 for the median user (their offset is inside the envelope the model
  already absorbs), potentially **several points for outlier users** whose
  systematic offset/aspect sits at or past the envelope edge — and it is
  exactly those users the aggregate never shows.
* **The lexicon side is the documented big-lever family.** λ/preset moves
  were the two largest levers ever measured (+2.7–4.6 and +1.39 t1); the
  t1–t5 gap (~5–6 pt) bounds what re-ranking can reclaim; OOV is 2.5–3.6 %
  of holdout rows against static tries and is certainly higher for a real
  user's proper nouns/jargon — every user word added converts a guaranteed
  miss into a candidate. The oracle-gap paper's "only the external LM
  helps" finding points the same way. **Per-user lexicon + rerank is the
  highest expected value per engineering unit**, realistic +0.5–2 t1 for an
  active user.
* **λ 4.0 amplifies user words** (`PHASE_G.md` §6 caveat): the app preset's
  3.6× larger λ multiplies top-of-scale (freq-255-clamped) user entries,
  and no campaign eval includes a user dictionary. Personal-lexicon v1 must
  therefore ship with a validation gate and a boost cap, not just an insert.
* **Compute is a non-issue.** The encoder is 0.215 ms/trace-class on
  laptop, ~7–10 ms end-to-end with the beam on JVM; a full gradient-free
  calibration fit (a few hundred decode evaluations) is seconds of CPU; a
  head-only gradient fine-tune over a few thousand cached traces is minutes.
  Everything schedules under charging+idle (WorkManager) with no felt
  battery cost.

### 2.4 The paths, surveyed

Common privacy properties (all paths): local by construction. Residual leak
vectors to close regardless of path: (1) **Android Auto Backup** — the
personalization stores and `swipe_ml_data.db` must be excluded via
`dataExtractionRules`/`fullBackupContent`, or "local-only" silently becomes
"in the user's Google backup"; (2) crash logs — never log trace contents or
learned words (mirror the clipboard PII gating); (3) the existing export
path writes plaintext JSON to `getExternalFilesDir()` — user-initiated,
but should carry a warning; (4) the missing retention cap on the swipe DB
(§2.2). Incognito/password fields are already handled by `LearningGate`.

Common label-noise control (all paths): the failure mode is the user
reinforcing their own typos (accepting a wrong word makes it a label).
Mitigations, all with existing app precedent: exclude commits that were
rolled back (`contextaware`'s `rollbackCommit` pattern); weight
suggestion-picks above silent accepts; require a word to be in-lexicon or
seen ≥2 times before it becomes a label (UserVocabulary's min-usage rule);
and gate every learned artifact behind a **validation check on the user's
own held-out recent pairs** — decode N recent (trace, word) pairs with and
without the candidate personalization and adopt only on improvement. That
gate is cheap (N × ~10 ms), exact, and doubles as drift/forgetting control.

| path | mechanism | value (bounded by §2.3) | forgetting risk | eng. cost | rank |
|---|---|---|---|---|---|
| **(c1) per-user lexicon + rerank** | merge user dictionary + `UserVocabulary` boosts into the CTC trie/score (freq clamp, boost cap); rerank via the reserved `alpha` slot or post-decode | **+0.5–2 t1** active users; converts OOV misses outright | none (model untouched); typo-reinforcement handled by §controls | S (days) — all hooks exist | **1 (v1)** |
| **(b4) per-user input calibration** | gradient-free fit (Nelder-Mead/CMA over 4–6 params) of `(sx, ox, sy, oy)` [+ optional shear k] in `CtcFeaturizer.normalizeRaw*`, maximizing committed-word beam rank over the frozen ONNX on ~50–200 stored pairs; clamp to the trained envelope (s∈[0.85, 1.11], |o|≤0.05) | ~0 median, **points for envelope-edge users**; C1 says don't expect aggregate movement | none (frozen model); bad fit caught by the validation gate | S (days): fitter + WorkManager job | **2 (v1.5)** |
| (b4′) per-key center offsets | 26×2 offsets on the `layout_keys` graph input (the model conditions on centers — Phase H trained exactly this invariance); closed-form from per-key touch statistics or same gradient-free fit | small-moderate; per-key systematic bias (e.g. always under 'p') is real but partially inside CENTER_NOISE σ0.01 tolerance | none; validation-gated | S–M | 3 |
| (c2) per-class emission recalibration | 27 biases (+ optional temperature) on sliced log-emissions from confusion stats, via a `CtcEmissionModel.emit()` decorator | small: the beam absorbs most per-class miscalibration; overlaps (b4′) | none; validation-gated | S | 3 |
| **(b1–b3) head-only gradient fine-tune** (BitFit/head, the honest "LoRA-sized" option) | export a personalization artifact with the pre-head hidden state `[32, ch]` as an extra output (app-side asset; shipped contract untouched); train `coeff_head`+`blank_head`+`lambda_head` (~5.3 k params at ch 80; bias-only ≈ 150 params as the cautious tier) in Kotlin with exact CTC α–β gradients; L2-SP anchor `μ‖θ−θ₀‖²` + rehearsal | the only path that can learn a user's *motor style* (the 20 pt HWS-type gap is style, not offset — §2.3); plausibly the largest ceiling, unproven | real — controlled by L2-SP + rehearsal buffer of frozen-base emissions on a reservoir of the user's own traces (self-distillation as a stability anchor is a different role from the refuted accuracy-KD, and stays license-clean: our own base) + validation gate + one-tap reset | M (weeks): Kotlin CTC DP + trainer + artifact export | **2 (v2)** |
| (a) full/partial fine-tune via ORT-training 1.19.2 | artifacts pipeline, hand-built CTC-as-ONNX-ops or CE-proxy loss | same ceiling as head-only, marginally higher | same controls needed | L + pins runtime to deprecated 1.19.2 AAR | 4 |
| (a′) ExecuTorch / LiteRT training | §2.1 | — | — | L, second runtime / TF port | 5 |
| (c3) residual-bank user-style synthesis + local head retrain | harvest user residuals vs ideal polylines (the `layout_aug` decomposition, ported), synthesize user-style traces for unseen words, retrain the head on them | augments (b1–b3)'s data; speculative | as (b1–b3) | L | 5 (v3 research) |

### 2.5 Recommendation

**v1 — "personal lexicon + calibration", no gradients, ~days of work,
measurably helps:**

1. **Wire the personal lexicon into the CTC path** (the missing 100 % of
   swipe-side coverage): merge system user dictionary + custom words +
   `UserVocabulary` into `CtcLexiconTrie` (clamped freq; model on
   `GeometricEngineAdapter.mergeUserWords`), and add a capped
   personalization term to the beam's final score (the reserved `alpha`
   slot, reading `WordPredictor.getPersonalizationBoostFor`). Validation-
   gate the boost scale per user (the λ 4.0 amplification caveat).
2. **Fix the label pipeline**: extend `MLDataCollector.collectAndStoreSwipeData`
   with (slate, chosenRank, acceptedTop1 flag), collect accepted top-1s
   (not only corrections), honor `rollbackCommit`, and enforce a retention
   cap + backup exclusion on `swipe_ml_data.db`.
3. **Per-user input calibration, gradient-free**: background WorkManager
   job (charging+idle) fits `(sx, ox, sy, oy)` over the frozen ONNX on the
   stored pairs, clamped to the trained envelope, adopted only if the
   held-out-pair validation gate improves; surfaced as one settings row
   with a reset. Expected: nothing for most users, real recoveries for
   envelope-edge users, zero risk (gate + clamp + frozen model).

**v2 — "personal head", the smallest genuine learner:** the Kotlin-native
head-only fine-tune of the table row (b1–b3): personalization artifact with
the hidden-state output, exact CTC gradients for the three heads (bias-only
first, full head ~5.3 k params second), AdamW-lite in Kotlin, L2-SP anchor
+ frozen-base rehearsal + the same validation gate and reset switch,
trained opportunistically on the (now correctly labeled) local store. This
is the first rung that can address motor *style* rather than offset — the
axis our per-source data says actually separates users — and it needs no
training framework, no runtime pin, and no new dependency.

Deliberately not proposed: ORT-training dependency (deprecated), full-model
on-device fine-tuning (all of the risk, little ceiling over the head at
these sizes), any pooled/federated variant (out of scope by definition).

---

*Prepared as a design study; every "run" in §1.3 and §2.5 is a proposal for
the next phase, pending owner prioritization. test-2400 is untouched and
stays untouched by everything in this document.*
