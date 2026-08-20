# Phase Q — SYNTH v3, the learned trace generator, and the sealed upper bound

**Opened:** 2026-08-20. **Workdir** `~/ctc-train`, **GPU** RTX 5080 Laptop (16 GB,
torch 2.8.0+cu128). The app repo `/home/will/git/swype/CleverKeys` is a
**read-only reference**.

**Mandate (user direction, verbatim intent):** *train a synthetic trace
generator using the futo + yandex swipes, generate new synthetic traces, train
on those to get closer to ru-real scores.* Phase P closed with v2 shipping at
**79.73** real in-dict t1 (ru, Yandex valid-10k, eval-only footing) against the
real-trained arm's **89.64** (PHASE_I_DATA §6, λ 1.1 footing — §5.3 below puts
it on the current preset). The question Phase Q exists to answer has two halves,
and they are on **different licence tracks by construction**.

---

## 0. The licence split — non-negotiable, designed around, stated first

`YANDEX_LICENSE_RESEARCH.md` binds this phase. Its recommendation §8.1 draws the
line at the training-pipeline boundary: **nothing derived from the Yandex corpus
enters a shipped artefact — no weights, no distilled teacher, no synth generator
fitted to Yandex residuals.** The ст. 1335.1 «в личных, научных, образовательных
целях» limb covers local research training for measurement; it does not cover
shipping, because every available permission theory is non-commercial and
GPL-3.0 is not.

Therefore two tracks, twins in code and hyperparameters, strangers in lineage:

* **SHIPPING TRACK.** Generator trained **only** on MIT data — the FUTO t3
  donor bank (927,869 traces) + HWS (76,748), the same corpora behind every
  shipped model. Its outputs may feed shipped decoders. Ship gate: the real ru
  probe against v2's 79.73 (§4).
* **RESEARCH TRACK (sealed).** A twin generator trained on the Yandex ru
  training sample (`cache_ru/train_yandex.npz`, 1,000,000 real ЙЦУКЕН rows) —
  legal under the научные carve-out **for measurement only**. Its weights, its
  samples, and any decoder trained on them are **permanently unshippable
  benchmark artifacts**: every file carries the `RESEARCH_ONLY` marker, lives
  under `~/ctc-train/research_only/`, never enters `artifacts/`, the registry,
  `exports/`, or any shipping lineage — the same discipline the FUTO-outputs
  rule established. Purpose: **the upper bound** — how much of the remaining ru
  gap does a *perfect in-domain* learned generator close? That one number
  prices all future shipping-track generator work (§5).

The two tracks never share weights, samples, caches or checkpoints. They share
only code, hyperparameters and the evaluation harness — which is exactly what
makes the comparison an experiment rather than an anecdote.

---

## 1. Design choice: conditional flow matching over the residual field

### 1.1 What the record already establishes

The choice is constrained by measurements this campaign has already paid for:

1. **The transplant mechanism is nearly exact on matched population.** The
   en→en control reads 0.5512 against a measured 0.50 floor (PHASE_P §2.5) —
   84 % gap closure. The ru residual (0.7412 MLP speed / 0.8125 GBM₁₇) is
   dominated by a **donor-population term ≈ 0.19** that no re-timing, matching
   or bandwidth stage can reach, because the donors are English.
2. **Transplant has a fidelity deficit and no diversity deficit** (PRDC recall
   0.916 vs a 0.919 real-vs-real control; WordGesture-GAN's recall is 0.258).
   Any learned replacement must not trade the axis we already win.
3. **Parametric synthesis ranges from useless to harmful; residual-realistic
   synthesis helps** — three independent confirmations (FUTO Appendix A, Apple
   ICASSP 2020: spline −2.7, residual-GAN **+3.6 and +4.6 at 10×**, at 2.2 M
   in-script traces; WordGesture-GAN's failed gates). Apple's is the standing
   evidence that a learned generator *can* beat handcrafted synthesis — given
   in-domain data, which is precisely what the research track has and the
   shipping track does not.
4. **The audit pre-registered the shape**: *"If a learned component ever lands,
   conditional flow matching over the 64×2 residual field is the shape to try —
   not an MDN-over-offsets (Graves) or a full DDPM"*
   (SYNTH_V2_RESEARCH_AUDIT §2.6). Phase Q takes that sentence literally.
5. **What v2 leaves on the table, by instrument**: cornering (`sharp_turns`
   at GBM importance 0.16–0.26 after all fixes), the speed–curvature coupling
   (slope KS 0.317, a defect re-timing *created* and only partly closed), and
   the GBM₂₃ residual 0.8750. On the shipping track these are the reachable
   targets; the population term is not.

### 1.2 The generator

**SYNTH v3 = a conditional rectified-flow (optimal-transport flow-matching)
model over the [2,64] time-uniform trace, expressed as a residual field from
the arc-uniform ideal-polyline reference, conditioned on the target word's
polyline geometry.** One model per track; no donors at generation time.

Representation, per row:

* `seq` = adjacent-collapsed key indices of the word; `V = centers[seq]` the
  ideal polyline on the **target layout** (QWERTY for the shipping-track
  training data; any script's layout at generation time — the conditioning is
  pure geometry, so cross-script transfer is by construction, exactly like the
  warp).
* `R ∈ [64,2]` — the polyline resampled at 64 **arc-uniform** points (S = 1 →
  the key centre repeated).
* `x₁ = (P − R)/σ` — the trace's residual field from that reference, scaled by
  one global scalar σ (the rms residual over the training bank, recorded in the
  checkpoint). Everything the trace *is* beyond its ideal geometry — dwell,
  overshoot, corner-cutting, jitter, tempo shape, the 60 Hz acquisition
  signature — lives in this field. **Timing included**: `P` is time-uniform and
  `R` arc-uniform, so dwells and decelerations appear as systematic
  displacement along the path. No separate re-timing stage, no duration model,
  no bandwidth stage — S4 and S5 exist in v2 because transplant moves *someone
  else's* residuals onto new geometry; a conditional density over residuals
  *given* geometry has nothing to re-time.
* Conditioning channels `c ∈ [9,64]`: `R` (2), unit tangent (2), normalized
  arc position (1), arc distance to the nearest polyline vertex (1), signed
  turn angle at that vertex (1), and broadcast `log1p(L_polyline)` (1) and
  `(S−1)/10` (1).

Model: a 1-D dilated residual conv net over the 64-sample axis (the same
inductive bias as the decoder it feeds): input `2 + 9` channels, hidden 128,
8 residual blocks with dilations 1,2,4,8,1,2,4,8 (kernel 5, GroupNorm, SiLU),
sinusoidal t-embedding injected per block via FiLM, output 2 channels ≈ 1.6 M
parameters. Objective: OT-CFM — `t ~ U(0,1)`, `x_t = (1−t)x₀ + t·x₁`,
`x₀ ~ N(0,I)`, MSE on `v_θ(x_t, t, c) − (x₁ − x₀)`. Sampling: 32 Euler steps
(pre-registered; a 16/64-step sensitivity row is reported once in the battery,
never used to select), then `P̂ = clip(R + σ·D̂, 0, 1)`.

Training (both tracks, identical): batch 512, 120,000 steps (~61 epochs of a
1 M bank), AdamW lr 3e-4 → cosine to 3e-6, weight decay 0.01, warmup 1,000,
EMA 0.999 (EMA weights are the generator), seed 1234, 2 % of rows held out for
CFM-loss monitoring only. Sampling seeds: noise generator 20260820 offset by
split; word draws keep v2's split seeds (1234/999/777) and v2's S0 verbatim.

### 1.3 What is deliberately kept from v2, and what is dropped

Kept: **S0/fix A verbatim** (`script_synth.token_mass`, the wordfreq draw, the
projection-mass fix, the G2 gate — a draw policy is orthogonal to a trace
generator); the npz schema and split discipline (`train`/`val`/`holdout`, seeds
1234/999/777) so `train.py` and every eval driver consume v3 caches unchanged;
the whole Phase-P gate harness (`synth_gap_audit.py` instruments, untouched).

Dropped: donor draw (S1/S2), warp (S3), re-timing (S4), bandwidth (S5) — the
model replaces the mechanism, not the scaffolding around it. Fix D's provenance
idea survives as per-row noise-seed provenance.

### 1.4 Alternatives considered and rejected, with reasons on the record

* **MDN-RNN (Graves lineage)** — rejected by the audit by name; autoregressive
  sampling is slow, MDNs mode-average exactly the jitter correlations that are
  load-bearing, and the one published swipe attempt in this family emits
  non-monotone time.
* **Full DDPM** — rejected by the audit by name; 100–1000 NFE for no measured
  fidelity need at d = 128.
* **Conditional VAE** — Gaussian-decoder blur is the precise failure mode the
  campaign has spent two phases measuring (it would smooth away `ac1`,
  `sharp_turns`, LDLJ); posterior collapse under strong geometry conditioning.
* **GAN (Apple's shape)** — the one family with a positive published result,
  but recall 0.258 (WordGesture-GAN) is the measured mode-collapse precedent,
  and an adversarial objective turns our gate classifier into a training
  signal — the Goodhart the battery exists to prevent. Flow matching gets the
  residual-realism without the discriminator.
* **Learned corrector on transplant output (design option G)** — keeps the
  donor-population term by construction (the base trace is still one English
  donor's); the conditional density is the cleaner test of whether
  geometry-conditional English motor law transfers.

### 1.5 The shipping-track hypothesis, stated honestly before the result

v2's residual on ru is mostly population, and a learned model trained on the
same English corpora does not change the population. What it *can* change:
(i) residuals become **geometry-conditional draws** rather than
nearest-donor-of-the-same-vertex-count transplants — ЙЦУКЕН's segment-geometry
distribution is sampled where transplant extrapolates; (ii) the cornering
signature (`sharp_turns`, the sc-coupling) is learned jointly with speed rather
than inherited from a stretched donor; (iii) no donor-scarcity tail (v2's
`no_donor` dropout and thin vertex-count strata disappear). **Pre-registered
expectation: −0.5 … +1.5 real t1 against 79.73, best estimate +0.5.** A miss is
a plausible and *valid terminal verdict* — v2 remains the generator, and the
audit's own priors (§2.6) point that way. The research twin is what makes even
that outcome decisive rather than ambiguous.

---

## 2. Pre-registered gates — shipping track

All instruments are Phase P's, bit-for-bit (`synth_gap_audit.py`): 5-fold
word-disjoint CV, final-epoch statistics, mandatory real-vs-real floor arm in
[0.48, 0.52], exact within-pair permutation null, the same 9,416 word-matched
Yandex pairs, the same metric batteries (17 + 6). The arm to beat is **v2
C+B′+S5** — v1 is history.

| gate | bar | v2's own reading (the reference) |
|---|---|---|
| **G1** endpoints | start/end-hit within 0.05 of v2's, or closer to real; wrong-geometry control start-hit < 0.05 | 0.7298 / 0.6335; control 0.0200 |
| **G2** length mix | max bucket deviation from wordfreq mass ≤ 0.03 (S0 is v2's code, so this is a regression check) | 0.001 |
| **G3** kinematics | the committed `G3_BARS` (step_cv < 0.15, step_max < 0.12, sharp_turns < 0.32, ac1 < 0.12, sc_slope < 0.35, sc_r2 < 0.40, minima/seg ± 0.10) — same absolute bars | 0.108 / 0.082 / 0.306 / 0.067 / 0.317 / 0.362 / 0.85 vs 0.77 |
| **G4** discriminability | **beat v2's point estimates**: GBM₁₇ mean < 0.8125 AND MLP-speed mean < 0.7412; GBM₂₃ reported never gated (v2: 0.8750); UCL₉₅ ≤ 0.60 stays an open-shortfall record, not a pass condition | 0.8125 / 0.7412 / 0.8750 |
| **GQ-D** diversity guard (new) | PRDC (MIT `prdc`, k = 5, on the matched pairs, real-vs-real control every run): **recall ≥ control − 0.05**. A learned generator must not trade away the axis transplant wins | v2 footing: recall 0.916 vs control 0.919 |
| **GQ-T** trivial-fit diagnostic (new, reported) | decoder greedy on its own v3 holdout, reported next to real-probe greedy; a >90 holdout greedy is FUTO's signature of a trivially fittable generator | v2: 50.98 holdout / 56.12 real |
| **G5-Q** ship gate | §4 — the only gate that decides shipping | 79.73 |

Failure handling, pre-registered: G1–G4 are CPU/minutes and may be re-run after
at most **one** documented hyperparameter repair round (the Phase-P amendment
discipline). The G5-Q retrain runs iff G1 passes and G3 passes on at least the
three speed/temporal bars (step_cv, step_max, ac1) — a generator that cannot
match marginals it was trained on is broken, not unlucky. G4 is reported either
way and does not block G5-Q (Phase P measured instrument disagreement:
the GBM₁₇ closed 22.6 % while the probe gained +2.31).

---

## 3. Pre-registered protocol — what runs, in order

```
Q2  ctc/synth_v3.py            build-cond / train / sample / matched / cache CLI
Q3a train ship generator       FUTO t3 + HWS full pool, 120k steps, detached
Q3b matched arms + battery     same 9,416 words; arms: v3_ship (+ Euler 16/64
                               sensitivity rows, reported only)
Q4a cache_ru_v3/               1M train rows (S0 wordfreq draw, seed 1234),
                               5k val (999), 10k holdout (777)
Q4b decoder retrain            Phase-O/P recipe VERBATIM (resbn:80, dil 1,2,4,8,
                               embed_hid 96, feat_v1, 94k steps, batch 256,
                               lr 3e-3, wd 0.01, warmup 1k, coupled affine,
                               no layout-alt, greedy sel, patience 40, seed 1234,
                               --workers 0), run phaseQ-ru-v3
Q4c real probe                 eval_script --preset ckdt --dump; paired McNemar
                               vs a phaseP-ru-v2full re-decode dump on the same
                               8,471 in-dict rows
Q5  research twin              §5, sealed
Q6  iff G5-Q ships: six scripts on the P6 pattern (full pool, holdout seeds
    unchanged), retrain, export, tiered registry supersede
```

---

## 4. G5-Q — the ship gate, pre-registered

Real Yandex valid-10k, eval-only footing, all 9,416 default-grid rows, 8,471
in-dict, fp32 export, CKDT preset (γ 1.05 / λ 2.0 / β 0.2 / 0.3734 / 0.9882),
`langpack-ru` 50 k trie — the Phase-P footing verbatim, decided on per-row
dumps.

* **SHIP** iff Δt1 ≥ **+1.0** over 79.73 (i.e. ≥ 80.73) **and** exact McNemar
  p < 0.05. +1.0 is the campaign's own single-seed resolution floor
  (PHASE_P §7.5); a smaller significant win is real but not distinguishable
  from seed luck at n = 1 training.
* 0 < Δ < +1.0 with p < 0.05: recorded as *candidate, below resolution* — v3
  does not supersede v2; a multi-seed phase may promote it later.
* Δ ≤ 0 or p ≥ 0.05: **terminal verdict — v2 remains the generator.**
* Corollary, reported not binding (Phase P's status): ≤3 stratum vs v2's 85.77.
* λ is **not** re-tuned, on the standing PHASE_P §4.2 refusal.

At most **one** amendment round before the second (final) G5-Q reading, and it
must change a *footing*, not a tuned parameter — the Phase-P precedent.

---

## 5. The research track — sealed, labelled, and what the number means

### 5.1 Protocol

The twin differs from the ship generator in exactly one thing: the training
corpus is `cache_ru/train_yandex.npz` (1,000,000 real default-grid ЙЦУКЕН rows,
conditioning polylines from `layouts/ru_jcuken_default.json`). Same
architecture, hyperparameters, steps, seeds, σ-recompute, sampler. Then:

1. matched battery on the same 9,416 words (expected near the 0.50 floor —
   reported, nothing gated; this arm calibrates what the battery reads when the
   population term is gone);
2. `cache_ru_v3yx_RESEARCH_ONLY/` — 1 M rows, same S0 draw, same seeds, so the
   **only** difference against `cache_ru_v3` is the generator's weights;
3. decoder retrain, recipe verbatim, run `phaseQ-ru-yxgen_RESEARCH_ONLY`;
4. real-probe decode with dump → **U, the upper bound**.

### 5.2 Labels (the FUTO-outputs rule pattern, applied)

Generator checkpoint, EMA weights, σ, every sampled npz, the decoder
checkpoint, its onnx and dumps: filename suffix `_RESEARCH_ONLY`, all under
`~/ctc-train/research_only/`, all untracked (the corpus rule already keeps
Yandex derivatives out of git), none ever copied into `artifacts/`, `exports/`,
the model registry, an app asset, or a donor bank. `synth_v3.py` enforces the
path prefix mechanically when the training corpus is flagged
`--research-yandex`: it refuses to write outside `research_only/` and stamps
`"license": "RESEARCH_ONLY — Yandex-derived; permanently unshippable"` into
every provenance blob. If any later phase wants a shipping v3, it retrains from
MIT data; there is no laundering path (YANDEX_LICENSE_RESEARCH §7a.5).

### 5.3 What U means, pre-registered before it is measured

Three numbers on one footing (same probe rows, same preset, same recipe;
`phaseIB-ru-real` is re-decoded at the current CKDT preset with a dump so the
ceiling is not quoted across a λ change):

```
79.73                v2 ship arm (transplant, English donors)
U                    decoder trained on the sealed in-domain learned generator
real-trained ceiling phaseIB-ru-real at the current preset (≈89.6 at λ 1.1)
```

* **U − 79.73** = the part of the gap a learned generator could close *given
  target-script motor data* — the value of in-domain data to the generator
  path, and the honest price tag on "collect ru swipes / FUTO ru prompts"
  (YANDEX_LICENSE_RESEARCH §8.7).
* **ceiling − U** = what generation itself costs even in-domain — if this is
  large, generator realism is not the binding constraint and only real
  *decoder* training data closes the rest.
* If **U ≈ 79.73** while the twin's battery sits near the floor: separability
  and probe accuracy have decoupled, and shipping-track generator iteration
  (v4+) is dead on arrival — the residual is data-domain, full stop.

Caveat stated now: U conflates generation fidelity with memorization of its
training corpus (the twin may reproduce near-real traces for words it saw).
That is acceptable *because* U is an upper bound — memorization only pushes it
up, and the bound is what is being bought.

---

## 6. Budget

| item | basis | estimate |
|---|---|---|
| generator training × 2 | 120k steps, batch 512, ~1.6 M params | 2–4 GPU-h each |
| sampling, 1 M rows × 2 | 32 NFE, batch 8192 | ~10–20 min each |
| battery runs | Phase-P measured | ~1 h CPU + minutes GPU each |
| decoder retrains × 2 (+1 re-decode) | Phase-O uk log | ~1–1.2 GPU-h each |
| six-script regen + retrain (iff ship) | P6 measured | ~7 GPU-h |
| disk | caches ~0.5 GB each, 94 GB free | fine |

---

## 7. Results

*(pre-registration ends here; everything below is filled in as it is measured)*
