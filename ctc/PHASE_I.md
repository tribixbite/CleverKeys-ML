# Phase I — capacity under the new budget: accuracy-first, ≤5 MB, layout-general

**Date:** 2026-08-09 · **Authority:** user directive of 2026-08-09. The 2×-speed
target that shaped Phases E–G was measured against the OLD transformer
(~178 ms/trace on-device); users cannot feel a sub-10 ms NN, so the encoder
latency budget is now **~10 ms — capacity is effectively unconstrained by
speed** (the 0.215 ms incumbent uses 2 % of it). Ship criterion: **highest
accuracy + maximum versatility** (cross-layout transfer per Phase H), model
size **≤5 MB, smaller preferred**. Unlimited training compute authorized.
**test-2400 is not read anywhere in this phase** — val-9918 + the six
alt-layout real corpora only; a final unsealing for the ultimate candidate is a
separate, user-approved step.

Phase I-A (this document) owns the training/export code; a concurrent Phase I-B
agent owns the data arms. Coordination is through commits.

## 0. What the record already says, and what it leaves open

* The t5-vs-params curve was monotone across 134 k → 689 k params
  (92.53 → 93.03, Phase E/F) and every capacity increment bought accuracy:
  ch 80 87.72 / ch 128 87.88 / ch 192 **88.06** val t1 seed-means (E1, AOSP).
  The ladder was never run above ch 192, and never *with* the Phase-H layout
  augmentation.
* ch 128's extra capacity previously bought **QWERTY memorization at the cost
  of transfer** (`ALT_LAYOUT_EVAL.md`: the 2.5×-smaller model beat it on every
  alt-layout, margin growing with layout difficulty). Phase H's augmentation
  changed the training distribution; whether capacity now converts to transfer
  instead of memorization is exactly the question the ladder answers per rung.
* `resbn80h` (ch 80 + layout aug) matched `resbn80g` on en_qwerty and beat the
  geometric engine on all six layouts. It is the baseline every rung is read
  against.

## 1. Protocol

Ladder rungs `phaseI-ch128 / ch192 / ch256` — the exact Phase-H winner recipe
at three widths: `resbn:{128,192,256}:1,2,4,8`, embed_hid 96, T3 + 3× HWS,
188,000 steps, batch 256, lr 3e-3, wd 0.01, warmup 1,000, coupled affine
sampler, **layout-alt p 0.5** (synth 2/3, real pool azerty/qwertz/german/
spanish; **dvorak held out**, unchanged), no KD, 5,000-row beam-t1 checkpoint
selection at the published preset, seed 1234 per rung; three seeds for the
final pick. Per rung, through the exported ONNX graph:

* `eval_beam.py` full val-9918, AOSP STRIP 146,964, E1 — the en_qwerty gate
  (bars unchanged: 85.52 / 91.54 / 92.80 / 89.29 / 83.57).
* `eval_altlayout.py` az26 arm, E1, in-dict, all five real alt-layout corpora
  + the dvorak app-98k-trie arm — the transfer axis, vs `resbn80h`'s seed-means
  and the geometric anchors.
* BN-fold drift printed and asserted (< 2e-3) at every export; sliced-view
  parity 100/100 argmax required.

Underfit watch: if a rung's train loss is still descending materially at
188 k, a longer-schedule arm is run for the winner (Phase F measured the
94 k → 188 k doubling at +0.5 for small students and the second doubling at a
quarter of that; the same curve is re-checked at capacity).

## 2. Size levers — measured (on `resbn80h_s1234`, ch 80, 1,142,727 B fp32)

The ≤5 MB budget at fp32 caps the ladder at ch 160 (4.23 MB); the interesting
rungs sit above it (ch 192 = 6.05 MB, ch 256 = 10.67 MB fp32). Two
**storage-only** levers were built into `quantize_onnx.py` (`--mode fp16w`,
`--mode int8w`): large fp32 initializers are stored fp16 (Cast restores fp32)
or per-channel symmetric int8 (DequantizeLinear restores fp32). All compute
stays float32 — deliberately sidestepping the activation-quantization failure
Phase F measured (int8 activations lost t5 at every size; the MASK_NEG −1e4
pad columns and the softplus gate are the worst tensors to scale into 8 bits).
ORT constant-folds the restore ops at session load, so steady-state compute
and memory equal the fp32 graph.

| variant | bytes | % | full val-9918 t1/t3/t5/≤3/4+ (E1, AOSP) | latency |
|---|---|---|---|---|
| fp32 (reference) | 1,142,727 | 100 | 87.66 / 92.24 / 93.05 / 90.88 / 85.99 | 0.215 ms class |
| **fp16w** | **589,406** | **51.6** | **87.68 / 92.24 / 93.06 / 90.88 / 86.02** | unchanged (paired busy-box bench 0.452 vs 0.457) |
| int8w | 317,476 | 27.8 | 87.48 / 92.19 / 93.04 / 90.85 / 85.73 | ~+10 % (DQ fold; re-measured on the winner) |

* **fp16w is free.** Five val deltas of +0.02 / 0.00 / +0.01 / 0.00 / +0.03 —
  weight rounding at rel ~5e-4 is invisible through the beam. Stress parity
  (100 random-layout draws): max sliced |Δ| 2.4e-2, argmax 100/100.
* **int8w costs −0.18 t1 / −0.26 4+** for a further half of the bytes. Its
  stress parity looks catastrophic (argmax 56/100) but that is the
  random-layout stress test exaggerating: on the real layout the beam absorbs
  nearly all of the weight-rounding noise. A trunk-only int8 scope
  (`--int8-scope trunk`, fp16 tail) does not improve stress parity — the
  sensitivity is in the trunk convolutions themselves, not the head.
  **Width-dependence, measured later on the ladder**: the int8 penalty is
  −0.18 t1 at ch 80, −0.20 at ch 192, and **zero at ch 256** (full-val
  88.63/92.61/93.26/91.12/87.33 vs fp32's 88.64/92.56/93.23/91.15/87.33,
  transfer within ±0.45 on all six layouts) — wider layers average the
  rounding noise away. fp16w is measured-free at every width.
* **Full-fp16 *compute* is a dead end on CPU EP, measured:** the converted
  graph (onnxconverter-common, keep_io_types) fails to load outright (MatMul
  type clash at the mask/Where boundary), the CPU EP's fp16 kernel coverage
  would upcast around most ops anyway, and its file (594,123 B) is not even
  smaller than fp16w's. Weights-only fp16 dominates it on every axis.

**Consequence for the ladder: with fp16w free, the budget admits ch 192 at
3.03 MB and ch 256 at 5.34 MB (ch 256 additionally needs int8w — 2.67 MB — or
a ~ch 248 width to sit under 5 MB).**

## 3. Export-code findings

* **The einsum-free head does not dominate at capacity.** ORT op-profile of the
  ch 192 artifact: Conv ≈ 74 % of run time (eight dense 5-tap trunk convs),
  `coeff_head` Gemm 2.6 %, everything else single digits. The fixed ~0.10 ms
  pre-trunk floor that mattered at 0.15 ms is 1 % of a 10 ms budget —
  irrelevant under the new constraint set.
* **ORT offline-optimized serialization is not a size lever**: 6,132,556 vs
  6,144,249 B on ch 192 (−0.2 %). It remains a session-load-time optimization
  for the app side, nothing more.
* **BN-fold at width**: drift asserted < 2e-3 per export; per-rung values
  recorded in §5.
* The `res` (GroupNorm) ch 192 artifact still carries the InstanceNormalization
  decomposition; the Phase-I rungs are `resbn`, which folds to zero
  normalization nodes — the profile above therefore *overstates* the non-Conv
  share a Phase-I ch 192 will have.

## 4. Training-code levers (I-A owns train.py)

* **Multi-layout checkpoint selection** (`--select-layout-probes`,
  `--select-layout-rows`, `--select-layout-weight`): PHASE_H §6.2 recorded that
  QWERTY-only selection does not order checkpoints by transfer (dvorak seed
  order ≠ val seed order; a mid-train checkpoint probed 90.4 dvorak vs the
  selected 88.85). The BeamValidator now optionally warps a val-prefix slice
  onto probe geometries (`synth:<seed>` fixed synthetic lattices and/or
  training-pool real layouts; **dvorak refused** — selecting on the held-out
  probe would void the transfer eval) and selects on
  `(t1_qwerty + w·mean(probe_t1)) / (1+w)`. Probe warping consumes no RNG the
  training loop sees, so a probe arm at the same seed follows the same weight
  trajectory as its QWERTY-selected twin — the comparison isolates selection.
  Measured in §6 (`phaseI-sel80` vs `phaseH-p50`).
* **T′ = 64 emission resolution** (`--t-out 64`, stride-1 stem):
  **contract-breaking** — the Kotlin slice, refinement-head input and export
  shapes all assume [·,32,·], so this is a measurement arm only; if it wins
  materially it is reported as an app decision, not adopted. Rationale: at
  T′ = 32 a long word must emit a letter nearly every frame; doubling frames
  relaxes the CTC alignment for exactly the 4+ stratum that binds. Measured in
  §6 (`phaseI-t64-80` vs `phaseH-p50`).
* **Coupled-sampler × layout-aug interaction — checked, healthy**: across
  2,000 synthetic geometries the per-axis feasible scale ceiling never
  collapses (min s_max_x 1.095, never binds at SCALE_LO 0.85; real layouts
  1.111–1.15), so the exact sampler retains its full range under resampled
  geometries. No pathology to fix.
* **lr/schedule at capacity**: rungs run the unchanged lr 3e-3 cosine for
  attribution; divergence/underfit at ch 256 would trigger a probe (none
  observed at launch — loss trajectories at 3–6 k steps are ordered
  ch 256 < ch 192 < ch 128 < ch 80, as capacity predicts).

## 5. The ladder — capacity converts, but the aug dose must scale with it

Seed 1234 per rung, full val-9918 (E1, AOSP) / alt-layout az26 in-dict (E1).
`ch80h` = `resbn80h` s1234 (the Phase-H baseline). All exports: BN-fold drift
1.5e-5–5.9e-5, sliced parity argmax 100/100 (residues in §7.3).

| rung | params | bytes fp32 | val t1/t3/t5/≤3/4+ | greedy | dvorak | azerty | qwertz | german | spanish | dvorak app-98k |
|---|---|---|---|---|---|---|---|---|---|---|
| ch80h (baseline) | 279,346 | 1,142,727 | 87.66/92.24/93.05/90.88/85.99 | 63.3 | 88.85 | 83.64 | 84.16 | 81.45 | 88.51 | 88.20 |
| `phaseI-ch128` | 685,090 | 2,762,279 | 88.08/92.44/93.20/91.18/86.48 | 70.6 | 89.70 | 82.63 | 81.21 | 80.04 | 86.52 | 88.77 |
| `phaseI-ch192` | 1,512,802 | 6,068,519 | 88.22/92.45/93.21/91.00/86.78 | 71.8 | 85.43 | 83.25 | 81.55 | 79.08 | 88.40 | 84.78 |
| `phaseI-ch256` | 2,668,194 | 10,685,479 | 88.64/92.56/93.23/91.15/87.33 | 75.8 | 87.95 | 81.87 | 79.95 | 78.81 | 88.51 | 87.83 |
| **`phaseI-ch192-p65`** ← the fix | 1,512,802 | 6,068,519 | **88.32/92.70/93.25/91.21/86.83** | 72.8 | **90.60** | **84.59** | **82.73** | **79.76** | **88.85** | **89.17** |

Reads:

* **The QWERTY axis is monotone in capacity, and steeper than the old
  no-aug curve**: 87.66 → 88.64 t1 (ch 80 → ch 256), 4+ 85.99 → 87.33. The
  ch 256 rung's single-seed val t1 88.64 already exceeds the old ch 192
  3-seed mean (88.06) and every test-validated model in `MODEL_COMPARISON.md`.
  Greedy rises 63 → 76 — capacity sharpens the emissions themselves.
* **At p = 0.5 the transfer axis breaks at ch 192**: dvorak 85.43 vs ch 80's
  88.85 — with 5.4× the parameters, the model re-learns QWERTY-specific
  shortcuts faster than 50 % resampled geometry can regularize them away. The
  `ALT_LAYOUT_EVAL.md` memorization story returns at capacity, shifted up two
  rungs by the augmentation.
* **Raising the dose to p = 0.65 fixes it and costs nothing**:
  `phaseI-ch192-p65` beats `phaseI-ch192` (p = 0.5) on **all eleven columns**
  — val +0.10/+0.25/+0.04/+0.21/+0.05 AND dvorak +5.17, azerty +1.34, qwertz
  +1.18, german +0.68, spanish +0.45, dvorak-app +4.39. The Phase-H ch 80
  dose-response ("p 0.5 dominates, val flat-to-up with p") extends: **the
  optimal dose scales with capacity.** ch 256's seed-volatile transfer (§7.1)
  says p > 0.5 is likely needed there too; not run — ch 192-p65 already sits
  on the budget frontier (§8).
* Euro-corpus columns carry the `ALT_LAYOUT_EVAL.md` §3 CKDT-λ confound and
  ~1 pt single-seed noise; dvorak (largest, English, cleanest) is the
  transfer axis that ordered these decisions.
* Underfit check: ch 256's train CTC loss at 188 k (0.545) is still above
  ch 192's own end-of-schedule value trend, and its selection-beam best came
  at epoch 39 of 41 — the biggest rung is not saturated at 188 k. Longer
  schedules at ch 256 remain headroom for a Phase J.

## 6. Training-lever probes (ch 80, seed 1234, same recipe as `phaseH-p50`)

Same-seed *re-runs* (cudnn nondeterminism means trajectories are not
bit-paired; deltas below carry that run-to-run noise, which Phase G measured
as sd 0.15–0.7 on val metrics across seeds).

### 6.1 `phaseI-t64-80` — T′ = 64 emission resolution (CONTRACT-BREAKING probe)

| | val t1/t3/t5/≤3/4+ | dvorak | azerty | qwertz | german | spanish | dvorak app |
|---|---|---|---|---|---|---|---|
| `phaseH-p50` (T′=32) | 87.66/92.24/93.05/90.88/85.99 | 88.85 | 83.64 | 84.16 | 81.45 | 88.51 | 88.20 |
| **`phaseI-t64-80`** | 87.85/92.36/93.22/90.79/**86.32** | **91.62** | **86.12** | **85.00** | 81.26 | 88.51 | **90.68** |
| Δ | +0.19/+0.12/+0.17/−0.09/**+0.33** | **+2.77** | **+2.48** | +0.84 | −0.19 | 0.00 | **+2.48** |

Doubling the emission frames helps exactly where the T′ = 32 alignment
pinches: the **4+ stratum (+0.33)** — a long word at T′ = 32 must emit a
letter nearly every frame — and, unexpectedly large, **cross-layout transfer
(+2.5–2.8 on the strong corpora)**: denser emissions give the lexicon beam
twice the evidence to recover from per-frame confusion on unfamiliar
geometry. Cost: the beam decodes 64 frames instead of 32 (~2× beam time;
measured 20 vs 30 tr/s in this harness — on-device that scales the 1.5–7 ms
beam, still far inside the 10 ms budget).

**Verdict: a real lever, reported as an APP DECISION, not adopted.** It
breaks the frozen I/O contract (`[1,32,·]` outputs, the Kotlin
`CtcEmissions.sliceFromHead` frame loop, the refine-head `[T′,92]` input) —
per the phase constraint this stays a measurement. If Phase J touches the
contract, T′ = 64 + capacity + p-scaled aug is the natural bundle to
re-ladder.

### 6.2 `phaseI-sel80` — multi-layout checkpoint selection

`--select-layout-probes synth:101,synth:202,azerty`, 2,000 warped rows each,
weight 1.0 (score = mean of canonical t1 and mean probe t1).

* **Within-run counterfactual (exact, no run noise)** — both metrics were
  logged at every eval: the QWERTY-only pick would be step 159 k (prefix t1
  85.18, probes 66.2/72.0/81.9); the mix picked step 188 k (85.14,
  66.6/72.8/82.1). **−0.04 canonical for +0.5 mean probe** — every
  selection-eligible late-cosine checkpoint is nearly equivalent; the big
  mid-train transfer peaks (Phase H's dvorak 90.4 @ 45 k) are never
  selection-eligible because canonical t1 keeps rising to the end.
* Corpus-level (vs `phaseH-p50`, run-noise-confounded): val +0.21 t1 /
  −0.06 t5; transfer up on five of six (dvorak +1.71, dvorak-app +2.11,
  azerty +0.76, spanish +0.74, qwertz 0.00, german −1.10).

**Verdict: consistent small positive on transfer at canonical-neutral cost —
kept available in `train.py`, not made default.** The mechanism PHASE_H §6.2
worried about (selection blind to transfer) turns out to matter little at the
end of a cosine schedule; it would matter for any future recipe that selects
mid-schedule or trains with early stopping.

### 6.3 Interactions checked

The coupled affine sampler stays exact under every resampled geometry (2,000
synthetic draws: min feasible s_max_x 1.095, never truncated to SCALE_LO;
reals 1.111–1.15) — no dose interaction to fix. Optimizer/schedule at
capacity: lr 3e-3 was stable to ch 256 (no divergence, no loss spikes); the
only capacity-schedule signal is the ch 256 underfit note in §5.

## 7. The winner — `resbn192i` = `resbn:192:1,2,4,8` + layout-alt p 0.65 — at three seeds

### 7.1 ch 256 at three seeds (the capacity frontier, for the record)

`phaseI-ch256[-s*]`, p = 0.5. Val (E1, AOSP): all five bars clear on every
seed with the campaign's largest margins.

| metric | s1234 | s4321 | s7777 | **seed-mean** | bar | Δ |
|---|---|---|---|---|---|---|
| t1 | 88.64 | 88.56 | 88.75 | **88.65** | 85.52 | **+3.13** |
| t3 | 92.56 | 92.64 | 92.64 | **92.61** | 91.54 | **+1.07** |
| t5 | 93.23 | 93.48 | 93.25 | **93.32** | 92.80 | **+0.52** |
| ≤3 | 91.15 | 91.27 | 91.35 | **91.26** | 89.29 | **+1.97** |
| 4+ | 87.33 | 87.15 | 87.39 | **87.29** | 83.57 | **+3.72** |

But its transfer is **seed-volatile at p = 0.5**: dvorak 87.95 / 88.52 /
**84.29** (seed-mean 86.92; app-trie 86.65) — the §5 dose lesson again, at a
width where p 0.65 was not run. ch 256 remains the QWERTY-accuracy frontier
(+0.33 t1 over the winner) and the natural Phase-J base *if* re-run at a
scaled dose; it is not the ship pick (size §8, transfer volatility here).

### 7.2 `resbn192i` at three seeds

`phaseI-ch192-p65[-s*]`, seeds 1234/4321/7777. Full val-9918, E1, AOSP,
exported ONNX.

| metric | s1234 | s4321 | s7777 | **seed-mean** | bar | Δ | worst seed | `resbn80h` seed-mean | **Δ** |
|---|---|---|---|---|---|---|---|---|---|
| t1 | 88.32 | 88.10 | 88.49 | **88.30** | 85.52 | **+2.78** | 88.10 PASS | 87.69 | **+0.61** |
| t3 | 92.70 | 92.45 | 92.64 | **92.60** | 91.54 | **+1.06** | 92.45 PASS | 92.22 | **+0.38** |
| t5 | 93.25 | 93.14 | 93.38 | **93.26** | 92.80 | **+0.46** | 93.14 PASS | 93.00 | **+0.26** |
| ≤3 | 91.21 | 91.30 | 91.30 | **91.27** | 89.29 | **+1.98** | 91.21 PASS | 90.79 | **+0.48** |
| 4+ | 86.83 | 86.45 | 87.03 | **86.77** | 83.57 | **+3.20** | 86.45 PASS | 86.08 | **+0.69** |

**All five bars clear on the seed mean and on every seed, with the largest
worst-seed t5 margin any model in the campaign has had (+0.34; the t5 knife
edge of Phases E–G is gone by a factor of ten).** Every metric beats
`resbn80h`'s seed-mean by +0.26…+0.69 — capacity converts under the scaled
dose. Against `resbn256i`'s seed-mean it gives up −0.35 t1 / −0.52 4+ and is
level on t3/t5/≤3, at 28 % of the fp32 bytes.

Alt-layouts (az26, in-dict, E1; seed-mean over the same three seeds; anchors
= geometric engine):

| layout | s1234 | s4321 | s7777 | **seed-mean** | geo anchor | Δ | `resbn80h` seed-mean |
|---|---|---|---|---|---|---|---|
| **dvorak** (held out) | 90.60 | 90.92 | 85.88 | **89.13** | 76.8 | **+12.3** | 90.01 |
| dvorak, app 98k trie | 89.17 | 90.31 | 85.10 | **88.20** | 76.8 | **+11.4** | 89.51 |
| azerty | 84.59 | 83.01 | 83.21 | **83.60** | 76.9 | **+6.7** | 84.27 |
| qwertz | 82.73 | 82.81 | 81.97 | **82.50** | 76.2 | **+6.3** | 84.36 |
| german | 79.76 | 79.58 | 79.58 | **79.64** | 71.1 | **+8.5** | 81.13 |
| spanish | 88.85 | 88.34 | 87.66 | **88.28** | 73.9 | **+14.4** | 88.43 |

Transfer sits within ~0.2–1.9 pt of `resbn80h` (the transfer champion at a
fifth of the capacity) and beats the geometric engine on all six layouts by
+6.3…+14.4. The dvorak seed spread (85.88–90.92) shows the transfer axis is
still not what checkpoint selection orders (§6.2) — the s7777 dip mirrors
ch 256's, at smaller amplitude. Trade stated exactly: **`resbn192i` buys
+0.61 QWERTY t1 / +0.69 4+ over `resbn80h` for ~1 pt of alt-layout transfer,
all of it far above the geometric-engine floor.**

### 7.3 Export parity at width, disclosed

fp32 accumulation-order residue on the sliced view grows with width; argmax
agreement is 100/100 on **every** export in this phase. Sliced max |Δ|:
ch 128 3.7e-5 · ch 192 5.3e-5 · ch 192-p65 1.3e-4 · ch 256 8.5e-4 ·
ch 256-s4321 8.4e-4 · ch 256-s7777 **2.3e-3** · t64 2.0e-4 · sel80 4.2e-5.
`export_onnx.py --parity-tol` makes the bound explicit per export (default
1e-4 unchanged); every accuracy number in this document was decoded through
the exported ONNX graph, so the residues are priced into the results.

### 7.4 Preset sweeps (winner emissions, tune val[0:2000], confirm [2000:4000], boundary-checked)

* **AOSP footing**: the E1-region wide sweep on `phaseI-ch256` converges to
  a point statistically identical to E1 (full-val 88.62 vs 88.64 at E1) —
  the fourth model family in a row for which **E1 transfers unchanged**; all
  benchmark numbers here are E1. On the winner itself the same wide sweep
  lands at `0.95/0.9/0.2/0.25/0.9882` = full-val 88.33/92.63/93.36/91.21/86.84
  vs E1's 88.32/92.70/93.25/91.21/86.83 — a tie; **E1 kept**.
* **App footing (`en_enhanced` 98,081 STRIP)**: the λ-scale finding
  reproduces at capacity — the compressed app-trie `log_freq` wants λ ≈ 3–4
  (`PHASE_F.md` §15.4 arithmetic). ch 256 interior winner
  `0.95 / 4.0 / 0.175 / 0.3734 / 0.9882` (holdout +2.45, full-val
  89.26 / 93.46 / 94.24 / 92.53 / 87.56). **Winner (`resbn192i`) interior
  app preset: `0.975 / 3.0 / 0.35 / 0.25 / 0.9882`** — holdout-confirmed
  (+2.55 on the untouched half), **full-val app
  89.23 / 93.54 / 94.30 / 92.53 / 87.52** (published-preset baseline 86.59).
  The λ 4.0 caveat travels: no eval includes a user dictionary, and large λ
  amplifies top-of-scale injected competitors (`PHASE_G.md` §6).

## 8. Ship recommendation

Idle-box latency (the `PHASE_F.md` §0 protocol, re-anchored: `resbn80h`
re-measures 0.213 ms on this instrument, matching its 0.215 ms class):

| model | params | fp32 bytes | **fp16w bytes** | idle mean / p90 | % of 10 ms budget |
|---|---|---|---|---|---|
| `resbn80h` | 279,346 | 1,142,727 | 589,406 | 0.213 / 0.223 ms | 2 % |
| `phaseI-ch128` | 685,090 | 2,762,279 | 1,399,197 | 0.423 / 0.439 ms | 4 % |
| **`resbn192i`** | 1,512,802 | 6,068,519 | **3,052,318** | **0.831 / 0.849 ms (fp16w identical: 0.831)** | 8 % |
| `resbn256i` | 2,668,194 | 10,685,479 | 5,360,800 (int8trunk 2,737,114) | 1.372 / 1.389 ms (int8trunk 1.540, +12 %) | 14–15 % |

**Ship pick: `resbn192i_s1234` stored as `resbn192i_s1234_fp16w.onnx` —
3,052,318 bytes (2.9 MiB), 0.83 ms encoder.**

* **Accuracy**: val seed-mean 88.30/92.60/93.26/91.27/86.77 — the highest
  3-seed val figures of the campaign at ≤5 MB, +0.61 t1 over the Phase-H
  incumbent, every bar on every seed, worst-seed t5 margin +0.34.
* **Versatility**: beats the geometric engine on all six measured layouts
  (+6.3…+14.4); dvorak held-out seed-mean 89.13. Routing stays "CTC
  everywhere a layout provides a-z key centers" (`PHASE_H.md` §7).
* **Size**: fp16w is measured-free on every axis (val Δ ≤ 0.02, transfer
  identical, latency identical, argmax 100/100) — 2.9 MiB is 28 % of the
  old transformer's encoder file alone and 61 % of `ch128`-fp32.
* **Latency**: 0.83 ms — 8 % of the felt-latency budget; the beam, not the
  NN, remains the end-to-end cost (`MODEL_COMPARISON.md` §3.2).
* **Presets that travel with it**: benchmark E1 unchanged; **app preset
  `gamma 0.975, lambda 3.0, beta 0.35, gammaPrune 0.25, betaPrune 0.9882`**
  (holdout-confirmed, full-val app 89.23/93.54/94.30/92.53/87.52). On
  adoption the golden fixture must be regenerated from `resbn192i_s1234` at
  exactly that preset (`MODEL_COMPARISON.md` §5.1 rule); not done here —
  `resbn192i` is **val + alt-layout validated only** and promotion over the
  test-validated `resbn80g` needs the owner's call on an unsealing.
* Middle grounds, priced: `resbn80h`-fp16w (0.58 MiB) gives back 0.61 t1 for
  best transfer; `resbn256i`-int8trunk (2.6 MiB) adds +0.35 t1 / +0.52 4+
  but its p=0.5 transfer is seed-volatile (dvorak seed-mean 86.92) and its
  int8 latency is +12 % — it is the Phase-J base, not the ship.

### Artifacts

| file | arm | sha256 |
|---|---|---|
| `artifacts/resbn192i_s1234.onnx` ← reference fp32 | `phaseI-ch192-p65` | `7436fdd2e1e29a930b02a93c09f993d75c4aa20087fbf5abe55e09b6594f7358` |
| `artifacts/resbn192i_s4321.onnx` | `phaseI-ch192-p65-s4321` | `cfeebdaac76df3a3c02a34a91f8dca5ca5b37a19792e6a965769d88a743c1df7` |
| `artifacts/resbn192i_s7777.onnx` | `phaseI-ch192-p65-s7777` | `adbab6c4dcb3544011cc11b217b05837085ef488370970bddd7acea89b8dc42b` |
| `artifacts/resbn192i_s1234_fp16w.onnx` ← **ship bytes** | fp16w of s1234 (`quantize_onnx.py --mode fp16w`) | `d55624cc5b53edce8fd8b24750c6f09d5c116edd8de911eef9f232cd16a84613` |
| `artifacts/resbn256i_s1234.onnx` (capacity frontier, fp32) | `phaseI-ch256` | `db5dfc771f00a90e4bda70730bf217514c168af519d41b176fbcaec95a0f7cd9` |

`resbn256i` s4321/s7777 stay in the workdir (10.7 MB each), shas recorded:
`910ad2f138e1911f56c6965bce06338ef160b0bb4ca9977eade4dc8208eb40ec` /
`5550d61c205bd2e75b0625a9f56397fdcc6463cbb1385204d9eb94c411dac06a`.

All committed artifacts: opset 17, static shapes, zero normalization nodes,
sliced-view argmax parity 100/100 (fp32 residues per §7.3; the fp16w file's
in-graph Casts constant-fold at session load — steady-state compute is fp32).

## 9. What Phase I did NOT do, and the Phase-J handoff

* **No test-2400 read** — nothing here is test-validated; `resbn80g` keeps
  that tier. The ultimate-candidate unsealing is a separate, user-approved
  step (`resbn192i` is the registered nominee).
* ch 256 at p 0.65 not run (the §5 dose lesson predicts it recovers transfer
  there too); ch 256 longer-than-188 k not run (underfit signal §5). These
  are the highest-value capacity follow-ups.
* T′ = 64 measured as an app decision (§6.1: +0.33 4+, +2.5–2.8 transfer,
  2× beam cost, contract-breaking) — the natural bundle with any Phase-J
  contract change.
* Multi-layout selection kept opt-in (§6.2); it matters more for any
  mid-schedule selection regime.
* Phase I-B handoff (their close): T3+3×HWS stays unfiltered
  (level-filtering negative); ru arms exist but **joint multi-script
  training needs per-row layout batching in `train.py`** — today one
  canonical geometry is a dataset-level constant. That is a contained change
  (per-sample `centers` already flow through `__getitem__`; the blocker is
  only the single `--layout` CLI and the per-dataset cache pairing) — flagged
  as the Phase-J train.py work item, not done here.
* Phase-J levers from the research scan (CR-CTC consistency regularization,
  FUTO-style augmentation extensions, blank-penalty shaping) and the scout's
  data arms (swipe2345q / realalt / HWS-frame) all stack on the `resbn192i`
  recipe: capacity + p-scaled layout aug is the base they should be measured
  against.

