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

## 5. The ladder (results pending — runs in flight)

## 6. Training-lever probes (results pending)

## 7. Winner at three seeds, preset sweep, artifacts (pending)

## 8. Ship recommendation (pending)
