# Phase B — architecture levers on T2

Base tier fixed at **T2** (385,021 rows, FUTO-only, session-excluded). Recipe frozen
at the Phase-A settings: ch 96 baseline, batch 256, lr 3e-3, wd 0.01, warmup 1,000,
47,000 steps, seed 1234, `--val-every 1500`. One variable per arm.

**Result: no arm is adopted. All three regress, and they regress under per-arm optimal
scoring too, so this is not a calibration artifact.** Detail below, because *how* they
regress is more useful than the fact that they do.

test-2400 was never decoded.

## 1. Arms

| arm | change | params |
|---|---|---|
| `phaseA-T2` (**B0**) | unchanged control | 394,114 |
| `phaseB-B1` | `path_features_v2` + key-proximity channels | 405,716 |
| `phaseB-B2` | ConvNeXt trunk, ch 128, 5 blocks, dil {1,2,3,5,8} | 570,818 |
| `phaseB-B3` | B1 + B2 | 585,940 |

B2/B3 land at 0.57–0.59 M, just under the 0.6–1.0 M the brief projected. The gap is the
GLU reading: `expand=4` widens the pointwise layer to `4*ch` and the GLU halves it back
to `2*ch`. Reading it as "4×ch *after* the GLU" (an `8*ch` projection) would give ~1.07 M.
The narrower reading was taken and is stated here so the number is not mistaken for a
different block.

All four export at opset 17 with **zero** `Einsum` / `BatchNormalization` / `Loop` /
`Scan` nodes and <1e-5 sliced parity at 50/50 argmax agreement, checked before any GPU
time was spent.

## 2. Results — frozen `enc` preset

| arm | wall | greedy | val t1 | t3 | t5 | FUTO t1 | HWS t1 | **T2-clean t1** | Δ clean |
|---|---|---|---|---|---|---|---|---|---|
| **B0** | 6.9 min | 59.62 | **80.86** | 88.52 | 90.48 | **92.59** | 69.21 | **79.99** | — |
| B1 | 6.7 min | 60.00 | 76.28 | 87.61 | 89.59 | 86.34 | 66.28 | 75.39 | **−4.60** |
| B2 | 13.9 min | **64.78** | 79.48 | **88.68** | 90.42 | 89.54 | **69.49** | 78.68 | **−1.31** |
| B3 | 13.2 min | 62.92 | 78.05 | 88.45 | **90.53** | 87.43 | 68.73 | 77.31 | **−2.68** |

Decision rule required a T2-clean t1 gain > 0.6 pt. Every arm is negative, and by margins
far outside the ~0.4 pt noise floor established in Phase A. **Adopt none.**

## 3. Greedy went up and beam went down

This is the result worth keeping.

| arm | greedy Δ vs B0 | beam t1 Δ vs B0 |
|---|---|---|
| B1 | **+0.38** | −4.58 |
| B2 | **+5.16** | −1.38 |
| B3 | **+3.30** | −2.81 |

B2 produces markedly better per-frame emissions — a 5.16 pt greedy gain is the largest
single-lever move in either phase — and still decodes *worse* through the lexicon beam.
Phase A already found greedy to be a loose proxy for beam top-1; Phase B shows the two can
move in **opposite** directions, which retires greedy as a model-selection metric here.
Checkpoints in this project are selected on `best val_greedy`, so this is not a reporting
detail: **B2's selected checkpoint is chosen by a metric that anti-correlates with the
metric that ships.**

## 4. Testing the calibration hypothesis — it is real, and it is not enough

A sharper model has a different emission calibration, and the `enc` scoring preset
(`gamma`, `lambda`, `beta`, prune params) was tuned for the original one. Before calling
Phase B negative, each arm got its own scoring sweep: tuned on val rows 0:4,959, confirmed
on the untouched 4,959:9,918.

| arm | frozen preset | re-tuned | gain from re-tuning | tuned ≤3 | tuned 4+ |
|---|---|---|---|---|---|
| B0 | 80.86 | 80.94 | **+0.08** | 84.74 | 78.97 |
| B1 | 76.28 | 77.30 | **+1.03** | 74.68 | 78.66 |
| B2 | 79.48 | 80.13 | **+0.65** | 81.00 | 79.68 |
| B3 | 78.05 | 79.03 | **+0.98** | 77.87 | 79.63 |

The hypothesis holds directionally: the new arms are genuinely mis-calibrated for the
frozen preset, gaining 0.65–1.03 pt from re-tuning where B0 gains 0.08 pt. Every tuned
preset moved in the same direction — lower `gamma` (0.41 → 0.25–0.31) and lower `beta`
(0.99 → 0.84) — i.e. the new models need *less* length compensation, which is what an
overconfident emission distribution looks like.

**But it does not rescue them.** Fully re-tuned, B2 still sits 0.81 pt below a re-tuned
B0. The gains held on the untouched holdout half (B2 +0.56, B1 +0.97, B3 +0.69), so the
re-tuning is real and not sweep overfit — the arms are simply worse.

### Where B2 loses: short words

Under both presets the loss is concentrated entirely in the short stratum.

| arm (tuned) | ≤3 letters | 4+ letters |
|---|---|---|
| B0 | **84.74** | 78.97 |
| B2 | 81.00 | **79.68** |

B2 is **+0.71 pt on 4+ and −3.74 pt on ≤3**. The larger receptive field (5 blocks,
dilations to 8, ch 128) helps exactly where long-range context exists and hurts where the
decision must be made from a few frames and the lexicon prior should dominate. A sharper,
more confident model over-commits on short words, where there are many lexicon candidates
within a small edit distance and the beam needs a soft distribution to rank them.

That decomposition, not the aggregate, is the actionable finding: **the ConvNeXt trunk is
not a failed idea, it is a mis-targeted one.** Anything that recovers short-word behaviour
(emission temperature, a length-conditioned score, or simply keeping ch 96 with the deeper
dilation ladder) is a live follow-up.

## 5. What was rejected and why

* **B1 (features) is the clear failure** — 4.6 pt down for +0.38 greedy. The added
  channels are dominated by the key-proximity field, which hands the trunk a soft
  argmax over keys at every frame. That is close to the answer the model is supposed to
  compute, and it appears to short-circuit learning the path dynamics that make the
  emission *distribution* informative. The ≤3 stratum collapses to 72.97 (B0: 84.69).
* **B3 confirms the levers do not compose** — it sits between its parents on every
  metric, so B2's trunk cannot repair what B1's features break.

## 6. Recommendation

* Keep the **B0 architecture** (ch 96, residual trunk, v1 features) as the Phase-C base.
* Retire `best val_greedy` as the checkpoint-selection metric; select on beam t1 over a
  val subset, or at minimum report both.
* Re-tune the scoring preset **per model** from here on. B0's 0.08 pt headroom is not a
  general property — a differently-calibrated model left 0.65–1.03 pt on the table.
* Short-word accuracy is the lever B2 exposed; a follow-up that keeps its 4+ gain without
  the ≤3 loss is worth more than either arm as built.

## 7. Reproduction

```bash
python train.py --train-npz train_t2.npz --run-name phaseB-B1 \
                --total-steps 47000 --val-every 1500 --seed 1234 --feat-version 2
python train.py --train-npz train_t2.npz --run-name phaseB-B2 --ch 128 \
                --block convnext --dilations 1,2,3,5,8 --total-steps 47000 \
                --val-every 1500 --seed 1234
python eval_arms.py --arms phaseA-T2,phaseB-B1,phaseB-B2,phaseB-B3 \
                    --own-mask T2 --also-masks T2 --rebuild-cache
python sweep_scoring.py --onnx ckpt/phaseB-B2/ctc_swipe_encoder.onnx \
                        --cache ckpt/phaseB-B2/sweep_emissions.npz \
                        --sweep-rows 0:4959 --holdout-rows 4959:9918
```
