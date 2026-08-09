# Phase H — layout-resampling augmentation: closing the dvorak gap

**Date:** 2026-08-09 · **Objective:** build the geometry-sampling augmentation
the training recipe named and skipped (`docs/guides/train-ctc-swipe-model.md`
§6, augmentation item 3), which `ALT_LAYOUT_EVAL.md` identified as the missing
stage behind the one decisive cross-layout loss: **dvorak t1 63.04 vs the
geometric engine's 76.8** (in-dict, real corpus). Slot permutation is slot
invariance, not layout invariance; the shared affine spans only axis-aligned
scale/translate/mirror; **key re-arrangement was never trained**. Phase H
trains it.

Recipe otherwise Phase G (`PHASE_G.md` §4): `resbn:80:1,2,4,8`, embed_hid 96,
T3 + 3× HWS, 188,000 steps, batch 256, lr 3e-3, wd 0.01, warmup 1,000, coupled
affine sampler, **no KD**, 5,000-row beam-t1 checkpoint selection (en_qwerty
val prefix, published preset), seeds 1234/4321/7777 for the winner.
**test-2400 is not read anywhere in this phase** — alt-layout corpora and
val-9918 only.

## 0. Verdict

**The dvorak gap is closed, with a wide margin to spare, at zero en_qwerty and
zero latency cost.** `resbn80h` = the Phase-G ship recipe + layout-resampling
augmentation at p 0.5 (dvorak held out of training):

* **dvorak (held-out transfer probe): t1 90.01 / t3 96.38 / t5 97.46**
  (3-seed mean, in-dict, AOSP trie) — up from 67.28 (`fast_resbn80`) and 63.04
  (ch 128). On the app-98k-trie footing the geo anchor was measured on:
  **89.51 vs 76.8 — +12.7**, where the previous best CTC read lost by 9.5–15.9.
* **Every alt-layout beats the geometric engine**: +7.4 azerty, +8.2 qwertz,
  +10.0 german, +12.7 dvorak, +14.5 spanish (t1 seed-means vs the current-basis
  anchors, still at the en_qwerty-fitted E1 preset — floors, per the §9 tuning
  asymmetry of `ALT_LAYOUT_EVAL.md`).
* **en_qwerty val seed-mean 87.69 / 92.22 / 93.00 / 90.79 / 86.08** — within
  0.06 of `resbn80g` on every metric (gate was ±0.3), all five val bars clear
  on every individual seed.
* **Latency unchanged**: the exported graph is node-for-node identical to
  `resbn80g` (231 nodes, weights differ); idle bench 0.216 ms vs 0.212 ms —
  measurement noise on the same 0.215 ms class. 279,346 params.
* The mean-key-displacement routing gate (`ALT_LAYOUT_EVAL.md` §8) is **no
  longer needed** for these weights: the displacement-0.4313 layout now
  decodes *better* than azerty/qwertz/german (§7 explains why).

The ship-candidate criterion pre-stated in §4 — dvorak ≥ 76.8 with en_qwerty
within 0.3 — is met on every seed individually, with ~13 points of headroom.
Evidence tier: **val + alt-layout corpora only.** test-2400 was not read, so
`resbn80h` is NOT test-validated; a promotion over `resbn80g` as the ship
candidate needs the owner's call on a fourth unsealing (not requested here,
and not pre-authorized).

## 1. The augmentation — what a sample becomes

With probability `p` (`--layout-alt-p`), a training sample is **re-targeted**:
an alternative 26-key geometry is drawn, the cached `[2,64]` QWERTY path is
warped onto it (§2), and the key centers shown to the model are swapped to the
drawn geometry. The word/CTC target is unchanged — the sample means "this word,
swiped on that keyboard". The re-targeting runs *before* the existing
augmentation stack, so the shared affine (with per-geometry feasible bounds —
`affine_axis_bounds`), path/center noise, mirroring and the p=0.5 slot
permutation apply identically to canonical and resampled geometries. With
probability `1-p` the sample stays canonical QWERTY — the fraction that guards
the en_qwerty bar.

Geometry source, given a re-target (`--layout-synth-frac`, default 2/3):

* **synthetic** (2/3): `synth_geometry()` — a plausible 3- or 4-row lattice
  (row-count patterns from the observed family: 10/9/7 … 8/7/6/5; x pitch
  0.085–0.10; per-row stagger ±0.04; per-key jitter σ 0.006; rows at
  `(2r+1)/2R`, matching both qwerty and the vendored dvorak) with the 26
  letters assigned by a **uniform random permutation**. An infinite family of
  re-arrangements — the strongest generalization signal available.
* **real** (1/3): uniformly from the vendored `futo_{azerty,qwertz,german,
  spanish}.json` az26 geometries — realistic structure, moderate displacement
  (0.058–0.107).

**dvorak is held out of the training pool** (and qwerty is the canonical
geometry, not an "alternative"). The dvorak real-corpus eval therefore remains
a **true transfer test**: the model has never seen dvorak's letter arrangement,
nor any real 4-row layout — only synthetic lattices whose *family* includes
4-row shapes. A variant that trains on dvorak itself would convert the eval
into a fitting exercise and say nothing about Colemak/Neo2/arbitrary user XML,
which is the property actually being bought.

## 2. The warp — residual re-anchoring on the word's ideal polyline

### 2.1 Design

The principled statement of "the same word swiped on a different keyboard": the
ideal QWERTY path for word *w* visits *w*'s key centers; the corresponding
alt-geometry path visits the same word's key centers **in the new geometry**;
the human deviation rides along. `layout_aug.warp_path` implements exactly
that decomposition:

1. **Correspondence** — every path point is assigned a (segment `j`, fraction
   `t`) on the source ideal polyline by a **monotone DP** (point-to-segment
   distance, segment index non-decreasing along the path) with the endpoints
   **pinned** to the first/last segment.
2. **Residual** — the point minus its polyline anchor, expressed in the
   segment's local tangent/normal frame.
3. **Re-anchoring** — the polyline is rebuilt through the same letters' centers
   on the target geometry; the anchor is re-placed by an arc-length remap that
   is **absolute (slope 1) within a 0.05 dwell radius of each vertex** and
   proportional across the segment middle; the residual is re-applied in the
   target segment's frame, in absolute units.

Three design points were each forced by a measured failure, not taste:

* **Monotone DP, not nearest-point projection.** A word that revisits a letter
  ("there") makes nearest-point assignment ambiguous; monotonicity resolves it
  in the order the finger actually travels.
* **Endpoint pins.** A segment that passes *over* a later key of the word
  breaks plain monotone assignment: in "has", the h→a segment runs straight
  through s, so the tail points stick to h→a and the re-anchored endpoint
  landed up to 0.83 from the target key (7–8 % of val words hit this). A swipe
  starts on its first key and ends on its last *by construction of the task*,
  so `j[0]=0, j[-1]=S-1` are boundary conditions, not heuristics. With the pin,
  endpoint residual distances transfer **exactly** (§2.3).
* **Vertex-absolute arc remap.** Projection absorbs an endpoint's tangential
  dwell into `t`; a purely proportional remap would scale that dwell by the
  target segment length and drag endpoints off their keys. Within a key
  half-width (every vendored layout shares rx 0.05), geometry is
  layout-invariant, so the remap has slope 1 there and stretches only the
  inter-key transit.

Residuals are carried in **absolute units** (motor noise scales with finger and
key size — near constant across layouts — not with inter-key distance) and in
the **movement frame** (rotated into the target segment): undershoot/overshoot
happens along the direction of travel *on the keyboard being typed on*. A
world-frame variant was measured (+0.05/+0.03 endpoint hit on dvorak) and
rejected as behaviorally wrong — it would give dvorak swipes overshoot along
QWERTY segment directions.

Alternative considered: the inverse-distance / thin-plate displacement field
over the 26 key-center correspondences that `ALT_LAYOUT_EVAL.md` §"What it
would take" sketched. It needs no target word, but it guarantees nothing about
the warped path visiting the word's keys in order — between anchors the field
is dominated by whatever unrelated keys sit near the transit, precisely where
QWERTY→dvorak scrambles neighborhoods most. Residual re-anchoring is the same
code size, exact where it matters (§2.2), and costs O(64·S) per sample —
**0.21 ms/item measured at p=0.5**, invisible next to the dataloader.

### 2.2 Exactness invariants (`layout_aug.py --selftest`)

* **Identity**: src = dst ⇒ warp is the identity. Measured max |Δ| = **0.0**.
* **Ideal → ideal**: a path lying on the source polyline lands on the target
  polyline. Measured max distance **3.9e-08**.
* 200 synthetic geometries: all 26 keys distinct, contained in [0,1]².

### 2.3 Endpoint-proximity validation (`layout_aug.py --validate`)

The frame metric of `ALT_LAYOUT_EVAL.md` §2 (nearest-key hit rate and distance
at the trace endpoints), on 2,000 warped val paths vs the real corpora:

| paths | geometry | start-hit | end-hit | start-d | end-d |
|---|---|---|---|---|---|
| source val (reference) | qwerty | 0.895 | 0.769 | 0.0686 | 0.0774 |
| **warped** | qwerty (identity) | 0.895 | 0.769 | 0.0686 | 0.0774 |
| **warped** | german | 0.871 | 0.732 | 0.0686 | 0.0774 |
| **warped** | azerty | 0.857 | 0.755 | 0.0686 | 0.0774 |
| **warped** | dvorak | 0.700 | 0.647 | 0.0680 | 0.0783 |
| **warped** | synthetic (seed 1234) | 0.691 | 0.641 | 0.0679 | 0.0762 |
| *real corpus* (ALT_LAYOUT §2) | german | 0.855 | 0.727 | 0.0561 | 0.0731 |
| *real corpus* | azerty | 0.870 | 0.788 | 0.0514 | 0.0699 |
| *real corpus* | dvorak | 0.793 | 0.973 | 0.1419 | 0.0341 |

**Distances transfer essentially unchanged on every target** (0.068/0.077 →
within ±0.001) — the warp adds no positional error. Hit rates on
german/azerty sit in the real-corpus band. On dvorak/synthetic the warped hit
rate (0.65–0.70) sits **below** the real band: the same absolute residual
meets a denser 4-row board (row pitch 0.248 vs 0.333) and different neighbor
identities, so qwerty-magnitude sloppiness is genuinely more confusable there
— real dvorak corpus swipes are far *more* precise than our qwerty val source
(end-d 0.034 vs 0.077). A pure-translation control (residual vector copied
unrotated onto the target key) bounds the decomposition cost: 0.871/0.733 on
dvorak. The gap to that control is the price of movement-frame realism, and it
errs toward *harder* training samples, not easier ones. Accepted and stated.

Visual check (not committed): warped paths track the target polylines with
carried-over corner cuts and jitter; occasional small kinks appear where the
assigned segment switches (the residual frame rotates discretely). Plausible
augmentation, not simulation.

## 3. Runs

`phaseH-p15 / p30 / p50` — seed 1234, the exact `phaseG-C80-188k-nokd`
invocation (188 k steps, val-every 3000, beam-val 5000 rows, published
selection preset) plus `--layout-alt-p {0.15, 0.3, 0.5}`. Concurrent with the
Phase-G §8 latency probes on the same GPU (both tiny models; throughput shared).

## 4. Evaluation protocol

Per candidate, all through the exported ONNX graph:

* `eval_beam.py` on **full val-9918**, AOSP STRIP trie, E1 preset — the
  en_qwerty gate. Bar: within 0.3 pt of `resbn80g`'s 87.72 val seed-mean.
* `eval_altlayout.py` az26 arm, E1 preset, in-dict protocol, all five real
  alt-layout corpora + the dvorak app-trie arm (`--lexicon dvorak=en`) for the
  geometric-engine comparison (76.8 anchor was measured against the app's 98k
  trie). The per-layout preset caveat from `ALT_LAYOUT_EVAL.md` §9 applies
  unchanged: E1 was fitted on en_qwerty, the geo anchors are self-tuned, so
  CTC numbers are floors.
* Ship-candidate criterion (pre-stated): **dvorak (held-out) t1 ≥ 76.8** while
  the en_qwerty gate holds; if unreachable, map the pareto and say so.

## 5. The p ablation — every arm transfers, p=0.5 dominates

Seed 1234, exported ONNX, E1 preset. Alt-layout numbers are in-dict az26
(`eval_altlayout.py`); val is full val-9918 all-rows (`eval_beam.py`), AOSP
trie. Baselines: `resbn80g` = the Phase-G ship candidate (val = 3-seed
seed-mean; alt-layout = `fast_resbn80` s1234, the closest measured proxy —
resbn80g itself was never alt-layout-evaled); ch128 = the D1 ship artifact
from `ALT_LAYOUT_EVAL.md`.

| arm | val-9918 t1/t3/t5/≤3/4+ | dvorak | azerty | qwertz | german | spanish | dvorak app-98k |
|---|---|---|---|---|---|---|---|
| ch128 (no layout aug) | 88.02 / 92.27 / 93.03 / 91.12 / 86.41 (s1234) | 63.04 | 75.31 | 76.66 | 72.08 | 81.34 | 60.93 |
| resbn80 (no layout aug) | 87.72 / 92.25 / 92.97 / 90.78 / 86.14 (g, seed-mean) | 67.28 | 76.03 | 78.77 | 76.17 | 82.37 | — |
| `phaseH-p15` | 87.31 / 92.29 / 93.00 / 90.82 / 85.48 | 86.94 | 82.49 | 82.81 | 79.08 | 85.55 | 85.75 |
| `phaseH-p30` | 87.57 / 92.14 / 93.01 / 90.68 / 85.95 | 88.36 | 83.01 | 82.65 | 80.31 | 88.74 | 86.28 |
| **`phaseH-p50`** | **87.66 / 92.24 / 93.05 / 90.88 / 85.99** | **88.85** | **83.64** | **84.16** | **81.45** | 88.51 | **88.20** |
| geo engine (anchors) | — | 76.8 | 76.9 | 76.2 | 71.1 | 73.9 | 76.8 |

Reads:

* **The dose-response is monotone on the transfer axis and nearly flat on the
  QWERTY axis.** dvorak climbs 86.94 → 88.36 → 88.85 as p rises, while
  en_qwerty t1 *rises* with p in this sweep (87.31 → 87.57 → 87.66) — the
  opposite of the naive dilution expectation. At 188 k steps the model is
  nowhere near capacity-limited by the canonical data (the Phase-G §3 result
  that 94 k→188 k bought +0.05 says the same thing); geometry diversity acts
  as regularization, not as a data tax. p15's val deficit (−0.41 vs the
  seed-mean bar) is single-seed noise territory, but it is in any case
  dominated by p50 on every single column.
* **p=0.5 wins everywhere except spanish** (−0.23 vs p30, inside noise) —
  best val, best dvorak, best on every other layout. **Winner: p=0.5**, taken
  to three seeds in §6.
* **Val greedy falls** (71.2 % ch128 → 63.3 % p50) while beam t1 holds: the
  emission head trades QWERTY-specific sharpness for cross-layout robustness,
  and the lexicon beam — the metric that ships — keeps everything it needs.
* Mid-train probe (p50 best.pt at 45 k steps, dvorak `--limit 800`): t1 90.41
  — most of the transfer is learned in the first quarter of the schedule.

### Against the geometric engine, single seed

`phaseH-p50` beats the geo anchors on **all six** layouts, including the one
the whole phase was for: dvorak **+12.1** at the AOSP footing, **+11.4** on
the like-for-like app-98k-trie footing (88.20 vs 76.8) — where ch128 lost by
15.9. The four previous wins widen to +6.7…+14.6. The `ALT_LAYOUT_EVAL.md` §9
tuning asymmetry still applies (E1 is en_qwerty-fitted; the anchors are
self-tuned), so these deltas remain floors.

## 6. The winner — `resbn80h` (= Phase-G recipe + layout-alt p 0.5) at three seeds

Fresh trainings at seeds 4321 / 7777 (`phaseH-p50-s*`), identical recipe.
Exported ONNX (sliced-view parity 100/100 argmax on all three), E1 preset.

### 6.1 en_qwerty — full val-9918, AOSP trie, all-rows

| metric | s1234 | s4321 | s7777 | **seed-mean** | val bar | worst seed | `resbn80g` seed-mean | **Δ** |
|---|---|---|---|---|---|---|---|---|
| t1 | 87.66 | 87.81 | 87.61 | **87.69** | 85.52 | 87.61 PASS | 87.72 | −0.03 |
| t3 | 92.24 | 92.20 | 92.21 | **92.22** | 91.54 | 92.20 PASS | 92.25 | −0.03 |
| t5 | 93.05 | 92.93 | 93.01 | **93.00** | 92.80 | 92.93 PASS | 92.97 | +0.03 |
| ≤3 (n=3,389) | 90.88 | 90.94 | 90.56 | **90.79** | 89.29 | 90.56 PASS | 90.78 | +0.01 |
| 4+ (n=6,529) | 85.99 | 86.18 | 86.08 | **86.08** | 83.57 | 85.99 PASS | 86.14 | −0.06 |

All five bars clear on the seed mean and on every seed. Against `resbn80g` the
five deltas are −0.06…+0.03 — statistically indistinguishable (resbn80g's own
seed sd is 0.15–0.73). **The layout augmentation is free on en_qwerty.**
Val greedy drops 7 pt (71.4 → 64.4 seed-mean) — the emission head gives up
QWERTY-specific sharpness the lexicon beam never needed.

### 6.2 Alt-layouts — in-dict az26, E1, all rows of every corpus

t1/t3/t5 per seed and seed-mean; anchors = geometric engine, current basis.

| layout | s1234 | s4321 | s7777 | **seed-mean** | geo anchor | **Δt1** |
|---|---|---|---|---|---|---|
| **dvorak** (held out) | 88.85/95.03/96.78 | 90.15/96.95/97.80 | 91.05/97.15/97.80 | **90.01/96.38/97.46** | 76.8/79.9/80.4 | **+13.2** |
| dvorak, app 98k trie | 88.20/93.33/95.36 | 89.62/94.91/97.03 | 90.72/96.46/97.80 | **89.51/94.90/96.73** | 76.8/79.9/80.4 | **+12.7** |
| azerty | 83.64/95.26/97.22 | 84.88/95.79/97.51 | 84.31/95.65/97.13 | **84.27/95.57/97.29** | 76.9/89.9/93.7 | **+7.4** |
| qwertz | 84.16/93.93/96.21 | 84.41/94.44/96.46 | 84.50/94.78/96.38 | **84.36/94.38/96.35** | 76.2/87.4/90.6 | **+8.2** |
| german | 81.45/92.36/94.72 | 80.95/92.27/94.45 | 80.99/92.45/94.27 | **81.13/92.36/94.48** | 71.1/81.7/84.3 | **+10.0** |
| spanish | 88.51/95.79/96.81 | 88.51/94.77/96.87 | 88.28/95.22/96.64 | **88.43/95.26/96.78** | 73.9/86.6/89.8 | **+14.5** |

Greedy t1 seed-means (the no-lexicon column `ALT_LAYOUT_EVAL.md` §8 demanded):
dvorak **42.5** (was 11.6), azerty 33.8 (22.1), qwertz 38.1 (27.2), german
26.2 (15.1), spanish 42.2 (31.9). The emissions themselves transfer now — the
beam is no longer doing all of the work on dvorak (42.5 greedy on dvorak ≈
2/3 of the 64.4 on en_qwerty, vs 1/6 before).

Two honest footnotes:

* dvorak seed-order effect: s7777 > s4321 > s1234 on dvorak while the val
  ordering is s4321 > s1234 > s7777 — cross-layout accuracy is not simply a
  function of the en_qwerty selection metric, and checkpoint selection still
  runs on the en_qwerty val prefix only. A transfer-aware selection metric is
  an open lever, not needed at these margins.
* dvorak now *out-decodes* german/azerty/qwertz. Not a paradox: dvorak's
  corpus is English against the largest, best-calibrated lexicon (146,964
  AOSP / 98k app), the German/French corpora carry the compressed-λ CKDT
  confound (`ALT_LAYOUT_EVAL.md` §3) and untypeable `ß` rows. Geometry
  stopped being the axis that orders the table — which is the point.

### 6.3 Artifacts, parity, latency

Same contract as every campaign artifact (opset 17, fp32, static shapes, zero
normalization nodes, 231-node graph **identical** to `resbn80g` — weights are
the only difference). Sliced-view parity at export: 100/100 argmax agreement,
max |Δ| ≤ 7.6e-05 on all three. Idle latency (`bench_latency.py`, paired):
`resbn80h_s1234` 0.216 ms mean / 0.229 p90 vs `resbn80g_s1234` 0.212 / 0.222
— same 0.215 ms class, as the identical graph dictates. 1,142,727 bytes.

```
3e215438f3c8fae1f249b91be3986bc30c027920f158371acaea0d159dbeff00  resbn80h_s1234.onnx
b3f30bcd33cd1137300b039ae166ccd9bdd7ea9117502c35f9d0d80d9a277331  resbn80h_s4321.onnx
1a1edac6f10f0fd88b427ce41b4808e46bef1e4209b4611dc7c9e81b5e5e94dd  resbn80h_s7777.onnx
```

Run logs and per-run JSON: `~/ctc-train/ckpt/phaseH-*` and
`~/ctc-train/altlayout/phaseH-*` (not committed).

## 7. Routing recommendation update — the displacement gate disappears

`ALT_LAYOUT_EVAL.md` §8 recommended routing by **mean a-z key displacement
from the training layout**: ≤ 0.11 → CTC, 0.43 (dvorak) → geometric engine.
On `resbn80h` weights that gate is obsolete:

* The displacement-0.4313 layout (dvorak) now decodes at 90.01 — *above*
  every 0.06–0.11-displacement layout and 13 pt above the geometric engine.
* The failure mode the gate guarded against (key re-arrangement) is exactly
  what the augmentation trains, including arrangements no human layout uses
  (random permutations), so Colemak / Neo2 / arbitrary user XML sit *inside*
  the training family rather than beyond a measured frontier. Colemak-class
  layouts still have no real corpus here — that eval gap remains open — but
  there is no longer a measured regime where the geometric engine wins.
* **Recommended routing on these weights: CTC everywhere a layout provides
  a-z key centers.** The two-line displacement check can be kept as a cheap
  telemetry signal, but nothing should route on it.

Residual caveats that travel with this: non-Latin scripts remain untested
(unchanged); `ß` remains untypeable on the a-z head (unchanged); per-layout
preset sweeps would likely add several points on the fr/de/es corpora (the λ
confound is untouched); and `resbn80h` is val-validated only — the en_qwerty
test-2400 tier stays with `resbn80g` unless a fourth unsealing is authorized.

## 8. What Phase H did NOT do

* No test-2400 read, no golden-fixture change, no app-preset re-sweep (the
  §6 app-trie dvorak numbers reuse Phase G's E1; the adopted app preset
  `0.9/4.0/0.25/0.25/0.9882` was fitted for `resbn80g` and would need a
  re-check before shipping `resbn80h`).
* No KD, no schedule change, no architecture change — the augmentation is the
  only delta vs `resbn80g`, which is what makes the attribution clean.
* No dvorak (or qwerty) geometry in the training pool, and no real corpus of
  any kind in training — the alt-layout corpora were spent exclusively on
  evaluation.
