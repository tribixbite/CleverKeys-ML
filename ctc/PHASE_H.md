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

*(filled in §5–§7 after the runs.)*

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

## 5. The p ablation

*(pending)*

## 6. The winner at three seeds

*(pending)*

## 7. Routing recommendation update

*(pending — does the mean-key-displacement gate move or disappear?)*
