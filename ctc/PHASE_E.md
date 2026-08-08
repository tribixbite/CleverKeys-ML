# Phase E — closing the gap to the FUTO ceiling on val-9918

Phase D ended at a seed-mean full-val top-1 of **84.81** against a FUTO ceiling of
84.83 measured on *test*-2400, and recommended not spending the test seal. Phase E
starts from a better-specified target: the ceiling re-measured on **our** val-9918.

## 0. The bar

FUTO's encoder+refinement ceiling, decoded on val-9918 (the app repo's committed
eval), is the five-number vector every Phase-E result is quoted against:

| metric | FUTO ceiling on val-9918 | Phase-D `D1` seed-mean | deficit |
|---|---|---|---|
| overall t1 | **85.52** | 84.81 | −0.71 |
| t3 | **91.54** | 91.01 | −0.53 |
| t5 | **92.80** | 92.33 | −0.47 |
| ≤3-char t1 (n=3,389) | **89.29** | 88.01 | −1.28 |
| 4+-char t1 (n=6,529) | **83.57** | 83.15 | −0.42 |

The gate for unsealing test-2400 is a **3-seed seed-mean on full val that beats all
five**. **test-2400 was not decoded in this phase**, whatever the gate says —
`eval_arms.py` and `train.py` both still refuse any split whose filename contains
`test`.

---

## 1. E1 — the scoring preset was the largest lever in the campaign

### What was measured

`sweep_scoring.py` re-tunes the trie beam's five scoring parameters on val rows
`0:4959`, confirms the winner on the untouched `4959:9918`, and reports full val.
Run on `phaseD-D1` (ch 128 on T3, seed 1234), five successive grids were needed
because the first four all put the optimum **on a grid boundary**:

| grid | γ span | β span | λ span | winner (γ, λ, β) | holdout-half Δt1 | full-val t1 |
|---|---|---|---|---|---|---|
| 1 (Phase-B width) | 0.20–0.51 | 0.79–1.08 | 0–0.035 | 0.4056, 0.035, 0.79 | **+0.50** | 84.67 |
| 2 | 0.25–0.56 | 0.50–0.99 | 0.026–0.10 | 0.61, 0.10, 0.50 | **+1.03** | 85.14 |
| 3 | 0.45–1.20 | 0.00–0.65 | 0.07–0.32 | 1.25, 0.32, 0.075 | **+2.98** | 86.76 |
| 4 | 1.00–3.00 | 0.00–0.20 | 0.25–1.50 | 1.225, 0.80, 0.125 | **+3.09** | 86.96 |
| 5 (fine, interior) | 1.10–1.40 | 0.05–0.25 | 0.50–1.30 | **1.05, 1.1, 0.2** | **+3.04** | **86.96** |

Grid 5 is the first whose winner is not on an edge, and grids 4 and 5 agree to
0.00 pt on full val, so the search is converged. The **adopted E1 preset** is

```
gamma = 1.05   lambda = 1.1   beta = 0.2   gammaPrune = 0.3734   betaPrune = 0.9882
```

against the published `encoderOnly`
`(0.4056, 0.0176, 0.9866, 0.4234, 1.0382)`.

### `phaseD-D1` under both presets — **full val-9918, seed 1234 only**

Every number in this table is all 9,918 val rows (not the 5,000-row selection
prefix) for the single seed-1234 checkpoint. It is **not** a seed-mean and must
not be read as one; the seed-mean is two tables below.

| preset | t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|
| published `enc` | 84.22 | 90.74 | 92.22 | 87.40 | 82.57 |
| **E1 tuned** | **86.96** | **91.85** | **92.78** | **89.70** | **85.54** |
| Δ | **+2.74** | **+1.11** | **+0.56** | **+2.30** | **+2.97** |

The gain is +2.44 on the rows it was fitted to and **+3.04 on the 4,959 rows it
never saw** — it is larger on the holdout half than on the tuning half, which is
the opposite of what sweep overfit looks like.

### The fast path was verified at the tuned preset, not just the published one

`sweep_scoring`/`eval_arms` re-score the terminal beam analytically instead of
re-running it per grid point. That identity was previously checked only at the
published preset, where the multipliers are small. Re-checked here at the tuned
preset through `eval_beam.py`'s **real per-row decoder** on val rows `0:2000`:

| path | t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|
| `eval_beam.py` (real beam, per row) | 87.15 | 92.05 | 92.80 | 89.40 | 85.94 |
| `sweep_scoring.py` (analytic re-score) | 87.15 | 92.05 | 92.80 | 89.40 | 85.94 |

Identical to the digit. The identity holds because `futo_viterbi_beam`'s per-frame
pruning key is a function of `(gammaPrune, betaPrune)` alone — `gamma`, `beta` and
`lambda` enter only the final score — which is visible directly in the vendored
source, and both prune params are held fixed between the two runs above.

### ⚠ Retraction — "there is no free win" was a grid-width artifact

`README.md` §"Scoring sweep" and `PHASE_B.md` §4 both concluded that the published
preset was already optimal for our emissions. `README.md` went further and quoted a
**headroom bound**: a grid run directly on all 9,918 val rows (selection on the
scored rows, an optimistic upper bound) "tops out at 81.78 top-1 vs 81.57
baseline, i.e. +0.21 pt maximum".

Every one of those grids spanned γ ∈ [0.30, 0.51], β ∈ [0.89, 1.08], λ ≤ 0.026 —
all centred on the published preset. The optimum for our emissions is at
γ ≈ 1.05, β ≈ 0.2, λ ≈ 1.1, which is **outside every grid the campaign had ever
run**. Re-swept on the *same* `r2` model with a wide grid, honest halves:

| `r2`, val-9918 | t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|
| published `enc` (the committed number) | 81.57 | 89.84 | 91.37 | 86.28 | 79.12 |
| re-tuned (γ 1.125, λ 1.2, β 0.25) | **86.14** | **91.01** | **92.12** | **89.94** | **84.16** |
| Δ full val | **+4.57** | +1.17 | +0.75 | +3.66 | +5.04 |
| Δ on the untouched holdout half | **+4.25** | +1.33 | +0.75 | +2.83 | +4.99 |

**The +0.21 pt "headroom bound" understated the real gain by a factor of ~20.**
The claim in `README.md` must be read as bounded by its grid, and is withdrawn.
So is `PHASE_B.md`'s "re-tuning does not rescue them" framing: the per-arm gains
it measured (0.08–1.03 pt) were all inside the same too-narrow box.

This does not change any Phase A–D **arm-vs-arm** conclusion — every arm in those
tables was decoded at the same published preset, so the mis-tuning is common-mode
and cancels. It changes every **absolute** number in the campaign, and it changes
the distance to the FUTO ceiling.

### `D1` at three seeds under the E1 preset — full val-9918

The preset was tuned on seed 1234's emissions; seeds 4321 and 7777 are a transfer
test for it. All rows are full val-9918; the seed-mean row is the only figure
comparable to the bar.

| arm | seed | t1 | t3 | t5 | ≤3 | 4+ | FUTO | HWS |
|---|---|---|---|---|---|---|---|---|
| `phaseD-D1` | 1234 | 86.96 | 91.85 | 92.78 | 89.70 | 85.54 | 94.60 | 79.38 |
| `phaseD-D1` | 4321 | 87.23 | 91.92 | 92.68 | 89.97 | 85.80 | 94.94 | 79.56 |
| `phaseD-D1` | 7777 | 87.45 | 92.03 | 92.91 | 90.06 | 86.09 | 95.00 | 79.94 |
| **seed-mean** | | **87.21** | **91.93** | **92.79** | **89.91** | **85.81** | 94.85 | 79.63 |
| — the bar | | 85.52 | 91.54 | 92.80 | 89.29 | 83.57 | — | — |
| **Δ vs bar** | | **+1.69** | **+0.39** | **−0.01** | **+0.62** | **+2.24** | | |

**Four of the five bars clear on E1 alone; t5 misses by 0.01 pt** (92.79 vs
92.80) — about one row in 9,918, averaged over three seeds. **The gate is not
passed**, and a 0.01 pt shortfall is far below anything this campaign can
resolve, so it should be read as "t5 is level with the ceiling", not as "t5 is
behind it". Either way the gate is a conjunction of five *strict* wins and this
is not one, so E1 alone does not unseal test-2400.

Note the E1 preset also helps the *harder* half more: on `D1` seed 1234 it moves
the FUTO half +2.35 (92.25 → 94.60) and the How-We-Swipe half **+3.13**
(76.25 → 79.38). A preset whose largest single parameter change is a 60× increase
in the word-frequency weight is the one most exposed to a holdout's vocabulary
distribution, so the fact that the out-of-distribution corpus gains *more* is the
main evidence that this is not a val-specific artifact. It is not proof, and §5
treats it as an open risk.

A t5-favoured point inside the flat region (γ 1.10, λ 1.1, β 0.15, at the
*published* prune params) was tested rather than assumed: it wins t5 on the tuning
half (92.92 vs 92.88) and **loses** it on full val (92.75 vs 92.78). The t5
differences across the flat region are not resolvable, so the preset was left at
the t1 optimum its own protocol selected, and no point was chosen for clearing a
bar it was measured against.

### The prune params, tested properly — mostly a null

`gamma` moved 0.41 → 1.05, and `gammaPrune` plays the same length-normalising role
during *survival*: a mis-tuned one drops candidates before the final score ever
sees them, which would cap t3/t5 no matter what the score does. Every previous
sweep could only search the published value ±0.05, so `sweep_scoring.py` gained
explicit `--grid-gamma-prune`/`--grid-beta-prune`. Swept 0.3734–1.35 on
`phaseE-E4-ch192`:

| gammaPrune | 0.3734 | 0.60 | 0.85 | 1.10 | 1.35 |
|---|---|---|---|---|---|
| best sweep-half t1 (betaPrune 0.5) | **88.47** | 88.36 | 88.30 | 87.60 | 79.75 |
| best sweep-half t1 (betaPrune 0.9882) | 88.40 | 88.32 | 87.46 | 75.38 | 14.62 |

`gammaPrune` wants to stay **low** — it is already near-optimal at the published
0.3734 and collapses catastrophically above ~1.0. `betaPrune` prefers 0.5 to the
published 0.9882 by 0.07 pt, which is noise. **The prune hypothesis is tested and
mostly dead**; unlike the score params, the published prune setting was close to
right. It is included in later grids anyway, since it costs one beam pass each.

---

## 2. E2 — the refinement head, retested on a strong base and re-tuned. Still null

`train_refine.py` was brought up to Phase-D/E standards for this arm: a step
budget instead of an epoch count, the T3 cache instead of T0, and **beam-t1
checkpoint selection at the E1 preset** rather than greedy (Phase 2 had selected
its head on the metric Phase B showed anti-correlates with the beam).

Trained on the frozen `phaseD-D1` seed-1234 base, 30,000 steps, batch 256:

| config | preset | t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|---|
| base `D1` | E1 tuned | **86.96** | **91.85** | **92.78** | **89.70** | **85.54** |
| + refine head | E1 tuned | 86.74 | 91.66 | 92.65 | 89.47 | 85.33 |
| + refine head | its **own** re-tuned preset | 86.81 | 91.69 | 92.68 | 89.61 | 85.36 |

**Negative on every metric and every stratum, under both presets.** The head's own
selection metric peaked at 86.86 on the 5,000-row val prefix against the base's
87.24 on the identical rows — it never once beat its own input.

The re-tune matters for fairness and was run: refined emissions have a different
calibration, so scoring them at a preset fitted to the unrefined ones would have
been a rigged comparison. It closes about a third of the gap and leaves the sign
unchanged.

Phase 2's original result was "+0.00 aggregate, **+1.00 on ≤3**", and that ≤3
profile is why the brief ranked this arm second. **It does not reproduce**: ≤3 is
−0.23 (E1 preset) / −0.09 (own preset). The stated reason for the original null —
FUTO's head recovered a base whose greedy was 43.96 %, while ours greedy-decodes
at 67 % — applies with more force to a base that is now stronger still, and the
one stratum that had looked promising was evidently the weak base's slack rather
than a property of the lever. **Phase 2 stays closed, now on much better
evidence.**

---

## 3. E5 and E4 — selection width and capacity

All arms in this section are **seed 1234, T3, 94,000 steps**, selected on beam
top-1 over a **5,000**-row val prefix at the published preset, and reported on
full val-9918 at the E1 preset. They are therefore paired with each other on
everything except the one variable named.

| arm | ch | params | t1 | t3 | t5 | ≤3 | 4+ | FUTO | HWS |
|---|---|---|---|---|---|---|---|---|---|
| `phaseD-D1` (2,000-row selection) | 128 | 689 k | 86.96 | 91.85 | 92.78 | 89.70 | 85.54 | 94.60 | 79.38 |
| `phaseE-E5base` (5,000-row selection) | 128 | 689 k | 87.19 | 92.09 | 92.87 | 90.06 | 85.71 | 94.80 | 79.64 |
| **`phaseE-E4-ch192`** | **192** | **1.525 M** | **87.67** | **92.11** | **92.90** | **90.35** | **86.28** | 95.22 | 80.16 |

**E5 (2,000 → 5,000-row selection prefix): +0.23 t1**, and positive on all five
metrics. Phase D predicted "worth ~0.2 pt" from the measured −0.17 pt that the
noisy 2,000-row rule had cost `D1`; the measured value is +0.23. One seed, so it
is not resolvable on its own, but it is a prediction made in advance that came
back the right size and sign, it costs ~3 s per validation, and it cannot be
harmful in expectation. **Adopted.**

**E4 (ch 128 → 192): +0.48 t1** paired against `E5base`, positive on all five
metrics, largest on 4+ (+0.57). That is inside the ~1 pt single-seed noise floor
and so is *not* resolvable at one seed — it goes to the 3-seed round rather than
being adopted here.

### Checkpoint selection is on a plateau, so the selection preset does not matter

Phase E reports at a preset far from the one the selection beam scores at, which
is a real mismatch. `train.py --select-preset` was added to close it, but the
measurement says there is nothing to close: for `E4`, the selected checkpoint
(step 87,000) and the final step (94,000) score **87.67 and 87.65** on full val
at the E1 preset — 0.02 pt apart. Under a cosine-to-zero schedule the last
~10,000 steps are indistinguishable, so which of them a selection rule picks
cannot move the result. Every Phase-E arm therefore keeps published-preset
selection, which also keeps them all mutually paired.
