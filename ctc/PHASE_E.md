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

---

## 4. E3 — the data-mix arms. Oversampling How-We-Swipe is the model-side win

Both arms are ch 128, seed 1234, 94,000 steps, 5,000-row selection — paired with
`phaseE-E5base` on everything but the training pool. Full val-9918, E1 preset.

| arm | training pool | rows | t1 | t3 | t5 | ≤3 | 4+ | FUTO | HWS |
|---|---|---|---|---|---|---|---|---|---|
| `phaseE-E5base` | T3 | 1,005,336 | 87.19 | 92.09 | 92.87 | 90.06 | 85.71 | 94.80 | 79.64 |
| `phaseE-E3a-T4` | T4 (curated FUTO) | 764,771 | 86.93 | 91.58 | 92.47 | 90.00 | 85.34 | 94.80 | 79.12 |
| **`phaseE-E3b-hws3x`** | **T3, HWS 3×** | **1,158,832** | **88.02** | **92.27** | **93.03** | **91.12** | **86.41** | **95.00** | **81.09** |

### E3a — curation does not pay at benchmark scale either

T4 is T3's contamination policy (exact-trace dedup only) applied to T1's curated
688,025-row FUTO pool instead of the raw corpus, so T3 vs T4 isolates the user's
curation at benchmark scale. **T4 is −0.26 t1 and negative on all five metrics.**
Phase A found the quality cascade net-harmful on the FUTO half at 385 k rows;
this reproduces the sign at 765 k rows against a different comparator. The
curation is not the lever — third independent look, third negative. `futo_leak_xy`
was **0** on the curated pool, confirming Phase A's finding that the user's
train/val FUTO pools were already exact-trace disjoint.

### E3b — 3× How-We-Swipe is +0.83 t1, and it fixes the half that was broken

Duplicating rows inside a tier jsonl would be undone by `prepare_data.py`'s exact
self-dedup, so the arm concatenates a standalone 76,748-row HWS cache onto T3
twice (`--train-npz train_t3.npz,train_t3hws.npz,train_t3hws.npz`) — exact 3×
repetition under a plain without-replacement shuffle rather than a weighted
sampler's with-replacement approximation.

**+0.83 t1, and positive on every metric and both strata** (≤3 **+1.06**, 4+
+0.70). The mechanism is legible and is the one the campaign has been describing
since Phase A: the holdout is ~50 % How-We-Swipe, our HWS half has trailed the
FUTO half by 15 pt throughout, and T3 is 92 % FUTO by row count. Oversampling the
under-represented corpus moves **HWS +1.45** (79.64 → 81.09) against FUTO +0.20 —
almost all of the aggregate gain comes from the half that was weak.

At +0.83 this is at the edge of the ~1 pt single-seed floor and is *not* resolved
by one seed. What raises confidence above the bare number is that all five
metrics, both strata and both corpora move the same way, with the largest move on
the exact sub-population the intervention targets. It goes into the final stack
and is tested at three seeds there.

> ⚠ **T3's disclosure carries to T3-3× and to T4 unchanged** (`PHASE_D.md` §2).
> None of these tiers applies session or participant exclusion, so every val row's
> contributor is in training. They are benchmark tiers, comparable with published
> FUTO numbers; **none can support a generalization claim.** Oversampling HWS
> makes this *worse* in one specific way worth stating: the HWS half is the more
> contributor-contaminated of the two (98.4 % of T0's HWS rows share a participant
> with the holdout), so tripling it triples the exposure of exactly those
> contributors. The +1.45 pt HWS gain is therefore an upper bound on what a new
> user would see, by an unknown margin.

### Checkpoint selection is on a plateau, so the selection preset does not matter

Phase E reports at a preset far from the one the selection beam scores at, which
is a real mismatch. `train.py --select-preset` was added to close it, but the
measurement says there is nothing to close: for `E4`, the selected checkpoint
(step 87,000) and the final step (94,000) score **87.67 and 87.65** on full val
at the E1 preset — 0.02 pt apart. Under a cosine-to-zero schedule the last
~10,000 steps are indistinguishable, so which of them a selection rule picks
cannot move the result. Every Phase-E arm therefore keeps published-preset
selection, which also keeps them all mutually paired.

### Latency, measured idle

Single-thread, batch-1, fixed-shape ONNX Runtime CPU, 300 runs × 3 interleaved
rounds, best round, machine idle.

| config | params | mean ms | p90 ms |
|---|---|---|---|
| `phaseE-E5base` (ch 128) | 689 k | 0.474 | 0.489 |
| `phaseE-E3b-hws3x` (ch 128) | 689 k | 0.470 | 0.485 |
| **`phaseE-E4-ch192`** | **1.525 M** | **0.898** | **0.914** |

ch 192 costs **1.9×** ch 128, landing at 0.90 ms — inside the 0.8–1.0 ms it was
budgeted at. An earlier reading of 1.54 ms / 1.90 ms p90 was taken with three
training runs on the box and is **withdrawn**: it measured contention, not the
graph. Latency figures in this campaign are only valid taken idle, which is what
the Phase-C protocol already said and what this nearly got wrong.

---

## 5. FINAL — the stacked configuration at three seeds

**Configuration.** ch 192 / `embed_hid` 192 residual trunk (E4), trained on T3
with its How-We-Swipe half oversampled 3× (E3b), 94,000 steps, batch 256, lr 3e-3,
wd 0.01, warmup 1,000, fp32, checkpoint selected on beam top-1 over a 5,000-row
val prefix (E5), decoded at the E1 preset. 1,525,378 params, 0.898 ms.
Seeds 1234 / 4321 / 7777.

E3a (T4) and E2 (the refinement head) were measured and **not** stacked — both
were negative.

### The preset used, and why it is the *less* contaminated choice

The E1 preset was tuned on `phaseD-D1`'s emissions — a **different model**
(ch 128, plain T3). Re-tuned from scratch on the final seed-1234 model, the sweep
lands at γ 0.975, λ 1.1, β 0.3, γp 0.3734, βp 0.5 and scores
**88.23 / 92.22 / 93.00** (≤3 91.12, 4+ 86.74) on full val, against
**88.22 / 92.23 / 93.08** (≤3 91.15, 4+ 86.71) for the transferred E1 preset —
identical within 0.08 pt on every metric.

Since re-tuning buys nothing measurable, the results below are reported at the
**transferred E1 preset**, which was never fitted to this model at all. That the
preset transfers unchanged across both a capacity change and a data-mix change is
also the best evidence available that it fits the task and the lexicon rather than
one model's quirks.

### Results — full val-9918, E1 preset

| metric | s1234 | s4321 | s7777 | **seed-mean** | sd | **the bar** | **Δ** | gate |
|---|---|---|---|---|---|---|---|---|
| overall t1 | 88.22 | 87.80 | 88.17 | **88.06** | 0.23 | 85.52 | **+2.54** | **PASS** |
| t3 | 92.23 | 92.34 | 92.38 | **92.32** | 0.08 | 91.54 | **+0.78** | **PASS** |
| t5 | 93.08 | 93.17 | 92.99 | **93.08** | 0.09 | 92.80 | **+0.28** | **PASS** |
| ≤3 t1 (n=3,389) | 91.15 | 90.62 | 90.82 | **90.86** | 0.27 | 89.29 | **+1.57** | **PASS** |
| 4+ t1 (n=6,529) | 86.71 | 86.34 | 86.80 | **86.62** | 0.24 | 83.57 | **+3.05** | **PASS** |

Per-source seed-mean: **FUTO 94.99**, **HWS 81.19** (against 92.87 / 76.80 at the
end of Phase D). Seed sd on t1 is **0.23**, against `D1`-on-T3's 0.56 in Phase D —
the oversampled mix is also markedly more stable across seeds.

### The same table on rows that were never used for anything

Full val is the basis the gate was specified on, but half of it (`0:4959`) fed the
E1 preset sweep and the first 5,000 rows fed checkpoint selection. Rows
`4959:9918` fed **neither** (a 41-row overlap aside). That subset is the strictest
estimate available:

| metric | seed-mean, holdout half | sd | the bar | Δ | gate |
|---|---|---|---|---|---|
| overall t1 | **87.58** | 0.32 | 85.52 | **+2.06** | **PASS** |
| t3 | **92.03** | 0.07 | 91.54 | **+0.49** | **PASS** |
| t5 | **92.85** | 0.11 | 92.80 | **+0.05** | **PASS** |
| ≤3 t1 | **90.67** | 0.53 | 89.29 | **+1.38** | **PASS** |
| 4+ t1 | **85.98** | 0.22 | 83.57 | **+2.41** | **PASS** |

All five clear on both bases. **t5 is the number to read carefully: +0.28 on full
val but +0.05 on untouched rows** — on the strictest reading top-5 is *level* with
the FUTO ceiling rather than above it. Every other metric keeps a margin far
outside anything this campaign can attribute to noise.

**test-2400 was not decoded.** `eval_arms.py` refuses any split whose filename
contains `test` and `train.py` refuses it as a selection split; neither guard was
touched or bypassed in this phase.

### The capacity lever, resolved at three paired seeds

The ch 128 control was run at the same three seeds on the same tier, so ch 192 vs
ch 128 is a fully paired test at the final data mix:

| metric | s1234 | s4321 | s7777 | mean Δ | paired t(2) |
|---|---|---|---|---|---|
| t1 | +0.20 | +0.17 | +0.19 | **+0.19** | 21.2 |
| t3 | −0.04 | +0.15 | +0.15 | +0.09 | 1.4 |
| t5 | +0.05 | +0.25 | +0.05 | +0.12 | 1.8 |
| ≤3 | +0.03 | −0.23 | −0.15 | **−0.12** | −1.5 |
| 4+ | +0.30 | +0.39 | +0.37 | **+0.35** | 13.0 |

**Capacity is real but much smaller than one seed suggested.** On plain T3 at
seed 1234 it measured +0.48 t1 (§3); paired at three seeds on the oversampled
tier it is **+0.19**. It is sign-stable on t1 and 4+ and the paired t clears the
4.30 threshold on both — though a t at n=3 rests on a 2-df variance estimate and
should be read as "consistent", not "precise". The stratum signature is the one
this campaign has seen from every sharpening lever since Phase B: **4+ +0.35, ≤3
−0.12**.

### ⚠ ch 128 also passes the gate, at half the latency

| config | params | ms | t1 | t3 | t5 | ≤3 | 4+ | all five? |
|---|---|---|---|---|---|---|---|---|
| the bar | — | — | 85.52 | 91.54 | 92.80 | 89.29 | 83.57 | — |
| **ch 128** + 3× HWS | 689 k | **0.470** | 87.88 | 92.23 | 92.96 | **90.98** | 86.26 | **PASS** |
| **ch 192** + 3× HWS | 1.525 M | 0.898 | **88.06** | **92.32** | **93.08** | 90.86 | **86.62** | **PASS** |

Both seed-means beat all five numbers. ch 192 buys +0.19 t1 for **1.9× the
encoder time and 2.2× the parameters**, and is actually *behind* on the ≤3
stratum. Reported as the headline because it is the best-measured configuration
and was the stacked arm the brief asked for, but **ch 128 is the better shipping
trade** and clears the same gate — that choice should be made on device budget,
not on these 0.19 pt.

---

## 6. Summary of decisions

* **Adopt the re-tuned scoring preset (E1)** — `gamma 1.05, lambda 1.1, beta 0.2,
  gammaPrune 0.3734, betaPrune 0.9882`. Worth **+2.7 to +4.6 pt top-1** depending
  on the model, more on untouched rows than on the tuning rows, verified against
  the real per-row decoder, and stable across a capacity change and a data-mix
  change. This is the largest single lever in the entire campaign, and it is
  free at inference.
* **Adopt 3× How-We-Swipe oversampling (E3b)** — +0.83 t1 at one seed, and the
  gain lands on the corpus half that was 15 pt behind. Also cuts seed sd from
  0.56 to 0.23.
* **Adopt the 5,000-row selection prefix (E5)** — +0.23 t1, the size and sign
  Phase D predicted in advance, for ~3 s per validation.
* **Adopt ch 192 only if the device budget allows it** — +0.19 t1 paired at three
  seeds for 1.9× encoder latency, and −0.12 on ≤3. ch 128 passes the same gate.
* **Reject T4 / curation at benchmark scale (E3a)** — −0.26 t1, negative on all
  five. Third independent negative for the user's quality cascade.
* **Reject the refinement head (E2)** — negative on every metric and stratum under
  both its own preset and E1's. Phase 2 stays closed.
* **The prune params are fine as published** — searched 0.25–1.35 for the first
  time; `gammaPrune` wants to stay low and collapses above ~1.0.

### Withdrawn claims

| claim | where | status |
|---|---|---|
| "the guide's free win does not exist for this model" | `README.md` | **withdrawn** — grid too narrow |
| "+0.21 pt maximum" scoring headroom on `r2` | `README.md` | **withdrawn** — real gain +4.25 pt on untouched rows |
| "re-tuning does not rescue them" (0.08–1.03 pt per arm) | `PHASE_B.md` §4 | **bounded by its grid**; the arm ranking still stands |
| ch 192 latency 1.54 ms / 1.90 ms p90 | this doc, earlier revision | **withdrawn** — measured under load; idle is 0.898 / 0.914 |
| ch 192 is worth +0.48 t1 | §3, one seed | superseded by **+0.19** at three paired seeds |

Every Phase A–D **arm-vs-arm** conclusion survives: all those arms were decoded at
the same published preset, so the mis-tuning was common-mode. Every **absolute**
number from those phases is understated by 2–5 pt.

### What is still not established

1. **Generalization.** T3-3× applies no session or participant exclusion, and
   oversampling HWS triples the exposure of the more contaminated corpus. These
   are benchmark numbers, comparable with published FUTO figures because the
   holdout traces are removed bit-exactly where FUTO kept them — **not** a claim
   about an unseen user.
2. **Preset transfer beyond this holdout.** λ moved 0.0176 → 1.1, a 60× increase
   in the word-frequency weight. It helps the out-of-distribution HWS half *more*
   than the FUTO half, which is the strongest evidence available that it is not a
   val artifact, but it is tuned on val-9918's vocabulary distribution and its
   behaviour on rare words and proper nouns is untested.
3. **The comparison is now asymmetric in our favour.** Our preset is tuned; the
   FUTO ceiling was measured at its own published preset. A fair rematch would
   re-tune both. This asymmetry did not exist in Phase D and does now.
4. **t5 on untouched rows is +0.05** — level with the ceiling, not above it.

---

## 7. Reproduction

```bash
python build_tiers.py --tiers t4,t3hws
python prepare_data.py --extra-train data/tier_t3hws.jsonl --out-name train_t3hws --jobs 10
python prepare_data.py --extra-train data/tier_t4.jsonl   --out-name train_t4   --jobs 10

# the final stacked arm, one seed
python train.py --train-npz train_t3.npz,train_t3hws.npz,train_t3hws.npz \
                --run-name phaseE-FINAL-s1234 --ch 192 --embed-hid 192 \
                --total-steps 94000 --val-every 3000 --batch 256 --lr 3e-3 \
                --weight-decay 0.01 --warmup 1000 --seed 1234 \
                --beam-val-rows 5000 --beam-jobs 10
python export_onnx.py --ckpt ckpt/phaseE-FINAL-s1234/best.pt \
                      --out ckpt/phaseE-FINAL-s1234/ctc_swipe_encoder.onnx

# the E1 preset sweep (note the WIDE grid — the narrow one hides the whole effect)
python sweep_scoring.py --onnx ckpt/phaseE-FINAL-s1234/ctc_swipe_encoder.onnx \
       --cache ckpt/phaseE-FINAL-s1234/eval_emissions.npz \
       --sweep-rows 0:4959 --holdout-rows 4959:9918 --full-rows 0:9918 \
       --grid-gamma 0.85,0.95,1.05,1.15,1.25,1.4 --grid-beta 0.05,0.125,0.2,0.3 \
       --grid-lambda 0.6,0.8,1.1,1.4,1.8 \
       --grid-gamma-prune 0.25,0.3734,0.5 --grid-beta-prune 0.5,0.9882

# report (test-2400 is refused)
python eval_arms.py --arms phaseE-FINAL-s1234,phaseE-FINAL-s4321,phaseE-FINAL-s7777 \
       --preset 1.05,1.1,0.2,0.3734,0.9882 --own-mask T0 --also-masks "" \
       --latency-runs 300
```
