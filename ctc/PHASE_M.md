# Phase M — close-out: five-seed footing, the E7 crown attempt, and the last knobs

**Date opened:** 2026-08-13 · **Authority:** coordinator directive of
2026-08-13 (user-approved close-out; the last training phase). Rules unchanged
from Phase L: pre-register before launch, gate-first blind order per pair, all
runs detached with `--workers 0`, commit + push at milestones, **test-2400
SEALED — the final unsealing is the orchestrator's act after this report, not
mine.** Prior record: `PHASE_L.md` (bar 2 met, bar 1 not met, E2 refuted).

## 0. What Phase M has to settle

1. **Are `L1 member A`'s two tie margins real?** Its 11/11 seed-mean rests on
   t3 **+0.000** (an exact tie) and qwertz **+0.007**. Three seeds cannot
   distinguish those from noise. Five can do better.
2. **Can one model wear the crown?** E7 — distil the *alignment-consistent*
   pair-average into a single student. PHASE_L §11.1 showed the teacher's
   gauge is now controllable, which removes by construction the confound that
   PHASE_L §1.3 (via Phase G) identified as fatal to the old KD refutation.
3. **The three unswept knobs** (Stage 2): `--pair-weight`, E4 `w_real`, E6.

## 1. STAGE 1 — pre-registration (committed BEFORE launch)

### 1.1 (a) Two more seeds of the L1 pair recipe

`v2pair-s5555` (`--seed 5555 --init-seed-a 7777 --init-seed-b 8888`) and
`v2pair-s9999` (`--seed 9999 --init-seed-a 1010 --init-seed-b 2020`),
otherwise the PHASE_L §3 command verbatim.

**Decision rule, fixed now.** Member A's eleven-bar tally is recomputed at
**five seeds**. The claim "first single model to clear all eleven campaign
bars on the seed-mean" **survives only if the five-seed mean still clears all
eleven**. If t3 or qwertz goes under, PHASE_L §15.4 is **retracted in place**
and the finalist reverts to a 10/11 or 9/11 statement. A tie that stays a tie
(|Δ| < 0.01) is reported as a tie, not as a pass.

### 1.2 (b) E7 — alignment-matched distillation

Teacher: **the L1 s1234 gated pair** (`pair_a_best.pt` + `pair_b_best.pt`,
measured 98.33 % per-frame agreement), passed to `train.py --kd-teacher` as a
two-checkpoint ensemble — whose target is `logsumexp − log N`, i.e. **exactly
the prob-averaged mix2 contract**. Our own models throughout; no FUTO output
anywhere. Student: single ch192 `resbn`, the L1 single-model recipe, 188 k.

| arm | init | `--kd-weight` | tests |
|---|---|---|---|
| `v2kd-initA-w1` | **member A** (`--init-from`, new flag) | 1.0 | the proposal's spec: gauge-matched student |
| `v2kd-fresh-w1` | fresh | 1.0 | does teacher-gauge-consistency alone suffice? |
| `v2kd-initA-w4` | member A | 4.0 | the never-swept knob (PHASE_F §11.3 used 1.0 twice) |

`--init-from` is added to `train.py` this commit: **weights only**, no
optimizer/step/RNG (unlike `--resume`), with a strict architecture check and
the source sha256 recorded into the checkpoint args.

**Decision rule, fixed now.** The single-seed gate is
**`v2kd-*` ≥ `L1 member A` s1234 on val t1 AND ≤3** (88.60 / 91.32). Any arm
that clears it goes to three seeds. **SUCCESS for the crown = a single model
meeting every number on the `mix2-i8f16` card** (88.68 / 92.61 / 93.46 /
91.30 / 87.32 + dvorak 91.94 / dv-app 91.53 / azerty 84.93 / qwertz 82.81 /
german 81.22 / spanish 89.59).

**Pre-stated expectation, recorded so it can be wrong** (the §14.1 forecast in
Phase L was wrong on its mechanism and that was worth more than being right):
**I expect E7 to beat member A on val t1 by a small margin and to FAIL the
crown on transfer** — the pair's dvorak/dvorak-app edge (+2…+3 over its own
members) is the part PHASE_L §11.1 attributes to two-point averaging, and a
single student has no second point to average with. I expect
`initA` > `fresh` if the gauge-matching argument is real.

## 2. Protocol carried

Blind order per pair (train → export → `pair_agreement.py` → commit gate and
band prediction → decode); both footings on every claim; sign-consistency for
promotion; negatives committed in place; retractions written where the
original claim lives. test-2400 sealed.

## 3. Stage-1 launch state (for a successor — the box reboots randomly)

Launched 2026-08-13, five arms, all detached, `--workers 0`, 188 k steps:
`v2pair-s5555`, `v2pair-s9999`, `v2kd-initA-w1`, `v2kd-fresh-w1`,
`v2kd-initA-w4`. **Resume = the identical launch command plus
`--resume ckpt/<run>/last.pt`** (pair arms use `train_v2.py`, KD arms
`train.py`); the exact argv of every arm is recorded in
`~/ctc-train/ckpt_<run>.launch.log` and in each checkpoint's `args`.
Harvest with `phaseL_eval.sh <run> {gate,pair,members}` for pairs; KD arms are
single models — export `best.pt` and run the val + layout battery
(`phaseJ_eval.sh <run>` does exactly that).

**`--init-from` verified at first eval** (the E7 correctness check): the
warm-started arms open at beam t1 **85.20 / 84.96** with CTC 0.617 / 0.601,
against the fresh arm's **80.80** and CTC 1.077 — i.e. the student really does
start inside member A's weights, and the KD arms are all training.

*(Stage-1 results land below. Stage 2 is pre-registered separately when slots
free.)*

## 4. STAGE 2 — pre-registration (committed BEFORE launch)

All four arms are the **L1 recipe at seed 1234**, one knob each, so every one
is paired against the already-decoded `v2pair-s1234` control.

| arm | knob | rationale |
|---|---|---|
| `v2pair-pw01` | `--pair-weight 0.1` | with pw 0.0 (`v2pair-pw0`), 0.3 (`v2pair-s1234`) and 1.0 already/also run, this completes a **four-point coupling-strength sweep at one seed** |
| `v2pair-pw10` | `--pair-weight 1.0` | the over-coupling end: does agreement saturate while member diversity (and the mix's transfer edge) collapses? |
| `v2pair-e4` | `--layout-synth-frac 0.615` | E4: `w_real` 0.217 → 0.25 at fixed `w_canon`, the proposal's §2.3-E4 arm |
| `v2pair-e6` | `--geo-align-weight 0.05` | E6: the geometric alignment prior, implemented in Phase L and never run |

**Decision rules, fixed now.**

* **pair-weight sweep** is a *measurement*, not a promotion candidate: report
  agreement, member solo scores and mix bars across {0, 0.1, 0.3, 1.0}, and
  apply the campaign's interior-optimum rule. If 0.3 is not interior-optimal
  the winner is registered for a three-seed test, not promoted on one seed.
  A specific pre-stated expectation: **agreement rises monotonically with the
  weight while the mix's transfer advantage falls at 1.0** (the
  coupling-vs-diversity trade PHASE_L §11.1 could not resolve).
* **E4** promotes only if the four euro axes gain with **no** val bar and no
  dvorak axis losing more than 0.15 — and then only to a registered
  three-seed test, since PHASE_L §15.5 just showed single-seed layout gains
  not reproducing.
* **E6 carries the proposal's own kill criterion verbatim: any val bar −0.15
  at one seed and it is dropped.** No second chance at another weight.


## 5. Stage-1 gates — measured and committed BEFORE decode

| pair | agreement | blank | letters | verdict |
|---|---|---|---|---|
| `v2pair-s5555` | **98.14 %** | 98.69 | 96.54 | PASS |
| `v2pair-s9999` | **98.25 %** | 98.79 | 96.60 | PASS |

**Eight of eight coupled pairs have now passed the gate** (98.05–98.33 %)
across five data seeds and two data mixes. **Committed prediction for both:
working band — val t1 ≥ 88.30 and ensemble greedy ≥ 55 %.** Decodes follow
this commit.

The three E7 students finished at 188 k with selection-prefix canonical t1
**86.38 (initA-w1) / 86.04 (fresh-w1) / 86.04 (initA-w4)** against member A's
**85.60** on the same 5,000-row prefix — so on the selection metric the
gauge-matched student is ahead of its own teacher's member. Full battery next.

## 6. STAGE 1 RESULTS

### 6.1 E7 — the crown attempt FAILS; and the gauge-matched *init* is unnecessary

Single seed (1234), full battery, fp32. Teacher = the s1234 pair (98.33 %).

| model | t1 | t3 | t5 | ≤3 | 4+ | dvorak | dv-app | azerty | qwertz | german | spanish | vs card | vs campaign |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| card | 88.68 | 92.61 | 93.46 | 91.30 | 87.32 | 91.94 | 91.53 | 84.93 | 82.81 | 81.22 | 89.59 | — | — |
| `memberA` (teacher's member) | 88.60 | 92.62 | 93.36 | 91.32 | 87.18 | 91.17 | 91.01 | 83.97 | 82.56 | 81.26 | 89.02 | 3/11 | 11/11 |
| **`v2kd-fresh-w1`** | **88.62** | 92.69 | **93.46** | 91.38 | 87.18 | **92.23** | **91.90** | 84.45 | **83.74** | 81.22 | 89.36 | **6/11** | **11/11** |
| `v2kd-initA-w1` | 88.59 | 92.72 | 93.43 | **91.41** | 87.12 | 91.86 | **91.53** | 84.21 | 83.24 | **81.63** | 88.17 | 5/11 | 10/11 |
| `v2kd-initA-w4` | 88.52 | 92.74 | **93.56** | 91.09 | 87.18 | **92.35** | **92.19** | 84.21 | 83.66 | 81.58 | 88.51 | 6/11 | 10/11 |

**CROWN: NOT WON.** No student meets every card number; the best is 6/11.
The residual gap is exactly where the proposal said it would be — **transfer**
(azerty −0.48, spanish −0.23) plus t1 −0.06.

**My committed forecast, scored — 1 right, 2 wrong:**

1. *"E7 beats member A on val t1 by a small margin"* — **WRONG.** It is a
   wash: 88.62 / 88.59 / 88.52 against 88.60. Distillation from the pair
   bought **nothing** on canonical val t1.
2. *"…and fails the crown on transfer"* — **RIGHT** on the outcome, but my
   mechanism was too strong. I argued a single student "has no second point
   to average with" and therefore could not inherit the pair's transfer edge.
   It inherited **most** of it: dvorak 91.17 → **92.23** (+1.06), dvorak-app
   +0.89, qwertz +1.18, azerty +0.48 over member A, at equal val. The pair's
   transfer advantage is **largely distillable**; it just stops ~0.5 short of
   the card on two axes.
3. *"initA > fresh if the gauge-matching argument is real"* — **WRONG, and
   this is the phase's most interesting negative.** The **fresh-init** student
   is the best of the three (11/11 campaign bars vs 10/11 for both warm
   starts). Gauge-matching the *student's initialization* is **unnecessary**;
   what matters is that the **teacher** is alignment-consistent. That is the
   cleaner form of the PHASE_L §1.3 corollary and it sharpens the Phase-G
   reinterpretation: the old KD refutation failed because its teacher was
   gauge-*inconsistent* (a cross-seed average), not because students need
   special initialization.

**Pre-registered gate applied as written** (`≥ member A on val t1 AND ≤3`):
`fresh-w1` **passes** (88.62 ≥ 88.60, 91.38 ≥ 91.32); `initA-w1` fails by
**0.01** on t1; `initA-w4` fails. → **`v2kd-fresh-w1` promoted to three
seeds** (`s4321`, `s7777` launched); the two `initA` arms are dropped, not
retuned.

### 6.2 Stage-1 pair predictions scored

| pair | agreement | prediction | outcome | verdict |
|---|---|---|---|---|
| `v2pair-s5555` | 98.14 % | working band | **88.65 / greedy ✓** | PASS |
| `v2pair-s9999` | 98.25 % | working band | **88.73 / greedy ✓** | PASS |

**8 of 8 gate predictions correct** across five seeds.

## 7. THE FIVE-SEED VERDICT — a retraction and a strengthened claim

### 7.1 ⚠ RETRACTION — PHASE_L §15.4's single-model all-eleven claim does NOT survive five seeds

The §1.1 rule fired. `L1 member A`, five seeds:

| bar | s1234 | s4321 | s7777 | s5555 | s9999 | **5-seed mean** | bar | Δ |
|---|---|---|---|---|---|---|---|---|
| t1 | 88.60 | 88.65 | 88.37 | 88.35 | 88.72 | 88.538 | 88.30 | +0.238 |
| **t3** | 92.62 | 92.64 | 92.54 | 92.51 | 92.57 | **92.576** | 92.60 | **−0.024 ✗** |
| t5 | 93.36 | 93.31 | 93.32 | 93.31 | 93.35 | 93.330 | 93.26 | +0.070 |
| ≤3 | 91.32 | 91.41 | 91.32 | 91.27 | 91.47 | **91.358** | 91.27 | +0.088 |
| 4+ | 87.18 | 87.21 | 86.84 | 86.84 | 87.29 | 87.072 | 86.77 | +0.302 |
| dvorak | 91.17 | 87.71 | 90.64 | 88.64 | 89.42 | 89.516 | 89.13 | +0.386 |
| dvorak-app | 91.01 | 87.51 | 90.03 | 88.07 | 88.85 | 89.092 | 88.20 | +0.892 |
| azerty | 83.97 | 83.59 | 83.78 | 83.73 | 83.64 | 83.742 | 83.60 | +0.142 |
| **qwertz** | 82.56 | 82.90 | 82.06 | 81.89 | 82.31 | **82.342** | 82.50 | **−0.158 ✗** |
| german | 81.26 | 79.99 | 80.90 | 80.04 | 80.35 | 80.509 | 79.64 | +0.869 |
| spanish | 89.02 | 88.62 | 87.66 | 88.05 | 88.11 | 88.294 | 88.28 | +0.014 |
| | | | | | **9/11** | | | |

**Both tie margins went under.** t3 (+0.000 at three seeds) is now **−0.024**;
qwertz (+0.007) is now **−0.158**. Per-seed tallies across five seeds are
**[11, 8, 8, 6, 8]** — the s1234 11/11 was the single-seed floor, exactly as
PHASE_K §8.3 warned and exactly as PHASE_L §15.4's own disclosure #2 flagged.

**RETRACTED, in place and everywhere it propagated:** "the campaign's first
single model to clear all eleven campaign bars on the seed-mean." The honest
replacement: **`L1 member A` is a 9/11 seed-mean single model** that clears
the ≤3 stratum (91.358, +0.088 — still a campaign first for that stratum on a
seed-mean) but misses t3 and qwertz. It therefore **does not supersede
`sw2345`** (10/11 seed-mean, missing ≤3): the two are non-dominating mirror
images, which is precisely the standing PHASE_K §9 described. The
single-model finalist question is **reopened**, pending the three-seed E7
result.

Retraction propagated this commit to `PHASE_L.md` §15.4, `RESULTS.md`,
`MODEL_COMPARISON.md` and `APP_INTEGRATION_PLAN.md` §9.

### 7.2 The PAIR claim strengthens at five seeds — 11/11 on EVERY SEED

The same two new seeds, evaluated as pairs:

| bar | 5-seed mean | campaign bar | Δ |
|---|---|---|---|
| t1 | **88.776** | 88.30 | +0.476 |
| t3 | **92.724** | 92.60 | +0.124 |
| t5 | **93.458** | 93.26 | +0.198 |
| **≤3** | **91.436** | 91.27 | **+0.166** |
| 4+ | **87.390** | 86.77 | +0.620 |
| dvorak | **91.339** | 89.13 | +2.209 |
| dvorak-app | **90.956** | 88.20 | +2.756 |
| azerty | **84.766** | 83.60 | +1.166 |
| qwertz | **84.128** | 82.50 | +1.628 |
| german | **81.764** | 79.64 | +2.124 |
| spanish | **89.078** | 88.28 | +0.798 |

**Per-seed tallies: [11, 11, 11, 11, 11] — all eleven campaign bars on all
five seeds**, with a seed-mean margin above +0.12 on every axis and above
+1.1 on five of six layouts. Nothing in Phases A–K approached this footing
(Phase J: 5/11 every-seed; Phase K's all-eleven was one configuration whose
recipe did not reproduce). Against the `mix2-i8f16` *card* the per-seed
tallies are [10, 8, 6, 4, 8] — bar 1 remains unmet and is now clearly unmeetable
by this recipe; the card's transfer numbers are a high-water single draw.

**The durable Phase L/M result is therefore about the pair, not the single
model:** coupled-pair training turns a lucky configuration into a recipe that
clears every campaign bar on every seed tested (5/5), at 4.39 MB.
Member B is 10/11 at the five-seed mean (per-seed [9, 10, 8, 8, 10]).
