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
| **qwertz** | 82.56 | 82.90 | 82.06 | 81.89 | 82.31 | **82.344** | 82.50 | **−0.156 ✗** |
| german | 81.26 | 79.99 | 80.90 | 80.04 | 80.35 | 80.509 | 79.64 | +0.869 |
| spanish | 89.02 | 88.62 | 87.66 | 88.05 | 88.11 | 88.294 | 88.28 | +0.014 |
| | | | | | **9/11** | | | |

**Both tie margins went under.** t3 (+0.000 at three seeds) is now **−0.024**;
qwertz (+0.007) is now **−0.156**. Per-seed tallies across five seeds are
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

## 8. STATE FOR A SUCCESSOR (written 2026-08-13, mid-phase)

**Six arms in flight**, all detached, `--workers 0`, 188 k, resume = identical
argv (in `~/ctc-train/ckpt_<run>.launch.log`) + `--resume ckpt/<run>/last.pt`:

| run | what | trainer | harvest |
|---|---|---|---|
| `v2kd-fresh-w1-s4321` | E7 3-seed | `train.py` | `~/ctc-train/phaseJ_eval.sh <run>` |
| `v2kd-fresh-w1-s7777` | E7 3-seed | `train.py` | same |
| `v2pair-pw01` | coupling 0.1 | `train_v2.py` | `phaseL_eval.sh <run> {gate,pair,members}` |
| `v2pair-pw10` | coupling 1.0 | `train_v2.py` | same |
| `v2pair-e4` | `w_real` 0.25 | `train_v2.py` | same |
| `v2pair-e6` | geo prior 0.05 | `train_v2.py` | same |

**Decision rules already committed** — §1.2 (E7 crown = every card number),
§4 (sweep is a measurement + interior-optimum rule; E4 needs euro gains with
no val/dvorak loss > 0.15 → 3 seeds; **E6 dies on any val bar −0.15**).

**What is settled and must not be re-litigated:** E1 confirmed (PHASE_L §11.1);
E2 refuted at 3 paired seeds (§15.5); the single-model all-eleven claim
**retracted** (§7.1); the pair at **11/11 × 5 seeds** (§7.2); E7 crown **not
won** (§6.1) with `fresh` ≥ `initA` — teacher gauge-consistency matters,
student init does not. Gate predictions stand at **8/8**.

**Ship candidate unchanged:** `v2pair-s1234` int8w+fp16w, 4.39 MB, artifacts
and sha256s in `PHASE_L.md` §16.1. **test-2400 SEALED — the final unsealing is
the orchestrator's act, not the agent's.**

## 9. STAGE-1b FINAL — the E7 student at three seeds WINS the single-model question

`v2kd-fresh-w1`, three seeds, full battery, fp32:

| bar | s1234 | s4321 | s7777 | **3-seed mean** | campaign bar | Δ |
|---|---|---|---|---|---|---|
| t1 | 88.62 | 88.88 | 88.75 | **88.750** | 88.30 | +0.450 |
| t3 | 92.69 | 92.80 | 92.83 | **92.773** | 92.60 | +0.173 |
| t5 | 93.46 | 93.45 | 93.51 | **93.473** | 93.26 | +0.213 |
| **≤3** | 91.38 | 91.44 | 91.30 | **91.373** | 91.27 | **+0.103** |
| 4+ | 87.18 | 87.55 | 87.43 | **87.387** | 86.77 | +0.617 |
| dvorak | 92.23 | 92.14 | 91.09 | **91.819** | 89.13 | +2.689 |
| dvorak-app | 91.90 | 91.45 | 89.95 | **91.100** | 88.20 | +2.900 |
| azerty | 84.45 | 84.69 | 84.45 | **84.530** | 83.60 | +0.930 |
| qwertz | 83.74 | 83.91 | 84.25 | **83.965** | 82.50 | +1.465 |
| german | 81.22 | 81.45 | 81.22 | **81.295** | 79.64 | +1.655 |
| spanish | 89.36 | 89.87 | 89.36 | **89.534** | 88.28 | +1.254 |
| | | | | **11/11** | | |

**Per-seed tallies: [11, 11, 11] — all eleven campaign bars on EVERY seed**,
seed-mean margins **+0.10 … +2.90**, smallest margin (≤3, +0.103) an order of
magnitude clear of the ties that sank the member-A claim.

**This is the single-model result the campaign was after, and it is robust
where the Phase-L one was luck.** It supersedes `sw2345` (10/11 seed-mean,
≤3 −0.07) as the **single-model finalist**, on both footings, at 1.5 M
parameters. Note what produced it: **distillation from an
alignment-consistent teacher** — the coupled pair. The pair is not just a
shipping configuration, it is a *training instrument* for single models.

**Crown, scored against the pre-stated definition:** SUCCESS required meeting
**every** `mix2-i8f16` card number. At the 3-seed mean the student beats **all
five val numbers** of the card (t1 88.750 > 88.68, t3, t5, ≤3 91.373 > 91.30,
4+) but misses four transfer axes (dvorak −0.12, dvorak-app −0.43, azerty
−0.40, spanish −0.06). **Crown NOT won** — per-seed card tallies [6, 8, 6].
**My §1.2 forecast — "beats member A on val, fails the crown on transfer" —
is confirmed at three seeds** on the transfer half; the val half was a wash
against member A at one seed but is a clear +0.21 at three.

## 10. STAGE-2 RESULTS — the coupling optimum is interior; E4 and E6 both die by their own rules

### 10.1 Coupling-weight sweep (four points, seed 1234, pair)

| `--pair-weight` | agreement | mix greedy | t1 | ≤3 | dvorak | azerty | campaign | card |
|---|---|---|---|---|---|---|---|---|
| 0.0 (control) | 92.09 % | **53.12** | 88.09 | 91.12 | 89.34 | 84.35 | 7/11 | 1/11 |
| 0.1 | 98.08 % | 73.72 | 88.84 | 91.44 | 92.96 | 84.07 | 11/11 | 9/11 |
| **0.3 (finalist)** | 98.33 % | 72.92 | **88.90** | **91.53** | **93.04** | 84.16 | **11/11** | **10/11** |
| 1.0 | **98.58 %** | 73.22 | 88.85 | 91.47 | 91.09 | **84.78** | 11/11 | 8/11 |

**My §4 pre-stated expectation is CONFIRMED on both halves:** agreement rises
**monotonically** with the weight (92.09 → 98.08 → 98.33 → 98.58) *and* the
mix's transfer advantage **falls at 1.0** (dvorak 93.04 → 91.09, −1.95) as
over-coupling collapses the member diversity that averaging feeds on. The
coupling-vs-diversity trade PHASE_L §11.1 could not resolve is now measured.
**0.3 is the interior optimum** under the campaign's interior-optimum rule —
the knob was already at its best value and the sweep closes it. No promotion,
by design.

### 10.2 E4 (`w_real` 0.217 → 0.25) — DROPPED

Rule: euro gains with **no** val bar and no dvorak axis losing more than 0.15.
Measured Δ vs the s1234 control: **dvorak −2.81, dvorak-app −2.85**,
qwertz −1.26, german −1.14, spanish −1.37, azerty +0.62, 4+ −0.15.
The rule is violated by nearly 20×, and only one of four euro axes even gains.
**Dropped.** Raising real-layout exposure at fixed `w_canon` buys one euro
axis and pays for it everywhere, dvorak worst — the mixture trade PHASE_J
§5.1b described, reproduced at ch 192.

### 10.3 E6 (geometric alignment prior, weight 0.05) — DROPPED by its kill criterion

The proposal's kill criterion: **any val bar −0.15 at one seed.** Measured Δ:
**t1 −0.21, t5 −0.16, ≤3 −0.18, 4+ −0.23** — four val bars past the
threshold, three of them past double it. **Dropped, and per the
pre-registration not retried at another weight.** The geometric prior's
attraction was that it would pin the gauge globally; E1 pins it without a
prior and without a val bill, so the motivation is gone as well as the
result. (Both E4 and E6 pairs still cleared 11/11 campaign bars — the coupled
recipe is robust enough to absorb two bad knobs, which is itself informative.)

## 11. PHASE M CLOSE — the ship menu and the final verdict

### 11.1 Packaging the promoted single model

`v2kd-fresh-w1` (s1234) at **fp16w, 3,052,318 B (2.91 MB)**: val
88.62 / 92.69 / 93.46 / 91.38 / 87.18, dvorak 92.19, azerty 84.45,
qwertz 83.74, german 81.17, spanish 89.36 — **free vs fp32** (largest delta
0.05). int8w is not retested here: PHASE_L §16 measured it costing a single
model the ≤3 bar, and this model's ≤3 margin (+0.103 seed-mean) has no room.

| file | bytes | sha256 |
|---|---|---|
| `phaseM_kd_fresh_w1_s1234_fp16w.onnx` | 3,052,318 | `84718e6e…549e88e5` |
| ~~`phaseM_kd_fresh_w1_fp16w_golden.json` (at **E1**)~~ | ~~140,480~~ | ~~`3788697f…495f058c`~~ |
| **`phaseM_kd_fresh_w1_fp16w_golden.json`** (regenerated 2026-08-14 at the **ship preset** `0.9,4.0,0.25,0.25,0.9882`) | **140,462** | **`2a449c4f2de19505131b396655ae01d3e3c325e40249446ff6e7a40c2b27559c`** |

**Fixture correction.** The first fixture was generated at **E1**, the
benchmark preset. `MODEL_COMPARISON.md` §5.1 requires the fixture to be
generated at the preset the app *ships*, which for this model is the app preset
(now test-validated as config B of the fourth unsealing, §12). It was
regenerated at that preset; the E1 row above is struck, not deleted. The
regeneration changes no measurement — only which configuration `CtcParityTest`
asserts against.

The three fp32 seed exports are also frozen in `artifacts/` (they are what the
fourth unsealing decoded), 6,068,519 B each:
`phaseM_kd_fresh_w1_s1234.onnx` `b71911da…`, `_s4321.onnx` `f7cb72c0…`,
`_s7777.onnx` `c55cc3b0…`.

### 11.2 THE SHIP MENU (all evidence, all footings)

| option | size | campaign bars | vs `mix2-i8f16` card | footing |
|---|---|---|---|---|
| **A. Coupled pair** `v2pair-s1234` int8w+fp16w | **4.39 MB** | **11/11 on 5 of 5 seeds** (5-seed mean +0.12…+2.76) | 10/11 at s1234, 7/11 seed-mean | strongest accuracy; two ONNX sessions, 1.79 ms |
| **B. Distilled single** `phaseM_kd_fresh_w1_s1234_fp16w` | **2.91 MB** | **11/11 on 3 of 3 seeds** (mean +0.10…+2.90) | 7/11 seed-mean (beats all 5 **val** numbers, misses 4 transfer) | one session, one graph; the campaign's first robust all-eleven single model |
| C. incumbent `mix2-i8f16` | 4.45 MB | 11/11 as **one configuration** | — (it is the card) | recipe did **not** reproduce (PHASE_K §8.5) |
| D. previous single finalist `sw2345` | 6.07 MB fp32 | 10/11 seed-mean (≤3 −0.07) | — | superseded by B |

**Recommendation.** **Ship B, the 2.91 MB distilled single model**, unless the
app wants the last few tenths and can afford two sessions, in which case
**A**. The reasoning, stated with its weaknesses: B is smaller than every
option, is a single graph on the frozen `[1,32,65]` contract with **no app
code change at all**, and is the only single model in the campaign's history
to clear all eleven bars **on every seed tested** — the property member A
failed to have. A is more accurate in absolute terms (s1234 t1 88.90 vs
88.62) and has the deeper seed evidence (5 seeds vs 3), so an accuracy-first
call picks A. **C is superseded on footing, not on numbers**: its card is a
high-water single draw whose recipe demonstrably does not reproduce, while A
and B reproduce by construction.

**Neither A nor B beats the C card on every axis** — both miss on transfer
(dvorak/dvorak-app/azerty/spanish, 0.06–0.43 at the seed-mean). That is the
honest residual, and it is why bar 1 and the crown are both recorded as
**not met** rather than argued around.

### 11.3 Every pre-registered bar and rule in Phases L+M, scored

| item | rule | outcome |
|---|---|---|
| Bar 1 (pair ≥ card, 2/3 seeds) | primary | **NOT MET** (per-seed 10/8/6/4/8 across five seeds) |
| Bar 2 (single ≥ 11 campaign bars, seed-mean) | secondary | **MET at 3 seeds → RETRACTED at 5** for member A (§7.1); **MET and robust** for the E7 student (§9) |
| Bar 3 / crown (single beats full card) | stretch | **NOT WON** (misses 4 transfer axes) |
| E1 coupling | attribution control | **CONFIRMED** (PHASE_L §11.1) |
| E2 synthesis | val gate | **REFUTED** at 3 paired seeds |
| E4 `w_real` | euro gain, no loss > 0.15 | **DROPPED** (dvorak −2.81) |
| E6 geo prior | any val bar −0.15 → die | **DROPPED** (four val bars past it) |
| coupling sweep | interior-optimum rule | **0.3 confirmed interior-optimal**; knob closed |
| E7 | ≥ member A on t1 and ≤3 → 3 seeds | gate passed by `fresh`, failed by both `initA`; 3 seeds → **11/11 every seed** |
| gate band predictions | PHASE_K §8.5 bands | **12 of 12 correct** above 98 % agreement across the two phases |

### 11.4 THE LEDGER IS EMPTY

Every item registered-not-run at the end of Phase L has been run and reported:
the L1 three-seed stage (→ five seeds), E2 at three seeds, the `--pair-weight`
sweep, E4, E6, E7 (single-seed gate → three seeds), and the extra seeds for
the tie margins. **Nothing is outstanding, nothing was quietly dropped, and no
element was retried after failing its rule.**

The one thing this campaign cannot self-certify: **test-2400 has never been
opened in Phases L or M** (ledger stays at 3 entries; `train_v2.py`,
`english_synth.py` and `pair_agreement.py` refuse test features by name). The
final unsealing pre-registration and the independent audit are the
orchestrator's acts, not the agent's.

## 12. The fourth unsealing — pre-registered in `UNSEALING_4.md`

The orchestrator's act, taken on the user's directive of 2026-08-13/14 (one
final pre-registered unsealing plus an adversarial audit, for whichever model
ships) and on §11.2's recommendation of **option B**, the distilled single
model. Subject: `v2kd-fresh-w1` at seeds 1234 / 4321 / 7777, two footings,
**six decodes, hard-capped**. Option A (the coupled pair) is **not** decoded
and stays val-only.

The pre-registration — authority, exact configs, frozen artifact sha256s,
numeric expectations with bands, the hard cap and the claim rules — is
`UNSEALING_4.md` §1–§7, **committed and pushed at `b91f179` before any
decode**. Results and the scored expectations are `UNSEALING_4.md` §8. Ledger
3 → 4; there is no fifth.

### 12.1 Outcome — the ship candidate is test-validated on both footings

Six decodes, one per (config, seed), no retries, no crash.

| footing | seed-mean | bar | Δ | every seed? |
|---|---|---|---|---|
| **A** — AOSP STRIP 146,964, E1 | **88.931 / 92.681 / 93.361 / 92.597 / 87.045** | published `84.83/91.04/92.08/89.57/82.40` | **+4.10 / +1.64 / +1.28 / +3.03 / +4.64** | **yes, all five** |
| **B** — app trie 98,081, shipping app preset | **89.306 / 93.792 / 94.500 / 93.701 / 87.045** | trie-matched `84.92/91.54/92.96/89.57/82.52` | **+4.39 / +2.25 / +1.54 / +4.13 / +4.53** | **yes, all five**; worst-seed t5 **+1.50** |
| **A vs equal footing** | same as A | val-tuned `87.12/92.29/92.96/89.94/85.68` | **+1.81 / +0.39 / +0.40 / +2.66 / +1.36** | **yes, all five**; McNemar **3/3, p < 0.001** |

**`phaseM_kd_fresh_w1` is TEST-VALIDATED**, and holds a **qualified
equal-footing win** — the second in the campaign after ch 192, resolved on 3 of
3 seeds instead of 2, at 2.91 MB instead of 6.14 MB.

Pre-registered expectations: **7 of 7 verdicts right**, band coverage **9 of
10**. The single band miss is config-A ≤3 (92.597 against a band top of 92.593,
an overshoot of 0.004 pt — a thirtieth of one row), and it is recorded as a
miss. §5.3's registered guess that ≤3 would be the one to miss was **right**:
its val→test shift is **+1.22 (A) / +1.14 (B)**, the largest ever measured on
this split, against ±0.35 for every other metric here.

Two limitations that must travel with the numbers: the equal-footing lead is
bought **entirely on the HWS corpus half** (FUTO's engine is +0.38 ahead on its
own half — `UNSEALING_4.md` §8.4), and **ch 192 keeps t5 by 0.14**. The pair
(option A) was **not** decoded and stays val-only, permanently.

§11.2's recommendation stands, now on test evidence: **ship B.**
