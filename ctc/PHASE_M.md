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
