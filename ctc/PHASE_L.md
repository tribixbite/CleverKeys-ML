# Phase L — pipeline v2 execution (coupled pairs)

**Date opened:** 2026-08-12 · **Authority:** orchestrator directive of
2026-08-12 ("execute the v2 pipeline the campaign architect proposed").
**Plan of record:** `PIPELINE_V2_PROPOSAL.md` — its experiment elements, bars
and protocol are the authority for this phase. **test-2400 stays SEALED**
(ledger stays at 3 entries; unsealing decisions are not this phase's).

## 0. Bars (verbatim from the proposal §2.6, which quotes PHASE_K §8.2)

**Primary — pair bar.** The S3/L3 gated pair meets or beats every number on
the `mix2-i8f16` card, on **≥ 2 of 3 seeds, every bar**, at ≤ 5 MB, with the
≤3 margin > +0.10:

| axis | mix2-i8f16 bar |
|---|---|
| val t1 | 88.68 |
| val t3 | 92.61 |
| val t5 | 93.46 |
| val ≤3 | 91.30 |
| val 4+ | 87.32 |
| dvorak | 91.94 |
| dvorak app-98k | 91.53 |
| azerty | 84.93 |
| qwertz | 82.81 |
| german | 81.22 |
| spanish | 89.59 |
| size | ≤ 5 MB |

**Secondary — single-model bar.** One member clears all **eleven campaign
bars** seed-mean: 88.30 / 92.60 / 93.26 / **91.27** / 86.77 · dvorak 89.13 ·
dvorak-app 88.20 · azerty 83.60 · qwertz 82.50 · german 79.64 · spanish 88.28.

**Stretch (judged unlikely by the proposal).** A single model beating the full
`mix2-i8f16` card; admissible route is E7 (pair → single distillation), run
only if the two bars above fall.

Both footings (seed-mean AND every-seed) are reported for every claim.

## 1. Protocol carried from the proposal §2.7

Three seeds for any promoted claim; sign-consistency for promotion; symmetric
application of decode-side changes to the incumbent; interior-optimum rule for
sweeps; per-trace dumps kept; **pre-registration committed before decodes**;
`--workers 0`; all trainings detached (`nohup setsid`); commit at every
milestone (the box reboots randomly — a successor resumes from these docs +
`last.pt` under identical args); test-2400 sealed.

**Evidence tiers used in this document:** *measured* (a number produced by a
run in this phase, log path given), *audited* (a number quoted from a closed
phase), *expected* (a pre-stated prediction, never a result). Negatives are
committed in place; nothing is silently retracted.

## 2. L0 — `train_v2.py` verification and smoke (DONE, 2026-08-12)

### 2.1 Static verification against the proposal and against `train.py`

Read line-by-line against `PIPELINE_V2_PROPOSAL.md` §2.3 and against the live
`train.py`/`model.py`/`export_onnx.py` APIs. Findings:

* **Formulas match the proposal exactly** — `L_ctc^m` is bit-identical to
  `train.py`'s `--short-loss-weight` branch (per-sample length-normalized,
  weighted mean, `w=1` ≡ stock reduction); `pair_kl` is the symmetric
  stop-grad KL in nats/frame normalized by `B·T′`; `λ_pair(step)` is the
  specified clip ramp; `geo_align_penalty` is the §2.3-E6 formula with blank
  exempt and pad slots masked.
* **No NaN hazard in the KL over pad columns**: `model.py` masks pad key
  slots with a *finite* `MASK_NEG = -1e4`, so `exp(target)·(target − input)`
  is `0·0`, not `0·(−inf)`. Confirmed empirically (no NaN in the smoke).
* **Fork-before-CUDA ordering is correct**: `BeamValidator` (which forks its
  beam pool) is constructed before the first `.to(device)`.
* **Checkpoints are toolchain-compatible**: member payloads carry every field
  `model.encoder_from_checkpoint` needs; `export_onnx.py` round-trips them
  (measured below).
* **No code changes were required before the first launch.**

### 2.2 Deviations from the proposal's stage list, declared up front

1. **Order.** The proposal's S1 (five *single-model* ablations, ~12 GPU-h)
   runs before its S2 pair arms. This phase runs the **coupled pair first**
   (L1), per the orchestrator's premise-first instruction and because a pair
   arm answers the primary bar directly while the S1 riders answer secondary
   ones. Element ablations that survive budget are folded in as *pair-level*
   paired comparisons (L2), which is the footing the product actually ships
   on.
2. **E4 (explicit mixture, `w_real` 0.25) is NOT in L1.** L1 keeps the
   audited finalist augmentation (`--layout-alt-p 0.65
   --layout-synth-frac 0.667`) so member A stays directly comparable to the
   audited `sw2345_s1234` single. E4 is a registered rider, not a default.
3. **E5 (layout-probe selection) IS on in L1**, per the proposal ("the right
   default for v2"). Attribution caveat recorded: L1's members therefore
   differ from the audited singles in *two* ways (coupling + selection
   metric), and the audited E5 price is −0.04 canonical for +0.5 probe mean
   (PHASE_I §6.2).
4. **`--val-every 4000`** (audited runs used 3000) — a selection *cadence*,
   traded for ~40 min of eval overhead across a 188 k pair run.

### 2.3 Smoke (measured) — the mechanism engages and nothing is destroyed

800 steps, `tier_sw5q.npz` (24,707 rows), ch 192, batch 256, ramp 100+100,
`--beam-val-rows 500`, probes `synth:101,azerty` (300 rows each),
`--workers 0`. Logs: `~/ctc-train/v2smoke.log`, `v2smoke_resume.log`,
`v2smoke0.log`.

| step | pw 0.3: agree | pw 0.3: ctc A/B | pw 0.0: agree | pw 0.0: ctc A/B |
|---|---|---|---|---|
| 200 | 89.4 % | 1.4091 / 1.4358 | 82.0 % | 1.4113 / 1.4357 |
| 400 | **95.5 %** | 1.1665 / 1.1983 | 86.0 % | 1.1903 / 1.2013 |
| 600 | 96.3 % | 1.1323 / 1.1455 | 90.7 % | 1.0774 / 1.1071 |
| 800 | **96.5 %** | 1.0204 / 1.0487 | 91.4 % | 1.0447 / 1.0597 |

The two runs share the data/augmentation seed and the member init seeds, so
this is a **paired** comparison; the only difference is `--pair-weight`.

* **The coupling does what it is for.** Per-frame argmax agreement runs
  +5.1…+9.5 pt above the uncoupled control at every matched step, and the
  coupled pair crosses the 0.95 gate by step 400 while the uncoupled pair
  never reaches it. `pair_score` is correctly `−1.00` (gate-rejected) at
  89.4 % and 82.0…91.4 %, and a real score once the gate passes.
* **The KL does not destroy the members** (the orchestrator's stop-early
  criterion is not triggered): CTC losses are equal-or-lower under coupling
  at 200/400/800 steps, and member beam scores at step 800 are 72.60/69.00
  (coupled) vs 68.00/67.80 (uncoupled) on a 500-row prefix — noisy at this
  scale, but nowhere near a collapse.
* **KL magnitude falls as agreement rises** (0.1919 → 0.0422 nats/frame),
  i.e. the term is being satisfied, not fought.
* **Checkpoint/resume verified**: killed and relaunched with identical args
  plus `--resume`; it reported `epoch 5, step 400, best {'a': 55.0, ...}`
  and continued, RNG restored.
* **Export/eval toolchain verified**: both member checkpoints export
  (`max |onnx − torch| = 1.53e-05`, argmax 100/100, BN folded), and the
  ensemble decode path runs on them —
  `eval_beam.py --ens-avg prob --limit 200` gives t1 79.50 / greedy 37.50
  on an 800-step toy pair (a gauge-incompatible mix reads greedy 9–20 %,
  PHASE_K §4.3, so the mix is healthy).

**L0 verdict: PASS.** The coupling premise is not refuted at smoke scale; the
implementation is faithful to the proposal; the trainer is safe to run
unattended.

## 3. L1 — the reference coupled pair (seed 1234)

Pre-registered here **before launch**. Arm `v2pair-s1234`:

```
python3 train_v2.py --run-name v2pair-s1234 \
  --train-npz train_t3.npz,train_t3hws.npz,train_t3hws.npz,tier_sw234.npz,tier_sw5q.npz \
  --ch 192 --block resbn --dilations 1,2,4,8 --t-out 32 \
  --layout-alt-p 0.65 --layout-synth-frac 0.6666666666666666 \
  --seed 1234 --init-seed-a 1111 --init-seed-b 2222 \
  --slw-a 1.0 --slw-b 1.5 \
  --pair-weight 0.3 --pair-ramp-start 5000 --pair-ramp-len 15000 \
  --total-steps 188000 --val-every 4000 --patience 40 \
  --beam-val-rows 5000 --beam-jobs 8 \
  --select-layout-probes synth:101,synth:202,azerty \
  --select-layout-rows 2000 --select-layout-weight 1.0 \
  --workers 0
```

**Pre-stated reading rules (fixed before any result):**

* **Coupling health gate (label-free, decided before any beam decode of the
  mix):** final per-frame argmax agreement ≥ 0.95 → the pair is mixable and
  the mix gets the full battery. < 0.95 → the coupling failed to pin the
  gauge at scale; report it as a negative and do not spend L3 on it.
* **Member-damage kill criterion:** if either member's canonical val t1 lands
  below 87.8 (i.e. > 0.5 below the 88.30 single-model bar, outside the
  audited 0.15–0.7 per-metric run-to-run sd), the KL is judged to be costing
  the members and `--pair-weight` drops to 0.1 for L2 rather than the E2 arm
  being added.
* **What L1 can and cannot settle:** L1 is one seed and one configuration. It
  can refute the premise, and it can show the pair reaching the card; it
  cannot establish the 2-of-3-seed primary bar — that is L3's job.

**Launched** 2026-08-12 ~21:31 (detached, `--workers 0`); log
`~/ctc-train/ckpt_v2pair-s1234.launch.log`, metrics
`~/ctc-train/ckpt/v2pair-s1234/metrics.jsonl`. Measured throughput at the
first eval: **11.8 steps/s** (4,000 steps in 339 s) ⇒ ~4.4 h of stepping plus
~47 × ~110 s of paired beam validation ≈ **5.9 h** — inside the proposal's
5 h/pair estimate class.

*(results land below when the run finishes)*

## 4. S0 — targeted English synthesis (`english_synth.py`, DONE 2026-08-12)

New generator committed this phase: `english_synth.py`, the proposal's E2
delta on `cyrillic_synth.py`. It reuses `build_donor_index`/`collapse` from
that module and `layout_aug.warp_path` verbatim; src and dst geometry are both
canonical QWERTY, so only the *word* changes and every Phase-H exactness
invariant carries. Donors are TRAIN caches only (the script refuses
`val.npz`/`test.npz` by name); the pools are training data exclusively —
selection and evaluation stay 100 % real.

| pool | rows | target words | unique words drawn | length profile |
|---|---|---|---|---|
| `cache/synth_en_short.npz` | 150,000 | 8,126 lexicon words of len ≤ 4 (3,920 of which have **no** real trace) | 7,919 | 1–4 (64 % len-4, 32 % len-3) |
| `cache/synth_en_tail.npz` | 150,000 | 121,499 lexicon words with < 3 real train traces (98,903 with **none**) | 74,314 | 2–23, mode 8 |

sha256: `78e0984e…8102d1` (short, 66,284,178 B), `92b89a56…c403b11b`
(tail, 68,360,181 B). Generation 3,181 rows/s (short) and 1,198 rows/s (tail),
`no_donor = 0` in both — the 1.21 M-trace donor pool covers every needed
vertex count (one 28-vertex lexicon word excepted, and it is never drawable).
Synthetic fraction of the v2 mix: 300,000 / 1,585,381 = **18.9 %**, under the
proposal's 25 % cap.

**Note for interpretation, stated before any training:** the two pools attack
*different* bars. `short` is the ≤3 lever (its draws are 1–4-letter words);
`tail` is overwhelmingly 5–12 letters, so it is a 4+/t1-and-lexicon-coverage
lever. Any ≤3 movement from `tail` would be incidental.

### 4.1 A gate was revised before use — recorded, not silent

The generator's first-written acceptance gate was "synthetic endpoint hit
rates within ±0.10 of the published real-en band (start 0.895 / end 0.769,
PHASE_H §2.3)". On the first 2,000-row dry run it **FAILED**: start-hit 0.756
(−0.139), end-hit 0.664 (−0.105). Two measurements taken *before* deciding
what to do about it:

1. **The published band is a corpus average over pools that differ
   materially.** Measured through the same code path: `train_t3` 0.948/0.833,
   `tier_sw234` 0.917/0.763, `tier_sw5q` 0.871/0.692. The honest comparator
   for a transplant is the donor pool the residuals came from (0.915/0.766),
   not a constant.
2. **The audited precedent is looser than the gate was.** The ru generator —
   the campaign's *proven* synthesis win (0 → ≈77.4) — scored synth
   0.7095/0.656 against its real corpus's 0.917/0.6465, i.e. a −0.21 start-hit
   gap, larger than the en generator's.

Diagnosis of *why* the hit rate lags while the distance does not: the
synthetic mean endpoint **distance** to the intended key matches the real pool
almost exactly (0.0583 vs 0.0589 start; 0.0779 vs 0.0792 end — a −0.0006 /
−0.0013 delta). Only the direction-sensitive nearest-key statistic lags,
because a residual of realistic magnitude, transplanted into a *different*
key neighbourhood, crosses a Voronoi boundary more often. The magnitude of
human motor deviation transfers; its local-neighbourhood direction does not.

The gate was therefore **replaced, and the replacement is stated in the code**
(`gate_report`): (a) displacement magnitude within 0.02 of the measured donor
pool; (b) wrong-geometry falsification ≥ 0.30 start-hit drop; (c) hit-rate gap
no worse than the ru precedent. **Precision about (c):** the 0.21 *start*-hit
ceiling is the audited ru number; the 0.15 *end*-hit ceiling is **not** a
precedent — the ru generator's end-hit slightly exceeded its real corpus
(0.656 vs 0.6465), so nothing measured bounds that axis and 0.15 is a chosen
tolerance of the same order. This matters concretely: `synth_en_tail`'s
end-hit gap is 0.126, so it passes the chosen tolerance and would *fail* a
"no worse than ru" reading of the end axis. Recorded rather than smoothed
over. This is a loosening, and it is disclosed as
one; the mitigating facts are that the original threshold was written before
either measurement above, that (a) is a *stricter* test of the thing that
matters (is the deviation human-sized?), and that **the decisive gate is not
any of these three** — it is the E2 on/off training ablation, which is what
the proposal always specified.

### 4.2 Measured gates (150 k rows each, 5,000-row validation sample)

| quantity | `short` | `tail` | real donor pool |
|---|---|---|---|
| start-hit | 0.761 | 0.773 | 0.915 / 0.916 |
| end-hit | 0.668 | 0.637 | 0.766 / 0.763 |
| start-distance | 0.0583 | 0.0587 | 0.0589 / 0.0586 |
| end-distance | 0.0779 | 0.0838 | 0.0792 / 0.0789 |
| **wrong-geometry (dvorak) start-hit** | **0.040** | **0.038** | — |
| gate (a) displacement magnitude | PASS | PASS | — |
| gate (b) wrong-geometry (Δ ≥ 0.30) | PASS (0.721) | PASS (0.735) | — |
| gate (c) ru precedent | PASS (0.154) | PASS (0.143) | — |

The falsification control is decisive: the same traces scored against dvorak
key centers hit 4 % of the time. The traces are geometrically anchored to
QWERTY, which is the only thing the endpoint check can establish.

## 5. PRE-REGISTRATION — two concurrent arms (committed BEFORE launch)

**Why concurrent.** Measured on this box while L1 runs: GPU utilization
**14–23 %**, 828 MiB of 16 GB, load 1.8 of 24 cores, 55 GB RAM free. The
coupled-pair loop is **CPU-bound in its single `--workers 0` feeder**, not
GPU-bound, so additional arms cost wall-clock only if they contend for cores.
Three arms need 3 feeder cores plus staggered beam pools. The Phase-K deadlock
(§4.5 there) was a persistent-worker failure; `--workers 0` from launch is the
standing mitigation and is used here. If L1's steps/s degrades more than 20 %
after a co-launch, the co-launched arm is killed and re-run serially — stated
now so the decision is not made after seeing results.

**L2 — `v2pair-e2-s1234`: L1 + the E2 pools, nothing else changed.** Identical
args to L1 except `--train-npz` gains `synth_en_short.npz,synth_en_tail.npz`.
Same data seed, same init seeds, same slw/pair weights, same selection. This
is the proposal's E2 ablation lifted to the pair level (the footing the
product ships on), and being paired with L1 it is a clean A/B of E2.

**L3 — `v2pair-pw0-s1234`: L1 with `--pair-weight 0`, the attribution
control.** Two differently-initialized members trained on *identical batches*
with no mutual KL. This is the arm that decides whether the coupling term —
rather than merely sharing a batch stream — is what pins the gauge. It is the
control the primary claim needs and it did not exist in the proposal's stage
list; it is added here because concurrency made it free in wall-clock.

**Pre-stated readings (fixed before any of the three finishes):**

* **E1 attribution (L1 vs L3).** If L3's final per-frame agreement is ≥ 0.95
  and its gated mix matches L1's, then the KL is *not* the active ingredient
  and E1 must be reported as "same-batch training suffices" — a negative for
  the proposal's central mechanism claim, to be written up as such.
  If L3 lands below the gate while L1 clears it, E1 is confirmed as the
  mechanism at scale.
* **E2 verdict (L1 vs L2).** Promotion requires L2 ≥ L1 on the ≤3 bar with no
  other val bar losing more than 0.15, at one seed, and is confirmed only at
  L4's three seeds. The `short` pool is the ≤3 lever; the `tail` pool is a
  4+/t1 lever (§4). A wash is a wash and will be recorded as one.
* **Seed footing.** All three arms are seed 1234. Nothing from this round is
  promotable on its own; the primary bar needs the three-seed stage.

## 6. PRE-REGISTRATION — the three-seed stage, launched speculatively

Measured: L1 runs at **18.4 steps/s with three arms up** (4,000 steps in 217 s
at step 12 k, vs 11.8 steps/s when it was alone and paying full eval cost),
i.e. the co-launch did not degrade it. Two further arms are therefore started
**before** the s1234 verdict, to buy wall-clock:

| arm | recipe | `--seed` | `--init-seed-a/b` |
|---|---|---|---|
| `v2pair-e2-s4321` | the L2 (E2) recipe | 4321 | 3333 / 4444 |
| `v2pair-e2-s7777` | the L2 (E2) recipe | 7777 | 5555 / 6666 |

**Why the E2 recipe and not L1's.** The proposal's §2.2 data mix *includes*
the synthesis pools, and its failure rule is "any element that misses its
ablation gate is dropped" — so the E2 recipe is the presumptive v2
configuration and L1 is its ablation, not the other way round. Stated as a
falsifiable commitment: **if E2 misses its gate at s1234 (§5), these two arms
become the recorded negative evidence for E2 at three seeds, and the
three-seed stage is re-run on the L1 recipe.** That is a real cost of being
wrong and it is accepted here rather than resolved by hindsight.

**Blind-gate commitment for the three-seed stage.** For every seed, the order
is: train → export → `pair_agreement.py` (label-free, no beam) → commit the
gate number and the band prediction → decode. The bands are the PHASE_K §8.5
bands, fixed here: agreement ≥ 95 % predicts val t1 ≥ 88.30 and ensemble val
greedy ≥ 55 %; agreement < 95 % predicts val t1 ≤ 87.5 and greedy ≤ 30 %.

## 7. All five arms completed 188 k — training-time record

| arm | final agreement | greedy A/B | selection-score A/B | gate-passing evals |
|---|---|---|---|---|
| `v2pair-s1234` (L1) | **98.34 %** | 72.70 / 71.33 | 82.07 / 82.00 | **46 of 47** |
| `v2pair-e2-s1234` (L2) | 98.23 % | 71.18 / 70.53 | 82.12 / 81.82 | — |
| `v2pair-e2-s4321` | 98.30 % | 71.48 / 70.58 | 81.81 / 82.14 | — |
| `v2pair-e2-s7777` | 98.19 % | 71.31 / 70.83 | 81.97 / 82.08 | 31 of 47 (from 76.7 % at 4 k) |
| `v2pair-pw0-s1234` (L3 control) | **92.09 %** | 72.62 / 71.79 | 81.66 / 81.89 | **2 of 47** |

Agreement trajectories tell the mechanism story on their own. Coupled arms
climb monotonically (L1: 93.5 → 97.7 → 98.3) and sit above the gate for
essentially the whole run. The uncoupled control **oscillates in a band that
never establishes itself above the gate** (89.1 → 92.8 → 93.7 → 89.9 → 91.8 →
92.09 final) and clears 0.95 at only two of 47 evaluations — *on identical
batches, identical data seed, identical everything but the KL*. And `s7777`,
which began at 76.7 % agreement — further apart than any pair the campaign has
recorded — was pulled over the gate by step 68 k and finished at 98.19 %.

## 8. THE GATE, MEASURED AND COMMITTED BEFORE ANY DECODE

Computed with `pair_agreement.py` from the exported ONNX graphs, 2,000 val
traces, **labels unused, no beam run yet**. This commit precedes every Phase-L
decode.

| configuration | agreement | blank-pattern | letters-both | verdict |
|---|---|---|---|---|
| `v2pair-s1234` selected pair | **98.33 %** | 98.83 | 96.85 | PASS |
| `v2pair-e2-s1234` selected pair | **98.23 %** | 98.75 | 96.73 | PASS |
| `v2pair-e2-s4321` selected pair | **98.20 %** | 98.74 | 96.65 | PASS |
| `v2pair-e2-s7777` selected pair | **98.18 %** | 98.70 | 96.71 | PASS |
| `v2pair-pw0` selected pair (its 136 k gate-passing eval) | 95.32 % | 95.86 | 96.22 | PASS (marginal) |
| **`v2pair-pw0` own-best members** (a@172 k + b@164 k) | **91.30 %** | 91.80 | 96.29 | **FAIL** |
| `v2pair-s1234` own-best members (reference) | 98.10 % | 98.60 | 96.80 | PASS |

**COMMITTED PREDICTIONS (before decoding), using the PHASE_K §8.5 bands:**

1. The four coupled pairs and the pw0 *selected* pair → **working band: val
   t1 ≥ 88.30 and ensemble val greedy ≥ 55 %.**
2. The **pw0 own-best mix → broken band: val t1 ≤ 87.5 and ensemble greedy
   ≤ 30 %.** This is the sharpest test in the phase: the same two members
   whose *solo* selection scores (81.66 / 81.89) are within a tenth of the
   coupled arms', predicted to fail catastrophically when averaged, purely
   because their alignment gauges never converged.

Note the diagnostic split: `letters_both` is ~96.2–96.9 % for *every*
configuration including the failing one — the members agree on letter identity
wherever both emit. The entire difference is in **`blank_pattern`** (98.8 % vs
91.8 %), i.e. in *where* the emissions sit. That is precisely the alignment
gauge, isolated.

## 9. Decode results (full battery: val-9918 AOSP/E1 + six layout bars)

All fp32 unless marked. Pairs are per-frame arithmetic probability averaging
of the two selected members (the mix2 contract), `--ens-avg prob`.

### 9.1 Pairs vs the `mix2-i8f16` card (the PRIMARY bar)

| config | t1 | t3 | t5 | ≤3 | 4+ | dvorak | dv-app | azerty | qwertz | german | spanish | tally |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **card** | 88.68 | 92.61 | 93.46 | 91.30 | 87.32 | 91.94 | 91.53 | 84.93 | 82.81 | 81.22 | 89.59 | — |
| L1 pair (no E2) | **88.90** | **92.86** | **93.58** | **91.53** | **87.53** | **93.04** | **92.76** | 84.16 ✗ | 83.91 | 82.08 | 89.87 | **10/11** |
| L2 pair (E2) s1234 | 88.85 | 92.79 | 93.42 ✗ | **91.59** | 87.43 | 92.76 | 92.35 | **85.02** | **84.84** | **82.22** | **90.44** | **10/11** |
| E2 pair s4321 | 88.52 ✗ | 92.71 | 93.35 ✗ | 91.15 ✗ | 87.15 ✗ | 92.88 | 92.35 | 84.35 ✗ | 83.91 | 82.08 | 88.51 ✗ | 5/11 |
| E2 pair s7777 | 88.49 ✗ | 92.72 | 93.35 ✗ | 91.12 ✗ | 87.12 ✗ | 89.82 ✗ | 89.21 ✗ | 84.98 | 85.09 | 81.95 | 89.76 | 5/11 |
| pw0 pair (control) | 88.09 ✗ | 92.59 ✗ | 93.32 ✗ | 91.12 ✗ | 86.52 ✗ | 89.34 ✗ | 88.81 ✗ | 84.35 ✗ | 84.41 | 80.99 ✗ | 88.96 ✗ | 1/11 |

### 9.2 The same configurations against the ELEVEN CAMPAIGN BARS

| config | tally | misses |
|---|---|---|
| **L1 pair (no E2)** | **11/11** | — |
| **L2 pair (E2) s1234** | **11/11** | — |
| E2 pair s4321 | 10/11 | ≤3 91.15 (−0.12) |
| E2 pair s7777 | 10/11 | ≤3 91.12 (−0.15) |
| pw0 pair (control) | 7/11 | t1, t3, ≤3, 4+ |

### 9.3 Members solo — the SECONDARY (single-model) bar, eleven campaign bars

| member | t1 | ≤3 | dvorak | spanish | tally |
|---|---|---|---|---|---|
| **L1 member A (slw 1.0)** | 88.60 | **91.32** | 91.17 | 89.02 | **11/11** |
| L1 member B (slw 1.5) | 88.47 | **91.53** | 92.80 | 89.42 | 9/11 (azerty, qwertz) |
| L2 member A | 88.35 | 91.30 | 91.58 | 88.62 | 10/11 (t5 −0.02) |
| **L2 member B (slw 1.5)** | 88.66 | **91.44** | 92.27 | 88.96 | **11/11** |
| s4321 member A / B | 88.40 / 88.10 | 91.03 / 91.12 | 92.47 / 90.35 | 87.83 / 88.34 | 8/11, 6/11 |
| s7777 member A / B | 88.14 / 88.38 | 90.85 / 91.38 | 89.54 / 88.93 | 88.79 / 88.68 | 7/11, 8/11 |
| pw0 member A / B (control) | 88.56 / 88.28 | 91.12 / 91.35 | 89.09 / 91.33 | 87.83 / 89.08 | 6/11, 7/11 |

**`L1 member A` and `L2 member B` each clear all eleven campaign bars as a
SINGLE 1.5 M-parameter model** — a thing no single model did in Phases A–K
(`sw2345` held 10/11 and `slw2` held ≤3 at the cost of four others). Both are
single-seed results and the campaign's own lesson applies: PHASE_K §8.3
watched an all-five-val s1234 sweep evaporate across seeds. The s4321/s7777
members here (6–8/11) are exactly that warning being right again — with the
caveat that those seeds ran the *E2* recipe, so "seed luck" and "recipe" are
not separated at the member level.

## 10. THE COMMITTED PREDICTIONS, SCORED

| configuration | agreement | prediction | outcome | verdict |
|---|---|---|---|---|
| L1 pair | 98.33 % | working: t1 ≥ 88.30, greedy ≥ 55 | 88.90 / 72.92 | **PASS** |
| L2 pair | 98.23 % | working | 88.85 / 72.03 | **PASS** |
| E2 s4321 pair | 98.20 % | working | 88.52 / 71.71 | **PASS** |
| E2 s7777 pair | 98.18 % | working | 88.49 / 71.84 | **PASS** |
| pw0 *selected* pair | 95.32 % | working | 88.09 / 53.12 | **MISS both** (−0.21 t1, −1.88 greedy) |
| pw0 *own-best* mix | 91.30 % | broken: t1 ≤ 87.5, greedy ≤ 30 | 87.64 / **29.10** | **greedy PASS, t1 MISS** (+0.14) |

**Four of four clean predictions above 98 % agreement. Both marginal cases
missed part of their band.** Stated as the qualification it is:

* The gate remains an excellent **ordinal** predictor. Agreement 98 % → mix
  greedy 71.7–72.9 and 10–11 bars; 95.3 % → greedy 53.1; 91.3 % → greedy
  **29.10**, a collapse from members that individually greedy at 72.6/71.8.
  The mechanism is confirmed harder than ever.
* But the PHASE_K §8.5 **numeric bands are calibrated for the extremes and do
  not describe the 91–96 % middle zone.** This phase is the **first time the
  broken half of those bands has ever been exercised** — Phase K's gate
  passed at 97 %, so its "< 95 % → t1 ≤ 87.5 / greedy ≤ 30" arm was a
  counterfactual that was never run. Run here, it is **half right**: the
  greedy collapse lands exactly as predicted (29.10 ≤ 30) while top-1
  degrades only to 87.64, not ≤ 87.5. The beam plus a 147 k-word trie
  recovers most of what a broken alignment costs greedy decoding.
* **This qualifies PHASE_K §8.5 without retracting it.** What was validated
  there (≥ 95 % predicts a working mix) held 4/4 again. What is newly
  measured is that "< 95 %" is a graded degradation, not a cliff, and that a
  *marginal* pass (95.32 %) does not guarantee the working band.

## 11. VERDICTS against the pre-stated bars

### 11.1 E1 (coupled-pair training) — CONFIRMED as the mechanism

The pre-registered attribution reading (§5) resolves cleanly in the
"E1 confirmed" branch. On **identical batches, identical data seed, identical
members-by-construction, differing only in `--pair-weight`**:

| quantity | coupled (L1) | uncoupled (L3) | Δ |
|---|---|---|---|
| final per-frame agreement | **98.34 %** | 92.09 % | +6.25 |
| evals above the 0.95 gate | 46 of 47 | **2 of 47** | — |
| mix val t1 | **88.90** | 87.64 (own-best) / 88.09 (selected) | **+1.26 / +0.81** |
| mix val greedy | **72.92** | **29.10** (own-best) | **+43.82** |
| mix tally, campaign bars | **11/11** | 7/11 | +4 |
| member A / B val t1 | 88.60 / 88.47 | 88.56 / 88.28 | +0.04 / +0.19 |
| member A dvorak | **91.17** | 89.09 | +2.08 |

**The coupling is the active ingredient, not batch sharing.** Sharing a batch
stream leaves the gauge unpinned — the control oscillates 89–94 % for 188 k
steps and never establishes itself over the gate. The KL pins it by step ~8 k
and holds it there. And the secondary, honestly-uncertain hypothesis (mutual
learning helps the *members*) is **weakly supported, not established**: +0.04
and +0.19 t1 are inside noise, but the transfer deltas (+2.08 dvorak on
member A) are larger than the axis's seed spread. One seed; not promoted.

The proposal's central claim — *make the campaign's best configuration a
reproducible recipe instead of a lucky draw* — is met: **4 of 4 coupled pairs
passed the gate (98.18–98.34 %), versus the campaign's historical 3-in-4**,
and the one pair that started furthest apart in campaign history (76.7 % at
4 k) was pulled to 98.19 %.

### 11.2 E2 (targeted English synthesis) — MISSES its gate by 0.01; NOT promoted

Pre-registered rule (§5): *L2 ≥ L1 on ≤3 with no other val bar losing more
than 0.15.*

| val bar | L1 | L2 | Δ | rule |
|---|---|---|---|---|
| ≤3 | 91.53 | **91.59** | +0.06 | ✓ required direction |
| t1 | 88.90 | 88.85 | −0.05 | ✓ |
| t3 | 92.86 | 92.79 | −0.07 | ✓ |
| **t5** | 93.58 | 93.42 | **−0.16** | **✗ (limit 0.15)** |
| 4+ | 87.53 | 87.43 | −0.10 | ✓ |

**Applied as written, E2 fails, and per the proposal's failure rule ("dropped,
not tuned-until-it-passes") it is NOT promoted.** The full disclosure, because
the margin is 0.01 and the gate was val-only:

* E2's measured effect is a **val wash with a euro-layout gain**: azerty
  **+0.86**, qwertz **+0.93**, spanish **+0.57**, german +0.14, against
  dvorak −0.28 and dvorak-app −0.41. On the full eleven-bar card comparison
  the E2 pair is *closer* to the card than L1 (its single miss is t5 by 0.04;
  L1's is azerty by 0.77).
* That gain is mechanistically coherent — 74 k unseen lexicon-tail word shapes
  and 8 k short-word shapes buy lexicon coverage, which is what the euro
  corpora stress — and it was **outside the scope the gate was written in**.
* I am not re-scoring E2 against a bar invented after seeing the numbers. The
  recorded verdict is: **fails its pre-registered gate; the layout evidence
  is a registered, unresolved question needing three seeds of the L1 recipe
  to answer.** Both facts stand in the record together.

### 11.3 Bar 1 (primary, the pair bar) — NOT MET

Requires: gated pair ≥ every number on the `mix2-i8f16` card, on ≥ 2 of 3
seeds, with ≤3 margin > +0.10.

* The three-seed stage ran the **E2 recipe** (registered in §6 as the
  presumptive configuration, with the cost of being wrong accepted in
  advance). Tally: s1234 10/11, s4321 5/11, s7777 5/11. **1 of 3 seeds.**
* The ≤3 sub-condition *is* met at s1234 (91.59, +0.29) and by L1 (91.53,
  +0.23) — both well past the +0.10 requirement — but fails at the other two
  seeds (91.15, 91.12).
* The L1 recipe, which produced the best single configuration in the phase
  (10/11 vs the card, 11/11 vs campaign bars), **has one seed**. Its
  three-seed stage is the registered-not-run measurement this phase ends on.

**Honest statement: the primary bar is not met.** The campaign-bar footing
tells the softer true story — *every* coupled pair cleared 10 or 11 of the
eleven campaign bars (11, 11, 10, 10), which matches and slightly exceeds
Phase K's recipe-level claim ("gated mixing clears 10–11 and beats every
single model") while now being **reliable by construction** rather than
gated-and-retried.

### 11.4 Bar 2 (secondary, single model, all eleven campaign bars) — MET AT ONE SEED, not on the stated seed-mean footing

`L1 member A` (11/11) and `L2 member B` (11/11) are the first single models in
the campaign to clear all eleven. The bar as written says *seed-mean*, and one
seed is not a seed-mean; the s4321/s7777 members (6–8/11) say plainly that
this can be seed luck. **Not promoted. Recorded as the strongest single-model
result the campaign has produced, pending three seeds.**

### 11.5 Bar 3 (stretch) and the E7 trigger — DOES NOT TRIGGER

The stretch (a single model beating the full `mix2-i8f16` card) did not fall:
`L1 member A` reads 88.60/91.32/91.17/83.97 against card numbers
88.68/91.30/91.94/84.93 — it beats the card's ≤3 but not its t1 or transfer,
exactly as the proposal predicted ("not expected to fall in v2's base
stages"). E7 was specified to run **only if bars 1–2 fall**; bar 1 is not met
and bar 2 is met only at one seed, so **E7 does not trigger and no
distillation was run.** The proposal's own fallback conclusion applies
verbatim: *the pair IS the product.*

## 12. The shippable candidate — `v2pair-s1234` at int8w + fp16w

Quantized exactly as the incumbent's packaging (member A int8w, member B
fp16w), then re-decoded in full:

| axis | card | **v2pair-s1234 i8f16** | Δ | campaign bar | Δ |
|---|---|---|---|---|---|
| val t1 | 88.68 | **88.86** | +0.18 | 88.30 | +0.56 |
| val t3 | 92.61 | **92.82** | +0.21 | 92.60 | +0.22 |
| val t5 | 93.46 | **93.59** | +0.13 | 93.26 | +0.33 |
| **val ≤3** | 91.30 | **91.56** | **+0.26** | 91.27 | **+0.29** |
| val 4+ | 87.32 | **87.46** | +0.14 | 86.77 | +0.69 |
| dvorak | 91.94 | **92.88** | +0.94 | 89.13 | +3.75 |
| dvorak app-98k | 91.53 | **92.59** | +1.06 | 88.20 | +4.39 |
| **azerty** | 84.93 | 84.11 | **−0.82 ✗** | 83.60 | +0.51 |
| qwertz | 82.81 | **84.41** | +1.60 | 82.50 | +1.91 |
| german | 81.22 | **82.26** | +1.04 | 79.64 | +2.62 |
| spanish | 89.59 | **89.76** | +0.17 | 88.28 | +1.48 |
| size | ≤ 5 MB | **4.39 MB** | −0.06 vs incumbent | ≤ 5 MB | ✓ |
| val greedy | — | 72.71 | — | — | — |

**10/11 against the card (azerty only), 11/11 against the campaign bars, at
4.39 MB** — 60 KB smaller than the incumbent `mix2-i8f16` packaging.
Quantization was free to three decimal places on val (fp32 88.90 → 88.86 t1,
≤3 91.53 → 91.56) and *helped* qwertz (83.91 → 84.41), consistent with
PHASE_K §4.6's finding that int8w+fp16w is the val-free packaging.

### 12.1 Artifacts (sha256, staged in `ctc/artifacts/`)

| file | bytes | sha256 |
|---|---|---|
| `phaseL_v2pair_s1234_a_int8w.onnx` | 1,554,355 | `01580189…8bead7c4` |
| `phaseL_v2pair_s1234_b_fp16w.onnx` | 3,052,318 | `59f40d95…c71b2db7` |
| `phaseL_v2pair_s1234_memberA.onnx` (the 11/11 single model, fp32) | 6,068,519 | `4b9d1ef2…36f3f6a7` |
| `phaseL_v2pair_i8f16_golden.json` (averaged-emission fixture, E1, 10 cases) | 140,476 | `7440873a…dc8dc749` |

Training-data provenance: `cache/synth_en_short.npz`
`78e0984e…8102d1`, `cache/synth_en_tail.npz` `92b89a56…c403b11b` (used by the
E2 arms only — the candidate above is the **L1** recipe and contains no
synthetic rows).

## 13. Registered, NOT run (the honest ledger of what would settle this)

1. **Three seeds of the L1 recipe** (no E2). The single most valuable missing
   measurement: the phase's best configuration has one seed, and bar 1 is
   unanswerable without it. ~10 GPU-h (two arms, concurrent).
2. **E2 at three seeds of its own**, to resolve the val-wash-vs-euro-gain
   question §11.2 leaves open.
3. `--pair-weight` sweep {0.1, 1.0}: the proposal's S2 knob was never swept —
   0.3 was used throughout and worked, so the shape of the
   coupling-strength/diversity trade is unmeasured.
4. E6 (geometric alignment prior) — implemented behind `--geo-align-weight`,
   default off, **never run**. E1 pinned the gauge well enough that its
   motivation weakened, but the "all models mutually mixable" prize is
   untested.
5. E4 (`w_real` 0.20/0.25/0.30) — not run; L1/L2 used the audited
   `synth_frac 0.667`.
6. E7 distillation — trigger not met, correctly not run.

**test-2400: SEALED throughout Phase L. Ledger stays at 3 entries. No script
in this phase opened it; `train_v2.py`, `english_synth.py` and
`pair_agreement.py` each refuse test features by name.**

## 14. PRE-REGISTRATION — the settlement measurement (committed BEFORE launch)

**Authority:** coordinator directive of 2026-08-13 releasing §13 item 1 (the
user's standing "no stones unturned" directive). Two arms, the **L1 recipe
verbatim** — no synthesis pools, `--layout-synth-frac 0.667`, slw 1.0/1.5,
`--pair-weight 0.3`, ramp 5000+15000, 188 k, `--val-every 4000`, E5 probes on,
`--workers 0`:

| arm | `--seed` | `--init-seed-a/b` | pairs with |
|---|---|---|---|
| `v2pair-s4321` | 4321 | 3333 / 4444 | `v2pair-e2-s4321` |
| `v2pair-s7777` | 7777 | 5555 / 6666 | `v2pair-e2-s7777` |

Seeds and inits deliberately match the E2 arms, so this stage **also** yields a
paired L1-vs-E2 comparison at all three seeds — which is the three-seed E2
measurement §13 item 2 asked for, obtained for free. (`--beam-jobs 6` vs L1's
8: infrastructure, not recipe; it changes no training math.)

### 14.1 What can and cannot happen — stated before the runs exist

**Bar 1 now has a sharp, asymmetric form, and it is worth being explicit that
it is stacked against a pass.** The already-decoded `v2pair-s1234` pair is
**10/11 against the card** — it misses azerty (84.16 vs 84.93). Bar 1 requires
≥ 2 of 3 seeds to meet **every** card number. Seed 1234 has therefore already
spent its chance. **Bar 1 can only be met if BOTH new seeds come in 11/11
against the card**, azerty included.

That is a demanding ask on precisely the axis where this recipe is weakest:
azerty is the one bar E2 fixed (+0.86) and L1 does not. A pre-stated
expectation, recorded so it can be wrong: **I expect bar 1 to fail on azerty
and the phase to close at "the coupled pair is a reliable 10–11-bar recipe
that does not dominate the s1234 mix2 configuration on every axis."** If
instead both seeds clear azerty ≥ 84.93, bar 1 falls and the v2 pair
supersedes `mix2-i8f16` as the ship configuration.

**Bar 2 gets its proper footing here too.** With three L1-recipe seeds, the
members' eleven-bar tally is computed **seed-mean** (the footing the bar
actually names), not just per-seed. `L1 member A`'s single-seed 11/11 either
survives averaging with two fresh seeds or it does not; PHASE_K §8.3's lesson
says to expect it may not.

**Protocol, unchanged:** train → export → `pair_agreement.py` (label-free, no
beam) → commit gate + band prediction → decode. Bands as always: ≥ 95 % →
t1 ≥ 88.30 and ensemble greedy ≥ 55 %; < 95 % → t1 ≤ 87.5 and greedy ≤ 30 %
(now known to be only half-calibrated below the gate, §10 — the prediction is
committed in its original form anyway, and scored the same way).

**Nothing else launches after this stage without new instruction.**
