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

*(results land below when the run finishes)*
