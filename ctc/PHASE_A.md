# Phase A — the data-tier ladder

One variable (the training pool), one frozen recipe, one holdout. Every number below is
measured, not projected; where a number is not trustworthy the reason is stated rather
than smoothed over.

**test-2400 was never decoded.** `eval_arms.py` refuses any split whose filename contains
`test`. Every figure here is val-9918.

---

## 1. Recipe (frozen across all five arms)

```
ch 96   embed_hid 96   batch 256   lr 3e-3   wd 0.01   warmup 1000 steps
AdamW + linear warmup -> cosine to 0 over a 47,000-step horizon
fp32, grad-clip 1.0, augmentation = slot permutation + rejection-sampled affine + noise
seed 1234   --total-steps 47000   --val-every 1500   patience 40 evals
```

`--total-steps` is what makes the ladder honest: the tiers differ by 3.5x in size, so an
epoch budget would have handed the big tiers 3.5x the gradient steps and the comparison
would have measured compute, not data. Every arm gets exactly 47,000 optimizer steps and
the same cosine horizon; only the pool underneath changes.

Checkpoint selection is `best val_greedy` on the 1,500-step evaluation grid.

## 2. Arms

| arm | cached rows | composition | contamination control |
|---|---|---|---|
| `phaseA-T0` | 109,600 | 55,438 HWS + 55,438 FUTO | none (the historical control) |
| `phaseA-T1` | 372,726 | HWS half + full curated FUTO | FUTO session exclusion only |
| `phaseA-T1strict` | 319,421 | 888 HWS + 318,566 FUTO | + HWS participant exclusion |
| `phaseA-T2` | 385,021 | FUTO only | session exclusion, hygiene filter only |
| `phaseA-T2b` | 285,929 | FUTO only | session exclusion + full quality cascade |

## 3. What the holdout can and cannot decide

This is the part that changed most during Phase A, and it changes how the table must be
read.

`val_clean[arm]` is the subset of val-9918 whose **contributor** appears nowhere in that
arm's training rows. It is rebuilt in `build_val_clean.py` from each row's own `session`
field — the canonical splits and both tier source pools carry it at 100 % coverage — after
the previous hash-based version was found to be wrong in the permissive direction:

* a tier's jsonl has no `session` field, so contributors were recovered by hashing tier
  rows against the corpus index;
* ~15 % of the FUTO pool was renormalised and no longer hashes to the corpus, so those
  rows resolved to nothing, contributed nothing to the tier's contributor set, and every
  val row from those contributors was scored as **clean while sitting in the training
  pool**.

| arm | val_clean (hash, old) | val_clean (session, correct) | of which FUTO / HWS |
|---|---|---|---|
| T0 | 176 | **164** (1.7 %) | 163 / 1 |
| T1 | 4,238 | **46** (0.5 %) | 43 / 3 |
| T1strict | — | **5,019** (50.6 %) | 43 / 4,976 |
| T2 | 9,213 | **9,300** (93.8 %) | 4,324 / 4,976 |
| T2b | 9,213 | **9,669** (97.5 %) | 4,693 / 4,976 |

Three consequences:

1. **T1 is not contributor-disjoint on the FUTO side.** `build_tiers.py` keeps the 102,826
   rows whose session could not be recovered, and they carry back ~2,000 of the very
   sessions the exclusion dropped: T1's FUTO contributor set is 9,877 of the corpus's
   10,889 sessions. `--strict-session` is what a genuinely disjoint T1 requires; T1strict
   fixes only the HWS side, which is why its clean FUTO count is still 43.
2. **Only T2/T2b support a generalization claim**, and their clean subset is 54 % HWS —
   a corpus they contain none of — so an undivided clean number is largely a cross-corpus
   transfer score. Every clean figure below is therefore also split by source.
3. **T0 and T1 clean subsets (164 and 46 rows) decide nothing.** At n=164 the standard
   error on a ~82 % rate is ±3.0 pt; at n=46 it is ±5.7 pt. They are reported for
   completeness and must not be used to rank arms.

Separately, 219 contributor sessions produced 249 val rows yet never entered
`futo_tainted_sessions.npz` — the holdout traces that would have exposed them are exactly
the ones whose hash no longer matches the corpus. The 27,356 corpus rows those sessions
hold remain inside every tier. The session-field masks above account for this; the tier
*builds* do not.

## 4. Results

Scoring preset: the published encoder-only `enc` preset, unchanged. Beam width 100.
`eval_arms.py` reproduces `eval_beam.py` exactly (checked on r2: 81.57 / 89.84 / 91.37
both ways), so these are directly comparable to every committed number.

### Aggregate and per-source

| arm | rows | wall | best greedy | val t1 | t3 | t5 | FUTO t1 | HWS t1 |
|---|---|---|---|---|---|---|---|---|
| `phaseA-T0` | 109,600 | 6.0 min | 58.42 | **82.12** | 90.27 | 91.60 | 88.00 | **76.29** |
| `phaseA-T1` | 372,726 | 7.0 min | **64.69** | **82.47** | **90.62** | **91.96** | 89.26 | 75.72 |
| `phaseA-T1strict` | 319,421 | 6.1 min | 61.09 | 80.67 | 88.61 | 90.41 | 91.24 | 70.18 |
| `phaseA-T2` | 385,021 | 6.9 min | 59.62 | 80.86 | 88.52 | 90.48 | **92.59** | 69.21 |
| `phaseA-T2b` | 285,929 | 6.3 min | 60.26 | 79.91 | 88.51 | 90.32 | 90.98 | 68.91 |

Reference: `r2` (the pre-Phase-A run, same recipe on T0 with an epoch budget) scores
81.57 / 89.84 / 91.37 with 58.57 greedy.

### On the contributor-clean subsets

`T2∩T2b` (n=9,300) is the only subset every arm can be scored on that is also large
enough to mean anything, so it is the cross-arm column. Split by source, because 4,976
of its 9,300 rows are HWS and three of the five arms contain almost no HWS.

| arm | own clean t1 (n) | shared-clean t1 (n=9,300) | shared FUTO (4,324) | shared HWS (4,976) |
|---|---|---|---|---|
| `phaseA-T0` | 83.54 (164) | 81.80 | 88.14 | **76.29** |
| `phaseA-T1` | 80.43 (46) | **81.91** | 89.04 | 75.72 |
| `phaseA-T1strict` | 70.31 (5,019) | 79.89 | 91.07 | 70.18 |
| `phaseA-T2` | 79.99 (9,300) | **79.99** | **92.39** | 69.21 |
| `phaseA-T2b` | 79.55 (9,669) | 79.03 | 90.68 | 68.91 |

### What the ladder actually measured

**The dominant variable is corpus mix, not corpus size.** Ordering the arms by how much
How-We-Swipe they contain — T0 and T1 (55,438 rows), T1strict (888), T2/T2b (0) — the two
halves of the holdout move in opposite directions and almost monotonically:

```
HWS in train:  55,438   55,438      888        0        0
FUTO t1:        88.00    89.26    91.24    92.59    90.98
HWS  t1:        76.29    75.72    70.18    69.21    68.91
```

Aggregate val is a ~50/50 mixture of the two, so it falls as FUTO accuracy rises. Going
from T0 to T2 buys **+4.6 pt on FUTO and costs −7.1 pt on HWS**. None of the aggregate
differences between T0, T1 and T2 are a statement about "more data"; they are a statement
about which corpus the training set is drawn from.

**T2 vs T2b is the one clean, unconfounded contrast** — same corpus, same contamination
control, same step budget, differing only by the recovered quality cascade. On the shared
clean subset **T2 beats T2b by 0.96 pt overall and 1.71 pt on the FUTO half** (n=4,324,
unpaired SE ≈ 0.43 pt, so ~4 SE; the paired comparison is tighter still). The cascade's
own gates reject 152,603 rows — overwhelmingly motion/geometry: `not_portrait` (53,464),
`bad_speed` (40,865), `too_many_points` (26,722), `bad_duration` (19,528), with the
lexical gates contributing only 11,567 — and leave T2b 99,526 rows smaller than T2 net.
It buys negative accuracy. **The quality cascade does not earn its keep.**

**Scale at fixed curation (T0 → T1) is worth about +0.35 pt aggregate** (+1.26 FUTO,
−0.57 HWS) for 3.4x the data, and both arms are contaminated, so even that is generous.
Greedy accuracy moves far more (58.42 → 64.69) than beam top-1 (82.12 → 82.47): the
lexicon beam already recovers most of what better emissions provide, so **greedy is a
poor proxy for the metric that ships**.

### Caveat on resolution — read before ranking anything

Re-evaluating a *single* arm's own checkpoints spans as much as the gaps between arms.
`phaseA-T0`'s final-step checkpoint scores 81.68 full-val t1 against 82.12 for its
best-greedy checkpoint, and the r2 reference — same data, same recipe, epoch budget —
scores 81.57. That is a **0.55 pt spread from checkpoint selection alone**, larger than
the T0→T1 difference. Aggregate differences below ~0.6 pt in this table are not
resolvable; the T2-vs-T2b FUTO gap (1.71 pt) and the corpus-mix effects (4–7 pt) are.

> ⚠ **Corrected after Phase C.** The 0.55 pt figure above counts checkpoint-selection
> spread only. Re-running the T2 arm at a second seed moved clean t1 by **1.05 pt** and
> FUTO t1 by 1.45 pt, so the true single-seed resolution limit is **~1 pt, not ~0.6**.
> Consequences for this table: the T0→T1 aggregate gap (+0.35) is **not** resolvable, and
> the T2-vs-T2b aggregate gap (0.96) only barely reaches the floor. **The T2-vs-T2b
> FUTO-half gap (1.71 pt) is the one that survives**, and it is what the "drop the cascade"
> recommendation should rest on. The corpus-mix effects (4–7 pt) are unaffected.

Related: the `seconds` field in `metrics.jsonl` is not the interval between validations —
`t0` is also reset at each epoch boundary, so with `--val-every` smaller than an epoch it
under-reports. Wall-clock above is taken from log file timestamps instead.

### Recommendation for Phase B

* **Drop the T2b quality cascade.** It is measurably harmful at fixed step budget.
* **Do not adopt T2 as-is.** It is the best FUTO model in the ladder and the worst HWS
  model; shipping it trades 7 pt on one user population for 4.6 pt on another.
* **Base tier should be the T2 FUTO pool re-merged with the HWS half**, with contamination
  control applied properly on *both* sides — `--strict-session` on FUTO (to close the
  102,826-unmapped-row hole that leaves T1 with 46 clean val rows) and participant
  exclusion on HWS. That is the arm this ladder implies but none of the five actually is.
* **Fix the holdout before Phase B ranks anything.** With a proper contributor-disjoint
  tier on both sides, val_clean should reach ~90 %+ for a mixed tier, which is what makes
  a mixed arm comparable to T2 at all. Today only T2/T2b have a usable clean subset.
* Re-run the winner at 2–3 seeds; a single seed cannot resolve the sub-1-pt differences
  this table is full of.

## 5. Protocol going forward

The rules Phase A settled, so later phases do not relitigate them:

* **Exact-trace dedup is non-negotiable** and is enforced twice — in `build_tiers.py` and
  again in `prepare_data.py`, so a tier physically cannot smuggle a holdout trace even if
  the tier builder missed one.
* **Session/participant exclusion defines a *generalization* arm.** Any arm claiming to
  measure generalization must exclude every contributor that touched the holdout, on
  *both* corpora, and must resolve contributors from the source `session` field rather
  than from a trace hash. An arm that cannot resolve a contributor must count that row as
  present, never as absent.
* **A benchmark arm is a different object from a generalization arm.** A future T3 would
  train the winning recipe *without* session exclusion, to be comparable with published
  FUTO numbers — which were produced by a model trained on the literal test traces. That
  asymmetry has to be disclosed at the point of comparison every time, because a T3-vs-FUTO
  number and a T2-vs-T2b number mean different things and only the latter is a claim about
  the model.
* **Report per-source.** The HWS and FUTO halves sit at a known ~0.064 systematic Y offset,
  and the arms move them in opposite directions. An aggregate val number hides that.

## 6. Reproduction

```bash
# tiers (‑‑suffix keeps a rebuild out of the file a run already trained on)
python build_tiers.py --tiers t1 --suffix _strict
python prepare_data.py --extra-train data/tier_t1_strict.jsonl \
                       --out-name train_t1_strict --jobs 10

# contributor masks (session fields, zero unresolved rows)
python build_val_clean.py --arms T0,T1,T1strict,T2,T2b

# one arm
python train.py --train-npz train_t2.npz --run-name phaseA-T2 \
                --total-steps 47000 --val-every 1500 --seed 1234

# eval (cached emissions; reproduces eval_beam.py exactly)
python export_onnx.py --ckpt ckpt/phaseA-T2/best.pt \
                      --out ckpt/phaseA-T2/ctc_swipe_encoder.onnx
python eval_arms.py --arms phaseA-T0,phaseA-T1,phaseA-T1strict,phaseA-T2,phaseA-T2b
```
