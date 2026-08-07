# Phase D — beam-selected checkpoints on the T3 benchmark tier

Phase D changes two things the earlier phases got wrong, and then runs the
architecture question again on a tier ~2.6× the size of anything used so far.

1. **Checkpoint selection is now beam top-1, not greedy.** Phase B measured an arm
   whose greedy accuracy rose 5.16 pt while its beam top-1 fell 1.38 pt, so
   `best val_greedy` was choosing checkpoints by a metric that anti-correlates
   with the metric that ships. Every Phase-D checkpoint is selected on
   `val_beam_t1` over a fixed 2,000-row val prefix.
2. **Decisions need three seeds.** Phase C measured a 1.05 pt clean-t1 swing
   between two runs differing only in seed. Single-seed differences below ~1 pt
   are not interpretable, so the two leading arms are run at seeds
   1234 / 4321 / 7777 and compared pairwise on the same seed set.

**test-2400 was never decoded.** `eval_arms.py` refuses any split whose filename
contains `test`, and `train.py` now refuses it as a selection split too.

---

## 1. Beam-selected checkpointing

At every `--val-every` step `train.py` runs the vendored `futo_viterbi_beam` over
the first 2,000 val rows, at the published `encoderOnly` preset, beam width 100,
`top_k=5`, and selects `best.pt` on the resulting top-1. Greedy accuracy is still
computed and logged at every point, so the greedy-vs-beam relationship stays
visible; it just no longer decides anything.

Mechanically the beam pool is forked **once**, at construction time, before the
model touches the GPU:

* the 146,964-word trie is inherited copy-on-write and is never pickled;
* the sliced `[N,32,27]` emissions live in a shared `RawArray` the parent
  overwrites in place, so no re-fork and no serialization happen per validation;
* no child ever inherits a live CUDA context.

**Correctness.** The validator reproduces the committed r2 numbers on the same
2,000 rows exactly: **81.55 / 89.85 / 91.65**, identical to the digit to the
figures in `README.md` §"Measured result on r2".

**Cost.** ~2 s per validation point at 12 processes on an idle machine (4.0 s at 6
processes with the box under a parallel featurization load). At `--val-every 3000`
over 94,000 steps that is 31 validations, i.e. **~1 minute added to a ~15-minute
run** — under 10 % overhead, and the pause is not overlapped with GPU steps
because it does not need to be.

---

## 2. T3 — the benchmark tier, and the disclosure it requires

### Build

| side | in | dropped | kept |
|---|---|---|---|
| FUTO swipe-1 `train.jsonl` | 939,550 | 4,709 `potentially_invalid_sentence`; 5,273 exact-trace holdout matches | **929,568** |
| How-We-Swipe full release (1,338 users) | 86,323 traces from 1,338 logs | 1,711 `len(word) < 2`; 6,457 exact-trace holdout matches | **78,155** |
| **tier jsonl** | | | **1,007,723** |
| after `prepare_data.py` | 1,007,723 | 43 empty word, 1 CTC-infeasible, 2,343 self-duplicates, **0 cross-split duplicates** | **1,005,336** (`cache/train_t3.npz`, 435 MB) |

Two build facts worth keeping:

* **The HWS parse is bit-exact.** `build_tiers.parse_hws_log` is a replica of
  `neural-swipe-typing/process_swipe_logs.py`; re-parsing the 1,052 logs the
  canonical splits were built from reproduces **all 60,303 unique traces of the
  61,597-row canonical pool, with zero misses**, and the first row compares equal
  dict-for-dict (word and every `{t,x,y}` float). The full release then yields
  **84,612** rows under the same filter (`is_err = 0`, ≥ 3 points,
  `len(word) ≥ 2`) against the canonical pool's 61,597 — the 286 users absent
  from the local log directory are worth +23,015 rows, exactly the figure
  `fetch_hws_full.py --analyze` projected.
* **The 6,457 HWS rows removed by exact dedup are the right ones.** The canonical
  holdout holds 6,159 HWS rows, which collapse to **6,140 unique traces** (11
  duplicate pairs in val, 1 in test). The release contains ~2 % duplicate traces
  overall (61,597 rows → 60,303 unique in the local pool), so 6,140 unique
  holdout traces are expected to appear ≈ 6,140 × 1.05 ≈ 6,450 times in the
  release. Measured: 6,457. The val/test HWS rows are literally drawn from these
  logs, and the dedup found them.

Exact-trace dedup was applied under **both** hash conventions in use in this
pipeline — `scan_futo_sessions`' `(word.lower(), x, y)` and `prepare_data`'s
`(word, x, y, t)` — and their union was taken. The `(word, x, y)` form ignores
timing and is therefore the stricter of the two: it caught every match on both
sides (`futo_leak_xy` 5,273 / `hws_leak_xy` 6,457) and the `(word,x,y,t)` form
added **zero** further rows. `prepare_data.py` then re-applies its own dedup
independently and reports `dropped_cross_split_duplicate = 0`, which is the
cross-check that the tier build missed nothing.

### ⚠ Disclosure — T3 is contributor-dirty by construction

**T3 applies no session or participant exclusion.** Every contributor who
produced a val or test trace also has other traces in T3, on both corpora. This
is deliberate, and it means:

* **T3 cannot support a generalization claim.** Phase A's ladder established that
  a generalization arm must exclude every contributor that touched the holdout on
  both corpora (T2/T2b do; T3 does not). A T3 number is an *upper* bound that
  includes whatever a model can memorise about a specific person's hand geometry
  from their other traces.
* **T3 exists to be comparable with the published FUTO baselines**, which were
  produced by models trained on the literal holdout traces. On that axis T3 is
  strictly the more conservative of the two: it removes the 12,299 holdout traces
  bit-exactly (11,730 of them found and dropped: 5,273 FUTO + 6,457 HWS
  instances), where the baselines kept them.
* **Every T3-vs-FUTO comparison must carry this paragraph.** A T3-vs-FUTO number
  and a T2-vs-T2b number are different objects; only the latter is a statement
  about the model. This is the asymmetry `PHASE_A.md` §5 said had to be disclosed
  at the point of comparison, every time.

Because of this, the arm-vs-arm comparisons in §3 are the load-bearing numbers
(all arms share the same tier and the same contamination, so the contamination
cancels), and the absolute level is not.

---

## 3. Arms and the seed-1234 round

Recipe frozen across every arm: **94,000 steps** (2× the Phase A–C budget, because
T3 is ~2.6× T2), batch 256, lr 3e-3, wd 0.01, warmup 1,000, fp32,
`--val-every 3000` (31 validation points), current augmentation (slot permutation
+ rejection-sampled affine + noise; **no** C1 path jitter, which Phase C killed).

| arm | tier | architecture | params |
|---|---|---|---|
| `phaseD-D0` | T3 | ch 96, residual trunk, v1 features (the Phase-A/B/C baseline) | 394,114 |
| `phaseD-D1` | T3 | **ch 128**, `embed_hid` 128, residual trunk | 689,282 |
| `phaseD-D2` | T3 | Phase-B ConvNeXt trunk, ch 128, 5 blocks, dil {1,2,3,5,8} | 570,818 |
| `phaseD-D3` | T3 | D1 (the D0–D2 winner) + EMA decay 0.999, evaluated on the averaged weights | 689,282 |
| `phaseD-T1bridge` | T1 | ch 96 — the D0 recipe on the Phase-A best tier | 394,114 |
| `phaseD-T1bridge128` | T1 | ch 128 — added to complete the arch × tier 2×2 (see below) | 689,282 |

`T1bridge128` was **not** in the original brief. It was added after `T1bridge`
(ch 96 on T1) beat `D0` (the same architecture on T3) by 0.89 pt: with the tier
question live, comparing tiers at one architecture only would have left the
result confounded with capacity. The 2×2 costs 15 GPU-minutes and removes that.

### Results — full val-9918, published `enc` preset, beam width 100

| arm | greedy | beam2000 t1 (selection) | **val t1** | t3 | t5 | ≤3 | 4+ | FUTO t1 | HWS t1 |
|---|---|---|---|---|---|---|---|---|---|
| `D0` (T3, ch96) | 67.20 | 84.30 | 83.15 | 90.81 | 92.10 | 84.83 | 82.28 | 91.00 | 75.36 |
| `D1` (T3, ch128) | 67.45 | 84.80 | **84.22** | 90.74 | 92.22 | 87.40 | 82.57 | **92.25** | 76.25 |
| `D2` (T3, convnext) | 70.65 | 84.45 | 83.35 | 90.98 | 92.20 | 84.60 | 82.71 | 91.42 | 75.34 |
| `D3` (T3, ch128+EMA) | 68.91 | 84.05 | 84.09 | **91.09** | **92.44** | 86.16 | **83.01** | 92.01 | 76.23 |
| `T1bridge` (T1, ch96) | 65.62 | 85.10 | 84.04 | 90.76 | 92.23 | **88.26** | 81.85 | 91.91 | 76.23 |
| `T1bridge128` (T1, ch128) | 69.89 | **85.35** | **84.29** | 90.90 | 92.13 | 88.02 | 82.36 | 91.30 | **77.33** |
| — reference: `phaseA-T1` (47 k steps) | 64.69 | — | 82.47 | 90.62 | 91.96 | 84.45 | 81.44 | 89.26 | 75.72 |
| — reference: `phaseA-T2` (47 k steps) | 59.62 | — | 80.86 | 88.52 | 90.48 | 84.69 | 78.88 | 92.59 | 69.21 |
| — target: FUTO ceiling (**test**-2400) | 69.12 | — | 84.83 | 91.04 | 92.08 | 89.57 | 82.40 | — | — |

Everything in the top block is val-9918; the FUTO row is test-2400 and is shown
only to indicate the target's shape. **They are not directly comparable** — see §5.

### What the round shows

**The step budget and the data both mattered, and neither is the whole story.**
The best pre-Phase-D arm was 82.47 (`phaseA-T1`, 47 k steps); the worst Phase-D
arm is 83.15. But `T1bridge` *is* `phaseA-T1`'s tier and architecture at 2× the
steps with beam selection, and it scores 84.04 — so **+1.57 pt of the improvement
is budget and selection, not data**.

**T3 did not beat T1.** At matched architecture and matched budget:

| architecture | T1 (374 k rows) | T3 (1.005 M rows) | Δ (T3 − T1) |
|---|---|---|---|
| ch 96 | 84.04 | 83.15 | **−0.89** |
| ch 128 | 84.29 | 84.22 | **−0.07** |

2.7× the training rows, including the whole FUTO corpus with **no** session
exclusion at all, buys nothing. At ch 96 it is 0.89 pt *worse*; at ch 128 the two
are a dead heat. Since T3 is strictly more contaminated than T1 (T1 excludes
369,459 session-tainted FUTO rows; T3 excludes none), this is a genuinely
negative result for raw FUTO volume: the curated, session-excluded, HWS-balanced
pool is at least as good and is far cheaper to train on. Whether the gap survives
three seeds is answered in §4.

**Capacity is the one lever that moved.** ch 96 → ch 128 is worth +1.07 pt on T3
and +0.25 pt on T1, at 1.75× the parameters. Only the T3 figure clears the ~1 pt
noise floor at one seed, and it is the only Phase-B/C/D architecture lever that
ever has.

**The ConvNeXt trunk regresses again, and beam selection was not the confound.**
Phase B's suspicion was that B2's greedy-selected checkpoint had been chosen by
the wrong metric. Under beam selection D2 still lands **−0.87 pt below D1** at
1.2× fewer parameters but 1.5× D0's inference cost, and its stratum signature is
the same one Phase B measured, just milder: vs D0, **−0.23 on ≤3 and +0.43 on
4+**. The hypothesis is tested and dead — the trunk is genuinely worse for
lexicon-decoded accuracy, not merely mis-selected.

**Greedy and beam still disagree.** D2 has the best greedy of any Phase-D arm
(73.00 at its final step) and the second-worst val t1. `T1bridge` has the *worst*
greedy (65.62) and the fourth-best val t1. Retiring greedy as the selection
metric was correct.

### The cost of beam-2000 selection — measured, not assumed

At n = 2,000 the binomial SE on an ~84 % rate is ~0.8 pt, which is large relative
to the plateau the cosine schedule produces. Three of the six arms selected a
checkpoint that was **not** the final step: D1 at step 54,000, D2 at 66,000, D3 at
63,000, `T1bridge128` at 72,000. For D1 that choice was measurable:

| D1 checkpoint | greedy | beam2000 t1 | full-val t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|---|---|
| beam-selected (step 54,000) | 67.45 | 84.80 | 84.22 | 90.74 | 92.22 | 87.40 | 82.57 |
| final step (94,000) | 70.31 | 84.70 | **84.39** | **91.14** | **92.45** | 86.46 | **83.32** |

The selection rule preferred a checkpoint 40,000 steps early on a 0.10 pt
beam-2000 margin, and it cost **−0.17 pt** on full val (and −0.40 / −0.23 on
t3/t5). That is small and well inside noise, but the direction is a warning: with
a cosine-to-zero schedule the final checkpoint is already near-optimal, so a
selection rule noisier than the plateau can only lose. **Recommendation for Phase
E: keep beam selection but raise the prefix to ~5,000 rows (SE ~0.5 pt, still
under 5 s per validation), or simply take the final step and use the beam curve
as a diagnostic.** Beam selection is still strictly better than greedy — greedy
would have picked the final step here for the wrong reason, and picked badly on
D2, where greedy rises monotonically while beam plateaus.
