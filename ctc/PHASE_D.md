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

> ⚠ **CORRECTED post-decode (`AUDIT_FINAL.md` §4).** The claim that exact-trace
> dedup "caught every match" is **false as written**. Both hash conventions keyed
> on the **raw** word while the CTC target is built from the a–z-normalized word,
> so `'arabian.'` in a tier did not match `'arabian'` in the holdout even though
> the two rows carry a bit-identical input tensor *and* the same label. **588 val
> rows and 145 test rows are in `train_t3` on that blind spot.** The key is fixed
> in `build_tiers.hash_row` / `prepare_data.trace_hash` as of the post-decode
> hygiene pass; the tiers on disk were **not** rebuilt, by the deliberate decision
> in `AUDIT_PREDECODE.md` §E. Measured effect: the leaked rows score **4.34 pt
> below** comparable non-leaked ones — no memorization signal — and removing all
> of them moves the headline **< 0.05 pt on val / 0.20 pt on test**, with all five
> bars still clearing on every one of six seeds.

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
369,459 session-tainted FUTO rows; T3 excludes none), this reads as a genuinely
negative result for raw FUTO volume.

> ⚠ **Corrected by §4.** At three seeds the ch-128 sign *reverses* (paired
> T3 − T1 = −0.07 / +0.31 / +1.06, mean **+0.43**) and the difference is not
> resolvable (paired t(2) = 1.31). **"T3 did not beat T1" is a one-seed artifact
> and must not be quoted.** The defensible statement is that the two tiers are
> indistinguishable at ch 128, which is still a negative result for volume — it
> just is not the stronger claim this paragraph originally made. The ch-96 row is
> single-seed on both sides and decides nothing on its own.

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

---

## 4. The seed round — three seeds on the two contenders

The two arms with the highest val t1 at seed 1234 were `D1` (ch 128 on T3) and
`T1bridge128` (ch 128 on T1) — the same architecture on different tiers, which
makes the seed round a **paired tier test at matched capacity**. Both were re-run
at seeds 4321 and 7777; the seed set is identical across arms, so every comparison
below is paired.

### Per-seed and seed-mean, full val-9918

| arm | seed | greedy | beam2000 | **val t1** | t3 | t5 | ≤3 | 4+ | FUTO | HWS |
|---|---|---|---|---|---|---|---|---|---|---|
| `D1` (T3, ch128) | 1234 | 67.45 | 84.80 | 84.22 | 90.74 | 92.22 | 87.40 | 82.57 | 92.25 | 76.25 |
| `D1` | 4321 | 69.66 | 85.60 | 84.87 | 91.11 | 92.35 | 87.90 | 83.29 | 93.08 | 76.71 |
| `D1` | 7777 | 69.96 | 86.50 | **85.34** | 91.18 | 92.42 | 88.73 | 83.58 | 93.28 | 77.45 |
| **`D1` mean (sd)** | | | 85.63 | **84.81** (0.56) | **91.01** | **92.33** | 88.01 | **83.15** | **92.87** | 76.80 |
| `T1bridge128` (T1, ch128) | 1234 | 69.89 | 85.35 | 84.29 | 90.90 | 92.13 | 88.02 | 82.36 | 91.30 | 77.33 |
| `T1bridge128` | 4321 | 69.27 | 85.55 | 84.55 | 90.84 | 92.28 | 88.34 | 82.59 | 91.58 | 77.57 |
| `T1bridge128` | 7777 | 67.87 | 85.20 | 84.28 | 90.64 | 91.90 | 88.91 | 81.88 | 91.68 | 76.93 |
| **`T1bridge128` mean (sd)** | | | 85.37 | **84.38** (0.15) | 90.79 | 92.10 | **88.42** | 82.27 | 91.52 | **77.28** |

### Paired T3 − T1 by seed

| metric | 1234 | 4321 | 7777 | mean |
|---|---|---|---|---|
| val t1 | −0.07 | +0.31 | +1.06 | **+0.43** |
| FUTO t1 | +0.95 | +1.50 | +1.60 | **+1.35** |
| HWS t1 | −1.09 | −0.86 | +0.52 | **−0.48** |
| ≤3 t1 | −0.62 | −0.44 | −0.18 | **−0.41** |
| 4+ t1 | +0.21 | +0.70 | +1.70 | **+0.87** |

**The aggregate tier difference is not resolvable.** Paired t(2) = 1.31 on val t1,
against the |t| > 4.30 a two-tailed 5 % test needs at n = 3. The sign also flips
between seeds. **The single-seed reading in §3 — "T3 did not beat T1" — does not
survive three seeds; the honest statement is that T3 and T1 are indistinguishable
at ch 128.** That correction matters, because the §3 reading was the more
interesting claim and it was wrong.

**What *is* consistent is the direction of the trade, and it is the same one
Phase A found.** T3 is 92 % FUTO by row count against T1's 85 %, and the two
halves move in opposite directions at every seed on FUTO (+0.95/+1.50/+1.60,
sign-stable) and at two of three on HWS. T3 also buys long words and sells short
ones: **4+ +0.87, ≤3 −0.41**, with 4+ positive at all three seeds. Adding
575,000 raw FUTO rows and dropping all session exclusion moves the corpus mix, not
the model quality — which is exactly Phase A's conclusion, now confirmed at 2.7×
the scale and with the contamination controls removed entirely.

**Seed variance is itself informative.** `D1` on T3 has sd 0.56 across seeds;
`T1bridge128` on T1 has sd **0.15**. T3's per-seed spread is nearly 4× T1's, and
its beam-2000 selection metric ranges 84.80–86.50. A tier that produces a 1.1 pt
range at fixed everything-else is a worse basis for a shipping decision than one
that produces a 0.3 pt range, independent of which has the higher mean. Note also
that the highest single number in the campaign (`D1`-7777, 85.34) sits at the top
of the noisier arm's range — precisely the kind of number that should not be
cherry-picked, and is not treated as the result here.

**EMA (D3) remains a null.** −0.13 pt vs D1 at seed 1234, having been +0.57 on
HWS across two seeds in Phase C. It was not promoted to the seed round because D2
and D3 both sat below the two arms above; it stays where Phase C left it —
free at inference, never yet worth adopting.

---

## 5. Comparability, and whether the test-2400 gate is warranted

### What separates these numbers from the FUTO ceiling

| | our best (`D1` seed-mean) | FUTO ceiling | gap |
|---|---|---|---|
| overall t1 | 84.81 | 84.83 | −0.02 |
| t3 | 91.01 | 91.04 | −0.03 |
| t5 | 92.33 | 92.08 | **+0.25** |
| ≤3 t1 | 88.01 | 89.57 | **−1.56** |
| 4+ t1 | 83.15 | 82.40 | **+0.75** |

**Three reasons this table is not yet a claim**, each of which pushes a different
way and none of which is small enough to ignore:

1. **Split.** Ours is val-9918; the ceiling is test-2400. The one arm measured on
   both — `r2` — scored val **81.57** and test **80.96**, i.e. **test ran 0.61 pt
   below val**. Applying that offset puts `D1`'s expected test t1 at **~84.2**,
   which is **below** the 84.83 ceiling, not level with it.
2. **Contamination.** T3 has no session exclusion, so every val row's contributor
   is in training (§2). T1 is better but not clean (its contributor-clean val
   subset is 46 rows). Both numbers are inflated by an unknown amount relative to
   a generalization claim. Against the *published FUTO baselines* this is the
   conservative direction — they trained on the literal holdout traces and we
   removed them bit-exactly — but it is not conservative in absolute terms.
3. **Lexicon.** Our trie is 146,964 words after a-z normalization; the published
   baselines used 131,544. A larger lexicon means more confusable candidates, so
   this one pushes our numbers *down* relative to the table. It is the only one of
   the three in our favour, and it is unquantified.

### Recommendation: **do not spend the test-2400 gate yet**

The gate bar was a seed-mean full-val t1 *comfortably* above 84. The measured
value is **84.81** — over the line, but the val→test offset eats 0.61 of the
0.81 pt of margin, leaving an expected test result of ~84.2 against a target of
84.83. Decoding test now most likely buys a documented near-miss and burns the
seal for a number the val evidence already predicts. Three things would change
that assessment, in descending order of expected value:

1. **Close the ≤3 gap.** It is the *only* stratum where we are behind, and we are
   behind by 1.56 pt while leading 4+ by 0.75. Short words are where the lexicon
   prior should dominate and where every sharpening lever this campaign has tried
   (ConvNeXt, batch/lr scale-up, raw FUTO volume) has hurt. Two untried,
   directly-targeted options: a length-conditioned score term in the beam (the
   Phase-2 refinement head gave **+1.00 on ≤3** for +0.00 aggregate — it was
   closed on the aggregate, but its stratum profile is exactly the shape needed
   now), and a T1/T3 blend, since T1 leads ≤3 at all three seeds and T3 leads 4+
   at all three.
2. **Keep scaling capacity.** ch 96 → ch 128 is the only architecture lever that
   has ever produced a gain here (+1.07 pt on T3 at seed 1234), at 0.31 → 0.49 ms
   single-thread CPU. ch 192/256 is untested and the encoder is not the per-swipe
   bottleneck — the 100-wide trie beam over 147 k words is.
3. **Fix the selection noise** (§3): a 5,000-row prefix, or final-step selection
   under the cosine-to-zero schedule. Worth ~0.2 pt for ~3 s per validation.

If the gate is spent anyway, spend it on **`D1` at all three seeds**, reporting
all three test numbers and their mean — not on the best seed. A single test
number from the noisier of the two arms would be the least informative thing this
campaign could produce.

### Latency

Single-thread, batch-1, fixed-shape ONNX Runtime CPU, 300 runs, machine idle.

| config | params | mean ms | p90 ms | vs `r2` |
|---|---|---|---|---|
| `r2` / `phaseA-T2` (ch 96 reference) | 394,114 | 0.307 | 0.320 | — |
| `phaseD-D0`, `phaseD-T1bridge` (ch 96) | 394,114 | 0.31–0.32 | 0.32–0.34 | ~0 % |
| **`phaseD-D1` and every ch-128 arm** | 689,282 | **0.48–0.52** | 0.50–0.56 | **+60 %** |
| `phaseD-D2` (ConvNeXt) | 570,818 | 0.48 | 0.49 | +57 % |

ch 128 costs ~60 % more encoder time for +1.07 pt (single seed) — a far better
trade than Phase B's ConvNeXt (+48 % for −1.31 pt), and still ~0.2 ms per swipe
on a desktop core. It is a real cost on a phone's little core and should be
re-measured there before shipping, but the trie beam dominates the per-swipe
budget either way.

---

## 6. Summary of decisions

* **Adopt beam-t1 checkpoint selection.** Verified exact against the committed r2
  numbers, ~10 % wall-clock overhead. Raise the prefix to ~5,000 rows in Phase E.
* **Adopt ch 128.** The only architecture lever in four phases that has produced a
  gain, at a defensible latency cost.
* **Do not adopt T3.** It is indistinguishable from T1 at matched capacity
  (paired t(2) = 1.31), needs 2.7× the training rows and all contamination
  controls removed to get there, and has ~4× T1's seed variance. Keep it as a
  *benchmark* tier only, always with the §2 disclosure attached.
* **Do not adopt the ConvNeXt trunk (D2) or EMA (D3).** D2's regression reproduces
  under beam selection, retiring the mis-selection hypothesis; D3 is a null.
* **Do not spend the test-2400 gate yet** — see §5.

---

## 7. Reproduction

```bash
# tier (adds T3; no session exclusion, both hash conventions for exact dedup)
python build_tiers.py --tiers t3
python prepare_data.py --extra-train data/tier_t3.jsonl --out-name train_t3 --jobs 16

# one arm (beam selection is on by default at --beam-val-rows 2000)
python train.py --train-npz train_t3.npz --run-name phaseD-D1 --ch 128 --embed-hid 128 \
                --total-steps 94000 --val-every 3000 --batch 256 --lr 3e-3 \
                --weight-decay 0.01 --warmup 1000 --seed 1234 --beam-jobs 12
python export_onnx.py --ckpt ckpt/phaseD-D1/best.pt \
                      --out ckpt/phaseD-D1/ctc_swipe_encoder.onnx

# eval (per-source, per-stratum, latency; test-2400 is refused)
python eval_arms.py --arms phaseD-D0,phaseD-D1,phaseD-D2,phaseD-D3,\
phaseD-T1bridge,phaseD-T1bridge128 --own-mask T0 --also-masks "" --latency-runs 300
```

> **Footnote — the `clean[T0]` column.** `eval_arms.py` reports every Phase-D arm
> on the 164-row T0-clean mask (`D0` 86.59, `D1` 89.02 / 89.63 / 89.02 by seed,
> `T1bridge128` 87.20 / 88.41 / 90.85, `D2` 85.98, `D3` 89.63). **These numbers
> decide nothing and are recorded only for continuity with Phase A.** The mask
> selects val rows whose contributor is absent from *T0*, which says nothing about
> T1 or T3 — both of which contain those contributors. At n = 164 the standard
> error on an ~88 % rate is ±2.5 pt, wider than every difference in the table.
> T3 is contributor-dirty by design and has no clean subset at all.
