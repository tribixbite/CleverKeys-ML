# Post-decode audit — the sealed test-2400 result, verified

Companion to `AUDIT_PREDECODE.md` (commit `77f856d`), which gated this decode.
Same auditor, same adversarial stance: credit only for verified confirmations or
concrete refutations.

**What was executed.** Six `eval_beam.py` runs, one per checkpoint, exactly as
pre-registered in `AUDIT_PREDECODE.md` §E: preset `1.05,1.1,0.2,0.3734,0.9882`, beam
width 100, `top_k` 8, the 146,964-word STRIP trie, the committed
`ckpt/<arm>/ctc_swipe_encoder.onnx`, no warm-ups, no second preset, per-trace dumps to
`ckpt/<arm>/test2400_e1.{log,jsonl}`.

Every number below marked **[measured]** was recomputed by this audit from the per-trace
dumps or from a fresh re-decode. **No log footer was trusted.**

| # | Item | Verdict |
|---|---|---|
| 1 | Aggregates, strata, OOV, seed-mean arithmetic recomputed from dumps | **CONFIRMED** |
| 2 | Per-source (futo/hws) split | **CONFIRMED — and it is the largest disclosure** |
| 3 | Single-decode integrity at the registered preset | **CONFIRMED** (bit-for-bit, 100/100 sampled rows) |
| 4 | The 145 T3-leaked test rows | **CONFIRMED harmless — stronger than on val** |
| 5 | Does the evidence support the claim *as registered*? | **YES** — see §7 |

---

## 1. The verified test table — CONFIRMED

Recomputed from the per-trace `test2400_e1.jsonl` dumps [measured]. Structural checks
first: each dump holds exactly **2,400** rows with contiguous `idx` 0…2399; all six dumps
carry an **identical row order and identical target words**; the length strata are
**≤3 n=815 / 4+ n=1,585**, matching the published bar's own n's exactly.

**OOV handling verified and conservative** [measured]. **86** of the 2,400 target words
are absent from the 146,964-word trie and therefore unreachable by the beam. All 86 carry
`rank = -1` and are **counted as misses**, not excluded. Total `rank = -1` is 150 (86 OOV
+ 64 in-vocabulary but outside the top-8); the remaining rank histogram is
{0: 2131, 1: 74, 2: 16, 3: 13, 4: 9, 5: 2, 6: 3, 7: 2}. Nothing is silently dropped.

### ch 192 (`phaseE-FINAL`, 1,525,378 params, 0.877 ms)

| seed | t1 | t3 | t5 | ≤3 (n=815) | 4+ (n=1585) | all five? |
|---|---|---|---|---|---|---|
| 1234 | 88.79 | 92.54 | 93.46 | 91.53 | 87.38 | yes |
| 4321 | 87.88 | 92.71 | 93.50 | 90.92 | 86.31 | yes |
| 7777 | 88.42 | 92.71 | 93.54 | 91.66 | 86.75 | yes |
| **seed-mean** | **88.36** | **92.65** | **93.50** | **91.37** | **86.81** | **yes** |
| seed sd | 0.46 | 0.10 | 0.04 | 0.39 | 0.54 | |
| **worst seed** | 87.88 | 92.54 | 93.46 | 90.92 | 86.31 | **yes** |
| the bar | 84.83 | 91.04 | 92.08 | 89.57 | 82.40 | |
| **Δ (mean)** | **+3.53** | **+1.61** | **+1.42** | **+1.80** | **+4.41** | |

### ch 128 (`phaseE-E3b-hws3x`, 689,282 params, 0.455 ms)

| seed | t1 | t3 | t5 | ≤3 | 4+ | all five? |
|---|---|---|---|---|---|---|
| 1234 | 88.04 | 92.08 | 92.96 | 91.29 | 86.37 | yes |
| 4321 | 87.83 | 92.46 | 93.12 | 90.55 | 86.44 | yes |
| 7777 | 87.88 | 92.46 | 92.92 | 91.41 | 86.06 | yes |
| **seed-mean** | **87.92** | **92.33** | **93.00** | **91.08** | **86.29** | **yes** |
| seed sd | 0.11 | 0.22 | 0.11 | 0.46 | 0.20 | |
| **worst seed** | 87.83 | 92.08 | 92.92 | 90.55 | 86.06 | **yes** |
| **Δ (mean)** | **+3.09** | **+1.29** | **+0.92** | **+1.51** | **+3.89** | |

**Seed-mean arithmetic verified** by recomputation from per-seed values; both means
reproduce the reported 88.36/92.65/93.50/91.37/86.81 and 87.92/92.33/93.00/91.08/86.29
exactly. Log footers agree with the dumps on every metric.

**All five bars clear on the seed-mean for both configurations, and — the stronger
statement — on every individual seed.** The minimum over six independent runs still
clears all five. That is a materially better outcome than the val gate, which had t5
riding at +0.28.

### The pre-registered expectation was wrong in our favour, and that must be said

`AUDIT_PREDECODE.md` §E registered: *"Expected outcome is a comfortable pass on t1 and 4+,
a likely pass on t3 and t5, and a coin-flip on ≤3"*, based on r2's val→test offset of
**−0.61** t1. The measured offset is **+0.30** (ch192) and **+0.04** (ch128) [measured] —
test ran at or *above* val, not below. Meanwhile the bar itself drops from val to test on
four of five metrics (t1 −0.69, t3 −0.50, t5 −0.72, 4+ −1.17; ≤3 +0.28). The margins
therefore widened on test for reasons roughly half attributable to the bar moving, not
only to the model. The registered ≤3 coin-flip resolved positive (+1.80), but see §5 — it
is the least resolved of the five.

---

## 2. Per-source split — CONFIRMED, and it is the disclosure that matters most

Pre-registered disclosure, from `cache/holdout_source_tags.json["test"]`
(futo 1,217 / hws 1,183, matching `DATA_TIERS.md` §1) [measured]:

| config | FUTO half (n=1,217) | HWS half (n=1,183) | spread |
|---|---|---|---|
| ch 192 | t1 **95.32** (t3 99.07, t5 99.48, ≤3 96.93, 4+ 94.27) | t1 **81.21** (86.05, 87.35, 83.48, 80.30) | **14.11 pt** |
| ch 128 | t1 **95.07** (98.88, 99.32, 96.72, 94.00) | t1 **80.56** (85.60, 86.50, 83.09, 79.55) | **14.51 pt** |

The aggregate 88.36 is the average of a 95.3 and an 81.2. On the How-We-Swipe half alone
the model scores **3.6–4.3 pt below the aggregate bar** — a comparison that is not
strictly valid, because the bar is published only in aggregate and cannot be decomposed,
but which shows that a single headline number over a two-distribution mixture with a
14-point internal spread is fragile to any mix difference. Val is 49.8 % FUTO and test
50.7 % FUTO; that 0.9 pt mix shift alone accounts for roughly +0.13 pt of the val→test
movement.

---

## 3. Single-decode integrity — CONFIRMED

**Artifact state.** Exactly 12 files in the workdir were created or modified after
00:50 [measured]: the 6 `test2400_e1.jsonl` and 6 `test2400_e1.log`. Nothing else — no
sweep json, no alternative-preset dump, no second output name. The only other test
artifact anywhere under `~/ctc-train` remains `ckpt/r2/test2400_onnx.{jsonl,log}` from
2026-08-07 06:20, the disclosed pre-campaign r2 decode.

**Timing.** The six mtimes are strictly monotonic and in exactly the registered loop
order: 01:03:38.47, 01:04:11.62, 01:04:39.64, 01:05:07.99, 01:05:35.08, 01:06:07.59.
Gaps of 33.1 / 28.0 / 28.4 / 27.1 / 32.5 s reconcile to the second with each log's own
reported throughput (78–94 tr/s over 2,400 rows) plus one trie load per run. The 149-s
window is fully accounted for; there is no slack for a seventh run, and a re-run of any
arm would have broken the monotonic order.

**Bit-for-bit re-decode.** The decisive check. From two different checkpoints I sampled
rows at random (seed 20260808) from the already-decoded dumps, re-featurized the raw test
traces, re-ran the committed ONNX graph and the unmodified vendored `futo_viterbi_beam`,
and compared the **entire `topk` list — every candidate string and every float32 score —
against the dump** [measured]:

| checkpoint | rows sampled | `greedy` matches | `topk` bit-for-bit @ **registered** preset | `topk` bit-for-bit @ published preset | `rank` matches |
|---|---|---|---|---|---|
| `phaseE-FINAL-s1234` | 60 | 60/60 | **60/60** | **0/60** | 60/60 |
| `phaseE-E3b-hws3x-s7777` | 40 | 40/40 | **40/40** | **0/40** | 40/40 |

100/100 sampled rows reproduce exactly from the named checkpoint at the registered
preset, and **0/100** reproduce at the published preset — so the match is informative,
not vacuous. The dumps are the genuine output of those checkpoints at
`1.05,1.1,0.2,0.3734,0.9882`. No preset substitution, no checkpoint swap, no post-hoc
editing of the dumps.

The one thing mtimes cannot exclude is an earlier run overwritten in place before
01:03:38. The perfectly-packed 149-s window, the registered ordering, and the absence of
any selection artifact make that implausible, but it rests on operator honesty rather
than on a mechanism — the same structural gap flagged pre-decode (`eval_beam.py` still
has no guard).

---

## 4. The T3-leaked test rows — CONFIRMED harmless, more clearly than on val

`AUDIT_PREDECODE.md` §3 refuted the "holdout traces removed bit-exactly" claim: **145 of
2,400 test rows (6.04 %)** sit in `train_t3` with a bit-identical input tensor and label,
because the dedup key uses the raw word (`'arabian.'`) while the training label uses the
a–z-normalized one (`'arabian'`). Recomputed on the actual test decode [measured, ch192
seed-mean; ch128 in brackets]:

| subset | n | t1 | ≤3 t1 | 4+ t1 |
|---|---|---|---|---|
| FUTO half, **leaked** (all 145 are FUTO-source) | 145 | **91.49** [91.49] | 83.33 (n=12) | 92.23 (n=133) |
| FUTO half, **not leaked** | 1,072 | **95.83** [95.55] | 97.28 | 94.72 |

Rows the model trained on score **4.34 pt lower** than comparable rows it did not — the
same sign as on val (−1.07), and larger. There is no memorization signal; if anything the
duplicated traces are harder ones.

**Bound on inflation, and the check that actually settles it** [measured]: excluding all
145 leaked rows,

| config | n | t1 | t3 | t5 | ≤3 | 4+ | all five, seed-mean | all five, **every seed** |
|---|---|---|---|---|---|---|---|---|
| ch 192, leak removed | 2,255 | 88.16 | 92.36 | 93.17 | 91.49 | 86.32 | **yes** | **yes** |
| ch 128, leak removed | 2,255 | 87.69 | 92.03 | 92.67 | 91.20 | 85.74 | **yes** | **yes** |

The whole effect of removing them is **−0.20 t1** (ch192) / −0.23 (ch128), and that is
composition — dropping easy FUTO rows — not memorization. **Even attributing the entire
−0.20 to leakage, all five bars still clear for both configurations on every seed.** The
defect is real and must be disclosed and fixed, but it cannot be the reason the gate
passed. This is a cleaner result than on val, where the leak-removed subset dipped below
the t5 bar.

---

## 5. Statistical resolution per bar — the honest caveat

Unpaired binomial SE against the published bar treated as a fixed estimate on the same
rows (conservative; FUTO's per-row output is unavailable, so a paired test is impossible)
[measured]:

| metric | n | ch192 Δ | SE | **z** | ch128 Δ | **z** |
|---|---|---|---|---|---|---|
| t1 | 2,400 | +3.53 | 0.98 | **3.6** | +3.09 | **3.1** |
| 4+ | 1,585 | +4.41 | 1.28 | **3.4** | +3.89 | **3.0** |
| t3 | 2,400 | +1.61 | 0.79 | 2.0 | +1.29 | 1.6 |
| t5 | 2,400 | +1.42 | 0.75 | 1.9 | +0.92 | 1.2 |
| ≤3 | 815 | +1.80 | 1.45 | 1.2 | +1.51 | 1.0 |

**Two of the five bars are statistically resolved** (t1 and 4+, both ≳3σ, for both
configurations). t3 and t5 are ~2σ and 1.2–1.9σ; **≤3 is 1.0–1.2σ and is not resolved at
all** — the ≤3 pass is a point estimate on 815 rows, exactly the fragility flagged
pre-decode. The correct statement is: *all five point estimates clear, on every seed;
two clear with statistical confidence, three are positive but within the noise the
n's admit.*

Seed variance is not the limiting factor (sd 0.04–0.54); row-sampling on a 2,400-row
split is.

---

## 6. The eight required disclosures, instantiated

1. **The preset asymmetry — quantified, and still the largest threat.** Our decode preset
   was fitted on val-9918 by a five-parameter grid search; the FUTO ceiling was measured
   at its own published preset. Measured control on val, ch192 3-seed mean at the
   **published** `encoderOnly` preset: **85.78 / 91.66 / 92.67 / 88.10 / 84.58** against
   the val bar 85.52 / 91.54 / 92.80 / 89.29 / 83.57 — **3 of 5 bars, with t5 −0.13 and
   ≤3 −1.19**, and t1/t3 clearing by 0.26/0.12, under one standard error. The tuning is
   worth **+2.29 pt top-1** on this exact model. Whether FUTO's emissions have comparable
   headroom under the same sweep is untested and, with no FUTO weights on this machine,
   untestable here. **A test pass does not resolve this**, and no headline may omit it.
2. **The T3 dedup defect.** 588 val and 145 test rows are in the training tier with a
   bit-identical input tensor and label; the dedup key uses the raw word, the training
   label the a–z-normalized word. Measured effect ≈ 0 (leaked test rows score 4.34 pt
   *below* comparable non-leaked ones; removing all 145 costs 0.20 pt and all five bars
   still clear on every seed). `PHASE_D.md` §2's "caught every match on both sides" and
   `PHASE_E.md` §5's "removed bit-exactly" are false as written and must be corrected.
3. **Contributor contamination.** T3 applies no session or participant exclusion
   (`build_tiers.py:549`); every contributor of every val and test row is in training, and
   3× HWS oversampling triples the exposure of the more contaminated corpus (98.4 % of HWS
   holdout rows share a participant with training). **No contributor-clean subset of val or
   test exists for this model.** These are benchmark numbers, comparable with published
   FUTO figures; **they are not a generalization claim about an unseen user.**
4. **The counter-asymmetry, in FUTO's favour.** 5,273 of the 12,299 unique holdout traces
   (43 %) are bit-exactly in the HF *train* split FUTO trained on; 0 in HF dev/test. We
   remove ours (bar the 145 above). The app repo's own eval notes describe the split as
   FUTO-held-out, which is incorrect.
5. **Statistical resolution.** §5: two of five bars resolved (t1, 4+); t3 ~2σ, t5
   1.2–1.9σ, **≤3 1.0–1.2σ, not resolved**.
6. **Lexicon.** Our runs and the val bar use the *same* 146,964-word STRIP trie — the
   `README.md` claim that our larger lexicon makes the numbers conservative does not apply
   to the val comparison and must be corrected. The test bar (84.83) was published on the
   131,544-word DROP trie and re-measured **unchanged** on the 146,964 one, so the overall
   test comparison is trie-neutral; its **strata were not republished post-fix**, so ≤3 and
   4+ on test are compared across normalizers.
7. **Seal hygiene.** One decode per checkpoint, verified bit-for-bit at the registered
   preset (§3). Prior contact with the split: the disclosed r2 decode (2026-08-07 06:20)
   and an **undisclosed 120-row smoke decode** at 04:20 EDT with a toy 898-word trie.
   `eval_beam.py` and `sweep_scoring.py` still carry **no test guard**; the seal was held
   by discipline. **7 traces are bit-exactly shared between val-9918 and test-2400**
   (0.29 % of the sealed set), so a sliver of test was inside the tuning corpus.
8. **Arm selection used full val.** The preset sweep (val `0:4959`) and checkpoint
   selection (5,000-row prefix) respected a holdout, but which arms were *stacked* — E3b,
   E5, ch192, and the rejections of T4 and E2 — was decided on full val-9918 tables.

Also disclose, from §1: the pre-registered val→test expectation (−0.61) was wrong in the
model's favour (+0.30 / +0.04), and roughly half the widening of the margins comes from
the bar itself falling on test.

---

## 7. Final verdict

**Does the evidence support the claim AS REGISTERED? — YES.**

The registered claim was, verbatim:

> On the sealed 2,400-row test split, the Phase-E configuration, decoded at the val-tuned
> E1 preset, is compared against FUTO's published encoder+refinement ceiling decoded at
> FUTO's published preset. A pass is not a claim of superiority on equal footing — the
> presets are not matched, and no attempt to re-tune FUTO's preset was possible.

Against that claim the evidence is clean and, on the points I could attack, stronger than
the val gate was:

* every headline number recomputed from per-trace dumps rather than log footers, and it
  reproduces;
* all five bars clear **on every one of six independent runs**, not merely on the mean;
* OOV targets are counted as misses, so the metric is not flattered by its own definition;
* the decode is provably a single pass per checkpoint at the registered preset,
  bit-for-bit on 100/100 sampled rows, with 0/100 matching under any other preset;
* the contamination defect I refuted pre-decode is measurably not the cause — removing
  every leaked row costs 0.20 pt and changes no verdict on any seed.

**What the evidence does not support, and what must never be written:** that this model
beats FUTO's decoder *on equal footing*. Our preset is tuned on the holdout family; theirs
is not. At the published preset the same model clears only 3 of 5 val bars. That
asymmetry is worth ~2.3 pt — comparable to the entire test margin on t1 and larger than
the margin on t3, t5 and ≤3 — and a test decode was never capable of resolving it. Three
of the five test bars are also within ~2σ, ≤3 within ~1σ.

**Recommended headline, and nothing stronger:** *"On the sealed test-2400 split, both
configurations exceed all five published FUTO-ceiling numbers on every seed — ch192
88.36/92.65/93.50 (≤3 91.37, 4+ 86.81) and ch128 87.92/92.33/93.00 (≤3 91.08, 4+ 86.29)
against 84.83/91.04/92.08/89.57/82.40 — decoded at a preset tuned on val while the
ceiling is quoted at its own published preset. On matched (published) presets the same
model clears 3 of 5. These are benchmark numbers on a contributor-contaminated tier, not
a generalization claim."*

**Required follow-up, independent of the claim:** fix `build_tiers.hash_row` and
`prepare_data.trace_hash` to key on `normalize_word(word)`; rebuild T3/T3hws; replace the
filename-substring guard with a row-count assertion in `eval_beam.py` and
`sweep_scoring.py`; and land the missing `ctc_golden.json` fixture so the Kotlin
featurizer-parity test actually runs.

**The seal is now spent. No further decode of test-2400 is legitimate** for any variant,
preset, checkpoint or stratum — including a "fair rematch" at the published preset, which
must be argued from the val control in §6.1.
