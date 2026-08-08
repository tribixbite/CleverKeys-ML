# Pre-decode adversarial audit — gating the test-2400 seal

**Auditor:** independent adversarial pass, 2026-08-08. **Mandate:** try to refute the
Phase-E claim before an irreversible test decode. Credit only for verified
confirmations or concrete refutations.

**Ground rules observed.** Read-only on everything except this file. The CleverKeys
app repo was read, never written. **test-2400 was never decoded, featurized for
scoring, or inspected row-wise** — the only contact with it was hashing
`cache/test.npz` / `data/test_hwsfuto.jsonl` for the contamination check (item 3),
which produces counts, not predictions. No decode output for test exists from this
audit.

Every number below marked **[measured]** was computed by this audit from the
artifacts. Numbers marked *[read]* are quotations and carry no evidentiary weight.

Artifacts produced (scratchpad, not committed): 6 × full-val `eval_beam.py` per-trace
dumps at the E1 preset, 3 × full-val dumps at the published preset, 6 independent ONNX
re-exports, leak masks, and a raw-corpus duplicate scan.

---

## Verdict summary

| # | Item | Verdict |
|---|---|---|
| 1 | Headline val numbers reproduce from artifacts | **CONFIRMED** (exact, 30/30 numbers) |
| 2 | No test leakage into tuning / model selection | **CONFIRMED with qualifications** |
| 3 | No training-tier row matches a holdout trace | **REFUTED** — 588 val + 145 test rows are in `train_t3`; measured accuracy impact ≈ 0 |
| 4 | The bar is quoted correctly | **CONFIRMED**; lexicon caveat is **misattributed** |
| 5 | Fairness asymmetries | **QUALIFIED — the decisive finding.** The gate is produced by the val-tuned preset; at the published preset only 3 of 5 bars clear |
| 6 | Seed protocol | **CONFIRMED** |
| 7 | Latency claims | **CONFIRMED** (measured slightly better) |
| 8 | I/O contract / "drop-in" | **CONFIRMED at the contract level, QUALIFIED as a shipping claim** |

**Overall: GO for the sealed decode — conditional on the disclosures in §D and on
restating the headline claim per §E.** The measurement machinery is sound and the val
numbers are real. What is *not* established, and cannot be established by decoding
test, is that this model beats FUTO's decoder on equal footing.

---

## 1. Reproduction of the headline val numbers — CONFIRMED, exactly

**Method (deliberately not the campaign's path).** For all six final checkpoints I
re-exported ONNX from `best.pt` with `export_onnx.py`, then ran `eval_beam.py` — the
real per-row decoder, not `sweep_scoring.py`'s analytic re-scoring — over all 9,918 val
rows at the E1 preset `1.05,1.1,0.2,0.3734,0.9882`, dumping per-trace ranks with
`--out`. All strata were recomputed by me from those dumps.

**Export determinism [measured].** All six re-exports are **byte-identical (sha256)** to
the committed `ckpt/<arm>/ctc_swipe_encoder.onnx`. Parameter counts confirmed:
ch 192 = **1,525,378**, ch 128 = **689,282**. Sliced-view torch/ONNX parity 1.6e-05 to
3.1e-05, argmax agreement 100/100 on every arm.

**Full val-9918, E1 preset [measured]:**

| arm | t1 | t3 | t5 | ≤3 (n=3389) | 4+ (n=6529) |
|---|---|---|---|---|---|
| `phaseE-FINAL-s1234` (ch192) | 88.22 | 92.23 | 93.08 | 91.15 | 86.71 |
| `phaseE-FINAL-s4321` | 87.80 | 92.34 | 93.17 | 90.62 | 86.34 |
| `phaseE-FINAL-s7777` | 88.17 | 92.38 | 92.99 | 90.82 | 86.80 |
| **seed-mean** | **88.06** | **92.31** | **93.08** | **90.86** | **86.61** |
| `phaseE-E3b-hws3x` (ch128) | 88.02 | 92.27 | 93.03 | 91.12 | 86.41 |
| `phaseE-E3b-hws3x-s4321` | 87.63 | 92.19 | 92.92 | 90.85 | 85.95 |
| `phaseE-E3b-hws3x-s7777` | 87.98 | 92.23 | 92.94 | 90.97 | 86.43 |
| **seed-mean** | **87.88** | **92.23** | **92.97** | **90.98** | **86.27** |

Every per-seed figure matches `PHASE_E.md` §4/§5 **to the digit**, and both seed-means
match the claimed 88.06/92.32/93.08/90.86/86.62 and 87.88/92.23/92.96/90.98/86.26 to
within one hundredth (rounding of the mean). Stratum n's 3,389 / 6,529 confirmed.

**Fast-path equivalence — verified far beyond the one config asked for.** The campaign's
sweep harness re-scores a terminal beam analytically; the audit's numbers come from the
unmodified vendored beam run once per row. They agree on **6 configs × 5 metrics = 30
numbers, exactly**, at the *tuned* preset on *full* val. The identity is not merely
plausible from the source, it is measured.

**Holdout-half table also reproduces [measured].** Rows `4959:9918`, ch192 seed-mean:
**87.58 / 92.03 / 92.85 / 90.67 / 85.98** — identical to `PHASE_E.md` §5. Restricting
further to rows `5000:9918` (removing the acknowledged 41-row overlap with the selection
prefix) changes nothing: 87.58 / 92.02 / 92.85 / 90.65 / 86.00.

**Paired capacity table reproduces [measured].** ch192 − ch128 per seed:
t1 +0.20/+0.17/+0.19 (mean **+0.19**), t3 −0.04/+0.15/+0.15, t5 +0.05/+0.25/+0.05,
≤3 +0.03/−0.23/−0.15 (mean **−0.12**), 4+ +0.30/+0.39/+0.37 (mean **+0.35**). Matches §5.

**Verdict: CONFIRMED.** I could not move any headline val number.

---

## 2. Tuning leakage — CONFIRMED, with three qualifications

**No test decode during the campaign.** Independent forensics over filesystem
timestamps, all 48 checkpoint logs, `~/.bash_history`, and the complete Claude
transcript record found exactly **one** full test decode in the workdir:
`ckpt/r2/test2400_onnx.{jsonl,log}` (2400 lines, 06:19–06:20 EDT, `80.96 / 89.79 /
91.12`) — the disclosed pre-campaign r2 run, produced ~90 min before Phase A began.
Every `phase*` result json carries `n_val: 9918`; no phase checkpoint dir holds a test
artifact; the only 2400-row jsonl in the workdir is r2's. `cache/test.npz`'s single
recorded post-creation read (07:43:28) is matched to the second to a
`len(d['target_lengths'])` row-count print.

**Qualification 2a — an undisclosed 120-row test decode.** At 04:20:53 and 04:21:12 EDT,
~3.5 h before Phase A, `eval_beam.py --test data/test_hwsfuto.jsonl --limit 120` was run
twice against a smoke checkpoint with an **898-word** toy trie (37.50 % t1). Those 120
real test traces were observed. Nothing derived from them appears anywhere and the trie
makes the output worthless for tuning, but `RESULTS.md`/`PHASE_*.md` state the split was
decoded only by r2. **Disclose.**

**Qualification 2b — the guard is weaker than the docs claim.** `PHASE_A/D/E` all state
"`eval_arms.py` refuses any split whose filename contains `test`". True, and `train.py`
/ `train_refine.py` carry the same check — but it is a `Path(...).name` substring test,
defeated by a rename or symlink, and **`eval_beam.py` (the script that performed the only
real test decode) and `sweep_scoring.py` have no guard at all**. The seal held by
operator discipline, not by enforcement. `eval_arms.py` does have an incidental stronger
control: it asserts the source-tag length equals the row count, which a 2400-row split
would fail.

**The untouched-rows claim — CONFIRMED [measured].** The E1 preset was swept on val
`0:4959` with confirmation on `4959:9918`; checkpoint selection used a 5,000-row prefix
(verified in every run banner: `beam-val: 5000 rows … preset (0.4056, 0.0176, 0.9866,
0.4234, 1.0382)`). Overlap between the selection prefix and the "untouched" half is
therefore rows 4959–4999 = **41 rows**, exactly as `PHASE_E.md` states, and excluding
them moves nothing (above).

**Qualification 2c — "untouched" covers the preset and the checkpoint, not the
configuration.** Every *arm* decision — adopt E3b, adopt E5, reject T4, reject E2, take
ch192 to three seeds — was made by reading **full val-9918** tables (`PHASE_E.md` §3/§4).
So rows `4959:9918` are untouched by preset fitting and checkpoint selection but *did*
inform which configuration was stacked. The holdout-half table is a strict estimate for
two of the three selection layers, not all three. The bias is small (a handful of binary
choices, each measured at ≤1 pt) but it is not zero, and the doc's framing ("rows that
were never used for anything") overstates it.

---

## 3. Data contamination — REFUTED as stated; impact measured at ≈ 0

This is where independent hashing found something the campaign's own checks could not.

### 3a. The claim

`PHASE_D.md` §2: dedup was applied under both hash conventions, "the `(word, x, y)` form
… **caught every match on both sides** … `prepare_data.py` then re-applies its own dedup
independently and reports `dropped_cross_split_duplicate = 0`, which is the cross-check
that the tier build missed nothing." `PHASE_E.md` §5 rests the whole T3-vs-FUTO
comparability argument on "the holdout traces are removed bit-exactly".

### 3b. What I measured

I hashed the **model's actual input and label** — the float32 `[2,64]` featurized path
plus the a–z-normalized word, i.e. exactly what `train.py` consumes — for every cached
tier and for val/test.

| tier | rows | rows identical to a **val** row | rows identical to a **test** row |
|---|---|---|---|
| `train_t1` | 372,726 | 0 | 0 |
| `train_t4` | 764,771 | 0 | 0 |
| `train_t3hws` | 76,748 | 0 | 0 |
| **`train_t3`** | **1,005,336** | **588** | **145** |

**588 of 9,918 val rows (5.93 %) and 145 of 2,400 test rows (6.04 %)** are present in
`train_t3` with a bit-identical input tensor and an identical training label. `train_t3`
is the tier every Phase-D and Phase-E model — including both final configurations — was
trained on.

### 3c. The mechanism, pinned to a concrete defect

A raw-corpus scan of all 1,007,723 `tier_t3.jsonl` rows [measured]:

```
val matches under (raw_word, x, y, t)      -> 0      <- what prepare_data checks
val matches under (norm_word, x, y)        -> 570
val matches under (norm_word, x, y, t)     -> 570
test: 0 / 139 / 139 respectively
```

Timestamps are irrelevant; **the word string is the whole story**. Examples [measured]:

```
val_idx=7468  val 'arabian'    t3 'arabian.'     173 identical points
val_idx=1172  val 'settlers'   t3 'settlers:'    109 identical points
val_idx=7727  val 'negatively' t3 'negatively,'  190 identical points
val_idx=6908  val 'languages'  t3 'languages,'   154 identical points
```

The canonical splits were built by FUTO's normalizer, which strips trailing punctuation;
the T3 build reads the raw corpus and keeps it. Both dedup keys are blind to this:
`build_tiers.hash_row` lowercases (`build_tiers.py:89`) but does not strip non-a–z, and
`prepare_data.trace_hash` uses the **raw** word (`prepare_data.py:119`). Training labels,
however, go through `normalize_word` (a–z strip, `prepare_data.py:41-47`). **The dedup
key and the label key disagree**, and every punctuation-suffixed duplicate of a holdout
trace walked straight through. `dropped_cross_split_duplicate = 0` is not a cross-check
that the build missed nothing; it is two keys sharing the same blind spot.

T1/T2/T4 are unaffected because they are built from the already-normalized curated pool.
This also means the **E3a (T3 vs T4) comparison is confounded**: T3 carries 588 memorized
val answers, T4 carries none.

### 3d. Impact — measured, and it is negligible

The obvious inference ("6 % of val is memorized, the numbers are inflated") does **not**
survive the control. All 588 leaked val rows are FUTO-source (0 are How-We-Swipe), and
the FUTO half is the easy half. Comparing like with like [measured, ch192 seed-mean]:

| subset | n | t1 | ≤3 t1 | 4+ t1 |
|---|---|---|---|---|
| FUTO half, **leaked** | 588 | 94.05 | 95.74 (n=47) | 93.90 (n=541) |
| FUTO half, **not leaked** | 4,354 | **95.12** | 96.76 | 93.81 |

Rows the model trained on score **1.07 pt lower** than comparable rows it did not; length
-controlled, the difference is +0.09 on 4+ and −1.02 on ≤3. ch128 shows the same sign
(93.59 vs 95.19). A 0.7–1.5 M-parameter model seeing each row once per epoch for 20
epochs out of 1.16 M rows does not memorize single traces. **Bound on the inflation of any
headline number from this leak: < 0.05 pt.**

Consequently the drop when leaked rows are excluded (ch192 t1 88.07 → 87.69, t5 93.08 →
92.74) is a **composition** effect — removing easy FUTO rows shifts the mix toward the
weaker HWS half — not evidence of memorization. It would be wrong to quote a "de-leaked"
number against a bar measured on the full 9,918 rows.

### 3e. Oversampling did not amplify anything — CONFIRMED [measured]

`train_t3hws` (76,748 rows) is a **strict subset of `train_t3`** — all 76,748 keys are
present in T3 — so `--train-npz train_t3.npz,train_t3hws.npz,train_t3hws.npz` is exactly
3× repetition, and the run banners confirm `train 1158832 = 1,005,336 + 2×76,748`. Since
`train_t3hws` has **zero** overlap with val/test under any key, oversampling leaves the
leaked-row count at 588 unchanged. The `PHASE_E.md` §4 description of the mechanism is
accurate.

### 3f. Two corpus facts worth recording [measured]

* `val_hwsfuto.jsonl` contains **11 internally duplicated traces** (9,918 rows → 9,907
  unique) and `test_hwsfuto.jsonl` **1** (2,400 → 2,399).
* **7 traces are bit-exactly shared between val-9918 and test-2400** (identical raw word,
  x, y, t). Since all tuning ran on val, 0.29 % of the sealed set is not sealed. Immaterial
  to any conclusion; disclose for completeness.

### 3g. Provenance verified

`train_t3.npz` provenance [measured]: `featurizer_sha256 eecfd240…` (matches the vendored
`futo_decoder_eval.py`), `layout_sha256 1965ecd5…`, `rows_in 1,007,723 → rows_out
1,005,336`, drops `43 empty / 1 infeasible / 2,343 self-dup / 0 cross-split`. The
independent dedup layer **did** run for T3; it was simply keyed wrong.

**Verdict: REFUTED as stated (the bit-exact-removal claim is false for ~6 % of both
holdouts), CONFIRMED as harmless (measured effect ≈ 0). Both halves must be disclosed.**

---

## 4. The bar itself — CONFIRMED; the lexicon caveat is misattributed

Both vectors are quoted correctly from the app repo's committed eval docs:

* **val-9918** `docs/eval/2026-07-24-test2400-head2head.md:107-119` — FUTO ceiling
  **85.52 / 91.54 / 92.80**, ≤3 t1 **89.29** (n=3,389), 4+ t1 **83.57** (n=6,529),
  macro 87.02. Stratum n's match my dumps exactly.
* **test-2400** same file `:37-44` and `docs/eval/futo-decoder-eval-notes.md:192-201` —
  **84.83 / 91.04 / 92.08**, ≤3 t1 **89.57** (n=815), 4+ t1 **82.40** (n=1,585).

Harness identity confirmed: `futo_decoder_ceiling.py` and `futo_decoder_eval.py` are
**md5-identical** between the app repo and this repo, so the bar and our numbers come
from the same beam (width 100, `top_k` 8). Our `eval_beam.py` default `--top-k 8` matches.

**The lexicon asymmetry runs the other way from how it is documented.** 131,544 and
146,964 are the *same* `en_wordlist.combined` (165,544 entries) under the pre- and
post-contraction-fix normalizers — DROP non-a–z surface forms vs STRIP to a–z. Verified
against the file itself: 165,544 total, **131,544** under DROP, **146,964** under STRIP.
The **val-9918 ceiling (85.52) was measured with the 146,964 STRIP trie** — the identical
trie we use. So `README.md:256-260`'s "our larger lexicon means our numbers are, if
anything, conservative" is **not applicable to the val comparison at all**: there is no
asymmetry there. For test-2400 the published 84.83 used the DROP trie, but the app repo
re-ran it post-fix and measured the ceiling **unchanged at 84.83** (floor 79.25 → 79.29),
so the direction is confirmed neutral there too — with the caveat that the *strata* were
not republished post-fix.

**Verdict: CONFIRMED for the numbers; the caveat text must be corrected — it claims a
conservatism that does not exist.**

---

## 5. Fairness asymmetries — QUALIFIED. This is the finding that matters

### 5a. The preset asymmetry is not a footnote; it *is* the result

`PHASE_E.md` §6 lists "the comparison is now asymmetric in our favour" as open risk #3.
That understates it. I measured the size of the asymmetry directly by decoding all three
ch192 seeds over full val at the **published** `encoderOnly` preset — the same footing on
which the FUTO ceiling was measured at *its* published preset:

| basis (ch192, 3-seed mean, full val-9918) | t1 | t3 | t5 | ≤3 | 4+ | bars cleared |
|---|---|---|---|---|---|---|
| the bar | 85.52 | 91.54 | 92.80 | 89.29 | 83.57 | — |
| **published preset** [measured] | **85.78** | **91.66** | **92.67** | **88.10** | **84.58** | **3 of 5** |
| Δ vs bar | +0.26 | +0.12 | **−0.13** | **−1.19** | +1.01 | |
| **E1 val-tuned preset** [measured] | **88.06** | **92.31** | **93.08** | **90.86** | **86.61** | **5 of 5** |
| the preset lever | **+2.29** | **+0.65** | **+0.41** | **+2.76** | **+2.04** | |

**At the published preset the final model fails the gate on t5 and on ≤3, and clears t1
and t3 by 0.26 and 0.12 pt — well under one standard error.** Every one of the five
passes is manufactured by the val-tuned preset, and the tuning lever (+2.3 pt t1) is
almost as large as the entire claimed margin (+2.55 pt t1).

The campaign's own preset-sensitivity data says the lever is model-dependent but always
large on *our* emissions: +4.57 t1 on `r2`, +2.74 on `phaseD-D1`, +2.29 here [measured].
Whether FUTO's `honorable_sturgeon`+`magic_macaw` emissions have comparable headroom under
a sweep on the same val rows is **unknown and untested**. It is not testable on this
machine — no FUTO `.pte` encoder/decoder exists under `/home/will` (only unrelated
`neural-swipe-typing` and `nema1` models). A fair rematch therefore cannot be run here.

`PHASE_E.md` §5's argument that the transferred E1 preset is "the less contaminated
choice" is sound *internally* (re-tuning on the final model changes ≤0.08 pt — confirmed
by `ckpt/phaseE-FINAL-s1234/sweep.json`, tuned `0.975,1.1,0.3,0.3734,0.5` → 88.23/92.22/
93.00 vs transferred 88.22/92.23/93.08). But "not tuned on *this model*" is not "not
tuned on *this val set*". The preset was fitted on val-9918 rows by a five-parameter grid
search; the bar was not.

**This is the single largest threat to the claim, and no test decode can resolve it.**

### 5b. The counter-asymmetry, in FUTO's favour

`DATA_TIERS.md:76-81`: **5,273 of the 12,299 unique holdout traces (43 %) sit bit-exactly
in the HF *train* split** FUTO trained on; 0 in HF dev/test. The app repo's own eval notes
assert the opposite (`futo-decoder-eval-notes.md:64-65` calls `test_hwsfuto` "held-out
FUTO test"), so the published ceiling carries an unstated home-field advantage that its
own documentation denies. We remove our copies (except the 588/145 of §3). This is real
and it cuts our way — but it is a *training-data* advantage for FUTO, while ours is a
*decode-tuning* advantage, and they are not commensurable.

### 5c. Contributor overlap — no clean subset exists, and the stress test is inconclusive
by construction

`build_tiers.py:549` puts `t3`, `t3hws`, `t4` in the `no_session` set: **no session or
participant exclusion at all.** T3 ingests the whole 939,550-row FUTO corpus and the full
1,338-participant How-We-Swipe release, so **every** contributor of **every** val and test
row is in training. There is no contributor-clean val subset for this model — the
requested "HWS rows from participants absent from training" is the empty set. Tripling the
HWS half triples the exposure of the more contaminated corpus (98.4 % of HWS holdout rows
share a participant with training).

The nearest available proxies are the pre-existing `val_clean_masks.json` masks, which are
clean with respect to *other* tiers' pools, not T3's — they do not remove contamination
for this model, and I report them only for completeness [measured, ch192 seed-mean]:

| subset | n | t1 | t3 | t5 | ≤3 | 4+ | all five? |
|---|---|---|---|---|---|---|---|
| FULL val | 9,918 | 88.07 | 92.31 | 93.08 | 90.86 | 86.61 | yes |
| holdout half `4959:` | 4,959 | 87.58 | 92.03 | 92.85 | 90.67 | 85.98 | yes |
| exact-leak removed | 9,330 | 87.69 | 91.96 | **92.74** | 90.79 | 85.96 | **no (t5)** |
| holdout half ∧ leak-removed | 4,662 | 87.23 | 91.66 | **92.48** | 90.56 | 85.37 | **no (t5)** |
| `val_clean` T2 mask | 9,300 | 87.56 | 91.91 | **92.70** | 90.37 | 86.08 | **no (t5)** |
| `val_clean` T2b mask | 9,669 | 87.91 | 92.17 | 92.95 | 90.77 | 86.38 | yes |
| FUTO half | 4,942 | 94.99 | 98.09 | 98.60 | 96.74 | 93.83 | — |
| **HWS half** | 4,976 | **81.19** | 86.58 | 87.60 | 82.70 | 80.59 | — |

Two things to read here. First, **t5 is the fragile metric on every stricter basis** —
it falls below the bar on four of the six subsets. Per §3d that is composition, not
memorization, and none of these subsets has a matching re-measured bar, so none of them
*refutes* the gate; they show it is not robust. Second, the aggregate win is carried
entirely by the FUTO half (94.99 vs 81.19 on HWS). The bar is not published per-source,
so this cannot be decomposed — but a 13.8 pt intra-holdout spread means the aggregate
comparison is sensitive to any mix difference between val and test (val 49.8 % FUTO,
test 50.7 % FUTO — close, but not identical).

### 5d. Statistical resolution of the gate [measured]

Treating the bar as a fixed published estimate on the same rows and using an unpaired
binomial SE (conservative; a paired test would be tighter, but FUTO's per-row output is
not available):

| metric | Δ vs bar | SE | z |
|---|---|---|---|
| t1 | +2.55 | 0.48 | **5.3** |
| t3 | +0.77 | 0.39 | 2.0 |
| t5 | +0.28 | 0.36 | **0.8** |
| ≤3 | +1.57 | 0.73 | 2.2 |
| 4+ | +3.04 | 0.62 | **4.9** |

Only t1 and 4+ are resolved. **t5 is indistinguishable from the ceiling** (the doc says
this too), and t3 / ≤3 are ~2σ. On test-2400 the n's are ~4× smaller, so every SE roughly
doubles: expect ±0.9 pt on overall metrics, ±1.4 pt on ≤3 (n=815).

**The ≤3 stratum is the specific test-side risk.** It is the only bar that is *higher* on
test (89.57) than on val (89.29), while every other test bar is lower. Our val ≤3 margin
is +1.57 at n=3,389; on test the bar rises 0.28 and our estimator's SE rises to ~1.0.
A ≤3 miss on test is a live possibility and must be pre-registered as such.

---

## 6. Seed protocol — CONFIRMED

All six runs have **distinct** `metrics.jsonl` trajectories (32 validation points each,
different `ctc_loss`, `val_greedy`, and `val_beam_t1` sequences from step 3,000 onward) —
they are independent training runs, not re-evaluations of one model. Distinct `best.pt`
sha/sizes and distinct ONNX hashes confirm it. Run banners show an identical recipe across
all six (`ch`/`embed_hid` per arm, `train 1158832`, 5,000-row beam selection at the
published preset, width 100). The ch192-vs-ch128 comparison is therefore fully paired.

Seed-mean arithmetic verified by recomputation from per-seed values: 88.06 / 92.31 /
93.08 / 90.86 / 86.61 and 87.88 / 92.23 / 92.97 / 90.98 / 86.27. Seed sd on t1 is **0.23**
(ch192) and **0.22** (ch128) — the claimed drop from Phase D's 0.56 holds.

The two non-result run dirs (`phaseE-E4-ch192-s4321`, killed at step 9,000;
`phaseE-E4-ch192-last`) are correctly flagged in `PHASE_E.md` §7 and are not quoted.

---

## 7. Latency — CONFIRMED

Independent measurement, ONNX Runtime CPU, `intra_op = inter_op = 1`, batch 1, fixed
shapes, 50 warmup + 3 × 300 runs, best round, machine idle (load average 1.9):

| config | claimed mean / p90 | **measured mean / p90** |
|---|---|---|
| ch 128 `phaseE-E3b-hws3x` | 0.470 / 0.485 ms | **0.455 / 0.471 ms** |
| ch 192 `phaseE-FINAL-s1234` | 0.898 / 0.914 ms | **0.877 / 0.894 ms** |

Both ~3 % better than claimed; the 1.9× ratio is confirmed (measured 1.93×). The
withdrawal of the earlier 1.54 ms reading as a contention artifact is corroborated —
I reproduced the inflation while six eval jobs were running.

---

## 8. I/O contract and the "drop-in" claim — CONFIRMED at the contract level, QUALIFIED as
a shipping claim

Graph introspection of both exported models [measured]: opset 17, inputs
`features [1,2,64] f32`, `layout_keys [1,64,2] f32`, `layout_mask [1,64] bool`; outputs
`log_emissions [1,32,65]`, `coefficients [1,32,64]`, `lambda [1,32,1]`; **zero `Einsum`
nodes** (audit fix #9 holds). This matches `README.md`'s contract table and the Kotlin
seam: `CtcEmissionModel.emit` takes a `[2,64]` flattened path plus the padded layout, and
`CtcEmissions.sliceFromHead(fullHead, numLetters, maxKeys = 64)` relocates blank from
column 64 to column `numLetters`. No contract break.

Three qualifications on "drop-in":

1. **There is no production `CtcEmissionModel` in the app.** `CtcSwipeDecoder.kt`'s own
   doc calls itself "deliberately dead code today"; nothing in the IME references it. The
   model fits a seam that is not yet wired.
2. **Adopting the E1 preset is an app-side change.** `CtcScoringParams.encoderOnly` ships
   `(0.4056, 0.0176, 0.9866, 0.4234, 1.0382)`; every Phase-E number requires
   `(1.05, 1.1, 0.2, 0.3734, 0.9882)`. `README.md`'s earlier "keep the published preset
   exactly as published" verdict is superseded but still sits in the file above the
   retraction note.
3. **Featurizer parity is asserted, not tested.** Audit finding #4 stands: the Kotlin
   `CtcParityTest` references a golden fixture that does not exist in the app tree, so the
   "bit-identical featurizer" guarantee has never actually run green. Every number here
   assumes it.

---

## D. Required disclosures for the final report

Non-negotiable, in the report body and not a footnote:

1. **The preset asymmetry, quantified.** "Our decode preset was tuned on val-9918 by a
   five-parameter grid search; the FUTO ceiling was measured at its own published preset.
   At the published preset our final model clears 3 of the 5 bars (t1 +0.26, t3 +0.12,
   t5 −0.13, ≤3 −1.19, 4+ +1.01). The tuning is worth +2.29 pt top-1 on this model. Whether
   FUTO's decoder has comparable headroom under the same sweep is untested." Any headline
   that says "beats the FUTO ceiling" without this sentence adjacent is misleading.
2. **The T3 dedup defect.** 588 val rows and 145 test rows are present in the training
   tier with a bit-identical input tensor and label, because the dedup key uses the raw
   word while the label uses the a–z-normalized word. State the measured impact honestly:
   leaked rows score *below* comparable non-leaked rows (94.05 vs 95.12), so the bound on
   inflation is <0.05 pt. Correct `PHASE_D.md` §2's "caught every match on both sides" and
   `PHASE_E.md` §5's "removed bit-exactly".
3. **Contributor contamination.** T3 applies no session or participant exclusion; every
   holdout contributor is in training, and 3× HWS oversampling triples the exposure of the
   more contaminated corpus. **No contributor-clean subset of val or test exists for this
   model.** These are benchmark numbers only.
4. **The counter-asymmetry.** 43 % of the holdout traces are in FUTO's own training corpus;
   the app repo's eval docs incorrectly describe the split as FUTO-held-out.
5. **Statistical resolution.** On val only t1 (z 5.3) and 4+ (z 4.9) are resolved; t5 is
   0.8σ, t3 2.0σ, ≤3 2.2σ. On test the SEs roughly double.
6. **Lexicon.** The val bar and our runs use the *same* 146,964-word trie — correct
   `README.md`'s claim of conservatism, which does not apply. The test bar was published on
   the 131,544 trie and re-measured unchanged (84.83) on the 146,964 one; its *strata* were
   not republished post-fix, so ≤3/4+ on test are compared across normalizers.
7. **Seal hygiene.** Disclose the pre-campaign 120-row test smoke decode (04:20 EDT, toy
   898-word trie) and the fact that `eval_beam.py` and `sweep_scoring.py` carry no test
   guard. Also: 7 traces are shared bit-exactly between val-9918 and test-2400.
8. **Arm selection used full val.** The `4959:9918` "untouched" table is untouched by the
   preset sweep and by checkpoint selection, but the *configuration* was chosen on full-val
   tables.

---

## E. Pre-registration for the single test decode

**Recommendation: GO.** The val result is real, reproducible to the digit, and the seal is
intact. The leak found in §3 does not warrant a rebuild — its measured effect is inside
0.05 pt, and retraining six runs would re-roll seed noise against a t5 margin of +0.28,
which is a worse trade than disclosing. Fix the dedup key *after* the decode.

**But the claim must change before the decode is spent.** Pre-register the wording, so
that the result cannot be re-framed after the numbers are seen:

> **Claim as registered:** on the sealed 2,400-row test split, the Phase-E configuration,
> decoded at the val-tuned E1 preset, is compared against FUTO's published encoder+
> refinement ceiling decoded at FUTO's published preset. A pass is not a claim of
> superiority on equal footing — the presets are not matched, and no attempt to re-tune
> FUTO's preset was possible (its weights are not available here).

**Configurations to decode — both, once each, three seeds each (6 runs, ~15 min total).**
Registering both is important: ch128 is the better shipping trade and registering only the
headline invites a post-hoc switch.

```bash
cd /home/will/git/CleverKeys-ML/ctc
for a in phaseE-FINAL-s1234 phaseE-FINAL-s4321 phaseE-FINAL-s7777 \
         phaseE-E3b-hws3x   phaseE-E3b-hws3x-s4321 phaseE-E3b-hws3x-s7777; do
  python3 eval_beam.py \
    --onnx ckpt/$a/ctc_swipe_encoder.onnx \
    --test data/test_hwsfuto.jsonl \
    --preset 1.05,1.1,0.2,0.3734,0.9882 \
    --beam-width 100 --top-k 8 \
    --out ckpt/$a/test2400_e1.jsonl \
    > ckpt/$a/test2400_e1.log 2>&1
done
```

Frozen before the run: preset `1.05,1.1,0.2,0.3734,0.9882`; beam width 100; `top_k` 8;
vocab `data/futo_en_wordlist.combined` (146,964-word STRIP trie); the committed
`ckpt/<arm>/ctc_swipe_encoder.onnx` (all six verified byte-identical to a fresh export
from `best.pt`); metric = seed-mean over 1234/4321/7777 of top-1/3/5 and the ≤3 (n=815) /
4+ (n=1585) strata; OOV counts as a miss.

**Also register, before seeing the result:**

* **One decode. No second preset, no second checkpoint, no `--limit` warm-up.** If the run
  crashes, the fix must not depend on any partial output.
* **Report all five numbers for both configurations regardless of outcome**, and report
  the per-source (futo/hws) split, which `holdout_source_tags.json["test"]` supports
  without a second decode.
* **Pre-registered expectations** (so a miss cannot be re-explained afterwards): r2's val→
  test offset was −0.61 t1; the test bars are lower than the val bars on t1/t3/t5
  (−0.69/−0.50/−0.72) but **higher on ≤3** (+0.28). Expected outcome is a comfortable pass
  on t1 and 4+, a likely pass on t3 and t5, and a **coin-flip on ≤3**. A 4-of-5 result is a
  4-of-5 result and must be reported as a failed gate, not as "essentially passing".
* **The published-preset control is already measured on val** (§5a) and should be quoted
  next to the test table. Do **not** spend a second test decode on it — one decode only.

**Post-decode, regardless of outcome:** fix `build_tiers.hash_row` and
`prepare_data.trace_hash` to key on `normalize_word(word)`, rebuild T3/T3hws, and add a
real guard (row-count assertion, not a filename substring) to `eval_beam.py` and
`sweep_scoring.py`.

---

## Appendix — what this audit could not check

* **A fair rematch is not runnable here.** FUTO's `honorable_sturgeon` / `magic_macaw`
  `.pte` files are not on this machine, so the ceiling's preset could not be re-tuned on
  val for a like-for-like comparison. This is the largest unresolved question and it is
  *not* resolved by decoding test.
* **The val-9918 bar has no committed reproduce recipe** in the app repo — it is prose in
  one table, with no per-trace cache and no script (unlike test-2400, which has one). I
  verified it is quoted correctly; I could not re-derive it.
* **Per-stratum t3/t5 for the ceiling on val are unpublished**, so only stratum top-1 can
  be gated.
* **Kotlin↔Python featurizer parity is untested** (missing golden fixture, app-repo work).
