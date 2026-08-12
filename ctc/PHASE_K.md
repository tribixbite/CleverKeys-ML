# Phase K — candidate-generation campaign (the ≤3 stone, attacked at its diagnosis)

**Date:** 2026-08-11 · **Authority:** orchestrator directive of 2026-08-11.
Phase J closed at 10/11 bars seed-mean (5/11 every-seed) with the ≤3 stratum
−0.07 and Cyrillic untouched, and diagnosed the ≤3 miss as a
**candidate-generation problem** (`PHASE_J.md` §9): five levers aimed at the
stratum all failed, and the decode-side sweep proved re-ranking of the
*existing* beam cannot buy it. Phase K attacks generation directly on three
Tier-1 tracks, with Tier-2 riders on spare GPU. **test-2400 stays sealed**
(ledger stays at 3 entries; any unsealing decision goes to the parent
orchestrator, not this campaign).

## 0. Bars (unchanged from `PHASE_J.md` §0/§8 — the audited footing)

en val-9918 (E1/AOSP, 3-seed means): **88.30 / 92.60 / 93.26 / 91.27 / 86.77**
(t1/t3/t5/≤3/4+) · dvorak **89.13** · dvorak-app **88.20** · azerty **83.60** ·
qwertz **82.50** · german **79.64** · spanish **88.28** · ru shippable
(λ = 2.0 footing, `PHASE_J.md` §6.9 confirm half) **77.92** · size ≤ 5 MB ·
latency < 50 ms. Incumbent `resbn192i`; finalist `sw2345` (10/11 seed-mean,
≤3 −0.07). Both footings (seed-mean AND every-seed) are reported for any claim.

## 1. The program

| track | idea | cost |
|---|---|---|
| **K1** | seed-ensemble emission averaging before the beam (eval-only) | CPU hours |
| **K2** | T′ = 64 contract-v2 retrain of the finalist recipe (`PHASE_I.md` §6.1 probe: +0.33 4+, +2.5–2.8 transfer at ch 80 — never measured on ≤3 at the modern recipe) | GPU, 188 k × ≥1 seed |
| **K3** | discriminative candidate rescorer: self-mined confusable slates from TRAIN-set emissions → tiny listwise ranker → top-k rerank; blend weight swept tune-half/confirm-half; **symmetric** (incumbent gets its own mined ranker) | GPU minutes + CPU |
| **K4** | riders: beam width 300 recheck, length-conditioned decode knobs, ≤3-weighted CTC loss, `sw2345` @ 280 k + soup, lr/wd micro-sweep | spare slots |

The contract change in K2 is pre-authorized by the user directive; it ships as
**contract-v2** (`[1,64,65]` head) beside the frozen v1, with the app-side
implication list in §K2 below.

## 2. Infrastructure (committed this phase)

* **Ensembles** (`eval_beam.py`, `eval_altlayout.py`): comma-separated
  `--onnx` + `--ens-avg {logprob, prob}`. `logprob` = mean of log-emissions
  **renormalized per frame** (log-softmax — required: the beam's `len^γ`
  normalization is not invariant to per-frame additive constants);
  `prob` = `logsumexp − log N`, normalized by construction. Single-model path
  regression-checked against the committed `sw2345` dump: first-500-rows t1
  identical (90.00).
* **Rescorer** (`ranker_features.py`, `mine_candidates.py`,
  `train_ranker.py`, `sweep_rerank.py`, `eval_beam.py --ranker-onnx`):
  the feature module replays the beam's per-word Viterbi state machine
  *exactly* — verified **0.00e+00** max deviation against 240 live beam final
  scores — so `forced_viterbi` is the beam's own path score for any word,
  kept or pruned. 14 features/candidate (alignment, mass spread, length,
  log-freq, slate rank/gaps, weakest-letter evidence, blank mass, short flag,
  T′). Miner is self-mining (our encoder, our beam, our train rows; no FUTO
  decoder output anywhere — license-clean by construction).
* **Training lever** (`train.py --short-loss-weight`): weighted-mean
  per-sample CTC loss with weight w on `len ≤ 3` targets; `w = 1` is
  bit-equal to the stock reduction; the logged `ctc_loss` stays unweighted
  for cross-arm comparability.
* **Ops fix:** `eval_beam.py`'s ORT sessions were uncapped and a 4-eval
  stampede measured **load 114** on the 24-core box; sessions now pin
  intra/inter-op threads to 1 like `eval_altlayout.py` always did.

## 3. Arms in flight (launched 2026-08-11 ~20:30, all detached per the
## continuity protocol)

| arm | what | seed |
|---|---|---|
| `phaseK-t64` | K2: finalist recipe + `--t-out 64` + snapshots | 1234 |
| `phaseK-sw2345-280k` | K4: finalist recipe on 280 k + snapshots (soup supply) | 1234 |
| `phaseK-sw2345-slw2` | K4: `--short-loss-weight 2.0` on the finalist recipe | 1234 |
| miner `mined_sw2345` | K3: 600 k train rows (ALL len ≤ 4 + 35 % of longer), E1 beam, k = 8 | — |

## 4. K1 — seed-ensemble emission averaging

*(numbers land below as the four full-val runs finish)*

### 4.1 Smoke (500-row val prefix) — the alignment-disagreement finding

| config | t1 | greedy |
|---|---|---|
| `sw2345` s1234 single | 90.00 | 74.20 |
| `sw2345` ×3 seeds, avg=**logprob** | **79.60 (−10.4)** | 32.80 |
| `sw2345` ×3 seeds, avg=**prob** | 90.00 | 38.20 |

Same-recipe seeds **disagree on CTC alignment**: each seed's letter peaks sit
on slightly different frames, so the geometric mean (logprob) multiplies
misaligned peaks into mush — catastrophic, outside any noise reading. The
arithmetic mean (prob) preserves whichever seed's peak is strongest
(logsumexp ≈ max) and survives — but the greedy collapse (74 → 38) shows the
averaged emissions are individually much blurrier; the lexicon beam absorbs
it. This is the CTC-ensembling classic (emission averaging wants a shared
alignment) reproduced at 1.5 M params.

### 4.2 Round 1, full val-9918 (E1/AOSP) — seed-ensembles REFUTED, and a
### cross-model surprise

Single-model references are the committed Phase-J dumps (same rows, preset,
trie).

| config | t1 | t3 | t5 | ≤3 | 4+ | greedy |
|---|---|---|---|---|---|---|
| `sw2345` s1234 (single) | 88.51 | 92.59 | 93.35 | 90.91 | 87.26 | 72.08 |
| `resbn192i` s1234 (single) | 88.32 | 92.70 | 93.25 | 91.21 | 86.83 | 72.75 |
| `sw2345` ×3 seeds, logprob | 77.02 | 90.94 | 92.32 | 78.52 | 76.24 | 29.26 |
| `sw2345` ×3 seeds, prob | 87.12 | 91.96 | 92.74 | 90.23 | 85.51 | 37.05 |
| `resbn192i` ×3 seeds, prob | 87.39 | 91.76 | 92.73 | 90.73 | 85.65 | 64.11 |
| **mix2 = `sw2345`+`resbn192i`, BOTH s1234, prob** | **88.66** | **92.63** | **93.42** | **91.41** | 87.23 | 68.12 |

**K1 as specified — averaging a model's own 3 seeds — is a clean negative in
both modes for both families** (−1.1 … −1.4 t1 prob-mode; logprob
catastrophic). The seeds do not share an alignment, and emission-space
averaging punishes that before the beam can help.

**The surprise: the cross-MODEL, same-SEED 2-mix clears all five val bars at
once, including ≤3** — Δ vs bars **+0.36 / +0.03 / +0.16 / +0.14 / +0.46**.
No Phase-J single model or seed-mean ever cleared ≤3; this configuration does
on its first measurement, with no tuning applied. Working mechanism
hypothesis: alignment phase is largely set by the init (the seed), not the
data mix — both members were seeded 1234 with the identical architecture, so
their letter peaks coincide and averaging sharpens instead of blurring.
Under test in round 2 (running): the s4321 and s7777 paired mixes (the
configuration's honest "seed axis"), a cross-seed control
(`sw2345` s1234 + `resbn192i` s7777 — the hypothesis predicts it fails), and
a 3-family s1234 mix (+`ch256-p65`).

Caveats registered up front: one measurement, and the mix2 artifact would be
2 × fp16w ≈ **6.1 MB > the 5 MB bar** as-is (int8-trunk or weight sharing
would be needed; unmeasured). Latency ≈ 2 × 0.84 ms encoder — trivially
inside budget. And the greedy drop (72 → 68) says even the same-seed average
blurs emissions slightly; the beam more than recovers it.

### 4.3 Round 2 — the mechanism is pair-compatibility, not "same seed"

| config | t1 | t3 | t5 | ≤3 | 4+ | greedy |
|---|---|---|---|---|---|---|
| mix2 s1234 pair | 88.66 | 92.63 | 93.42 | 91.41 | 87.23 | 68.12 |
| mix2 s4321 pair | **87.46** | 91.80 | 92.83 | 90.00 | 86.14 | **19.84** |
| mix2 s7777 pair | 88.57 | 92.72 | 93.49 | 91.30 | 87.15 | 61.35 |
| mix2 cross-seed (sw2345 s1234 + resbn s7777) | 86.75 | 91.58 | 92.55 | 89.55 | 85.30 | **9.27** |
| **mix3 s1234 (+`ch256-p65`)** | **88.88** | **92.73** | 93.35 | **91.53** | **87.50** | **74.85** |

* The cross-seed control fails as the alignment hypothesis predicts (greedy
  9.27: the members literally cannot agree on a frame labelling).
* **But "same seed" is NOT sufficient**: the s4321 pair fails the same way
  (greedy 19.84). Alignment compatibility is a property of the *pair*, not of
  the seed label.
* **The compatibility metric is per-FRAME, not per-string.** Whole-string
  greedy agreement is flat across all 21 member pairs (76.7–78.7 %) and
  predicts nothing. Per-frame argmax agreement (2,000 unlabeled val traces)
  separates perfectly, and the letter-identity agreement where both models
  emit is ~96 % for every pair — the disagreement is *where* letters and
  blanks sit, i.e. alignment phase:

  | pair | frame agreement | ensemble val t1 |
  |---|---|---|
  | s1234 pair | **96.9 %** | 88.66 ✓ |
  | s7777 pair | **96.1 %** | 88.57 ✓ |
  | s1234 + `ch256-p65` (mix3 edges 95.5 / 96.2 / 96.9 %) | | 88.88 ✓ |
  | s4321 pair | 88.8 % | 87.46 ✗ |
  | cross-seed control | 83.3 % | 86.75 ✗ |

  A label-free gate — *pair per-frame agreement ≥ ~95 %* — costs seconds,
  needs no val labels, and cleanly picks the working ensembles. (Registered
  as derived AFTER the round-2 outcomes were seen; any bar-claim built on it
  needs a fresh confirmation, e.g. new seeds gated blind.)
* On the honest **recipe-level seed-mean footing, mix2 fails t1**
  (mean 88.23 vs 88.30): as a *recipe*, cross-model mixing is not
  bar-clearing. What clears bars is a *specific compatible pair*, selected by
  the gate.

### 4.4 The s1234 mixes take ALL ELEVEN bars (single-configuration footing)

Alt-layout az26 in-dict E1 + dvorak vs app-98k trie
(`altlayout/phaseK-mix{2,3}s1234_*`):

| corpus | bar | mix2 s1234 | Δ | mix3 s1234 | Δ |
|---|---|---|---|---|---|
| val t1 | 88.30 | 88.66 | +0.36 ✓ | **88.88** | **+0.58 ✓** |
| val t3 | 92.60 | 92.63 | +0.03 ✓ | 92.73 | +0.13 ✓ |
| val t5 | 93.26 | 93.42 | +0.16 ✓ | 93.35 | +0.09 ✓ |
| **val ≤3** | 91.27 | **91.41** | **+0.14 ✓** | **91.53** | **+0.26 ✓** |
| val 4+ | 86.77 | 87.23 | +0.46 ✓ | 87.50 | +0.73 ✓ |
| dvorak | 89.13 | **92.27** | +3.14 ✓ | **92.02** | +2.89 ✓ |
| dvorak app-98k | 88.20 | **91.66** | +3.46 ✓ | **91.33** | +3.13 ✓ |
| azerty | 83.60 | 85.12 | +1.52 ✓ | 85.02 | +1.42 ✓ |
| qwertz | 82.50 | 82.90 | +0.40 ✓ | 82.73 | +0.23 ✓ |
| german | 79.64 | 81.22 | +1.58 ✓ | 81.17 | +1.53 ✓ |
| spanish | 88.28 | 89.59 | +1.31 ✓ | 89.76 | +1.48 ✓ |
| clearflow (floor 91.08) | — | 92.22 | — | 92.34 | — |
| kasroz (floor 92.07) | — | 91.94 | — | 91.80 | — |

**Every val bar and every layout bar falls, most by margins far outside the
measured seed spreads** (dvorak +2.9–3.1 against a 1.5–3 pt spread axis;
euro corpora +0.2–1.6). The transfer gains dwarf the val gains — averaging
across *models trained on different data mixes* smooths exactly the
layout-specific idiosyncrasies that single models trip on.

**Footing disclosures (why this is not yet a terminal-condition claim):**
1. These are deterministic single configurations; the bars are 3-seed means
   of a stochastic recipe. The configuration's own "seed axis" (§4.3) shows
   the *recipe* does not clear bars — the *gated pair* does. The gate is
   label-free but was identified after seeing these results (post-hoc for
   this phase; pre-registerable for any future confirmation).
2. **Size: NOT met as-is.** mix2 = 2 × fp16w = 5.82 MiB > 5 MB;
   mix3 ≈ 11 MB. int8 weight quantization (~×2 further) is the open route
   (`quantize_onnx.py`); accuracy after int8 is unmeasured on these members.
3. Cyrillic is untouched by K1 (stone 2 stands regardless).
4. Latency: 2–3 × 0.84 ms encoder — inside every budget; to be measured
   end-to-end for the record.

### 4.6 Size-compliant packagings hold the mix2 result

`quantize_onnx.py` weight-storage modes on both s1234 members, decoded through
the ensemble beam (full val-9918, E1/AOSP):

| packaging | bytes | t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|---|
| fp32 + fp32 (reference) | 12.14 MB | 88.66 | 92.63 | 93.42 | 91.41 | 87.23 |
| **int8w + int8w** | **3.11 MB** | 88.65 | 92.64 | 93.45 | 91.30 | 87.27 |
| int8w + fp16w | 4.45 MB | 88.68 | 92.61 | 93.46 | 91.30 | 87.32 |

Both compliant packagings clear all five val bars; ≤3 gives back ~0.11 to
weight rounding but stays over (+0.03). The int8w `resbn192i` member's rough
random-probe parity (argmax 76/100, max|Δ| 21) is real in the emissions and
almost invisible after the beam — same shape as the fp16w story in
`PHASE_J.md` §10. **The ≤5 MB size bar is met by the 3.11 MB int8w pair**
(alt-layout battery on this packaging: running).

## 5. K3 — discriminative rescorer, v1 results

Miner: 600,000 train rows (all len ≤ 4 + 35 % longer, seed 1234) through the
E1 beam at k = 8 → 94.2 % gold@1, 3.6 % gold@2–8, 2.2 % absent. Ranker:
5,185 params, listwise CE (err-weight 4, short-weight 2); dev pure-ranker
top-1 94.0 % (≈ ties the beam) and **puts gold first on 50.8 % of dev error
slates**. Blend `final' = beam + w·ranker`; w swept on val[0:4959], confirmed
on val[4959:9918], per protocol.

* **Tune half:** w = 0.05 wins min-margin (≤3 +0.29, t1 +0.10).
* **Confirm half (untouched):** ≤3 **+0.36**, t1 +0.21, t3 −0.04, t5 0.00 —
  the effect holds out of sample on the mined seed.
* **Frozen w = 0.05 applied to all three finalist seeds (full val):**

| seed | t1 Δ | ≤3 Δ | 4+ Δ |
|---|---|---|---|
| s1234 (mined) | +0.15 | **+0.33** | +0.06 |
| s4321 | −0.02 | −0.03 | −0.02 |
| s7777 | +0.12 | −0.17 | +0.28 |
| **seed-mean** | **+0.08** (88.59) | **+0.04** (91.24) | +0.11 (87.22) |

**Verdict v1: the ≤3 gain is sign-inconsistent across seeds** (the ranker was
trained on s1234's own emission quirks) — under the campaign's
sign-consistency rule it is NOT a promotable ≤3 lever as-is, and seed-mean ≤3
(91.24) still misses the bar by 0.03. t1/4+ tilt positive (2–3 of 3 seeds).
Registered next steps: (a) seed-general ranker (mine s4321 + s7777 emissions
too, retrain, same frozen-w protocol), (b) the symmetric incumbent ranker
(mining running), (c) rescorer × alt-layout interaction unmeasured — required
before any stacked-configuration claim.

## 4.5 Ops incident — all three GPU arms deadlocked at ~22:00–22:12

`phaseK-t64` froze at step 63 k, `280k` at 123 k, `slw2` at 54 k; GPU 0 %,
main procs alive, workers zombied. These were FRESH runs (not resumes), so
the known "persistent-worker resume deadlock" is **broader than resumes** on
this box — plausibly the persistent-worker pool generally, possibly
triggered by the large CPU eval load starting/stopping alongside. All three
trees were killed and resumed from `last.pt` (≤ 3 k steps lost each) under
identical args **plus `--workers 0`**, per the standing rule. Lesson
recorded: on this box, any train.py run that must survive unattended should
use `--workers 0` from launch; the throughput cost is the cheaper insurance.

## 5. Continuity

Same rules as `PHASE_J.md` §7: trainings `nohup setsid` + launch logs
(`ckpt_phaseK-*.launch.log`), resumed runs use `--workers 0`, no waiter
scripts, batteries run from a live orchestrator, state committed at every
milestone. Eval outputs under `~/ctc-train/phaseK/`.
