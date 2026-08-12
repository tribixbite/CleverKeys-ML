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

## 5. Continuity

Same rules as `PHASE_J.md` §7: trainings `nohup setsid` + launch logs
(`ckpt_phaseK-*.launch.log`), resumed runs use `--workers 0`, no waiter
scripts, batteries run from a live orchestrator, state committed at every
milestone. Eval outputs under `~/ctc-train/phaseK/`.
