# AUDIT_FINAL2 — adversarial audit of Phases L, M and the fourth unsealing

**Auditor:** independent adversarial session, 2026-08-14. **Scope:** the six
claims of the final-audit directive, covering `PHASE_L.md`, `PHASE_M.md`,
`UNSEALING_4.md`, `PIPELINE_V2_PROPOSAL.md`, `RESULTS.md`,
`MODEL_COMPARISON.md`, the seal ledger and the ship artifacts. **Constraint
honoured:** test-2400 was **not** decoded — every test-side check below is
recomputed from the six committed per-trace dumps; the ledger stands at
exactly 4 entries before and after this audit. All independent decodes in this
audit ran on **val-9918 only**.

## 0. Method

* Recomputed every fourth-unsealing test table from the per-row dumps
  (`~/ctc-train/ckpt/v2kd-fresh-w1{,-s4321,-s7777}/test2400_m_{e1,app}.jsonl`,
  6 × 2,400 rows), replicating `eval_beam`'s rank/stratum/OOV conventions from
  source (`rank_of`, `len_stratum`, `Tally`).
* Recomputed the McNemar b/c counts for **all three seeds** from the config-A
  dumps against `futo_verify/out/tuned_ceil_strip.jsonl` (dated Aug 8,
  pre-existing), under three matching conventions (exact case, lowercased,
  in-vocab-gated) — counts identical under all three.
* Recomputed the equal-footing bar itself from the FUTO per-row file:
  87.12 / 92.29 / 92.96 / 89.94 / 85.68 — matches `FAIR_REMATCH.md` §5.
* Independently re-decoded **val-9918** with the committed harness:
  s1234 fp32 at both footings, plus fp16w at both footings (four decodes).
* Independently re-ran `pair_agreement.py` on four pair configurations.
* Verified commit **and push** order against the GitHub PushEvent stream, file
  birth/mtimes (`statx`), the seal-guard lines in all six decode logs, ledger
  arithmetic, artifact sha256s, ONNX graph I/O, and the golden fixture's
  emissions against a live run of the ship artifact.

## 1. Alignment-coupling mechanism — **CONFIRMED**

* **Gate numbers reproduce.** `pair_agreement.py` re-run by this audit:
  `v2pair-s1234` selected pair **98.3 %** (doc 98.33), own-best 98.1;
  `v2pair-pw0` own-best **91.3 % FAIL** (doc 91.30), selected 95.3 (doc 95.32).
* **The collapse is real and recomputed from dumps**, not footers: pw0
  own-best mix greedy **29.10** / beam t1 **87.64** (n=9,918); pw0 selected
  53.12 / 88.09; L1 pair 72.92 / 88.90; member A solo 88.60. Same recipe,
  same data seed, differing in `--pair-weight` only → the KL is isolated as
  the cause. E1 attribution stands.
* **Coupling weight 0.3 interior-optimal — confirmed from primary logs/JSONs:**
  val t1 88.09 (pw 0) / 88.84 (0.1) / **88.90** (0.3) / 88.85 (1.0); agreement
  monotone 92.09 → 98.08 → 98.33 → 98.58; dvorak transfer 92.96 / **93.04** /
  91.09 — the pre-stated over-coupling collapse at 1.0 (−1.95) is in the
  altlayout JSONs exactly as published.
* **Qualification on "12/12 gate predictions".** The 12 band predictions are
  numerically correct (all twelve coupled pairs ≥ 98 %, all mixes in the
  working band — verified from gate logs and pair dumps). But only **8 of the
  12** were gate-committed-and-pushed *before* their decodes (Phase L round 1:
  commit 06:37:10 → decodes 06:40+; settlement: 12:23:36 → 12:26+; Phase M
  stage 1: 22:51:29 → 22:58+ — each push precedes its decodes). For the four
  **stage-2** arms (pw01/pw10/e4/e6) the gates were measured 09:35:31 and the
  pair decodes ran 09:39–09:42 with **no commit in between** (next commit
  09:45:54). Gate-before-decode held; the §2 "commit gate and band prediction
  → decode" blind step did **not**, for those four. Stage 2 was a
  pre-registered *measurement* with no promotion at stake, so no conclusion
  changes — but "12 of 12" should be read as 8 committed-blind + 4
  rule-implied.

## 2. Phase-L retraction and the pair claim — **CONFIRMED**

* **Retraction arithmetic checks.** Five-seed member-A means recomputed from
  the per-seed table: t3 **92.576** (−0.024 ✗), qwertz **82.344** (doc 82.342;
  a 0.002 rounding wobble from pre-rounded per-seed inputs — the miss stands
  either way). The §1.1 rule fired exactly as pre-registered; the retraction is
  propagated in place in `PHASE_L.md` §15.4, `RESULTS.md`,
  `MODEL_COMPARISON.md` and `APP_INTEGRATION_PLAN.md`.
* **The pair's 11/11-on-5/5-seeds claim checks.** Pair val t1 recomputed from
  the five per-trace dumps: 88.90 / 88.82 / 88.78 / 88.65 / 88.73 (mean
  **88.776** ✓); ≤3 91.53 / 91.47 / 91.47 / 91.30 / 91.41 (mean **91.436** ✓).
  Per-seed layout values spot-verified in the altlayout JSONs. No dump
  contradicts the [11,11,11,11,11] tally.

## 3. Phase M — **CONFIRMED**

* **The fresh-KD student's 3-seed battery is fully verified from primary
  files**: all five val axes recomputed from `val_dump_e1.jsonl` per seed
  (88.62/88.88/88.75 · 92.69/92.80/92.83 · 93.46/93.45/93.51 ·
  91.38/91.44/91.30 · 87.18/87.55/87.43 — exact match), and all six layout
  axes per seed from the altlayout JSONs (dvorak 92.23/92.14/91.09, dvorak-app
  91.90/91.45/89.95, azerty/qwertz/german/spanish likewise exact). Every seed
  clears every campaign bar: **[11,11,11] confirmed by recomputation.**
* **initA-vs-fresh verdict**: the pre-registered gate (`≥ member A on t1 AND
  ≤3`) was applied as written — fresh passes 88.62/91.38 vs 88.60/91.32;
  initA-w1 fails by 0.01 on t1; no retuning. The "teacher gauge-consistency
  matters, student init does not" reading follows the published numbers.
* **Negatives verified from logs**: E4 dvorak **−2.81** (90.23 vs 93.04),
  dvorak-app −2.85, qwertz −1.26, azerty +0.62 — rule violated, dropped ✓.
  E6 val deltas t1 −0.21 / t5 −0.16 / ≤3 −0.18 / 4+ −0.23 — four bars past the
  −0.15 kill criterion (which is verbatim from the proposal §E6), dropped with
  no re-weight retry ✓. pair-weight 1.0: see §1.
* **Crown scoring honest**: the student beats the card's five val numbers at
  the 3-seed mean but misses four transfer axes; recorded NOT WON ✓.

## 4. The fourth unsealing — **CONFIRMED**, with two errata

### 4.1 Protocol integrity — CONFIRMED, including the push

* Pre-registration commit `b91f179` authored **10:05:01** EDT; GitHub
  PushEvent with head `b91f179` at **10:05:20** EDT; the six dump files were
  **created 10:05:37–10:05:38** and completed 10:06:13–10:06:16. The
  pre-registration was committed *and pushed* before any decode, by 17
  seconds. Results commit `1642286` 10:13:51, pushed 10:14:03.
* **Exactly six new decodes.** Six dumps, 2,400 rows each; all six decode logs
  carry the `seal.py` 2400/2400-overlap + `--unseal-test` override lines; no
  other test-2400-dated output exists anywhere under `~/ctc-train` in the
  Aug 13–14 window. (An unlogged decode cannot be disproven in principle;
  every observable artifact is consistent with six.)
* **Ledger**: exactly **4 entries** plus the three disclosed `prior_contact`
  items; entry 4 is well-formed (authoriser, date, subject, both presets, both
  tries, publication sites, the no-fifth clause). Fingerprint set: 2,399
  unique hashes over 2,400 rows (the one known duplicate — the disclosed dedup
  defect); `seal.py` re-run reproduces the committed `words_sha256`
  (`f6053959…`).
* **The §1.2 pre-registration-grounding decodes were val-only**: all four
  `valB_*.log` files (10:01, before the pre-reg commit) are n=9,918 val runs
  at the app trie/preset. No seal warning appears in them (val's 7-trace
  overlap is under the guard's 1 % limit).

### 4.2 Test tables — CONFIRMED by full recomputation from dumps

Every cell of §8.1 and §8.2 (2 configs × 3 seeds × 5 metrics), every
seed-mean, sd and worst-seed value, and the greedy t1s (72.50/72.75/73.00)
reproduce **exactly** from the per-trace dumps. The per-source table (§8.4)
reproduces to ±0.01 (config-B HWS half computes 83.23 vs the doc's 83.24;
spread 11.97 ✓; the FUTO ceiling's 95.89/78.11 and the +0.38/+4.05 split ✓).
The trie-matched bar (84.92/91.54/92.96/89.57/82.52) was verified against the
original Phase-F FUTO log, and the equal-footing bar against the FUTO per-row
dump.

### 4.3 McNemar — counts CONFIRMED, two printed p-values WRONG (erratum)

Recomputed from the committed dumps, counts match the doc exactly and are
invariant to case/vocab conventions:

| seed | we win | they win | doc p | **correct exact two-sided p** |
|---|---|---|---|---|
| s1234 | 81 | 36 | 3.5e-05 | **3.87e-05** |
| s4321 | 89 | 43 | 1.4e-04 | **7.69e-05** |
| s7777 | 80 | 41 | 5.0e-04 | **4.99e-04** ✓ |

The s1234 and s4321 p-values printed in `UNSEALING_4.md` §8.3 and `RESULTS.md`
do not follow from their own printed counts under the exact two-sided binomial
(nor under mid-p, continuity-corrected or plain χ²). No verdict changes — all
three seeds resolve at p < 5e-4 (s4321 is in fact *more* significant than
printed), the 3/3 equal-footing resolution stands. **Correction required:**
fix the two p-values where they appear.

### 4.4 Expectation scoring — CONFIRMED

Band edges, point predictions, shift means and MAEs re-derived from §4.3's
shift table: all reproduce. Band coverage is genuinely **9 of 10** with the
single miss config-A ≤3 at 92.597 vs band top 92.593 (+0.004), honestly
recorded as a miss; all **7 verdicts** score as claimed (B2's +0.375 in
+0.1…+0.5; MAE 0.30 A vs 0.36 B). One rounding inconsistency: §8.5 states the
≤3 val→test lift as "+1.23/+1.14" while §8.6 states "+1.22/+1.14" (true values
+1.224/+1.138).

### 4.5 fp32-decode / fp16w-ship bridge — CONFIRMED by independent decode

This audit re-decoded val-9918 itself: **app footing** fp32 vs fp16w —
**89.20 / 93.63 / 94.37 / 92.59 / 87.44 both, delta 0.00 on all five** ✓
(matches the committed `valB_*` logs, which also match §4.2's table for all
three seeds); **E1 footing** fp32 vs fp16w — 88.62 / 92.69 / 93.46 / 91.38 /
87.18 both, sub-strata deltas ≤ 0.03, consistent with the ≤0.05 claim.

### 4.6 Fixture regeneration — CONFIRMED

`phaseM_kd_fresh_w1_fp16w_golden.json` (committed, sha256 `2a449c4f…` ✓)
carries `preset = [0.9, 4.0, 0.25, 0.25, 0.9882]` (the ship preset) and
`source_onnx_sha256 = 84718e6e…` (the ship artifact). This audit re-ran the
ship fp16w ONNX on the fixture's stored features: **emissions bit-identical
(max |diff| = 0)** on all four beam cases.

## 5. Documentation coherence — **QUALIFIED** (four stale/minor items)

The governing statements are correct: tier language at the top of
`RESULTS.md`, the `MODEL_COMPARISON.md` Phase-M addendum and §5 matrix, and
`UNSEALING_4.md` §8.7 all state the right tiers, and both limitations (HWS-half
-bought lead with FUTO +0.38 on its own half; ch 192 keeps t5 by 0.14) are
quoted at every top-level statement of the equal-footing win. Items needing
correction:

1. **`APP_INTEGRATION_PLAN.md` §9.5** still ends "test-2400 remains sealed; no
   number in §9 is test-validated" — stale since the fourth unsealing; the
   recommended asset in that very table is now test-validated.
2. **Stale "only configuration" lines not struck**: `MODEL_COMPARISON.md` §4.3
   ("ch 192 remains the only configuration with a (qualified) equal-footing
   win") and the same sentence in `RESULTS.md`'s Phase-G section. The §5
   matrix's copy of this claim *was* struck and corrected; these two were not.
   They sit in dated historical sections, but they are present-tense.
3. **The two wrong McNemar p-values** (§4.3 above) appear in `UNSEALING_4.md`
   §8.3/§8.6 and `RESULTS.md`.
4. Minor: the `MODEL_COMPARISON.md` §5 recommendation row asserts the
   equal-footing win with the t5 limitation implied ("4 of 5") but without the
   HWS-half limitation in the row itself (it is stated in the same file's
   addendum).

## 6. Ship-artifact contract compliance — **CONFIRMED**

* **Graph**: opset **17**; inputs `features [1,2,64] f32`,
  `layout_keys [1,64,2] f32`, `layout_mask [1,64] bool`; outputs
  `log_emissions [1,32,65] f32` (+ `coefficients`, `lambda`) — the frozen
  contract, blank at column 64 (fixture `numClasses` 27 = 26 letters + blank;
  greedy strings consistent with blank=64).
* **sha256s**: ship fp16w `84718e6e…` ✓ (3,052,318 B), fp32 seeds
  `b71911da…` / `f7cb72c0…` / `c55cc3b0…` ✓ (6,068,519 B each) — and the
  `artifacts/` copies are **byte-identical** to the `~/ctc-train/ckpt/*`
  exports the decodes actually ran on.
* **Fixture**: ship preset + correct source sha + bit-exact emissions (§4.6).

## 7. Independent execution performed by this audit

Six val-9918 decodes (s1234 fp32+fp16w × both footings, E1 fp16w), four
`pair_agreement.py` runs, full recomputation of all six test-2400 dump tables,
three-seed McNemar recomputation under three conventions, five-seed retraction
and pair arithmetic, seal fingerprint regeneration check, GitHub push-order
verification, artifact hashing, ONNX graph inspection, and fixture emission
replay. **Zero test-2400 decodes.**

## 8. Final statement

* **(a) The ship model's test-validated tier: SUPPORTED.** `phaseM_kd_fresh_w1`
  clears all five published-bar numbers on both footings on the seed-mean and
  every seed; the tables reproduce exactly from the committed per-row dumps;
  the protocol (pre-registered, pushed 17 s before the first decode, six
  decodes, no retries, ledger 3→4) held as written. The tier claim properly
  covers the fp32 graphs with a measured (and here independently reproduced)
  0.00-val-delta bridge to the fp16w ship artifact.
* **(b) The equal-footing win as stated: SUPPORTED, with one erratum.** All
  five equal-footing numbers clear on every seed; McNemar counts reproduce and
  resolve 3/3 — but two of the three printed p-values are wrong (correct:
  3.87e-05 / 7.69e-05 / 4.99e-04; all still < 5e-4, so the qualified-win rule
  fires identically). Both required limitations travel with the claim at every
  governing site.
* **(c) The campaign's terminal claims: SUPPORTED.** The coupling mechanism,
  the retraction, the pair's 5/5-seed footing, the distilled student's
  3/3-seed all-eleven result, and the E4/E6/E2 negatives all reproduce from
  primary artifacts. Required corrections are confined to: the two McNemar
  p-values, the stale `APP_INTEGRATION_PLAN.md` §9.5 seal sentence, two
  unstruck "only configuration" lines, one ±0.01 rounding (83.24→83.23,
  82.342→82.344, +1.23→+1.22), and the disclosed qualification that the four
  stage-2 gate predictions were rule-implied rather than committed-blind.
  None of these moves any verdict.
