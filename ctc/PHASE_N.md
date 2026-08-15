# Phase N — win the FUTO domain outright

**Status: PLAN OF RECORD, committed before execution.** Everything in §1–§9 is
fixed at commit time; each milestone additionally gets its own numeric
pre-registration committed before its decode (§5 protocol). Results are
appended as §10+ and may not restate or widen the plan.

**Date opened:** 2026-08-15 · **Authority:** the user's Phase-N directive —
*beat FUTO on every metric on the FUTO dataset itself.* Unlimited training
authorized. Workdir `~/ctc-train`, RTX 5080, runs detached with `--workers 0`,
milestone commits + pushes, honesty absolute.

## 0. The gap this phase exists to close

`UNSEALING_4.md` §8.4: the shipped model (`v2kd-fresh-w1`, the E7 distilled
single) holds a qualified equal-footing win over FUTO's val-tuned engine on
test-2400 — but the entire +1.81 aggregate lead is bought on the HWS half.
On **FUTO's own corpus half** (n = 1,217), FUTO's engine leads:

| engine | FUTO half t1 |
|---|---|
| FUTO ceiling, val-tuned | **95.89** |
| ours, config A (AOSP STRIP / E1) | 95.51 (−0.38) |
| ours, config B (app trie / app preset) | 95.21 (−0.68) |

Two honest facts about that number, recorded up front:

1. **−0.38 on n = 1,217 is ~4.6 rows and was never McNemar-resolved** — the
   per-source split was reported as a limitation, not tested for significance.
   The gap is plausibly real (it reproduces in sign across ch 128 / `resbn80g`
   / this model — FUTO led its own half at every unsealing) but its size is
   inside noise on that stratum.
2. **test-2400 can never be re-read** (ledger 4, final — `UNSEALING_4.md` §3,
   §7). So Phase N cannot be scored on the split where the gap was observed.
   It is scored on a benchmark with ~40× the FUTO-domain rows and full
   statistical power: FUTO's **official test split**.

## 1. The benchmark, defined precisely

### 1.1 Primary: the FUTO official test split, both engines through our harness

* **Rows:** `futo-org/swipe.futo.org` swipe-1 **`test.jsonl`, 49,970 rows**
  (already at `~/ctc-train/data/hf/test.jsonl`; sha256 recorded at N0),
  converted to the canonical eval schema by a committed converter
  (`convert_futo_official.py`): `x, y` verbatim (the HF frame **is** the
  canonical letter-area frame, proven bit-exact in `DATA_TIERS.md` §1),
  `t → t − t[0]`, `word → lower()`. Rows whose a–z-normalized target is empty
  or whose trace has < 2 points are dropped **with exact accounting** in the
  converter's committed stats. No other filtering — the official test is
  evaluated as FUTO shipped it, including landscape traces and
  dictionary-junk words (OOV-as-miss absorbs those symmetrically).
* **Dev split for all tuning:** swipe-1 **`dev.jsonl`, 54,269 rows**, same
  conversion. Iteration lives here (and on §1.2), never on test.
* **Metrics:** top-1 / top-3 / top-5, plus the **≤3-char** and **4+-char**
  strata (stratum n's recomputed on the converted file and frozen at N0).
  Same definitions as the whole campaign.
* **Decode footing, both engines:** beam width **100**, top-k **8**, **OOV
  counts as a miss**, **STRIP trie from FUTO's own `en_wordlist.combined`
  (146,964 words)** — FUTO's domain, FUTO's lexicon, identical for both.
* **Engines:**
  * ours — fp32 ONNX exports through `eval_beam.py` (the exact Phase-M
    machinery);
  * FUTO — their genuine published `.pte` weights
    (`honorable_sturgeon` + `magic_macaw`, hash-verified in
    `FUTO_WEIGHTS_VERIFICATION.md` §1) through the verified
    `futo_verify` harness. **Environment re-verified 2026-08-15**: both
    `.pte` load and execute under ExecuTorch 1.2.0 x86_64
    (encoder 26.1 ms, decoder 8.8 ms single-thread).
* **Symmetric preset tuning — the FAIR_REMATCH discipline.** Each engine's
  scoring preset is tuned on the official **dev** split with the same wide
  grid machinery that produced E1 and FUTO's val-tuned preset
  (`sweep_scoring.py` for ours, `futo_sweep.py` for theirs), same
  tune-half/holdout-half protocol, interior-optimum rule enforced. Neither
  engine ever gets a preset the other was denied the machinery to find.
* **Anchor:** the FUTO paper reports **92.94 (enc-only) / 93.30 (with
  refiner)** on this test split via their C++ beam. Our harness's read of
  their engine at the published preset is the cross-check; a large deviation
  is investigated before anything else proceeds (expected sources: OOV
  convention, case handling — our port reproduced their test-2400 numbers to
  the digit, so agreement should be close).

### 1.2 Secondary (dev-legal iteration): the FUTO half of val-9918

The 4,942 FUTO-tagged rows of `val_hwsfuto.jsonl`
(`cache/holdout_source_tags.json`). This is the val analog of the stratum
where the −0.38 was observed — note it passed the user's curation filters,
so it is a *cleaner* distribution than the official test. Used for fast
iteration and for bar B2. Our val decodes are free; FUTO's val-tuned per-row
val decode is run once at N1 (their engine, no seal implication).

### 1.3 What the two benchmarks are NOT

Not test-2400 (sealed forever). Not a generalization claim about unseen
users beyond what a held-out-contributor split supports. And the official
test is **unfiltered** FUTO data — numbers on it are not comparable to any
test-2400 or val-9918 number and will not be presented side by side without
that caveat.

## 2. Contamination — measured, not assumed

**Measured 2026-08-15** (sessions from each row's own `session` field vs the
10,889-session vocabulary of `cache/futo_session_index.npz`, the swipe-1
train index built by `scan_futo_sessions.py`):

| split | rows | sessions | sessions also in swipe-1 train | rows from shared sessions |
|---|---|---|---|---|
| official dev | 54,269 | 692 | **0** | **0** |
| official test | 49,970 | 697 | **0** | **0** |

**FUTO's official dev/test are session-disjoint from the train corpus.**
Since every FUTO-derived training row we have ever used comes from swipe-1
*train* (all tiers, `DATA_TIERS.md`), and our other corpora are HWS /
swipe-2..5 / our own synthesis, **no training tier shares a contributor with
the official dev or test splits.** The earlier scan already found **0**
exact-trace hits of our canonical holdout in official dev/test
(`DATA_TIERS.md` §2).

Remaining N0 checks, committed with results before any model touches dev:
exact-trace hash (the `scan_futo_sessions.trace_hash`) of every official
dev/test row against (a) the swipe-1 train index, (b) `tier_t3hws.jsonl`,
(c) the swipe-2..5 tier jsonls — expected 0 everywhere; any non-zero halts
the phase until explained. Input-file sha256s recorded.

**Symmetry note.** FUTO's own models were trained on the same swipe-1 train
split, so both engines have identical exposure to the benchmark's corpus and
zero exposure to its contributors. Our extra corpora (HWS, swipe-2..5,
synthesis) are legitimate training advantage, not contamination. Our tiers
additionally *lost* ~550 k train rows to our own holdout's session
exclusion — a handicap FUTO's training did not carry; also fine.

## 3. Seal discipline for the official test — pre-registered now

test-2400's guard generalized: at N0, `seal.py --emit` registers
**`futo-test-49970`** (content-addressed fingerprints of the converted
rows) in `test2400_seal.json`, with its own append-only `unsealings` ledger.
`eval_beam.py` then refuses the split without `--unseal-test`, exactly as for
test-2400.

**The rules, fixed before any number exists:**

1. **Our models decode futo-test only at named milestones — hard cap 3 for
   the phase:**
   * **M0** — baseline: the ship model `v2kd-fresh-w1` (3 seeds) at its
     dev-tuned preset. One config.
   * **M1** — optional mid-phase: at most one trained candidate (3 seeds,
     one config), only if a lever passes its dev/val gates.
   * **M2** — final: the Phase-N candidate (3 seeds, one config).
2. **Presets are frozen on dev and committed before the decode.** Emissions
   of futo-test may never be grid-scored: the analytic sweep machinery makes
   preset shopping on cached test emissions free, so the rule is stated in
   terms of scoring, not decoding — **no preset not named in the milestone's
   committed pre-registration is ever evaluated on futo-test.**
3. **FUTO's engine reads futo-test at exactly two presets** — published
   (the paper anchor) and dev-tuned (the bar) — once each, at N1, then its
   numbers are frozen for the phase.
4. **No retries, crashes reported as missing cells** — the UNSEALING_4 §3
   rule verbatim.
5. **Official dev is open** for both engines, unlimited (that is what a dev
   split is for). The fixed dev-subsample used for arm screening (§6) is the
   first 8,000 converted dev rows, frozen at N0.
6. **test-2400 is never touched.** Ledger 4, final. Consequence stated
   plainly: **the Phase-N candidate can never be test-2400-validated**; its
   no-regression bars (§7 B3/B4) are scored on val-9918 and the campaign's
   val batteries. If the user later wants a test-2400 claim for a new ship
   model, that is a user decision outside this plan.
7. Each milestone read appends a ledger entry naming authoriser, configs,
   and publication site.

## 4. Registered prohibitions (what must NOT be done)

* No training, checkpoint selection, preset fitting, data curation, or
  augmentation-statistic fitting on official **dev or test** rows or their
  sessions. (Augmentation statistics, if any lever needs them, come from
  swipe-1 **train** — same domain, zero benchmark contact.)
* No FUTO model output anywhere in training: no distillation from their
  engine, no pseudo-labels, no mined negatives, no selection signal. The
  license position of `FUTO_WEIGHTS_VERIFICATION.md` §0 carries unchanged —
  their weights are run for benchmarking only, their per-row outputs stay in
  the `futo_verify` scratch tree outside both repos.
* No asymmetric footing: any sweep/beam/lexicon our engine gets on this
  benchmark, theirs gets.
* No redefinition of metrics, strata, drop rules, or noise bands after M0
  numbers are seen.
* No re-decode of test-2400, no fifth read, under any framing.
* Teachers for any KD arm are our own models only.

## 5. Milestone protocol

Every milestone follows the campaign's unsealing pattern in miniature:
**(1)** all inputs frozen (checkpoints exported, sha256s recorded, dev-tuned
preset committed); **(2)** a numeric pre-registration section committed —
point predictions + bands per metric, derived from dev and val-futo-half
evidence only; **(3)** the decode, one shot; **(4)** results + scored
predictions appended; **(5)** ledger entry + push. Blind gate-first ordering
carries over wherever a pair gate exists (`pair_agreement.py` before decode).

## 6. Execution stages and levers

### N0 — infrastructure (no model contact with the benchmark)

1. `convert_futo_official.py` + converted `data/futo_dev.jsonl` /
   `data/futo_test.jsonl`, with row accounting, stratum n's, sha256s.
2. Seal registration `futo-test-49970`; verify the guard fires on a slice.
3. The §2 exact-trace hash checks, committed.
4. Freeze the 8,000-row dev screening prefix.

### N1 — measure the gap before touching a knob (dev + val only, plus the two FUTO test reads)

1. **FUTO engine:** cache dev emissions (floor + ceiling); sweep its preset
   on dev (tune/holdout halves, interior optimum). Decode futo-test at
   published + dev-tuned presets — its two frozen reads, the anchor and
   **the bar**. Also: one per-row val-9918 decode at its val-tuned preset
   (for the B2 bar and per-source strata; val is not sealed).
2. **Ship model:** cache dev emissions (3 seeds); sweep our preset on dev
   with the same machinery. No test contact yet.
3. **Gap decomposition on dev** (this is the lever-selection evidence):
   per-stratum (word length, orientation, speed quartiles, OOV), paired
   error overlap vs FUTO's dev output, and the same on the val FUTO half.
   The `DATA_TIERS.md` §3 quality-cascade facts frame the hypothesis space:
   the official splits contain exactly the traces the T2b motion/geometry
   cascade rejects (landscape, speed outliers), while our val FUTO half was
   curated — so the decomposition must separate "we lose on words/lexicon"
   from "we lose on motion regimes our curation under-weighted".
4. Ship model's val per-source numbers recomputed from existing dumps (no
   new decode) → the B3 HWS-half floor.

### M0 — baseline read (pre-registered per §5)

`v2kd-fresh-w1` 3 seeds, dev-tuned preset, futo-test, one config. This
prices the no-training lever first: **if symmetric dev tuning alone already
beats FUTO's dev-tuned engine on all five metrics every seed, the primary
goal is met by the ship model** and the phase reports that immediately —
remaining stages then only run if the user wants margin, not existence.

### N2 — training levers, screened on dev-8k + val (single seed s1234 each)

Control = the settled recipe (`PHASE_L.md` §3 mix at
`train_t3.npz, train_t3hws.npz ×2, tier_sw234.npz, tier_sw5q.npz` =
1,285,381 rows, ~72 % real-FUTO; coupled pair pw 0.3, slw 1.0/1.5; KD-fresh
student). The lever menu, with mechanisms matched to what N1 can show:

| arm | lever | mechanism |
|---|---|---|
| **N2a** | per-source loss weight | new `--source-loss-weight` in `train.py`/`train_v2.py`: a per-`--train-npz`-entry CTC loss weight (row→source is already recoverable via the dataset's cumulative-count searchsorted). Mix re-expressed with FUTO and HWS entries as separate files (`train_t3futo.npz` + HWS npz), verified row-equivalent to the control before the arm counts. FUTO-entry weight 1.5 — the slw2 move aimed at a source instead of a stratum |
| **N2b** | mix rebalance | drop the second `train_t3hws.npz` copy (the ×2 HWS double-count is the control's deliberate HWS emphasis; this measures what it costs on the FUTO domain) |
| **N2c** | domain-matched augmentation | only if N1 shows a motion-regime deficit: time-resampling / speed-profile jitter fitted to swipe-1 **train** percentiles (never dev), as a new aug alongside the existing rot/shear/timerev set |
| **N2d** | capacity | ch 256 under the settled pair+KD stack (Phase J priced raw ch 256 without the pair/KD instrument; that refutation does not cover this recipe). Run only if N2a/b/c leave a gap, because it also moves latency/size |

Choice rule: after N1, the two (at most three) arms whose mechanism matches
the measured error concentration are launched; launching anything outside
this menu requires a committed plan amendment first.

**Screening gate G-N2, fixed now** (single seed, each arm dev-swept
symmetrically with the control): dev-8k t1 ≥ control **+0.10** AND val
11-bar tally stays **11/11** AND val HWS-half t1 ≥ control **−0.20**. Pass →
the lever joins the N3 recipe; fail → recorded and dropped, no retuning.

### N3 — the proven stack with the winning levers

Coupled pair (s1234) with the surviving levers → agreement gate ≥ 95 %
(blind, committed before decode) → KD-fresh student w1 → if the student
clears G-N2 on dev-8k + full dev, **3 seeds** (1234/4321/7777). Optional M1
read if a mid-phase checkpoint on futo-test is worth one of the three reads
(default: skip, save the read for M2).

### N4 — M2 final read + close-out

Candidate 3 seeds, dev-tuned preset, futo-test, one config; bars scored
(§7); full val + layout batteries for B3; app-footing check for B4; export,
quantize (fp16w), golden fixture at the ship preset if the candidate is
recommended; docs (`RESULTS.md`, `MODEL_COMPARISON.md`) updated; ship
decision handed to the user with all footings stated.

## 7. The bars — pre-registered

* **B1 (primary — the user's goal):** on futo-test at M2 (or M0, if the
  ship model already does it), our candidate at its dev-tuned preset beats
  FUTO's engine at **its** dev-tuned preset on **all five** of
  t1 / t3 / t5 / ≤3 / 4+ — **seed-mean AND every seed** (the preferred
  footing), with exact paired McNemar on t1 resolving at p < 0.05 on ≥ 2 of
  3 seeds. All-five on seed-mean but not every seed, or McNemar unresolved,
  = a **qualified** FUTO-domain win, claimed at exactly that tier. n≈50 k
  resolves ~±0.15 pt on t1 — the power the 1,217-row stratum never had.
* **B2 (secondary):** val-9918 FUTO-half t1, candidate seed-mean ≥ FUTO's
  val-tuned engine on the same rows (the N1-measured bar) — the val analog
  of the +0.38 gap, closed.
* **B3 (floor — nothing given back):** all **11 campaign bars** cleared on
  **every seed** (the ship model's own property, [11,11,11]); AND vs the
  ship card seed-mean (88.750 / 92.773 / 93.473 / 91.373 / 87.387 + six
  layouts, `PHASE_M.md` §9) no axis lower by more than the pre-stated noise
  band: **0.15** on the five en axes, **0.75** on dvorak/dvorak-app (their
  measured seed sd reaches 0.52/0.85), **0.30** on
  azerty/qwertz/german/spanish; AND val HWS-half t1 ≥ ship model **−0.20**.
* **B4 (shipping footing):** at the app trie + shipping app preset, val
  seed-mean within **0.15** of the ship model's config-B card
  (89.377 / 93.680 / 94.467 / 92.563 / 87.727) on all five, or better.

A candidate that takes B1 by giving up B3/B4 is a failed candidate — the
user's directive is explicit that the overall/HWS win stays.

## 8. Cost estimate

N0+N1 are CPU-bound measurement (~1 day wall; the FUTO engine's 50 k-row
decode is ~35 min sharded ×24, dev sweeps ~1 h each). Screening arms are
~6–8 h GPU each (single model) / ~12 h (pair); N3 is one pair + three
students ≈ 2–3 days GPU. Whole phase ≈ 1–1.5 weeks with slack for one
surprise, well inside "unlimited training authorized".

## 9. Initial coarse expectations (sharpened per milestone, per §5)

Recorded so the phase's priors are scoreable: (a) the paper anchor
reproduces within ~0.3 pt; (b) dev tuning is worth +1.5…+2.5 t1 to both
engines (the FAIR_REMATCH lever transferred at that size); (c) the official
test lands well below the curated 95.x of the FUTO-half stratum for both
engines — mid-80s to low-90s t1 depending on OOV mass; (d) M0 is a genuine
coin-flip on t1 (we trail −0.38 on curated FUTO data but official-test's
uncurated regimes cut both ways), and if M0 loses any metric, 4+ or t1 are
the likely ones; (e) at least one of N2a/N2b passes G-N2. Exact bands per
milestone are committed before each decode, as always.

---

## 10. N0 — RESULT (executed 2026-08-15)

### 10.1 The benchmark files

`convert_futo_official.py` run on the HF originals
(`~/ctc-train/data/hf/{dev,test}.jsonl`); full accounting in
`~/ctc-train/data/futo_official_convert.stats.json` (sha256s of source and
output files included there):

| split | rows in | rows out | dropped <2 pts | dropped empty word | malformed | ≤3 (n) | 4+ (n) | sessions |
|---|---|---|---|---|---|---|---|---|
| **futo_dev** | 54,269 | **53,373** | 895 | 1 | 0 | 19,677 | 33,696 | 686 |
| **futo_test** | 49,970 | **49,208** | 759 | 3 | 0 | 17,906 | 31,302 | 687 |

The only material drop is single-point traces (~1.5 % of each split — taps,
undecodable by any featurizer here). Both engines are evaluated on the same
converted files, so every drop is symmetric. Caveat for the paper anchor:
FUTO's 92.94/93.30 was computed on all 49,970 rows; ours is on the 49,208
decodable ones (a ≤ ~0.2 pt definitional wobble, stated whenever the anchor
is quoted).

### 10.2 Seal registered and verified

`futo-test-49970` written into `test2400_seal.json`: 49,208 rows, 49,207
unique fingerprints (**one bit-exact duplicate trace inside the official
test split itself** — FUTO's own defect, kept as shipped),
`words_sha256 3b9f884b63549eb8…`. Guard verified: a 200-row slice is refused
at 100 % overlap with the split-generic refusal message (seal.py messages
made name-aware this commit). Cross-checks: `futo_dev` overlaps the sealed
set by 1 row (§10.3), `val_hwsfuto` by 0 — no false-trigger for dev-side
iteration. The frozen G-N2 screening prefix is `futo_dev.jsonl` rows
0–7,999, sha256 `1e7e9439…807d4495`.

### 10.3 Contamination — the §2 checks, executed

Method: 16-byte `scan_futo_sessions.trace_hash` (a–z-normalized word +
exact float64 x/y bytes) of every converted dev/test row, intersected with
(a) the full swipe-1 train index (939,550 rows), (b) every tier jsonl that
feeds or has fed a trainer.

| check | result |
|---|---|
| dev sessions ∩ train sessions | **0** of 692 |
| test sessions ∩ train sessions | **0** of 697 |
| dev sessions ∩ test sessions | **0** — FUTO's splits are fully contributor-disjoint |
| dev traces ∩ swipe-1 train | **0** |
| test traces ∩ swipe-1 train | **3** (bit-exact, across different sessions) |
| dev ∩ test traces | 1 |
| test traces ∩ `tier_t3futo` (ship mix) | **4 row-instances, all the word "a"** |
| test traces ∩ `tier_t2` | 2 (word "a"; tier not in the ship mix) |
| dev/test ∩ `tier_t3hws` / `tier_sw234` / `tier_sw5q` / `tier_t1` / `tier_t2b` | **0 everywhere** |

**Explanation of the non-zero cell, as §2 requires before proceeding.** The
3 shared traces are single-letter **"a" taps** whose 2-point quantized
coordinates collide bit-exactly across different contributors — not a split
leak (sessions are disjoint; a leak would come with its session siblings).
Exposure: ≤ 4 training row-instances out of 1,285,381 (0.0003 %) touching
at most 3 of 49,208 test rows (0.006 %), all in the ≤3 stratum, and **FUTO's
own models trained on the identical train rows** — the exposure is symmetric
to the row. Verdict: recorded, immaterial, phase proceeds. No row is
removed from either side (removing them would *desynchronize* us from the
benchmark FUTO's numbers describe).

### 10.4 Environment re-verification

ExecuTorch 1.2.0 x86_64 venv at `~/ctc-train/futo_verify/etvenv` loads and
executes both hash-verified `.pte` (encoder 26.1 ms, decoder 8.8 ms,
single-thread smoke) — the `FUTO_WEIGHTS_VERIFICATION.md` §2 environment is
intact. N0 complete; N1 (dev sweeps both engines, FUTO's two frozen
futo-test reads, gap decomposition) is next.
