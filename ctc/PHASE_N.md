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

## 11. N1 step 1 — RESULT: the symmetric dev sweeps (2026-08-15)

Both engines swept on official dev with **identical wide spans**
(γ 0–3.0 × λ 0–1.8 × β 0–1.3 × gp {0.05…0.5} × bp {0.25…1.2}; tune
half rows 0:26686, holdout 26686:53373), then a widened fine grid per the
interior-optimum rule: FUTO's wide winner hit the λ boundary (1.8, flagged
by the sweep), ours picked edge prune values on a flat prune surface
(spread ≤ 0.04). Artifacts:
`~/ctc-train/phaseN/{ours,futo}_dev_{wide,fine}.{json,log}`.

### 11.1 Adopted dev-tuned presets — both interior, holdout-confirmed

| engine | γ | λ | β | gp | bp | tune-half t1 | holdout-half t1 |
|---|---|---|---|---|---|---|---|
| **ours** (`v2kd-fresh-w1` s1234) | **0.725** | **1.75** | **0.35** | **0.05** | **1.2** | 79.40 (+3.11 over baseline) | 79.68 (+2.79) |
| **FUTO ceiling** | **0.65** | **2.2** | **0.55** | **0.3734** | **0.7** | 78.50 (+1.26 over published) | 79.01 (+1.13) |

Gains generalize tune→holdout for both (no sweep overfit), and both fine
winners reproduce their wide-grid full-dev numbers to ≤ 0.01 — converged.

### 11.2 Full official dev (53,373 rows), the N1 scoreboard

> **⚠ ERRATUM (2026-08-15, found the same day, before any test read).** The
> table as first committed was computed by the sweep scripts, whose target
> matching lowercases but does **not** a–z-normalize (`futo_sweep.py:130`,
> and the same convention inside `sweep_scoring`'s tally). Canonical splits
> are pre-normalized so this never mattered before; the raw official split
> is not — **14.11 %** of dev words carry punctuation ("don't", "the,"), and
> every such row was scored an automatic miss for **both** engines. The
> depressed numbers (ours 79.54 / FUTO 78.76 t1, and the whole first
> version of this table) were artifacts of that convention. Because the
> auto-miss set is identical for both engines and constant across presets,
> the **tuned presets remain valid argmaxes** and the inter-engine deltas
> were approximately preserved — the correction moves levels, not the
> ordering. Corrected numbers below are recomputed from **per-row dumps**
> (harness for FUTO, `eval_beam` for ours) with a–z normalization applied
> to both target and prediction — the campaign's standard convention. The
> FUTO harness path was verified digit-exact against the committed
> val-9918 record (87.48 / 92.31 / 93.03 / 89.76 / 86.29 reproduced) before
> being trusted for this.

Corrected scoreboard (normalized convention, per-row decodes, s1234):

| engine / preset | t1 | t3 | t5 | ≤3 (19,677) | 4+ (33,696) |
|---|---|---|---|---|---|
| FUTO, published | *(dev not decoded at published; test read in §12.1)* | | | | |
| **FUTO, dev-tuned** | **90.51** | **94.18** | **94.82** | **93.71** | **88.65** |
| ours s1234, dev-tuned | **91.25** | **95.02** | **95.57** | **95.86** | **88.55** |
| Δ (ours − FUTO) | **+0.74** | **+0.84** | **+0.75** | **+2.15** | **−0.10** |

**On FUTO's own dev split, at symmetric tuning, the ship model leads
t1/t3/t5/≤3 — t1 McNemar-resolved at p = 4.7e-18 — and loses 4+ by 0.10.**
The `UNSEALING_4.md` §8.4 FUTO-half pattern sharpens: the domain contest is
entirely about **words of 4+ characters at ordinary motion**; short words
are ours by +2.15 even on their data (§12.3).

### 11.3 The paper anchor cannot describe the raw split — flagged before any test read

Expectation §9(a) is already refuted on dev: FUTO's paper reports 92.94 t1
(enc-only) on their **test**, but their full engine at the published preset
scores 77.56 on raw dev, and target-OOV is only **3.39 %** (1,809/53,373 vs
the STRIP trie) — OOV cannot explain a ~15 pt difference. The paper number
is therefore almost certainly measured on a **quality-filtered subset**
(their training-time cascade — dictionary words, portrait, speed bounds —
`DATA_TIERS.md` §3) and/or a different scoring convention (in-vocab-only).
Consequence, stated before the test reads: the anchor comparison in §1.1
will be scored against the harness's **in-vocab** metrics and against a
cascade-filtered subset computed from the same frozen dumps (disclosed
analysis, not a bar; both engines see identical rows so every bar is
unaffected). Nothing about B1–B4 changes.

### 11.4 Ship-model val per-source baselines (from the existing s1234 E1 dump, no new decode)

val-9918 at AOSP STRIP / E1: **FUTO half 95.08 / 98.28 / 98.75 (t1/t3/t5,
n=4,942), HWS half 82.19 / 87.14 / 88.20 (n=4,976)** — aggregate 88.62
reproduces the PHASE_M §9 s1234 row exactly. These are the B2 our-side and
B3 HWS-floor reference points; FUTO's val-tuned per-row val decode lands
with the N1 step-2 reads.

## 12. N1 step 2 — RESULT: the frozen FUTO test reads, the B-bars, and the decomposition

Executed 2026-08-15: FUTO's engine decoded per-row (sharded ×12, `beamD`,
width 100, top-k 8, STRIP trie) on futo-test at the **published** preset and
at its **dev-tuned** preset — its two frozen reads under §3 rule 3, now
spent — plus open-split dumps: futo-dev @dev-tuned, val-9918 @val-tuned.
Dumps: `~/ctc-train/phaseN/futo_{test_published,test_devtuned,dev_devtuned,
val_valtuned}.jsonl`. All numbers below use the normalized-target
convention (§11.2 erratum).

### 12.1 The B1 bar — FUTO's engine on its own official test (49,208 rows)

| preset | t1 | t3 | t5 | ≤3 (17,906) | 4+ (31,302) | in-vocab t1 |
|---|---|---|---|---|---|---|
| published | 88.85 | 93.61 | 94.40 | 93.11 | 86.40 | 92.25 |
| **dev-tuned (THE BAR)** | **90.42** | **94.31** | **95.01** | **93.73** | **88.52** | 93.82 |

Dev-tuning is worth +1.57 t1 to FUTO on its own test — within the §9(b)
predicted lever size. Measured FUTO dev→test shift @dev-tuned:
−0.09 / +0.13 / +0.19 / +0.02 / −0.13 — the two splits are near-twins, as
their contributor-disjoint 50/50-style construction implies.

**The paper anchor reconciles.** FUTO's paper reports 92.94 (enc-only) /
93.30 (refined) on this split; the harness's **in-vocab** t1 is 92.25 at
the published preset and 93.82 dev-tuned. The paper's convention is
evidently in-vocab (or filtered) scoring — on the raw split with OOV-as-miss
its engine scores 88.85. §9(a) is scored **wrong as literally written**
(raw-convention reproduction was impossible for any engine) and **right in
substance** under the in-vocab reading (92.25 vs 92.94, within port/protocol
wobble of the C++ beam). All B-bars are unaffected: both engines are scored
on identical rows under identical conventions.

### 12.2 The B2 bar — val-9918 FUTO half, FUTO's val-tuned engine (real per-row decode)

| rows | t1 | t3 | t5 |
|---|---|---|---|
| val ALL (9,918) | 87.48 | 92.31 | 93.03 | — digit-exact vs the committed `FAIR_REMATCH.md` §2 val row |
| **val FUTO half (4,942) = B2 bar** | **95.65** | 98.73 | 99.05 |
| val HWS half (4,976) | 79.36 | 85.93 | 87.06 |

Ship model s1234 (E1) val FUTO half: 95.08 → **current B2 gap −0.57** (the
val analog of the −0.38; seed-mean pending s4321/s7777 val dumps).

### 12.3 Gap decomposition on dev (both engines dev-tuned, paired per row)

`phase_n_decomp.py`, full table in `~/ctc-train/phaseN/dev_decomp.json`:

| stratum | n | ours | FUTO | Δ | McNemar p |
|---|---|---|---|---|---|
| ALL | 53,373 | 91.25 | 90.51 | **+0.74** | 4.7e-18 |
| len ≤3 | 19,677 | 95.86 | 93.71 | **+2.15** | 6.0e-49 |
| len 4–5 | 13,515 | 89.88 | 90.10 | **−0.22** | 0.24 |
| len 6–8 | 14,069 | 88.49 | 88.54 | −0.05 | 0.78 |
| len ≥9 | 6,112 | 85.77 | 85.67 | +0.10 | 0.68 |
| in-vocab speed q1 (slow) | 12,891 | 94.92 | 91.23 | **+3.68** | 1.7e-67 |
| speed q2 / q3 / q4 | 12,891 ea | — | — | −0.21 / −0.16 / −0.26 | 0.13–0.33 |
| npts q1 (short traces) | 13,336 | 95.54 | 92.57 | **+2.97** | 4.2e-47 |
| npts q2 / q3 / q4 | ~12.7 k ea | — | — | −0.25 / +0.13 / +0.10 | ≥0.16 |
| landscape / portrait | 1,539 / 51,588 | — | — | +0.32 / +0.76 | 0.55 / 3.2e-18 |

**Reading.** Our entire lead lives in short words, short traces and the
slow-speed quartile — where FUTO's engine is weak and ours is not. On the
long-word bulk (4+ at ordinary speeds) the engines are statistically level,
with FUTO consistently a nose ahead in the middle bins (−0.05…−0.26, none
individually resolved, sign-consistent across five adjacent strata). 4+
aggregate: **88.55 vs 88.65, −0.10.** Orientation is NOT a factor (+0.32 on
landscape). There is no motion-regime catastrophe anywhere — the raw-domain
"collapse" feared after §11.3 was entirely the scoring-convention artifact.
**Phase N's real fight is ±0.2 pt on 4+-char words; everything else is
already won.**

## 13. M0 — PRE-REGISTRATION (committed BEFORE any decode of our model on futo-test)

Milestone read 1 of the §3 cap of 3. Subject: the **ship model**
`v2kd-fresh-w1`, the three frozen fp32 exports whose sha256s are committed
in `UNSEALING_4.md` §2.3 (`b71911da…`, `f7cb72c0…`, `c55cc3b0…`).

**The exact configuration, frozen now.** One config: STRIP trie 146,964,
beam 100, top-k 8, OOV = miss, normalized-target convention (§11.2), preset
**`0.725,1.75,0.35,0.05,1.2`** — the §11.1 dev-tuned preset, fitted on
s1234's dev emissions and applied to all three seeds unchanged (a slight
handicap to s4321/s7777, disclosed; FUTO's engine runs its own dev-tuned
optimum). **Hard cap: 3 decodes (one per seed), no retries — a crash is a
missing cell scored as a failure — no other preset, no grid scoring of test
emissions, nothing re-run after numbers are seen.** Published metrics are
recomputed from the per-row dumps under the stated convention
(`eval_beam`'s internal print uses raw-form targets and will read low; the
dump recomputation is the number, deterministic from committed artifacts).

**Numeric expectations.** Point prediction = our dev number (§11.2) + FUTO's
measured dev→test shift at its dev-tuned preset (§12.1) — the only
model-free transfer estimate available. Bands = point ± 0.45 (± 0.55 for
≤3, its documented record), covering seed spread (val sd ≤ 0.2) plus shift
uncertainty (FUTO's shifts were ≤ 0.19):

| metric | dev (s1234) | shift | **point** | **band** | FUTO bar | predicted Δ |
|---|---|---|---|---|---|---|
| t1 | 91.25 | −0.09 | **91.16** | 90.71–91.61 | 90.42 | **+0.74** |
| t3 | 95.02 | +0.13 | **95.15** | 94.70–95.60 | 94.31 | +0.84 |
| t5 | 95.57 | +0.19 | **95.76** | 95.31–96.21 | 95.01 | +0.75 |
| ≤3 | 95.86 | +0.02 | **95.88** | 95.33–96.43 | 93.73 | +2.15 |
| **4+** | 88.55 | −0.13 | **88.42** | 87.97–88.87 | 88.52 | **−0.10** |

**Registered expectations:**

* **M0-1:** t1, t3, t5, ≤3 all clear the FUTO bar on the seed-mean **and on
  every seed**, with t1 McNemar resolved p < 0.01 on every seed (dev net was
  +393 rows at p = 4.7e-18; the test analog is ~+360).
* **M0-2:** **4+ is the coin flip and the predicted MISS** (point −0.10
  against the bar, band spanning both outcomes). If B1 fails at M0, it fails
  on 4+ and nothing else.
* **M0-3:** consequence pre-stated: a 4-of-5 M0 is **not** a B1 pass and is
  written as the baseline it is; Phase-N training then targets 4+ per §12.3
  (levers N2a/N2b — FUTO-source emphasis and short-word/HWS de-emphasis —
  are the mechanism-matched arms; N2c is NOT indicated, §12.3 shows no
  motion-regime deficit). A clean 5-of-5 every-seed M0 means **B1 is met by
  the ship model with no training at all**; remaining work would be scoped
  to B2 and margin, and that decision is handed up.

The ledger entry for this read (and the two §12.1 FUTO reads) is appended to
`futo-test-49970.unsealings` with this commit and the decode follows it.

## 14. M0 — RESULT (decoded 2026-08-15, exactly as registered)

Three decodes, one per seed, no retries, no crash; seal override logged on
each; dumps `~/ctc-train/phaseN/m0_v2kd-fresh-w1{,-s4321,-s7777}.jsonl`.
Metrics recomputed from dumps under the registered convention; McNemar is
exact paired two-sided on t1 against FUTO's frozen dev-tuned test dump.

| metric | s1234 | s4321 | s7777 | **seed-mean** | bar | **Δ mean** | every seed? | band (§13) |
|---|---|---|---|---|---|---|---|---|
| t1 | 91.32 | 91.33 | 91.21 | **91.289** | 90.42 | **+0.87** | **yes** | in |
| t3 | 95.18 | 95.19 | 95.18 | **95.182** | 94.31 | **+0.87** | **yes** | in |
| t5 | 95.70 | 95.74 | 95.71 | **95.719** | 95.01 | **+0.71** | **yes** | in |
| ≤3 | 96.25 | 96.21 | 95.98 | **96.147** | 93.73 | **+2.42** | **yes** | in |
| **4+** | 88.51 | **88.54** | 88.48 | **88.510** | 88.52 | **−0.010** | **no** [miss, pass, miss] | in |

McNemar t1: +447 / +450 / +391 net rows, p = 1.7e-23 / 3.0e-24 / 1.7e-18 —
**resolved on every seed.**

**Verdict: B1 NOT MET at M0 — 4 of 5, and the registered coin flip fell as
registered.** All three §13 expectations score **RIGHT**: M0-1 (four metrics
clear every seed, McNemar every seed), M0-2 (4+ is the only miss — though
the measured −0.010 is twenty times closer than the −0.10 point prediction;
one seed individually clears), M0-3's consequence now binds. Band coverage
**10/10** (all five metrics, all in, on both footings of reading).

What the baseline already establishes, plainly: **on FUTO's own official
test split, at symmetric dev-tuned footing, the 2.91 MB ship model beats
FUTO's engine on t1/t3/t5/≤3 on every seed with the top-1 lead resolved at
p < 1e-17 — and the entire remaining contest is 0.010 pt (≈ 3 rows of
31,302) on 4+-char words.** The −0.38 FUTO-half story of `UNSEALING_4.md`
§8.4 is now measured at scale: it is a long-word residual, currently a
statistical tie, and everything else on their domain is won outright.

### 14.1 The branch taken (per the pre-registered M0-3 rule)

4-of-5 → Phase-N training proceeds, targeted at 4+, with the
mechanism-matched arms pre-selected in §13:

* **N2a** — FUTO-source loss emphasis: `--source-loss-weight` (new,
  per-`--train-npz`-entry CTC loss weight), mix re-expressed
  `train_t3futo.npz + train_t3hws.npz×3 + tier_sw234.npz + tier_sw5q.npz`
  with weights `1.5,1,1,1,1,1`. *(Corrected at launch, before any result:
  the first version of this bullet wrote HWS ×2, which would have silently
  dropped one effective HWS copy — `train_t3.npz` itself contains 77,467
  HWS rows, so composition-preserving re-expression needs ×3.)* Disclosed:
  the re-expressed pool is 1,284,662 rows = control −719 (0.06 %) — a
  documented delta, not a claimed identity.
* **N2b** — drop the second `train_t3hws.npz` copy (HWS ×1): measures what
  the control's deliberate HWS double-count costs on the FUTO domain.

Both are coupled-pair arms (`train_v2.py`, L1 recipe otherwise verbatim,
seed 1234), screened against the fully-evaluated `v2pair-s1234` control
under gate **G-N2** (§6-N2): dev-8k t1 ≥ control +0.10 **and** val 11-bar
tally 11/11 **and** val HWS-half t1 ≥ control −0.20 — with the arm's 4+
dev-8k delta reported as the primary curiosity even though the gate is t1.
N2c stays un-launched (§12.3: no motion-regime deficit exists); N2d waits
on the N2a/N2b verdict.

### 14.2 N2a/N2b launch record + pre-stated expectations (committed before launch)

`--source-loss-weight` implemented in `train_v2.py` this commit: per-entry
weight folded into the same weighted mean as slw (`weighted_ctc`), wrapper
dataset draws no randomness, empty flag reproduces the stock path (unit
check: src_w = 1 is bit-identical; the shuffle stream is untouched).

| arm | mix (`--train-npz`) | lever |
|---|---|---|
| `n2a-srcw15` | `train_t3futo,train_t3hws×3,tier_sw234,tier_sw5q` | `--source-loss-weight 1.5,1,1,1,1,1` |
| `n2b-hws1` | `train_t3,train_t3hws,tier_sw234,tier_sw5q` | HWS mass 18.0 % → 12.6 %, no new code |

Everything else is the `PHASE_L.md` §3 command verbatim (seed 1234, init
1111/2222, slw 1.0/1.5, pw 0.3 ramp 5000+15000, 188 k steps, `--workers 0`,
detached; resume = same argv + `--resume ckpt/<run>/last.pt`). Gate-first
blind order at harvest: `pair_agreement` ≥ 0.95 → commit gate + prediction →
decode dev-8k + val battery.

**Expectations, recorded to be scoreable.** N2a: dev-8k t1 +0.1…+0.3 and
4+ up, val FUTO-half +0.1…+0.4, val HWS-half −0.1…−0.4 (the gate's −0.20
floor is genuinely at risk — that is what the gate is for), val aggregate
≈ wash, 11/11 retained. N2b: the larger FUTO-domain shift of the two and
the larger HWS bill (−0.3…−0.8 — likely G-N2 failure on the HWS floor);
launched anyway because the pair (N2a = price of a *soft* reweight,
N2b = price of a *hard* one) brackets the trade the final recipe must buy.
If both pass, the stronger dev-8k 4+ wins; if both fail, N2d (capacity)
opens with a new registration.

### 14.3 N2 gates — measured and committed BEFORE any decode (2026-08-15)

Both arms reached 188 k clean (n2a header confirms the source-weight table:
FUTO 927,869 rows @1.5; totals 1,284,662 / 1,208,633 as registered).
Label-free per-frame agreement gates:

| pair | agreement | verdict |
|---|---|---|
| `n2a-srcw15` | **98.34 %** | PASS |
| `n2b-hws1` | **98.28 %** | PASS |

Ten of ten coupled pairs across three phases now pass the gate.
**Committed working-band prediction for both arms (the PHASE_K §8.5 band,
unchanged):** val pair-mix t1 ≥ 88.30 and ensemble greedy ≥ 55 %. Control
reference numbers, computed from the existing Phase-L dump (no decode):
`v2pair-s1234` pair val t1 **88.90** — FUTO half **95.59**, HWS half
**82.25**. G-N2's HWS floor is therefore **≥ 82.05**. Decodes follow this
commit: pair val-9918 + six layout bars per arm, dev-8k (frozen prefix) for
both arms and the control.

### 14.4 N2 RESULT — both arms FAIL G-N2; source reweighting is refuted for this domain

Decoded after the gate commit, exactly as registered (working band: both
pairs pass — val t1 88.47 / 88.45 ≥ 88.30, greedy ✓). G-N2 scoring, all
prongs, control = `v2pair-s1234`:

| prong | control | `n2a-srcw15` | `n2b-hws1` | rule |
|---|---|---|---|---|
| dev-8k pair t1 | 91.29 | 91.36 (**+0.07**) | 90.88 (**−0.41**) | ≥ +0.10 → **both FAIL** |
| val 11-bar tally | 11/11 | **9/11** (≤3 91.21 −0.06; t3 92.60 exact tie) | **10/11** (≤3 91.27 exact tie) | 11/11 → **both FAIL** |
| val HWS half | 82.25 | **81.61** | **81.53** | ≥ 82.05 → **both FAIL** |
| dev-8k 4+ (curiosity) | 88.80 | 88.80 (+0.00) | 88.95 (+0.15) | — |
| val FUTO half | 95.59 | 95.37 (−0.22) | 95.41 (−0.18) | — |

Layout detail: all six layout bars pass for both arms (n2a dvorak 92.47 /
dv-app 91.78 / azerty 84.55 / qwertz 83.91 / german 80.76 / spanish 89.76;
n2b 91.01 / 90.52 / 84.40 / 84.92 / 81.31 / 89.19) — the failures are
entirely on the en axes and the dev gate.

**§14.2 expectations, scored.** N2a: **wrong on all four sub-predictions**
— dev-8k below band, 4+ flat, val FUTO-half *negative* (−0.22), HWS worse
than band, 11/11 lost. N2b: "larger FUTO-domain shift" **wrong on t1**
(−0.41; its HWS cut cost dev ≤3 95.81 → 94.37 — the HWS corpus is
evidently where short-word skill comes from), **right** on dev 4+ being the
larger (+0.15), **right** on the HWS bill band (−0.72) and on the predicted
G-N2 failure. The durable finding mirrors Phase-M's E4: **the control
mixture is already at its optimum for this domain — pushing FUTO emphasis,
softly (loss weight) or hard (mass), buys nothing on dev and pays on both
val halves.** The 4+ residual is not a data-balance artifact. Both arms are
dropped, not retuned, per the rules.

One more measurement from the same dumps (no decode): the **pair lift over
the single model on dev-8k is +0.17 t1 / +0.42 ≤3 / +0.04 4+** — the
coupled pair does not close 4+ either; two-point averaging buys short
words, not long ones.

## 15. Plan amendment (committed before execution): N2e — the decode-side lever — then N2d

§14.2's rule fires: N2d (capacity) **opens**. But the M0/N2 evidence
identifies a cheaper, mechanism-matched lever that must be priced first,
because it costs hours and no training:

**N2e — B1-objective preset for our engine on dev.** M0's only miss is
−0.010 on 4+ while holding **+2.42 of slack on ≤3**; `FAIR_REMATCH.md`
showed scoring presets trade exactly this axis (λ/β moved ≤3 and 4+ in
opposite directions for both engines). `sweep_scoring.py --objective
minmargin --bars <FUTO dev numbers>` is a direct encoding of "clear every
bar". Protocol:

1. **Fix the §11.2 normalization defect at its root** — `sweep_scoring`
   (and `futo_sweep`) target construction gets the a–z normalization the
   rest of the campaign uses. Validation gate: the analytic path at the
   §11.1 preset must reproduce the dump-recomputed full-dev numbers
   (91.25 / 95.02 / 95.57 / 95.86 / 88.55) before any new sweep is trusted.
2. Sweep on dev (tune half 0:26686, holdout 26686:53373, interior-optimum
   rule) with `--objective minmargin`, bars = **FUTO's dev-tuned dev
   numbers `90.51, 94.18, 94.82, 93.71, 88.65`** (§11.2).
3. **Success = a holdout-confirmed interior preset whose five dev margins
   are all positive** (worst margin > 0 on the holdout half as well).
   Then **M2 pre-registers the ship model (same 3 seeds) at that preset** —
   B1 with zero training, B3/B4 untouched by construction (the app preset
   and artifact do not change; the benchmark preset is a benchmark
   configuration, exactly as E1 always was).
4. **Declared objective asymmetry:** FUTO's engine keeps its t1-optimal
   dev preset (an engine's own best tuning); ours optimizes the B1
   conjunction. Both engines had identical grid machinery and identical
   dev data; the difference is only which scalarization each side's
   objective calls for, and it is disclosed here in advance.
5. If no such preset exists, **N2d launches**: coupled pair at **ch 256**
   (control mix, L1 recipe otherwise, s1234) → gate → G-N2 screening →
   KD-fresh students; registered in full at that point.
