# Synthetic swipe-trace generator v2 — quality-gap analysis and design options

**Date:** 2026-08-19 · **Status: foundation document, AWAITING THE USER'S
DETAILED REQUIREMENTS before any implementation.** Nothing in Part 3 is
authorized to build; it is a prepared plan the user can steer or overrule.

**Method note.** Every number in Part 1 is measured, not quoted from memory:
the measurement script is `ctc/synth_gap_audit.py` (three stages: `data`,
`metrics`, `classifier`; outputs under `~/ctc-train/synth_gap/`). Russian is
the gold validator — the Yandex valid-10k (eval-only footing per
`YANDEX_LICENSE_RESEARCH.md`, the established 9,416 default-grid rows) gives
real ЙЦУКЕН swipes, and the v1 mechanism (`cyrillic_synth.py` semantics,
verbatim donor policy) synthesized **one trace per real row for the SAME word
on the SAME layout**, so every difference is generator error — no word, layout
or lexicon confound. GPU was idle throughout (Phase O's five trainings had
completed); no training run beyond the tiny MLP classifier was launched.

**v1 in one sentence.** A donor English trace whose collapsed polyline has the
same *vertex count* as the target word's is drawn uniformly, its tangent/normal
residuals are re-anchored onto the target word's ideal polyline by
`layout_aug.warp_path` (monotone-DP correspondence, endpoint pins,
vertex-absolute arc remap), and words are drawn by CKDT `255 − rank` weight.

---

## Part 1 — Quality-gap analysis

### 1.1 Word-matched distributional comparison

n = 9,416 matched pairs (real Yandex valid row vs v1 synth for the same word,
`ru_jcuken_default`). Features are the campaign's `[2,64]` time-uniform
samples, so step length = local speed in units of (board units per 1/63 of
trace duration); absolute duration is not in the features for either set.
KS = two-sample Kolmogorov–Smirnov statistic.

| metric | real mean | synth mean | real p50 | synth p50 | **KS** |
|---|---|---|---|---|---|
| **step_cv** (speed variability) | 0.741 | **1.354** | 0.729 | 1.175 | **0.597** |
| **step_max** (peak per-step speed) | 0.081 | **0.263** | 0.071 | 0.235 | **0.519** |
| **sharp_turns** (>60° between steps) | 1.71 | **5.71** | 1 | 5 | **0.443** |
| **turn_mean** (rad) | 0.153 | **0.319** | 0.142 | 0.293 | **0.405** |
| turn_total (rad) | 9.38 | 19.41 | 8.71 | 17.86 | 0.400 |
| straightness (path/ideal) | 1.234 | 1.353 | 1.179 | 1.261 | 0.174 |
| key_cover (word keys passed within rx) | 0.820 | 0.739 | 0.875 | 0.800 | 0.132 |
| end_d (endpoint→last key) | 0.104 | 0.078 | 0.078 | 0.065 | 0.123 |
| start_dwell (leading samples on 1st key) | 5.33 | 4.26 | **3** | **1** | 0.110 |
| dwell_frac (steps < 0.004) | 0.173 | 0.214 | 0.111 | 0.159 | 0.109 |
| path_len | 1.754 | 1.964 | 1.634 | 1.838 | 0.091 |
| step_mean | 0.0278 | 0.0312 | 0.0259 | 0.0292 | 0.091 |
| dwell_run_max | 6.94 | 8.24 | 4 | 5 | 0.086 |
| start_d (endpoint→first key) | 0.050 | 0.056 | 0.043 | 0.049 | 0.081 |
| end_dwell | 2.88 | 2.95 | 0 | 0 | 0.076 |
| speed_asym (first-16/last-16 step mean, p50) | 0.908 | 0.944 | — | — | 0.059 |
| dup_frac (zero-length steps) | 0.019 | 0.026 | 0 | 0 | 0.046 |

**The three biggest measured gaps, ranked:**

1. **The speed profile is wrong** (step_cv KS 0.60, step_max KS 0.52). Synth
   traces hit peak per-step speeds 3.2× real (0.263 vs 0.081 — a quarter of
   the board in one 1/63-of-duration step) and their speed variability is
   nearly double. Mechanism pinned in §1.2.
2. **The word-length mix is wrong by an order of magnitude** (§1.3): the v1
   training draw is 3.3 % ≤3-letter words against real usage's 35.6 % (this
   measurement) / 38.7 % (PHASE_O §2.1, decoded-rows footing).
3. **Synth transit is jagged where real transit is smooth** (sharp_turns
   3.3×, turn_mean 2.1×). Partly the movement-frame rotation at segment
   switches (the known Phase-H "kinks"), partly the sampling asymmetry of
   §1.5, partly stretched-transit aliasing from mechanism §1.2.

Secondary but real: the **start side** (real swipers plant the finger — median
3 leading samples inside the first key vs synth's 1; start_d 0.050 vs 0.056 —
the known PHASE_I_DATA §5 defect restated in dwell terms), the **end side in
the opposite direction** (real Yandex lifts late and sloppy, end_d 0.104 vs
synth's too-clean 0.078 — v1's "end-side matched" claim was about *hit rate*
0.656/0.647, which still holds; the distance distribution does not match), and
**key_cover** (synth paths miss one word-key in four within a key half-width
vs one in five real — transit fidelity, not endpoint fidelity).

### 1.2 The kinematic mechanism, pinned: vertex-count matching ignores geometry

v1 matches donors on collapsed vertex count ONLY. Measured over the same
9,416 matched draws (same RNG stream as the generated set):

| dst/src ideal-polyline length ratio | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| whole word | 0.36 | 0.63 | 0.82 | 1.05 | 1.67 |
| **per segment** | **0.16** | 0.50 | 0.83 | 1.27 | **3.63** |

**25.4 % of segments are compressed by more than 2× and 11.6 % stretched by
more than 2×.** `warp_path`'s arc remap scales the *between-dwell* sample
spacing by exactly `mid_ratio = (L_d − 2h)/(L_s − 2h)` — the residuals are
transferred in absolute units (correct, by design) but the **implicit speed
profile carried by the time-uniform spacing is multiplied by the segment
length ratio**, and nothing re-times it. A short English segment mapped onto a
long Russian one becomes a superhuman jump (the step_max blowup); a long one
mapped onto a short one becomes fake dwell (the dwell_frac excess: 0.214 vs
0.173). The whole-word ratio p50 of 0.82 also explains why synthetic paths are
*longer* than real ones for the same words (path_len +12 %): donors are drawn
from English words that are on average geometrically longer than the
frequency-weighted Russian targets, and the residual magnitudes ride along.

This is the single largest measured defect and it is **a matching + re-timing
bug, not a residual-transfer bug** — the Phase-H invariants (endpoints exact,
ideal→ideal, absolute near-key geometry) all still hold.

### 1.3 The word-length mix (train-draw defect, already flagged by Phase O)

| set | ≤3 | 4–6 | ≥7 | mean len |
|---|---|---|---|---|
| real Yandex valid (kept rows) | **0.356** | 0.438 | 0.205 | 4.78 |
| v1 train draw (CKDT `255 − rank`) | **0.033** | 0.291 | 0.675 | 7.91 |

The `255 − rank` weight is a *compressed dictionary rank*, not a token
frequency — the top word gets weight 255, the 50,000th gets 1, a ~255:1 range
where real token frequencies span ~10⁵:1. The result: a training corpus with
10.7× too few short words and a mean word two-thirds longer than real usage.
Phase O measured the downstream signature (§2.1: the ru model's synthesis
holdout is 3.3 % short words vs real 38.7 %, and the short-word stratum is
where the probe inversion is sharpest); PHASE_O already carries the corpus-
frequency fix as "the first thing a Phase P should do". Part 1 confirms the
size of the defect at the generator input.

### 1.4 Discriminability — the classifier IS the quality metric

2-layer MLP (128→64→2, Adam, 40 epochs — `synth_gap_audit.py --stage
classifier`), **word-matched AND word-disjoint**: both classes share the word
multiset, and train/test words never overlap, so the classifier cannot use
lexical identity or the length mix — only style. 15,740 train / 3,092 test
rows. 50 % = indistinguishable.

| feature view | dim | test acc |
|---|---|---|
| **speed profile only** (63 step lengths) | 63 | **0.904** |
| speed + turn angles | 125 | 0.880 |
| raw coordinates | 128 | 0.772 |
| turn angles only | 62 | 0.759 |
| endpoints (first/last 4 points) | 16 | 0.663 |
| *unmatched control* (v1 train draw vs real, coords, random split) | 128 | 0.900 |

**Verdict: v1 synthesis is 90 % separable from real traces, and the speed
profile alone carries essentially the whole signal.** The ablation ordering is
the same story as §1.1–1.2 from an independent instrument: kinematics (0.904)
≫ geometry (0.772) > local shape (0.759) > endpoints (0.663). Endpoint
statistics — the only distributional gate v1 ever had — are the *least*
discriminative view, which is exactly why v1 passed its gates while carrying a
KS-0.6 speed defect. The unmatched control (0.900 on coordinates alone, where
the matched coords view reads 0.772) shows the word/length mix adds a further
independently-detectable gap.

### 1.5 Facts the features drop — and the sampling asymmetry they hide

Raw real-trace facts (the generator has no time axis at all):

| | p1 | p25 | p50 | p75 | p99 |
|---|---|---|---|---|---|
| points/trace | 9 | 24 | 42 | 63 | 147 |
| duration ms | 137 | 398 | **701** | 1,029 | 2,112 |
| median inter-sample dt ms | 8 | 16.5 | **17** | 17 | 30 |
| max inter-sample gap ms | 13 | 26 | 40 | 65 | 146 |

Two reads. (a) **Real Russian swipes are fast**: median 701 ms vs the HWS
English median of 1,113 ms (PHASE_I_DATA §1) — the donor pool's tempo is a
different population's. Absolute duration never reaches the model, but tempo
*shape* does, via dwell/transit proportions. (b) **A sampling-rate asymmetry
the classifier can see**: the featurizer resamples to 60 Hz then to 64
index-uniform points. At median 701 ms a real ru trace has ~43 true 60 Hz
nodes upsampled to 64 — locally piecewise-linear, i.e. *smooth* at the step
scale — while English donors at median 1,113 ms carry ≥67 nodes and get
*downsampled*, keeping per-step jitter. Part of the turn_mean/sharp_turns gap
is therefore a resampling artifact baked into the donor bank, not human
behavior at all.

### 1.6 Downstream correlation — which gaps explain the known failures

* **The ~13-pt real-over-synth training gap** (PHASE_I_DATA §6: real arm 89.64
  vs synth arm 76.21 at λ 1.1; ~12 at matched λ 2.0). The synth-trained
  model's greedy collapses to 37 vs the real arm's 75 — its *emissions* are
  weak on real input, exactly what training on 90 %-separable kinematics
  predicts: the encoder learns speed/dwell statistics real traces do not have.
  The gap is plausibly decomposable as (i) length mix — training emphasis 20×
  off on a stratum that is 36–39 % of real usage; on the real probe the
  synth model's ≤3 stratum trails the real-data arm's by ~8–10 pts
  (86.44 vs ~94, different-footing caveat); (ii) kinematic mismatch — the
  capacity the model spends modeling stretched transits and fake dwell;
  (iii) an irreducible remainder (language-specific motor patterns, §1.7).
  Part 2's options are priced against components (i) and (ii).
* **The short-word probe inversion** (PHASE_O §2.1: synth holdout 62.31 ≤3 for
  the ru model vs 73.86 for English zero-shot, while REAL short words invert
  to 86.44 vs 85.22). Short synthetic words are short English traces with 1–3
  vertices re-anchored — the stratum where the donor draw is thinnest, where
  a single stretched segment dominates the whole trace, and where the length
  mix gives training almost no examples. All three defects concentrate there.
* **ru192 generator-artifact overfit** (PHASE_J §6.5: 2.4× capacity + 2× the
  schedule *lost* 2.68 real pts while gaining 3.1 greedy on the generator
  distribution). A 90 %-separable training distribution gives extra capacity
  something concrete and wrong to memorize: the stretch/kink signature. This
  is the quantitative reason "more capacity on v1 data" fails and any v2
  self-training loop must keep a real anchor (§2, option I).
* **The λ/probe inversions** (PHASE_O §2.1b/c): on the generator distribution
  emissions are artificially in-distribution and strong, so the holdout
  under-values the lexicon prior and over-credits capacity. A v2 that drives
  classifier separability toward 50 % attacks the root cause; the honest
  validation footing stays real-ru + English-matched regardless (§3 gates).

### 1.7 Design-assumption audit — what v1 structurally cannot represent

| # | assumption | what it misses | measurable? |
|---|---|---|---|
| A1 | residual bank is ENGLISH | language-specific motor patterns (Russian bigram habits, board density adaptation — real ru is *faster and more precise* than the English bank: §1.5, start_d/step stats) | ru only, via the residual gap after all fixable defects are fixed |
| A2 | donors drawn **i.i.d. per trace** | per-user style coherence — a real training corpus is ~10³ users × many traces with consistent style; v1 is ~10⁶ "users" with one trace each. Whatever per-style structure the encoder could exploit is absent; Yandex has no user ids so this cannot be measured on ru, but HWS user structure makes it buildable | structurally argued; buildable from HWS ids |
| A3 | donor match on vertex count only | segment-geometry compatibility → the §1.2 stretch defect; also letter/bigram-conditioned dynamics (common ru transitions get random English segment shapes; Fitts-type speed–distance coupling broken by the remap) | **measured** (§1.2) |
| A4 | time axis discarded; spacing = speed | duration/tempo distributions; the 60 Hz up/down-sampling asymmetry (§1.5); no dwell model beyond what donors happen to carry | **measured** (§1.5) |
| A5 | word draw = `255 − rank` | token-frequency length mix (§1.3); no per-session word co-occurrence | **measured** (§1.3) |
| A6 | endpoint stats as the only distribution gate | everything §1.4 shows endpoint stats are blind to | **measured** (§1.4) |

What v1 gets RIGHT and v2 must not lose: every output point is a re-anchored
**real human sample** — undershoot, corner-cutting, jitter and their
correlations are real, not modeled (the FUTO project's own verdict on min-jerk
synthesis was insufficient motor noise, and the one learned generator we
audited, WordGesture-GAN, failed our frame/endpoint gates badly:
DATASET_SCOUT §3 — start-hit 0.512 after a fitted affine, fixed 128 points,
2,100 ms median durations). Transplant beat both precisely because its noise
is human. Options are scored on preserving that property.

---

## Part 2 — Ranked design options for v2

Validation plan for EVERY option is the same triad, in this order of
authority: (1) the **ru real probe** (Yandex eval-only, in-dict t1 at the
CKDT λ 2.0 preset vs the 77.41 v1 baseline — `ru_synth_ch80`,
PHASE_I_DATA §9.2); (2) the **classifier score** (word-matched, word-disjoint
— §1.4 protocol; speed view is the binding one); (3) **downstream training**
(ch80/94k retrain per §3.4 cost). The synthesis holdout is used for nothing
except regression-testing the generator itself — Phase O proved it inverts
model comparisons.

| rank | option | targets (measured gap) | expected ru-probe gain | cost | human noise kept? | artifact risk |
|---|---|---|---|---|---|---|
| 1 | **A. corpus-frequency word draw** | §1.3 length mix (10.7× off) | **+1.5 … +3.5** | hours | yes (draw policy only) | low |
| 2 | **B. kinematic re-timing** | §1.1/1.2 speed profile (KS 0.60; classifier 0.90) | **+1 … +3** | 1–2 days | yes (donor tempo restored, not modeled) | low |
| 3 | **C. geometry-matched donor selection** | §1.2 stretch tail | +0.5 … +1.5 (partly overlaps B) | 1 day | yes | low (mild donor-diversity loss) |
| 4 | **D. session-coherent residual sampling** | A2 user coherence | 0 … +1 (unproven axis) | 1 day | yes | low |
| 5 | **E. start-side correction** | §1.1 start dwell/precision | 0 … +1 | 1–2 days | partial (edits residuals) | **medium** — only ru evidence exists; tuning to the sole validator is self-blinding |
| 6 | **F. bigram/segment-bank transplant** | A3 transit dynamics | uncertain | 3–5 days | partial (stitching breaks trace-level continuity) | medium |
| 7 | **G. learned corrector on transplant output** (hybrid) | residual realism broadly | 0 … +2 | ~1 week | partial | medium-high (Goodharts the classifier) |
| 8 | **H. conditional VAE/diffusion generator** | everything, in principle | unbounded, unproven | 2+ weeks | **no** (noise becomes model output) | high |
| 9 | **I. self-training loop** | — | blocked/limited (below) | high | — | high (ru192 precedent) |

### The options in detail

**A. Corpus-frequency word draw.** Replace `w = 255 − rank` with wordfreq
token frequency (Zipf-scale, available for every Phase-O language; for ru the
lexicon stays the app CKDT pack — only the *draw weight* changes, so the
no-corpus counterfactual is intact: token frequencies are lexicon-tier
knowledge, not swipe-corpus knowledge). Gate: generated length mix within
±5 pts of the wordfreq-implied token mass per length bucket. Expected gain
reasoning: training emphasis on a 36–39 % stratum goes from 3 % to ~35 %; the
synth model's real ≤3 stratum (86.44) has ~8 pts of headroom to the real-data
arm's; even half of it, on 36 % of rows, is ≈ +1.5 all-strata; the Phase-L
English precedent (synth_en_short targeting the same starvation) supports the
direction. Already pre-endorsed by PHASE_O §2.1's "recipe consequence".

**B. Kinematic re-timing.** Keep `warp_path` for geometry; restore the
donor's *time* parameterization afterwards. Formula: let `Q` be the warped
64-point path, `a_i` the donor's cumulative arc-length fraction at sample `i`
(computed on the DONOR path — this curve encodes its dwell/tempo pattern;
dwell = flat spans). Re-sample `Q`'s polyline at arc fractions `a_i` →
`Q′`. `Q′` has identical geometry (same polyline, same endpoints — Phase-H
invariants untouched) but per-step spacing that follows the donor's human
tempo instead of `mid_ratio`-scaled spacing. Kills the step_max/step_cv defect
by construction; also removes the fake-dwell excess. Classifier speed-view
accuracy is the gate (§3.3). Cost: ~30 lines + validation.

**C. Geometry-matched donor selection.** Index donors by (vertex count,
log-total-polyline-length bin); at draw time, reservoir-sample k = 16
candidates from the count-matched pool and pick the one minimizing
`Σ|log(L_d,seg/L_s,seg)|`. Bounds the §1.2 stretch tail without changing the
mechanism. Interacts with B: B fixes the *timing* of a stretch, C avoids
creating extreme stretches whose *residual magnitudes* are also
scale-inappropriate. Gate: per-segment ratio p95 < 2.0 (from 3.63).

**D. Session-coherent residual sampling.** Two-level draw: sample a donor
*user* (HWS has 1,338 user ids; FUTO has contributor/session structure via
`scan_futo_sessions.py`), then draw that user's traces for a block of ~50–200
consecutive synthetic rows, falling back to the global pool only when the user
lacks a vertex count/geometry match. Restores the user-consistency structure
real corpora have. Honest expectation: the CTC encoder sees single traces, so
the first-order gain may be small — the option is cheap, structurally right,
and matters more if any future consistency-exploiting training (curriculum,
per-user adaptation, discriminative reranking) lands.

**E. Start-side correction.** Real swipers plant the finger (median 3 leading
in-key samples vs synth 1). Two mechanisms, both parameterized and gated:
(i) prepend a start-dwell run drawn from the *donor corpus's own*
start-dwell distribution conditioned on trace length; (ii) shrink the
first-dwell-band residuals by a factor α ∈ [0.7, 1.0]. Danger, stated
plainly: the evidence that starts should be tighter is ru-only; fitting α on
the ru probe consumes the sole validator (the same trap as tuning λ on the
synthesis holdout, mirrored). Recommended footing: fit on English
(matched-real en→en transplant vs real en traces — measurable with the same
§1.4 protocol on FUTO/HWS), verify transfer on ru, never fit on ru.

**F. Bigram/segment-bank transplant.** Decompose donors into per-segment
transit units keyed by (length bin, turn-angle bin, entry/exit speed);
assemble target words segment-by-segment with continuity constraints; dwell
units inserted at vertices from a dwell bank. Directly attacks A3 (and would
give every rare bigram a *geometry-appropriate* transit). Costs trace-level
speed/style continuity — exactly the coherence D tries to add — and triples
the surface for stitching artifacts. Worth prototyping only if B+C leave the
classifier speed view above ~0.7.

**G. Learned corrector (hybrid transplant + learned).** Train a small
residual-correction network (input: transplanted trace + ideal polyline;
output: per-point delta, or a re-timing curve) with a moment-matching or
adversarial objective against REAL English traces — the pairing exists in
English only, which is license-clean (FUTO MIT + HWS MIT), then apply
cross-script. Keeps the human-noise base, learns only the correction.
Risk is Goodhart: optimizing against a discriminator turns the §1.4
classifier from instrument into training signal, and the ru probe must then
carry sole validation authority. Run only after A–D plateau, with the
classifier re-trained fresh (different seed/architecture) for evaluation than
the one used in training.

**H. Conditional VAE/diffusion over traces.** Full generative replacement,
conditioned on the ideal polyline; trainable on MIT English only, so
license-clean. Honest assessment against precedent: WordGesture-GAN — a
published, peer-reviewed attempt — measured locally at start-hit 0.512,
fixed 128 points, 2× human duration (DATASET_SCOUT §3); FUTO's own verdict on
min-jerk synthesis was insufficient motor noise; and our transplant already
beats both on every gate it fails. A learned generator would have to clear
the §1.4 classifier at ≤0.65 *and* the ru probe at ≥ v2-transplant to earn a
place, and nothing in the evidence says it starts anywhere near that. Park
unless the transplant paradigm plateaus measurably short.

**I. Self-training loops (generate → train → decode real → mine errors →
regenerate).** Mostly blocked, for reasons of record: for shipping models the
loop would route real Yandex signal into training (license-barred — the ru
model ships precisely because no Yandex row influences it); for the other
five scripts there is no real data to decode; and the ru192 lesson is that a
loop whose inner metric is generator-distributed amplifies artifacts
invisibly. The one viable variant: an **English** self-training loop (decode
real MIT English, mine error words, synthesize them — Phase L's
`english_synth --mode tail` is already half of this) used to improve the
*generator*, whose fixes then transfer cross-script by construction. Any such
loop must keep the ru probe entirely outside it as the untouched anchor.

---

## Part 3 — Recommended v2 architecture (spec, pre-registered expectations)

**⚠ AWAITING USER REQUIREMENTS — do not build from this section until the
user's rework request lands and is reconciled against it.**

Consistency note: PHASE_O §4.3 (committed concurrently by the Phase-O close)
independently ranks the corpus-frequency word draw as the #1 Phase-P action and
adds a step this spec adopts as G5's corollary: **re-run the ru probe
calibration after the fix** — if a length-correct, kinematically-repaired
generator restores rank-preservation against real data, the synthesis holdout
becomes a usable probe for the corpus-less scripts for the first time.

**Recommendation: v2 = transplant paradigm retained, options A + B + C + D
landed together as one generator revision (`script_synth.py v2` — same CLI,
same npz schema, same split/donor-side discipline), E/F/G as gated follow-ups
only if the ru probe says the headroom is still there.** Rationale: A–D fix
the three largest *measured* gaps, are individually cheap, all preserve v1's
load-bearing property (every point is a real human sample), and none
introduces a learned component that could Goodhart the validation.

### 3.1 Pipeline stages (per script)

```
S0 lexicon        registry load (unchanged) + wordfreq token frequency joined
                  per word; draw weight = token_freq (A), CKDT weight kept in
                  the npz for provenance
S1 donor index    (vertex_count, ⌊log1.25 L_polyline⌋) two-level index over the
                  donor side (train/holdout stride split unchanged);
                  per-user sub-index from HWS ids + FUTO session map (D)
S2 draw           word ~ token_freq; user ~ session block of 50–200 rows (D);
                  donor = argmin over k=16 reservoir candidates of
                  Σ_seg |log(L_d/L_s)|   (C)
S3 warp           layout_aug.warp_path — UNCHANGED (all Phase-H invariants)
S4 re-time        donor arc-progress curve a_i = cumarc_donor(i)/L_donor;
                  Q′ = resample(polyline(Q), a_i)   (B)
S5 clip + write   unchanged ([0,1] clip, npz schema, provenance json)
```

### 3.2 What changes in code

* `script_synth.py`: S1/S2/S4 (~150 lines); `--v1-compat` flag preserving the
  old draw for A/B ablations; provenance records generator version + option
  mask per npz.
* `cyrillic_synth.py`: untouched (historical record, per Phase-O convention).
* `synth_gap_audit.py`: becomes the standing gate harness (already committed).

### 3.3 Validation gates, in order, all pre-registered before any retrain

| gate | instrument | bar |
|---|---|---|
| G0 warp invariants | `layout_aug.py --selftest` semantics on v2 output path | identity exact; ideal→ideal < 1e-5 |
| G1 endpoint band | endpoint_stats vs ALT_LAYOUT §2 band | start/end-hit in band; wrong-geo control collapses (< 0.05) |
| G2 length mix | §1.3 table regenerated | each length bucket within ±5 pts of the wordfreq token mass; ru ≤3 in 30–40 % |
| G3 kinematic parity | §1.1 metrics, word-matched vs real ru | step_cv KS < 0.20 (from 0.60), step_max KS < 0.20 (from 0.52), sharp_turns KS < 0.25 (from 0.44) |
| G4 discriminability | §1.4 classifier, word-matched word-disjoint | **speed view ≤ 0.70** (from 0.904), coords ≤ 0.68 (from 0.772); en→en footing measured too |
| G5 downstream | ch80/94k retrain on v2 ru synth, real probe, λ 2.0 preset | in-dict t1 **≥ 78.9** (= +1.5 floor over 77.41); ≤3 stratum must not regress below 86.4 |

G1–G4 are generator-only (minutes, CPU). G5 is the binding gate and the only
GPU spend. Failure handling: if G5 fails while G2–G4 pass, the residual gap is
evidence for the A1 hypothesis (language-specific motor patterns) and prices
option E/F/G honestly; that outcome is informative, not wasted.

**Expected ru-probe band, stated before any build: +2 … +5 in-dict t1
(77.41 → 79.5–82.5), best estimate ≈ +3.** Reasoning chain: option A alone is
worth +1.5…+3.5 (§2.A); B+C's kinematic repair attacks the emissions collapse
(greedy 37 vs 75) whose beam-recoverable share is bounded by the λ lever's
+7.6 (PHASE_I_DATA §6) — call it +1…+3 after overlap with A; D ≈ 0 first
order. Anything above 83 would beat the estimate; below 79 falsifies the gap
decomposition of §1.6 and re-prices Part 2. The remaining ~7–10 pts to the
real-data arm's ≈ 90 is then the measured price of A1 — the part transplant
cannot fix without target-script motor data.

### 3.4 Cost at this box's measured throughput

| item | basis | cost |
|---|---|---|
| generator v2 implementation + unit gates | §3.2 scope | ~2–3 focused days |
| regeneration, 1 M rows + val/holdout, per script | v1 measured 1,141 rows/s single-core | **~16 min CPU** each |
| G1–G4 gate battery per script | this audit's runtime | ~10 min CPU + 2 min GPU (MLP) |
| G5 ru retrain (ch80, 94 k steps) | Phase-O uk log: 3,000 steps / 64–110 s | **~0.7–1.2 GPU-h** |
| ru A/B ablation (v1-compat vs v2, paired seed) | 2 × G5 | ~2 GPU-h |
| all six scripts regenerated + retrained (after ru gates pass) | 6 × (gen + train) | ~6 GPU-h + 2 h CPU |
| total for the full v2 rollout | | **≈ 10 GPU-h + 3 days engineering** |

### 3.5 Explicitly out of scope until the user's requirements arrive

Option E's α fitting footing, F/G/H go/no-go, any change to the split/donor-
side discipline, any new script, any training beyond G5's pre-registered
retrains, and any use of Yandex beyond the eval-only probe. The user's rework
request may also redefine the target property (e.g. personalization realism,
multi-finger, tempo conditioning) — the Part 1 instruments are built to
re-measure against whatever target is specified.

---

## Appendix — artifacts of this analysis

* `ctc/synth_gap_audit.py` — the three-stage measurement harness (committed).
* `~/ctc-train/synth_gap/{matched.npz, metrics.json, classifier.json}` —
  runtime outputs, not committed, regenerable in ~5 min from the checked-in
  script (`--stage data|metrics|classifier`, seed 1234).
* Baselines used, all pre-existing: `ru_synth_ch80` real-probe 77.41
  (PHASE_I_DATA §9.2), the Phase-O calibration grid (PHASE_O §2.1–2.4),
  PHASE_J §6.5 (ru192), PHASE_H §2 (warp validation), DATASET_SCOUT §3
  (WordGesture-GAN audit).
