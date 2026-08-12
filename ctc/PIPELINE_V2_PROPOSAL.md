# Pipeline v2 — final-architect retrospective and proposal

**Date:** 2026-08-12 · **Authority:** orchestrator directive of 2026-08-12
("review the entire campaign with fresh eyes; propose a from-scratch pipeline
only if genuinely sensible"). **Scope: design and proposal only — nothing in
this document was trained, no decode was run, test-2400 was not touched**
(ledger stays at 3 entries). Sources: `PHASE_A.md`–`PHASE_K.md`, `RESULTS.md`,
`MODEL_COMPARISON.md`, `FAIR_REMATCH.md`, `ALT_LAYOUT_EVAL.md`,
`DATASET_SCOUT.md`, `RESEARCH_SCAN.md`, `PHASE_I_DATA.md`,
`AUDIT_PREDECODE.md`, `AUDIT_FINAL.md`, `AUDIT_PHASEJ.md`, and the code
(`model.py`, `train.py`, `layout_aug.py`, `cyrillic_synth.py`,
`sweep_scoring.py`, `mine_candidates.py`, `ranker_features.py`,
`train_ranker.py`, `eval_beam.py`).

**Verdict up front:** a v2 pipeline IS sensible — not because a large accuracy
leap is available (the evidence says the QWERTY val axis is close to its data
ceiling) but because three structural moves the campaign discovered piecemeal
were never composed: (1) the alignment-compatibility mechanism can be
*trained in* instead of gated post-hoc, making the campaign's best
configuration (a mixable pair) a reproducible recipe rather than a lucky
draw; (2) the one scalable license-clean data lever the campaign built — the
residual-transplant synthesis engine that took Cyrillic from 0 to ≈77.4 — was
never pointed at English, where the ≤3 stratum and the lexicon tail are
measurably data-starved; (3) the short-word loss weighting works at W = 2 but
overshoots, and the registered W ∈ (1, 2) interpolation composes naturally
with pair training. `train_v2.py` (committed beside this document) implements
the pair trainer against the existing `train.py` infrastructure. Honest
headline assessment is in §2.6: a single model beating the full `mix2-i8f16`
card is judged **unlikely on the transfer axes**; the realistic and still
valuable bars are stated there, with a distillation contingency that is the
only credible route to the stretch goal.

---

# PART 1 — Retrospective: what the whole record says

## 1.1 The lever ledger, ranked by measured effect

Every number below is from the audited record; citations inline.

| rank | lever | measured worth | where |
|---|---|---|---|
| 1 | **decode-preset tuning (E1 discovery)** | +4.25 t1 (r2, holdout half), +2.29 (ch 192); symmetric: +1.94 to FUTO's ceiling, +7.38 to FUTO's floor | PHASE_E §1, FAIR_REMATCH §2–3 |
| 2 | **layout-resampling augmentation** | dvorak +13.2 t1 at zero en_qwerty cost | PHASE_H §6 |
| 3 | **synthesis (residual transplant, ru)** | 0 → ≈77.4 in-dict t1 with no real Cyrillic anywhere | PHASE_I_DATA §6, PHASE_J §6.9 |
| 4 | **data volume/mix** | T3+3×HWS +0.83 t1; sw234 +0.37 t1 ≙ 76 % more width; sw5q rescued 2 layout bars | PHASE_E §4, PHASE_J §6.1a |
| 5 | **capacity (dose-coupled)** | ch 80 → 256: +1.0 t1, only converts when the aug dose scales | PHASE_I §5 |
| 6 | **same-seed cross-model ensembling** | mix2: all 11 bars; layouts +1.3…+3.5 | PHASE_K §4.4, §8.2 |
| 7 | **≤3 loss weighting (slw2)** | ≤3 +0.12 seed-mean, every seed — the only training-objective lever that ever worked | PHASE_K §8.3 |
| 8 | **schedule 94 k → 188 k** | +0.5 t1 at ch ≤ 64; ~0 at ch 80+; 280 k a wash twice | PHASE_F §13, PHASE_J §6.1d |
| 9 | **architecture (resbn/BN-fold)** | free inference (−50 nodes), ~0 accuracy | PHASE_F §1 |
| — | everything else measured | null or negative | §1.8 below |

The rank ordering *is* the thesis: the two biggest levers are decode
calibration and data/augmentation construction; every attempt to be clever in
the objective or the architecture below rank 7 failed. Stated as the campaign
implicitly proved it:

> **The emission model is data-bound, the decode is preset-bound, and the
> architecture is a commodity.**

Supporting facts: `sw2345` is byte-identical in architecture to `resbn192i` —
the entire Phase-J gain is data (PHASE_J §10); E1 transferred unchanged across
**five** model families (PHASE_J §6.8b), so the preset is a property of the
emission/trie pair, not the model; ConvNeXt was refuted twice, separable
convs refuted, depth > 4 refuted, and the trunk that shipped is the first one
tried with the normalizer swapped (PHASE_B §3–5, PHASE_D §3, PHASE_F §3–4).
The decode-preset axis is now **saturated** — the wide-grid rule found the
interior optimum, the sweep is symmetric, and Phase J's minmargin sweep put
both engines back on their own E1 numbers within ±0.07. Nothing further is
buyable there except per-lexicon λ (a scale correction, done for app-en and
ru). What remains live is the data axis — which is what v2 must be about.

## 1.2 The dose-scaling law, re-read: a mixture-allocation problem wearing a scalar

The record: p 0.5 dominates at ch 80 (PHASE_H §5), breaks at ch 192
(dvorak 85.43 vs 88.85), p 0.65 repairs ch 192 on all eleven columns
(PHASE_I §5); at ch 256, p 0.65 vs p 0.80 trades dvorak (−1.0…−1.75) against
the four euro corpora (+0.3…+0.7), and **neither dose clears everything**
(PHASE_J §5.1b).

Fresh-eyes reading: the "dose" p is one scalar controlling exposure to a
*three-component mixture* with competing gradients — canonical QWERTY
(weight 1−p), synthetic random permutations (p·⅔, trains the
permuted-geometry axis = dvorak), real euro layouts (p·⅓, trains the
near-QWERTY-with-foreign-lexicon axis). Raising p to buy euro coverage
necessarily starves canonical AND raises the synthetic share; the ch-256
trade is exactly what a mis-parameterized mixture looks like, not a law of
nature. Two candidate dissolutions from the prompt, judged:

* **Per-batch multi-geometry** (each sample rendered on k geometries per
  batch) does not change the marginal training distribution — it reduces
  per-axis gradient variance and would smooth the trade, but the binding
  problem is allocation, not variance. Worth little alone.
* **Geometry-conditional normalization is a red herring.** The model already
  receives the key centers as input and scores keys through a learned
  geometry embedding (`model.py` key_embed); ALT_LAYOUT_EVAL §7 showed the
  failure was *exposure* to key re-arrangement, never a normalization
  deficiency (global affine distortion was survivable before Phase H;
  re-lettering was not).
* The **correct structural fix** is to promote the mixture to explicit
  weights `(w_canon, w_synth, w_real)` with the euro axis dosed
  independently of the permutation axis, plus the observation (recorded in
  PHASE_I §5 but never exploited) that part of the "euro" deficit is the
  CKDT-λ lexicon confound on the eval side, not emission quality.
* The strongest evidence the trade is *dissolvable* is mix2 itself: its
  transfer gains (+1.3…+3.5) dwarf its val gains because the two members sit
  at **different points of the data-mix trade** and averaging covers both
  (PHASE_K §4.4). A pair is a two-point mixture in model space. That is the
  bridge to §1.3.

## 1.3 Alignment is an unpinned gauge — and the campaign's KD refutation is partly a casualty of it

The Phase-K chain: same-recipe seed-ensembles are refuted in both averaging
modes (logprob catastrophic, prob −1.1…−1.4 t1) because seeds do not share a
CTC alignment; same-seed cross-model pairs can work; seed is neither
necessary nor sufficient (s4321 pair fails at 88.8 % frame agreement); the
real variable is **per-frame argmax agreement ≥ ~95 %**, a label-free gate
that predicted a fresh pair prospectively (PHASE_K §4.1–4.3, §8.5).

The clean theoretical statement: CTC's loss is a marginal over alignments and
never pins *which* alignment the model concentrates on — alignment phase is a
gauge freedom, fixed in practice by init + early optimization dynamics.
Emission-space averaging is only sound inside one gauge.

**The overlooked corollary — a genuinely new reading of an audited negative.**
Phase G's ensemble-teacher KD arm (teacher = per-frame *average of the three
ch-192 seeds' probabilities*) measured −0.45 t1 worse than single-teacher KD
(PHASE_G §3). Phase K then measured exactly what such a cross-seed average
looks like: blurred, alignment-incompatible emissions (greedy 72 → 37,
PHASE_K §4.2). The ensemble-teacher refutation is therefore **confounded by
alignment incompatibility** — the arm distilled from precisely the mush K1
later diagnosed. Likewise single-teacher KD's −0.5 at ch 80 (null at ch 64)
is a capacity-mediated result whose teacher's alignment gauge had no
relationship to the student's. What was refuted is "distill from an
arbitrary-gauge teacher into a capacity-sufficient student." What was never
tested is **alignment-matched distillation**: a student sharing the teacher's
gauge (co-trained, or gauge-pinned by construction). This matters for the
only credible single-model route to the mix2 numbers (§2.6).

**Can a from-scratch recipe pin the gauge deliberately? Yes, three ways, in
ascending order of intervention:**

1. **Gate-and-retry** (status quo): train pairs, gate at ≥95 %. Works 3-of-4
   times on the record; the failure mode (s4321) costs a full retrain.
2. **Co-training with a mutual per-frame KL on the SAME augmented view**
   (deep-mutual-learning shape). This directly optimizes the gate metric.
   Crucially it is *not* refuted by CR-CTC's failure: CR-CTC couples two
   *augmented views* through the model — at high capacity it fights the
   augmentation itself (its α·KL term punishes exactly the
   augmentation-induced variance the layout resampler needs the model to
   express), which is consistent with its sign flip between ch 80 and
   ch 192/256 (PHASE_J §6.4.1). A cross-model same-view KL cannot fight the
   data or the augmentation — both members see the identical view; it can
   only pull their gauges (and marginals, at high weight) together. Risk is
   the opposite one: too-high weight collapses the member diversity that
   makes averaging worth anything. Hence a ramped, moderate weight and
   member asymmetry by design (§2.3-E1).
3. **A geometric alignment prior**: letter mass constrained near the finger
   (`L_geo`, formula in §2.3-E6). Unlike speech, swipe input gives
   forced-alignment supervision for free — the path visits the word's keys
   in order, so "letter ℓ's emission frames sit where the path dwells near
   key ℓ" is a physically grounded prior that would pin the gauge globally
   (ALL models trained with it become mutually mixable, not just pairs).
   Counter-evidence to respect: the blank-penalty axis has a razor-sharp
   optimum at zero (PHASE_J §2), so the emission *structure* is delicately
   balanced — the prior must exempt blanks and enter at low weight,
   ablation-gated.

## 1.4 The ≤3 stone, re-read

The campaign's own diagnosis ("candidate generation, not re-ranking",
PHASE_J §9) was confirmed twice in Phase K — by the training-side lever
(slw2: ≤3 +0.12 seed-mean, every seed) and by configuration (mix2 +0.03).
Three sharpenings from fresh eyes:

* **It is not a T′-resolution problem.** The T′ = 32 pinch argument concerns
  *long* words (a letter nearly every frame); short traces have frames to
  spare. Measured: T′ = 64 moved ≤3 by −0.09 (Phase I probe) and +0.21
  (Phase K retrain) — both single-seed, both inside the measured
  run-to-run noise (sd 0.15–0.7 per metric, PHASE_I §6 preamble; 4+
  seed-sd 0.54, RESULTS Campaign-2 tables).
* **The T′ = 64 "flip-flop" needs no mechanistic story.** Both probes were
  single-seed re-runs under cudnn nondeterminism. The 4+ deltas (+0.33 at
  ch 80/p 0.5/T3+3×HWS vs −0.39 at ch 192/p 0.65/sw2345) straddle zero
  within the noise the campaign itself measured for exactly this stratum.
  What reproduced both times — and is therefore the real finding — is the
  **transfer gain** (+2.5–2.8 at ch 80; five of six layouts, german
  campaign-best 82.40 at ch 192) and the ~2× decode cost. T′ = 64 is a
  transfer lever with an unresolved val effect, correctly left as a
  contract-v2 option.
* **The durable ≤3 lesson is a two-part one**: (a) the stratum is
  *data-starved* — 34 % of val but a much smaller share of the FUTO train
  mix (PHASE_J §9), and no arm ever added short-word *data* (slw2 reweights
  the same rows; reweighting is emphasis, not information); (b) the
  weighting sits on a real frontier (slw2's designed bill: 4+ −0.13,
  spanish −0.66), and the registered W ∈ (1, 2) interpolation plus the
  mirror-image pair (`sw2345` 10/11 vs `slw2` 7/11-with-≤3) are the two
  never-run moves. A **length-conditioned beam** (wider/differently pruned
  for 2–3-key traces, where decode cost is trivial) also remains untried and
  is decode-side/app-side — registered again here.

## 1.5 Should the loss see the lexicon? No.

The greedy↔beam anti-correlation (B1 +0.38 greedy / −4.58 beam; B2 +5.16 /
−1.38; C3 best-greedy/worst-beam; PHASE_B §3, PHASE_C §5) is fully explained
as a *checkpoint-selection artifact* and was operationally solved by beam-t1
selection in Phase D. It is not evidence of exploitable sequence-level
training signal — RESEARCH_SCAN §1.1/§1.2's external evidence (MWER collapse
in 4/4 configs, N-best oracle gap ≈0.007 pp, 11 CTC-internal rescoring
strategies null) matches the campaign's internal measurement: the K3
rescorer, given 14 strong features including the beam's own exact Viterbi
path scores, bought +0.08 t1 — and bought the *incumbent the same* (+0.26),
i.e. the residual top-5 misrankings are dominated by information the trace
does not contain (dense short-word lexicon neighbourhoods, frequency ties),
not by ranking-model weakness. Lattice-constrained CTC and MWER-style
lexicon-aware losses are therefore **correctly excluded** from v2. The
lexicon's place is the decode (and λ per lexicon scale); the loss's job is
calibrated evidence.

Corollary worth writing down: **greedy accuracy is a diagnostic of emission
sharpness and alignment health, not a target.** The campaign's instinct here
was right, and v2 keeps beam-metric selection unchanged — with greedy
retained in logs precisely because it is the cheap alignment-collapse alarm
(mix greedy 9–20 % = incompatible pair; PHASE_K §4.3).

## 1.6 Where the next order of magnitude actually is

The real-data ceiling is reached: DATASET_SCOUT's inventory is exhaustive
("that is *all* of it") — ~1.30 M MIT-clean traces, all consumed; the last
+15 % of data (sw2345 pools) bought +0.21 seed-mean t1. The remaining routes,
priced by the record:

1. **Synthesis at scale, aimed at English.** The residual-transplant engine
   is the campaign's proven generator (ru: 0 → ≈77.4 with *zero* real target
   data; endpoint stats inside the real band; `cyrillic_synth.py` reuses
   `warp_path` verbatim). It was never pointed at the two places English is
   measurably starved: the ≤3 stratum and the lexicon tail
   (`MIN_WORD_FREQ = 3` discarded every word with < 3 traces — the leon-pool
   analysis showed ~7,400 such words exist in-corpus). Same-layout
   word-transplant synthesis (donor residual → different word's polyline on
   QWERTY and on the real euro layouts) is license-clean, costs ~15 min/M
   rows single-core, and adds *information* (new word shapes) where slw2
   could only add emphasis. Guards required — the ru capacity-overfit lesson
   (ch 192 on synth *lost* 2.7 pt in-dict while greedy rose; PHASE_J §6.5):
   cap the synthetic fraction, never select checkpoints on synthetic rows,
   ablate at one seed before scaling. Expected yield is sub-point but aimed
   at the exact standing stone (§2.3-E2).
2. **Collection.** The durable answer and out of this repo's control
   (swipe.futo.org accepts layouts on request — it shipped Shavian rows
   because someone asked). Any future app-side opt-in telemetry design
   should sample *short words and alt layouts* preferentially; recorded here
   so the next architect doesn't have to rediscover it.
3. **Personalization.** Scanned and correctly shelved (RESEARCH_SCAN Part 2:
   no exportable CTC loss, deprecated training runtimes; per-user offset
   worth ~nothing, per-user lexicon the real v1).

## 1.7 The audit of overlooked levers (things the campaign never touched)

Assessed against the record, with priors:

| candidate | prior | reasoning from evidence |
|---|---|---|
| co-trained pair / mutual KL | **high** | §1.3; the only mechanism that makes the campaign's best configuration a recipe |
| targeted en synthesis (short/tail) | **high** | §1.6; the one scalable data lever left |
| slw interpolation W ∈ (1, 2) | **high** | registered in PHASE_K §8.3, never run |
| alignment-matched distillation (pair → single) | **medium-high** | §1.3 corollary; the KD refutation does not cover it, but no direct evidence for it either |
| geometric alignment prior | **medium** | pins the gauge globally; risk documented (blank optimum sharpness) |
| length-conditioned beam | **medium** | decode-side, never tried, app decision; cheap |
| word-length auxiliary head | **low-medium** | could feed a length-conditioned decode; but K3's ranker already had length features and gained little |
| EMA at ch 192/188 k | **low** | null at ch 96/128 twice (PHASE_C §4, PHASE_D §4); soup (its cousin) sign-inconsistent at scale |
| label smoothing / entropy reg. on CTC | **low** | blank-penalty's sharp zero optimum says emission entropy is already where the beam wants it |
| warm restarts / SGDR, layer-wise lr | **low** | schedule axis closed (280 k wash twice); no optimization pathology observed anywhere except ch-256 underfit |
| frequency-domain / handcrafted features | **very low** | feature v2 was the campaign's worst arm (−4.60 t1); the trunk learns kinematics |
| curriculum by trace quality (distance col) | **very low** | exclusion-curation failed four independent times; ordering variants inherit the same mistaken premise (the "bad" rows carry the HWS-half signal) |
| license-clean contrastive negatives | **low** | K3 is the license-clean equivalent, measured symmetric and small |
| Mixup/manifold mixup, InterCTC, ASAM, Muon, label priors | **low** | RESEARCH_SCAN ranked, mechanisms subsumed or scale-inappropriate; CR-CTC's fate lowers the whole consistency-family prior at capacity |

## 1.8 What must not be retried (the refuted registry, consolidated)

KD from an arbitrary teacher at capacity-sufficient widths (−0.5 t1);
ensemble teachers built by cross-gauge averaging (−0.45); CR-CTC at capacity
(−1.6 euro at ch 256, retracted at ch 192); FUTO-parity augmentation bundle
(−0.46 t1, greedy −5.4); checkpoint soup (sign-inconsistent); EMA (twice
null); 280 k schedules (twice wash); exclusion curation (four negatives:
T2b, T4, englishLevel, motion gates a wash); key-proximity input features
(−4.60); ConvNeXt trunk (twice); depthwise-separable trunk (−0.61 at higher
latency); per-frame refinement head (negative at every modern standard);
whole-graph int8 activations (structural failure vs MASK_NEG); narrow preset
grids (the ~20×-understatement mistake — always widen to an interior
optimum); greedy-metric checkpoint selection; seed-ensemble emission
averaging; MWER-family fine-tuning (external + internal evidence); real
alt-layout training rows (costs the only never-seen eval corpora);
train-side-only HWS Y-frame correction (cannot answer its own question).

## 1.9 What the campaign got right (v2 keeps all of it)

Pre-registration before decodes; the seal discipline (3 reads in 6 phases,
each ordered and ledgered); seed-mean AND every-seed footings reported
together; the sign-consistency promotion rule; symmetric application of any
decode-side lever to the incumbent; the interior-optimum grid rule; beam-t1
checkpoint selection; label-free gates validated prospectively before use in
claims; adversarial audits that recompute from per-trace dumps; retractions
recorded in place. This methodological stack is worth more than any single
lever above rank 4 and is carried into §2.7 unchanged.

---

# PART 2 — The proposal: pipeline v2

## 2.0 One-paragraph summary

Train **coupled pairs**: two `resbn:192:1,2,4,8` encoders (the settled
architecture, T′ = 32 contract intact) on the sw2345 data mix **plus
targeted English synthesis pools**, on identical augmented batches, with
member-asymmetric short-word loss weights (1.0 / 1.5) and a ramped mutual
per-frame KL that trains in the ≥95 % frame-agreement the Phase-K gate
demands. Select the pair jointly on beam-t1 under the agreement gate; export
both members; ship as the mix2 contract (int8w + fp16w ≤ 5 MB, already
app-planned) — or, contingency, distill the pair's averaged emissions into
one gauge-matched single model. Augmentation keeps the Phase-H/I machinery
with the dose expressed as explicit three-way mixture weights. Everything
else — E1/app presets, beam, selection protocol, eval batteries, seal — is
unchanged.

## 2.1 Architecture

Unchanged: `resbn:192:1,2,4,8`, embed_hid 96, feat v1, T′ = 32, 1,512,802
params per member. Evidence: architecture search is exhausted (§1.1);
the contract is frozen app-side; capacity above ch 192 needs dose > 0.65 and
int8w and still trades euro-vs-dvorak (PHASE_I §5, §7.1). No new trunk work.

## 2.2 Data mix

| pool | rows | provenance |
|---|---|---|
| `train_t3` + 2×`train_t3hws` + `tier_sw234` + `tier_sw5q` | 1,285,381 | the audited sw2345 mix (PHASE_J §3.1) |
| **`synth_en_short`** (new) | ~150,000 | E2 below: residual-transplant, all 1–4-letter lexicon words |
| **`synth_en_tail`** (new) | ~150,000 | E2: lexicon words with < 3 real train traces |

Synthetic fraction ≈ 19 % of 1.59 M — under the 25 % cap motivated by the ru
overfit lesson. Checkpoint selection and every eval stay 100 % real.
Rebuild note: if the tiers are ever rebuilt, apply the dedup-key fix
(`normalize_word` in both hashes — the audited defect, AUDIT_PREDECODE §3)
rather than inheriting it a fourth time.

## 2.3 The elements, each with formula, expected gain, and evidence chain

### E1 — coupled-pair training (`train_v2.py`, implemented)

Two members A, B; same batches, same augmented view. Losses:

```
L_ctc^m  = Σ_i w_i·CTC_i / Σ_i w_i        CTC_i length-normalized per sample,
                                          w_i = slw_m if len_i ≤ 3 else 1
L_pair   = ½·[KL(sg(p_B) ‖ p_A) + KL(sg(p_A) ‖ p_B)] / (B·T′)     (nats/frame,
           all 65 columns; pad columns contribute 0)
L_total  = L_ctc^A + L_ctc^B + λ_pair(step)·L_pair
λ_pair(step) = pair_weight · clip((step − 5 000)/15 000, 0, 1)
```

The ramp exists because early CTC is blank-dominant and an immediate mutual
pull would lock a degenerate all-blank agreement. Init seeds are
deliberately *different* per member — a working pair then demonstrates the
coupling pins the gauge, not the init. Default `pair_weight 0.3` (the same
scale CR-CTC used; to be swept {0.1, 0.3, 1.0} in Stage 1). Selection: each
member on the standard 5 k-row beam-t1 (+ optional layout probes); the
**pair** jointly on mean member score gated by measured frame agreement
≥ 0.95 (the prospectively validated gate, PHASE_K §8.5). Cost ≈ 2× a single
run per step.

*Expected gain:* pair ≈ mix2-i8f16's numbers (val +0.3…+0.6 t1-class,
layouts +0.3…+3.3 over single-model bars) **reliably rather than 3-in-4** —
the claim is reproducibility, not new accuracy. Evidence chain: mix2 card
(PHASE_K §8.2) + the prospective gate confirmation (§8.5) + the fact that
fresh gated pairs landed 10–11/11 (§8.5). Secondary hypothesis, honestly
uncertain: mutual learning lifts each member ~+0.1–0.2 t1 over its solo twin
(deep-mutual-learning literature prior; zero in-campaign evidence — Stage 1
measures it; the design tension is coupling-strength vs member diversity,
which the record says is where mix2's transfer gains live).

### E2 — targeted English synthesis (delta on `cyrillic_synth.py`)

Mechanism already proven: donor English trace with matching
collapsed-polyline vertex count → virtual per-vertex alphabet →
`warp_path` residual transplant onto the target word's ideal polyline
(exactness invariants carry; `cyrillic_synth.py` docstring). The en
adaptation is an input change, not new math: `dst_centers = qwerty`
(and, for a euro-transfer variant, the four real euro layouts), word list =
(a) all 1–4-letter words of the AOSP/app lexicons, freq-weighted;
(b) lexicon words with < 3 real traces. Donor pool: the 1.0 M en traces
already indexed. Validation gate before any training: endpoint-proximity
within the real-en band (0.895/0.769 start/end-hit, PHASE_H §2.3), plus the
wrong-geometry falsification control.

*Expected gain:* ≤3 +0.2…+0.5 **without** slw2's spanish bill (it adds
information, not emphasis), small tail-word t1 tenths. Evidence chain: the
stratum responds to training-side signal (slw2, PHASE_K §8.3); new real data
converts at high efficiency (sw234 ≙ 76 % more width, PHASE_J §6.1a);
transplant realism is measured (ru endpoint stats; en→en is a strictly
easier transplant than en→ru). Risk: generator-artifact overfit lands
directly on val this time (same layout) — hence the cap, real-only
selection, and a single-seed on/off ablation gate.

### E3 — short-word weighting as member asymmetry

Member A slw 1.0, member B slw 1.5 (the registered W ∈ (1, 2) arm, carried
by one member instead of the whole model). The averaged pair merges the
mirror images (`sw2345` 10/11 vs `slw2` 7/11-with-≤3) instead of choosing
between them. *Expected:* the pair clears ≤3 with margin > mix2's +0.03,
without spanish going under. Evidence: slw2's every-seed ≤3 clear + mix2's
≤3 clear + the mirror-image structure (PHASE_K §8.3, §9). If the pair's
spanish dips (slw2's −0.66 diluted by averaging should halve, and E2 further
offsets), W_B drops to 1.25 — the interpolation the record already asked
for.

### E4 — augmentation as an explicit mixture

Same machinery (`LayoutAugmenter`, coupled affine sampler, slot permutation,
noise — all unchanged), dose expressed as
`(w_canon, w_synth, w_real) = (0.35, 0.40, 0.25)` — the p 0.65 equivalent
with the euro axis independently adjustable (p·synth_frac parameterization
maps 1:1; `--layout-alt-p 0.65 --layout-synth-frac 0.615` realizes this
split with zero code change). Stage-1 sweep moves only `w_real`
(0.20/0.25/0.30) at fixed `w_canon`. *Expected:* recover part of the ch-256
euro trade at ch 192 (azerty/qwertz/german +0.2…+0.8) without the dvorak
bill. Evidence: PHASE_J §5.1b's dose table shows the euro gains track the
real-layout exposure specifically; the confound caveat there is honored by
keeping dvorak the held-out axis.

### E5 — selection with layout probes (existing, opt-in → on)

`--select-layout-probes synth:101,synth:202,azerty` (weight 1.0): measured
−0.04 canonical for +0.5 mean probe (PHASE_I §6.2). With six layout bars in
the terminal condition, this is the right default for v2. Dvorak stays
refused as a probe.

### E6 — geometric alignment prior (optional, ablation-gated, default OFF)

```
L_geo = mean_t Σ_ℓ p_t(ℓ) · max(0, d(path_t, key_ℓ) − r),   r = 0.18, blank exempt
```

Pins letter mass near the finger — a physically grounded gauge fix unique to
swipe (the path visits the word's keys in order). If it works at small
weight (0.01–0.1), **every** v2 model becomes mutually mixable, not just
co-trained pairs, and the pair KL can be weakened. Risks honored: the blank
column is exempt (PHASE_J §2's sharp zero), and the term is Stage-1
ablation-gated with a kill criterion (any val bar −0.15 at one seed).
Implemented in `train_v2.py` behind `--geo-align-weight`.

### E7 — contingency: alignment-matched distillation (pair → single)

Only if the single-model stretch goal is pursued after Stage 2: distill the
**pair's averaged emissions** (a sharp, single-gauge teacher — the K1
mechanism guarantees it, unlike the Phase-G cross-gauge ensemble teacher)
into one student *initialized from member A* (gauge-matched by
construction), CTC + KD at swept weight (the never-swept knob,
PHASE_F §11.3). This is the only route the evidence admits to mix2-level
transfer in a single 3 MB model. Prior: medium; the Phase-G refutation does
not cover it (§1.3), but nothing supports it directly either. Single-seed
probe before any scale.

**Excluded, with reasons on file:** CR-CTC, soup, EMA, FUTO-parity augs,
MWER/lattice losses, curation, T′ = 64 (stays a contract-v2 app option — a
transfer lever with an unresolved val effect, §1.4), the K3 rescorer (kept
as an optional app-side +0.08/+0.11 t1/4+ add-on; it is symmetric and
composes with nothing here), real alt-layout training rows.

## 2.4 Training plan and cost

Throughput basis (measured on this box, RTX 5080 laptop): Phase-K ch-192
arms logged ~123 k steps in ~95 min ≈ 21.6 steps/s → a 188 k single run
≈ 2.4 h; a coupled pair ≈ 2× forward+backward ≈ **5 h/run**; synthesis
≈ 15 min/M rows (measured 1,141 rows/s); evals ≈ 0.5 h/battery (CPU,
parallel with GPU). `--workers 0` per the standing deadlock rule.

| stage | arms | GPU-h |
|---|---|---|
| S0 | build + validate `synth_en_short`/`tail` (endpoint gates) | ~0 (CPU) |
| S1 | 5 single-model ablations, 1 seed each: E2 on/off, W 1.25/1.5, E6 on/off, `w_real` sweep rider | ~12 |
| S2 | 2 pair arms (pair_weight 0.1/0.3; survivors of S1 folded in) | ~10 |
| S3 | winner pair × 3 seeds (fresh seeds; gate applied blind, pre-registered) | ~15 |
| S4 | full batteries, export, quantize, fixtures | ~3 + CPU |
| **total** | | **~40 GPU-h ≈ 2–3 days wall** |

## 2.5 Selection & shipping protocol

Per member: 5 k-row beam-t1 + layout probes (E5). Per pair: mean member
score, gated at frame agreement ≥ 0.95, computed label-free — and for S3 the
gate is applied **before any beam decode**, exactly the §8.5 blind protocol.
Ship packaging: int8w + fp16w pair (4.45 MB) or int8w + int8w (3.11 MB),
both audited size-compliant packagings of the mix2 contract
(PHASE_K §4.6) — the app integration plan already carries the dual-session
`CtcEmissionModel` seam and the averaged-emission golden-fixture format.

## 2.6 Success bars — stated before anything runs

All bars are the audited campaign numbers; both footings (seed-mean and
every-seed) reported for every claim; test-2400 untouched throughout — any
unsealing decision remains the orchestrator's, gated on the bars below
falling on val + layouts first.

1. **Pair bar (primary):** the S3 pair, gated blind, meets or beats every
   number on the `mix2-i8f16` card — val 88.68 / 92.61 / 93.46 / 91.30 /
   87.32, dvorak 91.94, dvorak-app 91.53, azerty 84.93, qwertz 82.81,
   german 81.22, spanish 89.59 — at ≤ 5 MB, on **2 of 3 seeds minimum,
   every bar**, with the ≤3 margin > +0.10 (mix2's +0.03 is two rows).
   *Judged reachable*: the fresh-pair experiment already landed 10–11/11
   without E2/E3, and E2+E3 aim at the one bar that varied.
2. **Single-model bar (secondary):** one member clears **all eleven
   campaign bars** seed-mean (i.e. `sw2345`'s tally plus the ≤3 bar —
   91.27 — that no single model has cleared while holding the other ten).
   *Judged plausible*: slw-1.5 + short-word synthesis attack ≤3's −0.07
   from two independent directions; the risk is E3's spanish bill, which
   averaging does not shield a *single* member from.
3. **Headline stretch (honest assessment):** a single model beating the
   full `mix2-i8f16` card — **not expected to fall in v2's base stages.**
   The gap is concentrated in transfer (mix2 dvorak +2.8 over the best
   single model), which the record attributes to two-point model averaging
   (§1.2); no single-model lever in the ledger moves dvorak +2–3 without a
   bill. The one admissible route is E7 distillation, run only if bars 1–2
   fall, with its own single-seed gate. If E7's probe fails, the correct
   conclusion — already defensible from this review — is that the pair IS
   the product: at 4.45 MB / 1.79 ms it is inside every budget, and
   "single model" is an aesthetic constraint, not a shipping one.

Failure handling: any element that misses its ablation gate is dropped, not
tuned-until-it-passes; a v2 that ends at "E1 reproduces mix2 reliably and
nothing else survived" is still a shipped result (reproducibility of the
best configuration) and will be written up as exactly that.

## 2.7 Protocol carried unchanged

Three seeds for any promoted claim; sign-consistency for promotion;
symmetric application of decode-side changes to the incumbent; interior-
optimum rule for any sweep; per-trace dumps kept for audit; pre-registration
committed before decodes; `--workers 0` for unattended runs; commit at every
milestone; test-2400 sealed — the gate for even proposing an unsealing is
bar 1 falling on every seed.

---

*Deliverables in this commit:* this document; `train_v2.py` (coupled-pair
trainer, E1+E3+E6, syntax- and import-verified against the existing
`train.py`/`model.py`/`layout_aug.py` infrastructure — **not executed**, per
the no-training scope). The E2 synthesis script is specified above as a
~60-line delta on `cyrillic_synth.py` (dst = qwerty/euro centers, en word
lists, en endpoint gates) and is deliberately left unwritten until its S0
validation gates are agreed, so that no synthetic English rows exist in the
tree before the guards do.
