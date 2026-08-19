# Synthetic swipe-generator v2 — research audit and amended spec

**Date:** 2026-08-19 · **Audits:** `SYNTH_V2_DESIGN.md` (fixes A–D, gates G0–G5)
· **Status:** amendments proposed, nothing built into the generator yet.

**Method note — this audit measured, it did not only read.** The four proposed
fixes were *implemented as prototypes and scored* on the same 9,416 word-matched
Yandex pairs the design's Part 1 used, before any literature verdict was
written. The harness is committed as `ctc/synth_retime_probe.py` (stages `law`,
`variants`, `classifier`, `gbm`, `coupling`, `enen`; outputs under
`~/ctc-train/synth_gap/`), and it
**asserts bit-identical reproduction of the v1 mechanism** (`max|Δ| = 0.00e+00`
against the committed `matched.npz`) so every delta below is attributable to the
fix and nothing else. No model training was run; the only GPU work is the gate
MLP the design already specifies (seconds). Literature and code were reviewed in
parallel — 42 repositories cloned to `~/ctc-train/research/` (Part 5).

**Headline.** The transplant paradigm survives the audit — a broad code review
of the learned-generator lineage found nothing that beats it, and produced new
evidence for *why*. Fix A is sound but its gate is self-contradictory. Fix B is
right in direction and **wrong in form**; the amended form is measured here.
Fix C is sound and cheap. Fix D is unfalsifiable at this stage and should not
gate anything. Two fixes the design does not contain are worth more than C and D
together. And the pre-registered gate **G4 (speed view ≤ 0.70) is not reachable
by A+B+C+D** — measured best 0.742 — for a reason this audit was able to
isolate and price: it is the English donor bank itself. A stronger classifier
than the committed one reads 0.854 on the same data, and tells us exactly what
is left: **cornering**, at three times the importance of any other feature.

---

## Part 1 — Verdicts on fixes A–D

Every number in this part is from `synth_retime_probe.py` at full n = 9,416
(`--rows 0`), scored against the same real Yandex rows, unless marked otherwise.

### 1.0 The two new controls that make the rest interpretable

The design had no floor for its classifier gate and no way to separate
*generator error* from *cross-script error*. Both are now measured.

**(i) Real-vs-real floor.** 923 of the 9,416 words have ≥2 real traces; pairing
one real trace against another real trace for the same word, through the
identical word-disjoint protocol, gives the empirical floor of the instrument:

| arm | speed view | coords | angles |
|---|---|---|---|
| **real vs real** (923 pairs, one per word) | 0.541 / 0.508 | 0.519 / 0.503 | 0.527 / 0.492 |
| **label-shuffled H0** (same rows) | 0.497 / 0.481 | 0.525 / 0.511 | 0.505 / 0.481 |
| real vs real, denser pairing (**3,194 pairs**) | 0.518 / 0.503 | 0.519 / 0.506 | 0.523 / 0.499 |

Cells are **max-over-epochs / final-epoch**. The denser arm pairs off *all*
traces of every word with ≥2 (⌊v/2⌋ pairs each) and is the better-powered
version; the 923-pair arm's max-statistic spread sits inside its own null, whose
95th percentile at that n is ≈ 0.545.

Two facts fall out. **The floor is 0.50** —
real-corpus heterogeneity does not put a floor above chance, so the whole
0.904 → 0.50 range is generator error and nothing else. And the committed
statistic (`max` over 40 epochs, `synth_gap_audit.train_mlp`) is a selection on
the test set: on the label-shuffled arm, where the truth is exactly 0.50, it
returns up to 0.525 (≈ +0.02 at this arm's 184 test pairs; smaller but non-zero
at the main gate's 1,546 test pairs). The final-epoch number on the same arm is 0.481–0.511.
Report the final-epoch accuracy against a permutation null (§3).

**(ii) English→English transplant control.** The design lists A1 (the residual
bank is English) as structurally unmeasurable. It is measurable: split HWS into
disjoint halves, treat one half as "real", transplant a *count-matched donor
from the other half* onto the **same word on the same QWERTY geometry**, and run
the identical battery. Every difference is generator error with **zero**
cross-script component.

| arm (n = 9,416 pairs each) | step_cv KS | step_max KS | sharp_turns KS | gate: speed | coords | angles |
|---|---|---|---|---|---|---|
| en→en, v1 mechanism | 0.535 | 0.511 | 0.279 | 0.824 | 0.682 | 0.676 |
| en→en, **amended B** (α=0.5, no C) | **0.010** | **0.076** | 0.257 | **0.604** | 0.669 | 0.651 |
| ru, v1 mechanism | 0.597 | 0.519 | 0.443 | 0.904 | 0.772 | 0.759 |
| ru, **amended B + C** | 0.125 | 0.096 | 0.380 | 0.748 | 0.685 | 0.728 |

This is the single most useful measurement in the audit. On matched population
the amended re-timing drives the speed-profile KS to **0.010** and the gate to
0.604 against a 0.50 floor — the mechanism is *essentially exact* when the donor
population matches the target. The Russian residual (0.748) is therefore
**not** generator error: ≈ 0.15 of the remaining speed-view separability is the
donor bank not matching the target population, i.e. hypothesis A1, now priced.
No amount of re-timing, donor matching or session coherence can remove it.

**Stated precisely, because the attribution matters.** The en→en arm differs
from the ru arm in *two* ways at once — script (Latin/QWERTY vs
Cyrillic/ЙЦУКЕН) and collection population (HWS-half vs HWS-half, against
HWS+FUTO vs Yandex). The control therefore bounds "cross-population +
cross-script" jointly and cannot split them. That is still decisive for the
engineering question, because both components are outside what the transplant
mechanism can reach: the only lever on either is target-script motor data.
It does mean the 0.15 should be quoted as *donor-population mismatch*, not
specifically as "Russians swipe differently".

### 1.1 Fix A — corpus-frequency word draw · verdict: **SOUND, gate is wrong**

Measured over the app's ru CKDT lexicon (49,704 projectable words), drawing by
`wordfreq.word_frequency(w, 'ru')` instead of `255 − rank`:

| draw | ≤3 | 4–6 | ≥7 | mean len |
|---|---|---|---|---|
| v1 (`255 − rank`) | 0.033 | 0.290 | 0.677 | 7.92 |
| **wordfreq token mass** | **0.268** | 0.376 | 0.356 | **5.74** |
| wordfreq^0.75 (damped) | 0.121 | 0.370 | 0.509 | 6.84 |
| real Yandex valid | 0.356 | 0.438 | 0.205 | 4.78 |

The fix closes **69 %** of the mean-length gap and raises the ≤3 stratum by 8×.
But gate **G2 as written is self-contradictory**: it demands "within ±5 pts of
the wordfreq-implied token mass per bucket" *and* "ru ≤3 in 30–40 %", and
wordfreq's own ≤3 mass is 26.8 % — outside that band. Both halves cannot hold.

The residual 26.8 → 35.6 pt gap is a **register** difference (wordfreq's ru
blend is written/subtitle text; swipe input is mobile chat), not a generator
defect, and closing it by tuning against the Yandex mix would consume the sole
validator. Amendment: keep the wordfreq draw, gate on the wordfreq mass ±3 pts,
and **record the real-usage mix as a known, unclosed 9-pt residual** rather than
pretending a band that the mechanism cannot hit. If register correction is
wanted later, the licence-clean route is a chat-register frequency list, chosen
without reference to Yandex.

### 1.2 Fix B — kinematic re-timing · verdict: **RIGHT IDEA, AMEND THE FORM**

As specified (§3.1 S4: resample the warped polyline at the donor's **global**
cumulative arc-length fractions) fix B works and clears its own gate G3:

| variant | step_cv | step_max | sharp_turns | turn_mean | straightness | path_len | gate: speed |
|---|---|---|---|---|---|---|---|
| v1 | 0.597 | 0.519 | 0.443 | 0.405 | 0.174 | 0.091 | 0.904 |
| **B_global** (as specified) | 0.122 | 0.095 | 0.404 | 0.293 | 0.076 | 0.052 | 0.775 |
| B_seg α=1 | 0.147 | 0.124 | 0.410 | 0.316 | 0.112 | 0.067 | 0.766 |
| **B_seg α=0.5** (amended) | 0.117 | 0.103 | 0.412 | 0.332 | 0.120 | 0.071 | **0.742** |
| C + B_seg α=0.5 | 0.125 | 0.096 | 0.380 | 0.308 | 0.087 | 0.059 | 0.748 |

But the specified form has a structural flaw the KS table hides. Copying the
donor's **global** arc-progress puts the donor's dwell spans at the donor's
*global* arc fractions, which need not be the target word's vertices. The
symptom is visible in a statistic the design's battery does not contain — the
count of local minima in the speed profile, i.e. how many *strokes* the trace
appears to contain:

| | speed-profile minima | per segment | speed autocorr lag-1 | KS(ac1) |
|---|---|---|---|---|
| **real** | 2.88 | **0.77** | **0.855** | — |
| v1 | 3.66 | 1.07 | 0.465 | **0.618** |
| B_global | 3.28 | 0.89 | 0.793 | 0.204 |
| **B_seg α=1** | 2.99 | **0.81** | 0.805 | 0.160 |
| B_seg α=0.5 | 3.15 | 0.85 | 0.820 | 0.115 |
| C + B_seg α=0.5 | 3.10 | 0.83 | **0.832** | **0.076** |

Note in passing that **lag-1 speed autocorrelation is the most discriminative
single statistic found anywhere in this campaign** — KS 0.618 on v1, higher than
step_cv's 0.597 — and the committed §1.1 battery does not measure it, because
all 17 of its metrics are marginal or aggregate and none looks at the *temporal
smoothness* of the speed sequence. That is exactly the structure an MLP over 63
step lengths exploits.

**Amendment B′ — vertex-aligned per-segment re-timing.** Reuse the monotone-DP
correspondence `warp_path` already computes; for each polyline segment *k* copy
the donor's **within-segment** arc-progress (so its dwells and corner
decelerations land on the corresponding *target* vertices), and reallocate the
sample budget *across* segments by `m_k ∝ n_k · ρ_k^α`, `ρ_k = ΔC_k/ΔA_k` the
target/donor traversed-arc ratio.

α is not a free parameter — it is a measured human invariant. Regressing
per-segment time share on per-segment ideal-length share *within* traces
(`--stage law`, `T_k ∝ L_k^α`):

| corpus | α | R² | footing |
|---|---|---|---|
| HWS (MIT) | **0.460** | 0.069 | fitting |
| FUTO (MIT) | **0.493** | 0.104 | fitting |
| real Russian | 0.447 | 0.101 | **transfer check only** |
| v1 synthetic | 0.271 | 0.023 | — |

Three real corpora, two scripts, two collection methods: **α ≈ 0.46–0.49**, and
Russian lands inside the English range. This is an isochrony-type invariant
(sublinear duration in movement amplitude, speed ∝ L^0.54) and it **transfers
across scripts**, so it can be fit on MIT English and applied to every script
without touching a validator. v1 sits at 0.271 — wrong, and with an R² four
times worse, i.e. v1's time allocation is not merely mis-sloped but noisy.

Use α = 0.5 (the fitted value rounded; the difference between 0.46 and 0.50 is
below the measurement's resolution). The measured penalty for getting it wrong
is real but modest — α = 1 costs +0.024 on the gate.

**Fix B is not a free win, and the design does not know it yet.** Re-timing
repairs the speed marginals and *opens a speed–curvature coupling defect of
comparable size* — slope KS 0.156 → 0.566 in the design's global form, 0.366 in
the amended one (§2.3). This does not change the recommendation to land B (the
gate, the KS battery and the downstream mechanism all improve), but it does mean
**B must not be validated on speed metrics alone**, and it is the reason §2.3
promotes a metric an earlier screen had proposed dropping.

**One honest cost of any re-timing.** v1's load-bearing property is that every
output point is a re-anchored *real human sample*. After re-timing, every output
point lies on the piecewise-linear interpolant *through* real human samples.
The residual field, its correlations and its magnitudes are untouched; the
sampling positions along it move. Given that real traces are themselves
interpolated onto a 60 Hz grid and then onto 64 points before the model ever
sees them (`futo_decoder_eval.featurize`), this is the same class of operation
the real pipeline already performs, and the en→en control (step_cv KS 0.010)
shows it does not introduce a detectable signature of its own.

### 1.3 Fix C — geometry-matched donor selection · verdict: **SOUND, keep, demote**

k = 16 reservoir candidates scored by `Σ_seg |log(L_dst/L_src)|` does what it
claims:

| per-segment dst/src length ratio | p5 | p25 | p50 | p75 | **p95** |
|---|---|---|---|---|---|
| v1 (count-matched only) | 0.16 | 0.50 | 0.83 | 1.27 | **3.63** |
| fix C (k = 16) | 0.29 | 0.72 | 0.91 | 1.07 | **1.72** |

Gate "per-segment ratio p95 < 2.0" is met with margin. But C **alone** barely
moves the instrument that matters (gate speed 0.904 → 0.879), because it
addresses the *magnitude* of a mis-timing that B removes entirely. Its measured
value is in the second-order metrics once B is in place: composed with B it
improves coords (0.729 → 0.685), sharp_turns (0.412 → 0.380), straightness
(0.120 → 0.087) and path_len (0.071 → 0.059), and it costs one day. Keep it —
but the design's ranking (C above D, both below B) understates how much of C's
headline value is really B's.

**One independent result argues C is worth more than our ru numbers show.** In a
separate English-only experiment (HWS, real geometries, only the time
parameterization varied), donor-arc-progress re-timing with a *random* donor
left a step-length classifier at 0.657, while the same re-timing with donors
**matched on (segment count, nearest total path length)** fell to **0.494 —
chance**. Our own en→en arm, which has B but *not* C, reads 0.604; the gap
between 0.604 and 0.494 is plausibly C's. On the ru footing that headroom is
masked by the donor-population term. Cheap, so build it.

### 1.4 Fix D — session-coherent residual sampling · verdict: **UNFALSIFIABLE NOW, DO NOT GATE ON IT**

Nothing in this audit can measure D: the Yandex validator has no user ids, and
the CTC encoder consumes single traces, so there is no first-order channel
through which per-user coherence can reach the loss. The design says this
already ("honest expectation: … the first-order gain may be small"). The
literature review turned the guess into a prediction, and it is negative for our
deployment case:

* **No published experiment tests user-coherent vs i.i.d. donor sampling for a
  gesture-typing decoder**, and no published swipe decoder conditions on a style
  embedding. Gboard's production personalization (Sivek & Riley, Proc. ACM HCI
  2022, 10.1145/3546737 — −0.5…−1.9 % words-modified-ratio in 10 of 11
  languages) is **tap-only** and names gesture personalization as open.
* The two nearest experiments that *isolate* the variable both split the same
  way: Li et al. (arXiv:2603.16883, IMU handwriting) get **−34.5 % relative CER
  writer-dependent and ≈ 0 writer-independent** from writer-consistent
  augmentation; Kohút et al. (ICDAR 2023, arXiv:2302.06318) get −9.22 % CER
  writer-dependent from a style-AdaIN block that "is not a suitable choice for
  writer-independent scenarios". **We ship to unseen users.**
* The style structure is nonetheless real — HWS shows ICC 0.29–0.60 on
  tempo/dwell/speed-CV axes — it is just unmonetizable without a runtime
  adaptation path.

Decision rule: **build D only because it is free** (a draw-policy change, ~1
day), record donor id in the npz provenance so a later personalization path can
use it, and **exclude D from the v2 acceptance criteria** — it can neither pass
nor fail them. One amendment worth taking: if D is built, prefer
**posture-coherent** to identity-coherent blocking. Touch offsets *sign-flip*
between thumb and index postures (Azenkot & Zhai, MobileHCI '12; Yin et al. CHI
'13 measured that pooling opposite-signed offsets made a key-adaptive spatial
model *worse* than no offset model at all, 8.02 % vs 7.85 % CER), HWS records
`swipeFinger`/`swipeHand` per donor, and posture — unlike identity — transfers
to unseen users.

---

## Part 2 — Enhancements the design does not contain, ranked

| # | enhancement | measured / expected value | cost | when |
|---|---|---|---|---|
| 1 | **Acquisition-bandwidth matching** (§2.1) | halves the residual cornering gap: sharp_turns KS 0.380 → 0.293, turn_mean 0.308 → 0.251, angles gate 0.728 → 0.675 (oracle bound) | donor-bank rebuild + ~40 lines | **v2 now** |
| 2 | **Vertex-aligned re-timing** instead of global (§1.2) | gate speed 0.775 → 0.742; stroke count 0.89 → 0.83 per segment; ac1 KS 0.204 → 0.076 | it *replaces* the design's S4, so ~0 extra | **v2 now** |
| 3 | **Speed-sequence + smoothness gate metrics** (§2.2, §3.2) | would have caught v1's defect without the classifier; ac1 KS 0.618 is the largest single-statistic gap in the campaign | ~1 hour | **v2 now** |
| 4 | **Real-vs-real floor + H0 arms on every gate run** (§3.1) | turns an uncalibrated threshold into a measurable gap-closure fraction | ~1 hour | **v2 now** |
| 5 | **PRDC with control + per-word authenticity** (§3.3) | independent fidelity/diversity axes; already showed transplant has **no** diversity deficit | ~half a day, MIT code exists | v2 if cheap, else v3 |
| 6 | Turn-angle covariate in the α regression (CLC's missing term) | unknown; tests whether cornering time needs its own law | ~2 hours | v3 |
| 7 | ΣΛ as a *diagnostic* vocabulary (§2.4) | only if the stroke-count proxy proves too crude | days; **no licence-clean implementation exists** | v3 at earliest |
| 3= | **Speed–curvature slope + R² in the gate** (§2.3) | non-discriminative on v1 (KS 0.156) but **KS 0.566 on re-timed output** — the defect fix B introduces and every other metric misses | ~1 hour | **v2 now, mandatory** |
| 8 | Two-thirds-law *re-timing*, min-jerk re-timing, SPARC, DTW, TimeGAN scores | **measured negative or non-discriminative**: power-law re-timing gives CV 1.21 (real 0.90), min-jerk 0.66; SPARC/DTW/TimeGAN consume gate budget and pass | — | **never** |

### 2.1 **Acquisition-bandwidth matching** — belongs in v2 · the largest unaddressed gap

`featurize` resamples a raw trace to 60 Hz — `n60 = max(2, round(dur/16.667)+1)`
nodes — *before* the 64-point index-uniform resample. A 701 ms real Russian
trace is an **upsampled** 43-node polyline (locally piecewise linear, smooth at
the step scale); a 1,113 ms English donor carries ≥67 nodes and is
**downsampled**, keeping per-step jitter. The design identifies this (§1.5) and
then proposes no fix for it. It is the reason the turn family barely responds to
A–D.

Oracle arm (each synthetic trace pushed back through a 60 Hz grid at its real
partner's duration — not shippable, it reads the validator; it *bounds* the
achievable gain):

| KS vs real | sharp_turns | turn_mean | turn_total | step_max | gate: angles |
|---|---|---|---|---|---|
| v1 | 0.443 | 0.405 | 0.400 | 0.519 | 0.759 |
| C + B′ (α=0.5), no bandwidth match | 0.380 | 0.308 | 0.306 | 0.096 | 0.728 |
| **+ oracle bandwidth match** | **0.293** | **0.251** | **0.252** | **0.067** | **0.675** |
| (same pair with the design's global B) | 0.376 → 0.288 | 0.270 → 0.207 | 0.267 → 0.207 | 0.091 → 0.059 | 0.722 → 0.660 |

So roughly **half of the residual cornering gap is an acquisition artefact, not
motor behaviour** — and G3's `sharp_turns < 0.25` bar is unreachable without
addressing it (and marginal even with the oracle). Shippable form: keep raw
timestamps in the donor bank (the caches currently store only `[2,64]`
features), draw a target duration from a model `T = f(path_len, S)` fit on MIT
English, and re-featurize the synthetic trace through the real 60 Hz chain.
Cost: donor-bank rebuild plus ~40 lines. **Do not** fit the duration model on
Yandex durations — that is the same validator burn the design correctly forbids
for option E's α.

### 2.2 **Speed-sequence and smoothness metrics in the gate** — belongs in v2 · near-free

See §3. `ac1` (KS 0.618 on v1) and the stroke-count proxy are the two statistics
that would have caught v1's defect ahead of the classifier, and both are ~10
lines. An independent screen of the wider metric literature converged on the
same conclusion from the other direction: log-dimensionless-jerk (LDLJ, KS
0.518) and the speed spectral centroid (KS 0.566) are strong, while SPARC
(0.195) and the two-thirds power-law exponent (0.025) are not.

### 2.3 The two-thirds power law — **useless against v1, essential against v2**

This is the audit's most surprising result and it reverses a conclusion twice.

**As a generator constraint: reject.** Re-timing a path to satisfy
v ∝ κ^(−1/3) directly was measured on real geometries and overshoots badly —
step-length CV 1.21 and peak/mean 5.92 against real 0.90 / 3.61, i.e. it
reproduces the *same* superhuman-speed defect v1 already has. Minimum-jerk
re-timing (Todorov & Jordan's along-a-fixed-path formulation, the principled
alternative) fails the other way: CV 0.66, peak 1.94, and a SPARC standard
deviation six times too small — smoothness-optimal profiles are far too uniform
to pass as human. Neither belongs in the generator.

**As a validation metric: the slope alone is worthless, and the pair
(slope, R²) is indispensable.** On v1 the coupling looks almost fine — measured
across corpora, real β ≈ −0.373 (FUTO) / −0.390 (HWS) against v1's −0.377, and
on our own matched ru set real −0.199 vs v1 −0.148 (KS 0.156). The transplant
inherits the invariant for free, because it moves real residuals onto a
polyline. That is why an early screen scored it "non-discriminative" and
proposed dropping it.

**Then re-timing breaks it.** Measured on the matched ru set:

| | speed–curvature slope | KS | fit R² | KS |
|---|---|---|---|---|
| real | −0.199 | — | 0.382 | — |
| v1 | −0.148 | 0.156 | 0.233 | 0.274 |
| **B_global** (as designed) | **+0.054** | **0.566** | 0.140 | 0.444 |
| B_seg α=0.5 | −0.024 | 0.436 | 0.149 | 0.419 |
| **C + B_seg α=0.5** | **−0.067** | **0.366** | 0.158 | 0.397 |
| C + B_seg + bandwidth | −0.096 | 0.306 | 0.188 | 0.351 |

The design's global form drives the slope through zero and *inverts* it: the
donor's velocity minima land at arbitrary points of the target geometry instead
of on its corners, so speed stops covarying with curvature at all. **Fix B
trades a speed-marginal defect (step_cv KS 0.597) for a speed–curvature coupling
defect (slope KS 0.566).** The vertex-aligned amendment recovers about a third
of that (0.566 → 0.366) precisely because it puts the donor's decelerations back
on the target's vertices, and the bandwidth stage another chunk (→ 0.306) — but
nothing tested closes it.

Two consequences, both load-bearing:

1. **The gate must include speed–curvature slope and R²**, or fix B will
   Goodhart it — a battery of speed marginals plus a speed-view classifier
   declares victory on exactly the axis B repairs while a new defect of similar
   size opens on the axis it does not measure. This is the same failure as
   v1's endpoint-only gating, one level up.
2. **The obvious repair does not work.** Warping the donor's time density by
   (κ+α)^β to force the coupling was tested: it restores slope and R² but pushes
   step-length CV back to 0.95–1.24 and the step-length classifier back to
   0.928. Rank-matching the donor's step-length marginal onto the target's
   curvature order is worse still (slope collapses to −0.04…−0.24, joint
   classifier 0.94). Both are negative results worth recording so they are not
   re-tried.

Estimator, for reproducibility: Menger curvature on the 64 samples
(`κ = 4·Area/(|ab||bc||ca|)`), `v_i = ½(d_i + d_{i+1})`, mask κ > 1e-3, OLS of
log v on log κ per trace. Savitzky–Golay differentiation (window 9, order 3)
gives the same conclusion. Note the standing estimator critique (Schaal &
Sternad 2001: a 4th-order Butterworth manufactures β = 0.327 from a β = 0
trajectory; Marken & Shaffer 2017 on the `V ≡ D^⅓R^⅓` identity) — which is
exactly why the metric is used *comparatively*, identically computed on both
populations, and never as an absolute standard.

### 2.4 Sigma-lognormal (Plamondon) velocity modelling — **v3 at the earliest, probably never**

The idea priced against fix B: fit ΣΛ parameters to real donors, learn the
parameter distributions, sample velocity profiles per synthetic trace. Against
the measured alternative it loses on every axis. Fix B costs ~30 lines, is
parameter-free (α is measured, not fit), and reaches step_cv KS **0.010** on the
matched-population control — a ΣΛ resampler cannot beat 0.010, and it would
replace *human* timing with *modelled* timing, forfeiting v1's load-bearing
property for no measured gain. There is a further structural objection: a ΣΛ
stroke is a circular arc with a lognormal speed profile, so the model ties
velocity to a specific geometry; borrowing only its velocity term and applying
it to a different path is not a use the model licenses.

Three further facts settle it.

* **Licensing is fatal.** Every ΣΛ implementation located is unusable:
  `research/sigma-lognormal` — the only readable *fitter*, a full Robust-XZERO
  at 1,080 LOC — has **no licence file at all**; `research/iDeLog` (Ferrer et
  al., TPAMI 2020) ships as **36 obfuscated MATLAB `.p` files inside a .rar**
  and also carries no licence; `research/calligraph` (a differentiable PyTorch
  ΣΛ synthesizer) is **GPL-3.0**; Plamondon's own g3 is a web service its
  authors state "cannot be made publicly available as a standalone software".
  Nothing on PyPI. **No MIT/Apache/BSD ΣΛ code exists.**
* **Fitting is affordable but not free:** measured 1.5–2.6 s/trace at beam
  width 3 (~14 lognormals to reach 25 dB SNR) — ≈ 55 CPU-hours for 10⁵ donors,
  about 3.5 h on 16 cores.
* **And state-of-the-art ΣΛ synthesis still loses to our exact discriminator.**
  Leiva, Diaz, Ferrer & Plamondon (ICPR 2020, arXiv:2010.13231) train a GRU on
  *velocity sequences* to separate human gestures from expertly fitted
  ΣΛ-synthetic ones and report **95.4 % / 87.0 % / 97.0 % / 93.4 %** across four
  corpora — "it is not what you write, but how you write it". That is
  ScriptStudio-grade extraction with literature-calibrated perturbations,
  failing the same test we are trying to pass.

Keep ΣΛ on the shelf as a *diagnostic* vocabulary (component count,
reconstruction SNR, the μ/σ distributions) if the stroke-count proxy in §3 ever
proves too crude. The MIT-licensed `research/handwriting-biometrics` is a
ready-made human-vs-ΣΛ discriminator harness if that check is ever wanted.

### 2.5 Fitts / CLC-calibrated transit durations — **subsumed by the measured α**

The CLC model (Cao & Zhai, CHI '07) predicts stroke-gesture production time from
geometry: `T_line = 68.8·L^0.469 ms` (R² 0.998; 0.394 on their practiced subset)
plus a curve term, and polylines at R² 0.960. Its line exponent **0.394–0.469 is
our α**, arrived at independently on a different apparatus. The scale constants
are stylus-on-tablet mm/ms and cancel exactly under our Σt = 1 normalization, so
only the exponent survives — and we already measured it, in our own units, on
our own corpora, with cross-script transfer demonstrated. Adopting the
parameterised model adds constants without adding information.

Two details from the primary source are worth keeping, both of which *close*
rather than open work:

* **Do not add a per-corner time constant.** Cao & Zhai measured one, found
  |T_corner| < 40 ms at every angle with an inconsistent sign, and **deleted it
  from the model**. The sublinear exponent already prices corners implicitly:
  splitting a segment in two costs `2^0.531 = +44.5 %` for free.
* **State which length the exponent is on.** Regressed on *ideal key-centre*
  distance the exponent is 0.39–0.49 (our 0.460/0.493/0.447 and CLC's
  0.394–0.469); regressed on *actually traced path* length the same data gives
  0.65–0.75. Same relationship, different denominator. Our S4 uses the ideal
  polyline, hence α ≈ 0.46–0.50.

Fitts proper does **not** apply within a swipe: every key-to-key index of
difficulty on a soft keyboard is under 3 bits, which is the ballistic
`MT ∝ √A` regime, and there is no acceptance width to substitute. Summed-Fitts
is presented in the literature as the *novice visual-tracing* model of gesture
typing, not the fluent one. Also worth recording: within-trace R² for the
length→time relation is only ≈ 0.58, so **42 % of within-trace timing variance
is not explained by geometry at all** — that residual is the donor's
idiosyncratic dwell and hesitation, which is exactly what fix B preserves and
what no geometric formula can synthesize. It is the quantitative reason to keep
a transplant at all.

### 2.6 Learned generators (options G/H) — **stay parked, and now for a better reason**

The code/paper review turned up new evidence that inverts the usual worry.
WordGesture-GAN (CHI '23) reports precision 0.973 / recall 0.258 — a 4× within-
word diversity collapse — and its released endpoint emits fixed 128-point traces
with 6.1 % negative Δt, i.e. non-monotone time. Our transplant fails in the
*opposite* direction: measured with PRDC against a real-vs-real control,
real-vs-synth recall is **0.916** against a control of 0.919 (density 0.666 vs
1.006, coverage 0.807 vs 0.964). **Transplant has a fidelity deficit and no
diversity deficit whatsoever.** Adopting a learned generator would trade away
the one axis we already win.

FUTO's appendix reports the same lesson from the other side, and it is worth
quoting exactly because the design doc paraphrases it loosely (arXiv:2606.25247
Appendix A, verified against the fetched paper text). Adding ~170 k
**IndicSwipe** synthetic swipes to an English-only baseline moved EN val
93.25 → 93.39 and EN test 93.13 → 93.22, while **RU val fell 77.15 → 76.68** and
CF val 96.45 → 96.32. Their diagnosis: the trajectories "are generated by a
parametric model of the target word's key sequence rather than recorded from
users, so they lack the motor noise, hesitation, and curvature mismatch that
real swipes exhibit", with "direct evidence" that the same encoder reaches
**96.5 % greedy-CTC top-1 on the held-out synthetic Tamil split, i.e. the
synthetic distribution is trivial to fit, even without a lexicon**. The
generator they indict is parametric, not min-jerk-specific and not learned — the
failure mode is *any* generator whose output is too easy, which is exactly what
the §3.3 diagnostic measures and exactly what PHASE_J §6.5's ru192 overfit was.

**The counter-evidence, stated because it is the strongest case against this
section.** Apple's swipe-synthesis work (Mehra et al., ICASSP 2020,
10.1109/ICASSP40776.2020.9053689) trained on 2.2 M real paths from 665 users and
reports: real-only top-1 62.2 %, **+ cubic-spline synthetic 59.5 % (it hurt)**,
**+ residual-realistic GAN synthetic 65.8 %, and 66.8 % at 10× volume**. So a
learned generator *can* beat parametric synthesis by ~6 points — given 2.2 M
real traces in the target script. That is precisely the resource the non-Latin
scripts do not have, which is why the transplant exists. The result is best read
as a third independent confirmation of the same axis (with FUTO Appendix A and
WordGesture-GAN): **parametric synthesis ranges from useless to harmful, and
residual-realistic synthesis helps.** Our transplant is on the correct side of
that line by construction, and it gets there without needing target-script data.

If a learned component ever lands, conditional
**flow matching** over the 64×2 residual field is the shape to try — not an
MDN-over-offsets (Graves) or a full DDPM.

---

## Part 3 — Validation-metric upgrades

### 3.1 The classifier gate is measuring the right thing the wrong way

Seven defects, each measured rather than asserted. (Footing note: the committed
harness's "15,740 train / 3,092 test" are *rows*; the split is 7,870 / **1,546
pairs**.)

| # | defect | evidence | fix |
|---|---|---|---|
| 1 | reports `max` test accuracy over 40 epochs — selection on the test set | under a true H0 the statistic centres at **0.512, not 0.500** (+1.2 pts), 95th pct 0.522; measured three ways that agree (toy simulation at matched n, 20 H0 trials of the exact architecture, and the real-data control arm rescaled) | select the epoch on a **validation split carved from training words**, report test accuracy there; or report the final epoch (they agree within 0.003) |
| 2 | no floor arm | the floor is **0.50** — measured, not assumed | **real-vs-real control mandatory every run**; if it lands outside [0.48, 0.52] the run is void |
| 3 | textbook null is the wrong null for matched pairs | pairs are dependent; per-pair variance under H0 is 0.077 vs the 0.125 an independent-rows model implies. `N(½, 1/(4n))` is valid but **13–22 % too wide**, i.e. too easy to pass | exact **within-pair permutation**: draw `s_i ~ Bern(½)` per word-pair and swap that pair's two labels, rerunning the whole procedure. 100 permutations = 0.9 min CPU, so there is no efficiency excuse |
| 4 | single arbitrary word split | across split seeds the accuracy sd is **0.016–0.029**, three times the binomial 0.009 — a single-seed CI is 3× too narrow | K-fold word-disjoint CV (every pair tested once), report mean and a one-sided 95 % UCL |
| 5 | 80/20 split | balanced splits are power-optimal (Kim et al. Thm 6.1), and 50/50 is the default in Lopez-Paz & Oquab's own released code | 50/50, or K-fold which dominates both |
| 6 | **the classifier is too weak — this inverts the gate's logic** | HistGBM beats the MLP on the same splits (speed+angles **0.931** vs 0.874; omnibus 0.911); kNN-5 reads only 0.605 on coords where GBM reads 0.777 — a kNN gate would have *passed* coords at τ=0.70 | see the principle below: max over a **pre-registered** classifier × view family |
| 7 | **the `acc_unmatched_coords` arm leaks** | `synth_gap_audit.py:396-399` splits by random *row*, not by word, and the two classes have different word distributions. Labelling real traces by an arbitrary word-half and splitting by row already reaches **0.732 on coords** from word memorisation alone | delete the arm, or make it word-disjoint. The design's "unmatched control 0.900" is not interpretable as a style measurement |

**The principle behind defect 6, stated for the record.** C2ST accuracy is a
*lower* bound on the Bayes accuracy `(1 + TV)/2`. A high accuracy therefore
certifies that the distributions differ; a low accuracy certifies nothing — it
is equally consistent with "the critic is weak". Our gate is an assertion of the
second kind, so it must use the **strongest practical classifier and then require
that one to fail**. Arora et al. (ICML 2017, arXiv:1703.00573 §3.4, Cor. 3.2)
make the sharp version: a capacity-limited critic's distance "can be small even
if μ, ν are not very close", and **more evaluation data cannot rescue it**. For a
gate, maximising a quantity you then require to be small is conservative, hence
safe; the multiplicity it introduces is handled by calibrating the *same*
max-statistic under permutation, not by avoiding it.

Two attributions to correct while we are here, so they do not propagate: the
`(1+TV)/2` identity is **not** in Lopez-Paz & Oquab (cite Sen et al.
arXiv:1806.09708 Lemma 2, or Bansal et al. arXiv:2507.17026 Cor. A.2), and the
Kim/Ramdas/Singh/Wasserman paper is **arXiv:1602.02210** (Ann. Statist. 49(1),
2021), not 1703.00573.

### 3.1b What the strongest instrument reads — and why it matters

`--stage gbm`: HistGradientBoosting on the 17 interpretable trace metrics, same
word-disjoint split, with permutation importances. It reads **higher than the
MLP on every variant**, and its importances name the residual defect:

| variant | MLP speed | **GBM (17 metrics)** | top-3 GBM importances |
|---|---|---|---|
| v1 | 0.904 | **0.914** | path_len .153, step_max .137, sharp_turns .112 |
| C only | 0.879 | 0.886 | path_len .171, step_max .147, sharp_turns .121 |
| B_global | 0.775 | 0.853 | **sharp_turns .314**, path_len .098, dwell_run_max .051 |
| B_seg α=0.5 | 0.742 | 0.858 | **sharp_turns .289**, path_len .096 |
| C + B_seg α=0.5 | 0.748 | 0.854 | **sharp_turns .258**, path_len .097 |
| **+ oracle bandwidth** | 0.759 | **0.819** | sharp_turns **.161**, dwell_run_max .065 |
| real vs real (923 pairs) | 0.541 | 0.538 | — |

Three readings, all consequential.

1. **The committed MLP under-reads by 6–11 points** once the speed defect is
   fixed (0.742 vs 0.858). A gate run on the MLP alone would overstate v2's
   progress by roughly a factor of two.
2. **After re-timing, one feature dominates: `sharp_turns`**, at importance
   0.26–0.31 — three times any other. The generator's remaining signature is
   cornering, not speed. That is an independent instrument arriving at the same
   conclusion as §2.1, and the bandwidth arm cuts that importance to 0.161,
   which is the mechanism-level confirmation that the defect is acquisition and
   not motor behaviour.
3. **Gap-closure depends on which instrument you ask.** Against the 0.50 floor,
   A+B+C closes **39 %** of the MLP speed-view gap but only **15 %** of the GBM
   gap; adding S5 takes the GBM figure to **23 %**. Any pre-registered bar must
   therefore name its instrument. Both are reported below.

### 3.2 Metrics to add to the §1.1 battery

Ranked by measured KS on v1 (the number is the discriminative power actually
observed, so this ranking is empirical, not editorial):

| metric | KS on v1 | why |
|---|---|---|
| **speed autocorr lag-1** (`ac1`) | **0.618** | highest of any single statistic; catches the temporal roughness the marginals miss |
| speed spectral centroid | 0.566 | frequency-domain form of the same defect |
| **LDLJ** (log dimensionless jerk) | 0.518 | smoothness; notably it gets *worse* under speed-only re-timing, so it is not redundant with `step_cv` |
| speed-profile minima per segment | 0.451 (as `n_crit_pts`) | stroke count — real ≈ 0.77/segment, v1 1.07 |
| accel p95, normalized peak-velocity position | 0.450 / 0.421 | standard kinematic descriptors that do separate here |
| **speed–curvature slope and R²** | 0.156 / 0.274 **on v1** — but **0.566 / 0.444 on re-timed output** | the one metric that is nearly blind to the old defect and wide awake to the new one (§2.3); it must be in the battery *before* B lands, not after |
| SPARC · deviation percentiles · launch accel | 0.195 · ≤0.05 · 0.059 | **drop** — they do not discriminate our generator; SPARC is designed to be noise-robust, which is precisely our signal (it does separate min-jerk synthesis from real, which is not the comparison we need) |

### 3.3 Distributional metrics beyond KS

Add **PRDC** (precision/recall/density/coverage) *with the real-vs-real control
run every time* — the control is not optional: `prdc`'s own self-test on two
identical Gaussians returns precision 0.804, so an uncontrolled PRDC number is
meaningless. Add **per-word authenticity** where the real corpus supports it
(FUTO has 4,603 English words with ≥20 real traces). Skip DTW/soft-DTW (invariant
to the property most likely wrong), TimeGAN's discriminative/predictive scores
(degenerate at dim = 2) and Context-FID (embedder-dominated).

Free extra: **FUTO's own diagnostic** — greedy CTC accuracy of a model *on the
synthetic distribution*. Their 96.5 % is the signature of a trivially fittable
generator; our own ru192 overfit (PHASE_J §6.5) is the same phenomenon seen from
the training side.

### 3.4 Re-registering the gates — and an uncomfortable fact

Two findings point in *opposite* directions and both are right.

**τ = 0.70 is far too permissive as a statistical bar.** Since
`TV ≥ 2·acc − 1`, a bar of 0.70 admits a total-variation distance of 0.40 —
40 % of the probability mass perfectly separable. The substantively defensible
bars are ≈ 0.60 ("acceptable", TV ≤ 0.20) and ≈ 0.55 ("good", TV ≤ 0.10), stated
as a one-sided 95 % upper confidence limit rather than a point estimate.

**And 0.70 is unreachable anyway.** The best measured transplant variant is
0.742 on ru (0.735 final-epoch), of which ≈ 0.15 is donor-population mismatch
(§1.0 ii) — a term no generator change can touch.

The honest resolution is not to move the bar until the generator passes. It is
to say plainly: **an English-donor transplant cannot be made statistically
indistinguishable from real Russian swipes, and v2 will not claim to be.** So
the gate is split into two roles it was conflating:

* **A progress meter, which v2 must move.**
  `gap-closure = (acc_v1 − acc_v2) / (acc_v1 − 0.50)`, floor measured, not
  assumed, **instrument named**: on the MLP speed view (0.904 → 0.748) the
  prototype closes **39 %**; on the GBM metric gate (0.914 → 0.854) it closes
  **15 %**, rising to **23 %** with the S5 bandwidth stage; on the English
  footing (0.824 → 0.604) it closes **68 %**. The spread between the first two
  is why the instrument has to be stated; the gap between the first and the last
  is the price of the donor bank.
* **A standard, which v2 will not meet, and which is recorded as not met.**
  UCL₉₅ ≤ 0.60. Reporting it as an open shortfall keeps the pressure on the one
  lever that could close it — target-script motor data — instead of retiring
  the question by redefining success.

The binding ship/no-ship decision therefore stays with **G5** (the real ru
probe), which is what it was always for.

---

## Part 4 — The amended v2 spec

### 4.1 Pipeline (diff against `SYNTH_V2_DESIGN.md` §3.1)

```
S0 lexicon     unchanged + wordfreq token frequency as the draw weight   (A)
S1 donor index (vertex_count, ⌊log1.25 L_polyline⌋) + per-user sub-index (C, D)
               CHANGED: the donor bank must also carry each donor's RAW
               timestamps / duration — required by S5 and absent today
S2 draw        word ~ token_freq;  user ~ session block  (D);
               donor = argmin over k=16 of Σ_seg |log(L_d/L_s)|          (C)
S3 warp        layout_aug.warp_path — UNCHANGED (all Phase-H invariants)
S4 re-time     CHANGED from the design: vertex-aligned PER-SEGMENT arc-progress
               transfer, cross-segment budget m_k ∝ n_k·ρ_k^α, α = 0.5    (B′)
               (design's global-arc-fraction form is measurably worse on
                stroke count, speed autocorrelation and the gate)
S5 acquire     NEW: draw duration T = f(path_len, S) fit on MIT English, then
               re-featurize through the real 60 Hz chain (n60 = round(T/16.667)+1
               nodes → 64 index-uniform)                                  (NEW §2.1)
S6 clip+write  unchanged, plus provenance: generator version, option mask,
               donor id, donor user id, drawn duration
```

### 4.2 Code changes

* `script_synth.py`: S1/S2/S4/S5 (~220 lines, up from the design's ~150).
* Donor-bank builder: retain `duration_ms` (and ideally raw points) per donor —
  **prerequisite for S5**, and the only change with a data-regeneration cost.
* `synth_gap_audit.py`: add `ac1`, spectral centroid, LDLJ, minima-per-segment
  to `trace_metrics`; add the real-vs-real and label-shuffled arms and the
  final-epoch statistic to `stage_classifier`; add a GBM gate.
* `synth_retime_probe.py` (committed with this audit): the A/B harness; its
  `retime_segment` is the reference implementation of S4.
* `cyrillic_synth.py`: untouched (historical record).

### 4.3 Amended gates

| gate | bar in `SYNTH_V2_DESIGN` | amended bar | why |
|---|---|---|---|
| G0 warp invariants | identity exact, ideal→ideal < 1e-5 | **unchanged** | still holds; S4/S5 act after the warp |
| G1 endpoint band | in-band, wrong-geo control collapses | **unchanged** | measured unaffected by B/C (start_d, end_d KS identical) |
| G2 length mix | ±5 pts of wordfreq mass **and** ru ≤3 ∈ 30–40 % | **±3 pts of wordfreq mass only**; record the 26.8 vs 35.6 register residual | the two halves are mutually unsatisfiable (§1.1) |
| G3 kinematics | step_cv < 0.20, step_max < 0.20, sharp_turns < 0.25 | step_cv < 0.15, step_max < 0.12 (**both already met**: 0.125/0.096); sharp_turns **< 0.32** with S5, and reported without it; **new**: ac1 KS < 0.12, minima/segment within ±0.10 of real, and **speed–curvature slope KS < 0.35 with R² KS < 0.40** | 0.25 on sharp_turns is unreachable even with an oracle (0.288); the coupling bars are new because re-timing *creates* that defect (§2.3) and the prototype sits at 0.306/0.351 with S5 |
| G4 discriminability | speed ≤ 0.70, coords ≤ 0.68, single MLP, max-over-epochs, one split | **protocol**: K-fold word-disjoint, max over a pre-registered classifier×view family, within-pair permutation null, real-vs-real validity arm in [0.48,0.52]. **Bar**, instrument-named: MLP speed-view gap-closure ≥ 35 %, **GBM metric-gate gap-closure ≥ 20 %**, en→en footing ≥ 65 %; UCL₉₅ ≤ 0.60 recorded as an **open shortfall**, not a pass condition | 0.70 is simultaneously too permissive as a standard (TV ≤ 0.40) and unreachable on ru (≈0.15 is donor-population mismatch). §3.4 |
| G5 downstream | in-dict t1 ≥ 78.9, ≤3 stratum ≥ 86.4 | **unchanged** — this is the binding gate and the only one that decides shipping | untouched by the audit |

**Revised expectation, pre-registered.** The design's +2…+5 band (best estimate
+3) is left standing, with the composition changed: fix A carries more of it than
the design assumed (it is the only fix that changes *what the model is trained
on* rather than how it moves), fix B's contribution is now known to be capped by
A1 on ru, and the new S5 is worth an unknown but non-zero amount that will show
up mostly in the turn family. If G5 lands below +1.5 while G2–G4 pass, the
en→en control says the remaining gap is the donor bank's language, and the next
lever is target-script motor data — not more generator engineering.

### 4.4 What is explicitly NOT recommended

ΣΛ velocity sampling (§2.4); **two-thirds-power-law re-timing and minimum-jerk
re-timing as generator mechanisms** (§2.3 — note the power law is nonetheless
*mandatory as a gate metric*, the two uses point opposite ways); curvature-warped
or rank-matched time densities (§2.3, both measured worse); CLC/Fitts
parameterisation on top of the measured α, and any per-corner time constant
(§2.5); any learned generator (§2.6); and **any parameter of any kind fit
against Yandex statistics** — tempo, duration, start-dwell, length mix. The α
exponent and the duration model are fit on MIT English and *checked* on Russian,
never fit on it.

---

## Part 5 — Repository and paper inventory

42 repositories were cloned to `~/ctc-train/research/` as read-only reference.
Licence status governs whether anything may be ported into the ML repo
(MIT/Apache preferred) or the GPL-3.0 app.

**Two-sample testing** (`~/ctc-train/research/two-sample/`): `mmdagg` (MIT,
Schrab et al. JMLR 2023 — the recommended MMD arm, no bandwidth choice to
defend), `DK-for-TST` (MIT, Liu et al. ICML 2020 deep-kernel test; note `np.int`
breakage in the C2ST helpers, not in `MMDu`/`mmd2_permutations`),
`torch-two-sample` (BSD-3 — multivariate energy distance with permutation
p-values; needs `setup.py build_ext --inplace`), `interpretable-test` (MIT,
ME/SCF localisation; dead `theano` import, use `GaussUMETest`), `kernel-gof`
(MIT). `classifier_tests` is Lopez-Paz & Oquab's own repo but has **no licence
file** and is Torch7/Lua — consult, never vendor. For the plain test,
`scipy.stats.permutation_test` + `sklearn.metrics.pairwise.rbf_kernel` is ~10
lines and needs none of them.

**Permissive, verified on disk (portable):** `eval/prdc` (MIT — the PRDC
implementation to use), `eval/alaa-faithful` (MIT, **but see below**),
`eval/diffusion-ts-metrics` (MIT), `siva82kb-SPARC` (ISC — smoothness/LDLJ
implementations), `two-sample/{mmdagg, DK-for-TST, interpretable-test,
kernel-gof}` (MIT), `two-sample/torch-two-sample` (BSD-3),
`handwriting-rnn/{Handwriting-synthesis, pytorch-handwriting-synthesis-toolkit}`
(MIT), `diffusion/{Diffusion-Handwriting-Generation, SketchKnitter}` (MIT),
`biometrics/AngularSpeedPowerLaw` (MIT), `generators/{Indic-Swipe-v2, traj_gen,
gesture-recognition}` (MIT), `handwriting-biometrics` (MIT).

**Blockers — action required before anything is reused:**

* `eval/alaa-faithful/metrics/improved_precision_recall.py` is **NVIDIA
  CC BY-NC 4.0** vendored inside an MIT repo (header intact). Delete that file
  if the clone is ever used; `prdc` reimplements the same metric under MIT.
* **IAM-OnDB-derived weights** ship inside three MIT-*code* clones
  (`Diffusion-Handwriting-Generation/weights/`, `Handwriting-synthesis/results/`,
  `pytorch-handwriting-synthesis-toolkit/docs/`). Code MIT, weights encumbered —
  never load them.
* **No licence at all** (= all rights reserved, read-only):
  `swipe/{neural-swipe-typing, indic-swipe, gesture_augmentation, swipetest,
  sigma-lognormal}`, `iDeLog`, `biometrics/{sapiagent, antal-mouse-dynamics}`,
  `BeCAPTCHA-Mouse`, `two-sample/classifier_tests`, `siva82kb-smoothness`.
  This includes **every** ΣΛ implementation found — a second reason §2.4 is
  parked.
* `calligraph` and `futo/swipe-library` are **GPL**; `futo/android-keyboard` is
  **FUTO Source First** (not open source).

**Motor-control / smoothness code:** `siva82kb-SPARC` (**ISC**, canonical —
`scripts/smoothness.py` has `sparc()` and `log_dimensionless_jerk()`, numpy
only, doctests reproduce; **vendor from here**). Its successor
`siva82kb-smoothness` has **no LICENSE file** despite the README claiming ISC —
do not use. Two implementation traps found in the SPARC paper itself: Eq. 2's
DLJ duration exponent (5) is dimensionally wrong and the reference code uses 3;
and the code normalises the spectrum by `max(Mf)`, not `Mf[0]`, so the speed
array must not be mean-centred. LDLJ is exactly invariant to both of our
normalizations and is the cleaner of the two here.

**Donor-corpus footing, re-verified:** How-We-Swipe **MIT** (OSF record,
`https://osf.io/sj67f/`; the OSF copy is additionally marked CC BY 4.0 —
1,338 users, 8 M+ touch points, raw timestamps present),
swipe.futo.org data **MIT**, FUTO model *weights* under a separate
attribution-mandatory licence. Yandex remains **eval-only**; a Kaggle mirror
relabelled it Apache-2.0, which is a re-uploader's claim and not a grant.

**Key papers.** Graves 2013 (arXiv:1308.0850); WordGesture-GAN, CHI '23
(10.1145/3544548.3581279, no code); FUTO Swipe (arXiv:2606.25247 — Appendix A is
the primary source for the synthetic-data negative result); Gesture2Text
(arXiv:2410.18099); Shen, Dudley & Kristensson ISMAR '21 (four-way synthesizer
comparison); SHARK² UIST '04 (10.1145/1029632.1029640); Cao & Zhai CLC, CHI '07;
Flash & Hogan 1985; Viviani & Terzuolo / Lacquaniti 1983 (two-thirds law) with
Schaal & Sternad 2001 as the standard critique; Plamondon's kinematic theory and
iDeLog (Ferrer et al.); Lopez-Paz & Oquab, C2ST (arXiv:1610.06545); Kim et al.,
classification accuracy as a proxy for two-sample testing; Alaa et al. ICML 2022
(alpha-precision/beta-recall/authenticity); Naeem et al. 2020 (PRDC);
Balasubramanian et al. 2015 (SPARC); Frank et al. 2013 (Touchalytics);
Bachert & Hesenius (synthetic-gesture evaluation taxonomy); Mehra et al. ICASSP
2020 (Apple, 10.1109/ICASSP40776.2020.9053689 — spline synthetic hurts, GAN
synthetic helps, at 2.2 M real traces); Sivek & Riley 2022 (10.1145/3546737,
Gboard tap personalization, gesture explicitly open); Kohút et al. ICDAR 2023
(arXiv:2302.06318) and Li et al. 2026 (arXiv:2603.16883) — writer-style gains
are writer-*dependent* only; Yin et al. CHI '13 (10.1145/2470654.2481384) and
Azenkot & Zhai MobileHCI '12 (per-user/per-posture touch offsets); Bi, Li & Zhai
CHI '13 (FFitts, σ_a = 0.94 mm 1-D / 1.5 mm 2-D). Methodology: Arora et al. ICML
2017 (arXiv:1703.00573 §3.4, Cor. 3.2 — a capacity-limited critic proves
nothing, and more data cannot fix it); Cawley & Talbot JMLR 2010 and Dwork et
al. Science 2015 (selection on the evaluation set); Gao, Schulman & Hilton ICML
2023 (arXiv:2210.10760 — proxy/gold divergence under optimization pressure);
Gretton et al. JMLR 2012 (MMD); Jitkrittum et al. (arXiv:1605.06796).

---

## Appendix — reproducing this audit

```bash
python3 ctc/synth_retime_probe.py --stage law          # the alpha ≈ 0.46 table (§1.2)
python3 ctc/synth_retime_probe.py --stage variants     # builds + KS-scores all variants
python3 ctc/synth_retime_probe.py --stage classifier   # gate + floor + H0 arms (§1.0i, §3.1)
python3 ctc/synth_retime_probe.py --stage gbm          # stronger gate + importances (§3.1b)
python3 ctc/synth_retime_probe.py --stage coupling     # speed-curvature slope/R2 (§2.3)
python3 ctc/synth_retime_probe.py --stage enen         # the A1 control (§1.0ii)
```

`variants` must run before `classifier`, `gbm` and `coupling` (they read
`variants.npz`); `law` and `enen` are independent.

Inputs: `~/ctc-train/synth_gap/matched.npz` (from `synth_gap_audit.py --stage
data`), `~/ctc-train/cache/train_t3{futo,hws}.npz`. Outputs:
`~/ctc-train/synth_gap/{law,variants_ks,variants_clf,variants_gbm,
variants_coupling,enen}.json` and `variants.npz`. Runtime ≈ 70 min CPU
(`gbm` is most of it) plus seconds of GPU; seed 1234 throughout, donor draw seed
20260819 (v1's). The `variants` stage asserts the v1 reproduction is bit-exact
and aborts if it is not.
