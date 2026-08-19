# Phase P — the amended synthetic generator v2, gated and shipped

**Opened:** 2026-08-19. **Workdir** `~/ctc-train`, **GPU** RTX 5080 Laptop (16 GB).
The app repo `/home/will/git/swype/CleverKeys` is a **read-only reference**.

Phase O closed with one instruction at the head of its Phase-P list: *fix the
generator's word draw, then re-run the ru calibration.* Two documents were
written before anything was built — `SYNTH_V2_DESIGN.md` measured what v1 gets
wrong and proposed fixes A–D, and `SYNTH_V2_RESEARCH_AUDIT.md` **built those
fixes as prototypes and scored them** before writing a verdict, amending three
of the four and adding two the design did not contain. Phase P implements the
union of the two, gates it on the amended battery, and puts it through the one
gate that decides shipping: the real Russian probe.

**Headline.** The generator v2 clears every pre-registered gate. On the real
Yandex probe — 9,416 rows, 8,471 in-dict, eval-only footing, no real Cyrillic
row anywhere in training — the ru model goes **77.42 → 79.73 in-dict top-1**
(+2.31, paired McNemar p = 2.6e-09), inside the pre-registered +2…+5 band. The
mechanism is not subtle: **greedy CTC on real Russian traces goes 37.07 →
56.12**. Fixing a generator that was 90 % separable from real traces on its
speed profile alone bought nineteen points of raw emission accuracy in a script
with no training data.

Two things are recorded as missed, not explained away: the G5 corollary on the
short-word stratum (85.77 against a bar of 86.4, though the paired test cannot
distinguish that regression from zero, p = 0.27), and the G4 *standard*
UCL₉₅ ≤ 0.60, which the research audit already predicted is unreachable with an
English donor bank and which this phase confirms.

---

## 1. What v2 is

Authority order: where `SYNTH_V2_RESEARCH_AUDIT.md` and `SYNTH_V2_DESIGN.md`
conflict, the audit wins, because it measured. The shipped pipeline is the union
of the two, stage by stage, with the flag letter that switches each one
(`script_synth.py --stages`, default `a,c,b,s5`):

| stage | what it does | flag | source |
|---|---|---|---|
| S0 | lexicon load unchanged + **wordfreq token mass** as the draw weight | `a` | design A |
| S1 | donor index by vertex count + polyline length, plus a per-contributor sub-index | `c`/`d` | design B/C/D |
| S2 | word ~ token mass; donor = argmin over k = 16 of `Σ_seg |log(L_dst/L_src)|` | `c`/`d` | design C/D |
| S3 | `layout_aug.warp_path` — **unchanged**, every Phase-H invariant intact | — | design |
| S4 | **vertex-aligned per-segment re-timing**, `m_k ∝ n_k·ρ_k^0.5` | `b` | **audit B′** (replaces the design's global form) |
| S5 | **acquisition-bandwidth matching** — draw a duration, re-featurize through the real 60 Hz chain | `s5` | **audit, new** |
| S6 | clip + write + per-row provenance (donor row, donor contributor, drawn duration) | — | both |

`--generator v1` reproduces the Phase-O mechanism **bit-exactly**, RNG call
sequence included, so every paired ablation below is a true control rather than
a re-implementation. Two selftests assert it (`--selftest-v1`):

* against `cache_ru_phaseO/train_synth.npz` (script_synth v1, 90/10 donor side): 20,000 rows, max|Δ| = 0;
* against `cache_ru_synth/train_synth.npz` (**`cyrillic_synth.py`**, full donor pool — the cache the 77.41 baseline was trained on): 20,000 rows, max|Δ| = 0.

The second is what makes §4's attribution possible: the *entire* difference
between this phase's v1 control and Phase I-B's shipped ru model is the donor
side, and nothing else.

### 1.1 Three implementation findings the two documents did not anticipate

**(a) Fix A has a projection trap that is silently catastrophic on Greek.**
Querying `wordfreq.word_frequency` with the *projected* lexicon form returns
**zero for 90 % of the el pack** — the projection strips accents and restores
word-final ς, while wordfreq's Greek list carries the accented, casefolded,
σ-final forms. The surviving 10 % collapses the draw onto 2–3-letter words
(≤3 mass 0.896, mean length 2.79) — a generator that would have trained Greek on
almost nothing but function words, passing every gate the design specified,
because none of them looks at the draw. The fix is to walk wordfreq's own list
and accumulate each **token's** mass into *its projection*, which is the correct
estimand anyway (the token mass of the projected equivalence class). Measured
after the fix: **zero** zero-frequency lexicon entries on all six scripts, and
ru reproduces the audit's 0.268 / 5.74 exactly.

| script | tokens walked | matched | zero-freq words | ≤3 mass | mean len |
|---|---|---|---|---|---|
| ru | 713,447 | 51,152 | 0 | 0.268 | 5.74 |
| el | 46,916 | 41,127 | 0 | 0.441 | 4.85 |
| uk | 443,616 | 50,097 | 0 | 0.309 | 5.34 |
| bg | 37,325 | 35,790 | 0 | 0.388 | 4.79 |
| mk | 260,128 | 49,995 | 0 | 0.418 | 4.73 |
| he | 591,944 | 50,930 | 0 | 0.385 | 4.06 |

**(b) The donor bank had to be rebuilt, and the rebuild is bit-clean.** S5 needs
raw durations, which `featurize` throws away. `prepare_data.py` now records
`duration_ms`, `n_points` and a contributor `group` per row, and the rebuild is
gated on `features`/`targets`/`target_lengths`/`words` being **bit-identical** to
the pre-Phase-P caches (`--verify-against`). They are. Measured while doing it:

| donor corpus | rows | duration p50 | contributor ids |
|---|---|---|---|
| FUTO `train_t3futo` | 927,869 | **787 ms** | 800,646 / 927,869 (86.3 %) |
| HWS `train_t3hws` | 76,748 | **1,117 ms** | 0 (the session index is FUTO-only) |

The audit quoted the donor bank's tempo as HWS's 1,113 ms against real ru's
701 ms. The bank is 92 % FUTO, whose median is **787 ms**, so the tempo mismatch
the design worried about is roughly a third of what was assumed — and at matched
*vertex count* the English donors are already close to real Russian at matched
word length (2-vertex 265 ms / 3-vertex 443 ms against real ≤3-letter 329 ms;
7-vertex 1,499 ms against real ≥7-letter 1,361 ms). This is why S5's shippable
form works at all.

**(c) The S5 duration law, fit on MIT English only.** `log T = a + b·log L +
c·log S` over 882,628 donors: **b = 0.262** at fixed polyline complexity
(R² 0.729; the length-only elasticity is 0.893, which is the number that
confounds length with vertex count). S5 keeps the donor's own tempo and prices
only the geometry change, `T_target = T_donor · (L_dst/L_src)^0.262`, so the
transplant's load-bearing property — the timing is a human's, not a model's —
survives. **No Yandex statistic enters the fit.** The result, measured against
the real partners' own durations on the 9,416 word-matched rows:

| | p25 | p50 | p75 | KS vs real |
|---|---|---|---|---|
| S5 drawn durations | 426 | **790** | 1,338 | **0.139** |
| real Yandex partners | 399 | **703** | 1,030 | — |

A duration model that has never seen a Russian trace lands its median within
12 % of the real one.

---

## 2. The gate battery, amended and run

`synth_gap_audit.py --stage v2` builds the word-matched arms **with the shipped
generator** (and asserts its v1 path reproduces `matched.npz` bit-exactly, so
the thing being gated is the thing that will be trained on);
`--stage gates` runs G1–G4. Because the words come from the real Yandex
partners, **fix A is out of scope for the matched arms by construction** — they
measure the mechanism (C, B′, S5). Fix A is gated separately by G2 on the
training draw, which is the only place a draw policy can be measured at all.

Every classifier defect the audit's §3.1 named is repaired: final-epoch accuracy
instead of max-over-epochs, a mandatory real-vs-real floor arm, an exact
within-pair permutation null, 5-fold word-disjoint CV with a one-sided 95 % UCL,
and a bar read as the max over a pre-registered classifier × view family.

### 2.1 G1 — endpoint band and the falsification control · **PASS**

| arm | start-hit | end-hit | start_d | end_d |
|---|---|---|---|---|
| real Yandex | **0.9151** | 0.6380 | — | — |
| v1 | 0.7105 | 0.6111 | — | — |
| **v2 (C+B′+S5)** | **0.7298** | **0.6335** | — | — |
| v2, wrong-geometry control | **0.0200** | 0.0060 | 0.43 | 0.56 |

v2 moves *toward* real on both ends, and the permuted-geometry control still
collapses to 2 % — the endpoint test is measuring the frame, not arithmetic.
(For the record, the "0.917" that circulates in Phase-O prose is the **real**
corpus's start-hit, not v1's.)

### 2.2 G2 — length mix · **PASS**, with the register residual recorded

Bar as amended: within **±3 pts of the wordfreq token mass** per bucket. The
design's second clause ("ru ≤3 in 30–40 %") was dropped because wordfreq's own
ru ≤3 mass is 26.8 % and the two halves are mutually unsatisfiable.

| set | ≤3 | 4–6 | ≥7 | mean len |
|---|---|---|---|---|
| wordfreq token mass (the target) | 0.268 | 0.376 | 0.356 | 5.74 |
| **v2 train draw** | **0.269** | **0.376** | **0.355** | **5.74** |
| v1 train draw (`255 − rank`) | 0.033 | 0.291 | 0.675 | 7.91 |
| real Yandex usage | 0.356 | 0.438 | 0.205 | 4.78 |

Max bucket deviation **0.001**. The remaining **+8.8 pt** gap to real usage is a
**register** difference — wordfreq's ru blend is written and subtitle text, swipe
input is mobile chat — and it is recorded as an open, unclosed residual rather
than tuned away, because tuning it against the Yandex mix would consume the sole
validator. The licence-clean route, if it is ever wanted, is a chat-register
frequency list chosen without reference to Yandex.

One measured cost of fix A worth stating: the draw now concentrates on frequent
words, so a 1 M-row training corpus covers **36,510** distinct words where v1
covered 49,508 of the same 49,704-word lexicon. The model learns emissions, not
vocabulary, and §4 shows the trade is strongly positive — but it is a real change
in what the corpus contains.

### 2.3 G3 — kinematic parity · **PASS on every bar**

KS against the real partners, n = 9,416 word-matched pairs. The six italicised
columns are the metrics Phase P added because the committed battery is entirely
marginal and aggregate and cannot see temporal structure.

| arm | step_cv | step_max | sharp_turns | turn_mean | *ac1* | *spec_centroid* | *ldlj* | *minima/seg (KS)* | *sc_slope* | *sc_r2* | min/seg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **real** | — | — | — | — | — | — | — | — | — | — | **0.77** |
| v1 | 0.597 | 0.519 | 0.443 | 0.405 | **0.618** | 0.657 | 0.588 | 0.419 | 0.156 | 0.274 | 1.13 |
| C only | 0.479 | 0.403 | 0.409 | 0.368 | 0.531 | 0.585 | 0.537 | 0.344 | 0.163 | 0.262 | 1.05 |
| B′ only | 0.117 | 0.103 | 0.412 | 0.332 | 0.115 | 0.207 | 0.202 | 0.171 | **0.436** | **0.419** | 0.91 |
| S5 only | 0.495 | 0.395 | 0.370 | 0.355 | 0.436 | 0.464 | 0.381 | 0.338 | 0.105 | 0.234 | 1.04 |
| C+B′ | 0.125 | 0.096 | 0.380 | 0.308 | 0.076 | 0.132 | 0.132 | 0.141 | 0.366 | 0.397 | 0.89 |
| **C+B′+S5** | **0.108** | **0.082** | **0.306** | **0.271** | **0.067** | **0.114** | **0.113** | **0.122** | **0.317** | **0.362** | **0.85** |

| bar (amended) | value | verdict |
|---|---|---|
| step_cv < 0.15 | 0.108 | PASS |
| step_max < 0.12 | 0.082 | PASS |
| sharp_turns < 0.32 (with S5) | **0.306** | PASS |
| ac1 KS < 0.12 | 0.067 | PASS |
| speed–curvature slope KS < 0.35 | 0.317 | PASS |
| speed–curvature R² KS < 0.40 | 0.362 | PASS |
| minima/segment within ±0.10 of real | 0.85 vs 0.77 | PASS |

Three readings the table earns.

1. **S5's shippable form beats its own oracle bound, nearly.** The audit's
   oracle arm — which reads the validator's actual durations and is therefore not
   shippable — bounded sharp_turns at 0.293. The English-fit duration model gets
   **0.306**, within 0.013 of a bound it was not allowed to use. Half the
   residual cornering gap really was an acquisition artefact, and it is now
   closed without touching a Yandex number.
2. **The coupling bars justified themselves.** Re-timing alone drives the
   speed–curvature slope KS to 0.436 — *worse* than v1's 0.156 — while every
   speed marginal it repairs shows a triumph. Without those two columns in the
   battery, fix B would have Goodharted the gate exactly as v1's endpoint-only
   gating did, one level up. C and S5 pull it back to 0.317, under the bar, but
   the defect is real and is not fully closed.
3. **`ac1` earned its place**: 0.618 on v1, the largest single-statistic gap
   anywhere in this campaign, and 0.067 after v2 — a 89 % reduction on the axis
   the committed 17-metric battery could not see at all.

### 2.4 G4 — discriminability · **PASS on every registered bar**

5-fold word-disjoint CV, final-epoch accuracy, n = 9,416 pairs.

**Validity arms (mandatory, every run).** Real-vs-real floor on 3,194
word-matched real pairs: MLP speed **0.4933**, GBM₁₇ **0.4972**, GBM₂₃ **0.4966**
— all inside [0.48, 0.52], so the run is valid and the floor is 0.50, measured
rather than assumed. Exact within-pair permutation null over 100 draws of the
GBM₁₇ fold-mean: mean **0.4999**, p95 **0.5062**, max **0.5133**. Every arm below
is far outside it.

| arm | MLP speed | MLP coords | MLP angles | **GBM₁₇** (registered) | GBM₂₃ (stronger) |
|---|---|---|---|---|---|
| v1 | 0.8766 | 0.7507 | 0.7497 | **0.9039** | 0.9328 |
| C only | 0.8542 | 0.6937 | 0.7332 | 0.8735 | 0.9132 |
| B′ only | 0.7256 | 0.7128 | 0.7264 | 0.8498 | 0.9113 |
| S5 only | 0.8334 | 0.7375 | 0.6895 | 0.8691 | 0.8981 |
| C+B′ | 0.7353 | 0.6821 | 0.7062 | 0.8364 | 0.8930 |
| **C+B′+S5** | **0.7412** | **0.6696** | **0.6400** | **0.8125** | **0.8750** |
| floor (real vs real) | 0.4933 | — | — | 0.4972 | 0.4966 |

| bar, instrument-named | measured | verdict |
|---|---|---|
| MLP speed-view gap-closure ≥ 35 % | **36.0 %** | PASS |
| **GBM₁₇** metric-gate gap-closure ≥ 20 % | **22.6 %** | PASS (the audit predicted 23 %) |
| en→en footing gap-closure ≥ 65 % | **84.2 %** | PASS |
| *(reported, never gated)* GBM₂₃ gap-closure | 13.4 % | — |
| **standard** UCL₉₅ ≤ 0.60 | 0.7467 / 0.8331 | **OPEN SHORTFALL** |

Two notes on instruments, both load-bearing.

* The 20 % bar was pre-registered against the audit's **17-metric** GBM, so the
  PASS is read from that instrument. The 23-metric battery — the same GBM with
  the six Phase-P metrics added — is a *strictly stronger* critic and closes only
  13.4 %. It is reported next to the registered number and **never gated**:
  inventing a bar after seeing the number is precisely the selection this battery
  exists to prevent. The honest summary is that a stronger critic sees more of
  what is left, which is what a stronger critic is for.
* The UCL₉₅ ≤ 0.60 standard is **not met and is recorded as not met**, exactly as
  the amendment requires. An English-donor transplant cannot be made
  statistically indistinguishable from real Russian swipes. The next section
  measures how much of that is the donor bank rather than the generator.

**The unmatched arm, repaired and re-read.** The audit's defect 7 was that
`acc_unmatched_coords` split by random *row* across two classes with different
word distributions, so it read 0.732 on coordinates from word memorisation alone
and its headline 0.900 was not interpretable as a style measurement. Splitting on
the union of words removes the memorisation channel and keeps the length-mix
signal, which is what the arm is for — it is the only view in the battery that
sees **fix A**, because the matched arms hold words fixed:

| training draw vs the real corpus, coords, word-disjoint | acc | gap-closure |
|---|---|---|
| v1 draw (`255 − rank`) | 0.8868 (0.900 with the leak) | — |
| **v2 draw (wordfreq token mass)** | **0.7206** | **43.0 %** |

That is a larger closure than the matched coords view (0.7507 → 0.6696, 21 %),
and it is the measurement that says fix A is doing real work rather than merely
satisfying its own gate.

### 2.5 The en→en control — where the residual actually lives

Split HWS in half, treat one half as "real", transplant a donor from the
disjoint other half onto the **same word on the same QWERTY geometry**. Every
difference is generator error with **zero** cross-script and zero
cross-population component. n = 9,416 pairs, same battery, final-epoch.

| arm | step_cv | step_max | sharp_turns | turn_mean | gate: speed | coords | angles |
|---|---|---|---|---|---|---|---|
| en→en, v1 | 0.535 | 0.511 | 0.279 | 0.252 | 0.8240 | 0.6657 | 0.6558 |
| en→en, B′ only | **0.010** | 0.076 | 0.257 | 0.150 | 0.5994 | 0.6545 | 0.6127 |
| en→en, **v2 (C+B′+S5)** | 0.030 | **0.048** | **0.172** | **0.117** | **0.5512** | **0.6078** | **0.5589** |
| ru, v2, for contrast | 0.108 | 0.082 | 0.306 | 0.271 | 0.7412 | 0.6696 | 0.6400 |

**On matched population the amended generator is nearly indistinguishable from
real traces: 0.5512 against a measured floor of 0.50, an 84 % gap closure.** The
Russian residual — 0.7412 — is therefore not generator error. It is the donor
bank not matching the target population, hypothesis A1, and the only lever on it
is target-script motor data. The audit priced that term at ≈0.15 of the speed
view with B alone; with the full v2 the split is 0.7412 − 0.5512 = **0.19**.

Note also that C is worth more than the ru footing shows, exactly as the audit
argued from an independent English experiment: en→en goes 0.5994 (B′ alone) →
0.5512 (C+B′+S5), a further 15 % of the remaining gap, invisible on ru because
the donor-population term masks it.

---

## 3. Fix D — built, off by default, and why

`--stages …,d,…` implements contributor-coherent donor blocks (50–200 rows per
contributor, falling back to the global pool when that contributor lacks the
vertex count). It runs: 10,832 groups over 79.7 % of the train-side bank, 19 %
fallback. It is **off in every shipped run**, for the reason the audit gave and
this phase did not disturb: the CTC encoder consumes single traces, so
per-contributor coherence has no first-order channel to the loss, it can neither
pass nor fail the acceptance criteria, and enabling it costs donor diversity
inside a block. What Phase P did take from the amendment is the cheap half —
**every generated row records its donor row and its donor contributor in the
npz**, so a later personalization or style-conditioned path has the structure
available without a regeneration.

The audit's preference for **posture**-coherent over identity-coherent blocking
is noted and not implemented: the contributor ids available here are FUTO
sessions, and HWS's `swipeFinger`/`swipeHand` fields are not plumbed through the
tier builder. Recorded as the shape any future attempt should take.

---

## 4. G5 — the ship gate

The only gate that decides shipping. Real Yandex valid-10k, eval-only footing
per `YANDEX_LICENSE_RESEARCH.md`, all 9,416 default-grid rows, 8,471 in-dict,
decoded through the exported **fp32** graph at the app's CKDT preset
(γ 1.05 / **λ 2.0** / β 0.2 / 0.3734 / 0.9882) on the `langpack-ru` 50 k trie.
Recipe verbatim from Phase O: `resbn:80`, dil 1,2,4,8, embed_hid 96, feat_v1,
94,000 steps, batch 256, lr 3e-3, wd 0.01, warmup 1,000, coupled affine sampler,
no layout-alt, greedy checkpoint selection, patience 40, seed 1234, `--workers 0`.

| arm | training cache | in-dict t1 | ≤3 | ≥4 | greedy | t3 | t5 |
|---|---|---|---|---|---|---|---|
| `ru_synth_ch80` — the registered baseline | `cache_ru_synth` (v1, **full** donor pool) | 77.42 | 86.47 | 71.70 | 37.07 | 89.06 | 91.76 |
| `phaseP-ru-v1ctl` — paired v1 control | `cache_ru_phaseO` (v1, 90/10 train side) | 75.73 | 83.66 | 70.71 | 31.34 | 88.44 | 90.93 |
| `phaseP-ru-v2` | `cache_ru_v2` (v2, 90/10 train side) | 78.87 | 83.60 | 75.88 | 55.67 | 90.73 | 93.13 |
| **`phaseP-ru-v2full` — SHIP** | `cache_ru_v2full` (v2, **full** donor pool) | **79.73** | 85.77 | **75.92** | **56.12** | **90.77** | **93.26** |

| paired comparison (McNemar, exact, n = 8,471) | Δ t1 | p |
|---|---|---|
| **v2 full pool vs the registered baseline** | **+2.31** | **2.6e-09** |
| v2 vs v1 at matched donor footing | **+3.14** | 6.4e-14 |
| the 90/10 donor split, v1 arm | **−1.69** | 5.2e-07 |
| v2 full pool vs v2 train side | +0.86 | 0.0023 |

**Verdict: G5 PASS.** 79.73 ≥ the 79.41 floor (+2 over 77.41), inside the
pre-registered +2…+5 band, best estimate was +3.

### 4.1 The registered amendment round, and why it was not probe-fitting

The first v2 arm came in at **78.87**, 0.54 short of the bar, and the paired
control said exactly where the missing point was. `script_synth --generator v1
--train-donor-side all` reproduces `cache_ru_synth` **bit-exactly** — the cache
behind the 77.41 baseline — so the only difference between that baseline and the
v1 control trained here is the donor pool: 1,004,617 traces against the 904,155
of Phase O's 90/10 train side. That costs **−1.69 real top-1 and −5.7 greedy**,
measured on the control arm, p = 5.2e-07. Phase O introduced the split for its
synthesis holdout's sake and **never trained on the resulting cache**, so this is
the first measurement of what it costs.

The one amendment round therefore changed **no generator parameter**: it put the
v2 arm on the baseline's own donor footing. The alternative reading needs no
amendment at all and gives the same answer — at matched footing v2 beats v1 by
**+3.14**, comfortably inside the band. Both are reported; neither is a choice
made after seeing which was larger, because both were computed together.

### 4.2 What actually improved, and the one thing that did not

**Emissions.** Greedy CTC on real Russian goes **37.07 → 56.12** with no real
Cyrillic row in training. Phase O's calibration had the shipped English model at
18.62 greedy and the v1 ru model at 37.13, and read the gap as "the emissions
really are script-specific". v2 says the emissions were also *generator*-limited,
by a factor almost as large again.

**The long stratum.** ≥4-letter words go 71.70 → 75.92 (+4.22, p = 3.6e-17).
That is where 61 % of real usage lives and where the v1 model was weakest.

**The short stratum — the registered corollary is MISSED.** G5's corollary said
the ≤3 stratum must not regress below 86.4; v2 reads **85.77**. Recorded as a
miss. Three facts around it, none of which retire it:

* the paired test cannot distinguish the regression from zero (b/c 210/187,
  p = 0.27, n = 3,281);
* the whole of it is carried by the donor-side term, not by v2 — the two
  train-side arms read 83.66 (v1) and 83.60 (v2), a dead heat;
* it is the stratum where the **lexicon prior**, not the encoder, does the work.
  v1 reached 86.47 on ≤3 words with a greedy of 37; the beam was carrying it. A
  model with 56 greedy leans on the prior less, and at λ = 2.0 — a value tuned
  in PHASE_J §6.9 against a *weak-emission* model — that is not obviously the
  right balance any more. **Re-tuning λ against the Yandex probe is refused**:
  λ is already one validator-fit parameter, and spending the validator again to
  recover 0.6 points on one stratum is exactly the trap this campaign keeps
  documenting. Registered as an open item for a phase that has a second real
  corpus.

---

## 5. The other five scripts

All five regenerated on v2, retrained on the **Phase-O recipe and the Phase-O
90/10 donor discipline** — deliberately, so that every per-script comparison
against Phase O changes exactly one variable, the generator. (ru ships from the
full-pool arm because its gate is real data; the five keep the split because
their holdout *is* their probe and donor-disjointness is what makes it worth
anything. The −1.69 real points §4.1 measures for that split is therefore left
on the table for these five, and is the registered recommendation for whoever
regenerates next.) **§8 executes that recommendation and reports what it is
worth: nothing this probe can see.**

Probe: each script's own 10,000-row v2 synthesis holdout, disjoint donor half,
independent word draw (seed 777), decoded through the exported fp32 graph at
γ 1.05 / λ 2.0 / β 0.2 / 0.3734 / 0.9882.

| script | K | greedy | **in-dict t1** | ≤3 | ≥4 | vs ch192 EN | vs ch80 EN | permuted geometry | ≥70 gate |
|---|---|---|---|---|---|---|---|---|---|
| **el** Greek | 25 | 69.01 | **90.69** | 95.79 | 86.70 | **+6.02** | **+6.92** | 0.00 | pass |
| **uk** Ukrainian | 31 | 55.09 | **87.97** | 92.73 | 85.88 | **+5.39** | **+7.22** | 0.02 | pass |
| **bg** Bulgarian | 30 | 55.97 | **82.26** | 85.64 | 80.12 | **+5.21** | **+7.29** | 0.00 | pass |
| **mk** Macedonian | 31 | 64.56 | **89.02** | 93.52 | 85.83 | **+5.57** | **+6.66** | 0.00 | pass |
| **he** Hebrew | 27 | 56.88 | **77.00** | 85.83 | 71.47 | **+8.06** | **+10.65** | 0.01 | **pass** (was FAIL) |
| *(ru, same probe family)* | 31 | 50.98 | *86.49* | *90.41* | *85.05* | *+3.87* | *+6.07* | *0.00* | *pass* |

**The sign flip is the result, not the level.** On Phase O's v1 holdouts every
script model **lost** to the 3×-capacity English ch192 zero-shot, by −0.56 to
−3.75, which was the single clearest symptom that the v1 holdout was measuring
"how English-shaped is this distribution" rather than "how good is this model".
On the v2 holdouts every script model **beats** ch192, by +5.2 to +8.1 — and on
Russian, where the same pair can be checked on real swipes, the holdout margin
(+3.87) and the real margin (+3.41) now agree to half a point. Greedy tells the
same story from the emissions side: the script models read 55–69 against the
English controls' 13–25.

**Hebrew clears its gate.** Phase O exported he flagged, at 65.36 against a
registered ≥70 band. On v2 it reads **77.00** at the same preset, +11.6 on a
probe that is not the same probe — the honest way to say it is that he is no
longer the outlier of the six on its own generator, and that its ≤3 stratum
(50.31 in Phase O, the diagnosis offered there) reads 85.83 now that the draw is
frequency-weighted. It remains the weakest of the six and remains unvalidated
against any real Hebrew swipe, because none exists.

**Falsification control.** `eval_script --permute-layout 4242` — every key centre
moved to some other key's position — collapses all five to **0.00–0.02** in-dict
t1 and 0.00 greedy, as it did in Phase O. The layout json is a testable claim and
it passes.

**Export gates.** Every fp32 export clears with **100/100 argmax agreement** on
the sliced contract view against real traces on the real layout, and the fp16w
ship bytes cost at most 0.03 t1:

| script | BN fold (sliced) | fp32 vs torch (sliced) | argmax | fp16w vs fp32 (white noise) | argmax | fp16w decode cost |
|---|---|---|---|---|---|---|
| ru | 1.20e-04 | 9.92e-05 | **100/100** | 1.06e-01 | 93/100 | 79.73 → 79.75 (+0.02, real probe) |
| el | 9.73e-05 | 8.58e-05 | **100/100** | 6.81e-02 | 96/100 | 90.69 → 90.68 (−0.01) |
| uk | 6.48e-05 | 1.28e-04 | **100/100** | 1.13e-01 | 95/100 | 87.97 → 87.97 (0.00) |
| bg | 4.71e-04 | 1.83e-04 | **100/100** | 5.49e-02 | 98/100 | 82.26 → 82.23 (−0.03) |
| mk | 8.01e-05 | 1.60e-04 | **100/100** | 5.42e-02 | 98/100 | 89.02 → 89.02 (0.00) |
| he | 1.04e-03 | **1.16e-03** | **100/100** | 5.41e-02 | 99/100 | 77.00 → 77.00 (0.00) |

**he's fp32 export needed the parity tolerance relaxed from 1e-3 to 2e-3**
(`--parity-tol 2e-3`) and is disclosed rather than quietly re-run: the measured
sliced residue is 1.16e-03 against a historical envelope of 0.8e-4…7.6e-4, with
argmax agreement 100/100 on **both** probes — which is the binding gate the
exporter's own docstring names. he is flagged for this in the registry.
`quantize_onnx.parity_vs_source` probes with white noise, which PHASE_J §5.2
established is not a calibrated stand-in in either direction; the decode column
is the binding evidence and it is free.

### 5.1 What these five numbers are NOT

**The Phase-O caveat carries over unchanged, and gets one turn worse.** Greek,
Ukrainian, Bulgarian, Macedonian and Hebrew have no real swipe corpus in
existence, so their only probe is a synthesis holdout — and a **v2** holdout is
generated by the **v2** generator. Every per-script number below is
generator-relative: it measures generalization to fresh samples of this
generator over a disjoint donor half and an independent word draw, and nothing
more. Phase O proved that such a probe inverts model comparisons on the capacity
axis and on λ. Nothing in Phase P rehabilitates it, and Phase P deliberately did
not re-run the calibration question on it, because the calibration Phase O ran
is the answer: **on the one script where both probes exist, the real probe is the
one that was right.**

What *is* transferable is the ru result: at matched capacity and matched recipe,
switching the generator from v1 to v2 is worth **+2.3 real top-1 and +19 real
greedy** on Russian. The five scripts share the generator, the donor bank, the
architecture and the recipe. They do not share the validation.

One more thing the v2 holdouts are not: comparable to Phase O's. A v1 model
scored on a v2 holdout is out of distribution — on ru the v1 ship model reads
80.87 there against the v2 model's 86.49, a 5.6-point gap where the real probe
says 2.3. Any cross-generation per-script table must therefore compare *margins
against a fixed control*, which is what §5's ch192/ch80 columns do, and never
levels.

---

## 6. Artifacts and reproduction

```bash
# donor bank (adds duration_ms / n_points / group; features asserted unchanged)
python3 ctc/prepare_data.py --extra-train data/tier_t3futo.jsonl \
    --out-name train_t3futo --jobs 10 \
    --group-index cache/futo_session_index.npz \
    --verify-against cache/train_t3futo.npz

# generator, v1 selftest then v2
python3 ctc/script_synth.py --code ru --generator v1 --train-donor-side all \
    --rows 20000 --splits train --cache cache_tmp --force \
    --selftest-v1 cache_ru_synth/train_synth.npz
python3 ctc/script_synth.py --code ru --cache cache_ru_v2full --rows 1000000 \
    --train-donor-side all

# gate battery (G1-G4)
python3 ctc/synth_gap_audit.py --stage v2       # word-matched arms, shipped generator
python3 ctc/synth_gap_audit.py --stage gates --permutations 100
python3 ctc/synth_retime_probe.py --stage enen  # the A1 control, with a v2 arm

# ship gate (G5)
python3 ctc/train.py --cache cache_ru_v2full --train-npz train_synth.npz \
    --layout ctc/layouts/ru_jcuken_default.json --run-name phaseP-ru-v2full \
    --batch 256 --lr 3e-3 --weight-decay 0.01 --warmup 1000 --ch 80 \
    --embed-hid 96 --feat-version 1 --block resbn --dilations 1,2,4,8 \
    --t-out 32 --total-steps 94000 --val-every 3000 --affine-sampler coupled \
    --layout-alt-p 0.0 --beam-val-rows 0 --patience 40 --seed 1234 --workers 0
python3 ctc/export_onnx.py --ckpt ckpt/phaseP-ru-v2full/best.pt \
    --out ckpt/phaseP-ru-v2full/ctc_swipe_encoder.onnx \
    --layout ctc/layouts/ru_jcuken_default.json \
    --parity-features cache_ru_v2full/val.npz
python3 ctc/eval_script.py --code ru --preset ckdt \
    --onnx ckpt/phaseP-ru-v2full/ctc_swipe_encoder.onnx \
    --probe data/yandex_val10k.jsonl
```

Committed evidence: `ctc/phase_p_gates.json` (the full G1–G4 record),
`ctc/phase_p_enen.json` (the A1 control), `ctc/phaseP_G5_*.json` (the four ship-gate
decodes), `ctc/phaseP_holdout_*.json` (the calibration re-run),
`ctc/phase_p_scripts.json` (the five-script battery). Runtime intermediates under
`~/ctc-train/synth_gap/` are regenerable from the committed scripts; seed 1234
throughout, donor-draw seed 20260819.

### 6.1 The v2 artifact registry

Generation-2 artifacts **supersede** Phase O's for deployment; both generations
stay in the registry with their tiers, because the v1 bytes are the ones every
prior number in `MODELS_TABLE.md` was measured on. Every graph is the standard
1,142,727-byte resbn80 export (the alphabet is data, not architecture).

| file | bytes | sha256 |
|---|---|---|
| `ru_synth_v2_ch80.onnx` | 1,142,727 | `763190f9bc9854a3183f10d7dba7d8e1de1c101812b5958ee9bdbb403b93089b` |
| `ru_synth_v2_ch80_fp16w.onnx` | 589,406 | `9004befb6ff07b744c65d3c13481539e758ebe10d4f47cbeffe68d39d12b0e52` |
| `ru_synth_v2_ch80_fp16w_golden.json` | 160,282 | `a5ed2b9f62843d085779f5ab7457e6608f5c47e8994c224146ebdaf32fcdb82d` |
| `el_synth_v2_ch80.onnx` | 1,142,727 | `ada06a627074d120fed77d128920e073270cf1caa5afeea285ff421945a99432` |
| `el_synth_v2_ch80_fp16w.onnx` | 589,406 | `a65151793bd78e0399b34dc2dede3da6a4a2a4d9ad48190a62cfdff75a770495` |
| `el_synth_v2_ch80_fp16w_golden.json` | 144,332 | `ee34b42260bbe53acea15c408c3c20ce73d1db6574c38436801b397985569262` |
| `uk_synth_v2_ch80.onnx` | 1,142,727 | `a9cf7ff49d1ac35a3e33921b4b6e74ce42a54bce50229101f410ee550f5529c8` |
| `uk_synth_v2_ch80_fp16w.onnx` | 589,406 | `e7941d310c9075adf97d31a14ae6da8d4e42282cfd5154d7e778414fd3679cbf` |
| `uk_synth_v2_ch80_fp16w_golden.json` | 155,680 | `6826e133d74a9551a8f41a8718b5d39fa2e19430adc93ae307705da346081db3` |
| `bg_synth_v2_ch80.onnx` | 1,142,727 | `b92cf65ff546db64af290c8fd2de04018977838d5c2d9b0eaf1ba4322090c82b` |
| `bg_synth_v2_ch80_fp16w.onnx` | 589,406 | `56d51194e22ee112dec868abaf9ddb91059d0dc752158552711db44f80935d4a` |
| `bg_synth_v2_ch80_fp16w_golden.json` | 154,872 | `c2fa0e387c15e16ad5c633542ee112089b815c87f1b06b674292d4dbe3c3deeb` |
| `mk_synth_v2_ch80.onnx` | 1,142,727 | `f2cd4cfa159039c8fe6d2326cb9377b0e2bf5afd4df07a5212716547b2a49e42` |
| `mk_synth_v2_ch80_fp16w.onnx` | 589,406 | `a3c96b5f98cbb66aad7a291ce8ecbc147d228085d4ad6eb5b27143402dca209c` |
| `mk_synth_v2_ch80_fp16w_golden.json` | 160,726 | `d1b25a309145feeca14be7356a9f2ac304ee0e4f605665de40933189885a67aa` |
| `he_synth_v2_ch80.onnx` (**flagged**, §5) | 1,142,727 | `863c5f4df524893141d34089ca6e12b248bac17af82cc5d651cb603c7b3b98bb` |
| `he_synth_v2_ch80_fp16w.onnx` (**flagged**) | 589,406 | `943ab4e36297c686a2af00a5bd5ec622a9671b9d2258b49297513aefe85f0c26` |
| `he_synth_v2_ch80_fp16w_golden.json` | 140,434 | `c0ff01294eccfefc54040b1ff8cf9d8266dd645f65305534bf0ab588d4f9e4b0` |

**§8.4 adds a third generation for the five non-ru scripts**
(`{el,uk,bg,mk,he}_synth_v2full_ch80*`), which supersedes the v2 rows above for
deployment. The v2 rows are **not** deleted — they are the bytes §5 was measured
on. ru is unaffected: `ru_synth_v2_ch80` is already the full-pool arm.

Every fixture is frozen at **γ 1.05 / λ 2.0 / β 0.2 / 0.3734 / 0.9882** on the
script's own lexicon weights, 10 cases each (5 pure-featurizer branch probes, 1
word-path featurizer case, 4 model-backed beam cases) — the same shape as Phase
O's, so the app-side `CtcParityTest` row needs a model/fixture swap and nothing
else. The alphabet strings, projection rules and per-script wiring of PHASE_O
§3.2–3.4 are **unchanged**: v2 changes the training distribution, not the
contract.

---

## 7. The ledger — what Phase P did NOT establish

1. **Four of the six scripts still have no accuracy measurement of any kind.**
   Only ru is real-validated. The v2 holdouts are v2-generated.
2. **UCL₉₅ ≤ 0.60 is not met** (0.7467 MLP speed / 0.8331 GBM₁₇). v2 is
   distinguishable from real Russian, and the en→en control says ≈0.19 of that is
   the donor bank's language, which no generator change can reach.
3. **The G5 ≤3 corollary is missed** at 85.77 against 86.4, non-significantly,
   with the cause attributed to the donor side rather than to v2.
4. **The register residual is open**: the wordfreq draw's ≤3 mass is 26.8 %
   against real usage's 35.6 %, and closing it would consume the validator.
5. **Single seed (1234) everywhere**, as in every prior phase. The campaign's own
   resolution floor is ~1 pt, which the +2.31 clears and the −0.70 stratum
   regression does not.
6. **Fix D is unmeasured**, by construction. Provenance is recorded so a later
   phase can use it; nothing here says it is worth anything.
7. **The 23-metric GBM still reads 0.875.** Cornering and the donor population
   are what is left, in that order, and §2.5 says only one of them is reachable.
8. **λ was not re-tuned** for a strong-emission model, and §4.2 argues it
   probably wants re-tuning. Deliberately left, to avoid spending the validator.
9. **The donor footing for the five is a transfer argument, not a measurement.**
   §8 puts them on the full pool and measures −0.09 pooled, p = 0.44, on 50,000
   paired holdout rows; the +0.86 that justifies the change is ru's, on real
   data. Their own probe cannot see it, which §8.2 argues is a fact about the
   probe.

---

## 8. P6 — the five scripts on the full donor pool

The one item §7 left as a registered recommendation rather than a finding: §4.1
priced Phase O's 90/10 donor split at **−1.69 real top-1** on Russian, and §5
left the five corpus-less scripts riding it. P6 puts them on
`--train-donor-side all`, the footing the shipped ru arm uses, and changes
**nothing else** — v2 stages `a,c,b,s5`, 1,000,000 train rows, the Phase-O
recipe verbatim (`resbn:80`, dil 1,2,4,8, embed_hid 96, feat_v1, 94,000 steps,
batch 256, lr 3e-3, wd 0.01, warmup 1,000, coupled affine sampler, no
layout-alt, greedy selection, patience 40, seed 1234, `--workers 0`), the same
CKDT preset, no λ sweep, no new lever. The donor side goes **904,155 →
1,004,617** traces, exactly as it did for ru.

**The comparison is exactly paired, and that is a fact rather than an
assumption.** `script_synth`'s `side_of` maps the *holdout* split to the
reserved donor half whatever `--train-donor-side` says, and each split's RNG is
seeded independently, so the flag cannot touch `holdout.npz`.
`phaseP6_assert_holdout.py` asserts it on all five: `features`, `targets`,
`target_lengths`, `words`, `donor_row`, `donor_group` and `drawn_duration_ms`
**bit-identical** to the P4 caches, 10,000 rows each. Every P4 and P6 number
below is therefore read off the *same rows*, and the McNemar cells are real
discordant pairs. As a free by-product, re-decoding the five P4 models on those
rows reproduces `phase_p_scripts.json` **to the last digit** (90.69 / 87.97 /
82.26 / 89.02 / 77.00), and the two English zero-shot controls reproduce to the
last digit as well.

### 8.1 The result: a null, on 50,000 paired rows

| script | in-dict t1, P4 → **P6** | Δ (exact McNemar, n = 10,000) | greedy, P4 → P6 | Δ greedy | ≤3 | ≥4 | vs ch192 EN | vs ch80 EN | permuted |
|---|---|---|---|---|---|---|---|---|---|
| **el** Greek | 90.69 → **90.78** | **+0.09** (b/c 209/218, p 0.70) | 69.01 → 69.16 | +0.15 (p 0.65) | 95.79 → 95.99 | 86.70 → 86.70 | +6.02 → **+6.11** | **+7.01** | 0.00 |
| **uk** Ukrainian | 87.97 → **87.67** | **−0.30** (302/272, p 0.23) | 55.09 → 55.28 | +0.19 (p 0.60) | 92.73 → 92.40 | 85.88 → 85.59 | +5.39 → **+5.09** | **+6.92** | 0.02 |
| **bg** Bulgarian | 82.26 → **82.52** | **+0.26** (360/386, p 0.36) | 55.97 → 55.65 | −0.32 (p 0.37) | 85.64 → 85.64 | 80.12 → 80.54 | +5.21 → **+5.47** | **+7.55** | 0.00 |
| **mk** Macedonian | 89.02 → **88.68** | **−0.34** (255/221, p 0.13) | 64.56 → 64.21 | −0.35 (p 0.29) | 93.52 → 93.18 | 85.83 → 85.49 | +5.57 → **+5.23** | **+6.32** | 0.01 |
| **he** Hebrew | 77.00 → **76.86** | **−0.14** (395/381, p 0.64) | 56.88 → **57.72** | **+0.84 (p 0.015)** | 85.83 → 85.34 | 71.47 → 71.55 | +8.06 → **+7.92** | **+10.51** | 0.01 |

| pooled over the five (not one model — five models on five probes) | n | b/c | Δ | p |
|---|---|---|---|---|
| **in-dict t1** | 50,000 | 1,521 / 1,478 | **−0.086** | **0.443** |
| greedy | 50,000 | 2,740 / 2,791 | +0.102 | 0.501 |
| ≤3 stratum | 19,325 | 383 / 349 | −0.176 | 0.223 |
| ≥4 stratum | 30,675 | 1,138 / 1,129 | −0.029 | 0.867 |

Not one per-script top-1 delta is significant; three of five are negative; the
pooled estimate is **−0.09 points with p = 0.44**. The only cell that moves is
he's greedy, +0.84 (p = 0.015), and a single nominally-significant cell out of
twenty is what a null looks like. t3/t5 move by the same nothing (el 96.34 →
96.63 / 97.24 → 97.40, uk 94.76 → 94.68 / 96.02 → 95.95, bg 94.25 → 94.03 /
96.15 → 96.25, mk 95.86 → 95.78 / 96.87 → 96.85, he 90.16 → 90.29 / 93.27 →
93.36), the ch192 margins stay in the +5.1 … +7.9 band that §5's sign flip
established, and the permuted-geometry falsification still collapses every model
to 0.00–0.02.

### 8.2 What the null does and does not mean

**It is not evidence that the donor pool does not matter.** It is evidence that
*this probe cannot see whether it matters*, and P6 makes that case stronger than
§5.1 could, because the P6 arm holds two advantages at once and still reads
flat:

1. it trains on 11 % more donors, the change §4.1 measured as worth +0.86 real
   top-1 on ru (p = 0.0023) and whose absence costs −1.69 on the v1 arm
   (p = 5.2e-07);
2. `--train-donor-side all` puts the holdout's reserved donor half **inside the
   training pool**, so the P6 holdout is no longer donor-disjoint and its
   numbers carry whatever a seen-donor advantage is worth here — an advantage
   the P4 arm did not have.

Both point the same way, both should inflate P6, and the measured pooled effect
is −0.09. The clean reading is the one Phase O and §5.1 already argued from two
other directions: a synthesis holdout measures *fit to the generator*, and the
donor pool's contribution to real-swipe accuracy is not in its field of view.
This is now the third independent demonstration — capacity (Phase O), λ
(Phase O), and donor footing (here) — and the first where the real-probe answer
for the *identical* change is known to be non-zero and the holdout still reads
zero.

**Why the P6 bytes are promoted anyway, stated as the transfer argument it is.**
Nothing in §8.1 argues for shipping P6 on its own evidence. The argument is
ru's: on the one script with a real corpus, this exact flag on this exact
generator, donor bank, architecture and recipe is worth **+0.86 real top-1**,
and the five scripts differ from ru only in alphabet, layout and lexicon. The
holdouts add the one thing they *can* establish — that the change costs nothing
measurable on the generator's own distribution, in either the ≤3 or the ≥4
stratum. Ship on the real measurement, not on the holdout; and if a real corpus
ever appears for any of the five, this is one of the first things it should
re-read.

### 8.3 Export gates — and the he flag retires for the new bytes

Every fp32 export clears at the **default 1e-3** tolerance with **100/100**
argmax agreement on the sliced contract view against real traces on the real
layout. **he needed no relaxation this time**: its sliced residue reads
**4.04e-04**, back inside the historical 0.8e-4…7.6e-4 envelope, against the
1.16e-03 that forced `--parity-tol 2e-3` in P4. The P4 `he_synth_v2_ch80` bytes
keep their flag — the exceedance was real and is a property of that checkpoint
— and the P6 bytes do not inherit it.

| script | BN fold (sliced) | fp32 vs torch (sliced) | argmax | parity tol | fp16w vs fp32 (white noise) | argmax | fp16w decode cost |
|---|---|---|---|---|---|---|---|
| el | 1.49e-04 | 1.04e-04 | **100/100** | 1e-3 | 5.49e-02 | 96/100 | 90.78 → 90.76 (−0.02) |
| uk | 1.16e-04 | 1.11e-04 | **100/100** | 1e-3 | 6.11e-02 | 97/100 | 87.67 → 87.68 (+0.01) |
| bg | 4.20e-04 | 7.78e-04 | **100/100** | 1e-3 | 1.15e-01 | 97/100 | 82.52 → 82.53 (+0.01) |
| mk | 1.79e-04 | 1.35e-04 | **100/100** | 1e-3 | 1.05e-01 | 97/100 | 88.68 → 88.69 (+0.01) |
| he | 3.20e-04 | **4.04e-04** | **100/100** | **1e-3** | 4.08e-02 | **100/100** | 76.86 → 76.87 (+0.01) |

The fp16w decode cost is ≤ 0.02 t1 everywhere and is *positive* on four of five,
which is the same noise floor P4 reported from the other side.

### 8.4 The P6 artifact registry

Generation-3 bytes for the five non-ru scripts. They **supersede** the
`*_synth_v2_ch80*` set for deployment; that set stays in the registry because
§5 was measured on it, exactly as the v1 set stays because Phase O was. ru is
absent by design — `ru_synth_v2_ch80` is already trained on the full pool
(§4.1), so P6 has nothing to change and its hash is untouched. Every graph is
the standard 1,142,727-byte resbn80 export; every fp16w is 589,406 B.

| file | bytes | sha256 |
|---|---|---|
| `el_synth_v2full_ch80.onnx` | 1,142,727 | `b5b94e541d5a5ab50bd24d084f43bf0e0d59210c054b788c4b771f6383c09fb7` |
| `el_synth_v2full_ch80_fp16w.onnx` | 589,406 | `6b82b4db9eb8bad9f9c075e5fcdf99a6a0f7fac5a076aebf647ee8f79b14eece` |
| `el_synth_v2full_ch80_fp16w_golden.json` | 144,407 | `d35b14e010512642765cae048fab2c2280c12c3a499f625d6f81e1a2bdcceb30` |
| `uk_synth_v2full_ch80.onnx` | 1,142,727 | `b39ae11291fbc4417b2d4036beae659204cb135d619508a4366bb25d3623ed09` |
| `uk_synth_v2full_ch80_fp16w.onnx` | 589,406 | `e3e6ff1cce3599c3d8299480bc6786cf1fb87f8970ca5d349dea3055bcdec50b` |
| `uk_synth_v2full_ch80_fp16w_golden.json` | 155,663 | `5afe7fd42cff3d81ab388872e23ca2361df2146dd6a8e12ef20cf54dd3746be8` |
| `bg_synth_v2full_ch80.onnx` | 1,142,727 | `84a2ce16ad0c0d43994425fd81d568114c44de8e203b18d6bc2521aaf28abbaa` |
| `bg_synth_v2full_ch80_fp16w.onnx` | 589,406 | `0345f222ca3e845b01a4f7d96ca4756c796cd38fbbe5eeb772bcc7da8aad558b` |
| `bg_synth_v2full_ch80_fp16w_golden.json` | 154,670 | `028b5ea8e2ebaccd978f84b97cd16dfb5eb15026bfb239fccf95f9fac65e1839` |
| `mk_synth_v2full_ch80.onnx` | 1,142,727 | `55dafbad9a2d4d2786de403db6ddb1c5775d00d53f1086edd79259cfea5c352f` |
| `mk_synth_v2full_ch80_fp16w.onnx` | 589,406 | `82d8eda4ee7739178187aab4ac5086ba2eda681e9e47ae413f6b6019a6dca57d` |
| `mk_synth_v2full_ch80_fp16w_golden.json` | 160,122 | `6000f88ba16ce742b13e94fae3818404999aefc494c70a8ddf3e8a916ec7cf74` |
| `he_synth_v2full_ch80.onnx` | 1,142,727 | `13a968d8114ae8f3c32a0eaa25af8943cc7ac6bd55e09f7f14289bca21f6a0e8` |
| `he_synth_v2full_ch80_fp16w.onnx` | 589,406 | `0835d3f575de5378ba78829b9249c22e81e8f7231e414c15682d3f33f9b45ffa` |
| `he_synth_v2full_ch80_fp16w_golden.json` | 140,247 | `adb5ea334fdff34fff378ce1edcedb9e2d6707b115c6c3a6649a5c28e28b7ae5` |

Fixtures are frozen at the same **γ 1.05 / λ 2.0 / β 0.2 / 0.3734 / 0.9882**, 10
cases each, same shape as P4's — the app-side row is a model-and-fixture swap
and nothing else. Alphabet strings, projection rules and the per-script wiring
of `PHASE_O.md` §3.2–3.4 remain unchanged.

### 8.5 Reproduction

Every driver is committed under `ctc/`, unlike Phase O's, which stayed in the
workdir:

```bash
# regenerate on the full donor pool (the only changed flag)
ctc/phaseP6_gen.sh el uk bg mk he                    # --train-donor-side all
python3 ctc/phaseP6_assert_holdout.py el uk bg mk he # must PASS before anything
ctc/phaseP6_train.sh el uk bg mk he                  # Phase-O recipe verbatim
ctc/phaseP6_eval.sh el uk bg mk he                   # exports, decodes, controls
ctc/phaseP6_paired.sh el uk bg mk he                 # P4 dumps on the same rows
python3 ctc/phaseP6_paired.py el uk bg mk he
python3 ctc/phaseP6_collect.py                       # -> phase_p6_scripts.json
ctc/phaseP6_artifacts.sh el                          # …and uk/bg/mk/he
```

Committed evidence: `ctc/phase_p6_scripts.json` — both footings per script, the
paired cells, the pooled test and the export gates, all parsed from the run's
own logs rather than transcribed by eye.

### 8.6 What P6 did NOT establish

1. **That the full donor pool helps these five.** It does not show that; it
   shows the probe cannot tell. The reason to believe it does is ru's real
   probe, and that reason is transfer, not measurement.
2. **Anything about real Greek, Ukrainian, Bulgarian, Macedonian or Hebrew
   swiping.** §5.1 carries over word for word: a v2 holdout is generated by v2,
   and four of the six scripts still have no accuracy measurement of any kind.
3. **A seed story.** Single seed 1234, as everywhere in this campaign, and the
   pooled ±0.09 is well inside the ~1 pt resolution floor a seed change would
   move things by.
4. **Anything about λ.** Not swept, not touched; §4.2's argument that λ 2.0 was
   tuned against a weak-emission model is still open and still waiting on a
   second real corpus.
