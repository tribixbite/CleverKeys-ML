# The fair rematch — both engines val-tuned, decoded on test-2400

**Date:** 2026-08-08 · **Companion to:** `FUTO_WEIGHTS_VERIFICATION.md` (which
established that FUTO's real weights run on this machine and reproduce the bar).

`AUDIT_FINAL.md` §6.1 and `AUDIT_PREDECODE.md` §5a both name the same unresolved
threat: **our test numbers were produced at a preset grid-fitted on val-9918, while
FUTO's ceiling was quoted at its published preset.** Both documents concluded the
rematch was impossible here — "no FUTO `.pte` encoder/decoder exists under
`/home/will`", "a fair rematch therefore cannot be run here", "it must be argued
from this val control". `RESULTS.md` carries the consequence as a standing
prohibition on any equal-footing claim.

The weights are now present and runnable. This document runs the rematch.

**The question is now answered. Both configurations still win all five metrics on
equal footing — but the margin shrinks by roughly two thirds, and for the shipping
candidate (ch 128) the win is no longer statistically resolvable at n = 2,400.**

---

## 0. Why this is legitimate, and what it cannot do

**Our models are frozen and were never re-decoded.** Every one of our numbers below
is re-read from the existing per-trace `test2400_e1.jsonl` dumps that the seal
produced and `AUDIT_FINAL.md` audited. No checkpoint was loaded, no beam was run on
our emissions, `eval_beam.py` and `eval_arms.py` were never invoked, and `seal.py`
was never bypassed. Nothing about our selection, training, or presets can change as
a result of this exercise — the E1 preset was fixed before the seal was spent and
remains exactly as adopted.

**Sweeping FUTO's preset can only make the bar harder.** The direction of the
change is known in advance: tuning a scoring preset on val cannot lower the
engine's accuracy, so this replaces the bar with a strictly stronger one. Running
our frozen numbers against a stronger bar is a test we can only fail, never game.
If we still clear it, the equal-footing question is answered; if we do not, that is
the finding and the "as registered" claim stays qualified exactly as it is today.

**License — unchanged and re-affirmed.** FUTO's weights were RUN for benchmarking
only. Their emissions, candidates, and predictions are benchmark artifacts held in
a scratch directory outside both repositories. **No FUTO output entered any
training loop, was saved as training data, or influenced any choice about our
models, presets, or checkpoints.** The sweep tunes *FUTO's own* scoring parameters
against *FUTO's own* emissions; nothing crosses over.

**What this still cannot resolve** is listed in §7 — chiefly that FUTO's published
stack has a third component (`hungry_jellyfish`, the context LM) that is not run
here, so the bar remains a floor on FUTO's full system.

---

## 1. Method — Phase E's own machinery, pointed at FUTO's emissions

The sweep reuses the ML repo's `sweep_scoring.py` **by import, not by
reimplementation**: `TraceCandidates`, `score_grid`, `strata`, `collect`,
`_init_worker` and `_raw_candidates` are the exact objects that produced the E1
preset. Only the emission source differs. The protocol is Phase E's:

- **Tune on `val[0:4959]`, confirm on the untouched `val[4959:9918]`,** report full
  val — the same three-way split, the same row ranges.
- **A winner that lands on a grid edge is not accepted.** Phase E needed five
  successive widenings before its optimum was interior; the same rule is applied
  here and forced three extra widenings on the floor (§3).
- **Beam width 100, top-k 8, OOV counted as a miss** — unchanged.
- The beam runs once per `(gammaPrune, betaPrune)` pair; `(gamma, beta, lambda)`
  enter only the final score and are re-scored analytically. This is Phase E's
  documented identity.

**Three independent checks that the fast path is exact:**

1. The vectorised scorer is asserted equal to `sweep_scoring.score_grid` on all five
   metrics to 1e-9, at two presets, at the start of every grid (printed as
   `[verify]` in every log).
2. The analytic path evaluated at FUTO's **published** preset on full val returns
   `85.54 / 91.52 / 92.78 / 89.29 / 83.60` — digit-identical to the independent
   full per-row decode in `FUTO_WEIGHTS_VERIFICATION.md` §4b.
3. On test-2400 at the **tuned** preset, the analytic path returns
   `87.12 / 92.29 / 92.96 / 89.94 / 85.68` and the real `futo_decoder_ceiling.py`
   per-row decode returns the same five numbers. Every headline test number below
   is from the **real decode**, not the analytic path.

Emissions were cached once per split per config (`[N,32,27]`, the exact array the
harness feeds the beam): 55 s for val ceiling, 39 s val floor, 17 s / 13 s test.
The seven grids cost ~28 min wall total on 22 processes.

---

## 2. The ceiling sweep — FUTO's optimum is interior, and it is worth ~2 pt

`encoder honorable_sturgeon + decoder magic_macaw`, STRIP trie (146,964 words — the
footing our own numbers used).

| grid | γ span | β span | λ span | winner (γ, λ, β, gp, bp) | edge? | holdout-half t1 | full-val t1 |
|---|---|---|---|---|---|---|---|
| G1 published-width | 0.45–0.74 | 0.63–0.83 | 0.007–0.028 | 0.45, 0.02, 0.63, .1902, 1.2727 | **γ, β** | 84.73 (−0.11) | 85.56 |
| G2 wide | 0.0–3.0 | 0.0–1.3 | 0.0–1.8 | 1.05, 1.1, 0.2, .3734, .9882 | none | 86.89 (+2.05) | 87.48 |
| G3 fine | 0.85–1.35 | 0.0–0.40 | 0.7–1.7 | 1.15, 1.3, 0.2, .3734, 0.7 | **bp** | 86.81 | 87.48 |
| **G4 fine, bp widened** | 0.95–1.30 | 0.05–0.30 | 0.9–1.5 | **1.15, 1.3, 0.2, .3734, 0.7** | **none** | **86.81 (+1.97)** | **87.48** |

G3 and G4 agree on the same interior winner and G2/G3/G4 agree to 0.00 pt on full
val, so the search is converged. **Adopted FUTO ceiling preset:
`gamma 1.15, lambda 1.3, beta 0.2, gammaPrune 0.3734, betaPrune 0.7`.**

Two things are worth pausing on. First, **the published-width grid finds nothing** —
it returns a boundary winner worth +0.14 on the tuning half and **−0.11 on the
holdout**, i.e. noise. This is the identical failure mode that produced our own
withdrawn "+0.21 pt maximum headroom" claim. Second, the wide grid's optimum for
FUTO's emissions is `γ 1.05, λ 1.1, β 0.2` — **our E1 preset, exactly**. The
mis-tuning was never specific to our model; the published preset is simply far from
optimal for this beam implementation on this data, for both engines.

### What tuning buys FUTO, versus what it bought us

| | t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|
| FUTO ceiling, published preset (full val) | 85.54 | 91.52 | 92.78 | 89.29 | 83.60 |
| **FUTO ceiling, val-tuned (full val)** | **87.48** | **92.31** | **93.03** | **89.76** | **86.29** |
| **Δ — the lever, for FUTO** | **+1.94** | +0.79 | +0.25 | +0.47 | **+2.69** |
| Δ — the same lever, for our ch192 (`RESULTS.md` §asymmetry) | +2.29 | +0.66 | +0.41 | +2.76 | +2.04 |

**The asymmetry was real and material.** FUTO's engine has +1.94 pt of preset
headroom against our +2.29 — the same order, not a rounding difference. Roughly
two thirds of our published test margin on t1 was an artifact of comparing a tuned
preset against an untuned one. The one place the lever is genuinely lopsided is the
≤3-char stratum: it moved our model +2.76 but moves FUTO only +0.47.

The gain generalises: it is +1.89 on the 4,959 rows fitted and **+1.97 on the 4,959
rows never seen**, larger on the holdout than on the tuning half — the opposite of
sweep overfit, exactly as Phase E observed for ours.

---

## 3. The floor sweep — the finding that revises the decomposition

`encoder honorable_sturgeon` only, FUTO Viterbi beam (config `beamB`; the sweep
machinery uses the Viterbi beam, which is also the beam our models were tuned
under, so `beamB` is the right encoder-only baseline here, not the logaddexp `A`).

Four grids were needed; the first three all landed on edges (λ and gammaPrune, then
γ/β/betaPrune, then λ/β), and each widening kept moving the number: full-val t1
85.53 → 85.74 → 85.97 → 85.97 (converged, interior).

**Adopted FUTO floor preset: `gamma 0.35, lambda 4.8, beta 1.6, gammaPrune 0.05,
betaPrune 1.4`.**

| | t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|
| FUTO floor, published preset (full val) | 78.59 | 88.15 | 90.24 | 81.35 | 77.16 |
| **FUTO floor, val-tuned (full val)** | **85.97** | **91.18** | **92.12** | **89.11** | **84.35** |
| Δ | **+7.38** | +3.03 | +1.88 | **+7.76** | +7.19 |
| *(for reference)* FUTO **ceiling** at its **published** preset | 85.54 | 91.52 | 92.78 | 89.29 | 83.60 |

**The val-tuned encoder alone (85.97) beats the published encoder+decoder ceiling
(85.54).** This materially revises `futo-decoder-eval-notes.md`'s per-lever
decomposition, which concluded that "the DECODER is where ALL the gain is" (+5.88 pt)
and the beam/scoring lever was ~neutral. That conclusion is **preset-conditional**.
Measured with each configuration tuned, `magic_macaw` is worth **+1.51 pt** on val
(87.48 − 85.97) and **+1.33 pt** on test (87.12 − 85.79), not +5.9. Most of what
looked like decoder gain was the published encoder-only preset being badly matched
to this beam — the refined 27-class posteriors happen to sit closer to what the
published scoring expects. The decoder is a real but modest lever, and the
0.55–0.76 pt the FUTO paper itself reports for it is much closer to our tuned
measurement than to our published-preset one.

---

## 4. The single test-2400 decode

One decode of **FUTO's engine** (not ours) at the val-tuned presets, via the real
`futo_decoder_ceiling.py` per-row path. All 2,400 rows, OOV = miss.

| engine / preset | trie | OOV | t1 | t3 | t5 | ≤3 (815) | 4+ (1585) |
|---|---|---|---|---|---|---|---|
| FUTO ceiling, published | STRIP | 86 | 84.92 | 91.38 | 92.42 | 89.94 | 82.33 |
| **FUTO ceiling, VAL-TUNED** | **STRIP** | **86** | **87.12** | **92.29** | **92.96** | **89.94** | **85.68** |
| Δ from tuning | | | **+2.20** | +0.91 | +0.54 | **0.00** | **+3.35** |
| FUTO ceiling, published | DROP | 99 | 84.83 | 91.04 | 92.08 | 89.57 | 82.40 |
| FUTO ceiling, VAL-TUNED | DROP | 99 | 86.46 | 91.58 | 92.21 | 88.96 | 85.17 |
| FUTO floor (beamB), published | STRIP | 86 | 79.08 | 88.50 | 90.33 | 81.84 | 77.67 |
| FUTO floor (beamB), VAL-TUNED | STRIP | 86 | 85.79 | 91.62 | 92.29 | 88.47 | 84.42 |
| *control:* FUTO ceiling @ **our** E1 preset | STRIP | 86 | 87.21 | 92.25 | 92.71 | 89.94 | 85.80 |

Tuning transfers from val to test almost exactly (+2.20 test vs +1.94 val on t1).
**The entire test gain is on long words** (4+ = +3.35); the ≤3 stratum does not move
at all (89.94 → 89.94). The DROP-trie column is supplied for continuity with the
older committed table only — our models were scored against the STRIP trie
(146,964 words, 86 OOV, confirmed in `AUDIT_FINAL.md` §1), so **STRIP is the matched
footing** and the headline comparison uses it.

The control row is worth noting: running FUTO's engine at *our exact E1 preset*
gives 87.21 — marginally **better** than its own swept optimum on t1. FUTO's
emissions and ours want essentially the same scoring parameters, so the comparison
below is not sensitive to which of the two tuned presets is used.

---

## 5. Equal footing — the verdict table

Both engines val-tuned, same 2,400 rows, same STRIP trie, same beam width, same
OOV-as-miss rule. Our numbers are re-read from the frozen dumps.

| engine | t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|
| **FUTO ceiling, val-tuned (the new bar)** | **87.12** | **92.29** | **92.96** | **89.94** | **85.68** |
| ours ch 192, seed-mean | 88.36 | 92.65 | 93.50 | 91.37 | 86.81 |
| ours ch 192, worst seed | 87.88 | 92.54 | 93.46 | 90.92 | 86.31 |
| ours ch 128 (ship candidate), seed-mean | 87.92 | 92.33 | 93.00 | 91.08 | 86.29 |
| ours ch 128, worst seed | 87.83 | 92.08 | 92.92 | 90.55 | 86.06 |
| ours `fast_resbn80`, seed-mean (config A) | 87.29 | 91.89 | 92.82 | 91.17 | 85.30 |
| ours `fast_resbn80`, worst seed | 86.75 | 91.42 | 92.62 | 90.80 | 84.67 |

`fast_resbn80` was test-validated by a separate session on 2026-08-08 (`RESULTS.md`
§"The second unsealing") at its config A — the AOSP STRIP 146,964 trie, the same
footing as everything else in this table. Its dumps were re-read, not re-decoded.
Its config B (app 98,081-word lexicon) is compared against a differently-measured
bar and is **not** covered here.

### Per-metric outcome, and how the margin moved

| metric | ch192 Δ vs **published** bar | ch192 Δ vs **val-tuned** bar | ch128 Δ vs published | ch128 Δ vs **val-tuned** | resbn80 Δ vs published | resbn80 Δ vs **val-tuned** |
|---|---|---|---|---|---|---|
| t1 | +3.53 | **+1.24 win** | +3.09 | **+0.79 win** | +2.46 | **+0.17 win** |
| t3 | +1.61 | **+0.36 win** | +1.29 | **+0.04 tie** | +0.85 | **−0.40 LOSS** |
| t5 | +1.42 | **+0.54 win** | +0.92 | **+0.04 tie** | +0.74 | **−0.14 LOSS** |
| ≤3 | +1.80 | **+1.43 win** | +1.51 | **+1.15 win** | +1.60 | **+1.23 win** |
| 4+ | +4.41 | **+1.14 win** | +3.89 | **+0.61 win** | +2.90 | **−0.38 LOSS** |

**`fast_resbn80` does not survive the rematch.** It clears all five bars against the
published preset and **fails three of five against the val-tuned one** — t3 −0.40,
t5 −0.14, 4+ −0.38 — keeping only t1 (+0.17, well inside noise) and ≤3 (+1.23). Its
paired McNemar against the val-tuned bar is unresolved on every seed and one seed is
**net negative** (s1234: 74 ours-only vs 83 FUTO-only, p = 0.52; s4321 +7, p = 0.62;
s7777 +14, p = 0.27). The 0.215 ms speed variant is, on equal footing, level with
FUTO's engine rather than ahead of it, and behind it on the deeper-list metrics.

Against the control (FUTO at our exact E1 preset) the picture is the same:
ch192 +1.15 / +0.40 / +0.79 / +1.43 / +1.01, ch128 +0.71 / +0.08 / +0.29 / +1.15 / +0.48.

**All five point estimates still favour us, for both configurations.** But ch 128's
t3 and t5 leads are **+0.04 pt — one trace in 2,400**. Those are ties in everything
but sign, and they should be written as ties.

### Paired significance — exact McNemar on top-1, per seed

FUTO's per-row output now exists, so the paired test `AUDIT_FINAL.md` §5 called
impossible can be run. It is far more sensitive than the unpaired SE used there
(~120–150 discordant pairs; the resolvable difference is ≈ 0.9–1.0 pt on t1).

| config | seed | ours-only | FUTO-only | net | p (exact, 2-sided) | |
|---|---|---|---|---|---|---|
| ch 192 | 1234 | 81 | 41 | +40 | **0.0004** | **resolved** |
| ch 192 | 4321 | 83 | 65 | +18 | 0.1621 | not resolved |
| ch 192 | 7777 | 84 | 53 | +31 | **0.0101** | **resolved** |
| ch 128 | 1234 | 73 | 51 | +22 | 0.0589 | not resolved |
| ch 128 | 4321 | 74 | 57 | +17 | 0.1619 | not resolved |
| ch 128 | 7777 | 73 | 55 | +18 | 0.1326 | not resolved |

By stratum, only ch192 seed 1234 resolves on all three slices (≤3 p=0.035, 4+
p=0.0055); seed 7777 resolves overall and on ≤3; seed 4321 resolves nothing.
**ch 128 resolves on no metric, on any seed.**

*(A majority-of-3-seeds pooling clears everything at p<0.05 — ch192 +2.00, ch128
+1.46 overall — but a 3-seed majority vote is not a shippable configuration and
overstates a single model. It is recorded here only so the number is not later
rediscovered and mistaken for the result.)*

### The honest statement

> On equal footing — both engines' scoring presets tuned by the same wide grid on
> the same val rows, decoded on the same sealed 2,400 rows against the same
> 146,964-word lexicon — **ch 192 leads FUTO's encoder+decoder engine on all five
> metrics, by +0.36 to +1.43 pt, and the top-1 lead is statistically resolved on
> two of three seeds. ch 128, the shipping candidate, leads on all five point
> estimates but by +0.04 pt on t3 and t5, and its lead is not statistically
> resolvable on any metric or any seed. `fast_resbn80` loses three of five.**
>
> The previously published margins (ch192 +3.53 t1, ch128 +3.09 t1, resbn80 +2.46)
> were inflated roughly threefold by the preset asymmetry. For ch 192 and ch 128
> the *ranking* survives the rematch and only the *size* of the lead does not. For
> `fast_resbn80` the ranking does not survive: its five-of-five pass was an artifact
> of the untuned bar.

The most durable result is the one the asymmetry never explained: **the ≤3-char
stratum**, where we lead by +1.43 / +1.15 and FUTO's preset lever is worth only
+0.47. That was the metric `AUDIT_FINAL.md` flagged as unresolved against the old
bar, and it is the metric that holds up best against the new one.

---

## 6. What changes in the record

- `RESULTS.md` §"The asymmetry" says the rematch "must be argued from this val
  control" and that "no second test decode may be spent on a fair rematch". The
  first clause is superseded — it was measured, not argued. The second was a rule
  about **our** seal, and it was not broken: no model of ours was decoded.
- `AUDIT_PREDECODE.md` §5a — "not testable on this machine … a fair rematch
  therefore cannot be run here" — is superseded on the facts. Its analysis of *why*
  the asymmetry mattered was correct, and its size estimate (~2.3 pt for us) was
  close: the net effect on the margin is ~2.3 pt on ch192 t1.
- `AUDIT_FINAL.md` §5's "a paired test is impossible — FUTO's per-row output is
  unavailable" is superseded; the paired test is in §5 above and it is stricter
  than the unpaired one it replaces.
- `futo-decoder-eval-notes.md`'s per-lever decomposition ("the DECODER is where ALL
  the gain is") is **preset-conditional** and should be read with §3 above.
- The prohibition in `RESULTS.md` §"The claim, verbatim as registered" on writing
  *that this model beats FUTO's decoder on equal footing* can now be replaced by
  the qualified statement in §5 — **for ch 192**. For **ch 128 it still stands**,
  because the ch128 lead does not resolve.

---

## 7. Caveats — all of which run against us

1. **FUTO's third model is not in the bar.** `hungry_jellyfish`, the context LM, is
   downloaded but unused, and `scoring.json` ships two presets that include it. The
   FUTO paper does not evaluate it, and our eval rows mostly lack the preceding-word
   context it consumes, so it is not straightforwardly runnable here — but the bar
   is a floor on FUTO's *full published stack*, not a ceiling.
2. **FUTO's floor sweep may not be exhausted.** The ceiling converged cleanly
   (interior, three grids agreeing). The floor needed four grids and was still
   creeping (+0.21, +0.23, +0.00) when it went interior; a further widening might
   find a little more.
3. **Our port is not FUTO's production C++.** The featurization and Viterbi beam are
   ports; FUTO's paper reports 93.30 on its own test split. Every FUTO number here
   remains a conservative estimate of its true engine.
4. **Our preset was tuned on val rows that overlap the seal by 7 traces** (0.29 %,
   `AUDIT_PREDECODE.md` §3f). FUTO's preset was now tuned on the same rows, so this
   contamination is symmetric — one of the few asymmetries this exercise removed.
5. **n = 2,400 is the binding constraint,** not seed variance (sd 0.04–0.54). The
   paired test resolves ≈ 1 pt on t1; three of the ten equal-footing comparisons sit
   below that.
6. This says nothing about latency, size, or on-device behaviour, where our 689 K
   / 0.455 ms ch 128 and FUTO's 939 K-parameter two-model stack are not comparable
   on accuracy alone.

---

## 8. Artifacts

Scratch tree `~/ctc-train/futo_verify/` (outside both repos; benchmark-only, never a
training input): `cache/{val,test}_{floor,ceiling}.npz` emission caches,
`sweep/*.json` all seven grids with their top-20 tables and boundary reports,
`out/tuned_*.jsonl` + `out/e1preset_*.jsonl` the test decodes, `futo_sweep.py`,
`rematch_table.py`, `strata_sig.py`, `cache_emissions.py`.

```bash
# ceiling sweep, converged grid (G4)
python futo_sweep.py --emissions cache/val_ceiling.npz --config ceiling \
  --rows ~/ctc-train/data/val_hwsfuto.jsonl --vocab ~/ctc-train/data/futo_en_wordlist.combined \
  --grid-gamma 0.95,1.00,1.05,1.10,1.15,1.20,1.25,1.30 --grid-beta 0.05,0.10,0.15,0.20,0.25,0.30 \
  --grid-lambda 0.9,1.0,1.1,1.2,1.3,1.4,1.5 \
  --grid-gamma-prune 0.25,0.30,0.3734,0.45 --grid-beta-prune 0.1,0.25,0.4,0.5,0.7,0.85,0.9882 \
  --sweep-rows 0:4959 --holdout-rows 4959:9918 --out sweep/ceil_g4.json

# the single test decode, real per-row harness, FUTO's val-tuned ceiling preset
python harness/futo_decoder_ceiling.py --encoder .../honorable_sturgeon/model_fp32.pte \
  --decoder .../magic_macaw/model_fp32.pte --layout en_qwerty.json \
  --vocab ~/ctc-train/data/futo_en_wordlist.combined --test ~/ctc-train/data/test_hwsfuto.jsonl \
  --config beamD --beam-width 100 --top-k 8 \
  --gamma 1.15 --lambda 1.3 --beta 0.2 --gamma-prune 0.3734 --beta-prune 0.7 \
  --out out/tuned_ceil_strip.jsonl
```
