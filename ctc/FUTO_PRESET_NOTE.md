# Note for the FUTO keyboard team — the published swipe scoring presets appear substantially mis-tuned

**Date:** 2026-08-09 · **Concerns:** `scoring.json` in `futo-org/futo-swipe` (HF), as
consumed by the `swipe-library` beam (`git_commit 86b375fbc0ad76fd6cc421b09f28a110c4e98367`)

## Summary

While benchmarking against your published swipe models we found that the
`encoderOnly` and `encoderDecoder` presets shipped in `scoring.json` sit far from
the optimum for **your own models' emissions** on large human-swipe evaluations. A
straightforward re-sweep of the five scoring parameters, tuned on a validation
split and confirmed on held-out rows, is worth about **+2 pt top-1 to the
encoder+decoder configuration** and about **+7 pt to the encoder-only
configuration**, with zero model changes — the gain is entirely in the beam's
final-scoring constants. The dominant issue is `lambda`, the word-frequency
weight: the published values (0.0176 / 0.0134) are roughly **two orders of
magnitude below optimal** for a log-frequency lexicon term, which effectively
disables the lexicon prior. A secondary consequence is that the published presets
substantially overstate the `magic_macaw` decoder's contribution relative to what
your own paper reports (details below).

We report this because we hit the *identical* failure mode in our own project
first and only found it by accident, so it seemed worth passing on. For context:
we build a from-scratch swipe engine trained only on the MIT-licensed
`swipe.futo.org` corpus, which is why we were measuring your models at all. Your
weights were run for **benchmarking only** under the FUTO Model Weights License
1.0; no output of any FUTO model entered any training loop.

## What we measured

All numbers below are **your models** (`honorable_sturgeon` encoder,
`magic_macaw` decoder, both sha256-verified against the HF repo and their own
`metadata.json`), decoded by a Python port of the `swipe-library` beam that
reproduces our previously committed published-preset numbers **to the digit**
(validation of the port; see Reproducibility). Scoring follows the library's
final score `s_ctc / L^gamma + beta·L + lambda·log_freq` with pruning
`s_ctc / max(d,1)^gammaPrune + betaPrune·d`; beam width 100, top-k 8,
out-of-vocabulary targets counted as misses.

Data: two disjoint splits drawn from the `swipe.futo.org` EN corpus — a
9,918-row dev split and a 2,400-row test split. The sweep tuned on the first
4,959 dev rows, confirmed on the untouched 4,959, and the test split was decoded
once at the resulting preset.

### Encoder + decoder ("ceiling", published `encoderDecoder` preset γ 0.5949, λ 0.0134, β 0.7271, γp 0.1902, βp 1.2727)

Swept optimum (interior on all five axes; grids G2–G4 agree to 0.00 pt):
**γ 1.15, λ 1.3, β 0.2, γp 0.3734, βp 0.7**.

| split | preset | top-1 | top-3 | top-5 | ≤3-char t1 | 4+-char t1 |
|---|---|---|---|---|---|---|
| dev 9,918 | published | 85.54 | 91.52 | 92.78 | 89.29 | 83.60 |
| dev 9,918 | val-tuned | **87.48** | 92.31 | 93.03 | 89.76 | 86.29 |
| dev Δ | | **+1.94** | +0.79 | +0.25 | +0.47 | +2.69 |
| test 2,400 | published | 84.92 | 91.38 | 92.42 | 89.94 | 82.33 |
| test 2,400 | val-tuned | **87.12** | 92.29 | 92.96 | 89.94 | 85.68 |
| test Δ | | **+2.20** | +0.91 | +0.54 | 0.00 | +3.35 |

The gain is not sweep overfit: on the dev split it is +1.89 on the 4,959 rows the
sweep fitted and **+1.97 on the 4,959 rows it never saw**, and it transfers to the
untouched test split at +2.20. The entire test gain lands on words of 4+
characters.

### Encoder only ("floor", published `encoderOnly` preset γ 0.4056, λ 0.0176, β 0.9866, γp 0.4234, βp 1.0382)

Swept optimum: **γ 0.35, λ 4.8, β 1.6, γp 0.05, βp 1.4** (interior after four
grid widenings).

| dev 9,918 | top-1 | top-3 | top-5 | ≤3 | 4+ |
|---|---|---|---|---|---|
| encoder-only, published | 78.59 | 88.15 | 90.24 | 81.35 | 77.16 |
| encoder-only, val-tuned | **85.97** | 91.18 | 92.12 | 89.11 | 84.35 |
| Δ | **+7.38** | +3.03 | +1.88 | +7.76 | +7.19 |
| *(reference)* encoder+decoder, **published** | 85.54 | 91.52 | 92.78 | 89.29 | 83.60 |

The striking row is the last one: **the val-tuned encoder alone (85.97 top-1)
beats the published-preset encoder+decoder stack (85.54)** on this data. On the
test split the same holds directionally (tuned encoder-only 85.79 vs published
encoder+decoder 84.92).

### Consequence: the decoder's measured contribution at matched tuning

At the published presets, `magic_macaw` appears to be worth **+5.88 pt** top-1
over the encoder-only beam on our test split — far more than the **+0.55–0.76 pt**
your paper reports for it. With each configuration at its own val-tuned preset,
the decoder's contribution is **+1.51 pt** (dev) / **+1.33 pt** (test) — much
closer to your paper's figure. Most of what looks like decoder gain at the
published presets is the encoder-only preset being badly matched to the beam
(the decoder's refined 27-class posteriors happen to sit closer to what the
published scoring constants expect, so it masks the mis-tuning). If you have
internal decompositions attributing large gains to the decoder at these presets,
they may be worth re-checking at matched tuning.

## Why a narrow re-check will not see this (we know, because it fooled us)

A sweep at the published grid width finds nothing: on your emissions, a grid
spanning γ 0.45–0.74, β 0.63–0.83, λ 0.007–0.028 returns a boundary winner worth
+0.14 on the tuning half and **−0.11 on the holdout half** — pure noise. The
optimum (γ ≈ 1.15, λ ≈ 1.3, β ≈ 0.2) lies outside every axis of that box.

We report this with some humility because our own project made exactly this
mistake: every scoring sweep we had ever run spanned γ ∈ [0.30, 0.51],
β ∈ [0.89, 1.08], λ ≤ 0.026 — all centered on your published values — and we
published a "+0.21 pt maximum headroom" bound on that basis. The true gain for
our model, once the grid was widened until the winner was interior, was +4.25 pt
on untouched rows; the bound had understated it ~20×. The procedural fix is
simple and is what produced every number above: **reject any winner that lands on
a grid edge, and keep widening until the optimum is interior.**

Two observations suggest the mis-tuning is a property of the beam + lexicon +
scoring form rather than of any particular model: (a) the wide-grid optimum for
your emissions (γ 1.05–1.15, λ 1.1–1.3, β 0.2) essentially coincides with the
optimum we independently derived for our own, differently-trained model; (b) as a
control, running your engine at *our* model's tuned preset scores 87.21 top-1 on
the test split — marginally above its own swept optimum — so the result is not
sensitive to the exact winner chosen. The prune parameters, by contrast, are
close to right as published (`gammaPrune` is near-optimal and collapses above
~1.0); the score-side constants, above all λ, are where the headroom is.

## Reproducibility

- **Artifacts.** `hf download futo-org/futo-swipe`:
  `honorable_sturgeon/model_fp32.pte` (2,649,856 B, sha256 `725242ba…`) and
  `magic_macaw/model_fp32.pte` (1,247,468 B, sha256 `01eaf16a…`), both matching
  the `file_hashes` in their own `metadata.json`, `git_commit 86b375fb…`,
  export 2026-04-20. Layout `en_qwerty.json` from
  `gitlab.futo.org/keyboard/swipe-library`.
- **Lexicon.** Your `en_wordlist.combined` from
  `gitlab.futo.org/keyboard/latinime` (165,544 `word=` entries), normalized by
  stripping apostrophes/hyphens to a–z → **146,964-word trie**. OOV targets
  (86/2,400 test, 336/9,918 dev) counted as misses throughout.
- **Data.** EN traces from the MIT `swipe.futo.org` corpus; our fixed splits of
  9,918 (dev) and 2,400 (test) rows. Strata: ≤3-char n = 3,389 / 815, 4+ n =
  6,529 / 1,585.
- **Harness.** Python port of the `swipe-library` featurization
  (`resampler.cpp`) and single-stream Viterbi trie beam (`beam_search.cpp`),
  under ExecuTorch **1.2.0** (x86_64 manylinux wheel, torch 2.11.0+cpu) — the
  `.pte` files run natively on x86_64, no aarch64 device needed. The port's
  validity check: at the published presets it reproduces our previously
  committed test-split numbers **to every published digit** (encoder+decoder
  84.83 / 91.04 / 92.08, ≤3 89.57, 4+ 82.40 on the pre-fix 131,544-word trie),
  and a re-run of a shard is bit-identical, so the pipeline is deterministic.
- **Hardware.** Intel Core Ultra 9 275HX (24 threads), Ubuntu 22.04/WSL2.
  Single-thread throughput is ~24 traces/s for encoder+decoder (the Python beam,
  not the `.pte`, is the bottleneck); six full runs over both splits — 24,636
  decodes — took 5 min 43 s wall sharded. The seven sweep grids took ~28 min
  wall on 22 processes (the beam runs once per
  `(gammaPrune, betaPrune)` pair; γ/λ/β only enter the final score and are
  re-scored analytically, an identity verified to 1e-9 against the full scorer
  and to the digit against per-row decodes).
- **Protocol.** Tune on dev rows 0:4959, confirm on 4959:9918, decode test once;
  boundary-rejection with successive grid widenings until the winner is interior.

## Caveats

1. **Your production C++ beam was not in the loop** — this is a port, and your
   paper reports 92.54 / 93.30 top-1 on your full test split, well above our
   absolute levels (our 2,400 rows are a harder subset and the port is not
   bit-identical to production). But the beam implementation is common-mode
   between the published and tuned presets, so the *delta* is the robust finding;
   the absolute numbers are conservative floors on your true engine.
2. **`hungry_jellyfish` (the context LM) was not run**, so nothing here speaks to
   the two `scoring.json` presets that include it.
3. **The tuned winners are dataset- and lexicon-dependent** (EN, QWERTY, our
   splits, the 146,964-word trie). The dev→test transfer held here
   (+1.94 → +2.20), but you would want to re-derive the exact constants on your
   own eval sets — the grids, protocol, and winners above should make that a
   short exercise. The qualitative finding (λ two orders of magnitude low, the
   optimum far outside the published neighborhood) seems unlikely to be
   split-specific.
4. **Our eval rows are not held out from your training**: ~43 % of the unique
   holdout traces appear bit-exactly in the HF train split your models trained
   on. That inflates absolute accuracies in your favor and is irrelevant to the
   within-model preset comparison, but these numbers should not be read as
   generalization measurements.
5. **The encoder-only sweep may not be exhausted** — it was still creeping
   (+0.21, +0.23, +0.00) when it went interior, so its +7.38 could be slightly
   understated.

## If you want the full method

Everything above is written up, with per-grid tables, boundary reports, and the
exact commands, in this repository:

- `ctc/FAIR_REMATCH.md` — the sweep on your emissions, the tuned-preset test
  decode, and the revised decoder decomposition.
- `ctc/FUTO_WEIGHTS_VERIFICATION.md` — artifact provenance/hashes, the x86_64
  ExecuTorch environment, and the digit-exact reproduction of the published-preset
  numbers.
- `ctc/PHASE_E.md` §1 — how the identical mis-tuning was found on our own models,
  including the withdrawn narrow-grid "headroom bound".
- `ctc/sweep_scoring.py`, `ctc/futo_decoder_eval.py`,
  `ctc/futo_decoder_ceiling.py` — the sweep machinery and the beam/featurization
  port.

Happy to share per-trace outputs or the emission caches if useful. Thanks for
publishing the models, the corpus, and the library openly — it is what made this
kind of measurement possible in the first place.
