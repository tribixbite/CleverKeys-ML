# Alternate layouts and languages — does the layout-agnostic claim hold?

The CTC encoders take the keyboard geometry as an **input** (`layout_keys [1,64,2]`
key centers + `layout_mask [1,64]`; emission column `c` is whatever key sits in slot
`c`) and were trained with slot-permutation + affine-jitter augmentation on
**en_qwerty only**. They are therefore layout-agnostic *by construction* — and were
validated on *nothing but* en_qwerty. This document supplies the missing evidence:
**real human swipes on five non-QWERTY layouts across four languages**, decoded
through the shipped artifact at the shipped preset.

Nothing here reads `test-2400`. Every number is either the (unsealed) en_qwerty
`val-9918` split or a FUTO `swipe-5` corpus that has never been part of any split.

## 0. Verdict

**Language-agnostic: yes. Layout-agnostic: only near QWERTY.**

Slot-permutation augmentation did *exactly* what it was designed to do and nothing
more. Scattering the keys into random slots of the 64 changes the gathered
emissions by **≤ 3.8e-6** — float32 noise, i.e. the model is *perfectly*
slot-invariant on every layout tested. But slot invariance is not layout
invariance, and the affine jitter that was supposed to supply the rest spans only
an axis-aligned scale/translate/mirror. Accuracy therefore decays monotonically
with how far a layout's key positions sit from QWERTY:

| layout | lang | mean key displacement vs qwerty | **t1** | greedy t1 |
|---|---|---|---|---|
| qwerty (control, val-9918) | en | 0.0000 | 91.11 | 72.83 |
| spanish | es | 0.0175 | **81.34** | 31.91 |
| qwertz | de | 0.0579 | **76.66** | 27.21 |
| azerty | fr | 0.1068 | **75.31** | 22.06 |
| german | de | 0.1071 | **72.08** | 15.14 |
| dvorak | en | **0.4313** | **63.04** | 11.64 |

Changing the *language* while holding geometry almost fixed (spanish, displacement
0.0175) costs ~10 pt of top-1. Changing the *geometry* while holding language *and
lexicon* fixed (dvorak, displacement 0.4313) costs ~28 pt. **Layout shift is the
failure mode; language shift is not.**

The greedy column is the honest one — it is the raw emission quality with no
lexicon to hide behind, and it falls from 72.8 % to 11.6 %. On dvorak the beam and
the 146,964-word English trie are doing essentially all of the work.

Against the shipped geometric engine the model wins or ties on four of five
layouts — often by a wide margin on top-3/top-5 — and **loses decisively on
dvorak**, the only layout whose geometry is genuinely novel.

And the ceiling is augmentation, not capacity: the 2.5×-smaller, 2.2×-faster
`fast_resbn80` is 0.47 pt *behind* on en_qwerty and 0.7–4.2 pt *ahead* on every
alt-layout, with its biggest margins on the hardest ones (§6). Extra parameters
bought QWERTY-specific memorization that actively hurts transfer.

## 1. What was actually run

| | |
|---|---|
| Encoder | `artifacts/ch128_s1234.onnx` (= `phaseE-E3b-hws3x` s1234, 689,282 params, the D1 ship artifact) |
| Preset | **E1** `gamma 1.05, lambda 1.1, beta 0.2, gammaPrune 0.3734, betaPrune 0.9882` |
| Beam | `futo_viterbi_beam`, width 100, top-k 8 — the same beam the Kotlin decoder is golden-parity-tested against |
| Harness | `eval_altlayout.py` (this repo), sharing `featurize` / `futo_viterbi_beam` / `LexTrie` / `Tally` with `eval_beam.py` |
| Corpora | `futo-org/swipe.futo.org` config `swipe-5` split `train` (MIT), `dual_finger = 0`, language-matched |
| Geometries | official FUTO `swipe-5/layouts/<layout>.json`, vendored to `ctc/layouts/` |
| Lexicons | en = the campaign's 146,964-word AOSP STRIP trie; fr/de/es = the app's bundled CKDT-v2 dictionaries |

### Corpus regeneration

`fetch_futo_multilayout.mjs` is vendored from the app repo's
`scripts/fetch_futo_multilayout_sample.mjs` with exactly one delta — the official
layout geometries are written to `ctc/layouts/` instead of the app repo's test
resources, because the app repo is a read-only reference here. Filters, row schema
and trace-cache paths are unchanged, so this harness and the app's Kotlin replay
read the **same** corpus files.

The five geometries fetched fresh are **byte-identical** (sha256) to the app repo's
committed `src/test/resources/layouts/futo_*.json`, so the geometry side of the
comparison against the geometric-engine anchors is exact.

Yield, reproduced locally:

| layout | lang | API rows | kept | dropped |
|---|---|---|---|---|
| dvorak | en | 2,809 | 2,535 | 2 few-points, 272 de-dup |
| azerty | fr | 2,542 | 2,291 | 44 few-points, 14 bad-word |
| qwertz | de | 1,402 | 1,356 | 1 few-points, 3 bad-word |
| german | de | 2,594 | 2,503 | 8 few-points, 4 bad-word |
| spanish | es | 2,029 | 1,883 | — |

The "API rows" column reproduces the fetch script's docstring counts exactly
(2809 / 2542 / 1402 / 2594 / 2029).

## 2. The frame mapping — established, not assumed

**This is the section that decides whether anything below is worth reading.** A
wrong coordinate frame does not crash; it produces plausible-looking garbage.

The corpus `x, y` are already normalized over the `[0,1]²` **letter area**, and the
FUTO layout JSONs give key centers in that same frame — the identical convention
`en_qwerty.json` uses for training. So the mapping is the **identity**, with no
rescale, no aspect correction, no origin shift.

That is a claim, so it is measured. For every trace, take the first and last path
point, find the nearest a-z key center, and ask whether it is the first / last
letter of the target word. A correct frame puts both fractions high; a wrong one
drives them toward chance (1/26 ≈ 0.038). Each corpus is also scored **against
QWERTY geometry** — a deliberately wrong frame — so the metric is shown to
discriminate rather than merely to be large.

```
corpus     geometry        n  start-hit   end-hit  start-d   end-d
qwerty     qwerty       2000      0.895     0.769   0.0686  0.0784   <- en_qwerty val reference
dvorak     dvorak       2535      0.793     0.973   0.1419  0.0341
           qwerty*      2535      0.127     0.054   0.4485  0.4178
azerty     azerty       2289      0.870     0.788   0.0514  0.0699
           qwerty*      2289      0.651     0.664   0.0875  0.0872
qwertz     qwertz       1338      0.907     0.763   0.0510  0.0817
           qwerty*      1338      0.879     0.756   0.0720  0.0885
german     german       2469      0.855     0.727   0.0561  0.0731
           qwerty*      2469      0.517     0.614   0.0989  0.0884
spanish    spanish      1882      0.913     0.761   0.0505  0.0776
           qwerty*      1882      0.810     0.737   0.0568  0.0814
* wrong-geometry falsification control
```

Every layout sits in the same band as the en_qwerty reference (start 0.79–0.91, end
0.73–0.97). **The frame is right.**

The falsification control behaves as it must, and how hard it collapses is itself
informative: it collapses completely for dvorak (0.127 / 0.054), substantially for
german (0.517) and azerty (0.651), and **barely at all** for qwertz (0.879) and
spanish (0.810). That is not a defect in the metric — it is the honest statement
that qwertz and spanish *are* QWERTY, geometrically:

| layout vs qwerty | mean a-z key displacement | max | keys moved > 0.02 | letter rows |
|---|---|---|---|---|
| spanish | **0.0175** | 0.0505 | 9 | 3 |
| qwertz | 0.0579 | 0.7527 | **2** (y↔z only) | 3 |
| azerty | 0.1068 | 0.6686 | 13 | 3 |
| german | 0.1071 | 0.7405 | 19 | 3 |
| dvorak | **0.4313** | 0.7908 | **26 (all)** | **4** |

This table is the experimental design. **dvorak** is a maximal geometry shift with
*zero* language shift and the *same* lexicon as the control — it isolates layout.
**spanish** is a near-zero geometry shift with a full language shift — it isolates
language. azerty and german move both.

## 3. Alphabet and lexicon policy

Emissions are a-z, as trained. A word is projected by NFD-decomposing, dropping
combining marks, then removing `'` and `-` (the STRIP convention the en trie already
uses): `café → cafe`, `niño → nino`, `don't → dont`. A word whose projection still
contains a non-a-z character has **no a-z spelling at all** — German `ß`, French
`œ`/`æ` — and is counted as **untypeable** rather than mangled into a false target.
The same projection is applied to the lexicon and to the corpus targets, so they
cannot disagree.

Lexicons are read straight from the app's bundled CKDT-v2 `*_enhanced.bin`
(`scripts/build_dictionary.py::write_v2_binary`: 48-byte header, then `uint16 len |
utf-8 | uint8 rank`, rank 0 = most frequent). `freq = 255 - rank` puts the CKDT rank
back on the same 1..255 magnitude scale as the AOSP `f=` field.

| lexicon | records | untypeable dropped | distinct a-z words |
|---|---|---|---|
| en (AOSP STRIP, the campaign trie) | — | — | **146,964** |
| fr (`fr_enhanced.bin`) | 40,000 | 31 (`œ`/`æ`) | 37,949 |
| de (`de_enhanced.bin`) | 40,000 | 0 | 39,594 |
| es (`es_enhanced.bin`) | 50,000 | 0 | 47,955 |

### A confound that must be stated up front

The E1 preset's `lambda = 1.1` was fitted against the AOSP trie's `log_freq`
distribution, whose spread is **sd 1.354**. The CKDT rank scale is far more
compressed:

| lexicon | log_freq min | median | max | **sd** |
|---|---|---|---|---|
| en_aosp | 0.000 | 3.912 | 5.403 | **1.354** |
| fr | 4.234 | 4.489 | 5.541 | **0.215** |
| de | 2.485 | 4.554 | 5.541 | **0.188** |
| es | 4.127 | 4.431 | 5.541 | **0.227** |

Only the *spread* of `lambda * log_freq` affects ranking (an additive constant
cancels), so at the same `lambda` the fr/de/es frequency prior carries roughly
**6-7× less ranking signal** than it does in English. Any fr/de/es deficit is
therefore partly a lexicon-scale artifact and not purely a language effect. §6
bounds that with a `lambda = 0` arm, which removes the frequency prior from *every*
layout and puts them all on identical footing.

## 4. Key-slot arms

Two arms are run per layout:

* **`az26`** — only the 26 a-z keys are given to the model (mask 26). This matches
  the training regime exactly: 26 active slots.
* **`full`** — the 26 a-z keys occupy slots 0..25 and the layout's *extra* letter
  keys (`'` on dvorak/azerty, `ä ö ü` on german, `ñ` on spanish) occupy slots 26+
  with mask true. The emission slice still reads columns 0..25 + blank, so the extra
  keys contribute nothing to the alphabet — they only tell the model "a key exists
  here", which is geometrically faithful to what the finger actually travelled over.
  qwertz has no extra letter keys, so its two arms are identical by construction.

## 5. Harness sanity control — the number that licenses everything else

Before any alt-layout figure is read, the harness must reproduce a published
en_qwerty result. `ch128_s1234` on **full val-9918** at the E1 preset, decoded
through `eval_beam.py` (the campaign harness, all rows in the denominator):

| | t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|
| published (`PHASE_F.md` §5/§14.1, `PHASE_E.md` E3b) | 88.02 | 92.27 | 93.03 | 91.12 | 86.41 |
| **reproduced here** | **88.02** | **92.27** | **93.03** | **91.12** | **86.41** |

Exact to the digit on all five figures. Greedy t1 71.24 %.

### The in-dict protocol, and its en_qwerty reference

The geometric-engine anchors this document compares against are **in-dict**: a
trace is decoded only if its target is in the layout's dictionary, and OOV rows are
excluded from the denominator rather than counted as failures. `eval_altlayout.py`
uses that same protocol, so its en_qwerty control differs from the table above:

| en_qwerty control, val-9918 | n | OOV | t1 | t3 | t5 | greedy |
|---|---|---|---|---|---|---|
| all-rows (`eval_beam.py`, the campaign number) | 9,918 | — | 88.02 | 92.27 | 93.03 | 71.24 |
| **in-dict (`eval_altlayout.py`, the comparator below)** | 9,582 | 336 (3.4 %) | **91.11** | **95.50** | **96.30** | **72.83** |

Both are the same model, preset, beam and trie; they differ only in whether the 336
OOV rows sit in the denominator. **91.11 is the number every alt-layout row below
should be read against.**

## 6. Results

`ch128_s1234`, E1 preset, beam 100, in-dict protocol, `az26` arm.

| layout | lang | lexicon | n | OOV | untypeable | **t1** | **t3** | **t5** | greedy |
|---|---|---|---|---|---|---|---|---|---|
| **qwerty** (control) | en | en 146,964 | 9,582 | 3.4 % | 0 | **91.11** | **95.50** | **96.30** | 72.83 |
| spanish | es | es 47,955 | 1,758 | 6.6 % | 1 (`ł`) | **81.34** | **93.91** | **95.68** | 31.91 |
| qwertz | de | de 39,594 | 1,187 | 11.1 % | 18 (`ß`) | **76.66** | **90.06** | **92.67** | 27.21 |
| azerty | fr | fr 37,949 | 2,090 | 8.7 % | 2 (`œ`) | **75.31** | **90.67** | **93.30** | 22.06 |
| german | de | de 39,594 | 2,199 | 10.8 % | 34 (`ß`) | **72.08** | **86.45** | **90.40** | 15.14 |
| dvorak | en | en 146,964 | 2,457 | 3.1 % | 0 | **63.04** | **73.71** | **75.17** | 11.64 |
| dvorak | en | *en 97,959 (app trie)* | 2,457 | 3.1 % | 0 | *60.93* | *72.61* | *74.77* | — |

### Versus the shipped geometric engine

⚠ **The anchors quoted in the brief are stale.** `docs/specs/geometric-swipe-engine.md`
§"Real-corpus replay — MULTI-LAYOUT" carries the pre-regeneration numbers, measured
against 25k fr/de dictionaries. `GeoRealCorpusMultiLayoutTest.kt:575-584` records a
**re-measurement after the dictionary regeneration to fr/de 40k + re-curated es 50k**
— which are exactly the dictionaries this evaluation uses. The current basis is the
correct comparator; the spec table was never updated.

| layout | geo t1/t3/t5 (current basis) | **CTC t1/t3/t5** | Δt1 | Δt3 | Δt5 |
|---|---|---|---|---|---|
| spanish | 73.9 / 86.6 / 89.8 | **81.34 / 93.91 / 95.68** | **+7.4** | **+7.3** | **+5.9** |
| german | 71.1 / 81.7 / 84.3 | **72.08 / 86.45 / 90.40** | **+1.0** | **+4.8** | **+6.1** |
| qwertz | 76.2 / 87.4 / 90.6 | **76.66 / 90.06 / 92.67** | **+0.5** | **+2.7** | **+2.1** |
| azerty | 76.9 / 89.9 / 93.7 | **75.31 / 90.67 / 93.30** | −1.6 | +0.8 | −0.4 |
| **dvorak** | 76.8 / 79.9 / 80.4 | **63.04 / 73.71 / 75.17** | **−13.8** | **−6.2** | **−5.2** |

(dvorak against the app's own 98k en trie — the exact lexicon the geo anchor used,
and near-identical coverage, 96.9 % vs 96.4 % — is worse still: −15.9 / −7.3 / −5.6.)

**On four of five layouts the CTC encoder matches or beats the shipped geometric
engine, with its largest wins on top-3/top-5. On dvorak it loses decisively.**

Note the tuning asymmetry (§9, first bullet): the geometric engine is at its own
shipped tuning while the CTC model is at a preset fitted on en_qwerty, so these
deltas understate the CTC side. That makes the four wins safe and the dvorak loss
the conservative reading of a real gap, not an artifact — but the *sizes* are
provisional until a per-layout sweep is run.

### The lexicon-scale confound does not change any of this

§3 warned that `lambda = 1.1` carries 6-7× less ranking signal on the compressed
CKDT scale than on the AOSP scale. Re-running everything at **`lambda = 0`** deletes
the frequency prior entirely, so every layout is scored identically:

| layout | t1 @ E1 (λ=1.1) | t1 @ λ=0 | Δ |
|---|---|---|---|
| qwerty (val[0:2000], in-dict n=1,928) | 91.55 | 89.32 | −2.2 |
| spanish | 81.34 | 73.04 | −8.3 |
| qwertz | 76.66 | 66.13 | −10.5 |
| azerty | 75.31 | 60.57 | −14.7 |
| german | 72.08 | 57.34 | −14.7 |
| dvorak | 63.04 | 43.92 | −19.1 |

The **ordering is identical** under both scorings, and dvorak is last by a wide
margin in both — while holding the *largest and best-calibrated* lexicon of the six
(146,964 words, full 0–5.40 `log_freq` spread). A deficit cannot be blamed on the
lexicon when the worst layout has the best lexicon. The confound is real in
magnitude but it does not touch the conclusion.

It is also worth noting what this table says on its own: the frequency prior is
worth more where the emissions are worse (−2.2 pt on qwerty, −19.1 pt on dvorak).
The lexicon is compensating for the encoder, not complementing it.

### The extra-key arm buys nothing

Giving the model the layout's non-a-z keys (`'`, `ä ö ü`, `ñ`) as extra masked
slots, so it knows a key exists where the finger actually went, moves nothing:

| layout | az26 t1 | full t1 | Δ |
|---|---|---|---|
| dvorak (+`'`) | 63.04 | 63.09 | +0.05 |
| azerty (+`'`) | 75.31 | 75.41 | +0.10 |
| german (+`ä ö ü`) | 72.08 | 72.08 | 0.00 |
| spanish (+`ñ`) | 81.34 | 81.11 | −0.23 |
| qwertz (no extra keys) | 76.66 | 76.66 | — |

All five deltas are inside noise. The `az26` arm is used for every headline number.

### The smaller, faster model transfers *better* — capacity is not the bottleneck

`fast_resbn80_s1234` (279,346 params, 0.215 ms — **2.5× smaller and 2.2× faster**
than the ch 128 ship artifact) was put through the identical pipeline, same preset,
same lexicons, same row sets:

| layout | n | ch128 t1 / t3 / t5 | **resbn80** t1 / t3 / t5 | Δt1 |
|---|---|---|---|---|
| qwerty (val[0:2000]) | 1,928 | 91.55 / 95.80 / 96.68 | 91.08 / 96.01 / 96.63 | **−0.47** |
| spanish | 1,758 | 81.34 / 93.91 / 95.68 | 82.37 / 94.82 / 96.25 | **+1.03** |
| qwertz | 1,187 | 76.66 / 90.06 / 92.67 | 78.77 / 91.74 / 94.44 | **+2.11** |
| azerty | 2,090 | 75.31 / 90.67 / 93.30 | 76.03 / 91.72 / 94.50 | **+0.72** |
| german | 2,199 | 72.08 / 86.45 / 90.40 | 76.17 / 88.77 / 92.00 | **+4.09** |
| dvorak | 2,457 | 63.04 / 73.71 / 75.17 | 67.28 / 77.70 / 79.08 | **+4.24** |

The 2.5×-smaller model is **behind on en_qwerty and ahead on every single
alt-layout**, and its margin grows monotonically with how hard the layout is
(+1.03 on the easiest, +4.24 on the hardest). This inverts the en_qwerty ordering:
Phase F measured resbn80 at −0.61 t1 against ch 128 on val-9918, and the ranking
reverses the moment the keyboard changes. (The −0.47 measured here on val[0:2000]
reproduces that direction independently.)

The reading that fits this and §7 together: **ch 128's extra capacity went into
QWERTY-specific structure that does not transfer.** It is a memorization effect, not
a better representation — and it is direct evidence that the cross-layout ceiling
here is set by *augmentation*, not by parameters. Scaling the model up would make
transfer worse, not better.

(**`fast_resbn80`'s en_qwerty standing is not settled and is not this document's
to settle** — it moved twice while this evaluation was being written, from
val-only to test-validated at `368426b`, then to *fails three of five against the
val-tuned FUTO bar* at `8d7462c`, which found its five-of-five pass to be an
artifact of an untuned opponent. `RESULTS.md` is the authority; nothing here
should be read as a claim about it. **The finding above is a within-this-document
paired comparison** — the same corpora, rows, preset, lexicons and harness for
both models — so it is unaffected by how the en_qwerty question resolves. It is
diagnostic, not a ship recommendation: the cross-layout numbers are single-seed.)

Even so, resbn80 on dvorak (67.28) still loses to the geometric engine's 76.8 by
9.5 pt. A better-transferring architecture narrows the dvorak gap; it does not close
it.

## 7. Why it fails — two diagnostic probes

### 7.1 Slot-permutation invariance is *perfect*, and irrelevant

Scatter the 26 keys into random slots of the 64 at inference — a fresh random
permutation per trace, exactly the train-time augmentation — and gather the emission
columns back. If the model reads slot index rather than geometry, this destroys it.

(The harness builds its `[32,27]` view by gathering the 26 columns holding a-z plus
the blank at index 64, rather than calling `slice_emissions`, because under a
permutation those columns are no longer `0..25`. On identity slots the gather is
asserted **bit-identical** to `slice_emissions(full, 26, 64)`, so every non-permuted
number in this document is on the campaign's exact code path.)

| corpus | traces | max abs Δ on the gathered `[32,27]` view | mean abs Δ |
|---|---|---|---|
| qwerty | 200 | 3.815e-06 | 5.7e-08 |
| dvorak | 200 | 1.907e-06 | 6.2e-08 |
| azerty | 200 | 3.815e-06 | 6.2e-08 |
| german | 200 | 3.815e-06 | 6.4e-08 |

That is float32 rounding. **Slot-permutation augmentation achieved its objective
completely — the model reads geometry, not slot index.** It is simply not the same
property as layout-invariance, and the campaign conflated the two. The model reads
*which key sits in which slot* perfectly, and still cannot decode a path over keys
arranged differently from QWERTY.

### 7.2 The affine envelope — global transforms are tolerated, re-arrangement is not

Apply one affine to **both** the en_qwerty val paths and the qwerty key centers —
precisely what `train.py`'s shared affine does — and vary it (val[0:1000],
in-dict n=970):

| affine (sx, sy, tx, ty) | in training distribution? | t1 | greedy |
|---|---|---|---|
| 1.00, 1.00, 0, 0 (identity) | yes | **91.75** | 73.81 |
| 1.00, 1.30, 0, 0 | no (sy above range) | 91.44 | 73.81 |
| 1.00, 0.70, 0, 0 | no (sy below range) | 90.21 | 69.90 |
| 0.85, 0.85, −0.05, −0.05 | **yes** (range corner) | 88.25 | 43.92 |
| 0.70, 1.00, 0, 0 | no (sx below range) | 85.98 | 22.58 |
| 1.00, 0.50, 0, 0 | no (far below) | 84.74 | 47.73 |
| 1.15, 1.15, +0.05, +0.05 | **no — rejected in training** (see below) | 80.21 | 28.04 |
| 0.50, 0.50, 0, 0 | no (far below) | 44.74 | 2.06 |

Two things fall out.

**(a) Global affine distortion is survivable.** Squashing the keyboard vertically to
70 % or stretching it to 130 % — both outside the nominal augmentation range — costs
0.3–1.5 pt of top-1. Even a 50 % vertical squash costs only 7 pt. So dvorak's
**4-row** geometry (row pitch 0.248 against QWERTY's 0.333) is *not* what breaks it.
What breaks it is that all 26 letters sit in different places. The model tolerates
moving the whole keyboard; it does not tolerate re-arranging the letters on it.

**(b) The rejection sampler silently truncated the x-augmentation.** `train.py`
rejection-samples the affine so every transformed center stays inside `[0,1]`
(audit fix #13). QWERTY's key centers already span `cx ∈ [0.05, 0.95]`, so almost
any horizontal *expansion* violates the bound and is thrown away. Simulating the
sampler 200,000 times against the real `en_qwerty.json` centers:

| | nominal range | **accepted** min / median / p95 / max | accepted mean |
|---|---|---|---|
| `sx` | 0.85 – 1.15 | 0.850 / 0.953 / 1.063 / **1.111** | **0.955** |
| `sy` | 0.85 – 1.15 | 0.850 / 1.000 / 1.135 / 1.150 | 1.000 |

**31.5 %** of first draws are rejected, and the survivors are biased toward
horizontal *compression*: the model has effectively never seen the keyboard stretched
wider, while the vertical axis kept its full nominal range. The measured asymmetry in
the probe table matches — `sy = 0.70` costs 4 pt of greedy, `sx = 0.70` costs 51.
(Part of that gap is intrinsic: 10 columns pack far more discriminative information
into x than 3 rows do into y, so horizontal compression destroys more signal than
vertical. The two causes are not separable from this data, but the sampler bias is
real and free to fix.)

## 8. Interpretation

**Does slot-permutation-only augmentation transfer? Partially, and predictably.**

The augmentation delivered two properties and neither is layout-invariance:

1. *Slot invariance* — complete, to float32 (§7.1). The model genuinely reads the
   `layout_keys` geometry to decide what an emission column means.
2. *Affine tolerance* — good, and wider than the training range in y (§7.2a).

What it never trained is **key re-arrangement**: a layout where letters occupy
different positions relative to one another. The shared affine is an axis-aligned
scale/translate/mirror of the whole board; no such transform, at any parameters,
turns QWERTY into Dvorak. The augmentation could not have taught it,
and empirically it did not.

**Which fails harder — layout shift or language shift? Layout, by ~3×.**

The two axes are cleanly separated by the corpus set:

* **dvorak/en** — maximal geometry shift (all 26 keys moved, mean displacement
  0.431), *zero* language shift, *identical* lexicon to the control:
  91.11 → 63.04, **−28.1 pt**.
* **spanish/es** — near-zero geometry shift (mean displacement 0.0175), complete
  language shift, different lexicon: 91.11 → 81.34, **−9.8 pt**, and that 9.8
  includes the lexicon change, the smaller vocabulary and the harder OOV profile,
  so it is an upper bound on the language effect.

Language is close to free. The four non-English layouts all beat or match the
shipped geometric engine. **A user typing French on AZERTY or German on QWERTZ is
already well served by this model today**; a user typing English on Dvorak is not.

The failure is also graded, not binary, and the grading is geometric:

```
mean key displacement vs qwerty:  0.0000  0.0175  0.0579  0.1068  0.1071  0.4313
                        top-1  :   91.11   81.34   76.66   75.31   72.08   63.04
                     greedy t1  :   72.83   31.91   27.21   22.06   15.14   11.64
```

Greedy — pure emission quality — collapses far faster than beam top-1, because the
lexicon absorbs the damage. That has a practical consequence: **any layout-transfer
claim measured only through a beam + a strong lexicon will look far healthier than
the model actually is.** Greedy should be reported alongside every cross-layout
number.

### What it would take to close the gap

The fix is already named in our own recipe and was never built.
`docs/guides/train-ctc-swipe-model.md` §6, augmentation item 3:

> *Optional later: sampling entirely different layout geometries (other language
> packs' key grids) with synthetic ideal paths — only needed when multi-language
> ships.*

That is exactly the missing stage, and this evaluation is the evidence that it is no
longer optional. Concretely, three changes, in cost order:

1. **Fix the rejection-sampler truncation** (≈10 lines). Scale about the key
   bounding-box centroid, or renormalize after transforming, instead of scaling
   about `0.5` and rejecting. Recovers the intended `sx` range that 31.5 % of draws
   currently throw away. Essentially free, and it should be done regardless.
2. **Layout-resampling augmentation** (the real fix, ≈1 day). Per sample, draw a
   target geometry — a real layout from the app's set, or a synthetic one (random
   assignment of the 26 letters over a random 3-or-4-row lattice with per-key
   jitter) — and warp the path from the QWERTY key positions onto the target ones
   with an inverse-distance / thin-plate displacement field. This runs on the cached
   `[2,64]` features, so **no re-featurization and no new data collection**: the
   `.npz` tiers are reused unchanged. Targets are already slot-space, so the CTC
   loss needs no change at all.
3. **Retrain and re-measure.** This harness is the evaluation.

The resbn80-vs-ch128 inversion (§6) says this is the *right* lever: capacity is
already past the point of diminishing — indeed negative — returns for transfer, so
the gain has to come from what the model is shown, not from how big it is.

**Cost.** Measured from the checkpoint metrics of the artifact under test
(`phaseE-E3b-hws3x`, RTX 5080 Laptop): **94,000 steps in 29.1 min wall**, of which
241 s is beam-validation. Three seeds is **~1.5 h of GPU**. The larger
`phaseF-N72-188k` doubled schedule is 70.8 min. So the retrain itself is
inconsequential; the cost is the ~1 day of augmentation-pipeline work, and the risk
is a regression on the en_qwerty bar, which is directly gated by the five val bars
(85.52 / 91.54 / 92.80, ≤3 89.29, 4+ 83.57) and by the sanity control in §5.

**Expected size of the win.** The task brief cites the FUTO paper (arXiv 2606.25247)
reporting that their full geometric augmentation moved Russian 40.5 → 77.2 and
ClearFlow 3.2 → 96.5 cross-layout. **Those figures were not verified locally** — no
copy of the paper is in this repo or the app repo — so they are quoted, not relied
on. What *is* locally established is the shape of the gap and that our augmentation
is a strict subset of theirs: the mechanism they credit for those gains (training
over many geometries) is precisely stage 2 above, and precisely the stage we skipped.

### What not to do

Do not read this as "ship the geometric engine for non-QWERTY". On four of five
layouts the CTC model already wins, usually by more on top-3/top-5 than it loses on
top-1. The correct routing today, on this evidence, is:

* **QWERTY-like Latin layouts (spanish, qwertz, azerty, german)** — route to the CTC
  model. It beats or matches the geometric engine and is 0.47 ms.
* **Dvorak, and by extension Colemak / Neo2 / arbitrary user XML** — do **not** route
  to the CTC model on current weights. The geometric engine is 13.8 pt better at
  top-1 there and is the right decoder until stage 2 above lands.

A cheap, principled gate is already available and needs no new model: **mean a-z key
displacement from the training layout**. Everything at ≤ 0.11 works; 0.43 does not.
That is a two-line check against the layout the IME already has in hand.

## 9. Open questions

* **⚠ Every CTC number here is at a preset tuned on a *different* layout, and the
  geometric anchors are not.** E1 was fitted on en_qwerty val-9918; it is applied
  unchanged to five layouts it never saw. The geometric engine's config A is its own
  shipped tuning. So this comparison is *tuned-vs-untuned in the geometric engine's
  favour* — the mirror image of the asymmetry `FAIR_REMATCH.md` found in the FUTO
  head-to-head, where roughly two thirds of a published margin turned out to be an
  untuned opponent. The λ=0 ablation in §6 bounds how big this lever is on these
  corpora and it is **large**: the frequency term alone is worth 8–19 pt per
  alt-layout against 2.2 pt on en_qwerty, so the alt-layout numbers are the ones
  most likely mis-tuned. Direction of the bias: **the CTC results below are a floor,
  not a ceiling.** A per-layout preset sweep (`sweep_scoring.py`, tune on half the
  corpus, confirm on the untouched half, reject grid-edge winners) is the obvious
  next run, and would firm up the four wins and probably narrow — though on a 13.8 pt
  gap, very unlikely to close — the dvorak loss.
* **Where between 0.11 and 0.43 does it break?** Only five layouts exist in this
  corpus and there is a wide gap between german (0.1071) and dvorak (0.4313). The
  routing threshold above is therefore bounded, not located. Colemak would land near
  the middle but has no real corpus here.
* **Non-Latin scripts are untested and out of scope.** The 64-slot contract fits
  Cyrillic's 33 keys mechanically, but there is no non-Latin training or evaluation
  data locally and none was fabricated. ЙЦУКЕН remains unmeasured for this model;
  `docs/specs/geometric-swipe-engine.md:726` records the same gap on the geometric
  side ("a JCUKEN real-corpus replay — CONFIRMED not on HuggingFace").
* **The `sx`-vs-`sy` asymmetry in §7.2b** mixes a fixable sampler bias with an
  intrinsic information-density difference between rows and columns. Separating them
  needs a retrain with the sampler fixed — item 1 above, which is cheap enough to
  just do.
* **`ß` is genuinely untypeable** on an a-z emission alphabet: 34 german and 18
  qwertz traces (2.1 % / 1.3 % of rows) can never be decoded correctly no matter how
  good the encoder gets. If German is a target, the alphabet needs a 27th emission
  column, which the 64-slot contract already accommodates.

## 10. Reproducing

```bash
# 1. corpora + official geometries (~10 min, network-bound)
node ctc/fetch_futo_multilayout.mjs

# 2. harness sanity control — must print 88.02 / 92.27 / 93.03
python3 ctc/eval_beam.py --onnx $PWD/ctc/artifacts/ch128_s1234.onnx \
    --test data/val_hwsfuto.jsonl --preset "1.05,1.1,0.2,0.3734,0.9882"

# 3. frame-mapping sanity (must precede any accuracy reading)
python3 ctc/eval_altlayout.py --sanity --qwerty-control data/val_hwsfuto.jsonl

# 4. the headline table
python3 ctc/eval_altlayout.py --arm az26 --arm full   # all five layouts, both arms
python3 ctc/eval_altlayout.py --layouts "" --qwerty-control data/val_hwsfuto.jsonl \
    --limit 9918                       # the in-dict en_qwerty comparator
python3 ctc/eval_altlayout.py --layouts dvorak --lexicon dvorak=en   # geo-comparable

# 5. confound control and probes
python3 ctc/eval_altlayout.py --preset "1.05,0.0,0.2,0.3734,0.9882" \
    --qwerty-control data/val_hwsfuto.jsonl          # lambda = 0
python3 ctc/eval_altlayout.py --layouts "" --qwerty-control data/val_hwsfuto.jsonl \
    --limit 1000 --affine "0.7,1.0,0.0,0.0"          # affine envelope
python3 ctc/eval_altlayout.py --perm-seed 42                     # slot-permutation probe

# 6. the second model
python3 ctc/eval_altlayout.py --onnx artifacts/fast_resbn80_s1234.onnx --arm az26
```

Raw run logs and per-run JSON: `~/ctc-train/altlayout/` (not committed).

