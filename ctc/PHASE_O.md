# Phase O — per-script CTC models for the non-Latin scripts the app can serve

**Opened:** 2026-08-18. **Workdir** `~/ctc-train`, **GPU** RTX 5080 Laptop (16 GB).
The app repo `/home/will/git/swype/CleverKeys` is a **read-only reference**.

Phase I-B proved one thing and left one question open. It proved that a script
with **no swipe corpus at all** can be launched from English motor residuals
transplanted onto its own layout — Cyrillic reached in-dict t1 **76.21** at
λ = 1.1, **≈ 77.4** once PHASE_J §6.9 tuned λ = 2.0, against a real Yandex
holdout with no real Cyrillic row anywhere in the pipeline. The open question
was scope: *which other scripts does that unlock, and what does each one
actually cost?*

Phase O answers that question script by script. Everything here is
**synthesis-only evidence** unless a section says otherwise — Russian remains
the single script in this campaign with a real eval corpus, and that corpus
(Yandex Cup 2023) is **eval-only, never training**, per `YANDEX_LICENSE_RESEARCH.md`.

---

## 1. O1 — INVENTORY

### 1.1 What "the app can actually serve" means, operationally

A script is servable when **both** halves exist:

**(a) A layout with extractable per-key geometry.** The app ships 86 layout XMLs
in `src/main/layouts/` (the dir `build.gradle`'s `copyLayoutDefinitions` copies
into `build/generated/layouts/res/raw`; `srcs/layouts/` is the upstream-style
source dir the *tests* read). 36 of them declare a non-Latin `script`.

**(b) A lexicon.** The app has exactly **seven bundled CKDT-v2 dictionaries**
(`src/main/assets/dictionaries/`: en 98,140 · es 50,000 · fr/de/it/pt/sv 40,000)
— all Latin — plus **19 importable langpack zips** in `scripts/dictionaries/`.
Of those 19, exactly **two are non-Latin: `langpack-ru.zip` (50,000) and
`langpack-el.zip` (39,860)**. There is no Hebrew, Arabic, Persian, Armenian,
Georgian, Devanagari, Bengali, Tamil, Kannada, Gujarati, Sinhala or Hangul
lexicon anywhere in the repo.

So on the strict reading, the answer to "every non-Latin script the app can
actually serve" is **two: Cyrillic-ru and Greek-el**. Everything else is
blocked on a dictionary — which for eleven further languages is a ~20-minute
job, because `wordfreq` is installed here and the app's own dictionary
pipeline (`scripts/build_wordlist.py` → `build_dictionary.py`) is itself a
wordfreq consumer whose rank formula this phase replicates byte-exactly
(`script_registry.frequency_to_rank`). Those are carried as **Tier B** below.

### 1.2 The geometry extractor, and why it is trustworthy

`app_layout.py` replicates the app's own key walk line for line:

* `a11y/KeyboardGeometry.computeKeyRects` — `xLeft = x + key.shift*keyWidth`,
  `xRight = xLeft + key.width*keyWidth`, `x = xRight`; `yTop = y + row.shift*rowHeight`,
  `yBottom = y + (row.shift+row.height)*rowHeight`, `y = yBottom`;
* `KeyboardData.Row.parse` / `Key.parse` defaults — row `height` 1 (clamped
  ≥ 0.5), row `shift` 0, row `scale` 0 = off (else `Row.updateWidth` scales key
  *widths* only), key `width` 1, key `shift` 0; centre value is `key0` **or** its
  synonym `c`;
* `swipe/CtcEngineAdapter.buildMappedLayout` — normalisation over the
  **bounding box of the letter keys only**, first-occurrence-wins in row-major
  order.

Margins and unit sizes cancel in that normalisation, so they are fixed at 0/1.
The bottom row, number row and numpad are injected at runtime and carry no
letters, so they cancel too.

**Validation — the extractor reproduces the frame the campaign trained in.**
Running it on the app's own `latn_qwerty_us.xml` and differencing against
`en_qwerty.json` — the geometry every English model in this campaign was
trained and evaluated on:

| comparison | max abs dx | max abs dy | mean euclid | max euclid |
|---|---|---|---|---|
| app `latn_qwerty_us.xml` vs `en_qwerty.json` | **4.7e-4** | 0.0 | 2.9e-4 | **4.7e-4** |
| app `cyrl_jcuken_ru.xml` vs Yandex-derived `ru_jcuken_default.json` | 3.4e-3 | 0.0 | 8e-4 | 3.4e-3 |

Row 1 and row 3 agree to the last bit; the 4.7e-4 is row 2 alone (a rounding
vintage in the FUTO-derived json). **The app frame and the training frame are
the same frame**, to 0.05 % of the board — three orders of magnitude inside the
measured affine-tolerance envelope. Every Phase-O layout json therefore lands
on the canonical `(2r+1)/2R` row family (0.1667 / 0.5 / 0.8333), which was
checked for all five generated layouts.

The ru row of that table is a **free retro-validation of Phase I-B**: the ru
model was trained on the *Yandex corpus'* grid, and the app's own ЙЦУКЕН grid
sits 3.4e-3 away from it. The ru model is deployable on the app's geometry
without a regeneration.

**Falsification control.** The same wrong-geometry control Phase I-B used
(`PHASE_I_DATA` §4: qwerty centres under a `letter → key i mod 26` map collapse
start-hit to 0.008 against 0.917 for the right frame) is re-run per script in
§2 as part of each script's endpoint-proximity gate — a frame is never accepted
on its row-position arithmetic alone.

### 1.3 Two app defects found while doing this

1. **The shipped Greek layout declares the wrong script.**
   `srcs/layouts/grek_qwerty.xml` says `script="greek"`; the copy that actually
   builds, `src/main/layouts/grek_qwerty.xml`, says `script="latin"` (verified
   identical in `build/generated/layouts/res/raw/`). The two files differ in
   nothing else. Today this is harmless — `SwipeEngineRouter` sends it down the
   CTC Latin path, `CtcEngineAdapter.buildMappedLayout` finds no a–z and returns
   null, and the swipe falls through to the geometric engine — but it is exactly
   the attribute a Greek CTC path would key on, so it must be fixed *before*
   any Greek model is wired in.
2. **The bundled Greek lexicon has no final sigma.** All 39,860 words in
   `langpack-el.zip` carry word-final **σ** where Greek orthography requires
   **ς** (`ξεκινώντασ`, `ὡσ`). This is documented in the app's own
   `scripts/build_wordlist.py::load_aosp`: wordfreq casefolds its corpus and
   Python casefolding maps ς→σ; the note calls the σ-final display forms "a
   documented wordfreq-status-quo caveat, not a fix target this round".
   For tap-typing that is a cosmetic misspelling. **For swipe-typing it is
   fatal**, and this phase measured how fatal: **25.7 % of the lexicon
   (25.4 % by frequency weight) is σ-final**, and σ and ς are *different keys in
   different rows* of the Greek layout (ς at the QWERTY-w position in row 1, σ at
   the s position in row 2). A σ-final lexicon would train and score one Greek
   word in four against the wrong endpoint. Phase O therefore restores final
   σ→ς by rule — a lossless repair, because modern Greek orthography is fully
   deterministic here — and the app must apply the same one-line rule when it
   builds the el trie (or regenerate the pack). See §3 for the integration note.

### 1.4 The non-Latin layout census

`app_layout.py --census` over all 86 shipped layouts. `centre` = letters on
`key0`/`c` (swipe-typeable); `corner-only` = letters that exist **only** on a
`key1..key8` slot. Corner letters are typeable by a directional flick but
**never by a swipe**, and the app's own alias table gives them the *host key's
centroid* — two letters at one coordinate, which no geometry-conditioned
encoder can separate. They are excluded from every Phase-O alphabet and their
cost is measured, not waved away.

| script | layouts | centre letters | corner-only | lexicon in repo | verdict |
|---|---|---|---|---|---|
| **Cyrillic** | 11 | 26–42 | varies | **ru 50 k (langpack)** | ru **DONE** (Phase I-B/J); uk/bg/mk buildable (Tier B); sr blocked; kk/mn/tj/os/as no lexicon source |
| **Greek** | 1 | **25** | none | **el 39,860 (langpack)** | **Tier A — the one new fully-served script** |
| Hebrew | 2 | 27 | none | none | Tier B (wordfreq `he`) |
| Arabic | 5 | 29–33 | **آ أ إ ذ** and more | none | Tier C — common letters are corner-only |
| Persian | 2 | 31 | ء آ ئ ژ | none | Tier C |
| Urdu | 1 | 26 | 18 incl. آ ث ح خ ذ ص ض ظ غ | none | Tier C — over 40 % of the alphabet is corner-only |
| Armenian | 1 | 38 | և | none | **blocked-on-dictionary** (no wordfreq `hy`) |
| Georgian | 2 | 26–32 | ჭ ჟ ღ შ ჩ ძ … | none | **blocked-on-dictionary** (no wordfreq `ka`) |
| Devanagari | 3 | **7–20** | 30–40 | none | **structurally blocked** |
| Bengali / Gujarati / Kannada / Tamil / Sinhala | 6 | **7–26** | 13–40 | none | **structurally blocked** |
| Hangul | 1 | 26 (jamo) | 25 | none | **structurally blocked** |
| Shavian | 1 | 39 | 9 | none | no lexicon; niche |

"Structurally blocked" is not a scheduling excuse, it is a property of the
layouts: `kann_kannada.xml` exposes **7** centre letters and 45 corner-only
ones, `deva_alt.xml` **8** centre and 40 corner. In those scripts the writing
system's units are simply not on the swipe surface — a swipe path cannot reach
them — so no amount of training data helps. They need a layout redesign (or a
conjunct-aware input model) before a swipe model is even a coherent request.
Arabic/Persian/Urdu sit one notch better: the *consonant skeleton* is mostly on
centre keys, but the hamza-carriers (أ إ آ) that a large share of words need are
on corners, so a swipe model would systematically fail those words. All three
are recorded as **priced, not attempted**.

### 1.5 The Phase-O work list, ranked

| # | script | letters | lexicon | tier | evidence available |
|---|---|---|---|---|---|
| 0 | **Cyrillic-ru** | 31 | app langpack-ru 50 k, CKDT | **DONE** — the worked example | synthesis + **real Yandex holdout** (unique) |
| 1 | **Greek-el** | **25** | app langpack-el 39,860, CKDT | **A** — bundled lexicon, perfect 1:1 layout fit, zero corner-only | **synthesis-holdout only** |
| 2 | **Ukrainian-uk** | 31 | wordfreq `uk`, app rank formula | **B** | synthesis-holdout only |
| 3 | **Hebrew-he** | 27 | wordfreq `he`, app rank formula | **B** — perfect fit, first abjad/RTL | synthesis-holdout only |
| 4 | **Bulgarian-bg** | 30 | wordfreq `bg`, app rank formula | **B** — perfect fit | synthesis-holdout only |
| 5 | **Macedonian-mk** | 31 | wordfreq `mk`, app rank formula | **B** — perfect fit | synthesis-holdout only |
| — | Serbian-sr | 30 | **none** — wordfreq `sh` yields **0** Cyrillic words in its top 80 k (it is a Latin-script list) | blocked | — |
| — | Armenian-hy, Georgian-ka | 38 / 32 | none | blocked-on-dictionary | — |
| — | Arabic/Persian/Urdu | 26–33 | none | priced, not attempted (corner-only letters) | — |
| — | Indic, Hangul | 7–26 | none | structurally blocked | — |

Measured costs that the ranking already accounts for:

* **uk** — ї and ґ are `loc` corner slots on `cyrl_jcuken_uk.xml`, so **4.03 %
  of the top-60 k Ukrainian vocabulary (4.16 % frequency-weighted) is
  permanently un-swipe-typeable** on this layout. That is a layout limitation,
  not a model limitation, and it is a ceiling on any uk number reported here.
* **bg / mk** — the only corner-only letters are the accented disambiguators
  ѝ (and ѐ for mk); they fold to и / е, which is what Bulgarian and Macedonian
  typists do on a keyboard without them. Effective loss ≈ 0.
* **el / he / sr** — zero corner-only letters. The layout is the alphabet.

### 1.6 Layout artifacts committed by O1

Generated with `app_layout.py --xml <file> --letters <alphabet>`; every one
lands on the `(2r+1)/2R` row family:

| script | layout json | source XML | letters | letter-box (units) |
|---|---|---|---|---|
| el | `layouts/el_qwerty.json` | `grek_qwerty.xml` | 25 | 10 × 3 |
| uk | `layouts/uk_jcuken.json` | `cyrl_jcuken_uk.xml` | 31 | 11 × 3 |
| bg | `layouts/bg_bds.json` | `cyrl_ueishsht.xml` | 30 | 11 × 3 |
| mk | `layouts/mk_lynyertdz.json` | `cyrl_lynyertdz_mk.xml` | 31 | 11 × 3 |
| he | `layouts/he_1.json` | `hebr_1_il.xml` | 27 | 11 × 3 |

All five load through the committed `train.py --layout` unchanged
(`load_layout_centers` accepts them; keys[] order == `letters`, so no emission
column is permuted). Every alphabet fits well inside the 64 slots the head
provides — the largest here is 31, the same as ru.

### 1.7 Lexicon inventory as loaded

`script_registry.py` output (projection applied identically to lexicon and
targets; weights on the CKDT `255 − rank` scale):

| script | source | records | distinct projectable | notes |
|---|---|---|---|---|
| el | app `langpack-el.zip` (CKDT v2) | 39,860 | **37,516** | 2,344 collapse under accent stripping; 0 unprojectable; 9,638 (25.7 %) σ-final → ς restored |
| uk | wordfreq `uk`, depth 54,599 | 50,000 | 49,955 | rank via the app's own formula |
| he | wordfreq `he`, depth 51,332 | 50,000 | 49,915 | niqqud stripped; final forms kept distinct |
| bg | wordfreq `bg`, depth 37,325 | 35,820 | 35,788 | list exhausts before the 50 k cap |
| mk | wordfreq `mk`, depth 52,172 | 50,000 | 49,963 | |
| sr | wordfreq `sh`, depth 54,841 | **0** | **0** | `sh` is a Latin-script list — Serbian is blocked |

**Honesty on the Tier-B lexicons:** a wordfreq top-N list is *not* what the app
ships. `build_wordlist.py` runs hunspell/aspell/pyspellchecker/AOSP oracles plus
per-language allow/block lists over its candidate stream; the Tier-B lists here
skip all of that, so they carry more corpus noise (typos, foreign tokens,
inflected junk) than a real pack would. The *frequency scale* is identical by
construction; the *word selection* is not. Every Tier-B number in §2 is a
number against a lexicon the app does not yet have.

---

## 2. O2 — PER-SCRIPT RESULTS

### 2.1 The calibration that governs every number below — and inverts one claim

Phase O's probe for a corpus-less script is a **synthesis holdout**: 10,000 rows
from the same generator as the training set, but over a **disjoint half of the
English donor pool** (a 90/10 stride split, so a holdout trace carries motor
noise from a human trace training never saw) and an **independent word draw**
(seed 777 vs 1234). It is the best probe that can exist without a corpus. The
first thing Phase O did was find out what such a number is worth — on Russian,
the one script that has **both** a synthesis holdout and a real corpus.

Four cells, one preset (the app's CKDT preset γ 1.05 / **λ 2.0** / β 0.2 /
0.3734 / 0.9882), one harness, one trie (`langpack-ru` 50 k), one geometry
(`ru_jcuken_default`). `eval_script.py`; the real probe is the untouched Yandex
valid-10k on the eval-only footing `YANDEX_LICENSE_RESEARCH.md` permits.

| model | ru **synthesis holdout** (n = 10,000) | ru **REAL** swipes (n = 8,471 in-dict) |
|---|---|---|
| `ru_synth_ch80` — the script-trained model | 81.10 | **77.41** |
| `phaseM_kd_fresh_w1` — the shipped **English** model, zero-shot | **83.38** | 76.32 |
| Δ (script − English) | **−2.28** | **+1.09** |
| paired exact McNemar | p = **7.1e-12** (English wins) | p = **0.0099** (script wins) |

The English model there is the shipped ch192; the fair, **capacity-matched**
control is `phaseH-p50` (ch 80, resbn, dil 1,2,4,8, embed_hid 96 — architecturally
identical to the ru model, trained on English with layout-alt p 0.5). On the same
real probe it reads **75.79** in-dict t1 (greedy 17.57), so:

| pair, real Yandex probe, n = 8,471 | Δ t1 | paired exact McNemar |
|---|---|---|
| ru-synth ch80 **vs** ch80 English zero-shot (capacity-matched) | **+1.62** | p = **1.4e-4** |
| ru-synth ch80 **vs** ch192 English zero-shot (the ship model) | +1.09 | p = 0.0099 |
| ch192 English **vs** ch80 English, both zero-shot | +0.53 | p = 0.11 (**n.s.**) |

Two clean reads: **per-script synthesis training is worth ≈ +1.6 real top-1 at
matched capacity**, and **English capacity buys nothing cross-script** — tripling
the English model's width moves zero-shot Cyrillic by half a point of noise.

**Both deltas are significant and they have opposite signs.** The synthesis
holdout does not merely flatter models — for the comparison that actually
matters (is a per-script model worth building?) **it returns the wrong answer,
with high confidence.**

Why: the holdout traces are English motor residuals re-anchored onto the target
script's polylines. A model trained on English motor statistics is, by
construction, *at home* on that distribution; a model trained on the same
synthetic distribution should be equally at home, but it has also spent capacity
on the generator's artefacts (PHASE_J §6.5 found the same overfitting-to-the-
generator effect when ru capacity was raised). Real human swipes in the target
script are a third distribution, and the two models' distance to it does not
order the same way.

**A third defect of the probe, measured: its length mix is wrong.** Words are
drawn by lexicon weight (`255 − rank`, the compressed CKDT scale), not by corpus
token frequency, so the holdout is **3.3 % short words (≤3 letters)** against the
real corpus's **38.7 %**. Stratified, the inversion is sharpest exactly where the
probe is thinnest:

| probe | model | ≤3 t1 | 4+ t1 |
|---|---|---|---|
| ru synthesis holdout | ru-synth ch80 | 62.31 (n = 329) | 81.74 |
| ru synthesis holdout | ch192 EN zero-shot | **73.86** | **83.70** |
| ru REAL | ru-synth ch80 | **86.44** (n = 3,281) | **71.70** |
| ru REAL | ch192 EN zero-shot | 85.22 | 70.69 |
| ru REAL | ch80 EN zero-shot | 83.11 | 71.16 |

Re-weighting the holdout to the real corpus's 38.7/61.3 length mix does **not**
rescue it (ru-synth 74.22 vs English 79.89) — the probe is wrong on short words
in *level*, not merely in *proportion*. Against real short Russian swipes the
script-trained model leads by +1.2/+3.3; against *synthetic* short Russian
"swipes" it trails by −11.6. Short synthetic words are short English traces with
their few vertices re-anchored, and that is evidently a distribution the English
model owns and the script-trained model has partly un-learned.

**Recipe consequence carried forward (not actioned this phase):** the synthesis
word draw should be weighted by corpus token frequency rather than by the
compressed dictionary rank, so that the generated corpus has a realistic length
mix. Re-generating and re-training all six scripts to test that was out of
budget here; it is the first thing a Phase P should do.

Three consequences, applied throughout §2:

1. **No synthesis-holdout number in this phase is a quality claim.** Each is
   reported next to the zero-shot English control on the identical probe, and
   the pair is read only as "these two models differ by X **on the generator**",
   never as "this script decodes at X".
2. **The zero-shot English control is not a straw man — it is the deployment
   alternative.** On real Russian the model the app already ships reaches
   **76.32** with nothing but the right layout and the right trie. Purpose-built
   synthesis training bought **+1.09** on top of that. By contrast, Phase I-B's
   *real-data* Russian arm reached 89.64 at λ 1.1 (≈ 90+ at λ 2.0). So on the
   only script where this can be checked: **real data is worth ~13 points;
   synthesis training is worth ~1.** The expensive half of "a script needs its
   own model" is the data, not the model.
3. **The guide's flat claim needs narrowing.** `ctc-architecture-and-multiscript-guide.md`
   §3.1 says "a model trained on Latin-arrangement geometry with English lexicon
   statistics does not zero-shot another script … do not route a non-Latin
   layout at CTC and hope." Measured, that is **too strong**: the shipped model
   zero-shots Cyrillic at 76.32 in-dict top-1 on real swipes — the same band as
   the shipped geometric engine's cross-layout anchors (71–77) and within 1.1 of
   the purpose-trained model. What survives of the claim is the *emissions*
   half: zero-shot greedy is **18.62** against the script-trained model's
   **37.13**, i.e. the English model is leaning almost entirely on the trie. The
   accurate statement is: *a Latin-trained model transfers to another script's
   geometry far better than expected, because layout-alt augmentation taught it
   to read key positions rather than slot indices — but its emissions are poor,
   so it needs a strong lexicon and it degrades faster than a script-trained
   model would when the lexicon is weak.*

This is genuinely out-of-distribution transfer, not a hidden in-distribution
case: the English model was trained with `k = 26` active slots drawn as a random
26-subset of the 64 (`train.py` slot-permutation augmentation), so it had never
seen 31 simultaneously-active slots nor ЙЦУКЕН key positions.

**Honesty limits of this calibration.** It is one script, one pair of models,
one preset. The ru synthesis holdout re-uses donor traces that the ru model saw
during its own training (paired with different words — `cyrillic_synth.py` used
the full donor pool, before Phase O introduced the 90/10 split), which if
anything inflates the ru model's holdout column and therefore *understates* the
sign reversal. And the script-trained side is the *synthesis*-trained ru model,
not the real-data one; a real-data model would win both columns comfortably.

### 2.1b The calibration grid, completed — and the warm-start arm

Two more models were trained/measured to fill the grid out. `phaseO-ru-initH` is
the ru synthesis recipe **warm-started from the English ch80 `phaseH-p50`**
(`--init-from`, weights only, same cache, same seed, same 94 k schedule — the
only difference is the initialisation). It is the one *recipe* change Phase O can
validate against real data, and it is the obvious thing to try once you know the
English model transfers.

| model (all ch 80 except where noted) | ru synthesis holdout | ru **REAL** | paired vs cold-start ru-synth on REAL |
|---|---|---|---|
| `ru_synth_ch80` — cold start | 81.10 | **77.41** | — |
| `phaseO-ru-initH` — warm start from English | **81.98** | 77.26 | **−0.14, p = 0.69 (a wash)** |
| `phaseH-p50` — English ch80, zero-shot | 76.24 | 75.79 | −1.62, p = 1.4e-4 |
| `phaseM_kd_fresh_w1` — English **ch192**, zero-shot | **83.38** | 76.32 | −1.09, p = 0.0099 |

* **Warm-starting from English is worth nothing on real data** (−0.14, p = 0.69)
  while looking like a +0.88 gain on the holdout. Registered as tried and
  refuted; do not warm-start.
* **The holdout over-credits capacity, and that is what flips the ranking.**
  English ch192 beats English ch80 by **+7.14 on the holdout** and by **+0.53
  (n.s.) on real swipes**. Every other pairwise sign is preserved between the two
  probes; only the capacity axis inverts, and it inverts hard enough to reorder
  the top of the table. Real ordering: ru-synth > initH > ch192-en > ch80-en.
  Holdout ordering: ch192-en > initH > ru-synth > ch80-en.

### 2.1c The λ sweep was run as registered — and the sweep itself is invalid

O2(e) registered a per-script λ sweep on a synthesis-val half (the PHASE_J §6.9
pattern). It was run for all five scripts, tune half `0:5000`, confirm half
`5000:10000`. **All five picked λ = 1.1, monotonically, with λ = 2.0 several
points worse** — flatly contradicting PHASE_J §6.9's finding that a CKDT
`255 − rank` lexicon wants λ = 2.0.

The control settles it. Same ru model, same trie, same code, two probes:

| λ | ru **synthesis holdout** (confirm half) | ru **REAL** (confirm half) |
|---|---|---|
| 1.1 | **85.98** | 76.72 |
| 2.0 | 81.28 | **77.92** |
| | holdout says 1.1, by **+4.70** | real says 2.0, by **+1.20** |

**The holdout's λ preference is inverted relative to real data**, and the
mechanism is plain: on the holdout the emissions are in-distribution and strong,
so a frequency prior only adds noise; on real swipes the emissions are degraded
and the prior carries real information. PHASE_I_DATA §6 saw the same shape from
the other side (λ worth +7.6 t1 to the synth-trained arm on real data vs +0.9 to
the real-trained arm).

**Adopted: λ = 2.0 for every Phase-O script**, i.e. the app's existing CKDT-scale
preset `1.05, 2.0, 0.2, 0.3734, 0.9882`, because it is the only λ any *real* data
has ever endorsed and every Phase-O lexicon is on the same CKDT scale. The
per-script sweeps are reported and discarded. All golden fixtures are frozen at
λ = 2.0.

### 2.2 The five new scripts — results

Recipe, identical for all five and **verbatim from `phaseIB-ru-synth`'s stored
args**: `resbn:80` dil 1,2,4,8, embed_hid 96, feat_v1, 94,000 steps, batch 256,
lr 3e-3, wd 0.01, warmup 1,000, coupled affine sampler, **no layout-alt** (single
geometry), greedy checkpoint selection (`--beam-val-rows 0`), patience 40,
seed 1234, single seed. Only `--workers` differs (0 vs 8, per the ops protocol),
which changes the augmentation RNG stream but not its distribution.

Probe: each script's own 10,000-row synthesis holdout, disjoint donor half,
independent word draw, decoded at the adopted preset (λ = 2.0) through the
exported fp32 graph. **Read every row of this table through §2.1: these are
generator numbers, and the generator over-credits capacity and misranks models.**

| script | K | greedy | **in-dict t1** | t3 | t5 | ≤3 t1 | 4+ t1 | ≥70 gate |
|---|---|---|---|---|---|---|---|---|
| **el** Greek | 25 | 35.87 | **82.54** | 92.97 | 94.93 | 58.44 | 83.14 | pass |
| **uk** Ukrainian | 31 | 31.98 | **79.27** | 91.91 | 94.05 | 62.30 | 79.93 | pass |
| **bg** Bulgarian | 30 | 26.86 | **71.80** | 88.56 | 92.18 | 49.09 | 72.57 | pass |
| **mk** Macedonian | 31 | 29.39 | **71.69** | 88.33 | 91.80 | 46.15 | 72.65 | pass |
| **he** Hebrew | 27 | 37.91 | **65.36** | 85.10 | 90.13 | 50.31 | 66.39 | **FAIL** (70.28 at λ 1.1) |
| *(ru, for scale)* | 31 | 29.73 | *81.10* | *92.16* | *94.08* | *62.31* | *81.74* | *pass* |

**Hebrew is the one gate failure**, and it fails only at the adopted preset — at
the holdout-optimal λ = 1.1 it reads 70.28/71.12 and would pass. Since §2.1c
established that the holdout's λ choice is not to be trusted, the honest
statement is that **he sits at or just below the registered band and is the
weakest of the five**; its artifacts are produced and registered, flagged. Its
≤3 stratum (50.31) is the likely cause: Hebrew is an abjad, its words are short,
and short words are where a 27-key board with a 49,915-word lexicon has least
evidence to work with.

### 2.3 Every script against its controls — the comparison that is actually meaningful

Same probe, same preset, three models per script: the script's own model, the
shipped English ch192 zero-shot, and the capacity-matched English ch80 zero-shot.

| script | script model | ch192 EN zero-shot | Δ vs ch192 | **ch80 EN zero-shot** | **Δ vs ch80 (matched)** | greedy: script / ch192 / ch80 |
|---|---|---|---|---|---|---|
| el | 82.54 | 83.10 | −0.56 | 76.56 | **+5.98** | 35.87 / 16.60 / 9.71 |
| uk | 79.27 | 81.41 | −2.14 | 74.20 | **+5.07** | 31.98 / 15.24 / 7.77 |
| bg | 71.80 | 74.09 | −2.29 | 66.53 | **+5.27** | 26.86 / 12.97 / 7.28 |
| mk | 71.69 | 72.67 | −0.98 | 65.19 | **+6.50** | 29.39 / 13.70 / 6.90 |
| he | 65.36 | 69.11 | −3.75 | 58.04 | **+7.32** | 37.91 / 26.19 / 16.36 |
| ru (calibrated) | 81.10 | 83.38 | −2.28 | 76.24 | **+4.86** | 29.73 / 14.27 / 7.07 |

The pattern is the same in all six scripts, ru included, and ru is the one where
we know what it means on real data:

* **against the capacity-matched control the script model wins everywhere**, by
  +4.9 to +7.3 on the holdout; on real Russian that same comparison is **+1.62,
  p = 1.4e-4** — same sign, magnitude deflated ~3×;
* **against the 3×-capacity ship model the script model loses everywhere**, by
  −0.6 to −3.8 on the holdout; on real Russian that same comparison is **+1.09,
  p = 0.0099** — *opposite* sign, because the holdout over-credits capacity
  (§2.1b);
* **greedy is 2–4× higher for every script model than for either English
  control.** The emissions really are script-specific; the lexicon beam is what
  lets English keep up.

Every el/uk/bg/mk/he number above sits inside the band the ru calibration would
predict, so nothing here is anomalous — but nothing here is a measurement of
Greek, Ukrainian, Bulgarian, Macedonian or Hebrew swipe accuracy either. **What
these five rows license is exactly one claim: each script's model behaves, on its
own generator, the way the Russian model behaved on the Russian generator — and
the Russian model turned out to be worth +1.6 real points over the
capacity-matched English alternative.**

### 2.4 Falsification: the geometry is load-bearing, decisively

`eval_script.py --permute-layout 4242` shuffles the key centres so slot *c* gets
some other key's position, with no key left in place, and decodes again. If a
model were reading slot indices rather than geometry, or if a layout json were
merely decorative, this would barely move.

| probe | in-dict t1 | greedy |
|---|---|---|
| ru model, **real** Yandex traces, true geometry | 77.41 | 37.13 |
| ru model, **real** Yandex traces, permuted geometry | **0.00** | **0.00** |
| el / uk / bg / mk / he, own holdout, permuted geometry | **0.00** each | **0.00** each |

Zero, in every case, on every script. The endpoint-proximity control of §1.4 says
the *traces* are on the claimed frame; this says the *model* cannot function
without it. Together they close O2(a): a Phase-O layout json is a testable claim
and it passes.

### 2.5 Export gates

`export_onnx.py` with `--parity-features cache_<code>/val.npz` — the assertion
runs on the **sliced contract view** with **real traces on the real layout**, and
argmax agreement is the binding gate. Then `quantize_onnx.py --mode fp16w`.

| script | BN fold (sliced) | fp32 vs torch (sliced) | argmax | fp16w vs fp32 (sliced) | argmax | fp16w decode cost |
|---|---|---|---|---|---|---|
| el | 1.35e-04 | 1.54e-04 | **100/100** | 6.86e-02 | 99/100 | 82.54 → 82.55 (**+0.01**) |
| uk | 2.14e-04 | 1.26e-04 | **100/100** | 1.78e-01 | 95/100 | 79.27 → 79.27 (**0.00**) |
| bg | 6.03e-04 | 5.11e-04 | **100/100** | 2.85e-02 | 97/100 | 71.80 → 71.81 (**+0.01**) |
| mk | 3.72e-04 | 2.62e-04 | **100/100** | 5.65e-02 | 98/100 | 71.69 → 71.71 (**+0.02**) |
| he | 5.84e-04 | 2.10e-04 | **100/100** | 2.19e-04\* | 99/100 | 65.36 → 65.36 (**0.00**) |

\* he's fp16w residue printed 8.39e-02; the 2.10e-04 column is its fp32 probe.

Every fp32 export clears the 1e-3 tolerance with **100/100 argmax**. The fp16w
residues are large (2.9e-2 … 1.8e-1) and 95–99/100 on argmax — **disclosed, not
hidden**, exactly as the ru export disclosed 1.16e-01 / 98-100. Note also that
`quantize_onnx.parity_vs_source` probes with **white noise**, which PHASE_J §5.2
established is not a calibrated stand-in in either direction. The binding
evidence is therefore the decode, and the decode is free: **on 10,000-row
holdouts the fp16w bytes move top-1 by at most 0.02 points on any script**, and
uk and he move it by exactly zero.

### 2.6 Artifacts

`ctc/artifacts/`, all five at the standard 1,142,727-byte resbn80 graph (the
alphabet is data, not architecture — the same graph size ru, and every other
resbn80 in the campaign, produces):

| file | bytes | sha256 |
|---|---|---|
| `el_synth_ch80.onnx` | 1,142,727 | `857f8b7c710ec4ef3615b83eb9382a6355794ad1157c811be43709f2ddb23417` |
| `el_synth_ch80_fp16w.onnx` | 589,406 | `c8bff5d9cdf16428f99010d8b2414d5e27556339583760278fa3afb300520580` |
| `el_synth_ch80_fp16w_golden.json` | 144,593 | `9321eeda90a3d0ada68cf1720c8fedcf6154a11f3ddf36d0b961b0d8b0f38fa5` |
| `uk_synth_ch80.onnx` | 1,142,727 | `c466de5c4ba2a83728249db1e37bb64f4f9a6a27f7b7de851cf158684a28e201` |
| `uk_synth_ch80_fp16w.onnx` | 589,406 | `02246bbdd95682c45f432fbf7f8e51b51ad590a76f4a2091fe1bd65d3a282511` |
| `uk_synth_ch80_fp16w_golden.json` | 156,073 | `c5e2a3d37e9e90db80b9bfd8d4dd0e085f81faff609c85bae1773e143ac3fad0` |
| `bg_synth_ch80.onnx` | 1,142,727 | `f6fd179e7b634d8a04fb9c86de4380546a93940dc2277c9ab735e23b57440e40` |
| `bg_synth_ch80_fp16w.onnx` | 589,406 | `947dd44f8c122cfdb5d1ddeb270b94e938afddddddc4868f3b0706e29ec2ddd0` |
| `bg_synth_ch80_fp16w_golden.json` | 155,130 | `53901d76a4447aa055f11e712d586c51c955e61743821e04b5d6165d8ca57056` |
| `mk_synth_ch80.onnx` | 1,142,727 | `6f14cf78ac7fc473dc4178d8432f428cc6cc281622f8daefc27778bd770a0fd3` |
| `mk_synth_ch80_fp16w.onnx` | 589,406 | `be98fcc32ef9e9182915a0504e22b0390dce019ff83603dbc6caa2ae054eb682` |
| `mk_synth_ch80_fp16w_golden.json` | 160,894 | `ed852963cd018e3133281a29a3182f3c4c1fc9ae8380503fee471efee8b29374` |
| `he_synth_ch80.onnx` | 1,142,727 | `37a0518b466265c5a429879648674b186ee77cd6a2a095635a153cd5458ef149` |
| `he_synth_ch80_fp16w.onnx` | 589,406 | `384608660170cdfe7b77c730a255ad046041ba9ddee7e934668ee539a8cef202` |
| `he_synth_ch80_fp16w_golden.json` | 140,235 | `7950e656b1ac3767582416ed3d231f39202f7a71f34a5e20faf8ee9f463cb7b7` |

Every golden fixture is frozen at **γ 1.05 / λ 2.0 / β 0.2 / 0.3734 / 0.9882**
(§2.1c) on the script's real lexicon weights, with 10 cases each (5 pure-featurizer
branch probes + 1 word-path featurizer case + 4 model-backed beam cases).

`phaseO-ru-initH` is **not** promoted to `artifacts/` — it is a refuted arm
(§2.1b) and lives only at `~/ctc-train/ckpt/phaseO-ru-initH/`.

### 2.7 Evidence tier — say it exactly this way

> Greek, Ukrainian, Bulgarian, Macedonian and Hebrew CTC are
> **synthesis-trained, synthesis-holdout-only, single-seed, and calibrated
> against Russian rather than measured on their own script.**

Unpacked, and none of it is negotiable:

* **No real swipe data exists in any of these scripts.** `DATASET_SCOUT.md` §4.4
  searched and came up empty for Arabic, Hebrew, Greek, Thai, Hangul, Devanagari,
  Japanese and Chinese; Yandex-ru is the only real non-Latin corpus in existence
  under any terms, and it is eval-only.
* **The only probe available is one this phase proved unreliable** — it inverts
  the capacity axis (§2.1b), inverts the λ choice (§2.1c), and carries a length
  mix 12× off real usage (§2.1). Its numbers are reported because suppressing
  them would be worse, not because they measure accuracy.
* **Single seed (1234).** Every other campaign bar is a seed-mean bar; these are
  not.
* **Four of the five lexicons are not the app's.** uk/bg/mk/he run on wordfreq
  top-N lists put on the app's rank formula, with none of `build_wordlist.py`'s
  spell-check oracles or allow/block lists. el is the app's own pack, repaired.
* **No test split, ever.** There is no sealed Cyrillic/Greek/Hebrew test set and
  none can be created. These models can never be called "test-validated".
* **No on-device measurement.** No latency, no memory, no instrumented run. The
  graphs are byte-identical in size to the ru model, which is half the ship
  model, so the expectation is favourable — expectation is not measurement.

---

## 3. O3 — APP-INTEGRATION NOTES

### 3.1 What is shared by every script (do this once)

None of it is ML work; all of it is app work, and it dominates the cost of
adding a script. Verified against the app at `9a6ffdd2` (the audit head).

| # | change | file | today |
|---|---|---|---|
| 1 | per-script alphabet instead of `'a'..'z'` | `swipe/CtcEngineAdapter.kt` — `letterOf` returns `c.takeIf { it in 'a'..'z' }`, `buildMappedLayout` uses `FloatArray(26)`/`BooleanArray(26)` and returns null unless all 26 are `seen` | hard-codes 26 |
| 2 | per-script routing | `swipe/SwipeEngineRouter.kt` — only `isSwipeTypingSupportedForLayout` (QWERTY-Latin) or `isLatinScript(script)` reach `Engine.CTC` | `script="cyrillic"`/`"greek"` fall to geometric |
| 3 | per-language model asset | `CtcEngineAdapter.MODEL_ASSET` is a single constant | one model only |
| 4 | per-script language support | `CtcLanguageSupport.SUPPORTED` is a compiled-in `linkedMapOf` of en/fr/de/es(/it/pt/sv) | not extensible at runtime |
| 5 | per-script preset | `swipe/ctc/CtcScoringParams.kt` — `tunedRuCkdt` exists but `presetFor` can never return it | unreachable |
| 6 | fixture↔model↔preset triple per script | `CtcParityTest` | one row |
| 7 | trie width | `swipe/ctc/CtcLexiconTrie.kt` | **already done** — the 26-child clamp was removed and replaced by a constructor check against the emission-head width |
| 8 | `CtcLayout` | `alphabet: CharArray` + parallel centre arrays | **already generic** |

**The model's slot order IS the app's alphabet array.** Every Phase-O layout
json lists its letters in **codepoint-sorted order**, and emission column *c* is
`letters[c]`. The app's per-script `ALPHABET` must be that same string, in that
same order, or every decode is silently permuted. The strings are given per
script in §3.2.

**The geometry needs no app-side change.** `app_layout.py` replicates
`KeyboardGeometry.computeKeyRects` + `buildMappedLayout` exactly, and reproduces
`en_qwerty.json` from the app's own QWERTY XML to 4.7e-4 (§1.2) — so the
`layout_keys` the app computes at runtime for these layouts *is* the geometry the
models were trained on. Locale extra keys and the bottom/number/numpad rows do
not perturb it: extra keys land in corner slots (never `key0`), and the
normalisation box is built from letter-centre rects only.

### 3.2 Per-script wiring table

| script | layout XML (`src/main/layouts/`) | alphabet / slot order (codepoint-sorted) | K | lexicon the app must have | preset |
|---|---|---|---|---|---|
| ru | `cyrl_jcuken_ru.xml` | `абвгдежзийклмнопрстуфхцчшщыьэюя` | 31 | `langpack-ru.zip` (importable today) | `tunedRuCkdt` = 1.05 / **2.0** / 0.2 / 0.3734 / 0.9882 |
| el | `grek_qwerty.xml` | `αβγδεζηθικλμνξοπρςστυφχψω` | 25 | `langpack-el.zip` **with final sigma repaired** (§1.3) | CKDT preset, λ per §2.2 |
| uk | `cyrl_jcuken_uk.xml` | `абвгдежзийклмнопрстуфхцчшщьюяєі` | 31 | **none — must be built** (`build_wordlist.py --lang uk`, the `cyrillic` script gate already exists) | CKDT preset |
| bg | `cyrl_ueishsht.xml` | `абвгдежзийклмнопрстуфхцчшщъьюя` | 30 | **none — must be built** (`cyrillic` gate exists) | CKDT preset |
| mk | `cyrl_lynyertdz_mk.xml` | `абвгдежзиклмнопрстуфхцчшѓѕјљњќџ` | 31 | **none — must be built** (`cyrillic` gate exists) | CKDT preset |
| he | `hebr_1_il.xml` | `אבגדהוזחטיךכלםמןנסעףפץצקרשת` | 27 | **none — must be built**, and `build_wordlist._is_script_word` needs a **new `hebrew` branch** (0x0590–0x05FF); it currently `raise`s on any script but latin/greek/cyrillic | CKDT preset |

### 3.3 Two app fixes that must land before any of this

1. **`src/main/layouts/grek_qwerty.xml` declares `script="latin"`.** One-word
   fix to `greek` (matching `srcs/layouts/`). Until then the Greek layout is
   indistinguishable from Latin at the router.
2. **Final sigma in the Greek lexicon.** Either regenerate `langpack-el` with ς
   preserved, or apply the deterministic repair when the el trie is built:
   a word ending in `σ` gets that `σ` rewritten to `ς`. 25.7 % of the pack's
   words are affected. Without it, a Greek swipe model is scored against the
   wrong key, in the wrong row, for one word in four — and the *user* will swipe
   to ς because that is where Greek orthography puts it.

### 3.4 Per-script projection rules the app must mirror

The projection is applied to the lexicon **and** to anything compared against a
decode; the campaign's ALT_LAYOUT §3 policy. From `script_registry.py`:

* **all scripts** — lowercase; strip `- ' ’ ʼ ‘ \``.
* **el, he** — NFD, drop combining marks (`Mn`), NFC. Safe here because no
  letter's identity depends on a mark: Greek accents/diaeresis and Hebrew niqqud
  are not keys.
* **ru, bg, mk** — **no NFD** (it would decompose й into и + breve). Character
  folds instead: ru ё→е, ъ→ь; bg ѝ→и; mk ѐ→е, ѝ→и.
* **el only** — after mark stripping, word-final `σ` → `ς`.
* **uk** — no folds; words containing ї or ґ are **rejected as untypeable**
  (4.03 % of the vocabulary, §1.5). If the app wants them, it needs the
  corner-alias path, and that is a *different input mode* (flick), not a swipe.

### 3.5 What the Termux agent needs, per script, in one list

For each script: the layout XML is already in the app; the model, the fixture and
the preset are in `ctc/artifacts/`; the alphabet string is §3.2; the projection is
§3.4. What is *missing* is the lexicon for four of the six.

| script | model asset | golden fixture | preset | lexicon status | blocking app work |
|---|---|---|---|---|---|
| ru | `ru_synth_ch80_fp16w.onnx` | `ru_synth_ch80_fp16w_golden.json` | `tunedRuCkdt` | `langpack-ru.zip` exists (import) | §3.1 items 1–6 |
| el | `el_synth_ch80_fp16w.onnx` | `el_synth_ch80_fp16w_golden.json` | same numbers as `tunedRuCkdt` | `langpack-el.zip` exists, **needs the ς repair** | §3.1 items 1–6 **+ the `script="latin"` fix** |
| uk | `uk_synth_ch80_fp16w.onnx` | `uk_synth_ch80_fp16w_golden.json` | same | **must be built** (`build_wordlist.py --lang uk`) | §3.1 items 1–6 |
| bg | `bg_synth_ch80_fp16w.onnx` | `bg_synth_ch80_fp16w_golden.json` | same | **must be built** | §3.1 items 1–6 |
| mk | `mk_synth_ch80_fp16w.onnx` | `mk_synth_ch80_fp16w_golden.json` | same | **must be built** | §3.1 items 1–6 |
| he | `he_synth_ch80_fp16w.onnx` (**flagged**, §2.2) | `he_synth_ch80_fp16w_golden.json` | same | **must be built**, plus a `hebrew` branch in `build_wordlist._is_script_word` | §3.1 items 1–6 |

Sizes: every model is **589,406 B** as fp16w — a fifth of the shipped English
model's 2.91 MB. Six scripts is 3.5 MB of assets, which is the argument for
gating them behind the langpack import rather than bundling all of them.

**And the cheaper option, which the evidence supports:** items 1–6 of §3.1 plus a
layout and a trie get a non-Latin script to ≈ 76 real top-1 **with the model the
app already ships** (measured on Russian, §2.1). The per-script models add ≈ +1.6
on top of that (measured on Russian, p = 1.4e-4; inferred elsewhere). If the app
work has to be staged, stage the wiring first and the per-script models second —
the wiring is where all the accuracy is.

---

## 4. O3 — CLOSE

### 4.1 What Phase O delivered

* **An inventory** (§1) that answers "which non-Latin scripts can the app serve"
  with evidence rather than a guess, including four categories of *cannot* with
  the reason measured for each.
* **A validated geometry extractor** (§1.2) that reproduces the training frame
  from the app's own XML to 4.7e-4, retro-validating the ru model's deployability
  as a by-product.
* **Five new per-script models**, trained, gated, exported with fp16w ship bytes
  and golden fixtures (§2.2, §2.5, §2.6): el, uk, bg, mk, he.
* **A calibration** (§2.1) that tells anyone reading those five numbers exactly
  how much to trust them — which turned out to be the most valuable thing in the
  phase, because the answer is "less than you would have assumed".
* **Two app defects** found and quantified (§1.3), one of which would have
  silently corrupted a quarter of all Greek decodes.
* **Two registered arms refuted** with the rule applied as written (§2.1b,
  §2.1c), and **one campaign claim narrowed** by measurement (§2.1, point 3).

### 4.2 The ledger of what is NOT established

1. **No accuracy number for any of the five new scripts.** Only generator numbers
   and a Russian-derived expectation. Nothing here says Greek swipe typing works.
2. **Single seed everywhere.** The campaign's own resolution floor (~1 pt,
   Phase C) exceeds several of the deltas in §2.3.
3. **Four lexicons are not the app's.** uk/bg/mk/he ride wordfreq lists without
   `build_wordlist.py`'s oracles; a real pack would differ in word selection
   (though not in frequency scale, which is replicated byte-exactly).
4. **No on-device anything.** No latency, no memory, no instrumented test.
5. **Hebrew is below the registered gate** at the adopted preset and is exported
   flagged rather than withheld, because the gate is defined on a probe this
   phase discredited — withholding it would over-trust the same probe in the
   opposite direction.
6. **The ru synthesis holdout reuses donor traces** the ru model trained on
   (paired with different words) because `cyrillic_synth.py` predates the 90/10
   donor split. This inflates the ru holdout column and therefore *understates*
   the sign reversal — the finding is conservative, not flattered.
7. **A Phase I-B runtime cache was clobbered** and disclosed rather than quietly
   repaired: `cache_ru/train_synth.npz` was restored byte-identical from
   `cache_ru_synth/`, `cache_ru/val.npz` (the 5,000-row real-data selection val)
   was not regenerated. `python3 prepare_yandex.py --val-rows 5000` rebuilds it,
   but that command also rewrites the vendored `layouts/ru_jcuken_default.json`,
   so run it and then check `git status` before committing anything.

### 4.3 What a Phase P should do first, in order

1. **Fix the generator's word draw** — weight by corpus token frequency instead
   of the compressed dictionary rank, so the synthesis has a realistic length mix
   (§2.1). This is the cheapest change with the clearest expected payoff, and it
   is testable against real Russian.
2. **Re-run the ru calibration after (1).** If a length-correct generator
   restores rank-preservation against the real corpus, the synthesis holdout
   becomes a usable probe and every corpus-less script gets a trustworthy number
   for the first time. If it does not, stop reporting holdout numbers at all and
   report only the capacity-matched zero-shot delta.
3. **Collect real data for one non-Latin script.** Everything above is
   downstream of the fact that ~13 of the ~15 available points live in real data
   (`DATASET_SCOUT.md` §4.4 makes the case that causing collection at
   `swipe.futo.org` is the only clean route).
4. **Do not** re-run capacity, warm-starting, or per-script λ sweeps against a
   synthesis holdout. All three are now measured to be probe artefacts.
