# Dataset scout — every additional swipe corpus we could train on

**Date:** 2026-08-10. **Scope:** find data beyond the current pool (FUTO swipe-1
939,550 + How-We-Swipe full release 84,612 basic-hygiene traces; Yandex = eval-only
per `YANDEX_LICENSE_RESEARCH.md`; `futo-org/swipe-negatives` = model-output rule).
**Nothing here was trained.** Every number below is either measured locally on the
files themselves or fetched from the source API today; second-hand claims are
labelled as such.

---

## 0. Verdict up front

1. **The "leon" dataset is FUTO swipe-1 with every row duplicated exactly 10×.**
   Measured, not inferred: the 1,734,660 local rows collapse to **173,463 unique
   traces**, 173,460 of them at multiplicity exactly 10 (`§1`). 95.53 % of those
   unique traces are **bit-exact members of our own FUTO pools**; the residual
   4.47 % are FUTO rows our normaliser dropped as `too_rare`. Zero new human
   swipes. And **1,352 of our 12,299 holdout traces (11.0 %) sit inside it**, each
   ×10. **Verdict: do not train on it, not even as a measured arm** — the arm would
   measure an 11 % leak, and its unique content is a strict subset of data we
   already hold at 4.4× the scale. The "unknown licence / 30-point" description in
   `DATA_TIERS.md` §5 was wrong on both counts and is corrected in §1.4.
2. **The real find is FUTO swipe-2/3/4/5: 175,870 rows added 2026-06-15, MIT, same
   schema, same frame, and structurally impossible to contaminate our holdout**
   (§2). swipe-4 is *purpose-built confusable words*; swipe-3 maximises unique
   words incl. deliberate misspellings; swipe-5 is 11 layouts / 8 languages with
   official geometries, including **11,805 clearflow and 1,058 kasroz rows on
   layouts we have never seen**. This is the top trial arm.
3. **The local WordGesture-GAN pull is not usable.** 49,228 traces, all GAN
   outputs scraped from a demo endpoint (no licence), one trace per word, fixed
   128 points, and its frame needs a fitted affine after which endpoint accuracy
   is still **start-hit 0.512 / end-hit 0.520** vs the 0.79–0.91 real-corpus band
   (§3). Our own residual-transplant synth is better licensed *and* geometrically
   better. **Verdict: drop it.**
4. **There is no cleanly-licensed real human swipe corpus in any non-Latin script,
   anywhere.** Exhaustively searched (§4). Yandex remains the only real Cyrillic
   data and remains eval-only. The multi-script programme stays on the synthesis
   path — with one new option: swipe.futo.org is still open to new
   languages/layouts, i.e. **the only route to clean non-Latin data is to help
   cause its collection.**

---

## 1. The "leon" dataset — identified, measured, rejected

### 1.1 What it is

`/home/will/git/swype/cc-old-data/train_leon_filtered_norm.jsonl` derives from the
HuggingFace dataset **`leonweber/swipe`**, whose git clone is sitting next to it at
`cc-old-data/leonweber/swipe_dataset/` (remote `https://huggingface.co/datasets/leonweber/swipe`,
5 commits, all "Upload dataset", 2025-06-09). Owner is **Leon Weber-Genzel**, an NLP
researcher (biomedical NLP / bigscience / bigbio), not a keyboard person. There is
no README prose, no paper, and **no `license` field in the card metadata — none
declared**.

Its purpose is visible from his companion model
`leonweber/bge-base-en-v1.5-futo-swipe-base` (2025-06-14, `dataset_size:9395500`,
MultipleNegativesRankingLoss): he built a *retrieval-style* swipe decoder that ranks
candidate words from a masked sentence plus a key-letter string. That is why the
schema carries four columns FUTO's does not — `masked_sentence`,
`trajectory_sampled`, `trajectory_sampled_keys`, `trajectory_word` — on top of
FUTO's exact 12 (`id, session, timestamp, word, canvas_width, canvas_height,
orientation, data[{t,x,y}], sentence, word_idx, potentially_invalid_sentence,
distance`).

### 1.2 It is futo swipe-1 ×10 — the arithmetic and the hashes

| | leonweber/swipe | futo-org/swipe.futo.org swipe-1 | ratio |
|---|---|---|---|
| train | 9,395,500 | 939,550 | **×10** |
| validation | 542,690 | 54,269 | **×10** |
| test | 499,700 | 49,970 | **×10** |

The web sweep confirmed row-level identity: leonweber rows 0–3 are all futo
swipe-1 `id=65`, `session anon-session-6f52996c…`, word "The",
`timestamp 1724390287154`, `distance 17.02327802608763` — byte-identical, with
identical raw `data`. Only the *derived* columns differ per copy: each of the 10
copies carries an independent random 30-point resampling of the same trajectory and
its nearest-key string. **The 10× is not augmentation of the trajectory; it is ten
re-rolls of a preprocessing column over one unchanged trace.**

Locally measured on the full 1,734,660-row `filtered_norm` train+val
(`scratchpad/leon_overlap.py`, bit-exact key over `word` + every `(t,x,y)`):

```
rows                      : 1,734,660
unique traces (t,x,y)     :   173,463
duplicate-multiplicity    : {10: 173,460 ; 20: 3}     ← exactly ×10, no exceptions
unique traces (geom only) :   173,463                 ← geometry-dedup finds nothing extra
unique sessions           :     2,447
unique words              :    22,233
```

The corroborating tell is in `filtered/leon_filtered_norm_stats.json`: **every
single drop counter is a multiple of 10** (`invalid_sentence 47,090 ·
invalid_word 14,240 · invalid_length 48,420 · not_portrait 90,460 ·
too_long_duration 38,050 · too_slow_speed 43,930 · too_fast_speed 720 ·
trace_too_short 8,260 · trace_too_long 3,350`), and `too_rare` is **1** — because
the word-frequency count ran over the ×10 corpus, so nothing could be rare.

### 1.3 Overlap with what we already have — and with our holdout

Measured against our own pools (same normaliser, so a bit-exact key is a valid
identity test):

| check | result |
|---|---|
| leon unique traces ∈ FUTO pools (688,023 + 76,448) | **165,717 / 173,463 = 95.53 %** |
| leon unique traces ∈ HWS train pool (54,378) | **0** |
| leon unique traces in **neither** | 7,746 |
| **our val+test holdout traces found inside leon (exact `t,x,y`)** | **1,352 / 12,299 = 11.0 %** |
| same, geometry-only key (time-free) | 1,352 (identical — no time-resampled variants) |

The 7,746 residual is not new data. Their ids are all in FUTO's id space
(371 – 280,530), they are **7,400 distinct words at 1–2 traces each** — `acosta`,
`stifled`, `guinean`, `berninis`, `tyndale`, `czechs`, `morimoto` — i.e. exactly
the rare-word tail our own `MIN_WORD_FREQ = 3` gate discarded (41,585 rows) and
leon's ×10-inflated frequency count did not.

So the *entire* informational content of the leon corpus is
**173,463 FUTO swipe-1 traces ⊂ the 939,550 we already have in full**, at 2,447
sessions instead of 10,889.

### 1.4 Two corrections to `DATA_TIERS.md` §5

`DATA_TIERS.md:193` says the file was deliberately excluded as "30-point
standardised, unknown licence". Both halves are wrong:

* **Not 30-point.** The local jsonl was built from the raw `data` column, not the
  `trajectory_sampled` column. Measured point counts on the file: **min 8, median
  57, max 470**, with genuine irregular device timestamps (`t: 0, 7, 24, 43, 63,
  78 …`). The 30-point standardisation is a property of the `trajectory_sampled`
  column we never touched, and the "exactly 30 points" line in
  `leonweber/FINAL_DATASET_ANALYSIS_REPORT.md` refers to that column.
* **Licence is not unknown in substance.** The repo declares nothing, but it is an
  unlicensed *redistribution of MIT-licensed FUTO data*, so the underlying data
  terms are MIT and unproblematic. The reason to refuse it is contamination and
  redundancy, not licensing.

The exclusion was right; the stated reasons were not. **The 30-point timing concern
that motivated the "is it worth trying anyway?" question is moot** — the file has
real timing. The featurizer question is therefore not even reached.

### 1.5 "Is it worth trying anyway?" — no, and here is the design that would be needed

For completeness, the arm that *would* be defensible costs more than it can return:
you would have to (a) drop all 1,352 leaked holdout traces and their siblings,
(b) dedup ×10 down to 173,463, (c) subtract the 165,717 already in T1–T4 — leaving
**7,746 rare-word FUTO traces we deliberately filtered out**, which is a re-run of
the `MIN_WORD_FREQ` gate, not a new corpus. Phase A already measured the FUTO
lexical gates as costing ~11.5 k rows for no gain, and this campaign has now
recorded **four** exclusion-style curation negatives. A 7,746-row rare-word arm is
below the ~1 pt Phase-C resolution floor by an order of magnitude.

> Note on the general principle, since it will come up again: our features are
> `[2, 64]` — **x and y only** (`featurize` → `resample_to_60hz(t)` →
> `resample_fixed` index-uniform to 64). There are no dx/dy/speed/angle channels.
> Timing enters *only* through the 60 Hz step, which converts dwell into point
> density along the path. So a corpus with synthetic uniform `t` does not merely
> lose a few features — it erases the single channel by which corner dwell reaches
> the model, and its resampled geometry becomes arc-length-uniform rather than
> time-uniform. That is a real defect for any future 30-point-style source, it just
> does not apply to this file.

---

## 2. FUTO remainders — the actual opportunity

### 2.1 swipe-1 has NOT grown; four new configs appeared

Verified today against the HF API (`/size`, `/api/datasets`, commit list):

| config | train rows | added | contents |
|---|---|---|---|
| **swipe-1** | **939,550** (+ test 49,970, validation 54,269) | 2025-03-11 | our snapshot — **unchanged**, LFS oids identical to the original upload |
| **swipe-2** | **28,095** | 2026-06-15 | "more informal language" — 8 k Amazon-Reviews-2023 + 8 k tv_dialogue sentences |
| **swipe-3** | **38,228** | 2026-06-15 | "maximize unique words"; 30 % random-5-word, 70 % Urban Dictionary / OpenWebText; **deliberate misspellings, swiped as spelled** |
| **swipe-4** | **50,300** | 2026-06-15 | **"easily confusable words"** — sentences are sequences of swipe-negatives' confusables |
| **swipe-5** | **59,247** | 2026-06-15 | 11 layouts / 8 languages + 2,708 dual-finger nintype rows |
| | **175,870 new** | | |

Licence: card `license: mit`, repo `LICENSE` = standard MIT, "Copyright (c) 2025
FUTO". Last modified 2026-06-25 (added a paper link — **"FUTO Swipe:
Layout-Agnostic Neural Swipe Decoding", Miller & Kostarevas, arXiv 2606.25247**).
Note their *model* release `futo-org/futo-swipe` is under the FUTO Model Weights
License and is still off-limits under our §0 recipe rule; the **data stays MIT** —
the weights licence explicitly excludes datasets.

### 2.2 Format — drop-in, verified on a live row

swipe-2/3/4 carry 11 columns (swipe-1 minus `potentially_invalid_sentence`);
swipe-5 carries 14 (adds `language`, `dual_finger`, `layout`). A live swipe-3 row:

```json
{"id":1,"session":"anon-session-cbbd4f27…","timestamp":1772228161870,"word":"attend",
 "canvas_width":426.0,"canvas_height":170.40000915527344,"orientation":"portrait-primary",
 "data":[{"t":1772228161620,"x":0.11971830985915492,"y":0.6255866775734765}, …],
 "sentence":"attend a very large baptist church on wed nights that bemoans",
 "word_idx":0,"distance":74.3253582909639}
```

Same **canonical letter-area frame** (`x,y` already normalised over the letter
area), same per-point epoch-ms `t`, same `canvas_height ≈ 170`. Our existing
normaliser applies unchanged. Note: **no filtering has been applied to swipe-2..5**;
FUTO's own README says to gate on the `distance` field (dual-finger rows carry the
sentinel `distance = 100004`).

### 2.3 Contamination risk — structurally near-zero, and measured

These are *new collection runs* conducted after our holdout was carved from
swipe-1 + HWS, so exact-trace overlap is impossible by construction. Session-id
overlap was checked directly: 68 distinct session ids sampled across seven offsets
in each of the four configs, grepped against the raw 5.1 GB swipe-1
`train.jsonl` → **0 hits**. Session namespaces are per-run.

The unresolvable residual is *person*-level: the same volunteer may have joined
several runs under different anonymous session ids. That is the same class of leak
the T3 benchmark tier already accepts by design (`PHASE_A.md` §5), and it is
strictly smaller here than inside swipe-1 itself.

**One real hazard.** swipe-5's dvorak / azerty / qwertz / german / spanish rows
**are our alt-layout evaluation set** (`ALT_LAYOUT_EVAL.md` §1: 2,809 / 2,542 /
1,402 / 2,594 / 2,029 API rows). Training on them destroys that measurement. Any
swipe-5 arm must either exclude those five layouts or carve a disjoint split and
re-baseline the alt-layout table.

### 2.4 swipe-5 layout census — verified row-by-row via the filter API

| layout | rows | status for us |
|---|---|---|
| qwerty | 33,526 | **free to train** — not in any eval |
| **clearflow** | **11,805** | **free, and a layout we have never seen** |
| dvorak | 2,809 | ⚠ alt-layout eval set |
| azerty | 2,641 | ⚠ alt-layout eval set |
| german | 2,594 | ⚠ alt-layout eval set |
| qwertz | 2,283 | ⚠ alt-layout eval set |
| spanish | 2,029 | ⚠ alt-layout eval set |
| **kasroz** | **1,058** | **free, unseen layout** |
| toki_pona | 450 | free, unseen |
| lithuanian_qwerty | 50 | too small |
| shavian | 2 | negligible |
| *dual_finger = 0* | 56,539 of 59,247 | 2,708 two-finger rows to exclude |

Languages: en 47,364 · de 5,207 · fr 3,124 · es 2,029 · pl 1,019 · tok 450 ·
lt 52 · shaw 2. Official geometries for **all eleven** layouts are published at
`swipe-5/layouts/*.json` in the same `cx/cy/rx/ry` normalised form as our
`en_qwerty.json` — so nothing has to be reconstructed. Each of swipe-2..5 ships as
a **single unsplit JSONL** (`swipe-N/swipeN.jsonl`) plus a `sentences/` directory;
swipe-5 adds `layouts/`.

*(The 11,805 clearflow figure is my own `where "layout"='clearflow'` count against
the filter API today; the OSS sweep independently reported 11,028, presumably after
some gating. Either way it is the second-largest layout in the config and an order
of magnitude more real alt-layout data than any single layout in our current eval
suite.)*

**Parsing hazard — schema fork.** For `dual_finger = 1` rows (2,708 of 59,247) the
`data` field stops being an array of points and becomes
`{"L": [[{x,y,t},…],…], "R": […]}` — a list of *separate strokes* per hand, with
the sentinel `distance = 100004`. Any loader must branch on this or it will crash
or, worse, silently mis-parse. Simplest correct policy for a first arm: filter
`dual_finger = 0` and drop the 2,708 (a genuinely different input mode our model
does not represent).

**Why this matters more than raw count.** Phase H's whole layout-agnosticism story
rests on `layout_aug` *synthetic* geometry warping, validated against alt-layout
corpora we only ever **evaluated** on. clearflow (11,805) + kasroz (1,058) +
toki_pona (450) are **real human swipes on layouts no arm has ever trained on**, and
they are not part of any committed eval. That is the first chance to test
"synthetic layout-alt vs real alt-layout data" without burning a benchmark.

### 2.5 Expected value, by weakness

| new pool | rows | weakness it addresses |
|---|---|---|
| swipe-4 confusables | 50,300 | the decoder's top-1-vs-top-3 gap: 92.24 t3 against 87.66 t1 means the beam *has* the word and ranks it second. Confusable-pair training data is the most direct attack on that 4.6 pt spread we have ever had access to. |
| swipe-3 unique words + misspellings | 38,228 | long/rare-word tail; OOV robustness; the misspelling rows are the only data anywhere that teaches "swipe what is written, not what is meant" |
| swipe-2 informal | 28,095 | vocabulary/register mismatch — swipe-1's sentences are formal, and the app's users are not |
| swipe-5 qwerty | 33,526 | plain scale on the primary layout, from a fresh contributor cohort (helps the known **HWS-half deficit**: HWS t1 81.09 vs FUTO t1 94.27) |
| swipe-5 clearflow/kasroz/toki_pona | 13,313 | real alt-layout training data — the untested half of the layout-agnostic claim |

---

## 3. WordGesture-GAN — the 49,228 local traces, assessed and rejected

`/home/will/git/swype/neural-swipe-typing/scripts/curlWGG/wgg_swipes_normalized.jsonl`,
**49,228 rows**, 317 MB.

**Provenance (from the scripts, not from recollection).** `fetch_wgg_swipes.py`
POSTs `{"word": w, "std_dev": "1"}` to `http://wordgesturegan.com/gesture_from_word`
at 100 req/s for the 49,296 words of `en_enhanced.txt` (67 failures: all 26 single
letters plus non-ASCII). These are **model outputs from a live demo endpoint**, not
recorded human swipes.

**Licence: none, and worse than none.** No terms were offered by the endpoint; the
site is now HTTP 500 / connection-refused, so the traces cannot even be
re-derived. The CHI 2023 authors released no data, no generator and no official
code. On top of that it is squarely inside our **model-output rule** — the same
principle that keeps `futo-org/swipe-negatives` out (those were mined with the
SwipeALot encoder). WGG was itself trained on a filtered ~38 k subset of How-We-Swipe,
so it is a model derivative of a corpus we already hold in full.

**Technical assessment (measured on 4,000–6,000 rows against `en_qwerty.json`):**

| property | value | comment |
|---|---|---|
| points/trace | **exactly 128, always** | fixed-length GAN output |
| traces/word | **exactly 1** (4,000 rows → 4,000 distinct words) | no user variation, no repetition — the *opposite* of what training needs |
| coordinate frame | not ours | raw endpoint-proximity **start-hit 0.008**, i.e. the falsification-control level from `PHASE_I_DATA` §4 |
| frame after fitted affine | `x: s=1.086 o=−0.045`, `y: s=1.221 o=+0.008` | needed a real correction, so the `[-1,1]→[0,1]` conversion in `normalize_wgg_swipes.py` is wrong |
| endpoints after that affine | **start-hit 0.512 / end-hit 0.520**, d 0.068 / 0.073 | real-corpus band is 0.79–0.91 start; our own synth Cyrillic already hits 0.710/0.656 |
| duration | median 2,100 ms (p99 3,689) | ~2× the HWS median of 1,113 ms — slow, smooth, un-human |

The "46 % corrupt" note in `neural-swipe-typing/memory/history2.md:283` is about a
**different, earlier** 86,061-sample synthetic mixture (negative pixel coordinates,
39,896 bad-timing rows), not this pull — but the frame defect measured above is the
same family of bug and this pull has it too.

**Verdict: drop it.** Unlicensed model output, one sample per word, fixed 128
points, a frame that needs fitting and is *still* worse than the synthetic pipeline
we already validated in Phase H/I-B. Our residual-transplant synth
(`cyrillic_synth.py`, `layout_aug.warp_path`) dominates it on licence, on endpoint
fidelity, and on being reproducible.

---

## 4. Everything else — the systematic sweep

Two independent sweeps were run (HF Hub census; academic/repository sweep across
arXiv, ACM, OSF, Zenodo, figshare, Dryad, Harvard Dataverse, IEEE DataPort, Kaggle,
GitHub). Findings below are attributed; the two agreed everywhere they overlapped.

### 4.1 Real human corpora (the complete list)

| corpus | traces | format / frame | timestamps | licence | script |
|---|---|---|---|---|---|
| **futo-org/swipe.futo.org** | 1,262,000 (swipe-1..5) | jsonl/parquet, **normalised letter-area** + official layout JSONs | per-point epoch ms | `mit`, "Copyright (c) 2025 FUTO" | Latin (en/de/es/fr/pl/lt/tok/shaw) |
| **How-We-Swipe** (OSF `sj67f`) | 109,275 valid / 1,338 users / 11,227 words | per-user `.log` + `.json`, **integer px relative to the keyboard box** | ms on every touchstart/move/end, + touch-ellipse radii | OSF licence record: **MIT**, © Leiva/Kim/Cui/Bi/Oulasvirta 2021 | English |
| **Yandex Cup 2023 NeuroSwipe** | ~6.0 M (labels on val only) | integer px + key hitboxes | yes | **none granted**; ToU bars commercial (`YANDEX_LICENSE_RESEARCH.md`) | Cyrillic ЙЦУКЕН |
| **TU Delft VR word-gesture** (4TU DOI `10.4121/2e0d26b2…v1`) | **3,129 / 21 users** | plain-text logs, `POINTS:` = **normalised 0–1, pre-resampled**, median 31 pts | **none per-point** (one epoch-ms per gesture) | `"license": {"name": "CC0"}` | English, **mid-air VR** |

That is *all* of it. Everything else is synthetic, is metadata, or is a model.

**How-We-Swipe layout note (new, useful).** The keyboard geometry is not shipped
with the OSF release, but the academic sweep reconstructed it empirically and it
matches `luileito/swipetest`'s `js/keyboard-impl.js`: `key_w = kb_w/10`,
`key_h = kb_h/3.5`, row centres at `y = 1/7, 3/7, 5/7`, row x-offsets `0 / 0.5 /
2.0` keys. That is an independent handle on the ~0.064 systematic Y offset between
our HWS and FUTO halves (`DATA_TIERS.md` §1, `CoordAlignmentVsFuto.md`) — the
correction was derived once and never applied, and this gives a second, principled
derivation of it.

**Our HWS headroom is genuinely exhausted**: Phase I-B already consumed the full
1,338-user release (84,612 basic-hygiene traces) and measured englishLevel
filtering as a consistent negative.

### 4.2 Synthetic / derived / not-trajectories (all rejected)

* `futo-org/swipe-negatives` — 143,794 rows, `apache-2.0` (**note: a different
  licence from the corpus's MIT**), but it is `word → top-128 confusables +
  neg_sims`, **no trajectories at all**, mined from `dleemiller/SwipeALot-base`
  embeddings. Model-output rule stands. *However*: it is exactly the hard-negative
  resource a lexicon **reranker** would want, and swipe-4 is its real-swipe
  counterpart. Worth revisiting if a reranker is ever built — see §5 note.
* `kuroqg/wgk-ja-trajectories` — 256 GB, `cc-by-sa-4.0`, **synthetic**
  minimum-jerk Japanese over 4 layouts from Wikipedia. Only new 2026 release
  besides FUTO's. Synthetic-from-a-generator is what our own pipeline already does,
  better validated. **>5 GB — flagged, not downloaded.**
* `AI4Bharat/Indic-Swipe` v1 — 193,658 + 104,412 words, 7 Indic languages,
  **synthetic** minimum-jerk `.xlsx`, **no licence stated**, Google-Drive hosted.
  **Indic-Swipe-v2** (pushed 2026-07-13, MIT) is an Android IME + synthetic
  *generator*, not a corpus.

  > **The FUTO paper's negative result on this, and why it does not indict our
  > synth.** FUTO report IndicSwipe-style synthetic training as ineffective, with
  > the diagnosis **"insufficient motor noise"**. That is a statement about
  > *minimum-jerk generators*, which produce an idealised velocity profile and then
  > add parametric noise. Our pipeline is a different class: `cyrillic_synth.py` /
  > `layout_aug.warp_path` **transplant real human residuals** from donor traces
  > onto new polylines, so the motor noise is measured, not modelled — and Phase I-B
  > measured the result at in-dict t1 **76.21** on real Cyrillic with no real
  > Cyrillic anywhere in its training path. The FUTO finding is corroborating
  > evidence for that design choice, not against it. It is also a reason not to
  > bother with `kuroqg/wgk-ja-trajectories` or IndicSwipe: both are exactly the
  > generator class FUTO measured as failing.
* `dleemiller/SwipeALot-base` — apache-2.0 **model**, not data.
* Kaggle NeuroSwipe mirrors — `sharthz23` 5.18 GB ("Unknown") and `e0xextazy`
  1.84 GB (**relabelled "Apache 2.0"**). A third-party uploader's relabel cannot
  cure Yandex's underlying terms; treat as unreliable provenance, not a grant.
  Nothing changes the eval-only verdict.

### 4.3 Chased to the end and confirmed unavailable

* **WordGesture-GAN** (CHI 2023) — no traces, no generator, no official code; site
  down. Only GitHub hit is a third-party re-implementation with no data.
* **Gesture2Text** (arXiv 2410.18099) — nothing released. Its "Mobile Phone WGK
  Dataset" (95,649 samples) **is How-We-Swipe**; its four XR sets (~162 k) and 100 k
  synthetic set are internal to Meta Reality Labs.
* **Shen et al. AdaptiKeyboard** AR corpora — called "public" by Gesture2Text; no
  resolvable download exists.
* **SHARK² / Kristensson & Zhai lineage** — **never deposited any public corpus.**
  ShapeWriter died post-acquisition; the Bi/Zhai Stony Brook lineage consumes data
  and does not release it. The `aaronlaursen/SHARK2` GitHub repo referenced in our
  old `Gesture-Keyboard-Traj-Gen/find_real_datasets.py` is an implementation, not a
  corpus.
* **How-We-Type** (Aalto, Jiang/Oulasvirta) — **tap typing only, zero swipe
  component**; the project FAQ states the gesture condition is not covered. Not
  applicable.
* **Other Leiva datasets** — `G3` is unistroke *pen* gestures, CC-BY 3.0, wrong
  modality. Nothing else of his is word-gesture.
* Google-era corpora (Reyal / Quinn / Alsharif) — never released.
* Exhaustive negatives, do not repeat: Kaggle API (12 queries) → only the two
  NeuroSwipe mirrors. HF (50+ hits across name/full-text/tag/author search) → only
  the futo pair, leonweber, and the synthetic ja set. Zenodo, figshare, Dryad,
  Harvard Dataverse → **zero** word-gesture records. OSF → only `sj67f`. IEEE
  DataPort → only BB-MAS (biometrics). GitHub repo/code/topic search → engines and
  IMEs only.

### 4.4 Non-Latin scripts — the answer is no

**No real human swipe corpus in any non-Latin script exists under a clean licence.**
Searched and empty: Arabic, Hebrew, Greek, Thai, Hangul, Devanagari (real),
Japanese (real), Chinese pinyin. Yandex is the only real non-Latin corpus and it is
eval-only. Indic-Swipe is synthetic and unlicensed; the ja set is synthetic.

Two consequences:

1. The Phase I-B conclusion holds unchanged — **synthesis is the multi-script
   path**, and it was already measured at in-dict t1 76.21 with no real data
   anywhere in the pipeline, closing to ~90 once real data exists.
2. **A new lever appeared.** **FUTO is the only open-source keyboard project that
   collects *and publishes* traces**, the collection is **still live** (both
   `swipe.futo.org` and `swype.futo.org`, now multilingual and two-finger), and
   swipe-5 proves they will ship a new script on request — it contains **2 Shavian
   rows**, i.e. a script added because someone asked. The only route to clean
   non-Latin real data is to cause its collection there. That is a people-action,
   not a download; it is flagged here because it is the highest-leverage item in
   this whole document and has no engineering substitute.

   Why no one else can supply it (first-hand from the OSS sweep):

   | project | shape-writing status |
   |---|---|
   | **FUTO** | collects and publishes — the only one |
   | HeliBoard | glide comes from a **closed Google `.so`** the user side-loads at runtime; GPL app, nothing bundled, nothing collected |
   | OpenBoard | glide typing is an **unimplemented TODO** |
   | FlorisBoard | geometric SHARK²-lineage decoder (resample-200, bbox-normalise, Gaussian PDFs, `PRUNING_LENGTH_THRESHOLD = 8.42`) — algorithm only, no corpus. *Citation correction for our own notes:* that lineage is **Kristensson & Zhai** (via the AnySoftKeyboard PR #1870 heritage), **not** Bi & Zhai. |
   | AnySoftKeyboard | corner-matching heuristic, no corpus |
   | Thumb-Key / Simple Keyboard / Unexpected-Keyboard | no shape writing at all |

---

## 5. Ranked trial-arm shortlist

Recipe assumed frozen per Phase G/H (`resbn:80:1,2,4,8`, embed_hid 96, batch 256,
lr 3e-3, coupled affine sampler, layout-alt p 0.5, seed 1234) so arms are
comparable to the control. The **~1 pt Phase-C resolution floor** applies to every
one of these — anything expected to move less than that needs multi-seed or should
not be run.

| # | arm | data | cost | what it tests | expected |
|---|---|---|---|---|---|
| **1** | **`swipe234`** | T3 pool + swipe-2/3/4 (116,623 raw → ~95 k after `distance` gating + our normaliser) | 1 fetch (~0.6 GB) + 1 train | Does 10 % more data, deliberately enriched in **confusables, misspellings and informal register**, close the t1↔t3 gap? | The best-motivated arm in the campaign. Contamination-free by construction. If any single arm moves t1 by >1 pt, this is it. |
| **2** | **`swipe5-qwerty`** | + swipe-5 qwerty, `dual_finger=0` (~32 k) | fetch + train (can fold into #1) | Fresh-cohort scale on the primary layout; the HWS-half deficit is a cohort-diversity problem and this is a new cohort | Modest but cheap; **fold into #1 as `swipe2345q`** unless you specifically want to attribute the delta. |
| **3** | **`realalt`** | + swipe-5 **clearflow 11,805 + kasroz 1,058 + toki_pona 450**, holding out 20 % of each | fetch + train + a new eval split | **Real** alt-layout training data vs our synthetic `layout_aug` — the untested half of the layout-agnosticism claim, on layouts that are in no committed eval | The most *scientifically* valuable arm. Does not touch the dvorak/azerty/qwertz/german/spanish eval set. |
| **4** | HWS frame correction | existing data | 0 fetch, 1 train | Apply the ~0.064 Y offset (now independently re-derivable from the `swipetest` geometry in §4.1) before featurizing the HWS half | Zero-data-cost attack on the 13-point FUTO-vs-HWS gap. Cheapest ratio in the list. |
| **5** | *(conditional)* confusable reranker | swipe-negatives 143,794 + swipe-4 | design work | Only if a lexicon reranker is ever built — and only after resolving whether swipe-negatives' SwipeALot provenance trips the model-output rule (**it currently does; do not use it without an explicit ruling**) | Parked. |
| — | ~~leon~~ | — | — | — | **Rejected** (§1): 11 % holdout leak, 95.5 % redundant, residual is 7,746 rare-word rows we filtered on purpose. |
| — | ~~WordGesture-GAN~~ | — | — | — | **Rejected** (§3): unlicensed model output, 1 trace/word, fixed 128 pts, endpoints at 0.51 after a fitted affine. |
| — | ~~Indic-Swipe / wgk-ja~~ | — | — | — | **Rejected**: synthetic; our own synth is validated and better licensed. |
| — | ~~TU Delft VR~~ | — | — | — | Out-of-domain (mid-air VR), 3,129 rows, no per-point timing. CC0 and interesting as a *generalisation curiosity eval*; useless for training. |

**Recommended order:** run **#1 merged with #2** as one `swipe2345q` arm first — it
is one fetch, one training run, no eval is put at risk, and it is the only arm with
a first-principles reason to move the metric that has been stuck. Then **#4** (free).
Then **#3**, which needs a new held-out split defined before it can be trained.

**Do not download** `kuroqg/wgk-ja-trajectories` (256 GB) or either Kaggle NeuroSwipe
mirror. The swipe-2..5 fetch is ~0.6 GB and is the only download this document
recommends.

---

## 6. Method notes / reproducibility

* Leon measurements: `scratchpad/leon_overlap.py` (bit-exact blake2b key over
  `word` + every `(t,x,y)`; a geometry-only key was run in parallel and found no
  additional matches, i.e. there are no time-resampled variants). Full pass over
  1,734,660 leon rows × 764,471 FUTO pool rows × 54,378 HWS rows × 12,299 holdout
  traces.
* FUTO counts: `datasets-server.huggingface.co/size` per config,
  `/filter?where="layout"='…'` per layout, `/first-rows` for the schema,
  `huggingface.co/api/datasets/futo-org/swipe.futo.org` for `cardData.license`.
  All fetched 2026-08-10.
* Session-disjointness of swipe-2..5 vs swipe-1: 68 session ids sampled across
  seven offsets in each config, `grep -F -f` against the raw 5.1 GB swipe-1
  `train.jsonl` → 0 hits.
* WGG frame: least-squares fit of first/last trace points against
  `en_qwerty.json` first/last-letter key centres over 6,000 traces, then the
  `PHASE_H` §2.3 endpoint-proximity metric re-run under the fitted affine.
* §4 is the product of two independent web sweeps; where they are the sole source
  the text says so.
