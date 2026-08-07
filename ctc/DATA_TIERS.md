# Data provenance, contamination audit, and the training tiers

Status: **data built, nothing trained yet.** Everything below was verified against the
files themselves — no claim here rests on a script's comments or on recollection.

---

## 1. Provenance of the canonical `*_hwsfuto.jsonl`

The canonical splits are a **deliberately balanced 50/50 merge** of two corpora, not a
random subsample of FUTO:

| split | rows | HWS half | FUTO half |
|---|---|---|---|
| `train_hwsfuto.jsonl` | 110,876 | 55,438 | 55,438 |
| `val_hwsfuto.jsonl` | 9,918 | 4,976 | 4,942 |
| `test_hwsfuto.jsonl` | 2,400 | 1,183 | 1,217 |

Verified by hashing every canonical row against the two source pools: **100 % of rows
are accounted for** (0 rows matched neither pool). Build chain:

```
HF futo-org/swipe.futo.org  train.jsonl (939,550)
  -> filter_and_normalize_dataset.py   -> 764,473 kept
  -> ~90/10 pool split                 -> train_futo_filtered_norm.jsonl  688,025
                                          val_futo_filtered_norm.jsonl     76,448
How-We-Swipe swipetraces/*.log (1,052 logs)
  -> process_swipe_logs.py (drops is_err=1)  -> 62,880
  -> drop len(word)==1                       -> 61,597
  -> pool split                              -> 55,438 train / 6,159 val
create_combined_dataset.py (seed 42): FUTO down-sampled to match HWS 1:1
  -> train_combined.jsonl == train_hwsfuto.jsonl (byte-identical)
  -> val_combined.jsonl -> head-2400 = test, tail-9918 = val
```

### Coordinate frame — verified, not assumed

The HF corpus coordinates are **already the canonical letter-area frame**. Proven by
matching 50 canonical rows to HF corpus rows on an affine-invariant key
(word, n_points, duration) and then comparing values:

* per-row least-squares fit of `local = s·hf + o` gives **s = 1.0000000000,
  o = 0.0000000000, residual ≤ 8.9e-16** for both x and y across all 50 pairs;
* **50/50 are bit-exact** on every x and y value.

So the transform is: `x, y` copied verbatim; `t → t − t[0]` (float ms); `word → lower()`.
No canvas normalisation, no crop, no aspect correction — and no `4/3` constant exists
anywhere in the pipeline. Independently, FUTO's own `swipe-5/layouts/qwerty.json` is
**byte-identical** to our vendored `en_qwerty.json` (sha256 `1965ecd5…`), so the key
geometry and the data frame are the same object.

### Two corrections to the record

1. **The splits are not "held-out FUTO".** They are 49 % How-We-Swipe. Any statement
   that a number on `test_hwsfuto` measures "the FUTO floor" is measured against a
   **two-distribution mixture**, and the HWS half sits at a known ~0.064 systematic Y
   offset from the FUTO half (documented in `cc-old-data/howweswipe/CoordAlignmentVsFuto.md`;
   the correction was produced *after* the training data and was never applied). This is
   the most likely driver of the per-source accuracy gap.
2. **Native-speaker filtering never happened.** The How-We-Swipe participant metadata
   (`metadata.tsv`, OSF `wy9q8`) is not present on this machine and contains the only
   language field; no script references nativeness. Positive disproof: accented and
   non-English words survive in the shipped training data (`estáis`, `María`, `línea`,
   `हैं`). The HWS half received **no word cleanup at all** beyond dropping
   `len(word)==1`; the dictionary/junk-word filtering applied to the **FUTO half only**.

---

## 2. Contamination audit

`scan_futo_sessions.py` indexes all 939,550 corpus rows by trace hash → contributor
session (10,889 sessions, 8.7 s over 5.2 GB with 20 workers).

| check | result |
|---|---|
| canonical holdout traces (val+test, unique) | 12,299 |
| …found bit-exactly in HF **train** | 5,273 |
| …found in HF **dev** / **test** | **0** / **0** |
| …not in any HF split (HWS-derived, cannot leak via swipe-1) | 7,026 |
| contributor sessions touched by a holdout trace | **3,044 of 10,889 (28.0 %)** |
| corpus rows in those sessions | **552,515 (58.8 %)** |

Session exclusion costs 547,242 rows *beyond* exact dedup, because the leaked sessions
are the large ones (181 rows/session vs 86 overall). It is still applied: a session is
one volunteer's run, so leaving siblings in lets the model memorise a specific person's
hand geometry on the exact words we score.

### ⚠ The existing baseline T0 is itself contaminated

| source | T0 rows | sharing a contributor with val/test |
|---|---|---|
| FUTO half | 55,438 | 28,565 |
| HWS half | 55,438 | **54,550 (98.4 %)** |
| **total** | **110,876** | **83,115 (75.0 %)** |

The merge used a plain global `random.shuffle` — not grouped by session or participant —
so 924 HWS participants appear in both train and holdout. **Every committed number
(val 81.57, test 80.96) is measured with a 75 % contributor overlap.**

**Consequence for the tier ladder:** T1/T2 are contributor-disjoint by construction, T0
is not. Comparing them on the current val is therefore **confounded** — T0 enjoys a leak
the others were denied, so a tie would actually favour the bigger tier. Before the ladder
decides anything, we need either a contributor-disjoint holdout carved from the existing
val, or T0 re-run with the same session exclusion. Recommend the former (keeps the
committed numbers comparable and costs no training).

> ⚠ **Superseded — "by construction" was wrong for T1.** Measuring contributors from each
> row's own `session` field (rather than from a trace hash) in Phase A showed T1's FUTO
> contributor set is **9,877 of the corpus's 10,889 sessions**: the 102,826 rows whose
> session could not be recovered are kept by default, and they carry back ~2,000 of the
> sessions the exclusion had dropped. T1's contributor-clean val subset is **46 rows, not
> the 4,238 the hash-based mask reported**. A genuinely disjoint T1 needs
> `--strict-session`. T2/T2b *are* clean (93.8 % / 97.5 % of val). Separately, 219 sessions
> produced 249 val rows without ever entering `futo_tainted_sessions.npz`, so their 27,356
> corpus rows sit inside every tier. See `PHASE_A.md` §3.

---

## 3. The tiers

| tier | jsonl rows | cached npz | cache file | composition | filters |
|---|---|---|---|---|---|
| **T0** | 110,876 | 109,600 | `train.npz` (48 MB) | 55,438 HWS + 55,438 FUTO | user's curation, 1:1 balance cap |
| **T1** | 374,004 | **372,726** | `train_t1.npz` (165 MB) | 55,438 HWS + 318,566 FUTO | user's curation at full scale + contamination controls |
| **T2** | 385,458 | **385,021** | `train_t2.npz` (166 MB) | FUTO only | basic hygiene only (`potentially_invalid_sentence`) + contamination controls |
| **T2b** | 285,932 | **285,929** | `train_t2b.npz` (127 MB) | FUTO only | T2 + the full recovered quality cascade |

Drop accounting:

```
T1 : futo_in 688,025 | leak 0 | session-taint 369,459 | unmapped 102,826 (kept) | kept 318,566
T2 : rows_in 939,550 | invalid_sentence 4,709 | leak 5,273 | session-taint 544,110 | kept 385,458
T2b: rows_in 939,550 | invalid_sentence 4,709 | leak 5,152 | session-taint 491,154
     not_portrait 53,464 | bad_speed 40,865 | too_many_points 26,722 | bad_duration 19,528
     not_in_dictionary 7,166 | invalid_word 4,401 | canvas_wide 457 | canvas_dims 0
     kept 285,932   (dictionary = 458,325 terms)
```

T2b's quality cascade costs **99,526 rows on top of T2** (385,458 → 285,932). The two
biggest single gates are `not_portrait` (53,464) and `bad_speed` (40,865); the
dictionary/word gates together drop only 11,567, so the curation is overwhelmingly a
*motion/geometry* filter rather than a lexical one. That makes T2 vs T2b a clean test of
whether those motion gates earn their keep, independent of vocabulary coverage.

**T1 has zero exact-trace contamination** (0 of 12,299 holdout traces, 0.0000 %) — the
user's FUTO train/val pools were already disjoint. Its 102,826 "unmapped" rows are the
~15 % the normaliser altered enough to lose bit-identity with the raw corpus, so their
session cannot be recovered; they are kept by default and `--strict-session` drops them.

Note T1 (374 k) and T2 (385 k) end up nearly the same size: **once session-disjointness
is enforced, the user's curation only costs ~11 k rows.** The "more data vs better data"
contrast is therefore much weaker than the raw 110 k-vs-939 k framing suggested — the
real cost of a clean evaluation is the 58.8 % session exclusion, not the curation.

Every tier is still **3.4–3.5× T0**, so the scale-up lever is real; it is the *spread
between* the tiers that is narrow. The informative contrasts are therefore T0→T1 (scale
at fixed curation, but confounded by §2's T0 leak) and T2→T2b (curation at fixed scale,
unconfounded — both are contributor-disjoint).

### Recovered filter criteria (FUTO half, verbatim)

```
MIN_WORD_LEN, MAX_WORD_LEN = 2, 20      MIN_POINTS, MAX_POINTS = 8, 512
MIN_DURATION_MS, MAX_DURATION_MS = 40, 4000
MIN_SPEED, MAX_SPEED = 0.001, 0.01      MAX_CANVAS_WIDTH = 900   MIN_WORD_FREQ = 3
orientation == "portrait-primary";  canvas_width > canvas_height
word ∈ (NLTK words ∪ wordfreq top-400k), after stripping '.,;:!?()
```
`distance`, `word_idx` and `sentence` are **not** used — only `potentially_invalid_sentence`.

### Expansion headroom

* **HWS: essentially none.** All 61,597 filtered HWS rows are already consumed by T0
  (55,438 train + 6,159 holdout), zero left over. The remaining headroom is 15,301
  participant-flagged `is_err=1` traces (undesirable) and whatever sessions the full OSF
  release holds beyond the 1,052 local logs — fetching that release plus `metadata.tsv`
  is also the only route to real native-language filtering.
* **FUTO: large but gated by session exclusion.** 688,025 filtered rows exist; only
  55,438 (8.1 %) were used, and 318,566 survive contributor-disjointness.

---

## 4. Per-source holdout tags

`cache/holdout_source_tags.json` tags every canonical val/test row `futo` | `hws`, so
every eval from here reports per-source accuracy (val 4,942 futo / 4,976 hws;
test 1,217 futo / 1,183 hws). Given the ~0.064 Y offset between the halves, aggregate
numbers hide two different problems.

## 5. Licensing

`swipe.futo.org` is MIT; How-We-Swipe is its own open release. Deliberately **not** used:
`train_leon_filtered_norm.jsonl` (30-point standardised, unknown licence) and
`futo-org/swipe-negatives` (mined with FUTO model embeddings — the model-output rule).
