# Phase I-B — data quality (HWS filtering arms) + language versatility (Cyrillic)

**Date:** 2026-08-09 · Concurrent with Phase I-A (capacity; owns `train.py`).
I-B owns data-prep/corpus tooling, all in new files. **test-2400 is not read
anywhere in this phase.**

Two questions:

1. **The user's original HWS filtering intent, finally measurable.** The
   canonical HWS training rows only ever got basic hygiene; the intended
   native-speaker / quality filtering never happened because `metadata.tsv`
   was absent (`DATA_TIERS.md` §1.2). The full OSF release
   (`fetch_hws_full.py`) carries `englishLevel` for all 1,338 participants —
   so build measured arms and judge them.
2. **The true remaining agnosticism gap: non-Latin scripts.** ЙЦУКЕН was
   "untested, no data locally" (`ALT_LAYOUT_EVAL.md` §9). Answer whether real
   Cyrillic data is acquirable (it is — §4), whether synthetic Cyrillic can be
   built from English motor residuals (§5), and what a first honest decode
   measurement says (§6).

## 0. Verdict

*(filled at close of phase — see §3/§6 result tables)*

---

## 1. HWS filtering arms — construction

Builder: `build_hws_arms.py` (imports the audited keep-path from
`build_tiers.py`; the control arm's output is verified **byte-identical** to
`tier_t3hws.jsonl`, which is what licenses the other arms as same-code-path).

Fixed across all arms: the FUTO side of the adopted T3+3×HWS training pool
(`tier_t3futo.jsonl` = the FUTO-only prefix of `tier_t3.jsonl`, split at the
recorded `futo_kept` line count, HWS-tail sha256-verified) and the frozen
Phase-G/H recipe: `resbn:80:1,2,4,8`, embed_hid 96, 188,000 steps, batch 256,
lr 3e-3, wd 0.01, warmup 1,000, coupled affine sampler, layout-alt p 0.5, no
KD, 5,000-row beam-t1 selection, seed 1234. Composition per arm:
`--train-npz train_t3futo.npz,hws_<arm>.npz,hws_<arm>.npz,hws_<arm>.npz`
(FUTO + 3× the arm's HWS pool — the same structure as the adopted
`train_t3.npz,train_t3hws.npz,train_t3hws.npz`). Arms were trained from a git
worktree pinned at `d7faa75` — the exact code state that trained the control —
because I-A's train.py edits were landing concurrently.

| arm | HWS users | jsonl rows | cached rows | total train rows |
|---|---|---|---|---|
| **control** (= `phaseH-p50`, already trained) | 1,338 | 78,155 | 76,748 | 1,158,832 |
| **quality** (all levels + HWS-derived motion gates) | 1,338 | 76,805 | 75,519 | 1,154,426 |
| **nativeadv** (englishLevel native+advanced) | 755 | 47,185 | 46,395 | 1,067,054 |
| **native** (native only) | 413 | 27,222 | 26,820 | 1,008,329 |

Two disclosed asymmetries vs the control:

* **Dedup-hash vintage.** `train_t3.npz` predates the campaign-2
  normalized-word hash fix and was deliberately never rebuilt
  (AUDIT_PREDECODE §E); the arm caches use the corrected hash. Net effect:
  the arm FUTO side is 719 rows smaller than the control's joint build
  (927,869 + 76,748 = 1,004,617 vs 1,005,336) — 0.06 % of the pool, bounded
  by the audit at <0.05 val pt.
* **Contributor overlap moves with the filter.** T3-family tiers are
  contributor-dirty by design (benchmark tier, `PHASE_A.md` §5). 890 val-HWS
  contributors (337 native / 243 advanced / 260 intermediate / 49 beginner /
  1 na) are inside the control's training pool; a level arm removes the
  excluded levels' val contributors from training, so its HWS-half val is
  *cleaner* than the control's. Per-level and per-overlap slices are
  therefore reported alongside the aggregates (`hws_arm_report.py`).

### The quality gates — derived from HWS, not copied from FUTO

Phase A measured the FUTO motion cascade **negative** (−0.96 clean overall,
−1.71 FUTO-half at fixed scale), and its thresholds are demonstrably
mis-calibrated for HWS: the FUTO speed floor (0.001 u/ms) sits at the HWS
**25th percentile** and would discard a quarter of the corpus. Measured HWS
distributions (84,612 basic-hygiene traces of the full release; letter-area
units):

| metric | p0.5 | p1 | p25 | p50 | p75 | p99 | p99.5 | p99.9 |
|---|---|---|---|---|---|---|---|---|
| duration ms | 109 | 158 | 611 | 1,113 | 2,042 | 8,231 | 9,997 | 15,055 |
| points | 6 | 9 | 35 | 62 | 109 | 361 | 425 | 605 |
| path len | 0.024 | 0.143 | 0.98 | 1.80 | 2.89 | 6.18 | 6.72 | 8.16 |
| speed u/ms | 0.0002 | 0.0003 | 0.0010 | 0.0015 | 0.0021 | 0.0039 | 0.0043 | 0.0081 |

Chosen gates trim only degenerate tails (~1.9 % total; drop accounting per
gate in `hws_arm_quality.stats.json`): duration [150, 10,000] ms; points
[8, 512]; path length ≥ 0.10 (a near-zero path against a multi-letter word is
a tap, not a swipe); speed [0.0002, 0.008] u/ms. Drops: bad_points 778,
bad_duration 514, short_path 97, bad_speed 101.

Self-reported English level of the release (per-user `.json`, all 1,338):
native 413 / advanced 342 / intermediate 363 / beginner 219 / na 1. Median
trace duration rises monotonically from native (973 ms) to beginner
(1,433 ms) — the level field is behaviorally real, not just self-report noise.

## 2. HWS arms — evaluation protocol

Per arm, through the exported ONNX graph: `eval_beam.py` full val-9918, AOSP
STRIP trie, E1 preset, with `--out` per-trace dumps; `hws_arm_report.py`
joins per-source (futo/hws), per-englishLevel, and contributor-overlap
slices; `eval_altlayout.py` az26 E1 on all five real alt-layout corpora +
dvorak app-98k-trie arm. Control numbers are `phaseH-p50`'s own (same code,
same seed, same eval harness).

## 3. HWS arms — results

*(pending — training in flight at six-way GPU contention with I-A's capacity
runs)*

## 4. Cyrillic data acquisition — the Yandex Cup 2023 corpus is LIVE

The acquisition pointer sits in the vendored 7th-place solution repo
(`/home/will/git/neural-swipe-typing`, and the `swype/` fork):
`download_original_data.py` fetches the organizer's public Yandex Disk link.
Both were probed on 2026-08-09:

| | |
|---|---|
| URL | `https://disk.yandex.ru/d/IYiSpLob-zAxqg` (Disk API download endpoint) |
| payload | `data.zip`, 1,745,670,429 B |
| sha256 | `2e65d7a28ec737f208d2553c24e062d64ab1c71173c2609d0d3878f123b37521` — **matches the Disk API's published checksum** after download |
| contents | `train.jsonl` 6,000,000 rows (17.6 GB) · `valid.jsonl` 10,000 rows + `valid.ref` targets · `test.jsonl` (no targets) · `voc.txt` 503,598 words |
| grids | `default` (31 letter keys, no ё/ъ; **93.8 %** of train) and `extra` (33; 6.2 %), both 1080×667 px with per-key hitboxes, constant across rows |
| targets | а-я minus ё (organizers folded ё); ъ ~0.06 %; `-` ~0.3 % |
| license | **none stated.** Competition data, publicly distributed by the organizer via the solution repos' documented link. Treat as research-use-only; nothing trained on it ships without an explicit owner decision. |
| local | `~/ctc-train/data/yandex_cup/` (not committed) |
| secondary mirror | the solution repo's Google-Drive "preprocessed" folder (`download_dataset_preprocessed.py`) — same data post `separate_grid.py`; not needed once the primary verified |

The earlier note "the corpus is off-HuggingFace" conflated hosts: it was
never on HF; the organizer's Yandex Disk link has been live since 2023-10-25
(zip mtime) and still is. The geometric-engine spec's parallel claim
(`geometric-swipe-engine.md:726`, "JCUKEN real-corpus replay — CONFIRMED not
on HuggingFace") is true but was never a statement about this link.

Conversion: `prepare_yandex.py`. Coordinates land in the **canonical
letter-area frame** (x over grid width, y over the letter-key block, rows at
cy 0.167/0.5/0.833 — the exact (2r+1)/2R family every trained geometry lives
in; a naive `y/height` would squash the letter block to 46 % of the unit
square, outside the measured affine-tolerance envelope). Projection for
targets and lexicon identically (ALT_LAYOUT §3 policy): lowercase, strip
`-`/`'`, ё→е, ъ→ь, **no unicode NFD** (it would decompose й into и+breve).
Alphabet = the 31 default-grid letters, alphabetical slot order. Geometries
vendored as `layouts/ru_jcuken_default.json` / `_extra.json` (generated from
the corpus' own embedded grids; loads through the campaign `load_layout`).

### Frame mapping — established, not assumed (the ALT_LAYOUT §2 discipline)

Endpoint-proximity on 2,000 converted valid rows against the vendored jcuken
geometry, with a deliberately wrong geometry (qwerty centers, `ru letter i →
qwerty key i mod 26`) as the falsification control:

| geometry | start-hit | end-hit | start-d | end-d |
|---|---|---|---|---|
| **jcuken (claimed frame)** | **0.917** | 0.647 | 0.0491 | 0.1069 |
| wrong-geo control | 0.008 | 0.004 | 0.5095 | 0.4215 |
| en_qwerty val reference (ALT_LAYOUT §2) | 0.895 | 0.769 | 0.0686 | 0.0784 |

Start-hit sits at the top of the real-corpus band (0.79–0.91 across six
corpora) and the control collapses to below-chance — the frame is right. The
lower end-hit (0.65, end-d 0.107) is a property of this corpus (late finger
lift), not of the mapping; dvorak showed the mirror-image skew (0.79 start /
0.97 end).

The model needs **no change**: emission column `c` is whatever key sits in
slot `c`, so a 31-letter layout uses 31 of the 64 slots. The committed
`train.py` runs Cyrillic untouched (`--layout` + `--cache cache_ru`
+ `--beam-val-rows 0`, greedy selection — the in-train beam validator's vocab
loader is a-z-hardcoded, so the lexicon beam runs offline in
`eval_cyrillic.py` instead, generic `LexTrie` + `futo_viterbi_beam` over
Cyrillic). App side: `CtcLayout.kt` is already alphabet-agnostic (CharArray);
the gap there is a ru trie + layout wiring, not engine work.

## 5. Synthetic Cyrillic — residual transplant (`cyrillic_synth.py`)

The counterfactual: a script with **no corpus at all**. English human
residuals are re-anchored onto ideal polylines of Russian words on the ЙЦУКЕН
geometry. `layout_aug.warp_path` is reused **verbatim** through per-vertex
virtual indices: a donor English trace whose collapsed polyline has the same
vertex count as the Russian word's gets virtual ids `0..S`, with
`src_virtual[i] = qwerty[donor_seq[i]]`, `dst_virtual[i] =
jcuken[ru_seq[i]]`. The monotone-DP correspondence, endpoint pins,
vertex-absolute arc remap and movement-frame residual transfer are the same
code Phase H validated; donor match is purely structural (vertex count), the
correspondence is geometric, letter identity never enters.

Words are drawn from the app's bundled langpack-ru CKDT-v2 dictionary
(49,704 projectable words, weight = 255 − rank) — deliberately NOT the
Yandex word distribution, because in the no-corpus counterfactual only a
lexicon exists. Donor pool: `train_t3futo.npz` + `train_t3hws.npz`
(1,004,617 traces, vertex counts 1–24; lexicon coverage complete except 31
words of vertex count 19–22).

What CAN be validated (because real data exists here): endpoint-proximity
stats vs the real corpus, and the decisive train-on-synth → eval-on-real
measurement (§6). What CANNOT: per-script motor idiosyncrasies (Cyrillic
swipers may deviate differently); transfer of this validation to scripts
whose geometry departs further than ЙЦУКЕН's.

### Generation + endpoint validation (1,000,000 rows, seed 1234)

1,141 rows/s single-core; zero no-donor rejections. Endpoint proximity
(PHASE_H §2.3 frame metric), 2,000 rows each, jcuken geometry:

| paths | start-hit | end-hit | start-d | end-d |
|---|---|---|---|---|
| **synthetic** (en residuals → ru words) | 0.710 | 0.656 | 0.0557 | 0.0783 |
| **real** (Yandex valid-10k) | 0.917 | 0.647 | 0.0491 | 0.1069 |
| en source band (ALT_LAYOUT §2, reference) | 0.895 | 0.769 | 0.0686 | 0.0774 |

Distances transfer at English magnitudes (0.056/0.078 vs the en source's
0.069/0.077) — the transplant adds no positional error, same as Phase H. The
synthetic **end** side matches the real corpus almost exactly (hit 0.656 vs
0.647). The **start** side is markedly sloppier than real Russian swipes
(0.710 vs 0.917): real Yandex starts are very precise (d 0.049), English
start residuals are not. As in Phase H, the mismatch errs toward *harder*
training samples, not easier ones. Training-time selection for the synth arm
runs on a 5,000-row synth val (seed 999, its own cache dir) — no real
Cyrillic row touches the synth arm before its final eval, keeping the
no-corpus counterfactual intact.

## 6. Cyrillic decode — first measurements

Two arms, both `resbn:80:1,2,4,8` embed_hid 96 at **94,000 steps** (the
Phase-G measurement that 188 k buys +0.05 at this width licenses the half
schedule for a first measurement), no layout-alt (single geometry), coupled
affine + slot permutation + noise unchanged, greedy checkpoint selection
(`--beam-val-rows 0` — the in-train beam validator is en-only), committed
`train.py` untouched at `d7faa75`:

* `phaseIB-ru-real` — 1,000,000 real default-grid rows, selection on the
  5,000 train-derived real rows;
* `phaseIB-ru-synth` — 1,000,000 synthetic rows, selection on 5,000
  synthetic rows (seed 999): **no real Cyrillic sample enters this arm
  before its final eval.**

Final eval, both arms: the untouched valid-10k (9,416 default-grid rows),
`eval_cyrillic.py`, app-ru-50k CKDT trie and voc-503k flat trie footings, E1
preset + λ=0 control, in-dict and all-rows protocols.

*(results pending)*

## 7. Data-asset inventory (this phase)

| asset | rows/size | license | where |
|---|---|---|---|
| HWS full release (already fetched, Phase pre-I) | 1,338 users / 86,323 traces | MIT (OSF sj67f) | `~/ctc-train/data/hws_full/` |
| Yandex Cup 2023 NeuroSwipe | 6.01 M curves / 1.63 GB zip | unstated (see §4) | `~/ctc-train/data/yandex_cup/` |
| app langpack-ru CKDT v2 | 50 k words | app asset (read-only) | app repo `scripts/dictionaries/langpack-ru.zip` |
| Yandex `voc.txt` | 503,598 words | with corpus | same dir |

## 8. Commits

* `b606879` pm: phase i-b todos
* `0f36913` `build_hws_arms.py` + arm builds (control byte-identical check)
* `7de2ee5` cyrillic pipeline: `prepare_yandex.py`, `cyrillic_synth.py`,
  `eval_cyrillic.py`, vendored ru geometries
* `50c561a` letter-area frame fix for the ru conversion
* *(this file + results: pending)*
