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

1. **The user's intended native-speaker filtering, finally measured: it is a
   (consistent, noise-floor-magnitude) negative.** Both level arms lose every
   val slice — including the native-speaker rows themselves, on a
   leak-matched comparison. The HWS-derived motion gates are a statistical
   tie with a mildly positive point estimate. **Keep the full-release HWS
   pool; do not filter by englishLevel** (§3).
2. **Real Cyrillic data exists and is downloadable today**: the Yandex Cup
   2023 corpus (6.0 M ЙЦУКЕН swipes, sha256-verified, license unstated —
   research-use caution) (§4).
3. **Cyrillic decodes at English-class accuracy with zero model changes**:
   in-dict t1 **89.64** / t3 95.82 / t5 96.97 (app-ru 50k trie), greedy
   75.2, from a 94 k-step resbn80 trained on 1 M real rows through the
   committed `train.py` (§6).
4. **The no-corpus counterfactual works at geometric-engine level**: a model
   trained purely on synthetic Cyrillic (English residuals transplanted onto
   ru polylines, `warp_path` reused verbatim) decodes real swipes at in-dict
   t1 **76.21** — no real Cyrillic sample anywhere in its pipeline (§5–6).
   Synthesis can bootstrap a script; real data is worth ~13 t1 on top.

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

Full val-9918, exported ONNX, AOSP trie, E1, all rows (`eval_beam.py` dumps →
`hws_arm_report.py`; raw JSON `cache/phase_ib_arm_report.json`):

| arm | val t1/t3/t5 | FUTO t1 | **HWS t1** | native | advanced | intermediate | beginner (n=142) |
|---|---|---|---|---|---|---|---|
| **control** | 87.66 / 92.24 / 93.05 | 94.27 | **81.09** | 81.97 | 80.04 | 81.72 | 71.83 |
| **quality** | **87.71 / 92.29 / 93.11** | **94.58** | 80.89 | 81.48 | **80.12** | 81.22 | **76.06** |
| **nativeadv** | 87.30 / 92.12 / 92.99 | 94.39 | 80.25 | 81.43 | 78.94 | 80.44 | 73.24 |
| **native** | 87.33 / 92.00 / 92.84 | 94.56 | 80.14 | 81.34 | 78.72 | 80.51 | 72.54 |

Alt-layout suite (in-dict az26, E1, t1; control = `phaseH-p50` s1234; PHASE_H
seed spread on dvorak alone was 88.85–91.05, which bounds how much this table
can resolve):

| arm | dvorak | azerty | qwertz | german | spanish | dvorak app-98k |
|---|---|---|---|---|---|---|
| control | 88.85 | 83.64 | 84.16 | 81.45 | 88.51 | 88.20 |
| quality | 90.84 | 84.74 | 83.49 | 80.99 | 86.18 | 89.99 |
| nativeadv | 91.66 | 83.11 | 82.48 | 80.04 | 87.32 | 90.72 |
| native | 90.03 | 84.98 | 81.89 | 80.95 | 87.88 | 89.62 |

### Reads (single seed; the Phase-C resolution floor of ~1 pt applies)

* **englishLevel filtering is a consistent negative.** Both level arms lose
  on every aggregate val metric, lose the HWS half (−0.84 / −0.95), and —
  decisive — lose on the **native-speaker val rows themselves** (81.97 →
  81.43 / 81.34). The leak asymmetry cannot explain that slice: native
  contributors are inside the training pool of *all* three arms (T3-family
  is contributor-dirty), so the native-row comparison is leak-matched, and
  it still favors keeping everyone's data. Non-native swipes are not noise
  for native users; they are more of the same motor signal. Individually
  each delta sits at the noise floor; the *direction* is consistent across
  all seven slices of both arms. This is the fourth exclusion-style curation
  negative in the campaign (T2b cascade, T4, KD... and now levels).
* **The HWS-derived quality gates are a wash with a mildly positive point
  estimate** — best-or-tied on all three aggregate val metrics (+0.05 /
  +0.05 / +0.06), +0.31 FUTO, −0.20 HWS, and +4.2 on the beginner slice
  (n=142, SE ≈ 3.6 — not resolvable). Unlike the FUTO cascade (−0.96 clean
  at 26 % of the pool), trimming 1.7 % of degenerate tails costs nothing
  measurable in either direction.
* **Contributor-overlap disclosure:** on the level arms the excluded-level
  val rows are contributor-clean while the control's are not; their
  held-out-contributor HWS slices score 79.81 (nativeadv, n=1,550) and 79.30
  (native, n=2,913) vs 80.44 / 81.34 for the still-leaked slices — so part
  of the level arms' intermediate/beginner deficit is leak removal, which is
  exactly why the leak-matched native slice above is the load-bearing
  comparison.
* Cross-layout: no arm separates from the control beyond the known seed
  spread; the transfer property is insensitive to the HWS composition.

**Recommendation to I-A:** keep the current tier (T3 + 3× full-release HWS)
for the capacity runs. The quality-gated HWS pool
(`hws_quality.npz`) is an acceptable drop-in (statistically a tie, tiny
positive point estimate) — adopt only if a rebuild is happening anyway. Do
NOT adopt englishLevel filtering at any threshold.

Artifacts (runtime dir, not committed): `ckpt/phaseIB-{nativeadv,native,
quality}/ctc_swipe_encoder.onnx`, sha256 `e024ac35…` / `0cd1771d…` /
`f4c0500e…`; per-trace dumps `val_dump_e1.jsonl` alongside; alt-layout JSON
in `~/ctc-train/altlayout/phaseIB-*`.

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
| license | **none stated — and researched in full: `YANDEX_LICENSE_RESEARCH.md`.** No grant exists anywhere (contest, Cup regulations, Disk link, solution repos, Kaggle mirror = "License: Unknown"). Background terms are restrictive, not permissive: Yandex's services agreement authorises only *personal non-commercial* use of content reached via its services, and the corpus is a protected database under ГК РФ ст. 1334 (6 M rows ≫ the 10 k presumption; term to ~2039) whose ст. 1335.1 carve-outs cover research/education but not a shipped product. **Verdict: research + held-out-eval only; synth-only for anything that ships.** |
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

### `phaseIB-ru-real` — real-data Cyrillic, valid-10k (9,416 rows), first read

Training: 94 k steps, best greedy 70.04 % (selection val) @ epoch 21. ONNX
`cb8ece6b…` (1,142,727 B — byte-size-identical graph to every resbn80
artifact; the alphabet is data, not architecture).

| lexicon | λ | OOV | greedy | in-dict t1/t3/t5 | all-rows t1 | ≤3 t1 (n) | 4+ t1 |
|---|---|---|---|---|---|---|---|
| app-ru 50k (CKDT) | 1.1 (E1) | 945 (10.0 %) | **75.23** | **89.64 / 95.82 / 96.97** | 80.64 | 94.12 (3,281) | 86.80 |
| app-ru 50k | 0 | 945 | 75.23 | 88.76 / 95.11 / 96.38 | 79.85 | 93.51 | 85.76 |
| voc 503k (flat) | 1.1 | **0** | 71.54 | **84.11 / 92.11 / 93.91** | 84.11 | 91.36 (3,356) | 80.10 |
| voc 503k | 0 | 0 | 71.54 | 84.11 / 92.11 / 93.91 | 84.11 | 91.36 | 80.10 |

Reads:

* **Cyrillic decodes at English-class accuracy.** In-dict 89.64 on the 50k
  app trie vs the en_qwerty in-dict control's 91.11 (147k trie, ch128 at a
  swept preset) — at half the schedule, greedy checkpoint selection, no
  layout-alt, and an unswept preset. Greedy 75.2 beats the en control's 72.8:
  the emissions themselves are excellent on ЙЦУКЕН.
* The voc footing (every target reachable, zero OOV) reads 84.11 all-rows t1
  against a 10× larger, frequency-free lexicon — the flat-frequency rows are
  bit-identical at λ=1.1 and λ=0, confirming the λ term is inert there
  (uniform log-freq cancels in ranking).
* E1 transfers: λ is worth +0.88 in-dict t1 on the CKDT trie. No per-language
  preset sweep was run (tuning-asymmetry rules as in ALT_LAYOUT §9 — these
  are floors).

### `phaseIB-ru-synth` — the no-corpus counterfactual, same valid-10k

Trained on 1,000,000 **synthetic** rows only (English residuals transplanted
onto ru polylines), checkpoint selected on synthetic val — no real Cyrillic
sample touched this arm before this table. Same recipe, same eval:

| lexicon | λ | greedy | in-dict t1/t3/t5 | all-rows t1 | ≤3 t1 | 4+ t1 |
|---|---|---|---|---|---|---|
| app-ru 50k | 1.1 (E1) | 37.07 | **76.21 / 88.53 / 91.42** | 68.56 | 83.75 | 71.45 |
| app-ru 50k | 0 | 37.07 | 68.65 / 84.70 / 88.70 | 61.76 | 72.54 | 66.18 |
| voc 503k | 1.1 | 35.16 | 61.09 / 77.64 / 82.53 | 61.09 | 70.50 | 55.87 |

**Verdict on the synthesis path:** a script with no swipe corpus at all can
be launched from English motor residuals at **in-dict t1 ≈ 76** on the app
lexicon — the same accuracy class as the shipped geometric engine's
cross-layout anchors (71–77) — and closes to ≈ 90 once real data exists
(the paired real arm above). The synth-vs-real gap (−13.4 in-dict t1) is
the honest price of the counterfactual: greedy collapses to 37 (emissions
carry English-magnitude start noise on a denser board, §5) and the lexicon
does the rest, exactly the pattern ALT_LAYOUT §8 warned about — so a
synth-trained ship would lean hard on its trie, and the λ prior is worth
+7.6 t1 there vs +0.9 on the real arm. The mixed-data lever
(synth pre-train → small real fine-tune) is the obvious next rung and was
not run (GPU shared with I-A's capacity ladder).

*(HWS arm results pending)*

### Cyrillic artifacts + what a joint multi-script model still needs

`ckpt/phaseIB-ru-real/ctc_swipe_encoder.onnx` sha256 `cb8ece6b…`,
`ckpt/phaseIB-ru-synth/ctc_swipe_encoder.onnx` sha256 `d78a9fb9…` (runtime
dir; both the standard 1,142,727-byte resbn80 graph). These are ru-only
prototypes: one model per script. A single model serving both scripts needs
per-row layout/alphabet batching in `train.py` (I-A's file — a dataset-level
change, the model itself is already alphabet-free), plus ru entries in the
app's layout→`CtcLayout` wiring and a ru trie (`CtcLayout.kt` is already
char-generic). The extra-grid (33-letter) geometry is vendored but untrained;
ё/ъ remain projected (ё→е, ъ→ь) exactly as the corpus itself does.

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
