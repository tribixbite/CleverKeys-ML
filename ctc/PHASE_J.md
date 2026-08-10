# Phase J — the final convergence campaign

**Date:** 2026-08-10 · **Authority:** user directive of 2026-08-09/10 (unlimited
compute; terminal condition = high-confidence SOTA for what existing usable
datasets and research admit — a ≤5 MB, <50 ms model beating the incumbents on
ALL spreads and layouts/languages). **test-2400 is not read before §"Final
verification"** — if and only if every bar is beaten on val + alt-layout + ru,
a pre-registered unsealing closes the campaign.

## 0. The bars (incumbents to beat; seed-mean over 3 seeds, every-seed preferred)

| axis | incumbent | bar |
|---|---|---|
| en_qwerty full val-9918 (E1, AOSP) | `resbn192i` (`PHASE_I.md` §7.2) | **88.30 / 92.60 / 93.26 / 91.27 / 86.77** |
| dvorak held-out / dvorak app-98k | `resbn192i` seed-mean | **89.13 / 88.20** |
| azerty / qwertz / german / spanish | `resbn192i` seed-mean | **83.60 / 82.50 / 79.64 / 88.28** |
| Cyrillic, shippable (NO Yandex training rows) | `phaseIB-ru-synth` real-val probe | **in-dict t1 76.21** (app-ru 50k, E1) |
| size / latency | — | **≤5 MB** (fp16w free; int8-trunk free at ch 256) / <50 ms |

New this phase (no incumbent — floors established here): realalt heldouts
clearflow / kasroz (§3.3).

## 1. Protocol

One variable per arm; ~1 pt single-seed noise floor → paired seeds for close
calls; sign-consistency across the five val metrics for promotion; 3 seeds for
the final stack. Base recipe = `resbn192i` (`resbn:{ch}:1,2,4,8`, embed_hid 96,
T3+3×HWS, 188 k steps, batch 256, lr 3e-3, wd 0.01, warmup 1 k, coupled affine,
layout-alt p per arm, no KD, 5 k-row beam-t1 selection at the published
selection preset, seed 1234). Eval battery per arm (`phaseJ_eval.sh`): exported
ONNX → full val-9918 (E1/AOSP) + 7 alt-layout corpora (5 classic + the new
realalt heldouts) + dvorak app-trie arm.

## 2. Free lever — blank-penalty decode axis: REFUTED (zero GPU)

`sweep_scoring.py --blank-grid` (constant offset on the blank column of the
sliced log-emissions, BEFORE the beam — enters the DP and pruning, so each
offset runs its own beam pass). `resbn192i_s1234` val emissions, E1, sweep
half / holdout half (2,000 rows each):

| offset | sweep t1/t3/t5 | holdout t1/t3/t5 |
|---|---|---|
| **0 (incumbent)** | **88.75 / 92.75 / 93.55** | **88.05 / 92.45 / 93.25** |
| −0.10 | 88.75 / 92.75 / 93.45 | 88.20 / 92.15 / 93.15 |
| +0.10 | 88.65 / 92.55 / 93.60 | 87.85 / 92.40 / 93.20 |
| ±0.25 | −0.5…−1.2 t1 | both halves down |
| ±0.5 | −5.5…−11 t1 | — |
| −1 / −2 / +1 | catastrophic (51/21/45 t1) | — |

**Zero is a sharp optimum; ±0.1 is inside noise and sign-inconsistent across
halves.** The decode-side peakiness hypothesis (RESEARCH_SCAN #5) gets no
support: the emissions' blank calibration is already matched to the lexicon
beam. This lowers the prior on CR-CTC's mechanism (iii) (blank-mass shaping);
its self-distillation mechanisms remain the live question (§4 arm).

## 3. Data (DATASET_SCOUT arms — built, verified)

### 3.1 swipe-2/3/4/5 pools (`build_swipe2345.py`)

Fetched 2026-08-10 (MIT; row counts match the scout exactly: 28,095 / 38,228 /
50,300 / 59,247). Gates: dual-finger (2,708), `distance >= 100000` sentinel
codes (measured to be FUTO validity-failure encodings 100001–3; swipe-1 itself
carries only 1.9 % after FUTO's upstream filtering vs 7–12 % here — dropping
them reproduces the validity class swipe-1 was already filtered by; an
invalid-label gate, not curation), ≥3 points, word ≥2. Verified **here, not
inherited from the scout**:

* **Session disjointness on the complete sets**: all sessions of each run ∩ the
  full 10,889-session raw swipe-1 train corpus = **0, 0, 0, 0** (the scout had
  sampled 68 ids).
* **Holdout trace overlap** (both hash conventions): **0** everywhere.
* **Endpoint proximity** (PHASE_H §2.3): sw234 0.936/0.784, sw5q 0.906/0.774 —
  at/above the 0.79–0.91 real-corpus band; frame confirmed.

Pools: `tier_sw234` **101,842** rows · `tier_sw5q` **24,707** (qwerty **en
only** — pl/fr/de-on-qwerty excluded because a-z stripping would corrupt their
labels, e.g. 'über'→'ber') · realalt `clearflow` 9,483 train / 1,715 heldout ·
`kasroz` 245 / 774 (session-disjoint 80/20 splits) · `toki_pona` 382 =
**zero-shot eval only** (2 sessions, unsplittable; 14-letter alphabet). The
five eval-suite layouts (11,015 rows) are structurally excluded from training.

### 3.2 HWS Y-frame correction (`tier_t3hws_yfix`)

The swipetest geometry (DATASET_SCOUT §4.1: rows at y = 1/7, 3/7, 5/7 of the
keyboard box) says the HWS frame error is a **scale**, not the assumed 0.064
offset: y_canonical = y_hws · 7/6. Verified on the data — median first-touch y
per keyboard row: HWS raw **0.144 / 0.433 / 0.701** (≈ 1/7, 3/7, 5/7); ×7/6 →
**0.168 / 0.505 / 0.818** vs canonical 0.167 / 0.5 / 0.833 (FUTO reference
0.182 / 0.522 / 0.852 — touches sit slightly below-center there too). Arm built
(`hws_yfix.npz`, 76,748 rows — cache-count-identical to the I-B control).
Caveat registered in advance: **the val HWS half keeps the uncorrected frame**
(benchmark comparability), so this arm tests train-side frame unification only;
a train-only fix may well lose the HWS half by construction.

### 3.3 realalt zero-shot floors (before any realalt training)

`resbn192i_s1234`, az26 in-dict, E1, AOSP trie, session-disjoint heldouts:

| layout | n | OOV % | t1 | t3 | t5 | greedy |
|---|---|---|---|---|---|---|
| clearflow | 1,670 | 2.6 | **91.08** | 98.26 | 98.92 | 32.6 |
| kasroz | 744 | 3.9 | **90.19** | 97.58 | 98.66 | 32.4 |

Frame sanity 0.99 start-hit; wrong-geometry falsification controls collapse to
0.000. **Zero-shot transfer onto never-seen real layouts already exceeds the
dvorak transfer number** — the synthetic layout-alt training distribution
covers these geometries well. (Both corpora are small and single-cohort; ±0.7–
1.1 pt binomial SE at these n.)

## 4. Training levers implemented (`train.py`, committed d29d648)

* **CR-CTC** (`--cr-alpha`, Spec A): dual views sharing the layout draw + slot
  permutation (columns mean the same key in both), independent affine / noise /
  frame-hold masking; symmetric stop-grad frame KL over all 65 columns.
* **FUTO-parity augs** (Spec B): `--aug-shear` (exact containment range),
  `--aug-rot` (containment-rejection), `--aug-timerev` (frames AND target
  reversed, before the layout warp), `--aug-maskhold` (frame-hold spans).
* **Per-source layouts** (`--train-layouts`): each `--train-npz` pool can carry
  its own geometry — realalt pools and non-Latin scripts join one run; non-26-
  key sources skip the a-z layout resampler. This is the PHASE_I §9 "per-row
  layout batching" work item, resolved at per-source granularity (every pool is
  single-geometry).
* **`--snapshot-every`** + `soup_checkpoints.py`: greedy beam-selected
  checkpoint soup with BN re-estimation over augmented train rows.
* **Regression guard**: with all new flags off, the dataset emits tensors
  BIT-IDENTICAL to the pre-J code (0/10 mismatches, plain + layout-alt paths).

## 5. Arms in flight (round 1, launched 2026-08-10, seed 1234)

| arm | question |
|---|---|
| `phaseJ-ch256-p65` | dose-scaling law at ch 256 (PHASE_I §9 #1): does p 0.65 fix the seed-volatile transfer the way it did at ch 192? |
| `phaseJ-ch256-p80` | coarse dose sweep upper point at ch 256 |
| `phaseJ-ch192-p80` | is p 0.65 already the ch 192 optimum? |
| `phaseJ-cr80` | CR-CTC α 0.2 + per-view frame-hold masking at ch 80 vs `phaseH-p50` |

All four reached the 188 k step budget (the orchestrator process died mid-
campaign on an API-credit failure; the trainings survived it). Selection-beam
top-1 (the in-training 5 k-row selector, NOT the benchmark — full-val battery
in §5.1):

| arm | sel-beam t1 @best | same-recipe reference | Δ sel |
|---|---|---|---|
| `phaseJ-ch256-p65` | **87.02** (ep 39/41) | `phaseI-ch256` p 0.5 86.78 | **+0.24** |
| `phaseJ-ch256-p80` | 86.76 (ep 28) | `phaseI-ch256` p 0.5 86.78 | −0.02 |
| `phaseJ-ch192-p80` | 86.32 (ep 35) | `resbn192i` p 0.65 86.56 | −0.24 |
| `phaseJ-cr80` | 84.86 (ep 37) | `phaseH-p50` 85.48 | **−0.62** |

Reads (selection metric only; ~1 pt noise floor applies, and this metric is a
5 k-row subset, so these are direction hints pending §5.1):
* the §5-of-PHASE_I **dose-scaling law extends to ch 256**: p 0.65 is the best
  of the three doses measured there, p 0.8 is not better than p 0.5 →
  **p 0.65 looks like a plateau optimum, not a monotone trend**;
* at ch 192 p 0.8 is *worse* than p 0.65 — consistent with the same plateau,
  and it confirms 0.65 was not an under-shoot at that width;
* **CR-CTC α 0.2 at ch 80 is negative on the QWERTY selector** (−0.62, i.e.
  at the noise floor and the wrong sign). Its train CTC loss ends at
  0.6959 vs `phaseH-p50`'s 0.5986 — the consistency term is acting as a strong
  regularizer on a model that is not overfitting. Transfer is the remaining
  live question for it (§5.1); the §2 blank-axis refutation had already
  lowered the prior on its mechanism (iii).

### 5.1 Round-1 full battery *(running 2026-08-10 16:13, results pending)*

`phaseJ_eval.sh` on all four: exported ONNX → full val-9918 (E1/AOSP) → 7 alt-
layout corpora (dvorak/azerty/qwertz/german/spanish + clearflow/kasroz) →
dvorak app-98k trie.

## 6. Arms in flight (round 2, launched 2026-08-10 16:09, seed 1234, ch 192 /
## p 0.65 base unless noted)

| arm | rows | question |
|---|---|---|
| `phaseJ-sw234` | 1,260,674 | does the new swipe-2/3/4 pool (+101,842, §3.1) convert? |
| `phaseJ-yfix` | 1,158,113 | HWS Y-scale correction ×7/6 (§3.2), train-side only |
| `phaseJ-realalt` | 1,188,016 | clearflow/kasroz train rows on their own geometry (3× = 29,184 rows) via `--train-layouts` |
| `phaseJ-ch256-280k` | 1,158,832 | ch 256 + p 0.65 on a **280 k** schedule (PHASE_I §5 underfit signal) + `--snapshot-every 4` soup supply |

Composition notes: the yfix arm swaps the base's `train_t3 + hws×2` for
`train_t3futo + hws_yfix×3` — a 719-row (0.06 %) hygiene delta against the base
that is far below the noise floor; the realalt heldouts stop being zero-shot for
that arm only (§3.3 keeps the zero-shot floors for everything else).

## 7. Continuity protocol (after the 2026-08-10 orchestrator loss)

Trainings are launched **detached** (`nohup setsid` + per-run
`ckpt_<run>.launch.log`) so an orchestrator death cannot kill them; the
orchestrator polls those logs synchronously. Round scripts:
`phaseJ_round2.sh`, `phaseJ_eval_round1.sh` in the workdir. A killed run is
resumed with `--resume ckpt/<run>/last.pt` under the **identical** run name and
args. State is committed at every milestone so a successor can take over from
this file alone.
