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

### 5.1 Round-1 full battery

Exported ONNX → full val-9918 (E1/AOSP) → 7 alt-layout corpora → dvorak app-98k
trie. Single seed (1234) throughout, so the ~1 pt floor applies to every Δ;
paired references are the same-seed runs, not the 3-seed bars.

| model | val t1/t3/t5/≤3/4+ | greedy | dvorak | azerty | qwertz | german | spanish | clearflow | kasroz | dvorak-app |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `resbn192i` s1234 (base) | 88.32/92.70/93.25/91.21/86.83 | 72.8 | 90.60 | 84.59 | 82.73 | 79.76 | 88.85 | 91.08* | 90.19* | 89.17 |
| `phaseI-ch256` s1234 (p 0.5) | 88.64/92.56/93.23/91.15/87.33 | 75.8 | 87.95 | 81.87 | 79.95 | 78.81 | 88.51 | — | — | 87.83 |
| **`phaseJ-ch256-p65`** | **88.69/92.75/93.37/91.21/87.38** | 75.0 | 89.66 | 82.44 | 79.61 | 78.72 | 87.66 | 91.50 | 90.32 | 88.89 |
| `phaseJ-ch256-p80` | 88.31/92.61/93.38/90.94/86.95 | 70.1 | 88.12 | **83.92** | **82.90** | **79.99** | **89.02** | 88.68 | 90.99 | 86.45 |
| `phaseJ-ch192-p80` | 88.10/92.51/93.33/90.88/86.66 | 70.3 | 90.72 | 83.83 | 83.07 | 80.67 | 88.23 | 92.46 | 92.61 | 90.15 |
| `phaseH-p50` ch 80 (CR base) | 87.66/92.24/93.05/90.88/85.99 | 63.3 | 88.85 | 83.64 | 84.16 | 81.45 | 88.51 | — | — | 88.20 |
| `phaseJ-cr80` (CR-CTC α 0.2) | 87.46/92.15/92.89/90.85/85.69 | 65.5 | **91.98** | 84.59 | 83.74 | 80.95 | 88.40 | 93.05 | 90.05 | **91.94** |

\* §3.3 zero-shot floors, measured on the same checkpoint before this round.

**(a) ch 256 + p 0.65 dominates ch 256 + p 0.5 on all five val metrics**
(+0.05/+0.19/+0.14/+0.06/+0.05) and recovers 1.7 pt of dvorak (87.95 → 89.66,
app-trie 87.83 → 88.89). The PHASE_I §5 dose-scaling law holds at the top rung.
Against the campaign bars it is the **first model to beat the val bar on t1,
t3, t5 and 4+ simultaneously** (+0.39/+0.15/+0.11/+0.61) — ≤3 is −0.06, i.e. a
tie. But **it loses the euro-layout axis**: azerty −1.16, qwertz −2.89, german
−0.92, spanish −0.62 vs the 3-seed bars (and vs `resbn192i`'s own s1234, so it
is not a seed artifact). Capacity buys English accuracy and dvorak; it does not
buy the CKDT-λ-confounded euro corpora. **Not promotable as-is.**

**(b) p 0.8 is past the optimum for val — but the dose axis is NOT closed on
transfer, and the selection metric hid that.** At ch 192, p 0.8 costs val
(−0.22/−0.19/+0.08/−0.33/−0.17 vs the same seed, four of five down) and buys a
little transfer (dvorak-app +0.98, qwertz +0.34, german +0.91, clearflow +1.38,
kasroz +2.42; azerty −0.76, spanish −0.62). At ch 256 the same step is far more
consequential than the 5 k-row selector suggested (it read p 80 ≈ p 50):

| ch 256 dose | val t1 / 4+ | dvorak / app | azerty | qwertz | german | spanish |
|---|---|---|---|---|---|---|
| p 0.50 | 88.64 / 87.33 | 87.95 / 87.83 | 81.87 | 79.95 | 78.81 | 88.51 |
| p 0.65 | **88.69 / 87.38** | **89.66 / 88.89** | 82.44 | 79.61 | 78.72 | 87.66 |
| p 0.80 | 88.31 / 86.95 | 88.12 / 86.45 | **83.92** | **82.90** | **79.99** | **89.02** |

**p 0.8 is the only ch-256 point that beats all four euro bars** (83.60 / 82.50
/ 79.64 / 88.28 → +0.32 / +0.40 / +0.35 / +0.74), and it clears the val t1/t3/t5
and 4+ bars too (+0.01/+0.01/+0.12/+0.18) — it misses only ≤3 (90.94 vs 91.27)
and the two dvorak columns (−1.01 / −1.75). p 0.65 is the mirror image: val and
dvorak yes, euro no. So at capacity **the dose trades the permuted-geometry axis
(dvorak) against the near-QWERTY-with-foreign-lexicon axis (euro)**, and neither
dose clears everything.

Honesty caveats on that read: single seed each, and PHASE_I §7.2 measured a
dvorak seed spread of 85.88–90.92 at fixed settings, so the 1.5 pt dvorak
difference is *inside* the seed noise while the euro deltas (+1.3…+3.3 on four
of four, same sign) are not. The euro corpora also carry the
`ALT_LAYOUT_EVAL.md` §3 CKDT-λ confound. **Neither dose is promotable on one
seed; the p 0.65-vs-0.8 call at ch 256 is a paired-seed question, and it is the
one the §7 stack has to resolve.** The earlier reading of this section — "the
dose axis is closed" — was written off the 5 k-row selector before this battery
existed and is **retracted**.

**(c) CR-CTC is a transfer lever, not an accuracy lever — the strongest one
measured so far.** At ch 80, α 0.2 costs a small but sign-consistent amount on
every val metric (−0.20/−0.09/−0.16/−0.03/−0.30) and buys **dvorak +3.13 and
dvorak-app +3.74**, azerty +0.95, clearflow ≈ +2 (vs the ch192 floor), against
qwertz −0.42, german −0.50, spanish −0.11. Its end-of-schedule train CTC loss
is 0.696 vs 0.599 — the consistency term is a strong regularizer, which is why
it costs in-distribution accuracy and pays out-of-distribution. This is the
same shape as the PHASE_I §6.1 T′ = 64 result (small val, large transfer) but
without the contract break. **The refuted blank axis (§2) was mechanism (iii);
the self-distillation mechanisms are the ones that carry the effect.**

**Round-2 consequence:** the campaign's binding constraint is the transfer axis
at capacity — no single ch 256 dose clears both dvorak and euro — and CR-CTC is
the only measured lever that moves transfer by points rather than tenths, at a
val cost of tenths. Round 3 is `cr192` (α 0.2 at the ship width, paired vs
`resbn192i`) and `cr256` (α 0.2 on the `ch256-p65` frontier) plus `futoaug`.
Carried forward as required work before any promotion: a **`cr256-p80` arm** (if
CR-CTC's transfer gain is additive with the high dose, that bundle clears both
axes at once) and **paired seeds on the ch 256 p 0.65-vs-0.8 call**, since one
seed cannot separate a 1.5 pt dvorak difference from the 5 pt seed spread that
axis is known to have.

### 5.2 Export parity, corrected (affects the PHASE_I §7.3 record)

`phaseJ-ch256-p65` failed the export gate at 9.2e-3 — and the failure turned out
to be the gate's fault. `export_onnx.py` fed `torch.rand` for **both** features
and `layout_keys`; a white-noise trajectory on 64 random key positions is not
the operating distribution. Re-probing on real val traces at the real layout
centers (commit `1f05ea1`):

| model | real-trace sliced max abs | white-noise probe | argmax |
|---|---|---|---|
| `phaseJ-cr80` (ch 80) | 8.0e-5 | 4.2e-5 | 100/100 both |
| `resbn192i` (ch 192) | 1.49e-4 | 1.34e-4 | 100/100 both |
| `phaseJ-ch192-p80` (ch 192) | 1.60e-4 | 3.4e-5 | 100/100 both |
| `phaseJ-ch256-p65` (ch 256) | 1.47e-4 | 5.0e-3…9.7e-3 (run-varying) | 100/100 both |

**On the operating distribution the residue is essentially width-flat
(0.8–1.6e-4), and PHASE_I §7.3's "the residue grows with width" was a property
of the retired probe, not of the exports.** The white-noise probe is not
conservative in either direction (it under-reported ch 192 by 5×). The script
now asserts magnitude on real traces (`--parity-features`, default
`cache/val.npz`), keeps the noise probe as a printed diagnostic, asserts argmax
100/100 on **both**, defaults `--parity-tol` to 5e-4, and measures BN-fold drift
on the sliced contract view (the raw 65-wide max had been reporting the 9.77e-4
float32 ULP of the −1e4 pad columns as "drift"). No accuracy number anywhere in
the campaign moves: argmax parity was and is 100/100, and every published
number was decoded through the exported graph.

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

**Queued work (all detached, no orchestrator needed to keep it alive):**

* `phaseJ_queue3.sh` — waits for the three 188 k round-2 arms to print
  `reached step budget`, then `exec`s `phaseJ_round3.sh` (`phaseJ-cr192`,
  `phaseJ-cr256`, `phaseJ-futoaug`). `phaseJ-ch256-280k` keeps a 4th slot the
  whole time; the 5080 saturates at four concurrent arms (98 % util, 2.4 GB of
  16 GB — compute-bound, so a fifth buys nothing).
* Per-arm battery: `phaseJ_eval.sh <run>` (env `PARITY_TOL` overrides the
  export tolerance). Cyrillic battery: `phaseJ_eval_ru.sh <run> [layout]` — E1
  + app-ru-50k, the footing the 76.21 bar was set on.
* `jsum.py <run>…` in the workdir prints the one-line val + alt-layout summary
  used in the tables above.
* `phaseJ_queue4.sh` — round 4, scheduled **per slot** rather than per round
  (the non-CR arm finishes hours before the 2×-cost CR arms, and a freed slot
  should not idle): `futoaug` → `phaseJ-joint` (en+ru single model), `cr192` →
  `phaseJ-ru192` (ru-only ch 192 rung), `cr256` → `phaseJ-cr256-p80` (the
  §5.1b bundle: CR-CTC on the only dose that clears the euro bars).
* Battery chains run themselves as arms land: `phaseJ_eval_round2.sh` and
  `phaseJ_eval_round34.sh` (the latter routes `phaseJ-ru192` to the ru battery
  and gives `phaseJ-joint` both). So the whole of rounds 2–4 — train **and**
  eval — completes with no orchestrator attached; a successor's first act
  should be `jsum.py` over the finished runs plus `tail` of the `*.launch.log`s.
* Still unstarted and needing a decision-maker: the checkpoint soup (supply is
  accumulating in `ckpt/phaseJ-ch256-280k/`, `soup_checkpoints.py --run`), the
  winner stack + 3 seeds, preset sweeps, the pre-registered unsealing, docs.

**Early mid-schedule signal on round 2** (selection beam at ~66 k of 188 k,
noisy, recorded so a successor knows what to expect): `sw234` 84.64 and
`realalt` 85.18 vs the base's 84.30 at the same step; `yfix` 82.90, i.e. clearly
down — consistent with the §3.2 pre-registered caveat that the val HWS half
keeps the uncorrected frame.
