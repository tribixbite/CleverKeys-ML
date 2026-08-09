# Phase G — the affine-sampler fix and the upgraded student recipe

**Date:** 2026-08-09 · **Authority:** user directive of 2026-08-09 ("retrain and
reexport and re-run tests on new onnx (resbn80)"), which also pre-authorizes a
third unsealing of test-2400 **gated on the val bars** (§7).

Phase F closed with `fast_resbn80` (`resbn:80:1,2,4,8`, 94 k steps, 279,346
params, 0.215 ms) as the test-validated speed candidate. Phase G asks: with the
recipe upgraded — the latent affine-sampler bug fixed, the 188 k schedule that
was measured *after* `resbn80` was trained, and the distillation levers that were
never ablated — how much better can the same latency class get, and can the
0.186 ms class reach the margins the 0.215 ms class holds today?

The val bar is unchanged from Phase E/F (FUTO ceiling, published preset,
val-9918, AOSP STRIP trie): **85.52 / 91.54 / 92.80 / ≤3 89.29 / 4+ 83.57**.
The margins to beat are the shipping `resbn80`'s val seed-mean:
**87.47 / 92.13 / 92.89 / 90.35 / 85.98**. The stretch bar is the equal-footing
val bar — FUTO's ceiling val-tuned by the same wide grid (`FAIR_REMATCH.md` §2):
**87.48 / 92.31 / 93.03 / 89.76 / 86.29**.

Every accuracy number below is full val-9918 at the E1 preset
(`1.05, 1.1, 0.2, 0.3734, 0.9882`), beam 100, top-k 8, OOV = miss, through the
exported ONNX graph, unless a sweep section says otherwise. test-2400 is not
touched anywhere before §7, and §7 runs only if the §5 gate passes.

---

## 1. The affine-sampler bug, fixed and verified

### 1.1 The bug (found in `ALT_LAYOUT_EVAL.md` §7.2b)

`train.py`'s shared affine — the augmentation that moves the path AND the key
centers together — drew `(sx, sy) ~ U(0.85, 1.15)²` and `(tx, ty) ~ U(±0.05)²`
independently, then rejection-tested "every transformed center in [0,1]" up to
10 times, falling back to the identity (audit fix #13). en_qwerty's centers span
`cx ∈ [0.05, 0.95]`, so a horizontal *expansion* almost always violated the
bound: **31.4 % of first draws were rejected**, and the survivors were biased
toward compression. The model effectively never saw the keyboard stretched
wider, while the y axis kept its full nominal range — and the §7.2 probe table
showed the matching asymmetry (sx = 0.70 costs 51 pt of greedy, sy = 0.70 costs
4).

### 1.2 The fix — sample the feasible region exactly, couple translate to scale

`--affine-sampler coupled` (the new default; `legacy` reproduces Phase A–F).
Per axis, with `lo/hi` the min/max center and the scale about 0.5:

* the **feasible scale ceiling** is precomputed:
  `s_max = min( 1/(hi−lo), (0.5+T)/(0.5−lo), (0.5+T)/(hi−0.5) )` with
  `T = TRANS_ABS = 0.05`;
* `s ~ U(0.85, min(1.15, s_max))`;
* the **translate is then drawn from the exact interval that keeps every center
  inside [0,1] at that scale**, intersected with the nominal ±0.05 window:
  `t ~ U( max(−T, −((lo−0.5)s + 0.5)), min(T, 0.5 − (hi−0.5)s) )`.

Acceptance is 1.0 by construction — no rejection loop, no identity fallback, no
compression bias.

**One geometric fact the fix makes explicit rather than hiding:** for en_qwerty
the x span is 0.90, so `s_max_x = 1/0.90 = 1.1111`. A horizontal scale of 1.15
**cannot** keep 26 centers whose span is 0.90 inside a unit interval at any
translate — the nominal upper bound was always infeasible in x on this layout.
The legacy sampler dealt with that by silently re-rolling; the fixed sampler
realizes exact uniformity over the feasible set. (Scaling about the bounding-box
centroid — the other fix `ALT_LAYOUT_EVAL.md` suggested — is the identity here:
en_qwerty's centers are symmetric about 0.5 on both axes.)

### 1.3 Before/after, measured (`affine_stats.py`, 200,000 draws, en_qwerty)

| | acceptance | sx mean | sx median | sx p95 | sx max | sy mean | sy max |
|---|---|---|---|---|---|---|---|
| legacy (Phase A–F) | 68.6 % first-draw | **0.9554** | 0.9530 | 1.0634 | 1.1109 | 1.0000 | 1.1500 |
| **coupled (Phase G)** | **100 %** | **0.9807** | 0.9808 | 1.0980 | 1.1111 | 1.0003 | 1.1500 |

The legacy figures reproduce `ALT_LAYOUT_EVAL.md` §7.2b to the digit (mean
0.955, max 1.111, 31.5 %→31.4 % rejects). The coupled sampler realizes
`sx ~ U(0.85, 1.1111)` exactly; sy was never truncated (its span bound is 1.5)
and is unchanged. Containment (`every transformed center ∈ [0,1]`) is asserted
on all 200,000 draws of both samplers. The translate becomes slightly
scale-coupled (its window shrinks toward 0 as sx → 1.1111), which is the price
of exactness; its marginal p95 moves 0.0434 → 0.0418.

The fix is therefore a *distribution repair*, not a range extension: what
changes is the density on `sx ∈ [1.0, 1.111]` (roughly doubled) and the removal
of the compression bias — not the support. §3 measures what that is worth in
accuracy; the honest prior from the small distribution shift is "small".

---

## 2. What Phase G folds in from the campaign record, and why

| lever | source | verdict here |
|---|---|---|
| `resbn` trunk, dilations 1,2,4,8, embed_hid 96 | `PHASE_F.md` §10 | kept — strict win |
| T3 + 3× HWS oversampling | `PHASE_E.md` §4 | kept — +0.83 t1 |
| 5,000-row beam-t1 checkpoint selection | `PHASE_E.md` §3 | kept — +0.23 t1 |
| **188 k schedule** | `PHASE_F.md` §13: +0.5 t1 for small students, measured at ch 56/64 *after* `resbn80` was frozen at 94 k | **applied to ch 80 for the first time** (§3) |
| KD from our ch 192, weight 1.0, temp 2 | `PHASE_F.md` §4 | **ablated for the first time** (§3) — Phase F's largest stated evidence hole (§11.3) |
| KD temperature 4 | `PHASE_F.md` §13.2 | excluded — measured −0.59 t1 |
| 280 k schedule | `PHASE_F.md` §13.1 | excluded — second doubling bought a quarter of the first; +0.02 t5 for +50 % GPU |
| post-training int8 | `PHASE_F.md` §2/§5 | excluded — loses t5 at every size |
| depthwise-separable trunk | `PHASE_F.md` §3/§4 | excluded — −0.61 t1 at higher latency |
| feature v2, EMA, path-only jitter | `PHASE_B.md`/`PHASE_C.md` | excluded — null or negative inside the seed floor |
| 5-block trunks | `PHASE_F.md` §14 | excluded — depth past 4 blocks does not pay at these widths |
| per-model preset re-sweep | `PHASE_E.md` §5 found it transfers; `PHASE_F.md` §15.4 found λ 2.5 worth +1.01 on the app trie | **re-run wide for the new model on both tries** (§6) |

---

## 3. The lever table — paired single-seed arms at ch 80 / 188 k

All arms: `resbn:80:1,2,4,8`, embed_hid 96, T3 + 3× HWS, 188,000 steps, batch
256, lr 3e-3, wd 0.01, warmup 1,000, 5,000-row beam-t1 selection, seed 1234.
Full val-9918, E1 preset, decoded through the exported ONNX graph. KD (where on)
is teacher `phaseE-FINAL-s1234`, weight 1.0, temp 2 — the exact Phase-F setting.

| arm | sampler | KD | t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|---|---|
| `phaseF-I-resbn80x4` @94k (the shipping baseline) | legacy | ch192 s1234 | 87.41 | 92.18 | 92.85 | 90.38 | 85.86 |
| **A** `phaseG-A80-188k-legacy` | legacy | ch192 s1234 | 87.46 | 92.28 | 92.95 | 90.76 | 85.74 |
| **B** `phaseG-B80-188k` | **coupled** | ch192 s1234 | 87.52 | 92.15 | 92.76 | 91.03 | 85.69 |
| **C** `phaseG-C80-188k-nokd` | **coupled** | **none** | **88.04** | **92.39** | **93.18** | **91.30** | **86.35** |

Paired reads (single seed; sign-consistency across the five metrics is the
evidence, not any one 0.1):

* **188 k schedule at ch 80** (A − baseline): **+0.05 t1** — the +0.5 measured
  at ch 56/64 (`PHASE_F.md` §13) does *not* transfer to ch 80 *with KD on*.
* **Affine fix, KD on** (B − A): +0.06 t1, −0.13 t3, −0.19 t5, +0.27 ≤3 —
  mixed, ~null, as the small distribution repair (§1.3) predicted.
* **KD ablation** (C − B): **+0.52 t1, +0.24 t3, +0.42 t5, +0.27 ≤3, +0.66 4+
  — dropping distillation wins on every metric.** Phase F's largest stated
  evidence hole (§11.3: "the distillation contribution is unmeasured") resolves
  in the direction nobody assumed: at ch 80 / 188 k, KD from the ch 192 teacher
  was *capping* the student, not helping it. The selection-prefix beam
  (published preset) ranked C *last* of the three (84.84 vs 86.02/85.76) while
  full-val E1 ranks it first by half a point — one more instance of the
  selection-metric/report-metric divergence Phase B first flagged.

| **D** `phaseG-D80-188k-kdens` | **coupled** | **3-seed ch192 ensemble** | 87.07 | 91.97 | 92.80 | 91.18 | 84.94 |

* **Ensemble teacher** (D − B): −0.45 t1, −0.18 t3, −0.75 4+ — averaging the
  three ch 192 seeds' probabilities makes the teacher *worse* for the student,
  not better. Both KD arms lose to no-KD; the teacher choice does not rescue
  the lever. (D also trained ~2× slower per step — three teacher forwards —
  so the lever costs GPU and accuracy at once.)

| **E** `phaseG-E80-188k-legacy-nokd` | legacy | **none** | 87.94 | 92.33 | 92.98 | 91.12 | 86.29 |

### 3.2 The 2×2 factorial, attributed

|  | KD on | KD off | Δ (KD off − on) |
|---|---|---|---|
| legacy sampler | A 87.46 | E 87.94 | **+0.48** |
| coupled sampler | B 87.52 | C **88.04** | **+0.52** |
| Δ (coupled − legacy) | +0.06 | +0.10 | |

* **KD removal is the dominant lever: +0.48 / +0.52 t1**, consistent across
  both sampler states, and positive on all five metrics in the no-KD column
  (C − B: +0.52/+0.24/+0.42/+0.27/+0.66; E − A: +0.48/+0.05/+0.03/+0.36/+0.55).
* **The affine fix is a small consistent positive: +0.06 / +0.10 t1**, and
  without KD it is positive on all five (C − E: +0.10/+0.06/+0.20/+0.18/+0.06,
  including the +0.20 on t5 — the metric the fix's x-expansion coverage most
  plausibly serves). With KD it is mixed on t3/t5. Both deltas are far inside
  the single-seed floor; the sign-consistency and the mechanism (§1) are the
  evidence. Its *cross-layout* value (`ALT_LAYOUT_EVAL.md` predicted the sampler
  bias hurts transfer most) is not measured here and remains the stronger
  argument for it.
* **The 188 k schedule at ch 80 with KD was +0.05** (A vs the 94 k baseline) —
  the +0.5 measured at ch 56/64 in Phase F does not transfer to ch 80 with KD
  on. No 94 k no-KD arm was run, so schedule×KD at ch 80 is not decomposed;
  the winner recipe simply takes 188 k, which is at worst neutral.

---

## 4. The winner — `resbn:80:1,2,4,8`, 188 k, coupled sampler, **no distillation** — at three seeds

Fresh trainings at seeds 1234 / 4321 / 7777 (`phaseG-C80-188k-nokd[-s*]`).
Full val-9918, E1 preset, AOSP STRIP trie, exported ONNX graphs
(sliced-view parity 100/100 argmax on all three).

| metric | s1234 | s4321 | s7777 | **seed-mean** | sd | the bar | **Δ** | worst seed | resbn80@94k seed-mean (the incumbent) | **Δ vs incumbent** |
|---|---|---|---|---|---|---|---|---|---|---|
| overall t1 | 88.04 | 87.82 | 87.31 | **87.72** | 0.37 | 85.52 | **+2.20** | 87.31 PASS | 87.47 | **+0.25** |
| t3 | 92.39 | 92.27 | 92.09 | **92.25** | 0.15 | 91.54 | **+0.71** | 92.09 PASS | 92.13 | **+0.12** |
| t5 | 93.18 | 92.83 | 92.90 | **92.97** | 0.18 | 92.80 | **+0.17** | 92.83 PASS | 92.89 | **+0.08** |
| ≤3 t1 (n=3,389) | 91.30 | 91.09 | 89.94 | **90.78** | 0.73 | 89.29 | **+1.49** | 89.94 PASS | 90.35 | **+0.43** |
| 4+ t1 (n=6,529) | 86.35 | 86.12 | 85.94 | **86.14** | 0.21 | 83.57 | **+2.57** | 85.94 PASS | 85.98 | **+0.16** |

**All five bars clear on the seed mean and on every individual seed, with
margins larger than the incumbent's on all five.** The t5 margin roughly
doubles (+0.17 vs +0.09) and the worst-seed t5 margin moves 0.05 → 0.03 —
still a knife edge, as every model in this family has been on t5. Same graph,
same 279,346 params, same 0.215 ms latency class as the incumbent: the gain is
free at inference.

Per-source seed-mean t1: FUTO **94.56**, HWS **80.86** (incumbent: 94.79 /
80.29) — the upgrade comes from the harder HWS half (+0.57), with the FUTO half
giving back 0.23.

## 5. The gate, and the stretch bar

* **Primary target (the §7 decode gate): PASS.** 3-seed val seed-mean clears
  all five val bars, every seed clears individually, and all five margins are
  ≥ the shipping `resbn80`'s (`PHASE_F.md` §8).
* **Stretch — the equal-footing val bar** (FUTO ceiling val-tuned,
  `FAIR_REMATCH.md` §2: 87.48 / 92.31 / 93.03 / 89.76 / 86.29): **NOT met on
  the seed mean** — t1 +0.24 and ≤3 +1.02 clear, but t3 −0.06, t5 −0.06 and
  4+ −0.15 do not. Seed 1234 alone clears all five (88.04/92.39/93.18/
  91.30/86.35), but a single seed is not the gate and is not claimed. The
  0.215 ms class gets *level* with the val-tuned FUTO ceiling (three metrics
  inside ±0.15) where the incumbent was clearly behind it (−0.18 t3 / −0.14 t5
  / −0.31 4+ on val seed-means); it does not beat it.
* **Stretch — all-bars at ≤0.15 ms: not attempted.** Nothing in the lever
  table moves t5 by the ~0.2 pt that class is short; §8's ch 64 probe tests
  the 0.162 ms class instead.

---

## 8. The latency push — the upgraded recipe below 0.215 ms

### 8.1 `resbn:72:1,2,4,8` at the Phase-G recipe (`phaseG-F72-188k-nokd`) — seed 1234

229,642 params, 944,487 bytes, 0.186 ms class. Full val-9918, E1, AOSP:

| | t1 | t3 | t5 | ≤3 | 4+ | bars |
|---|---|---|---|---|---|---|
| `phaseF-N72-188k` (KD, legacy sampler) s1234 | 87.25 | 92.24 | 92.96 | 90.44 | 85.59 | 5/5 |
| **`phaseG-F72-188k-nokd` s1234** | **87.53** | **92.33** | **93.01** | **90.62** | **85.92** | **5/5** |
| Δ (recipe upgrade at ch 72) | +0.28 | +0.09 | +0.05 | +0.18 | +0.33 | |
| incumbent `resbn80`@94k **seed-mean** (the §5 margin target) | 87.47 | 92.13 | 92.89 | 90.35 | 85.98 | |

At one seed the 0.186 ms class matched the incumbent 0.215 ms model's
seed-mean, so it was promoted to three seeds.

### 8.2 `resbn72g` at three seeds — all five bars, every seed, at 0.184 ms

Seeds 4321/7777 were killed by a host reboot at step 39,000 and resumed from
`last.pt` per §7.α. Full val-9918, E1, AOSP; **val-only** (no further test
decode — the third unsealing is spent):

| metric | s1234 | s4321 | s7777 | **seed-mean** | the bar | **Δ** | worst seed | incumbent `resbn80`@94k seed-mean | **Δ vs incumbent** |
|---|---|---|---|---|---|---|---|---|---|
| t1 | 87.53 | 87.46 | 87.88 | **87.62** | 85.52 | **+2.10** | 87.46 PASS | 87.47 | **+0.15** |
| t3 | 92.33 | 92.11 | 92.22 | **92.22** | 91.54 | **+0.68** | 92.11 PASS | 92.13 | **+0.09** |
| t5 | 93.01 | 93.06 | 92.98 | **93.02** | 92.80 | **+0.22** | 92.98 PASS | 92.89 | **+0.13** |
| ≤3 | 90.62 | 90.26 | 90.56 | **90.48** | 89.29 | **+1.19** | 90.26 PASS | 90.35 | **+0.13** |
| 4+ | 85.92 | 86.00 | 86.49 | **86.14** | 83.57 | **+2.57** | 85.92 PASS | 85.98 | **+0.16** |

**The headline of the latency push: at 0.184 ms, `resbn72g` clears all five
val bars on every seed AND exceeds the incumbent 0.215 ms `fast_resbn80`'s
seed-mean on all five metrics.** Its worst-seed t5 margin is **+0.18** — the
Phase-F `resbn72`'s was +0.01 (one row), and even `resbn80`'s was +0.05. The
t5 knife edge this family has ridden since Phase E is, at this width, simply
gone. Against the old-recipe `phaseF-N72-188k` seed-mean
(87.27/92.09/92.87/90.49/85.60) the upgrade is +0.35/+0.13/+0.15/−0.01/+0.54.

Against `resbn80g` (§4) it is −0.10 t1 / −0.03 t3 / **+0.05 t5** / −0.30 ≤3 /
0.00 4+ — statistically level everywhere but ≤3. Evidence tiers differ:
`resbn80g` is **test-validated**, `resbn72g` is **val-only** and may not be
decoded on test.

One export note, disclosed: `resbn72g_s4321`'s sliced-view parity sits at the
tolerance boundary — across ~500 random draws its max |onnx−torch| ranged
6.7e-05 to **2.14e-04** (two draws above the 1e-4 assert), argmax unchanged
**500/500**, and the exported bytes are deterministic (identical sha across
re-exports). The graph is fine; the fp32 margin on this seed is just thinner
than the other five Phase-G exports (all ≤ 9.7e-05).

### 8.3 The ch 64 probe — the ≤0.162 ms answer does NOT change

`phaseG-H64-188k-nokd` (185,058 params, **0.161 ms**), seed 1234, same recipe:

| | t1 | t3 | t5 | ≤3 | 4+ | bars |
|---|---|---|---|---|---|---|
| `phaseF-L64-188k` (KD, legacy) s1234 | 87.19 | 92.09 | 92.76 | 90.29 | 85.59 | 4/5 |
| **`phaseG-H64-188k-nokd` s1234** | 87.17 | 91.83 | **92.70** | 90.12 | 85.65 | **4/5 (t5 −0.10)** |

**The no-KD gain does not transfer to ch 64** (−0.02 t1, −0.26 t3, −0.06 t5
against its KD twin, single seed) — KD's harm is capacity-dependent: negative
at ch 80 (−0.5 t1), ~null-to-positive at ch 64. Consistent with the standard
distillation picture: a teacher helps where the student lacks capacity to fit
the data and hurts where it doesn't. Consequently **Phase F's ≤0.15 ms verdict
stands unchanged under the upgraded recipe**: t5 remains the binding miss below
~0.18 ms, and the bar-clearing frontier moves only from 0.186 to **0.184 ms**
(the same graph, re-measured; the recipe change is what upgraded its margins).

### 8.4 The measured Phase-G frontier (idle, §0-of-Phase-F protocol)

| ms | model | params | bytes | t1 | t5 | bars | seeds | tier |
|---|---|---|---|---|---|---|---|---|
| 0.161 | `resbn64g` | 185,058 | 766,727 | 87.17 | 92.70 | 4/5 | 1 | val-only |
| **0.184** | **`resbn72g`** | 229,642 | 944,487 | **87.62*** | **93.02*** | **5/5, every seed** | 3 | val-only |
| 0.213 | **`resbn80g`** ← ship | 279,346 | 1,142,727 | **87.72*** | **92.97*** | **5/5, every seed** | 3 | **test-validated, both footings** |
| 0.455† | ch 128 (Phase E) | 689,282 | 2,799,865 | 87.88* | 92.96* | 5/5 | 3 | test-validated |

`*` seed-mean. † Phase-E harness figure. The 0.215 ms class now sits 0.16 t1
under the 0.455 ms ch 128 (was 0.41 under), and the 0.184 ms class 0.26 under.

### 3.1 Why Phase F adopted KD in the first place — and why this is not a contradiction

Phase F never ran a no-KD control (`PHASE_F.md` §7.1/§11.3 said so explicitly);
KD was adopted on the *a-priori* argument that a 5–6×-larger teacher should help
a small student, and every Phase-F arm inherited it. The ablation here is the
first measurement, and it says the CTC task + 3×-HWS tier already carry more
signal than the teacher's soft targets at this capacity: the KD term was pulling
the student toward a teacher whose *own* HWS half sits 15 pt below its FUTO
half, and toward its calibration rather than the data's. Phase-F *arm-vs-arm*
conclusions survive (all arms shared the same teacher — common-mode), but every
Phase-F absolute number at ≤280 k params should be read as ~0.5 t1 understated,
and the "capacity crosses t5 at 210–230 k" boundary (§14) was measured *with*
the KD handicap and may sit lower without it.

---

## 6. The per-model preset sweep — E1 confirmed on AOSP; the app footing wants λ 4

`sweep_scoring.py` on arm C's emissions, wide grids, tune val`[0:4959]` /
confirm `[4959:9918]`, boundary-reject (grids widened until the winner is
interior).

**AOSP STRIP 146,964 (the benchmark footing).** Grid 6γ × 4β × 5λ × 6 prune
pairs. The winner is **E1 exactly** — γ 1.05, λ 1.1, β 0.2 — with the prune
pair flat to ±0.06 (gp 0.25 ties gp 0.3734 at 88.57 on the sweep half; full-val
88.10 vs 88.04, inside noise). Third model in a row (`PHASE_E.md` §5, `PHASE_F.md`
§4) for which E1 transfers unchanged on this trie: **keep E1 for every
benchmark-footing number.** No golden-fixture churn from this footing.

**App `en_enhanced` 98,081 STRIP (the shipping footing).** First grid put the
winner at λ = 4.0 — the grid edge — so λ was widened to {3,4,5,6,8} with γ/β
refined; the winner is then interior and stable:

```
gamma 0.9, lambda 4.0, beta 0.25, gammaPrune 0.25, betaPrune 0.9882
```

| arm C, app trie, full val | t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|
| E1 (λ 1.1) | 87.37 | 92.90 | 93.86 | 90.50 | 85.74 |
| **app-tuned (λ 4.0)** | **88.76** | **93.22** | **94.18** | **92.42** | **86.86** |
| Δ | **+1.39** | +0.32 | +0.32 | **+1.92** | **+1.12** |

The gain is confirmed on the untouched holdout half (+4.15 t1 over the
published-preset baseline there vs +4.66 on the tuning half — no sweep-overfit
signature) and is positive on all five metrics. This is `PHASE_F.md` §15.4's
λ-scale finding (slope-matched λ ≈ 3.8–6.2 for the compressed app-trie
`log_freq`) landing where the arithmetic said it would, now with γ/β allowed to
adapt (γ 1.05 → 0.9, β 0.2 → 0.25).

**Preset decision.** Phase F §15.4 declined a λ-only +1.09 because the golden
fixture, `CtcScoringParams`, and every published number were frozen at λ = 1.1
for an already-shipped model. Phase G ships a **new** model, so the fixture is
regenerated regardless and the divergence cost collapses. Decision:

* **Benchmark footing (AOSP trie): E1, unchanged** — the sweep converged to it.
* **Shipping footing (app trie): adopt `0.9 / 4.0 / 0.25 / 0.25 / 0.9882`** as
  the app preset for this model, worth +1.39 t1 / +0.32 t5 on the configuration
  users actually run. The golden fixture for the shipped model must be
  regenerated at THIS preset (fixture must match what the app runs).
* Caveat that travels with λ 4.0: the user-dictionary merge injects
  top-of-scale (freq 255) competitors, and a 3.6× larger λ amplifies them.
  No eval here includes a user dictionary (`PHASE_F.md` §15.5, unchanged).

---

## 4. (reserved — winner recipe at three seeds)

## 5. (reserved — the val gate)

## 6. (reserved — the per-model preset sweep)

## 7. Pre-registration — the third unsealing of test-2400 (`resbn80g`)

**Written and committed before any decode runs.** Everything below is fixed at
commit time; §7.5 (results) is appended afterwards and may not restate the plan.

### 7.1 Authority, and why this is not iterative tuning

**The user — who owns this benchmark — directed on 2026-08-09: "retrain and
reexport and re-run tests on new onnx (resbn80)", gating the decode on the
3-seed val seed-mean clearing all five val bars.** §4/§5 record that gate as
passed, with margins ≥ the incumbent's on all five. That directive is the entire
authority for this decode, exactly as the user's order was for the second
unsealing (`PHASE_F.md` §16.1).

Why this is a first decode, not a selection loop: the `phaseG-C80-188k-nokd`
recipe (architecture, width, dilations, schedule, **no distillation**, coupled
affine sampler, per-seed checkpoints selected on beam top-1 over a 5,000-row
*val* prefix) and both decode presets were fixed on val-9918 and committed
(§3–§6) before this section was written; the artifact sha256s are frozen in
§7.3. No Phase-G model has ever touched test-2400. The result cannot feed back:
Phase G's training is closed. What it costs is stated: **test-2400 will have
been read three times**, and every future claim rests on a yet more worn split.

### 7.2 The claim being registered

> On the 2,400-row test split, `resbn80g` (279,346 params, 0.215 ms class),
> decoded at the frozen presets below, is compared against (a) FUTO's published
> encoder+refinement ceiling at FUTO's published preset — the published bar —
> and (b) the val-tuned equal-footing bar from `FAIR_REMATCH.md` (config A
> only, where that bar exists). A pass on (a) moves `resbn80g` to
> **test-validated**. Comparison (b) is reported for calibration; §5 already
> shows the val seed-mean does not clear the equal-footing val bar, so no
> equal-footing superiority claim is being attempted, and none may be written
> from this decode.

### 7.3 The runs — hard cap, one decode each, no iteration

**Maximum 2 configurations × 3 seeds = 6 decodes. Nothing more.** No fourth
seed, no alternate preset, no `--limit` warm-up, no retry on partial output.

| # | trie | preset | bar |
|---|---|---|---|
| A | AOSP STRIP 146,964 (`data/futo_en_wordlist.combined`) | E1 `1.05,1.1,0.2,0.3734,0.9882` | published `84.83/91.04/92.08/89.57/82.40`; equal-footing `87.12/92.29/92.96/89.94/85.68` |
| B | app `en_enhanced.json` STRIP 98,081, `--vocab-kind json-strip` | **the adopted app preset** `0.9,4.0,0.25,0.25,0.9882` (§6) | trie-matched published-preset bar `84.92/91.54/92.96/89.57/82.52` (`PHASE_F.md` §15.2) |

Config B decodes at the preset the app will actually ship (§6). Its bar is
FUTO's published preset on the same trie, so config B is a tuned-vs-published
comparison — **the asymmetry is declared here, in advance**: no val-tuned FUTO
bar exists on the app trie, and config B supports a shipping-validation claim,
not an equal-footing one.

Frozen: beam width 100, top-k 8, OOV = miss, strata ≤3 n=815 / 4+ n=1,585,
seed-mean over 1234/4321/7777, per-source split read from
`cache/holdout_source_tags.json["test"]` without extra decodes. Artifacts
(byte-identical to `ckpt/<arm>/ctc_swipe_encoder.onnx`, sliced-view parity
100/100 at export):

```
330cadfbaa7334eaeaeab93762084181b70710fe9d59cbd69600a6de468fe1a0  resbn80g_s1234.onnx
c9379c60a23bec4ca300512d2930b7a724aad91b761597972446a6577f5d5bab  resbn80g_s4321.onnx
3e303d46abaff4bfe31779de35fb9fc81e63f1ae8fd5ab554a9db205f167191a  resbn80g_s7777.onnx
```

```bash
for a in phaseG-C80-188k-nokd phaseG-C80-188k-nokd-s4321 phaseG-C80-188k-nokd-s7777; do
  python3 eval_beam.py --onnx ckpt/$a/ctc_swipe_encoder.onnx \
    --test data/test_hwsfuto.jsonl --preset 1.05,1.1,0.2,0.3734,0.9882 \
    --beam-width 100 --top-k 8 --unseal-test --out ckpt/$a/test2400_g_e1.jsonl
  python3 eval_beam.py --onnx ckpt/$a/ctc_swipe_encoder.onnx \
    --test data/test_hwsfuto.jsonl --preset 0.9,4.0,0.25,0.25,0.9882 \
    --vocab <app en_enhanced.json> --vocab-kind json-strip \
    --beam-width 100 --top-k 8 --unseal-test --out ckpt/$a/test2400_g_app.jsonl
done
```

### 7.4 Pre-stated expectations (so a miss cannot be re-explained afterwards)

The val→test shift used is the one measured **on this same architecture class**
at the second unsealing (`PHASE_F.md` §16.5: config A −0.18/−0.24/−0.07/+0.82/
−0.68; config B −0.42/−0.11/−0.26/+0.25/−0.74), not the ch128/ch192 shift that
§16.4 wrongly extrapolated from.

* Config A val seed-mean 87.72/92.25/92.97/90.78/86.14 → **predicted test
  87.54 / 92.01 / 92.90 / 91.60 / 85.46**, i.e. +2.71/+0.97/+0.82/+2.03/+3.06
  over the published bar — expected pass on all five, t5 narrowest.
* Config B val seed-mean (app preset: 88.76+88.69+88.18, 93.22+93.30+93.00,
  94.18+94.02+93.99, 92.42+92.12+91.21, 86.86+86.90+86.61 → 88.54 / 93.17 /
  94.06 / 91.92 / 86.79) → **predicted test 88.12 / 93.06 / 93.80 / 92.17 /
  86.05**, i.e. +3.20/+1.52/+0.84/+2.60/+3.53 over the trie-matched bar —
  expected pass on all five.
* Against the equal-footing bar (config A only): predicted +0.42 t1, −0.28 t3,
  −0.06 t5, +1.66 ≤3, −0.22 4+ — **expected NOT to clear all five**, matching
  §5's val verdict. This is stated so a mixed result there cannot later be
  spun either way.

**A 4-of-5 result against the published bar is a failed gate** and will be
written as one. All five numbers for both configurations are reported
regardless of outcome; the tier moves only if config A clears all five on the
seed mean *and* on every seed — the same rule as both prior unsealings.

### 7.α Workdir note — reboots

The host rebooted twice during this phase. The first cost nothing (no run in
flight was lost); the second killed `phaseG-F72-188k-nokd-s4321`/`-s7777` at
step 39,000 and `phaseG-H64-188k-nokd` at its start. The F72 seeds were resumed
from `last.pt` with identical run names and arguments (`--total-steps` keys the
cosine schedule to the restored global step, so the resumed halves follow the
identical LR trajectory — the Phase-E §7 protocol); H64 was restarted from
scratch. All six §7.3 decodes had already completed and been committed before
the second reboot; nothing in §7.5 is affected.

### 7.5 Result — `resbn80g` clears all five test bars on both footings, on every seed

Run 2026-08-09 exactly as registered: six decodes, one per (config, seed), no
warm-up, no retry. `seal.py` logged the 2,400/2,400 overlap and the
`--unseal-test` override on each; the ledger entry is
`test2400_seal.json["test-2400"]["unsealings"][2]`. OOV = miss (86 rows config
A, 64 config B). Greedy-CTC 70.96 / 69.71 / 67.96 %.

#### Config A — AOSP STRIP 146,964, E1 preset

| metric | s1234 | s4321 | s7777 | **seed-mean** | worst | **published bar** | **Δ** | gate |
|---|---|---|---|---|---|---|---|---|
| t1 | 87.83 | 88.08 | 87.12 | **87.68** | 87.12 | 84.83 | **+2.85** | **PASS** |
| t3 | 92.29 | 92.38 | 91.88 | **92.18** | 91.88 | 91.04 | **+1.14** | **PASS** |
| t5 | 93.04 | 92.92 | 92.50 | **92.82** | 92.50 | 92.08 | **+0.74** | **PASS** |
| ≤3 (n=815) | 91.17 | 91.04 | 90.18 | **90.80** | 90.18 | 89.57 | **+1.23** | **PASS** |
| 4+ (n=1,585) | 86.12 | 86.56 | 85.55 | **86.08** | 85.55 | 82.40 | **+3.68** | **PASS** |

**All five clear on the seed mean and on every individual seed → `resbn80g` is
test-validated.** Against the incumbent `fast_resbn80`'s config-A test
seed-mean (87.29/91.89/92.82/91.17/85.30): **+0.39 t1, +0.29 t3, +0.00 t5,
−0.37 ≤3, +0.78 4+** — the upgrade transfers to test on t1/t3/4+, t5 is level,
and ≤3 gives back a third of its val gain.

#### Config B — the shipping configuration: app trie 98,081 at the app-tuned preset

| metric | s1234 | s4321 | s7777 | **seed-mean** | worst | **trie-matched bar** | **Δ** | gate |
|---|---|---|---|---|---|---|---|---|
| t1 | 88.42 | 88.29 | 87.71 | **88.14** | 87.71 | 84.92 | **+3.22** | **PASS** |
| t3 | 93.17 | 93.38 | 93.12 | **93.22** | 93.12 | 91.54 | **+1.68** | **PASS** |
| t5 | 94.08 | 93.92 | 93.71 | **93.90** | 93.71 | 92.96 | **+0.94** | **PASS** |
| ≤3 (n=815) | 92.15 | 92.27 | 91.17 | **91.86** | 91.17 | 89.57 | **+2.29** | **PASS** |
| 4+ (n=1,585) | 86.50 | 86.25 | 85.93 | **86.23** | 85.93 | 82.52 | **+3.71** | **PASS** |

**All five clear, on every seed, with the worst-seed t5 margin at +0.75 —
against the +0.08 knife edge the incumbent shipped with.** The declared
asymmetry stands: our preset is val-tuned, the bar's is published; config B is
a shipping validation, not an equal-footing claim.

#### The predictions, scored

| | t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|
| config A, predicted → measured | 87.54 → **87.68** (+0.14) | 92.01 → **92.18** (+0.17) | 92.90 → **92.82** (−0.08) | 91.60 → **90.80** (−0.80) | 85.46 → **86.08** (+0.62) |
| config B, predicted → measured | 88.12 → **88.14** (+0.02) | 93.06 → **93.22** (+0.16) | 93.80 → **93.90** (+0.10) | 92.17 → **91.86** (−0.31) | 86.05 → **86.23** (+0.18) |

Using the same-architecture val→test shift fixed the aggregate metrics (config
B within 0.2 everywhere; config A within 0.2 on t1/t3/t5) but the ≤3 stratum
again moved oppositely to its prediction (−0.80) — the short-word stratum's
val→test behaviour remains the unstable one across all three unsealings.

#### Equal footing (config A vs the val-tuned bar 87.12/92.29/92.96/89.94/85.68)

Seed-mean Δ: **+0.56 t1, −0.11 t3, −0.14 t5, +0.86 ≤3, +0.40 4+** — 3 of 5,
as §7.4 predicted ("expected NOT to clear all five"; the miss set differs:
4+ came in positive, ≤3's margin halved). Exact paired McNemar on t1 against
FUTO's per-row val-tuned output: **+17 (p 0.17), +23 (p 0.052), +0 (p 1.00)**
— level-to-slightly-ahead, resolved on no seed. That is still a real move from
the incumbent, whose McNemar had one net-*negative* seed and which lost t3/t5/4+
by 0.14–0.40: `resbn80g` turns three equal-footing losses into two −0.1 ties
and a +0.40 win. **No equal-footing superiority claim is made**, per §7.2.

#### Per-source

Seed-mean t1 by corpus half: config A **94.80 FUTO / 80.36 HWS** (spread
14.44), config B **94.55 / 81.54** (13.01). The 14-point internal spread is
unchanged from every prior read of this split; the app footing narrows it by
~1.4 pt from the HWS side.
