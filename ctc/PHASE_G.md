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

*(pending: E — legacy + no-KD, the fourth factorial cell, which attributes C's
gain between the sampler and the KD removal.)*

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

## 7. (reserved — pre-registration of the third unsealing; written and committed
before any decode, only if §5 passes)
