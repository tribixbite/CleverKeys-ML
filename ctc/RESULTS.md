# CTC Swipe Encoder — Training Results

# Campaign 2 (2026-08-07/08): FUTO ceiling beaten as registered

**Status: the sealed test-2400 decode happened once, was pre-registered, and was
independently audited post-hoc.** Both shipping configurations exceed all five
published FUTO-ceiling numbers — on the seed-mean *and* on every one of six
individual runs.

> **Amended 2026-08-08 — test-2400 has since been read a second time, on the
> user's explicit order, for `fast_resbn80` only.** See "The second unsealing"
> below and `PHASE_F.md` §16. The split has now been read twice; the ledger of
> both reads is in `test2400_seal.json["test-2400"]["unsealings"]`. Nothing else
> was decoded, and no third unsealing is contemplated.

Read in order: `PHASE_A.md` → `PHASE_B.md` → `PHASE_C.md` → `PHASE_D.md` →
`PHASE_E.md`, then `AUDIT_PREDECODE.md` (the adversarial audit that gated the
decode) and `AUDIT_FINAL.md` (the post-decode verification). `DATA_TIERS.md` has
the provenance and contamination audit.

## The claim, verbatim as registered

Registered in `AUDIT_PREDECODE.md` §E **before** the decode, and reproduced from
`AUDIT_FINAL.md` §7:

> **Claim as registered:** on the sealed 2,400-row test split, the Phase-E
> configuration, decoded at the val-tuned E1 preset, is compared against FUTO's
> published encoder+refinement ceiling decoded at FUTO's published preset. A pass
> is not a claim of superiority on equal footing — the presets are not matched,
> and no attempt to re-tune FUTO's preset was possible (its weights are not
> available here).

`AUDIT_FINAL.md` §7 verdict: **"Does the evidence support the claim AS REGISTERED?
— YES."** It also states what may never be written: *that this model beats FUTO's
decoder on equal footing.* See "The asymmetry" below.

## Verified test-2400 results

Every number recomputed by the audit from the per-trace `test2400_e1.jsonl` dumps,
not from log footers. 2,400 rows, strata ≤3 n=815 / 4+ n=1,585 (matching the bar's
own n's). **86 out-of-vocabulary targets are counted as misses**, not excluded.

### ch 192 — `phaseE-FINAL`, 1,525,378 params, 0.877 ms

| seed | t1 | t3 | t5 | ≤3 | 4+ | all five? |
|---|---|---|---|---|---|---|
| 1234 | 88.79 | 92.54 | 93.46 | 91.53 | 87.38 | yes |
| 4321 | 87.88 | 92.71 | 93.50 | 90.92 | 86.31 | yes |
| 7777 | 88.42 | 92.71 | 93.54 | 91.66 | 86.75 | yes |
| **seed-mean** | **88.36** | **92.65** | **93.50** | **91.37** | **86.81** | **yes** |
| seed sd | 0.46 | 0.10 | 0.04 | 0.39 | 0.54 | |
| worst seed | 87.88 | 92.54 | 93.46 | 90.92 | 86.31 | **yes** |
| **the bar** | 84.83 | 91.04 | 92.08 | 89.57 | 82.40 | |
| **Δ** | **+3.53** | **+1.61** | **+1.42** | **+1.80** | **+4.41** | |

### ch 128 — `phaseE-E3b-hws3x`, 689,282 params, 0.455 ms

| seed | t1 | t3 | t5 | ≤3 | 4+ | all five? |
|---|---|---|---|---|---|---|
| 1234 | 88.04 | 92.08 | 92.96 | 91.29 | 86.37 | yes |
| 4321 | 87.83 | 92.46 | 93.12 | 90.55 | 86.44 | yes |
| 7777 | 87.88 | 92.46 | 92.92 | 91.41 | 86.06 | yes |
| **seed-mean** | **87.92** | **92.33** | **93.00** | **91.08** | **86.29** | **yes** |
| seed sd | 0.11 | 0.22 | 0.11 | 0.46 | 0.20 | |
| worst seed | 87.83 | 92.08 | 92.92 | 90.55 | 86.06 | **yes** |
| **Δ** | **+3.09** | **+1.29** | **+0.92** | **+1.51** | **+3.89** | |

### Per-source — the aggregate hides a 14-point internal spread

| config | FUTO half (n=1,217) | HWS half (n=1,183) | spread |
|---|---|---|---|
| ch 192 | t1 **95.32** (t3 99.07, t5 99.48, ≤3 96.93, 4+ 94.27) | t1 **81.21** (86.05, 87.35, 83.48, 80.30) | **14.11 pt** |
| ch 128 | t1 **95.07** (98.88, 99.32, 96.72, 94.00) | t1 **80.56** (85.60, 86.50, 83.09, 79.55) | **14.51 pt** |

The 88.36 headline is the average of a 95.3 and an 81.2. On the How-We-Swipe half
alone the model is 3.6–4.3 pt *below* the aggregate bar.

## The second unsealing — `fast_resbn80` is test-validated (2026-08-08)

### The disclosure, first

**Who ordered it.** The user, who owns this benchmark, explicitly ordered
test-validation of `fast_resbn80`. That is the entire authority for the decode.
`AUDIT_FINAL.md` §7 had declared the seal spent and `PHASE_F.md` §11.1 had said
throughout that no Phase-F artifact may be quoted as test-validated; both
statements are superseded **for this one model only**, by instruction, not by
anything that changed in the evidence.

**Why it is not iterative tuning.** The seal doctrine exists to stop a selection
loop from touching test. This decode is not one: `fast_resbn80`'s architecture,
width, dilations, schedule, distillation teacher/weight/temperature, its three
per-seed checkpoints (selected on beam top-1 over a 5,000-row **val** prefix) and
the E1 preset it decodes at were all fixed on val-9918 and frozen in `PHASE_F.md`
§8/§9 before this task existed; the published artifact sha256 are unchanged. The
first unsealing decoded **ch 128 and ch 192 only** — no `resbn` model had ever
touched test-2400, so this is a *first* decode for this model rather than a
re-decode of a tuned variant. Nothing was being chosen: the preset is frozen, the
seeds are the three already published, Phase F is closed, and the result cannot
feed back into any model or hyper-parameter.

**What was pre-stated, before the numbers were seen.** `PHASE_F.md` §16 was
written and **committed before the decode ran** (commit `50c303a`). It registered:
the claim wording; a hard cap of **2 configurations × 3 seeds = 6 decodes, one
each, no warm-up, no retry on partial output**; the frozen preset, beam width,
top-k, artifacts and metric definitions; the requirement to report all five
numbers for both configurations regardless of outcome; that **a 4-of-5 result is a
failed gate**; and a numeric prediction — config-A seed-mean
87.64/92.35/93.12/90.66/86.09, derived from val plus the val→test shift the first
unsealing showed. **That prediction was wrong in the unfavourable direction on
four of five metrics** (measured −0.35/−0.46/−0.30/+0.51/−0.79 against it), which
is recorded in `PHASE_F.md` §16.5 rather than quietly dropped.

**What it costs.** test-2400 has now been read twice. Every future claim rests on
a more worn split, and that is the price of this table.

### The numbers

`fast_resbn80` — 279,346 params, 0.215 ms, seeds 1234/4321/7777, E1 preset, beam
100, top-k 8, OOV counted as a miss. Two configurations: **A** the AOSP STRIP
146,964-word trie (protocol-identical to the Phase-E decode, comparable to the
published bar) and **B** the shipping lexicon `en_enhanced.json` (98,081 words),
compared against a bar re-measured on that same trie from FUTO's real weights
(`PHASE_F.md` §15.2 — benchmarking only, no training contact).

| config A (AOSP 146,964) | s1234 | s4321 | s7777 | **seed-mean** | sd | worst | bar | **Δ** | z |
|---|---|---|---|---|---|---|---|---|---|
| t1 | 86.75 | 87.42 | 87.71 | **87.29** | 0.40 | 86.75 | 84.83 | **+2.46** | **3.4** |
| t3 | 91.42 | 92.12 | 92.12 | **91.89** | 0.33 | 91.42 | 91.04 | **+0.85** | 1.5 |
| t5 | 92.62 | 92.83 | 93.00 | **92.82** | 0.15 | 92.62 | 92.08 | **+0.74** | 1.3 |
| ≤3 (n=815) | 90.80 | 91.53 | 91.17 | **91.17** | 0.30 | 90.80 | 89.57 | **+1.60** | 1.5 |
| 4+ (n=1,585) | 84.67 | 85.30 | 85.93 | **85.30** | 0.51 | 84.67 | 82.40 | **+2.90** | **3.0** |

| config B (app 98,081) | s1234 | s4321 | s7777 | **seed-mean** | sd | worst | bar | **Δ** | z |
|---|---|---|---|---|---|---|---|---|---|
| t1 | 85.96 | 86.38 | 87.21 | **86.51** | 0.52 | 85.96 | 84.92 | **+1.59** | 2.2 |
| t3 | 91.92 | 92.42 | 92.50 | **92.28** | 0.26 | 91.92 | 91.54 | **+0.74** | 1.3 |
| t5 | 93.04 | 93.33 | 93.38 | **93.25** | 0.15 | 93.04 | 92.96 | **+0.29** | 0.6 |
| ≤3 (n=815) | 90.18 | 90.55 | 91.53 | **90.76** | 0.57 | 90.18 | 89.57 | **+1.19** | 1.1 |
| 4+ (n=1,585) | 83.79 | 84.23 | 84.98 | **84.33** | 0.49 | 83.79 | 82.52 | **+1.81** | 1.9 |

**All five bars clear, on the seed mean and on every individual seed, under both
lexicons.** `fast_resbn80`'s evidence tier is therefore **test-validated**, and it
joins ch 128 and ch 192 as the only configurations that are. Two things to read
carefully: config B's **top-5 worst-seed margin is +0.08 pt — two rows of 2,400**,
the same knife edge Phase F flagged on val; and the statistical resolution is
weaker than the first unsealing's — only t1 and 4+ resolve at z > 2 under config
A, and nothing does under config B.

Per-source seed-mean top-1: config A **94.63** FUTO / **79.74** HWS (spread 14.89),
config B **93.37** / **79.46** (13.91), against ch 128's 95.07 / 80.56. The
14-point internal spread is unchanged, and the app lexicon costs the FUTO half
1.26 pt while leaving the HWS half flat.

Against the ch 128 anchor at the same trie, `fast_resbn80` is **−0.63 t1 / −0.44
t3 / −0.18 t5 / +0.09 ≤3 / −0.99 4+** — the val-measured −0.61 t1 trade transfers
to test essentially unchanged. **`fast_resbn72` (0.186 ms) and every other Phase-F
artifact remain val-only and were not decoded.**

## Statistical resolution — three of five bars are not resolved

Unpaired binomial SE against the published bar treated as a fixed estimate on the
same rows (`AUDIT_FINAL.md` §5; a paired test is impossible — FUTO's per-row output
is unavailable):

| metric | n | ch192 Δ | SE | **z** | ch128 Δ | **z** |
|---|---|---|---|---|---|---|
| t1 | 2,400 | +3.53 | 0.98 | **3.6 — resolved** | +3.09 | **3.1 — resolved** |
| 4+ | 1,585 | +4.41 | 1.28 | **3.4 — resolved** | +3.89 | **3.0 — resolved** |
| t3 | 2,400 | +1.61 | 0.79 | 2.0 | +1.29 | 1.6 |
| t5 | 2,400 | +1.42 | 0.75 | 1.9 | +0.92 | 1.2 |
| ≤3 | 815 | +1.80 | 1.45 | **1.2 — not resolved** | +1.51 | **1.0 — not resolved** |

The correct statement is: **all five point estimates clear, on every seed; two
clear with statistical confidence (t1, 4+); three are positive but within the noise
the row counts admit.** Seed variance is not the limiting factor (sd 0.04–0.54);
row sampling on a 2,400-row split is.

## The asymmetry: the published-preset control

Our decode preset was fitted on val-9918 by a five-parameter grid search; the FUTO
ceiling is quoted at its own published preset. The control, measured on **val**
(ch192, 3-seed mean) at the published `encoderOnly` preset (`AUDIT_FINAL.md` §6.1):

| | t1 | t3 | t5 | ≤3 | 4+ | bars cleared |
|---|---|---|---|---|---|---|
| published preset (matched footing) | 85.78 | 91.66 | 92.67 | 88.10 | 84.58 | **3 of 5** (t5 −0.13, ≤3 −1.19) |
| E1 tuned preset | 88.06 | 92.32 | 93.08 | 90.86 | 86.62 | 5 of 5 |

**The tuning is worth +2.29 pt top-1 on this exact model** — comparable to the
entire test margin on t1 and larger than the margin on t3, t5 and ≤3.

### ⚠ SUPERSEDED 2026-08-08 — the rematch was run, and the asymmetry was material

This section previously said FUTO's headroom was "untested and, with no FUTO weights
on this machine, untestable here", and that no test decode may be spent on a fair
rematch. FUTO's weights were downloaded, hash-verified and re-run
(`FUTO_WEIGHTS_VERIFICATION.md`), and their scoring preset was then swept on val by
the same wide grid that produced E1 (`FAIR_REMATCH.md`). **Our models were not
re-decoded** — the frozen `test2400_e1.jsonl` dumps were re-read — so our seal was
not touched; only FUTO's fixed third-party engine was decoded again.

**The equal-footing question is answered.** Tuning is worth **+1.94 pt t1 to FUTO**
against +2.29 to us — the same order. Against the val-tuned bar (test-2400, STRIP
trie, both engines tuned on the same val rows):

| | t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|
| FUTO ceiling, val-tuned (the equal-footing bar) | 87.12 | 92.29 | 92.96 | 89.94 | 85.68 |
| ch 192 Δ | **+1.24** | **+0.36** | **+0.54** | **+1.43** | **+1.14** |
| ch 128 Δ | **+0.79** | **+0.04** | **+0.04** | **+1.15** | **+0.61** |

All five point estimates still favour us for both configurations, but the margins
shrink by roughly two thirds, and ch 128's t3/t5 leads (+0.04 = one trace) are ties.
Exact paired McNemar on top-1 — now possible, since FUTO's per-row output exists —
**resolves ch 192 on two of three seeds and ch 128 on none.**

The prohibition on writing *that this model beats FUTO's decoder on equal footing*
is lifted **for ch 192, qualified**; it **still stands for ch 128**, the shipping
candidate, whose lead is not statistically resolvable at n = 2,400. See
`FAIR_REMATCH.md` §5 for the verdict table and §7 for the caveats (chief among them:
FUTO's context LM is still not in the bar).

## ⚠ Retraction — the old "+0.21 pt maximum headroom" scoring claim

The previous edition of this file and `README.md` recommended keeping
`CtcScoringParams.encoderOnly` unchanged, on the basis that a two-pass val sweep
plus a full-val headroom grid "bounded the best reachable gain at +0.21 pt top-1".

**That bound is withdrawn.** Every grid behind it spanned γ ∈ [0.30, 0.51],
β ∈ [0.89, 1.08], λ ≤ 0.026 — all centred on the published preset. The optimum for
our emissions is at **γ ≈ 1.05, β ≈ 0.2, λ ≈ 1.1**, outside every grid the campaign
had run. Re-swept wide on the *same* `r2` model the gain is **+4.25 pt top-1 on
untouched val rows** — the bound understated it by ~20×. See `PHASE_E.md` §1.

Arm-vs-arm conclusions in phases A–D are unaffected (all arms were decoded at the
same preset, so the mis-tuning was common-mode). Every **absolute** number in those
phases is understated by 2–5 pt.

## Shipping recommendation

**Ship ch 128** — `artifacts/ch128_s1234.onnx`, 689,282 params, **0.455 ms**
single-thread batch-1 CPU. It clears all five bars on every seed, and ch 192 buys
only +0.19 t1 on val (paired, three seeds) for 1.9× the encoder time and 2.2× the
parameters — while being *behind* on the ≤3 stratum. ch 192
(`artifacts/ch192_s1234.onnx`, 0.877 ms) is the max-accuracy alternative if the
device budget allows.

**Or ship the Phase-F speed variant, with its weaker evidence stated.**
`artifacts/fast_resbn72_s1234.onnx` — 229,642 params, **0.186 ms**, 0.94 MB —
clears all five **val** bars on the seed mean and on every individual seed, at
**2.55× the speed, 33 % of the parameters and 34 % of the bytes** of ch 128, for
−0.61 t1 on the val seed-mean (87.27 vs 87.88, both three seeds). It uses the
`resbn` trunk (dense convolutions with BatchNorms folded into them at export, so
the graph carries no normalization node), 188 k training steps, and is distilled
from our own ch 192 checkpoint. **It has never been decoded on test-2400 and never
may be** — the seal is spent — so unlike ch 128 it carries val evidence only.

**`fast_resbn80` is the better-evidenced of the two speed variants, and as of
2026-08-08 it is test-validated.** `artifacts/fast_resbn80_s1234.onnx` — 279,346
params, **0.215 ms**, 1.1 MB, 2.20× — clears all five bars on **val and on
test-2400**, on every seed, at both the AOSP tuning lexicon and the shipped
`en_enhanced.json` (see "The second unsealing" above). Against `resbn72` its val
t5 seed-mean is statistically the same (92.89 vs 92.87) but its **worst seed
clears the val t5 bar by 0.05 pt against `resbn72`'s 0.01** — and unlike
`resbn72` it now has test evidence. If APK size or latency matters more than the
0.63 t1 gap to ch 128, this is the variant to take.

`PHASE_F.md` has the frontier and the negative results. Everything at or under
0.15 ms misses top-5 — by 0.19 pt at 0.141 ms and 0.13 pt (three seeds) at
0.162 ms — and it stays missed after tripling the training schedule to 280 k steps
(+0.06 t5) or doubling the distillation temperature (−0.20 t5). The constraint is
capacity: t5 crosses the bar at 210–230 k parameters.

**Set `CtcScoringParams` to the E1 preset** — this is a required change, not an
option, for either artifact: at the published preset the same model clears only
3 of 5 bars.

```kotlin
CtcScoringParams(gamma = 1.05, lambda = 1.1, beta = 0.2, alpha = 0.0,
                 gammaPrune = 0.3734, betaPrune = 0.9882)
```

## Artifacts

`artifacts/`, all opset 17, fp32, static shapes `[1,2,64]/[1,64,2]/[1,64]` →
`[1,32,65]/[1,32,64]/[1,32,1]`, zero `Einsum`. Byte-identical to the checkpoints
the audited decode ran on (verified by sha256 against `ckpt/<arm>/`).

| file | arm | params | bytes | sha256 |
|---|---|---|---|---|
| `ch128_s1234.onnx` ← **ship** | `phaseE-E3b-hws3x` | 689,282 | 2,799,865 | `6c1144949e545f626419e1fa7b29e80f9ecf3e303886f30411fc37ae72c45c51` |
| `ch128_s4321.onnx` | `phaseE-E3b-hws3x-s4321` | 689,282 | 2,799,865 | `1eac209332fe6fd52eb7edf2ce52ae77a52552956fdfe7f333d74f2cf46ecce6` |
| `ch128_s7777.onnx` | `phaseE-E3b-hws3x-s7777` | 689,282 | 2,799,865 | `8e910571b748290cb09fdd09e5531cc2aad6d5c09c7fd9d83d57c84ad67dda8b` |
| `ch192_s1234.onnx` | `phaseE-FINAL-s1234` | 1,525,378 | 6,144,249 | `d5b5f10ea16f08743d0742b3c60aa37a469ada11c418a7f459d5ae4cff20c666` |
| `ch192_s4321.onnx` | `phaseE-FINAL-s4321` | 1,525,378 | 6,144,249 | `b020b841abfb011779e2584e418cc651bfcac988a06bfcff2aeea5862bfabab3` |
| `ch192_s7777.onnx` | `phaseE-FINAL-s7777` | 1,525,378 | 6,144,249 | `a182191152ad77b233a73bc79750b0dda51bdbcf7fcb76ddaaad6d17016eee79` |
| `ctc_model_golden.json` | golden fixture, from `ch128_s1234` **at the E1 preset** | — | 140,204 | `a18ea58cd662b0e18b6daadaf417361f93fd0b146ce6478d4d6a62e7e185fa8a` |
| `ctc_swipe_encoder.onnx` | ⚠ **superseded** pre-campaign `r2` | 394,114 | 1,619,140 | `fcf1633167b10f5c28e7c4dc16a9bba178bacc9e2b76efb06d792162dc99d0b7` |

Phase-F additions. Same contract, opset 17, fp32, plus zero normalization nodes
(BatchNorm folded at export). Full table, parity checks and the frontier in
`PHASE_F.md` §6/§8/§9. The `resbn72` rows are **val-validated only** and have
never been decoded on test-2400; the `resbn80` rows were test-validated by the
second unsealing (`PHASE_F.md` §16.5).

| file | arm | params | bytes | ms | all five val bars | test |
|---|---|---|---|---|---|---|
| `fast_resbn72_s1234.onnx` ← Phase-F candidate | `phaseF-N72-188k` | 229,642 | 944,487 | 0.186 | **yes**, every seed | never decoded |
| `fast_resbn72_s4321.onnx` | `phaseF-N72-188k-s4321` | 229,642 | 944,487 | 0.186 | **yes** | never decoded |
| `fast_resbn72_s7777.onnx` | `phaseF-N72-188k-s7777` | 229,642 | 944,487 | 0.186 | **yes** | never decoded |
| `fast_resbn80_s1234.onnx` — wider t5 margin | `phaseF-I-resbn80x4` | 279,346 | 1,142,727 | 0.215 | **yes**, every seed | **all five, both lexicons** |
| `fast_resbn80_s4321.onnx` | `phaseF-FINAL-resbn80x4-s4321` | 279,346 | 1,142,727 | 0.215 | **yes** | **all five** |
| `fast_resbn80_s7777.onnx` | `phaseF-FINAL-resbn80x4-s7777` | 279,346 | 1,142,727 | 0.215 | **yes** | **all five** |
| `fast_resbn64_188k_s1234.onnx` ⚠ frontier evidence | `phaseF-L64-188k` | 185,058 | 766,727 | 0.162 | **no** — t5 92.76 vs 92.80 |
| `fast_resbn56_188k_s1234.onnx` ⚠ frontier evidence | `phaseF-L56-188k` | 145,594 | 609,445 | 0.142 | **no** — t5 92.65 vs 92.80 |

`ctc_model_golden.json` records its own `source_onnx_sha256` and `preset`, and was
regenerated at `1.05,1.1,0.2,0.3734,0.9882` — the fixture must match the preset the
app actually ships, or the parity test asserts against a configuration nothing runs.
For G3 it was regenerated once more (same model, same preset — the 4 beam cases are
byte-identical) to add the 6 `"featurize"`-kind cases `CtcParityTest` requires and a
top-level `layout` block (the exact en_qwerty letters/centers the emissions were
generated against) for the app-side ONNX-backed `CtcEmissionModel` parity test. See
`APP_INTEGRATION_PLAN.md`.
Note `model_cat` decodes to `car`: these are synthetic straight-line paths, and the
fixture is a **parity** artifact (Kotlin must reproduce Python bit-for-bit), not an
accuracy artifact.

## Caveats that travel with every number above

1. **Preset asymmetry** — the largest threat; quantified above at ~2.3 pt.
2. **Contributor contamination.** T3 applies no session or participant exclusion;
   every contributor of every val and test row is in training, and 3× HWS
   oversampling triples the exposure of the more contaminated corpus. **No
   contributor-clean subset of val or test exists for this model.** These are
   benchmark numbers comparable with published FUTO figures — **not a
   generalization claim about an unseen user.**
3. **The dedup defect.** 588 val / 145 test rows sat in `train_t3` with a
   bit-identical input tensor and label, because the dedup keyed on the raw word
   and the label on the a–z-normalized one. **Key fixed** in
   `build_tiers.hash_row` / `prepare_data.trace_hash`; **tiers deliberately not
   rebuilt** (`AUDIT_PREDECODE.md` §E). Measured effect: leaked rows score 4.34 pt
   *below* comparable non-leaked ones, and removing all of them costs < 0.05 pt on
   val / 0.20 pt on test with all five bars still clearing on every seed.
4. **The counter-asymmetry, in FUTO's favour.** 5,273 of the 12,299 unique holdout
   traces (43 %) are bit-exactly in the HF *train* split FUTO trained on; 0 in HF
   dev/test. The app repo's description of the split as FUTO-held-out is incorrect.
5. **Lexicon.** Our runs and the val bar use the *same* 146,964-word STRIP trie, so
   `README.md`'s "our larger lexicon makes these conservative" does **not** apply to
   the val comparison. The test bar was published on the 131,544-word DROP trie and
   re-measured unchanged on the 146,964 one, so the overall test comparison is
   trie-neutral; its **strata were not republished**, so ≤3 and 4+ on test are
   compared across normalizers. **The app will not ship that trie** — it ships the
   bundled 98,140-entry `en_enhanced.json` (98,081 words after a–z stripping),
   whose byte frequencies are floored at 134–255 and whose `log_freq` spread is
   therefore 0.64 against the AOSP trie's 5.40, an 8× collapse of the scale the
   E1 `lambda = 1.1` was fitted on. That was validated end-to-end in `PHASE_F.md`
   §15: the app trie has **fewer** OOV targets (2.52 % of val vs 3.39 %), the bar
   was re-measured on it from FUTO's real weights so the comparison stays
   trie-matched, and **both ship candidates clear all five bars on every seed at
   the unchanged preset**. A λ-only re-sweep is worth +0.6 to +1.1 t1 at λ 2.0–2.5
   and is documented there, but is **not** taken: every number in this file and
   the golden fixture are quoted at λ = 1.1.
6. **Arm selection used full val.** The preset sweep (val `0:4959`) and checkpoint
   selection (5,000-row prefix) respected a holdout, but *which* arms were stacked
   was decided on full val-9918 tables.
7. **Seal hygiene.** One decode per checkpoint, verified bit-for-bit at the
   registered preset on 100/100 sampled rows, with 0/100 matching under any other
   preset. Prior contact: the disclosed pre-campaign `r2` decode and an undisclosed
   120-row smoke decode with a toy 898-word trie. **7 traces are bit-exactly shared
   between val-9918 and test-2400.** During the post-decode hygiene pass, 3 test
   rows were decoded to verify the new `--unseal-test` override branch; no number
   from that run appears anywhere. **Second unsealing, 2026-08-08:** six more
   decodes (`fast_resbn80` × 3 seeds × 2 lexicons) on the user's order, plus three
   re-scores of FUTO's own cached ceiling emissions on test-2400 to obtain a
   trie-matched bar (an external reference, no CleverKeys model involved). Both
   reads, and the prior contact above, are now logged in
   `test2400_seal.json["test-2400"]` under `unsealings` / `prior_contact`;
   `seal.py --emit` preserves that ledger across a fingerprint regeneration.

## Next — app-side (not this repo)

1. **G3 wiring.** Drop `ch128_s1234.onnx` into the `CtcEmissionModel` seam; the I/O
   contract is unchanged from `r2`, so no Kotlin signature moves.
2. **Update `CtcScoringParams`** to `gamma 1.05, lambda 1.1, beta 0.2, alpha 0.0,
   gammaPrune 0.3734, betaPrune 0.9882`. **Required** — the published preset costs
   ~2.3 pt and drops the model to 3 of 5 bars.
3. **Land the golden fixture.** Commit `artifacts/ctc_model_golden.json` as
   `src/test/resources/ctc/ctc_golden.json`. `CtcParityTest` currently fails its own
   file-existence assertion (audit finding #4), so featurizer parity is **untested
   today**; this is what makes it run.
4. **`NOTICE` attribution.** `futo-org/swipe.futo.org` corpus (**MIT**) and
   How-We-Swipe / OSF `sj67f` (**MIT**, © 2021 Leiva/Kim/Cui/Bi/Oulasvirta). No FUTO
   weights or model outputs were used anywhere in training (guide §0), so the FUTO
   Model Weights License is **not** implicated; the decode *algorithms* ported from
   the GPL-3.0 `swipe-library` are already committed on the app side.
5. **Re-measure latency on a phone little core.** 0.455 ms is a desktop x86 core; the
   trie beam over 147 k words, not the encoder, dominates the per-swipe budget.
   On device the beam runs over the **98 k** app trie, which is a third smaller.
6. **O3 is closed: ship `dictionaries/en_enhanced.json`** via
   `CtcLexiconTrie.loadStrippingNonAlphabet`, with the preset unchanged. Validated
   in `PHASE_F.md` §15 on val (both candidates, three seeds each) and in §16.5 on
   test (`fast_resbn80`, three seeds). No new dictionary asset is needed and no
   λ change is required.

---

# Campaign 1 (2026-08-07) — superseded

> ⚠ Kept for provenance. Its ship candidate (`r2`, ch 96) and its
> "keep the published scoring preset" recommendation are both **superseded** by
> Campaign 2 above; its absolute accuracy numbers are quoted at the mis-tuned
> published preset and are understated by 2–5 pt.


From-scratch, license-clean CTC swipe-emission encoder for CleverKeys' `swipe/ctc/`
Kotlin decode module. Recipe: CleverKeys `docs/guides/train-ctc-swipe-model.md`
(@ app-repo HEAD `79ddfb0f`), with the 18 audit fixes documented in `README.md`.
Trained ONLY on the MIT `futo-org/swipe.futo.org`-derived hwsfuto splits — no FUTO
weights or model outputs anywhere in the loop (guide §0). MIT corpus attribution
must be added to the app repo `NOTICE` when the model ships there.

## Ship candidate

**`artifacts/ctc_swipe_encoder.onnx`** — run r2, ch=96, 0.39 M params, fp32,
1,619,140 bytes, opset 17, static shapes `[1,2,64]/[1,64,2]/[1,64]` →
`[1,32,65]/[1,32,64]/[1,32,1]`, 0 Einsum. sha256
`fcf1633167b10f5c28e7c4dc16a9bba178bacc9e2b76efb06d792162dc99d0b7`.

Scoring params: **unchanged published preset** `CtcScoringParams.encoderOnly`
(gamma 0.4056, lambda 0.0176, beta 0.9866, alpha 0.0, gammaPrune 0.4234,
betaPrune 1.0382) — a two-pass val sweep + a full-val headroom grid bounded the
best reachable gain at +0.21 pt top-1 (at a top-3/5 cost); flat optimum, keep as-is.

**`artifacts/ctc_model_golden.json`** — 4 model-backed golden cases
(cat/the/hello/keyboard) in the `CtcParityTest` fixture schema, for the app-side
G3 `CtcEmissionModel` parity test. sha256 `a76ae8eb19195e3fdbd7229c014f2eeda9ccec15f045ecfaf699983712e02498`.

## Hardware / wall-clock

RTX 5080 Laptop 16 GB (WSL2), torch 2.8.0+cu128. ~4 s/epoch on the deduped
109,600-row train split — the guide's "an evening" estimate was ~100× high;
every run below cost minutes.

## Data

Canonical splits (`{train,val,test}_hwsfuto.jsonl` 110,876/9,918/2,400), featurized
with the exact `futo_decoder_eval.featurize` port. Train deduped: 298 cross-split
leaks into val/test + 977 exact self-duplicates removed → 109,600 rows (audit #3;
val/test untouched, so all numbers remain comparable to the committed baselines).

## Runs

| run | config | best val greedy | full-val beam t1/t3/t5 |
|---|---|---|---|
| r1 | ch 96, cosine horizon 300, early-stopped @93 | 58.24 % | (1000-row probe 83.5/91.2/93.1) |
| **r2** | ch 96, horizon 110 (fully annealed) | 58.57 % | **81.57 / 89.84 / 91.37** |
| r3 | ch 128, horizon 110 | 60.77 % | 81.27 / 89.73 / 91.41 |

r3's +2.2 greedy did not survive the trie beam (−0.30 t1); r2 wins the gate metric
at half the params.

## Gates

- **G2 (training feasibility): PASS.** Bar was top-1 within ~2 pt of the FUTO
  enc-only floor (77–79). Measured **81.57** on full val-9,918 — above the floor
  itself, with a *larger* lexicon (146,964-word trie vs the baselines' 131,544;
  conservative direction — see Caveats).
- **Export parity: PASS.** Sliced-[32,27] max |onnx−torch| 3.81e-05, argmax
  100/100; ONNX full-val eval reproduces the torch numbers to every printed digit.
- **G4 (phase-2 refinement head): MISS — phase 2 closed.** Frozen per-frame head
  (15.6 K params): +0.9 greedy, **+0.0 beam** (28 fixed / 28 broken per 2,000 rows,
  both scoring presets tried). End-to-end `--unfreeze-after` fine-tune: +0.25
  greedy, below threshold. Root cause: FUTO's +5.88 pt lever came off a 43.96 %
  greedy base; ours is at 58.6 %, so a per-frame head has nothing to fix. The one
  untried structural idea is temporal context in the head (FUTO's magic_macaw is
  a DFSMN). Consequence per the decision doc §4: ship enc-only behind the
  confidence-gated cascade/router.

## Report numbers — test-2400 (one-shot, ONNX, same split/harness as all committed baselines)

| Engine | t1 | t3 | t5 | ≤3-char t1 | 4+-char t1 |
|---|---|---|---|---|---|
| FUTO ceiling (enc+refine) | 84.83 | — | — | 89.57 | 82.40 |
| **ours enc-only (r2)** | **80.96** | **89.79** | **91.12** | **85.89** | **78.42** |
| FUTO floor (enc-only) | 79.25 | 87.71 | 89.58 | 82.45 | 77.60 |
| CleverKeys shipped neural | 74.62 | — | — | 89.45 | 67.00 |
| CleverKeys geometric | 67.50 | — | — | 69.33 | 66.56 |

Greedy 58.92 % (FUTO floor anchor: 43.96 %). Beats the FUTO enc-only floor on
every metric and both strata; +6.3 overall / +11.4 on 4+-char vs shipped neural;
loses ≤3-char to neural (85.89 vs 89.45) — the router hedge stands.

## Caveats

1. **Lexicon mismatch vs published baselines**: our fetched
   `en_wordlist.combined` (gitlab.futo.org master) normalizes to a 146,964-word
   trie; the committed baselines used a 131,544-word variant. Larger trie = more
   confusables, so our numbers are conservative. Match the exact baseline lexicon
   before quoting deltas to the second decimal.
2. The app repo's `CtcParityTest` fixture `src/test/resources/ctc/ctc_golden.json`
   was found missing from the tree (audit #4) — regenerate/commit it during G3.
3. Per-trace eval dumps are LOCAL-ONLY at `~/ctc-train/ckpt/r2/`
   (`val_full[_onnx].jsonl`, `test2400_onnx.jsonl`) per project convention.

## Next (app-side, G3/G5 — not this repo)

Copy `artifacts/ctc_swipe_encoder.onnx` → app `src/main/assets/models/`, implement
`CtcEmissionModel` over onnxruntime-android, wire `ctc` into `swipe_engine_mode`,
golden-parity-test against `artifacts/ctc_model_golden.json`, G3 latency gate,
add MIT corpus attribution to NOTICE.
