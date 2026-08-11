# Model comparison — speed and accuracy, every candidate, every footing

**Date:** 2026-08-09 · **Phase-J addendum:** 2026-08-11 · **Status:** standalone
reference; no new measurement was run for it. Every number below is quoted from
a committed document and is traceable to a named section (the `source` column or
the citation on each table). Where a value is arithmetic over quoted values (a
mean of three published per-seed numbers, a difference of two published
seed-means) it is marked **[derived]** and the inputs are named.

> **Addendum notice.** §§0–7 were written at the close of Phase G, when every
> candidate was a **test-validated** model. Phases H, I and J added models that
> are **val + alt-layout validated only** — the seal was not spent again — so
> they can appear on the val footings (§0.1, §2.4, §2.8) and nowhere on the
> test footings (§2.1–§2.3). The Phase-J finalist `sw2345` is the current best
> model **on val and on alt-layouts**; it is **not** test-validated and must
> never be quoted as such. `resbn80g` remains the best-evidenced small model and
> ch 192 / ch 128 / `resbn80g` remain the only test-validated configurations.

This document exists because the campaign record is spread across nine phase
documents with three different accuracy footings, two different latency
harnesses, and one model (`resbn80g`) that supersedes another (`fast_resbn80`)
without replacing it in the older tables. It answers one question: **for each
candidate, how fast is it, how accurate is it, on which footing, and how good is
the evidence.**

Authorities, in precedence order: `RESULTS.md` (the record of what is
test-validated), `PHASE_J.md` (the current val + alt-layout frontier and the
campaign bars), `PHASE_I.md` (the incumbent `resbn192i`), `PHASE_G.md` (the
test-validated model), `FAIR_REMATCH.md` (the
equal-footing bar), `PHASE_F.md` (latency protocol + frontier),
`THREEWAY_AUDIT.md` (the old shipped NN, and the latency non-comparability
statement), `AUDIT_FINAL.md` (statistics and disclosures), `ALT_LAYOUT_EVAL.md`
(cross-layout transfer). Where this document and one of those disagree, they
are right and this is stale.

---

## 0. Reading rules — the three footings, and why columns may never be mixed

Almost every wrong comparison in this project comes from putting two footings in
one column. There are three, and they are not interchangeable:

| footing | our preset | FUTO's preset | trie | what a win means |
|---|---|---|---|---|
| **A — published bar** | val-tuned (E1) | FUTO's published | AOSP STRIP 146,964 | our tuned engine beats their untuned one. The tuning lever alone is worth **+2.29 pt t1** to us (`AUDIT_FINAL.md` §6.1) and **+1.94 pt t1** to them (`FAIR_REMATCH.md` §2). |
| **B — shipping / trie-matched** | val-tuned (E1, or the app preset for `resbn80g`) | FUTO's published, **re-measured on the same app trie** | app `en_enhanced.json` STRIP 98,081 | our shipping configuration clears the bar *on the lexicon users actually run*. Still tuned-vs-published: **not** an equal-footing claim (`PHASE_G.md` §7.3). |
| **C — equal footing** | val-tuned (E1) | **val-tuned by the same wide grid on the same val rows** | AOSP STRIP 146,964 | a genuine engine-vs-engine comparison. This is the only footing on which a superiority claim is admissible, and only `ch 192` has earned a (qualified) one (`FAIR_REMATCH.md` §5). |

The bars themselves, so no table below has to restate them:

| bar | t1 | t3 | t5 | ≤3 | 4+ | source |
|---|---|---|---|---|---|---|
| **test-2400, published** (FUTO ceiling, published preset, DROP 131,544) ‡ | 84.83 | 91.04 | 92.08 | 89.57 | 82.40 | `RESULTS.md` §"Verified test-2400 results" |
| **test-2400, trie-matched** (FUTO ceiling, published preset, app `en_enhanced` 98,081) | 84.92 | 91.54 | 92.96 | 89.57 | 82.52 | `PHASE_F.md` §15.2 |
| **test-2400, equal-footing** (FUTO ceiling, **val-tuned**, STRIP 146,964) | 87.12 | 92.29 | 92.96 | 89.94 | 85.68 | `FAIR_REMATCH.md` §4/§5 |
| **val-9918, published** (FUTO ceiling, published preset, STRIP 146,964) | 85.52 | 91.54 | 92.80 | 89.29 | 83.57 | `PHASE_F.md` §0 |
| **val-9918, trie-matched** (app `en_enhanced` 98,081) | 85.59 | 91.82 | 93.20 | 89.05 | 83.80 | `PHASE_F.md` §15.2 |
| **val-9918, equal-footing** (FUTO ceiling, **val-tuned**, STRIP) | 87.48 | 92.31 | 93.03 | 89.76 | 86.29 | `FAIR_REMATCH.md` §2 |

‡ The published test bar was measured on the 131,544-word DROP trie. Re-measured
on the 146,964-word STRIP trie our models actually used, FUTO's ceiling at its
published preset gives **84.92 / 91.38 / 92.42 / 89.94 / 82.33** (`FAIR_REMATCH.md`
§4). `AUDIT_FINAL.md` §6.6 calls the *overall* comparison trie-neutral on that
basis (+0.09 t1); the strata were never republished post-fix, so ≤3 and 4+ on test
are compared across normalizers. Every Δ quoted against "the published bar" in
this document uses the 84.83 row, as the campaign did.

Metric definitions, constant everywhere: beam width 100, top-k 8, **OOV against
the engine's own lexicon counted as a miss**, strata ≤3-char / 4+-char split at
n = 815 / 1,585 on test-2400 and n = 3,389 / 6,529 on val-9918, seed-mean over
seeds 1234 / 4321 / 7777.

### 0.1 The second kind of bar — the campaign's own incumbents (Phases H–J)

The bars above are FUTO's engine. From Phase H onward the campaign also carries
**internal** bars: the previous best CleverKeys model's seed-mean on each axis,
which a challenger must **beat** (not tie) to be promoted. Phase J's set — the
**eleven en bars** (5 val metrics + 6 alt-layout corpora) that the "10 of 11"
tally counts, plus the Cyrillic axis and the size/latency gate, which are
separate conditions and are **not** in that tally. All are `resbn192i` Phase I-A
seed-means except the Cyrillic one (`PHASE_J.md` §0, `RESULTS.md` §Phase I-A):

| axis | bar | source |
|---|---|---|
| val-9918 t1 / t3 / t5 / ≤3 / 4+ (E1, AOSP) | 88.30 / 92.60 / 93.26 / 91.27 / 86.77 | `RESULTS.md` §Phase I-A |
| dvorak held out / dvorak app-98k | 89.13 / 88.20 | same |
| azerty / qwertz / german / spanish | 83.60 / 82.50 / 79.64 / 88.28 | same |
| Cyrillic in-dict t1 (app-ru 50 k, real Yandex val, **eval-only**) | 76.21 | `PHASE_J.md` §0, §6.5 |
| size / latency | ≤5 MB / <50 ms | `PHASE_J.md` §0 |

These are **not** FUTO bars and the two kinds may not be mixed in one column: an
internal bar says "better than our last model", a FUTO bar says "better than the
external reference". Phase J's terminal condition — and its pre-registered
test-2400 unsealing — required **every one** of them. Ten of the eleven en bars
fell and the Cyrillic bar did not, so the condition was not met and the seal was
not opened; see §2.8.

---

## 1. Model cards

| model | arch | params | bytes | laptop latency (enc-only) | evidence tier | preset it is quoted at |
|---|---|---|---|---|---|---|
| **`sw2345`** ← the Phase-J finalist | `resbn:192:1,2,4,8`, embed_hid 96, BN folded at export; `resbn192i` recipe + the `tier_sw234` (101,842 rows) and `tier_sw5q` (24,707 rows) pools; 1,285,381 train rows (`PHASE_J.md` §3.1, §6.6.1) | **1,512,802** | **3,052,318** (fp16w ship artifact; 6,068,519 fp32) | **0.842 ms** mean / 0.859 p90 fp16w; 0.816 fp32 (`PHASE_F.md` §0 protocol) | **val + alt-layout validated only, NOT test-validated**, 3 seeds (`RESULTS.md` §Phase J) | benchmark **E1**; no app-trie sweep run for it |
| `resbn192i` — the Phase-I incumbent / bar-holder | `resbn:192:1,2,4,8`, embed_hid 96, layout-alt p 0.65 | not published in `RESULTS.md` §Phase I-A | 3,052,318 (fp16w ship artifact) | not published | **val + alt-layout only, NOT test-validated**, 3 seeds | benchmark **E1**; app preset `0.975 / 3.0 / 0.35 / 0.25 / 0.9882` |
| **`resbn80g`** ← the Phase-G candidate | `resbn:80:1,2,4,8`, embed_hid 96, 4 dilated blocks, BN folded at export | **279,346** | **1,142,727** | **0.215 ms** class — *inherited, not re-measured* (identical graph and parameter count to `fast_resbn80`; `RESULTS.md` §Phase G, `PHASE_G.md` §4) | **test-validated on both footings**, 3 seeds (third unsealing, `PHASE_G.md` §7.5) | benchmark **E1** `1.05 / 1.1 / 0.2 / 0.3734 / 0.9882`; **app** `gamma 0.9, lambda 4.0, beta 0.25, alpha 0.0, gammaPrune 0.25, betaPrune 0.9882` (`PHASE_G.md` §6) |
| `fast_resbn80` ⚠ **superseded** | same graph, 94 k steps, KD from ch 192, legacy affine sampler | 279,346 | 1,142,727 | **0.215 ms** (mean, `PHASE_F.md` §0 protocol / §6 frontier) | test-validated, 3 seeds (second unsealing, `PHASE_F.md` §16.5) — **superseded as the speed-class ship candidate** (`RESULTS.md` §Phase G) | E1 on both footings |
| `ch 128` | `res:128`, GroupNorm, 4 blocks (`phaseE-E3b-hws3x`) | 689,282 | 2,799,865 | **0.455 ms** (audit protocol) / 0.472–0.475 ms (Phase-F harness) | test-validated, 3 seeds (first unsealing) | E1 |
| `ch 192` | `res:192` (`phaseE-FINAL`) | 1,525,378 | 6,144,249 | **0.877 ms** (audit protocol) / 0.911–0.934 ms (Phase-F harness) | test-validated, 3 seeds (first unsealing) | E1 |
| old shipped NN (transformer) | `swipe_encoder_android.onnx` + `swipe_decoder_android.onnx`, d_model 256, 6 enc + 4 dec layers | decoder **4.2 M int8 + 146 K fp**; encoder param count not published — compare by bytes | 5,317,537 + 4,975,510 = **10,293,047** | **not encoder-only comparable** — its committed figures are full-pipeline: ~178 ms/trace on-device, ~55 ms/trace on the audit laptop | test-measured by the app repo's production-equivalent harness (`docs/eval/2026-07-24-test2400-head2head.md`) | its own shipped production beam-6 config |
| FUTO ceiling (`honorable_sturgeon` + `magic_macaw`) | encoder + DFSMN refinement, `.pte` XNNPACK | 635 K + 304 K = **939 K** | 2,649,856 + 1,247,468 = **3,897,324** | **no committed latency figure** (its evals ran uninstrumented in proot) | external bar, two versions: **published preset** (as published) and **val-tuned** (measured here from FUTO's real hash-verified weights, `FUTO_WEIGHTS_VERIFICATION.md` / `FAIR_REMATCH.md`) | published preset; val-tuned = `gamma 1.15, lambda 1.3, beta 0.2, gammaPrune 0.3734, betaPrune 0.7` (`FAIR_REMATCH.md` §2) |
| *(context)* `fast_resbn72` | `resbn:72:1,2,4,8` @188 k | 229,642 | 944,487 | 0.186 ms | **val-only**, never decoded on test | E1 |
| *(context)* geometric SHARK2 | pure-JVM, no NN | — | — | — | test-measured (head2head doc) | its own shipped tuning |

Sizes for `ch*`/`resbn*` are from `RESULTS.md` §Artifacts and `PHASE_F.md` §9;
old-NN and FUTO sizes from `THREEWAY_AUDIT.md` §3 (verified by that audit).

### 1.1 Phase-G artifacts and the fixture

| file | arm | sha256 | source |
|---|---|---|---|
| `artifacts/resbn80g_s1234.onnx` ← **ship** | `phaseG-C80-188k-nokd` | `330cadfbaa7334eaeaeab93762084181b70710fe9d59cbd69600a6de468fe1a0` | `RESULTS.md` §Phase G / `PHASE_G.md` §7.3 |
| `artifacts/resbn80g_s4321.onnx` | `phaseG-C80-188k-nokd-s4321` | `c9379c60a23bec4ca300512d2930b7a724aad91b761597972446a6577f5d5bab` | same |
| `artifacts/resbn80g_s7777.onnx` | `phaseG-C80-188k-nokd-s7777` | `3e303d46abaff4bfe31779de35fb9fc81e63f1ae8fd5ab554a9db205f167191a` | same |
| `artifacts/ctc_model_golden.json` | golden parity fixture, from `resbn80g_s1234` **at the app preset** (139,728 bytes) | `ce3b5456ad13543ac09ac8c2610374bd8847b15f740f9004a98efea59d74f134` | `RESULTS.md` §Phase G |

All three ONNX files: opset 17, fp32, static shapes
`[1,2,64]/[1,64,2]/[1,64]` → `[1,32,65]/[1,32,64]/[1,32,1]`, in-graph
`log_softmax`, blank at column 64, **zero normalization nodes** (BatchNorm folded
into the preceding convolution at export), sliced-view parity 100/100 argmax
against torch. The I/O contract is unchanged from `r2` and from every Phase-E/F
artifact — **no Kotlin signature moves** (`PHASE_F.md` §9).

---

## 2. Accuracy

### 2.1 Footing A — test-2400, AOSP STRIP 146,964, our E1 vs FUTO's published preset

Seed-means over 1234/4321/7777. The bar is 84.83 / 91.04 / 92.08 / 89.57 / 82.40.

| model | t1 | t3 | t5 | ≤3 (n=815) | 4+ (n=1,585) | greedy t1 | all five, every seed? | source |
|---|---|---|---|---|---|---|---|---|
| ch 192 | **88.36** | **92.65** | **93.50** | **91.37** | **86.81** | 74.56 | yes | `RESULTS.md` §"Verified test-2400 results" |
| ch 128 | 87.92 | 92.33 | 93.00 | 91.08 | 86.29 | 70.47 | yes | same |
| **`resbn80g`** | **87.68** | **92.18** | **92.82** | **90.80** | **86.08** | 69.54 **[derived]** | **yes** | `PHASE_G.md` §7.5 config A (greedy = mean of the published 70.96/69.71/67.96) |
| `fast_resbn80` ⚠ superseded | 87.29 | 91.89 | 92.82 | 91.17 | 85.30 | — | yes | `RESULTS.md` §"The second unsealing" config A |
| FUTO ceiling (**published preset** — the bar) | 84.83 | 91.04 | 92.08 | 89.57 | 82.40 | 69.12 | — | `THREEWAY_AUDIT.md` §1 |
| FUTO floor (enc-only, textbook trie beam, published preset) | 79.25 | 87.71 | 89.58 | 82.45 | 77.60 | 43.96 | — | same |
| old shipped NN (its own 98,140-word dict, prod beam-6) | 74.62 | 84.33 | 87.42 | 89.45 | 67.00 | n/a | — | same (≤3 t3/t5 95.46/96.32; 4+ t3/t5 78.61/82.84) |
| *(context)* geometric SHARK2 | 67.50 | 78.88 | 81.79 | 69.33 | 66.56 | — | — | same |

Worst-seed values for `resbn80g` on this footing: 87.12 / 91.88 / 92.50 / 90.18 /
85.55 — every one above the bar (`PHASE_G.md` §7.5).

> The old-NN row sits in this table only because the head2head registered
> "own lexicon, OOV = miss" as the cross-engine rule. It is a **different
> lexicon** (98,140 `en_enhanced` vs our 146,964 AOSP) and the difference runs
> *against* us: 86 forced OOV misses on our side vs 64 on the old NN's, roughly
> 0.9 pt (`THREEWAY_AUDIT.md` §4.3). See §2.2 for the lexicon-matched version.

### 2.2 Footing B — test-2400, app `en_enhanced.json` STRIP 98,081, trie-matched bar

**The two model rows here are at different presets and are not comparable to each
other.** `resbn80g` is at the app preset (λ 4.0), `fast_resbn80` at E1 (λ 1.1).
Both are compared against the same trie-matched, published-preset bar
84.92 / 91.54 / 92.96 / 89.57 / 82.52.

| model | preset | t1 | t3 | t5 | ≤3 | 4+ | worst-seed t5 margin | source |
|---|---|---|---|---|---|---|---|---|
| **`resbn80g`** | **app** `0.9/4.0/0.25/0.25/0.9882` | **88.14** | **93.22** | **93.90** | **91.86** | **86.23** | **+0.75** | `PHASE_G.md` §7.5 config B |
| Δ vs bar | | **+3.22** | **+1.68** | **+0.94** | **+2.29** | **+3.71** | | |
| `fast_resbn80` ⚠ superseded | E1 | 86.51 | 92.28 | 93.25 | 90.76 | 84.33 | **+0.08** | `RESULTS.md` §"The second unsealing" config B |
| Δ vs bar | | +1.59 | +0.74 | +0.29 | +1.19 | +1.81 | | |
| old shipped NN (same dictionary asset) | its own | 74.62 | 84.33 | 87.42 | 89.45 | 67.00 | — | `THREEWAY_AUDIT.md` §1 |

This footing is the **lexicon-matched** comparison against the old shipped NN:
both run the bundled `en_enhanced` dictionary (98,140 entries; 98,081 words after
a–z stripping) and both take **exactly 64 forced OOV misses** on test-2400
(`PHASE_F.md` §15.1, `THREEWAY_AUDIT.md` §4.3). The OOV handicap of §2.1 is gone
here.

`resbn80g`'s per-seed config-B values: t1 88.42 / 88.29 / 87.71, t3
93.17 / 93.38 / 93.12, t5 94.08 / 93.92 / 93.71, ≤3 92.15 / 92.27 / 91.17, 4+
86.50 / 86.25 / 85.93 (`PHASE_G.md` §7.5).

### 2.3 Footing C — test-2400, equal footing (both engines val-tuned, STRIP 146,964)

Bar: **87.12 / 92.29 / 92.96 / 89.94 / 85.68** (`FAIR_REMATCH.md` §4, real per-row
decode of FUTO's weights, not the analytic path).

| model | Δ t1 | Δ t3 | Δ t5 | Δ ≤3 | Δ 4+ | bars cleared | paired McNemar on t1, per seed | source |
|---|---|---|---|---|---|---|---|---|
| ch 192 | **+1.24** | **+0.36** | **+0.54** | **+1.43** | **+1.14** | 5 of 5 | +40 p 0.0004 · +18 p 0.16 · +31 p 0.0101 → **resolved on 2 of 3** | `FAIR_REMATCH.md` §5 |
| ch 128 | +0.79 | +0.04 | +0.04 | +1.15 | +0.61 | 5 of 5 point estimates, but t3/t5 are **+0.04 = one trace in 2,400 — ties** | +22 p 0.059 · +17 p 0.162 · +18 p 0.133 → **resolved on none** | same |
| **`resbn80g`** | **+0.56** | **−0.11** | **−0.14** | **+0.86** | **+0.40** | **3 of 5** | **+17 p 0.17 · +23 p 0.052 · +0 p 1.00 → resolved on none** | `PHASE_G.md` §7.5 §"Equal footing" |
| `fast_resbn80` ⚠ superseded | +0.17 | **−0.40** | **−0.14** | +1.23 | **−0.38** | 2 of 5 (fails three) | unresolved on every seed, **one seed net negative** (−9) | `FAIR_REMATCH.md` §5 |

**What may and may not be written.** `FAIR_REMATCH.md` §6 lifts the
equal-footing prohibition **for ch 192, qualified**; it **stands for ch 128**
(the lead does not resolve); and `fast_resbn80` "must not be described as beating
FUTO at all". For **`resbn80g`, `PHASE_G.md` §7.2/§7.5 registers in advance and
repeats afterwards that no equal-footing superiority claim is made or
permitted.** The admissible sentence is: *`resbn80g` is level with FUTO's
val-tuned engine where `fast_resbn80` was behind it — three losses become two
−0.1 ties and a +0.40 win — and the paired test resolves on no seed.*

### 2.4 val-9918, AOSP STRIP 146,964 — the footing every model has

Seed-means at E1 (ours) / published preset (FUTO). Bar: 85.52 / 91.54 / 92.80 /
89.29 / 83.57.

| model | t1 | t3 | t5 | ≤3 (n=3,389) | 4+ (n=6,529) | source |
|---|---|---|---|---|---|---|
| **`sw2345`** ⚠ val + alt-layout only | **88.51** | **92.67** | **93.37** | 91.20 | **87.11** | `RESULTS.md` §Phase J; `PHASE_J.md` §6.6.1 |
| `resbn192i` ⚠ val + alt-layout only | 88.30 | 92.60 | 93.26 | **91.27** | 86.77 | `RESULTS.md` §Phase I-A |
| ch 192 | **88.06** | **92.32** | **93.08** | 90.86 | **86.62** | `PHASE_E.md` §5 via `THREEWAY_AUDIT.md` §2 |
| ch 128 | 87.88 | 92.23 | 92.96 | **90.98** | 86.26 | same |
| **`resbn80g`** | **87.72** | **92.25** | **92.97** | **90.78** | **86.14** | `PHASE_G.md` §4 |
| `fast_resbn80` ⚠ superseded | 87.47 | 92.13 | 92.89 | 90.35 | 85.98 | `PHASE_F.md` §8 |
| `fast_resbn72` ⚠ val-only | 87.27 | 92.09 | 92.87 | 90.49 | 85.60 | `PHASE_F.md` §14.1 |
| FUTO ceiling, published preset (**the bar**) | 85.52 | 91.54 | 92.80 | 89.29 | 83.57 | `PHASE_F.md` §0 |
| FUTO ceiling, **val-tuned** (footing C bar) | 87.48 | 92.31 | 93.03 | 89.76 | 86.29 | `FAIR_REMATCH.md` §2 |
| FUTO floor, published preset | 78.84 | 88.01 | 90.11 | 81.17 | 77.62 | `THREEWAY_AUDIT.md` §2 |
| FUTO floor, **val-tuned** | 85.97 | 91.18 | 92.12 | 89.11 | 84.35 | `FAIR_REMATCH.md` §3 |
| old shipped NN | 76.01 | 85.53 | 87.82 | 89.23 | 69.15 | `THREEWAY_AUDIT.md` §2 |
| *(context)* geometric | 67.69 | 78.36 | 81.49 | 70.23 | 66.37 | same |

The top two rows are on this footing but **not on this evidence tier**: neither
has been decoded on test-2400 and neither may be quoted against a test bar. They
are the only two models in this table that beat the published val bar by more
than 2.5 pt t1, and against each other the comparison that matters is the
campaign's internal one (§0.1, §2.8): `sw2345` beats `resbn192i` on t1, t3, t5
and 4+ and **loses `≤3` by 0.07**.

`resbn80g` per-seed val (config A footing): t1 88.04 / 87.82 / 87.31, worst seed
87.31, all five clear on every seed (`PHASE_G.md` §4).

Against the **equal-footing val bar**, `resbn80g`'s seed-mean clears t1 (+0.24)
and ≤3 (+1.02) but misses t3 (−0.06), t5 (−0.06) and 4+ (−0.15) — *not met*
(`PHASE_G.md` §5). Seed 1234 alone clears all five; a single seed is not the gate
and is not claimed.

### 2.5 val-9918, app `en_enhanced` 98,081 — the shipping footing on val

Bar (trie-matched, published preset): 85.59 / 91.82 / 93.20 / 89.05 / 83.80.

| model / preset | t1 | t3 | t5 | ≤3 | 4+ | source |
|---|---|---|---|---|---|---|
| `resbn80g` @ **app preset**, seed-mean | **88.54** | **93.17** | **94.06** | **91.92** | **86.79** | `PHASE_G.md` §7.4 **[derived** from the per-seed values quoted there: 88.76/88.69/88.18 etc.**]** |
| `resbn80g` (arm C, s1234) @ E1 | 87.37 | 92.90 | 93.86 | 90.50 | 85.74 | `PHASE_G.md` §6 |
| `resbn80g` (arm C, s1234) @ **app preset** | 88.76 | 93.22 | 94.18 | 92.42 | 86.86 | same — **the preset is worth +1.39 t1 / +0.32 t5 / +1.92 ≤3 / +1.12 4+** |
| ch 128 @ E1, seed-mean | 87.96 | 92.77 | 93.67 | 91.49 | 86.12 | `PHASE_F.md` §15.3 |
| `fast_resbn80` @ E1, seed-mean | 86.93 | 92.39 | 93.51 | 90.51 | 85.07 | same |

The app preset's gain is confirmed on the untouched holdout half (+4.15 t1 over
the published-preset baseline there vs +4.66 on the tuning half — no
sweep-overfit signature) and is positive on all five metrics (`PHASE_G.md` §6).

### 2.6 Per-source — the aggregate hides a ~14-point internal spread

Seed-mean top-1 by corpus half. test-2400 is 1,217 FUTO rows / 1,183 How-We-Swipe
rows.

| model / footing | FUTO half | HWS half | spread | source |
|---|---|---|---|---|
| ch 192 (test, A) | 95.32 | 81.21 | 14.11 | `RESULTS.md` §"Per-source" |
| ch 128 (test, A) | 95.07 | 80.56 | 14.51 | same |
| **`resbn80g` (test, A)** | **94.80** | **80.36** | **14.44** | `PHASE_G.md` §7.5 §"Per-source" |
| **`resbn80g` (test, B — app trie)** | **94.55** | **81.54** | **13.01** | same |
| `fast_resbn80` (test, A) | 94.63 | 79.74 | 14.89 | `RESULTS.md` §"The second unsealing" |
| `fast_resbn80` (test, B) | 93.37 | 79.46 | 13.91 | same |
| `resbn80g` (val) | 94.56 | 80.86 | 13.70 **[derived]** | `PHASE_G.md` §4 |
| `fast_resbn80` (val) | 94.79 | 80.29 | 14.50 **[derived]** | same |

The 87.68 headline of §2.1 is the average of a 94.8 and an 80.4. On the
How-We-Swipe half alone every model in this family sits **below** the aggregate
bar. Note also where the Phase-G upgrade came from: on val it is **+0.57 on the
harder HWS half** with the FUTO half giving back 0.23 (`PHASE_G.md` §4) — the
gain is on the hard corpus, which is the direction one wants.

### 2.7 Statistical resolution

Two different tests, two different bars — do not merge them.

**Unpaired binomial z against the *published* bar** (footing A; FUTO's per-row
output was unavailable when this was computed, so the bar is treated as a fixed
estimate on the same rows) — `AUDIT_FINAL.md` §5:

| metric | n | SE | ch 192 z | ch 128 z |
|---|---|---|---|---|
| t1 | 2,400 | 0.98 | **3.6 resolved** | **3.1 resolved** |
| 4+ | 1,585 | 1.28 | **3.4 resolved** | **3.0 resolved** |
| t3 | 2,400 | 0.79 | 2.0 | 1.6 |
| t5 | 2,400 | 0.75 | 1.9 | 1.2 |
| ≤3 | 815 | 1.45 | 1.2 not resolved | 1.0 not resolved |

For `fast_resbn80` the same test gives t1 z 3.4 and 4+ z 3.0 under config A, and
**nothing resolves under config B** (`RESULTS.md` §"The second unsealing").
`PHASE_G.md` does not publish z-scores for `resbn80g`; its margins on footing A
(+2.85 t1, +3.68 4+) are of the same order as `fast_resbn80`'s and larger, so the
same two-of-five picture is the conservative read — but that is an inference, not
a published number.

**Exact paired McNemar against the *val-tuned* bar** (footing C; possible only
because FUTO's per-row output now exists) — `FAIR_REMATCH.md` §5 and
`PHASE_G.md` §7.5: see the last column of §2.3 above. It is far more sensitive
than the unpaired test (~120–150 discordant pairs; resolvable difference ≈ 0.9–1.0
pt on t1), and **the only model it resolves is ch 192, on two of three seeds.**

Seed variance is not the limiting factor anywhere (sd 0.04–0.73). Row sampling on
a 2,400-row split is.

### 2.8 The Phase-J finalist `sw2345` — 10 of 11 internal bars, and the two that stand

**Evidence tier: val + alt-layout validated only. NOT test-validated.**
test-2400 was **not** unsealed in Phase J: the pre-registered rule was that the
seal opens if and only if **all** the §0.1 bars fall, and two did not. Nothing
below may be compared against any test bar, and `resbn80g` retains the
test-validated tier (`RESULTS.md` §Phase J, `PHASE_J.md` header + §0).

**val-9918, E1 / AOSP, exported ONNX, seeds 1234 / 4321 / 7777**
(`PHASE_J.md` §6.6.1):

| metric | s1234 | s4321 | s7777 | **seed-mean** | bar | Δ |
|---|---|---|---|---|---|---|
| t1 | 88.51 | 88.57 | 88.46 | **88.51** | 88.30 | **+0.21** |
| t3 | 92.59 | 92.72 | 92.70 | **92.67** | 92.60 | **+0.07** |
| t5 | 93.35 | 93.48 | 93.28 | **93.37** | 93.26 | **+0.11** |
| ≤3 | 90.91 | 91.24 | 91.44 | **91.20** | 91.27 | **−0.07 — MISS** |
| 4+ | 87.26 | 87.18 | 86.90 | **87.11** | 86.77 | **+0.34** |

**Alt-layout, az26 in-dict, E1, same three seeds** — all six clear:

| corpus | bar | `sw2345` 3-seed | Δ |
|---|---|---|---|
| dvorak (held out of training) | 89.13 | **89.87** | **+0.74** |
| dvorak, app-98k trie | 88.20 | **88.98** | **+0.78** |
| azerty | 83.60 | **83.81** | **+0.21** |
| qwertz | 82.50 | **83.01** | **+0.51** |
| german | 79.64 | **80.64** | **+1.00** |
| spanish | 88.28 | **88.45** | **+0.17** |

Two new real-layout corpora carry **no incumbent** — their zero-shot floors were
established this phase (91.08 / 90.19, `PHASE_J.md` §3.3) — so they are
informational and outside the tally: `sw2345` scores **clearflow 91.06** and
**kasroz 92.07**. Both are small, single-cohort corpora (±0.7–1.1 pt binomial
SE).

**The two stones that stand, stated plainly:**

1. **`≤3` misses by 0.07 pt.** On a 3,389-row stratum that is roughly two rows,
   and it is a *tie* in every practical sense — but the campaign's condition
   says *beat*, so it is a **miss** and is not rounded away. Every lever tried
   against it failed: layout-alt dose, CR-CTC, FUTO-parity augmentations, the
   checkpoint soup (sign-inconsistent, mean −0.10) and a stratum-aware
   `minmargin` decode sweep over the E1 region, which bought **+0.03** where
   ~+0.33 was needed (`PHASE_J.md` §6.7, §6.4.1, §6.6.2, §6.8b). The decode
   sweep is the diagnostic one: gamma and beta re-rank candidates by length and
   **cannot conjure a short candidate the beam never generated**, so
   `PHASE_J.md` §9 reads the residue as a candidate-generation problem (§7).
2. **The Cyrillic bar (76.21 in-dict t1) is NOT beaten.** Capacity on synthetic
   ru made it worse (ch 192 / 188 k → 73.53), and a joint en+ru single model
   ties at best on ru (76.56, +0.35, inside one binomial SE at n = 8,471, and
   *behind* the bar on t3/t5) while costing **−0.42 en val t1** against a 0.3
   tolerance (`PHASE_J.md` §6.5, §6.8). No Yandex rows are used in training
   anywhere; Yandex val is eval-only.

**The ru λ correction — a real, model-independent gain that changes no verdict.**
Every ru number in this campaign, the 76.21 bar included, was decoded at **E1's
λ = 1.1**, while the app ru lexicon stores `freq = 255 − rank`. A symmetric
sweep over both ru models (tuned on val rows 0:4708, confirmed on the untouched
4708:9416) puts the optimum at **λ = 2.0**, worth about **+1.2 to the synth-only
bar-holder on both halves** (`PHASE_J.md` §6.9):

| λ | `phaseIB-ru-synth` tune / confirm | joint en+ru tune / confirm |
|---|---|---|
| 1.1 (as published) | 75.73 / 76.70 | 76.77 / 76.34 |
| **2.0** | **76.91 / 77.92** | **77.83 / 78.23** |

**The correct shippable Cyrillic figure is therefore ≈ 77.4, not 76.21** — and
because the lever is model-independent it lifts the challenger equally, so the
bar rises with it and the Cyrillic axis remains **not beaten**. Any app-side
Cyrillic decode should use λ ≈ 2.0 on the ru lexicon; this is a *per-language*
preset finding and does **not** touch E1 on the en footings.

**Bar tally: 10 of the 11 en bars** — 4 of 5 val, 6 of 6 alt-layout — **and the
Cyrillic bar, which is a separate axis, also stands**. Phase J did **not** meet
its terminal condition, and no part of it is test-validated.

**Size and latency:**

| model | params | bytes | Phase-F-class ms | ≤5 MB gate |
|---|---|---|---|---|
| `sw2345` fp32 | 1,512,802 | 6,068,519 | 0.816 mean / 0.830 p90 | no |
| **`sw2345` fp16w** ← ship | 1,512,802 | **3,052,318** | 0.842 mean / 0.859 p90 | **yes (2.91 MiB)** |

fp16w is accuracy-free on this model by its own measurement (val-9918
88.51/92.58/93.35/90.91/87.26 vs fp32 88.51/92.59/93.35/90.91/87.26) but **3 %
slower**, unlike the "identical" reported for `resbn192i` in `PHASE_I.md` §8.

---

## 3. Speed

### 3.1 The laptop encoder-only ladder

Protocol (`PHASE_F.md` §0, itself the `AUDIT_PREDECODE.md` §7 protocol): ONNX
Runtime `CPUExecutionProvider`, `intra_op = inter_op = 1`, batch 1, fixed shapes,
50 warmup calls then 3 rounds × 300 timed calls, mean and p90 of the best round,
machine idle. **The harness floor is 0.007 ms** — a no-op ONNX graph carrying the
exact production I/O signature — so every figure below is graph work, not
instrument overhead.

The Phase-F harness reads **~3 % high** against the `AUDIT_PREDECODE.md` §7
numbers (ch 128: 0.472–0.475 vs 0.455; ch 192: 0.911–0.934 vs 0.877). All Phase-F
figures are internally consistent because they come from one harness on one
machine; **use the Phase-F column for ratios.**

| model | Phase-F mean ms | p90 | audit-protocol mean ms | params | bytes | bars on val (footing A) | source |
|---|---|---|---|---|---|---|---|
| *(no-op graph — harness floor)* | 0.007 | 0.007 | — | 0 | — | — | `PHASE_F.md` §0 |
| `resbn:48:1,2,4,8` | 0.122 | 0.130 | — | 111,250 | 472,645 | 3/5 | `PHASE_F.md` §6 |
| `resbn:56:1,2,4,8` @280 k — best ≤0.15 ms | 0.142 | 0.149 | — | 145,594 | 609,445 | 4/5 (t5 −0.13) | same |
| `resbn:64:1,2,4,8` @188 k (3 seeds) | 0.162 | 0.172 | — | 185,058 | 766,727 | 4/5 (t5 −0.13, no seed clears) | `PHASE_F.md` §6/§8 |
| **`fast_resbn72`** — fastest 5/5 | **0.186** | 0.195 | — | 229,642 | 944,487 | 5/5 (val only) | `PHASE_F.md` §14.1 |
| **`fast_resbn80` / `resbn80g`** | **0.215** | 0.224 | — | 279,346 | 1,142,727 | 5/5 | `PHASE_F.md` §6/§8; `resbn80g` inherits the class (`PHASE_G.md` §4) |
| ch 128 fp32 | 0.475 | 0.490 | **0.455** | 689,282 | 2,799,865 | 5/5 | `PHASE_F.md` §6 |
| ch 192 fp32 | 0.920 | 0.937 | **0.877** | 1,525,378 | 6,144,249 | 5/5 | same |
| ⚠ pre-campaign `r2` ch 96 | 0.306 | 0.318 | — | 394,114 | 1,619,140 | 3/5 | same |
| **`sw2345`** (Phase-J finalist) | 0.816 | 0.830 | — | 1,512,802 | 6,068,519 fp32 / 3,052,318 fp16w | 4/5 vs the **campaign** bars (§2.8) — not the FUTO bar | `RESULTS.md` §Phase J |

Ratios that follow, on the Phase-F column: `resbn80g` is **2.21× faster than
ch 128** and **4.28× faster than ch 192**, at 41 % and 18 % of their bytes.
`fast_resbn72` is 2.55× faster than ch 128 at 34 % of its bytes.

Two structural facts about this ladder (`PHASE_F.md` §1, §7): two-thirds of
ch 128's time is eight dense 5-tap trunk convolutions, and **~0.10 ms of any
budget is spent before the first trunk block** on work the I/O contract fixes
(key-embed MLP, masking, heads, `LogSoftmax`). That floor is why ≤0.15 ms with
all five bars clearing was not reached: t5 crosses the bar at 210–230 k
parameters, measured *with* the KD handicap that Phase G later showed was
costing ~0.5 t1 (`PHASE_G.md` §3.1), so the boundary may sit lower without it.

### 3.2 Per-stage, and the end-to-end paths that were measured

Only three whole-pipeline measurements exist, on three different runtimes. None
is a phone.

| runtime | engine | featurize | NN | beam | total | source |
|---|---|---|---|---|---|---|
| **headless Chrome, WASM single-thread** (web demo, mean of 10 decodes of `keyboard`) | CTC ch 128 | 0.01 ms | 1.52 ms | 1.60 ms | **3.13 ms** | app repo `web_demo/README.md` §Latency |
| same | CTC `fast_resbn80` | 0.01 ms | 0.76 ms | 1.55 ms | **2.32 ms** | same |
| same | old transformer | 0.01 ms | 356.10 ms † | — | **356.11 ms** | same († its beam is interleaved with per-step decoder sessions, so there is no separable beam term) |
| **desktop JVM, ORT Java API, `intra_op=inter_op=1`, WSL x86_64** (temporary probe, deleted after use) | ch 128 / ch 192 / `fast_resbn80` / `fast_resbn72` encoder-only | — | 0.554 / 1.001 / **0.257** / 0.211 ms | 7.3 ms mean (p90 12.6) at beam 100 over the 98 k app trie | **9.6 ms** NN + beam | app repo `memory/todo.md` §"CTC on-device latency measurement (G3)" |
| **Python, laptop x86** (campaign decode harness) | ch 128 encoder + Python beam-100 over the 147 k trie | — | — | — | ~11–13 ms/trace (78–94 traces/s, single process) | `THREEWAY_AUDIT.md` §3 |
| **Python, laptop x86** | old transformer, full production decode | — | — | — | ~55 ms/trace (1,097 traces/min, 4 threads) | same |
| **on-device Termux ARM64** | old transformer, full production decode | — | — | — | **~178 ms/trace** (337.8 traces/min, 4 threads, paired benchmark) | `THREEWAY_AUDIT.md` §3, quoting the app repo's `2026-08-06-offline-decoder-speedup.md` |

The desktop JVM probe reads **14–22 % high** against the Phase-F laptop table
(0.554 vs 0.455, 0.257 vs 0.215) with the **ratios preserved** — which is the
useful part: the Kotlin/ORT-Java path reproduces the ranking, and the trie build
costs 90 ms once per process. The web-demo numbers use `fast_resbn80`, not
`resbn80g`; the two are the same graph, so the encoder column should carry over,
but that has not been measured.

**The beam, not the encoder, is the budget.** In both end-to-end measurements the
trie beam is 1.5 ms (WASM) to 7.3 ms (JVM) against an encoder of 0.8–1.5 ms and
0.26–0.55 ms respectively. A 0.24 ms encoder saving (ch 128 → `resbn80g` on the
Phase-F scale) is 2–3 % of the end-to-end path on the JVM measurement. Size, not
latency, is the stronger argument for the small model.

### 3.3 On-device: nothing is measured, and that is the honest state

* **No committed on-device latency exists for any CTC model.** The instrumented
  benchmark (`src/androidTest/kotlin/tribixbite/cleverkeys/swipe/ctc/CtcOnnxLatencyBenchmarkTest.kt`,
  four models × two session configs, plus the full
  featurize → NN → slice → beam@100 path over the real bundled 98 k trie) is
  written, compiled, and packaged; **the emulator.wtf run is blocked** on
  `EW_API_TOKEN` / `ew-cli` not being present on the WSL box (app repo
  `memory/todo.md`). Until it runs, every on-device statement is extrapolation.
* **No committed on-device latency exists for FUTO's engine** either — its evals
  ran uninstrumented in a proot container (`THREEWAY_AUDIT.md` §3).
* The old NN's ~178 ms/trace is the only real phone number in this document, and
  it is a **full pipeline** figure (encoder + autoregressive beam-6 decoder +
  rerank), not an encoder-only one.

### 3.4 The non-comparability caveat, stated once and applying to all of §3

`THREEWAY_AUDIT.md` §3, verbatim in substance: **these columns are not
cross-comparable and no single-number speedup claim is legitimate.** Our figures
are encoder-only forward pass, single-thread batch-1, laptop x86 through the
Python ORT binding; the old NN's committed figures are the full decode pipeline on
a phone at 4 threads; FUTO has no figure at all. What *is* legitimately
comparable: (a) **file size** — `resbn80g` is 11.1 % of the old NN's 10.3 MB and
29 % of FUTO's 3.9 MB two-model stack; (b) **same-machine full-pipeline
throughput**, where the CTC path ran ~4–5× faster than the old NN's production
decode on the same laptop — across different code paths, so indicative only;
(c) the web-demo column, where all three engines ran in the same browser on the
same trace (~110–150× for CTC, but on synthetic constant-speed trajectories that
are out of distribution for the transformer — the web-demo README says explicitly
not to read its accuracy table as model quality, and the same caution applies
loosely to its latency ratio).

---

## 4. The deltas that matter

### 4.1 `resbn80g` vs `fast_resbn80` — what the Phase-G recipe bought

Same graph, same 279,346 params, same 0.215 ms class. The recipe changed in
three ways (`PHASE_G.md` §2/§3): 94 k → **188 k steps**, the legacy affine
sampler → the **coupled** one, and **distillation removed**.

The 2×2 factorial at ch 80 / 188 k, single seed, full val, E1 (`PHASE_G.md` §3.2):

| | KD on | KD off | Δ (KD off − on) |
|---|---|---|---|
| legacy sampler | A 87.46 | E 87.94 | **+0.48** |
| coupled sampler | B 87.52 | **C 88.04** | **+0.52** |
| Δ (coupled − legacy) | +0.06 | +0.10 | |

* **KD removal is the dominant lever**, +0.48/+0.52 t1, positive on all five
  metrics in the no-KD column. This resolves Phase F's largest stated evidence
  hole (`PHASE_F.md` §11.3, "the distillation contribution is unmeasured") in the
  direction nobody assumed: the ch 192 teacher was *capping* the ch 80 student.
* **The ensemble teacher is worse still**: averaging the three ch 192 seeds'
  probabilities costs −0.45 t1 / −0.18 t3 / −0.75 4+ against the single teacher,
  at ~2× the per-step GPU cost (`PHASE_G.md` §3).
* **The affine fix is a small consistent positive**, +0.06/+0.10 t1, and without
  KD positive on all five (C − E: +0.10/+0.06/+0.20/+0.18/+0.06). It is a
  *distribution repair*, not a range extension: the legacy sampler rejected 31.4 %
  of first draws and biased sx toward compression (mean 0.9554); the coupled one
  accepts 100 % and realizes `sx ~ U(0.85, 1.1111)` exactly (`PHASE_G.md` §1.3).
* **188 k at ch 80 with KD on was worth +0.05** — the +0.5 measured at ch 56/64 in
  `PHASE_F.md` §13 does not transfer to this width. No 94 k no-KD arm was run, so
  schedule × KD at ch 80 is not decomposed.

What it bought where it counts:

| comparison | t1 | t3 | t5 | ≤3 | 4+ | source |
|---|---|---|---|---|---|---|
| val seed-mean, footing A | **+0.25** | +0.12 | +0.08 | **+0.43** | +0.16 | `PHASE_G.md` §4 |
| **test seed-mean, footing A** | **+0.39** | **+0.29** | **+0.00** | **−0.37** | **+0.78** | `PHASE_G.md` §7.5 |
| equal footing (footing C), Δ of Δs | +0.39 | +0.29 | +0.00 | −0.37 | +0.78 | derived from §2.3 |

And the one that matters most for shipping: on the **app footing**, the
**worst-seed top-5 margin is +0.75 for `resbn80g` against +0.08 for
`fast_resbn80`** — two rows of 2,400 vs eighteen. `fast_resbn80` shipped on a
knife edge that Phase F had flagged on val and that survived into test; that
knife edge is gone. On val the t5 margin roughly doubled (+0.17 vs +0.09) though
the worst-seed val t5 margin only moved 0.05 → 0.03 (`PHASE_G.md` §4).

Read the ≤3 column honestly: **−0.37 on test**, i.e. the upgrade's +0.43 val gain
on short words did not merely fail to transfer, it reversed. `PHASE_G.md` §7.5
notes the short-word stratum has moved opposite to its prediction on all three
unsealings.

### 4.2 `resbn80g` vs ch 128 — is the 2.45×-smaller model level now?

ch 128 is 2.47× the parameters, **2.45× the bytes** and 2.21× the latency.

| footing | Δ t1 | Δ t3 | Δ t5 | Δ ≤3 | Δ 4+ | source |
|---|---|---|---|---|---|---|
| **val-9918, AOSP, E1** | **−0.16** | **+0.02** | **+0.01** | −0.20 | −0.12 | §2.4 **[derived]** |
| **test-2400, AOSP, E1** | **−0.24** | −0.15 | −0.18 | −0.28 | −0.21 | §2.1 **[derived]** |
| *(for contrast)* `fast_resbn80` vs ch 128, test, AOSP | −0.63 | −0.44 | −0.18 | +0.09 | −0.99 | `RESULTS.md` §"The second unsealing" |

**On val the small model is level** — t3 and t5 are ties in everything but sign
(+0.02, +0.01), t1 is −0.16. **On test it is not quite level**: uniformly behind
by 0.15–0.28 on all five. The gap narrows sharply against `fast_resbn80`'s
(mean |Δ| 0.21 vs 0.47) on four of five metrics; the exception is ≤3, where
`fast_resbn80` was marginally *ahead* of ch 128 (+0.09) and `resbn80g` is behind
(−0.28).

Two cross-checks that go the other way and matter for a ship decision:

* **On the app trie** (the lexicon that ships), the two models have not been
  compared at a common preset for `resbn80g`. ch 128's app-trie val seed-mean at
  E1 is 87.96/92.77/93.67/91.49/86.12; `resbn80g`'s at E1, single seed, is
  87.37/92.90/93.86/90.50/85.74 (`PHASE_F.md` §15.3, `PHASE_G.md` §6) — small
  model ahead on t3/t5, behind on t1/≤3/4+. At its own app preset `resbn80g`
  moves to 88.76/93.22/94.18/92.42/86.86 (s1234), ahead of ch 128 on all five —
  but ch 128 was never swept on the app trie, so **that is a tuned-vs-untuned
  comparison and not admissible as a model-quality claim.**
* **Cross-layout, ch 128 loses to the small model outright** — see §6.5.

Verdict: **level on val, ~0.2 pt behind on test, at 41 % of the bytes and 45 % of
the encoder time.** That is the trade, stated exactly.

### 4.3 `resbn80g` vs the equal-footing FUTO bar

Quoted precisely, from `PHASE_G.md` §7.5:

> Seed-mean Δ: **+0.56 t1, −0.11 t3, −0.14 t5, +0.86 ≤3, +0.40 4+** — 3 of 5, as
> §7.4 predicted. Exact paired McNemar on t1 against FUTO's per-row val-tuned
> output: **+17 (p 0.17), +23 (p 0.052), +0 (p 1.00)** — level-to-slightly-ahead,
> resolved on no seed. … **No equal-footing superiority claim is made.**

The move from the incumbent is real and worth naming: `fast_resbn80` lost t3, t5
and 4+ by 0.14–0.40 and had a **net-negative** McNemar seed; `resbn80g` turns
those three losses into two −0.1 ties and a +0.40 win, with no negative seed. It
is *level*, not ahead. ch 192 remains the only configuration with a (qualified)
equal-footing win.

### 4.4 Everything vs the old shipped transformer — the rout

Lexicon-matched (footing B, both engines on the bundled `en_enhanced`
dictionary, both taking 64 forced OOV misses):

| metric | old shipped NN | **`resbn80g`** | Δ |
|---|---|---|---|
| t1 | 74.62 | **88.14** | **+13.52** |
| t3 | 84.33 | **93.22** | **+8.89** |
| t5 | 87.42 | **93.90** | **+6.48** |
| ≤3 t1 | 89.45 | **91.86** | **+2.41** |
| 4+ t1 | **67.00** | **86.23** | **+19.23** |

On footing A (our AOSP trie, an ~0.9 pt OOV handicap *against* us) the same
comparison is +13.06 / +7.85 / +5.40 / +1.35 / +19.08 **[derived]** from §2.1.

`THREEWAY_AUDIT.md` §5 resolved the ch 192 version of this gap at **z ≈ 12 on t1
and z ≈ 14 on 4+**, called it "the only pairing that is unambiguous", and
concluded: *"Every axis — accuracy, size, speed, strata — favours the CTC models;
there is no registered caveat that could plausibly reverse a 14-pt resolved gap.
Replacing the shipped transformer is supported without qualification."*
`resbn80g` sits 0.24 t1 below ch 128, which sits 0.44 below ch 192 — the ordering
is unchanged and the gap to the transformer is ~13 pt for every one of them, at
**11 % of its bytes**.

The old NN's failure mode is structural: 4+-char top-1 collapses to 67.00 on test
/ 69.15 on val against our 86.08 / 86.14 — a ~19-pt gap, independently reproduced
by the audit's fresh 500-row val re-run (old NN 65.4 on 4+, n = 315). Its one
durable strength, short words (≤3 89.45), is now also beaten by every CTC model
in this document.

---

## 5. Ship recommendation matrix

**Read this first, post-Phase-J.** The matrix below ranks by *evidence quality*
as well as accuracy, and the two now diverge: the most accurate models on val and
alt-layouts (`sw2345`, then `resbn192i`) are **not test-validated**, while every
test-validated model is behind them on val. That is a deliberate state — the
seal was not spent, because Phase J's terminal condition was not met (§2.8).
Choosing `sw2345` means choosing a model whose accuracy claim rests on val-9918
plus seven alt-layout corpora and on no sealed split at all.

| priority | pick | why | what must move with it |
|---|---|---|---|
| **Best measured accuracy, accepting val-only evidence** | **`sw2345`** (Phase-J finalist, 1,512,802 params, 3,052,318 B fp16w / 2.91 MiB, 0.842 ms) | Best val seed-mean in the campaign (88.51 / 92.67 / 93.37 / 91.20 / 87.11) and **6 of 6 alt-layout bars**, 3 seeds. Beats the `resbn192i` incumbent on four of five val metrics. | **E1** preset (no app-trie sweep has been run for it); a golden fixture regenerated from it at whatever preset ships. **Never quote it as test-validated**, and it **misses the `≤3` bar by 0.07** — see §2.8. Cyrillic on this family is unresolved and the ru path wants **λ ≈ 2.0**, not E1's 1.1. |
| **Accuracy first, device budget permits ~0.9 ms encoder** | **ch 192** (`ch192_s1234.onnx`, 1,525,378 params, 6.14 MB, 0.877 ms) | The only configuration with a **qualified equal-footing win** over FUTO (all five, McNemar resolved on 2 of 3 seeds). Best on every test metric. | E1 preset; golden fixture regenerated from ch 192 at whatever preset ships. |
| **Accuracy first, balanced size** | **ch 128** (689,282 params, 2.80 MB, 0.455 ms) | Campaign-2's shipping pick. Test-validated, clears all five on every seed on both tries, +0.19 t1 behind ch 192 on val for 1.9× less encoder time. Equal-footing lead does **not** resolve on any metric or seed — do not claim superiority for it. | E1 preset; the existing ch 128 fixture. |
| **Balanced — the recommendation** | **`resbn80g`** (`resbn80g_s1234.onnx`, 279,346 params, 1.14 MB, **0.215 ms class**) | Test-validated on **both** footings, every seed. 2.45× smaller and 2.21× faster than ch 128 for −0.24 t1 on test / −0.16 on val. On the shipping footing it clears the trie-matched bar by +3.22/+1.68/+0.94/+2.29/+3.71 with a **+0.75 worst-seed t5 margin**. Level with FUTO's val-tuned engine. | **`CtcScoringParams(gamma = 0.9, lambda = 4.0, beta = 0.25, alpha = 0.0, gammaPrune = 0.25, betaPrune = 0.9882)`** *and* the fixture `ctc_model_golden.json` sha256 `ce3b54…`, regenerated from `resbn80g_s1234` at that preset. |
| **Smallest that clears the val bars** | `fast_resbn72` (229,642 params, 0.94 MB, 0.186 ms) | Fastest 5/5 configuration measured. | **Val evidence only, permanently** — never decoded on test, and the third unsealing is spent. Its Phase-G re-trained sibling is still in progress (§7). |
| **Do not ship** | `fast_resbn80` | Superseded by `resbn80g` at identical cost. Its five-of-five pass was against the *published* bar; on equal footing it fails three of five, and its app-footing worst-seed t5 margin was +0.08. | — |

### 5.1 The fixture-and-preset rule

**The golden fixture and the scoring preset move together, always.** The fixture
records its own `source_onnx_sha256` and `preset`; `CtcParityTest` asserts Kotlin
reproduces Python bit-for-bit *at that preset*. Shipping the model at one preset
and the fixture at another means the parity test asserts against a configuration
nothing runs (`RESULTS.md` §Artifacts).

Concretely, for `resbn80g`:

* app runtime preset **must** be `0.9 / 4.0 / 0.25 / 0.25 / 0.9882`;
* fixture **must** be `artifacts/ctc_model_golden.json` sha256 `ce3b5456…`,
  which was generated from `resbn80g_s1234` at exactly that preset;
* every **benchmark-footing** number in this repo stays at **E1**
  (`1.05 / 1.1 / 0.2 / 0.3734 / 0.9882`) — the per-model sweep converged to E1 on
  the AOSP trie for the third model in a row (`PHASE_G.md` §6), so no
  benchmark-side churn is required.

**E1 has now transferred unchanged for a fifth model family.** Phase J swept the
E1 region symmetrically over both the finalist and the incumbent with a
stratum-aware `minmargin` objective (maximise the worst margin over the five val
bars), tuned on val`[0:4959]` and confirmed on val`[4959:9918]`: **both models
landed back on their own E1 numbers to within ±0.07 on every metric**
(`PHASE_J.md` §6.8b). This is the strongest evidence in the campaign that E1 is
a property of the emission/trie pair rather than of any individual model, and
that no per-model benchmark retuning is required. It applies to the **en**
footings only: the ru lexicon is a different scale and wants λ ≈ 2.0 (§2.8).

Note that `RESULTS.md` §"Next — app-side" still says "drop `ch128_s1234.onnx`
into the `CtcEmissionModel` seam" and "update `CtcScoringParams` to gamma 1.05,
lambda 1.1 …". That is Campaign-2 text; **Phase G supersedes it** for the
speed-class pick and for the app preset (`RESULTS.md` §Phase G says so
explicitly). The I/O contract is identical either way, so the choice is an asset
swap plus two constants.

### 5.2 Caveat that travels with λ 4.0 specifically

The user-dictionary merge injects top-of-scale (freq 255) competitors, and a 3.6×
larger λ amplifies them. **No evaluation anywhere in this campaign includes a
user dictionary** (`PHASE_F.md` §15.5, `PHASE_G.md` §6). This is the one preset
risk that the sweep could not price.

---

## 6. Caveats register

Every number in this document travels with all of these.

### 6.1 Preset asymmetry — the largest threat on footings A and B

Our decode preset was fitted on val-9918 by a five-parameter grid search; the
published bar is quoted at FUTO's own preset. The lever is worth **+2.29 pt t1 to
us** (`AUDIT_FINAL.md` §6.1) and **+1.94 pt t1 to FUTO** (`FAIR_REMATCH.md` §2) —
the same order, not a rounding difference. Roughly **two thirds of the published
test margin on t1 was an artifact of comparing a tuned preset against an untuned
one**. At the published preset our own ch 192 clears only **3 of 5** val bars.
Footing C exists precisely to remove this, and only footing C numbers are
asymmetry-free. **Footing B remains asymmetric** — no val-tuned FUTO bar exists on
the app trie — and `PHASE_G.md` §7.3 declared that in advance.

### 6.2 Contributor contamination — these are benchmark numbers, not generalization claims

T3 applies no session or participant exclusion: every contributor of every val and
test row is in training, and 3× HWS oversampling triples the exposure of the more
contaminated corpus (98.4 % of HWS holdout rows share a participant with
training). **No contributor-clean subset of val or test exists for this model.**
The counter-asymmetry runs in FUTO's favour: 5,273 of 12,299 unique holdout traces
(43 %) are bit-exactly in the HF *train* split FUTO trained on (0 in HF dev/test),
so the app repo's description of the split as FUTO-held-out is incorrect.
Separately, the dedup-key defect left 588 val / 145 test rows in `train_t3` with
bit-identical tensors; measured effect is *negative* (leaked rows score 4.34 pt
**below** comparable non-leaked ones) and removing all of them costs < 0.05 pt on
val / 0.20 pt on test with all five bars still clearing.
(`AUDIT_FINAL.md` §6.2–6.4, `RESULTS.md` §Caveats 2–4.)

### 6.3 A worn test split — three reads

test-2400 has now been decoded **three** times: ch 128 + ch 192 (first unsealing,
pre-registered in `AUDIT_PREDECODE.md` §E), `fast_resbn80` (second, `PHASE_F.md`
§16, on the user's order), `resbn80g` (third, `PHASE_G.md` §7, pre-authorized by
the user's 2026-08-09 directive and gated on the val bars). Each was
pre-registered before the decode, hard-capped at 6 decodes, and logged in
`test2400_seal.json["test-2400"]["unsealings"]`. `RESULTS.md` §Phase G states the
price plainly: **"it is a worn split and a fourth read needs a better reason than
any of the first three had."** Also on the record: 7 traces are bit-exactly shared
between val-9918 and test-2400 (0.29 %), so a sliver of test sat inside the tuning
corpus — symmetrically for both engines since the rematch, since FUTO's preset was
tuned on the same rows.

### 6.4 Per-source spread — a 13–15 pt gap the aggregates hide

See §2.6. Every aggregate top-1 in this document is the average of a ~94–95 on the
FUTO half and a ~80–82 on the How-We-Swipe half. On the HWS half alone these
models sit *below* the aggregate bar. Any product claim derived from the aggregate
is a claim about a 50/50 mixture of two very different corpora.

### 6.5 `resbn80g`'s cross-layout transfer is UNMEASURED

> **Still true for `resbn80g` itself**, but no longer true of the family: Phases
> H, I and J measured cross-layout transfer on every candidate they produced,
> and the Phase-H layout-resampling augmentation was built precisely to close
> the dvorak gap this section describes. The Phase-J finalist clears all six
> alt-layout bars (§2.8), and the numbers below are the *pre-augmentation*
> picture. What remains unmeasured is `resbn80g` — the test-validated model —
> which was trained before that augmentation existed.

`ALT_LAYOUT_EVAL.md` evaluated **`ch128_s1234`** and **`fast_resbn80_s1234`**
only. **No Phase-G model has been run on any alternate layout.** Its findings must
not be extrapolated to `resbn80g` — even though `resbn80g` is the same graph as
`fast_resbn80`, it was trained with the *fixed affine sampler*, which
`ALT_LAYOUT_EVAL.md` §7.2b identified as the likely cross-layout culprit, and
`PHASE_G.md` §3.2 says explicitly that "its *cross-layout* value … is not measured
here and remains the stronger argument for it". The measured picture, for the two
models that were run (single-seed, in-dict protocol, E1, `az26` arm):

| layout | lang | mean key displacement vs qwerty | ch 128 t1 | `fast_resbn80` t1 | Δ |
|---|---|---|---|---|---|
| qwerty (val[0:2000], n=1,928) | en | 0.0000 | 91.55 | 91.08 | **−0.47** |
| spanish | es | 0.0175 | 81.34 | 82.37 | **+1.03** |
| qwertz | de | 0.0579 | 76.66 | 78.77 | **+2.11** |
| azerty | fr | 0.1068 | 75.31 | 76.03 | **+0.72** |
| german | de | 0.1071 | 72.08 | 76.17 | **+4.09** |
| dvorak | en | 0.4313 | 63.04 | 67.28 | **+4.24** |

Source: `ALT_LAYOUT_EVAL.md` §0 and §6. The verdict there is
**"language-agnostic: yes; layout-agnostic: only near QWERTY"** — slot-permutation
invariance is perfect (≤ 3.8e-6 change in emissions) but irrelevant, and accuracy
decays monotonically with key displacement; greedy top-1 falls from 72.8 % to
11.6 % on dvorak, where the beam and the English trie do essentially all the work.
The 2.5×-smaller model transfers **better** on every alt-layout, with its margin
growing monotonically with layout difficulty — read as ch 128's extra capacity
having gone into QWERTY-specific memorization. Against the shipped geometric
engine the CTC model wins or ties on four of five layouts and **loses dvorak by
13.8 pt** (`ALT_LAYOUT_EVAL.md` §6). If `resbn80g` ships and alternate layouts
matter, **that evaluation needs re-running on it.**

### 6.6 Lexicon and OOV bookkeeping

Our footing-A runs and the val bar use the *same* 146,964-word AOSP STRIP trie, so
"our larger lexicon makes these conservative" does **not** apply on val. The
published test bar was measured on the 131,544-word DROP trie and re-measured
**unchanged overall** on STRIP, so the overall test comparison is trie-neutral —
but **its strata were never republished**, so ≤3 and 4+ on test are compared
across normalizers. The app trie is a third smaller and yet covers *more* targets
(64 OOV on test vs 86; 250 val vs 336), and its `log_freq` spread is 0.64 against
AOSP's 5.40 — an 8× scale collapse, which is exactly why λ moves from 1.1 to 4.0
on that footing. (`AUDIT_FINAL.md` §6.6, `PHASE_F.md` §15.1/§15.4,
`PHASE_G.md` §6.)

### 6.7 Selection hygiene, and what remains unattributed

Preset sweeps ran on val`[0:4959]` and were confirmed on the untouched
val`[4959:9918]`; checkpoint selection used beam top-1 over a 5,000-row val
prefix. But **which arms were stacked** was decided on full val-9918 tables
(`AUDIT_FINAL.md` §6.8). Phase-F absolute numbers at ≤280 k params should be read
as ~0.5 t1 understated, because every Phase-F arm carried the KD handicap that
Phase G measured as negative (`PHASE_G.md` §3.1) — Phase-F *arm-vs-arm*
conclusions survive, being common-mode. The KD **weight** was never swept, only
the temperature (negative) and now the on/off ablation (negative).

### 6.8 What the FUTO bar is not

`hungry_jellyfish`, FUTO's context LM, is downloaded but not run — our eval rows
mostly lack the preceding-word context it consumes. **The bar is a floor on FUTO's
full published stack, not a ceiling.** Our featurization and Viterbi beam are
ports, not FUTO's production C++, and FUTO's paper reports 93.30 on its own test
split against our port's 84.83–87.12 here, so every FUTO number in this document
is a conservative estimate of that engine (`FAIR_REMATCH.md` §7).

---

## 7. In progress / not yet known

* **int8-trunk for the Phase-J finalist** — not built. Its fp32/fp16w bytes and
  latency are measured and filled in (§1, §2.8, §3.1); int8-trunk was never
  needed, because fp16w already clears ≤5 MB at 2.91 MiB.
* **`sw2345` on any test footing** — it has never been decoded on test-2400 and
  under the campaign's own pre-registered rule it may not be, because the `≤3`
  and Cyrillic bars did not fall (§2.8). Its §2.1–§2.3 rows do not exist and must
  not be filled in from val figures.
* **The `≤3` stratum** — the finalist's one en shortfall (−0.07). Five levers
  have been measured against it and all five failed (§2.8). `PHASE_J.md` §9
  diagnoses the residue as a **candidate-generation** problem, not a training or
  re-ranking one, and leaves three untried directions on the register: T′ = 64
  emission resolution (contract-breaking, an app decision), a length-conditioned
  beam, and a ≤3-specific training signal.
* **Cyrillic** — the 76.21 bar stands. The λ ≈ 2.0 per-language preset finding
  (§2.8) is a genuine ~1 pt gain for whatever ru model ships, but it lifts every
  model equally and so resolves nothing. `sw2345` itself has no Cyrillic
  evaluation; the ru models are separate runs.
* **`sw2345` on the app trie** — no app-trie preset sweep has been run for it,
  so its only published footing is E1 / AOSP.
* **Latency stretch probes at ≤0.215 ms under the Phase-G recipe.** `PHASE_G.md`
  §8.1 has `phaseG-F72-188k-nokd` at seed 1234 (87.53/92.33/93.01/90.62/85.92,
  5/5 val bars, 0.186 ms class) — at one seed the 0.186 ms class *matches the
  incumbent 0.215 ms model's seed-mean*. **Seeds 4321/7777 are pending**, and the
  §8.2 ch 64 probe (0.162 ms class, the arm whose t5 sat at 92.76–92.78 against a
  92.80 bar through all of Phase F) is **pending** — it was restarted from scratch
  after the second host reboot (`PHASE_G.md` §7.α). Both are **val-only either
  way**: the third unsealing is spent, so no Phase-G latency probe can become
  test-validated. A concurrent session owns `PHASE_G.md` and `RESULTS.md`'s top
  section for these; when they land, §1, §3.1 and §5 of this document need a
  re-read.
* **On-device latency for any CTC model** — blocked, §3.3.
* **Cross-layout transfer for `resbn80g`** — unmeasured, §6.5.
* **`resbn80g` vs ch 128 on the app trie at a common preset** — ch 128 was never
  swept on that trie, so the only common-preset comparison available is at E1,
  single seed for `resbn80g` (§4.2).
* **Statistical resolution (unpaired z) for `resbn80g` against the published
  bar** — not published in `PHASE_G.md`; only the paired McNemar against the
  val-tuned bar is.

---

## 8. Source index

| this document | reads from |
|---|---|
| §0 bars | `RESULTS.md` §"Verified test-2400 results"; `PHASE_F.md` §0, §15.2; `FAIR_REMATCH.md` §2, §4 |
| §0.1 campaign bars | `PHASE_J.md` §0, §6.5; `RESULTS.md` §Phase I-A |
| §2.8 Phase-J finalist | `RESULTS.md` §Phase J; `PHASE_J.md` §3.1, §3.3, §6.5, §6.6.1, §6.6.2, §6.7, §6.8, §6.8b, §6.9 |
| §1 model cards | `RESULTS.md` §Phase J + §Phase I-A + §Phase G + §Artifacts; `PHASE_F.md` §6, §8, §9; `THREEWAY_AUDIT.md` §3 |
| §2.1 | `RESULTS.md` §"Verified test-2400 results", §"The second unsealing"; `PHASE_G.md` §7.5; `THREEWAY_AUDIT.md` §1 |
| §2.2 | `PHASE_G.md` §7.5 config B; `RESULTS.md` §"The second unsealing" config B; `PHASE_F.md` §15.1–15.2 |
| §2.3 | `FAIR_REMATCH.md` §4–§6; `PHASE_G.md` §7.2, §7.5 |
| §2.4 | `THREEWAY_AUDIT.md` §2 (quoting `PHASE_E.md` §5, `PHASE_F.md` §8/§14.1); `PHASE_G.md` §4–§5; `FAIR_REMATCH.md` §2–§3; `RESULTS.md` §Phase J + §Phase I-A |
| §2.5 | `PHASE_G.md` §6, §7.4; `PHASE_F.md` §15.2–15.3 |
| §2.6 | `RESULTS.md` §"Per-source", §"The second unsealing"; `PHASE_G.md` §4, §7.5 |
| §2.7 | `AUDIT_FINAL.md` §5; `FAIR_REMATCH.md` §5; `PHASE_G.md` §7.5 |
| §3.1 | `PHASE_F.md` §0, §1, §6, §7; `PHASE_G.md` §4 |
| §3.2–3.3 | app repo `web_demo/README.md` §Latency; app repo `memory/todo.md` §"CTC on-device latency measurement (G3)"; `THREEWAY_AUDIT.md` §3 |
| §3.4 | `THREEWAY_AUDIT.md` §3, §4.5 |
| §4.1 | `PHASE_G.md` §1.3, §2, §3, §3.1, §3.2, §4, §7.5 |
| §4.2 | §2.1/§2.4 of this doc; `PHASE_F.md` §15.3; `PHASE_G.md` §6 |
| §4.3 | `PHASE_G.md` §7.5; `FAIR_REMATCH.md` §5 |
| §4.4 | `THREEWAY_AUDIT.md` §1, §4.3, §5, §6.2 |
| §5 | `RESULTS.md` §Phase J + §Phase G + §"Shipping recommendation"; `PHASE_G.md` §6; `PHASE_J.md` §6.8b; `FAIR_REMATCH.md` §5 |
| §6 | `AUDIT_FINAL.md` §5–§7; `RESULTS.md` §Caveats + §Phase G; `FAIR_REMATCH.md` §7; `PHASE_F.md` §11, §15; `PHASE_G.md` §3.1, §6, §7.1; `ALT_LAYOUT_EVAL.md` §0, §6, §7.2b |
| §7 | `PHASE_G.md` §7.α, §8.1; app repo `memory/todo.md` |
