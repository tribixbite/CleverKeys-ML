# MODELS_TABLE — the definitive registry of every trained model and configuration

**Date:** 2026-08-15 · **Scope:** every model, ablation arm and multi-model
configuration produced by the CTC swipe-encoder campaign, Phases A → M, plus the
two pre-campaign runs and the FUTO reference engine it was measured against.
**Status:** standalone registry. **No measurement was run for it and nothing was
re-decoded.** Every number is quoted from a committed document and carries the
document and section it came from. Where a value was never recorded, the cell
reads **not recorded** — it is never reconstructed, inferred or rounded in from
a neighbouring model.

> **Scope boundary — Phase N is in flight and is NOT in this registry.** A
> concurrent session opened `PHASE_N.md` ("win the FUTO domain outright") on
> 2026-08-15, with N0 complete (official FUTO dev/test converted,
> `futo-test-49970` sealed) and N1 dev sweeps running. Nothing from Phase N —
> no model, no arm, no split, no bar — appears below. When Phase N closes, this
> file needs a §4.14 and its own footing entry: **the `futo-test-49970` split
> is a different benchmark from test-2400 and its numbers may not be placed in
> any column here.**

This file supersedes the scattered per-phase model lists for the purpose of
*"which models exist, what did each cost, what did each score, and how good is
the evidence"*. It does **not** supersede the phase documents themselves: where
this file and `RESULTS.md` / `PHASE_*.md` / `UNSEALING_4.md` disagree, **they are
right and this is stale**.

## How to read a row

* **Footings may never be mixed in one column.** Every accuracy cell names its
  footing. The legend is §7; read it before quoting anything.
* **Latency** is laptop, encoder-only, single-thread, batch-1 CPU ONNX Runtime.
  Each cell says which harness produced it and whether it was **measured** for
  that model or **inherited** from an identical exported graph. Encoder latency
  is *not* the decode budget — see `MODEL_COMPARISON.md` §3.2.
* **Evidence tier** is one of: **test-validated** (decoded on the sealed
  test-2400 under a pre-registered unsealing), **val-only** (val-9918 and/or
  alt-layout corpora only — and permanently so; there is no fifth unsealing),
  **superseded**, **refuted / died**, **control**, or **reference (opponent)**.
* **[derived]** marks a value this file computed arithmetically from published
  numbers (a mean of published per-seed values, a mean of the four euro-layout
  corpora). The inputs are always named in the same cell.
* **Euro-mean** = arithmetic mean of the four euro-layout corpora
  (azerty / qwertz / german / spanish) top-1. It is always **[derived]**; no
  phase document publishes it as a metric.
* `≤3` / `4+` are the short-word (n=3,389 val / 815 test) and long-word
  (n=6,529 val / 1,585 test) strata of top-1.

## The seal ledger, in one place

`test2400_seal.json["test-2400"]["unsealings"]` — the sealed split was read
**four times, and there is no fifth**:

| # | date | models decoded | authority | published in |
|---|---|---|---|---|
| 1 | 2026-08-08 | ch 192 (`phaseE-FINAL`×3) + ch 128 (`phaseE-E3b-hws3x`×3), 6 decodes | pre-registration `AUDIT_PREDECODE.md` §E | `RESULTS.md` §"Verified test-2400 results"; `AUDIT_FINAL.md` |
| 2 | 2026-08-08 | `fast_resbn80` ×3 seeds × 2 configs = 6 decodes | the user's explicit order | `PHASE_F.md` §16; `RESULTS.md` §"The second unsealing" |
| 3 | 2026-08-09 | `resbn80g` (`phaseG-C80-188k-nokd`×3) × 2 configs = 6 decodes | the user's 2026-08-09 directive, gated on the val bars | `PHASE_G.md` §7; `RESULTS.md` §Phase G |
| 4 | 2026-08-14 | `phaseM_kd_fresh_w1` (`v2kd-fresh-w1`×3, fp32) × 2 configs = 6 decodes | the user's 2026-08-13/14 directive, subject fixed by `PHASE_M.md` §11.2 | `UNSEALING_4.md` §8; `RESULTS.md` §"The fourth unsealing" |

Everything not in that table is **val-only permanently**. In particular the
coupled pair `v2pair-s1234` — the campaign's most accurate configuration on val
— was deliberately left sealed (`UNSEALING_4.md` §1, `RESULTS.md`
§"The fourth unsealing").

---

## 1. The ship model and the active candidates

The campaign closed on 2026-08-14 with a four-option ship menu (`PHASE_M.md`
§11.2, re-affirmed on test evidence at §12.1: **"ship B"**). Rows below are the
options as that section states them, plus the two small-budget picks that remain
correct under a tighter size/latency budget.

| rank | model / configuration | phase | size | evidence tier | eleven campaign bars | vs the `mix2-i8f16` card | source |
|---|---|---|---|---|---|---|---|
| **B — THE RECOMMENDATION** | **`phaseM_kd_fresh_w1_s1234_fp16w`** — one ch192 model distilled from the coupled pair | M | **2.91 MB** (3,052,318 B), one ONNX session, one graph, frozen `[1,32,65]` contract | **TEST-VALIDATED on both footings, every seed** (4th unsealing) | **11/11 on 3 of 3 seeds and the seed-mean**, smallest margin ≤3 **+0.103** | 7/11 seed-mean — beats **all five val** numbers, misses 4 transfer axes by 0.06–0.43 → **crown NOT won** | `PHASE_M.md` §9, §11.2, §12.1; `UNSEALING_4.md` §8 |
| A — accuracy-first alternative | **`v2pair-s1234` i8f16** — member A int8w + member B fp16w, per-frame probability averaging before the beam | L | **4.39 MB**, **two** ONNX sessions, 1.79 ms (no protocol cited in `PHASE_L/M.md`) | **val-only, permanently** — deliberately not decoded, and there is no fifth unsealing | **11/11 on 5 of 5 seeds**, five-seed mean margins +0.124 … +2.756 | 10/11 at s1234; **7/11 at the seed-mean** (`PHASE_M.md` §11.2). At the three-seed mean the card keeps dvorak −0.34, dvorak-app −0.32, spanish −0.40, azerty −0.05 (`PHASE_L.md` §15.3); at the five-seed mean those gaps widen to **−0.601 / −0.574 / −0.512 / −0.164 [derived** from `PHASE_M.md` §7.2 against the card**]**. → **bar 1 NOT met** | `PHASE_M.md` §7.2, §11.2; `PHASE_L.md` §12 |
| C — the incumbent recorded ship configuration | **`mix2-i8f16`** — `sw2345_s1234` int8w + `resbn192i_s1234` fp16w, prob-averaged | K | 4.45 MB, two sessions, **1.79 ms** encoder (0.930 + 0.858, measured) | val + alt-layout only | 11/11 **as one configuration** — the recipe does not reproduce: the s4321 pair fails its gate at 88.8 % agreement (`PHASE_K.md` §4.3). (`PHASE_M.md` §11.2 pins the non-reproduction on "PHASE_K §8.5", but §8.5's blind s5555 gate in fact **passed**, at 10/11 — the non-reproduction evidence is §4.3.) | it *is* the card | `PHASE_K.md` §4.3, §8.2, §8.5; `PHASE_M.md` §11.2 |
| D — previous single finalist | **`sw2345`** (fp16w) | J | 2.91 MB fp16w / 6.07 MB fp32 | val + alt-layout only | 10/11 seed-mean (≤3 −0.07); **5/11 every-seed** | — | `PHASE_J.md` §8; `PHASE_M.md` §11.2 |
| small-budget pick, best evidence | **`resbn80g`** | G | 1.14 MB, **0.213 ms** | **test-validated** (3rd unsealing) | n/a — predates the eleven-bar set; clears all five *val* bars and both test footings | — | `PHASE_G.md` §4, §7.5 |
| smallest configuration clearing all five val bars | **`resbn72g`** (`phaseG-F72-188k-nokd`) 0.184 ms · **`fast_resbn72`** 0.186 ms | G / F | 0.94 MB | **val-only permanently** | n/a — five val bars, every seed | — | `PHASE_G.md` §8.1; `PHASE_F.md` §14.1 |

**What must ship with option B** (fixture and preset move together,
`MODEL_COMPARISON.md` §5.1): app runtime preset
`CtcScoringParams(gamma = 0.9, lambda = 4.0, beta = 0.25, alpha = 0.0, gammaPrune = 0.25, betaPrune = 0.9882)`
— fitted on `resbn80g`, **never swept for this model family**, and now
test-validated on this model as config B — plus the fixture
`artifacts/phaseM_kd_fresh_w1_fp16w_golden.json` (140,462 B, sha256
`2a449c4f2de19505131b396655ae01d3e3c325e40249446ff6e7a40c2b27559c`),
regenerated at exactly that preset. The earlier 140,480-byte fixture generated
at **E1** is struck (`PHASE_M.md` §11.1). Benchmark numbers stay at E1.
**int8w is not available to option B** — it was not built, because
`PHASE_L.md` §16 measured int8w costing a single model the ≤3 bar and this
model's ≤3 margin is +0.103 (`PHASE_M.md` §11.1).

---

## 2. The test-validated tier — the only five models ever decoded on test-2400

Five models, four unsealings. **No other CleverKeys model in this file has a
test-2400 number and none ever will** — the two exceptions are not counter-
examples: `r2`'s pre-campaign decode (§4.0) is a *disclosed prior contact* in
the seal ledger rather than an unsealing, and §6's FUTO rows are the external
reference, decoded from FUTO's own weights. Config A = AOSP STRIP 146,964 at E1; config B = app
`en_enhanced` 98,081 (at E1 for `fast_resbn80`, at the app preset
`0.9/4.0/0.25/0.25/0.9882` for `resbn80g` and `phaseM_kd_fresh_w1`). Bars: A
published `84.83/91.04/92.08/89.57/82.40`; B trie-matched
`84.92/91.54/92.96/89.57/82.52`; equal-footing (both engines val-tuned)
`87.12/92.29/92.96/89.94/85.68`.

| model | phase | arch (ch · blocks · T′) | params | bytes fp32 / fp16w / int8w | laptop ms (protocol) | val-9918 t1/t3/t5/≤3/4+ (E1, AOSP) | test-2400 config A (unsealing) | test-2400 config B | equal footing (McNemar on t1) | alt-layout dvorak / euro-mean | seeds | evidence tier | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **`phaseM_kd_fresh_w1`** (ship: `_s1234_fp16w`) | M | `resbn:192:1,2,4,8`, embed_hid 96, T′ 32 | 1,512,802 | 6,068,519 / **3,052,318** / not built | **0.83 ms class — INHERITED** (identical graph to `resbn192i`, measured 0.831 / 0.849 p90 idle, `PHASE_I.md` §8); no per-model bench exists in `PHASE_L/M.md` | **88.750 / 92.773 / 93.473 / 91.373 / 87.387** (`PHASE_M.md` §9 via `UNSEALING_4.md` §4.1). App-trie/app-preset val: 89.377 / 93.680 / 94.467 / 92.563 / 87.727 (`UNSEALING_4.md` §4.2) | **88.931 / 92.681 / 93.361 / 92.597 / 87.045** — all five, every seed, Δ +4.10/+1.64/+1.28/+3.03/+4.64 (**unsealing 4**, `UNSEALING_4.md` §8.1) | **89.306 / 93.792 / 94.500 / 93.701 / 87.045**, all five every seed, worst-seed t5 **+1.50** (`UNSEALING_4.md` §8.2) | **+1.81/+0.39/+0.40/+2.66/+1.36**, all five every seed; McNemar **3 of 3** — +45 (p 3.87e-05), +46 (7.69e-05), +39 (4.99e-04) → **qualified equal-footing win** (`UNSEALING_4.md` §8.3 + erratum) | dvorak **91.82**, dvorak-app 91.10 · euro-mean **84.83 [derived** from azerty 84.53 / qwertz 83.97 / german 81.30 / spanish 89.53, `RESULTS.md` §Phase M**]** | 1234 / 4321 / 7777 | **test-validated** (both footings, every seed) | `UNSEALING_4.md` §2.3 (params, bytes, sha256), §4.1–4.2 (val), §8 (test); `RESULTS.md` §"The fourth unsealing", §Phase M |
| ch 192 (`phaseE-FINAL`) | E | `res:192` (GroupNorm family), embed_hid 192, T′ 32 | 1,525,378 | 6,144,249 / not recorded / not recorded | **0.877 ms** audit protocol (`AUDIT_PREDECODE.md` §7) / 0.920 Phase-F harness; 0.898 measured in `PHASE_E.md` §4 | 88.06 / 92.32 / 93.08 / 90.86 / 86.62 (`PHASE_E.md` §5) | **88.36 / 92.65 / 93.50 / 91.37 / 86.81**, all five every seed (**unsealing 1**, `RESULTS.md` §"Verified test-2400 results") | not decoded on the app trie | +1.24/+0.36/+0.54/+1.43/+1.14; McNemar **2 of 3** (+40 p 4e-4 · +18 p 0.16 · +31 p 0.0101) → qualified win (`FAIR_REMATCH.md` §5) | not recorded (never alt-layout evaluated) | 1234 / 4321 / 7777 | **test-validated**; keeps test t5 (93.50, +0.14 over the ship model) | `RESULTS.md` §"Verified test-2400 results"; `FAIR_REMATCH.md` §5 |
| ch 128 (`phaseE-E3b-hws3x`) | E | `res:128` (GroupNorm), 4 blocks, T′ 32 | 689,282 | 2,799,865 / not recorded / not recorded | **0.455 ms** on the `AUDIT_PREDECODE.md` §7 audit protocol; **0.472–0.475** on the Phase-F harness (`PHASE_F.md` §0); `PHASE_E.md` §4's own idle reading is 0.470 | 87.88 / 92.23 / 92.96 / 90.98 / 86.26 (`PHASE_E.md` §5) | **87.92 / 92.33 / 93.00 / 91.08 / 86.29**, all five every seed (**unsealing 1**) | not decoded on the app trie (its val app-trie seed-mean is 87.96/92.77/93.67/91.49/86.12, `PHASE_F.md` §15.3) | +0.79/+0.04/+0.04/+1.15/+0.61 — t3/t5 are one-trace ties; McNemar resolves on **none** → no superiority claim admissible | dvorak **63.04** (single seed s1234) · euro-mean **76.35 [derived** from 75.31/76.66/72.08/81.34, `ALT_LAYOUT_EVAL.md` §6**]** | 1234 / 4321 / 7777 (alt-layout: s1234 only) | **test-validated**; Campaign-2 ship pick, superseded | `RESULTS.md` §"Verified test-2400 results"; `FAIR_REMATCH.md` §5; `ALT_LAYOUT_EVAL.md` §6 |
| **`resbn80g`** (`phaseG-C80-188k-nokd`) | G | `resbn:80:1,2,4,8`, embed_hid 96, 4 blocks, T′ 32, BN folded | 279,346 | 1,142,727 / not recorded / not recorded | **0.213 ms** idle, `PHASE_F.md` §0 protocol — MEASURED (`PHASE_G.md` §8); 0.212 / 0.222 p90 paired `bench_latency.py` (`PHASE_H.md` §6); quoted as the "0.215 ms class" | 87.72 / 92.25 / 92.97 / 90.78 / 86.14, all five every seed (`PHASE_G.md` §4). App trie at the app preset, 3 seeds: 88.54 / 93.17 / 94.06 / 91.92 / 86.79 (`PHASE_G.md` §7.4) | **87.68 / 92.18 / 92.82 / 90.80 / 86.08**, all five every seed, Δ +2.85/+1.14/+0.74/+1.23/+3.68 (**unsealing 3**, `PHASE_G.md` §7.5) | **88.14 / 93.22 / 93.90 / 91.86 / 86.23** at the app preset, Δ +3.22/+1.68/+0.94/+2.29/+3.71, worst-seed t5 +0.75 | +0.56/−0.11/−0.14/+0.86/+0.40 = **3 of 5**; McNemar +17 (p 0.17), +23 (0.052), +0 (1.00) → **resolved on no seed; no superiority claim permitted** | **not recorded — never alt-layout evaluated** (`PHASE_H.md` §5 used `fast_resbn80` as proxy) | 1234 / 4321 / 7777 | **test-validated** (both footings, every seed); superseded as ship pick by Phase M, still the pick under a ~1 MB budget | `PHASE_G.md` §4, §6, §7.5; `RESULTS.md` §Phase G |
| `fast_resbn80` (`phaseF-I-resbn80x4` + `phaseF-FINAL-resbn80x4-s{4321,7777}`) | F | `resbn:80:1,2,4,8`, embed_hid 96, 4 blocks, T′ 32; 94 k steps, KD from ch 192, legacy affine sampler | 279,346 | 1,142,727 / not recorded / not recorded | **0.215 ms** mean / 0.224 p90, `PHASE_F.md` §0 protocol — MEASURED | 87.47 / 92.13 / 92.89 / 90.35 / 85.98 (`PHASE_F.md` §8) | **87.29 / 91.89 / 92.82 / 91.17 / 85.30**, all five every seed vs the *published* bar (**unsealing 2**, `RESULTS.md` §"The second unsealing"); z: t1 3.4, 4+ 3.0, rest unresolved | **86.51 / 92.28 / 93.25 / 90.76 / 84.33** at **E1** (not the app preset); worst-seed t5 margin **+0.08**; nothing resolves at z>2 | +0.17/**−0.40**/**−0.14**/+1.23/**−0.38** = **2 of 5, fails three**; one seed net-negative → "**must not be described as beating FUTO at all**" (`RESULTS.md` §"The asymmetry"; `FAIR_REMATCH.md` §5 words it as "does not survive the rematch") | dvorak **67.28** (s1234) · euro-mean **78.34 [derived** from 76.03/78.77/76.17/82.37, `ALT_LAYOUT_EVAL.md` §6**]** | 1234 / 4321 / 7777 (alt-layout: s1234) | **test-validated but SUPERSEDED** by `resbn80g` at identical cost | `RESULTS.md` §"The second unsealing"; `PHASE_F.md` §16.5; `FAIR_REMATCH.md` §5; `ALT_LAYOUT_EVAL.md` §6 |

**Config-A test seed-means, all five models on one footing** (`UNSEALING_4.md`
§8.5): `phaseM_kd_fresh_w1` 88.931 / 92.681 / 93.361 / 92.597 / 87.045 · ch 192
88.36 / 92.65 / **93.50** / 91.37 / 86.81 · ch 128 87.92 / 92.33 / 93.00 /
91.08 / 86.29 · `resbn80g` 87.68 / 92.18 / 92.82 / 90.80 / 86.08 ·
`fast_resbn80` 87.29 / 91.89 / 92.82 / 91.17 / 85.30. The ship model is best on
four of five; **ch 192 keeps t5 by 0.14**.

---

## 3. The val-validated finalists — best models that were never decoded on test

Every row here is **val-only permanently**. All val numbers are val-9918 at E1
on the AOSP STRIP 146,964 trie unless the cell says otherwise; alt-layout is the
`az26` in-dict protocol at E1, with `dvorak-app` on the 98,081 app trie. The
eleven-bar tally is against the Phase-I `resbn192i` seed-means
(88.30/92.60/93.26/91.27/86.77 + dvorak 89.13 / dvorak-app 88.20 / azerty 83.60
/ qwertz 82.50 / german 79.64 / spanish 88.28).

| model | phase | arch (ch · blocks · T′) | params | bytes fp32 / fp16w / int8w | laptop ms (protocol) | val-9918 t1/t3/t5/≤3/4+ (E1, AOSP) | test-2400 | alt-layout dvorak (app) / euro-mean | bars | seeds | tier | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **`sw2345`** (`phaseJ-sw2345`) | J | `resbn:192:1,2,4,8`, embed_hid 96, T′ 32; `resbn192i` recipe + `tier_sw234` (101,842) + `tier_sw5q` (24,707); 1,285,381 train rows | **1,512,802** | 6,068,519 / **3,052,318** / 1,554,355 (int8w exported in Phase K) | fp32 **0.816 / 0.830 p90**; fp16w **0.842 / 0.859** — MEASURED, `PHASE_F.md` §0 idle protocol (fp16w is **3 % slower**, not identical) | **88.51 / 92.67 / 93.37 / 91.20 / 87.11** (per-seed 88.51/88.57/88.46 …) | **never decoded** — the pre-registered rule required all bars and ≤3 + Cyrillic did not fall | dvorak **89.87** (app 88.98) · euro-mean **83.98 [derived** 83.81/83.01/80.64/88.45**]**; informational zero-shot clearflow 91.06, kasroz 92.07 | **10/11 seed-mean; 5/11 every-seed** (miss ≤3 −0.07) | 1234 / 4321 / 7777 | val + alt-layout only; single-model finalist until Phase M | `PHASE_J.md` §8, §10; `PHASE_K.md` §7 (int8w); `RESULTS.md` §Phase J |
| `resbn192i` (`phaseI-ch192-p65`) — the bar-holder | I-A | `resbn:192:1,2,4,8`, embed_hid 96, T′ 32, layout-alt **p 0.65**, 188 k, no KD | **1,512,802** | 6,068,519 / **3,052,318** / 1,554,355 (Phase-K export) | **0.831 / 0.849 p90** MEASURED (`PHASE_I.md` §8; fp16w identical); re-measured 0.819 / 0.833 in `PHASE_J.md` §10 | **88.30 / 92.60 / 93.26 / 91.27 / 86.77** — all five, every seed, worst-seed t5 +0.34. App trie at its own app preset `0.975/3.0/0.35/0.25/0.9882`: 89.23 / 93.54 / 94.30 / 92.53 / 87.52 | never decoded | dvorak **89.13** (app 88.20) · euro-mean **83.51 [derived** 83.60/82.50/79.64/88.28**]** | it **is** the eleven-bar set | 1234 / 4321 / 7777 | val + alt-layout only; registered nominee for an unsealing that was never spent on it | `PHASE_I.md` §7.2, §7.4, §8; `RESULTS.md` §Phase I-A |
| `resbn256i` (`phaseI-ch256`) — the capacity frontier | I-A | `resbn:256:1,2,4,8`, embed_hid 96, T′ 32, layout-alt **p 0.50** (p 0.65 never run at this width) | **2,668,194** | 10,685,479 / 5,360,800 / **int8-trunk 2,737,114** (int8w measured **free** at this width: 88.63/92.61/93.26/91.12/87.33, `PHASE_I.md` §2) | **1.372 / 1.389 p90**; int8-trunk 1.540 (+12 %) — MEASURED (`PHASE_I.md` §8) | **88.65 / 92.61 / 93.32 / 91.26 / 87.29** — best QWERTY val of the ladder | never decoded | dvorak seed-mean **86.92** (app 86.65), **transfer-volatile** (87.95/88.52/84.29) · euro-mean **82.29 [derived, s1234 only,** `PHASE_I.md` §5**:** 81.87/79.95/78.81/88.51**]** | not tallied — over budget at fp32/fp16w | 1234 / 4321 / 7777 | val + alt-layout only; **not shipped** (budget + transfer volatility); designated the Phase-J base | `PHASE_I.md` §7.1, §8 |
| `resbn80h` (`phaseH-p50`) | H | `resbn:80:1,2,4,8`, embed_hid 96, 4 blocks, T′ 32, `--layout-alt-p 0.5`; graph node-for-node identical to `resbn80g` | **279,346** | 1,142,727 / **589,406** / **317,476** (fp16w free; int8w −0.18 t1 / −0.26 4+) | **0.216 / 0.229 p90** MEASURED paired against `resbn80g`'s 0.212 / 0.222 (`PHASE_H.md` §6); re-measured 0.213 / 0.223 in `PHASE_I.md` §8 | **87.69 / 92.22 / 93.00 / 90.79 / 86.08** — all five, every seed; vs `resbn80g` −0.03/−0.03/+0.03/+0.01/−0.06 | never decoded | dvorak **90.01** (app **89.51**) — the gap closed from 67.28; dvorak greedy 11.6 → 42.5 · euro-mean **84.55 [derived** 84.27/84.36/81.13/88.43**]** | predates the eleven-bar set; beats the geometric engine on **all six** layouts | 1234 / 4321 / 7777 | val + alt-layout only | `PHASE_H.md` §5–§6; `PHASE_I.md` §2, §7.2; `RESULTS.md` §Phase H |
| `resbn72g` (`phaseG-F72-188k-nokd`) | G | `resbn:72:1,2,4,8`, embed_hid 96, 4 blocks, T′ 32, coupled sampler, no KD, 188 k | **229,642** | 944,487 / not recorded / not recorded | **0.184 ms** idle MEASURED (`PHASE_G.md` §8) | **87.62 / 92.22 / 93.02 / 90.48 / 86.14** — all five val bars, every seed, worst-seed t5 **+0.18**; exceeds `fast_resbn80`'s seed-mean on all five while being 14 % faster | never decoded (3rd unsealing spent) | not recorded | five val bars | 1234 / 4321 / 7777 | val-only permanently; **s4321 export parity is at the tolerance boundary** (occasional draws 2.1e-04 vs a 1e-4 assert, argmax 500/500) | `PHASE_G.md` §8.1–§8.2; `RESULTS.md` §Phase G addendum |
| `fast_resbn72` (`phaseF-N72-188k`) | F | `resbn:72:1,2,4,8`, embed_hid 96, 4 blocks, T′ 32, 188 k, KD from ch 192, legacy sampler | **229,642** | 944,487 / not recorded / not recorded | **0.186** mean (`PHASE_F.md` §9, §14.1 — §14's table prints 0.185) / **0.195 p90** (§6) — MEASURED on the §0 protocol | **87.27 / 92.09 / 92.87 / 90.49 / 85.60** — all five, every seed; **worst-seed t5 margin +0.01 = one row in 9,918**; seed sd on t1 (0.03) is the campaign's tightest | never decoded | not recorded | five val bars | 1234 / 4321 / 7777 | val-only permanently — "fastest 5/5 configuration measured" | `PHASE_F.md` §14.1, §9 |
| `phaseL memberA` (ship form `phaseL_memberA_s1234_fp16w`) | L | single ch192 `resbn` member A (slw 1.0) of the L1 coupled pair, T′ 32, `train_v2.py`, 188 k | **not recorded** (`PHASE_L.md` gives only "1.5 M-parameter"; same 6,068,519-byte graph) | 6,068,519 / **3,052,318** / **1,554,355** — **int8w costs it the ≤3 bar** (91.32 → 91.24) and −0.78 dvorak | **not recorded** — no single-session measurement and no protocol for this model in `PHASE_L.md` or `PHASE_M.md` (the only ms figure in either is the *pair's* 1.79 ms at `PHASE_M.md` §11.2, itself unprotocolled) | **5-seed mean 88.538 / 92.576 / 93.330 / 91.358 / 87.072** (3-seed was 88.54 / 92.600 / 93.33 / 91.35 / 87.08) | never decoded | dvorak **89.516** (app 89.092) · euro-mean **83.72 [derived** 83.742/82.344/80.509/88.294**]** | **⚠ 11/11 at three seeds → RETRACTED to 9/11 at five** (t3 −0.024, qwertz −0.156); per-seed [11, 8, 8, 6, 8] | 1234 / 4321 / 7777 / 5555 / 9999 | val-only; **claim retracted in place** — does not supersede `sw2345`. What survives: it clears **≤3 on a five-seed mean (91.358, +0.088)** | `PHASE_L.md` §15.4, §16; `PHASE_M.md` §7.1 |
| `phaseK-sw2345-slw2` (`phaseK_slw2_s1234.onnx`) | K | finalist recipe + `--short-loss-weight 2.0` (≤3-weighted CTC), ch 192, T′ 32 | not recorded | 6,068,519 / not measured / not measured | not recorded (arch unchanged) | **88.27 / 92.59 / 93.31 / 91.39 / 86.64** — **the only single model to clear the val ≤3 bar on EVERY seed** (91.47/91.38/91.32) | never decoded | dvorak **90.07** (app 89.68) · euro-mean **83.67 [derived** 83.92/82.87/80.26/87.62**]** | **7/11** (loses t1 −0.03, t3 −0.01, 4+ −0.13, spanish −0.66) | 1234 / 4321 / 7777 | val-only; the mirror-image counter-finalist to `sw2345` | `PHASE_K.md` §8.3, §7 |

---

## 4. The full historical ladder, by phase

Every arm the campaign ever trained, with at least its headline number. Read the
**footing banner** at the head of each phase before comparing anything across
phases: the decode preset changed once (published → E1, at Phase E), and that
change alone is worth **+2 to +5 pt** on every absolute number below it.

### 4.0 Pre-campaign (Campaign 1, 2026-08-07) — superseded

**Footing:** val-9918, **published `encoderOnly` preset**
(`0.4056 / 0.0176 / 0.9866 / 0.4234 / 1.0382`), 146,964-word trie, beam 100.
Absolute numbers here are understated by 2–5 pt against everything from Phase E
on (`RESULTS.md` §"⚠ Retraction — the old +0.21 pt maximum headroom claim").
No alt-layout evaluation existed yet.

| run | arch | params | bytes fp32 | laptop ms | val-9918 t1/t3/t5 | test-2400 | seeds | verdict / what it proved | source |
|---|---|---|---|---|---|---|---|---|---|
| `r1` | ch 96, cosine horizon 300, early-stopped @ epoch 93 | not recorded | not exported | — | 1,000-row probe only: 83.5 / 91.2 / 93.1; best val greedy **58.24** | — | 1 | under-annealed; superseded by `r2` at the same width | `RESULTS.md` §Campaign 1 §Runs |
| **`r2`** ← Campaign-1 ship candidate | ch 96, horizon 110 (fully annealed), residual trunk, v1 features | **394,114** | **1,619,140** (`ctc_swipe_encoder.onnx`, sha `fcf16331…`) | **0.306 / 0.318 p90** MEASURED on the Phase-F harness (`PHASE_F.md` §6) | **81.57 / 89.84 / 91.37**, greedy 58.57. At its own re-tuned preset: 86.14 / 91.01 / 92.12 / ≤3 89.94 / 4+ 84.16 (`PHASE_E.md` §1) | **80.96 / 89.79 / 91.12**, ≤3 85.89, 4+ 78.42 — a **disclosed prior contact**, logged under `test2400_seal.json` `prior_contact`, **not an unsealing** | 1 | the G2 feasibility pass; and the model whose wide re-sweep (**+4.25 pt on untouched val rows**) destroyed the old "+0.21 pt maximum headroom" bound | `RESULTS.md` §Campaign 1; `PHASE_E.md` §1 |
| `r3-ch128` | ch 128, horizon 110 | not recorded | not exported | — | 81.27 / 89.73 / 91.41, greedy **60.77** | — | 1 | **died**: +2.2 greedy did not survive the trie beam (−0.30 t1) — the campaign's first demonstration that emission sharpness ≠ decoded accuracy | `RESULTS.md` §Campaign 1 §Runs |
| `r2-refine` (G4 phase-2 head) | frozen per-frame refinement head on `r2` | **15.6 K** as recorded — `RESULTS.md` §Gates prints no exact count (the checkpoint holds 15,571) | `ctc_refine_head.onnx` **63,617 B** (measured by `stat` on `~/ctc-train/ckpt/r2-refine/`, 2026-08-15) | — | **+0.9 greedy, +0.0 beam** (28 fixed / 28 broken per 2,000 rows, both presets) | — | 1 | **died — phase 2 closed.** FUTO's +5.88 lever came off a 43.96 % greedy base; ours was at 58.6, so a per-frame head has nothing to fix | `RESULTS.md` §Campaign 1 §Gates |
| `refine-unfreeze` | same head, end-to-end `--unfreeze-after` fine-tune | — | not exported | — | +0.25 greedy, below threshold | — | 1 | **died** — same conclusion by the other route | same |
| `smoke` / `v2smoke` / `v2smoke0` / `phaseJ-smoke{1..4}` | toy / plumbing runs | — | — | — | see `PHASE_L.md` §2.3 for the `v2smoke` pair-coupling smoke (agreement 89.4 → 96.5 % coupled vs 82.0 → 91.4 % uncoupled at 800 steps) | — | — | **not results** — mechanism and plumbing checks only | `PHASE_L.md` §2.3 |

### 4.1 Phase A — the data-tier ladder

**Footing:** val-9918, published `enc` preset, beam 100, ~147 k trie. Recipe
frozen for all five arms: ch 96, embed_hid 96, batch 256, lr 3e-3, wd 0.01,
warmup 1 k, 47,000 steps, seed 1234, checkpoint = best val **greedy** (a metric
later retired for anti-correlating with the shipping metric).
**No arm in Phase A has a test-2400 number, an alt-layout number, a byte count
or a T′ statement** — `eval_arms.py` refuses any split whose filename contains
`test`. **≤3/4+ strata exist only for T2**, and only because `PHASE_B.md` §4–§5
re-swept its preset. `PHASE_A.md` itself records **no parameter count for any
arm**; T2's 394,114 comes from `PHASE_B.md` §1 / `PHASE_C.md` §7. The seed-4321
replicate's full-val t1/t3/t5 and greedy appear in **no `ctc/*.md` file** — they
are `phase_c_seed_results.json`, and the same is true of `phaseC-C2-s4321`
(`PHASE_C.md` §4 publishes only its clean and per-corpus columns, so do not read
that doc's clean t3 87.43 as this table's full-val t3).

| arm | tier / pool | params | laptop ms | val-9918 t1 / t3 / t5 | FUTO half / HWS half t1 | shared-clean (n=9,300) t1 | greedy | seeds | verdict | source |
|---|---|---|---|---|---|---|---|---|---|---|
| `phaseA-T0` | 109,600 rows, 55,438 HWS + 55,438 FUTO, **no contamination control** | not recorded | not measured | **82.123 / 90.270 / 91.601** | 88.001 / **76.286** | 81.796 | 58.42 | 1234 | the contaminated historical control; best HWS, worst FUTO of the ladder | `PHASE_A.md` §4 |
| `phaseA-T1` | 372,726 rows, HWS half + full curated FUTO, FUTO session exclusion | not recorded | not measured | **82.466 / 90.623 / 91.964** | 89.255 / 75.723 | 81.914 | **64.69** | 1234 | highest aggregate t1 in the ladder, but **not contributor-disjoint** (46 clean rows) and its +0.35 over T0 is inside the later-measured ~1 pt seed floor | `PHASE_A.md` §4 |
| `phaseA-T1strict` | 319,421 rows (888 HWS + 318,566 FUTO), FUTO session **+** HWS participant exclusion | not recorded | not measured | 80.672 / 88.607 / 90.411 | **91.238** / 70.177 | 79.892 | 61.09 | 1234 | **died** — fixes only the HWS side (clean FUTO stays at 43 rows); shows the corpus-mix effect — against T1 it is **−5.54 HWS for +1.98 FUTO**, against T0 **−6.11 for +3.24** | `PHASE_A.md` §4 |
| **`phaseA-T2`** (= Phase-B/C control **B0**) | 385,021 rows, FUTO-only, session exclusion + hygiene | **394,114** | **0.307 / 0.320 p90** MEASURED (`PHASE_C.md` §7); parity 2.48e-05, argmax 100/100 | 80.863 / 88.516 / 90.482; ≤3 **84.69** at the frozen preset (84.74 re-tuned, 4+ 78.97) | **92.594** (campaign best) / 69.212 | 79.989 | 59.62 | **1234, 4321** | the ladder's winner on the decision metric and the campaign's worst HWS model — "do not adopt T2 as-is" | `PHASE_A.md` §4; `PHASE_B.md` §5 (frozen-preset ≤3), §4 (re-tuned strata) |
| `phaseA-T2-s4321` | as T2, seed 4321 only | 394,114 | not separately measured | 79.814 / 88.435 / 90.250 | 91.137 / 68.569 | **78.935** | 60.67 | 4321 | **the run that set the campaign's noise floor** — 1.05 pt of clean-t1 spread from the seed alone, retroactively invalidating every sub-1-pt claim in A, B and C | `PHASE_C.md` §4 |
| `phaseA-T2b` | 285,929 rows, FUTO-only + **full quality cascade** (rejects 152,603 rows) | not recorded | not measured | 79.905 / 88.506 / 90.321 | 90.975 / 68.911 | 79.032 | 60.26 | 1234 | **died — the quality cascade buys negative accuracy**; the surviving evidence is the FUTO-half 1.71 pt (≈4 SE) | `PHASE_A.md` §4 |

### 4.2 Phase B — architecture levers on T2

**Footing:** as Phase A. Adoption rule: T2-clean t1 gain **> 0.6 pt** — a
threshold later shown to sit *below* the single-seed noise floor it was applied
to. All four arms export at opset 17 with zero `Einsum`/`BatchNormalization`.
No test-2400, no alt-layout, no byte counts, no ≤3/4+ except where shown.

| arm | arch | params | laptop ms | val-9918 t1/t3/t5 | Δ clean t1 vs B0 | greedy (Δ) | ≤3 / 4+ (re-tuned preset) | seeds | verdict | source |
|---|---|---|---|---|---|---|---|---|---|---|
| `phaseB-B1` | B0 trunk + `path_features_v2` + key-proximity channels | **405,716** | **0.332 / 0.345** (+8.5 % vs `r2`) — MEASURED **in Phase C** (`PHASE_C.md` §7); `PHASE_B.md` records no latency at all | 76.275 / 87.608 / 89.595 | **−4.60** | 60.00 (+0.38) | 74.68 / 78.66 (frozen-preset ≤3 collapses to **72.97**) | 1234 | **died, the clearest failure** — the proximity field hands the trunk a soft argmax over keys and short-circuits learning path dynamics | `PHASE_B.md` §2 (val), §4 (re-tuned strata), §5 (verdict); `PHASE_C.md` §7 (latency) |
| `phaseB-B2` | **ConvNeXt trunk, ch 128, 5 blocks, dilations {1,2,3,5,8}** | **570,818** | **0.453 / 0.468** (+48 %) — MEASURED in `PHASE_C.md` §7 | 79.482 / 88.677 / 90.421 | −1.31 (marginal) | **64.78 (+5.16)** | 81.00 / 79.68 | 1234 | **died as built, most informative arm in the phase** — largest greedy gain anywhere while beam accuracy falls; "not a failed idea, a mis-targeted one" | `PHASE_B.md` §2 (val), §4 (re-tuned strata), §4.1 (verdict); `PHASE_C.md` §7 (latency) |
| `phaseB-B3` | B1 features + B2 trunk | **585,940** | **0.475 / 0.489** (+55 %) — MEASURED in `PHASE_C.md` §7 | 78.050 / 88.445 / 90.532 | −2.68 | 62.92 (+3.30) | 77.87 / 79.63 | 1234 | **died** — sits between its parents on every metric; the two levers do not compose | `PHASE_B.md` §2 (val), §4 (re-tuned strata), §5 (verdict); `PHASE_C.md` §7 (latency) |

### 4.3 Phase C — training-procedure levers on T2

**Footing:** as Phase A. All four arms are **394,114 params and export
identically to `phaseA-T2`** — every Phase-C lever is training-time only, so
inference cost is unchanged. No test-2400, no alt-layout, no ≤3/4+.

| arm | change | laptop ms | val-9918 t1/t3/t5 | clean t1 (Δ vs B0) | HWS half t1 | greedy | seeds | verdict | source |
|---|---|---|---|---|---|---|---|---|---|
| `phaseC-C1` | path-only offset/scale jitter (σ_off 0.02, σ_scale 0.05), keys untouched | not measured — INHERITED from the identical export | 80.480 / 88.324 / 89.978 | 79.656 (−0.33) | 69.273 | 60.00 | 1234 | **died, and usefully** — designed to absorb the measured ~0.064 HWS-vs-FUTO y-offset, moved HWS by **+0.06**, killing the "registration problem" hypothesis | `PHASE_C.md` §2 (values), §3 (verdict) |
| `phaseC-C2` | EMA decay 0.999, evaluated and exported on the averaged weights | **0.308 / 0.319** (+0.7 %) MEASURED, and parity 2.29e-05 / argmax 100/100 — both `PHASE_C.md` §7 | 80.702 / 88.929 / 90.603 | 79.914 (−0.08) | **70.117 (+0.91)** | 60.32 | **1234, 4321** | **not adopted** — only HWS keeps its sign across seeds and its +0.57 two-seed mean sits under the +0.6 bar; free at inference | `PHASE_C.md` §4, §6 |
| `phaseC-C2-s4321` | as C2, seed 4321 | not measured | 80.046 / 88.052 / 90.038 | 79.237 | 68.790 | 59.87 | 4321 | the replicate that removed most of C2's claim (clean t3 +0.47 → −0.35) | `PHASE_C.md` §4 |
| `phaseC-C3` | C1 jitter + C2 EMA, batch 1024, lr 6e-3, 94 k steps (2× steps, 8× samples) | not measured — INHERITED | 80.470 / 88.294 / 89.917 | 79.527 (−0.46) | **68.629** (worst in phase) | **63.40** (best greedy in the campaign) | 1234 | **died** — 8× samples buys the sharpest emissions and the worst HWS number; second independent demonstration that sharpness ≠ decoded accuracy. Confounded (carries C1's jitter inseparably) | `PHASE_C.md` §2 (values), §5 (verdict) |

### 4.4 Phase D — capacity, tiers, selection

**Footing:** val-9918 at the **published `enc` preset**, ~147 k trie, beam 100,
`top_k=5`. Latency protocol: single-thread batch-1 ONNX CPU, 300 runs, idle.
**test-2400 was not decoded in Phase D.** No alt-layout, no byte counts, no T′.
Phase E later re-decoded the D1 checkpoints at E1; both footings are shown, and
**every E1 column below is `PHASE_E.md` §1, not `PHASE_D.md`.**
**Provenance of the latency column:** `PHASE_D.md` §5 publishes *ranges* only
(e.g. D0 "0.31–0.32 / 0.32–0.34"); the three-decimal mean/p90 pairs tabulated
here — and `phaseD-D1-last`'s per-corpus 92.43 / 76.41 — reproduce exactly from
`phase_d_results.json`, which is the actual source for those cells.

| arm | arch | params | laptop ms (MEASURED) | val-9918 t1/t3/t5/≤3/4+ (published preset) | same checkpoint at **E1** | FUTO / HWS t1 | seeds | verdict | source |
|---|---|---|---|---|---|---|---|---|---|
| `phaseD-D0` | ch 96, residual trunk, v1 features, tier T3 | **394,114** | 0.307 / 0.321 | **83.15 / 90.81 / 92.10 / 84.83 / 82.28** | — | 91.00 / 75.36 | 1234 | the T3 baseline at ch 96; lost to ch 128 by 1.07 and to its own arch on T1 by 0.89 | `PHASE_D.md` §3 |
| **`phaseD-D1`** | ch 128, embed_hid 128, residual trunk, T3 | **689,282** | 0.491 / 0.510 | **84.22 / 90.74 / 92.22 / 87.40 / 82.57** | **86.96 / 91.85 / 92.78 / 89.70 / 85.54** | 92.25 / 76.25 | 1234 | Phase D's winner; the base for the E1 preset sweep and the E2 refine head | `PHASE_D.md` §3; `PHASE_E.md` §1 |
| `phaseD-D1-s4321` | as D1 | 689,282 | 0.481 / 0.497 | 84.87 / 91.11 / 92.35 / 87.90 / 83.29 | 87.23 / 91.92 / 92.68 / 89.97 / 85.80 | 93.08 / 76.71 | 4321 | middle seed | `PHASE_D.md` §4 (published preset); `PHASE_E.md` §1 (E1 column) |
| `phaseD-D1-s7777` | as D1 | 689,282 | 0.515 / 0.564 | 85.34 / 91.18 / 92.42 / 88.73 / 83.58 | 87.45 / 92.03 / 92.91 / 90.06 / 86.09 | 93.28 / 77.45 | 7777 | explicitly flagged not-to-be-cherry-picked | `PHASE_D.md` §4 (published preset); `PHASE_E.md` §1 (E1 column) |
| **`phaseD-D1` seed-mean** | — | 689,282 | — | **84.81 (sd 0.56) / 91.01 / 92.33 / 88.01 / 83.15** | **87.21 / 91.93 / 92.79 / 89.91 / 85.81** — Δ vs the E-phase bar +1.69/+0.39/**−0.01**/+0.62/+2.24 | 92.87 / 76.80 | 3 | the Phase-D headline (84.81 vs FUTO ceiling 84.83); under E1 it clears **four of five** — t5 misses by 0.01, so the gate did not open on E1 alone | `PHASE_D.md` §4–§5; `PHASE_E.md` §1 |
| `phaseD-D1-last` | D1's final-step export | 689,282 | 0.486 / 0.501 | 84.39 / 91.14 / 92.45 / 86.46 / 83.32 | — | 92.43 / 76.41 | 1234 | proved the 2,000-row beam-selection rule cost **−0.17 pt** full-val t1 | `PHASE_D.md` §3 |
| `phaseD-D2` | Phase-B ConvNeXt trunk, ch 128, **5 blocks**, dil {1,2,3,5,8} | **570,818** | 0.477 / 0.491 | 83.35 / 90.98 / 92.20 / 84.60 / 82.71 | — | 91.42 / 75.34 | 1234 | **died** — −0.87 below D1 under beam selection, killing the "B2 was merely mis-selected" hypothesis | `PHASE_D.md` §3 |
| `phaseD-D3` | D1 + EMA 0.999, evaluated on averaged weights | **689,282** | 0.492 / 0.509 | 84.09 / 91.09 / 92.44 / 86.16 / 83.01 | — | 92.01 / 76.23 | 1234 | **died as a null** (−0.13 vs D1); not promoted to the seed round — "free at inference, never yet worth adopting" | `PHASE_D.md` §3 (values), §4 (verdict) |
| `phaseD-T1bridge` | ch 96 (D0 recipe) on tier **T1** (374 k rows) | **394,114** | 0.318 / 0.336 | 84.04 / 90.76 / 92.23 / **88.26** / 81.85 | — | 91.91 / 76.23 | 1234 | beat D0 by 0.89 → forced `T1bridge128` into existence; isolated **+1.57 pt of Phase-D gain as budget+selection, not data** | `PHASE_D.md` §3 |
| `phaseD-T1bridge128` | ch 128 on tier T1 | **689,282** | 0.490 / 0.507 | 84.29 / 90.90 / 92.13 / 88.02 / 82.36 | — | 91.30 / **77.33** | 1234 | the T1 side of the arch × tier 2×2 at matched capacity | `PHASE_D.md` §3 |
| `phaseD-T1bridge128-s4321` | as above | 689,282 | 0.495 / 0.513 | 84.55 / 90.84 / 92.28 / 88.34 / 82.59 | — | 91.58 / 77.57 | 4321 | — | `PHASE_D.md` §4 |
| `phaseD-T1bridge128-s7777` | as above | 689,282 | 0.486 / 0.502 | 84.28 / 90.64 / 91.90 / **88.91** / 81.88 | — | 91.68 / 76.93 | 7777 | — | `PHASE_D.md` §4 |
| **`phaseD-T1bridge128` seed-mean** | — | 689,282 | — | **84.38 (sd 0.15) / 90.79 / 92.10 / 88.42 / 82.27** | — | 91.52 / **77.28** | 3 | **the tier verdict**: paired T3 − T1 mean +0.43, t(2) = 1.31 against a 4.30 threshold → **T1 and T3 are indistinguishable at ch 128**; the §3 one-seed "T3 did not beat T1" claim was retracted | `PHASE_D.md` §4 |

### 4.5 Phase E — the preset retraction, the tier stack, and Campaign 2's finals

**Footing:** val-9918 at the **E1 preset** (`1.05 / 1.1 / 0.2 / 0.3734 /
0.9882`) — adopted in this phase and unchanged for the rest of the campaign.
Bar (FUTO ceiling, published preset, val): **85.52 / 91.54 / 92.80 / 89.29 /
83.57**. Latency protocol: 300 runs × 3 interleaved rounds, best round, idle.
No alt-layout evaluation existed yet, and **`PHASE_E.md` prints no byte count
anywhere** — the fp32 sizes below are `PHASE_F.md` §6.
**`PHASE_E.md` §0 and §5 state, twice and emphatically, that test-2400 was NOT
decoded in Phase E.** The first unsealing is a separate, post-phase act,
pre-registered in `AUDIT_PREDECODE.md` §E and reported in `RESULTS.md`; the
Phase-E checkpoints are its subject, not its author. Those numbers are in §2.

| arm | arch / change | params | bytes fp32 | laptop ms | val-9918 t1/t3/t5/≤3/4+ (E1) | FUTO / HWS t1 | seeds | verdict | source |
|---|---|---|---|---|---|---|---|---|---|
| `phaseE-E2-refine` | refinement head on the frozen `phaseD-D1` s1234 base, 30 k steps | not recorded | not exported | not recorded | at E1 **86.74 / 91.66 / 92.65 / 89.47 / 85.33**; at its own re-tuned preset 86.81 / 91.69 / 92.68 / 89.61 / 85.36 — base was 86.96 / 91.85 / 92.78 / 89.70 / 85.54 | — | 1 | **died** — negative on every metric under both presets, and never beat its own input on its own selection metric (86.86 vs 87.24). Phase-2's "+1.00 on ≤3" does **not** reproduce | `PHASE_E.md` §2 |
| **`phaseE-E5base`** | ch 128, T3, 94 k, only change vs D1 = **5,000-row** beam-selection prefix | 689,282 | not printed in `PHASE_E.md`; the export is the 2,799,865-byte ch-128 graph (§8.1) | **0.474 / 0.489** MEASURED (`PHASE_E.md` §4) | **87.19 / 92.09 / 92.87 / 90.06 / 85.71** | 94.80 / 79.64 | 1234 | **adopted** — +0.23 t1 over the 2,000-row rule, positive on all five, matching Phase D's advance prediction in size and sign | `PHASE_E.md` §3 |
| `phaseE-E4-ch192` | **ch 192**, embed_hid 192, plain T3, 94 k | **1,525,378** | not recorded | **0.898 / 0.914** MEASURED idle (an earlier 1.54 ms reading is **withdrawn** — measured under load) | **87.67 / 92.11 / 92.90 / 90.35 / 86.28** | 95.22 / 80.16 | 1234 | +0.48 over E5base at one seed — inside the noise floor, so sent to the seed round rather than adopted; the +0.48 later became **+0.19 at three paired seeds** | `PHASE_E.md` §3–§4 |
| `phaseE-E4-ch192-last` | E4's final-step export | 1,525,378 | — | — | 87.65 (vs the selected checkpoint's 87.67) | — | — | **not an arm** — the selected-vs-final checkpoint check | `PHASE_E.md` §4 |
| `phaseE-E4-ch192-s4321` | hedge run | — | — | — | — | — | — | **not a result** — deliberately killed at step 9,000 of 94,000; never evaluated | `PHASE_E.md` §7 |
| `phaseE-E3a-T4` | ch 128 on **tier T4** (dedup-only policy on T1's curated FUTO pool; 764,771 rows) | 689,282 | not printed in `PHASE_E.md`; the export is the 2,799,865-byte ch-128 graph (§8.1) | not recorded | **86.93 / 91.58 / 92.47 / 90.00 / 85.34** | 94.80 / 79.12 | 1234 | **died** — −0.26 vs E5base, negative on all five; the third independent negative for the quality-cascade family | `PHASE_E.md` §4 |
| **`phaseE-E3b-hws3x`** (= `ch128_s1234`) | ch 128 on T3 with a 76,748-row HWS cache concatenated **twice** → 1,158,832 rows | **689,282** | **2,799,865** | **0.470 / 0.485** MEASURED | **88.02 / 92.27 / 93.03 / 91.12 / 86.41** | 95.00 / **81.09** | 1234 (+2 more as the ch-128 control) | **adopted** — +0.83 t1, positive on every metric and both strata; **HWS +1.45 vs FUTO +0.20** — the gain lands on the half that was 15 pt behind. Disclosure: 3× oversampling triples exposure of the more contaminated corpus, so +1.45 is an upper bound | `PHASE_E.md` §4 |
| **`phaseE-FINAL-s1234/-s4321/-s7777`** (= ch 192) | E4 arch on the E3b tier, 94 k, E5 selection | **1,525,378** | **6,144,249** | **0.898 ms** — INHERITED from the E4 measurement (same graph) | s1234 88.22 / 92.23 / 93.08 / 91.15 / 86.71 · s4321 87.80 / 92.34 / 93.17 / 90.62 / 86.34 · s7777 88.17 / 92.38 / 92.99 / 90.82 / 86.80 → **seed-mean 88.06 / 92.32 / 93.08 / 90.86 / 86.62**, all five PASS, and all five PASS again on the untouched holdout half (87.58 / 92.03 / 92.85 / 90.67 / 85.98, t5 only **+0.05**) | 94.99 / 81.19 | 1234 / 4321 / 7777 | the Campaign-2 headline; its checkpoints were **test-validated at the first unsealing** (§2) — an act registered by `AUDIT_PREDECODE.md` §E and reported in `RESULTS.md`, not performed in Phase E | `PHASE_E.md` §5 (val); `PHASE_F.md` §6 (bytes); `RESULTS.md` §"Verified test-2400 results" (test) |
| ch 128 control, 3 seeds (s1234 member = `phaseE-E3b-hws3x`) | ch 128 on the FINAL tier | 689,282 | 2,799,865 | 0.470 ms | **seed-mean 87.88 / 92.23 / 92.96 / 90.98 / 86.26** — all five PASS. Paired ch192 − ch128: t1 **+0.19** (t(2)=21.2), t3 +0.09, t5 +0.12, **≤3 −0.12**, 4+ +0.35 | — | 3 | named "the better shipping trade" — same gate at **half the latency and 2.2× fewer params**, and *ahead* on ≤3; became Campaign-2's ship pick | `PHASE_E.md` §5 |

---

### 4.6 Phase F — the latency frontier (the `resbn` family)

**Footing:** val-9918 at E1, AOSP STRIP **146,964**, beam 100, through the
exported ONNX. Bar: 85.52 / 91.54 / 92.80 / 89.29 / 83.57.
**Latency protocol (§0, the campaign's protocol of record from here on):**
`bench_latency.py`, ORT `CPUExecutionProvider`, `intra_op = inter_op = 1`,
batch 1, fixed shapes, 50 warmup + 3 rounds × 300 timed calls, mean/p90 of the
best round, machine idle. The harness reads **~3 % high** against
`AUDIT_PREDECODE.md` §7 — use the Phase-F column for *ratios*.
**Every Phase-F absolute number should be read as ~0.5 t1 understated**: every
arm carried the KD handicap that Phase G later measured as negative, and that
is common-mode, so arm-vs-arm conclusions survive (`MODEL_COMPARISON.md` §6.7).
**Alt-layout: not measured for any Phase-F arm** except `fast_resbn80_s1234`
(in `ALT_LAYOUT_EVAL.md`). **test-2400: only `fast_resbn80`** (second
unsealing, §2). All arms are seed 1234 unless a seed list is given.
Shared recipe: T3 + 3× HWS, 94 k steps (unless a step count is shown), batch
256, lr 3e-3, wd 0.01, warmup 1 k, `embed_hid` 96, KD from
`phaseE-FINAL-s1234` at weight 1.0 / temp 2.0, 5,000-row beam-t1 selection.

**Instrument and quantization arms (not new trainings):**

| arm | bytes | laptop ms | val-9918 t1/t3/t5/≤3/4+ | bars | verdict | source |
|---|---|---|---|---|---|---|
| no-op graph carrying the production I/O signature | — | **0.007 / 0.007** | n/a | — | the harness floor: every figure below is graph work, not instrument overhead | `PHASE_F.md` §0 |
| ch 128 **int8 dynamic** | 2,737,242 | 0.471 / 0.487 | not measured | — | −0.8 % latency; ORT dynamic only quantizes MatMul/Gemm on a 66 %-Conv graph | `PHASE_F.md` §2.1 |
| ch 128 **int8 static, whole graph** | 863,996 | 0.269 / 0.282 | **0.00 ×5** | 0/5 | **died structurally** — `MASK_NEG = −1e4` + in-graph log-softmax means a uint8 step ≈ 39 nats swallows every real log-prob | `PHASE_F.md` §2.2 |
| ch 128 int8 static, tail fp32 | 918,415 | 0.279 / 0.294 | 86.12 / 91.53 / 92.61 / 89.08 / 84.58 | 3/5 | −1.90 t1 | `PHASE_F.md` §2.2 |
| ch 128 int8 static, tail+norms+stem fp32 (best exclusion set found) | 905,293 | 0.273 / 0.285 | 87.04 / 91.98 / **92.72** / 90.09 / 85.46 | 4/5 (t5) | −0.98 t1 — still not free | `PHASE_F.md` §2.2, §6 |
| ch 192 int8 dynamic | 6,043,994 | 0.906 / 0.927 | not measured | — | −3.0 % latency | `PHASE_F.md` §2.1 |
| ch 192 int8 static, whole graph | 1,710,334 | 0.428 / 0.445 | **0.00 ×5** | 0/5 | same structural failure | `PHASE_F.md` §2 |
| ch 192 int8 static, tail fp32 | 1,801,552 | 0.439 / 0.457 | 86.77 / 91.71 / **92.62** / 89.32 / 85.45 | 4/5 (t5) | — | `PHASE_F.md` §2 |
| `resbn:56` **int8** (arm G) | 290,872 | 0.126 / 0.136 | 84.77 / 90.99 / 92.05 / 87.49 / 83.35 | **0/5** | **F3 is the worst family in the phase** — Δ t1 −1.48 | `PHASE_F.md` §5 |
| `resbn:80` **int8** (arm I) | 434,986 | 0.146 / 0.155 | 86.38 / 91.63 / 92.50 / 88.70 / 85.17 | 3/5 | Δ t1 −1.03; the int8 accuracy cost does not shrink with the model while the latency benefit vanishes | `PHASE_F.md` §5 |
| ORT offline graph-optimization serialization (`--optimize-out cache/ch128_ortopt.onnx`) | byte count **not recorded** in Phase F (`PHASE_I.md` §3 later measured the same lever on **ch 192**: 6,132,556 vs 6,144,249 B, −0.2 %) | **no runtime effect** — means identical inside noise | — | — | **not a size or speed lever** — it moves session-load work only | `PHASE_F.md` §1; `PHASE_I.md` §3 |

**Trained arms:**

| arm (run dir) | arch | params | bytes fp32 | laptop ms | val-9918 t1/t3/t5/≤3/4+ | bars | seeds | verdict | source |
|---|---|---|---|---|---|---|---|---|---|
| `phaseF-A-resbn64x3` (arm A) | `resbn:64:1,2,4` @94 k | **143,714** | 600,196 | **0.134 / 0.141** | 85.89 / 91.48 / 92.50 / 88.76 / 84.41 | 2/5 | 1234 | **died** — width at 3 blocks loses to depth at 4 | `PHASE_F.md` §4, §6 |
| `phaseF-B-resbn48x4` (arm B) | `resbn:48:1,2,4,8` @94 k | **111,250** | 472,645 | **0.122 / 0.130** | 86.39 / 91.41 / 92.38 / 89.82 / 84.61 (own-preset re-sweep buys +0.12: 86.51 / 91.46 / 92.41 / 89.88 / 84.76) | 3/5 | 1234 | best sub-0.13 ms model; **proves depth > width** (+0.50 over A at 0.012 ms less); dominates the old `r2` | `PHASE_F.md` §4, §6 |
| `phaseF-C-dwsep128x4` (arm C) | `dwsep:128:1,2,4,8` @94 k | **97,826** | 413,034 | **0.142 / 0.150** | 85.78 / 91.39 / 92.27 / 88.40 / 84.42 | 2/5 | 1234 | **died — refutes the brief's depthwise-separable hypothesis**: −0.61 t1 vs B at +0.020 ms (dense throughput ~68 GMAC/s vs separable ~38) | `PHASE_F.md` §4, §10 |
| `phaseF-G-resbn56x4` (arm G) | `resbn:56:1,2,4,8` @94 k | **145,594** | 609,445 | **0.141 / 0.150** | 86.25 / 91.67 / **92.61** / 89.52 / 84.55 | 4/5 (t5 −0.19) | 1234 | the largest `resbn` inside the ≤0.15 ms target — and it misses t5 | `PHASE_F.md` §4, §6 |
| `phaseF-D-resbn64x4` + `phaseF-FAST-resbn64x4-s{4321,7777}` (arm D / FAST) | `resbn:64:1,2,4,8` @94 k | **185,058** | 766,727 | **0.162 / 0.172** | s1234 86.70 / 91.84 / **92.78** / 89.44 / 85.28; **3-seed mean 86.82 / 91.85 / 92.67 / 89.86 / 85.24** | **4/5 — t5 −0.13, no seed clears** | 1234 / 4321 / 7777 | the knife edge the phase resolved by seeding — the reason 0.162 ms is not the frontier | `PHASE_F.md` §4, §8 |
| `phaseF-I-resbn80x4` + `phaseF-FINAL-resbn80x4-s{4321,7777}` = **`fast_resbn80`** | `resbn:80:1,2,4,8` @94 k | **279,346** | 1,142,727 (int8 tail+stem-fp32 434,986) | **0.215 / 0.224** — §8's heading reads 0.215; the 0.213 figure appears once in the document, at §14.1, as a parenthetical claim *about* §8 | seed-mean **87.47 / 92.13 / 92.89 / 90.35 / 85.98** — 5/5 every seed, worst-seed t5 **+0.05**; app trie seed-mean 86.93 / 92.39 / 93.51 / 90.51 / 85.07, 5/5 every seed | **5/5** | 1234 / 4321 / 7777 | Phase F's conservative pick; **test-validated at the second unsealing** (§2), then superseded by `resbn80g` at identical cost | `PHASE_F.md` §8, §15.3, §16.5 |
| `phaseF-L48x5-188k` | `resbn:48:1,2,4,8,16` @188 k (5 narrow blocks) | **134,578** | 567,368 | **0.141 / 0.148** (§13 prints 0.139) | 86.64 / 91.80 / **92.53** / 89.73 / 85.04 | 4/5 | 1234 | extra depth past 4 blocks does not buy t5 | `PHASE_F.md` §13, §6 |
| `phaseF-L56-188k` (`fast_resbn56_188k_s1234.onnx`) | `resbn:56:1,2,4,8` @188 k | **145,594** | 609,445 | 0.144 (§13) / **0.142 (§9 artifact table** — §6's 0.142 row is the @280 k twin, not this arm**)** | **86.79 / 91.83 / 92.65 / 90.26 / 84.99**; Δ over 94 k = +0.54 / +0.16 / **+0.04** / +0.74 / +0.44 | 4/5 | 1234 | published as **frontier evidence, not a ship candidate** | `PHASE_F.md` §13, §9 |
| `phaseF-M56-280k` (arm M) — best ≤0.15 ms | `resbn:56:1,2,4,8` @**280 k** | 145,594 | 609,445 | **0.142 / 0.149** | **86.83 / 91.85 / 92.67 / 90.23 / 85.07**; Δ 94 k → 280 k = +0.58 / +0.18 / **+0.06 t5** against a 0.19 deficit | 4/5 | 1234 | the "taken to exhaustion" arm — 3× schedule buys ¼ of what 2× did → **the constraint is capacity, not optimization** | `PHASE_F.md` §13.1 |
| `phaseF-P56-188k-T4` | `resbn:56:1,2,4,8` @188 k, **KD temperature 4** | 145,594 | 609,445 | not separately measured (same graph) | 86.20 / 91.66 / **92.45** / 90.38 / 84.03 — Δ vs T=2 **−0.59 / −0.17 / −0.20 / +0.12 / −0.96** | — | 1234 | **died** — negative on 4 of 5 including the metric it targeted. Confounded: T² scaling quadruples the effective KD weight | `PHASE_F.md` §13.2 |
| `phaseF-L64-188k` (`fast_resbn64_188k_s1234.onnx`) | `resbn:64:1,2,4,8` @188 k | **185,058** | 766,727 | 0.161 (§13) / **0.162 (§9 artifact table** — §6's 0.162 row is arm D/FAST @94 k, a different arm**)** | **87.19 / 92.09 / 92.76 / 90.29 / 85.59**; Δ over 94 k = +0.49 / +0.25 / **−0.02 t5** / +0.85 / +0.31 | 4/5 | 1234 | frontier evidence only | `PHASE_F.md` §13, §9 |
| `phaseF-O56x5-188k` | `resbn:56:1,2,4,8,16` @188 k | **177,290** | **720,000 as printed** in `PHASE_F.md` §6 — a suspiciously round figure; the exported graph at `~/ctc-train/ckpt/phaseF-O56x5-188k/` measures **737,512 B** (`stat`, 2026-08-15) | **0.166** (§14) / 0.165 / 0.175 (§6) | 87.07 / 91.92 / **92.74** / 90.20 / 85.45 | 4/5 (t5 −0.06) | 1234 | depth does not substitute for width past 4 blocks — 4×(1+2+4+8) = 60 frames already covers T = 32 | `PHASE_F.md` §14, §6 |
| `phaseF-Q68-188k` | `resbn:68:1,2,4,8` @188 k | **206,710** | **852,927 as printed** (`PHASE_F.md` §6); the exported graph measures **853,047 B** (`stat`, 2026-08-15) | **0.176 / 0.185** | 87.27 / 91.98 / **92.74** / 90.20 / 85.74 | 4/5 (t5 −0.06) | 1234 | the lower probe that pins the t5 crossing to **207 k – 230 k params / 0.176 – 0.186 ms** | `PHASE_F.md` §14 |
| `phaseF-N72-188k` + `-s4321` + `-s7777` = **`fast_resbn72`** | `resbn:72:1,2,4,8` @188 k | **229,642** | 944,487 | **0.186 / 0.195** (§14 prints 0.185) | **87.27 / 92.09 / 92.87 / 90.49 / 85.60** — 5/5 every seed | **5/5** | 1234 / 4321 / 7777 | the fastest 5/5 configuration ever measured; **val-only permanently** | `PHASE_F.md` §14.1, §9 |
| `resbn:40:1,2,4,8,1,2` (6 blocks, ch 40) | — | — | — | — | — | — | — | **never trained** — killed for GPU budget; §14's depth finding says it would not have paid | `PHASE_F.md` §7.1 |
| the no-KD ablation at the final architecture | — | — | — | — | — | — | — | **never completed in Phase F** — named "the largest remaining hole in the evidence"; Phase G closed it, and the answer was that KD had been costing ~0.5 t1 | `PHASE_F.md` §7.1, §11.3 |

**The t5-vs-capacity curve — the phase's whole story** (`PHASE_F.md` §14):
134.6 k → 92.53 · 145.6 k → 92.65 · 177.3 k → 92.74 · 185.1 k → 92.76 ·
206.7 k → 92.74 · 229.6 k → **92.96** · 279.3 k → 92.85 · 689.3 k → 93.03.
It crosses the 92.80 bar at **210–230 k parameters**, and ~0.10 ms of any
budget is spent before the first trunk block on work the I/O contract fixes.

**Random-init architecture pricing** (`arch_latency.py`, `PHASE_F.md` §3 — these
are latency-only measurements on untrained graphs, no accuracy attaches to
them): `res:128:1,2,4,8` 685,090 p / 0.468 ms · `dwsep:96:1,2,4` 53,986 /
**0.100** · `dwsep:96:1,2,4,8` 64,258 / 0.114 · `dwsep:112:1,2,4,8` 80,018 /
0.129 · `dwsep:128:1,2,4,8` 97,826 / 0.141 · `dwsep:160:1,2,4` 112,226 / 0.143 ·
`dwsep:144:1,2,4,8` 117,682 / 0.156 · `dwsep:160:1,2,4,8` 139,586 / 0.174 ·
`dwsep:128:1,2,4,8,1,2,4,8` 168,994 / 0.228 · `dwsep:192:1,2,4,8` 189,538 /
0.208 · `resbn:48:1,2,4,8` 111,250 / 0.120 · `resbn:64:1,2,4` 143,714 / 0.135 ·
`resbn:64:1,2,4,8` 185,058 / 0.160 · `resbn:80:1,2,4` 214,866 / 0.175 ·
`resbn:80:1,2,4,8` 279,346 / 0.210 · `resbn:96:1,2,4,8` 394,114 / 0.267.
`embed_hid` sub-arms on `dwsep:128:1,2,4,8`: 96 → 0.141, 64 → 0.138,
48 → 0.134 — never spent on a training run.

> **Known source discrepancy, carried not resolved:** `PHASE_F.md` prints ch 128's
> parameter count as **689,282** (trained, §4/§6) and **685,090** (the §3
> random-init spec row). `PHASE_I.md` §5 also gives 685,090 for its own
> `phaseI-ch128`. Both are printed; this file uses 689,282 for the Phase-E
> ch 128 artifact and 685,090 for `phaseI-ch128`, as each document does.

### 4.7 Phase G — the recipe upgrade (KD removed, affine sampler fixed)

**Footing:** as Phase F (val-9918, E1, AOSP STRIP 146,964). Bars unchanged.
Recipe for every arm: `resbn:{ch}:1,2,4,8`, embed_hid 96, 188 k steps,
batch 256, lr 3e-3, wd 0.01, warmup 1 k, T3 + 3× HWS, 5,000-row beam-t1
selection. **All single-seed (s1234) unless noted. No Phase-G arm was ever
alt-layout evaluated.**

| arm | sampler | KD | params | bytes fp32 | laptop ms | val-9918 t1/t3/t5/≤3/4+ | seeds | verdict | source |
|---|---|---|---|---|---|---|---|---|---|
| `phaseG-A80-188k-legacy` | legacy | on (ch 192 s1234, w 1.0, T 2) | 279,346 (class) | 1,142,727 (class) | INHERITED 0.215 class | **87.46 / 92.28 / 92.95 / 90.76 / 85.74** | 1234 | isolates the 188 k schedule with KD on — worth only **+0.05 t1**; Phase F's +0.5 at ch 56/64 does not transfer to ch 80 | `PHASE_G.md` §3 |
| `phaseG-B80-188k` | **coupled** | on | 279,346 | 1,142,727 | INHERITED | **87.52 / 92.15 / 92.76 / 91.03 / 85.69** | 1234 | B−A isolates the affine fix with KD on: +0.06 t1, mixed elsewhere | `PHASE_G.md` §3 |
| **`phaseG-C80-188k-nokd` = `resbn80g`** | coupled | **off** | **279,346** | **1,142,727** | **0.213** MEASURED | s1234 **88.04 / 92.39 / 93.18 / 91.30 / 86.35**; 3-seed mean **87.72 / 92.25 / 92.97 / 90.78 / 86.14** (worst seed 87.31 / 92.09 / 92.83 / 89.94 / 85.94 — all five PASS) | 1234 / 4321 / 7777 | the winner; **test-validated** — see §2 | `PHASE_G.md` §3–§4, §7.5 |
| `phaseG-D80-188k-kdens` | coupled | **3-seed ch 192 ensemble teacher** | 279,346 | 1,142,727 | not recorded (~2× training cost) | **87.07 / 91.97 / 92.80 / 91.18 / 84.94** | 1234 | **died** — the ensemble teacher is *worse* than the single one (−0.45 t1, −0.18 t3, −0.75 4+) at ~2× the per-step GPU cost | `PHASE_G.md` §3 |
| `phaseG-E80-188k-legacy-nokd` | legacy | off | 279,346 | 1,142,727 | not recorded | **87.94 / 92.33 / 92.98 / 91.12 / 86.29** | 1234 | the 2×2's fourth cell — confirms KD removal is +0.48 with the legacy sampler too, and the affine fix is +0.10 without KD | `PHASE_G.md` §3 |
| `phaseG-F72-188k-nokd` (+ s4321, s7777) = **`resbn72g`** | coupled | off | **229,642** | 944,487 | **0.184** MEASURED | 3-seed mean **87.62 / 92.22 / 93.02 / 90.48 / 86.14** — 5/5 every seed | 1234 / 4321 / 7777 | the val-only latency frontier — see §3 | `PHASE_G.md` §8.1 |
| `phaseG-H64-188k-nokd` = `resbn64g` | coupled | off | **185,058** | 766,727 | **0.161** MEASURED | **87.17 / 91.83 / 92.70 / 90.12 / 85.65** | 1234 | **died on t5 (−0.10)** — the no-KD gain does *not* transfer to ch 64 (−0.02 / −0.26 / −0.06 vs its KD twin `phaseF-L64-188k`): **KD's harm is capacity-dependent**, and Phase F's "≤0.15 ms is unreachable with the bar intact" verdict stands under the upgraded recipe | `PHASE_G.md` §8.3 |

**The 2×2 factorial, in one place** (`PHASE_G.md` §3.2, ch 80 / 188 k / s1234 /
full val / E1): legacy+KD **A 87.46** · legacy−KD **E 87.94** (+0.48) ·
coupled+KD **B 87.52** · coupled−KD **C 88.04** (+0.52); sampler Δ +0.06 with
KD, +0.10 without. **KD removal is the dominant lever** and it resolves Phase F's
largest stated evidence hole in the direction nobody assumed: the ch 192 teacher
was *capping* the ch 80 student. The affine fix is a distribution repair, not a
range extension — the legacy sampler rejected 31.4 % of first draws and biased
sx toward compression (mean 0.9554); the coupled one accepts 100 % and realizes
`sx ~ U(0.85, 1.1111)` exactly.

**Levers excluded in Phase G with the measurement that killed them**
(`PHASE_G.md` §2): KD temp 4 (−0.59 t1), the 280 k schedule (+0.02 t5 for +50 %
GPU), post-training int8 (loses t5 at every size), the depthwise-separable trunk
(−0.61 t1 at higher latency), feature v2 / EMA / path-only jitter
(null-or-negative), 5-block trunks.

### 4.8 Phase H — layout-resampling augmentation

**Footing:** val-9918 at E1 / AOSP for the en column; alt-layout is `az26`
in-dict at E1, all rows of each corpus, with `dvorak-app` on the 98,081 app
trie. Recipe = `phaseG-C80-188k-nokd` verbatim plus `--layout-alt-p`; geometry
source on re-target is 2/3 synthetic random-permutation lattices + 1/3 real
azerty/qwertz/german/spanish, with **dvorak held out of training as a true
transfer probe**. Warp cost 0.21 ms/item at p 0.5. Every arm's exported graph
is **node-for-node identical to `resbn80g`** (231 nodes) — 279,346 params,
1,142,727 B. **test-2400 was not read in Phase H.**

| arm | dose | laptop ms | val-9918 t1/t3/t5/≤3/4+ | dvorak (app-98k) | azerty / qwertz / german / spanish | euro-mean | seeds | verdict | source |
|---|---|---|---|---|---|---|---|---|---|
| `phaseH-p15` | p 0.15 | INHERITED (identical graph) | 87.31 / 92.29 / 93.00 / 90.82 / 85.48 | **86.94** (85.75) | 82.49 / 82.81 / 79.08 / 85.55 | **82.48 [derived]** | 1234 | already +19.7 dvorak over `fast_resbn80`; dominated by p 0.5 on every column | `PHASE_H.md` §5 |
| `phaseH-p30` | p 0.30 | INHERITED | 87.57 / 92.14 / 93.01 / 90.68 / 85.95 | **88.36** (86.28) | 83.01 / 82.65 / 80.31 / **88.74** | **83.68 [derived]** | 1234 | middle of a monotone dose-response; loses to p 0.5 everywhere except spanish (−0.23, inside noise) | `PHASE_H.md` §5 |
| **`phaseH-p50` = `resbn80h`** | p 0.50 | **0.216 / 0.229** MEASURED, paired against `resbn80g`'s 0.212 / 0.222 | 3-seed **87.69 / 92.22 / 93.00 / 90.79 / 86.08** | **90.01** (**89.51**) | 84.27 / 84.36 / 81.13 / 88.43 | **84.55 [derived]** | 1234 / 4321 / 7777 | the winner — see §3. Dvorak greedy 11.6 → 42.5; the CTC model now beats the geometric engine on **all six** layouts, obsoleting the mean-key-displacement routing gate | `PHASE_H.md` §5–§6 |

Also measured and **rejected** in Phase H: a world-frame residual warp variant
(+0.05/+0.03 endpoint hit on dvorak, rejected as behaviourally wrong);
inverse-distance / thin-plate fields (rejected at design). Warp exactness
self-tests: identity max |Δ| = 0.0, ideal→ideal max distance 3.9e-08, 200
synthetic geometries all-26-distinct and contained (`PHASE_H.md` §2).

---

### 4.9 Phase I — capacity (I-A), data curation and Cyrillic (I-B)

**Footing:** val-9918 at E1 / AOSP STRIP **146,964**; alt-layout `az26` in-dict
at E1; `dvorak app-98k` on the 98,081 app trie. Latency = the `PHASE_F.md` §0
idle-box protocol, re-anchored on this instrument. **The latency constraint was
retired in this phase** (capacity is bounded by **size ≤ 5 MB**, not speed).
**test-2400 was not read anywhere in Phase I or I-B.**

**I-A — the capacity ladder** (all: embed_hid 96, T′ 32, 188 k steps, batch 256,
lr 3e-3, wd 0.01, warmup 1 k, coupled affine, no KD, 5,000-row beam-t1
selection; tier = T3 + 3× full-release HWS, 1,158,832 rows):

| arm | ch · dose | params | bytes fp32 / fp16w / int8 | laptop ms | val-9918 t1/t3/t5/≤3/4+ | dvorak (app) / euro-mean | seeds | verdict | source |
|---|---|---|---|---|---|---|---|---|---|
| `phaseH-p50` (`resbn80h`, the baseline) | 80 · p 0.50 | 279,346 | 1,142,727 / **589,406** / **317,476** | **0.213 / 0.223** MEASURED | s1234 87.66 / 92.24 / 93.05 / 90.88 / 85.99; 3-seed 87.69 / 92.22 / 93.00 / 90.79 / 86.08 | 90.01 (89.51) / 84.55 [derived] | 3 | the transfer champion at 1/5 the capacity — the rung every other is read against | `PHASE_I.md` §2, §5, §7.2 |
| `phaseI-ch128` | 128 · p 0.50 | **685,090** | 2,762,279 / 1,399,197 / not recorded | **0.423 / 0.439** MEASURED | **88.08 / 92.44 / 93.20 / 91.18 / 86.48**, greedy 70.6 | 89.70 (88.77) / **82.60 [derived** 82.63/81.21/80.04/86.52**]** | 1234 | the middle rung; priced but never the pick | `PHASE_I.md` §5, §8 |
| `phaseI-ch192` | 192 · **p 0.50** | 1,512,802 | 6,068,519 / — / — | INHERITED from the p 0.65 twin (identical arch) | **88.22 / 92.45 / 93.21 / 91.00 / 86.78**, greedy 71.8 | **85.43** (84.78) / **83.07 [derived** 83.25/81.55/79.08/88.40**]** | 1234 | **died on transfer** — at p 0.50 the dvorak axis breaks at ch 192 (85.43 vs ch 80's 88.85): 5.4× params re-learns QWERTY shortcuts faster than 50 % resampled geometry regularizes them away | `PHASE_I.md` §5 |
| **`phaseI-ch192-p65` = `resbn192i`** | 192 · **p 0.65** | 1,512,802 | 6,068,519 / **3,052,318** / — | **0.831 / 0.849** MEASURED (fp16w identical) | **88.30 / 92.60 / 93.26 / 91.27 / 86.77** — see §3 | 89.13 (88.20) / 83.51 [derived] | 3 | **the governing law of the phase: capacity converts to accuracy, but the augmentation dose must scale with it.** Beats its p 0.50 twin on **all eleven** measured columns | `PHASE_I.md` §5, §7.2 |
| `phaseI-ch256` = `resbn256i` | 256 · p 0.50 | **2,668,194** | 10,685,479 / 5,360,800 / **int8-trunk 2,737,114** | **1.372 / 1.389**; int8-trunk 1.540 | **88.65 / 92.61 / 93.32 / 91.26 / 87.29** — see §3 | 86.92 (86.65) / 82.29 [derived, s1234] | 3 | the QWERTY frontier, transfer-volatile at its unscaled dose; **p 0.65 was never run at this width** | `PHASE_I.md` §7.1 |
| `phaseI-t64-80` | 80 · p 0.50, **T′ = 64** (`--t-out 64`, stride-1 stem) | not recorded | **1,150,923** (`stat` on `~/ctc-train/ckpt/phaseI-t64-80/`, 2026-08-15; not printed in `PHASE_I.md`) | encoder not recorded; **beam cost ~2×** measured (20 vs 30 traces/s) | **87.85 / 92.36 / 93.22 / 90.79 / 86.32** — Δ vs `phaseH-p50` +0.19/+0.12/+0.17/−0.09/**+0.33** | **91.62** (90.68) / **85.22 [derived** 86.12/85.00/81.26/88.51**]** | 1234 | a real lever (+0.33 on the binding 4+ stratum, +2.5–2.8 transfer) that **breaks the frozen `[1,32,·]` I/O contract** → reported as an **app decision**, not adopted | `PHASE_I.md` §6.1 |
| `phaseI-sel80` | 80, multi-layout checkpoint selection (`--select-layout-probes synth:101,synth:202,azerty`) | not recorded (arch = `resbn80h`) | not recorded | not recorded | **absolute values not recorded** — only Δ vs `phaseH-p50`: **+0.21 t1 / −0.06 t5** | Δ dvorak +1.71, dvorak-app +2.11, azerty +0.76, spanish +0.74, qwertz 0.00, german −1.10 (run-noise confounded) | 1234 | consistent small positive on transfer at canonical-neutral cost — **kept available in `train.py`, not made default** | `PHASE_I.md` §6.2 |

Storage-variant findings, measured on real graphs (`PHASE_I.md` §2): **fp16w is
free at every width**; **int8w penalty is width-dependent** — −0.18 t1 at ch 80,
−0.20 at ch 192, **0.00 at ch 256**; int8 trunk-only does not improve stress
parity (the sensitivity is in the trunk convs, not the head); a full-fp16
*compute* graph (594,123 B) **fails to load** on the CPU EP (MatMul type clash
at the mask/`Where` boundary) and is not even smaller than fp16w — a dead end.

**I-B — HWS filtering arms.** All four share the frozen Phase-G/H recipe at
ch 80, p 0.50, 188 k, **seed 1234, single seed each**, from a worktree pinned at
`d7faa75`. **≤3/4+ strata, params, bytes, latency and test-2400 are not
recorded for any I-B arm.**

| arm | HWS pool | train rows | val t1/t3/t5 | FUTO / HWS half | dvorak (app) | azerty / qwertz / german / spanish | verdict | source |
|---|---|---|---|---|---|---|---|---|
| control (= `phaseH-p50`) | all 1,338 users, hygiene only | 1,158,832 | 87.66 / 92.24 / 93.05 | 94.27 / **81.09** | 88.85 (88.20) | 83.64 / **84.16** / **81.45** / **88.51** | the reference | `PHASE_I_DATA.md` §3 |
| `phaseIB-quality` | all levels + motion gates (~1.9 % trimmed) | 1,154,426 | **87.71 / 92.29 / 93.11** | **94.58** / 80.89 | 90.84 (89.99) | 84.74 / 83.49 / 80.99 / 86.18 | a **statistical tie** with a mildly positive point estimate (+0.05 val, +4.2 beginner at SE ≈ 3.6) — acceptable drop-in, adopt only on a rebuild | `PHASE_I_DATA.md` §3 |
| `phaseIB-nativeadv` | englishLevel ∈ {native, advanced}, 755 users | 1,067,054 | 87.30 / 92.12 / 92.99 | 94.39 / 80.25 | **91.66** (**90.72**) | 83.11 / 82.48 / 80.04 / 87.32 | **died** — negative on **all seven** HWS slices including the leak-matched native-speaker rows | `PHASE_I_DATA.md` §3 |
| `phaseIB-native` | englishLevel = native, 413 users | 1,008,329 | 87.33 / 92.00 / 92.84 | 94.56 / 80.14 | 90.03 (89.62) | **84.98** / 81.89 / 80.95 / 87.88 | **died** — same, harder. Verdict: **do not filter by englishLevel at any threshold**; the fourth exclusion-style curation negative of the campaign | `PHASE_I_DATA.md` §0, §3 |

**I-B — the Cyrillic prototypes.** Both: `resbn:80:1,2,4,8`, embed_hid 96,
**94 k steps**, no layout-alt, greedy checkpoint selection, byte-identical
1,142,727-byte graph ("the alphabet is data, not architecture"). Eval =
untouched Yandex `valid-10k`, **9,416 default-grid rows, EVAL-ONLY — no Yandex
row is used in training anywhere**. Footing: app-ru 50 k CKDT trie at E1
(λ = 1.1) unless stated.

| arm | training data | greedy | in-dict t1 / t3 / t5 (app-ru 50 k, λ 1.1) | on Yandex `voc` 503 k | verdict | source |
|---|---|---|---|---|---|---|
| `phaseIB-ru-real` | 1,000,000 **real** Yandex default-grid rows | **75.23** | **89.64 / 95.82 / 96.97** (all-rows 80.64; ≤3 94.12, 4+ 86.80) | 84.11 / 92.11 / 93.91 | Cyrillic decodes at English-class accuracy with **zero model changes** — but it is corpus-licensed out of anything shippable | `PHASE_I_DATA.md` §6 |
| **`phaseIB-ru-synth`** ← the Cyrillic bar-holder | 1,000,000 **purely synthetic** rows (English motor residuals transplanted onto ru polylines, `cyrillic_synth.py`); selection on 5,000 synthetic rows, seed 999; **no real Cyrillic sample seen before final eval** | 37.07 | **76.21 / 88.53 / 91.42** (all-rows 68.56; ≤3 83.75, 4+ 71.45); at λ = 0 it falls to 68.65 | 61.09 / 77.64 / 82.53 | synth-only launches a script at in-dict t1 ≈ 76 — the same class as the shipped geometric engine's cross-layout anchors. **Synth-vs-real gap = −13.4 in-dict t1.** This 76.21 is the Cyrillic bar for the rest of the campaign | `PHASE_I_DATA.md` §6 |

### 4.10 Phase J — the convergence campaign

**Footing:** full val-9918 at E1 / AOSP via exported ONNX; alt-layout `az26`
in-dict at E1. **`PHASE_J.md` never writes the literal trie sizes** — it says
"E1/AOSP" and "app-98k". Bars = the eleven `resbn192i` seed-means (§0).
**test-2400 was NOT unsealed in Phase J** — the pre-registered rule required
*all* bars, ≤3 and Cyrillic did not fall, no pre-registration was filed and no
ledger entry was appended. **Latency was measured for exactly three artifacts**
(§10); every other arm's latency is *not recorded* — the phase never claims one.
**Params are recorded only for `resbn192i` / `sw2345` (both 1,512,802).**

| arm | round | change vs the base recipe | val-9918 t1/t3/t5/≤3/4+ | dvorak (app) / euro-mean | seeds | verdict | source |
|---|---|---|---|---|---|---|---|
| `phaseI-ch256` (carried in) | ref | ch 256, p 0.50 | 88.64 / 92.56 / 93.23 / 91.15 / 87.33 | 87.95 (87.83) / 82.29 [derived] | 1234 | the p 0.50 control point of the ch-256 dose sweep | `PHASE_J.md` §5.1 |
| `phaseJ-ch256-p65` | 1 | ch 256, p 0.65 | **88.69 / 92.75 / 93.37 / 91.21 / 87.38** | 89.66 (88.89) / **82.11 [derived** 82.44/79.61/78.72/87.66**]** | 1234 | first model to beat the val bar on t1, t3, t5 and 4+ simultaneously — and **loses all four euro bars**. Not promotable | `PHASE_J.md` §5.1a |
| `phaseJ-ch256-p80` | 1 | ch 256, p 0.80 | 88.31 / 92.61 / 93.38 / 90.94 / 86.95 | 88.12 (86.45) / **83.96 [derived** 83.92/82.90/79.99/89.02**]** | 1234 | **the only ch-256 point that beats all four euro bars** — and misses ≤3 and both dvorak columns. Not promotable; the paired-seed test that would have settled p 0.65-vs-0.80 was **never run** | `PHASE_J.md` §5.1b |
| `phaseJ-ch192-p80` | 1 | ch 192, p 0.80 | 88.10 / 92.51 / 93.33 / 90.88 / 86.66 | 90.72 (90.15) / **83.95 [derived** 83.83/83.07/80.67/88.23**]** | 1234 | p 0.80 is worse than p 0.65 at ch 192 on val but buys transfer → **0.65 is a plateau optimum, not an undershoot** | `PHASE_J.md` §5.1b |
| `phaseJ-cr80` | 1 | ch 80 + **CR-CTC α 0.2** (dual views, symmetric stop-grad frame KL) | 87.46 / 92.15 / 92.89 / 90.85 / 85.69 | **91.98** (**91.94**) / **84.42 [derived** 84.59/83.74/80.95/88.40**]** | 1234 | read at the time as "**the strongest transfer lever measured**" (dvorak **+3.13**, dvorak-app +3.74 for −0.20 val t1) — **later retracted** | `PHASE_J.md` §5.1c, §6.4.1a |
| `phaseJ-sw234` (+ s4321, s7777) | 2 | + `tier_sw234` (101,842 new FUTO rows) → 1,260,674 rows | s1234 88.69 / 92.66 / 93.30 / 91.32 / 87.32; **3-seed mean 88.54 / 92.65 / 93.35 / 91.25 / 87.12** — ≤3 misses by **one row** (−0.02) | 3-seed 89.92 (89.34) / **83.75 [derived** 83.40/83.01/80.58/88.00**]** | 1234 / 4321 / 7777 | **8/11 seed-mean** — reaches `ch256-p65`'s val t1 with **57 % of the parameters**; briefly the finalist, then displaced by `sw2345` | `PHASE_J.md` §6.1a, §6.6, §6.6.1 |
| `phaseJ-realalt` | 2 | + clearflow/kasroz train rows on their own geometry (29,184 rows) → 1,188,016 | 88.42 / 92.70 / 93.37 / 91.35 / 86.89 | 88.89 (88.44) / **83.74 [derived** 83.11/83.91/80.45/87.49**]**; clearflow 96.35 / kasroz 94.89 — **in-domain, NOT transfer** | 1234 | val-neutral-to-positive but **not worth its price** — it burns the campaign's only two never-seen real-layout eval corpora for 2.5 % of the mix | `PHASE_J.md` §6.1b |
| `phaseJ-yfix` | 2 | HWS y-frame ×7/6 train-side correction → 1,158,113 rows | **87.43 / 91.94 / 92.72 / 90.50 / 85.83** — Δ **−0.89 / −0.76 / −0.53 / −0.71 / −1.00** | 87.46 (86.69) / **83.11 [derived** 82.58/81.97/79.49/88.40**]** | 1234 | **rejected — the largest coherent negative in the phase, and the arm cannot answer its own question**: the val HWS half keeps the uncorrected frame, so a train-only fix is penalised by construction | `PHASE_J.md` §6.1c |
| `phaseJ-ch256-280k` | 2 | ch 256, p 0.65, **280 k steps** + 23 snapshots | 88.61 / 92.66 / 93.38 / 91.30 / 87.21 — Δ vs its own 188 k twin −0.08/−0.09/+0.01/+0.09/−0.17 | 88.60 (87.67) / **82.86 [derived** 82.92/81.89/78.76/87.88**]** | 1234 | **the schedule axis is closed** — 49 % more schedule is a tie at best; train-loss headroom is not val headroom | `PHASE_J.md` §6.1d |
| `phaseJ-cr192` | 3 | ch 192 + CR-CTC α 0.2 | 88.12 / 92.33 / 93.16 / 91.27 / 86.49 | 88.97 (87.63) — **dvorak −1.63** / **83.67 [derived** 84.59/81.47/80.26/88.34**]** | 1234 | **the attribution control that killed CR-CTC** — the ch-80 transfer gain does not survive capacity. §5.1c retracted | `PHASE_J.md` §6.4.1a |
| `phaseJ-sw234-cr` | 3 | sw234 data + CR-CTC at ch 192 ("the stack candidate") | 87.81 / 92.32 / 93.03 / 90.71 / 86.31 — **−0.88 t1 vs its no-CR twin** | 89.34 (88.89) / **83.81 [derived** 84.40/82.48/80.13/88.23**]** | 1234 | **died** — negative data × consistency interaction; the two levers are not additive | `PHASE_J.md` §6.4.1a |
| `phaseJ-cr256-p80` | 3 | ch 256, p 0.80 + CR-CTC (the frontier bundle) | **t1 88.10 only** — t3/t5/≤3/4+ **not recorded** | 89.42 (88.85) / **82.45 [derived** 82.30/81.97/78.67/86.86**]** — four of four euro axes negative vs `ch256-p80` | 1234 | **died** — "the ch 256 frontier is now out of candidates"; needs int8-trunk to make ≤5 MB and **no int8 measurement exists** | `PHASE_J.md` §6.4.1b |
| **`phaseJ-sw2345`** (+ s4321, s7777) | 3 | + `tier_sw234` **and** `tier_sw5q` (24,707 qwerty-en rows) → 1,285,381 rows | 3-seed **88.51 / 92.67 / 93.37 / 91.20 / 87.11** | 89.87 (88.98) / 83.98 [derived] | 1234 / 4321 / 7777 | **the finalist** — see §3 | `PHASE_J.md` §8, §10 |
| `phaseJ-sw2345-snap` | 3 | `sw2345` s1234 re-run, differing only by cudnn nondeterminism | 88.54 / 92.73 / 93.39 / 91.06 / 87.23 | **88.24** / spanish **87.71** — "markedly worse transfer" | 1234 | recorded as the uncomfortable reminder that **even a fixed seed does not fix the transfer axis to better than ~1–3 pt** in this harness | `PHASE_J.md` §8 |
| `phaseJ-sw234-p80` | — | sw234 data at p 0.80 | 88.32 / 92.47 / 93.32 / **91.03** / 86.92 — ≤3 **−0.29** vs its p 0.65 twin | alt-layout decode **not recorded** | 1234 | **dose cannot be the ≤3 repair** — it makes ≤3 worse | `PHASE_J.md` §6.7b |
| `phaseJ-futoaug` | — | Spec-B FUTO-parity augmentation bundle (shear ±0.1, rot ±8°, time-reversal p 0.25, frame-hold masking) | **87.86 / 92.28 / 93.04 / 91.00 / 86.23** — Δ −0.46 / −0.42 / −0.21 / −0.21 / −0.60; greedy −5.4 | not recorded | 1234 | **rejected** — the coupled affine sampler already covers the useful part of that space | `PHASE_J.md` §6.7a |
| `phaseJ-ru192` | — | Cyrillic capacity rung: ch 192 / 188 k / synth-only | Cyrillic in-dict **73.53 / 86.80 / 90.17**, greedy 40.18 — Δ vs the 76.21 bar **−2.68 / −1.73 / −1.25 / +3.11 greedy**; `last.pt` 73.30 | n/a (ru-only) | 1 | **a negative** — more capacity and twice the schedule made the emissions better and the answers worse: overfitting to the synthetic generator, and the checkpoint-selection explanation is **refuted by measurement** on `last.pt` | `PHASE_J.md` §6.5 |
| `phaseJ-joint` | — | en + 1,000,000 synthetic ru rows, **one 65-wide head** for both scripts | ru in-dict **76.56** (+0.35, inside one binomial SE at n = 8,471) / t3 88.16 (−0.37) / t5 91.12 (−0.30), greedy 23.68 (−13.39); **en val 87.90 / 92.49 / 93.24 / 90.50 / 86.55 = −0.42 en t1 against a 0.3 tolerance** | not recorded | 1 | **not adopted** — a tie at best on ru while failing the en tolerance. Carries its own retraction: a running 2,000-row figure of "77.40" was wrong; the completed 9,416-row decode is 76.56 | `PHASE_J.md` §6.8 |
| `phaseJ-sw234-snap` | — | ship-seed re-run for soup supply | — | — | — | **KILLED** before completion when the finalist flipped to `sw2345`; no numbers exist | `PHASE_J.md` §6.6.1 |
| `phaseJ-smoke{1,2,3,4}` | — | plumbing | — | — | — | **not results** | run dirs only |

**The λ correction for Cyrillic, model-independent** (`PHASE_J.md` §6.9; tuned
on ru val rows 0:4708, confirmed on the untouched 4708:9416): every ru number
ever published — the 76.21 bar included — was decoded at E1's λ = 1.1, while the
app ru lexicon stores `freq = 255 − rank`. At **λ = 2.0** the synth-only
bar-holder reads 76.91 tune / **77.92** confirm and the joint model 77.83 /
78.23. **The honest shippable Cyrillic figure is ≈ 77.4, not 76.21** — but the
lever lifts challenger and bar equally, so **the Cyrillic axis remains not
beaten**.

**Phase-J levers measured against the ≤3 stone, all five negative or
sign-inconsistent** (`PHASE_J.md` §9): layout-alt dose p 0.65→0.80 **−0.29** ·
CR-CTC α 0.2 at ch 192 **−0.50** · FUTO-parity augmentations **−0.21** ·
checkpoint soup **+0.14 / −0.33, sign-inconsistent** · a stratum-aware
`minmargin` decode sweep over the E1 region **+0.03 where ~+0.33 was needed**.
The sweep is the diagnostic: gamma and beta re-rank candidates by length and
**cannot conjure a short candidate the beam never generated** — so the residue
is a **candidate-generation** problem.

**Checkpoint soups (Phase J)** — greedy soup over per-run snapshots with BN
re-estimation over 20,480 augmented train rows:

| soup | members | selection-t1 gain | full-val t1/t3/t5/≤3/4+ | verdict | source |
|---|---|---|---|---|---|
| `phaseJ-ch256-280k` soup | 4 of 23 snapshots (192 k / 204 k / 252 k / 264 k) | **+0.50** | **88.99 / 92.54 / 93.23 / 91.68 / 87.59** — the highest full-val t1 and 4+ of the entire campaign | on a non-candidate parent; would still miss t3 and t5 | `PHASE_J.md` §6.3 |
| `phaseJ-sw234` s4321 soup | 2 members | +0.16 | 88.49 / 92.75 / 93.33 / **91.32** / 87.01 (≤3 **+0.14**) | — | `PHASE_J.md` §6.6.2 |
| `phaseJ-sw234` s7777 soup | 4 members | +0.14 | 88.26 / 92.51 / 93.22 / **90.91** / 86.89 (≤3 **−0.33**) | — | `PHASE_J.md` §6.6.2 |
| **verdict** | — | — | — | **the soup does not generalise**: ≤3 is sign-inconsistent (mean −0.10) → **not promotable** under the campaign's own rule, and §6.3's +0.38 is retracted as over-read from one non-candidate run | `PHASE_J.md` §6.6.2 |

---

### 4.11 Phase K — mixing, the ≤3 training lever, T′ = 64, and a rescorer

**Footing:** every val number is val-9918 at E1 / AOSP; alt-layout `az26`
in-dict at E1, `dvorak-app` on the app-98k trie. **`PHASE_K.md` never writes the
literal trie sizes.** **test-2400 sealed for the entire phase** — nothing in
Phase K read, loaded or hashed it. Configurations (the mixes) are in §5.

| arm | change | params | bytes | laptop ms | val-9918 t1/t3/t5/≤3/4+ | dvorak (app) / euro-mean | bars | seeds | verdict | source |
|---|---|---|---|---|---|---|---|---|---|---|
| `phaseK-t64` (`phaseK_t64_s1234_contractv2.onnx`) | finalist recipe + `--t-out 64`; head `[1,64,65]` | not recorded | **6,076,715** fp32 | encoder **1.588 ms** (1.9× the T′ 32 twin); whole decode ≈ 2.1× (**29.0 vs 60.7 traces/s**, same box) — MEASURED | **88.32 / 92.57 / 93.31 / 91.12 / 86.87** — Δ vs the `sw2345` twin −0.19/−0.02/−0.04/**+0.21**/**−0.39** | 90.96 (90.27) / **84.60 [derived** 84.40/83.07/**82.40 german, campaign best**/88.51**]** | **8/11** — all six layout bars, misses val t3 and ≤3 | 1234 | the Phase-I T′=64 transfer promise **reproduces**; its 4+ promise **flips sign** (+0.33 at ch 80 → −0.39 at ch 192). **Documented, not promoted**; breaks `CtcEmissions.sliceFromHead` and any `[·,32,·]` assert | `PHASE_K.md` §6, §8.4 |
| `phaseK-sw2345-280k` | finalist recipe on 280 k steps + snapshots | not recorded | no artifact staged | not recorded | 88.37 / 92.44 / 93.27 / 91.18 / 86.92 — Δ −0.14/−0.15/−0.08/+0.27/−0.34 | not recorded | — | 1234 | **died as a wash — the third independent 280 k negative.** Its checkpoint soup bought +0.16 selection t1: a wash again | `PHASE_K.md` §6, §8.1 |
| **`phaseK-sw2345-slw2`** (+ s4321, s7777) | finalist recipe + `--short-loss-weight 2.0` | not recorded | **6,068,519** fp32 | not recorded (arch unchanged) | 3-seed **88.27 / 92.59 / 93.31 / 91.39 / 86.64** — **≤3 clears on EVERY seed** (91.47 / 91.38 / 91.32) | 90.07 (89.68) / 83.67 [derived] | **7/11** | 1234 / 4321 / 7777 | see §3 — proof the ≤3 stratum responds to a **training-side** signal; the s1234 all-five-val sweep was seed luck | `PHASE_K.md` §8.3 |
| `phaseK-sw2345-s5555` | finalist recipe at a never-before-used seed, for the blind gate test | not recorded | not recorded | not recorded | **88.65 / 92.64 / 93.38 / 91.47 / 87.18** | not recorded (member alone) | — | 5555 | trained only to supply the prospective pair-gate test | `PHASE_K.md` §8.5 |
| `phaseK-resbn192i-s5555` | incumbent recipe at seed 5555 | not recorded | not recorded | not recorded | **88.52 / 92.67 / 93.42 / 91.00 / 87.23** | not recorded | — | 5555 | same | `PHASE_K.md` §8.5 |
| ranker v1 (`mined_sw2345`) | 14-feature listwise rescorer, listwise CE, err-weight 4 / short-weight 2 | **5,185** | not recorded | ~0.3 ms of feature work | seed-mean Δ at frozen w = 0.05: **+0.08 t1 / +0.04 ≤3 (91.24) / +0.11 4+**; per-seed ≤3 +0.33 / −0.03 / −0.17 | — | — | mined from s1234 only (600 k slates) | **died as a ≤3 lever** — sign-inconsistent across seeds, and 91.24 still misses the bar by 0.03 | `PHASE_K.md` §5 |
| `phaseK_ranker_sw2345_2seed.onnx` | same ranker, mined from s1234 **and** s4321 (1.2 M slates) | not restated | **21,782** | ~0.3 ms | seed-mean **+0.08 / +0.04 / +0.02 / +0.04 / +0.11**; per-seed ≤3 +0.30 / 0.00 / −0.18 | — | — | applied to 3 seeds | **not the ≤3 lever** even seed-general; **is** a small sign-consistent t1/t5/4+ lever (3-of-3 seeds each) | `PHASE_K.md` §5.1 |
| `phaseK_ranker_resbn192i.onnx` | the symmetric control ranker, mined from the incumbent's own emissions | not recorded | **21,782** | ~0.3 ms | incumbent 88.32 → **88.58** (+0.26); ≤3 91.21 → 91.35 (+0.14) | — | — | 1234 | proves the rescorer is **symmetric** — "a field-shifting lever, not a ranking-shifting one"; any bar comparison involving a rescorer must use rescored bars | `PHASE_K.md` §5.1 |

**The pair-compatibility gate, discovered in Phase K** (`PHASE_K.md` §4.3,
§8.5): per-frame argmax agreement on 2,000 **unlabelled** val traces, threshold
≈ 95 %. Whole-string greedy agreement is flat across all 21 member pairs
(76.7–78.7 %) and predicts nothing; letter-identity agreement where both emit is
~96 % for every pair — the disagreement is purely *where* letters and blanks
sit. Measured: s1234 pair 96.9 ✓ · s7777 pair 96.1 ✓ · mix3 edges
95.5 / 96.2 / 96.9 ✓ · **s4321 pair 88.8 ✗** · cross-seed control **83.3 ✗** ·
s5555 pair 97.0 ✓ (predicted **before** the decode, git `3156080`).

### 4.12 Phase L — pipeline v2 (coupled-pair training)

**Footing:** val-9918 at E1 / AOSP ("a 147 k-word trie" as the doc writes it) +
the six layout bars; `dvorak app-98k` on the app trie. Arch, identical for every
arm: `--ch 192 --block resbn --dilations 1,2,4,8 --t-out 32`, `train_v2.py`,
188 k steps, `--pair-weight 0.3 --pair-ramp-start 5000 --pair-ramp-len 15000`,
members differing by init seed and by `--slw-a 1.0 / --slw-b 1.5`.
**`PHASE_L.md` records no parameter count (only "1.5 M") and NO LATENCY AT ALL
— no ms figure, no protocol, for any arm.** **test-2400 sealed throughout.**
Configurations are in §5.

| arm | change | val-9918 pair t1/t3/t5/≤3/4+ | dvorak (app) / euro-mean | agreement (gate) | campaign bars / vs card | seeds | verdict | source |
|---|---|---|---|---|---|---|---|---|
| **`v2pair-s1234`** (L1) | the reference coupled pair | **88.90 / 92.86 / 93.58 / 91.53 / 87.53**, mix greedy 72.92 | 93.04 (92.76) / **85.01 [derived** 84.16/83.91/82.08/89.87**]** | training-final **98.34 %**, 46 of 47 evals over the gate; ONNX gate 98.33 % PASS | **11/11 campaign**, 10/11 vs card (azerty) | 1234 | the phase's best configuration; the arm that confirmed E1; later the **KD teacher** for Phase M | `PHASE_L.md` §3, §7–§9 |
| `v2pair-s1234` **member A** (slw 1.0) | solo member | s1234 88.60 / 92.62 / 93.36 / 91.32 / 87.18 | 91.17 (91.01) | — | 11/11 campaign at s1234 (`PHASE_L.md` §9.3); **3/11 vs card** (`PHASE_M.md` §6.1 — `PHASE_L.md` computes no card tally for member A) | 1234 (+4 more in M) | the promoted single model — see §3, and the retraction | `PHASE_L.md` §9.3, §15.4; `PHASE_M.md` §6.1 |
| `v2pair-s1234` **member B** (slw 1.5) | solo member | s1234 t1 88.47, ≤3 **91.53** | dvorak 92.80, spanish 89.42 | — | 9/11 at s1234 (misses azerty, qwertz); **10/11 at the 3-seed mean**, holding better ≤3 (91.43) and dvorak (91.33) than member A | 1234 | the fp16w half of the shippable pair | `PHASE_L.md` §9.3, §15.4 |
| `v2pair-e2-s1234` (L2) | L1 **+ the E2 English-synthesis pools** (`synth_en_short` 150 k + `synth_en_tail` 150 k = 18.9 % of the mix) | **88.85 / 92.79 / 93.42 / 91.59 / 87.43**, greedy 72.03 | 92.76 (92.35) / **85.63 [derived** 85.02/84.84/82.22/90.44**]** | 98.23 % PASS | 11/11 campaign, 10/11 vs card (t5 by 0.04) | 1234 | **fails its pre-registered E2 gate by 0.01** (t5 −0.16 against a 0.15 limit) → not promoted | `PHASE_L.md` §5, §9, §11.2 |
| `v2pair-e2-s4321` | E2 recipe, seed 4321 | 88.52 / 92.71 / 93.35 / 91.15 / 87.15, greedy 71.71 | 92.88 (92.35) / **84.71 [derived** 84.35/83.91/82.08/88.51**]** | 98.20 % PASS | 10/11 campaign (≤3 −0.12), **5/11 vs card** | 4321 | with s7777, the evidence that E2's s1234 result was seed luck | `PHASE_L.md` §9 |
| `v2pair-e2-s7777` | E2 recipe, seed 7777 | 88.49 / 92.72 / 93.35 / 91.12 / 87.12, greedy 71.84 | 89.82 (89.21) / **85.45 [derived** 84.98/85.09/81.95/89.76**]** | 98.18 % PASS — but **31 of 47** evals, starting at 76.7 % at 4 k: the furthest-apart pair in campaign history, pulled over the gate by step 68 k | 10/11 campaign (≤3 −0.15), 5/11 vs card | 7777 | same | `PHASE_L.md` §7, §9 |
| **`v2pair-pw0-s1234`** (L3) | L1 args with **`--pair-weight 0`** — identical batches, no mutual KL | selected pair 88.09 / 92.59 / 93.32 / 91.12 / 86.52, greedy **53.12**; own-best mix **t1 87.64, greedy 29.10** (members solo greedy 72.6 / 71.8) | 89.34 (88.81) / **84.68 [derived** 84.35/84.41/80.99/88.96**]** | **92.09 %**, **2 of 47** evals over the gate; selected pair 95.32 % (marginal PASS), own-best 91.30 % **FAIL** | 7/11 campaign, 1/11 vs card | 1234 | **the attribution control the whole E1 verdict rests on** — the KL, not batch sharing, pins the alignment gauge | `PHASE_L.md` §5, §7–§11.1 |
| `v2pair-s4321` | L1 recipe verbatim, seed 4321 (settlement) | **88.82 / 92.80 / 93.48 / 91.47 / 87.44**, greedy 73.14 | 90.44 (90.07) / **84.93 [derived** 85.02/84.25/81.49/88.96**]** | 98.05 % PASS, 38 of 47 | **11/11 campaign**, 8/11 vs card | 4321 | azerty — the axis the phase forecast would fail — passed; the miss came on dvorak/dvorak-app/spanish instead | `PHASE_L.md` §14–§15 |
| `v2pair-s7777` | L1 recipe verbatim, seed 7777 (settlement) | **88.78 / 92.60 / 93.42 / 91.47 / 87.38**, greedy 72.84 | 91.33 (90.80) / **85.15 [derived** 85.45/84.33/82.08/88.74**]** | 98.15 % PASS, 37 of 47 | **11/11 campaign**, 6/11 vs card | 7777 | — | `PHASE_L.md` §14–§15 |

**Data artifacts produced (and refuted) by Phase L.**
`cache/synth_en_short.npz` — 150,000 rows over 8,126 lexicon words of length
≤ 4, **66,284,178 B**, sha `78e0984e…`; `cache/synth_en_tail.npz` — 150,000 rows
over 121,499 words with < 3 real train traces, **68,360,181 B**, sha
`92b89a56…`. **E2 is REFUTED at three paired seeds**: sign-consistently
**−0.21 t1 / −0.12 t5 / −0.22 4+**; its single-seed ≤3 (+0.06) and euro gains
(azerty +0.86, qwertz +0.93) did not reproduce. 300 k licence-clean
endpoint-validated rows bought a negative, and `english_synth.py` plus the pools
stay committed as the documented negative.

**Registered in Phase L and NOT run there** (all subsequently executed in
Phase M, none quietly dropped): the `--pair-weight` sweep {0.1, 1.0}; **E6** the
geometric alignment prior (implemented behind `--geo-align-weight`, default off);
**E4** the `w_real` arm; **E7** distillation (trigger unmet at the time); a
fourth and fifth seed for the two tie margins.

### 4.13 Phase M — the close (distillation, the coupling sweep, and two deaths)

**Footing:** val-9918 at E1 / AOSP for §6.1 / §7 / §9 / §10 (the footing label
is not printed in `PHASE_M.md` itself — it is established by the audit's
recomputation from `val_dump_e1.jsonl`, `AUDIT_FINAL2.md` §3); the app footing
is the 98,081 trie at `0.9/4.0/0.25/0.25/0.9882`. **Latency is not recorded for
any Phase-M arm** — the only ms figure in the document is option A's 1.79 ms,
and it carries no protocol, device or measured/inherited label.

| arm | stage | change | val-9918 t1/t3/t5/≤3/4+ (s1234 unless noted) | dvorak (app) / euro-mean | bars | seeds | verdict | source |
|---|---|---|---|---|---|---|---|---|
| `v2pair-s5555` | 1a | two more seeds of the L1 pair recipe | pair t1 **88.65** (`PHASE_M.md` §6.2); ≤3 **91.30** — recorded not in `PHASE_M.md` but in `AUDIT_FINAL2.md` §2, recomputed from the per-trace dumps | not recorded per axis | gate 98.14 % PASS, working-band prediction correct | 5555 | one of the two seeds that turned the 3-seed member-A claim into a 5-seed retraction | `PHASE_M.md` §1.1, §5, §6.2; `AUDIT_FINAL2.md` §2 |
| `v2pair-s9999` | 1a | as above | pair t1 **88.73** (`PHASE_M.md` §6.2); ≤3 **91.41** (`AUDIT_FINAL2.md` §2, from the dumps) | not recorded per axis | gate 98.25 % PASS | 9999 | with s5555 completes the 5-seed footing; **8 of 8 gate predictions correct** across five seeds | `PHASE_M.md` §1.1, §5, §6.2; `AUDIT_FINAL2.md` §2 |
| **`v2kd-fresh-w1`** (+ `-s4321`, `-s7777`) = **`phaseM_kd_fresh_w1`** | 1, E7 | single ch192 `resbn`, `train.py`, 188 k, **distilled from the L1 s1234 gated pair** (`pair_a_best.pt` + `pair_b_best.pt`, 98.33 % agreement) passed as a two-checkpoint ensemble whose target is `logsumexp − log N` — the prob-averaged mix2 contract; `--kd-weight 1.0`; **fresh init** | s1234 88.62 / 92.69 / 93.46 / 91.38 / 87.18; **3-seed mean 88.750 / 92.773 / 93.473 / 91.373 / 87.387**. App footing s1234 **89.20 / 93.63 / 94.37 / 92.59 / 87.44**, fp32 ≡ fp16w to 0.00 — recorded not in `PHASE_M.md` but in `UNSEALING_4.md` §2.2 / §4.2 and `AUDIT_FINAL2.md` §4.5 | 3-seed **91.819** (91.100) / **84.83 [derived** 84.530/83.965/81.295/89.534**]** | **11/11, per-seed [11, 11, 11]**, smallest margin ≤3 **+0.103**; card per-seed [6, 8, 6] | 1234 / 4321 / 7777 | **the ship model** — see §1 and §2. Selection-prefix t1 86.04 vs member A's 85.60 | `PHASE_M.md` §6.1, §9, §11.1, §12; `UNSEALING_4.md` §2.2, §4.2; `AUDIT_FINAL2.md` §4.5 |
| `v2kd-initA-w1` | 1, E7 | same, but **initialized from member A** via the new `--init-from` (weights only, no optimizer/step/RNG, strict arch check, source sha recorded) | **88.59** / 92.72 / 93.43 / **91.41** (best ≤3 of the three students) / 87.12 | 91.86 (91.53) / **84.31 [derived** 84.21/83.24/81.63/88.17**]** | 10/11 campaign, 5/11 card | 1234 | **fails the gate by 0.01 on t1** (88.59 < 88.60) → dropped, **not retuned**. Its real value is the negative: **gauge-matching the student's initialization is unnecessary** — what matters is that the *teacher* is alignment-consistent. Warm starts open at beam t1 85.20 / 84.96 (CTC 0.617 / 0.601) vs fresh 80.80 (CTC 1.077), so `--init-from` demonstrably worked | `PHASE_M.md` §1.2, §3, §6.1 |
| `v2kd-initA-w4` | 1, E7 | as above with **`--kd-weight 4.0`** — the never-swept knob | 88.52 / 92.74 / **93.56** (best t5 in the phase) / 91.09 / 87.18 | **92.35** (**92.19**) — best transfer in the phase / **84.49 [derived** 84.21/83.66/81.58/88.51**]** | 10/11 campaign, 6/11 card | 1234 | **fails the gate** (t1 and ≤3) → dropped, not retuned. KD weight 4.0 does not buy the gate | `PHASE_M.md` §1.2, §6.1 |
| `v2pair-pw01` | 2 | coupling sweep, `--pair-weight 0.1` | t1 88.84, ≤3 91.44 | dvorak 92.96, azerty 84.07 | 11/11 campaign, 9/11 card | 1234 | sweep point; agreement 98.08 % | `PHASE_M.md` §10.1 |
| `v2pair-pw10` | 2 | coupling sweep, `--pair-weight 1.0` | t1 88.85, ≤3 91.47 | dvorak **91.09** (−1.95 vs 0.3), azerty **84.78** | 11/11 campaign, 8/11 card | 1234 | agreement is **highest** here (98.58 %) and the mix's transfer edge **collapses** — over-coupling kills the diversity the averaging feeds on | `PHASE_M.md` §10.1 |
| `v2pair-e4` | 2 | **E4** `w_real` 0.217 → 0.25 (`--layout-synth-frac 0.615`) | Δ vs the control: 4+ −0.15 (other absolutes not recorded) | **dvorak −2.81** (absolute **90.23** vs the control's 93.04 — recorded not in `PHASE_M.md` but in `AUDIT_FINAL2.md` §3, from the logs), dvorak-app −2.85, qwertz −1.26, german −1.14, spanish −1.37, **azerty +0.62** | still 11/11 campaign as a pair | 1234 | **DROPPED** — the rule ("euro gains with no axis losing more than 0.15") is violated by nearly 20×, and only one of four euro axes even gains | `PHASE_M.md` §10.2; `AUDIT_FINAL2.md` §3 |
| `v2pair-e6` | 2 | **E6** geometric alignment prior, `--geo-align-weight 0.05` | Δ **t1 −0.21, t5 −0.16, ≤3 −0.18, 4+ −0.23** (absolutes not recorded) | not recorded | still 11/11 campaign as a pair | 1234 | **DROPPED by its own kill criterion** ("any val bar −0.15 at one seed and it is dropped; no second chance at another weight") — four val bars past it, three past double it. And the motivation is gone too: E1 pins the gauge without a prior and without a val bill | `PHASE_M.md` §10.3 |

**The coupling sweep, in one place** (`PHASE_M.md` §10.1, seed 1234, val-9918):
pw 0.0 → agreement 92.09 %, mix greedy **53.12**, t1 88.09, 7/11 · pw 0.1 →
98.08 %, 73.72, 88.84, 11/11 · **pw 0.3 → 98.33 %, 72.92, 88.90, 11/11 (the
finalist)** · pw 1.0 → **98.58 %**, 73.22, 88.85, 11/11 but dvorak −1.95.
**Agreement rises monotonically with the weight while the transfer edge
collapses at 1.0 → 0.3 is the interior optimum, and the knob is closed.** The
sweep was pre-registered as a *measurement*, so no promotion follows from it.

**Every pre-registered bar and rule of Phases L+M, scored** (`PHASE_M.md` §11.3):
bar 1 (pair ≥ card on ≥2/3 seeds) **NOT MET** · bar 2 (a single model ≥ all
eleven campaign bars, seed-mean) **MET at 3 seeds → RETRACTED at 5 for member A**,
**MET and robust for the E7 student** · bar 3 / crown (a single model beating the
full card) **NOT WON** · E1 coupling **CONFIRMED** · E2 synthesis **REFUTED** ·
E4 **DROPPED** · E6 **DROPPED** · coupling weight **0.3 confirmed interior-optimal**
· E7 gate passed by `fresh`, failed by both `initA` · gate-band predictions
**12 of 12 correct** above 98 % agreement — qualified by the audit as
**8 committed-blind + 4 rule-implied**, because the four stage-2 gates were
measured 09:35:31 with the pair decodes at 09:39–09:42 and **no commit in
between** (`AUDIT_FINAL2.md` §1). **The ledger is empty**: every item registered
and not run at the end of Phase L was run in Phase M, and no element was retried
after failing its rule.

---

### 4.14 Phase N — beating FUTO on FUTO's own test

Recorded in full in `PHASE_N.md` §19; no new promotable model. Terminal
standing: B1 = 4 of 5 metrics outright on every seed on FUTO's official test
(t1 +0.87, McNemar p ≤ 1.7e-18 all seeds), 4+ a statistical tie (−0.010
seed-mean); B2 not closed (−0.365 seed-mean). Levers screened and all closed:
`n2a`/`n2b` source reweighting **refuted on every prong**, `n2e`/`n2e-b`
min-margin preset **refuted at an 88.63 dev-4+ emissions ceiling**, `n2d` ch256
capacity **real (+0.29 dev-8k 4+, in band)** but `n3` students **fail the
distillation gate** and `n3b` fails holdout confirmation. Two of three milestone
reads unspent; test-2400 sealed at ledger 4 throughout.

### 4.15 Phase O — per-script models for the non-Latin scripts (2026-08-18/19)

**Footing:** each script's **own 10,000-row synthesis holdout** (disjoint donor
half, independent word draw), decoded at the app CKDT preset
`1.05 / 2.0 / 0.2 / 0.3734 / 0.9882` through the exported fp32 graph. **These
are generator numbers and they are not comparable to any val/test row in this
table** — `PHASE_O.md` §2.1 measures the probe inverting both the capacity axis
and the λ choice against real data. Russian rows are the calibration anchor and
carry a real-corpus column (Yandex, eval-only). All arms: `resbn:80` dil 1,2,4,8,
embed_hid 96, 94 k steps, greedy selection, **single seed 1234**, no layout-alt.
Lexicon provenance (applies to every uk/bg/mk/he holdout row here and in
§4.16–§4.17): measured on raw wordfreq top-50k lexicons — the shipped ARC-056
CKDT packs (app commit `86156ea3`, 2026-09-01) did not exist yet.

| arm | K | script holdout t1/t3/t5 | greedy | ch192-EN zero-shot (same probe) | ch80-EN zero-shot (same probe) | real-corpus column | verdict |
|---|---|---|---|---|---|---|---|
| **`phaseO-el-synth`** = `el_synth_ch80` | 25 | **82.54** / 92.97 / 94.93 | 35.87 | 83.10 (−0.56) | 76.56 (**+5.98**) | none exists | exported; the only new script with a bundled app lexicon |
| **`phaseO-uk-synth`** = `uk_synth_ch80` | 31 | **79.27** / 91.91 / 94.05 | 31.98 | 81.41 (−2.14) | 74.20 (**+5.07**) | none exists | exported; ceiling from ї/ґ being corner-only (4.03 % of vocabulary) |
| **`phaseO-bg-synth`** = `bg_synth_ch80` | 30 | **71.80** / 88.56 / 92.18 | 26.86 | 74.09 (−2.29) | 66.53 (**+5.27**) | none exists | exported |
| **`phaseO-mk-synth`** = `mk_synth_ch80` | 31 | **71.69** / 88.33 / 91.80 | 29.39 | 72.67 (−0.98) | 65.19 (**+6.50**) | none exists | exported |
| **`phaseO-he-synth`** = `he_synth_ch80` | 27 | **65.36** / 85.10 / 90.13 | 37.91 | 69.11 (−3.75) | 58.04 (**+7.32**) | none exists | exported **flagged** — the only ≥70-gate failure at the adopted preset (70.28 at λ 1.1) |
| `phaseIB-ru-synth` = `ru_synth_ch80` (Phase I-B model, re-probed here) | 31 | 81.10 / 92.16 / 94.08 | 29.73 | 83.38 (−2.28) | 76.24 (**+4.86**) | **77.41** in-dict t1 (8,471 rows) | the calibration anchor |
| `phaseO-ru-initH` | 31 | 81.98 / 92.95 / — | 30.72 | — | — | **77.26** | **REFUTED** — warm start from the English ch80 `phaseH-p50` is +0.88 on the holdout and **−0.14 on real (p = 0.69)**. Not promoted |
| `phaseM_kd_fresh_w1` (ship model) zero-shot on ru | — | 83.38 | 14.27 | — | — | **76.32** | the deployment alternative: the shipped model reaches 76.32 on real Russian with only a layout and a trie |
| `phaseH-p50` (English ch80) zero-shot on ru | — | 76.24 | 7.07 | — | — | **75.79** | capacity-matched control |

**Paired tests on the real Russian probe** (n = 8,471, exact McNemar):
ru-synth vs ch80-EN **+1.62, p = 1.4e-4** · ru-synth vs ch192-EN **+1.09,
p = 0.0099** · ch192-EN vs ch80-EN +0.53, **p = 0.11 (n.s.)** · ru-initH vs
ru-synth −0.14, **p = 0.69 (n.s.)**. On the *synthesis holdout* the second of
those flips to **−2.28, p = 7.1e-12**.

**Falsification, all six scripts:** with key centres permuted
(`eval_script.py --permute-layout`), every model reads **0.00 t1 / 0.00 greedy**.

**Export gates, all five new scripts:** fp32 vs torch on real traces at the real
layout, sliced 1.26e-4 … 5.11e-4, **argmax 100/100 every script**; fp16w residues
2.85e-2 … 1.78e-1 at 95–99/100 argmax, and **free at the decode** (10 k-row top-1
moves ≤ 0.02 on every script). All five graphs are the standard 1,142,727-byte
resbn80; fp16w 589,406 B.


### 4.16 Phase P — the generator rebuilt, and the only real-data number that moved (2026-08-19)

**Two footings, and they must never be mixed.** The ru rows carry a **real**
Yandex column (eval-only, 9,416 rows / 8,471 in-dict, app CKDT preset
`1.05 / 2.0 / 0.2 / 0.3734 / 0.9882`, exported fp32 graph) — that is the ship
gate and the only accuracy claim in this section. The five other scripts carry
their **own v2 synthesis holdout**, which is generated by the generator being
evaluated and is therefore generator-relative; `PHASE_P.md` §5.1 states the
limits. All arms: `resbn:80` dil 1,2,4,8, embed_hid 96, 94 k steps, batch 256,
lr 3e-3, greedy selection, **single seed 1234**, no layout-alt — Phase O's recipe
verbatim.

**The ship gate (real Russian).**

| arm | training cache | in-dict t1 | t3 | t5 | ≤3 | ≥4 | greedy | verdict |
|---|---|---|---|---|---|---|---|---|
| `phaseIB-ru-synth` = `ru_synth_ch80` | v1, **full** donor pool | 77.42 | 89.06 | 91.76 | 86.47 | 71.70 | 37.07 | the registered baseline |
| `phaseP-ru-v1ctl` | v1, 90/10 train side | 75.73 | 88.44 | 90.93 | 83.66 | 70.71 | 31.34 | the paired control; prices the donor split |
| `phaseP-ru-v2` | **v2**, 90/10 train side | 78.87 | 90.73 | 93.13 | 83.60 | 75.88 | 55.67 | same-footing v2 arm |
| **`phaseP-ru-v2full`** = `ru_synth_v2_ch80` | **v2**, **full** donor pool | **79.73** | **90.77** | **93.26** | 85.77 | **75.92** | **56.12** | **SHIP** — G5 PASS (bar ≥ 79.41) |

**Paired tests, exact McNemar, n = 8,471:** v2full vs baseline **+2.31,
p = 2.6e-09** · v2 vs v1 at matched footing **+3.14, p = 6.4e-14** · the 90/10
donor split **−1.69, p = 5.2e-07** · v2full vs v2 train-side +0.86, p = 0.0023 ·
the ≤3 stratum, v2full vs baseline **−0.70, p = 0.27 (n.s.)** · the ≥4 stratum
**+4.22, p = 3.6e-17**.

**The five corpus-less scripts, on their own v2 holdouts** (10,000 rows, disjoint
donor half, independent word draw, λ 2.0). The column that carries information is
the **margin against a fixed English control**, not the level: Phase O's v1
holdouts had every script model *losing* to ch192 (−0.56 … −3.75), and the same
comparison on ru inverted against real data. On v2 every one wins, and on ru the
holdout margin (+3.87) matches the real margin (+3.41).

| arm | K | holdout t1 / t3 / t5 | greedy | ch192-EN zero-shot (Δ) | ch80-EN zero-shot (Δ) | permuted geometry | vs Phase O's ch192 Δ |
|---|---|---|---|---|---|---|---|
| **`phaseP-el-v2`** = `el_synth_v2_ch80` | 25 | **90.69** / 96.34 / 97.24 | 69.01 | 84.67 (**+6.02**) | 83.77 (**+6.92**) | 0.00 | was −0.56 |
| **`phaseP-uk-v2`** = `uk_synth_v2_ch80` | 31 | **87.97** / 94.76 / 96.02 | 55.09 | 82.58 (**+5.39**) | 80.75 (**+7.22**) | 0.02 | was −2.14 |
| **`phaseP-bg-v2`** = `bg_synth_v2_ch80` | 30 | **82.26** / 94.25 / 96.15 | 55.97 | 77.05 (**+5.21**) | 74.97 (**+7.29**) | 0.00 | was −2.29 |
| **`phaseP-mk-v2`** = `mk_synth_v2_ch80` | 31 | **89.02** / 95.86 / 96.87 | 64.56 | 83.45 (**+5.57**) | 82.36 (**+6.66**) | 0.00 | was −0.98 |
| **`phaseP-he-v2`** = `he_synth_v2_ch80` | 27 | **77.00** / 90.16 / 93.27 | 56.88 | 68.94 (**+8.06**) | 66.35 (**+10.65**) | 0.01 | was −3.75; **clears the ≥70 gate it failed** |
| `phaseP-ru-v2full` on the ru v2 holdout | 31 | 86.49 | 50.98 | 82.62 (+3.87) | 80.42 (+6.07) | 0.00 | was −2.28 |

*(every cell read from `ctc/phase_p_scripts.json`; the ru holdout row is the
calibration anchor and its real column is the ship-gate table above.)*

**P6 — the same five on the full donor pool** (`PHASE_P.md` §8). The five were
regenerated with `--train-donor-side all`, the footing the shipped ru arm uses
and the one §4.1 prices at +0.86 real top-1, and retrained on the same recipe
with nothing else changed. The holdout is provably the **same 10,000 rows**
(`--train-donor-side` cannot reach the holdout split; asserted bit-identical),
so this is an exact paired comparison — and re-decoding the P4 models on it
reproduces every published number to the digit.

| arm | in-dict t1, P4 → **P6** | Δ, exact McNemar (n = 10,000) | greedy | vs ch192 EN | permuted | fp16w cost |
|---|---|---|---|---|---|---|
| `phaseP6-el-v2full` = `el_synth_v2full_ch80` | 90.69 → **90.78** | +0.09 (p 0.70) | 69.01 → 69.16 | +6.11 | 0.00 | −0.02 |
| `phaseP6-uk-v2full` = `uk_synth_v2full_ch80` | 87.97 → **87.67** | −0.30 (p 0.23) | 55.09 → 55.28 | +5.09 | 0.02 | +0.01 |
| `phaseP6-bg-v2full` = `bg_synth_v2full_ch80` | 82.26 → **82.52** | +0.26 (p 0.36) | 55.97 → 55.65 | +5.47 | 0.00 | +0.01 |
| `phaseP6-mk-v2full` = `mk_synth_v2full_ch80` | 89.02 → **88.68** | −0.34 (p 0.13) | 64.56 → 64.21 | +5.23 | 0.01 | +0.01 |
| `phaseP6-he-v2full` = `he_synth_v2full_ch80` | 77.00 → **76.86** | −0.14 (p 0.64) | 56.88 → **57.72** (p 0.015) | +7.92 | 0.01 | +0.01 |
| **pooled** | — | **−0.086, p 0.443** (50,000 paired rows) | +0.102, p 0.501 | — | — | — |

**A null, and an informative one.** The P6 arm trains on 11 % more donors *and*
loses the holdout's donor-disjointness (the reserved half is now inside the
training pool), so both confounds point up — and the pooled effect is −0.09 with
p = 0.44. The change that is worth **+0.86 real top-1 on ru** is invisible to
this probe, which is the third independent demonstration (after capacity and λ
in Phase O) that a synthesis holdout does not rank what real swipes rank. The
P6 bytes are promoted on ru's real measurement plus the holdout's evidence that
the change costs nothing — **not** on the holdout showing a gain. Cells from
`ctc/phase_p6_scripts.json`.

**Export gates.** Every fp32 export is **100/100 argmax** on the sliced contract
view against real traces on the real layout. fp16w ship bytes cost ≤ 0.03 t1 and
are exactly free on uk, mk and he. **he's P4 bytes are flagged**: their fp32
sliced residue is 1.16e-03 against a historical envelope of 0.8e-4…7.6e-4, so
that export needed `--parity-tol 2e-3`; argmax is 100/100 on both probes and the
exceedance is disclosed, not smoothed. **The P6 he export needed no relaxation**
— 4.04e-04 at the default 1e-3, back inside the envelope — so the flag does not
carry to `he_synth_v2full_ch80`. Every graph is the standard 1,142,727-byte
resbn80; fp16w 589,406 B. Hashes in `PHASE_P.md` §6.1 (v2) and §8.4 (v2full).

**All three generations stay in the registry.** The v1 artifacts
(`*_synth_ch80*`) are the bytes every Phase-O row above was measured on; the v2
artifacts (`*_synth_v2_ch80*`) are the bytes §5 / the P4 table was measured on;
the v2full artifacts (`{el,uk,bg,mk,he}_synth_v2full_ch80*`) supersede them for
deployment. ru has only two generations, because `ru_synth_v2_ch80` is already
the full-pool arm and P6 left it untouched. Alphabet strings, projection rules
and the app-side wiring of `PHASE_O.md` §3.2–3.4 are unchanged throughout — the
generator and its donor footing change the training distribution, not the
contract.


### 4.17 Phase Q — the learned generator, and the sealed upper bound (2026-08-20)

Full record: `PHASE_Q.md`. SYNTH v3 replaces the transplant with a conditional
rectified-flow model over the residual field (1.94 M params, trained on FUTO t3
+ HWS only — the MIT shipping track), plus an acquisition imprint (duration law
+ dwell snap, both fit on the generator's own corpus). Same footing rules as
§4.16: the ru rows carry the **real** Yandex column (eval-only, 8,471 in-dict,
CKDT preset, fp32 graph) and are the only accuracy claims; the five others
carry their own **v3** synthesis holdout, generator-relative as ever.

**The ship gate (real Russian), all rows paired on the same 8,471 rows:**

| arm | training cache | in-dict t1 | t3 | t5 | ≤3 | ≥4 | greedy | verdict |
|---|---|---|---|---|---|---|---|---|
| `phaseP-ru-v2full` = `ru_synth_v2_ch80` | v2 transplant | 79.73 | 90.77 | 93.26 | 85.77 | 75.92 | 56.12 | the registered baseline |
| **`phaseQ-ru-v3`** = `ru_synth_v3_ch80` | **v3 learned**, `cache_ru_v3` | **85.07** | **93.35** | **95.16** | **89.15** | **82.49** | **65.66** | **SHIP** — G5-Q PASS (bar ≥ 80.73). **3-seed mean 85.30 ± 0.207** (s4321 85.36, s7777 85.47; s1234 is the *lowest* draw, neither replicate distinguishable at p 0.24 / 0.11) — `PHASE_Q.md` §10.2 |
| `phaseQ-ru-yxgen_RESEARCH_ONLY` (sealed, unshippable) | v3 twin trained on 1 M real Yandex rows | *85.95* | *93.74* | *95.37* | *90.95* | *82.79* | *69.72* | **THE UPPER BOUND** — measurement only |
| `phaseIB-ru-real` re-decode at this preset | 1 M real Yandex rows (unshippable) | *88.69* | *95.28* | *96.82* | *93.90* | *85.39* | *75.23* | the ceiling |

**Paired tests, exact McNemar, n = 8,471:** v3 vs v2full **+5.34, p = 5.4e-53**
(greedy +9.54, p = 1.4e-100; ≤3 +3.38, p = 1.5e-11 — clears the 86.4 corollary
v2 missed; ≥4 +6.57, p = 8.8e-44) · U vs v3 **+0.89, p = 0.0025** (≥4 stratum
+0.31, p = 0.47 — indistinguishable) · ceiling vs U **+2.74, p = 5.6e-23**.
Decomposition: of the 8.96-point v2→ceiling gap, the English-trained learned
generator closes 5.34 (86 % of the 6.22 any learned generator of this family
could reach), the in-domain data adds 0.89, and generation itself costs 2.74.
fp16w decode cost +0.01. λ untouched — and in the closing round λ was
**swept and still untouched**: monotone decreasing across {1.1 … 4.0} on the
probe's tune half, optimum off-grid low, adoption refused by the interior-optimum
rule, incumbent 2.0 carrying a measured −0.63 shortfall (`PHASE_Q.md` §9.7).

**The battery** (`phaseQ_gates_v3.json`): G1 PASS (start-hit 0.885 vs real
0.915; v2 0.730) · G3 6/7 with step_cv 0.165 vs 0.15 the sole MISS and
sharp_turns 0.080 / turn_mean 0.083 / sc-coupling 0.250/0.217 all far past v2 ·
G4 **GBM₁₇ 0.7212 vs v2's 0.8125 — the lowest reading on the registered
instrument in the campaign** (45 % gap-closure vs v1), GBM₂₃ agrees at 0.7943,
MLP-speed 0.7640 MISS (English tempo texture — the axis the probe proved
non-binding) · GQ-D PRDC recall 0.879 PASS · GQ-T clean (holdout greedy 56.52
*below* real greedy 65.66). The §2 proceed-rule deviation is disclosed in
`PHASE_Q.md` §7.2. The sealed twin's calibration battery reads near-floor
(MLP speed 0.598, coords 0.501, GBM₁₇ 0.613) — what remains in-domain is the
model family's own over-smoothing (ac1, stroke count), the named mechanism
behind the 2.74.

**The five corpus-less scripts, on their own v3 holdouts** (10,000 rows, CKDT,
fresh noise + fresh word draw; levels are generator-relative — margins against
the EN zero-shot controls on the same rows are the only cross-generation
comparator, and they **widen** against P6's):

| arm | holdout t1 / t3 / t5 (s1234) | **3-seed mean ± sd** | greedy | ch192-EN (Δ at mean) | ch80-EN (Δ) | P6's ch192 Δ | permuted | fp16w cost |
|---|---|---|---|---|---|---|---|---|
| **`phaseQ-el-v3`** = `el_synth_v3_ch80` | **92.12** / 97.23 / 97.92 | **92.19 ± 0.070** | 74.17 | 85.11 (**+7.08**) | 86.05 (+6.14) | +6.11 | 0.00 | 0.00 |
| **`phaseQ-uk-v3`** = `uk_synth_v3_ch80` | **88.96** / 94.97 / 95.93 | **89.12 ± 0.266** | 60.75 | 75.94 (**+13.18**) | 77.05 (+12.07) | +5.09 | 0.00 | 0.00 |
| **`phaseQ-bg-v3`** = `bg_synth_v3_ch80` | **86.76** / 95.87 / 97.04 | **86.91 ± 0.180** | 65.28 | 76.71 (**+10.20**) | 76.03 (+10.88) | +5.47 | 0.00 | 0.00 |
| **`phaseQ-mk-v3`** = `mk_synth_v3_ch80` | **91.55** / 97.26 / 97.97 | **91.56 ± 0.121** | 71.66 | 86.55 (**+5.01**) | 85.98 (+5.58) | +5.23 | 0.00 | 0.00 |
| **`phaseQ-he-v3`** = `he_synth_v3_ch80` | **80.69** / 92.25 / 94.72 | **80.43 ± 0.238** | 64.03 | 64.64 (**+15.79**) | 63.89 (+16.54) | +7.92 | 0.00 | 0.00 |
| *(ru, on its own v3 holdout — the GQ-T probe, EN controls read for the first time in §10.1)* | *88.14* / 94.45 / 95.48 | ***87.93 ± 0.254*** | *56.52* | *74.91* (***+13.02***) | *75.82* (+12.11) | — | *0.00* | — |

*(cells from `ctc/phase_q_scripts.json`; the greedy columns — 60–74 against the
EN controls' 13–32 — are the emissions-side confirmation that the texture moved
toward the target scripts and away from English, the direction ru's real probe
verified at +5.34.)*

**The closing round (`PHASE_Q.md` §8/§10) replicated all six decoders at seeds
4321 and 7777** — 12 runs, recipe verbatim, caches untouched, `--seed` the only
varying quantity. **Every tier replicates**, all twelve exports cleared the
default 1e-3 tolerance at 100/100 argmax, and none of the four pre-registered
anomaly rules fired, so **s1234 remains the shipped artifact** and no fixture was
regenerated. The seed-mean is now the quoted tier; the s1234 column is kept
because it is what the shipped bytes measure. Two findings came out of the
replication itself: the campaign's **±1.0 single-seed resolution floor is ~5×
too wide** on this instrument (measured ru real-probe seed sd **0.207**, so ~±0.4
at 95 %) — which does not re-decide any past gate but does make §9.7's −0.63 λ
shortfall larger than three seed sds — and **seed variance lives almost entirely
in short words** (≤3 sd 0.509 against ≥4 sd 0.049), the same stratum as Phase P's
missed corollary and the λ sweep's largest movement.

**Export gates.** Every fp32 export **100/100 argmax at the default 1e-3**
(he 3.57e-04, inside the envelope — the v2-era flag does not recur); fp16w
≤ 0.01 t1 everywhere. Standard 1,142,727-byte resbn80 graphs, fp16w 589,406 B;
hashes in `PHASE_Q.md` §7.7. **Generation 4 (`*_synth_v3_ch80*`) supersedes
v2full for deployment on all six scripts**; every earlier generation stays in
the registry with the numbers that were measured on it. **Nothing in
`artifacts/` derives from Yandex** — the twin generator, its samples and its
decoder live untracked under `~/ctc-train/research_only/`, permanently
unshippable per `YANDEX_LICENSE_RESEARCH.md` §8.1 / `PHASE_Q.md` §0.

## 5. Configurations — mixes, pairs and blends, with their member composition

A **configuration** is not a model: it is a named set of members plus a rule for
combining them. The campaign produced two combination rules and one packaging
rule:

* **prob averaging (the `mix2` contract)** — per-frame **arithmetic** mean of
  the members' emission probabilities (`logsumexp − log N`) **before the single
  beam**, via `eval_beam.py --onnx a,b[,c] --ens-avg prob`. Normalized by
  construction.
* **logprob averaging** — per-frame **geometric** mean, i.e. mean of
  log-emissions renormalized per frame. Required renormalization because the
  beam's `len^γ` term is not invariant to a per-frame additive constant.
  Empirically catastrophic on same-recipe seeds.
* **additive rerank** — `final' = beam_score + w · ranker_score` over the beam's
  top-k slate, `w` frozen at 0.05. Not a beam union; the candidate set is
  unchanged.

**Nothing in this section is test-validated.** Every configuration is val-only,
permanently.

| configuration | phase | member composition (exact) | rule | combined bytes | combined latency | val-9918 t1/t3/t5/≤3/4+ (E1, AOSP) | dvorak (app) / euro-mean | agreement gate | bars | source |
|---|---|---|---|---|---|---|---|---|---|---|
| **`mix2-i8f16`** ← the incumbent recorded ship configuration, "the card" | K | `artifacts/phaseK_sw2345_s1234_int8w.onnx` (**1,554,355 B**) **+** `artifacts/phaseK_resbn192i_s1234_fp16w.onnx` (**3,052,318 B**) | prob | **4.45 MB** ✓ ≤5 MB | **1.79 ms** encoder = 0.930 + 0.858, two sequential ONNX sessions, beam unchanged — MEASURED | **88.68 / 92.61 / 93.46 / 91.30 / 87.32** — **the first entry ever to beat the val ≤3 bar** (+0.03) | **91.94** (**91.53**) / **84.64 [derived** 84.93/82.81/81.22/89.59**]** | 96.9 % | **11/11** + size + latency | `PHASE_K.md` §8.2 |
| `mix2` s1234, fp32 + fp32 | K | `sw2345_s1234` fp32 + `resbn192i_s1234` fp32 | prob | 12.14 MB ✗ | not separately measured | **88.66 / 92.63 / 93.42 / 91.41 / 87.23**, greedy 68.12 | 92.27 (91.66) / **84.71 [derived** 85.12/82.90/81.22/89.59**]** | 96.9 % | 11/11 accuracy, **fails ≤5 MB** | `PHASE_K.md` §4.2, §4.4 |
| `mix2` s1234, int8w + int8w | K | `phaseK_sw2345_s1234_int8w` + `phaseK_resbn192i_s1234_int8w` (1,554,355 B each) | prob | **3.11 MB** — the largest size margin | not separately measured | 88.65 / 92.64 / **93.45** / 91.30 / 87.27 — all five val bars | qwertz misses | 96.9 % | **10/11** | `PHASE_K.md` §4.6, §8.2 |
| `mix2` **s4321** pair | K | `sw2345` s4321 + `resbn192i` s4321 | prob | — | — | 87.46 / 91.80 / 92.83 / 90.00 / 86.14, greedy **19.84** | — | **88.8 % ✗** | **0/5 val** | `PHASE_K.md` §4.3 |
| `mix2` **s7777** pair | K | `sw2345` s7777 + `resbn192i` s7777 | prob | — | — | **88.57 / 92.72 / 93.49 / 91.30 / 87.15**, greedy 61.35 | not recorded | 96.1 % ✓ | all five val bars; layout battery not recorded | `PHASE_K.md` §4.3 |
| `mix2` **cross-seed control** | K | `sw2345` s1234 + `resbn192i` **s7777** | prob | — | — | 86.75 / 91.58 / 92.55 / 89.55 / 85.30, greedy **9.27** | — | **83.3 % ✗** | 0/5 — **failed by design**, the alignment hypothesis's predicted failure | `PHASE_K.md` §4.3 |
| `mix2` **s5555** pair — the blind-gate confirmation | K | `phaseK-sw2345-s5555` + `phaseK-resbn192i-s5555`, both fresh 188 k trainings at a never-used seed | prob | not recorded | not recorded | **88.72 / 92.73 / 93.45 / 91.18 / 87.44**, greedy 68.40 | 91.58 (91.05) / **84.31 [derived** 84.02/83.15/80.63/89.42**]** | **97.0 %, measured and committed BEFORE any decode** (git `3156080`) | **10/11** — all six layouts, ≤3 −0.09 | `PHASE_K.md` §8.5 |
| `mix3` s1234 | K | `sw2345` s1234 + `resbn192i` s1234 + **`ch256-p65`** | prob | **≈ 11 MB ✗** | not measured (estimated 2–3 × 0.84 ms) | **88.88 / 92.73 / 93.35 / 91.53 / 87.50**, greedy **74.85** — the best val t1 and ≤3 of any Phase-K configuration | 92.02 (91.33) / **84.67 [derived** 85.02/82.73/81.17/89.76**]** | pairwise edges 95.5 / 96.2 / 96.9 | **11/11 accuracy, size NOT met** | `PHASE_K.md` §4.3, §4.4 |
| `sw2345` ×3 seeds, **logprob** | K | `sw2345` s1234 + s4321 + s7777 | logprob | — | — | **77.02 / 90.94 / 92.32 / 78.52 / 76.24**, greedy **29.26** | — | — | **0/5 val — REFUTED.** No alt-layout battery was run for it, so an eleven-bar tally is not measurable from the source | `PHASE_K.md` §4.1–§4.2 |
| `sw2345` ×3 seeds, **prob** | K | same three | prob | — | — | 87.12 / 91.96 / 92.74 / 90.23 / 85.51, greedy 37.05 | — | — | **0/5 — REFUTED** | `PHASE_K.md` §4.2 |
| `resbn192i` ×3 seeds, prob | K | `resbn192i` s1234 + s4321 + s7777 | prob | — | — | 87.39 / 91.76 / 92.73 / 90.73 / 85.65, greedy 64.11 | — | — | **0/5 — REFUTED in both families**: same-recipe seeds do not share a CTC alignment | `PHASE_K.md` §4.2 |
| **`v2pair-s1234` i8f16** ← ship option A | L | `artifacts/phaseL_v2pair_s1234_a_int8w.onnx` (**member A, int8w, 1,554,355 B**, sha `01580189…`) **+** `artifacts/phaseL_v2pair_s1234_b_fp16w.onnx` (**member B, fp16w, 3,052,318 B**, sha `59f40d95…`) | prob | **4.39 MB** — 60 KB *less* than the card; the byte sum 1,554,355 + 3,052,318 = **4,606,673 B [derived]** | 1.79 ms as quoted in `PHASE_M.md` §11.2 with **no protocol cited** | **88.86 / 92.82 / 93.59 / 91.56 / 87.46**, greedy 72.71 — **≤3 +0.29 over the campaign bar is the largest ≤3 margin the campaign recorded** | **92.88** (**92.59**) / **85.14 [derived** 84.11/84.41/82.26/89.76**]** | 98.33 % | **11/11 campaign**, 10/11 vs card (azerty −0.82) at **one seed** | `PHASE_L.md` §12, §16.1 |
| `v2pair` three-seed pair aggregate | L | the fp32 pairs of `v2pair-s1234`, `-s4321`, `-s7777` | prob | — | — | seed-mean **88.83 / 92.75 / 93.49 / 91.49 / 87.45** | 91.60 (91.21) / **85.03 [derived** 84.88/84.16/81.89/89.19**]** | 98.05–98.33 %, **6 of 6** over the gate | **11/11 campaign on EVERY SEED**; **7/11 vs card** at the seed-mean, per-seed 10/8/6 → **bar 1 NOT met** | `PHASE_L.md` §15.2–§15.3 |
| `v2pair` five-seed pair aggregate | M | + `v2pair-s5555`, `v2pair-s9999` | prob | 4.39 MB | — | 5-seed mean **88.776 / 92.724 / 93.458 / 91.436 / 87.390** | 91.339 (90.956) / **84.93 [derived** 84.766/84.128/81.764/89.078**]** | 8 of 8 pairs over the gate across five seeds and two data mixes | **11/11 on 5 of 5 seeds**, margins +0.124 … +2.756; vs card per-seed [10, 8, 6, 4, 8] | `PHASE_M.md` §7.2 |
| `v2pair-pw0` — **selected pair** | L | the trainer's own selected members of the uncoupled control, at its 136 k gate-passing eval | prob | not recorded | — | 88.09 / 92.59 / 93.32 / 91.12 / 86.52, greedy **53.12** | 89.34 (88.81) / 84.68 [derived] | **95.32 % (marginal PASS)** | 7/11 campaign, 1/11 card | `PHASE_L.md` §8–§10 |
| `v2pair-pw0` — **own-best mix** | L | member a @172 k + member b @164 k of the same uncoupled run | prob | not recorded | — | **t1 87.64, greedy 29.10** — against members whose solo greedy is 72.6 / 71.8 | — | **91.30 % FAIL** | — | `PHASE_L.md` §8, §10 |
| `phaseL_memberA_s1234_fp16w` | L | one member (member A of `v2pair-s1234`), fp16w | single-model packaging | **3,052,318 B (2.91 MB)**, sha `127874dd…`; fixture `phaseL_memberA_fp16w_golden.json` 140,225 B, sha `7c3948c6…` | not recorded | see §3 | see §3 | — | **11/11 seed-mean at 3 seeds → 9/11 at 5 (RETRACTED)** | `PHASE_L.md` §16.1; `PHASE_M.md` §7.1 |
| `sw2345` + `phaseK_ranker_sw2345_2seed` | K | `sw2345` encoder + a 21,782-byte rescorer | additive rerank, w = 0.05 | encoder + **21.8 KB** | + ~0.3 ms feature work | seed-mean t1 88.51 → **88.59**, t3 +0.04, t5 +0.02, ≤3 +0.04 (**91.24 — still 0.03 under the bar**), 4+ +0.11 | — | — | not tallied; **rescorer × alt-layout interaction is UNMEASURED**, and that measurement is required before any stacked-configuration claim | `PHASE_K.md` §5.1 |
| `resbn192i` + `phaseK_ranker_resbn192i` | K | the incumbent + its own symmetric rescorer | additive rerank, w = 0.05 | encoder + 21.8 KB | + ~0.3 ms | 88.32 → **88.58** (+0.26); ≤3 91.21 → 91.35 | — | — | exists to enforce the symmetric-application rule; **the rescorer shifts the field, not the ranking** | `PHASE_K.md` §5.1 |
| rescorer stacked on `mix2` | K | the mix2 ensemble + a rescorer | prob + rerank | — | — | "**t3/t5 tenths**, flat on t1/≤3 — the ensemble already harvests the ranker's signal"; **no numeric table was recorded** | — | — | — | `PHASE_K.md` §8.1 |

**Why the pair configuration is nonetheless better-founded than the card**
(`PHASE_L.md` §11.1, `PHASE_M.md` §11.2): `mix2-i8f16` was explicitly a
configuration **whose recipe does not reproduce** — its s4321 twin fails the
gate at 88.8 % agreement and collapses to greedy 19.84, and the ≥95 % gate was
derived post-hoc (then vindicated prospectively on s5555). The Phase-L pair
**reproduces by construction**: pair compatibility is *trained in*, not gated
for — six of six coupled pairs cleared the gate at 98.05–98.33 % (eight of eight
by the end of Phase M), against the paired `--pair-weight 0` control's 92.09 %.

---

## 6. The opponent — FUTO reference rows

**"Floor" and "ceiling" are not published-vs-tuned.** They are
**encoder-only vs encoder+decoder**, orthogonal to the preset axis:

* **config A (floor)** — `honorable_sturgeon` encoder alone, decoded by a
  *textbook* logaddexp CTC prefix beam over a log-freq lexicon trie
  (`futo_decoder_eval.py`).
* **config B (`beamB`)** — the same encoder-only emissions decoded by **FUTO's
  own single-stream Viterbi trie beam** (MAX-merge, length-aware prune). This
  isolates the beam lever.
* **config D (ceiling)** — encoder **+ `magic_macaw` DFSMN refinement** +
  the same Viterbi beam (`futo_decoder_ceiling.py`). "Full ceiling".

Shared artifact facts. Bytes, sha256 and provenance are hash-verified against
FUTO's own `metadata.json` (`FUTO_WEIGHTS_VERIFICATION.md` §1); the **parameter
counts below are `THREEWAY_AUDIT.md` §3**, which that file does not record.
Encoder `honorable_sturgeon/model_fp32.pte`
**635 K params / 2,649,856 B**, sha `725242ba…`; decoder
`magic_macaw/model_fp32.pte` **304 K params / 1,247,468 B**, sha `01eaf16a…`;
**total 939 K params / 3,897,324 B**; export 2026-04-20, git
`86b375fbc0ad76fd6cc421b09f28a110c4e98367`. **No on-device latency figure
exists for FUTO's engine in either repo** — its evals ran uninstrumented in a
proot container; the only speed number is host throughput, **26.8 traces/s
(floor) / 23.7 traces/s (ceiling)** (`FUTO_WEIGHTS_VERIFICATION.md` §2),
single-threaded, x86_64, ExecuTorch 1.2.0 — and that is **Python-beam-bound,
not model-bound**.

| reference row | config | preset | params / bytes | val-9918 t1/t3/t5/≤3/4+ | test-2400 t1/t3/t5/≤3/4+ | trie | role | source |
|---|---|---|---|---|---|---|---|---|
| FUTO **floor**, published | A | `encoderOnly` `γ 0.4056 / λ 0.0176 / β 0.9866` — **config A's textbook logaddexp beam uses no prune pair**; the `0.4234 / 1.0382` pair belongs to config B (`FUTO_WEIGHTS_VERIFICATION.md` §3) | 635 K / 2,649,856 | **78.84 / 88.01 / 90.11 / 81.17 / 77.62** (reproduced 78.82 / 88.00 / 90.10 / 81.20 / 77.58) | **79.25 / 87.71 / 89.58 / 82.45 / 77.60** (DROP 131,544; greedy-CTC t1 **43.96 as committed, 43.83 on the reproduction** — the one metric of this row that did not reproduce to the digit); on AOSP STRIP **79.29 / 87.96 / 89.88 / 82.58 / 77.60** | AOSP STRIP / DROP | the G2 feasibility anchor of Campaign 1 | `FUTO_WEIGHTS_VERIFICATION.md` §4a–§4c; `THREEWAY_AUDIT.md` §1–§2 |
| FUTO floor, `beamB` | B | same three, **plus** the prune pair `γp 0.4234 / βp 1.0382` | same | **78.59 / 88.15 / 90.24 / 81.35 / 77.16** | STRIP **79.08 / 88.50 / 90.33 / 81.84 / 77.67** | AOSP STRIP | isolates the beam lever: **B − A = −0.29 pt t1 as committed** (−0.25 on the reproduction), both on the DROP trie; the STRIP figures in this row give −0.21 — FUTO's bespoke beam is neutral-to-negative on top-1 | `FAIR_REMATCH.md` §3; `FUTO_WEIGHTS_VERIFICATION.md` §4a, §4c |
| FUTO floor, **val-tuned** | B | swept by *our* grid on FUTO's own emissions → **γ 0.35, λ 4.8, β 1.6, γp 0.05, βp 1.4** (interior after **three** widenings — four grids, 85.53 → 85.74 → 85.97 → 85.97) | same | **85.97 / 91.18 / 92.12 / 89.11 / 84.35** (Δ +7.38 / +3.03 / +1.88 / +7.76 / +7.19) | **85.79 / 91.62 / 92.29 / 88.47 / 84.42** | AOSP STRIP | **the val-tuned encoder alone (85.97) beats the published encoder+decoder ceiling (85.54)**. Caveat on record: this sweep **may not be exhausted** — it was still creeping when it went interior | `FAIR_REMATCH.md` §3, §7 |
| **FUTO ceiling, published — THE BAR** | D | `encoderDecoder` **`0.5949 / 0.0134 / 0.7271 / 0.1902 / 1.2727`** | 939 K / 3,897,324 | **85.52 / 91.54 / 92.80 / 89.29 / 83.57** (reproduced here 85.54 / 91.52 / 92.78 / 89.29 / 83.60 — within ±0.03). On the **app 98,081** trie: **85.59 / 91.82 / 93.20 / 89.05 / 83.80** | **84.83 / 91.04 / 92.08 / 89.57 / 82.40** on DROP 131,544 — **reproduced exactly, every digit**; greedy-CTC t1 **69.12**. On AOSP STRIP: 84.92 / 91.38 / 92.42 / 89.94 / 82.33. On the **app 98,081** trie: **84.92 / 91.54 / 92.96 / 89.57 / 82.52** | DROP / AOSP STRIP / app | **footing-A and footing-B bar.** Verification verdict: "the campaign's bar numbers are CONFIRMED on this hardware, with FUTO's genuine published weights, verified by hash" | `FUTO_WEIGHTS_VERIFICATION.md` §4a–§4c, §6; `PHASE_F.md` §15.2, §0 |
| **FUTO ceiling, val-tuned — the equal-footing bar** | D | swept by the same grid that produced E1 → **γ 1.15, λ 1.3, β 0.2, γp 0.3734, βp 0.7** (interior) | 939 K / 3,897,324 | **87.48 / 92.31 / 93.03 / 89.76 / 86.29** — Δ from tuning **+1.94 / +0.79 / +0.25 / +0.47 / +2.69**; generalizes (+1.89 on the fitted rows, **+1.97 on the 4,959 never seen**) | **87.12 / 92.29 / 92.96 / 89.94 / 85.68** — from a **real per-row decode**, not the analytic path. On DROP: 86.46 / 91.58 / 92.21 / 88.96 / 85.17 | AOSP STRIP | **footing-C bar — the only footing on which a superiority claim is admissible.** At matched tuning `magic_macaw` is worth **+1.51 val / +1.33 test**, not +5.9 (`FAIR_REMATCH.md` §3) | `FAIR_REMATCH.md` §2, §3, §4 |
| control: FUTO ceiling at **our E1 preset** | D | `1.05 / 1.1 / 0.2 / 0.3734 / 0.9882` | — | full-val t1 **87.48** (holdout half 86.89) — the G2 row of the sweep; t3/t5/≤3/4+ not recorded | **87.21 / 92.25 / 92.71 / 89.94 / 85.80** — marginally **better on t1 than FUTO's own swept optimum** | AOSP STRIP | a sensitivity control, not a bar. Note the wide-grid optimum on FUTO's emissions (γ 1.05, λ 1.1, β 0.2) is *exactly* our E1 triple | `FAIR_REMATCH.md` §2, §4 |
| **FUTO's own paper numbers** (as they claim them) | their production stack | theirs | — | — | **92.54 / 93.30 top-1 on their own test split** (*FUTO Swipe: Layout-Agnostic Neural Swipe Decoding*, Miller & Kostarevas, arXiv 2606.25247 — bibliographic data from `DATASET_SCOUT.md` and `ALT_LAYOUT_EVAL.md` §8, not from the accuracy cites) | their own | **not comparable to any row above** — different split, different code. Our port scores their engine at 84.83–87.12, so **every FUTO number in this file is a conservative estimate of that engine** | `FUTO_PRESET_NOTE.md` §Caveats; `FAIR_REMATCH.md` §7; `MODEL_COMPARISON.md` §6.8 |
| their claimed decoder contribution | — | — | — | — | **+0.55 – 0.76 pt** for `magic_macaw` — against the +5.88 their published presets imply and the +1.33–1.51 measured at matched tuning | — | the discrepancy that motivated the whole rematch | `FAIR_REMATCH.md` §3; `FUTO_PRESET_NOTE.md` |
| their cross-layout claims | — | — | — | — | Russian 40.5 → 77.2, ClearFlow 3.2 → 96.5 | — | **quoted, not relied on** — no copy of the paper is in this repo and the figures were never verified locally | `ALT_LAYOUT_EVAL.md` §8 |
| `hungry_jellyfish` (FUTO's context LM) | — | — | **not recorded** | never run | never run | — | downloaded and **not used** — our eval rows mostly lack the preceding-word context it consumes. Consequence: **the bar is a floor on FUTO's full published stack, not a ceiling** | `FUTO_WEIGHTS_VERIFICATION.md` §1; `FAIR_REMATCH.md` §7 |
| FUTO on any non-QWERTY layout | — | — | — | — | — | — | **NOT RECORDED.** No alternate-layout FUTO evaluation exists in any document here — that no such run was ever made is an **inference from the absence of a row**, not a statement either doc makes. The alt-layout comparator is the shipped geometric SHARK2-family engine, not FUTO | `ALT_LAYOUT_EVAL.md` §6, §9 |

**The non-FUTO comparators, for completeness:**

| reference | t1 / t3 / t5 / ≤3 / 4+ on test-2400 | val-9918 | alt-layout t1 | source |
|---|---|---|---|---|
| old shipped CleverKeys NN (transformer, `swipe_encoder_android` + `swipe_decoder_android`, d_model 256, 6 enc + 4 dec, **10,293,047 B**, ~178 ms/trace on device, full pipeline) | **74.62 / 84.33 / 87.42 / 89.45 / 67.00** (its own 98,140-word dict, production beam-6) | 76.01 / 85.53 / 87.82 / 89.23 / 69.15 | — | `THREEWAY_AUDIT.md` §1–§3 |
| shipped geometric engine (SHARK2 family, pure JVM, no NN) | 67.50 / 78.88 / 81.79 / 69.33 / 66.56 | 67.69 / 78.36 / 81.49 / 70.23 / 66.37 | dvorak **76.8** / azerty 76.9 / qwertz 76.2 / german 71.1 / spanish 73.9 → euro-mean **74.53 [derived]** — and it is **self-tuned** while every CTC alt-layout number is at an en_qwerty-fitted preset, so all CTC deltas against it are **floors** | `THREEWAY_AUDIT.md` §1–§2; `ALT_LAYOUT_EVAL.md` §6, §9 |

---

## 7. Footings legend

### 7.1 The three cross-engine footings

`MODEL_COMPARISON.md` §0, whose first line is the reason this legend exists:
*"Almost every wrong comparison in this project comes from putting two footings
in one column. There are three, and they are not interchangeable."*

| footing | our preset | FUTO's preset | trie | what a win means |
|---|---|---|---|---|
| **A — published bar** | val-tuned (E1) | FUTO's published | AOSP STRIP **146,964** | our tuned engine beats their untuned one. The tuning lever alone is worth **+2.29 pt t1 to us** and **+1.94 pt t1 to them** — both **val-9918** figures (`AUDIT_FINAL.md` §6.1; `FAIR_REMATCH.md` §2); on test the FUTO lever is **+2.20** (`FAIR_REMATCH.md` §4). |
| **B — shipping / trie-matched** | val-tuned E1, or the app preset where one was fitted | FUTO's published, **re-measured on the same app trie** | app `en_enhanced.json` STRIP **98,081** | our shipping configuration clears the bar *on the lexicon users actually run*. Still tuned-vs-published: **not** an equal-footing claim. |
| **C — equal footing** | val-tuned (E1) | **val-tuned by the same wide grid on the same val rows** | AOSP STRIP 146,964 | a genuine engine-vs-engine comparison. **The only footing on which a superiority claim is admissible** — held, qualified, by ch 192 and by `phaseM_kd_fresh_w1`. |

A → B changes **the lexicon only**; A → C changes **the preset only**. They are
**not composable**: no val-tuned FUTO bar exists on the app trie, so footing B
is permanently asymmetric and `PHASE_G.md` §7.3 declared that in advance.

### 7.2 The bars, so no table has to restate them

| bar | t1 | t3 | t5 | ≤3 | 4+ | source |
|---|---|---|---|---|---|---|
| test-2400, **published** (FUTO ceiling, published preset, DROP 131,544) | 84.83 | 91.04 | 92.08 | 89.57 | 82.40 | values `RESULTS.md` §"Verified test-2400 results"; the DROP-131,544 attribution `FAIR_REMATCH.md` §4 / `PHASE_F.md` §15.2 |
| test-2400, **trie-matched** (app `en_enhanced` 98,081) | 84.92 | 91.54 | 92.96 | 89.57 | 82.52 | `PHASE_F.md` §15.2 |
| test-2400, **equal-footing** (FUTO ceiling val-tuned, STRIP) | 87.12 | 92.29 | 92.96 | 89.94 | 85.68 | `FAIR_REMATCH.md` §4/§5 |
| val-9918, **published** (FUTO ceiling, published preset, STRIP) | 85.52 | 91.54 | 92.80 | 89.29 | 83.57 | `PHASE_F.md` §0 |
| val-9918, **trie-matched** (app 98,081) | 85.59 | 91.82 | 93.20 | 89.05 | 83.80 | `PHASE_F.md` §15.2 |
| val-9918, **equal-footing** (FUTO ceiling val-tuned, STRIP) | 87.48 | 92.31 | 93.03 | 89.76 | 86.29 | `FAIR_REMATCH.md` §2 |

**The eleven campaign bars** — a *different kind of bar*: the previous best
CleverKeys model's seed-mean on each axis, which a challenger must **beat**, not
tie. All are `resbn192i` Phase I-A seed-means. The two kinds may never be mixed
in one column: an internal bar says "better than our last model", a FUTO bar
says "better than the external reference".

| axis | bar | axis | bar |
|---|---|---|---|
| val-9918 t1 | 88.30 | dvorak (held out of training) | 89.13 |
| val-9918 t3 | 92.60 | dvorak, app-98k trie | 88.20 |
| val-9918 t5 | 93.26 | azerty | 83.60 |
| val-9918 ≤3 | **91.27** | qwertz | 82.50 |
| val-9918 4+ | 86.77 | german | 79.64 |
| | | spanish | 88.28 |

Counted **separately**, not in the 11-bar tally: **Cyrillic in-dict t1 76.21**
(`phaseIB-ru-synth`, app-ru 50 k trie, E1, real Yandex val rows, **eval-only**)
— corrected to **≈ 77.4 at λ = 2.0**, which lifts challenger and bar equally —
and the **size / latency gate** (≤ 5 MB, < 50 ms).

**Seed-mean vs every-seed.** The campaign's bars are stated as **seed-means over
1234 / 4321 / 7777**, with every-seed "preferred". The two disagree sharply:
`sw2345` is 10/11 seed-mean and **5/11 every-seed**. Any "N/11" quoted without
its footing is under-specified — `AUDIT_PHASEJ.md` §7 recommends the fuller
phrasing, and its corrections list makes attaching the footing a required change.

### 7.3 Presets

| name | γ, λ, β, γ-prune, β-prune | where it applies |
|---|---|---|
| published `encoderOnly` | 0.4056, 0.0176, 0.9866, 0.4234, 1.0382 | Campaign 1 and Phases A–D (**every absolute number there is understated by 2–5 pt**) |
| published `encoderDecoder` | 0.5949, 0.0134, 0.7271, 0.1902, 1.2727 | the FUTO ceiling bar |
| **E1** — the benchmark preset | **1.05, 1.1, 0.2, 0.3734, 0.9882** | every CleverKeys number from Phase E on, on the AOSP trie. **It transferred unchanged for five model families in a row** — a symmetric stratum-aware sweep landed both the Phase-J finalist and the incumbent back on their own E1 numbers — `PHASE_J.md` §6.8b summarizes that as "within ±0.07 on every metric", though its own table's largest deviation is **0.13** (`sw2345` 4+ 87.13 vs 87.26). Either way it is the strongest evidence that E1 is a property of the emission/trie pair rather than of any model |
| **app preset** | **0.9, 4.0, 0.25, 0.25, 0.9882** | the shipping footing. Fitted on `resbn80g` (`PHASE_G.md` §6) and never swept for **this** family, and now test-validated on `phaseM_kd_fresh_w1` as config B. (`resbn192i` did receive its own app-footing sweep — next row — so "never swept again" would be wrong.) λ moves 1.1 → 4.0 because the app trie's `log_freq` spread is 0.64 against AOSP's 5.40 — an 8× scale collapse |
| `resbn192i` app preset | 0.975, 3.0, 0.35, 0.25, 0.9882 | holdout-confirmed for `resbn192i` only (`PHASE_I.md` §7.4) |
| ru | λ ≈ **2.0** | the Cyrillic path only — the app ru lexicon stores `freq = 255 − rank`. Does **not** touch E1 on the en footings. **Re-swept on generation 4 and left unchanged** (`PHASE_Q.md` §9.7): on the real ru probe's tune half, in-dict t1 is monotone decreasing across {1.1 … 4.0} (85.65 → 80.69), so λ\* = 1.1 is a grid endpoint, the pre-registered interior-optimum rule refused adoption, and the confirm half was not spent. The constant stands with a **measured, unconfirmed −0.63 t1 shortfall** — λ 2.0 was fitted to a greedy-37 model and generation 4 reads greedy 66 |

**The λ 4.0 caveat that travels with the app preset:** the user-dictionary merge
injects top-of-scale (freq 255) competitors and a 3.6× larger λ amplifies them.
**No evaluation anywhere in this campaign includes a user dictionary.** This is
the one preset risk the sweep could not price.

### 7.4 Metric definitions, constant everywhere

Beam width **100**, top-k **8**, **OOV against the engine's own lexicon counted
as a miss** (86 rows under config A, 64 under config B on test-2400; 336 vs 250
on val-9918). Strata split at **n = 815 / 1,585** on test-2400 and
**n = 3,389 / 6,529** on val-9918. Per-source halves: test-2400 is 1,217 FUTO
rows / 1,183 How-We-Swipe rows; val-9918 is 4,942 / 4,976. Alt-layout uses the
`az26` arm, in-dict protocol (OOV excluded from the denominator), at E1.

### 7.5 Caveats that travel with every number in this file

1. **Preset asymmetry** — the largest threat on footings A and B; on val-9918
   ~2.3 pt to us and ~1.9 pt to FUTO (on test the FUTO lever is +2.20). Roughly **two thirds of the published test margin on t1 was
   an artifact of comparing a tuned preset against an untuned one**.
2. **Contributor contamination** — T3 applies no session or participant
   exclusion; every contributor of every val and test row is in training, and
   3× HWS oversampling triples the exposure of the more contaminated corpus.
   **These are benchmark numbers, not a generalization claim about an unseen
   user.** The counter-asymmetry runs in FUTO's favour: 5,273 of 12,299 unique
   holdout traces (43 %) are bit-exactly in the HF *train* split FUTO trained on.
3. **The dedup defect** — 588 val / 145 test rows sat in `train_t3` with
   bit-identical tensors. Key fixed, tiers deliberately not rebuilt; measured
   effect is *negative* (leaked rows score 4.34 pt **below** comparable
   non-leaked ones).
4. **A worn split** — test-2400 has been read four times, and 7 traces are
   bit-exactly shared between val-9918 and test-2400 (0.29 %).
5. **The per-source spread** — the campaign's aggregate top-1s are averages of a
   **~94–95** on the FUTO half and a **~80–82** on the How-We-Swipe half
   (`MODEL_COMPARISON.md` §6.4); early-phase arms sit far wider (Phase A spans
   92.59 / 69.21 to 88.00 / 76.29). The narrowest spread ever recorded **on
   test-2400** is **11.97**, by the ship model at the shipping footing.
6. **Latency is not cross-comparable across engines.** Ours is encoder-only,
   single-thread batch-1 laptop x86 through the Python ORT binding; the old NN's
   committed figures are a full decode pipeline on a phone at 4 threads; FUTO
   has no figure at all. **No on-device latency exists for any CTC model** — the
   instrumented benchmark is written and packaged but its run was blocked.
   And **the beam, not the encoder, is the budget**, paired per runtime: WASM
   beam **1.5 ms** against an encoder of 0.8–1.5 ms; desktop-JVM beam **7.3 ms**
   against an encoder of 0.26–0.55 ms.

---

## 8. Appendix — run inventory and artifact hashes

### 8.1 Every run directory under `~/ctc-train/ckpt/`

147 run directories exist. Those **promoted to `ctc/artifacts/`** are listed in
§8.2; everything else is local-only and is registered in §4 with the best value
its phase document recorded. Exported-graph sizes below are `stat` measurements
taken 2026-08-15 on `~/ctc-train/ckpt/<run>/ctc_swipe_encoder.onnx` — they are
file facts, not new evaluations, and they are given because several phase
documents do not print a byte count for their own arms.

| exported fp32 bytes | runs with that graph |
|---|---|
| 1,619,140 | `r2`, `phaseA-T0`, `phaseA-T1`, `phaseA-T1strict`, `phaseA-T2`, `phaseA-T2b`, `phaseA-T2-s4321`, `phaseC-C1`, `phaseC-C2`, `phaseC-C2-s4321`, `phaseC-C3`, `phaseD-D0`, `phaseD-T1bridge` |
| 1,676,794 / 2,332,892 / 2,404,624 | `phaseB-B1` / `phaseB-B2` / `phaseB-B3` |
| 2,332,892 | `phaseD-D2` |
| 2,799,865 | `phaseD-D1{,-last,-s4321,-s7777}`, `phaseD-D3`, `phaseD-T1bridge128{,-s4321,-s7777}`, `phaseE-E3a-T4`, `phaseE-E3b-hws3x{,-s4321,-s7777}`, `phaseE-E5base` |
| 6,144,249 | `phaseE-E4-ch192{,-last}`, `phaseE-FINAL-s{1234,4321,7777}` |
| 413,034 – 1,142,727 | the Phase-F ladder (see §4.6 for the per-arm mapping) |
| 1,142,727 | all Phase-G ch-80 arms, all `phaseH-p*`, all four `phaseIB-*` en arms, both `phaseIB-ru-*`, `phaseI-sel80` |
| 2,762,279 / 6,068,519 / 10,685,479 | `phaseI-ch128` / `phaseI-ch192{,-p65,-p65-s4321,-p65-s7777}` / `phaseI-ch256{,-s4321,-s7777}` |
| 1,150,923 | `phaseI-t64-80` (T′ = 64 head) |
| 6,068,519 | every Phase-J ch-192 arm, `phaseJ-ru192` (as `ctc_swipe_encoder_ru.onnx`), the Phase-K ch-192 arms, and all three `v2kd-fresh-w1*` |
| 6,076,715 | `phaseK-t64` (T′ = 64, contract-v2) |
| 6,068,519 × 2 (as `a.onnx` / `b.onnx`) | every `v2pair-*` run — the pair trainer writes two member graphs, so these dirs have no `ctc_swipe_encoder.onnx` |
| no export | `r1`, `r2-refine` (writes `ctc_refine_head.onnx`, 63,617 B), `r3-ch128`, `refine-unfreeze`, `smoke`, `phaseE-E2-refine`, `phaseE-E4-ch192-s4321` (killed at step 9 k), `phaseJ-cr256`, `phaseJ-sw234-snap` (killed), `phaseJ-smoke{1..4}`, `v2smoke`, `v2smoke0` |

### 8.2 Committed artifacts and their sha256 (`ctc/artifacts/`, verified 2026-08-15)

| file | bytes | sha256 |
|---|---|---|
| `ch128_s1234.onnx` | 2,799,865 | `6c1144949e545f626419e1fa7b29e80f9ecf3e303886f30411fc37ae72c45c51` |
| `ch128_s4321.onnx` | 2,799,865 | `1eac209332fe6fd52eb7edf2ce52ae77a52552956fdfe7f333d74f2cf46ecce6` |
| `ch128_s7777.onnx` | 2,799,865 | `8e910571b748290cb09fdd09e5531cc2aad6d5c09c7fd9d83d57c84ad67dda8b` |
| `ch192_s1234.onnx` | 6,144,249 | `d5b5f10ea16f08743d0742b3c60aa37a469ada11c418a7f459d5ae4cff20c666` |
| `ch192_s4321.onnx` | 6,144,249 | `b020b841abfb011779e2584e418cc651bfcac988a06bfcff2aeea5862bfabab3` |
| `ch192_s7777.onnx` | 6,144,249 | `a182191152ad77b233a73bc79750b0dda51bdbcf7fcb76ddaaad6d17016eee79` |
| `ctc_swipe_encoder.onnx` ⚠ pre-campaign `r2` | 1,619,140 | `fcf1633167b10f5c28e7c4dc16a9bba178bacc9e2b76efb06d792162dc99d0b7` |
| `fast_resbn56_188k_s1234.onnx` | 609,445 | `ecd317b4ab0b40673f760c7cdae8eb65f55f15c2e5c90c278daa07ae434f779b` |
| `fast_resbn64_188k_s1234.onnx` | 766,727 | `0a773948b1195436897a19b1f3824433cc2a72dd9bdd71f4fbd23574d87836c3` |
| `fast_resbn72_s1234.onnx` | 944,487 | `6567366b61bbbd04b5353f7f780aedb9aa507f7a87f52a381089cb54bf510985` |
| `fast_resbn72_s4321.onnx` | 944,487 | `0697af644212d09ec5592b0c4018f4b98089abc72a7a4b66df2f9d4d6cd5fa7f` |
| `fast_resbn72_s7777.onnx` | 944,487 | `02c20784287aa831e20835ec391fd91c9144b5d12f6e798d711eff51d9ae4f7b` |
| `fast_resbn80_s1234.onnx` | 1,142,727 | `5e8c88756cbad5a5a8b8b3f289a990174fa6f3b6edfead46d8dbdb2927fb06f2` |
| `fast_resbn80_s4321.onnx` | 1,142,727 | `ca7a670095dae41ed441eaca22cd0a5be6cdd620826f1d1bc0b49c0d9f72a35d` |
| `fast_resbn80_s7777.onnx` | 1,142,727 | `a0d0c894a1cfd616f939644cd9c63cbe5910c3846ca2b542e55b43d2f278f4d0` |
| `resbn72g_s1234.onnx` | 944,487 | `30b5f3de7831d8137d2e0a9403f3d93ec5b22524db0fba1d76729ab9b09d8043` |
| `resbn72g_s4321.onnx` | 944,487 | `b5ad0911db7ee47c0c6da7c668c62a69eb76b30ab3477053029f9a54c473b987` |
| `resbn72g_s7777.onnx` | 944,487 | `b232a158c620b70e59a2f6d30746f9305231d33af7f0196d48f88879dc1248a2` |
| `resbn80g_s1234.onnx` | 1,142,727 | `330cadfbaa7334eaeaeab93762084181b70710fe9d59cbd69600a6de468fe1a0` |
| `resbn80g_s4321.onnx` | 1,142,727 | `c9379c60a23bec4ca300512d2930b7a724aad91b761597972446a6577f5d5bab` |
| `resbn80g_s7777.onnx` | 1,142,727 | `3e303d46abaff4bfe31779de35fb9fc81e63f1ae8fd5ab554a9db205f167191a` |
| `resbn80h_s1234.onnx` | 1,142,727 | `3e215438f3c8fae1f249b91be3986bc30c027920f158371acaea0d159dbeff00` |
| `resbn80h_s4321.onnx` | 1,142,727 | `b3f30bcd33cd1137300b039ae166ccd9bdd7ea9117502c35f9d0d80d9a277331` |
| `resbn80h_s7777.onnx` | 1,142,727 | `1a1edac6f10f0fd88b427ce41b4808e46bef1e4209b4611dc7c9e81b5e5e94dd` |
| `resbn192i_s1234.onnx` | 6,068,519 | `7436fdd2e1e29a930b02a93c09f993d75c4aa20087fbf5abe55e09b6594f7358` |
| `resbn192i_s4321.onnx` | 6,068,519 | `cfeebdaac76df3a3c02a34a91f8dca5ca5b37a19792e6a965769d88a743c1df7` |
| `resbn192i_s7777.onnx` | 6,068,519 | `adbab6c4dcb3544011cc11b217b05837085ef488370970bddd7acea89b8dc42b` |
| `resbn192i_s1234_fp16w.onnx` | 3,052,318 | `d55624cc5b53edce8fd8b24750c6f09d5c116edd8de911eef9f232cd16a84613` |
| `resbn256i_s1234.onnx` | 10,685,479 | `db5dfc771f00a90e4bda70730bf217514c168af519d41b176fbcaec95a0f7cd9` |
| `sw2345_s1234.onnx` | 6,068,519 | `96dd27ece698fa981530639700e66e0689acd2d3f024ad214e8a79b3fa083a30` |
| `sw2345_s1234_fp16w.onnx` | 3,052,318 | `2e820c121fc69ae95a9b2e22444fe14c47f5c5253df4696a0d0a432e364fc7b8` |
| `phaseK_sw2345_s1234_int8w.onnx` | 1,554,355 | `9a8edefa3ed4d8dd26eba7871670aeff231f5a50eeb22c0dce3ed5e443a86bf9` |
| `phaseK_resbn192i_s1234_fp16w.onnx` | 3,052,318 | `d55624cc5b53edce8fd8b24750c6f09d5c116edd8de911eef9f232cd16a84613` — **byte-identical to `resbn192i_s1234_fp16w.onnx`** (same hash; two names for one file) |
| `phaseK_resbn192i_s1234_int8w.onnx` | 1,554,355 | `ce225924f7601ab7889edb348808905d03ea596185166a5444dcadaba66736a3` |
| `phaseK_slw2_s1234.onnx` | 6,068,519 | `54ff81f022d96639a24516d83f8321e3b7bf16df95e4dcbd5894d63dd1387eb7` |
| `phaseK_t64_s1234_contractv2.onnx` | 6,076,715 | `747718419ff910b504ad919494252344be85918e6da4413493cf79215270d7cf` |
| `phaseK_ranker_sw2345_2seed.onnx` | 21,782 | `b8add7523fd504b621de2a939322abb693a1dbd47b6cb62d7755910611efae71` |
| `phaseK_ranker_resbn192i.onnx` | 21,782 | `11775853b5cc76173f67787bed8bcb548a6332e90d4604322f927636af4384da` |
| `phaseL_memberA_s1234_fp16w.onnx` | 3,052,318 | `127874ddef80a7eb847d8321e75fe78af1b1a8ba6298d3dd2994b04e10116a16` |
| `phaseL_v2pair_s1234_a_int8w.onnx` | 1,554,355 | `015801894968b1bb4a9da691a9e0a0d46b3156be4255ce9cafee5bfb8bead7c4` |
| `phaseL_v2pair_s1234_b_fp16w.onnx` | 3,052,318 | `59f40d95604969914e4307a6d5c9129c804b576e75db8d68b8737d74c71b2db7` |
| **`phaseM_kd_fresh_w1_s1234_fp16w.onnx`** ← **SHIP** | 3,052,318 | `84718e6ebc8020176f27b9668e50922a765c96838307b640a8db9ab0549e88e5` |
| `phaseM_kd_fresh_w1_s1234.onnx` (fp32, decoded) | 6,068,519 | `b71911da3407abc0b113bbc662a1929953b04dcaf7650d848a7e897605a9bf80` |
| `phaseM_kd_fresh_w1_s4321.onnx` (fp32, decoded) | 6,068,519 | `f7cb72c07e1d5a920e5ceb93b4f6cf241bf0c9dcc630bcd1117d4fdf38d2daf1` |
| `phaseM_kd_fresh_w1_s7777.onnx` (fp32, decoded) | 6,068,519 | `c55cc3b055cf2db2b198c03b3fae688aad1930058dfed3902296aa08fd6510d7` |

**Golden fixtures** (each records its own `source_onnx_sha256` and `preset`;
fixture and preset move together):

| fixture | bytes | sha256 | generated from / at |
|---|---|---|---|
| **`phaseM_kd_fresh_w1_fp16w_golden.json`** ← **SHIP** | 140,462 | `2a449c4f2de19505131b396655ae01d3e3c325e40249446ff6e7a40c2b27559c` | the ship fp16w artifact **at the app preset** `0.9/4.0/0.25/0.25/0.9882` |
| `ctc_model_golden.json` | 139,728 | `ce3b5456ad13543ac09ac8c2610374bd8847b15f740f9004a98efea59d74f134` | `resbn80g_s1234` at the app preset |
| `phaseL_v2pair_i8f16_golden.json` | 140,476 | `7440873afcce2b38dff7ee3cb130a3da8965c3f5a8654ff987ebe1fbdc8dc749` | the L1 pair, **averaged emissions**, E1 |
| `phaseL_memberA_fp16w_golden.json` | 140,225 | `7c3948c691447b3e901eefc1df58c3b5d34496405c5f7b91f6298890e7a184c2` | member A fp16w, E1 |
| `phaseK_mix2i8f16_golden.json` | 140,497 | `e3c2a351be195b6d08d424d6a1db0cf38622d2dda2ed8c546b12221aa32febeb` | the `mix2-i8f16` **averaged** head, E1 |
| `phaseK_t64_golden_contractv2.json` | 229,413 | `22124ebbcb5be3f5fc6174a05331ed68effb17954f1d26ced424106a8adf9a42` | `phaseK_t64_s1234_contractv2`, **frames = 64** |
| `sw2345_s1234_golden_CANDIDATE.json` | 140,098 | `b397715091b0ccb26be802842a6b3048efbeba7fbc3fd19572face62f12b47b7` | `sw2345_s1234`, candidate fixture (never promoted) |
| `ru_synth_ch80_fp16w_golden.json` | 160,876 | `041c20722a957d1341108eb969dc677a123363011094ad05b36fdc1baa1050b0` | `ru_synth_ch80_fp16w` on `ru_jcuken_default`, at the app's `tunedRuCkdt` preset (1.05 / **2.0** / 0.2 / 0.3734 / 0.9882), CKDT frequency scale — see §8.2b |

#### 8.2b The Russian export (2026-08-18)

`phaseIB-ru-synth` exported to the ship contract. **Evidence tier: val-only,
permanently; single seed (1234); license-clean synthesis training; Yandex =
eval-only per `YANDEX_LICENSE_RESEARCH.md`.** Full derivation in
`PHASE_I_DATA.md` §9; the audience-facing writeup is
`ctc-architecture-and-multiscript-guide.md` §4.

| file | bytes | sha256 |
|---|---|---|
| `artifacts/ru_synth_ch80.onnx` (fp32 source) | 1,142,727 | `d78a9fb9f8e170595a7714220cf5fd9dfc2324935900aec6cb6d7a2ec1a36666` |
| `artifacts/ru_synth_ch80_fp16w.onnx` (ship bytes) | 589,406 | `84ac284d4f0d0cb86061df9c557507e1489ab93a75b40885a4431976cee32469` |

The fp32 re-export is **byte-identical** to the artifact the 2026-08-09 training
run produced (`d78a9fb9…`, §8.2's "not committed" list) — a free determinism
check on the whole export path. fp16w decode is free: in-dict t1/t3/t5 on the
confirm half are identical to fp32 (**77.92 / 89.50 / 92.00**) despite a
1.16e-01 sliced emission residue and 2 argmax flips per 100 real traces.
Full-set in-dict t1 **77.41** (9,416 rows, 8,471 decoded, app-ru 50 k CKDT trie,
λ = 2.0). **This is the only non-Latin ONNX the campaign has produced that is
shippable at all** — `phaseIB-ru-real` (89.64) is Yandex-license-blocked
forever, and `phaseJ-joint` (78.23 confirm) was rejected on its −0.42 en
regression against a 0.3 tolerance.

Also present in the working directory but **not committed** (too large):
`resbn256i_s4321.onnx` sha `910ad2f138e1911f56c6965bce06338ef160b0bb4ca9977eade4dc8208eb40ec`
and `resbn256i_s7777.onnx` sha `5550d61c205bd2e75b0625a9f56397fdcc6463cbb1385204d9eb94c411dac06a`
(`PHASE_I.md` §8), plus the I-B arm exports (`phaseIB-nativeadv` `e024ac35…`,
`phaseIB-native` `0cd1771d…`, `phaseIB-quality` `f4c0500e…`, `phaseIB-ru-real`
`cb8ece6b…`, `phaseIB-ru-synth` `d78a9fb9…`) — abbreviated in
`PHASE_I_DATA.md` §3/§6 and not reproduced in full anywhere.

### 8.3 What this registry could not source

Recorded here so that a gap is never mistaken for a value:

* **fp16w / int8w byte counts** for every Phase-A→G model — the quantization
  levers did not exist until Phase I.
* **Latency for every Phase-L and Phase-M arm.** `PHASE_L.md` contains no ms
  figure at all; `PHASE_M.md` contains exactly one (option A's 1.79 ms) and
  gives it **no protocol, device, or measured/inherited label**. The ship
  model's "0.83 ms class" in §1–§2 of this file is **inherited** from
  `resbn192i`'s measured 0.831 ms on an identical graph, and is labelled as such.
* **Exact parameter counts** for `phaseK-t64`, `phaseK-sw2345-slw2`,
  `phaseK-sw2345-280k`, `phaseI-t64-80`, `phaseI-sel80`, every I-B arm, both
  Phase-L members, and every Phase-M pair — the documents give only the family
  and, for the ship model, "1.5 M" (`UNSEALING_4.md` §2.3 does pin the ship
  model at **1,512,802**).
* **Per-arm latency in Phases A (except T2), B (measured in C), D2/D3-class
  variants, G arms A/B/D/E, and H p15/p30** — never benchmarked individually;
  identical-graph inheritance is the only basis and is labelled where used.
* **Absolute val values for `phaseI-sel80`** — only deltas were recorded.
* **t3/t5/≤3/4+ for `phaseJ-cr256-p80`** — only t1 was recorded.
* **≤3/4+ strata for every Phase-A and Phase-C arm**, and for every I-B arm.
* **Alt-layout for `resbn80g`, `resbn72g`, `resbn64g`, `fast_resbn72`, every
  Phase-A→F arm except `ch128_s1234` and `fast_resbn80_s1234`,
  `phaseJ-futoaug`, `phaseJ-sw234-p80`, `phaseJ-ru192`, `phaseJ-joint`, the
  Phase-K s5555 members solo, and every rescorer-stacked configuration.**
* **Any on-device latency, for any model, ours or FUTO's.**
* **`resbn56x5`'s byte count**: `PHASE_F.md` §6 prints the round figure
  **720,000**; the exported graph measures **737,512 B**. Similarly `resbn68`
  prints 852,927 against a measured 853,047. Both are recorded as printed, with
  the measurement beside them.
* **The B5 qwertz regression magnitude** is stated inconsistently inside
  `PHASE_K.md` — −0.11 at §8.2 and −0.5 at §8.1. No third measurement resolves
  it; both are recorded as written.
* **ch 128's parameter count** is printed as both 689,282 and 685,090 inside
  `PHASE_F.md`. Both are recorded, each where its own document uses it.
* **`phaseK-t64`'s bar tally.** §4.11 copies `PHASE_K.md` §8.1's **8/11**
  faithfully, but that arm's own published numbers clear 3 val bars + 6 layout
  bars = **9/11**. The arithmetic error is in `PHASE_K.md`, not here, so the
  figure is carried as printed rather than silently corrected.
