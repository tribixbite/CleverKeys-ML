# CTC Swipe Encoder — Training Results (2026-08-07)

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
