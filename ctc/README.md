# CTC swipe encoder — from-scratch training pipeline

Trains a layout-agnostic CTC swipe-emission encoder from scratch and exports it to
ONNX with the exact I/O signature the CleverKeys Kotlin decoder
(`src/main/kotlin/tribixbite/cleverkeys/swipe/ctc/`) already implements and
parity-tests, so the result drops into the `CtcEmissionModel` seam unchanged.

## Provenance

| Item | Source |
|---|---|
| Recipe | CleverKeys `docs/guides/train-ctc-swipe-model.md` @ `79ddfb0f` |
| `futo_decoder_eval.py`, `futo_decoder_ceiling.py` | vendored **verbatim** from CleverKeys `scripts/` @ `79ddfb0f` (files last touched in `f29e956b`) — the featurizer/trie/beam the Kotlin module is a port of |
| `en_qwerty.json` | verbatim copy of CleverKeys `src/test/resources/layouts/futo_qwerty.json` @ `79ddfb0f` (full-precision canonical centers) |
| Training data | `{train,val,test}_hwsfuto.jsonl` (110 876 / 9 918 / 2 400), derived from `futo-org/swipe.futo.org` (MIT) |
| Lexicon | AOSP-format `en_wordlist.combined` |

The two vendored harness files are byte-identical to their sources (no header was
added, so behaviour and any future `diff` against upstream stay clean):

```
eecfd2406f3fa021448a497519ab796377e830916ee8661a60be1ea755d57f46  futo_decoder_eval.py
ea6e184bf3925cdbcb5fa6f188de35592f3cab2ea3aa4e34c535370deef7ab1f  futo_decoder_ceiling.py
1965ecd59c9e4bff89446bb56ff3a2d0070b16eeae4ce424ce08b06ed6864632  en_qwerty.json
```

**Licensing (hard constraint, from the recipe §0):** train from scratch on the data
only. Never initialize from, fine-tune on, or distill FUTO's published weights or
*any* of their model outputs — the FUTO Model Weights License defines derivative
models to include anything trained on model outputs, which would re-import their
licence. Decode *algorithms* were ported from the GPL-3.0 `swipe-library` and are
already committed on the app side; only the I/O contract is copied here.

### How-We-Swipe full release (acquired, **not yet wired into any tier**)

`fetch_hws_full.py` reproduces the download + verification of the *complete*
How-We-Swipe release (OSF project [`sj67f`](https://osf.io/sj67f/), **MIT**,
© 2021 Leiva/Kim/Cui/Bi/Oulasvirta) into `~/ctc-train/data/hws_full/` — 1 GB,
~22 s on a gigabit link. Run `--analyze` for the participant/yield breakdown.

The pool used by T0 (`1 052` logs → `61 597` rows) is a strict **subset** of the
release's **1 338** participants; the missing 286 users are worth **+23 015** rows
(+37 %) under the identical filter. Every participant ships a per-user `.json`
carrying `englishLevel`, so native-speaker filtering needs no extra source:
413 native / 342 advanced / 363 intermediate / 219 beginner (1 `NA`), yielding
**29 853** native-only and **51 559** native+advanced rows across the full release.
Note `metadata.tsv` covers only the **909**-user subset the paper analysed, not all
1 338 — use the per-user `.json` for coverage.

## Audit fixes applied to the recipe

The recipe was audited before its first GPU run; the numbered fixes below are the
deltas baked into these scripts. Each is also flagged in the relevant source file.

| # | Fix |
|---|---|
| **1** | `export_onnx.py` asserted torch/ONNX parity over the raw 65-wide head, whose 38 pad columns sit at ≈ −1.0e4 where the float32 ULP is 9.77e-4 — the 1e-4 tolerance could never pass and failed deterministically. Parity is now asserted on the sliced `[32,27]` contract view (what `CtcEmissions.sliceFromHead` feeds the beam), plus argmax agreement on that view. |
| **2** | **The rank defect.** The recipe scored keys by sampling a *fixed* 8×8 cosine field at the key centers. At the 26 canonical en_qwerty centers that basis has **rank 23, not 26** (singular values 24–26 are exactly zero; rank 25 at `NUM_FREQ=9`, 26 only at `NUM_FREQ=10`). Since `lambda` is a per-frame scalar, three emission directions — concentrated on `d/f/g/h/j/k` and `e/r/t/y/u/i` — were structurally unreachable regardless of width, depth, epochs or data. `model.py` now embeds each key from its own geometry, `(cx, cy)` + the 64 cosine features → `Linear(66,96) → GELU → Linear(96,64)`, and scores with `coeff @ keyEmbed^T * lambda`. The exported `coefficients` head stays 64 wide so the phase-2 `[T',92]` contract is untouched. |
| **3** | The committed splits are **not disjoint**: 35 bit-exact traces (word + full x/y/t) are shared between train and test-2400 and 186 between train and val, plus 1 060 exact self-duplicates inside train. `prepare_data.py` now drops train rows whose content hash appears in val/test and train's own repeats. val/test are never filtered. |
| **5** | Checkpoints were model-only. They now carry model / optimizer / scheduler / step / epoch / best / best_epoch / args / RNG (torch, cuda, numpy, python) and are written atomically (tmp + `os.replace`); `--resume PATH` restores all of it and refuses an arch-defining mismatch (`ch`, `embed_hid`). |
| **7** | bf16 autocast dropped — measured 0.9 % per-frame argmax disagreement vs fp32 for zero benefit on a 0.39 M-param model whose epochs take ~4 s. Pure fp32. |
| **8** | Run isolation: `ckpt/<run-name>/{last.pt,best.pt,metrics.jsonl}`, one metrics line per epoch. Previously every iteration clobbered a single `ckpt/best.pt`. |
| **9** | The scoring `einsum` became a `MatMul` (folded into fix #2) — the exported graph now contains **zero `Einsum` nodes**, which are not XNNPACK-delegable and block fusion. |
| **10** | The recipe budgeted 5–10 min/epoch ("an evening" for 80 epochs); measured reality on an RTX 5080 Laptop is **~4 s/epoch** over the full split. Defaults are now `--epochs 300 --patience 40`. |
| **11** | `MAX_WORD_LEN = 28` allowed CTC-infeasible targets (`len + adjacent_repeats > 32`), which `zero_infinity=True` silently zeroes. Replaced by the exact feasibility rule (0 rows affected in these splits; it matters for the HF scale-up). |
| **12** | Warmup is `(step+1)/warmup`, so the first optimizer step no longer runs at lr 0. |
| **13** | The shared affine is rejection-sampled (≤10 tries, identity on failure) so all 26 transformed key centers stay inside `[0,1]`; centers are never clipped into their neighbours. The path is still clipped after noise. |
| **14** | `persistent_workers=True`; the `NpzFile` handle is closed after the arrays are copied out. |
| **15** | `eval_beam.py --out PATH` dumps per-trace `{idx, word, greedy, topk, rank}` JSONL for error analysis. Beam width stays 100 for baseline comparability. |
| **16** | Every script takes `--workdir` (default `~/ctc-train`) and resolves `data/`, `cache/`, `ckpt/` under it; the layout defaults to the `en_qwerty.json` beside the scripts; `export_onnx.py` gained argparse. Scripts run from any cwd. |
| **17** | Each `.npz` embeds a provenance JSON: source path/size/mtime, all row and drop counts, and the sha256 of the vendored featurizer and the layout file. |
| — | `torch.onnx.export(..., dynamo=False)` is now explicit (torch 2.9 flips the default to the dynamo exporter). |

Non-fixes worth knowing (audit findings #4, #6 also applied):

* **#6** the recipe's Appendix A layout rounded the canonical centers (`a.cx = 0.10`
  vs the real `0.10046728971962617`, `cy 0.1667` vs `1/6`). We vendor the committed
  layout file verbatim instead.
* **#4 (known issue, app-repo matter, deliberately NOT fixed here)** the recipe and
  `CtcParityTest` both reference `src/test/resources/ctc/ctc_golden.json` and
  `scratchpad/gen_ctc_golden.py` — **neither exists in the CleverKeys tree**, so
  `CtcParityTest` currently fails its own file-existence assertion and the
  "featurizer is bit-identical" guarantee is untested. `make_golden.py` here emits
  the matching schema; regenerating and committing that fixture is app-repo work.

## Workdir convention

Code lives in this repo; all runtime artifacts live under `--workdir` (default
`~/ctc-train`) and are never committed:

```
~/ctc-train/
├── data/
│   ├── {train,val,test}_hwsfuto.jsonl      # symlinks to the corpus
│   └── futo_en_wordlist.combined           # AOSP lexicon for the eval beam
├── cache/{train,val,test}.npz              # prepare_data.py output
├── ckpt/<run-name>/{last.pt,best.pt,metrics.jsonl}
├── ctc_swipe_encoder.onnx
└── ctc_model_golden.json
```

## Run order

```bash
python prepare_data.py                                     # cache + dedup + provenance
python train.py --run-name base                            # ~4 s/epoch, 300 epochs default
python eval_beam.py --ckpt ckpt/base/best.pt --test data/val_hwsfuto.jsonl   # G2 gate
python export_onnx.py --ckpt ckpt/base/best.pt             # + sliced parity check
python eval_beam.py --onnx ctc_swipe_encoder.onnx --test data/val_hwsfuto.jsonl
python eval_beam.py --onnx ctc_swipe_encoder.onnx --test data/test_hwsfuto.jsonl  # report once
python make_golden.py                                      # golden traces for Kotlin
```

Resume a run (the cosine horizon follows the new `--epochs`, so extending a finished
run re-warms the LR before decaying again):

```bash
python train.py --resume ckpt/base/last.pt --epochs 500 --run-name base
```

Use `--limit 2000` on `eval_beam.py` for quick iteration — prefix slicing is
distributionally representative (≤3-char share 35 % at N=120 vs 34 % over full test).

## Phase 2 — the refinement head (guide §11)

`train_refine.py` freezes a trained base encoder and trains our `magic_macaw`
analogue on top: per frame it consumes
`concat(sliced_emissions[27] | coefficients[64] | lambda[1]) = [32, 92]` and emits
refined `log_probs[32, 27]` that **replace** the emissions before the beam.
Head = `LayerNorm(92) → Linear(92,128) → GELU → Linear(128,27) → log_softmax`
(15.6 K params). CTC blank is **26** here — the head works on the already-sliced
view, not the 65-wide head.

**Canonical-QWERTY gating.** The base encoder is layout-agnostic because it trains
with slot permutation, but the sliced 27-class view is only the alphabet under the
canonical *identity* slot assignment. Refinement therefore trains with
`permute=False` (geometric jitter stays on), and the head — exactly like FUTO's
layout-fingerprint-gated `magic_macaw` — is valid for canonical QWERTY only. Other
layouts must fall back to the encoder-only path.

```bash
python train_refine.py --base-ckpt ckpt/r2/best.pt --run-name r2-refine   # ~2.7 s/epoch
python export_refine_onnx.py --ckpt ckpt/r2-refine/best.pt                # [1,32,92] -> [1,32,27]
python eval_beam.py --ckpt ckpt/r2/best.pt --refine-ckpt ckpt/r2-refine/best.pt \
       --test data/val_hwsfuto.jsonl                                      # G4 probe
```

`eval_beam.py --refine-ckpt/--refine-onnx` swaps the refined output in before
greedy/beam and auto-selects the `encoderDecoder` scoring preset (gamma 0.5949,
lambda 0.0134, beta 0.7271, gammaPrune 0.1902, betaPrune 1.2727); `--scoring
enc|dec` overrides. `--unfreeze-after N` optionally unfreezes the base at 0.1×
the head lr after epoch N (default off).

### Scoring sweep — **no free win; keep the published preset** (2026-08-07)

`sweep_scoring.py` grid-searches the beam's scoring params on **val** (test is never
touched). Two things make an exhaustive grid nearly free:

* Emissions are computed once by the ONNX encoder and cached as the sliced
  `[N,32,27]` contract view.
* The beam runs once per *prune* setting, not once per grid point. In
  `futo_viterbi_beam` the per-frame pruning key depends only on
  `(gammaPrune, betaPrune)`; `(gamma, beta, lambda)` enter solely through the final
  score `raw / max(L,1)**gamma + beta*L + lambda*log_freq`. So the **unmodified
  vendored beam** is called with `gamma=beta=lambda=0` and `top_k=beam_width`,
  returning every terminal-beam word with its raw path score, and the grid is then
  re-scored analytically. Verified exact: this path reproduces `eval_beam.py`'s full
  val-9918 numbers to the digit (81.57 / 89.84 / 91.37, strata 86.28 / 79.12) in
  seconds rather than 45 minutes.

Result on r2 (coarse 5×5×3 grid, then prune params ±0.05 with a local refinement):

| preset | sweep half (0:2000) | holdout half (2000:4000) | FULL val (9918) |
|---|---|---|---|
| published `encoderOnly` | 81.55 | 82.70 | **81.57** |
| tuned (γ 0.275, λ 0.026, β 0.84, γp 0.3734, βp 0.9882) | 82.00 (+0.45) | 82.75 (+0.05) | 81.56 (**−0.01**) |

The +0.45 pt on the rows it was fitted to is 0.5 standard errors (SE ≈ 0.87 pt at
n=2000), evaporates to +0.05 on untouched rows, and is −0.01 on full val.

**Headroom bound.** To settle whether *any* setting helps, a wider grid (9 γ × 9 β
× 5 λ = 405 points, plus 9 prune settings × 125 local points) was run directly on
all 9 918 val rows — i.e. selecting on the same rows it is scored on, an optimistic
upper bound rather than an honest estimate. The best point reachable that way is
**81.78 top-1 vs 81.57 baseline: +0.21 pt maximum**, and it costs −0.01 on both
top-3 and top-5 (it trades 4+-char accuracy 79.12 → 78.89 for ≤3-char 86.28 →
87.34). The two sweeps also disagree on which point wins (γ 0.275/β 0.84 vs
γ 0.4056/β 0.69), which is what a flat objective surface looks like.

**Verdict: keep `CtcScoringParams.encoderOnly` exactly as published.** It is already
at the optimum within noise for our emissions; the guide's "free win" does not exist
for this model, and the ≤0.21 pt that selection-on-val can manufacture is not worth
a preset divergence from the committed Kotlin default.

```
CtcScoringParams(gamma = 0.4056, lambda = 0.0176, beta = 0.9866,
                 alpha = 0.0, gammaPrune = 0.4234, betaPrune = 1.0382)  // unchanged
```

### Measured result on r2 — **G4 misses** (2026-08-07)

60 epochs, 2.7 s/epoch, head best val greedy **58.91 %** @ epoch 40 (base r2 was
58.00 %). G4 probe on the identical first 2 000 val rows:

| config | greedy | top-1 | top-3 | top-5 |
|---|---|---|---|---|
| enc-only r2 (enc preset) | 58.55 % | **81.55** | 89.85 | 91.65 |
| refined, `dec` preset (auto) | 59.15 % | **81.55** | 89.80 | 91.30 |
| refined, `enc` preset (control) | 59.15 % | **81.35** | 89.65 | 91.40 |

Delta vs the ≥ +4 pt bar: **+0.00 pt** (dec) / −0.20 pt (enc). Per-row churn is
28 fixed / 28 broken — net zero. By stratum (dec): ≤3-char 83.24 → **84.24**
(+1.00), 4+-char 80.65 → **80.11** (−0.54).

The likely reason the lever does not reproduce: FUTO's +5.88 pt came off a base
whose greedy was only 43.96 %, i.e. emissions with a lot of per-frame slack for a
refiner to recover. Our base already greedy-decodes at 58.0 %, so there is far
less headroom, and the head converges to roughly reproducing its input. The two
scoring presets bracket the result, so this is not a preset-mismatch artifact.

**End-to-end fine-tune probe** (`--unfreeze-after 10 --epochs 40`): unfreezing the
base at 0.1× the head lr pushed train CTC loss 0.316 → 0.276 but val greedy only
58.91 % → **59.16 %** (+0.25 pt, best @ epoch 22) — train loss improving while val
stalls is mild overfitting, expected once the slot-permutation regularizer is gone.
Below the +0.5 pt threshold agreed for a beam probe, so none was run.

**Phase 2 is closed.** Both the frozen head and the end-to-end fine-tune are nulls,
and the scoring sweep above shows the decode side is already at its optimum. If
phase 2 is ever revisited, the one untried structural change is temporal context in
the head — FUTO's `magic_macaw` is a DFSMN, ours is strictly per-frame by design.

## Contract the export must satisfy

| Tensor | Shape | dtype | Meaning |
|---|---|---|---|
| in `features` | `[1, 2, 64]` | float32 | resampled path, x row then y row, `[0,1]` |
| in `layout_keys` | `[1, 64, 2]` | float32 | key centers in emission-column order, pad `(0,0)` |
| in `layout_mask` | `[1, 64]` | bool | true for the K real key slots |
| out `log_emissions` | `[1, 32, 65]` | float32 | log-softmaxed; **blank at column 64** |
| out `coefficients` | `[1, 32, 64]` | float32 | spatial coefficients (phase-2 refinement input) |
| out `lambda` | `[1, 32, 1]` | float32 | per-frame positive gate (phase-2 input) |

Hard invariants: column `c < K` is `alphabet[c]` (alphabetical `a..z` for en_qwerty);
blank is at column 64, never 0; `log_emissions` is already log-softmaxed in-graph;
pad slots carry ~zero probability; featurization must be the vendored port.

## Baselines to beat (same 2 400-row test split)

| Engine | overall t1 | ≤3-char t1 | 4+-char t1 | greedy |
|---|---|---|---|---|
| FUTO ceiling (enc + refine) | 84.83 | 89.57 | 82.40 | 69.12 % |
| FUTO floor (enc only) — G2 reference | 79.25 | 82.45 | 77.60 | 43.96 % |
| CleverKeys shipped neural (beam 6) | 74.62 | 89.45 | 67.00 | — |
| CleverKeys geometric SHARK2 | 67.50 | 69.33 | 66.56 | — |

**Caveat on comparability:** the staged `en_wordlist.combined` yields a
**146 964-word** trie after a-z normalization, whereas the published baselines were
measured against a 131 544-word lexicon. A larger lexicon means more confusable
candidates, so numbers from this pipeline are, if anything, conservative relative to
the table — but for a formal G2/G4 claim the baseline lexicon should be matched.

## The scale-up campaign

**test-2400 remains sealed.** Only the pre-campaign `r2` run was ever decoded on
it (val 81.57 → test **80.96**, i.e. test runs ~0.61 pt *below* val); every phase
below is measured on val-9918 alone.

| doc | question | outcome |
|---|---|---|
| `DATA_TIERS.md` | provenance + contamination audit | splits are 49 % How-We-Swipe, not "held-out FUTO"; T0 has 75 % contributor overlap with the holdout |
| `PHASE_A.md` | which training pool | corpus **mix** dominates, not size; the quality cascade is net harmful |
| `PHASE_B.md` | architecture levers | all three regress; greedy and beam top-1 move in **opposite** directions |
| `PHASE_C.md` | training-procedure levers | all inside a seed-noise floor of **~1 pt**, 2× the previously assumed figure |
| `PHASE_D.md` | beam-selected checkpoints, the T3 benchmark tier, 3 seeds | **ch 128 adopted**; T3 (1.0 M rows) indistinguishable from T1; test gate **not** spent |
| `PHASE_E.md` | scoring re-tune, data mix, capacity, refinement head | **the published scoring preset was badly mis-tuned for our emissions** (+2.7–4.6 pt); 3× How-We-Swipe oversampling +0.83; the 3-seed stack **beats all five FUTO-ceiling numbers on val-9918**; test-2400 still **not** decoded |

Best measured configuration: `phaseE-FINAL` — ch 192 residual trunk, 1.525 M
params, T3 with its How-We-Swipe half oversampled 3×, 94 k steps, checkpoint
selected on beam top-1 over 5,000 val rows, decoded at the **Phase-E re-tuned
preset** (`gamma 1.05, lambda 1.1, beta 0.2, gammaPrune 0.3734, betaPrune
0.9882`). Seed-mean val-9918 **88.06 t1 / 92.32 t3 / 93.08 t5** (≤3 90.86,
4+ 86.62) over seeds 1234/4321/7777, at 0.898 ms single-thread CPU. The ch 128
variant scores 87.88 / 92.23 / 92.96 (≤3 90.98, 4+ 86.26) at 0.470 ms and clears
the same bar.

> ⚠ **The scoring-sweep section above is superseded by `PHASE_E.md` §1.** Its
> "no free win" verdict and its "+0.21 pt maximum headroom" bound were both
> artifacts of a grid centred on the published preset; re-swept wide on the same
> `r2` model the gain is **+4.25 pt top-1 on untouched val rows**. Every absolute
> accuracy number quoted elsewhere in this README is measured at the mis-tuned
> published preset and is understated by 2–5 pt. Arm-vs-arm comparisons are
> unaffected — the mis-tuning was common-mode.
