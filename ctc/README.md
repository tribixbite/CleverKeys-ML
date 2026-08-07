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
