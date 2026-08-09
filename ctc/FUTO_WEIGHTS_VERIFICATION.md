# FUTO weights verification — re-running the bar on this machine

**Date:** 2026-08-08 · **Host:** x86_64 (Intel Core Ultra 9 275HX, 24 threads), Ubuntu
22.04 under WSL2 · **Verdict: the committed bar numbers are CONFIRMED.**

The campaign's accuracy bar (the "FUTO ceiling") was measured on an aarch64 phone inside a
proot Ubuntu sandbox. Every gate in Phase D/E/F is scored against it, so the bar itself is
load-bearing and had never been independently re-run. This document records an independent
re-execution of FUTO's *actual* published model weights, on this machine, against both
evaluation splits, using the same harness that produced the committed numbers.

---

## 0. License and scope statement (read first)

**FUTO's model weights are covered by the FUTO Model Weights License 1.0 and were used here
for BENCHMARKING ONLY.** Specifically, and without exception:

- The weights were **run** to produce accuracy measurements. Nothing more.
- **No output of a FUTO model entered any training loop.** No distillation, no
  pseudo-labelling, no data augmentation, no teacher signal of any kind.
- **No output of a FUTO model was saved as training data.** The per-trace prediction files
  live in a scratch directory outside both repositories and are not, and will not be,
  ingested by `prepare_data.py`, `build_tiers.py`, or any trainer.
- **No output of a FUTO model influenced model selection, checkpoint selection, scoring-preset
  selection, or any other decision about our models.** This exercise verifies numbers that
  were already frozen and already published; it is a measurement of the measuring stick, not
  an input to anything downstream.

**Seal scope.** Our campaign's test-2400 seal forbids decoding *our* models on test.
Running FUTO's fixed, third-party engine on test-2400 to reproduce an already-committed bar
is verification, not a decode of our models, and is explicitly authorized for this task.
**No decision about our models flows from the test-2400 numbers below.** Our own models were
not run at all during this exercise; `seal.py` and `eval_beam.py` were never invoked. The
app repo (`/home/will/git/swype/CleverKeys`) was treated as read-only reference; its scripts
were copied verbatim into a scratch tree and executed from there, unmodified.

---

## 1. Provenance and integrity of the artifacts

Downloaded fresh from the Hugging Face repo named in the app repo's eval notes
(`docs/eval/futo-decoder-eval-notes.md`), into `~/ctc-train/futo_verify/artifacts/`:

```
hf download futo-org/futo-swipe --local-dir ~/ctc-train/futo_verify/artifacts
```

| Artifact | Size (B) | sha256 | Documented value | Match |
|---|---|---|---|---|
| `honorable_sturgeon/model_fp32.pte` (encoder) | 2,649,856 | `725242bab5d14345e96ff214e8de2bfbc1f962c232d320df9c24cb82ffd1fbaf` | `2,649,856` / `725242ba…` | **YES** |
| `magic_macaw/model_fp32.pte` (decoder) | 1,247,468 | `01eaf16ac4bc0f1ed0698c240807f0e95e6d427bcf6de04983ffc50736744d85` | `1,247,468` / `01eaf16a…` | **YES** |

Both hashes also match the `file_hashes` block inside each model's own
`metadata.json`, which is a second, independent attestation shipped by FUTO. Both metadata
files record `git_commit 86b375fbc0ad76fd6cc421b09f28a110c4e98367` — the same commit cited
in the eval notes — and `export_timestamp 2026-04-20`.

Supporting artifacts:

| File | Provenance | Check |
|---|---|---|
| `scoring.json` | same HF repo | contents reproduced verbatim in §3; the harness's hard-coded constants match it exactly |
| `en_qwerty.json` (layout) | `ctc/en_qwerty.json`, from `gitlab.futo.org/keyboard/swipe-library` | sha256 `1965ecd59c9e4bff89446bb56ff3a2d0070b16eeae4ce424ce08b06ed6864632` — matches the `layout_sha256 1965ecd5…` recorded in `AUDIT_PREDECODE.md` §3g |
| `en_wordlist.combined` | `gitlab.futo.org/keyboard/latinime`, already present at `~/ctc-train/data/` | 165,544 `word=` entries — matches the documented count |
| `hungry_jellyfish` (context LM) | downloaded with the repo | **not used** — the paper does not evaluate it and the bar does not include it |

The wordlist normalizer counts also reconcile exactly with `AUDIT_PREDECODE.md` §4:
165,544 raw entries → **131,544** unique words under the pre-fix DROP normalizer (keep only
surface forms that are already pure a–z) and **146,964** under the post-fix STRIP normalizer
(strip apostrophes/hyphens to a–z). Both tries were built and used below, because the
committed test-2400 table and the committed val-9918 row were measured under *different*
normalizers — see §4.

---

## 2. Environment — ExecuTorch runs these `.pte` natively on x86_64

The eval notes claim the `.pte` require an aarch64 proot sandbox. That constraint was
device-specific (Termux/bionic), not architectural. On this box:

| Component | Version | Note |
|---|---|---|
| Python | 3.10.12 | venv at `~/ctc-train/futo_verify/etvenv` |
| `executorch` | **1.2.0** (`cp310-cp310-manylinux_2_28_x86_64`) | same version family as the phone run |
| `torch` | **2.11.0+cpu** | the ABI pin documented in the eval notes holds on x86_64 too |
| `numpy` | 2.2.6 | |

The system-wide `executorch 0.7.0` **cannot** load these files —
`XNNCompiler.cpp: Unhandled node type` / `Init failed for backend XnnpackBackend: 0x11`.
1.2.0 loads both cleanly. Undeclared runtime deps had to be installed by hand
(`ruamel.yaml`, `flatbuffers`, `pyyaml`, `tabulate`, `scikit-learn==1.7.1`, `torchao`);
without them the `_portable_lib.so` import raises a misleading "Prebuilt … is not found".

I/O signatures observed at load, matching the documented contract exactly:

```
encoder honorable_sturgeon  in: features[1,2,64], layout_keys[1,64,2], layout_mask[1,64] (bool)
                           out: log_emissions[1,32,65], coefficients[1,32,64], lambda[1,32,1]
decoder magic_macaw         in: decoder_input[1,32,92]   out: log_probs[1,32,27]
```

The `Encoder` class in `futo_decoder_eval.py` needed **no adaptation** — the ExecuTorch
Python API (`Runtime.get().load_program(path).load_method("forward").execute(tuple)`) is
unchanged between the aarch64 1.2.0 wheel and the x86_64 one.

**Throughput.** Single-threaded, one process: **26.8 traces/s** floor, **23.7 traces/s**
ceiling (encoder + decoder + width-100 trie beam; the Python beam, not the `.pte`, is the
bottleneck). Sharded 6-way (test) and 12-way (val) across the 24 threads, all six full runs
— 24,636 decodes — completed in **5 min 43 s wall / 73 min CPU**. Re-running one val shard
produced a **bit-identical** output file, so the pipeline is deterministic and every delta
below is genuinely cross-platform, not run-to-run noise.

---

## 3. Configuration — matched to the committed runs

Read straight out of FUTO's `scoring.json` and identical to the harness's hard-coded
defaults (`futo_decoder_ceiling.py:60-64`), which is how the committed numbers were produced:

| Config | Model path | gamma | lambda | beta | gamma_prune | beta_prune |
|---|---|---|---|---|---|---|
| **A — floor** | encoder only, textbook logaddexp CTC prefix beam | 0.4056 | 0.0176 | 0.9866 | — | — |
| **B — beamB** | encoder only, FUTO Viterbi trie beam | 0.4056 | 0.0176 | 0.9866 | 0.4234 | 1.0382 |
| **D — ceiling** | encoder + `magic_macaw`, FUTO Viterbi beam | 0.5949 | 0.0134 | 0.7271 | 0.1902 | 1.2727 |

Beam width **100**, top-k **8**, OOV target counted as a **miss** (never skipped) — all as
committed. Harness files copied verbatim from the app repo
(`futo_decoder_eval.py` md5 `0138ddfd…` verified identical to
`/home/will/git/swype/CleverKeys/scripts/futo_decoder_eval.py`); no script in either
repository was modified. Sharding used only the harness's own `--skip`/`--limit`, and every
record carries its absolute `idx`, so the merged files are index-complete.

---

## 4. Reproduced results vs the committed bar

### 4a. test-2400, DROP trie (131,544) — the exact configuration of the committed table

The committed test-2400 table in `docs/eval/2026-07-24-test2400-head2head.md:37-44` and
`docs/eval/futo-decoder-eval-notes.md:192-201` was measured **before** the contraction fix,
i.e. with the DROP normalizer. To compare like with like I rebuilt that trie without
touching the script, by pre-filtering the wordlist to the 134,906 lines whose surface form
is already pure a–z (→ 131,544 unique words, matching the documented figure).

| Metric | Committed | Reproduced | Δ |
|---|---|---|---|
| **CEILING (D) overall t1** | **84.83** | **84.83** | **0.00** |
| **CEILING t3** | **91.04** | **91.04** | **0.00** |
| **CEILING t5** | **92.08** | **92.08** | **0.00** |
| **CEILING ≤3 t1** (n=815) | **89.57** | **89.57** | **0.00** |
| **CEILING 4+ t1** (n=1,585) | **82.40** | **82.40** | **0.00** |
| ceiling ≤3 t3/t5 | 94.36 / 94.72 | 94.36 / 94.72 | 0.00 / 0.00 |
| ceiling 4+ t3/t5 | 89.34 / 90.73 | 89.34 / 90.73 | 0.00 / 0.00 |
| ceiling in-vocab t1/t3/t5 | 88.48 / 94.96 / 96.05 | 88.48 / 94.96 / 96.05 | 0.00 |
| ceiling greedy-CTC t1 | 69.12 | 69.12 | 0.00 |
| ceiling macro t1 (58 words ≥5 ex) | 91.39 | 91.39 | 0.00 |
| FLOOR (A) overall t1 | 79.25 | 79.25 | 0.00 |
| FLOOR t3 | 87.71 | 87.71 | 0.00 |
| FLOOR t5 | 89.58 | 89.62 | +0.04 |
| FLOOR ≤3 t1 / 4+ t1 | 82.45 / 77.60 | 82.45 / 77.60 | 0.00 / 0.00 |
| FLOOR in-vocab t1/t3/t5 | 82.66 / 91.48 / 93.44 | 82.66 / 91.48 / 93.48 | 0 / 0 / +0.04 |
| FLOOR greedy-CTC t1 | 43.96 | 43.83 | **−0.13** |
| FLOOR macro t1 | 83.28 | 83.28 | 0.00 |
| beamB (B) t1/t3/t5 | 78.96 / 88.17 / 90.12 | 79.00 / 88.17 / 90.17 | +0.04 / 0.00 / +0.05 |
| N / OOV / errors | 2400 / 99 / 0 | 2400 / 99 / 0 | — |

Per-lever decomposition also reproduces: lever 2 (FUTO Viterbi beam) **B−A = −0.25 pt**
(committed −0.29), lever 1 (`magic_macaw`) **D−B = +5.83 pt** (committed +5.88), total
**D−A = +5.58 pt** (committed +5.58, exact). The campaign's conclusion that the *decoder*
is the entire lever and the bespoke beam is ~neutral on top-1 is reproduced unchanged.

**Every published digit of the test-2400 ceiling — the five headline bar numbers, all six
stratum sub-metrics, in-vocab, greedy, and macro — reproduces exactly.** The two non-zero
deltas are the floor's top-5 (+0.04 = one trace of 2,400) and the floor's greedy-CTC
(−0.13 = three traces of 2,400), both discussed in §5.

### 4b. val-9918, STRIP trie (146,964) — the ceiling row that is the Phase-E/F bar

The committed val row (`docs/eval/2026-07-24-test2400-head2head.md:107-119`) was measured
**after** the contraction fix, with the STRIP trie — the same trie the current harness
builds by default, and the same one our own `eval_beam.py` uses.

| Metric | Committed (the bar) | Reproduced | Δ |
|---|---|---|---|
| **CEILING overall t1** | **85.52** | **85.54** | **+0.02** |
| **CEILING t3** | **91.54** | **91.52** | **−0.02** |
| **CEILING t5** | **92.80** | **92.78** | **−0.02** |
| **CEILING ≤3 t1** (n=3,389) | **89.29** | **89.29** | **0.00** |
| **CEILING 4+ t1** (n=6,529) | **83.57** | **83.60** | **+0.03** |
| ceiling macro t1 (276 words ≥5 ex) | 87.02 | 87.24 | +0.22 |
| FLOOR overall t1 | 78.84 | 78.82 | −0.02 |
| FLOOR t3 | 88.01 | 88.00 | −0.01 |
| FLOOR t5 | 90.11 | 90.10 | −0.01 |
| FLOOR ≤3 t1 | 81.17 | 81.20 | +0.03 |
| FLOOR 4+ t1 | 77.62 | 77.58 | −0.04 |
| FLOOR macro t1 | 78.05 | 78.00 | −0.05 |
| N / OOV / errors | 9,918 / — / — | 9,918 / 336 / 0 | — |

Stratum n's (3,389 / 6,529) and the macro word count (276) match the committed values
exactly, confirming the same rows and the same selection rule.

**All five bar metrics land within ±0.03 pt.** The largest deviation on any *bar* metric,
either split, is 0.04 pt.

### 4c. test-2400 at the STRIP trie (for completeness)

Not a committed table, but the app repo states the post-fix re-run left the overall numbers
"essentially unchanged (floor 79.25→**79.29**, ceiling 84.83→**84.83**)" while noting the
strata were never republished. Reproduced here:

| Config | t1 | t3 | t5 | ≤3 t1 | 4+ t1 | OOV |
|---|---|---|---|---|---|---|
| floor | **79.29** (committed 79.29, exact) | 87.96 | 89.88 | 82.58 | 77.60 | 86 |
| beamB | 79.08 | 88.50 | 90.33 | 81.84 | 77.67 | 86 |
| **ceiling** | **84.92** (committed "unchanged at 84.83", Δ +0.09) | 91.38 | 92.42 | 89.94 | 82.33 | 86 |

The post-fix floor reproduces to the digit. The post-fix ceiling is +0.09 pt (two traces of
2,400) above the "unchanged at 84.83" note — consistent with the same drift seen elsewhere,
and with that note being a rounded restatement rather than a republished measurement. This
row fills in the strata the app repo left unpublished.

---

## 5. Analysis of the deltas

**No delta exceeds 0.2 pt on any bar metric on either split.** The flag threshold set for
this task was not tripped. Two observations are worth recording anyway:

1. **The residual is a handful of individual traces flipping, not a systematic offset.**
   Every non-zero delta is an exact multiple of one trace: 0.04 pt = 1/2400, 0.13 pt =
   3/2400, 0.02 pt = 2/9918, 0.03 pt = 3/9918. The signs are mixed (+0.02 t1 but −0.02 t3 on
   the val ceiling; +0.03 ≤3 but −0.04 4+ on the val floor), which is the signature of
   near-tie reorderings, not of a featurization or configuration difference. A configuration
   mismatch would move whole strata coherently.

2. **The mechanism is XNNPACK kernel selection.** The `.pte` are XNNPACK-delegated; the
   delegate dispatches to NEON micro-kernels on the phone and AVX2/AVX-512 ones here. These
   accumulate float32 in different orders, so log-emissions differ in the last bits, and a
   trace whose top-2 beam candidates are within that margin can swap. The greedy-CTC number
   is the most exposed metric of all — it is a bare per-timestep argmax with no lexicon,
   beam, or score margin to absorb a tie — and it is indeed where the largest delta appears
   (floor greedy −0.13). Consistently, the *ceiling's* greedy is exact (69.12), because the
   `magic_macaw` decoder's refined 27-class posteriors are far less tie-prone than the raw
   65-class encoder head. The pattern is internally coherent.

3. **The one number above 0.2 pt is a macro, and macro amplifies single traces.**
   Val ceiling macro is +0.22 (87.24 vs 87.02). Macro is an unweighted mean over 276
   per-word rates, so a single flip in a word with 5 examples moves it by
   (1/5)/276 = 0.072 pt. The +0.22 is ~3 net flips in low-count words — the same 2–3 traces
   already visible in the micro numbers, amplified ~10× by the small denominators. Macro is
   not one of the five bar metrics and no gate is scored on it. The test macros (both floor
   83.28 and ceiling 91.39, over 58 words) reproduce exactly, which would be very unlikely
   if there were any real divergence in the pipeline.

**Consequence for the gates.** The narrowest margin any of our models ever had against the
bar was Phase-E's t5 at **+0.28 pt** (93.08 vs 92.80); Phase-F's `resbn:72` cleared t5 by
**+0.16**. Measured bar drift on t5 is **0.02 pt** — an order of magnitude smaller than the
tightest margin. **No gate outcome recorded in `PHASE_E.md` or `PHASE_F.md` changes.**

---

## 6. Verdict

**The campaign's bar numbers are CONFIRMED on this hardware, with FUTO's genuine published
weights, verified by hash.**

- The committed **test-2400 ceiling (84.83 / 91.04 / 92.08, ≤3 89.57, 4+ 82.40)** reproduces
  **exactly, to every published digit**, including all stratum sub-metrics, in-vocab,
  greedy (69.12), and macro (91.39).
- The committed **test-2400 floor (79.25 / 87.71 / 89.58)** reproduces exactly on t1 and t3;
  t5 is +0.04 (one trace).
- The committed **val-9918 ceiling row (85.52 / 91.54 / 92.80, ≤3 89.29, 4+ 83.57)** — the
  bar all Phase-E/F gates are scored against — reproduces within **±0.03 pt on all five
  metrics**.
- The committed **val-9918 floor row** reproduces within ±0.04 pt on all five.
- The **per-lever decomposition** (beam ≈ neutral, decoder = the whole +5.6 pt) reproduces.
- The bar was **not** in fact locked to aarch64: ExecuTorch 1.2.0 has x86_64 wheels and runs
  both `.pte` natively, so the bar is now re-runnable on this box in ~6 minutes and no
  longer rests solely on the phone runs.

Nothing about the campaign's conclusions needs revision on account of the bar.

The claim in `AUDIT_PREDECODE.md` §5a that "no FUTO `.pte` encoder/decoder exists under
`/home/will`" and that a fair rematch "cannot be run here" is now **out of date**: the
weights are present and runnable. That audit's substantive point survives untouched, though
— it concerns the *preset asymmetry* (our E1 preset was grid-fitted on val-9918 rows while
FUTO's ceiling used its published preset), and that asymmetry is a question about tuning
fairness, not about whether the bar reproduces. This verification says the bar is the right
number for FUTO's published preset. It says nothing about what FUTO's engine would score
under an equivalent sweep, and no sweep of FUTO's preset was run — doing so would risk
turning a benchmarking run into an input to our own selection, which the license statement
in §0 forbids.

---

## 7. Reproduce

```bash
hf download futo-org/futo-swipe --local-dir ~/ctc-train/futo_verify/artifacts
sha256sum ~/ctc-train/futo_verify/artifacts/{honorable_sturgeon,magic_macaw}/model_fp32.pte

python3 -m venv ~/ctc-train/futo_verify/etvenv
~/ctc-train/futo_verify/etvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
~/ctc-train/futo_verify/etvenv/bin/pip install executorch==1.2.0 --no-deps
~/ctc-train/futo_verify/etvenv/bin/pip install ruamel.yaml flatbuffers pyyaml tabulate scikit-learn==1.7.1 torchao --no-deps

# harness copied verbatim from the APP repo, run from the scratch tree:
cp /home/will/git/swype/CleverKeys/scripts/futo_decoder_{eval,ceiling,ceiling_metrics}.py \
   ~/ctc-train/futo_verify/harness/

# ceiling, val-9918 (STRIP trie = the file as shipped)
OMP_NUM_THREADS=1 ~/ctc-train/futo_verify/etvenv/bin/python \
  ~/ctc-train/futo_verify/harness/futo_decoder_ceiling.py \
  --encoder .../honorable_sturgeon/model_fp32.pte --decoder .../magic_macaw/model_fp32.pte \
  --layout ctc/en_qwerty.json --vocab ~/ctc-train/data/futo_en_wordlist.combined \
  --test ~/ctc-train/data/val_hwsfuto.jsonl --out val_beamD.jsonl \
  --config beamD --beam-width 100 --top-k 8 --threads 1

# test-2400 at the pre-fix DROP trie: pre-filter the wordlist to lines whose `word=` value
# is already pure a-z (134,906 lines -> 131,544 unique), then pass that file as --vocab.
```

Full scratch tree, including per-trace prediction files for all nine runs, the shard driver,
and the logs: `~/ctc-train/futo_verify/` (deliberately outside both repositories, and — per
§0 — never to be used as training input).
