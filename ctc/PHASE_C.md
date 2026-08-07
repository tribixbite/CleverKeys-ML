# Phase C — training-procedure levers on T2

Phase B adopted nothing, so "the best B config" is **B0** — the unchanged Phase-A T2 arm,
which no B arm beat. Phase C therefore runs on B0's architecture (ch 96, residual trunk,
v1 features), tier T2, recipe otherwise frozen. That reading is stated up front because
the brief conditioned Phase C on Phase B landing positive; it did not, and running the
procedure levers on the surviving baseline is the useful action rather than stopping.

test-2400 was never decoded.

## 1. Arms

| arm | change |
|---|---|
| `phaseA-T2` (**B0**) | control |
| `phaseC-C1` | + path-only offset/scale jitter (σ_offset 0.02, σ_scale 0.05; keys untouched) |
| `phaseC-C2` | + EMA, decay 0.999, **evaluated and exported on the averaged weights** |
| `phaseC-C3` | C1 + C2, batch 1024, lr 6e-3, 94,000 steps (2× steps, 8× samples) |

All four are 394,114 params and export identically — every Phase-C lever is training-time
only, so **inference cost is unchanged**.

## 2. Results — frozen `enc` preset

| arm | greedy | val t1 | t3 | t5 | FUTO t1 | HWS t1 | **T2-clean t1** | Δ clean |
|---|---|---|---|---|---|---|---|---|
| **B0** | 59.62 | **80.86** | 88.52 | 90.48 | **92.59** | 69.21 | **79.99** | — |
| C1 | 60.00 | 80.48 | 88.32 | 89.98 | 91.76 | 69.27 | 79.66 | −0.33 |
| C2 | 60.32 | 80.70 | **88.93** | **90.60** | 91.36 | **70.12** | 79.91 | −0.08 |
| C3 | **63.40** | 80.47 | 88.29 | 89.92 | 92.39 | 68.63 | 79.53 | −0.46 |

On the clean-mask t1 the decision rule uses, **no arm gains**, so no rerun threshold is
even triggered and **no arm is adopted by the rule**. All three sit within −0.46 pt of
B0 — inside the ~1 pt seed-noise floor measured in §4 below, so none of these three
numbers is individually interpretable.

## 3. C1 is a clean negative, and that is worth knowing

The path-only jitter was designed against a specific, measured defect: the HWS half sits
~0.064 off the FUTO half in y against the same layout, and the existing shared affine
moves the path and the key centers *together*, so the model never trains on a path/layout
disagreement. The hypothesis was that training that tolerance would recover HWS accuracy.

**It moved HWS by +0.06 pt.** That is nothing. The 20+ pt per-source gap is therefore
**not** a path/layout registration problem that augmentation can absorb — it is a genuine
distribution difference in how the two populations swipe. This closes a hypothesis that
would otherwise have been worth several more experiments.

## 4. C2 (EMA): promising at one seed, downgraded by the second

At seed 1234 it is neutral on the aggregate and positive on exactly the axis the campaign
cares about — the only such result in either phase. The seed check below then removes most
of that claim, so read this table together with the one that follows it, not on its own.

| metric | B0 | C2 | Δ |
|---|---|---|---|
| clean t1 | 79.99 | 79.91 | −0.08 |
| clean t3 | 87.91 | **88.38** | **+0.47** |
| clean t5 | 89.94 | **90.10** | +0.16 |
| **HWS t1** (generalization gauge) | 69.21 | **70.12** | **+0.91** |
| FUTO t1 | 92.59 | 91.36 | −1.23 |

EMA trades in-distribution FUTO accuracy for out-of-distribution HWS accuracy. That is
the textbook signature of a flatter, lower-variance solution, and it is precisely the
trade the brief asked for — "the HWS-side number is our generalization gauge, and
robustness is where the win lives". The aggregate clean t1 hides it because that subset is
46 % FUTO and 54 % HWS, so the two moves nearly cancel.

C2 also improves t3 and t5 here, which the beam's top-3/top-5 suggestion slots consume
directly, and it costs **zero** inference time. (The t3/t5 gain does not survive the second
seed; the HWS gain does, at reduced size.)

### Seed confirmation — and the noise floor it exposed

C2 and a matched control were re-run at seed 4321. The gain does **not** hold cleanly, and
the reason is more important than the result.

| config | seed | clean t1 | clean t3 | HWS t1 | FUTO t1 |
|---|---|---|---|---|---|
| B0 | 1234 | 79.99 | 87.91 | 69.21 | 92.59 |
| B0 | 4321 | 78.94 | 87.78 | 68.57 | 91.14 |
| C2 | 1234 | 79.91 | 88.38 | 70.12 | 91.36 |
| C2 | 4321 | 79.24 | 87.43 | 68.79 | 91.38 |

Paired C2 − B0 by seed:

| metric | seed 1234 | seed 4321 | mean |
|---|---|---|---|
| clean t1 | −0.08 | +0.30 | **+0.11** |
| clean t3 | +0.47 | −0.35 | +0.06 |
| HWS t1 | +0.91 | +0.22 | **+0.57** |

Only the HWS direction keeps its sign across both seeds, and its mean gain is +0.57 —
below the +0.6 adoption bar even on the metric it helps. The t3 gain flips sign entirely,
so C2's apparent top-3 improvement at seed 1234 was noise.

> ### ⚠ The measured seed noise floor is ~1 pt, not ~0.4 pt
>
> The two B0 runs differ **only** in seed and land 1.05 pt apart on clean t1
> (79.99 vs 78.94), 1.45 pt apart on FUTO t1, and 0.64 pt apart on HWS t1.
>
> My Phase-A caveat put the resolution limit at ~0.4–0.5 pt, inferred from binomial
> standard error and checkpoint spread. That estimate was **too optimistic by roughly
> 2×** — it counted sampling noise on a fixed model but not run-to-run optimization
> variance. Corrected: **single-seed differences below ~1 pt on clean t1 are not
> interpretable.**
>
> This applies retroactively to the whole campaign:
> * **Phase C is entirely inside the noise floor.** C1/C2/C3 span −0.46 to −0.08; none of
>   it is resolvable. Phase C measured nothing at one seed per arm.
> * **Phase B's B2 (−1.31) is marginal**, only just outside it. B1 (−4.60) and B3 (−2.68)
>   remain solidly outside and stand as real regressions.
> * **Phase A's T0→T1 gap (+0.35)** and the T2-vs-T2b aggregate gap (0.96) are weaker than
>   reported. The T2-vs-T2b *FUTO-half* gap (1.71 pt) survives; the aggregate does not
>   clear the bar on its own.
> * The decision rule's +0.6 pt threshold sits **below** the noise floor of the single-seed
>   design it was applied to. Any future arm needs 2–3 seeds before it can clear a
>   sub-1-pt bar at all.

## 5. C3 shows the budget/batch scale-up is not free

C3 has the best greedy in the whole campaign (63.40) and the *worst* HWS number in Phase C
(68.63, −0.58 vs B0). 8× the samples at batch 1024 / lr 6e-3 buys a sharper model that
generalizes slightly worse — the same greedy-up/beam-down pattern Phase B found, now
reproduced by a pure optimization change with the architecture held fixed. Combined with
Phase B's B2, that is two independent demonstrations that **on this task, per-frame
emission sharpness and lexicon-decoded accuracy are not the same objective**, and the
larger-compute direction optimizes the wrong one.

C3 also inherits C1's jitter, which C1 showed to be worthless here, so part of its deficit
may be that rather than the batch/lr change; the two were not separated because the brief
specified them together.

## 6. Recommendation

* **By the decision rule: adopt nothing.** No arm cleared +0.6 pt clean t1; none was even
  positive.
* **C2 (EMA) is the only arm worth keeping on the table, but the two-seed check downgrades
  it from "the win" to "weakly promising".** Its HWS gain keeps its sign at both seeds but
  averages +0.57, under the +0.6 bar; its t3 gain was seed noise. It is free at inference
  and free to re-test, so it is the natural thing to fold into a multi-seed Phase D — not
  something to adopt on this evidence.
* **Drop C1's path jitter** — hypothesis tested and dead.
* **Do not scale batch/steps** on this recipe without re-checking the beam metric; C3 shows
  greedy improving while the shipping metric does not.
* Retire `best val_greedy` for checkpoint selection (carried over from Phase B; C3 makes
  the case again).
* **Run 3 seeds per arm from here on.** One seed cannot clear a sub-1-pt bar, so the
  single-seed design used in Phases B and C could not have detected the effect sizes it
  was asked to detect. This is the campaign's main methodological debt.

## 7. Winner, parity and latency

No arm displaced **B0 = `phaseA-T2`**, which therefore remains the campaign's best config
on the decision metric. Both it and the recommended-alternative C2 were re-exported and
parity-checked on the sliced contract view:

| config | params | sliced max abs err | argmax agreement |
|---|---|---|---|
| `phaseA-T2` | 394,114 | 2.48e-05 | 100/100 |
| `phaseC-C2` | 394,114 | 2.29e-05 | 100/100 |

Latency proxy: exported graph, ONNX Runtime CPU, **1 intra-/inter-op thread**, batch 1,
fixed shapes — one call per swipe, the shape the IME actually makes. 300 runs per round,
3 interleaved rounds so drift hits every config equally, best round reported, machine idle.

| config | params | mean ms | p90 ms | vs r2 |
|---|---|---|---|---|
| `r2` (shipped reference) | 394,114 | **0.306** | 0.319 | — |
| `phaseA-T2` (**winner**) | 394,114 | **0.307** | 0.320 | +0.3 % |
| `phaseC-C2` (EMA) | 394,114 | 0.308 | 0.319 | +0.7 % |
| `phaseB-B1` | 405,716 | 0.332 | 0.345 | +8.5 % |
| `phaseB-B2` | 570,818 | 0.453 | 0.468 | **+48 %** |
| `phaseB-B3` | 585,940 | 0.475 | 0.489 | **+55 %** |

The winner is latency-identical to the shipped r2 encoder, as it must be — same
architecture, different data. Worth recording that B2 would have cost **+48 % inference
time for −1.31 pt accuracy**, so it was never a live trade even before the accuracy result
came in; and every Phase-C lever is training-time only, so C2 would be free.

Absolute numbers are a desktop x86 core and are useful only as a *relative* proxy; a
phone's little core is several times slower. The encoder is not the bottleneck either way —
the 100-wide trie beam over 147 k words dominates the per-swipe cost.

## 8. Reproduction

```bash
python train.py --train-npz train_t2.npz --run-name phaseC-C1 --total-steps 47000 \
                --val-every 1500 --seed 1234 --path-offset-sigma 0.02 --path-scale-sigma 0.05
python train.py --train-npz train_t2.npz --run-name phaseC-C2 --total-steps 47000 \
                --val-every 1500 --seed 1234 --ema-decay 0.999
python train.py --train-npz train_t2.npz --run-name phaseC-C3 --total-steps 94000 \
                --val-every 3000 --batch 1024 --lr 6e-3 --seed 1234 \
                --path-offset-sigma 0.02 --path-scale-sigma 0.05 --ema-decay 0.999
python eval_arms.py --arms phaseA-T2,phaseC-C1,phaseC-C2,phaseC-C3 \
                    --own-mask T2 --also-masks T2 --rebuild-cache --latency-runs 200
```
