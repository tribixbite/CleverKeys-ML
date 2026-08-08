# Phase F — latency

Phase E ended with two configurations that clear all five FUTO-ceiling bars on
val-9918 and on the (now spent) test-2400 seal: ch 192 at **0.877 ms** and ch 128
at **0.455 ms** single-thread batch-1 CPU. Phase F asks a different question:
**how much of that time can be removed without falling back under the bar?**

The target set for this phase is **≤ 0.15 ms** — half the Campaign-1 `r2`
artifact's 0.306 ms, and about a third of the ch 128 ship candidate.

> ⚠ **The evidence in this document is val-only, and that is structural.**
> `test-2400` is **sealed-spent** (`AUDIT_FINAL.md` §7): the one legitimate decode
> was executed and audited, and no variant introduced after it may be measured on
> that split — `seal.py` enforces this by hashing the rows a run actually loads.
> Every Phase-F accuracy number is therefore full val-9918 at the E1 preset, and
> the bar it is compared against is the **val** bar. The test-validated anchors
> remain ch 128 (`phaseE-E3b-hws3x`) and ch 192 (`phaseE-FINAL`); a Phase-F
> artifact is a *val-validated* variant of them and must never be quoted as
> test-validated.

## 0. The bar, the preset and the measurement protocol

The bar is unchanged from Phase E §0 — FUTO's encoder+refinement ceiling decoded
on val-9918:

| metric | bar |
|---|---|
| overall t1 | **85.52** |
| t3 | **91.54** |
| t5 | **92.80** |
| ≤3-char t1 (n=3,389) | **89.29** |
| 4+-char t1 (n=6,529) | **83.57** |

Every accuracy figure below is **full val-9918** decoded at the **E1 preset**
(`gamma 1.05, lambda 1.1, beta 0.2, gammaPrune 0.3734, betaPrune 0.9882`), beam
width 100, the 146,964-word STRIP trie, through the *exported ONNX graph* — the
artifact whose latency is being measured, not a torch twin.

**Latency protocol** (`bench_latency.py`, the `AUDIT_PREDECODE.md` §7 protocol):
ONNX Runtime `CPUExecutionProvider`, `intra_op = inter_op = 1`, batch 1, fixed
shapes, 50 warmup calls then 3 rounds × 300 timed calls, reporting the mean and
p90 of the **best round**, machine idle.

**This harness reads ~3 % high against the numbers in `AUDIT_PREDECODE.md` §7**
(ch 128 measures 0.472–0.475 ms here against the audit's 0.455; ch 192 measures
0.911–0.934 against 0.877). The `r2` artifact measures **0.303 ms** here against
the 0.306 on record, so the scale is the same one the ≤0.15 ms target was set on.
All Phase-F comparisons are internally consistent because they all come from this
harness on this machine; where a Phase-E number is quoted it is re-measured here
rather than copied.

### What the metric does and does not include

A no-op ONNX graph carrying the exact production I/O signature
(`features [1,2,64]` + `layout_keys [1,64,2]` + `layout_mask [1,64]` in,
`log_emissions [1,32,65]` out, no arithmetic) measures the harness floor — the
Python binding, ORT's `Run()` dispatch and the input copies. That floor is
reported in §1 and is a **common-mode addition to every row of every table**,
including the Phase-D/E baselines the target was derived from. It is not
subtracted anywhere; it is stated so the numbers are not over-read as pure
arithmetic.

---

## 1. Where the 0.455 ms actually goes — profile first

`bench_latency.py --profile` runs the ch 128 ship candidate under ORT's own
profiler (300 instrumented calls) and aggregates the trace by op type. Profiling
instruments every kernel, so the instrumented total (561 us/call) runs ~20 % above
the un-instrumented wall time (0.463 ms in the same session); read the **shares**.

| op type | share | us/call | what it is |
|---|---|---|---|
| **`Conv`** | **66.1 %** | 371.3 | 8 dense 5-tap trunk convs (~46 us each) + the stem (4.8) |
| `Gelu` | 6.2 % | 35.0 | 9 activations, ~3.3 us each |
| `Mul` | 5.5 % | 30.9 | GroupNorm scale, the lambda gate |
| `Gemm` | 4.6 % | 25.9 | key-embed MLP (18.0) + coeff/lambda/blank heads |
| `Add` | 4.6 % | 25.8 | residual adds + GroupNorm shift |
| `Reshape` | 3.4 % | 19.4 | almost entirely the GroupNorm decomposition |
| `InstanceNormalization` | 2.1 % | 11.7 | the GroupNorm kernel itself |
| everything else (18 op types) | 7.5 % | 42.0 | Slice/Where/Concat/Squeeze/Split/Cos/Pad/LogSoftmax/CumSum/… |

The eight slowest nodes are the eight trunk convolutions, at 44.5–47.0 us each and
7.9–8.4 % apiece; the ninth is the key-embed `Gemm` at 9.3 us.

**Three things follow, and they set the whole phase.**

1. **The trunk convolutions are the target.** Two-thirds of the time is eight dense
   `Conv1d(ch, ch, 5)` calls. Nothing else can pay for a 3× reduction.
2. **`GroupNorm` is not the villain, but it is not free either.** ONNX has no
   GroupNorm before opset 21, so torch exports each one as
   `Reshape → InstanceNormalization → Reshape → Mul → Add`: **five nodes, nine
   times over**. Summing the shares plausibly attributable to it
   (`InstanceNormalization` + `Reshape` + roughly half of `Mul`/`Add`) puts the
   normalizer at **~10 %** of the graph. Worth removing, not worth a phase.
3. **Node count is a first-class cost at batch 1.** After ORT's own
   `ORT_ENABLE_ALL` pass the ch 128 graph still holds **142 nodes** for 32 frames
   of work; the smallest ops in the table run at 0.1–3 us, which is dispatch, not
   arithmetic. An architecture that does the same work in fewer nodes wins twice.

### The ORT-preoptimized graph is not a latency lever

Serializing the `ORT_ENABLE_ALL` graph offline (`--optimize-out`) and loading
*that* changes **session-load** time, not steady-state latency: ORT applies the
same passes either way, and the measured means are identical inside noise (§2).
It is reported because the brief asked, and because the serialized graph is the
right object to *count nodes* in — the as-exported file's 312 nodes include 113
`Constant` nodes that never execute.

### What node folding is worth, structurally

| trunk | optimized nodes | normalization nodes |
|---|---|---|
| `res` + GroupNorm (ch 128, the ship candidate) | **142** | 9 × (`InstanceNormalization` + 2 `Reshape` + `Mul` + `Add`) |
| `resbn` (dense conv + **folded** BatchNorm), ch 64 × 3 | **92** | **0** |
| `dwsep` (depthwise + pointwise + folded BatchNorm), ch 128 × 4 | **97** | **0** |

BatchNorm in `eval()` is a per-channel affine over running statistics, so it fuses
exactly into the preceding convolution — the fold is asserted numerically at export
(`max |Δlog_emissions| ≈ 1.5e-05`, `export_onnx.py`). Training still gets a real
normalizer; inference gets none. This is the change that makes the residual ~34 %
of non-`Conv` time tractable, and it costs nothing at all.

---

## 2. F1 — post-training quantization of the shipped models

`quantize_onnx.py`. Static calibration draws 1,024 random rows from the training
cache (`train_t3hws.npz`) and feeds them through the graph **at the canonical
layout**, one row at a time, which is the shape of call inference actually makes.
Weights are per-channel int8, activations uint8, QDQ format. Accuracy is full
val-9918 at the E1 preset, seed 1234, decoded through the quantized graph itself.

| arm | mean ms | p90 | bytes | t1 | t3 | t5 | ≤3 | 4+ | all five? |
|---|---|---|---|---|---|---|---|---|---|
| the bar | — | — | — | 85.52 | 91.54 | 92.80 | 89.29 | 83.57 | — |
| **ch 128 fp32** (`phaseE-E3b-hws3x`, s1234) | 0.475 | 0.492 | 2,799,865 | 88.02 | 92.27 | 93.03 | 91.12 | 86.41 | **yes** |
| ch 128 int8 **dynamic** | 0.471 | 0.487 | 2,737,242 | — | — | — | — | — | n/a |
| ch 128 int8 **static, whole graph** | 0.269 | 0.282 | 863,996 | **0.00** | 0.00 | 0.00 | 0.00 | 0.00 | **no** |
| ch 128 int8 static, tail fp32 | 0.279 | 0.294 | 918,415 | 86.12 | 91.53 | **92.61** | **89.08** | 84.58 | **no (t5, ≤3)** |
| ch 128 int8 static, tail+norms+stem fp32 | *(§2.3)* | | 905,293 | 87.04 | 91.98 | **92.72** | 90.09 | 85.46 | **no (t5)** |
| **ch 192 fp32** (`phaseE-FINAL-s1234`) | 0.934 | 0.951 | 6,144,249 | 88.22 | 92.23 | 93.08 | 91.15 | 86.71 | **yes** |
| ch 192 int8 **dynamic** | 0.906 | 0.927 | 6,043,994 | — | — | — | — | — | n/a |
| ch 192 int8 static, whole graph | 0.428 | 0.445 | 1,710,334 | **0.00** | 0.00 | 0.00 | 0.00 | 0.00 | **no** |
| ch 192 int8 static, tail fp32 | *(§2.3)* | | 1,801,552 | 86.77 | 91.71 | **92.62** | 89.32 | 85.45 | **no (t5)** |

### 2.1 Dynamic int8 is a null, and it was predictable

ORT's dynamic path quantizes `MatMul`/`Gemm`/`Attention`/`LSTM` only. On a graph
that is 66 % `Conv` there is nothing for it to do: **−0.8 % on ch 128, −3.0 % on
ch 192**, inside the measurement's own spread. (Asking it to quantize `Conv`
as well emits `ConvInteger`, for which this ORT build has no CPU kernel — the
session fails to load at all, so the arm is `MatMul`/`Gemm`-only by necessity, not
by choice.) No accuracy run was spent on it.

### 2.2 Quantizing the whole graph destroys the model — and the reason is structural

Both whole-graph static arms score **exactly 0.00** on all five metrics. This is
not a calibration failure that a better method would fix. Pad key slots are masked
with `MASK_NEG = -1e4` and the head is log-softmaxed **in graph**, so the tensor a
uint8 affine quantizer must cover spans roughly `[-1e4, 0]`. At 256 levels the step
is ~39 nats — and every real log-probability lives in `[-20, 0]`, i.e. inside a
single quantization bucket. The emissions come out constant and the beam returns
nothing. The contract's own `MASK_NEG` convention is what makes the tail
unquantizable.

Excluding the tail (key-embed MLP, the three heads, the scoring `MatMul`, `Where`,
`Concat`, `Softplus`, `LogSoftmax`) restores a working model at **−1.90 pt**;
additionally excluding the `GroupNorm` decomposition and the stem brings it to
**−0.98 pt**. Both still lose **t5** against the bar (92.72 / 92.61 vs 92.80) at
seed 1234, and the tail-only variant also loses ≤3.

### 2.3 Verdict on F1

Static int8 is a real **1.8–2.2× latency lever** (ch 128 0.475 → 0.269, ch 192
0.934 → 0.428) at ~⅓ the file size, but on these models it costs **1.0–1.9 pt
top-1** and drops **t5 below the bar**. It also does not reach the target: the
fastest bar-relevant int8 arm is 0.27 ms, **1.8× over 0.15 ms**. F1 cannot deliver
this phase's goal on its own, and it cannot be shipped as-is without falling under
the gate.

---

## 3. F2 — pricing the architecture before training it

Latency is a property of the graph, not of the weights, so `arch_latency.py`
builds each candidate at **random init**, folds its BatchNorms, exports it with
the production settings and times it with the §0 protocol. The whole width/depth
grid was therefore priced in minutes, before a single GPU hour was spent, and the
training budget went only to configurations already known to fit.

Two trunk families were priced against the `res` baseline:

* **`dwsep`** — depthwise 5-tap + pointwise 1×1, foldable BatchNorm. One block is
  `ch·T·(5 + ch)` MACs against `res`'s `2·ch·ch·5·T`: **9.4× fewer at ch 128**.
* **`resbn`** — the *original dense* block, unchanged except that its GroupNorms
  become foldable BatchNorms. Same MACs as `res`, five ONNX nodes instead of
  fourteen.

| spec | params | fp32 ms | p90 | int8 ms | p90 |
|---|---|---|---|---|---|
| `res:128:1,2,4,8` (the ship candidate's shape) | 685,090 | 0.468 | 0.484 | 0.261 | 0.275 |
| `dwsep:96:1,2,4` | 53,986 | **0.100** | 0.106 | 0.106 | 0.113 |
| `dwsep:96:1,2,4,8` | 64,258 | 0.114 | 0.121 | 0.123 | 0.134 |
| `dwsep:112:1,2,4,8` | 80,018 | 0.129 | 0.138 | — | — |
| `dwsep:128:1,2,4,8` | 97,826 | 0.141 | 0.150 | 0.136 | 0.146 |
| `dwsep:160:1,2,4` | 112,226 | 0.143 | 0.151 | — | — |
| `dwsep:144:1,2,4,8` | 117,682 | 0.156 | 0.164 | — | — |
| `dwsep:160:1,2,4,8` | 139,586 | 0.174 | 0.184 | 0.154 | 0.164 |
| `dwsep:192:1,2,4,8` | 189,538 | 0.208 | 0.219 | 0.168 | 0.180 |
| `dwsep:128:1,2,4,8,1,2,4,8` | 168,994 | 0.228 | 0.238 | 0.205 | 0.217 |
| `resbn:48:1,2,4,8` | 111,250 | **0.120** | 0.128 | — | — |
| **`resbn:64:1,2,4`** | **143,714** | **0.135** | 0.145 | — | — |
| `resbn:64:1,2,4,8` | 185,058 | 0.160 | 0.168 | — | — |
| `resbn:80:1,2,4` | 214,866 | 0.175 | 0.186 | — | — |
| `resbn:80:1,2,4,8` | 279,346 | 0.210 | 0.221 | — | — |
| `resbn:96:1,2,4,8` | 394,114 | 0.267 | 0.281 | — | — |

### The result that redirected the phase: separable convolution is *not* the win

The brief's hypothesis was depthwise-separable. Measured, **dense convolution is
the more parameter-efficient shape per microsecond on this graph**:

| at ~0.14 ms | params |
|---|---|
| `dwsep:128:1,2,4,8` | 97,826 |
| **`resbn:64:1,2,4`** | **143,714 (+47 %)** |

The reason is that both shapes have the *same* MAC-to-parameter ratio — `T = 32`,
because every weight is reused once per frame — so separability buys no arithmetic
per parameter; it only trades a well-vectorized dense kernel for a memory-bound
depthwise one plus a smaller dense one. Measured throughput bears that out:
**~68 GMAC/s for the dense trunk against ~38 GMAC/s for the separable one.** What
separability *does* buy is a smaller model at a given width, which is not what this
phase is short of. The `dwsep` trunk was kept as a trained control (arm C) rather
than dropped on the strength of the argument alone.

### There is a hard floor at ~0.10 ms, and it is not the trunk

`dwsep:96:1,2,4` — 54 k parameters, a trunk that is nearly free — still measures
**0.100 ms**. Roughly two thirds of the ≤0.15 ms budget is spent before the first
trunk block: the in-graph path featurization, the per-key embedding MLP over
`layout_keys` (a `[1,64,66] → [1,64,96] → [1,64,64]` pair of `Gemm`s that ORT
cannot constant-fold, because `layout_keys` is a graph **input** and must stay one
for layout-agnosticism), the three output heads, the scoring `MatMul`, the mask
`Where` and the `LogSoftmax` — plus the harness floor of §1. **Any target below
~0.10 ms would require changing the I/O contract, which this phase may not do.**

Shrinking the key-embed MLP is the only lever inside that floor and it is small:
`embed_hid` 96 → 64 → 48 moves `dwsep:128:1,2,4,8` 0.141 → 0.138 → 0.134 ms. Not
spent — 0.007 ms is not worth a capacity cut anywhere else.

---

## 4. F2 — the distilled students

**Training.** Identical to the Phase-E final recipe except for the trunk: T3 with
its How-We-Swipe half oversampled 3× (`train_t3.npz,train_t3hws.npz,train_t3hws.npz`),
94,000 steps, batch 256, lr 3e-3, wd 0.01, warmup 1,000, fp32, checkpoint selected
on beam top-1 over a 5,000-row val prefix at the published preset. Seed 1234 for
the arm comparison.

**Distillation.** `train.py --kd-teacher ckpt/phaseE-FINAL-s1234/best.pt
--kd-weight 1.0 --kd-temp 2.0`. The teacher is **our own** ch 192 Phase-E
checkpoint — never FUTO weights and never FUTO outputs, which the FUTO Model
Weights License would treat as making the student a derivative. It runs frozen in
`eval()` on the *same augmented batch* as the student, so both see the identical
slot permutation and geometric jitter and the KD term compares two distributions
over the same column assignment. The loss is
`KL(teacher‖student)` at temperature 2 on the log-softmaxed head, scaled by `T²`
and normalized **per frame** (summed over classes, averaged over batch × frames)
so a quoted `--kd-weight` means the same thing at any frame count. Taken over all
65 columns, which is identical to the 27-wide sliced view: the 38 pad columns sit
at the finite `MASK_NEG` for *both* models, so `exp(teacher/T)` is exactly 0 there
and they contribute nothing. The teacher's sha256 is recorded in every student
checkpoint.

### Round 1 — the two structural questions, answered

| arm | trunk | params | ms (idle) | t1 | t3 | t5 | ≤3 | 4+ | bars |
|---|---|---|---|---|---|---|---|---|---|
| the bar | | | | 85.52 | 91.54 | 92.80 | 89.29 | 83.57 | |
| **A** | `resbn:64:1,2,4` | 143,714 | 0.135 | 85.89 | 91.48 | 92.50 | 88.76 | 84.41 | **2/5** |
| **B** | `resbn:48:1,2,4,8` | 111,250 | **0.120** | **86.39** | 91.41 | 92.38 | **89.82** | **84.61** | **3/5** |
| **C** | `dwsep:128:1,2,4,8` | 97,826 | 0.141 | 85.78 | 91.39 | 92.27 | 88.40 | 84.42 | **2/5** |

* **Depth beats width at equal latency.** B is four blocks at ch 48; A is three at
  ch 64. B is **0.015 ms cheaper and +0.50 t1**, and wins on ≤3 by 1.06.
* **Depthwise-separable is the wrong shape here**, as §3 predicted from throughput
  and as C confirms after training: −0.61 t1 against B while costing 0.021 ms more.
  The brief's F2 hypothesis is refuted on this graph.
* **t1, 4+ and (for B) ≤3 clear the bar. t3 and t5 do not.** Every student in the
  phase is short on exactly those two, by 0.13 and 0.42 pt for B. That is the whole
  remaining problem, and it is a model problem, not a decode problem — see next.

### The scoring preset still transfers, so t3/t5 cannot be recovered at decode time

The E1 preset was tuned on a **5–6× larger** model. Re-swept from scratch on B's
own emissions (`sweep_scoring.py`, wide grid, tuned on val `0:4959`, confirmed on
the untouched `4959:9918`) the optimum moves to γ 0.95, λ 1.1, β 0.3, γp 0.25,
βp 0.5 and scores **86.51 / 91.46 / 92.41** (≤3 89.88, 4+ 84.76) on full val
against **86.39 / 91.41 / 92.38** (≤3 89.82, 4+ 84.61) for the transferred E1
preset: **+0.12 t1, +0.05 t3, +0.03 t5.** Nothing. As in Phase E §5, the preset is
reported *transferred*, not re-fitted, and the deficit on t3/t5 is real.

### Round 2 — capacity, and the knife-edge on t5

Round 1 says the shape is settled (`resbn`, four dilated blocks) and the only
remaining variable is how much of it fits in the budget. Arm **D** takes the same
shape to ch 64 — 185 k parameters at **0.160 ms**, i.e. deliberately *over* the
target — to locate where the bar is actually cleared:

| arm | trunk | params | ms (idle) | t1 | t3 | t5 | ≤3 | 4+ | bars |
|---|---|---|---|---|---|---|---|---|---|
| the bar | | | | 85.52 | 91.54 | 92.80 | 89.29 | 83.57 | |
| **D** | `resbn:64:1,2,4,8` | 185,058 | 0.160 | 86.70 | 91.84 | **92.78** | 89.44 | 85.28 | **4/5** |

**D misses by 0.02 pt on t5 and nothing else** — one row in 9,918 — while clearing
t1 by +1.18, t3 by +0.30, ≤3 by +0.15 and 4+ by +1.71. That is the same knife-edge
Phase E hit at E1 (t5 92.79 vs 92.80), and it fixes the shape of the answer: at
around 0.16 ms this architecture is *level* with the FUTO ceiling on top-5 and
above it on everything else. The ≤0.15 ms question is therefore decided by
whatever separates a 0.149 ms model from a 0.160 ms one, plus whatever a
training-side lever can add for free.

---
