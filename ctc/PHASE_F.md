# Phase F — latency

Phase E ended with two configurations that clear all five FUTO-ceiling bars on
val-9918 and on the (now spent) test-2400 seal: ch 192 at **0.877 ms** and ch 128
at **0.455 ms** single-thread batch-1 CPU. Phase F asks a different question:
**how much of that time can be removed without falling back under the bar?**

The target set for this phase is **≤ 0.15 ms** — half the Campaign-1 `r2`
artifact's 0.306 ms, and about a third of the ch 128 ship candidate.

**Result in one paragraph.** The target is **not** reachable with the bar intact.
The best model at or under 0.15 ms (`resbn:56:1,2,4,8`, 0.141 ms) clears four of
the five bars and misses top-5 by 0.19 pt; the next size up, at 0.162 ms, misses
top-5 by 0.13 pt at three seeds. The fastest configuration that clears **all five
on the seed mean and on every individual seed** is `resbn:80:1,2,4,8` at
**0.213 ms / 279 k parameters / 1.1 MB** — **2.23× faster and 2.45× smaller than
the Phase-E ch 128 ship candidate for −0.41 t1**. Two of the three levers the brief
proposed were measurably the wrong ones (post-training int8, depthwise-separable
convolution); the win came from a dense trunk whose BatchNorms fold away at export,
plus self-distillation from our own ch 192 checkpoint. §6 is the measured frontier,
§7 states what was not reached and §7.1 the one promising lever that GPU budget cut
short.

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
0.911–0.934 against 0.877). The `r2` artifact measures **0.306 ms** here — the figure on record to the digit —
so the scale is exactly the one the ≤0.15 ms target was set on.
All Phase-F comparisons are internally consistent because they all come from this
harness on this machine; where a Phase-E number is quoted it is re-measured here
rather than copied.

### The harness itself costs 0.007 ms, so every number below is the graph

A no-op ONNX graph carrying the exact production I/O signature
(`features [1,2,64]` + `layout_keys [1,64,2]` + `layout_mask [1,64]` in,
`log_emissions [1,32,65]` out, no arithmetic) measures what the Python binding,
ORT's `Run()` dispatch and the input copies cost on their own:

```
noop-contract-shapes    mean 0.007 ms   p90 0.007 ms
```

**0.007 ms — under 5 % of the ≤0.15 ms target and under 1.5 % of the ch 128
baseline.** This was measured rather than assumed precisely because the phase turns
on a ~0.10 ms floor (§3): if that floor had been harness overhead the whole
analysis would have been an artifact of the instrument. It is not. The floor is
real graph work.

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
| ch 128 int8 static, tail+norms+stem fp32 | 0.273 | 0.285 | 905,293 | 87.04 | 91.98 | **92.72** | 90.09 | 85.46 | **no (t5)** |
| **ch 192 fp32** (`phaseE-FINAL-s1234`) | 0.934 | 0.951 | 6,144,249 | 88.22 | 92.23 | 93.08 | 91.15 | 86.71 | **yes** |
| ch 192 int8 **dynamic** | 0.906 | 0.927 | 6,043,994 | — | — | — | — | — | n/a |
| ch 192 int8 static, whole graph | 0.428 | 0.445 | 1,710,334 | **0.00** | 0.00 | 0.00 | 0.00 | 0.00 | **no** |
| ch 192 int8 static, tail fp32 | 0.439 | 0.457 | 1,801,552 | 86.77 | 91.71 | **92.62** | 89.32 | 85.45 | **no (t5)** |

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

| arm | trunk | params | ms (idle, trained export) | t1 | t3 | t5 | ≤3 | 4+ | bars |
|---|---|---|---|---|---|---|---|---|---|
| the bar | | | | 85.52 | 91.54 | 92.80 | 89.29 | 83.57 | |
| **A** | `resbn:64:1,2,4` | 143,714 | 0.134 | 85.89 | 91.48 | 92.50 | 88.76 | 84.41 | **2/5** |
| **B** | `resbn:48:1,2,4,8` | 111,250 | **0.122** | **86.39** | 91.41 | 92.38 | **89.82** | **84.61** | **3/5** |
| **C** | `dwsep:128:1,2,4,8` | 97,826 | 0.142 | 85.78 | 91.39 | 92.27 | 88.40 | 84.42 | **2/5** |

* **Depth beats width at equal latency.** B is four blocks at ch 48; A is three at
  ch 64. B is **0.012 ms cheaper and +0.50 t1**, and wins on ≤3 by 1.06.
* **Depthwise-separable is the wrong shape here**, as §3 predicted from throughput
  and as C confirms after training: −0.61 t1 against B while costing 0.020 ms more.
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

| arm | trunk | params | ms (idle, trained export) | t1 | t3 | t5 | ≤3 | 4+ | bars |
|---|---|---|---|---|---|---|---|---|---|
| the bar | | | | 85.52 | 91.54 | 92.80 | 89.29 | 83.57 | |
| **D** | `resbn:64:1,2,4,8` | 185,058 | 0.162 | 86.70 | 91.84 | **92.78** | 89.44 | 85.28 | **4/5** |

**D misses by 0.02 pt on t5 and nothing else** — one row in 9,918 — while clearing
t1 by +1.18, t3 by +0.30, ≤3 by +0.15 and 4+ by +1.71. That is the same knife-edge
Phase E hit at E1 (t5 92.79 vs 92.80), and it fixes the shape of the answer: at
around 0.16 ms this architecture is *level* with the FUTO ceiling on top-5 and
above it on everything else.

Arm **G** is the same shape at ch 56 — the largest `resbn` trunk that comfortably
fits the budget, at **0.141 ms**:

| arm | trunk | params | ms (idle, trained export) | t1 | t3 | t5 | ≤3 | 4+ | bars |
|---|---|---|---|---|---|---|---|---|---|
| the bar | | | | 85.52 | 91.54 | 92.80 | 89.29 | 83.57 | |
| **G** | `resbn:56:1,2,4,8` | 145,594 | **0.141** | 86.25 | 91.67 | **92.61** | 89.52 | 84.55 | **4/5** |

Inside the target, **four of five bars clear and t5 misses by 0.19 pt.**

Arm **I** takes the same shape past the target, to ch 80 at **0.213 ms**, to find
where all five actually clear:

| arm | trunk | params | ms (idle, trained export) | t1 | t3 | t5 | ≤3 | 4+ | bars |
|---|---|---|---|---|---|---|---|---|---|
| the bar | | | | 85.52 | 91.54 | 92.80 | 89.29 | 83.57 | |
| **I** | `resbn:80:1,2,4,8` | 279,346 | 0.213 | **87.41** | **92.18** | **92.85** | **90.38** | **85.86** | **5/5** |
| ch 128 `res` (Phase-E ship candidate, s1234) | 689,282 | 0.475 | 88.02 | 92.27 | 93.03 | 91.12 | 86.41 | 5/5 |

**All five clear at 0.213 ms with 279 k parameters — 2.23× faster than the ch 128
ship candidate at 41 % of its size, for −0.61 t1.** That is the phase's shippable
result, and it is not the target.

---

## 5. F3 — the student stacked with static int8. Rejected

The brief expected F3 to be the winner. It is the worst arm in the phase, for two
independent reasons that only show up once the student exists.

| model | fp32 t1 | int8 t1 | Δ | int8 bars |
|---|---|---|---|---|
| `G` `resbn:56:1,2,4,8` | 86.25 | 84.77 | **−1.48** | 0/5 |
| `I` `resbn:80:1,2,4,8` | 87.41 | 86.38 | **−1.03** | 3/5 (loses t5 and ≤3) |

*(static QDQ, per-channel int8 weights, uint8 activations, 1,024 calibration rows,
tail and stem in fp32 — the best exclusion set found in §2.)*

1. **The accuracy cost does not shrink with the model.** int8 takes 1.0–1.5 pt off
   a student, the same order as it took off ch 128 — but the student has no margin
   to spend. `I` goes from 5/5 to 3/5.
2. **The latency benefit almost vanishes at this size.** Measured on the random-init
   grid in §3, int8 moves `dwsep:128:1,2,4,8` from 0.141 to 0.136 ms and makes
   `dwsep:96:1,2,4` *slower* (0.100 → 0.106). Once the trunk is small, the graph is
   dominated by the fixed head/featurizer work and by per-node dispatch, and QDQ
   **adds** `QuantizeLinear`/`DequantizeLinear` nodes — 361 of them on the ch 128
   graph. Quantization is a lever for arithmetic-bound graphs; these are not.

**F3 is rejected.** The stack is strictly worse than the fp32 student it is built
from.

---

## 6. The measured latency/accuracy frontier

Every row is seed 1234, full val-9918, E1 preset, decoded through the exported
graph; latency is the §0 protocol on an idle machine. **The ≤0.15 ms target is not
reached with all five bars clearing** — §7 states that plainly — so per the brief
this frontier is the result.

All latencies here are the **trained exports**, re-measured in a single idle pass
(the §3 grid was priced at random init and some of it under load; where the two
differ, this table is authoritative).

| ms | p90 | model | params | bytes | t1 | t3 | t5 | ≤3 | 4+ | bars |
|---|---|---|---|---|---|---|---|---|---|---|
| — | — | **the bar** | — | — | 85.52 | 91.54 | 92.80 | 89.29 | 83.57 | — |
| 0.007 | 0.007 | *(no-op graph — the harness floor)* | 0 | — | — | — | — | — | — | — |
| **0.122** | 0.130 | `resbn:48:1,2,4,8` (B) | 111,250 | 472,645 | 86.39 | 91.41 | 92.38 | 89.82 | 84.61 | 3/5 |
| 0.126 | 0.136 | `resbn:56:1,2,4,8` **int8** (G) | 145,594 | 290,872 | 84.77 | 90.99 | 92.05 | 87.49 | 83.35 | 0/5 |
| 0.134 | 0.141 | `resbn:64:1,2,4` (A) | 143,714 | 600,196 | 85.89 | 91.48 | 92.50 | 88.76 | 84.41 | 2/5 |
| **0.141** | 0.150 | **`resbn:56:1,2,4,8` (G)** — best ≤0.15 ms | 145,594 | 609,445 | 86.25 | 91.67 | **92.61** | 89.52 | 84.55 | **4/5** |
| 0.142 | 0.150 | `dwsep:128:1,2,4,8` (C) | 97,826 | 413,034 | 85.78 | 91.39 | 92.27 | 88.40 | 84.42 | 2/5 |
| 0.146 | 0.155 | `resbn:80:1,2,4,8` **int8** (I) | 279,346 | 434,986 | 86.38 | 91.63 | 92.50 | 88.70 | 85.17 | 3/5 |
| 0.162 | 0.172 | `resbn:64:1,2,4,8` (D/FAST) | 185,058 | 766,727 | 86.82* | 91.85* | **92.67*** | 89.86* | 85.24* | **4/5** |
| **0.213** | 0.223 | **`resbn:80:1,2,4,8` (I/FINAL)** — fastest 5/5 | 279,346 | 1,142,727 | **87.47*** | **92.13*** | **92.89*** | **90.35*** | **85.98*** | **5/5** |
| 0.273 | 0.285 | ch 128 `res` **int8** (tail+norms+stem fp32) | 689,282 | 905,293 | 87.04 | 91.98 | 92.72 | 90.09 | 85.46 | 4/5 |
| 0.306 | 0.318 | ⚠ pre-campaign `r2` ch 96, at **its own** re-tuned preset† | 394,114 | 1,619,140 | 86.14 | 91.01 | 92.12 | 89.94 | 84.16 | 3/5 |
| 0.475 | 0.490 | ch 128 `res` fp32 — the Phase-E ship candidate | 689,282 | 2,799,865 | 88.02 | 92.27 | 93.03 | 91.12 | 86.41 | 5/5 |
| 0.920 | 0.937 | ch 192 `res` fp32 — the Phase-E headline | 1,525,378 | 6,144,249 | 88.22 | 92.23 | 93.08 | 91.15 | 86.71 | 5/5 |

`*` = **seed-mean over three seeds** (§8). Every other accuracy row is seed 1234
only. The ch 128 and ch 192 rows are their seed-1234 members, quoted here so the
whole table is single-seed-comparable; their own 3-seed means are in `PHASE_E.md`
§5 (ch 128 87.88 / 92.23 / 92.96, ch 192 88.06 / 92.32 / 93.08).

† The `r2` row is quoted from `PHASE_E.md` §1 at the preset that sweep fitted to
`r2`'s own emissions, not at E1, because E1 was never measured on it. It is the
artifact the ≤0.15 ms target was derived from and is included for that reason
alone; it is not paired with the rest of the table.

Read across it:

* **t5 is the binding bar wherever the model is otherwise good enough.** At the two
  configurations that clear four bars — 0.141 ms and 0.162 ms — t5 is the *only*
  miss, by 0.19 pt and (at three seeds) 0.13 pt. Below ~0.14 ms t3 goes with it, and
  below ~0.135 ms so does ≤3. t1 and 4+ hold +0.3 to +1.9 pt of margin the whole way down; the
  ordering of which bar fails first is stable across all seven trained models.
* **The knee is between 0.162 and 0.213 ms.** Above it all five clear; below it t5
  does not, and §8 resolves the 0.162 ms case at three seeds rather than leaving it
  on a one-seed coin flip.
* **The fastest configuration that clears all five is 2.23× faster than the
  Phase-E ship candidate** and 4.32× faster than the ch 192 headline, at 41 % and
  18 % of their parameters, in **41 %** and **19 %** of the file bytes.
* **The old `r2` artifact is dominated** by the 0.122 ms student, which is 2.5×
  faster, 3.4× smaller and +0.25 t1 (across presets, so read the accuracy loosely).
  The 0.306 ms figure the target was derived from was never a frontier point; it
  was one model, trained on the pre-campaign data with the pre-campaign recipe.

---

## 7. Was ≤0.15 ms reachable? — the honest answer

**No, not with all five bars clearing, on the evidence this phase gathered.**

The best model measured at or under 0.15 ms is `resbn:56:1,2,4,8` at 0.141 ms:
four of five bars clear and **top-5 falls 0.19 pt short** (92.61 vs 92.80). The
next configuration up, at 0.162 ms, was taken to **three seeds** precisely because
its single-seed t5 (92.78) sat 0.02 pt from the bar and a one-seed reading could
not settle it: the seed-mean is **92.67, and no seed clears** (§8). The first
configuration in this family that does clear all five, on the mean and on every
seed, is **0.213 ms**.

So the frontier is not merely "0.15 ms was missed narrowly" — the bar-clearing
boundary sits at **~1.4× the target**, and it is now measured on both sides with a
seeded run, not extrapolated.

What was tried, and what each was worth:

| lever | outcome |
|---|---|
| depthwise-separable trunk (the brief's F2 hypothesis) | **negative** — −0.61 t1 vs dense at *higher* latency (§3, §4) |
| dense trunk with foldable BatchNorm (`resbn`) | **the win** — 142 → 92 ONNX nodes, +47 % parameters per microsecond |
| depth over width at fixed budget | **+0.50 t1** (4×ch48 over 3×ch64), and cheaper |
| self-distillation from our ch 192 | adopted throughout (see §7.1 for what is *not* established about it) |
| post-training static int8 on the shipped models (F1) | 1.8–2.2× latency, **−1.0 to −1.9 pt**, loses t5 |
| student + static int8 (F3) | **worst arm in the phase** — −1.0 to −1.5 pt for ~0 latency |
| re-tuning the scoring preset for the student | **+0.12 t1** — nothing; the E1 preset transfers |
| shrinking the key-embed MLP (`embed_hid` 96→48) | 0.007 ms; not spent |
| ORT offline graph-optimization serialization | **no runtime effect**; it moves session-load work only |

And the structural reason the target is out of reach on this contract: **~0.10 ms
of the 0.15 ms budget is spent before the first trunk block** (§3), and it is real
graph work, not instrument overhead — the no-op graph with the same I/O costs
0.007 ms (§0). Profiled on the 54 k-parameter `dwsep:96:1,2,4`, that 0.099 ms
splits as `Conv` 31 %, `Gelu` 15 %, `Gemm` 15 % (the key-embed MLP alone is
18.7 us), `Mul` 8 %, and a long tail of `Where`/`Slice`/`Concat`/`Add`/`Cos`/`Pad`/
`LogSoftmax` at 1–5 % each. Every one of those except the trunk is fixed by the
I/O contract, which this phase is not permitted to change. That leaves ~0.05 ms of
trunk, and 0.05 ms of trunk is about 110–145 k parameters — which lands at arm B's
3/5 and arm G's 4/5.

### 7.1 Levers that were started and cut for GPU budget — not nulls, *untested*

Three runs were launched and killed before completion when the box was
oversubscribed. They are listed so the phase is not read as having exhausted the
space:

* **`resbn:56:1,2,4,8` at 188,000 steps** (double the campaign budget). Every
  student here **underfits badly** — final train CTC loss 0.42–0.47 against the
  ch 192 teacher's 0.30 — so a longer schedule is the most likely remaining source
  of the 0.19 pt, and it is **free at inference**. Killed at step ~86,000. This is
  the single most promising untested lever in the phase.
* **`resbn:48:1,2,4,8,16` and `resbn:40:1,2,4,8,1,2`** (5- and 6-block narrow
  trunks). Both were trailing `resbn:56:1,2,4,8` at step 21,000 and were killed
  there; that is a trajectory, not a result.
* **the no-distillation ablation** at the final architecture. KD was used in every
  trained arm, so **this phase cannot say how much of the students' accuracy comes
  from the teacher** and how much from the architecture and data. That is a real
  gap in the evidence and is not hidden.

> **§13 supersedes the first bullet.** The extended schedule was subsequently run
> to completion at 188,000 steps on three architectures. It is a real **+0.5 t1**
> and it does **not** move t5. The conclusion of §7 survives it.

---

## 8. The two candidates at three seeds

Both are `resbn`, four dilated blocks (1, 2, 4, 8), `embed_hid` 96, feature v1,
distilled from `phaseE-FINAL-s1234` at weight 1.0 / temperature 2, T3 with 3× HWS,
94,000 steps, batch 256, lr 3e-3, wd 0.01, warmup 1,000, fp32, checkpoint selected
on beam top-1 over a 5,000-row val prefix. Fresh trainings at seeds 1234 / 4321 /
7777. Full val-9918 at the E1 preset.

### FINAL — `resbn:80:1,2,4,8`, 279,346 params, **0.213 ms** — all five clear

| metric | s1234 | s4321 | s7777 | **seed-mean** | sd | the bar | **Δ** | gate | worst seed |
|---|---|---|---|---|---|---|---|---|---|
| overall t1 | 87.41 | 87.31 | 87.70 | **87.47** | 0.20 | 85.52 | **+1.95** | **PASS** | 87.31 **PASS** |
| t3 | 92.18 | 92.15 | 92.05 | **92.13** | 0.07 | 91.54 | **+0.59** | **PASS** | 92.05 **PASS** |
| t5 | 92.85 | 92.91 | 92.91 | **92.89** | 0.03 | 92.80 | **+0.09** | **PASS** | 92.85 **PASS** |
| ≤3 t1 (n=3,389) | 90.38 | 90.29 | 90.38 | **90.35** | 0.05 | 89.29 | **+1.06** | **PASS** | 90.29 **PASS** |
| 4+ t1 (n=6,529) | 85.86 | 85.76 | 86.31 | **85.98** | 0.29 | 83.57 | **+2.41** | **PASS** | 85.76 **PASS** |

**All five clear on the seed mean and on every individual seed.** Per-source
seed-mean: FUTO **94.79**, HWS **80.29** (against the ch 128 anchor's 95.0 / 81.1).

The margin to read carefully is **t5 at +0.09**. It is small — but the seed sd on
t5 is 0.03, the smallest of the five, and the worst of the three seeds still clears
by 0.05. Compare the ch 128 anchor, whose val t5 margin is +0.16, and Phase E's E1
arm, which *failed* this bar by 0.01. This is a pass, and a narrow one.

### FAST — `resbn:64:1,2,4,8`, 185,058 params, **0.162 ms** — four of five

| metric | s1234 | s4321 | s7777 | **seed-mean** | sd | the bar | **Δ** | gate |
|---|---|---|---|---|---|---|---|---|
| overall t1 | 86.70 | 86.80 | 86.95 | **86.82** | 0.13 | 85.52 | **+1.30** | **PASS** |
| t3 | 91.84 | 91.87 | 91.83 | **91.85** | 0.02 | 91.54 | **+0.31** | **PASS** |
| t5 | 92.78 | 92.58 | 92.65 | **92.67** | 0.10 | 92.80 | **−0.13** | **FAIL** |
| ≤3 t1 | 89.44 | 89.82 | 90.32 | **89.86** | 0.44 | 89.29 | **+0.57** | **PASS** |
| 4+ t1 | 85.28 | 85.24 | 85.20 | **85.24** | 0.04 | 83.57 | **+1.67** | **PASS** |

This arm is why the phase can answer its question rather than guess at it. At one
seed FAST's t5 read **92.78** — a 0.02 pt miss that a single run cannot distinguish
from a pass. At three seeds it is **92.67 ± 0.10, and no seed clears**: the
knife-edge resolves to a miss, and it does so on the metric with the smallest seed
variance in the table. The 0.162 ms configuration does **not** clear the bar, and
the ≤0.15 ms configurations are all below it.

---

## 9. Artifacts and export parity

`artifacts/`, all opset 17, fp32, static shapes `[1,2,64]/[1,64,2]/[1,64]` →
`[1,32,65]/[1,32,64]/[1,32,1]`, in-graph `log_softmax`, blank at column 64, zero
`Einsum`, and — new in Phase F — **zero normalization nodes**, the BatchNorms
having been folded into their convolutions at export.

| file | arm | params | bytes | ms | sha256 |
|---|---|---|---|---|---|
| `fast_resbn80_s1234.onnx` ← **the Phase-F candidate** | `phaseF-I-resbn80x4` | 279,346 | 1,142,727 | 0.213 | `5e8c88756cbad5a5a8b8b3f289a990174fa6f3b6edfead46d8dbdb2927fb06f2` |
| `fast_resbn80_s4321.onnx` | `phaseF-FINAL-resbn80x4-s4321` | 279,346 | 1,142,727 | 0.213 | `ca7a670095dae41ed441eaca22cd0a5be6cdd620826f1d1bc0b49c0d9f72a35d` |
| `fast_resbn80_s7777.onnx` | `phaseF-FINAL-resbn80x4-s7777` | 279,346 | 1,142,727 | 0.213 | `a0d0c894a1cfd616f939644cd9c63cbe5910c3846ca2b542e55b43d2f278f4d0` |
| `fast_resbn64_s1234.onnx` | `phaseF-D-resbn64x4` | 185,058 | 766,727 | 0.162 | `1c39b49a7673695dab046eabc303f3428a30eec691cdfdf51af037fc0685cd79` |
| `fast_resbn64_s4321.onnx` | `phaseF-FAST-resbn64x4-s4321` | 185,058 | 766,727 | 0.161 | `3045207fa5b95874eb111eb817199a6f4c9ae048cd8b35665bc35021e490a382` |
| `fast_resbn64_s7777.onnx` | `phaseF-FAST-resbn64x4-s7777` | 185,058 | 766,727 | 0.161 | `f03639aaea8ea91b871a56133b8838963e5ee8680c7f6050962aec966719a399` |
| `fast_resbn56_s1234.onnx` ⚠ **under the bar** | `phaseF-G-resbn56x4` | 145,594 | 609,445 | 0.141 | `e8b52c80e9950eb8d6d1ac886493826764de957a10eeb46f764fcf8bb7c3ec8a` |

The `fast_resbn64_*` and `fast_resbn56_*` files are published as **frontier
evidence, not as ship candidates** — neither configuration clears t5 (§8, §6).

### Parity, per the audit conventions

Every export ran the two `export_onnx.py` checks plus the Phase-F fold check, over
100 random `(features, layout_keys)` draws each:

| arm | BN pairs folded | max abs Δ from folding | sliced `[32,27]` max abs vs torch | argmax agreement |
|---|---|---|---|---|
| `fast_resbn80_s1234` | 9 | 9.77e-04 | 2.57e-05 | **100/100** |
| `fast_resbn80_s4321` | 9 | 9.77e-04 | 3.62e-05 | **100/100** |
| `fast_resbn80_s7777` | 9 | 9.77e-04 | 4.01e-05 | **100/100** |
| `fast_resbn64_s1234` | 9 | 4.77e-06 | 2.29e-05 | **100/100** |
| `fast_resbn64_s4321` | 9 | 7.63e-06 | 2.38e-05 | **100/100** |
| `fast_resbn64_s7777` | 9 | 8.11e-06 | 2.86e-05 | **100/100** |
| `fast_resbn56_s1234` | 9 | 8.11e-06 | 3.43e-05 | **100/100** |

Parity is asserted on the **sliced 27-column contract view** — what
`CtcEmissions.sliceFromHead` feeds the Kotlin beam — not the raw 65-wide head,
whose 38 pad columns sit at ≈ −1e4 where the float32 ULP is 9.8e-4 (audit fix #1).
The three `resbn80` folds report exactly that ULP, 9.77e-04, because their fold
check is taken on the **full** head where a one-ULP move in a pad column is the
largest single difference; the contract view moves by 2.6e-05 to 4.0e-05, four
orders of magnitude inside the 1e-4 tolerance, with every frame's argmax unchanged
on all 100 draws.

The I/O contract is **unchanged** from `r2` and from the Phase-E artifacts: same
input names, shapes and dtypes, same three outputs, same blank column, same opset.
No Kotlin signature moves.

---

## 10. Decisions

* **Adopt the `resbn` trunk for any future encoder in this family.** Swapping
  GroupNorm for BatchNorm and folding it at export removes 50 ONNX nodes and every
  normalization kernel from the graph at **zero cost to accuracy or training**, and
  it is the change that made every other Phase-F result possible. It is a strict
  improvement on the `res` trunk and should be used even at ch 128 or ch 192 if
  those are ever retrained.
* **Reject depthwise-separable convolution for this graph** (F2's stated
  hypothesis). Dense and separable share the same MAC-per-parameter ratio at
  `T = 32`, so separability buys nothing arithmetic and costs kernel efficiency:
  measured 68 GMAC/s dense vs 38 GMAC/s separable, and −0.61 t1 after training.
* **Reject post-training int8, at every size** (F1 and F3). It is a genuine
  1.8–2.2× on the big graphs but costs 1.0–1.9 pt and always loses t5; on the
  small graphs it costs the same accuracy for no latency at all.
* **Prefer more blocks to more channels** at a fixed latency budget.
* **Do not change the scoring preset for a smaller model.** Re-tuned from scratch
  on a 6×-smaller student the E1 preset is worth +0.12 t1 — inside noise, and a
  second preset divergence is not worth that.
* **The ≤0.15 ms target is not met with the bar intact.** No under-bar model is
  put forward as a ship candidate; §6's frontier is the deliverable, and §7.1 names
  the one lever that might still close the 0.19 pt.
* **If a faster encoder is wanted, ship `fast_resbn80_s1234.onnx`** — 0.213 ms,
  279 k parameters, 1.1 MB, all five val bars on the seed mean and on every seed,
  **2.23× faster and 2.45× smaller than the Phase-E ch 128 candidate for −0.41 t1
  on the seed mean** (87.47 vs 87.88, both 3-seed). The decision is a genuine trade, not a free
  win, and it must be taken with §11.1 in view: ch 128 is *test*-validated and this
  is not.

---

## 11. What this phase does **not** establish

1. **All of it is val-only, by construction.** `test-2400` is sealed-spent
   (`AUDIT_FINAL.md` §7). No Phase-F model has ever been decoded on it and none may
   be. The test-validated anchors are still ch 128 and ch 192 from Phase E; a
   Phase-F artifact is a val-validated variant, and any claim about it must say so.
2. **Single seed for the frontier.** Every row of §6 except the two candidates in
   §8 is one seed. Phase D measured the single-seed noise floor at **~1 pt top-1**,
   which is larger than several of the gaps in that table. The frontier's *shape*
   is robust (it is monotone in capacity across seven models); individual 0.1–0.3 pt
   differences in it are not.
3. **The distillation contribution is unmeasured** (§7.1). Every trained arm used
   the same teacher at the same weight and temperature; neither the KD weight nor
   the temperature was swept, and no no-teacher control completed.
4. **Everything Phase E did not establish still applies unchanged** — the preset
   asymmetry (our decode preset is tuned on the holdout family, FUTO's is not), the
   T3 contributor contamination, and the fact that these are benchmark numbers and
   not a generalization claim about an unseen user. Distillation adds one more:
   these students inherit whatever the ch 192 teacher learned from a
   contributor-contaminated tier, so their agreement with it is not independent
   evidence of anything.
5. **Desktop x86, single core, through the Python binding.** §0 quantifies the
   harness floor. On a phone the ratio between the encoder and the 147 k-word trie
   beam is what matters, and the beam — not the encoder — dominates the per-swipe
   budget. A 0.3 ms saving on the encoder may be invisible end-to-end.

---

## 12. Reproduction

```bash
# profile the incumbent before optimizing it
python bench_latency.py --onnx ckpt/phaseE-E3b-hws3x/ctc_swipe_encoder.onnx \
       --profile --optimize-out cache/ch128_ortopt.onnx

# price an architecture before training it (random init; latency is graph-only)
python arch_latency.py --quant --spec resbn:80:1,2,4,8 --spec dwsep:128:1,2,4,8

# the F1 arms
python quantize_onnx.py --onnx ckpt/phaseE-E3b-hws3x/ctc_swipe_encoder.onnx \
       --mode static --calib-npz cache/train_t3hws.npz --calib-rows 1024 \
       --exclude-tail --exclude InstanceNormalization,norm,stem \
       --out quant/ch128_stat_tail_norm_stem.onnx

# a distilled student (teacher is OUR ch192 checkpoint, never FUTO output)
python train.py --train-npz train_t3.npz,train_t3hws.npz,train_t3hws.npz \
       --run-name phaseF-FINAL-resbn80x4-s1234 --block resbn --ch 80 \
       --dilations 1,2,4,8 --total-steps 94000 --val-every 3000 --batch 256 \
       --lr 3e-3 --weight-decay 0.01 --warmup 1000 --seed 1234 \
       --beam-val-rows 5000 --beam-jobs 8 \
       --kd-teacher ckpt/phaseE-FINAL-s1234/best.pt --kd-weight 1.0 --kd-temp 2.0
python export_onnx.py --ckpt ckpt/phaseF-I-resbn80x4/best.pt \
       --out ckpt/phaseF-I-resbn80x4/ctc_swipe_encoder.onnx     # folds BN, asserts parity

# report (test-2400 is refused by seal.py's content guard)
python eval_arms.py --arms phaseF-I-resbn80x4 --preset 1.05,1.1,0.2,0.3734,0.9882 \
       --own-mask T0 --also-masks= --rebuild-cache
```

---

## 13. The extended-schedule round — a real +0.5 t1 that does **not** buy t5

§7.1 flagged the training budget as the most promising untested lever: every
student's final train CTC loss sits far above the teacher's, so they looked
undertrained rather than under-capacity. With the GPU budget lifted, the schedule
was doubled to **188,000 steps** (cosine keyed to 188 k, warmup unchanged at 1,000,
everything else identical to §4 including the teacher, weight and temperature) on
three architectures, seed 1234.

### Did the underfit close?

| arm | params | train CTC @94k | train CTC @188k | Δ |
|---|---|---|---|---|
| `resbn:48:1,2,4,8,16` | 134,578 | — | 0.4358 | — |
| `resbn:56:1,2,4,8` | 145,594 | 0.4425 | **0.4284** | −0.0141 |
| `resbn:64:1,2,4,8` | 185,058 | 0.4178 | **0.4039** | −0.0139 |
| *reference:* `resbn:80:1,2,4,8` @94k | 279,346 | 0.3816 | — | — |
| *reference:* ch 128 `res` @94k | 689,282 | 0.3017 | — | — |
| *reference:* ch 192 `res` @94k (the teacher) | 1,525,378 | 0.2422 | — | — |

**Barely.** Doubling the schedule bought **0.013–0.014** of train CTC, while the
step from ch 56 to ch 80 buys **0.061** and the step to ch 128 buys **0.141**. The
gap to the teacher is dominated by capacity, not by optimization: these models are
**under-capacity, not undertrained**, and the "underfit" reading in §7.1 was wrong
in its mechanism even though the experiment was worth running.

### What it bought on the bar — full val-9918, E1 preset, seed 1234

| arm | params | ms (idle) | t1 | t3 | t5 | ≤3 | 4+ | bars |
|---|---|---|---|---|---|---|---|---|
| the bar | | | 85.52 | 91.54 | 92.80 | 89.29 | 83.57 | |
| `resbn:48:1,2,4,8,16` @188k | 134,578 | **0.139** | 86.64 | 91.80 | 92.53 | 89.73 | 85.04 | **4/5** |
| `resbn:56:1,2,4,8` @94k (G) | 145,594 | 0.141 | 86.25 | 91.67 | 92.61 | 89.52 | 84.55 | 4/5 |
| **`resbn:56:1,2,4,8` @188k** | 145,594 | 0.144 | **86.79** | **91.83** | **92.65** | **90.26** | **84.99** | **4/5** |
| Δ (188k − 94k) | | | **+0.54** | **+0.16** | **+0.04** | **+0.74** | **+0.44** | |
| `resbn:64:1,2,4,8` @94k (D) | 185,058 | 0.162 | 86.70 | 91.84 | 92.78 | 89.44 | 85.28 | 4/5 |
| **`resbn:64:1,2,4,8` @188k** | 185,058 | 0.161 | **87.19** | **92.09** | **92.76** | **90.29** | **85.59** | **4/5** |
| Δ (188k − 94k) | | | **+0.49** | **+0.25** | **−0.02** | **+0.85** | **+0.31** | |

**The extended schedule is worth about +0.5 t1, +0.2 t3, +0.8 ≤3 and +0.4 4+ —
and +0.04 / −0.02 on t5.** It is a genuine, free-at-inference gain on four of the
five metrics and it moves the fifth by nothing at all. Neither arm's seed-1234
result clears all five, so under the pre-agreed decision rule **neither earned a
seed round**.

### Why t5 is the metric that does not respond

Across the 94 k series `t5 − t1` shrinks monotonically with capacity — **6.36** pt
at 145 k parameters, **6.08** at 185 k, **5.44** at 279 k, **5.01** at 689 k — so
the headroom between "in the beam's top five" and "ranked first" is a capacity
property. The extended schedule shrinks that gap too (`resbn:56` 6.36 → 5.86,
`resbn:64` 6.08 → 5.57) but does so **entirely by raising t1**, leaving the top of
the distribution where it was. Longer training sharpens the top of the distribution, which is
what t1, ≤3 and 4+ read; keeping the right word inside a 100-wide beam's **top
five** depends on how well the whole emission distribution is shaped, and that is
a capacity property. This is consistent with Phase E, where the same metric was
the one riding at +0.01 and +0.05 margins on far larger models.

**§7's verdict stands, now on stronger evidence: ≤0.15 ms does not clear all five
bars, and the reason is capacity, which no training-side lever recovers.**

---
