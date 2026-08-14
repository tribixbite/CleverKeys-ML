# The fourth — and final — unsealing of test-2400

**Status: PRE-REGISTRATION. Written and committed BEFORE any decode of the
sealed split.** Everything in §1–§7 is fixed at commit time. §8 (results) is
appended afterwards and may not restate, reinterpret or quietly widen the plan.

Precedents and protocol: `PHASE_F.md` §16 (second unsealing), `PHASE_G.md` §7
(third unsealing), `AUDIT_FINAL.md` §7 (the seal declared spent), `seal.py`
(the content-addressed guard and the append-only ledger).

**Ledger state at the time of writing: 3 entries** — verified against
`test2400_seal.json["test-2400"]["unsealings"]` (2026-08-08 ×2, 2026-08-09),
plus the three disclosed `prior_contact` items. This decode adds the **fourth**
entry and nothing beyond it.

---

## 1. Authority

Two things, together, and nothing else:

1. **The user's directive of 2026-08-13/14**, approving *one final
   pre-registered unsealing plus an adversarial audit, for whichever model
   ships.* The benchmark's owner is the only party who can spend a spent seal;
   this is that authorisation, and it is explicitly a **single** authorisation.
2. **The Phase-M ship recommendation** (`PHASE_M.md` §11.2): **option B — the
   distilled single model** `v2kd-fresh-w1`, shipped as
   `phaseM_kd_fresh_w1_s1234_fp16w.onnx`, 2.91 MB, one ONNX session, no app
   code change.

The directive names the *shipping* model as the subject. Option A (the coupled
pair) is **not** decoded here and remains val-only; if the orchestrator later
prefers A, that is a decision taken on val evidence, not on this decode. Only
one model is unsealed because only one model ships.

### 1.1 Why this is a first decode and not a selection loop

* `v2kd-fresh-w1` has **never touched test-2400 in any capacity**. Its
  architecture, teacher, `--kd-weight`, initialization, schedule, per-seed
  checkpoint selection (beam top-1 on a 5,000-row *val* prefix) and its
  promotion from one seed to three were all fixed on val-9918 and on the
  alt-layout batteries, and are committed in `PHASE_M.md` §1.2, §6.1 and §9
  — all of which predate this file.
* **Nothing can feed back.** Phase M is closed; `PHASE_M.md` §11.4 records the
  ledger of registered-not-run items as empty and states that no further
  training happens. The outcome of this decode cannot select a model, a seed, a
  checkpoint, a preset or a lexicon, because every one of those was frozen
  before this section was written, and no arm remains in flight.
* **What it costs, stated plainly.** test-2400 will have been read **four**
  times. Every claim resting on it is a claim on a worn split, and this
  document does not pretend otherwise. There is no fifth read: the campaign
  ends with the split more worn than any single result deserves, and that is
  the price of validating what ships.

### 1.2 Measurements taken BEFORE this pre-registration (disclosed, not hidden)

The config-B footing (§2) had never been measured for this model on **val**
either, and a numeric expectation cannot be written without it. Four
**val-9918** decodes — the unsealed split, no protocol constraint — were run
immediately before this file was written, at the app trie and the app preset:
three seeds fp32 plus one fp16w repeat of s1234. Their outputs are in §4.2 and
in `~/ctc-train/valB_*.log`. This is measurement, not tuning: no preset, seed,
checkpoint or lexicon was chosen from them; they exist only so that §5's
predictions are grounded rather than invented. The precedent is `PHASE_G.md`
§6, which fit the app preset on val before §7 registered the decode.

---

## 2. The exact configurations

Both configurations decode the **same three fp32 ONNX graphs** — the ones the
whole Phase-M val and alt-layout record was measured on.

| # | footing | trie | preset | bar it is measured against |
|---|---|---|---|---|
| **A** | benchmark / equal-footing | AOSP `data/futo_en_wordlist.combined` **STRIP, 146,964 words** | **E1** `1.05, 1.1, 0.2, 0.3734, 0.9882` | (a) FUTO published `84.83 / 91.04 / 92.08 / 89.57 / 82.40`; (c) FUTO **val-tuned, equal footing** `87.12 / 92.29 / 92.96 / 89.94 / 85.68` (`FAIR_REMATCH.md` §5) |
| **B** | shipping | app `en_enhanced.json` **STRIP, 98,081 words**, `--vocab-kind json-strip` | **the current shipping app preset** `0.9, 4.0, 0.25, 0.25, 0.9882` | (b) trie-matched published-preset bar `84.92 / 91.54 / 92.96 / 89.57 / 82.52` (`PHASE_F.md` §15.2) |

### 2.1 The config-B preset is transplanted, and that is declared in advance

**No app-trie preset sweep has ever been run for this model or its family.**
`MODEL_COMPARISON.md` records the same gap for `sw2345` explicitly, and nothing
in Phases J–M closed it. The preset used for config B is the one fitted on
`resbn80g` in `PHASE_G.md` §6 and adopted as the app preset in `RESULTS.md` and
`MODEL_COMPARISON.md` §5.1 — i.e. **the current shipping app preset, fitted on
a different model**. Per the orchestrator's instruction, config B runs at that
preset and the fact is noted here rather than fixed by a new sweep.

Consequence, stated before the numbers exist: config B is a **tuned-for-another-
model** footing. It is not an equal-footing comparison (the bar's preset is
FUTO's published one), and it plausibly *understates* what this model would do
at its own app optimum. No claim in §8 may convert that into a favourable
adjustment; the measured number is the number.

### 2.2 Numeric format: fp32 decoded, fp16w shipped

The ship artifact is **fp16w** (`phaseM_kd_fresh_w1_s1234_fp16w.onnx`,
3,052,318 B). All six decodes here use the **fp32** graphs, for strict
comparability with the Phase-M val battery and with all three prior unsealings,
and because a single dtype keeps the six-decode cap intact.

The bridge is measured, not assumed:

* **E1 footing**, `PHASE_M.md` §11.1: fp16w vs fp32 on val, largest delta 0.05
  across eleven axes.
* **App footing**, measured 2026-08-14 for this file (§4.2): fp16w vs fp32 on
  val-9918 at the app trie and app preset — **identical to 0.00 on all five
  metrics** (89.20 / 93.63 / 94.37 / 92.59 / 87.44 both).

Any test number here therefore describes the shipped model to within a measured
≤0.05, and §8 must say so rather than eliding the distinction.

### 2.3 Frozen artifacts — sha256 committed before the decode

Byte-identical copies of `~/ctc-train/ckpt/<run>/ctc_swipe_encoder.onnx`
(exported from `best.pt`, sliced-view parity checked at export), 6,068,519
bytes each, 1,512,802 parameters, opset 17, frozen `[1,32,65]` contract:

```
b71911da3407abc0b113bbc662a1929953b04dcaf7650d848a7e897605a9bf80  phaseM_kd_fresh_w1_s1234.onnx   (run v2kd-fresh-w1)
f7cb72c07e1d5a920e5ceb93b4f6cf241bf0c9dcc630bcd1117d4fdf38d2daf1  phaseM_kd_fresh_w1_s4321.onnx   (run v2kd-fresh-w1-s4321)
c55cc3b055cf2db2b198c03b3fae688aad1930058dfed3902296aa08fd6510d7  phaseM_kd_fresh_w1_s7777.onnx   (run v2kd-fresh-w1-s7777)
84718e6ebc8020176f27b9668e50922a765c96838307b640a8db9ab0549e88e5  phaseM_kd_fresh_w1_s1234_fp16w.onnx   (the ship artifact; NOT decoded here)
```

### 2.4 Everything else, frozen

Beam width **100**; top-k **8**; **OOV counts as a miss** (86 rows under config
A, 64 under config B, per the two prior unsealings); metric = seed-mean over
**1234 / 4321 / 7777** of top-1/3/5 plus the **≤3 (n=815)** and **4+ (n=1,585)**
strata; per-source (FUTO / HWS half) reported from
`cache/holdout_source_tags.json["test"]` **without any extra decode**; the seal
guard logs the 2,400/2,400 overlap and the `--unseal-test` override on every
run.

### 2.5 The commands

```bash
CTC=/home/will/git/CleverKeys-ML/ctc; WD=/home/will/ctc-train
VOCAB=/home/will/git/swype/CleverKeys/src/main/assets/dictionaries/en_enhanced.json
for a in v2kd-fresh-w1 v2kd-fresh-w1-s4321 v2kd-fresh-w1-s7777; do
  # config A
  python3 eval_beam.py --onnx $WD/ckpt/$a/ctc_swipe_encoder.onnx \
    --test data/test_hwsfuto.jsonl --preset 1.05,1.1,0.2,0.3734,0.9882 \
    --beam-width 100 --top-k 8 --unseal-test --out $WD/ckpt/$a/test2400_m_e1.jsonl
  # config B
  python3 eval_beam.py --onnx $WD/ckpt/$a/ctc_swipe_encoder.onnx \
    --test data/test_hwsfuto.jsonl --preset 0.9,4.0,0.25,0.25,0.9882 \
    --vocab $VOCAB --vocab-kind json-strip \
    --beam-width 100 --top-k 8 --unseal-test --out $WD/ckpt/$a/test2400_m_app.jsonl
done
```

---

## 3. THE HARD CAP

**Maximum 2 configurations × 3 seeds = 6 decodes of test-2400. One each.
Nothing more.**

* No fourth seed. No alternate preset. No alternate trie. No `--limit` warm-up.
* **No retries.** If a run crashes, that (config, seed) cell is reported as
  **missing** and the gate is evaluated as a failure on it. Partial output is
  not used and the run is not restarted.
* **No iteration.** The numbers are read once, written down, and that is the
  end. Nothing is re-decoded after the numbers are seen, for any reason,
  including a result that looks wrong.
* The exact-paired McNemar of §6.3 is computed from the **already-written**
  config-A dumps against a **pre-existing** FUTO per-row file
  (`~/ctc-train/futo_verify/out/tuned_ceil_strip.jsonl`, the val-tuned ceiling
  on the same 2,400 rows). It is arithmetic on committed dumps, **not** a
  seventh decode.

---

## 4. The val numbers this decode is predicted from

### 4.1 Config-A footing (val-9918, AOSP STRIP, E1) — `PHASE_M.md` §9

| metric | s1234 | s4321 | s7777 | **seed-mean** |
|---|---|---|---|---|
| t1 | 88.62 | 88.88 | 88.75 | **88.750** |
| t3 | 92.69 | 92.80 | 92.83 | **92.773** |
| t5 | 93.46 | 93.45 | 93.51 | **93.473** |
| ≤3 | 91.38 | 91.44 | 91.30 | **91.373** |
| 4+ | 87.18 | 87.55 | 87.43 | **87.387** |

### 4.2 Config-B footing (val-9918, app trie 98,081, app preset) — measured 2026-08-14, §1.2

| metric | s1234 | s4321 | s7777 | **seed-mean** | s1234 **fp16w** |
|---|---|---|---|---|---|
| t1 | 89.20 | 89.54 | 89.39 | **89.377** | 89.20 |
| t3 | 93.63 | 93.78 | 93.63 | **93.680** | 93.63 |
| t5 | 94.37 | 94.53 | 94.50 | **94.467** | 94.37 |
| ≤3 | 92.59 | 92.77 | 92.33 | **92.563** | 92.59 |
| 4+ | 87.44 | 87.87 | 87.87 | **87.727** | 87.44 |

Greedy-CTC val t1 72.35 / 72.35 / 72.37 %.

### 4.3 Every val→test shift this campaign has ever measured

| unsealing | model / footing | t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|---|
| 1 | ch 192, config A | +0.30 | +0.33 | +0.42 | +0.51 | +0.19 |
| 1 | ch 128, config A | +0.04 | +0.10 | +0.04 | +0.10 | +0.03 |
| 2 | `fast_resbn80`, config A | −0.18 | −0.24 | −0.07 | **+0.82** | −0.68 |
| 2 | `fast_resbn80`, config B (E1 preset, app trie) | −0.42 | −0.11 | −0.26 | +0.25 | −0.74 |
| 3 | `resbn80g`, config A | −0.04 | −0.07 | −0.15 | +0.02 | −0.06 |
| 3 | `resbn80g`, config B (app preset, app trie) | −0.40 | +0.05 | −0.16 | −0.06 | −0.56 |
| | **config-A mean (4)** | **+0.03** | **+0.03** | **+0.06** | **+0.36** | **−0.13** |
| | **config-B mean (2)** | **−0.41** | **−0.03** | **−0.21** | **+0.10** | **−0.65** |
| | **observed range (all 6)** | −0.42 … +0.30 | −0.24 … +0.33 | −0.26 … +0.42 | −0.06 … +0.82 | −0.74 … +0.19 |

`PHASE_F.md` §16.5 already concluded that this shift is **not** a stable
per-split constant that extrapolates across architectures, and `PHASE_G.md`
§7.5 added that **the ≤3 stratum is the unstable one** — it moved *opposite* to
its prediction at both the second (+0.51 vs prediction) and the third (−0.80)
unsealings. Bands below are therefore stated from the observed range across all
six prior observations, and **≤3's band is deliberately widened past it**.

---

## 5. NUMERIC EXPECTATIONS — stated so a miss cannot be re-explained afterwards

Point prediction = §4 val seed-mean + the matching config's mean shift. Band =
val seed-mean + the observed range across all six prior shifts (§4.3), with ≤3
widened by a further ±0.4 for its documented record of surprising in both
directions.

### 5.1 Config A (AOSP STRIP, E1)

| metric | val | **point prediction** | **band** | vs published bar | vs equal-footing bar |
|---|---|---|---|---|---|
| t1 | 88.750 | **88.78** | 88.33 – 89.05 | +3.95 | +1.66 |
| t3 | 92.773 | **92.80** | 92.53 – 93.10 | +1.76 | +0.51 |
| t5 | 93.473 | **93.53** | 93.21 – 93.89 | +1.45 | +0.57 |
| ≤3 | 91.373 | **91.74** | **90.91 – 92.59** | +2.17 | +1.80 |
| 4+ | 87.387 | **87.26** | 86.65 – 87.58 | +4.86 | +1.58 |

**Registered expectation A1:** config A clears **all five** published-bar
numbers on the seed-mean and on every individual seed. Margins are predicted
large (+1.45 … +4.86); the narrowest is t5.

**Registered expectation A2 — the one that is genuinely new:** config A clears
**all five equal-footing** numbers on the seed-mean. This is the first
unsealing in which that is predicted *in advance* — `PHASE_G.md` §7.4 predicted
the opposite for `resbn80g`, correctly. The basis is that this model's **val**
seed-mean already clears the equal-footing **val** bar
(87.48 / 92.31 / 93.03 / 89.76 / 86.29, `FAIR_REMATCH.md` §2) on all five, by
+1.27 / +0.46 / +0.44 / +1.61 / +1.10 — which `resbn80g`'s did not. Even at the
pessimistic end of every band, all five still clear (t3 +0.24 and t5 +0.25 are
the two that could go under on an individual seed).

**Registered expectation A3:** exact paired McNemar on t1 against FUTO's
val-tuned per-row output resolves at p < 0.05 on **2 or 3 of the 3 seeds**. A
+1.66 pt predicted gap is ~40 rows of 2,400; ch 192 resolved 2 of 3 at +1.24,
`resbn80g` resolved 0 of 3 at +0.56.

### 5.2 Config B (app trie 98,081, current shipping app preset)

| metric | val | **point prediction** | **band** | vs trie-matched bar |
|---|---|---|---|---|
| t1 | 89.377 | **88.97** | 88.96 – 89.68 | +4.05 |
| t3 | 93.680 | **93.65** | 93.44 – 94.01 | +2.11 |
| t5 | 94.467 | **94.26** | 94.21 – 94.89 | +1.30 |
| ≤3 | 92.563 | **92.66** | **92.10 – 93.78** | +3.09 |
| 4+ | 87.727 | **87.08** | 86.99 – 87.92 | +4.56 |

**Registered expectation B1:** config B clears all five trie-matched numbers on
the seed-mean and on every seed, with the worst-seed t5 margin above +1.0 (the
incumbent shipped on +0.08; `resbn80g` on +0.75).

**Registered expectation B2:** config B's t1 exceeds config A's t1 by
+0.1 … +0.5 — the app trie is a third the size and covers more of these targets,
and that has held on every prior read.

### 5.3 Where I expect to be wrong

Recorded so it is scoreable: the campaign's forecasts have failed on **≤3** at
two of three unsealings, in opposite directions. If any single prediction in
§5.1/§5.2 misses its band, ≤3 is the one I expect it to be. I also expect the
config-B point predictions to be worse than the config-A ones, because the
config-B shift rests on two observations at two *different* presets, one of
which (unsealing 2) was not even an app-preset run.

---

## 6. THE RULES — what may be claimed, decided now

### 6.1 Tier rule (identical to all three prior unsealings)

`phaseM_kd_fresh_w1` moves from **val-only** to **test-validated** **iff config
A clears all five published-bar numbers on the seed-mean AND on each of the
three individual seeds.** A 4-of-5 result on either footing is a **failed
gate** and is written as a failed gate. All ten numbers (5 metrics × 2 configs)
are published regardless of outcome.

### 6.2 Shipping-footing rule

Config B supports a **shipping validation** — "the configuration users would
run clears the bar on the lexicon that ships" — under the same all-five,
every-seed rule. It is **not** an equal-footing claim: our preset is tuned (for
another model, §2.1), the bar's is FUTO's published one. That asymmetry is
declared here, in advance, exactly as `PHASE_G.md` §7.3 declared it.

### 6.3 Equal-footing rule

Against `87.12 / 92.29 / 92.96 / 89.94 / 85.68` (both engines val-tuned, same
2,400 rows, same STRIP trie, same beam, same OOV rule):

* **all five on the seed-mean AND every seed, AND McNemar p < 0.05 on ≥ 2 of 3
  seeds** → a **qualified equal-footing win**, the same tier of claim ch 192
  holds and no stronger.
* **all five but McNemar resolves on ≤ 1 seed** → "level-to-ahead on equal
  footing"; **no superiority claim**.
* **fewer than five** → no equal-footing claim of any kind; the misses are
  named.

McNemar is exact, paired, two-sided, on top-1 correctness per row, against
`futo_verify/out/tuned_ceil_strip.jsonl`.

### 6.4 What this decode may never do

Select anything; justify a retrain; authorise a fifth read; be quoted for any
model other than `v2kd-fresh-w1`; or be presented without the caveats that
travel with every test-2400 number (T3 contributor contamination, the dedup
defect, the ~14-point FUTO/HWS internal spread, the preset asymmetry on
published-bar comparisons, and the fact that these are benchmark numbers and
not a generalization claim about an unseen user).

### 6.5 Regardless of outcome

The golden parity fixture is regenerated from the **ship artifact at the ship
preset** (`MODEL_COMPARISON.md` §5.1 — fixture and preset move together). The
currently committed `phaseM_kd_fresh_w1_fp16w_golden.json` was generated at
**E1**, which is the benchmark preset, not the app preset the fixture rule
points at; that is resolved in §8 as a chore, independent of the numbers.

---

## 7. The ledger entry

Appended to `test2400_seal.json["test-2400"]["unsealings"]` as entry **n = 4**,
with: authoriser, date, what was decoded, both presets, both tries, and the
publication site. Entries are never removed, and there is no fifth.

---

## 8. RESULT

*(appended after the six decodes; nothing above this line was edited afterwards)*

Run 2026-08-14, exactly as registered: **six decodes, one per (config, seed),
no warm-up, no retry, no fourth seed, nothing re-run after the numbers were
seen, and no crash.** `seal.py` logged the 2,400/2,400 overlap and the
`--unseal-test` override on every one. Ledger entry
`test2400_seal.json["test-2400"]["unsealings"][3]`. Dumps:
`~/ctc-train/ckpt/v2kd-fresh-w1{,-s4321,-s7777}/test2400_m_{e1,app}.jsonl`.
Greedy-CTC t1 **72.50 / 72.75 / 73.00 %** (val was 72.35 / 72.35 / 72.37).

Every table below is recomputed from the per-row dumps and reproduces
`eval_beam`'s printed metrics exactly.

### 8.1 Config A — AOSP STRIP 146,964 at E1, against the published bar

| metric | s1234 | s4321 | s7777 | **seed-mean** | sd | worst | **published bar** | **Δ mean** | **Δ worst** | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| t1 | 89.00 | 89.04 | 88.75 | **88.931** | 0.158 | 88.75 | 84.83 | **+4.10** | +3.92 | **PASS** |
| t3 | 92.71 | 92.62 | 92.71 | **92.681** | 0.048 | 92.62 | 91.04 | **+1.64** | +1.58 | **PASS** |
| t5 | 93.42 | 93.29 | 93.38 | **93.361** | 0.064 | 93.29 | 92.08 | **+1.28** | +1.21 | **PASS** |
| ≤3 (n=815) | 92.76 | 92.76 | 92.27 | **92.597** | 0.283 | 92.27 | 89.57 | **+3.03** | +2.70 | **PASS** |
| 4+ (n=1,585) | 87.07 | 87.13 | 86.94 | **87.045** | 0.096 | 86.94 | 82.40 | **+4.64** | +4.54 | **PASS** |

**All five clear on the seed-mean and on every individual seed. The §6.1 tier
rule is met: `phaseM_kd_fresh_w1` is TEST-VALIDATED.**

### 8.2 Config B — the shipping footing: app trie 98,081 at the shipping app preset

| metric | s1234 | s4321 | s7777 | **seed-mean** | sd | worst | **trie-matched bar** | **Δ mean** | **Δ worst** | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| t1 | 89.38 | 89.29 | 89.25 | **89.306** | 0.064 | 89.25 | 84.92 | **+4.39** | +4.33 | **PASS** |
| t3 | 93.75 | 93.79 | 93.83 | **93.792** | 0.042 | 93.75 | 91.54 | **+2.25** | +2.21 | **PASS** |
| t5 | 94.50 | 94.54 | 94.46 | **94.500** | 0.042 | 94.46 | 92.96 | **+1.54** | +1.50 | **PASS** |
| ≤3 (n=815) | 93.74 | 93.87 | 93.50 | **93.701** | 0.187 | 93.50 | 89.57 | **+4.13** | +3.93 | **PASS** |
| 4+ (n=1,585) | 87.13 | 86.94 | 87.07 | **87.045** | 0.096 | 86.94 | 82.52 | **+4.53** | +4.42 | **PASS** |

**All five clear, every seed. Worst-seed t5 margin +1.50** — against
`resbn80g`'s +0.75 and the Campaign-2 incumbent's +0.08 knife edge. §6.2 is
met: this is a **shipping validation** on the lexicon users run. It is **not**
an equal-footing claim, and the preset is the one fitted on `resbn80g`
(§2.1) — not this model's own optimum, which has never been sought.

### 8.3 Equal footing — the campaign's second qualified win, and its cleanest

Against `87.12 / 92.29 / 92.96 / 89.94 / 85.68` (both engines val-tuned, same
2,400 rows, same STRIP trie, same beam, same OOV rule — `FAIR_REMATCH.md` §5):

| metric | ours (mean) | worst seed | bar | **Δ mean** | **Δ worst** |
|---|---|---|---|---|---|
| t1 | 88.931 | 88.75 | 87.12 | **+1.81** | +1.63 |
| t3 | 92.681 | 92.62 | 92.29 | **+0.39** | +0.33 |
| t5 | 93.361 | 93.29 | 92.96 | **+0.40** | +0.33 |
| ≤3 | 92.597 | 92.27 | 89.94 | **+2.66** | +2.33 |
| 4+ | 87.045 | 86.94 | 85.68 | **+1.36** | +1.26 |

**All five clear on the seed-mean and on every seed.** Exact paired two-sided
McNemar on t1 against FUTO's val-tuned per-row output:

| seed | we win | they win | net | p |
|---|---|---|---|---|
| s1234 | 81 | 36 | **+45** | **3.5e-05** |
| s4321 | 89 | 43 | **+46** | **1.4e-04** |
| s7777 | 80 | 41 | **+39** | **5.0e-04** |

**Resolved on 3 of 3 seeds at p < 0.001.** Under §6.3 the permitted claim is a
**qualified equal-footing win** — the same tier ch 192 holds and *no stronger*,
as registered. What is new is only the resolution: ch 192 resolved 2 of 3, this
resolves 3 of 3. This is the campaign's first model to hold that win at
**2.91 MB** rather than 6.14 MB, and the first to hold it while also being
test-validated on the shipping footing.

### 8.4 The honest part of the equal-footing win: it is entirely the HWS half

Per-source top-1, seed-mean, from `cache/holdout_source_tags.json["test"]` (no
extra decode):

| engine / config | FUTO half (n=1,217) | HWS half (n=1,183) | spread |
|---|---|---|---|
| **FUTO ceiling, val-tuned** (the equal-footing bar) | **95.89** | **78.11** | 17.78 |
| ours, config A | 95.51 | **82.16** | 13.34 |
| ours, config B | 95.21 | **83.24** | **11.97** |
| *(prior reads: ch 128 95.07/80.56; `resbn80g` A 94.80/80.36, B 94.55/81.54)* | | | |

**On FUTO's own corpus half, FUTO's val-tuned engine beats us by +0.38.** The
entire +1.81 aggregate equal-footing lead is bought on the HWS half (+4.05).
Anyone reading §8.3 as "our model is better at swipe decoding" should read this
table first: what is demonstrated is better *coverage across two corpora*, and
one of the two corpora is FUTO's own. The 14-point internal spread that every
prior read of this split reported is, however, genuinely narrower here —
**11.97 at the shipping footing is the smallest ever recorded on test-2400**.

### 8.5 Against every other model ever decoded on this split

Config A (all at AOSP STRIP / E1, seed-means):

| model | file size | t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|---|
| **`phaseM_kd_fresh_w1`** (this decode) | **2.91 MB** fp16w | **88.931** | **92.681** | 93.361 | **92.597** | **87.045** |
| ch 192 (unsealing 1) | 6.14 MB fp32 | 88.36 | 92.65 | **93.50** | 91.37 | 86.81 |
| ch 128 (unsealing 1) | 2.80 MB fp32 | 87.92 | 92.33 | 93.00 | 91.08 | 86.29 |
| `resbn80g` (unsealing 3) | 1.14 MB fp32 | 87.68 | 92.18 | 92.82 | 90.80 | 86.08 |
| `fast_resbn80` (unsealing 2) | 1.14 MB fp32 | 87.29 | 91.89 | 92.82 | 91.17 | 85.30 |

**Best test numbers the campaign has ever measured on four of five metrics**,
at less than half of ch 192's bytes; **t5 is −0.14 behind ch 192** and that is
the one place a previous model stays ahead. On the shipping footing it beats
`resbn80g`'s config B on all five (+1.17 / +0.57 / +0.60 / +1.84 / +0.82).

The **≤3 stratum** — the stone this campaign chased for four phases — lands at
**92.60 (config A) / 93.70 (config B)**, +1.23 / +1.14 above its own val
figure and 1.2–1.9 pt above any prior model's test ≤3.

### 8.6 The pre-registered expectations, scored

| # | registered in §5 | outcome | verdict |
|---|---|---|---|
| A1 | config A clears all five published, seed-mean **and** every seed; t5 narrowest | all five, every seed; t5 margin +1.28 is indeed the narrowest of the five | **RIGHT** |
| A2 | config A clears all five **equal-footing** on the seed-mean | all five on the seed-mean **and** on every seed | **RIGHT** |
| A3 | McNemar resolves on 2 or 3 of 3 seeds | 3 of 3, p ≤ 5.0e-04 | **RIGHT** |
| B1 | config B clears all five, every seed, worst-seed t5 > +1.0 | all five, every seed, worst-seed t5 **+1.50** | **RIGHT** |
| B2 | config B t1 exceeds config A t1 by +0.1 … +0.5 | **+0.375** | **RIGHT** |
| §5.3 | "if one prediction misses its band, ≤3 is the one" | ≤3 is the **only** band miss, and it missed by the largest point error on both footings (+0.86 A, +1.04 B) while no other metric erred past 0.34 | **RIGHT** |
| §5.3 | config-B point predictions worse than config-A's | mean absolute error **0.36 (B) vs 0.30 (A)** | **RIGHT (marginally)** — though B was 5/5 in-band and A 4/5 |

**Point predictions vs measured:**

| | t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|
| config A, predicted → measured | 88.78 → **88.931** (+0.15) | 92.80 → **92.681** (−0.12) | 93.53 → **93.361** (−0.17) | 91.74 → **92.597** (**+0.86**) | 87.26 → **87.045** (−0.21) |
| config B, predicted → measured | 88.97 → **89.306** (+0.34) | 93.65 → **93.792** (+0.14) | 94.26 → **94.500** (+0.24) | 92.66 → **93.701** (**+1.04**) | 87.08 → **87.045** (−0.03) |

**Band coverage: 9 of 10.** The single miss is **config A ≤3**, which came in
at 92.597 against a band top of 92.593 — an overshoot of **0.004 pt**, i.e.
**one thirtieth of a single row** of the 815-row stratum, against a band that
had already been widened by 0.4 for exactly this metric. It is recorded as a
**miss** because that is what the rule says, and the rule is not adjusted after
the fact. What it actually establishes is the substantive finding:

**The ≤3 stratum has now been the outlier at three unsealings running, and its
val→test shift is not merely unstable but consistently the largest of the five:
+0.82 (U2), +0.02 (U3), and now +1.22 (A) / +1.14 (B) — the biggest yet.**
Short words are systematically easier on test-2400 than on val-9918, by an
amount no other metric approaches (every other shift here is within ±0.35), and
four unsealings of evidence say a ≤3 prediction should carry a ±1.3 band, not
±0.8. That is the durable methodological output of this decode.

Measured val→test shifts for this model, added to the §4.3 table:

| model / footing | t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|
| `phaseM_kd_fresh_w1`, config A | +0.18 | −0.09 | −0.11 | **+1.22** | −0.34 |
| `phaseM_kd_fresh_w1`, config B | −0.07 | +0.11 | +0.03 | **+1.14** | −0.68 |

### 8.7 What the test evidence now supports, at what tier

| claim | tier | basis |
|---|---|---|
| `phaseM_kd_fresh_w1` beats FUTO's **published** encoder+refinement ceiling on all five metrics | **test-validated**, seed-mean and every seed | §8.1 |
| …and on the **app lexicon** at the shipping preset, against the trie-matched published bar | **test-validated (shipping footing)**, every seed | §8.2 |
| …and on **equal footing** (both engines val-tuned) | **qualified equal-footing win**, McNemar-resolved 3/3 — the registered ceiling on this claim, not a general superiority claim | §8.3, §6.3 |
| the equal-footing lead is a *coverage* result, not a decoding-quality result | **stated as a limitation**: FUTO's engine is +0.38 ahead on its own corpus half | §8.4 |
| the model is the campaign's best on test-2400 | **true on 4 of 5 config-A metrics**; ch 192 keeps t5 by 0.14 | §8.5 |
| the fp16w **ship artifact** carries these numbers | **inferred**, not directly decoded: fp32 was decoded; fp16w ≡ fp32 to 0.00 on all five at the app footing on val and to ≤0.05 at E1 (§2.2) | §2.2 |
| anything about `v2pair-s1234` (option A, the coupled pair) | **val-only, unchanged.** It was not decoded and never will be | §1 |
| anything about a fifth read | **not authorised, and none is contemplated** | §3, §7 |

Every Campaign-2 caveat travels unchanged: T3 contributor contamination, the
dedup defect, the preset asymmetry on published-bar comparisons, the per-source
spread of §8.4, and the standing fact that these are benchmark numbers on a
worn split rather than a generalization claim about an unseen user.
**test-2400 has now been read four times. There is no fifth.**
