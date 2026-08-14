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
