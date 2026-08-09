# Three-way audit — our CTC models vs FUTO's engine vs the shipped transformer NN

**Auditor stance: adversarial verification.** Every number below is either (a)
recomputed by this audit from per-trace artifacts, (b) quoted from a committed
document with its provenance named, or (c) freshly measured on non-test data. No
new decode of test-2400 was run — the seal is spent (`AUDIT_FINAL.md` §7) and
`seal.py` guards it; the test numbers for our models come from the existing
audited `test2400_e1.jsonl` dumps, which this audit re-read and re-scored
independently.

Date: 2026-08-08. Auditor hardware for anything marked [this audit]: Intel Core
Ultra 9 275HX (WSL2), onnxruntime 1.22.1.

## 0. What was independently recomputed vs quoted

Recomputed by this audit, from artifacts, not doc footers:

* **All six test-2400 dumps** (`~/ctc-train/ckpt/{phaseE-FINAL,phaseE-E3b-hws3x}{,-s4321,-s7777}/test2400_e1.jsonl`):
  2,400 rows each, contiguous `idx` 0–2399; target words match the raw
  `~/ctc-train/data/test_hwsfuto.jsonl` **2,400/2,400 exactly**; strata n=815/1,585;
  t1/t3/t5, ≤3, 4+, per-source futo/hws splits, and seed means all reproduce the
  committed tables in `RESULTS.md`/`AUDIT_FINAL.md` **to the second decimal, every
  cell checked**.
* **OOV accounting**: rebuilt the STRIP-normalized lexicon from
  `~/ctc-train/data/en_wordlist.combined` → exactly **146,964** words; exactly
  **86** of the 2,400 test targets are OOV, all carried `rank=-1` (counted as
  misses); rank histogram matches `AUDIT_FINAL.md` §1 ({-1:150, 0:2131, 1:74,
  2:16, 3:13, 4:9, 5:2, 6:3, 7:2}).
* **Greedy top-1 from the dumps** (not previously published for these models —
  see §1 table).
* **Artifact identity**: `artifacts/ch128_s1234.onnx`, `ch192_s1234.onnx`,
  `fast_resbn72_s1234.onnx` sha256-match the checkpoint ONNX files the audited
  decode ran on, byte-for-byte.
* **Old-NN sanity re-run** [this audit]: the shipped transformer was re-run
  locally through the app repo's own production-equivalent harness
  (`tools/test_cli_predict.py --frame-remap identity --training-features
  --production`, encoder+decoder from `src/main/assets/models/`) on a fresh
  random 500-row sample of val-9918 (seed 20260808) — see §6. Result is
  consistent with the committed val figures within sampling noise.

Quoted (cannot be recomputed here): FUTO floor/ceiling and old-NN full-split
numbers. FUTO's per-row caches and the old NN's full-split dumps live on-device
(`~/.cache/cleverkeys-test/`, Termux), not on this machine; the committed docs in
the app repo are the source of record. The app repo was treated read-only.

## 1. Test-2400 — the verified three-way table

Same 2,400 traces for every engine, original file order, **OOV-vs-own-lexicon
counted as a miss** (the registered cross-engine methodology of
`docs/eval/2026-07-24-test2400-head2head.md`). Strata: ≤3-char n=815, 4+ n=1,585.

| engine | t1 | t3 | t5 | ≤3 t1 | 4+ t1 | greedy t1 | provenance |
|---|---|---|---|---|---|---|---|
| **ours ch192** (3-seed mean) | **88.36** | **92.65** | **93.50** | **91.37** | **86.81** | 74.56 | recomputed [this audit] from `test2400_e1.jsonl` dumps; sealed decode 2026-08-08, RTX-5080-laptop box, E1 preset, 146,964-word STRIP trie |
| **ours ch128** (3-seed mean) | **87.92** | **92.33** | **93.00** | **91.08** | **86.29** | 70.47 | same |
| FUTO ceiling (enc + `magic_macaw` DFSMN + FUTO Viterbi beam) | 84.83 | 91.04 | 92.08 | 89.57 | 82.40 | 69.12 | `docs/eval/futo-decoder-eval-notes.md` "CEILING 2026-07-31"; FUTO's actual `.pte` weights via ExecuTorch 1.2.0, on-device proot Ubuntu ARM64; FUTO's published scoring preset; 131,544-word DROP lexicon (re-measured **unchanged overall** on the 146,964 STRIP one; strata not republished) |
| FUTO floor (enc-only, textbook trie beam) | 79.25 | 87.71 | 89.58 | 82.45 | 77.60 | 43.96 | same doc, "FINAL RESULTS" section |
| old shipped NN (transformer, prod beam-6) | 74.62 | 84.33 | 87.42 | 89.45 | **67.00** | — | `docs/eval/2026-07-24-test2400-head2head.md`; `tools/test_cli_predict.py --production` on-device Termux ARM64, 98,140-word `en_enhanced` dict (t3/t5 strata: ≤3 95.46/96.32, 4+ 78.61/82.84) |
| (context: our geometric SHARK2) | 67.50 | 78.88 | 81.79 | 69.33 | 66.56 | — | same head2head doc |

Worst single seed still clears every FUTO-ceiling number: ch192 min
87.88/92.54/93.46/90.92/86.31; ch128 min 87.83/92.08/92.92/90.55/86.06
(recomputed). Per-source seed-mean (recomputed): ch192 FUTO-half 95.32 / HWS-half
81.21; ch128 95.07 / 80.56 — a ~14-pt internal spread the aggregate hides.

Greedy note: our greedy values are exact-string collapsed-argmax matches
recomputed from the dumps (per-seed ch192 75.46/73.00/75.21; ch128
71.08/69.62/70.71); they were not previously published for these models. The
FUTO greedy anchors are that engine's own committed values. The old NN has no
greedy equivalent (autoregressive decoder, not CTC).

**Statistical resolution** (unpaired binomial, per `AUDIT_FINAL.md` §5):
ours-vs-FUTO-ceiling resolves on t1 (z≈3.1–3.6) and 4+ (z≈3.0–3.4) only; t3/t5/≤3
are positive but within noise. Ours-vs-old-NN is resolved beyond any doubt:
Δt1 +13.74 → z≈12, Δ4+ +19.81 → z≈14 [this audit].

## 2. Val-9918 — the verified table, including the val-only fast models

Full held-out val split, N=9,918, OOV=miss, strata ≤3 n=3,389 / 4+ n=6,529. Our
rows are 3-seed means at the E1 preset on the 146,964-word STRIP trie; FUTO rows
use FUTO's published preset and its own a-z-normalized lexicon.

| engine | t1 | t3 | t5 | ≤3 t1 | 4+ t1 | provenance |
|---|---|---|---|---|---|---|
| **ours ch192** | **88.06** | **92.32** | **93.08** | 90.86 | **86.62** | `PHASE_E.md` §5, 3 seeds |
| **ours ch128** | 87.88 | 92.23 | 92.96 | **90.98** | 86.26 | `PHASE_E.md` §5, 3 seeds |
| ours fast_resbn80 ⚠ val-only | 87.47 | 92.13 | 92.89 | 90.35 | 85.98 | `PHASE_F.md` §8, 3 seeds |
| ours fast_resbn72 ⚠ val-only | 87.27 | 92.09 | 92.87 | 90.49 | 85.60 | `PHASE_F.md` §14.1, 3 seeds |
| FUTO ceiling | 85.52 | 91.54 | 92.80 | 89.29 | 83.57 | app repo `2026-07-24-test2400-head2head.md` "val corroboration" (run 2026-08-06, on-device ExecuTorch, a-z-normalized lexicon) |
| FUTO floor | 78.84 | 88.01 | 90.11 | 81.17 | 77.62 | same |
| old shipped NN | 76.01 | 85.53 | 87.82 | 89.23 | 69.15 | same (production beam-6, on-device) |
| (context: our geometric) | 67.69 | 78.36 | 81.49 | 70.23 | 66.37 | same |

The **fast_resbn80/72 rows are val-validated only and must never be quoted as
test results** — they were trained after the test seal was spent and `seal.py`
forbids decoding them on test. Their t5 margins over the FUTO-ceiling bar are
+0.09 and +0.07 with worst-seed margins of +0.05 and +0.01 — real passes under
the registered gate, but with no margin to spend.

Strictest view for our headline models: on the val rows never used for preset
tuning or checkpoint selection (`4959:9918`), ch192 still clears all five
(87.58/92.03/92.85, ≤3 90.67, 4+ 85.98 — `PHASE_E.md` §5), with t5 at +0.05.

## 3. Latency & size

**These columns are not cross-comparable and no single-number speedup claim is
legitimate.** Our figures are *encoder-only forward pass*, single-thread batch-1,
laptop x86 (Core Ultra 9 275HX class, WSL2, ONNX Runtime); the old NN's committed
figures are the *full decode pipeline* (encoder + autoregressive beam-6
transformer decoder + rerank) on a phone (Termux ARM64, 4 threads). No committed
on-device latency exists for FUTO's engine (its evals ran uninstrumented in proot),
nor for our CTC models (never yet run on-device; the 147k-word trie beam — not the
encoder — is expected to dominate the per-swipe budget there, `PHASE_F.md` §11.5).

| model | files / params | bytes on disk | measured latency | provenance |
|---|---|---|---|---|
| old shipped NN | `swipe_encoder_android.onnx` + `swipe_decoder_android.onnx` (d_model 256, 6 enc + 4 dec layers; decoder already int8-dynamic: 4.2M int8 + 146K fp params) | 5,317,537 + 4,975,510 = **10,293,047** | full pipeline: **~178 ms/trace** on-device (337.8 traces/min, 4 threads, paired benchmark, `2026-08-06-offline-decoder-speedup.md`); ~437 ms/trace in the head2head run (~17 min / 2,336, incl. warmup); **~55 ms/trace** on this audit's laptop (1,097 tr/min, 4 threads) [this audit] | sizes verified [this audit]; app spec's "<100 ms total" is a target, not a measurement |
| ours ch192 | 1,525,378 params | 6,144,249 | 0.877 ms enc-only (0.911–0.934 on the Phase-F harness) | `AUDIT_PREDECODE.md` §7 / `PHASE_F.md` §0 |
| ours ch128 (ship) | 689,282 | 2,799,865 | 0.455 ms enc-only (0.472–0.475 Phase-F harness) | same |
| ours fast_resbn80 ⚠ val-only | 279,346 | 1,142,727 | 0.215 ms enc-only | `PHASE_F.md` §8/§14 |
| ours fast_resbn72 ⚠ val-only | 229,642 | 944,487 | 0.186 ms enc-only | `PHASE_F.md` §14.1 |
| FUTO encoder + decoder | 635K + 304K params (`.pte`, XNNPACK-delegated) | 2,649,856 + 1,247,468 = 3,897,324 | **no committed figure** (ExecuTorch on-device; eval throughput not recorded) | `futo-decoder-eval-notes.md` Phase 1–2 |

What *is* legitimately comparable: (a) file size — ch128 is 27% of the old NN's
bytes, fast_resbn72 is 9%; (b) same-machine full-pipeline throughput [this
audit]: the old NN's full production decode ran at ~55 ms/trace on this laptop
while the campaign's full CTC decode (encoder + Python beam-100 over the 147k
trie) logged 78–94 traces/s ≈ 11–13 ms/trace on the same box (`test2400_e1.log`
throughput, single process) — directionally a ~4–5× faster pipeline despite a
17× wider beam, but across different code paths (Python beam vs Python
autoregressive ONNX loop), so treat as indicative only.

## 4. Consolidated caveat register (travels with every claim above)

1. **Preset asymmetry (ours-vs-FUTO, the largest threat).** Our decode preset
   (E1) was grid-searched on val-9918; FUTO's ceiling is quoted at its own
   published preset. Measured control: at FUTO's published preset our ch192
   clears only **3 of 5** val bars (85.78/91.66/92.67, ≤3 88.10, 4+ 84.58); the
   tuning is worth **+2.29 t1** — comparable to the whole test t1 margin. Whether
   FUTO's emissions have similar sweep headroom is untestable here (no weights on
   this machine). `AUDIT_FINAL.md` §6.1. No headline may omit this.
2. **Contributor contamination, both directions.** Ours: T3 applies no
   participant exclusion — every contributor of every val/test row is in our
   training data (98.4% of HWS holdout rows share a participant with training);
   no contributor-clean subset exists. Theirs: **5,273 of 12,299 unique holdout
   traces (43%) are bit-exactly in the HF train split FUTO trained on** (0 in HF
   dev/test) — the app repo's "FUTO-held-out" description is wrong. Bit-exact
   leakage on our side (145 test rows via the dedup-key defect) was measured
   harmless: leaked rows score 4.34 pt *below* comparable non-leaked ones, and
   removing them costs 0.20 t1 with all bars still clearing on every seed.
   All numbers are benchmark numbers, **not** generalization claims about unseen
   users.
3. **Lexicon differences.** Ours/val-bar: same 146,964-word STRIP trie (the "our
   larger lexicon is conservative" claim does **not** apply on val). Test bar:
   published on the 131,544-word DROP trie, re-measured unchanged overall on the
   STRIP one, but **its ≤3/4+ strata were never republished** — test strata are
   compared across normalizers. Old NN: its own 98,140-word `en_enhanced` dict,
   which — despite being smallest — covers the test targets *best* (64 OOV
   forced misses vs our 86, a ~0.9 pt handicap **against us** in §1; our lead
   survives it ~15×) [this audit].
4. **Val-only status of the fast models.** fast_resbn72/80 (and every Phase-F
   number) have zero test evidence, structurally and permanently — the seal is
   spent. Also unmeasured: the no-distillation control (all Phase-F students
   were distilled from our ch192; the teacher's contribution is unattributed).
5. **Latency non-comparability.** §3: encoder-only laptop ms vs full-pipeline
   phone ms. Neither our models nor FUTO has committed on-device numbers; the old
   NN has no committed encoder-only number. Statistical: 3 of 5 test bars
   (t3/t5/≤3) vs FUTO are within ~2σ; ≤3 within ~1.2σ — point-estimate wins only.
6. **Harness provenance.** The "old shipped NN" numbers come from the app repo's
   Python production-equivalent harness (validated against the shipped config,
   one immaterial −0.12 pt alpha deviation documented), not from the Android app
   itself. FUTO's numbers come from a port of its C++ beam running FUTO's real
   weights; its ceiling remains 8.5 pt below FUTO's own paper number on FUTO's
   full test split, so it is a **conservative** estimate of FUTO-on-this-data
   (partly harder subset, partly port fidelity). Arm *selection* in our campaign
   used full val; the preset sweep and checkpoint selection respected a holdout.
   7 traces are bit-shared between val-9918 and test-2400.

## 5. Verdicts

**Ours vs FUTO (as registered).** Verified as registered, and only as
registered: on the sealed test-2400, both ch192 and ch128 exceed all five
published FUTO-ceiling numbers on every one of six independent seeds — t1 +3.53
/ +3.09, 4+ +4.41 / +3.89 statistically resolved; t3, t5, ≤3 positive but within
noise — **with our preset tuned on the holdout family while FUTO's is not**. At
matched (published) presets the same model clears 3 of 5 on val. The one
sentence that may never be written is that we beat FUTO's decoder on equal
footing; the evidence cannot support it and the seal prevents ever testing it on
test-2400. The FUTO ceiling itself is a conservative floor on FUTO's true
engine (paper: 93.30 on its full test split), which cuts the other way.

**Ours vs the old shipped NN — unambiguous, and the only pairing that is.**
Same traces, same OOV-as-miss rule, a lexicon handicap in the old NN's favour,
and the gap is still **+13.7 t1 on test (88.36 vs 74.62; z≈12)** and +12.1 on
val (88.06 vs 76.01). The old NN's failure mode is structural: **4+-char top-1
collapses to 67.00 on test / 69.15 on val, vs our 86.81 / 86.62 — a ~20-pt gap**
(z≈14) — reproduced independently by this audit's fresh 500-row val re-run (old
NN 65.4 on 4+, n=315). Even where the old NN is best — short words, ≤3 89.45 —
both our models beat it (91.37 / 91.08) as well as being ~27% of its bytes
(ch128), with the same-machine full-pipeline throughput favouring the CTC path
~4–5× (§3, indicative). Every axis — accuracy, size, speed, strata — favours the
CTC models; there is no registered caveat that could plausibly reverse a 14-pt
resolved gap. Replacing the shipped transformer is supported without
qualification; the only open question is on-device end-to-end latency of the
trie beam, which was never the transformer's strength either.

**Old NN vs FUTO.** FUTO's engine beats the shipped transformer decisively at
both floor and ceiling on overall t1 (ceiling +10.2 test / +9.5 val), entirely
on long words (82.40 vs 67.00 test). The transformer's one durable strength is
short words, where it ties FUTO's ceiling (89.45 vs 89.57 test; 89.23 vs 89.29
val) and beats FUTO's floor by ~7 pt — the frequency/context rerank, not the
network, is doing that work. Directionally the same preset caveat applies in
FUTO's favour here too (the old NN's beam params are its own tuned production
config; FUTO's are its published ones), but no plausible tuning closes 15 pt on
4+.

## 6. Findings, including anything contrary to the campaign's claims

1. **No numerical discrepancy found.** Every recomputed cell (120+ across six
   dumps: aggregates, strata, per-source, seed means, OOV counts, rank
   histograms, artifact hashes, lexicon size) matches the committed tables
   exactly. The campaign's arithmetic survives adversarial recomputation.
2. **The old-NN val anchor reproduces.** Fresh local re-run [this audit], random
   500 val rows (seed 20260808), shipped ONNX + production-equivalent path:
   t1/t3/t5 OOV-as-miss **74.0 / 84.6 / 87.2** vs committed full-val
   76.01/85.53/87.82 — within ~1σ of sampling noise on every metric (SE ≈1.9 pt
   at n=500); ≤3 88.65 vs 89.23; 4+ 65.4 vs 69.15 (n=315, ≈1.4σ). The committed
   old-NN numbers are genuine, and were not unflattering to the old NN.
3. **Previously unpublished: our greedy top-1** (§1) — ch192 ~74.6, ch128 ~70.5
   on test-2400. Both sit *above* FUTO's ceiling greedy (69.12) and far above its
   floor greedy (43.96). Consistent with the campaign's account of why FUTO's
   refinement head had leverage (43.96 base) and ours did not (Phase-1 G4 miss).
4. **The OOV asymmetry (§4.3) was nowhere stated**: the cross-engine "own
   lexicon, OOV=miss" rule forces 86 misses on us vs 64 on the old NN — the
   three-way tables are mildly tilted *against* our models, not for them. Worth
   one line in any future headline; changes no verdict.
5. Known-and-disclosed items verified still true rather than re-litigated: the
   withdrawn "+0.21 headroom" scoring claim (`RESULTS.md` retraction), the false
   "removed bit-exactly" dedup sentences in `PHASE_D/E.md` (corrected via
   `AUDIT_FINAL.md` §6.2), the undisclosed-then-disclosed 120-row smoke decode,
   and the seal-hygiene note that 3 test rows were re-decoded post-audit to test
   the `--unseal-test` guard branch (no number from that run is used anywhere,
   including here).

*Files this audit read but did not modify: everything under
`/home/will/git/swype/CleverKeys` (read-only), `~/ctc-train/ckpt/*/test2400_e1.jsonl`
(existing audited dumps), `~/ctc-train/data/{test,val}_hwsfuto.jsonl`,
`en_wordlist.combined`, `cache/holdout_source_tags.json`. New decodes run: old-NN
val-500 sample only. Test-2400 was not decoded by any engine for this audit.*
