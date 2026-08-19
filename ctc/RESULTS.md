# CTC Swipe Encoder — Training Results

# Phase O (2026-08-18/19): five new non-Latin scripts — and the discovery that the probe they are measured on is unreliable

Full record: `ctc/PHASE_O.md`. Phase I-B showed a script with no corpus can be
launched from English motor residuals (Cyrillic, in-dict t1 76.21 → 77.41 at the
tuned λ). Phase O asks which other scripts that unlocks and what each costs.

**Inventory.** Of 36 non-Latin layouts the app ships, exactly **two scripts have
both a layout and a lexicon in-repo**: Russian (done) and **Greek**. Five more
are one wordfreq list away (uk/he/bg/mk on the app's own rank formula; Serbian is
blocked because wordfreq's `sh` list has **zero** Cyrillic words in its top 80 k).
Armenian and Georgian are blocked on a dictionary. Devanagari, Bengali, Gujarati,
Kannada, Tamil, Sinhala and Hangul are **structurally** blocked — those layouts
expose 7–20 letters on centre keys and put the rest on corner slots a swipe
cannot reach. Arabic/Persian/Urdu are priced and not attempted (the hamza
carriers أ إ آ are corner-only).

**The extractor is validated, not asserted.** `app_layout.py` replicates the
app's own `KeyboardGeometry.computeKeyRects` + `buildMappedLayout` and reproduces
`en_qwerty.json` from the app's `latn_qwerty_us.xml` to **4.7e-4** — the app frame
and the training frame are the same frame. As a free by-product, the app's own
ЙЦУКЕН grid sits **3.4e-3** from the Yandex grid the ru model trained on, so that
model is deployable on app geometry as exported.

**The central result: the synthesis holdout is not a trustworthy probe.** On
Russian — the only script with both a synthesis holdout and a real corpus — the
same two models rank in opposite orders, both significantly:

| model | ru synthesis holdout | ru **REAL** swipes |
|---|---|---|
| `ru_synth_ch80` (script-trained) | 81.10 | **77.41** |
| `phaseM_kd_fresh_w1` (shipped **English**, zero-shot) | **83.38** | 76.32 |
| Δ / paired exact McNemar | −2.28, **p = 7.1e-12** | +1.09, **p = 0.0099** |

Three defects, each measured: it **over-credits capacity** (English ch192 beats
English ch80 by +7.14 on the holdout and by +0.53, n.s., on real swipes — which
is exactly what flips the ranking); it **inverts the λ choice** (holdout prefers
λ 1.1 by +4.70, real data prefers λ 2.0 by +1.20); and its **length mix is 12×
off** (3.3 % short words vs real usage's 38.7 %).

**Consequences that change the campaign's picture.** The shipped English model
reaches **76.32 in-dict top-1 on real Russian swipes** with nothing but the right
layout and the right trie — genuinely out of distribution (it was trained with 26
active slots, never 31), in the shipped geometric engine's 71–77 band, and within
1.1 of the purpose-trained model. `ctc-architecture-and-multiscript-guide.md`
§3.1's "a Latin-trained model does not zero-shot another script" is **too strong
as stated**; what survives is the emissions half — zero-shot greedy is 18.62
against the script model's 37.13, so English is leaning on the trie. The price
list for a new script now reads: **layout + trie wiring ≈ 76, per-script
synthesis ≈ +1.6 (p = 1.4e-4 at matched capacity), real data ≈ +13.** The
expensive half is the data, not the model.

**Five new models, exported.** Same recipe as `phaseIB-ru-synth`, verbatim,
single seed. On their own 10 k synthesis holdouts at the adopted preset:
**el 82.54 · uk 79.27 · bg 71.80 · mk 71.69 · he 65.36** in-dict t1. Four clear
the registered ≥70 gate; **he fails it** at the adopted preset (70.28 at λ 1.1)
and is exported flagged. Against the capacity-matched English control every
script model wins by +4.9…+7.3; against the 3×-capacity ship model every one
loses by −0.6…−3.8 — the exact pattern the ru calibration says to expect, and the
reason neither column is quoted as an accuracy.

**Two registered arms refuted.** The per-script λ sweep picked 1.1 in all five
scripts and the ru control showed that preference is inverted against real data —
λ 2.0 adopted, sweeps discarded, all fixtures frozen at 2.0. Warm-starting from
the English ch80 (`phaseO-ru-initH`) is +0.88 on the holdout and **−0.14 on real
(p = 0.69)**; not promoted.

**Falsification.** With key centres permuted, every model on every script decodes
at **0.00 top-1 and 0.00 greedy**. The geometry is entirely load-bearing.

**Two app defects found.** The shipped `grek_qwerty.xml` declares
`script="latin"` (the `srcs/` copy says `greek`), and `langpack-el` carries **no
final sigma** — 25.7 % of the pack is σ-final where Greek writes ς, and the two
are different keys in different rows, so an unrepaired lexicon would train and
score one Greek word in four against the wrong endpoint. Phase O restores final
σ→ς by rule; the app must do the same.

**Evidence tier.** el/uk/bg/mk/he are **synthesis-trained, synthesis-holdout-only,
single-seed, and calibrated against Russian rather than measured on their own
script**. No real swipe corpus exists in any of them (`DATASET_SCOUT.md` §4.4),
no sealed test split can ever exist, four of the five lexicons are not the app's,
and no on-device measurement was taken.

---

# The fourth unsealing (2026-08-14) — the shipped model is test-validated on both footings, and takes a qualified equal-footing win at 2.91 MB

**The final read of test-2400. The ledger closes at 4 and there is no fifth.**
Pre-registered in `ctc/UNSEALING_4.md` §1–§7 and **pushed at `b91f179` before
any decode**; executed exactly as registered — six decodes, one per (config,
seed), no warm-up, no retry, no crash. Authority: the user's directive of
2026-08-13/14 (one final pre-registered unsealing plus an adversarial audit,
for whichever model ships) applied to the `PHASE_M.md` §11.2 recommendation,
**option B — the pair-distilled single model** `phaseM_kd_fresh_w1_s1234_fp16w`
(2.91 MB, 1.5 M params, one ONNX session).

| footing | seed-mean (1234/4321/7777) | bar | Δ | worst-seed status |
|---|---|---|---|---|
| val, AOSP, E1 | 88.750 / 92.773 / 93.473 / 91.373 / 87.387 | campaign 88.30/92.60/93.26/91.27/86.77 | +0.45/+0.17/+0.21/+0.10/+0.62 | all five, every seed |
| **test, AOSP, E1 (config A)** | **88.931 / 92.681 / 93.361 / 92.597 / 87.045** | published `84.83/91.04/92.08/89.57/82.40` | **+4.10 / +1.64 / +1.28 / +3.03 / +4.64** | **all five clear, every seed** |
| **test, app trie, app preset (config B)** | **89.306 / 93.792 / 94.500 / 93.701 / 87.045** | trie-matched `84.92/91.54/92.96/89.57/82.52` | **+4.39 / +2.25 / +1.54 / +4.13 / +4.53** | **all five clear, every seed; worst-seed t5 +1.50** |
| **test, equal footing (config A)** | same as config A | val-tuned `87.12/92.29/92.96/89.94/85.68` | **+1.81 / +0.39 / +0.40 / +2.66 / +1.36** | **all five clear, every seed** |

**Evidence tier: `phaseM_kd_fresh_w1` moves from val-only to TEST-VALIDATED**,
on both footings, on the seed-mean and on every individual seed — the
pre-registered rule, unchanged from all three prior unsealings.

**Equal footing — a qualified win, and what it is not.** Exact paired two-sided
McNemar on top-1 against FUTO's val-tuned per-row output resolves on **3 of 3
seeds**: +45 (p 3.87e-05), +46 (p 7.69e-05), +39 (p 4.99e-04) — *corrected
2026-08-14; the first write-up printed 3.5e-05 / 1.4e-04 / 5.0e-04, two of them
hand-transcribed from a four-decimal print at precision that had not been
computed. Counts unchanged, verdict unchanged; erratum in `UNSEALING_4.md`
§8.3.* Under the rule
registered in `UNSEALING_4.md` §6.3 the permitted claim is a **qualified
equal-footing win** — the same tier ch 192 holds and no stronger — now held at
**2.91 MB instead of 6.14 MB** and resolved on three seeds instead of two.
**Two limitations travel with it, and must be quoted with it:**

* **The entire lead is bought on the HWS corpus half.** Per-source top-1:
  FUTO's val-tuned engine **95.89 futo / 78.11 hws**; ours **95.51 / 82.16**
  (config A) and **95.21 / 83.23** (config B). *On FUTO's own corpus half
  FUTO's engine beats us by +0.38.* What is demonstrated is better coverage
  across two corpora, not better decoding per se. (The internal spread does
  narrow to **11.97** at the shipping footing — the smallest ever recorded on
  this split, against 13.0–14.9 for every prior read.)
* **ch 192 keeps top-5** (93.50 vs 93.361, −0.14). On the other four config-A
  metrics this model is the best the campaign has ever measured on test-2400.

**The ≤3 stratum, at last, and a methodological finding.** ≤3 lands at
**92.597** (config A) / **93.701** (config B) — 1.2–1.9 pt above any prior
model's test ≤3, and the stone Phases J–M chased is emphatically cleared on the
sealed split. It is also the campaign's most badly predicted metric for the
third unsealing running: its val→test shift is **+1.22 / +1.14** here against
±0.35 for every other metric, and four unsealings now say short words are
systematically easier on test-2400 than on val-9918. A ≤3 prediction wants a
±1.3 band, not ±0.8.

**Pre-registered expectations: 7 of 7 verdicts right; band coverage 9 of 10.**
The one band miss is config-A ≤3, which overshot the (already widened) band top
by **0.004 pt** — a thirtieth of one row of 815. It is reported as a miss
because the rule says so. Full scoring: `UNSEALING_4.md` §8.6.

**What was NOT decoded.** The coupled pair `v2pair-s1234` (option A, more
accurate on val, 4.39 MB, two sessions) was deliberately left sealed and is
**val-only permanently**. Every other val-only artifact in this repo stays
val-only for the same reason: there is no fifth unsealing.

**Ship artifacts and the fixture** (fixture and preset move together —
`MODEL_COMPARISON.md` §5.1):

| file | bytes | sha256 |
|---|---|---|
| `phaseM_kd_fresh_w1_s1234_fp16w.onnx` ← **ship** | 3,052,318 | `84718e6ebc8020176f27b9668e50922a765c96838307b640a8db9ab0549e88e5` |
| `phaseM_kd_fresh_w1_fp16w_golden.json` (at the ship preset) | 140,462 | `2a449c4f2de19505131b396655ae01d3e3c325e40249446ff6e7a40c2b27559c` |
| `phaseM_kd_fresh_w1_s1234.onnx` (fp32, decoded) | 6,068,519 | `b71911da3407abc0b113bbc662a1929953b04dcaf7650d848a7e897605a9bf80` |
| `phaseM_kd_fresh_w1_s4321.onnx` (fp32, decoded) | 6,068,519 | `f7cb72c07e1d5a920e5ceb93b4f6cf241bf0c9dcc630bcd1117d4fdf38d2daf1` |
| `phaseM_kd_fresh_w1_s7777.onnx` (fp32, decoded) | 6,068,519 | `c55cc3b055cf2db2b198c03b3fae688aad1930058dfed3902296aa08fd6510d7` |

```kotlin
// shipping preset, unchanged from resbn80g and now test-validated on this model
CtcScoringParams(gamma = 0.9, lambda = 4.0, beta = 0.25, alpha = 0.0,
                 gammaPrune = 0.25, betaPrune = 0.9882)
```

Disclosures that are part of the record, not footnotes to it: the **fp32**
graphs were decoded and the **fp16w** artifact ships — bridged by a measured
**0.00** val delta on all five at the app footing and ≤0.05 at E1, not by
assumption; the config-B preset was fitted on `resbn80g` and **has never been
swept for this model family**, so config B is a transplanted-preset footing
that plausibly understates it; and every Campaign-2 caveat travels unchanged
(T3 contributor contamination, the dedup defect, the preset asymmetry on
published-bar comparisons, benchmark numbers rather than a generalization claim
about an unseen user). **test-2400 has now been read four times.**

# Phase M (2026-08-14): the close — a distilled single model takes all eleven bars on EVERY seed, the coupling optimum is interior, E4/E6/E2 all die by their own rules

Full record `ctc/PHASE_M.md`. Twelve arms, every one pre-registered.

* **NEW SINGLE-MODEL FINALIST — `phaseM_kd_fresh_w1_s1234_fp16w` (2.91 MB)**:
  a single 1.5 M-param model **distilled from the coupled pair**, clearing
  **all eleven campaign bars on ALL THREE seeds and on the seed-mean**
  (88.750 / 92.773 / 93.473 / **91.373** / 87.387 · dvorak 91.82 ·
  dvorak-app 91.10 · azerty 84.53 · qwertz 83.97 · german 81.30 ·
  spanish 89.53; margins +0.10 … +2.90). It supersedes `sw2345` (10/11).
  The mechanism: the teacher was **alignment-consistent** (a gated coupled
  pair) — and the student's own init turned out **not** to matter (fresh beat
  warm-started).
* **⚠ RETRACTION:** Phase L's single-model all-eleven claim (`L1 member A`)
  **does not survive five seeds** — t3 −0.024, qwertz −0.156, **9/11**.
  Retracted in place; see PHASE_M §7.1.
* **The pair strengthens: 11/11 campaign bars on 5 of 5 seeds**, seed-mean
  margins +0.12 … +2.76, at 4.39 MB.
* **Coupling weight: 0.3 is interior-optimal** on a four-point sweep
  (0/0.1/0.3/1.0). Agreement rises monotonically with the weight
  (92.09 → 98.58 %) while the mix's transfer edge **collapses at 1.0**
  (dvorak −1.95) — over-coupling kills the diversity averaging feeds on.
* **E4 `w_real` DROPPED** (dvorak −2.81); **E6 geometric prior DROPPED** by
  its own kill criterion (four val bars past −0.15); **E2 synthesis REFUTED**
  at three paired seeds. No element was retried after failing.
* **Crown NOT won:** the best single model beats all five of the
  `mix2-i8f16` card's **val** numbers at the seed-mean but misses four
  transfer axes (0.06–0.43). Bar 1 (pair ≥ card, 2/3 seeds) **not met**.
* **Gate band predictions: 12 of 12 correct** above 98 % agreement.
* **test-2400 SEALED throughout L and M** (ledger 3). Unsealing is the
  orchestrator's act — **taken on 2026-08-14 for the shipped model only**;
  see "The fourth unsealing" at the top of this file and `UNSEALING_4.md`.
  The single model is now **test-validated**; the pair stays val-only.

# Phase L (2026-08-13): pipeline v2 — the alignment gauge is TRAINABLE, the ≤3 stone falls to a single model on the seed-mean, and English synthesis is refuted

**Settled at three seeds.** Seven 188 k arms, every one pre-registered before
launch, every gate committed before its decode. Full record `ctc/PHASE_L.md`.

* **⚠ BAR 2: met at three seeds, RETRACTED at five** (PHASE_M.md §7.1). The
  two tie margins went under with two more seeds — t3 **−0.024**, qwertz
  **−0.156** — leaving `L1 member A` at **9/11 seed-mean** (per-seed
  [11, 8, 8, 6, 8]). It does **not** supersede `sw2345` (10/11, missing ≤3);
  the two are non-dominating mirror images. What survives: member A clears
  the **≤3 stratum on a five-seed mean (91.358, +0.088)**.
* **BAR 1 NOT MET.** The gated pair had to beat every `mix2-i8f16` card
  number on ≥2/3 seeds; per-seed tallies were 10/11, 8/11, 6/11.
* **All FIVE L1 pairs clear all eleven CAMPAIGN bars on EVERY SEED**
  (11/11 × 5, five-seed mean margins +0.12 … +2.76 — PHASE_M.md §7.2) — a footing nothing in Phases A–K reached (Phase J: 5/11
  every-seed). The pair seed-mean beats all five card val numbers
  (t1 +0.15, ≤3 **+0.19**) at 4.39 MB; the card keeps a 0.3–0.4 edge on
  dvorak/dvorak-app/spanish. Ship standing unchanged pending an orchestrator
  decision; the single-model finalist changes.
* **The CTC alignment gauge is trainable.** Coupled pairs (mutual per-frame
  KL, identical batches) reach **98.05–98.33 % agreement, 6 of 6 over the
  gate**; the paired `--pair-weight 0` control — identical in every other
  respect — sits at **92.09 %** (2 of 47 evals) and its mix collapses to
  **greedy 29.10** against the coupled pair's 72.92. Pair mixability stops
  being a lucky draw and becomes a recipe.
* **Targeted English synthesis (E2) is REFUTED** at three paired seeds:
  sign-consistently **−0.21 t1 / −0.12 t5 / −0.22 4+**. Its single-seed ≤3
  (+0.06) and euro-layout gains (azerty +0.86, qwertz +0.93) **did not
  reproduce**. 300 k license-clean endpoint-validated rows bought a negative;
  `english_synth.py` and the pools stay committed as the documented negative.
* **`int8w` is not free for a single model** (≤3 91.32 → 91.24, below bar;
  dvorak −0.78) — only for the averaged pair, correcting a natural
  generalization of PHASE_K §4.6.
* **PHASE_K §8.5 qualified, not retracted:** its working band held 6/6 above
  98 % agreement, but this phase is the first to exercise the *broken* band
  and both marginal cases (95.32 %, 91.30 %) missed part of it. The gate is a
  reliable **ordinal** predictor; the numeric bands describe the extremes.
* **test-2400 remains SEALED** (ledger at 3; nothing in Phase L opened it).

## Superseded Phase-L interim entry (single-seed, kept for the record)

Full record: `ctc/PHASE_L.md`; plan of record `ctc/PIPELINE_V2_PROPOSAL.md`.
Five 188 k arms, all pre-registered before launch. Headlines with footings:

* **Coupled-pair training (E1) works, and the coupling — not batch sharing —
  is what does it.** Two encoders trained on identical batches with a ramped
  mutual per-frame KL land at **98.18–98.34 % per-frame agreement, 4 of 4
  pairs over the gate** (the campaign's historical rate was 3-in-4, by luck).
  The paired `--pair-weight 0` control, *identical in every other respect*,
  finishes at **92.09 %** and clears the gate at only **2 of 47** evaluations;
  its own-best mix collapses to **greedy 29.10** (its members greedy
  72.6/71.8) at t1 87.64, while the coupled mix reads **greedy 72.92 / t1
  88.90**. The proposal's central claim — make the best configuration a
  recipe rather than a lucky draw — is met.
* **Candidate `v2pair-s1234` (int8w + fp16w, 4.39 MB)**: val **88.86 / 92.82 /
  93.59 / 91.56 / 87.46**, dvorak 92.88, dvorak-app 92.59, azerty 84.11,
  qwertz 84.41, german 82.26, spanish 89.76 — **11/11 campaign bars**, and
  10/11 against the `mix2-i8f16` card (azerty −0.82) at 60 KB *less* size.
  The ≤3 stratum reads **91.56 (+0.29 over the campaign bar, +0.26 over the
  card)** — the largest ≤3 margin the campaign has recorded.
* **Two single models clear ALL ELEVEN campaign bars** (`L1 member A`,
  `L2 member B`) — no single model had ever done this (`sw2345` held 10/11;
  `slw2` held ≤3 at the cost of four). **Footing: one seed.** The s4321/s7777
  members read 6–8/11, so seed luck is not excluded; **not promoted**.
* **The primary bar is NOT met.** It required the gated pair to beat every
  `mix2-i8f16` number on ≥ 2 of 3 seeds; the three-seed stage ran the E2
  recipe and returned 10/11, 5/11, 5/11. Every coupled pair cleared 10 or 11
  *campaign* bars (11, 11, 10, 10), which matches Phase K's recipe-level
  claim while now being reliable by construction.
* **Targeted English synthesis (E2) misses its pre-registered gate by 0.01**
  (t5 −0.16 against a 0.15 limit) and is **not promoted**. Disclosed with it:
  its measured effect is a val wash (t1 −0.05) plus a euro-layout gain
  (azerty +0.86, qwertz +0.93, spanish +0.57) that the val-only gate did not
  scope. Generator, pools and endpoint gates are committed
  (`english_synth.py`).
* **A qualification of PHASE_K §8.5, not a retraction.** Phase L is the first
  time the *broken* half of that pre-registration's bands has been exercised
  (K's gate passed, so it was counterfactual). At 91.30 % agreement the
  greedy prediction holds exactly (29.10 ≤ 30) but t1 lands at 87.64, not
  ≤ 87.5; and a *marginal* 95.32 % pass missed both working-band thresholds
  (88.09 / 53.12). The gate is a reliable **ordinal** predictor; the numeric
  bands describe the extremes, not the 91–96 % middle.
* **test-2400 remains SEALED** (ledger at 3 entries; nothing in Phase L
  opened it).

# Phase K (2026-08-12): a two-model configuration takes all 11 en bars (single-configuration footing); the ≤3 training lever works; the seed-mean stone still stands

Full record: `ctc/PHASE_K.md`. Headlines, with footings stated:

* **`mix2-i8f16` — `sw2345_s1234` (int8w) + `resbn192i_s1234` (fp16w),
  per-frame probability averaging of the two emission heads before the one
  beam — clears ALL ELEVEN en bars**: val 88.68 / 92.61 / 93.46 / **91.30** /
  87.32 (≤3 **beaten +0.03** — the first time anything beat that stratum) and
  every layout bar by +0.31…+3.33. **4.45 MB, 1.79 ms encoder.** Footing:
  a deterministic single configuration against seed-mean bars; the mix
  *recipe* does NOT clear bars (the s4321 pair fails — pair compatibility is
  per-frame alignment agreement, measurable label-free at ≥95 %, a gate
  derived post-hoc); disclosures in `PHASE_K.md` §8.2.
* **Seed-ensemble averaging (same recipe, 3 seeds) is REFUTED** in both
  averaging modes for both families — seeds do not share a CTC alignment.
* **`--short-loss-weight 2.0`** (≤3-weighted CTC loss) breaks the ≤3 stone
  for a single model: seed-mean **91.39 (+0.12), clearing on EVERY seed**
  (91.32–91.47) — the campaign's first every-seed ≤3 clear. The bill is the
  designed trade: t1 −0.03, t3 −0.01, 4+ −0.13, spanish −0.66 seed-mean →
  **7/11 overall**; `sw2345` (10/11, ≤3 −0.07) remains the single-model
  finalist, with `slw2` the mirror-image counter-finalist. The s1234
  all-five-val sweep did not survive the seeds (single-seed floor).
* **T′ = 64 contract-v2 retrain**: the Phase-I transfer promise reproduces
  (all six layout bars, german 82.40 — campaign best) but 4+ **flips sign**
  (−0.39) and val t3/≤3 bars miss; ~2.1× decode cost measured. Documented,
  not promoted (`artifacts/phaseK_t64_golden_contractv2.json`, frames=64).
* **Self-mined discriminative rescorer** (21.8 KB second ONNX, top-k rerank):
  sign-consistent +0.08 t1 / +0.02 t5 / +0.11 4+ seed-mean; **NOT a ≤3
  lever** (sign-inconsistent); the incumbent, offered its own symmetric
  rescorer, gains the same (+0.26 t1 s1234) — a field-shifting lever, not a
  ranking-shifting one. Flat when stacked on the ensemble.
* **Seed-mean footing: the ≤3 stone STANDS** (best seed-mean 91.24, −0.03).
  **Cyrillic: untouched** (bar ≈77.4 full / 77.92 confirm-half at λ=2.0).
  **test-2400 remains SEALED** (ledger at 3 entries; any unsealing is the
  orchestrator's decision).

# Phase J (2026-08-11): the convergence campaign closes 10 of 11 bars on the seed-mean footing (5 of 11 every-seed) — `sw2345`, and the two stones that stand

**The campaign's terminal condition was NOT met.** Phase J was run under the
directive "high-confidence SOTA for what existing usable datasets and research
admit" — a ≤5 MB model beating the incumbents on **every** spread, layout and
language — with a **pre-registered rule that test-2400 is unsealed if and only
if all bars fall**. The finalist beats **10 of the 11 en bars **on the seed-mean footing** (the campaign's stated bar footing; on the stricter every-seed reading it is **5 of 11** — only t1, t5, 4+, dvorak-app and german clear on all three seeds)**; the `≤3` (words
of ≤3 letters) val stratum misses by **0.07 pt**, and the **Cyrillic bar is not
beaten**. The rule therefore did not fire: **test-2400 was NOT read, nothing in
Phase J is test-validated, and `resbn80g` retains the test-validated tier.**
Full record: `ctc/PHASE_J.md`.

**`sw2345`** (arm `phaseJ-sw2345`) = the `resbn192i` recipe — `resbn:192:1,2,4,8`,
embed_hid 96, T3 + 3×HWS, 188 k steps, batch 256, lr 3e-3, wd 0.01, warmup 1 k,
coupled affine sampler, layout-alt **p 0.65**, no KD, 5 k-row beam-t1 checkpoint
selection, **E1** decode preset — plus the two new FUTO data pools built this
phase (`PHASE_J.md` §3.1): **`tier_sw234`** (101,842 rows from swipe-2/3/4) and
**`tier_sw5q`** (24,707 rows, swipe-5 **qwerty-en only**). **1,512,802 params,
1,285,381 train rows.** Session-disjointness against the swipe-1 train corpus
and zero holdout-trace overlap were verified on the complete pools, not
inherited from the scout.

## The 3-seed val table — four bars fall, `≤3` does not

Seeds 1234 / 4321 / 7777, full val-9918, E1 / AOSP trie, exported ONNX. Bars are
`resbn192i`'s Phase I-A seed-means.

| metric | s1234 | s4321 | s7777 | **seed-mean** | bar | Δ |
|---|---|---|---|---|---|---|
| t1 | 88.51 | 88.57 | 88.46 | **88.51** | 88.30 | **+0.21** |
| t3 | 92.59 | 92.72 | 92.70 | **92.67** | 92.60 | **+0.07** |
| t5 | 93.35 | 93.48 | 93.28 | **93.37** | 93.26 | **+0.11** |
| ≤3 (n=3,389) | 90.91 | 91.24 | 91.44 | **91.20** | 91.27 | **−0.07 — MISS** |
| 4+ (n=6,529) | 87.26 | 87.18 | 86.90 | **87.11** | 86.77 | **+0.34** |

**The `≤3` miss is recorded as a miss, not rounded away.** Every lever measured
against that stratum failed: layout-alt dose (worse), CR-CTC (worse),
FUTO-parity augmentations (worse), the checkpoint soup (sign-inconsistent across
seeds, mean −0.10), and a stratum-aware `minmargin` decode sweep over the E1
region, which moved it **+0.03** where roughly +0.33 was needed
(`PHASE_J.md` §6.7, §6.4.1, §6.6.2, §6.8b). The sweep is the diagnostic result:
gamma and beta re-rank candidates by length and cannot conjure a short candidate
the beam never generated, so `PHASE_J.md` §9 reads the shortfall as a
**candidate-generation** problem and leaves three untried directions on the
register — T′ = 64 emission resolution (contract-breaking, an app decision), a
length-conditioned beam, and a ≤3-specific training signal.

## Alt-layout — all six bars fall

Same three seeds, az26 in-dict protocol, E1, seed-means:

| corpus | bar | `sw2345` 3-seed | Δ |
|---|---|---|---|
| dvorak (held out of training) | 89.13 | **89.87** | **+0.74** |
| dvorak, app-98k trie | 88.20 | **88.98** | **+0.78** |
| azerty | 83.60 | **83.81** | **+0.21** |
| qwertz | 82.50 | **83.01** | **+0.51** |
| german | 79.64 | **80.64** | **+1.00** |
| spanish | 88.28 | **88.45** | **+0.17** |

Two further real-layout corpora were built and evaluated this phase and have
**no incumbent** — their zero-shot floors were established here
(`PHASE_J.md` §3.3, 91.08 / 90.19 on `resbn192i` s1234), so they are
informational and are **not** part of the 11-bar tally: `sw2345` scores
**clearflow 91.06** and **kasroz 92.07**. Both corpora are small and
single-cohort (±0.7–1.1 pt binomial SE).

**Tally: 10 of the 11 en bars on the seed-mean footing — 4 of 5 val, 6 of 6
alt-layout, `≤3` −0.07.** Every-seed, the reading is **5 of 11**: only t1, t5,
4+, dvorak-app and german clear the bar on all three seeds. Both footings are
reported because the campaign's bars are seed-means but Phase I-A preferred
every-seed, and the two disagree sharply here. The
Cyrillic axis is counted separately and it also stands (below).

## Cyrillic — the bar is not beaten, and the published ru number was under-tuned

The Cyrillic bar is `phaseIB-ru-synth`'s **in-dict t1 76.21** (app-ru 50 k trie,
E1, real Yandex val rows; **EVAL-ONLY — no Yandex training rows anywhere**, per
`YANDEX_LICENSE_RESEARCH.md`). **It stands.** Two routes were tried and both
closed (`PHASE_J.md` §6.5, §6.8):

* **more capacity on synthetic ru made it worse** — ch 192 / 188 k scores
  73.53 in-dict t1 against the ch 80 / 94 k bar-holder's 76.21, while its
  *greedy* number improves by 3.1 pt: overfitting to the synthetic generator,
  confirmed on `last.pt` (73.30), so not a checkpoint-selection artefact;
* a **joint en+ru single model** (one 65-wide head, 1 M synthetic ru rows on
  ru_jcuken) reaches ru t1 76.56 — **+0.35, inside one binomial SE at
  n = 8,471, and behind the bar on t3/t5** — while costing **−0.42 en val t1**
  against a stated tolerance of 0.3. **Not adopted.** A running 2,000-row figure
  of 77.40 was briefly carried for this model and is **wrong**; the completed
  9,416-row decode is 76.56.

**Correction, model-independent and worth shipping.** Every ru number ever
published in this campaign — **the 76.21 bar included** — was decoded at **E1's
λ = 1.1**, while the app ru lexicon stores `freq = 255 − rank`, the compressed
CKDT scale that wants a larger λ. A symmetric λ sweep over both ru models (tuned
on val rows 0:4708, confirmed on the untouched 4708:9416, `PHASE_J.md` §6.9)
puts the optimum at **λ = 2.0**, worth about **+1.2 to the synth-only model on
both halves**:

| λ | `phaseIB-ru-synth` tune / confirm | joint en+ru tune / confirm |
|---|---|---|
| 1.1 (as published) | 75.73 / 76.70 | 76.77 / 76.34 |
| **2.0** | **76.91 / 77.92** | **77.83 / 78.23** |
| 3.0 | 75.82 / — | 76.39 / — |

**The honest shippable Cyrillic number is therefore ≈ 77.4 in-dict t1, not
76.21** — a full point of free accuracy for the app's Cyrillic path. It does
**not** change the verdict: the lever lifts the challenger equally, so the bar
rises with it and the Cyrillic axis is still **NOT beaten**.

## Size and latency

Measured on the `PHASE_F.md` §0 idle-box protocol (ORT CPU, 3 rounds):

| artifact | params | bytes | mean / p90 | ≤5 MB? |
|---|---|---|---|---|
| `sw2345_s1234.onnx` fp32 | 1,512,802 | 6,068,519 | 0.816 / 0.830 ms | no (6.07 MB) |
| **`sw2345_s1234_fp16w.onnx`** ← ship bytes | 1,512,802 | **3,052,318 (2.91 MiB)** | 0.842 / 0.859 ms | **yes** |
| `resbn192i` fp32 (incumbent, re-measured here) | 1,512,802 | 6,068,519 | 0.819 / 0.833 ms | no |

The finalist is architecturally identical to the incumbent — the entire Phase-J
gain is training data, so it costs nothing at inference. **fp16w is free on
accuracy, measured on this model rather than inherited:** val-9918 decodes
88.51/92.58/93.35/90.91/87.26 through the fp16w graph against fp32's
88.51/92.59/93.35/90.91/87.26. Two asterisks, both disclosed in `PHASE_J.md`
§10: fp16w is **3 % slower** here (not "identical" as Phase I reported for
`resbn192i`), and its weight-rounding residue is 2.30e-02 sliced with argmax
100/100 — real in the emissions, invisible after the beam.

sha256: fp32 `96dd27ece698fa981530639700e66e0689acd2d3f024ad214e8a79b3fa083a30`,
fp16w `2e820c121fc69ae95a9b2e22444fe14c47f5c5253df4696a0d0a432e364fc7b8`.

## Also on record — the levers that failed, and one correction to Phase I

* **CR-CTC is dropped.** Its large transfer gain at ch 80 (+3.13 dvorak) does
  **not** survive capacity: at ch 192 it is dvorak −1.63, and on the ch 256
  high-dose bundle it destroys the euro advantage that was the bundle's whole
  purpose (−1.62 / −0.93 / −1.32 / −2.16). The §5.1c "strongest transfer lever
  measured" reading is **retracted** (`PHASE_J.md` §6.4.1).
* **The checkpoint soup does not generalise.** +0.50 selection t1 / +0.38 `≤3`
  on the `ch256-280k` arm; on the finalist's recipe the paired seeds give `≤3`
  +0.14 and −0.33 — sign-inconsistent, so **not promotable** under the
  campaign's own rule (`PHASE_J.md` §6.6.2).
* **Rejected:** the FUTO-parity augmentation bundle (every val metric down,
  greedy −5.4), the HWS Y-frame ×7/6 train-side correction (rejected, and the
  arm cannot answer its own question — the val HWS half keeps the uncorrected
  frame), the 280 k schedule extension at ch 256 (a tie), real-alt-layout
  training rows (val-neutral, and it costs the campaign its only two never-seen
  real-layout eval corpora), and the blank-penalty decode axis (zero is a sharp
  optimum).
* **E1 transferred unchanged for a fifth model family.** A symmetric
  stratum-aware sweep over the E1 region landed both the finalist and the
  incumbent back on their own E1 numbers to within ±0.07 on every metric
  (`PHASE_J.md` §6.8b) — the strongest evidence yet that E1 is a property of the
  emission/trie pair, not of an individual model.
* **Export-parity correction to `PHASE_I.md` §7.3.** Its "the residue grows with
  width" finding was an artefact of a white-noise export probe. On real val
  traces at real layout centers the residue is essentially width-flat
  (0.8–1.6e-4) and argmax parity was and is 100/100, so **no accuracy number
  anywhere in the campaign moves** (`PHASE_J.md` §5.2).

Evidence tier: **val + alt-layout corpora only. test-2400 was NOT read** — the
pre-registered unsealing requires all bars, `≤3` and Cyrillic did not fall, and
so the seal was not spent. **No pre-registration was filed and no
`test2400_seal.json` unsealing entry was appended**, because the gate's
precondition never came true; the split has still been read exactly three times.
`resbn80g` keeps the test-validated tier; `sw2345` may not be quoted as
test-validated, and no equal-footing claim against FUTO is made for it.

# Phase I-A (2026-08-10): capacity under the accuracy-first mandate — `resbn192i`

**The latency constraint is retired** (user directive: the 2× target was vs
the ~178 ms transformer; sub-10 ms is imperceptible, so capacity is bounded by
**size ≤5 MB**, not speed). Phase I-A ran the capacity ladder UP with the
Phase-H layout augmentation and found the governing law: **capacity converts
to accuracy, but the augmentation dose must scale with it** — at p 0.5 the
held-out dvorak transfer breaks at ch 192 (85.43 vs ch 80's 88.85); raising
the dose to **p 0.65 fixes it and costs nothing** (beats the p 0.5 twin on
all eleven measured columns). Full record: `ctc/PHASE_I.md`.

**`resbn192i`** = `resbn:192:1,2,4,8`, layout-alt **p 0.65**, otherwise the
Phase-G/H recipe. 3 seeds, exported ONNX, E1 / AOSP:

| footing | seed-mean | vs `resbn80h` | vs bars |
|---|---|---|---|
| full val-9918 | **88.30 / 92.60 / 93.26 / 91.27 / 86.77** | +0.61/+0.38/+0.26/+0.48/+0.69 | all five, **every seed**; worst-seed t5 margin **+0.34** |
| dvorak (held out) / app-98k | **89.13** / 88.20 | −0.9 / −1.3 | geo anchor 76.8: **+12.3 / +11.4** |
| azerty / qwertz / german / spanish | 83.60 / 82.50 / 79.64 / 88.28 | −0.2…−1.9 | all beat geo by +6.3…+14.4 |

**Ship bytes: `artifacts/resbn192i_s1234_fp16w.onnx` — 3,052,318 B, 0.831 ms
idle (identical to fp32 on val, transfer, latency; argmax 100/100).** App
preset for it: `0.975 / 3.0 / 0.35 / 0.25 / 0.9882` (holdout-confirmed;
full-val app-trie 89.23 / 93.54 / 94.30 / 92.53 / 87.52). Benchmark preset
stays E1 (fourth family in a row).

Also on record: `resbn256i` (ch 256, p 0.5) — the QWERTY frontier at val
seed-mean **88.65 / 92.61 / 93.32 / 91.26 / 87.29** but transfer-volatile at
its unscaled dose (dvorak seed-mean 86.92) and 10.7 MB fp32 / 5.36 MB fp16w /
2.74 MB int8-trunk (int8 measured **free** at this width); size levers
fp16w (free at every width) and weight-only int8 in `quantize_onnx.py`;
T′ = 64 probe (+0.33 4+, +2.5–2.8 transfer, 2× beam cost — **contract-
breaking, an app decision**); multi-layout checkpoint selection in
`train.py` (small positive, opt-in). Export parity residue grows with width
(disclosed per artifact; argmax 100/100 everywhere) — **withdrawn by Phase J
§5.2**: that was a property of the retired white-noise probe, and on real
traces the residue is width-flat.

Evidence tier: **val + alt-layout corpora only. test-2400 was NOT read** —
`resbn80g` keeps the test-validated tier; `resbn192i` is the registered
nominee for a user-approved final unsealing. **Phase J supersedes this as the
val + alt-layout frontier** (`sw2345`, above) and holds `resbn192i`'s seed-means
as its bars; the unsealing was **not** executed, because Phase J's `≤3` and
Cyrillic bars did not fall.

# Phase H (2026-08-09): layout-resampling augmentation — the dvorak gap closed, `resbn80h`

**The one decisive cross-layout loss is gone.** `ALT_LAYOUT_EVAL.md` measured
dvorak t1 63.04 (ch128) / 67.28 (resbn80) against the geometric engine's 76.8
and traced the cause to untrained **key re-arrangement**. Phase H built the
augmentation the recipe had named and skipped: per sample, with p 0.5, the
cached QWERTY path is warped onto an alternative geometry (residual
re-anchoring on the word's ideal polyline — `ctc/layout_aug.py`) — synthetic
random letter arrangements (2/3) + real azerty/qwertz/german/spanish (1/3),
**dvorak held out of training as a true transfer probe**. Recipe otherwise
identical to `resbn80g` (188 k, coupled sampler, no KD). Full record:
`ctc/PHASE_H.md`.

`resbn80h`, 3 seeds (1234/4321/7777), E1 preset, exported ONNX:

| footing | seed-mean | anchor / bar | Δ |
|---|---|---|---|
| **dvorak** real corpus, in-dict, AOSP trie | **90.01 / 96.38 / 97.46** | geo engine 76.8/79.9/80.4 | **+13.2 t1** |
| **dvorak**, app 98k trie (the anchor's own footing) | **89.51 / 94.90 / 96.73** | 76.8/79.9/80.4 | **+12.7 t1** |
| azerty / qwertz / german / spanish t1 | **84.27 / 84.36 / 81.13 / 88.43** | geo 76.9 / 76.2 / 71.1 / 73.9 | +7.4 / +8.2 / +10.0 / +14.5 |
| en_qwerty full val-9918, AOSP, E1 | **87.69 / 92.22 / 93.00 / 90.79 / 86.08** | val bars 85.52/91.54/92.80/89.29/83.57 | all five clear, **every seed**; vs `resbn80g` seed-mean −0.03/−0.03/+0.03/+0.01/−0.06 |

**The CTC model now beats the geometric engine on all six layouts measured**,
at the en_qwerty-fitted preset (floors, per the tuning asymmetry), with the
en_qwerty val indistinguishable from `resbn80g` and latency identical by
construction (same 231-node graph; idle bench 0.216 vs 0.212 ms). Dvorak
greedy (no lexicon) went 11.6 → 42.5. The `ALT_LAYOUT_EVAL.md` §8
mean-key-displacement routing gate is **obsolete on these weights**: route CTC
everywhere a layout provides a-z key centers (non-Latin scripts remain
untested; `ß` remains untypeable).

Artifacts (opset 17, fp32, parity 100/100, 0.215 ms class, 279,346 params):

| file | arm | sha256 |
|---|---|---|
| `resbn80h_s1234.onnx` | `phaseH-p50` | `3e215438f3c8fae1f249b91be3986bc30c027920f158371acaea0d159dbeff00` |
| `resbn80h_s4321.onnx` | `phaseH-p50-s4321` | `b3f30bcd33cd1137300b039ae166ccd9bdd7ea9117502c35f9d0d80d9a277331` |
| `resbn80h_s7777.onnx` | `phaseH-p50-s7777` | `1a1edac6f10f0fd88b427ce41b4808e46bef1e4209b4611dc7c9e81b5e5e94dd` |

Evidence tier: **val + alt-layout real corpora. test-2400 was NOT read** —
`resbn80h` is not test-validated, and `resbn80g` keeps the test-validated
ship-candidate tier. Promoting `resbn80h` (the strictly better cross-layout
model at equal en_qwerty val) needs an owner-authorized fourth unsealing, and
its app-trie preset (`0.9/4.0/…` was fitted for `resbn80g`) needs a re-check
first.

# Phase G (2026-08-09): the upgraded 0.215 ms candidate — `resbn80g` — test-validated, third unsealing

**`resbn80g` supersedes `fast_resbn80` as the speed-class ship candidate.**
Same graph, same 279,346 params, same 0.215 ms latency class — retrained with
the Phase-G recipe: 188 k steps, the fixed (coupled) affine sampler, and **no
distillation** (the first-ever KD ablation measured KD at −0.5 t1; the ensemble
teacher was −0.45 worse still). Full record: `ctc/PHASE_G.md`.

Evidence tier: **test-validated on both footings** — the third unsealing of
test-2400 was pre-authorized by the user's directive of 2026-08-09 ("retrain
and reexport and re-run tests on new onnx (resbn80)"), gated on the val bars,
pre-registered in `PHASE_G.md` §7 (committed `46aecb1` before the decode), and
executed exactly as registered (6 decodes, one per config × seed; ledger
`test2400_seal.json["test-2400"]["unsealings"][2]`).

| footing | seed-mean (s1234/s4321/s7777) | bar | Δ | worst-seed status |
|---|---|---|---|---|
| val, AOSP, E1 | **87.72 / 92.25 / 92.97 / 90.78 / 86.14** | 85.52/91.54/92.80/89.29/83.57 | +2.20/+0.71/+0.17/+1.49/+2.57 | all five clear, every seed |
| test, AOSP, E1 (config A) | **87.68 / 92.18 / 92.82 / 90.80 / 86.08** | published 84.83/91.04/92.08/89.57/82.40 | **+2.85/+1.14/+0.74/+1.23/+3.68** | all five clear, every seed |
| test, app trie, **app preset** (config B) | **88.14 / 93.22 / 93.90 / 91.86 / 86.23** | trie-matched 84.92/91.54/92.96/89.57/82.52 | **+3.22/+1.68/+0.94/+2.29/+3.71** | all five clear, every seed; worst-seed t5 margin **+0.75** vs the incumbent's +0.08 |

Against the incumbent `fast_resbn80` at the same test footing (config A):
**+0.39 t1 / +0.29 t3 / +0.00 t5 / −0.37 ≤3 / +0.78 4+.** Against the
**equal-footing** bar (both engines val-tuned, `FAIR_REMATCH.md`:
87.12/92.29/92.96/89.94/85.68): **+0.56 / −0.11 / −0.14 / +0.86 / +0.40 — 3 of
5, McNemar unresolved on every seed (+17 p 0.17, +23 p 0.052, +0 p 1.00).**
`resbn80g` is *level* with FUTO's val-tuned engine where `fast_resbn80` was
behind it (three losses become two −0.1 ties and a win) — but **no
equal-footing superiority claim is made or permitted** for it. ~~ch 192 remains
the only configuration with a (qualified) equal-footing win.~~ **⚠ SUPERSEDED
2026-08-14:** the Phase-M distilled single model
`phaseM_kd_fresh_w1_s1234_fp16w` now holds the same qualified win, all five on
every seed with McNemar resolved **3 of 3** (ch 192 resolved 2 of 3), at
2.91 MB — see "The fourth unsealing" at the top of this file.

**The preset changes for the app.** The per-model sweep (`PHASE_G.md` §6)
confirms E1 on the AOSP/benchmark footing and finds the app-trie optimum at

```kotlin
// shipping preset for the app lexicon (en_enhanced.json), resbn80g
CtcScoringParams(gamma = 0.9, lambda = 4.0, beta = 0.25, alpha = 0.0,
                 gammaPrune = 0.25, betaPrune = 0.9882)
```

worth +1.39 t1 / +0.32 t5 on val over E1 on the shipping footing, confirmed on
the untouched holdout half and then on test (config B above). The golden
fixture was regenerated from `resbn80g_s1234` **at this preset** —
`artifacts/ctc_model_golden.json`, sha256
`ce3b5456ad13543ac09ac8c2610374bd8847b15f740f9004a98efea59d74f134` — and must
ship with it (fixture and preset move together).

Phase-G artifacts (same contract, opset 17, fp32, zero normalization nodes,
sliced-view parity 100/100 at export):

| file | arm | sha256 |
|---|---|---|
| `resbn80g_s1234.onnx` ← **ship** | `phaseG-C80-188k-nokd` | `330cadfbaa7334eaeaeab93762084181b70710fe9d59cbd69600a6de468fe1a0` |
| `resbn80g_s4321.onnx` | `phaseG-C80-188k-nokd-s4321` | `c9379c60a23bec4ca300512d2930b7a724aad91b761597972446a6577f5d5bab` |
| `resbn80g_s7777.onnx` | `phaseG-C80-188k-nokd-s7777` | `3e303d46abaff4bfe31779de35fb9fc81e63f1ae8fd5ab554a9db205f167191a` |

Every Campaign-2 caveat travels unchanged (contributor contamination, the
preset asymmetry on published-bar comparisons, per-source 14-pt spread —
config A 94.80/80.36, config B 94.55/81.54). test-2400 has now been read
**three** times; it is a worn split and a fourth read needs a better reason
than any of the first three had.

**Addendum — the Phase-G latency frontier (val-only below 0.213 ms).**
`resbn72g` (`phaseG-F72-188k-nokd*`, 229,642 params, **0.184 ms**, 944,487
bytes) — the same upgraded recipe at ch 72 — clears **all five val bars on the
seed mean and on every individual seed** (87.62 / 92.22 / 93.02 / 90.48 /
86.14; worst-seed t5 margin **+0.18** against the Phase-F `resbn72`'s +0.01)
and **exceeds the old 0.215 ms `fast_resbn80`'s seed-mean on all five
metrics while being 14 % faster**. It is **val-only** — the third unsealing is
spent and it may not be decoded on test. The ch 64 probe (0.161 ms) stays 4/5
(t5 92.70): the no-KD gain does not transfer to ch 64, KD's harm is
capacity-dependent, and Phase F's "≤0.15 ms is not reachable with the bar
intact" verdict is unchanged under the upgraded recipe. Full tables:
`PHASE_G.md` §8.

| file | arm | ms | sha256 |
|---|---|---|---|
| `resbn72g_s1234.onnx` — fastest all-bars, val-only | `phaseG-F72-188k-nokd` | 0.184 | `30b5f3de7831d8137d2e0a9403f3d93ec5b22524db0fba1d76729ab9b09d8043` |
| `resbn72g_s4321.onnx` | `phaseG-F72-188k-nokd-s4321` | 0.184 | `b5ad0911db7ee47c0c6da7c668c62a69eb76b30ab3477053029f9a54c473b987` |
| `resbn72g_s7777.onnx` | `phaseG-F72-188k-nokd-s7777` | 0.184 | `b232a158c620b70e59a2f6d30746f9305231d33af7f0196d48f88879dc1248a2` |

(Idle re-measurement of the Phase-G graphs: `resbn80g` 0.213 ms, `resbn72g`
0.184 ms, `resbn64g` 0.161 ms — the Phase-F class figures within the ±0.002 ms
harness spread. `resbn72g_s4321`'s export parity margin is thinner than the
other five Phase-G exports — occasional random draws reach 2.1e-04 on the
sliced view against the 1e-4 assert, argmax 500/500 unchanged, deterministic
bytes; disclosed in `PHASE_G.md` §8.2.)

The sections below are the Campaign-2 record. Where they name `fast_resbn80`
as the speed candidate or λ = 1.1 as the app preset, **Phase G supersedes
them**; the numbers themselves remain the audited record of that campaign.

---

# Campaign 2 (2026-08-07/08): FUTO ceiling beaten as registered

**Status: the sealed test-2400 decode happened once, was pre-registered, and was
independently audited post-hoc.** Both shipping configurations exceed all five
published FUTO-ceiling numbers — on the seed-mean *and* on every one of six
individual runs.

> **Amended 2026-08-08 — test-2400 has since been read a second time, on the
> user's explicit order, for `fast_resbn80` only.** See "The second unsealing"
> below and `PHASE_F.md` §16. The split has now been read twice; the ledger of
> both reads is in `test2400_seal.json["test-2400"]["unsealings"]`. Nothing else
> was decoded, and no third unsealing is contemplated.

Read in order: `PHASE_A.md` → `PHASE_B.md` → `PHASE_C.md` → `PHASE_D.md` →
`PHASE_E.md`, then `AUDIT_PREDECODE.md` (the adversarial audit that gated the
decode) and `AUDIT_FINAL.md` (the post-decode verification). `DATA_TIERS.md` has
the provenance and contamination audit.

## The claim, verbatim as registered

Registered in `AUDIT_PREDECODE.md` §E **before** the decode, and reproduced from
`AUDIT_FINAL.md` §7:

> **Claim as registered:** on the sealed 2,400-row test split, the Phase-E
> configuration, decoded at the val-tuned E1 preset, is compared against FUTO's
> published encoder+refinement ceiling decoded at FUTO's published preset. A pass
> is not a claim of superiority on equal footing — the presets are not matched,
> and no attempt to re-tune FUTO's preset was possible (its weights are not
> available here).

`AUDIT_FINAL.md` §7 verdict: **"Does the evidence support the claim AS REGISTERED?
— YES."** It also states what may never be written: *that this model beats FUTO's
decoder on equal footing.* See "The asymmetry" below.

## Verified test-2400 results

Every number recomputed by the audit from the per-trace `test2400_e1.jsonl` dumps,
not from log footers. 2,400 rows, strata ≤3 n=815 / 4+ n=1,585 (matching the bar's
own n's). **86 out-of-vocabulary targets are counted as misses**, not excluded.

### ch 192 — `phaseE-FINAL`, 1,525,378 params, 0.877 ms

| seed | t1 | t3 | t5 | ≤3 | 4+ | all five? |
|---|---|---|---|---|---|---|
| 1234 | 88.79 | 92.54 | 93.46 | 91.53 | 87.38 | yes |
| 4321 | 87.88 | 92.71 | 93.50 | 90.92 | 86.31 | yes |
| 7777 | 88.42 | 92.71 | 93.54 | 91.66 | 86.75 | yes |
| **seed-mean** | **88.36** | **92.65** | **93.50** | **91.37** | **86.81** | **yes** |
| seed sd | 0.46 | 0.10 | 0.04 | 0.39 | 0.54 | |
| worst seed | 87.88 | 92.54 | 93.46 | 90.92 | 86.31 | **yes** |
| **the bar** | 84.83 | 91.04 | 92.08 | 89.57 | 82.40 | |
| **Δ** | **+3.53** | **+1.61** | **+1.42** | **+1.80** | **+4.41** | |

### ch 128 — `phaseE-E3b-hws3x`, 689,282 params, 0.455 ms

| seed | t1 | t3 | t5 | ≤3 | 4+ | all five? |
|---|---|---|---|---|---|---|
| 1234 | 88.04 | 92.08 | 92.96 | 91.29 | 86.37 | yes |
| 4321 | 87.83 | 92.46 | 93.12 | 90.55 | 86.44 | yes |
| 7777 | 87.88 | 92.46 | 92.92 | 91.41 | 86.06 | yes |
| **seed-mean** | **87.92** | **92.33** | **93.00** | **91.08** | **86.29** | **yes** |
| seed sd | 0.11 | 0.22 | 0.11 | 0.46 | 0.20 | |
| worst seed | 87.83 | 92.08 | 92.92 | 90.55 | 86.06 | **yes** |
| **Δ** | **+3.09** | **+1.29** | **+0.92** | **+1.51** | **+3.89** | |

### Per-source — the aggregate hides a 14-point internal spread

| config | FUTO half (n=1,217) | HWS half (n=1,183) | spread |
|---|---|---|---|
| ch 192 | t1 **95.32** (t3 99.07, t5 99.48, ≤3 96.93, 4+ 94.27) | t1 **81.21** (86.05, 87.35, 83.48, 80.30) | **14.11 pt** |
| ch 128 | t1 **95.07** (98.88, 99.32, 96.72, 94.00) | t1 **80.56** (85.60, 86.50, 83.09, 79.55) | **14.51 pt** |

The 88.36 headline is the average of a 95.3 and an 81.2. On the How-We-Swipe half
alone the model is 3.6–4.3 pt *below* the aggregate bar.

## The second unsealing — `fast_resbn80` is test-validated (2026-08-08)

### The disclosure, first

**Who ordered it.** The user, who owns this benchmark, explicitly ordered
test-validation of `fast_resbn80`. That is the entire authority for the decode.
`AUDIT_FINAL.md` §7 had declared the seal spent and `PHASE_F.md` §11.1 had said
throughout that no Phase-F artifact may be quoted as test-validated; both
statements are superseded **for this one model only**, by instruction, not by
anything that changed in the evidence.

**Why it is not iterative tuning.** The seal doctrine exists to stop a selection
loop from touching test. This decode is not one: `fast_resbn80`'s architecture,
width, dilations, schedule, distillation teacher/weight/temperature, its three
per-seed checkpoints (selected on beam top-1 over a 5,000-row **val** prefix) and
the E1 preset it decodes at were all fixed on val-9918 and frozen in `PHASE_F.md`
§8/§9 before this task existed; the published artifact sha256 are unchanged. The
first unsealing decoded **ch 128 and ch 192 only** — no `resbn` model had ever
touched test-2400, so this is a *first* decode for this model rather than a
re-decode of a tuned variant. Nothing was being chosen: the preset is frozen, the
seeds are the three already published, Phase F is closed, and the result cannot
feed back into any model or hyper-parameter.

**What was pre-stated, before the numbers were seen.** `PHASE_F.md` §16 was
written and **committed before the decode ran** (commit `50c303a`). It registered:
the claim wording; a hard cap of **2 configurations × 3 seeds = 6 decodes, one
each, no warm-up, no retry on partial output**; the frozen preset, beam width,
top-k, artifacts and metric definitions; the requirement to report all five
numbers for both configurations regardless of outcome; that **a 4-of-5 result is a
failed gate**; and a numeric prediction — config-A seed-mean
87.64/92.35/93.12/90.66/86.09, derived from val plus the val→test shift the first
unsealing showed. **That prediction was wrong in the unfavourable direction on
four of five metrics** (measured −0.35/−0.46/−0.30/+0.51/−0.79 against it), which
is recorded in `PHASE_F.md` §16.5 rather than quietly dropped.

**What it costs.** test-2400 has now been read twice. Every future claim rests on
a more worn split, and that is the price of this table.

### The numbers

`fast_resbn80` — 279,346 params, 0.215 ms, seeds 1234/4321/7777, E1 preset, beam
100, top-k 8, OOV counted as a miss. Two configurations: **A** the AOSP STRIP
146,964-word trie (protocol-identical to the Phase-E decode, comparable to the
published bar) and **B** the shipping lexicon `en_enhanced.json` (98,081 words),
compared against a bar re-measured on that same trie from FUTO's real weights
(`PHASE_F.md` §15.2 — benchmarking only, no training contact).

| config A (AOSP 146,964) | s1234 | s4321 | s7777 | **seed-mean** | sd | worst | bar | **Δ** | z |
|---|---|---|---|---|---|---|---|---|---|
| t1 | 86.75 | 87.42 | 87.71 | **87.29** | 0.40 | 86.75 | 84.83 | **+2.46** | **3.4** |
| t3 | 91.42 | 92.12 | 92.12 | **91.89** | 0.33 | 91.42 | 91.04 | **+0.85** | 1.5 |
| t5 | 92.62 | 92.83 | 93.00 | **92.82** | 0.15 | 92.62 | 92.08 | **+0.74** | 1.3 |
| ≤3 (n=815) | 90.80 | 91.53 | 91.17 | **91.17** | 0.30 | 90.80 | 89.57 | **+1.60** | 1.5 |
| 4+ (n=1,585) | 84.67 | 85.30 | 85.93 | **85.30** | 0.51 | 84.67 | 82.40 | **+2.90** | **3.0** |

| config B (app 98,081) | s1234 | s4321 | s7777 | **seed-mean** | sd | worst | bar | **Δ** | z |
|---|---|---|---|---|---|---|---|---|---|
| t1 | 85.96 | 86.38 | 87.21 | **86.51** | 0.52 | 85.96 | 84.92 | **+1.59** | 2.2 |
| t3 | 91.92 | 92.42 | 92.50 | **92.28** | 0.26 | 91.92 | 91.54 | **+0.74** | 1.3 |
| t5 | 93.04 | 93.33 | 93.38 | **93.25** | 0.15 | 93.04 | 92.96 | **+0.29** | 0.6 |
| ≤3 (n=815) | 90.18 | 90.55 | 91.53 | **90.76** | 0.57 | 90.18 | 89.57 | **+1.19** | 1.1 |
| 4+ (n=1,585) | 83.79 | 84.23 | 84.98 | **84.33** | 0.49 | 83.79 | 82.52 | **+1.81** | 1.9 |

**All five bars clear, on the seed mean and on every individual seed, under both
lexicons.** `fast_resbn80`'s evidence tier is therefore **test-validated**, and it
joins ch 128 and ch 192 as the only configurations that are. Two things to read
carefully: config B's **top-5 worst-seed margin is +0.08 pt — two rows of 2,400**,
the same knife edge Phase F flagged on val; and the statistical resolution is
weaker than the first unsealing's — only t1 and 4+ resolve at z > 2 under config
A, and nothing does under config B.

Per-source seed-mean top-1: config A **94.63** FUTO / **79.74** HWS (spread 14.89),
config B **93.37** / **79.46** (13.91), against ch 128's 95.07 / 80.56. The
14-point internal spread is unchanged, and the app lexicon costs the FUTO half
1.26 pt while leaving the HWS half flat.

Against the ch 128 anchor at the same trie, `fast_resbn80` is **−0.63 t1 / −0.44
t3 / −0.18 t5 / +0.09 ≤3 / −0.99 4+** — the val-measured −0.61 t1 trade transfers
to test essentially unchanged. **`fast_resbn72` (0.186 ms) and every other Phase-F
artifact remain val-only and were not decoded.**

## Statistical resolution — three of five bars are not resolved

Unpaired binomial SE against the published bar treated as a fixed estimate on the
same rows (`AUDIT_FINAL.md` §5; a paired test is impossible — FUTO's per-row output
is unavailable):

| metric | n | ch192 Δ | SE | **z** | ch128 Δ | **z** |
|---|---|---|---|---|---|---|
| t1 | 2,400 | +3.53 | 0.98 | **3.6 — resolved** | +3.09 | **3.1 — resolved** |
| 4+ | 1,585 | +4.41 | 1.28 | **3.4 — resolved** | +3.89 | **3.0 — resolved** |
| t3 | 2,400 | +1.61 | 0.79 | 2.0 | +1.29 | 1.6 |
| t5 | 2,400 | +1.42 | 0.75 | 1.9 | +0.92 | 1.2 |
| ≤3 | 815 | +1.80 | 1.45 | **1.2 — not resolved** | +1.51 | **1.0 — not resolved** |

The correct statement is: **all five point estimates clear, on every seed; two
clear with statistical confidence (t1, 4+); three are positive but within the noise
the row counts admit.** Seed variance is not the limiting factor (sd 0.04–0.54);
row sampling on a 2,400-row split is.

## The asymmetry: the published-preset control

Our decode preset was fitted on val-9918 by a five-parameter grid search; the FUTO
ceiling is quoted at its own published preset. The control, measured on **val**
(ch192, 3-seed mean) at the published `encoderOnly` preset (`AUDIT_FINAL.md` §6.1):

| | t1 | t3 | t5 | ≤3 | 4+ | bars cleared |
|---|---|---|---|---|---|---|
| published preset (matched footing) | 85.78 | 91.66 | 92.67 | 88.10 | 84.58 | **3 of 5** (t5 −0.13, ≤3 −1.19) |
| E1 tuned preset | 88.06 | 92.32 | 93.08 | 90.86 | 86.62 | 5 of 5 |

**The tuning is worth +2.29 pt top-1 on this exact model** — comparable to the
entire test margin on t1 and larger than the margin on t3, t5 and ≤3.

### ⚠ SUPERSEDED 2026-08-08 — the rematch was run, and the asymmetry was material

This section previously said FUTO's headroom was "untested and, with no FUTO weights
on this machine, untestable here", and that no test decode may be spent on a fair
rematch. FUTO's weights were downloaded, hash-verified and re-run
(`FUTO_WEIGHTS_VERIFICATION.md`), and their scoring preset was then swept on val by
the same wide grid that produced E1 (`FAIR_REMATCH.md`). **Our models were not
re-decoded** — the frozen `test2400_e1.jsonl` dumps were re-read — so our seal was
not touched; only FUTO's fixed third-party engine was decoded again.

**The equal-footing question is answered.** Tuning is worth **+1.94 pt t1 to FUTO**
against +2.29 to us — the same order. Against the val-tuned bar (test-2400, STRIP
trie, both engines tuned on the same val rows):

| | t1 | t3 | t5 | ≤3 | 4+ |
|---|---|---|---|---|---|
| FUTO ceiling, val-tuned (the equal-footing bar) | 87.12 | 92.29 | 92.96 | 89.94 | 85.68 |
| ch 192 Δ | **+1.24** | **+0.36** | **+0.54** | **+1.43** | **+1.14** |
| ch 128 Δ | **+0.79** | **+0.04** | **+0.04** | **+1.15** | **+0.61** |
| `fast_resbn80` Δ (config A) | +0.17 | **−0.40** | **−0.14** | +1.23 | **−0.38** |

ch 192 and ch 128 still favour us on all five point estimates, but the margins shrink
by roughly two thirds and ch 128's t3/t5 leads (+0.04 = one trace) are ties.
**`fast_resbn80` fails three of five against the tuned bar** — its five-of-five pass
in "The second unsealing" above holds only against the *published* preset. Exact
paired McNemar on top-1 — now possible, since FUTO's per-row output exists —
**resolves ch 192 on two of three seeds, ch 128 on none, and `fast_resbn80` on none
(one seed net negative).**

The prohibition on writing *that this model beats FUTO's decoder on equal footing*
is lifted **for ch 192, qualified**; it **still stands for ch 128**, whose lead is
not statistically resolvable at n = 2,400, and **`fast_resbn80` must not be
described as beating FUTO at all** — on equal footing it is level on t1 and behind
on t3, t5 and 4+. This bears directly on the shipping choice: the 0.215 ms variant's
accuracy case rests entirely on the untuned comparison. See `FAIR_REMATCH.md` §5 for
the verdict table and §7 for the caveats (chief among them: FUTO's context LM is
still not in the bar, so it remains a floor on FUTO's full stack).

## ⚠ Retraction — the old "+0.21 pt maximum headroom" scoring claim

The previous edition of this file and `README.md` recommended keeping
`CtcScoringParams.encoderOnly` unchanged, on the basis that a two-pass val sweep
plus a full-val headroom grid "bounded the best reachable gain at +0.21 pt top-1".

**That bound is withdrawn.** Every grid behind it spanned γ ∈ [0.30, 0.51],
β ∈ [0.89, 1.08], λ ≤ 0.026 — all centred on the published preset. The optimum for
our emissions is at **γ ≈ 1.05, β ≈ 0.2, λ ≈ 1.1**, outside every grid the campaign
had run. Re-swept wide on the *same* `r2` model the gain is **+4.25 pt top-1 on
untouched val rows** — the bound understated it by ~20×. See `PHASE_E.md` §1.

Arm-vs-arm conclusions in phases A–D are unaffected (all arms were decoded at the
same preset, so the mis-tuning was common-mode). Every **absolute** number in those
phases is understated by 2–5 pt.

## Shipping recommendation

**Ship ch 128** — `artifacts/ch128_s1234.onnx`, 689,282 params, **0.455 ms**
single-thread batch-1 CPU. It clears all five bars on every seed, and ch 192 buys
only +0.19 t1 on val (paired, three seeds) for 1.9× the encoder time and 2.2× the
parameters — while being *behind* on the ≤3 stratum. ch 192
(`artifacts/ch192_s1234.onnx`, 0.877 ms) is the max-accuracy alternative if the
device budget allows.

**Or ship the Phase-F speed variant, with its weaker evidence stated.**
`artifacts/fast_resbn72_s1234.onnx` — 229,642 params, **0.186 ms**, 0.94 MB —
clears all five **val** bars on the seed mean and on every individual seed, at
**2.55× the speed, 33 % of the parameters and 34 % of the bytes** of ch 128, for
−0.61 t1 on the val seed-mean (87.27 vs 87.88, both three seeds). It uses the
`resbn` trunk (dense convolutions with BatchNorms folded into them at export, so
the graph carries no normalization node), 188 k training steps, and is distilled
from our own ch 192 checkpoint. **It has never been decoded on test-2400 and never
may be** — the seal is spent — so unlike ch 128 it carries val evidence only.

**`fast_resbn80` is the better-evidenced of the two speed variants, and as of
2026-08-08 it is test-validated.** `artifacts/fast_resbn80_s1234.onnx` — 279,346
params, **0.215 ms**, 1.1 MB, 2.20× — clears all five bars on **val and on
test-2400**, on every seed, at both the AOSP tuning lexicon and the shipped
`en_enhanced.json` (see "The second unsealing" above). Against `resbn72` its val
t5 seed-mean is statistically the same (92.89 vs 92.87) but its **worst seed
clears the val t5 bar by 0.05 pt against `resbn72`'s 0.01** — and unlike
`resbn72` it now has test evidence. If APK size or latency matters more than the
0.63 t1 gap to ch 128, this is the variant to take.

`PHASE_F.md` has the frontier and the negative results. Everything at or under
0.15 ms misses top-5 — by 0.19 pt at 0.141 ms and 0.13 pt (three seeds) at
0.162 ms — and it stays missed after tripling the training schedule to 280 k steps
(+0.06 t5) or doubling the distillation temperature (−0.20 t5). The constraint is
capacity: t5 crosses the bar at 210–230 k parameters.

**Set `CtcScoringParams` to the E1 preset** — this is a required change, not an
option, for either artifact: at the published preset the same model clears only
3 of 5 bars.

```kotlin
CtcScoringParams(gamma = 1.05, lambda = 1.1, beta = 0.2, alpha = 0.0,
                 gammaPrune = 0.3734, betaPrune = 0.9882)
```

## Artifacts

`artifacts/`, all opset 17, fp32, static shapes `[1,2,64]/[1,64,2]/[1,64]` →
`[1,32,65]/[1,32,64]/[1,32,1]`, zero `Einsum`. Byte-identical to the checkpoints
the audited decode ran on (verified by sha256 against `ckpt/<arm>/`).

| file | arm | params | bytes | sha256 |
|---|---|---|---|---|
| `ch128_s1234.onnx` ← **ship** | `phaseE-E3b-hws3x` | 689,282 | 2,799,865 | `6c1144949e545f626419e1fa7b29e80f9ecf3e303886f30411fc37ae72c45c51` |
| `ch128_s4321.onnx` | `phaseE-E3b-hws3x-s4321` | 689,282 | 2,799,865 | `1eac209332fe6fd52eb7edf2ce52ae77a52552956fdfe7f333d74f2cf46ecce6` |
| `ch128_s7777.onnx` | `phaseE-E3b-hws3x-s7777` | 689,282 | 2,799,865 | `8e910571b748290cb09fdd09e5531cc2aad6d5c09c7fd9d83d57c84ad67dda8b` |
| `ch192_s1234.onnx` | `phaseE-FINAL-s1234` | 1,525,378 | 6,144,249 | `d5b5f10ea16f08743d0742b3c60aa37a469ada11c418a7f459d5ae4cff20c666` |
| `ch192_s4321.onnx` | `phaseE-FINAL-s4321` | 1,525,378 | 6,144,249 | `b020b841abfb011779e2584e418cc651bfcac988a06bfcff2aeea5862bfabab3` |
| `ch192_s7777.onnx` | `phaseE-FINAL-s7777` | 1,525,378 | 6,144,249 | `a182191152ad77b233a73bc79750b0dda51bdbcf7fcb76ddaaad6d17016eee79` |
| `ctc_model_golden.json` | golden fixture, from `ch128_s1234` **at the E1 preset** | — | 140,204 | `a18ea58cd662b0e18b6daadaf417361f93fd0b146ce6478d4d6a62e7e185fa8a` |
| `ctc_swipe_encoder.onnx` | ⚠ **superseded** pre-campaign `r2` | 394,114 | 1,619,140 | `fcf1633167b10f5c28e7c4dc16a9bba178bacc9e2b76efb06d792162dc99d0b7` |

Phase-F additions. Same contract, opset 17, fp32, plus zero normalization nodes
(BatchNorm folded at export). Full table, parity checks and the frontier in
`PHASE_F.md` §6/§8/§9. The `resbn72` rows are **val-validated only** and have
never been decoded on test-2400; the `resbn80` rows were test-validated by the
second unsealing (`PHASE_F.md` §16.5).

| file | arm | params | bytes | ms | all five val bars | test |
|---|---|---|---|---|---|---|
| `fast_resbn72_s1234.onnx` ← Phase-F candidate | `phaseF-N72-188k` | 229,642 | 944,487 | 0.186 | **yes**, every seed | never decoded |
| `fast_resbn72_s4321.onnx` | `phaseF-N72-188k-s4321` | 229,642 | 944,487 | 0.186 | **yes** | never decoded |
| `fast_resbn72_s7777.onnx` | `phaseF-N72-188k-s7777` | 229,642 | 944,487 | 0.186 | **yes** | never decoded |
| `fast_resbn80_s1234.onnx` — wider t5 margin | `phaseF-I-resbn80x4` | 279,346 | 1,142,727 | 0.215 | **yes**, every seed | **all five, both lexicons** |
| `fast_resbn80_s4321.onnx` | `phaseF-FINAL-resbn80x4-s4321` | 279,346 | 1,142,727 | 0.215 | **yes** | **all five** |
| `fast_resbn80_s7777.onnx` | `phaseF-FINAL-resbn80x4-s7777` | 279,346 | 1,142,727 | 0.215 | **yes** | **all five** |
| `fast_resbn64_188k_s1234.onnx` ⚠ frontier evidence | `phaseF-L64-188k` | 185,058 | 766,727 | 0.162 | **no** — t5 92.76 vs 92.80 |
| `fast_resbn56_188k_s1234.onnx` ⚠ frontier evidence | `phaseF-L56-188k` | 145,594 | 609,445 | 0.142 | **no** — t5 92.65 vs 92.80 |

`ctc_model_golden.json` records its own `source_onnx_sha256` and `preset`, and was
regenerated at `1.05,1.1,0.2,0.3734,0.9882` — the fixture must match the preset the
app actually ships, or the parity test asserts against a configuration nothing runs.
For G3 it was regenerated once more (same model, same preset — the 4 beam cases are
byte-identical) to add the 6 `"featurize"`-kind cases `CtcParityTest` requires and a
top-level `layout` block (the exact en_qwerty letters/centers the emissions were
generated against) for the app-side ONNX-backed `CtcEmissionModel` parity test. See
`APP_INTEGRATION_PLAN.md`.
Note `model_cat` decodes to `car`: these are synthetic straight-line paths, and the
fixture is a **parity** artifact (Kotlin must reproduce Python bit-for-bit), not an
accuracy artifact.

## Caveats that travel with every number above

1. **Preset asymmetry** — the largest threat; quantified above at ~2.3 pt.
2. **Contributor contamination.** T3 applies no session or participant exclusion;
   every contributor of every val and test row is in training, and 3× HWS
   oversampling triples the exposure of the more contaminated corpus. **No
   contributor-clean subset of val or test exists for this model.** These are
   benchmark numbers comparable with published FUTO figures — **not a
   generalization claim about an unseen user.**
3. **The dedup defect.** 588 val / 145 test rows sat in `train_t3` with a
   bit-identical input tensor and label, because the dedup keyed on the raw word
   and the label on the a–z-normalized one. **Key fixed** in
   `build_tiers.hash_row` / `prepare_data.trace_hash`; **tiers deliberately not
   rebuilt** (`AUDIT_PREDECODE.md` §E). Measured effect: leaked rows score 4.34 pt
   *below* comparable non-leaked ones, and removing all of them costs < 0.05 pt on
   val / 0.20 pt on test with all five bars still clearing on every seed.
4. **The counter-asymmetry, in FUTO's favour.** 5,273 of the 12,299 unique holdout
   traces (43 %) are bit-exactly in the HF *train* split FUTO trained on; 0 in HF
   dev/test. The app repo's description of the split as FUTO-held-out is incorrect.
5. **Lexicon.** Our runs and the val bar use the *same* 146,964-word STRIP trie, so
   `README.md`'s "our larger lexicon makes these conservative" does **not** apply to
   the val comparison. The test bar was published on the 131,544-word DROP trie and
   re-measured unchanged on the 146,964 one, so the overall test comparison is
   trie-neutral; its **strata were not republished**, so ≤3 and 4+ on test are
   compared across normalizers. **The app will not ship that trie** — it ships the
   bundled 98,140-entry `en_enhanced.json` (98,081 words after a–z stripping),
   whose byte frequencies are floored at 134–255 and whose `log_freq` spread is
   therefore 0.64 against the AOSP trie's 5.40, an 8× collapse of the scale the
   E1 `lambda = 1.1` was fitted on. That was validated end-to-end in `PHASE_F.md`
   §15: the app trie has **fewer** OOV targets (2.52 % of val vs 3.39 %), the bar
   was re-measured on it from FUTO's real weights so the comparison stays
   trie-matched, and **both ship candidates clear all five bars on every seed at
   the unchanged preset**. A λ-only re-sweep is worth +0.6 to +1.1 t1 at λ 2.0–2.5
   and is documented there, but is **not** taken: every number in this file and
   the golden fixture are quoted at λ = 1.1.
6. **Arm selection used full val.** The preset sweep (val `0:4959`) and checkpoint
   selection (5,000-row prefix) respected a holdout, but *which* arms were stacked
   was decided on full val-9918 tables.
7. **Seal hygiene.** One decode per checkpoint, verified bit-for-bit at the
   registered preset on 100/100 sampled rows, with 0/100 matching under any other
   preset. Prior contact: the disclosed pre-campaign `r2` decode and an undisclosed
   120-row smoke decode with a toy 898-word trie. **7 traces are bit-exactly shared
   between val-9918 and test-2400.** During the post-decode hygiene pass, 3 test
   rows were decoded to verify the new `--unseal-test` override branch; no number
   from that run appears anywhere. **Second unsealing, 2026-08-08:** six more
   decodes (`fast_resbn80` × 3 seeds × 2 lexicons) on the user's order, plus three
   re-scores of FUTO's own cached ceiling emissions on test-2400 to obtain a
   trie-matched bar (an external reference, no CleverKeys model involved). Both
   reads, and the prior contact above, are now logged in
   `test2400_seal.json["test-2400"]` under `unsealings` / `prior_contact`;
   `seal.py --emit` preserves that ledger across a fingerprint regeneration.

## Next — app-side (not this repo)

1. **G3 wiring.** Drop `ch128_s1234.onnx` into the `CtcEmissionModel` seam; the I/O
   contract is unchanged from `r2`, so no Kotlin signature moves.
2. **Update `CtcScoringParams`** to `gamma 1.05, lambda 1.1, beta 0.2, alpha 0.0,
   gammaPrune 0.3734, betaPrune 0.9882`. **Required** — the published preset costs
   ~2.3 pt and drops the model to 3 of 5 bars.
3. **Land the golden fixture.** Commit `artifacts/ctc_model_golden.json` as
   `src/test/resources/ctc/ctc_golden.json`. `CtcParityTest` currently fails its own
   file-existence assertion (audit finding #4), so featurizer parity is **untested
   today**; this is what makes it run.
4. **`NOTICE` attribution.** `futo-org/swipe.futo.org` corpus (**MIT**) and
   How-We-Swipe / OSF `sj67f` (**MIT**, © 2021 Leiva/Kim/Cui/Bi/Oulasvirta). No FUTO
   weights or model outputs were used anywhere in training (guide §0), so the FUTO
   Model Weights License is **not** implicated; the decode *algorithms* ported from
   the GPL-3.0 `swipe-library` are already committed on the app side.
5. **Re-measure latency on a phone little core.** 0.455 ms is a desktop x86 core; the
   trie beam over 147 k words, not the encoder, dominates the per-swipe budget.
   On device the beam runs over the **98 k** app trie, which is a third smaller.
6. **O3 is closed: ship `dictionaries/en_enhanced.json`** via
   `CtcLexiconTrie.loadStrippingNonAlphabet`, with the preset unchanged. Validated
   in `PHASE_F.md` §15 on val (both candidates, three seeds each) and in §16.5 on
   test (`fast_resbn80`, three seeds). No new dictionary asset is needed and no
   λ change is required.

---

# Campaign 1 (2026-08-07) — superseded

> ⚠ Kept for provenance. Its ship candidate (`r2`, ch 96) and its
> "keep the published scoring preset" recommendation are both **superseded** by
> Campaign 2 above; its absolute accuracy numbers are quoted at the mis-tuned
> published preset and are understated by 2–5 pt.


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
