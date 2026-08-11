# AUDIT_PHASEJ — final adversarial audit of the Phase I/J record and the terminal verdict

**Date:** 2026-08-11 · **Auditor:** independent adversarial session (no shared
state with the campaign orchestrator) · **Scope:** the claimed terminal state
"Phase J closed at 10/11 bars, terminal condition NOT met, test-2400 sealed",
the finalist `sw2345` record, the two stones, the six retractions, and the
documentation set (`PHASE_J.md`, `PHASE_I.md`, `RESULTS.md`,
`MODEL_COMPARISON.md`, `APP_INTEGRATION_PLAN.md`). **test-2400 was not decoded,
loaded, or hashed by this audit**; the app repo was not modified; the only file
written is this one.

**Method.** Every headline number was traced to its committed eval dump or run
log under `/home/will/ctc-train` (`ckpt/*/val9918_e1.log`, `altlayout/*.json`,
`ckpt/*/ru_*lam*.json`, soup and sweep logs), and the finalist's entire
seed-1234 battery was **re-executed from scratch by this audit** through the
committed harness (`eval_beam.py`, `eval_altlayout.py`) against the committed
artifact `artifacts/sw2345_s1234.onnx` — not read off the campaign's dumps.

---

## 1. The finalist 3-seed table — CONFIRMED, with an exact independent replication

* **Independent re-run (this audit, from the committed fp32 artifact,
  sha256 `96dd27…`):** full val-9918 at E1/AOSP reproduced the published
  numbers **digit-for-digit**: t1/t3/t5 = 88.51 / 92.59 / 93.35, ≤3 = 90.91,
  4+ = 87.26, greedy 72.08. The full alt-layout battery (az26/E1: dvorak,
  azerty, qwertz, german, spanish, clearflow, kasroz; plus dvorak vs the app
  98k trie) also reproduced **exactly**: 91.09 / 84.02 / 82.14 / 80.76 /
  88.00 / 91.44 / 91.40 / 89.05. The pipeline is deterministic and the
  committed artifact is byte-identical to the eval'd graph
  (`ckpt/phaseJ-sw2345/ctc_swipe_encoder.onnx` has the same sha256).
* **Per-seed values, all 11 bars × 3 seeds:** every cell of the `PHASE_J.md`
  §8 table matches its underlying dump (`val9918_e1.log` for the three seeds;
  `altlayout/phaseJ-sw2345{,-s4321,-s7777}_{az26_e1,dvorak_en98k}.json`).
  33/33 val cells and 24/24 layout cells checked — zero discrepancies.
* **Seed-mean arithmetic:** recomputed for all 11 bars; every mean and every Δ
  in §8 is correct (≤3 unrounded gap is −0.073 ≈ 2.4 rows; "two rows" is a
  fair statement of it).

## 2. The bars — CONFIRMED, measured on comparable footings

* The five val bars (88.30 / 92.60 / 93.26 / 91.27 / 86.77) recompute exactly
  from the incumbent's three per-seed dumps
  (`ckpt/phaseI-ch192-p65{,-s4321,-s7777}/val9918_e1.log`: 88.32/88.10/88.49
  etc.), matching `PHASE_I.md` §7.2.
* The six layout bars (89.13 / 88.20 / 83.60 / 82.50 / 79.64 / 88.28)
  recompute exactly from the incumbent's alt-layout JSONs (dvorak-app
  unrounded mean 88.197 → 88.20).
* **Footing symmetry verified:** incumbent and finalist alt-layout JSONs carry
  the identical preset (1.05, 1.1, 0.2, 0.3734, 0.9882), identical corpora
  (row counts 2535/2291/1356/2503/1883 and OOV counts equal on both sides),
  identical lexicons, same AOSP trie for val. The Cyrillic bar 76.21 matches
  `ckpt/phaseIB-ru-synth/eval_app_e1.json` (n = 9,416, decoded 8,471, E1,
  app-ru 50k) — the same footing `phaseJ_eval_ru.sh` pins.

## 3. Seal integrity — CONFIRMED

* `test2400_seal.json` ledger holds **exactly 3** unsealing entries
  (2026-08-08 ×2, 2026-08-09), each with authority, preset, trie, and
  publication pointer. No fourth entry.
* **Newest test-decode artifact on disk:** `ckpt/phaseG-*/test2400_g_app.jsonl`
  at 2026-08-09 04:10 — the third unsealing. Nothing test-related is newer;
  no Phase-J directory contains a test dump; `phaseJ_eval.sh` and the round
  scripts reference no test file; no Phase-J log contains an `--unseal-test`
  invocation or a "SEALED split" override message.
* **The two invalid sweep incidents** (the first `minmargin` runs on the wrong
  grid, one per model — `ckpt/phaseJ-sw2345-minmargin.log` and
  `ckpt/phaseI-ch192-p65-minmargin.log`) both operated on
  `sweep_emissions.npz` built from **val-9918** ("rows: 9918" in both logs).
  No test contact.
* Minor observation, pre-existing and benign: the ledger stores 2,399 unique
  fingerprints for n = 2,400 because `seal.py --emit` writes
  `sorted(set(fps))` and the split contains one duplicated (word, x, y) trace
  pair (consistent with the duplicate-trace findings in `AUDIT_FINAL.md`).
  This does not weaken the overlap guard.

## 4. The refutations — CONFIRMED against committed outputs

* **Soup (§6.6.2):** `phaseJ-sw234-s4321-soup.log` → 2 members, 86.42 vs
  86.26 (+0.16); `-s7777-soup.log` → 4 members, 86.00 vs 85.86 (+0.14);
  `sw234-soupval.log` full-val → s4321 soup 88.49/92.75/93.33/**91.32**/87.01,
  s7777 soup 88.26/92.51/93.22/**90.91**/86.89. The ≤3 deltas +0.14 / −0.33
  are exactly as recorded; sign-inconsistent; retraction sound.
* **CR-CTC (§6.4.1):** all six rows of the retraction table (val + 6 layout
  columns for `cr192`, `sw234-cr`, `cr256-p80`, against `sw234`, `sw2345`, and
  the base) match their JSON dumps cell-for-cell. The ch-80 positive
  (+3.13 dvorak) and its reversal at ch 192 (88.97 vs base 90.60) are both in
  the dumps. Retraction sound.
* **Stratum-aware preset sweep (§6.8b):** both E1-region logs verified —
  finalist tuned (1.100/0.90/0.175) → full-val 88.44/92.55/93.27/**90.94**/87.13;
  incumbent tuned (0.975/1.10/0.325) → 88.35/92.57/93.25/**91.18**/86.89.
  Identical 6×6×5 grid, identical tune/holdout split, both models — the
  symmetry claim holds, and ≤3 moved +0.03 as recorded.
* **Blank-penalty axis (§2):** `ckpt/phaseI-ch192-p65/blank_grid{,_fine}.json`
  match the table (0 → 88.75/92.75/93.55 sweep, 88.05/92.45/93.25 holdout;
  ±0.1 inside noise and sign-inconsistent; −1/−2/+1 → 51.4/21.1/45.1).
* **Round-1/2/3 arm tables:** final val blocks of all 13 arm logs
  (`sw234`, `yfix`, `realalt`, `ch256-280k`, `futoaug`, `sw234-p80`, `cr192`,
  `sw234-cr`, `ch256-p65`, `ch256-p80`, `ch192-p80`, `cr80`, `cr256-p80`)
  match §5.1/§6.1/§6.4/§6.7 exactly.

## 5. The ru λ = 2.0 claim — CONFIRMED, symmetric, no training contamination

* All tune/confirm cells of §6.9 match `ckpt/{phaseIB-ru-synth,phaseJ-joint}/
  ru_{tune,confirm}_lam*.json`: synth 75.73/76.70 → **76.91/77.92** at λ 2.0
  (3.0: 75.82; 4.0: 73.88); joint 76.77/76.34 → **77.83/78.23** (3.0: 76.39;
  4.0: 74.50). Both models swept by the same script (`ru_lambda_sweep.sh`) on
  the same row slices (0:4708 tune / 4708:9416 confirm) at the same grid —
  symmetric. Confirm-half t3/t5 at λ 2.0 (joint 88.94/91.49 vs synth
  89.50/92.00) support the "bar rises with it" verdict; +0.92/+0.31 leads are
  magnitude-inconsistent as stated.
* **Yandex is eval-only, verified:** the finalist trained on
  `train_t3 + 2×hws + tier_sw234 + tier_sw5q` (launch log, 1,285,381 rows —
  matches §8); `phaseJ-joint` trained on `cache_ru_synth/train_synth.npz`
  (synthetic transplant, English donor traces + ru wordlist —
  `cyrillic_synth.py`); `phaseJ-ru192` on `--cache cache_ru_synth`. No
  `train_yandex` reference appears in any Phase-J launch log or round script.
  `cache_ru/train_yandex.npz` exists but was consumed only by the disclosed
  Phase-I `phaseIB-ru-real` probe, which is not the bar and not shipped.
* ru negatives verified from dumps: `phaseJ-ru192` best 73.53/86.80/90.17
  (greedy 40.18), `last.pt` 73.30 (greedy 39.94) — the selector refutation
  is real. Joint full decode 76.56/88.16/91.12 (greedy 23.68) — the corrected
  figure, not the retracted 77.40.

## 6. Documentation coherence — CONFIRMED, one citation defect

* `RESULTS.md` top section, `MODEL_COMPARISON.md` §0.1/§1/§2.8, and
  `APP_INTEGRATION_PLAN.md` §7/O9/O10 all carry the same numbers as
  `PHASE_J.md` §8–10 (checked cell-by-cell for the two 3-seed tables, the ru
  λ table, sizes, latencies, sha256s), the same evidence-tier language
  ("val + alt-layout only, NOT test-validated", `resbn80g` keeps the tier),
  and the same two-stone verdict. No document states or implies the terminal
  condition was met; no equal-footing claim against FUTO is made for
  `sw2345` anywhere (`RESULTS.md` says so explicitly; `MODEL_COMPARISON.md`
  §2 marks the row "val + alt-layout only").
* Artifact claims verified directly: file sizes 6,068,519 / 3,052,318 B and
  both sha256s match; the fp16w full-val decode log (`ckpt/fp16w_val.log`)
  shows 88.51/92.58/93.35/90.91/87.26 — the "free on accuracy" table is
  real, and the 3 % latency and 2.30e-02 residue disclosures are present.
* **Defect (correction required):** `MODEL_COMPARISON.md` cites
  `PHASE_J.md` §6.6.1 as the source of the finalist's 3-seed tables (lines
  ~104, ~215, ~323 and the §5 crosswalk). §6.6.1 holds the **superseded
  2-seed** figures; the numbers actually printed are §8's (correct) ones.
  The citations should read §8. Substance unaffected.

## 7. The honest-verdict framing — QUALIFIED (verdict stands; the tally is footing-dependent)

* "Terminal condition NOT met" is **correct and robust**: ≤3 misses on the
  seed-mean (−0.07, and on 2 of 3 individual seeds), and the Cyrillic bar is
  untouched by any Phase-J model on its full decode. Every stricter reading
  makes the verdict *more* not-met. The gate discipline followed: no
  pre-registration was filed, no ledger entry appended, nothing in Phase J is
  quoted as test-validated.
* "10 of 11 bars" is a **seed-mean claim**, correct on the footing the bars
  were defined on (§0: "seed-mean over 3 seeds"). But §0 also says
  "every-seed preferred", and on that preferred reading the tally is
  **5 of 11** (t3, dvorak, azerty, qwertz, spanish each have at least one
  seed under the bar, in addition to ≤3). The per-seed columns are printed in
  §8, so nothing is concealed — but the headline number inherits the weaker
  footing without saying so. Several seed-mean margins (t3 +0.07, t5 +0.11,
  spanish +0.17, azerty +0.21) are inside the noise the campaign itself
  measures (~1 pt single-seed; 1.7–2 pt seed spreads on the layout axes), so
  "10/11" should be read as a point estimate, not a resolved superiority
  claim per bar. Recommended phrasing anywhere the tally is quoted:
  "10/11 on seed-means (5/11 on the stricter every-seed reading)".
* The ≤3 diagnosis ("candidate-generation, not re-ranking") is supported by
  the evidence hierarchy verified in §4 above — five levers measured against
  the stratum, none moved it, and the decode-side sweep (the diagnostic one)
  was symmetric. The three named residual routes (T′ = 64, length-conditioned
  beam, ≤3-weighted training) are correctly registered as untried.
* **The six self-retractions are real and evidenced**, each with its
  refuting measurement committed: (1) "dose axis closed" (§5.1b), (2) CR-CTC
  as a general transfer lever (§6.4.1), (3) the soup as the ≤3 route
  (§6.6.2), (4) the joint-ru "≈77.4" running figure (§6.8, commit
  `cb568c9`), (5) the §6.6 sw234-finalist reading and §6.6.1's 2-seed
  figures (§8, commit `b3ca05a`), (6) `PHASE_I.md` §7.3's width-residue
  claim (§5.2). A seventh correction — the wrong-grid first sweep (§6.8b) —
  is also on record.
* One discrepancy against the *briefed* terminal state (not the record): the
  brief circulated as "missing ≤3 by ~0.2–0.3". That matches the superseded
  2-seed estimate (−0.20); the record's final 3-seed number is **−0.07**.
  The record is right; summaries should quote −0.07.

---

## Corrections required

1. `MODEL_COMPARISON.md`: change the finalist-table citations from
   `PHASE_J.md` §6.6.1 to §8 (three occurrences plus the §5 crosswalk row).
2. Anywhere the 10/11 tally is quoted without the per-seed table (pm.md,
   briefings), attach the footing: seed-mean; every-seed is 5/11.
3. Optional: note in `seal.py`'s docstring that the ledger's 2,399 unique
   fingerprints for n = 2,400 reflect one duplicated trace pair, so the next
   auditor does not re-derive it.

## Final statement

**The record supports the campaign's terminal claims as written.** The
finalist's full battery was independently re-executed by this audit and
reproduced digit-for-digit; every bar, every refutation, and the ru λ result
trace to committed dumps measured on symmetric footings; the seal ledger
stands at exactly three entries with no test contact after the third
unsealing; and the documentation set carries the verdict without
overstatement. The only substantive caveat is footing transparency on the
"10/11" headline (item 7), and the only defect found is a stale section
citation (item 6) — neither changes the verdict: **terminal condition NOT
met, two stones standing, test-2400 sealed.**
