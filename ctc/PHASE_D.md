# Phase D — beam-selected checkpoints on the T3 benchmark tier

Phase D changes two things the earlier phases got wrong, and then runs the
architecture question again on a tier ~2.6× the size of anything used so far.

1. **Checkpoint selection is now beam top-1, not greedy.** Phase B measured an arm
   whose greedy accuracy rose 5.16 pt while its beam top-1 fell 1.38 pt, so
   `best val_greedy` was choosing checkpoints by a metric that anti-correlates
   with the metric that ships. Every Phase-D checkpoint is selected on
   `val_beam_t1` over a fixed 2,000-row val prefix.
2. **Decisions need three seeds.** Phase C measured a 1.05 pt clean-t1 swing
   between two runs differing only in seed. Single-seed differences below ~1 pt
   are not interpretable, so the two leading arms are run at seeds
   1234 / 4321 / 7777 and compared pairwise on the same seed set.

**test-2400 was never decoded.** `eval_arms.py` refuses any split whose filename
contains `test`, and `train.py` now refuses it as a selection split too.

---

## 1. Beam-selected checkpointing

At every `--val-every` step `train.py` runs the vendored `futo_viterbi_beam` over
the first 2,000 val rows, at the published `encoderOnly` preset, beam width 100,
`top_k=5`, and selects `best.pt` on the resulting top-1. Greedy accuracy is still
computed and logged at every point, so the greedy-vs-beam relationship stays
visible; it just no longer decides anything.

Mechanically the beam pool is forked **once**, at construction time, before the
model touches the GPU:

* the 146,964-word trie is inherited copy-on-write and is never pickled;
* the sliced `[N,32,27]` emissions live in a shared `RawArray` the parent
  overwrites in place, so no re-fork and no serialization happen per validation;
* no child ever inherits a live CUDA context.

**Correctness.** The validator reproduces the committed r2 numbers on the same
2,000 rows exactly: **81.55 / 89.85 / 91.65**, identical to the digit to the
figures in `README.md` §"Measured result on r2".

**Cost.** ~2 s per validation point at 12 processes on an idle machine (4.0 s at 6
processes with the box under a parallel featurization load). At `--val-every 3000`
over 94,000 steps that is 31 validations, i.e. **~1 minute added to a ~15-minute
run** — under 10 % overhead, and the pause is not overlapped with GPU steps
because it does not need to be.

---

## 2. T3 — the benchmark tier, and the disclosure it requires

### Build

| side | in | dropped | kept |
|---|---|---|---|
| FUTO swipe-1 `train.jsonl` | 939,550 | 4,709 `potentially_invalid_sentence`; 5,273 exact-trace holdout matches | **929,568** |
| How-We-Swipe full release (1,338 users) | 86,323 traces from 1,338 logs | 1,711 `len(word) < 2`; 6,457 exact-trace holdout matches | **78,155** |
| **tier jsonl** | | | **1,007,723** |
| after `prepare_data.py` | 1,007,723 | 43 empty word, 1 CTC-infeasible, 2,343 self-duplicates, **0 cross-split duplicates** | **1,005,336** (`cache/train_t3.npz`, 435 MB) |

Two build facts worth keeping:

* **The HWS parse is bit-exact.** `build_tiers.parse_hws_log` is a replica of
  `neural-swipe-typing/process_swipe_logs.py`; re-parsing the 1,052 logs the
  canonical splits were built from reproduces **all 60,303 unique traces of the
  61,597-row canonical pool, with zero misses**, and the first row compares equal
  dict-for-dict (word and every `{t,x,y}` float). The full release then yields
  **84,612** rows under the same filter (`is_err = 0`, ≥ 3 points,
  `len(word) ≥ 2`) against the canonical pool's 61,597 — the 286 users absent
  from the local log directory are worth +23,015 rows, exactly the figure
  `fetch_hws_full.py --analyze` projected.
* **The 6,457 HWS rows removed by exact dedup are the right ones.** The canonical
  holdout holds 6,159 HWS rows, which collapse to **6,140 unique traces** (11
  duplicate pairs in val, 1 in test). The release contains ~2 % duplicate traces
  overall (61,597 rows → 60,303 unique in the local pool), so 6,140 unique
  holdout traces are expected to appear ≈ 6,140 × 1.05 ≈ 6,450 times in the
  release. Measured: 6,457. The val/test HWS rows are literally drawn from these
  logs, and the dedup found them.

Exact-trace dedup was applied under **both** hash conventions in use in this
pipeline — `scan_futo_sessions`' `(word.lower(), x, y)` and `prepare_data`'s
`(word, x, y, t)` — and their union was taken. The `(word, x, y)` form ignores
timing and is therefore the stricter of the two: it caught every match on both
sides (`futo_leak_xy` 5,273 / `hws_leak_xy` 6,457) and the `(word,x,y,t)` form
added **zero** further rows. `prepare_data.py` then re-applies its own dedup
independently and reports `dropped_cross_split_duplicate = 0`, which is the
cross-check that the tier build missed nothing.

### ⚠ Disclosure — T3 is contributor-dirty by construction

**T3 applies no session or participant exclusion.** Every contributor who
produced a val or test trace also has other traces in T3, on both corpora. This
is deliberate, and it means:

* **T3 cannot support a generalization claim.** Phase A's ladder established that
  a generalization arm must exclude every contributor that touched the holdout on
  both corpora (T2/T2b do; T3 does not). A T3 number is an *upper* bound that
  includes whatever a model can memorise about a specific person's hand geometry
  from their other traces.
* **T3 exists to be comparable with the published FUTO baselines**, which were
  produced by models trained on the literal holdout traces. On that axis T3 is
  strictly the more conservative of the two: it removes the 12,299 holdout traces
  bit-exactly (11,730 of them found and dropped: 5,273 FUTO + 6,457 HWS
  instances), where the baselines kept them.
* **Every T3-vs-FUTO comparison must carry this paragraph.** A T3-vs-FUTO number
  and a T2-vs-T2b number are different objects; only the latter is a statement
  about the model. This is the asymmetry `PHASE_A.md` §5 said had to be disclosed
  at the point of comparison, every time.

Because of this, the arm-vs-arm comparisons in §3 are the load-bearing numbers
(all arms share the same tier and the same contamination, so the contamination
cancels), and the absolute level is not.

---

## 3. Arms

*(filled in below)*
