# Phase A — the data-tier ladder

One variable (the training pool), one frozen recipe, one holdout. Every number below is
measured, not projected; where a number is not trustworthy the reason is stated rather
than smoothed over.

**test-2400 was never decoded.** `eval_arms.py` refuses any split whose filename contains
`test`. Every figure here is val-9918.

---

## 1. Recipe (frozen across all five arms)

```
ch 96   embed_hid 96   batch 256   lr 3e-3   wd 0.01   warmup 1000 steps
AdamW + linear warmup -> cosine to 0 over a 47,000-step horizon
fp32, grad-clip 1.0, augmentation = slot permutation + rejection-sampled affine + noise
seed 1234   --total-steps 47000   --val-every 1500   patience 40 evals
```

`--total-steps` is what makes the ladder honest: the tiers differ by 3.5x in size, so an
epoch budget would have handed the big tiers 3.5x the gradient steps and the comparison
would have measured compute, not data. Every arm gets exactly 47,000 optimizer steps and
the same cosine horizon; only the pool underneath changes.

Checkpoint selection is `best val_greedy` on the 1,500-step evaluation grid.

## 2. Arms

| arm | cached rows | composition | contamination control |
|---|---|---|---|
| `phaseA-T0` | 109,600 | 55,438 HWS + 55,438 FUTO | none (the historical control) |
| `phaseA-T1` | 372,726 | HWS half + full curated FUTO | FUTO session exclusion only |
| `phaseA-T1strict` | 319,421 | 888 HWS + 318,566 FUTO | + HWS participant exclusion |
| `phaseA-T2` | 385,021 | FUTO only | session exclusion, hygiene filter only |
| `phaseA-T2b` | 285,929 | FUTO only | session exclusion + full quality cascade |

## 3. What the holdout can and cannot decide

This is the part that changed most during Phase A, and it changes how the table must be
read.

`val_clean[arm]` is the subset of val-9918 whose **contributor** appears nowhere in that
arm's training rows. It is rebuilt in `build_val_clean.py` from each row's own `session`
field — the canonical splits and both tier source pools carry it at 100 % coverage — after
the previous hash-based version was found to be wrong in the permissive direction:

* a tier's jsonl has no `session` field, so contributors were recovered by hashing tier
  rows against the corpus index;
* ~15 % of the FUTO pool was renormalised and no longer hashes to the corpus, so those
  rows resolved to nothing, contributed nothing to the tier's contributor set, and every
  val row from those contributors was scored as **clean while sitting in the training
  pool**.

| arm | val_clean (hash, old) | val_clean (session, correct) | of which FUTO / HWS |
|---|---|---|---|
| T0 | 176 | **164** (1.7 %) | 163 / 1 |
| T1 | 4,238 | **46** (0.5 %) | 43 / 3 |
| T1strict | — | **5,019** (50.6 %) | 43 / 4,976 |
| T2 | 9,213 | **9,300** (93.8 %) | 4,324 / 4,976 |
| T2b | 9,213 | **9,669** (97.5 %) | 4,693 / 4,976 |

Three consequences:

1. **T1 is not contributor-disjoint on the FUTO side.** `build_tiers.py` keeps the 102,826
   rows whose session could not be recovered, and they carry back ~2,000 of the very
   sessions the exclusion dropped: T1's FUTO contributor set is 9,877 of the corpus's
   10,889 sessions. `--strict-session` is what a genuinely disjoint T1 requires; T1strict
   fixes only the HWS side, which is why its clean FUTO count is still 43.
2. **Only T2/T2b support a generalization claim**, and their clean subset is 54 % HWS —
   a corpus they contain none of — so an undivided clean number is largely a cross-corpus
   transfer score. Every clean figure below is therefore also split by source.
3. **T0 and T1 clean subsets (164 and 46 rows) decide nothing.** At n=164 the standard
   error on a ~82 % rate is ±3.0 pt; at n=46 it is ±5.7 pt. They are reported for
   completeness and must not be used to rank arms.

Separately, 219 contributor sessions produced 249 val rows yet never entered
`futo_tainted_sessions.npz` — the holdout traces that would have exposed them are exactly
the ones whose hash no longer matches the corpus. The 27,356 corpus rows those sessions
hold remain inside every tier. The session-field masks above account for this; the tier
*builds* do not.

## 4. Results

<!-- RESULTS_TABLE -->

## 5. Protocol going forward

The rules Phase A settled, so later phases do not relitigate them:

* **Exact-trace dedup is non-negotiable** and is enforced twice — in `build_tiers.py` and
  again in `prepare_data.py`, so a tier physically cannot smuggle a holdout trace even if
  the tier builder missed one.
* **Session/participant exclusion defines a *generalization* arm.** Any arm claiming to
  measure generalization must exclude every contributor that touched the holdout, on
  *both* corpora, and must resolve contributors from the source `session` field rather
  than from a trace hash. An arm that cannot resolve a contributor must count that row as
  present, never as absent.
* **A benchmark arm is a different object from a generalization arm.** A future T3 would
  train the winning recipe *without* session exclusion, to be comparable with published
  FUTO numbers — which were produced by a model trained on the literal test traces. That
  asymmetry has to be disclosed at the point of comparison every time, because a T3-vs-FUTO
  number and a T2-vs-T2b number mean different things and only the latter is a claim about
  the model.
* **Report per-source.** The HWS and FUTO halves sit at a known ~0.064 systematic Y offset,
  and the arms move them in opposite directions. An aggregate val number hides that.

## 6. Reproduction

```bash
# tiers (‑‑suffix keeps a rebuild out of the file a run already trained on)
python build_tiers.py --tiers t1 --suffix _strict
python prepare_data.py --extra-train data/tier_t1_strict.jsonl \
                       --out-name train_t1_strict --jobs 10

# contributor masks (session fields, zero unresolved rows)
python build_val_clean.py --arms T0,T1,T1strict,T2,T2b

# one arm
python train.py --train-npz train_t2.npz --run-name phaseA-T2 \
                --total-steps 47000 --val-every 1500 --seed 1234

# eval (cached emissions; reproduces eval_beam.py exactly)
python export_onnx.py --ckpt ckpt/phaseA-T2/best.pt \
                      --out ckpt/phaseA-T2/ctc_swipe_encoder.onnx
python eval_arms.py --arms phaseA-T0,phaseA-T1,phaseA-T1strict,phaseA-T2,phaseA-T2b
```
