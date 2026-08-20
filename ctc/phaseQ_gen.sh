#!/usr/bin/env bash
# Phase Q — per-script v3 cache generation (SHIPPING TRACK).
#
# The learned generator (ckpt/synthq_gen_ship/gen.pt, MIT-trained: FUTO t3 +
# HWS) sampled onto each script's own layout, with the MIT acquisition imprint
# (duration law + dwell snap, PHASE_Q.md §2 repair round).  Same rows/splits/
# seeds as every prior generation: 1,000,000 train (seed 1234), 5,000 val
# (999), 10,000 holdout (777); noise seeds 20260820+{0,1,2}.
#
# There is no donor split any more — the model has no donors.  The holdout is
# fresh noise + a fresh word draw from the SAME generator weights, so it is
# generator-relative in the same way (and one turn more strongly than) the v2
# holdouts were; §5.1 of PHASE_P.md carries over word for word.  Cross-
# generation comparisons must use margins against the EN zero-shot controls,
# never levels.
#
# Runs SEQUENTIALLY on purpose: each sampler saturates the GPU.
set -u
CTC=/home/will/git/CleverKeys-ML/ctc
GEN=ckpt/synthq_gen_ship/gen.pt
IMP=ckpt/synthq_gen_ship/imprint_mit.json
cd "$CTC" || exit 1
for code in "$@"; do
  echo "### gen $code"
  python3 synth_v3.py sample-cache --gen "$GEN" --imprint "$IMP" \
    --code "$code" --cache "cache_${code}_v3" --rows 1000000 \
    > "$HOME/ctc-train/phaseQ_gen_${code}.log" 2>&1 \
    || { echo "GEN FAILED: $code"; tail -5 "$HOME/ctc-train/phaseQ_gen_${code}.log"; exit 1; }
  grep -E "train:|wrote" "$HOME/ctc-train/phaseQ_gen_${code}.log" | tail -2
done
echo PHASEQ-GEN-DONE
