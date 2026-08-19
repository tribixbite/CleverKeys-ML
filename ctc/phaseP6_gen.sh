#!/usr/bin/env bash
# Phase P6 — regenerate the five corpus-less scripts on the FULL donor pool.
#
# Exactly the P4 generation with one flag changed: --train-donor-side all, the
# footing the shipped ru arm (cache_ru_v2full) uses.  Everything else — v2
# stages a,c,b,s5, 1,000,000 train rows, 5,000 val, 10,000 holdout, the split
# seeds — is held.  The holdout ALWAYS draws from the reserved donor half
# (script_synth side_of maps holdout -> "holdout" regardless of this flag), so
# each cache_<code>_v2full/holdout.npz must come out bit-identical to the P4
# cache_<code>_v2/holdout.npz; phaseP6_assert_holdout.py checks that, and it is
# what makes P4 vs P6 a paired comparison on the same 10,000 rows.
set -u
CTC=/home/will/git/CleverKeys-ML/ctc
cd "$CTC" || exit 1
for code in "$@"; do
  nohup setsid python3 script_synth.py \
    --code "$code" --cache "cache_${code}_v2full" \
    --rows 1000000 --train-donor-side all \
    > "$HOME/ctc-train/phaseP6_gen_${code}.log" 2>&1 &
  echo "launched gen $code -> ~/ctc-train/phaseP6_gen_${code}.log"
done
