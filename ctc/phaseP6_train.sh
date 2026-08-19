#!/usr/bin/env bash
# Phase P6 — per-script training launcher, the Phase-O/P recipe VERBATIM.
# Identical to phaseO_train.sh / the P4 launches in every argument; the only
# difference is --cache cache_<code>_v2full (the full-donor-pool regeneration)
# and the run name.  resbn ch80 dil 1,2,4,8, embed_hid 96, feat_v1, 94,000
# steps, batch 256, lr 3e-3, wd 0.01, warmup 1,000, coupled affine sampler, no
# layout-alt, greedy checkpoint selection, patience 40, seed 1234, --workers 0.
set -u
CTC=/home/will/git/CleverKeys-ML/ctc
cd "$CTC" || exit 1
for code in "$@"; do
  case "$code" in
    el) layout=layouts/el_qwerty.json ;;
    uk) layout=layouts/uk_jcuken.json ;;
    bg) layout=layouts/bg_bds.json ;;
    mk) layout=layouts/mk_lynyertdz.json ;;
    he) layout=layouts/he_1.json ;;
    *)  echo "unknown script $code" >&2; exit 2 ;;
  esac
  run="phaseP6-${code}-v2full"
  nohup setsid python3 train.py \
    --layout "$CTC/$layout" \
    --cache "cache_${code}_v2full" \
    --train-npz train_synth.npz \
    --run-name "$run" \
    --epochs 1000000 --batch 256 --lr 3e-3 --weight-decay 0.01 --warmup 1000 \
    --ch 80 --embed-hid 96 --feat-version 1 --block resbn --dilations 1,2,4,8 \
    --total-steps 94000 --val-every 3000 \
    --affine-sampler coupled --layout-alt-p 0.0 \
    --beam-val-rows 0 --patience 40 --seed 1234 --workers 0 \
    > "$HOME/ctc-train/ckpt_${run}.launch.log" 2>&1 &
  echo "launched $run -> ~/ctc-train/ckpt_${run}.launch.log"
done
