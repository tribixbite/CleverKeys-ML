#!/usr/bin/env bash
# Phase Q — per-script training launcher, the Phase-O/P recipe VERBATIM.
# Identical to phaseP6_train.sh in every training argument; only the cache
# (cache_<code>_v3, the learned-generator synthesis) and the run name change.
#
# SEED (env, default 1234) is the closing round's Q-A axis (PHASE_Q.md §8): the
# ONLY quantity that varies across the replication triple 1234/4321/7777.  With
# SEED unset this file is byte-equivalent in effect to the launcher that
# produced the shipped s1234 runs — same run names, same arguments.  Replicates
# get the `-s<seed>` suffix so the s1234 checkpoints (and their measured export
# hashes) are never overwritten.
#
# `ru` is a case here for the same reason: its s1234 run was launched by hand
# from PHASE_P.md's reproduction block, and the replicates must not be.
set -u
CTC=/home/will/git/CleverKeys-ML/ctc
SEED=${SEED:-1234}
cd "$CTC" || exit 1
for code in "$@"; do
  case "$code" in
    ru) layout=layouts/ru_jcuken_default.json ;;
    el) layout=layouts/el_qwerty.json ;;
    uk) layout=layouts/uk_jcuken.json ;;
    bg) layout=layouts/bg_bds.json ;;
    mk) layout=layouts/mk_lynyertdz.json ;;
    he) layout=layouts/he_1.json ;;
    *)  echo "unknown script $code" >&2; exit 2 ;;
  esac
  run="phaseQ-${code}-v3"
  [ "$SEED" != "1234" ] && run="${run}-s${SEED}"
  nohup setsid python3 train.py \
    --layout "$CTC/$layout" \
    --cache "cache_${code}_v3" \
    --train-npz train_synth.npz \
    --run-name "$run" \
    --epochs 1000000 --batch 256 --lr 3e-3 --weight-decay 0.01 --warmup 1000 \
    --ch 80 --embed-hid 96 --feat-version 1 --block resbn --dilations 1,2,4,8 \
    --total-steps 94000 --val-every 3000 \
    --affine-sampler coupled --layout-alt-p 0.0 \
    --beam-val-rows 0 --patience 40 --seed "$SEED" --workers 0 \
    > "$HOME/ctc-train/ckpt_${run}.launch.log" 2>&1 &
  echo "launched $run -> ~/ctc-train/ckpt_${run}.launch.log"
done
