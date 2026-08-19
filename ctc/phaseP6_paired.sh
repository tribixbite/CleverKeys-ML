#!/usr/bin/env bash
# Phase P6 — re-decode the P4 (90/10 donor split) models on the SAME holdout
# rows with a per-row dump, so P4 vs P6 can be read as a paired McNemar rather
# than two independent point estimates.  P4 saved summaries but no dumps; the
# decode is deterministic, so this reproduces its numbers as a side effect and
# any disagreement with phase_p_scripts.json is itself a finding.
set -u
CTC=/home/will/git/CleverKeys-ML/ctc
O=/home/will/ctc-train/evalP6
cd "$CTC" || exit 1
mkdir -p "$O"
for code in "$@"; do
  probe="$HOME/ctc-train/cache_${code}_v2/holdout.npz"
  python3 eval_script.py --code "$code" \
    --onnx "$HOME/ctc-train/ckpt/phaseP-${code}-v2/ctc_swipe_encoder.onnx" \
    --probe "$probe" --preset ckdt --progress 0 \
    --dump "$O/dump_${code}_p4.jsonl" --out-json "$O/${code}_p4_repro.json" \
    > /dev/null 2>&1
  echo "  $code P4 repro t1 $(python3 -c "import json;print(json.load(open('$O/${code}_p4_repro.json'))['indict_t1'])")"
done
echo PHASEP6-PAIRED-DUMPS-DONE
